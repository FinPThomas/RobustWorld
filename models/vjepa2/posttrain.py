"""Post-train V-JEPA 2's predictor on the clips, so its imagined future can learn the blockade.

The encoder stays frozen; only the predictor (context tokens -> future tokens) is trained.
That keeps the measuring stick fixed (CLAUDE.md): the evaluation decoder is fitted on frozen
context-half features, so it is bit-for-bit the same before and after post-training. It also
makes training cheap, because every clip's encoder features are computed once and cached.

Objectives (`--loss`), all learned purely from the video:
    l1      the V-JEPA latent prediction loss restricted to the future half. The predictor sees the
            context half's tokens (encoded on their own, so nothing leaks from the future) and
            predicts the encoder's tokens of the real target half; L1, as in V-JEPA 2.
    commit  "commit to a ball": L1 averages over possible futures and is dominated by the static
            table (the ball is ~1.5% of tokens), so the pretrained predictor hedges to "no ball".
              - motion weighting: each target token's L1 is weighted up where the features change
                from one step to the next, in the real future OR in the prediction (so a phantom
                ball is as costly as a missed one). No tracker, no labels.
              - in-batch contrast (InfoNCE): the imagined future must be nearer its own real
                future than other training clips' real futures (motion-weighted distance). An
                average of "through" and "hidden" is equally far from both, so hedging costs.
    codes   discrete targets: a codebook (k-means, `--codes` entries) is fitted on each split's
            training clips' real target tokens (half sampled uniformly, half from the tokens that
            change most between steps, so the ball gets codes; no labels). The predictor is trained
            with cross-entropy to pick each target token's code, and at inference every imagined
            token is snapped to its most likely code. A categorical choice can't average "ball" and
            "no ball", so the prediction has to commit.

Inference modes, stored in each checkpoint and used by ball_probe_cv.py:
    --rollout  predict one V-JEPA step (2 frames) at a time, feeding each prediction back in as
               context (snapped to codes with --loss codes), instead of all target steps at once.
               Trained the same way, with the fed-back predictions detached.
    (default)  all target steps in one pass, as in V-JEPA 2.

Architecture additions (label-free, trained with the predictor; both start as the pretrained output):
    --copy-gate      a per-token gate mixes each predicted token with the last context step's token,
                     so the static table and plank can be copied exactly and the predictor's capacity
                     and loss go to what moves (a "clean" background).
    --hypotheses K   K futures from one prediction (low-rank per-token adapters) and a picker that
                     scores them from the predicted change. Trained winner-takes-all: the hypothesis
                     nearest the real future gets the loss and the picker learns which one wins, so
                     hypotheses specialise ("comes through" / "stays hidden") instead of averaging.
                     At inference the picker's choice is the imagined future; the real future is
                     never looked at. Like L1, it learns only from the training clips' own video.

`--epochs 0` saves the pretrained predictor with a run's inference mode (and codebook): the "before"
for that variant. experiments.py runs the whole before/after grid.

Splits: one predictor per cross-validation fold (robust_world.eval.ball.cv_folds, the same
folds the evaluation uses), each trained only on that fold's training clips. Each checkpoint
records its training clip ids, and ball_probe_cv.py refuses a checkpoint whose clips overlap
the fold it scores. `--all` trains one predictor on every clip, for scoring new sessions.

Steps:
    python models/vjepa2/posttrain.py encode               # cache encoder features (GPU, once)
    python models/vjepa2/posttrain.py train --run l1       # checkpoints/vjepa2/l1/fold{k}.pt
    python models/vjepa2/posttrain.py train --run commit --loss commit
    python models/vjepa2/ball_probe_cv.py --predictor-run checkpoints/vjepa2/l1
    python models/vjepa2/blocker_figs.py --per-clip outputs/vjepa2/ball_probe_cv/l1/per_clip.json --label l1

Held-out L1 ("surprise", label-free) is logged per fold for the pretrained and post-trained
predictor, split by outcome, in checkpoints/vjepa2/<run>/log.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.ball import OUTCOMES, cv_folds  # noqa: E402

MANIFEST = REPO / "data" / "processed" / "clips" / "manifest_right.jsonl"
CKPT_ROOT = REPO / "checkpoints" / "vjepa2"


def cache_dir(model_id: str) -> Path:
    """Shared with ball_probe_cv.py, so clips are only ever encoded once."""
    return REPO / "outputs" / "vjepa2" / "cache" / model_id.replace("/", "--")


def training_clips(manifest: Path = MANIFEST) -> list[dict]:
    clips = [json.loads(line) for line in manifest.open()]
    return [c for c in clips if c.get("include") and c["outcome"] in OUTCOMES]


def cache_matches(enc: dict, clip: dict, tubelet: int = 2) -> bool:
    """A cached encoding is only reused if it was made from a clip of the same length and split
    (clips keep their ids when they are re-cut, e.g. from 2 s to 3 s)."""
    n_ctx = clip["context_frames"][1] + 1
    return (enc["real"].shape[0] * tubelet == clip["n_frames"]
            and "context" in enc and enc["context"].shape[0] * tubelet == n_ctx)


def load_cached(cache: Path, clip: dict) -> dict | None:
    path = cache / f"{clip['clip_id']}.pt"
    if not path.exists():
        return None
    enc = torch.load(path)
    return enc if cache_matches(enc, clip) else None


def masks(n_ctx_tok: int, n_all_tok: int, batch: int, device) -> tuple[list, list]:
    ctx = torch.arange(n_ctx_tok, device=device).unsqueeze(0).repeat(batch, 1)
    tgt = torch.arange(n_ctx_tok, n_all_tok, device=device).unsqueeze(0).repeat(batch, 1)
    return [ctx], [tgt]


def imagine(predictor, context: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Context token grids [B, ctx_steps, g, g, D] -> imagined target grids [B, target_steps, g, g, D]."""
    b, cs, g, _, d = context.shape
    per_step = g * g
    cm, tm = masks(cs * per_step, (cs + target_steps) * per_step, b, context.device)
    pred = predictor(encoder_hidden_states=context.reshape(b, cs * per_step, d),
                     context_mask=cm, target_mask=tm).last_hidden_state
    return pred.reshape(b, target_steps, g, g, -1)


def target_space(x: torch.Tensor) -> torch.Tensor:
    """V-JEPA's prediction space: each token layer-normalised (no learned scale). V-JEPA 2 was
    pretrained to predict F.layer_norm(encoder output), and the Hugging Face encoder returns the
    un-normalised output, so real targets are mapped into this space before they are compared."""
    return torch.nn.functional.layer_norm(x.float(), x.shape[-1:]).to(x.dtype)


def to_encoder_space(pred: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Map predicted tokens (target space) back to encoder-output scale, giving each token the
    mean and spread of the same position in `like` (the latest real or fed context step).
    Used when a prediction is fed back in as context. [B, 1, g, g, D], [B, g, g, D]."""
    mu = like.float().mean(-1, keepdim=True)[:, None]
    sd = like.float().std(-1, correction=0, keepdim=True)[:, None]
    return (target_space(pred).float() * sd + mu).to(like.dtype)


def load_predictor(model, checkpoint: Path):
    """Load a post-trained predictor into `model` (in place); returns the checkpoint's metadata,
    including its inference mode ("rollout", "codebook", "heads")."""
    ck = torch.load(checkpoint, map_location="cpu")
    model.predictor.load_state_dict(ck["predictor"])
    meta = {k: v for k, v in ck.items() if k not in ("predictor", "heads", "heads_cfg")}
    meta["heads"] = None
    if ck.get("heads") is not None:
        meta["heads"] = Heads(**ck["heads_cfg"])
        meta["heads"].load_state_dict(ck["heads"])
        meta["heads"].eval()
    return meta


def saved_mode(mode: dict) -> dict:
    """A run's inference mode as stored in a checkpoint (the heads as weights + settings)."""
    heads = mode.get("heads")
    return {"rollout": mode.get("rollout", False), "codebook": mode.get("codebook"),
            "heads": heads.state_dict() if heads is not None else None,
            "heads_cfg": heads.cfg if heads is not None else None}


class Heads(torch.nn.Module):
    """Optional additions on the predictor's output (--copy-gate, --hypotheses; see the docstring).
    Near-identity at the start: the gate is shut (bias -8) and the hypotheses' adapters are ~0."""

    def __init__(self, d: int, copy_gate: bool = False, hypotheses: int = 1, rank: int = 64):
        super().__init__()
        self.cfg = {"d": d, "copy_gate": copy_gate, "hypotheses": hypotheses, "rank": rank}
        self.k = hypotheses
        self.gate = torch.nn.Linear(d, 1) if copy_gate else None
        if self.gate is not None:
            torch.nn.init.zeros_(self.gate.weight)
            torch.nn.init.constant_(self.gate.bias, -8.0)
        if hypotheses > 1:
            g = torch.Generator().manual_seed(0)       # small, different starts break the symmetry
            self.down = torch.nn.Parameter(torch.randn(hypotheses, d, rank, generator=g) / math.sqrt(d))
            self.up = torch.nn.Parameter(torch.randn(hypotheses, rank, d, generator=g) * 1e-3)
            self.picker = torch.nn.Linear(d, hypotheses)
            torch.nn.init.zeros_(self.picker.weight)
            torch.nn.init.zeros_(self.picker.bias)

    def forward(self, pred: torch.Tensor, last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """pred [B, T, g, g, D] and the last context step [B, g, g, D], both in target space
        -> hypotheses [K, B, T, g, g, D] and picker logits [B, K] (None with one hypothesis)."""
        x, last = pred.float(), last.float()[:, None]
        hyps, logits = x[None], None
        if self.k > 1:
            low = torch.einsum("btxyd,kdr->kbtxyr", x, self.down.float())
            hyps = x[None] + torch.einsum("kbtxyr,krd->kbtxyd", low, self.up.float())
            logits = self.picker((x - last).mean((1, 2, 3)))          # from the predicted change
        if self.gate is not None:
            a = torch.sigmoid(self.gate(hyps))
            hyps = hyps + a * (last[None] - hyps)
        return hyps, logits


def pick(hyps: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
    """The picker's choice per clip: [K, B, ...] -> [B, ...]."""
    if logits is None:
        return hyps[0]
    idx = logits.argmax(-1)
    return hyps[idx, torch.arange(hyps.shape[1], device=hyps.device)]


def apply_heads(heads, pred: torch.Tensor, context: torch.Tensor):
    """-> (hypotheses [K, B, ...], logits) for a raw prediction; the last context step is the copy source."""
    return heads.to(pred.device)(pred, target_space(context[:, -1].float()))


def snap(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Replace every token [..., D] with its most likely code's centroid (cosine)."""
    idx = code_logits(x, codebook, 1.0).argmax(-1)
    return codebook.to(x.device, x.dtype)[idx]


def code_logits(x: torch.Tensor, codebook: torch.Tensor, tau: float) -> torch.Tensor:
    c = torch.nn.functional.normalize(codebook.to(x.device, torch.float32), dim=-1)
    return torch.nn.functional.normalize(x.float(), dim=-1) @ c.T / tau


def rollout(predictor, context: torch.Tensor, target_steps: int, codebook=None) -> torch.Tensor:
    """Predict one step at a time, feeding each (detached, snapped if a codebook is given) prediction
    back in as context. Returns the raw per-step predictions [B, target_steps, g, g, D]."""
    out, seen = [], context
    for _ in range(target_steps):
        step = imagine(predictor, seen, 1)
        out.append(step)
        fed = snap(step.detach(), codebook) if codebook is not None else step.detach()
        seen = torch.cat([seen, to_encoder_space(fed, seen[:, -1])], 1)
    return torch.cat(out, 1)


def imagine_mode(predictor, context: torch.Tensor, target_steps: int, mode: dict) -> torch.Tensor:
    """The imagined target as a checkpoint's inference mode produces it (what the evaluation reads)."""
    codebook, heads = mode.get("codebook"), mode.get("heads")
    pred = (rollout(predictor, context, target_steps, codebook) if mode.get("rollout")
            else imagine(predictor, context, target_steps))
    if heads is not None:
        pred = pick(*apply_heads(heads, pred, context))
    return snap(pred, codebook) if codebook is not None else pred


def fit_codebook(data, k: int, seed: int, n_tokens: int = 50_000, iters: int = 15, device="cpu") -> torch.Tensor:
    """Spherical k-means on real target tokens of `data` (a Cached set): half drawn uniformly, half
    from the tokens that change most from the previous step. -> centroids [k, D] in feature space."""
    g = torch.Generator().manual_seed(seed)
    per_clip = max(1, n_tokens // (2 * len(data)))
    picks = []
    for i in range(len(data)):
        _, tgt, last, _ = data[i]
        flat = tgt.reshape(-1, tgt.shape[-1])
        motion = step_change(tgt[None], last[None])[0].reshape(-1)
        picks.append(flat[torch.randperm(len(flat), generator=g)[:per_clip]])
        picks.append(flat[motion.topk(min(per_clip, len(flat))).indices])
    x = torch.cat(picks).to(device)
    xn = torch.nn.functional.normalize(x, dim=-1)
    c = xn[torch.randperm(len(xn), generator=g)[:k].to(device)].clone()
    for _ in range(iters):
        assign = (xn @ c.T).argmax(-1)
        for j in range(len(c)):
            m = assign == j
            if m.any():
                c[j] = torch.nn.functional.normalize(xn[m].mean(0), dim=0)
    assign = (xn @ c.T).argmax(-1)
    cent = torch.stack([x[assign == j].mean(0) if (assign == j).any() else x[j] for j in range(len(c))])
    return cent.cpu()


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """V-JEPA's latent L1 between imagined and real target tokens."""
    return (pred - target).abs().mean()


def step_change(seq: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """Per-token feature change from the previous step, [B, T, g, g, D] -> [B, T, g, g];
    `first` [B, g, g, D] is the step before seq[:, 0]."""
    prev = torch.cat([first[:, None], seq[:, :-1]], 1)
    return (seq - prev).abs().mean(-1)


def motion_weights(target, pred, last_real, alpha: float) -> torch.Tensor:
    """1 + alpha * (token motion / its mean over the clip), motion = max of real and predicted change.
    Detached: the weights say where to look, they are not themselves optimised."""
    with torch.no_grad():
        m = torch.maximum(step_change(target, last_real), step_change(pred.float(), last_real))
        return 1 + alpha * m / m.mean((1, 2, 3), keepdim=True).clamp_min(1e-6)


def weighted_l1(pred, target, w) -> torch.Tensor:
    return ((pred - target).abs().mean(-1) * w).sum() / w.sum()


def contrast(pred, target, negatives, w, tau: float) -> torch.Tensor:
    """InfoNCE over motion-weighted L1 distances: row i's positive is its own real future, the rest
    are the other clips in the batch plus `negatives` [K, T, g, g, D] (real futures of other
    training clips)."""
    cands = torch.cat([target, negatives], 0) if negatives is not None else target       # [C, T, g, g, D]
    d = torch.stack([((pred[i:i + 1] - cands).abs().mean(-1) * w[i]).sum((1, 2, 3)) / w[i].sum()
                     for i in range(len(pred))])                                         # [B, C]
    return torch.nn.functional.cross_entropy(-d / tau, torch.arange(len(pred), device=pred.device))


def heads_objective(args, hyps, logits, target, last_real, negatives=None, codebook=None):
    """Winner-takes-all over hypotheses, per clip: the hypothesis nearest the real future gets the
    loss (the others a small share, so none is abandoned) and the picker learns to choose it."""
    if logits is None:
        return objective(args, hyps[0], target, last_real, negatives, codebook)
    total, parts, wins = 0.0, [], []
    for b in range(target.shape[0]):
        per = [objective(args, hyps[k, b:b + 1], target[b:b + 1], last_real[b:b + 1], negatives, codebook)
               for k in range(len(hyps))]
        vals = torch.stack([v for v, _ in per])
        win = int(vals.detach().argmin())
        pick_ce = torch.nn.functional.cross_entropy(logits[b:b + 1].float(), torch.tensor([win], device=vals.device))
        total = total + vals[win] + args.wta_relax * vals.mean() + pick_ce
        parts.append({**per[win][1], "pick_ce": pick_ce.item(), "winner": win})
        wins.append(win)
    part = {k: float(np.mean([p[k] for p in parts])) for k in parts[0] if k != "winner"}
    part["winner_spread"] = len(set(wins)) / len(hyps)
    return total / target.shape[0], part


def objective(args, pred, target, last_real, negatives=None, codebook=None) -> tuple[torch.Tensor, dict]:
    pred = pred.float()
    if args.loss == "codes":
        logits = code_logits(pred, codebook, args.code_tau)
        labels = code_logits(target, codebook, 1.0).argmax(-1)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return ce, {"ce": ce.item(), "l1": loss_fn(pred, target).item()}
    if args.loss == "l1":
        loss = loss_fn(pred, target)
        return loss, {"l1": loss.item()}
    w = motion_weights(target, pred, last_real, args.motion_alpha)
    wl1 = weighted_l1(pred, target, w)
    nce = contrast(pred, target, negatives, w, args.tau)
    return wl1 + args.contrast_weight * nce, {"l1": loss_fn(pred, target).item(), "wl1": wl1.item(),
                                              "nce": nce.item()}


# ---------------------------------------------------------------------------------------------

def encode_all(args) -> None:
    from transformers import AutoVideoProcessor, VJEPA2Model
    from ball_probe import encode
    from robust_world.eval.io import read_video
    from run import pick_device

    device = pick_device()
    model = VJEPA2Model.from_pretrained(args.model_id).to(device).eval()
    proc = AutoVideoProcessor.from_pretrained(args.model_id)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    cache = cache_dir(args.model_id)
    cache.mkdir(parents=True, exist_ok=True)
    clips = training_clips(args.manifest)
    t0, done = time.perf_counter(), 0
    for i, c in enumerate(clips):
        if load_cached(cache, c) is None:
            frames = read_video(REPO / c["path"])
            enc = encode(model, frames, c["context_frames"][1] + 1, mean, std, device, imagine=True)
            torch.save(enc, cache / f"{c['clip_id']}.pt")
            done += 1
        print(f"\r  {i + 1}/{len(clips)} clips ({done} newly encoded, {time.perf_counter() - t0:.0f}s)",
              end="", flush=True)
    print(f"\n-> {cache}")


class Cached(torch.utils.data.Dataset):
    """(context grid, real target grid, last real context step, index) per clip, read from the
    encoding cache on demand. Context stays in encoder space (the predictor's input); the target and
    last step are in target space (what the predictor outputs)."""

    def __init__(self, clips: list[dict], cache: Path):
        self.clips, self.cache = clips, cache
        missing = [c["clip_id"] for c in clips if load_cached(cache, c) is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips not encoded (e.g. {missing[:3]}); "
                                    "run `posttrain.py encode` first")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        enc = load_cached(self.cache, self.clips[i])
        cs = enc["context"].shape[0]
        return (enc["context"].float(), target_space(enc["real"][cs:].float()),
                target_space(enc["real"][cs - 1].float()), i)


@torch.no_grad()
def held_out_l1(predictor, data: Cached, device, autocast, mode: dict) -> list[float]:
    """L1 between the imagined target, in the run's inference mode, and the real one (label-free)."""
    predictor.eval()
    if mode.get("heads") is not None:
        mode["heads"].eval()
    out = []
    for i in range(len(data)):
        ctx, tgt, _, _ = data[i]
        with autocast():
            pred = imagine_mode(predictor, ctx[None].to(device), tgt.shape[0], mode)
        out.append(float(loss_fn(pred.float(), tgt[None].to(device))))
    return out


def by_outcome(clips, values) -> dict:
    return {o: round(float(np.mean([v for c, v in zip(clips, values) if c["outcome"] == o])), 5)
            for o in OUTCOMES if any(c["outcome"] == o for c in clips)}


OVERFIT_RISE = 0.02      # validation L1 this far above its best by the last epoch -> overfitting
GAP_WARN = 1.25          # validation L1 / training-clip L1 above this -> memorising the training clips


def check_split(train: list[dict], test: list[dict]) -> None:
    """Leakage guard: no clip, and no pass of the same recording, on both sides of a split."""
    def key(c):
        return (c.get("source_video"), c.get("pass_id", c["clip_id"]))
    shared = {c["clip_id"] for c in train} & {c["clip_id"] for c in test}
    shared |= {str(k) for k in {key(c) for c in train} & {key(c) for c in test}}
    if shared:
        raise ValueError(f"train and held-out share {len(shared)} clips/passes, e.g. {sorted(shared)[:3]}")


def overfit_report(val_l1: list[float], train_l1: list[float], best_v: float, best_epoch: int) -> dict:
    """Overfitting signs from per-epoch validation and training-clip L1 (same clip count, eval mode)."""
    last_v = val_l1[-1] if val_l1 else best_v
    i = max(best_epoch - 1, 0)
    gap = val_l1[i] / train_l1[i] if val_l1 else 1.0
    return {"val_rise_from_best": round(last_v / best_v - 1, 4), "val_over_train": round(gap, 3),
            "flag": bool(last_v > best_v * (1 + OVERFIT_RISE) or gap > GAP_WARN)}


def split_validation(train: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold back part of the TRAINING clips to watch for overfitting and pick the best epoch.
    The fold's held-out clips are never used for that (they are what gets scored)."""
    n = int(round(len(train) * frac))
    if frac <= 0 or n < 2 or len(train) - n < 2:
        return train, []
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    val = set(order[:n])
    return [c for i, c in enumerate(train) if i not in val], [c for i, c in enumerate(train) if i in val]


def train_one(model, base_state, train: list[dict], test: list[dict], args, device, tag: str) -> tuple[dict, dict]:
    """Train one split. Returns (log, inference mode {"rollout", "codebook"}).

    Guards: train and held-out must not share clips or passes; a validation part of the training
    clips is scored every epoch (L1 in target space), the best epoch's weights are kept
    (--patience stops early), and the log flags overfitting (validation L1 rising from its best,
    or far above the L1 on training clips)."""
    check_split(train, test)
    cache = cache_dir(args.model_id)
    model.predictor.load_state_dict(base_state)
    pred = model.predictor
    train, val = split_validation(train, getattr(args, "val_frac", 0.1), args.seed)
    if getattr(args, "train_frac", 1.0) < 1.0:     # data-efficiency runs: a seeded share of the training clips
        order = list(range(len(train)))
        random.Random(args.seed + 1).shuffle(order)
        train = [train[i] for i in sorted(order[:max(2, int(round(len(train) * args.train_frac)))])]
    tr, te = Cached(train, cache), Cached(test, cache) if test else None
    va = Cached(val, cache) if val else None
    tr_probe = Cached(train[:max(len(val), 2)], cache) if val else None    # same-size sample of training clips
    codebook = (fit_codebook(tr, args.codes, args.seed, device=device) if args.loss == "codes" else None)
    heads = None
    if getattr(args, "copy_gate", False) or getattr(args, "hypotheses", 1) > 1:
        heads = Heads(tr[0][1].shape[-1], getattr(args, "copy_gate", False), getattr(args, "hypotheses", 1)).to(device)
    mode = {"rollout": bool(getattr(args, "rollout", False)), "codebook": codebook, "heads": heads}

    use_amp = device == "cuda"
    # bf16 only where it is native (A100/L4 and newer). On a T4 bf16 is emulated and attention falls
    # back to a kernel that stores the full token-by-token matrix, which runs out of memory.
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if getattr(args, "grad_checkpoint", True):
        # Recompute each layer's activations in the backward pass instead of keeping them: a little
        # slower, far less memory (3 s clips are 6,144 predictor tokens).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    log = {"tag": tag, "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    if te:
        before = held_out_l1(pred, te, device, autocast, mode)
        log["held_out_l1_pretrained"] = by_outcome(test, before)

    groups = [{"params": list(pred.parameters())}]
    if heads is not None:                     # new weights start from scratch: a higher learning rate
        groups.append({"params": list(heads.parameters()), "lr": args.lr * 30, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                         num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed))
    total = max(1, args.epochs * math.ceil(len(loader) / args.accum))
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    bank = None
    if args.loss == "commit" and args.negatives:
        # Real futures of every training clip, kept in fp16 on the CPU; K are drawn per step.
        bank = torch.stack([tr[i][1].half() for i in range(len(tr))])
        if len(bank) <= args.batch_size:
            bank = None                           # too few clips for outside negatives
    rng = torch.Generator().manual_seed(args.seed)
    log["train_loss"], log["train_parts"], log["val_l1"], log["train_clip_l1"] = [], [], [], []
    log["epochs"] = []                            # one detailed record per epoch (see epoch_record)
    log["val_clip_ids"] = [c["clip_id"] for c in val]
    best = (float(np.mean(held_out_l1(pred, va, device, autocast, mode))), 0, None, None) if va else None
    if va:
        log["val_l1_pretrained"] = round(best[0], 5)
    patience = getattr(args, "patience", 3)
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        pred.train()
        if heads is not None:
            heads.train()
        losses, parts, grads = [], [], []
        t_epoch = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, (ctx, tgt, last, ids) in enumerate(loader):
            ctx, tgt, last = ctx.to(device), tgt.to(device), last.to(device)
            neg = None
            if bank is not None:                  # other clips only: never a clip's own future
                others = torch.tensor([j for j in range(len(bank)) if j not in set(ids.tolist())])
                idx = others[torch.randperm(len(others), generator=rng)[:args.negatives]]
                neg = bank[idx].to(device).float()
            with autocast():
                imagined = (rollout(pred, ctx, tgt.shape[1], codebook) if mode["rollout"]
                            else imagine(pred, ctx, tgt.shape[1]))
            if heads is not None:
                hyps, logits = apply_heads(heads, imagined, ctx)
                loss, part = heads_objective(args, hyps, logits, tgt, last, neg, codebook)
            else:
                loss, part = objective(args, imagined, tgt, last, neg, codebook)
            parts.append(part)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                grads.append(float(torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(loss.item())
        log["train_loss"].append(round(float(np.mean(losses)), 5))
        if parts:
            log["train_parts"].append({k: round(float(np.mean([p[k] for p in parts])), 5) for k in parts[0]})
        print(f"  [{tag}] epoch {epoch + 1}/{args.epochs}: loss {log['train_loss'][-1]:.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in (log["train_parts"][-1] if parts else {}).items())
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        rec = epoch_record(epoch + 1, log, grads, opt, pred, base_state, device, time.perf_counter() - t_epoch)
        log["epochs"].append(rec)
        print(f"  [{tag}]   grad norm mean {rec['grad_norm_mean']} max {rec['grad_norm_max']} "
              f"(clipped {rec['grad_clipped_share']:.0%}, non-finite steps {rec['nonfinite_steps']}), "
              f"weights moved {rec['weight_change']:.2e}, lr {rec['lr']:.1e}, "
              f"peak GPU memory {rec['peak_gpu_gb']} GB, {rec['seconds']}s", flush=True)
        if va:
            v = float(np.mean(held_out_l1(pred, va, device, autocast, mode)))
            t = float(np.mean(held_out_l1(pred, tr_probe, device, autocast, mode)))
            log["val_l1"].append(round(v, 5)), log["train_clip_l1"].append(round(t, 5))
            rec["val_l1"], rec["train_clip_l1"] = round(v, 5), round(t, 5)
            print(f"  [{tag}]   validation L1 {v:.4f} (training clips {t:.4f})", flush=True)
            if v < best[0]:
                best = (v, epoch + 1, {k: x.detach().to("cpu", copy=True) for k, x in pred.state_dict().items()},
                        {k: x.detach().to("cpu", copy=True) for k, x in heads.state_dict().items()} if heads else None)
            elif patience and epoch + 1 - best[1] >= patience:
                print(f"  [{tag}]   no validation gain for {patience} epochs: stopping early", flush=True)
                break
    if va:
        log["best_epoch"] = best[1]
        if best[2] is not None and getattr(args, "keep_best", True):
            pred.load_state_dict(best[2])             # weights from the best validation epoch
            if heads is not None:
                heads.load_state_dict(best[3])
        elif best[1] == 0 and getattr(args, "keep_best", True):
            pred.load_state_dict(base_state)          # never beat the pretrained predictor
        log["overfit"] = overfit_report(log["val_l1"], log["train_clip_l1"], best[0], best[1])
        if log["overfit"]["flag"]:
            o = log["overfit"]
            print(f"  [{tag}] WARNING overfitting: validation L1 rose {o['val_rise_from_best']:.1%} from its best "
                  f"(epoch {best[1]}); validation / training-clip L1 {o['val_over_train']}. "
                  f"Kept epoch {best[1]}.", flush=True)
    if te:
        log["held_out_l1_posttrained"] = by_outcome(test, held_out_l1(pred, te, device, autocast, mode))
        print(f"  [{tag}] held-out L1 pretrained {log['held_out_l1_pretrained']} -> "
              f"post-trained {log['held_out_l1_posttrained']}")
    log["seconds"] = round(time.perf_counter() - t0, 1)
    log["final_weight_change"] = weight_change(pred, base_state)
    return log, mode


@torch.no_grad()
def weight_change(pred, base_state: dict) -> float:
    """||weights - pretrained|| / ||pretrained|| over the whole predictor."""
    num = sum(float((v.float() - base_state[k].float()).norm()) ** 2 for k, v in pred.state_dict().items())
    den = sum(float(v.float().norm()) ** 2 for v in base_state.values())
    return round(math.sqrt(num / max(den, 1e-12)), 6)


def epoch_record(epoch: int, log: dict, grads: list[float], opt, pred, base_state: dict, device, seconds: float) -> dict:
    finite = [g for g in grads if math.isfinite(g)]
    return {"epoch": epoch, "train_loss": log["train_loss"][-1],
            **({"parts": log["train_parts"][-1]} if log["train_parts"] else {}),
            "grad_norm_mean": round(float(np.mean(finite)), 4) if finite else None,
            "grad_norm_max": round(float(np.max(finite)), 4) if finite else None,
            "grad_clipped_share": round(sum(g > 1.0 for g in finite) / max(len(finite), 1), 3),
            "nonfinite_steps": len(grads) - len(finite), "optimizer_steps": len(grads),
            "lr": opt.param_groups[0]["lr"], "weight_change": weight_change(pred, base_state),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
            "seconds": round(seconds, 1)}


@torch.no_grad()
def scale_check(predictor, enc: dict, device) -> dict:
    """Per-token spread of the pretrained prediction vs the real target, raw and in target space."""
    cs = enc["context"].shape[0]
    real = enc["real"][cs:].float()
    pred = imagine(predictor.eval(), enc["context"][None].float().to(device), real.shape[0])[0].float().cpu()
    sd = lambda x: round(float(x.std(-1).mean()), 4)  # noqa: E731
    return {"pred_spread": sd(pred), "target_spread_raw": sd(real), "target_spread_ln": sd(target_space(real)),
            "l1_vs_raw": round(float((pred - real).abs().mean()), 4),
            "l1_vs_ln": round(float((pred - target_space(real)).abs().mean()), 4)}


RESUME_KEYS = ("epochs", "loss", "lr", "rollout", "copy_gate", "hypotheses", "train_frac", "codes", "seed")


def resumable(path: Path, args, train: list[dict]) -> dict | None:
    """With --resume: the saved log of a split already trained with the same settings and clips."""
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu")
    saved = ck.get("args", {})
    same = all(saved.get(k) == getattr(args, k, None) for k in RESUME_KEYS)
    return ck.get("log") if same and ck.get("train_clip_ids") == [c["clip_id"] for c in train] else None


def train_all(args) -> None:
    from transformers import VJEPA2Model
    from run import pick_device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()
    model = VJEPA2Model.from_pretrained(args.model_id)
    model.encoder = None                      # frozen and already cached; frees memory
    model.to(device)
    base_state = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}
    clips = training_clips(args.manifest)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    print(f"code version {commit.stdout.strip() or 'unknown'}", flush=True)
    sc = scale_check(model.predictor, load_cached(cache_dir(args.model_id), clips[0]), device)
    print(f"scale check: predicted token spread {sc['pred_spread']}, real target {sc['target_spread_raw']} raw / "
          f"{sc['target_spread_ln']} layer-normalised (training and scoring use the layer-normalised space)",
          flush=True)
    out = CKPT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    info = {"run": args.run, "code_version": commit.stdout.strip() or "unknown", "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_clips": len(clips),
            "clips_per_outcome": {o: sum(c["outcome"] == o for c in clips) for o in OUTCOMES},
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "scale_check": sc, "folds": []}
    t_run = time.perf_counter()
    splits = ([("all", list(range(len(clips))), [])] if args.all else
              [(f"fold{k}", tr, te) for k, (tr, te) in enumerate(cv_folds(clips, args.folds, args.seed))
               if args.fold is None or k == args.fold])
    print(f"post-training the V-JEPA 2 predictor on {device}: {len(clips)} clips, run '{args.run}', "
          f"{len(splits)} split(s)")
    logs = []
    for tag, tr, te in splits:
        train, test = [clips[i] for i in tr], [clips[i] for i in te]
        done = resumable(out / f"{tag}.pt", args, train) if args.resume else None
        if done is not None:
            print(f"  [{tag}] already trained with these settings: skipping (--resume)", flush=True)
            log = done
        else:
            log, mode = train_one(model, base_state, train, test, args, device, tag)
            torch.save({"predictor": model.predictor.state_dict(), **saved_mode(mode), "model_id": args.model_id,
                        "split": tag, "train_clip_ids": [c["clip_id"] for c in train],
                        "args": vars(args) | {"manifest": str(args.manifest)}, "log": log}, out / f"{tag}.pt")
        logs.append(log)
        (out / "log.json").write_text(json.dumps(logs, indent=1))
        info["folds"].append({"split": tag, "seconds": log["seconds"], "best_epoch": log.get("best_epoch"),
                              "overfit": log.get("overfit"), "final_weight_change": log["final_weight_change"],
                              "held_out_l1_pretrained": log.get("held_out_l1_pretrained"),
                              "held_out_l1_posttrained": log.get("held_out_l1_posttrained"),
                              "peak_gpu_gb": max((e["peak_gpu_gb"] or 0 for e in log["epochs"]), default=None)})
        info["seconds"] = round(time.perf_counter() - t_run, 1)
        (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"-> {out}/ ({', '.join(t for t, _, _ in splits)}.pt, log.json, run_info.json)")


def main(argv: list[str] | None = None) -> int:
    from run import MODEL_ID

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["encode", "train"])
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--run", default="l1", help="checkpoints go to checkpoints/vjepa2/<run>/")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, help="train only this fold")
    p.add_argument("--all", action="store_true", help="one predictor on every clip (no held-out fold)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss", choices=["l1", "commit", "codes"], default="l1")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                   help="keep activations instead of recomputing them (faster, needs a big GPU)")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="share of each fold's training clips held back to watch for overfitting (0 = off)")
    p.add_argument("--patience", type=int, default=3, help="stop after this many epochs without validation gain")
    p.add_argument("--no-keep-best", dest="keep_best", action="store_false",
                   help="keep the last epoch instead of the best validation epoch")
    p.add_argument("--rollout", action="store_true", help="predict one step at a time, feeding predictions back")
    p.add_argument("--copy-gate", action="store_true", help="per-token gate that can copy the last context step")
    p.add_argument("--hypotheses", type=int, default=1, help="K futures, trained winner-takes-all, with a picker")
    p.add_argument("--wta-relax", type=float, default=0.05, help="hypotheses: share of the loss for non-winners")
    p.add_argument("--train-frac", type=float, default=1.0, help="use this share of each fold's training clips")
    p.add_argument("--resume", action="store_true", help="skip splits already trained with the same settings")
    p.add_argument("--codes", type=int, default=256, help="codes: codebook size")
    p.add_argument("--code-tau", type=float, default=0.05, help="codes: softmax temperature (cosine)")
    p.add_argument("--motion-alpha", type=float, default=4.0, help="commit: extra weight on moving tokens")
    p.add_argument("--contrast-weight", type=float, default=0.1, help="commit: weight of the InfoNCE term")
    p.add_argument("--tau", type=float, default=0.02, help="commit: InfoNCE temperature (in L1 units)")
    p.add_argument("--negatives", type=int, default=8, help="commit: other clips' futures per step")
    args = p.parse_args(argv)
    encode_all(args) if args.stage == "encode" else train_all(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

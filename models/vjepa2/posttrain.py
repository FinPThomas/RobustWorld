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


def load_predictor(model, checkpoint: Path):
    """Load a post-trained predictor into `model` (in place); returns the checkpoint's metadata,
    including its inference mode ("rollout", "codebook")."""
    ck = torch.load(checkpoint, map_location="cpu")
    model.predictor.load_state_dict(ck["predictor"])
    return {k: v for k, v in ck.items() if k != "predictor"}


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
        seen = torch.cat([seen, fed.to(seen.dtype)], 1)
    return torch.cat(out, 1)


def imagine_mode(predictor, context: torch.Tensor, target_steps: int, mode: dict) -> torch.Tensor:
    """The imagined target as a checkpoint's inference mode produces it (what the evaluation reads)."""
    codebook = mode.get("codebook")
    pred = (rollout(predictor, context, target_steps, codebook) if mode.get("rollout")
            else imagine(predictor, context, target_steps))
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
    """(context grid, real target grid) per clip, read from the encoding cache on demand."""

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
        return enc["context"].float(), enc["real"][cs:].float(), enc["real"][cs - 1].float(), i


@torch.no_grad()
def held_out_l1(predictor, data: Cached, device, autocast, mode: dict) -> list[float]:
    """L1 between the imagined target, in the run's inference mode, and the real one (label-free)."""
    predictor.eval()
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


def train_one(model, base_state, train: list[dict], test: list[dict], args, device, tag: str) -> tuple[dict, dict]:
    """Train one split. Returns (log, inference mode {"rollout", "codebook"})."""
    cache = cache_dir(args.model_id)
    model.predictor.load_state_dict(base_state)
    pred = model.predictor
    tr, te = Cached(train, cache), Cached(test, cache) if test else None
    codebook = (fit_codebook(tr, args.codes, args.seed, device=device) if args.loss == "codes" else None)
    mode = {"rollout": bool(getattr(args, "rollout", False)), "codebook": codebook}

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    log = {"tag": tag, "n_train": len(train), "n_test": len(test)}
    if te:
        before = held_out_l1(pred, te, device, autocast, mode)
        log["held_out_l1_pretrained"] = by_outcome(test, before)

    opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    log["train_loss"], log["train_parts"] = [], []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        pred.train()
        losses, parts = [], []
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
            loss, part = objective(args, imagined, tgt, last, neg, codebook)
            parts.append(part)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
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
    if te:
        log["held_out_l1_posttrained"] = by_outcome(test, held_out_l1(pred, te, device, autocast, mode))
        print(f"  [{tag}] held-out L1 pretrained {log['held_out_l1_pretrained']} -> "
              f"post-trained {log['held_out_l1_posttrained']}")
    log["seconds"] = round(time.perf_counter() - t0, 1)
    return log, mode


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
    out = CKPT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    splits = ([("all", list(range(len(clips))), [])] if args.all else
              [(f"fold{k}", tr, te) for k, (tr, te) in enumerate(cv_folds(clips, args.folds, args.seed))
               if args.fold is None or k == args.fold])
    print(f"post-training the V-JEPA 2 predictor on {device}: {len(clips)} clips, run '{args.run}', "
          f"{len(splits)} split(s)")
    logs = []
    for tag, tr, te in splits:
        train, test = [clips[i] for i in tr], [clips[i] for i in te]
        log, mode = train_one(model, base_state, train, test, args, device, tag)
        torch.save({"predictor": model.predictor.state_dict(), **mode, "model_id": args.model_id, "split": tag,
                    "train_clip_ids": [c["clip_id"] for c in train], "args": vars(args) | {"manifest": str(args.manifest)},
                    "log": log}, out / f"{tag}.pt")
        logs.append(log)
        (out / "log.json").write_text(json.dumps(logs, indent=1))
    print(f"-> {out}/ ({', '.join(t for t, _, _ in splits)}.pt, log.json)")


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
    p.add_argument("--rollout", action="store_true", help="predict one step at a time, feeding predictions back")
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

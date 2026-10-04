"""Probing V-JEPA 2's frozen encoder: which readout best locates the ball, and is the blockade in it?

Probing only (no post-training). Every probe here obeys CLAUDE.md: it is fitted on REAL
features of the CONTEXT half encoded on its own, labelled with the ball cells of the same
frames, through eval_decoder.examples_from(). Probes never see predictions, target-half labels
or outcomes; those are only used to SCORE them on held-out clips (cv_folds, 5 folds, seed 0).

Stages (each resumable; `all` runs them in order and keeps the machine awake):
  encode     per clip: context-only and full-clip encodings at several encoder layers, and the
             pretrained predictor's imagined target (outputs/vjepa2/probe_cache/, ~80 MB a clip)
  baseline   colour tracker + straight line + plank/blockade rules, blockade range fitted on each
             fold's training outcomes (a reference model, like the TAPNext rule baseline)
  sweep      probe grid: layer x linear/MLP x token/3x3/global input x raw/layer-norm x
             per-cell sigmoid/one-ball softmax. Scored on held-out context frames, on real target
             frames (including far-side cells no probe ever saw a ball in) and, for the last
             layer, on the pretrained predictor's imagined target
  blockade   label-free blockade analyses: real-frame P(blocked) against where the path crosses
             the plank, per-cell distinctiveness of static context features, and per-cell
             surprise (imagined vs real) by outcome
  report     outputs/vjepa2/probing/summary.{md,json} and figures

    python models/vjepa2/probing.py all
    python models/vjepa2/probing.py encode --limit 10        # quick check
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.ball import (MANIFEST, ball_labels, cv_folds, included_clips, load_tracking,  # noqa: E402
                                    source_indices, video_of)
from robust_world.eval.io import read_video  # noqa: E402

OUT = REPO / "outputs" / "vjepa2" / "probing"
CACHE = REPO / "outputs" / "vjepa2" / "probe_cache"
LAYERS = ("L4", "L8", "L12", "L16", "L20", "last")     # hidden_states index, or the encoder's output
GRID = 16


def keep_awake(on: bool = True) -> None:
    """Ask Windows not to sleep while this process runs (no setting is changed)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------------------- encode

def layer_tokens(out, name: str) -> torch.Tensor:
    return out.last_hidden_state if name == "last" else out.hidden_states[int(name[1:])]


@torch.inference_mode()
def encode_clip(model, frames, n_ctx: int, mean, std) -> dict:
    from run import to_pixels
    g = 16
    per = g * g
    ctx_steps, all_steps = n_ctx // 2, len(frames) // 2
    t = time.perf_counter()
    ctx = model(to_pixels(frames[:n_ctx], mean, std, "cpu"), skip_predictor=True, output_hidden_states=True)
    full = model(to_pixels(frames, mean, std, "cpu"), skip_predictor=True, output_hidden_states=True)
    t_enc = time.perf_counter() - t
    pred = model.predictor(
        encoder_hidden_states=ctx.last_hidden_state,
        context_mask=[torch.arange(ctx_steps * per).unsqueeze(0)],
        target_mask=[torch.arange(ctx_steps * per, all_steps * per).unsqueeze(0)]).last_hidden_state
    out = {}
    for name in LAYERS:
        out[f"ctx_{name}"] = layer_tokens(ctx, name).reshape(ctx_steps, g, g, -1)
        out[f"tgt_{name}"] = layer_tokens(full, name).reshape(all_steps, g, g, -1)[ctx_steps:]
    out["imag"] = pred.reshape(all_steps - ctx_steps, g, g, -1)
    return {k: v.half().numpy() for k, v in out.items()}, {"encode_s": round(t_enc, 1),
                                                           "predict_s": round(time.perf_counter() - t - t_enc, 1)}


def cmd_encode(args) -> None:
    from transformers import AutoVideoProcessor, VJEPA2Model
    from run import MODEL_ID
    torch.set_num_threads(args.threads)
    clips = included_clips(args.manifest)[:args.limit]
    todo = [c for c in clips if not (CACHE / c["clip_id"] / "done.json").exists()]
    log(f"encode: {len(clips)} clips, {len(todo)} to do")
    if not todo:
        return
    model = VJEPA2Model.from_pretrained(MODEL_ID).eval()
    proc = AutoVideoProcessor.from_pretrained(MODEL_ID)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    t0 = time.perf_counter()
    for i, c in enumerate(todo):
        try:
            frames = read_video(REPO / c["path"])
            arrays, timing = encode_clip(model, frames, c["context_frames"][1] + 1, mean, std)
        except Exception as e:                                   # one bad clip shouldn't stop the night
            log(f"encode {c['clip_id']} FAILED: {e!r}")
            continue
        d = CACHE / c["clip_id"]
        d.mkdir(parents=True, exist_ok=True)
        for k, v in arrays.items():
            np.save(d / f"{k}.npy", v)
        (d / "done.json").write_text(json.dumps({**timing, "n_frames": len(frames)}))
        el = time.perf_counter() - t0
        log(f"encode {i + 1}/{len(todo)} {c['clip_id']} {timing} | {el / 60:.0f} min, "
            f"~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left")


def load(clip_id: str, key: str, mmap: bool = False) -> np.ndarray:
    return np.load(CACHE / clip_id / f"{key}.npy", mmap_mode="r" if mmap else None)


def encoded_clips(args) -> list[dict]:
    clips = included_clips(args.manifest)[:args.limit]
    missing = [c["clip_id"] for c in clips if not (CACHE / c["clip_id"] / "done.json").exists()]
    if len(missing) > len(clips) // 10:
        raise SystemExit(f"{len(missing)} clips not encoded yet (e.g. {missing[:3]}); run `encode` first")
    if missing:
        log(f"leaving out {len(missing)} clips that failed to encode: {missing} (folds change accordingly)")
    return [c for c in clips if c["clip_id"] not in missing]


def truth(clips: list[dict]) -> tuple[list[dict], dict]:
    """Per clip: ball cells and centres per 2-frame step (tracker), split into context and target."""
    from robust_world.eval.ball import far_cells, near_cells
    track, passes, scene = load_tracking(video_of(clips))
    rows = []
    for c in clips:
        n_ctx = c["context_frames"][1] + 1
        src = source_indices(c, passes, c["n_frames"], n_ctx)
        lab, centres = ball_labels(src, track)
        k = n_ctx // 2
        ctx_px = [(track[i][1], track[i][2]) if track[i][0] else None for i in src[:n_ctx]]
        radius = float(np.median([track[i][3] for i in src[:n_ctx] if track[i][0]] or [20.0]))
        rows.append({"clip_id": c["clip_id"], "outcome": c["outcome"], "labels": lab, "centres": centres,
                     "ctx_steps": k, "ctx_centres_px": ctx_px, "radius": radius})
    return rows, {"scene": scene, "far": far_cells(scene), "near": near_cells(scene)}


# ------------------------------------------------------------------------------------------ probes

class Config:
    """layer, model (linear|mlp), input (token|nbhd|global), norm (raw|ln), head (sigmoid|softmax)."""
    FIELDS = ("layer", "model", "input", "norm", "head")

    def __init__(self, layer, model, input, norm, head):
        self.layer, self.model, self.input, self.norm, self.head = layer, model, input, norm, head

    @property
    def name(self) -> str:
        return "-".join(getattr(self, f) for f in self.FIELDS)

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.FIELDS}


class Probe(torch.nn.Module):
    """Cell logits [B, g, g] and a no-ball logit [B] from a step's token grid [B, g, g, D].
    token/nbhd: one shared map per token (on its own, or averaged over its 3x3 neighbourhood);
    global: the step's mean token -> all cells at once (no spatial prior)."""

    def __init__(self, d: int, g: int, cfg: Config, mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.cfg, self.g = cfg, g
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        n_out = g * g + 1 if cfg.input == "global" else 1
        hidden = 512 if cfg.input == "global" else 128
        self.f = (torch.nn.Linear(d, n_out) if cfg.model == "linear" else
                  torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_out)))
        self.null = torch.nn.Parameter(torch.zeros(1))

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.cfg.norm == "ln":
            x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        return (x - self.mu) / self.sd

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Parameter-free part: norm, standardise, pool. [B, g, g, D] -> [B, g, g, D] or [B, D]."""
        x = self.prep(x)
        if self.cfg.input == "global":
            return x.mean((1, 2))
        if self.cfg.input == "nbhd":
            x = torch.nn.functional.avg_pool2d(x.permute(0, 3, 1, 2), 3, 1, 1, count_include_pad=False).permute(0, 2, 3, 1)
        return x

    def head(self, z: torch.Tensor):
        if self.cfg.input == "global":
            o = self.f(z.float())
            return o[:, :-1].reshape(-1, self.g, self.g), o[:, -1]
        return self.f(z.float()).squeeze(-1), self.null.expand(z.shape[0])

    def forward(self, x: torch.Tensor):
        return self.head(self.features(x))

    def loss(self, cells, null, y):
        if self.cfg.head == "sigmoid":
            return torch.nn.functional.binary_cross_entropy_with_logits(cells, y)
        logits = torch.cat([cells.flatten(1), null[:, None]], 1)
        flat = y.flatten(1)
        target = torch.cat([flat, (flat.sum(1, keepdim=True) == 0).float()], 1)
        target = target / target.sum(1, keepdim=True)            # one ball: spread over its cells, or "no ball"
        return -(target * torch.log_softmax(logits, 1)).sum(1).mean()

    @torch.no_grad()
    def maps(self, x: torch.Tensor):
        """P(ball in cell) [B, g, g] and P(a ball is visible) [B]."""
        cells, null = self(x)
        if self.cfg.head == "sigmoid":
            m = torch.sigmoid(cells)
            return m, m.flatten(1).max(1).values
        p = torch.softmax(torch.cat([cells.flatten(1), null[:, None]], 1), 1)
        return p[:, :-1].reshape(-1, self.g, self.g), 1 - p[:, -1]


def fit_probe(cfg: Config, examples: list, seed: int = 0) -> Probe:
    """Fit on EvalExample objects only (real context-half features, same-frame labels)."""
    import eval_decoder
    if not examples or not all(isinstance(e, eval_decoder.EvalExample) for e in examples):
        raise TypeError("fit_probe() takes EvalExample objects built with eval_decoder.examples_from()")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d, g = examples[0].context_features.shape[-1], examples[0].context_features.shape[1]
    x = [e.context_features for e in examples]
    y = torch.from_numpy(np.concatenate([e.context_labels for e in examples])).float()
    probe = Probe(d, g, cfg, torch.zeros(d), torch.ones(d))
    with torch.no_grad():
        steps = [(i, s) for i, e in enumerate(x) for s in range(len(e))]
        sub = torch.stack([x[i][s] for i, s in (steps[j] for j in rng.choice(len(steps), min(300, len(steps)), replace=False))])
        z = probe.prep(sub).reshape(-1, d)
        probe.mu, probe.sd = z.mean(0), z.std(0) + 1e-4
        # The parameter-free part (norm, standardise, pool) once, so each epoch only runs the head.
        feats = torch.cat([probe.features(e).half() for e in x])
        rate = float(y.mean().clamp(1e-4, 1 - 1e-4))
        last = probe.f if cfg.model == "linear" else probe.f[-1]       # start at the base rate, not 50%
        if cfg.model == "linear":
            last.weight.zero_()
        last.bias.fill_(float(np.log(rate / (1 - rate))) if cfg.head == "sigmoid" else 0.0)
    epochs, lr = (4, 3e-3) if cfg.model == "linear" else (3, 1e-3)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    for _ in range(epochs):
        for ids in torch.from_numpy(rng.permutation(len(feats))).split(64):
            opt.zero_grad()
            probe.loss(*probe.head(feats[ids]), y[ids]).backward()
            opt.step()
    return probe.eval()


def apply(probe: Probe, feats) -> tuple[np.ndarray, np.ndarray]:
    m, v = [], []
    for i in range(0, len(feats), 64):
        a, b = probe.maps(torch.from_numpy(np.asarray(feats[i:i + 64])))
        m.append(a.numpy())
        v.append(b.numpy())
    return np.concatenate(m), np.concatenate(v)


def imagined_like(imag: np.ndarray, ctx_last_step: np.ndarray, cfg: Config) -> np.ndarray:
    """The predictor outputs layer-normalised tokens. Raw-feature probes read them after mapping each
    token to the mean and spread of the same position in the last real context step."""
    if cfg.norm == "ln":
        return imag
    t = torch.from_numpy(imag).float()
    t = torch.nn.functional.layer_norm(t, t.shape[-1:])
    like = torch.from_numpy(ctx_last_step).float()
    return (t * like.std(-1, correction=0, keepdim=True) + like.mean(-1, keepdim=True)).half().numpy()


# ----------------------------------------------------------------------------------------- scoring

def auc(y, s) -> float:
    from robust_world.eval.ball import roc_auc_score_safe
    return round(float(roc_auc_score_safe(np.asarray(y).reshape(-1), np.asarray(s).reshape(-1))), 4)


def location_metrics(maps: list, vis: list, labels: list, centres: list, far: np.ndarray) -> dict:
    """Threshold-free where-is-the-ball scores over all steps of the given clips."""
    lab, mp, vs = np.concatenate(labels), np.concatenate(maps), np.concatenate(vis)
    present = lab.any((1, 2))
    cell = 512 / GRID
    errs, far_errs = [], []
    for m, cs in zip(maps, centres):
        for s, c in enumerate(cs):
            if c is None:
                continue
            y, x = np.unravel_index(m[s].argmax(), m[s].shape)
            e = float(np.hypot((x + 0.5) * cell - c[0], (y + 0.5) * cell - c[1]))
            errs.append(e)
            if far[min(int(c[1] // cell), GRID - 1), min(int(c[0] // cell), GRID - 1)]:
                far_errs.append(e)
    out = {"cell_auroc": auc(lab, mp), "visible_auroc": auc(present, vs),
           "phantom_rate": round(float((vs[~present] > 0.5).mean()), 4) if (~present).any() else None,
           "hit_rate": round(float(np.mean(np.array(errs) <= 48)), 4) if errs else None,
           "error_px": round(float(np.median(errs)), 1) if errs else None, "n_ball_steps": len(errs)}
    if lab[:, far].any():
        out["far_cell_auroc"] = auc(lab[:, far], mp[:, far])
        out["far_hit_rate"] = round(float(np.mean(np.array(far_errs) <= 48)), 4) if far_errs else None
        out["n_far_ball_steps"] = len(far_errs)
    return out


def outcome_rows(maps: list, rows: list, cfg: Config, far, near) -> tuple[list, list]:
    """Per clip readouts: native (sigmoid: max cell; softmax: probability mass) and one-ball (share)."""
    from robust_world.eval.ball import one_ball_readouts
    native, share = [], []
    for m, r in zip(maps, rows):
        if cfg.head == "sigmoid":
            f, n = (m * far).max((1, 2)), (m * near).max((1, 2))
        else:
            f, n = (m * far).sum((1, 2)), (m * near).sum((1, 2))
        base = {"clip_id": r["clip_id"], "outcome": r["outcome"]}
        native.append({**base, "far": f.round(4).tolist(), "near": n.round(4).tolist()})
        share.append({**base, **one_ball_readouts(m, far, near)})
    return native, share


def outcome_metrics(native: list, share: list) -> dict:
    from robust_world.eval.ball import three_way_metrics
    y = [r["outcome"] == "through" for r in native]
    tw, tws = three_way_metrics(native), three_way_metrics(share)
    return {"outcome_auroc": auc(y, [max(r["far"]) for r in native]),
            "outcome_auroc_one_ball": auc(y, [max(r["far"]) for r in share]),
            "p_correct_balanced": tw["p_correct_balanced"], "balanced_accuracy": tw["balanced_accuracy"],
            "p_correct_balanced_one_ball": tws["p_correct_balanced"],
            "balanced_accuracy_one_ball": tws["balanced_accuracy"]}


# ------------------------------------------------------------------------------------------- sweep

MIDDLE = ("L12", "L16", "L20", "L8", "L4")


def grid(phase: str = "all") -> list[Config]:
    """Every config, most informative first, so a partial night still answers the main questions.
    "core": everything on the last layer (the only one imagined features can be read from) and
    layer-normed linear probes on the other layers; "rest": raw linear and MLP probes there."""
    inputs, heads = ("token", "nbhd", "global"), ("sigmoid", "softmax")
    cells = [(i, n, h) for i in inputs for n in ("ln", "raw") for h in heads]
    core = [Config("last", "linear", *c) for c in cells]
    core += [Config(layer, "linear", i, "ln", h) for layer in MIDDLE for i in inputs for h in heads]
    core += [Config("last", "mlp", *c) for c in cells]
    rest = [Config(layer, "linear", i, "raw", h) for layer in MIDDLE for i in inputs for h in heads]
    rest += [Config(layer, "mlp", i, "ln", h) for layer in MIDDLE for i in ("token", "nbhd") for h in heads]
    return {"core": core, "rest": rest, "all": core + rest}[phase]


def run_config(cfg: Config, clips, rows, geo, ctx_feats: list, folds, out_dir: Path) -> dict:
    import eval_decoder
    far, near = geo["far"], geo["near"]
    n = len(clips)
    res = {k: [None] * n for k in ("ctx_m", "ctx_v", "tgt_m", "tgt_v", "img_m", "img_v")}
    t0 = time.perf_counter()
    for tr, te in folds:
        examples = [eval_decoder.examples_from({"context": torch.from_numpy(ctx_feats[i])}, rows[i]["labels"])
                    for i in tr]
        probe = fit_probe(cfg, examples)
        for i in te:
            cid = clips[i]["clip_id"]
            res["ctx_m"][i], res["ctx_v"][i] = apply(probe, ctx_feats[i])
            res["tgt_m"][i], res["tgt_v"][i] = apply(probe, load(cid, f"tgt_{cfg.layer}"))
            if cfg.layer == "last":
                res["img_m"][i], res["img_v"][i] = apply(probe, imagined_like(load(cid, "imag"), ctx_feats[i][-1], cfg))
    k = rows[0]["ctx_steps"]
    ctx_lab = [r["labels"][:k] for r in rows]
    tgt_lab = [r["labels"][k:] for r in rows]
    ctx_c = [r["centres"][:k] for r in rows]
    tgt_c = [r["centres"][k:] for r in rows]
    out = {**cfg.as_dict(), "name": cfg.name, "n_clips": n, "seconds": round(time.perf_counter() - t0, 1),
           "context": location_metrics(res["ctx_m"], res["ctx_v"], ctx_lab, ctx_c, far)}
    out["target"] = location_metrics(res["tgt_m"], res["tgt_v"], tgt_lab, tgt_c, far)
    out["target"].update(outcome_metrics(*outcome_rows(res["tgt_m"], rows, cfg, far, near)))
    arrays = {"tgt_maps": np.stack(res["tgt_m"]).astype(np.float16), "tgt_vis": np.stack(res["tgt_v"])}
    if cfg.layer == "last":
        out["imagined"] = location_metrics(res["img_m"], res["img_v"], tgt_lab, tgt_c, far)
        out["imagined"].update(outcome_metrics(*outcome_rows(res["img_m"], rows, cfg, far, near)))
        arrays |= {"img_maps": np.stack(res["img_m"]).astype(np.float16), "img_vis": np.stack(res["img_v"])}
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "maps" / f"{cfg.name}.npz", **arrays)
    (out_dir / "sweep" / f"{cfg.name}.json").write_text(json.dumps(out, indent=1))
    return out


def cmd_sweep(args) -> None:
    torch.set_num_threads(args.threads)
    clips = encoded_clips(args)
    rows, geo = truth(clips)
    folds = cv_folds(clips, args.folds)
    (OUT / "sweep").mkdir(parents=True, exist_ok=True)
    todo = [c for c in grid(args.phase) if args.only is None or args.only in c.name]
    todo = [c for c in todo if not (OUT / "sweep" / f"{c.name}.json").exists()]
    log(f"sweep: {len(clips)} clips, {len(todo)} configs to do")
    loaded, feats = None, None
    for j, cfg in enumerate(todo):
        if cfg.layer != loaded:
            feats = None
            feats = [load(c["clip_id"], f"ctx_{cfg.layer}") for c in clips]
            log(f"sweep: loaded layer {cfg.layer}")
            loaded = cfg.layer
        try:
            r = run_config(cfg, clips, rows, geo, feats, folds, OUT)
        except Exception as e:
            log(f"sweep {cfg.name} FAILED: {e!r}")
            continue
        t = r["target"]
        log(f"sweep {j + 1}/{len(todo)} {cfg.name} ({r['seconds']:.0f}s): ctx cell {r['context']['cell_auroc']}, "
            f"tgt cell {t['cell_auroc']}, far cell {t.get('far_cell_auroc')}, hit {t['hit_rate']}, "
            f"outcome {t['outcome_auroc']}, P(correct) {t['p_correct_balanced']}"
            + (f" | imagined outcome {r['imagined']['outcome_auroc']}" if "imagined" in r else ""))
        if not args.no_report:
            cmd_report(args)


# ---------------------------------------------------------------------------------------- baseline

def tapnext_rules():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tapnext_rule_predict", REPO / "models" / "tapnext_rule" / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_baseline(args) -> None:
    """The TAPNext rule baseline with the colour tracker's context centres in place of TAPNext
    (no GPU needed). "blockade" fits the blockade range on each fold's training OUTCOMES, so it is
    a reference model, not an evaluation tool (CLAUDE.md rule 7)."""
    from robust_world.eval.ball import readouts, summary_metrics, three_way_metrics
    clips = included_clips(args.manifest)[:args.limit]
    rows, geo = truth(clips)
    tr = tapnext_rules()
    scene, far, near = geo["scene"], geo["far"], geo["near"]
    k = rows[0]["ctx_steps"]
    n_ctx, n_target = 2 * k, clips[0]["n_frames"] - 2 * k
    recs = []
    for r in rows:
        path = tr.continue_path({"centres": r["ctx_centres_px"]}, scene, n_target)
        recs.append({**r, "path": path, "crossing_y": path["crossing_y"]})
    folds = cv_folds(clips, args.folds)
    result = {"name": "tracker + straight line + plank/blockade rules (blockade fitted on outcomes)",
              "crossing_y": {r["clip_id"]: r["crossing_y"] for r in recs}}
    for variant in ("continue", "blockade"):
        out_rows, maps, blockades = [None] * len(recs), [None] * len(recs), []
        for tr_idx, te_idx in folds:
            b = tr.fit_blockade([recs[i] for i in tr_idx]) if variant == "blockade" else None
            blockades.append([round(v, 1) for v in b] if b else None)
            for i in te_idx:
                fr = tr.apply_rules(recs[i]["path"], recs[i]["radius"], scene, b)
                maps[i] = tr.cell_maps(fr, n_ctx)
                out_rows[i] = {"clip_id": recs[i]["clip_id"], "outcome": recs[i]["outcome"], **readouts(maps[i], far, near)}
        m = summary_metrics(out_rows, [r["labels"][k:] for r in recs], maps)
        m.update({kk: v for kk, v in three_way_metrics(out_rows).items() if kk in ("p_correct_balanced", "balanced_accuracy")})
        m["location"] = location_metrics(maps, [x.max((1, 2)) for x in maps], [r["labels"][k:] for r in recs],
                                         [r["centres"][k:] for r in recs], far)
        m["blockade_per_fold"] = blockades
        result[variant] = m
        log(f"baseline {variant}: outcome AUROC {m['outcome_auroc']}, P(correct) {m['p_correct_balanced']}, "
            f"balanced acc {m['balanced_accuracy']}, blockade {blockades}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline.json").write_text(json.dumps(result, indent=1))


# ---------------------------------------------------------------------------------------- blockade

def sweep_results() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((OUT / "sweep").glob("*.json"))]


def best(results: list[dict], part: str, key: str, where=lambda r: True) -> dict | None:
    ok = [r for r in results if part in r and r[part].get(key) is not None and where(r)]
    return max(ok, key=lambda r: r[part][key]) if ok else None


def cell_centres(poly: np.ndarray) -> np.ndarray:
    import cv2
    c = 512 / GRID
    return np.array([[cv2.pointPolygonTest(poly, ((x + 0.5) * c, (y + 0.5) * c), False) >= 0
                      for x in range(GRID)] for y in range(GRID)])


def cmd_blockade(args) -> None:
    """Label-free looks at the blockade. Nothing here is fitted; outcomes and the outcome-fitted
    blockade range from the baseline are only used to group, colour and score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clips = encoded_clips(args)
    rows, geo = truth(clips)
    far, scene = geo["far"], geo["scene"]
    poly = np.array(scene["occluder_polygon"], np.float32)
    plank = cell_centres(poly)
    k = rows[0]["ctx_steps"]
    base = json.loads((OUT / "baseline.json").read_text())
    ranges = [b for b in base["blockade"]["blockade_per_fold"] if b]
    lo, hi = (float(np.median([b[0] for b in ranges])), float(np.median([b[1] for b in ranges]))) if ranges else (None, None)
    cross = np.array([base["crossing_y"].get(r["clip_id"]) or np.nan for r in rows], float)
    outc = np.array([r["outcome"] for r in rows])
    colours = {"through": "#2a9d8f", "hidden": "#e63946", "bounce": "#f4a261"}
    res = {"blockade_range_y": [lo, hi]}
    results = sweep_results()

    # 1. P(blocked) against where the straight path crosses the plank: real frames vs imagined.
    real = best(results, "target", "outcome_auroc")
    imag = best(results, "imagined", "outcome_auroc")
    panels = [("truth", (outc != "through").astype(float), None)]
    for label, cfg, part, key in (("real target frames", real, "target", "tgt"),
                                  ("imagined (pretrained predictor)", imag, "imagined", "img")):
        if cfg is None:
            continue
        z = np.load(OUT / "maps" / f"{cfg['name']}.npz")
        m = z[f"{key}_maps"].astype(np.float32)
        farscore = (m * far).max((2, 3)) if cfg["head"] == "sigmoid" else (m * far).sum((2, 3))
        pb = 1 - farscore.max(1)
        panels.append((f"{label}\n{cfg['name']}", pb, cfg))
        inside = ~np.isnan(cross) & (cross >= (lo or 0)) & (cross <= (hi or 0))
        res[f"p_blocked_{key}"] = {
            "config": cfg["name"], "auroc_blocked": auc(outc != "through", pb),
            "mean_inside_range": round(float(pb[inside].mean()), 3) if inside.any() else None,
            "mean_outside_range": round(float(pb[~inside & ~np.isnan(cross)].mean()), 3)}
        if key == "tgt":           # where the real-frame probe last sees the ball
            vis = z["tgt_vis"]
            lost = []
            for i in range(len(rows)):
                seen = np.where(vis[i] > 0.5)[0]
                if len(seen):
                    y, x = np.unravel_index(m[i, seen[-1]].argmax(), (GRID, GRID))
                    lost.append(((x + 0.5) * 32, (y + 0.5) * 32))
                else:
                    lost.append(None)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
    for ax, (title, pb, _) in zip(np.atleast_1d(axes), panels):
        for o, col in colours.items():
            sel = outc == o
            ax.scatter(cross[sel], pb[sel], s=14, c=col, label=o, alpha=0.8)
        if lo is not None:
            ax.axvspan(lo, hi, color="grey", alpha=0.2, label="blockade range (fitted on outcomes)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("y where the straight path crosses the plank (px)")
    np.atleast_1d(axes)[0].set_ylabel("P(blocked)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "blocked_vs_crossing.png", dpi=110)
    plt.close(fig)

    # 2. Where the ball is last seen in the target half: tracker truth vs the real-frame probe.
    if real is not None:
        bg = read_video(REPO / clips[0]["path"])[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        truth_lost = []
        for r in rows:
            cs = [c for c in r["centres"][k:] if c is not None]
            truth_lost.append(cs[-1] if cs else None)
        for ax, pts, title in ((axes[0], truth_lost, "tracker: last seen in target half"),
                               (axes[1], lost, f"real-frame probe: last seen\n{real['name']}")):
            ax.imshow(bg)
            ax.plot(*np.vstack([poly, poly[:1]]).T, c="w", lw=1)
            for o, col in colours.items():
                xy = np.array([p for p, oo in zip(pts, outc) if p is not None and oo == o])
                if len(xy):
                    ax.scatter(xy[:, 0], xy[:, 1], s=12, c=col, label=o, alpha=0.8)
            if lo is not None:
                ax.axhspan(lo, hi, color="w", alpha=0.15)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "where_ball_last_seen.png", dpi=110)
        plt.close(fig)

    # 3. Is the blockade visible in static context features? Per-cell distinctiveness: 1 - cosine
    # between a cell's mean (ball-free) token and the mean of its 8 neighbours, per layer.
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(3.2 * len(LAYERS), 3.4))
    rows_in = np.zeros(GRID, bool)
    if lo is not None:
        rows_in[int(lo // 32):int(hi // 32) + 1] = True
    res["distinctiveness"] = {}
    for ax, layer in zip(axes, LAYERS):
        s, n = np.zeros((GRID, GRID, 1024)), np.zeros((GRID, GRID, 1))
        for r, c in zip(rows, clips):
            x = torch.from_numpy(load(c["clip_id"], f"ctx_{layer}")).float()
            x = torch.nn.functional.layer_norm(x, x.shape[-1:]).numpy()
            busy = torch.nn.functional.max_pool2d(torch.from_numpy(r["labels"][:k]).float()[:, None], 3, 1, 1)[:, 0].numpy() > 0
            free = ~busy[..., None]
            s += (x * free).sum(0)
            n += free.sum(0)
        mean = s / np.maximum(n, 1)
        unit = mean / np.linalg.norm(mean, axis=-1, keepdims=True)
        dist = np.zeros((GRID, GRID))
        for y in range(GRID):
            for x in range(GRID):
                nb = [mean[yy, xx] for yy in range(y - 1, y + 2) for xx in range(x - 1, x + 2)
                      if (yy, xx) != (y, x) and 0 <= yy < GRID and 0 <= xx < GRID]
                v = np.mean(nb, 0)
                dist[y, x] = 1 - float(unit[y, x] @ (v / np.linalg.norm(v)))
        on = plank & rows_in[:, None]
        off = plank & ~rows_in[:, None]
        res["distinctiveness"][layer] = {
            "plank_in_range": round(float(dist[on].mean()), 4) if on.any() else None,
            "plank_out_of_range": round(float(dist[off].mean()), 4) if off.any() else None,
            "plank_row_profile": [round(float(dist[y][plank[y]].mean()), 4) if plank[y].any() else None for y in range(GRID)]}
        ax.imshow(dist, cmap="magma")
        ax.plot(*(np.vstack([poly, poly[:1]]) / 32 - 0.5).T, c="c", lw=0.8)
        if lo is not None:
            ax.axhline(lo / 32 - 0.5, c="w", ls="--", lw=0.8)
            ax.axhline(hi / 32 - 0.5, c="w", ls="--", lw=0.8)
        ax.set_title(f"{layer}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Static context features: how different each cell is from its neighbours "
                 "(dashed: blockade range fitted on outcomes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "static_distinctiveness.png", dpi=110)
    plt.close(fig)

    # 4. Surprise (pretrained predictor, label-free): per-cell L1 between imagined and real target.
    sur = []
    for c in clips:
        a = torch.from_numpy(load(c["clip_id"], "imag")).float()
        b = torch.from_numpy(load(c["clip_id"], "tgt_last")).float()
        a, b = (torch.nn.functional.layer_norm(t, t.shape[-1:]) for t in (a, b))
        sur.append((a - b).abs().mean(-1).mean(0).numpy())
    sur = np.stack(sur)
    means = {o: sur[outc == o].mean(0) for o in colours if (outc == o).any()}
    fig, axes = plt.subplots(1, len(means) + 1, figsize=(3.4 * (len(means) + 1), 3.4))
    vmin, vmax = min(m.min() for m in means.values()), max(m.max() for m in means.values())
    for ax, (o, m) in zip(axes, means.items()):
        ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"surprise, {o}", fontsize=9)
    if "hidden" in means and "through" in means:
        d = means["hidden"] - means["through"]
        lim = np.abs(d).max()
        axes[-1].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[-1].set_title("hidden - through", fontsize=9)
        res["surprise_plank_row_hidden_minus_through"] = [
            round(float(d[y][plank[y]].mean()), 4) if plank[y].any() else None for y in range(GRID)]
    for ax in axes:
        ax.plot(*(np.vstack([poly, poly[:1]]) / 32 - 0.5).T, c="w", lw=0.8)
        if lo is not None:
            ax.axhline(lo / 32 - 0.5, c="w", ls="--", lw=0.8)
            ax.axhline(hi / 32 - 0.5, c="w", ls="--", lw=0.8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "surprise_by_outcome.png", dpi=110)
    plt.close(fig)
    (OUT / "blockade.json").write_text(json.dumps(res, indent=1))
    log(f"blockade: {json.dumps({k: v for k, v in res.items() if k.startswith('p_blocked')})}")


# ------------------------------------------------------------------------------------------ report

KEYS = [("context", "cell_auroc", "ctx cell AUROC"), ("target", "cell_auroc", "tgt cell AUROC"),
        ("target", "far_cell_auroc", "far cell AUROC"), ("target", "hit_rate", "tgt hit"),
        ("target", "far_hit_rate", "far hit"), ("target", "error_px", "err px"),
        ("target", "outcome_auroc", "outcome AUROC"), ("target", "p_correct_balanced", "P(correct)"),
        ("imagined", "far_cell_auroc", "imag far cell"), ("imagined", "outcome_auroc", "imag outcome"),
        ("imagined", "p_correct_balanced_one_ball", "imag P(correct, 1 ball)")]
FACTORS = {"model": ("linear", "mlp"), "input": ("token", "nbhd", "global"), "norm": ("raw", "ln"),
           "head": ("sigmoid", "softmax")}


def paired(results: list[dict], factor: str, part: str, key: str) -> dict:
    """Mean change in a metric when only `factor` changes (from its first level to each other level)."""
    levels = FACTORS[factor]
    index = {tuple(r[f] for f in Config.FIELDS): r for r in results}
    out = {}
    for lv in levels[1:]:
        diffs = []
        for r in results:
            if r[factor] != levels[0]:
                continue
            other = index.get(tuple(lv if f == factor else r[f] for f in Config.FIELDS))
            a, b = r.get(part, {}).get(key), (other or {}).get(part, {}).get(key)
            if a is not None and b is not None:
                diffs.append(b - a)
        if diffs:
            out[f"{lv} - {levels[0]}"] = {"mean": round(float(np.mean(diffs)), 4), "n": len(diffs),
                                          "wins": int(np.sum(np.array(diffs) > 0))}
    return out


def cmd_report(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = sweep_results()
    if not results:
        return
    fmt = lambda v: "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["# V-JEPA 2 encoder probing", "",
             f"{results[0]['n_clips']} right-segment clips, 5-fold CV (`cv_folds`, seed 0). Every probe is fitted on real "
             "context-half features (encoded alone) with same-frame ball cells (`eval_decoder.examples_from`), "
             "then scored on held-out clips: context frames, real target frames (far-side cells never had a "
             "ball in training) and, for the last layer, the pretrained predictor's imagined target.", ""]
    base_path = OUT / "baseline.json"
    if base_path.exists():
        b = json.loads(base_path.read_text())
        lines += ["## Reference: tracker + rules", "",
                  "| variant | outcome AUROC | P(correct) | balanced acc | tgt cell AUROC | far cell AUROC | hit |",
                  "|---|---|---|---|---|---|---|"]
        for v in ("continue", "blockade"):
            m = b[v]
            lines.append(f"| {v}{' (blockade fitted on outcomes)' if v == 'blockade' else ''} | {fmt(m['outcome_auroc'])} | "
                         f"{fmt(m.get('p_correct_balanced'))} | {fmt(m.get('balanced_accuracy'))} | "
                         f"{fmt(m['future_cell_auroc'])} | {fmt(m['location'].get('far_cell_auroc'))} | {fmt(m['location'].get('hit_rate'))} |")
        lines += ["", "TAPNext + coded blockade on the earlier 88-clip set: outcome AUROC 0.98 "
                  "(`docs/results/ball_comparison.md`).", ""]
    lines += ["## Factor effects (paired: configs identical except for one factor)", "",
              "| factor | change | far cell AUROC | outcome AUROC (real) | tgt hit | imag outcome AUROC |", "|---|---|---|---|---|---|"]
    for f in FACTORS:
        cols = [paired(results, f, "target", "far_cell_auroc"), paired(results, f, "target", "outcome_auroc"),
                paired(results, f, "target", "hit_rate"), paired(results, f, "imagined", "outcome_auroc")]
        for ch in cols[0]:
            cell = lambda d: f"{d[ch]['mean']:+.3f} ({d[ch]['wins']}/{d[ch]['n']})" if ch in d else ""  # noqa: E731
            lines.append(f"| {f} | {ch} | " + " | ".join(cell(d) for d in cols) + " |")
    lines += ["", "Each cell: mean change, and how many pairs improved out of all pairs.", "",
              "## All configs", "", "| config | " + " | ".join(k[2] for k in KEYS) + " |",
              "|---|" + "---|" * len(KEYS)]
    for r in sorted(results, key=lambda r: -(r["target"].get("far_cell_auroc") or 0)):
        lines.append(f"| {r['name']} | " + " | ".join(fmt(r.get(p, {}).get(k)) for p, k, _ in KEYS) + " |")
    bl = OUT / "blockade.json"
    if bl.exists():
        lines += ["", "## Blockade (label-free)", "", "```json", bl.read_text().strip(), "```"]
    figs = sorted(p.name for p in OUT.glob("*.png"))
    lines += ["", "## Figures", ""] + [f"![{f}]({f})" for f in figs]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))

    # Metrics across layers for the linear probes.
    layers = [l for l in LAYERS if any(r["layer"] == l for r in results)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (part, key, title) in zip(axes, [("context", "cell_auroc", "context cell AUROC"),
                                             ("target", "far_cell_auroc", "far-side cell AUROC (real target)"),
                                             ("target", "outcome_auroc", "outcome AUROC (real target)")]):
        for i in FACTORS["input"]:
            for n in FACTORS["norm"]:
                for h in FACTORS["head"]:
                    ys = [next((r[part].get(key) for r in results if (r["layer"], r["model"], r["input"], r["norm"], r["head"])
                                == (l, "linear", i, n, h)), None) for l in layers]
                    if any(y is not None for y in ys):
                        ax.plot(layers, [np.nan if y is None else y for y in ys], marker="o",
                                ls="-" if n == "ln" else "--", label=f"{i} {n} {h}")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=6)
    fig.suptitle("Linear probes by encoder layer (solid: layer-normed, dashed: raw)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "layers.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------- publish

def cmd_publish(args) -> None:
    """Copy scores and figures to results/<date>_<time>_vjepa-probing/ (the same folder on every call
    of one night) with scripts/save_results.py, then commit only results/ and push."""
    import subprocess
    sys.path.insert(0, str(REPO / "scripts"))
    import save_results
    marker = OUT / "results_folder.txt"
    into = marker.read_text().strip() if marker.exists() else None
    dest = save_results.save("vjepa-probing", "V-JEPA 2 encoder probing (models/vjepa2/probing.py), CPU overnight",
                             into=into or save_results.new_id(REPO / "results", "vjepa-probing"))
    marker.write_text(dest.name)
    if args.no_push:
        return
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)  # noqa: E731
    git("add", "results")
    r = git("commit", "-q", "-m", f"Probing results: {dest.name}", "--", "results")
    if r.returncode == 0:
        for attempt in range(4):
            r = git("push", "origin", "HEAD")
            if r.returncode == 0:
                break
            time.sleep(2 ** (attempt + 1))
    log(f"publish: {dest.name} " + ("pushed" if r.returncode == 0 else f"not pushed: {(r.stderr or r.stdout).strip()[-200:]}"))


def cmd_all(args) -> None:
    cmd_encode(args)
    cmd_baseline(args)
    for phase in ("core", "rest"):
        args.phase = phase
        for stage in (cmd_sweep, cmd_blockade, cmd_report, cmd_publish):
            try:
                stage(args)
            except Exception as e:
                log(f"{stage.__name__} FAILED: {e!r}")


# ------------------------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    global OUT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["encode", "baseline", "sweep", "blockade", "report", "publish", "all"])
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--limit", type=int, default=None, help="first N clips only (quick checks)")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--folds", type=int, default=5, help="CV folds (5 for real runs; fewer only for quick checks)")
    p.add_argument("--phase", default="all", choices=["core", "rest", "all"], help="sweep: which part of the grid")
    p.add_argument("--no-push", action="store_true", help="publish: save under results/ but don't commit or push")
    p.add_argument("--only", default=None, help="sweep: only configs whose name contains this")
    p.add_argument("--no-report", action="store_true", help="sweep: don't rewrite the report after each config")
    p.add_argument("--out", type=Path, default=OUT, help="where scores and figures go (quick checks elsewhere)")
    args = p.parse_args(argv)
    OUT = args.out
    keep_awake(True)
    try:
        globals()[f"cmd_{args.stage}"](args)
    finally:
        keep_awake(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

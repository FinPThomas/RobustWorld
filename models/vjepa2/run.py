"""V-JEPA 2: does the model expect the ball to come out from under the plank?

V-JEPA 2 predicts in representation space rather than pixels, so it is scored rather
than rendered:

1. Surprise. The context half is encoded on its own (no peeking at the future), the
   predictor imagines the target half's features, and those are compared with the
   encoder's features of the real full clip (L1, per target time step). This is the
   V-JEPA "intuitive physics" protocol. A copy-the-last-context-step baseline gives the
   scale.
2. Predicted outcome. A linear probe learns through-vs-hidden from features of *real*
   target halves; applied to the *imagined* target halves, it reads off what V-JEPA
   predicts happens. Cross-validated, so a clip is never scored by a probe that saw it.
   A probe on context features alone shows how much the context gives away.

Frames are resized (not cropped) to 256x256 so the ball's exit at the frame edge is kept.

    python models/vjepa2/run.py                       # all included clips in the manifest
    python models/vjepa2/run.py --sample data/eval/sample5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.io import read_video  # noqa: E402

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"     # or facebook/vjepa2-vitg-fpc64-256 (1B)
SIZE = 256
GRID = 4                                        # spatial pooling grid for probe features


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clips(manifest: Path | None, sample: Path | None) -> list[dict]:
    if sample:
        s = json.loads((sample / "sample.json").read_text())
        return [{**c, "video": sample / "clips" / f"{c['clip_id']}.mp4"} for c in s["clips"]]
    records = [json.loads(line) for line in manifest.open()]
    return [{**r, "video": REPO / r["path"]} for r in records if r.get("include")]


def to_pixels(frames: list[np.ndarray], mean, std, device) -> torch.Tensor:
    x = np.stack([cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32) / 255
    x = (x - mean) / std
    return torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).to(device)   # [1, T, C, H, W]


def pooled(tokens: torch.Tensor, steps: int, grid_hw: int) -> dict[str, np.ndarray]:
    """[1, steps*grid_hw^2, D] -> mean-pooled and GRIDxGRID spatially pooled features (time-averaged)."""
    d = tokens.shape[-1]
    t = tokens.reshape(steps, grid_hw, grid_hw, d).float()
    k = grid_hw // GRID
    grid = t.mean(0).reshape(GRID, k, GRID, k, d).mean((1, 3)).reshape(-1)
    return {"mean": t.mean((0, 1, 2)).cpu().numpy(), "grid": grid.cpu().numpy()}


@torch.no_grad()
def score_clip(model, frames: list[np.ndarray], n_ctx: int, mean, std, device) -> dict:
    cfg = model.config
    tubelet, patch = cfg.tubelet_size, cfg.patch_size
    grid_hw = SIZE // patch
    per_step = grid_hw * grid_hw
    ctx_steps, all_steps = n_ctx // tubelet, len(frames) // tubelet
    n_ctx_tok, n_all_tok = ctx_steps * per_step, all_steps * per_step

    full = model(to_pixels(frames, mean, std, device), skip_predictor=True).last_hidden_state
    ctx = model(to_pixels(frames[:n_ctx], mean, std, device), skip_predictor=True).last_hidden_state
    pred = model.predictor(
        encoder_hidden_states=ctx,
        context_mask=[torch.arange(n_ctx_tok, device=device).unsqueeze(0)],
        target_mask=[torch.arange(n_ctx_tok, n_all_tok, device=device).unsqueeze(0)],
    ).last_hidden_state
    target = full[:, n_ctx_tok:]
    tgt_steps = all_steps - ctx_steps

    def per_step_l1(a, b):
        return (a - b).abs().mean(-1).reshape(tgt_steps, per_step).mean(-1).float().cpu().numpy()

    last_ctx = ctx[:, -per_step:].repeat(1, tgt_steps, 1)
    return {
        "surprise": per_step_l1(pred, target),
        "copy_baseline": per_step_l1(last_ctx, target),
        "features": {"context": pooled(ctx, ctx_steps, grid_hw),
                     "target": pooled(target, tgt_steps, grid_hw),
                     "predicted": pooled(pred, tgt_steps, grid_hw)},
    }


def probe(results: list[dict], kind: str, folds: int, seed: int) -> dict:
    """Cross-validated through-vs-hidden probes. Trained on real target (or context) features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([r["outcome"] == "hidden" for r in results], dtype=int)
    feats = {name: np.stack([r["features"][name][kind] for r in results])
             for name in ("context", "target", "predicted")}
    n_splits = min(folds, int(y.sum()), int(len(y) - y.sum()))
    p_target, p_imagined, p_context = (np.zeros(len(y)) for _ in range(3))
    for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(y, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        clf.fit(feats["target"][train], y[train])
        p_target[test] = clf.predict_proba(feats["target"][test])[:, 1]
        p_imagined[test] = clf.predict_proba(feats["predicted"][test])[:, 1]
        ctx_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        ctx_clf.fit(feats["context"][train], y[train])
        p_context[test] = ctx_clf.predict_proba(feats["context"][test])[:, 1]

    def acc(p):
        return float(((p > 0.5) == y).mean())
    return {"features": kind, "folds": n_splits, "chance": float(max(y.mean(), 1 - y.mean())),
            "real_target_acc": acc(p_target), "imagined_target_acc": acc(p_imagined),
            "context_only_acc": acc(p_context), "p_hidden_imagined": p_imagined.tolist(),
            "p_hidden_context": p_context.tolist()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest.jsonl")
    p.add_argument("--sample", type=Path, help="score an eval sample instead of the whole manifest")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from transformers import AutoVideoProcessor, VJEPA2Model

    device = pick_device()
    clips = load_clips(args.manifest, args.sample)
    model = VJEPA2Model.from_pretrained(args.model_id).to(device).eval()
    proc = AutoVideoProcessor.from_pretrained(args.model_id)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    print(f"V-JEPA 2 ({args.model_id}) on {device}; {len(clips)} clips")

    results = []
    t0 = time.time()
    for i, c in enumerate(clips):
        frames = read_video(c["video"])
        n_ctx = c["context_frames"][1] + 1
        r = score_clip(model, frames, n_ctx, mean, std, device)
        results.append({"clip_id": c["clip_id"], "outcome": c["outcome"], **r})
        print(f"\r  {i + 1}/{len(clips)} clips ({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"model_id": args.model_id, "n_clips": len(results), "seconds": round(time.time() - t0, 1)}
    for outcome in ("through", "hidden"):
        group = [r for r in results if r["outcome"] == outcome]
        if group:
            summary[f"surprise_{outcome}"] = np.mean([r["surprise"] for r in group], 0).round(4).tolist()
            summary[f"copy_baseline_{outcome}"] = np.mean([r["copy_baseline"] for r in group], 0).round(4).tolist()
    outcomes = {r["outcome"] for r in results}
    if {"through", "hidden"} <= outcomes and len(results) >= 10:
        summary["probe"] = {kind: probe(results, kind, args.folds, args.seed) for kind in ("grid", "mean")}

    per_clip = []
    for i, r in enumerate(results):
        row = {"clip_id": r["clip_id"], "outcome": r["outcome"],
               "surprise": round(float(r["surprise"].mean()), 4),
               "surprise_per_step": r["surprise"].round(4).tolist(),
               "copy_baseline": round(float(r["copy_baseline"].mean()), 4)}
        if "probe" in summary:
            row["p_hidden_imagined"] = round(summary["probe"]["grid"]["p_hidden_imagined"][i], 3)
        per_clip.append(row)
    (args.out / "per_clip.json").write_text(json.dumps(per_clip, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(args.out / "features.npz", **{
        f"{r['clip_id']}/{name}/{kind}": v
        for r in results for name, fk in r["features"].items() for kind, v in fk.items()})

    print(f"  surprise (mean L1, imagined vs real target):")
    for outcome in ("through", "hidden"):
        if f"surprise_{outcome}" in summary:
            print(f"    {outcome:8s} {np.mean(summary[f'surprise_{outcome}']):.4f}   "
                  f"(copy-last-context baseline {np.mean(summary[f'copy_baseline_{outcome}']):.4f})")
    if "probe" in summary:
        for kind, pr in summary["probe"].items():
            print(f"  probe [{kind}] chance {pr['chance']:.2f}: real target {pr['real_target_acc']:.2f}, "
                  f"V-JEPA imagined target {pr['imagined_target_acc']:.2f}, context only {pr['context_only_acc']:.2f}")
    print(f"-> {args.out}/ (summary.json, per_clip.json, features.npz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

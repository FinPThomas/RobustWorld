"""V-JEPA 2 surprise: how far is the imagined future from the real one? (label-free)

The context half is encoded on its own (no peeking at the future), the predictor imagines
the target half's features, and those are compared with the encoder's features of the real
full clip (L1, per target time step). This is the V-JEPA "intuitive physics" protocol and
involves no trained readout. A copy-the-last-context-step baseline gives the scale.

Where V-JEPA puts the ball is read by ball_probe_cv.py with the frozen evaluation decoder
(eval_decoder.py). No readout here is trained on outcomes or on model predictions: learning
the blockade is for post-training the world model, never for the evaluation.

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
GRID = 4                                        # spatial pooling grid for the saved feature summaries


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest_right.jsonl")
    p.add_argument("--sample", type=Path, help="score an eval sample instead of the whole manifest")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2")
    p.add_argument("--model-id", default=MODEL_ID)
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

    per_clip = []
    for i, r in enumerate(results):
        row = {"clip_id": r["clip_id"], "outcome": r["outcome"],
               "surprise": round(float(r["surprise"].mean()), 4),
               "surprise_per_step": r["surprise"].round(4).tolist(),
               "copy_baseline": round(float(r["copy_baseline"].mean()), 4)}
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
    print(f"-> {args.out}/ (summary.json, per_clip.json, features.npz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

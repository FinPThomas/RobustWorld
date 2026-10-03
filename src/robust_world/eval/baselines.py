"""Trivial predictors: lower bounds for real models, and a way to test the grid without a GPU."""

from __future__ import annotations

import time
from pathlib import Path

from .io import load_sample, read_clip, write_prediction


def hold_last_frame(sample_dir: Path, out_dir: Path) -> None:
    """Predict that nothing moves: repeat the last context frame."""
    sample = load_sample(sample_dir)
    n_target = sample["target_frames"][1] - sample["target_frames"][0] + 1
    for clip in sample["clips"]:
        t0 = time.time()
        context, _ = read_clip(sample, clip["clip_id"])
        write_prediction(out_dir, clip["clip_id"], [context[-1]] * n_target, context,
                         list(range(len(context))),
                         {"model": "hold", "label": "Hold last frame", "seconds": round(time.time() - t0, 2)})
    print(f"hold-last-frame baseline -> {out_dir}")

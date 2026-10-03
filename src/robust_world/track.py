"""Stage: per-frame ball position and hand-in-shot signal -> track.csv + backgrounds.npz."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .media import iter_frames, probe
from .paths import rel
from .scene import background_for, detect_ball, detect_scene_breaks, foreground_mask, segment_backgrounds

FIELDS = ["frame", "t", "visible", "x", "y", "area", "radius", "occ_dist", "n_blobs", "fg_area"]


def run(video: Path, scene: dict, track_csv: Path, bg_npz: Path) -> None:
    info = probe(video)
    fps = info["fps"]
    total = int(round(info["duration"] * fps))

    breaks = scene["background_breaks_s"]
    if breaks == "auto":
        breaks = detect_scene_breaks(video)
        print(f"  scene changes detected at: {', '.join(f'{b:.0f}s' for b in breaks) or 'none'}")
    starts = [0] + [int(round(t * fps)) for t in breaks]
    bgs = segment_backgrounds(video, starts, total)
    np.savez_compressed(bg_npz, backgrounds=bgs, segment_starts=np.array(starts), fps=fps)
    starts = np.array(starts)

    seen = 0
    with track_csv.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=FIELDS)
        out.writeheader()
        for idx, rgb in enumerate(iter_frames(video)):
            row = detect_ball(rgb, scene)
            row["fg_area"] = int(foreground_mask(rgb, background_for(idx, bgs, starts), row, scene).sum())
            seen += row["visible"]
            out.writerow({"frame": idx, "t": round(idx / fps, 3), **row})
            if idx % 1000 == 0:
                print(f"\r  {idx}/{total} frames", end="", flush=True)
    print(f"\r  {idx + 1} frames @ {fps:g} fps, ball visible in {seen / (idx + 1):.1%}")
    print(f"  -> {rel(track_csv)}, {rel(bg_npz)}")


def load(track_csv: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(track_csv.open()):
        r = {k: float(v) if v else None for k, v in r.items()}
        r["frame"] = int(r["frame"])
        r["visible"] = int(r["visible"])
        rows.append(r)
    return rows

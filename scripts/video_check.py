"""Look at a new video before its scene file is set: frames where the ball is most likely in view, with a grid.

Picks, from 2 frames a second of data/interim/<name>/<name>_512.mp4, the frames with the most strongly
coloured pixels in the hue range given (default orange: H 0-22 of OpenCV's 0-179, S >= 140, V >= 90), and saves
them as one gridded sheet plus the hue/saturation/value spread of those pixels, so the occluder outline and
the ball's colour range can be set from it.

    python scripts/video_check.py other --out results/other_ball_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from robust_world.media import iter_frames  # noqa: E402


def grid(im: np.ndarray) -> np.ndarray:
    im = im.copy()
    for v in range(0, im.shape[0], 32):
        col = (255, 255, 255) if v % 64 == 0 else (110, 110, 110)
        cv2.line(im, (v, 0), (v, im.shape[0] - 1), col, 1)
        cv2.line(im, (0, v), (im.shape[1] - 1, v), col, 1)
        if v % 64 == 0:
            cv2.putText(im, str(v), (v + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            cv2.putText(im, str(v), (2, v + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return im


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lo", type=int, nargs=3, default=[0, 140, 90])
    p.add_argument("--hi", type=int, nargs=3, default=[22, 255, 255])
    p.add_argument("--n", type=int, default=16)
    args = p.parse_args(argv)
    video = REPO / "data" / "interim" / args.name / f"{args.name}_512.mp4"
    best, pix = [], []
    for i, f in enumerate(iter_frames(video, vf="fps=2")):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, tuple(args.lo), tuple(args.hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n = int((m > 0).sum())
        if n:
            pix.append(hsv[m > 0][:: max(1, n // 50)])
        best.append((n, i / 2, f if n else None))
        best = sorted(best, key=lambda b: -b[0])[: args.n * 3]
    # spread the picks out in time (at least 5 s apart)
    picks = []
    for n, t, f in best:
        if f is not None and all(abs(t - t2) >= 5 for _, t2, _ in picks):
            picks.append((n, t, f))
        if len(picks) == args.n:
            break
    tiles = []
    for n, t, f in sorted(picks, key=lambda x: x[1]):
        im = grid(f)
        cv2.putText(im, f"t={t:.1f}s  {n} px", (6, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "frames.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    allpix = np.concatenate(pix) if pix else np.zeros((0, 3))
    stats = {"frames_scanned": i + 1, "seconds": (i + 1) / 2, "picked_times_s": sorted(t for _, t, _ in picks),
             "hsv_percentiles_5_50_95": np.percentile(allpix, [5, 50, 95], axis=0).round().tolist() if len(allpix) else None}
    (args.out / "frames.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())

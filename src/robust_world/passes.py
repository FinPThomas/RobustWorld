"""Stage: group the ball track into passes at the occluder -> passes.json.

A pass is a visible run that ends against the occluder. Its outcome depends on
what is seen next: "through" (reappears on the far side), "bounce" (reappears on
the same side) or "hidden" (no reappearance within max_hidden_s, e.g. blocked).
occlusion_start_frame is the first frame of the final approach where the ball's
outline touches the occluder polygon. All frames are source-video frames.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .paths import rel

EDGE_MARGIN = 30      # px from the frame border that counts as "at the edge"
TOUCH_TOL = 15        # px slack when deciding a run starts/ends at the occluder
MIN_RUN = 5           # frames; shorter visible runs are not a real approach
MAX_GAP = 2           # missed detections tolerated inside one run


def split_runs(track: list[dict]) -> list[list[dict]]:
    runs, cur, gap = [], [], 0
    for r in track:
        if r["visible"]:
            if cur and gap > MAX_GAP:
                runs.append(cur)
                cur = []
            cur.append(r)
            gap = 0
        else:
            gap += 1
    if cur:
        runs.append(cur)
    return runs


def side_fn(poly):
    """side(x, y) -> 'R' or 'L' of the occluder's long axis (top-edge midpoint to bottom-edge midpoint)."""
    ys = [p[1] for p in poly]
    top = [p for p in poly if p[1] == min(ys)]
    bot = [p for p in poly if p[1] == max(ys)]
    x0, y0 = sum(p[0] for p in top) / len(top), min(ys)
    x1, y1 = sum(p[0] for p in bot) / len(bot), max(ys)
    return lambda x, y: "R" if (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) < 0 else "L"


def find(track: list[dict], fps: float, scene: dict, frame_size: int,
         max_hidden_s: float = 1.5) -> list[dict]:
    side = side_fn(scene["occluder_polygon"])

    def at_edge(r):
        return min(r["x"], r["y"], frame_size - r["x"], frame_size - r["y"]) < EDGE_MARGIN + r["radius"]

    runs = split_runs(track)
    passes = []
    for i, a in enumerate(runs):
        end = a[-1]
        if end["occ_dist"] < -TOUCH_TOL or len(a) < MIN_RUN:
            continue
        k = len(a) - 1
        while k > 0 and a[k - 1]["occ_dist"] >= 0:
            k -= 1
        onset = a[k]
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        b = None
        if nxt and (nxt[0]["frame"] - end["frame"]) / fps <= max_hidden_s and nxt[0]["occ_dist"] >= -TOUCH_TOL:
            b = nxt
        outcome = "hidden" if b is None else (
            "through" if side(b[0]["x"], b[0]["y"]) != side(end["x"], end["y"]) else "bounce")
        after = (runs[i + 2] if i + 2 < len(runs) else None) if b else nxt
        passes.append({
            "id": len(passes),
            "outcome": outcome,
            "side_in": side(a[0]["x"], a[0]["y"]),
            "entry_frame": a[0]["frame"],
            "entry_at_edge": at_edge(a[0]),
            "occlusion_start_frame": onset["frame"],
            "hidden_frame": end["frame"] + 1,
            "reappear_frame": b[0]["frame"] if b else None,
            "exit_frame": b[-1]["frame"] if b else None,
            "exit_at_edge": at_edge(b[-1]) if b else None,
            "onset_xy": [onset["x"], onset["y"]],
            "prev_ball_frame": runs[i - 1][-1]["frame"] if i > 0 else None,
            "next_ball_frame": after[0]["frame"] if after else None,
        })
    return passes


def run(track: list[dict], fps: float, scene: dict, frame_size: int, out: Path) -> list[dict]:
    passes = find(track, fps, scene, frame_size)
    out.write_text(json.dumps({"fps": fps, "passes": passes}, indent=1))

    def secs(xs):
        xs = [x / fps for x in xs]
        return f"median {statistics.median(xs):.2f}s ({min(xs):.2f}-{max(xs):.2f})" if xs else "n/a"

    counts = {o: sum(p["outcome"] == o for p in passes) for o in ("through", "hidden", "bounce")}
    print(f"  {len(passes)} passes: " + ", ".join(f"{n} {o}" for o, n in counts.items()))
    print(f"  entry -> occlusion start: {secs([p['occlusion_start_frame'] - p['entry_frame'] for p in passes])}")
    thr = [p for p in passes if p["outcome"] == "through"]
    print(f"  occlusion start -> reappear (through): "
          f"{secs([p['reappear_frame'] - p['occlusion_start_frame'] for p in thr])}")
    print(f"  -> {rel(out)}")
    return passes

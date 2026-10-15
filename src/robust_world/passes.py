"""Stage: group the ball track into passes at the occluder -> passes.json.

A pass is a visible run that starts clear of the occluder and reaches it: the ball's outline
overlaps the occluder polygon (occlusion_start_frame, the onset), or the ball vanishes right at
it. The outcome is what the ball does by itself within BOUNCE_S of the onset:
    "through"  vanishes and reappears on the far side (within max_hidden_s of vanishing)
    "bounce"   rolls clear again on the entry side, either straight back without ever fully
               vanishing (touch_only) or after briefly going under
    "hidden"   neither: it stays under the occluder, or rests against its edge (with a sliver
               showing) until it is picked up. The blocked case.
All frames are source-video frames.
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
BOUNCE_S = 1.5        # s after the onset by which a bouncing ball must be clear of the occluder


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

    def first_clear(run, start=0):
        return next((r for r in run[start:] if r["occ_dist"] < -TOUCH_TOL), None)

    runs = split_runs(track)
    passes = []
    for i, a in enumerate(runs):
        end = a[-1]
        if len(a) < MIN_RUN or a[0]["occ_dist"] >= -TOUCH_TOL:
            # Too short, or starts at the occluder (a ball emerging, or one stuck at the edge
            # flickering in and out of view): not an approach.
            continue
        k = next((k for k, r in enumerate(a) if r["occ_dist"] >= 0), None)
        if k is None:
            if end["occ_dist"] < -TOUCH_TOL:
                continue                       # never reached the occluder
            k = len(a) - 1                     # vanished right at it before overlapping
        onset = a[k]
        deadline = onset["frame"] + BOUNCE_S * fps
        back = first_clear(a, k + 1)
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        vanished = back is None                # still at the occluder when the run ended
        b = reappear = None
        outcome = "hidden"
        if back is not None:
            # Rolled clear without vanishing: a bounce if quick, else it rested at the edge and
            # was moved (picked up) later, so it was blocked.
            if back["frame"] <= deadline:
                outcome, reappear = "bounce", back
        elif nxt and (nxt[0]["frame"] - end["frame"]) / fps <= max_hidden_s:
            # Far side: the ball came through, even if it's first detected some way out
            # (supports/shadow under the occluder can hide it as it emerges).
            # Same side: a bounce only if it reappears at the occluder and rolls clear in time.
            # A sliver at the edge that never gets clear (or only when picked up) is blocked.
            if side(nxt[0]["x"], nxt[0]["y"]) != side(end["x"], end["y"]):
                outcome, b = "through", nxt
            elif nxt[0]["occ_dist"] >= -TOUCH_TOL:
                clear = first_clear(nxt)
                if clear is not None and clear["frame"] <= deadline:
                    outcome, b = "bounce", nxt
            if b:
                reappear = b[0]
        after = (runs[i + 2] if i + 2 < len(runs) else None) if b else nxt
        exit_run = b if b else (a if outcome == "bounce" else None)
        passes.append({
            "id": len(passes),
            "outcome": outcome,
            "touch_only": outcome == "bounce" and not vanished,
            "side_in": side(a[0]["x"], a[0]["y"]),
            "entry_frame": a[0]["frame"],
            "entry_at_edge": at_edge(a[0]),
            "occlusion_start_frame": onset["frame"],
            "hidden_frame": end["frame"] + 1 if vanished else None,
            "reappear_frame": reappear["frame"] if reappear else None,
            "exit_frame": exit_run[-1]["frame"] if exit_run else None,
            "exit_at_edge": at_edge(exit_run[-1]) if exit_run else None,
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
    print(f"  bounces without ever vanishing: {sum(p['touch_only'] for p in passes)}; "
          f"rolled in from R {sum(p['side_in'] == 'R' for p in passes)}, L {sum(p['side_in'] == 'L' for p in passes)}")
    print(f"  entry -> occlusion start: {secs([p['occlusion_start_frame'] - p['entry_frame'] for p in passes])}")
    thr = [p for p in passes if p["outcome"] == "through"]
    print(f"  occlusion start -> reappear (through): "
          f"{secs([p['reappear_frame'] - p['occlusion_start_frame'] for p in thr])}")
    print(f"  -> {rel(out)}")
    return passes

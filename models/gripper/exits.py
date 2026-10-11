"""Real exits from the tracker, in the plank's far-edge frame (env.py), for the simulator and the replay.

Reads only the tracker fields of a per_clip.json (true_pos_context, true_pos, side_in, outcome, fold);
any scoring run's file has them. For each pass that comes out on the far side: where along the edge it
first shows past it, when (seconds after the decision point at the end of the context half) and its
velocity over the next visible step. Also writes each pass's real track after it comes out, for replay.

    python models/gripper/exits.py --per-clip results/2026-10-04_2110_twoday/vjepa2/plan/scores/both-before/per_clip.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STEP_S = 1 / 8
CTX_STEPS = 12


class Edge:
    """The plank's far edge for a ball from side_in: s along it from its top end, d past it."""

    def __init__(self, polygon: list, side_in: str):
        p = np.array(polygon, float)
        top, bot = p[p[:, 1] == p[:, 1].min()], p[p[:, 1] == p[:, 1].max()]
        if side_in == "R":
            a, b = top[top[:, 0].argmin()], bot[bot[:, 0].argmin()]
        else:
            a, b = p[p[:, 0].argmax()], bot[bot[:, 0].argmax()]
        self.a, self.dir = a, (b - a) / np.linalg.norm(b - a)
        n = np.array([self.dir[1], -self.dir[0]])
        self.normal = n if (n[0] < 0) == (side_in == "R") else -n

    def sd(self, xy) -> np.ndarray:
        q = np.asarray(xy[:2], float) - self.a
        return np.array([q @ self.dir, q @ self.normal])

    def vec(self, v) -> np.ndarray:
        v = np.asarray(v, float)
        return np.array([v @ self.dir, v @ self.normal])


def step_time(k: int) -> float:
    """Seconds after the decision point of combined (context + target) step k's centroid."""
    return (k - CTX_STEPS + 0.25) * STEP_S


def real_exit(row: dict, edge: Edge):
    """(exit dict, track [(t, s, d)]) for a pass whose tracked ball shows past the far edge, else None."""
    pts = row["true_pos_context"] + row["true_pos"]
    for k, p in enumerate(pts):
        if p is not None and edge.sd(p)[1] >= 0:
            track = [(step_time(j), *edge.sd(q)) for j, q in enumerate(pts) if j >= k and q is not None]
            if len(track) < 2:
                return None
            (t0, s0, d0), (t1, s1, d1) = track[:2]
            v = [(s1 - s0) / (t1 - t0), (d1 - d0) / (t1 - t0)]
            return {"s": s0, "t": t0, "vs": v[0], "vd": v[1], "d0": d0}, track
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    ap.add_argument("--slope", type=float, default=87.0, help="px/s^2 towards image -x (real tracker)")
    ap.add_argument("--out", type=Path, default=HERE / "exits.json")
    args = ap.parse_args()
    poly = json.loads(args.scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}
    rows = json.loads(args.per_clip.read_text())
    exits, tracks = [], {}
    for r in rows:
        side = r.get("side_in", "R")
        got = real_exit(r, edges[side])
        if got is None or got[0]["t"] < 0 or got[0]["vd"] <= 0:
            continue
        e, track = got
        exits.append({"clip_id": r["clip_id"], "side": side, "fold": r["fold"], "outcome": r["outcome"],
                      **{k: round(float(v), 2) for k, v in e.items()}})
        tracks[r["clip_id"]] = [[round(float(x), 2) for x in p] for p in track]
    a = np.array([[e["s"], e["t"], e["vs"], e["vd"]] for e in exits])
    out = {"source": str(args.per_clip.relative_to(REPO)) if args.per_clip.is_absolute() else str(args.per_clip),
           "n": len(exits), "outcomes": {o: sum(e["outcome"] == o for e in exits) for o in ("through", "hidden", "bounce")},
           "prior": {"s": float(np.median(a[:, 0])), "s_std": float(a[:, 0].std()), "t": float(np.median(a[:, 1])),
                     "vs": float(np.median(a[:, 2])), "vd": float(np.median(a[:, 3]))},
           "slope_acc": {s: edges[s].vec([-args.slope, 0]).round(2).tolist() for s in "RL"},
           "exits": exits, "tracks": tracks}
    args.out.write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k not in ("exits", "tracks")}, indent=1))


if __name__ == "__main__":
    main()

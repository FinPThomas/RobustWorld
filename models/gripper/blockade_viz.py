"""Moving view of the blockade test (blockade_eval.py): held-out real passes, some that come out and some that
are blocked, each played three times side by side with the forecast from before post-training, after (C) and
a perfect one.

Each panel first shows the decision the gripper takes from the forecast alone (p = how likely the ball comes
out; GO or STAY, by blockade_eval's fixed expected-reward rule), then what follows:
    comes out, GO      the experiment's frozen PPO policy tries the catch (replayed, as in blockade_eval)
    comes out, STAY    the ball rolls through while the gripper waits parked: let go (-1)
    blocked,   GO      the gripper sets off for the forecast exit and nothing comes: wasted trip (-1)
    blocked,   STAY    it waits parked and nothing comes: held back (+0.5)
The parked gripper is drawn below the catch zone. A ball that is let go is drawn on a straight line from its
real exit. Scoring only, nothing is fitted (CLAUDE.md rules 1 and 6).

    python models/gripper/blockade_viz.py --plan results/2026-10-04_2110_twoday/vjepa2/plan --tag pb_early1_wide06
writes <plan>/gripper/<tag>/blockade.mp4 and blockade.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
T_END = 1.6                                    # s: as blockade_eval / BLOCK_END, every real exit is gone by then
COLS = ("before", "after (C)", "perfect")


def still(E, fc: dict, true: dict, park: np.ndarray, go: bool, result: dict) -> dict:
    """A pass the PPO policy does not play: the gripper stays parked, or (go) drives to the forecast exit."""
    steps, g, t = [], park.copy(), 0.0
    target = np.array([fc["x"], E.ZONE / 2])
    while t <= T_END + 1e-9:
        b = None
        if np.isfinite(true["t"]) and t >= true["t"]:
            k = t - true["t"]
            b = [true["x"] + true["vx"] * k, true["vy"] * k]
            if b[1] > E.ZONE + 0.04:
                b = None
        steps.append({"t": t, "g": g.tolist(), "gap": (E.GAP_MIN + E.GAP_MAX) / 2 if go else E.GAP_MIN, "b": b,
                      "seen": b is not None})
        if go:
            d = target - g
            n = np.linalg.norm(d)
            g = g + (d / n * min(n, E.GRIP_SPEED * E.DT) if n > 0 else 0)
        t += E.DT
    return {"fc": fc, "true": true, "steps": steps, "result": result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--through", type=int, default=3, help="passes that come out to show (the fold's first ones)")
    ap.add_argument("--blocked", type=int, default=3, help="blocked passes to show (the fold's first ones)")
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = ap.parse_args()
    args.env = args.params = args.policy = None
    sys.path.insert(0, str(HERE))
    import viz
    viz.setup(args)
    sys.path.insert(0, str(REPO / "src"))
    import blockade_eval as BE
    import evaluate as V
    from exits import Edge
    E = V.E
    res_dir = args.plan / "gripper" / args.tag
    sim = json.loads((res_dir / "gripper.json").read_text())["sim"]["PPO policy"]
    us = np.array(sorted(float(u) for u in sim))
    cs = np.array([sim[str(u)]["caught"] for u in us])
    pol = V.load_policies(HERE / "runs" / args.tag)[args.fold]
    exits = E.load_exits()
    prior = exits["prior"]
    by_id = {e["clip_id"]: e for e in exits["exits"]}
    edges = {s: Edge(json.loads(args.scene.read_text())["occluder_polygon"], s) for s in "RL"}
    rows = {k: {r["clip_id"]: r for r in json.loads((args.plan / "scores" / v / "per_clip.json").read_text())}
            for k, v in (("before", V.BEFORE), ("after (C)", V.AFTER))}
    held = sorted((r for r in rows["before"].values() if r["fold"] == args.fold), key=lambda r: r["clip_id"])
    shown = ([r for r in held if r["outcome"] == "through" and r["clip_id"] in by_id][: args.through]
             + [r for r in held if r["outcome"] != "through"][: args.blocked])
    env = E.CatchEnv(exits)
    park = np.array([E.exit_to_m(prior)["x"], E.ZONE + 0.03])
    out = []
    for r in shown:
        cid, blocked = r["clip_id"], r["outcome"] != "through"
        e = by_id.get(cid)
        true = dict(E.exit_to_m(prior), t=np.inf) if blocked else E.exit_to_m(e)
        recs, labels = [], []
        for src in COLS:
            if src == "perfect":
                p, fc = (0.0, V.prior_forecast(prior)) if blocked else (1.0, V.true_forecast(e))
            else:
                row = rows[src][cid]
                p = BE.peak_far(row)
                fc = V.wm_forecast(row, edges[r.get("side_in", "R")], prior) or V.prior_forecast(prior)
            go, stay = BE.go_value(p, float(np.interp(fc["u"], us, cs)))
            go = go > stay
            if blocked:
                res = ({"caught": False, "good": False, "label": f"WASTED TRIP ({-BE.WASTED:+.1f})"} if go
                       else {"caught": False, "good": True, "label": f"HELD BACK ({BE.HOLD:+.1f})"})
                rec = still(E, fc, true, park, go, res)
            elif go:
                rec = viz.record(E, env, pol, {"exit": e, "forecast": fc})
                ep = rec["result"]
                rec["result"] = dict(ep, label=(f"PICKED UP ({1 + E.EARLY * (1 - ep['depth']):+.1f})" if ep["caught"]
                                                else f"tried, missed ({-BE.TRY_MISS:+.1f})"))
            else:
                rec = still(E, fc, true, park, False, {"caught": False, "good": False, "label": f"LET GO ({-BE.LET_GO:+.1f})"})
            rec["result"]["why"] = rec["result"]["label"]
            recs.append(rec)
            labels.append(f"{src}: p={p:.2f} {'GO' if go else 'STAY'}")
        out.append((f"{cid} ({'blocked' if blocked else 'comes out'})", recs, labels))
    env.close()

    import cv2
    from robust_world.media import write_video
    frames = []
    for name, recs, labels in out:
        n = max(len(x["steps"]) for x in recs) + 2 * viz.FPS
        for i in range(n):
            f = np.concatenate([viz.panel(E, x, i, f"{name}  {lab}") for x, lab in zip(recs, labels)], axis=1)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    write_video(res_dir / "blockade.mp4", frames, viz.FPS)
    viz.COLS = COLS
    viz.sheet(E, [(n, recs) for n, recs, _ in out], res_dir / "blockade.png",
              f"Blockade test, {args.tag}, fold {args.fold}: GO or STAY from the forecast, then the frozen policy")
    print("wrote", res_dir / "blockade.mp4", "and blockade.png", flush=True)


if __name__ == "__main__":
    main()

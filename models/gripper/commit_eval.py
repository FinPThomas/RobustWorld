"""Score the commit-or-stay gripper (commit_env.py): does the world model's blockade knowledge help?

Every real pass, through or blocked, replayed with the policy that never saw its fold:
    through  the ball comes out at its real exit (tracker), as in evaluate.py
    blocked  (hidden or bounce) the ball never comes out
Forecast sources:
    no forecast   p = BASE (the share of passes that come out), the median exit
    before / C    p = the frozen decoder's peak P(ball beyond the plank) over the imagined steps (the fixed
                  score behind the through-vs-blocked AUROC); exit forecast as evaluate.wm_forecast
    perfect       p = 1 or 0 from the real outcome, the real exit
Reported: pick-up rate on through passes, commit rate on blocked passes (moved for a ball that never
came), mean reward. Ground truth only scores (CLAUDE.md rule 6); policies trained in simulation only.

    GRIPPER_ENV=commit python models/gripper/commit_eval.py --plan results/2026-10-04_2110_twoday/vjepa2/plan --tag commit_base
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ["GRIPPER_ENV"] = "box"            # evaluate.py's forecast helpers, with the box task's constants
import evaluate as V  # noqa: E402
os.environ["GRIPPER_ENV"] = "commit"
import commit_env as E  # noqa: E402
from exits import Edge  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCES = {"no forecast": None, "before": V.BEFORE, "after (C)": V.AFTER, "perfect": "tracker"}


def peak_far(row: dict) -> float:
    v = [x for x in row.get("imagined_far", []) if x is not None]
    return float(max(v)) if v else 0.0


def passes(plan: Path, scene: Path) -> tuple[list, dict]:
    """[(clip_id, fold, blocked, exit or None, {source: (p, forecast)})] for every scorable pass."""
    exits = E.load_exits()
    prior = exits["prior"]
    by_id = {e["clip_id"]: e for e in exits["exits"]}
    poly = json.loads(scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}
    rows = {k: {r["clip_id"]: r for r in json.loads((plan / "scores" / v / "per_clip.json").read_text())}
            for k, v in SOURCES.items() if v not in (None, "tracker")}
    truth = rows["before"]
    out = []
    for cid, r in sorted(truth.items()):
        blocked = r["outcome"] != "through"
        if not blocked and cid not in by_id:
            continue                                  # came out but the tracker has no usable exit
        e = by_id.get(cid)
        side = r.get("side_in", "R")
        src = {"no forecast": (E.BASE, V.prior_forecast(prior)),
               "perfect": (0.0, V.prior_forecast(prior)) if blocked else (1.0, V.true_forecast(e))}
        for k in ("before", "after (C)"):
            row = rows[k][cid]
            src[k] = (peak_far(row), V.wm_forecast(row, edges[side], prior) or V.prior_forecast(prior))
        out.append((cid, r["fold"], blocked, e, src))
    return out, prior


def replay(items: list, drivers: dict) -> dict:
    env = E.CatchEnv()
    res = {}
    for name, per_fold in drivers.items():
        res[name] = {}
        for s in SOURCES:
            got = []
            for cid, fold, blocked, e, src in items:
                p, fc = src[s]
                opt = {"p": p, "forecast": fc, "blocked": blocked}
                if not blocked:
                    opt["exit"] = e
                r = E.run_episode(env, per_fold(fold), opt, seed=0)
                got.append(dict(r, clip=cid, blocked=blocked))
            res[name][s] = got
    return res


def summary(eps: list) -> dict:
    thr = [r for r in eps if not r["blocked"]]
    blk = [r for r in eps if r["blocked"]]
    return {"through": len(thr), "blocked": len(blk),
            "pickup_through": round(float(np.mean([r["caught"] for r in thr])), 3),
            "commit_through": round(float(np.mean([r["committed"] for r in thr])), 3),
            "commit_blocked": round(float(np.mean([r["committed"] for r in blk])), 3),
            "reward": round(float(np.mean([r["reward"] for r in eps])), 3)}


def sim_curve(drivers: dict, n: int = 200) -> dict:
    env = E.CatchEnv()
    out = {}
    for name, per_fold in drivers.items():
        out[name] = {}
        for sharp in (0.0, 2.5, 5.0):
            for u in (0.0, 1.0):
                eps = [E.run_episode(env, per_fold(i % 5), {"u": u, "sharp": sharp}, seed=9000 + i) for i in range(n)]
                out[name][f"sharp {sharp}, u {u}"] = summary([dict(r, blocked=r["blocked"]) for r in eps])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = ap.parse_args()
    ppo = V.load_policies(HERE / "runs" / args.tag)
    drivers = {"commit if p > 0.5 (rule)": lambda f: E.commit_if_likely}
    if ppo:
        drivers = {"PPO policy": lambda f: ppo[f], **drivers}
    items, _ = passes(args.plan, args.scene)
    eps = replay(items, drivers)
    out = {"tag": args.tag, "params": json.loads(os.environ.get("GRIPPER_PARAMS", "{}")),
           "settings": {k: getattr(E, k) for k in ("PARK_DX", "FA", "COMMIT_D", "BASE", "S_MAX", "EARLY", "WIDE", "MISS")},
           "real": {d: {s: summary(v) for s, v in per.items()} for d, per in eps.items()},
           "sim": sim_curve(drivers)}
    dest = args.plan / "gripper" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "commit.json").write_text(json.dumps(out, indent=1))
    (dest / "commit_episodes.json").write_text(json.dumps(
        {d: {s: [{k: r[k] for k in ("clip", "blocked", "caught", "committed", "why", "reward")} for r in v]
             for s, v in per.items()} for d, per in eps.items()}))
    L = [f"# Commit-or-stay gripper: {args.tag}", "",
         f"Real passes: {out['real'][next(iter(drivers))]['perfect']['through']} come out, "
         f"{out['real'][next(iter(drivers))]['perfect']['blocked']} are blocked. The gripper starts {E.PARK_DX} m away and "
         f"must commit before it can see the ball; committing to a blocked pass costs {E.FA}, missing a ball that comes out "
         f"costs {E.MISS}. Generated by models/gripper/commit_eval.py.", "",
         "| driver | forecast | pick-up (comes out) | commits when blocked | mean reward |", "|---|---|---|---|---|"]
    for d, per in out["real"].items():
        for s, m in per.items():
            L.append(f"| {d} | {s} | {m['pickup_through']:.0%} | {m['commit_blocked']:.0%} | {m['reward']:.2f} |")
    L += ["", "## Simulation (sharp = how informative p is; u = exit forecast uncertainty)", "",
          "| driver | setting | pick-up | commits when blocked | mean reward |", "|---|---|---|---|---|"]
    for d, per in out["sim"].items():
        for s, m in per.items():
            L.append(f"| {d} | {s} | {m['pickup_through']:.0%} | {m['commit_blocked']:.0%} | {m['reward']:.2f} |")
    (dest / "commit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

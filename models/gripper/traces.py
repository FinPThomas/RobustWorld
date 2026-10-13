"""Step-by-step traces of one gripper experiment's policy, for the interactive policy viewer.

Records, at every policy step (1/16 s), where the gripper is, how wide its fingers are and where the ball
is, plus each episode's forecast, true exit and outcome. Two sets, both with the experiment's frozen PPO
folds (no training here; real forecasts only drive the frozen policy, CLAUDE.md rules 1 and 6):

    sim   the same simulated balls at forecast uncertainty u = 0, 0.5 and 1 (seeded, one per fold in turn)
    real  every real pass that comes out, driven by: no forecast, the world model before post-training,
          C after post-training, and the tracker's own exit (a perfect forecast)

Run with the experiment's simulator and overrides, as overnight.py does:

    GRIPPER_ENV=box GRIPPER_PARAMS='{"WIDE": 0.6}' python models/gripper/traces.py \
        --plan results/2026-10-04_2110_twoday/vjepa2/plan --tag box_wide06
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import evaluate as V
from evaluate import E
from exits import Edge

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
U_SHOW = [0.0, 0.5, 1.0]
N_BALLS = 10


def rounded(x, n=4):
    return round(float(x), n)


def episode(env, policy, options, seed) -> dict:
    obs, _ = env.reset(seed=seed, options=options)
    steps, total = [], 0.0

    def snap(closing=0):
        g, b = env.grip_xy(), env.ball_pos()
        steps.append([rounded(env.t, 3), rounded(g[0]), rounded(g[1]), rounded(env.gap()), rounded(b[0]),
                      rounded(b[1]), rounded(b[2]), closing])

    while True:
        snap()
        a = policy(env, obs)
        closing = int(a[3] > 0.5 and bool(env.visible()))
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            snap(closing)
            break
    fc = {k: rounded(env.fc[k]) for k in ("x", "t", "vx", "vy", "sig_s", "sig_t", "u")}
    true = {k: rounded(env.true[k]) for k in ("x", "t", "vx", "vy")}
    res = {k: (rounded(v, 3) if isinstance(v, float) else v) for k, v in info.items()}
    return {"forecast": fc, "true": true, "result": dict(res, reward=rounded(total, 3)), "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    pols = V.load_policies(HERE / "runs" / args.tag)
    if not pols:
        raise SystemExit(f"no policies in runs/{args.tag}")
    exits = E.load_exits()
    prior = exits["prior"]
    poly = json.loads(args.scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}

    out = {"tag": args.tag, "env": os.environ.get("GRIPPER_ENV", "pybullet"),
           "params": json.loads(os.environ.get("GRIPPER_PARAMS", "{}")),
           "settings": {k: getattr(E, k) for k in ("ZONE", "GAP_MIN", "GAP_MAX", "BALL_R", "DT", "OBS_DELAY",
                                                   "SIG_S", "SIG_T", "SIG_V", "EARLY", "WIDE", "MISS")},
           "sim": [], "real": []}
    folds = sorted(pols)
    envs = {f: E.CatchEnv(dict(exits, exits=[e for e in exits["exits"] if e["fold"] == f])) for f in folds}
    for i in range(N_BALLS):
        f = folds[i % len(folds)]
        for u in U_SHOW:
            ep = episode(envs[f], pols[f], {"u": u}, seed=777 + i)
            out["sim"].append(dict(ep, ball=i, u=u, fold=f))
    for env in envs.values():
        env.close()

    scores = args.plan / "scores"
    sources = {"no forecast": None, "before": V.BEFORE, "after (C)": V.AFTER, "perfect": "tracker"}
    rows = {k: {r["clip_id"]: r for r in json.loads((scores / v / "per_clip.json").read_text())}
            for k, v in sources.items() if v not in (None, "tracker")}
    env = E.CatchEnv(exits)
    for e in exits["exits"]:
        for src, v in sources.items():
            if v is None:
                fc = V.prior_forecast(prior)
            elif v == "tracker":
                fc = V.true_forecast(e)
            else:
                fc = V.wm_forecast(rows[src][e["clip_id"]], edges[e["side"]], prior) or V.prior_forecast(prior)
            ep = episode(env, pols[e["fold"]], {"exit": e, "forecast": fc}, seed=0)
            out["real"].append(dict(ep, clip=e["clip_id"], source=src, fold=e["fold"]))
    env.close()

    dest = args.out or args.plan / "gripper" / args.tag / "traces.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")))
    caught = {s: sum(r["result"]["caught"] for r in out["real"] if r["source"] == s) for s in sources}
    print(args.tag, "real pick-ups:", caught, "of", len(exits["exits"]), "->", dest, flush=True)


if __name__ == "__main__":
    main()

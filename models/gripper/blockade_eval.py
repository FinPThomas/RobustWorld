"""Blockade test: does knowing which balls will be blocked make the gripper smarter?

Every real pass, through or blocked (hidden or bounce). Before the ball could come out, the gripper must
decide from the forecast alone whether to set off (it cannot react in time once it sees the ball):
    ball comes out, set off   the experiment's frozen PPO policy tries to pick it up (real exit, replayed)
                              picked up: +1 + its early bonus; tried and missed: -TRY_MISS
    ball comes out, stayed    -LET_GO (a ball let go)
    blocked, set off          -WASTED (a wasted trip)
    blocked, stayed           +HOLD (correctly held back)
Decision (fixed, nothing fitted): set off iff the expected reward of going beats staying, with p = the
forecast's P(the ball comes out) and c = the policy's simulated pick-up rate at the forecast's uncertainty
u (its own gripper.json simulation curve, no real data):
    p (c (1 + EARLY/2) - (1 - c) TRY_MISS) - (1 - p) WASTED  >  -p LET_GO + (1 - p) HOLD
Forecast sources: no forecast (p = BASE, the share of passes that come out, and the median exit),
before / after (C) (p = the frozen decoder's peak P(ball beyond the plank) over the imagined steps, the
fixed score behind the through-vs-blocked AUROC; exit forecast as evaluate.wm_forecast), perfect (p = 1 or
0, the real exit). Ground truth only scores (CLAUDE.md rule 6).

    GRIPPER_ENV=pybullet GRIPPER_PARAMS='{"EARLY": 1.0, "WIDE": 0.6}' python models/gripper/blockade_eval.py \
        --plan results/2026-10-04_2110_twoday/vjepa2/plan --tag pb_early1_wide06
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate as V
from evaluate import E
from exits import Edge

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRY_MISS, LET_GO, WASTED, HOLD = 0.5, 1.0, 1.0, 0.5
BASE = 0.42                       # 143 of 337 passes come out
SOURCES = {"no forecast": None, "before": V.BEFORE, "after (C)": V.AFTER, "perfect": "tracker"}


def peak_far(row: dict) -> float:
    v = [x for x in row.get("imagined_far", []) if x is not None]
    return float(max(v)) if v else 0.0


def go_value(p: float, c: float) -> tuple[float, float]:
    go = p * (c * (1 + E.EARLY / 2) - (1 - c) * TRY_MISS) - (1 - p) * WASTED
    stay = -p * LET_GO + (1 - p) * HOLD
    return go, stay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = ap.parse_args()
    res_dir = args.plan / "gripper" / args.tag
    sim = json.loads((res_dir / "gripper.json").read_text())["sim"]["PPO policy"]
    us = np.array(sorted(float(u) for u in sim))
    cs = np.array([sim[str(u)]["caught"] for u in us])
    pols = V.load_policies(HERE / "runs" / args.tag)
    exits = E.load_exits()
    prior = exits["prior"]
    by_id = {e["clip_id"]: e for e in exits["exits"]}
    poly = json.loads(args.scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}
    rows = {k: {r["clip_id"]: r for r in json.loads((args.plan / "scores" / v / "per_clip.json").read_text())}
            for k, v in SOURCES.items() if v not in (None, "tracker")}
    env = E.CatchEnv(exits)
    out, per_clip = {}, []
    for src in SOURCES:
        tally = {"through": 0, "blocked": 0, "set_off_through": 0, "set_off_blocked": 0, "picked_up": 0, "reward": 0.0}
        for cid, r in sorted(rows["before"].items()):
            blocked = r["outcome"] != "through"
            if not blocked and cid not in by_id:
                continue
            e = by_id.get(cid)
            if src == "no forecast":
                p, fc = BASE, V.prior_forecast(prior)
            elif src == "perfect":
                p, fc = (0.0, V.prior_forecast(prior)) if blocked else (1.0, V.true_forecast(e))
            else:
                row = rows[src][cid]
                p = peak_far(row)
                fc = V.wm_forecast(row, edges[r.get("side_in", "R")], prior) or V.prior_forecast(prior)
            c = float(np.interp(fc["u"], us, cs))
            go, stay = go_value(p, c)
            set_off = go > stay
            tally["blocked" if blocked else "through"] += 1
            if blocked:
                tally["set_off_blocked"] += set_off
                rew = -WASTED if set_off else HOLD
                caught = False
            elif set_off:
                tally["set_off_through"] += 1
                ep = E.run_episode(env, pols[e["fold"]], {"exit": e, "forecast": fc}, seed=0)
                caught = bool(ep["caught"])
                tally["picked_up"] += caught
                rew = 1 + E.EARLY * (1 - ep["depth"]) if caught else -TRY_MISS
            else:
                caught, rew = False, -LET_GO
            tally["reward"] += rew
            per_clip.append({"clip": cid, "source": src, "blocked": blocked, "p": round(p, 3), "c": round(c, 3),
                             "set_off": bool(set_off), "caught": caught, "reward": round(float(rew), 3)})
        n = tally["through"] + tally["blocked"]
        out[src] = {"passes": n, "through": tally["through"], "blocked": tally["blocked"],
                    "set_off_through": round(tally["set_off_through"] / tally["through"], 3),
                    "set_off_blocked": round(tally["set_off_blocked"] / tally["blocked"], 3),
                    "picked_up": round(tally["picked_up"] / tally["through"], 3),
                    "mean_reward": round(tally["reward"] / n, 3)}
    env.close()
    res = {"tag": args.tag, "rewards": {"pick_up": f"1 + {E.EARLY} x earliness", "try_miss": -TRY_MISS,
                                        "let_go": -LET_GO, "wasted_trip": -WASTED, "hold_back": HOLD},
           "base": BASE, "sim_pickup_by_u": dict(zip(us.tolist(), cs.tolist())), "real": out}
    (res_dir / "blockade.json").write_text(json.dumps(res, indent=1))
    (res_dir / "blockade_clips.json").write_text(json.dumps(per_clip))
    L = [f"# Blockade test: {args.tag}", "",
         f"{out['perfect']['through']} real passes come out, {out['perfect']['blocked']} are blocked. The gripper decides "
         f"from the forecast alone whether to set off. Rewards: picked up +1 + {E.EARLY} x earliness, tried and missed "
         f"-{TRY_MISS}, ball let go -{LET_GO}, wasted trip -{WASTED}, held back on a blocked pass +{HOLD}. "
         "Generated by models/gripper/blockade_eval.py.", "",
         "| forecast | set off when it comes out | set off when blocked (wasted) | picked up | mean reward |",
         "|---|---|---|---|---|"]
    for s, m in out.items():
        L.append(f"| {s} | {m['set_off_through']:.0%} | {m['set_off_blocked']:.0%} | {m['picked_up']:.0%} | {m['mean_reward']:+.2f} |")
    (res_dir / "blockade.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Quick reward / setting sweep in simulation only: which settings make the forecast and its uncertainty
matter to the policy? (So that a better world-model forecast can show up as more pick-ups.)

For each setting: train one fold briefly (train.py), then on that fold's held-out simulated balls measure,
at forecast uncertainty u = 0, 0.5 and 1, the PPO pick-up rate and finger opening, next to the reactive
rule (no forecast at all). A setting is useful when
    - u = 0 beats u = 1 by a wide margin (a perfect forecast is worth a lot), and
    - u = 1 still beats or matches reactive (the policy does not depend on a forecast it cannot trust), and
    - the opening (or catch depth) changes with u (the policy reads the uncertainty).
Simulation only: no real forecasts and no real outcomes are used, so choosing a setting here cannot leak
anything about the world models into the policy (CLAUDE.md rules 1 and 7).

    python models/gripper/sweep.py --steps 400000 --envs 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = {"U_INDEP": 1, "HONEST_SIG": 1}
SETTINGS = {
    "e1w06_s025": {"EARLY": 1.0, "WIDE": 0.6, "OBS_DELAY": 0.25},
    "e1w06_s025_slow": {"EARLY": 1.0, "WIDE": 0.6, "OBS_DELAY": 0.25, "GRIP_SPEED": 0.15},
    "e2w1_s025": {"EARLY": 2.0, "WIDE": 1.0, "OBS_DELAY": 0.25},
    "e1w06_s0375_slow": {"EARLY": 1.0, "WIDE": 0.6, "OBS_DELAY": 0.375, "GRIP_SPEED": 0.15},
    # every setting above still opens the fingers fully: limit how far they open, so where matters
    "gap08_e1_s025": {"EARLY": 1.0, "WIDE": 0.6, "OBS_DELAY": 0.25, "GAP_MAX": 0.08},
    "gap065_e1_s025": {"EARLY": 1.0, "WIDE": 0.6, "OBS_DELAY": 0.25, "GAP_MAX": 0.065},
    "gap08_e1_s0125": {"EARLY": 1.0, "WIDE": 0.6, "GAP_MAX": 0.08},
}
SCORE = r"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import evaluate as V
E = V.E
exits = E.load_exits()
env = E.CatchEnv(dict(exits, exits=[e for e in exits["exits"] if e["fold"] == 0]))
pol = V.load_policies(__import__("pathlib").Path(sys.argv[2]))[0]
out = {}
for name, drv in (("ppo", pol), ("reactive", E.reactive), ("trust", E.trust_forecast)):
    for u in (0.0, 0.5, 1.0):
        res = [E.run_episode(env, drv, {"u": u}, seed=7000 + i) for i in range(int(sys.argv[3]))]
        s = V.summary(res)
        out[f"{name} u={u}"] = {k: s[k] for k in ("caught", "reward", "opening", "catch_depth")}
print(json.dumps(out))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--sim", default="box", choices=("box", "pybullet"))
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "gripper_sweep")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for name, ovr in SETTINGS.items():
        if args.only and name not in args.only:
            continue
        res_f = args.out / f"{name}.json"
        if res_f.exists():
            continue
        env = dict(os.environ, GRIPPER_PARAMS=json.dumps({**BASE, **ovr}), GRIPPER_ENV=args.sim)
        runs = args.out / name
        print(f"== {name}: {ovr}", flush=True)
        subprocess.run([py, str(HERE / "train.py"), "--fold", "0", "--steps", str(args.steps), "--envs", str(args.envs),
                        "--out", str(runs)], env=env, cwd=HERE, check=True)
        r = subprocess.run([py, "-c", SCORE, str(HERE), str(runs), str(args.episodes)], env=env, cwd=HERE,
                           check=True, capture_output=True, text=True)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        res_f.write_text(json.dumps({"settings": {**BASE, **ovr}, "steps": args.steps, "sim": args.sim, **res}, indent=1))
        print(name, json.dumps(res), flush=True)
    rows = ["| setting | PPO u=0 | u=0.5 | u=1 | reactive | trust u=0 | opening u=0 -> 1 | depth u=0 -> 1 |",
            "|---|---|---|---|---|---|---|---|"]
    for f in sorted(args.out.glob("*.json")):
        d = json.loads(f.read_text())
        rows.append(f"| {f.stem} | {d['ppo u=0.0']['caught']:.0%} | {d['ppo u=0.5']['caught']:.0%} | {d['ppo u=1.0']['caught']:.0%} "
                    f"| {d['reactive u=0.0']['caught']:.0%} | {d['trust u=0.0']['caught']:.0%} "
                    f"| {d['ppo u=0.0']['opening']} -> {d['ppo u=1.0']['opening']} "
                    f"| {d['ppo u=0.0']['catch_depth']} -> {d['ppo u=1.0']['catch_depth']} |")
    (args.out / "SWEEP.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()

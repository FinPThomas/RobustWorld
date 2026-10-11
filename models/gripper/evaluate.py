"""Score the exit-and-catch gripper (PyBullet, env.py): in simulation against forecast uncertainty, and on real clips with
each world model's forecast (before vs after post-training).

1. Simulation curve. For each uncertainty level u (forecast sigma = u x (SIG_S m, SIG_T s, SIG_V of the
   speed), env.py), episodes drawn from the held-out fold's real exits: catch rate, reward, and how the
   policy behaves (opening width, how deep in the zone it catches). This is P(pick-up | the forecast's
   error distribution).
2. Real replay. Every real pass that comes out (exits.json): the ball comes out at that pass's real
   exit (tracker: where, when, how fast) and rolls from there. The policy is the
   one whose training never saw that clip's fold. Forecast per world model, read from per_clip.json
   with the frozen decoder (ball_probe_cv.py) and fixed rules, no fitting (CLAUDE.md rules 1-6):
       where/when  the first imagined step whose decoded ball (peak >= CONF) is past the far edge;
                   imagined_far_pos (far-side peak) is used when the scoring run saved it
       speed       from the next such step, else the median real exit speed (exits.json "prior")
       u           from imagined_far_spread (the far-side decoded map's spread) when saved; else 1 - peak
       no ball past the edge: the prior forecast (median exit) with u = 1
   Plus two references: the tracker's own exit (u = 0, a perfect forecast) and no forecast at all.
3. Bridge. Each world model's real forecast errors (scored with the tracker) placed on the simulation
   curve: the catch rate the simulation predicts for that error size, next to the replayed one.

    python models/gripper/evaluate.py --plan results/2026-10-04_2110_twoday/vjepa2/plan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import env as E
from exits import Edge

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONF = 0.5
U_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SIM = 60                        # episodes per fold per u (300 per u in all)
BEFORE, AFTER = "both-before", "lossmix_both-after"


def load_policies(run: Path) -> dict:
    from stable_baselines3 import PPO
    pols = {}
    for f in sorted(run.glob("fold*/policy.zip")):
        model = PPO.load(f, device="cpu")
        pols[int(f.parent.name[4:])] = (lambda m: (lambda env, obs: m.predict(obs, deterministic=True)[0]))(model)
    return pols


def summary(res: list) -> dict:
    c = [r for r in res if r["caught"]]
    tried = [r for r in res if r["opening"] is not None]
    return {"n": len(res), "caught": round(float(np.mean([r["caught"] for r in res])), 3),
            "reward": round(float(np.mean([r["reward"] for r in res])), 3),
            "catch_depth": round(float(np.mean([r["depth"] for r in c])), 3) if c else None,
            "opening": round(float(np.mean([r["opening"] for r in tried])), 3) if tried else None,
            "travel_m": round(float(np.mean([r["travel"] for r in res])), 3)}


def sim_curve(exits: dict, drivers: dict) -> dict:
    folds = sorted({e["fold"] for e in exits["exits"]})
    envs = {f: E.CatchEnv(dict(exits, exits=[e for e in exits["exits"] if e["fold"] == f])) for f in folds}
    out = {}
    for name, per_fold in drivers.items():
        out[name] = {}
        for u in U_GRID:
            res = []
            for f in folds:        # the same episodes for every driver: seeded per fold, u and episode
                res += [E.run_episode(envs[f], per_fold(f), {"u": u}, seed=10_000 * f + 100 * int(u * 10) + i)
                        for i in range(N_SIM)]
            out[name][str(u)] = summary(res)
    for env in envs.values():
        env.close()
    return out


def wm_forecast(row: dict, edge: Edge, prior: dict):
    pos = row.get("imagined_far_pos") or row["imagined_pos"]
    pts = []
    for k, p in enumerate(pos):
        if p is None or p[2] < CONF:
            continue
        sd = edge.sd(p)
        if sd[1] >= 0:
            pts.append(((k + 0.25) * E.DT * 2, sd, p[2], k))
    if not pts:
        return None
    t0, sd0, peak, k0 = pts[0]
    # speed over the longest stretch of confident imagined steps past the edge (one step is too noisy on
    # the decoder's 32 px cells); a single such step falls back to the median real exit speed
    v = (pts[-1][1] - sd0) / (pts[-1][0] - t0) if len(pts) > 1 else np.array([prior["vs"], prior["vd"]])
    if "imagined_far_spread" in row:      # fixed scale: a spread of SIG_S (in px) counts as u = 1
        u = float(np.clip(row["imagined_far_spread"][k0] * E.M_PER_PX / E.SIG_S, 0, 1))
    else:
        u = float(np.clip(1 - peak, 0, 1))
    return to_m({"s": float(sd0[0]), "t": float(t0), "vs": float(v[0]), "vd": float(v[1])}, u)


def to_m(f: dict, u: float) -> dict:
    """A forecast in px (along / past the plank edge) as the env's forecast in metres, with its sigmas."""
    m = E.exit_to_m(f)
    return {"x": m["x"], "t": m["t"], "vx": m["vx"], "vy": m["vy"],
            "sig_s": E.SIG_S * u, "sig_t": E.SIG_T * u, "sig_v": E.SIG_V * u, "u": u}


def prior_forecast(prior: dict) -> dict:
    return to_m(prior, 1.0)


def true_forecast(e: dict) -> dict:
    return to_m(e, 0.0)


def replay(exits: dict, forecasts: dict, drivers: dict) -> dict:
    """forecasts: clip_id -> forecast dict. Returns per driver the summary over all exiting clips."""
    out = {}
    for name, per_fold in drivers.items():
        res = []
        env = E.CatchEnv(exits)
        for e in exits["exits"]:
            res.append(E.run_episode(env, per_fold(e["fold"]), {"exit": e,
                                                                "forecast": forecasts[e["clip_id"]]}, seed=0))
        env.close()
        out[name] = summary(res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=HERE / "runs" / "ppo")
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    ap.add_argument("--quick", action="store_true", help="fewer simulated episodes (a smoke test)")
    args = ap.parse_args()
    global N_SIM
    if args.quick:
        N_SIM = 6
    exits = E.load_exits()
    prior = exits["prior"]
    poly = json.loads(args.scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}
    ppo = load_policies(args.policy)
    drivers = {"reactive": lambda f: E.reactive, "trust forecast": lambda f: E.trust_forecast}
    if ppo:
        drivers = {"PPO policy": lambda f: ppo[f], **drivers}
    out = {"settings": {k: getattr(E, k) for k in ("ZONE", "GRIP_SPEED", "GAP_MIN", "GAP_MAX", "FINGER_SPEED",
                                                   "EARLY", "WIDE", "MISS", "MOVE", "SIG_S", "SIG_T", "SIG_V", "OBS_DELAY")},
           "policy_folds": sorted(ppo), "exits": exits["n"]}
    print("simulation curve...", flush=True)
    out["sim"] = sim_curve(exits, drivers)

    print("real replay...", flush=True)
    ids = {e["clip_id"]: e for e in exits["exits"]}
    sources = {"tracker exit (perfect forecast)": {i: true_forecast(e) for i, e in ids.items()},
               "no forecast": {i: prior_forecast(prior) for i in ids}}
    errors = {}
    scores = args.plan / "scores"
    for d in sorted(scores.iterdir()):
        f = d / "per_clip.json"
        if not f.exists():
            continue
        rows = {r["clip_id"]: r for r in json.loads(f.read_text())}
        if not all(i in rows for i in ids) or "imagined_pos" not in next(iter(rows.values())):
            continue
        fc, errs = {}, []
        for i, e in ids.items():
            got = wm_forecast(rows[i], edges[e["side"]], prior)
            fc[i] = got or prior_forecast(prior)
            if got:
                t = E.exit_to_m(e)
                errs.append([abs(got["x"] - t["x"]) / E.M_PER_PX, abs(got["t"] - t["t"]),
                             np.hypot(got["vx"] - t["vx"], got["vy"] - t["vy"]) / np.hypot(t["vx"], t["vy"])])
        sources[d.name] = fc
        er = np.array(errs) if errs else np.zeros((0, 3))
        errors[d.name] = {"coverage": round(len(errs) / len(ids), 3),
                          "median_err_s_px": round(float(np.median(er[:, 0])), 1) if errs else None,
                          "median_err_t_s": round(float(np.median(er[:, 1])), 3) if errs else None,
                          "median_err_v_frac": round(float(np.median(er[:, 2])), 2) if errs else None}
    out["replay"] = {src: replay(exits, fc, drivers) for src, fc in sources.items()}
    out["forecast_errors"] = errors

    # bridge: where each world model's error size sits on the simulation curve
    if ppo:
        curve = out["sim"]["PPO policy"]
        us = np.array(U_GRID)
        cr = np.array([curve[str(u)]["caught"] for u in U_GRID])
        for name, er in errors.items():
            if er["median_err_s_px"] is None:
                pred = float(cr[-1])
            else:
                u_star = min(er["median_err_s_px"] * E.M_PER_PX / (0.6745 * E.SIG_S), 1.0)   # median |N(0, s)| = 0.6745 s
                pred = er["coverage"] * float(np.interp(u_star, us, cr)) + (1 - er["coverage"]) * float(cr[-1])
                er["matched_u"] = round(u_star, 3)
            er["sim_predicted_caught"] = round(pred, 3)

    dest = args.plan / "gripper"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "gripper.json").write_text(json.dumps(out, indent=1))
    write_md(out, dest / "gripper.md")
    plot(out, dest / "gripper.png")
    print((dest / "gripper.md").read_text())


def write_md(out: dict, path: Path) -> None:
    s = out["settings"]
    L = ["# Exit-and-catch gripper (PyBullet)", "",
         f"Catch zone {s['ZONE']} m past the plank edge (ball 0.04 m), gripper {s['GRIP_SPEED']} m/s, finger gap "
         f"{s['GAP_MIN']}-{s['GAP_MAX']} m closing at {s['FINGER_SPEED']} m/s per finger, sight lag {s['OBS_DELAY']} s. "
         f"Reward: lifted and held = 1 + {s['EARLY']} x earliness - {s['WIDE']} x opening; miss = -{s['MISS']}. "
         f"{out['exits']} real passes come out. PPO folds: {out['policy_folds']}. Generated by models/gripper/evaluate.py.", "",
         "## Simulation: catch rate vs forecast uncertainty u", "",
         "| driver | " + " | ".join(f"u={u}" for u in U_GRID) + " |", "|---|" + "---|" * len(U_GRID)]
    for d, c in out["sim"].items():
        L.append(f"| {d} | " + " | ".join(f"{c[str(u)]['caught']:.0%}" for u in U_GRID) + " |")
    if "PPO policy" in out["sim"]:
        c = out["sim"]["PPO policy"]
        L += ["", "PPO behaviour vs u (on catches): opening " + ", ".join(f"{c[str(u)]['opening']}" for u in U_GRID)
              + "; catch depth " + ", ".join(f"{c[str(u)]['catch_depth']}" for u in U_GRID) + " (0 = at the edge, 1 = zone end); "
              "travel (m) " + ", ".join(f"{c[str(u)]['travel_m']}" for u in U_GRID) + "."]
    drivers = list(out["sim"])
    L += ["", "## Real replay: catch rate (reward) by forecast source", "",
          "| forecast | " + " | ".join(drivers) + " |", "|---|" + "---|" * len(drivers)]
    for src, r in out["replay"].items():
        L.append(f"| {src} | " + " | ".join(f"{r[d]['caught']:.0%} ({r[d]['reward']:.2f})" for d in drivers) + " |")
    L += ["", "## World-model forecast errors (scored with the tracker) and the simulation's prediction", "",
          "| model | has a forecast | median error along edge | time | speed | sim-predicted catch |", "|---|---|---|---|---|---|"]
    for n, e in out["forecast_errors"].items():
        L.append(f"| {n} | {e['coverage']:.0%} | {e['median_err_s_px']} px | {e['median_err_t_s']} s | "
                 f"{e['median_err_v_frac']} | {e.get('sim_predicted_caught', '')} |")
    path.write_text("\n".join(L) + "\n")


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    col = {"PPO policy": "tab:blue", "trust forecast": "tab:green", "reactive": "tab:orange"}
    for d, c in out["sim"].items():
        ax[0].plot(U_GRID, [c[str(u)]["caught"] for u in U_GRID], "o-", color=col.get(d), label=d)
    ax[0].set(xlabel="forecast uncertainty u (sigma = u x 12 cm, 0.3 s, 50% speed)", ylabel="pick-up rate",
              title="Simulation", ylim=(0, 1))
    ax[0].legend(fontsize=8)
    if "PPO policy" in out["sim"]:
        c = out["sim"]["PPO policy"]
        ax[1].plot(U_GRID, [c[str(u)]["opening"] for u in U_GRID], "o-", label="finger opening at grasp (0 narrow, 1 wide)")
        ax[1].plot(U_GRID, [c[str(u)]["catch_depth"] for u in U_GRID], "s-", label="grasp depth in zone (0 early, 1 late)")
        ax[1].set(xlabel="forecast uncertainty u", title="How the PPO policy adapts", ylim=(0, 1))
        ax[1].legend(fontsize=8)
    srcs = list(out["replay"])
    d = "PPO policy" if "PPO policy" in out["sim"] else "trust forecast"
    ax[2].barh(range(len(srcs)), [out["replay"][s][d]["caught"] for s in srcs], color="tab:blue")
    ax[2].set_yticks(range(len(srcs)), srcs, fontsize=7)
    ax[2].set(xlabel=f"pick-up rate on real replayed passes ({d})", xlim=(0, 1), title="Real clips, by forecast source")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()

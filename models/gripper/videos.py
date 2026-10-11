"""Videos of the gripper policy, to watch how it behaves.

1. uncertainty.mp4: the same simulated ball three times, side by side, with the forecast's uncertainty
   u = 0, 0.5 and 1. The orange disc on the plank edge is the forecast exit, its radius one sigma.
2. before_after.mp4: real passes (each ball comes out at its real exit), side by side, with the forecast from the
   world model before post-training and after (C), each through the same frozen policy.

    python models/gripper/videos.py --plan results/2026-10-04_2110_twoday/vjepa2/plan [--fold 0] [--gui]
--gui instead opens PyBullet's own window and plays the uncertainty episodes live.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

import env as E
import evaluate as V
from exits import Edge

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from robust_world.media import write_video  # noqa: E402

FPS = 16


def marker(env: E.CatchEnv) -> int:
    r = max(env.fc["sig_s"], 0.006)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=0.001, rgbaColor=[1, 0.55, 0, 0.6],
                              physicsClientId=env.cid)
    return p.createMultiBody(0, -1, vis, [env.fc["x"], 0.0, 0.001], physicsClientId=env.cid)


def episode_frames(env, policy, options, seed, label) -> tuple[list, dict]:
    frames = []
    obs, _ = env.reset(seed=seed, options=options)
    m = marker(env)
    info = {}
    while True:
        frames.append(env.render())
        obs, r, term, trunc, info = env.step(policy(env, obs))
        if term or trunc:
            break
    frames += [env.render()] * 8
    p.removeBody(m, physicsClientId=env.cid)
    verdict = "PICKED UP" if info.get("caught") else f"missed ({info.get('why')})"
    out = []
    for f in frames:
        f = np.ascontiguousarray(f)
        cv2.putText(f, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(f)
    for f in out[-8:]:
        cv2.putText(f, verdict, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 220, 60) if info.get("caught") else (230, 60, 60), 2, cv2.LINE_AA)
    return out, info


def side_by_side(clips: list[list]) -> list:
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    return [np.concatenate([c[i] for c in clips], axis=1) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=HERE / "runs" / "ppo")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = ap.parse_args()
    exits = E.load_exits()
    pols = V.load_policies(args.policy)
    pol = pols.get(args.fold) or E.trust_forecast
    held = dict(exits, exits=[e for e in exits["exits"] if e["fold"] == args.fold])
    dest = args.plan / "gripper"
    dest.mkdir(parents=True, exist_ok=True)

    if args.gui:
        env = E.CatchEnv(held, gui=True)
        for i in range(args.episodes):
            for u in (0.0, 0.5, 1.0):
                E.run_episode(env, lambda en, o: (__import__("time").sleep(E.DT), pol(en, o))[1], {"u": u}, seed=i)
        return

    env = E.CatchEnv(held)
    video = []
    for i in range(args.episodes):
        row = [episode_frames(env, pol, {"u": u}, 500 + i, f"u = {u}")[0] for u in (0.0, 0.5, 1.0)]
        video += side_by_side(row)
    write_video(dest / "uncertainty.mp4", video, FPS)
    print("wrote", dest / "uncertainty.mp4", flush=True)

    poly = json.loads(args.scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}
    runs = {}
    for name in (V.BEFORE, V.AFTER):
        rows = {r["clip_id"]: r for r in json.loads((args.plan / "scores" / name / "per_clip.json").read_text())}
        runs[name] = rows
    video = []
    for e in held["exits"][: args.episodes]:
        row = []
        for name, label in ((V.BEFORE, "before post-training"), (V.AFTER, "after (C)")):
            fc = V.wm_forecast(runs[name][e["clip_id"]], edges[e["side"]], exits["prior"]) or V.prior_forecast(exits["prior"])
            opts = {"exit": e, "forecast": fc}
            row.append(episode_frames(env, pol, opts, 0, f"{e['clip_id']}: {label}")[0])
        video += side_by_side(row)
    write_video(dest / "before_after.mp4", video, FPS)
    print("wrote", dest / "before_after.mp4", flush=True)


if __name__ == "__main__":
    main()

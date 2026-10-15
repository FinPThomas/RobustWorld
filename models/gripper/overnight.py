"""Overnight queue for the exit-and-catch gripper: train, score and push one experiment after another.

Each experiment is a simulator (PyBullet env.py, or the plain-Python box_env.py), a set of setting
overrides (GRIPPER_PARAMS: reward weights, finger opening, sight lag, ...) and a training length. For
each one, in order: train 5 folds (train.py; resumes from checkpoints, skips finished folds), score it
(evaluate.py --tag <name>), record videos for PyBullet runs, write SUMMARY.md across all experiments,
then commit and push. A finished experiment (its gripper.json exists) is skipped, so the queue can be
restarted at any time and never repeats work. "init" continues training from another experiment's
finished policies (e.g. longer training).

Every policy is trained on simulated forecasts only (the true exit plus noise of the stated size);
real world-model forecasts are used only to score (CLAUDE.md rule 1).

    python models/gripper/overnight.py --plan results/2026-10-04_2110_twoday/vjepa2/plan [--envs 3] [--no-push]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (name, simulator, overrides, steps per fold, init from, why)
EXPERIMENTS = [
    ("pb_base", "pybullet", {}, 1_500_000, None,
     "PyBullet, default reward (early bonus 0.5, wide-grip cost 0.3, miss -1); touching the ball before the grasp is a miss"),
    ("box_base", "box", {}, 1_000_000, None, "the same task in plain-Python kinematics"),
    ("box_early1", "box", {"EARLY": 1.0}, 1_000_000, None, "bigger bonus for an early pick-up"),
    ("box_wide06", "box", {"WIDE": 0.6}, 1_000_000, None, "wide grip costs more"),
    ("box_wide01", "box", {"WIDE": 0.1}, 1_000_000, None, "wide grip nearly free"),
    ("box_miss2", "box", {"MISS": 2.0}, 1_000_000, None, "a miss costs twice as much"),
    ("box_sight025", "box", {"OBS_DELAY": 0.25}, 1_000_000, None, "slower sight (4 frames): prediction matters more"),
    ("pb_early1_wide06", "pybullet", {"EARLY": 1.0, "WIDE": 0.6}, 1_500_000, None,
     "PyBullet with early pick-ups and narrow grips rewarded more"),
    ("pb_sight025", "pybullet", {"OBS_DELAY": 0.25}, 1_500_000, None, "PyBullet, slower sight"),
    # Real forecasts place the ball well but time it and give its speed far less precisely than their one u
    # says, and the policies above only ever saw one u for all three. Fix: state each part's own sigma
    # from the forecast's resolution (evaluate.py HONEST_SIG), and train with separate uncertainties.
    ("box_base_honest", "box", {"HONEST_SIG": 1}, 1_000_000, "box_base",
     "box_base's policies unchanged, real forecasts stated with each part's own sigma"),
    ("box_indep", "box", {"U_INDEP": 1, "HONEST_SIG": 1}, 1_000_000, None,
     "trained with separate uncertainties for where / when / speed; honest sigmas"),
    ("box_sight025_indep", "box", {"OBS_DELAY": 0.25, "U_INDEP": 1, "HONEST_SIG": 1}, 1_000_000, None,
     "box_indep with slower sight (4 frames)"),
    # BLOCK experiments (blocked passes, parked gripper, PPO end to end) are left out: PPO learned never to set off
    # (box_block_cloud, box_block_letgo*_cloud). The blockade is scored instead by blockade_eval.py, run on every
    # experiment below: decide to set off from the forecast alone, then the frozen policy picks up.
    ("pb_early1_wide06_indep", "pybullet", {"EARLY": 1.0, "WIDE": 0.6, "U_INDEP": 1, "HONEST_SIG": 1}, 1_500_000,
     None, "pb_early1_wide06 trained with separate uncertainties; honest sigmas"),
    ("pb_narrow", "pybullet", {"GAP_MAX": 0.09}, 1_500_000, None, "fingers open at most 9 cm"),
    # Fast along the edge, slow towards / away from the plank: the gripper must commit to a depth on the forecast
    # (when the ball comes out) and can only correct sideways once it sees the ball; an early catch earns EARLY.
    ("box_slowy", "box", {"EARLY": 1.0, "WIDE": 0.6, "GRIP_SPEED": 0.35, "GRIP_SPEED_Y": 0.05}, 1_000_000, None,
     "fast sideways (0.35 m/s), slow towards the plank (0.05 m/s): commit to a depth early"),
    ("pb_slowy", "pybullet", {"EARLY": 1.0, "WIDE": 0.6, "GRIP_SPEED": 0.35, "GRIP_SPEED_Y": 0.05}, 1_500_000, None,
     "box_slowy in PyBullet"),
    # A speed-limited gripper: it moves at 0.15 m/s and its fingers close at 0.1 m/s each (a wide grip takes about
    # 0.4 s to close), so it cannot snap onto the ball and must be in place, and closing, early.
    ("box_slowgrip", "box", {"EARLY": 1.0, "WIDE": 0.6, "GRIP_SPEED": 0.15, "FINGER_SPEED": 0.1}, 1_000_000, None,
     "speed-limited gripper: 0.15 m/s, fingers close at 0.1 m/s"),
    ("pb_slowgrip", "pybullet", {"EARLY": 1.0, "WIDE": 0.6, "GRIP_SPEED": 0.15, "FINGER_SPEED": 0.1}, 1_500_000, None,
     "box_slowgrip in PyBullet"),
    ("pb_early1_wide06_3m", "pybullet", {"EARLY": 1.0, "WIDE": 0.6}, 3_000_000, "pb_early1_wide06",
     "pb_early1_wide06 (best on the blockade test) trained on to 3M steps per fold"),
    ("pb_base_4m", "pybullet", {}, 4_000_000, "pb_base", "pb_base trained on to 4M steps per fold"),
]


FULL = False                     # --full: normal priority, torch threads uncapped


def sh(cmd: list, env: dict | None = None, check: bool = True) -> int:
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    # below-normal priority on Windows (unless --full), so the browser stays responsive while the laptop trains
    flags = 0 if FULL else getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    r = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, creationflags=flags)
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")
    return r.returncode


def push(paths: list, message: str) -> None:
    sh(["git", "add", "-f", *paths], check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        return
    sh(["git", "commit", "-q", "-m", message])
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    for wait in (0, 2, 4, 8, 16):
        time.sleep(wait)
        sh(["git", "pull", "-q", "--no-rebase", "origin", branch], check=False)   # "HEAD" would pull the default branch
        if sh(["git", "push", "-q", "origin", "HEAD"], check=False) == 0:
            return
    print("!! push failed; will retry with the next experiment", flush=True)


def summary(plan: Path) -> Path:
    rows = ["# Gripper experiments: summary", "",
            "Pick-up rate. Simulation: PPO at forecast uncertainty u = 0 / 0.5 / 1 (and the finger opening it "
            "uses at u = 0 vs 1). Real passes: PPO driven by the pretrained world model's forecast (before), "
            "by C's (after), and by the tracker's own exit (perfect).", "",
            "| experiment | why | sim u=0 | u=0.5 | u=1 | opening u=0 -> 1 | real: before | after (C) | perfect |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, kind, ovr, steps, init, why in EXPERIMENTS:
        f = plan / "gripper" / name / "gripper.json"
        if not f.exists():
            rows.append(f"| {name} | {why} | (queued) | | | | | | |")
            continue
        g = json.loads(f.read_text())
        s, rp = g["sim"].get("PPO policy"), g["replay"]
        if s is None:
            continue
        pick = lambda src: rp.get(src, {}).get("PPO policy", {})  # noqa: E731
        pct = lambda v: "" if not v or v.get("caught") is None else (  # noqa: E731
            f"{v['caught']:.0%}" + (f" ({v['wasted']:.0%} wasted, reward {v['reward_all']:.2f})" if "wasted" in v else ""))
        s = {u: (dict(c, caught=c["caught"]) if c["caught"] is not None else c) for u, c in s.items()}
        rows.append(f"| {name} | {why} | {pct(s['0.0'])} | {pct(s['0.5'])} | {pct(s['1.0'])} "
                    f"| {s['0.0']['opening']} -> {s['1.0']['opening']} | {pct(pick('both-before'))} "
                    f"| {pct(pick('lossmix_both-after'))} | {pct(pick('tracker exit (perfect forecast)'))} |")
    out = plan / "gripper" / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--envs", type=int, default=2, help="simulator processes per trainer (one trainer at a time)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--only", nargs="*", help="run just these experiments")
    ap.add_argument("--steps-scale", type=float, default=1.0, help="multiply training steps (smoke tests)")
    ap.add_argument("--quick", action="store_true", help="quick scoring (smoke tests)")
    ap.add_argument("--full", action="store_true", help="full force: normal priority, torch threads uncapped")
    args = ap.parse_args()
    global FULL
    FULL = args.full
    plan = args.plan if args.plan.is_absolute() else REPO / args.plan
    py = sys.executable
    for name, kind, ovr, steps, init, why in EXPERIMENTS:
        if args.only and name not in args.only:
            continue
        dest = plan / "gripper" / name
        if (dest / "gripper.json").exists():
            print(f"== {name}: done already", flush=True)
            continue
        print(f"== {name}: {why}", flush=True)
        # one OpenBLAS thread per process: its per-thread buffers ran a 7.5 GB laptop out of memory with
        # several queues of 4-6 processes each (torch keeps its own threads for the PPO updates)
        # torch is capped at 2 threads too, leaving cores free for Fin's browser
        env = dict(os.environ, GRIPPER_PARAMS=json.dumps(ovr), GRIPPER_ENV=kind,
                   OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                   **({} if args.full else {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
                                            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "2")}))
        runs = HERE / "runs" / name
        if init:
            for f in sorted((HERE / "runs" / init).glob("fold*/policy.zip")):
                d = runs / f.parent.name
                if not (d / "policy.zip").exists() and not list(d.glob("ckpt_*_steps.zip")):
                    d.mkdir(parents=True, exist_ok=True)
                    info = json.loads((f.parent / "train_info.json").read_text())
                    shutil.copy(f, d / f"ckpt_{info['steps']}_steps.zip")
        for fold in range(5):
            if not (runs / f"fold{fold}" / "policy.zip").exists():
                sh([py, HERE / "train.py", "--fold", fold, "--steps", int(steps * args.steps_scale), "--envs", args.envs, "--out", runs], env)
                if not args.no_push:
                    push([runs / f"fold{fold}" / "policy.zip", runs / f"fold{fold}" / "train_info.json"],
                         f"Gripper {name}: fold {fold} trained")
        sh([py, HERE / "evaluate.py", "--plan", plan, "--policy", runs, "--tag", name, *(["--quick"] if args.quick else [])], env)
        if kind == "pybullet":
            sh([py, HERE / "videos.py", "--plan", plan, "--policy", runs, "--tag", name], env, check=False)
        sh([py, HERE / "blockade_eval.py", "--plan", plan, "--tag", name], env, check=False)
        sh([py, HERE / "blockade_eval.py", "--plan", plan, "--tag", name, "--v2"], env, check=False)
        s = summary(plan)
        if not args.no_push:
            push([dest, s], f"Gripper {name}: scored on simulation and real passes")
    summary(plan)
    print("== queue finished", flush=True)


if __name__ == "__main__":
    main()

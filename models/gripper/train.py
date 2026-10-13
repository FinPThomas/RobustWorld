"""Train the exit-and-catch gripper policy with PPO in the PyBullet simulator (env.py), on CPU.

One policy per CV fold (robust_world.eval.ball.cv_folds, as stored in each exit's "fold"): fold k's
policy never saw the exits of fold k's clips, and only fold k's policy drives those clips in the real
replay (evaluate.py). Checkpoints every --save-every steps; a finished fold is skipped, a stopped
one resumes from its last checkpoint.

    python models/gripper/train.py                       # all 5 folds, 1M steps each
    python models/gripper/train.py --fold 0 --envs 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import os

if os.environ.get("GRIPPER_ENV") == "box":      # plain-Python kinematics instead of PyBullet (box_env.py)
    import box_env as E
elif os.environ.get("GRIPPER_ENV") == "commit":   # box kinematics with blocked passes (commit_env.py)
    import commit_env as E
else:
    import env as E

HERE = Path(__file__).resolve().parent


def exits_without(fold: int) -> dict:
    ex = E.load_exits()
    return dict(ex, exits=[e for e in ex["exits"] if e["fold"] != fold])


def train_fold(fold: int, steps: int, out: Path, save_every: int, envs: int, seed: int) -> None:
    d = out / f"fold{fold}"
    done = d / "policy.zip"
    if done.exists():
        print(f"fold {fold}: done already ({done})", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    exits = exits_without(fold)
    venv = make_vec_env(lambda: E.CatchEnv(exits), n_envs=envs, seed=seed + fold,
                        vec_env_cls=SubprocVecEnv if envs > 1 else None)
    ckpts = sorted(d.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        model = PPO.load(ckpts[-1], env=venv, device="cpu")
        left = steps - model.num_timesteps
        print(f"fold {fold}: resuming at {model.num_timesteps} steps", flush=True)
    else:
        # FAST (GRIPPER_PARAMS): half the update passes over each rollout, one minibatch per rollout; about 2x
        # faster on CPU, where PPO's updates rather than the simulator are the bottleneck
        fast = json.loads(os.environ.get("GRIPPER_PARAMS", "{}")).get("FAST", 0)
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512 * envs if fast else 512, n_epochs=5 if fast else 10,
                    gamma=0.99, ent_coef=0.003,
                    learning_rate=3e-4, policy_kwargs={"net_arch": [128, 128]}, seed=seed + fold, device="cpu",
                    verbose=1)
        left = steps
    t0 = time.time()
    if left > 0:
        model.learn(left, reset_num_timesteps=not ckpts,
                    callback=CheckpointCallback(max(save_every // envs, 1), str(d), name_prefix="ckpt"))
    model.save(done)
    for c in d.glob("ckpt_*_steps.zip"):
        c.unlink()
    (d / "train_info.json").write_text(json.dumps({"fold": fold, "steps": int(model.num_timesteps),
                                                   "minutes": round((time.time() - t0) / 60, 1),
                                                   "train_exits": len(exits["exits"])}))
    print(f"fold {fold}: trained {model.num_timesteps} steps in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> None:  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "ppo")
    ap.add_argument("--save-every", type=int, default=100_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    folds = [args.fold] if args.fold is not None else sorted({e["fold"] for e in E.load_exits()["exits"]})
    for f in folds:
        train_fold(f, args.steps, args.out, args.save_every, args.envs, args.seed)


if __name__ == "__main__":     # SubprocVecEnv needs this guard on Windows
    main()

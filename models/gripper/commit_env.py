"""Commit-or-stay gripper: the exit-and-catch task with blocked passes, so knowing the blockade pays.

Box kinematics (box_env.py), with three changes:
1. Blocked passes. Some balls never come out (the hidden blocker stops them). The policy sees the
   forecast's P(the ball comes out) as one more observation, p.
2. The gripper cannot react in time. It starts parked PARK_DX along the edge from the median exit, too far
   to reach the ball after first seeing it, so it has to commit on the forecast.
3. Committing to a blocked pass costs FA (it moved for a ball that never came). Staying parked on a blocked
   pass costs nothing; a ball that comes out and is not picked up costs MISS, as before.

Training (simulation only, CLAUDE.md rule 1): p is drawn first, calibrated by construction (logit p ~
N(logit BASE, s^2), s ~ U(0, S_MAX)), then the ball comes out with probability p. Sharp p (large s) is an
informative forecast, p near BASE an uninformative one. The exit forecast is the usual noisy forecast.
Real replay (commit_eval.py): p is the frozen decoder's peak P(ball beyond the plank) over the imagined
steps, the same fixed score as the through-vs-blocked AUROC; nothing is fitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import box_env as B
from box_env import exit_to_m, load_exits as _load_exits  # noqa: F401

HERE = Path(__file__).resolve().parent
PARK_DX = 0.18          # m along the edge from the median exit: ~0.7 s away at GRIP_SPEED
FA = 1.0                # cost of committing to a ball that never comes out
COMMIT_D = 0.03         # moved this far from the park = committed
BASE = 0.42             # share of passes that come out (143 of 337 clips): the sim prior and "no forecast"
S_MAX = 5.0
T_BLOCKED = 1.6         # s: a blocked pass ends here (every real exit is out and gone by then)
globals().update({k: v for k, v in json.loads(os.environ.get("GRIPPER_PARAMS", "{}")).items() if k in globals()})

# the box task's constants and helpers, for scripts that read them from this module
for _k in ("ZONE", "GAP_MIN", "GAP_MAX", "BALL_R", "DT", "OBS_DELAY", "SIG_S", "SIG_T", "SIG_V", "EARLY", "WIDE",
           "MISS", "MOVE", "M_PER_PX", "GRIP_SPEED", "FINGER_SPEED"):
    globals()[_k] = getattr(B, _k)


def load_exits(path: Path = HERE / "exits.json") -> dict:
    return _load_exits(path)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class CatchEnv(B.CatchEnv):
    def __init__(self, exits: dict | None = None, seed: int | None = None, gui: bool = False):
        super().__init__(exits, seed, gui)
        self.park = np.array([self.prior["x"] + PARK_DX, ZONE / 2])
        self.rest = self.park.copy()
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(-10, 10, (18,), np.float32)

    def reset(self, seed=None, options=None):
        o = dict(options or {})
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if "exit" in o or "blocked" in o:           # replay: a real pass, its p and forecast
            self.p = float(o["p"])
            self.blocked = bool(o.get("blocked"))
            if self.blocked:
                o = {"exit": {"s": 0, "t": 99.0, "vs": 0, "vd": 1}, "forecast": o["forecast"]}
            obs, info = super().reset(options={"exit": o["exit"], "forecast": o["forecast"]})
        else:
            s = float(o.get("sharp", self.rng.uniform(0, S_MAX)))
            self.p = float(1 / (1 + np.exp(-(logit(BASE) + self.rng.normal(0, s)))))
            self.blocked = bool(self.rng.uniform() >= self.p)
            obs, info = super().reset(options={k: v for k, v in o.items() if k == "u"})
        if self.blocked:
            self.true = dict(self.true, t=99.0)      # never comes out
        self.g = self.park.copy()
        self.committed = False
        return self._obs(), info

    def step(self, a):
        obs, r, term, trunc, info = super().step(a)
        self.committed |= bool(np.linalg.norm(self.g - self.park) > COMMIT_D)
        if term:
            info = dict(info, committed=self.committed, blocked=self.blocked)
            self.result = info
            return self._obs(), r, term, trunc, info
        if self.blocked and self.t >= T_BLOCKED:
            r -= FA if self.committed else 0.0
            self.result = {"caught": False, "why": "false alarm" if self.committed else "stayed (blocked)",
                           "depth": None, "opening": None, "t": self.t, "travel": self.travel,
                           "committed": self.committed, "blocked": True}
            return self._obs(), float(r), True, False, dict(self.result)
        return self._obs(), r, term, trunc, info

    def _obs(self):
        return np.r_[super()._obs(), self.p].astype(np.float32)


# ---- scripted references --------------------------------------------------------------------------
def always_commit(env, obs):
    """Trust the exit forecast and always go for it (ignores p)."""
    return B.trust_forecast(env, obs)


def commit_if_likely(env, obs):
    """Go for it only if p > 0.5; otherwise stay parked."""
    if env.p <= 0.5 and not env.visible():
        return np.array([0, 0, -1.0, -1.0])
    return B.trust_forecast(env, obs)


run_episode = B.run_episode

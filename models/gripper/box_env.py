"""The same exit-and-catch task without a physics engine: plain-Python kinematics ("box" version).

Same scene, exits, forecast, observations, actions and reward as env.py (PyBullet), so policies and
results can be compared one to one, but contact is replaced by geometry: the ball rolls in a straight
line at its exit velocity, the fingers are two 1.2 x 4 cm boxes either side of the gripper centre, and
- the ball is knocked (a miss) if it overlaps a finger before the grasp;
- a grasp closes both fingers at FINGER_SPEED; it holds if, when the fingers reach the ball, the ball's
  centre is still between the fingers' ends (along y) and the closing fingers did not clip it before
  (along x, |dx| < gap/2 - radius at the start).
About 50x faster than PyBullet, so it suits wide reward and setting sweeps; PyBullet checks they hold
with real contact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
M_PER_PX = 0.04 / 44
BALL_R = 0.02
ZONE = 0.12
DT = 1 / 16
SUBSTEPS = 8
GRIP_SPEED = 0.25
GAP_MIN, GAP_MAX = 0.045, 0.13
FINGER_SPEED = 0.25
FINGER_W, FINGER_L = 0.012, 0.04
EARLY, WIDE, MISS, MOVE = 0.5, 0.3, 1.0, 0.05
SIG_S, SIG_T, SIG_V = 0.12, 0.3, 0.5
OBS_NOISE = 0.003
OBS_DELAY = 0.125
MAX_T = 2.5
globals().update({k: v for k, v in json.loads(os.environ.get("GRIPPER_PARAMS", "{}")).items() if k in globals()})


def load_exits(path: Path = HERE / "exits.json") -> dict:
    return json.loads(path.read_text())


def exit_to_m(e: dict) -> dict:
    return {"x": e["s"] * M_PER_PX, "t": e["t"], "vx": e["vs"] * M_PER_PX, "vy": e["vd"] * M_PER_PX}


class CatchEnv(gym.Env):
    def __init__(self, exits: dict | None = None, seed: int | None = None, gui: bool = False):
        self.exits = exits or load_exits()
        self.ex = [exit_to_m(e) for e in self.exits["exits"]]
        self.prior = exit_to_m(self.exits["prior"])
        self.rest = np.array([self.prior["x"], ZONE / 2])
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-10, 10, (17,), np.float32)
        self.rng = np.random.default_rng(seed)

    # same episode set-up as env.py
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = options or {}
        if "exit" in o:
            self.true, self.fc = exit_to_m(o["exit"]), dict(o["forecast"])
        else:
            e = dict(self.ex[self.rng.integers(len(self.ex))])
            j = self.rng.normal(1, 0.1, 2)
            e = {"x": e["x"] + self.rng.normal(0, 0.01), "t": max(e["t"] + self.rng.normal(0, 0.03), 0.0),
                 "vx": e["vx"] * j[0], "vy": max(e["vy"] * j[1], 0.03)}
            self.true = e
            u = float(o.get("u", self.rng.uniform(0, 1)))
            sv = SIG_V * u * np.hypot(e["vx"], e["vy"])
            self.fc = {"x": e["x"] + self.rng.normal(0, SIG_S * u + 1e-9), "t": e["t"] + self.rng.normal(0, SIG_T * u + 1e-9),
                       "vx": e["vx"] + self.rng.normal(0, sv + 1e-9), "vy": e["vy"] + self.rng.normal(0, sv + 1e-9),
                       "sig_s": SIG_S * u, "sig_t": SIG_T * u, "sig_v": SIG_V * u, "u": u}
        self.t, self.g, self.w = 0.0, self.rest.copy(), GAP_MIN
        self.seen, self.travel, self.result = [], 0.0, None
        return self._obs(), {}

    def ball_at(self, t):
        e = self.true
        if t < e["t"]:
            return None
        k = t - e["t"]
        return np.array([e["x"] + e["vx"] * k, e["vy"] * k])

    def ball_pos(self):
        b = self.ball_at(self.t)
        return np.r_[b, BALL_R] if b is not None else np.array([0.0, -0.3, BALL_R])

    def grip_xy(self):
        return self.g.copy()

    def gap(self):
        return self.w

    def _touches_finger(self, b, g, w) -> bool:
        d = b - g
        if abs(d[1]) > FINGER_L / 2 + BALL_R:
            return False
        inner, outer = w / 2, w / 2 + FINGER_W
        return abs(d[0]) + BALL_R > inner and abs(d[0]) - BALL_R < outer

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        v = a[:2] * GRIP_SPEED
        n = np.linalg.norm(v)
        v = v * GRIP_SPEED / n if n > GRIP_SPEED else v
        target_w = GAP_MIN + (a[2] + 1) / 2 * (GAP_MAX - GAP_MIN)
        g0 = self.g.copy()
        for _ in range(SUBSTEPS):
            h = DT / SUBSTEPS
            self.g = self.g + v * h
            self.g[1] = np.clip(self.g[1], 0.0, ZONE)
            self.w += np.clip(target_w - self.w, -0.5 * h, 0.5 * h)
            self.t += h
            b = self.ball_at(self.t)
            if b is not None:
                self.seen.append((self.t, b + self.rng.normal(0, OBS_NOISE, 2)))
                if self._touches_finger(b, self.g, self.w):
                    return self._finish(-MOVE * np.linalg.norm(self.g - g0) / 0.1, False, "knocked the ball")
        r = -MOVE * float(np.linalg.norm(self.g - g0)) / 0.1
        self.travel += float(np.linalg.norm(self.g - g0))
        if a[3] > 0.5 and self.visible():
            return self._grasp(r)
        b = self.ball_at(self.t)
        if b is not None and (b[1] > ZONE + BALL_R or abs(b[0] - 0.2) > 0.45):
            return self._finish(r, False, "left the zone")
        if self.t >= MAX_T:
            return self._finish(r, False, "time out")
        return self._obs(), r, False, False, {}

    def _grasp(self, r):
        b0 = self.ball_at(self.t)
        depth = float(np.clip(b0[1] / ZONE, 0, 1)) if b0 is not None else 1.0
        open_frac = float(np.clip((self.w - GAP_MIN) / (GAP_MAX - GAP_MIN), 0, 1))
        if b0 is None:
            return self._finish(r, False, "grasp missed", depth, open_frac)
        dx = b0[0] - self.g[0]
        if abs(dx) + BALL_R > self.w / 2:             # a finger clips the ball as it starts to close
            return self._finish(r, False, "grasp missed", depth, open_frac)
        t_close = (self.w / 2 - BALL_R + abs(dx)) / FINGER_SPEED / 2 + 0.01
        b1 = self.ball_at(self.t + t_close)
        if abs(b1[1] - self.g[1]) < FINGER_L / 2:
            return self._finish(r + 1 + EARLY * (1 - depth) - WIDE * open_frac, True, "held", depth, open_frac)
        return self._finish(r, False, "grasp missed", depth, open_frac)

    def _finish(self, r, ok, why, depth=None, opening=None):
        if not ok:
            r -= MISS
        self.result = {"caught": ok, "why": why, "depth": depth, "opening": opening, "t": self.t, "travel": self.travel}
        return self._obs(), float(r), True, False, dict(self.result)

    def visible(self):
        return [s for s in self.seen if s[0] <= self.t - OBS_DELAY + 1e-9]

    def _obs(self):
        f = self.fc
        seen = self.visible()
        if seen:
            recent = [x for x in seen if x[0] >= seen[-1][0] - 0.25]
            if len(recent) >= 3:
                t = np.array([x[0] for x in recent]) - recent[-1][0]
                coef = np.linalg.lstsq(np.c_[np.ones_like(t), t], np.array([x[1] for x in recent]), rcond=None)[0]
                bp, bv = coef[0], coef[1]
            else:
                bp, bv = seen[-1][1], np.array([f["vx"], f["vy"]])
        else:
            bp, bv = np.zeros(2), np.zeros(2)
        g = self.g
        return np.array([self.t, f["x"] / 0.4, f["t"] - self.t, f["vx"] / 0.2, f["vy"] / 0.2,
                         f["sig_s"] / SIG_S, f["sig_t"] / SIG_T, f["sig_v"] / SIG_V,
                         g[0] / 0.4, g[1] / ZONE, (self.w - GAP_MIN) / (GAP_MAX - GAP_MIN),
                         float(bool(seen)), bp[0] / 0.4, bp[1] / ZONE, bv[0] / 0.2, bv[1] / 0.2,
                         (bp[0] - g[0]) / 0.1 if seen else 0.0], np.float32)

    def close(self):
        pass


# ---- scripted reference policies (fixed rules) -----------------------------------------------------
def _go(env, target):
    v = (np.asarray(target) - env.grip_xy()) / DT / GRIP_SPEED
    n = np.linalg.norm(v)
    return v / n if n > 1 else v


def _predict(x, y, vx, vy, aim_y):
    k = max((aim_y - y) / vy, 0.05) if vy > 0.01 else 0.05
    return x + vx * k, k


AIM = 0.5 * ZONE                 # scripted policies wait for the ball at mid-zone
ENTRY = AIM - 0.02               # where the ball passes the fingers' near ends: line up for that point


def _ball_now(env, obs):
    """Latest sighting carried forward over the sight lag."""
    seen = env.visible()
    x, y = seen[-1][1]
    vx, vy = obs[14] * 0.2, obs[15] * 0.2
    lag = env.t - seen[-1][0]
    return np.array([x + vx * lag, y + vy * lag]), vx, vy


def _close_when_between(env, now) -> float:
    d = now - env.grip_xy()
    return 1.0 if abs(d[0]) < env.gap() / 2 - BALL_R * 0.7 and abs(d[1]) < 0.01 else -1.0


def reactive(env, obs):
    """Waits at rest; once it sees the ball, lines up with where it will cross mid-zone, half open,
    and grasps when the ball is between the fingers."""
    if not env.visible():
        return np.array([0, 0, 0.0, -1.0])
    now, vx, vy = _ball_now(env, obs)
    tx, _ = _predict(now[0], now[1], vx, vy, ENTRY) if now[1] < ENTRY else (now[0], 0)
    return np.r_[_go(env, (tx, AIM)), 0.0, _close_when_between(env, now)]


def trust_forecast(env, obs):
    """Lines up with where the forecast says the ball crosses mid-zone, fairly narrow; corrects from
    sightings once it sees the ball."""
    if env.visible():
        a = reactive(env, obs)
        a[2] = -0.4
        return a
    f = env.fc
    tx, _ = _predict(f["x"], 0.0, f["vx"], f["vy"], ENTRY)
    return np.r_[_go(env, (tx, AIM)), -0.4, -1.0]


def run_episode(env, policy, options=None, seed=None, frames=None):
    obs, _ = env.reset(seed=seed, options=options)
    total = 0.0
    while True:
        a = policy(env, obs)
        if frames is not None:
            frames.append(env.render())
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            if frames is not None:
                frames.append(env.render())
            return dict(info, reward=total)

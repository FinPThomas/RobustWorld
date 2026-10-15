"""Exit-and-catch in PyBullet: a ball comes out from under the plank and a two-finger gripper has to
grasp it and lift it before it rolls out of the catch zone.

Scene (metres; 1 ball diameter = 4 cm, set from the ball's ~44 px diameter mid-table, so 1 px = 0.9 mm):
flat table (no slope), the plank as a box along y < 0 whose edge is y = 0, and a ball that comes out
from under it at the real exits of the tracker (exits.json: where along the edge, when after the
decision point, how fast), resampled with a little jitter in training; in the real replay
(evaluate.py) each real pass's own exit, after which the ball rolls freely. The gripper is a floating two-finger
gripper (x, y, z slides plus two finger slides) working in the catch zone 0 <= y <= ZONE. It sees the
ball only after the ball comes out, two video frames late.

Each episode the policy is also told a forecast of where, when and how fast the ball comes out, and how
uncertain that forecast is (one sigma for each). In simulation the forecast is the true exit plus noise
of exactly that size, so the stated uncertainty is honest. Nothing here sees a world model's real
forecasts against real outcomes (CLAUDE.md rule 1); evaluate.py feeds those in only to score.

Actions (every 1/16 s, all in [-1, 1]): gripper velocity in x and y, finger opening, and "grasp now"
(> 0.5, honoured once the gripper has seen the ball). A grasp closes the fingers and lifts; it is the
episode's one attempt.

Reward: a ball lifted and held earns 1 + EARLY x (how early in the zone it was grasped, 0..1) minus
WIDE x (how wide the fingers were open, 0..1). A miss costs MISS: the ball dropped or slipped, left the
zone, or was touched by the gripper before the grasp (a knocked rolling ball is lost, so the gripper
cannot just block it and pick it up at rest). Moving costs a little. A wide opening also takes longer
to close, so it is a real trade-off.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p

HERE = Path(__file__).resolve().parent
M_PER_PX = 0.04 / 44             # ball diameter 4 cm ~ 44 px
BALL_R = 0.02
ZONE = 0.12                      # catch zone depth past the plank edge: 3 ball diameters
DT = 1 / 16                      # one video frame per policy step
SUBSTEPS = 8                     # physics at 128 Hz
GRIP_SPEED = 0.25                # m/s
GRIP_SPEED_Y = 0                 # > 0: speed towards / away from the plank (m/s) when it differs from
                                 # the lateral GRIP_SPEED: a slow y means committing to a depth early
SEEN_SPEED_X = 0                 # > 0: once the ball is in sight, sideways speed is capped at this (m/s): the
                                 # gripper must already be lined up on its forecast, it can only fine-tune
GAP_MIN, GAP_MAX = 0.045, 0.13   # finger gap (inner faces); the ball is 0.04 wide
FINGER_SPEED = 0.25              # m/s per finger when closing
EARLY, WIDE, MISS, MOVE = 0.5, 0.3, 1.0, 0.05
SIG_S, SIG_T, SIG_V = 0.12, 0.3, 0.5     # forecast sigma at uncertainty 1: m along the edge, s, fraction of speed
OBS_NOISE = 0.003                # m
OBS_DELAY = 0.125                # s: seeing the ball lags by 2 frames
MAX_T = 2.5
U_INDEP = 0                      # 1: training draws a separate uncertainty for where, when and speed
# BLOCK = 1: the pass may be blocked (never comes out), and the gripper must commit before it can see.
# The forecast adds p_out, how likely the ball is to come out. The gripper waits parked PARK m beyond the
# zone, too far to get there after seeing the ball, so it has to set off on the forecast. Setting off
# (leaving the park by more than 3 cm) is a commitment: a blocked pass then costs COMMIT (a wasted trip),
# a missed ball costs MISS. Staying parked costs nothing either way (a ball let go).
BLOCK, PARK, COMMIT, BLOCK_END = 0, 0.15, 1.0, 1.6
LETGO = 0.0                      # BLOCK: cost of letting a ball that comes out go by (never setting off)
HOVER_Z = 0.0                   # gripper frame z while hunting (fingers reach the table)
LIFT_Z = 0.08

# Experiment overrides (overnight.py): GRIPPER_PARAMS='{"EARLY": 1.0}' replaces these constants, in every
# training process (subprocesses inherit the environment variable).
globals().update({k: v for k, v in json.loads(os.environ.get("GRIPPER_PARAMS", "{}")).items() if k in globals()})

GRIPPER_URDF = """<?xml version="1.0"?>
<robot name="gripper">
  <link name="world"/>
  <link name="sx"><inertial><mass value="0.1"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="sy"><inertial><mass value="0.1"/><inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="palm">
    <inertial><mass value="0.3"/><inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.075"/><geometry><box size="0.16 0.04 0.012"/></geometry><material name="grey"><color rgba="0.4 0.4 0.45 1"/></material></visual>
    <collision><origin xyz="0 0 0.075"/><geometry><box size="0.16 0.04 0.012"/></geometry></collision>
  </link>
  <link name="left">
    <inertial><mass value="0.05"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="-0.006 0 0.037"/><geometry><box size="0.012 0.04 0.07"/></geometry><material name="dark"><color rgba="0.15 0.15 0.2 1"/></material></visual>
    <collision><origin xyz="-0.006 0 0.037"/><geometry><box size="0.012 0.04 0.07"/></geometry></collision>
  </link>
  <link name="right">
    <inertial><mass value="0.05"/><inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.006 0 0.037"/><geometry><box size="0.012 0.04 0.07"/></geometry><material name="dark"><color rgba="0.15 0.15 0.2 1"/></material></visual>
    <collision><origin xyz="0.006 0 0.037"/><geometry><box size="0.012 0.04 0.07"/></geometry></collision>
  </link>
  <joint name="x" type="prismatic"><parent link="world"/><child link="sx"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="200" velocity="2"/></joint>
  <joint name="y" type="prismatic"><parent link="sx"/><child link="sy"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="200" velocity="2"/></joint>
  <joint name="z" type="prismatic"><parent link="sy"/><child link="palm"/><axis xyz="0 0 1"/><limit lower="0" upper="0.3" effort="200" velocity="2"/></joint>
  <joint name="fl" type="prismatic"><parent link="palm"/><child link="left"/><axis xyz="1 0 0"/><limit lower="-0.07" upper="-0.004" effort="20" velocity="0.5"/></joint>
  <joint name="fr" type="prismatic"><parent link="palm"/><child link="right"/><axis xyz="1 0 0"/><limit lower="0.004" upper="0.07" effort="20" velocity="0.5"/></joint>
</robot>
"""
JX, JY, JZ, JL, JR = range(5)


def load_exits(path: Path = HERE / "exits.json") -> dict:
    return json.loads(path.read_text())


def exit_to_m(e: dict) -> dict:
    """An exit from exits.json (px, px/s along / past the plank edge) in metres in this scene."""
    return {"x": e["s"] * M_PER_PX, "t": e["t"], "vx": e["vs"] * M_PER_PX, "vy": e["vd"] * M_PER_PX}


def noisy_forecast(e: dict, o: dict, rng) -> dict:
    """The true exit plus Gaussian noise of exactly the stated sigmas. One uncertainty u for all three
    parts, or (U_INDEP, training only) a separate one for where, when and speed, so the policy learns to
    weigh each part by its own stated sigma (a real forecast can place the ball well but time it badly)."""
    u = float(o.get("u", rng.uniform(0, 1)))
    us, ut, uv = rng.uniform(0, 1, 3) if U_INDEP and "u" not in o else (u, u, u)
    sv = SIG_V * uv * np.hypot(e["vx"], e["vy"])
    return {"x": e["x"] + rng.normal(0, SIG_S * us + 1e-9), "t": e["t"] + rng.normal(0, SIG_T * ut + 1e-9),
            "vx": e["vx"] + rng.normal(0, sv + 1e-9), "vy": e["vy"] + rng.normal(0, sv + 1e-9),
            "sig_s": SIG_S * us, "sig_t": SIG_T * ut, "sig_v": SIG_V * uv, "u": u}


def draw_pass(ex: list, o: dict, rng) -> tuple[dict, dict]:
    """A simulated pass: (true exit, forecast). The exit is a real one with a little jitter. With BLOCK, the
    forecast also states p_out ~ Beta(0.5, 0.5) (mostly confident, sometimes not) and the ball comes out with
    exactly that probability, so p_out is honest; a blocked pass gets t = inf (it never comes out), and its
    forecast still names an exit, as a mistaken world model would."""
    e = dict(ex[rng.integers(len(ex))])
    j = rng.normal(1, 0.1, 2)
    e = {"x": e["x"] + rng.normal(0, 0.01), "t": max(e["t"] + rng.normal(0, 0.03), 0.0),
         "vx": e["vx"] * j[0], "vy": max(e["vy"] * j[1], 0.03)}
    fc = noisy_forecast(e, o, rng)
    if BLOCK:
        fc["p_out"] = float(o.get("p_out", rng.beta(0.5, 0.5)))
        if rng.uniform() >= fc["p_out"]:
            e = dict(e, t=np.inf)
    return e, fc


def block_obs(env) -> list:
    """The extra observation with BLOCK: the forecast's p_out."""
    return [env.fc.get("p_out", 0.5)] if BLOCK else []


def park_xy(prior: dict) -> np.ndarray:
    return np.array([prior["x"], ZONE + PARK]) if BLOCK else np.array([prior["x"], ZONE / 2])


def y_max() -> float:
    return ZONE + PARK if BLOCK else ZONE


class CatchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, exits: dict | None = None, seed: int | None = None, gui: bool = False):
        self.exits = exits or load_exits()
        self.ex = [exit_to_m(e) for e in self.exits["exits"]]
        pr = exit_to_m(dict(self.exits["prior"], s=self.exits["prior"]["s"]))
        self.prior = pr
        self.rest = park_xy(pr)
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-10, 10, (17 + bool(BLOCK),), np.float32)
        self.rng = np.random.default_rng(seed)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        self._build()

    def _build(self):
        c = self.cid
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setTimeStep(DT / SUBSTEPS, physicsClientId=c)
        table = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.01], physicsClientId=c)
        tv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.01], rgbaColor=[0.75, 0.6, 0.45, 1], physicsClientId=c)
        self.table = p.createMultiBody(0, table, tv, [0.2, 0.1, -0.01], physicsClientId=c)
        pv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.05, 0.004], rgbaColor=[0.55, 0.35, 0.2, 1], physicsClientId=c)
        self.plank = p.createMultiBody(0, -1, pv, [0.2, -0.05, 0.06], physicsClientId=c)       # visual only, above the ball
        zv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.001, 0.0005], rgbaColor=[0.9, 0.2, 0.2, 1], physicsClientId=c)
        p.createMultiBody(0, -1, zv, [0.2, ZONE, 0.0006], physicsClientId=c)                   # end of the catch zone
        bs = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R, physicsClientId=c)
        bv = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R, rgbaColor=[0.1, 0.3, 0.9, 1], physicsClientId=c)
        self.ball = p.createMultiBody(0.05, bs, bv, [0, -0.3, BALL_R], physicsClientId=c)
        p.changeDynamics(self.ball, -1, lateralFriction=0.8, rollingFriction=1e-5, spinningFriction=1e-5,
                         physicsClientId=c)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
            f.write(GRIPPER_URDF)
        self.grip = p.loadURDF(f.name, useFixedBase=True, physicsClientId=c)
        for j in (JL, JR):
            p.changeDynamics(self.grip, j, lateralFriction=1.5, physicsClientId=c)

    # ---- helpers ------------------------------------------------------------------------------------
    def _drive(self, j, target, vmax, force=200):
        p.setJointMotorControl2(self.grip, j, p.POSITION_CONTROL, targetPosition=float(target),
                                maxVelocity=vmax, force=force, physicsClientId=self.cid)

    def _set_gap(self, gap, vmax=1.0, force=20):
        h = gap / 2 + 0.0  # inner faces sit at the joint positions
        self._drive(JL, -h, vmax, force), self._drive(JR, h, vmax, force)

    def gap(self) -> float:
        return p.getJointState(self.grip, JR, physicsClientId=self.cid)[0] - p.getJointState(self.grip, JL, physicsClientId=self.cid)[0]

    def grip_xy(self) -> np.ndarray:
        return np.array([p.getJointState(self.grip, JX, physicsClientId=self.cid)[0],
                         p.getJointState(self.grip, JY, physicsClientId=self.cid)[0]])

    def ball_pos(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self.ball, physicsClientId=self.cid)[0])

    def _sim(self, n=SUBSTEPS):
        for _ in range(n):
            if not self.out and self.t >= self.true["t"] - 1e-9:
                self._release()
            p.stepSimulation(physicsClientId=self.cid)
            self.t += DT / SUBSTEPS
            if self.out:
                self.seen.append((self.t, self.ball_pos()[:2] + self.rng.normal(0, OBS_NOISE, 2)))
                if not self.grasping and p.getContactPoints(self.ball, self.grip, physicsClientId=self.cid):
                    self.knocked = True

    def _release(self):
        e = self.true
        self.out = True
        p.resetBasePositionAndOrientation(self.ball, [e["x"], 0.0, BALL_R], [0, 0, 0, 1], physicsClientId=self.cid)
        p.resetBaseVelocity(self.ball, [e["vx"], e["vy"], 0], [-e["vy"] / BALL_R, e["vx"] / BALL_R, 0],
                            physicsClientId=self.cid)

    # ---- episode ------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = options or {}
        if "exit" in o:                              # replay: a real exit (px), or None if blocked, and a forecast
            self.true = exit_to_m(o["exit"]) if o["exit"] is not None else dict(self.prior, t=np.inf)
            self.fc = dict(o["forecast"])
        else:
            self.true, self.fc = draw_pass(self.ex, o, self.rng)
        c = self.cid
        p.resetBasePositionAndOrientation(self.ball, [0, -0.3, BALL_R], [0, 0, 0, 1], physicsClientId=c)
        p.resetBaseVelocity(self.ball, [0, 0, 0], [0, 0, 0], physicsClientId=c)
        for j, v in ((JX, self.rest[0]), (JY, self.rest[1]), (JZ, HOVER_Z), (JL, -GAP_MIN / 2), (JR, GAP_MIN / 2)):
            p.resetJointState(self.grip, j, v, physicsClientId=c)
        self._drive(JZ, HOVER_Z, 1.0)
        self._set_gap(GAP_MIN)
        self.t, self.out, self.seen, self.travel = 0.0, False, [], 0.0
        self.grasping, self.knocked, self.result, self.committed = False, False, None, False
        return self._obs(), {}

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        g = self.grip_xy()
        v = a[:2] * GRIP_SPEED
        n = np.linalg.norm(v)
        v = v * GRIP_SPEED / n if n > GRIP_SPEED else v
        if GRIP_SPEED_Y:                             # fast along the edge, slow towards / away from the plank
            v = np.array([a[0] * GRIP_SPEED, a[1] * GRIP_SPEED_Y])
        if SEEN_SPEED_X and self.visible():
            v[0] = np.clip(v[0], -SEEN_SPEED_X, SEEN_SPEED_X)
        if (g[1] <= 0.0 and v[1] < 0) or (g[1] >= y_max() and v[1] > 0):
            v[1] = 0.0                               # stays inside the catch zone (and its park)
        for j, vj in ((JX, v[0]), (JY, v[1])):
            p.setJointMotorControl2(self.grip, j, p.VELOCITY_CONTROL, targetVelocity=float(vj), force=200,
                                    physicsClientId=self.cid)
        gap = GAP_MIN + (a[2] + 1) / 2 * (GAP_MAX - GAP_MIN)
        self._set_gap(gap, vmax=0.5)
        self._sim()
        r = -MOVE * float(np.linalg.norm(self.grip_xy() - g)) / 0.1
        self.travel += float(np.linalg.norm(self.grip_xy() - g))
        self.committed |= bool(BLOCK) and self.grip_xy()[1] < y_max() - 0.03
        if a[3] > 0.5 and self.visible():
            return self._grasp(r)
        if not np.isfinite(self.true["t"]) and self.t >= BLOCK_END:
            return self._blocked(r)
        if self.knocked:
            return self._finish(r, False, "knocked the ball")
        b = self.ball_pos()
        if self.out and (b[1] > ZONE + BALL_R or b[1] < -0.05 or abs(b[0] - 0.2) > 0.45):
            return self._finish(r, False, "left the zone")
        if self.t >= MAX_T:
            return self._finish(r, False, "time out")
        return self._obs(), r, False, False, {}

    def _grasp(self, r):
        """Close the fingers (FINGER_SPEED each), then lift; success = ball held clear of the table."""
        b0 = self.ball_pos()
        depth = float(np.clip(b0[1] / ZONE, 0, 1)) if self.out else 1.0
        open_frac = float(np.clip((self.gap() - GAP_MIN) / (GAP_MAX - GAP_MIN), 0, 1))
        self.grasping = True
        self._set_gap(0.0, vmax=FINGER_SPEED, force=15)
        g = self.grip_xy()
        self._drive(JX, g[0], 0.5), self._drive(JY, g[1], 0.5)
        steps = int(np.ceil(self.gap() / 2 / FINGER_SPEED / (DT / SUBSTEPS))) + 10
        self._sim(steps)
        self._drive(JZ, LIFT_Z, 0.4)
        self._sim(int(0.35 / (DT / SUBSTEPS)))
        b = self.ball_pos()
        held = b[2] > BALL_R + 0.03 and bool(p.getContactPoints(self.ball, self.grip, physicsClientId=self.cid))
        if held:
            rew = 1 + EARLY * (1 - depth) - WIDE * open_frac
            return self._finish(r + rew, True, "held", depth=depth, opening=open_frac)
        return self._finish(r, False, "grasp missed", depth=depth, opening=open_frac)

    def _finish(self, r, ok, why, depth=None, opening=None):
        if not ok and BLOCK and not self.committed:
            why = "let it go"                        # never set off: no attempt; costs LETGO
            r -= LETGO
        elif not ok:
            r -= MISS
        self.result = {"caught": ok, "why": why, "depth": depth, "opening": opening, "t": self.t, "travel": self.travel,
                       "out": True, "committed": self.committed}
        return self._obs(), float(r), True, False, dict(self.result)

    def _blocked(self, r):
        """The ball never came out: a wasted trip if the gripper set off, else nothing."""
        r -= COMMIT if self.committed else 0.0
        self.result = {"caught": False, "why": "blocked: wasted trip" if self.committed else "blocked: stayed parked",
                       "depth": None, "opening": None, "t": self.t, "travel": self.travel, "out": False,
                       "committed": self.committed}
        return self._obs(), float(r), True, False, dict(self.result)

    @staticmethod
    def _fit(seen, f):
        """Ball position and velocity from a straight-line fit to the sightings of the last 0.25 s (one
        sighting per physics step), which averages out the tracker noise."""
        recent = [x for x in seen if x[0] >= seen[-1][0] - 0.25]
        if len(recent) < 3:
            return seen[-1][1], np.array([f["vx"], f["vy"]])
        t = np.array([x[0] for x in recent]) - recent[-1][0]
        xy = np.array([x[1] for x in recent])
        A = np.c_[np.ones_like(t), t]
        coef = np.linalg.lstsq(A, xy, rcond=None)[0]
        return coef[0], coef[1]

    def visible(self) -> list:
        return [s for s in self.seen if s[0] <= self.t - OBS_DELAY + 1e-9]

    def _obs(self):
        f = self.fc
        seen = self.visible()
        if seen:
            bp, bv = self._fit(seen, f)
        else:
            bp, bv = np.zeros(2), np.zeros(2)
        g = self.grip_xy()
        return np.array([self.t, f["x"] / 0.4, f["t"] - self.t, f["vx"] / 0.2, f["vy"] / 0.2,
                         f["sig_s"] / SIG_S, f["sig_t"] / SIG_T, f["sig_v"] / SIG_V,
                         g[0] / 0.4, g[1] / ZONE, (self.gap() - GAP_MIN) / (GAP_MAX - GAP_MIN),
                         float(bool(seen)), bp[0] / 0.4, bp[1] / ZONE, bv[0] / 0.2, bv[1] / 0.2,
                         (bp[0] - g[0]) / 0.1 if seen else 0.0, *block_obs(self)], np.float32)

    def render(self, width=480, height=360):
        c = self.cid
        view = p.computeViewMatrix([0.2, 0.33, 0.42], [0.2, 0.03, 0.0], [0, -1, 0], physicsClientId=c)
        proj = p.computeProjectionMatrixFOV(55, width / height, 0.05, 2.0, physicsClientId=c)
        img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=c)[2]
        return np.asarray(img, np.uint8).reshape(height, width, 4)[..., :3]

    def close(self):
        try:
            p.disconnect(self.cid)
        except p.error:
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
    if f.get("p_out", 1.0) < 0.5:               # BLOCK: the forecast says it stays hidden: stay parked
        return np.array([0.0, 0.0, -0.4, -1.0])
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

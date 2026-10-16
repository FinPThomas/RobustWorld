"""The exit-and-catch gripper: the physics grasp works, and world-model forecasts are read by fixed rules
that never look at outcomes."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("gymnasium")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models" / "gripper"))
import env as E  # noqa: E402
import evaluate as V  # noqa: E402
from exits import Edge  # noqa: E402


def test_a_grasp_around_a_still_ball_lifts_it():
    env = E.CatchEnv(seed=0)
    env.reset(options={"exit": {"s": env.rest[0] / E.M_PER_PX, "t": 0.0, "vs": 0.0, "vd": 1.0}, 
                       "forecast": V.prior_forecast(env.exits["prior"])})
    for _ in range(3):                    # the gripper may only grasp once it has seen the ball (sight lag)
        env.step([0, 0, 1, -1])
    g = env.grip_xy()
    E.p.resetBasePositionAndOrientation(env.ball, [g[0], g[1], E.BALL_R], [0, 0, 0, 1], physicsClientId=env.cid)
    E.p.resetBaseVelocity(env.ball, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)
    *_, info = env.step([0, 0, 1, 1])
    env.close()
    assert info["caught"]


def test_forecast_ignores_outcome_labels():
    import json
    poly = json.loads((REPO / "configs" / "scenes" / "whole.json").read_text())["occluder_polygon"]
    edge = Edge(poly, "R")
    row = {"outcome": "through", "imagined_pos": [[400.0, 250.0, 0.9]] * 3 + [[60.0, 250.0, 0.9], [30.0, 250.0, 0.9]]}
    prior = E.load_exits()["prior"]
    a = V.wm_forecast(row, edge, prior)
    b = V.wm_forecast(dict(row, outcome="hidden"), edge, prior)
    assert a is not None and a == b
    assert V.wm_forecast(dict(row, imagined_pos=[[400.0, 250.0, 0.9]] * 5), edge, prior) is None

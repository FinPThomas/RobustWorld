"""The pick-up test is a fixed rule: it fits nothing and never reads outcomes to drive the gripper."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models" / "vjepa2"))
import pickup  # noqa: E402

POLY = json.loads((REPO / "configs" / "scenes" / "whole.json").read_text())["occluder_polygon"]


def _row(side, through, outcome="through"):
    xs = np.linspace(330, 20, 12) if side == "R" else np.linspace(30, 480, 12)
    if not through:
        xs = np.full(12, 330.0 if side == "R" else 30.0)
    track = [[float(x), 250.0] for x in xs]
    return {"clip_id": f"{side}{through}", "side_in": side, "outcome": outcome,
            "true_pos": track, "true_pos_context": [[400.0, 250.0]] * 12,
            "imagined_pos": [p + [0.9] for p in track], "imagined_far": [0.9 if through else 0.0] * 12}


def test_catch_line_is_beyond_the_plank_on_the_far_side():
    right, left = pickup.Line(POLY, "R"), pickup.Line(POLY, "L")
    assert right.past([20, 250]) > 0 > right.past([330, 250])
    assert left.past([480, 250]) > 0 > left.past([30, 250])


def test_perfect_forecast_matches_oracle_and_ignores_outcome_labels():
    rows = [_row(s, t) for s in "RL" for t in (True, False)]
    e = pickup.evaluate(rows, POLY, "model", None)
    assert pickup.summarise(e)["pickup_auc"] == 1.0
    relabelled = [dict(r, outcome="hidden") for r in rows]
    e2 = pickup.evaluate(relabelled, POLY, "model", None)
    assert all(np.array_equal(e[k], e2[k]) for k in e)


def test_slow_gripper_cannot_reach_a_far_crossing():
    assert pickup.gripper_at(0.0, 300.0, 0.5, 100) == 50.0
    assert pickup.gripper_at(0.0, 300.0, 0.5, None) == 300.0

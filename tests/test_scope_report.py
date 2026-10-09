"""scope_report.py (narrow gap, both sides) and slope.py (table slope) from per-clip files."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "vjepa2"))
import scope_report  # noqa: E402


def clip(i, side, outcome, y, acc, imagined_acc):
    """A ball on open table well left of the plank (x 20-100 px), moving at 10 px/step with acceleration acc."""
    sign = -1 if side == "R" else 1               # from the right it rolls left
    x0 = 100 if side == "R" else 20
    xs = [x0 + sign * (10 * t + 0.5 * acc * t * t) for t in range(8)]
    xi = [x0 + sign * (10 * t + 0.5 * imagined_acc * t * t) for t in range(8)]
    p = 0.9 if outcome == "through" else 0.1
    return {"clip_id": f"c{i}", "outcome": outcome, "side_in": side,
            "real_one_ball_far": [p], "imagined_one_ball_far": [p * 0.8], "real_one_ball_near": [0.0],
            "imagined_one_ball_near": [0.0],
            "true_pos": [[x, 300.0] for x in xs], "real_pos": [[x, 300.0, 0.9] for x in xs],
            "imagined_pos": [[x, 300.0, 0.5] for x in xi]}, {"clip_id": f"c{i}", "entry_xy": [200.0, y]}


def test_scope_report(tmp_path):
    rows, cross = [], []
    for i in range(12):
        side = "R" if i < 8 else "L"
        outcome = "through" if i % 2 else "hidden"
        y = 340.0 if outcome == "through" else 250.0
        acc = 1.0 if side == "R" else -1.0             # downhill from the right, uphill from the left
        r, c = clip(i, side, outcome, y, acc, acc * 0.5)
        rows.append(r), cross.append(c)
    plan = tmp_path / "plan"
    for name in ("both-before", "commit_both-after"):
        (plan / "scores" / name).mkdir(parents=True)
        (plan / "scores" / name / "per_clip.json").write_text(json.dumps(rows))
        (plan / "blocker" / name).mkdir(parents=True)
        (plan / "blocker" / name / "crossing.json").write_text(json.dumps({"passes": cross}))
    assert scope_report.main(["--plan", str(plan)]) == 0
    res = json.loads((plan / "scope" / "scope.json").read_text())
    narrow = res["height"]["commit_both-after / R"]["narrow gap (315-367)"]
    assert narrow["n"] == 4 and abs(narrow["mean"] - 0.72) < 1e-6
    assert (plan / "scope" / "height_profile.png").exists()


def test_slope_recovers_a_known_slope(tmp_path):
    """Balls rolling left speed up by s, rolling right slow down by s, with friction growing with speed: the
    tracker's tracks give s back; an imagined track with half the slope gives about half."""
    import numpy as np
    import slope

    rng = np.random.default_rng(0)
    s_true, rows = 0.4, []

    def track(x0, v0, sign, s, n=12):
        xs, x, v = [], x0, v0
        for _ in range(n):
            xs.append(x)
            x += sign * v
            v += -0.02 * v + (s if sign < 0 else -s)
            v = max(v, 0.5)
        return xs

    for i in range(60):
        sign = -1 if i % 2 == 0 else 1                       # left (downhill) or right (uphill)
        x0 = 500 if sign < 0 else 300                        # right of the plank (x > 270), on open table
        v0 = rng.uniform(6, 14)
        xt, xi = track(x0, v0, sign, s_true), track(x0, v0, sign, s_true / 2)
        y = 300.0
        rows.append({"clip_id": f"c{i}", "outcome": "through", "side_in": "R",
                     "true_pos_context": [[x0 - sign * 5 * k, y] for k in range(6, 0, -1)][:0],
                     "true_pos": [[x, y] for x in xt], "real_pos": [[x, y, 0.9] for x in xt],
                     "imagined_pos": [[x, y, 0.5] for x in xi]})
    for name in ("both-before", "commit_both-after"):
        (tmp_path / "scores" / name).mkdir(parents=True)
        (tmp_path / "scores" / name / "per_clip.json").write_text(json.dumps(rows))
    assert slope.main(["--plan", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "slope" / "slope.json").read_text())["sources"]
    s0 = res["S0 tracker (all steps)"]["1d"]["s"]
    assert abs(s0 - s_true) < 0.05
    assert abs(res["S0 tracker (all steps)"]["2d"]["s_left"] - s_true) < 0.05
    assert abs(res["S4 imagined, A both directions"]["1d"]["s"] - s_true / 2) < 0.05
    assert (tmp_path / "slope" / "acc_vs_speed.png").exists()

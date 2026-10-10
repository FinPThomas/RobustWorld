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


def test_slope_test_recovers_slope_through_the_camera_tilt(tmp_path):
    """A tilted camera (ball radius changing across the frame) and a known slope: slope_test.py maps positions onto
    the table with the ball's size and gives the slope back; without that mapping the pixels would mislead."""
    import csv

    import numpy as np
    import slope_test

    a_r, b_r, c_r = -0.004, 0.012, 5.0                      # radius px = a x + b y + c (bigger lower down)
    s_true, fric = 0.01, 0.006                              # ball diameters / step^2
    rng = np.random.default_rng(1)
    fps = 16.0
    dt = 1 / (fps * slope_test.STEP_S)                      # steps per frame

    def to_image(q):                                        # inverse of Table: p = 2 q r(p)
        q = np.asarray(q, float)
        M = np.eye(2) - 2 * np.outer(q, [a_r, b_r])
        return np.linalg.solve(M, 2 * q * c_r)

    rows, frame = [], 0
    for k in range(80):
        y = rng.uniform(120, 400)
        left = k % 2 == 0
        x0 = 480.0 if left else 300.0
        q = np.array([x0, y]) / (2 * (a_r * x0 + b_r * y + c_r))
        v = rng.uniform(0.25, 0.6) * (-1 if left else 1)
        for _ in range(40):                                 # gap: ball out of view
            rows.append({"frame": frame, "t": frame / fps, "visible": 0, "x": "", "y": "", "radius": "", "fg_area": 0})
            frame += 1
        while True:
            p = to_image(q)
            if not (296 < p[0] < 484):
                break
            r = a_r * p[0] + b_r * p[1] + c_r
            rows.append({"frame": frame, "t": frame / fps, "visible": 1, "x": p[0] + rng.normal(0, 0.3),
                         "y": p[1] + rng.normal(0, 0.3), "radius": r + rng.normal(0, 0.1), "fg_area": 0})
            frame += 1
            along = (s_true if left else -s_true) - fric    # acceleration along the motion: slope -/+ friction
            v += (-1 if left else 1) * along * dt
            q = q + np.array([v * dt, 0.0])
    track = tmp_path / "track.csv"
    with track.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t", "visible", "x", "y", "area", "radius", "occ_dist", "n_blobs",
                                          "fg_area"])
        w.writeheader()
        for r in rows:
            w.writerow({"area": "", "occ_dist": "", "n_blobs": 1, **r})
    assert slope_test.main(["--plan", str(tmp_path / "plan"), "--track", str(track)]) == 0
    res = json.loads((tmp_path / "plan" / "slope_test" / "slope_test.json").read_text())
    assert res["calibration"]["ok"] and res["calibration"]["r2"] > 0.9
    assert abs(res["real"]["s"] - s_true) < 0.004, res["real"]

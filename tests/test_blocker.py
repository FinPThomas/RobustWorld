"""Blocker outline in the scene file, and the crossing geometry behind the blocker figures."""

import json
import sys
from pathlib import Path

import numpy as np

from robust_world.scene import blocker_array, load_scene

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vjepa2"))
import blocker_figs  # noqa: E402

FPS = 30.0
# Vertical bar occluder at x 200-260, blocker under it at y 200-300.
OCCLUDER = [[200, 0], [260, 0], [260, 512], [200, 512]]
BLOCKER = [[210, 200], [250, 200], [250, 300], [210, 300]]


def write_scene(tmp_path, **extra):
    p = tmp_path / "scene.json"
    p.write_text(json.dumps({"occluder_polygon": OCCLUDER, **extra}))
    return load_scene(p)


def test_blocker_states(tmp_path):
    assert write_scene(tmp_path)["blocker"] is None                           # not recorded
    assert write_scene(tmp_path, blocker_polygon="none")["blocker"] is None   # no blocker
    s = write_scene(tmp_path, blocker_polygon=BLOCKER)
    assert s["blocker"].shape == (4, 2)
    assert blocker_array({"blocker_polygon": BLOCKER}).dtype == np.float32


def roll(y, n=60, x0=500.0, vx=-8.0, r=10.0):
    """Ball rolling left along row y, as tracker rows (visible, x, y, r)."""
    return [(1, x0 + vx * i, float(y), r) for i in range(n)]


def test_line_fit_and_crossing():
    fit = blocker_figs.line_fit(roll(250), 0, 20, FPS)
    assert abs(fit[0] - 340) < 1e-6 and abs(fit[2] + 8) < 1e-6
    entry, exit_ = blocker_figs.crossing(fit, OCCLUDER)
    assert abs(entry[0] - 260) <= 2 and abs(exit_[0] - 200) <= 2
    assert blocker_figs.crossing((400.0, 250.0, 8.0, 0.0, 10.0), OCCLUDER) is None   # rolling away
    assert blocker_figs.crossing(None, OCCLUDER) is None
    assert blocker_figs.line_fit([(0, 0, 0, 0)] * 30, 0, 20, FPS) is None            # never seen


def test_hits_blocker():
    assert blocker_figs.hits_blocker((260, 250), (199, 250), 10, BLOCKER)
    assert not blocker_figs.hits_blocker((260, 400), (199, 400), 10, BLOCKER)
    assert blocker_figs.hits_blocker((260, 195), (199, 195), 10, BLOCKER)          # grazes the edge


def test_along_plank_and_summary():
    axis = blocker_figs.plank_axis(OCCLUDER)                 # top-edge midpoint (230, 0), pointing down
    assert abs(blocker_figs.along((230, 120), axis) - 120) < 1e-6
    assert abs(blocker_figs.plank_length(OCCLUDER) - 512) < 1e-6
    s = blocker_figs.summarise([], OCCLUDER, BLOCKER)
    assert s["blocker_span_px"] == [200.0, 300.0] and s["auroc_imagined"] is None


def test_pass_row_reads_far_side_peak():
    clip = {"clip_id": "c", "outcome": "hidden", "pass_id": 0, "context_frames": [0, 15], "n_frames": 32,
            "fps": 16, "source_fps": FPS}
    passes = {0: {"occlusion_start_frame": 21}}
    probe = {"real_far": [0.0, 0.1], "imagined_far": [0.2, 0.7]}
    row = blocker_figs.pass_row(clip, probe, roll(250, n=200), passes, OCCLUDER, BLOCKER)
    assert row["true"] == 1 and row["real"] == 0.9 and row["imagined"] == 0.3
    assert row["known_blocker"] == 1 and abs(row["exit_along"] - 250) < 1


# Bounce as a third outcome in the shared evaluation.
from robust_world.eval import ball  # noqa: E402


def test_returned_needs_gone_then_back():
    assert ball.returned([0.9, 0.9, 0.9, 0.9]) < 0.5         # still rolling in at the start of the target
    assert ball.returned([0.9, 0.2, 0.0, 0.0]) < 0.5         # went under and stayed (hidden or through)
    assert ball.returned([0.9, 0.1, 0.0, 0.9]) > 0.5         # gone, then back out: bounce


def test_outcome_metrics_with_bounce():
    rows = [{"outcome": "through", "far": [0, 0.9], "near": [0.9, 0]},
            {"outcome": "hidden", "far": [0, 0], "near": [0.9, 0]},
            {"outcome": "bounce", "far": [0, 0], "near": [0.9, 0, 0.8]},
            {"outcome": "through", "far": [0, 0.8], "near": [0.9, 0]}]
    m = ball.outcome_metrics(rows)
    assert m["outcome_auroc"] == 1.0 and m["blocked_called_through"] == 0
    assert m["bounce_auroc"] == 1.0 and m["outcome3_accuracy"] == 1.0
    m2 = ball.outcome_metrics([{k: v for k, v in r.items() if k != "near"} for r in rows])
    assert "bounce_auroc" not in m2                           # no near readouts: binary only


def test_near_and_far_cells_are_opposite_sides():
    scene = {"occluder_polygon": OCCLUDER}
    far, near = ball.far_cells(scene), ball.near_cells(scene)
    assert far[8, 0] and near[8, 15] and not (far & near).any()


def test_cv_folds_unchanged_for_two_outcomes():
    from sklearn.model_selection import StratifiedKFold
    clips = [{"outcome": o} for o in ["through"] * 14 + ["hidden"] * 6]
    y = np.array([c["outcome"] == "through" for c in clips], int)
    old = list(StratifiedKFold(5, shuffle=True, random_state=0).split(y, y))
    new = ball.cv_folds(clips)
    assert all((a[1] == b[1]).all() for a, b in zip(old, new))

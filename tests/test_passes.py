"""Pass detection on synthetic tracks (no video needed).

Run: .venv/bin/python -m pytest tests/   (or: .venv/bin/python tests/test_passes.py)
"""

import numpy as np

from robust_world.passes import find, side_fn

FPS = 30.0
SIZE = 512
# Vertical bar occluder at x 200-260.
SCENE = {"occluder_polygon": [[200, 0], [260, 0], [260, 512], [200, 512]]}
POLY = np.array(SCENE["occluder_polygon"], dtype=np.float32)


def occ_dist(x, y, r):
    import cv2
    return cv2.pointPolygonTest(POLY, (float(x), float(y)), True) + r


def roll(xs, y=250, r=20):
    rows = []
    for x in xs:
        if x is None:
            rows.append({"visible": 0})
        else:
            rows.append({"visible": 1, "x": x, "y": y, "radius": r, "occ_dist": occ_dist(x, y, r)})
    for i, row in enumerate(rows):
        row["frame"] = i
    return rows


def visible_where(xs):
    """Ball positions, hidden while its centre is over the occluder."""
    return [None if 200 - 10 < x < 260 + 10 else x for x in xs]


def test_side():
    side = side_fn(SCENE["occluder_polygon"])
    assert side(400, 100) != side(50, 100)


def test_through_pass():
    xs = [None] * 10 + visible_where(list(range(500, -20, -10))) + [None] * 10
    xs = [x if x is None or 0 <= x <= SIZE else None for x in xs]
    (p,) = find(roll(xs), FPS, SCENE, SIZE)
    assert p["outcome"] == "through"
    assert p["entry_at_edge"] and p["exit_at_edge"]
    # Onset is the first frame the ball's outline (r=20) touches the bar at x=260.
    onset_x = 500 - 10 * (p["occlusion_start_frame"] - 10)
    assert onset_x <= 280 and onset_x + 10 > 280


def test_blocked_pass_is_hidden():
    xs = [None] * 10 + visible_where(list(range(500, 240, -10))) + [None] * 120
    (p,) = find(roll(xs), FPS, SCENE, SIZE)
    assert p["outcome"] == "hidden"
    assert p["reappear_frame"] is None


def test_bounce():
    xs = [None] * 10 + list(range(500, 280, -10)) + [None] * 5 + list(range(290, 520, 10))
    xs = [x if x is None or x <= SIZE else None for x in xs]
    (p,) = find(roll(xs), FPS, SCENE, SIZE)
    assert p["outcome"] == "bounce"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")

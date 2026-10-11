"""nn_locate: sub-cell ball positions from nearest real context patches, built from the context half only."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vjepa2"))
import nn_locate  # noqa: E402


def synthetic(rng, n_steps, g=16, d=32):
    """Features whose ball cell carries the ball's sub-cell offset; returns feats, centres (px)."""
    feats = torch.from_numpy(rng.normal(0, 1, (n_steps, g, g, d))).float()
    centres = []
    cell = 512 / g
    for t in range(n_steps):
        x, y = rng.uniform(100, 400, 2)
        cx, cy = int(x // cell), int(y // cell)
        ox, oy = x / cell - cx - 0.5, y / cell - cy - 0.5
        feats[t, cy, cx, :2] = torch.tensor([ox, oy]) * 40
        feats[t, cy, cx, 2:] *= 0.05
        centres.append((x, y))
    return feats, centres


def test_bank_locates_within_cell():
    rng = np.random.default_rng(0)
    bank = nn_locate.Bank([synthetic(rng, 12) for _ in range(150)])
    feats, centres = synthetic(rng, 12)
    maps = np.zeros((12, 16, 16))
    for t, (x, y) in enumerate(centres):
        maps[t, int(y // 32), int(x // 32)] = 0.9
    pos = bank.locate(feats, maps)
    err = [np.hypot(p[0] - c[0], p[1] - c[1]) for p, c in zip(pos, centres)]
    coarse = [np.hypot((int(c[0] // 32) + 0.5) * 32 - c[0], (int(c[1] // 32) + 0.5) * 32 - c[1]) for c in centres]
    assert np.median(err) < 0.5 * np.median(coarse)


def test_bank_sees_only_context():
    """Built from (context features, context centres): extra target-half centres change nothing."""
    rng = np.random.default_rng(1)
    clips = [synthetic(rng, 6) for _ in range(20)]
    a = nn_locate.Bank(clips)
    b = nn_locate.Bank([(f, c + [(1.0, 1.0)] * 6) for f, c in clips])
    assert torch.equal(a.keys, b.keys) and np.array_equal(a.offs, b.offs)


def test_no_position_without_visible_ball():
    rng = np.random.default_rng(2)
    bank = nn_locate.Bank([synthetic(rng, 12) for _ in range(5)])
    feats, _ = synthetic(rng, 3)
    assert bank.locate(feats, np.full((3, 16, 16), 0.05)) == [None, None, None]

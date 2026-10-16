"""other_ball.py end to end on a tiny random V-JEPA 2 (no download, CPU), and its colour direction's isolation."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "vjepa2"))
import generalise  # noqa: E402
import other_ball  # noqa: E402
import plan  # noqa: E402
import posttrain  # noqa: E402
from test_interpret_generalise import ALL, CTX, D, G, SCENE, tiny_model  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    g = torch.Generator().manual_seed(0)
    colour = torch.randn(D, generator=g)                          # the new ball's tokens are shifted along this
    tracks, manifests = {}, {"right": [], "left": [], "orange": []}
    for video, n in (("whole", 24), ("orange", 8)):
        track, passes = [], {}
        for i in range(n):
            side = "L" if video == "whole" and i >= 20 else "R"
            start = len(track)
            for k in range(ALL * 2):
                track.append((1, 400 - 20 * k if side == "R" else 60 + 20 * k, 150.0 + 7 * i, 10.0))
            passes[i] = {"id": i, "occlusion_start_frame": start + CTX * 2, "side_in": side}
            real = torch.randn(ALL, G, G, D, generator=g)
            if video == "orange":
                real += 3 * colour
            cid = f"{video}{i}"
            torch.save({"real": real.half(), "context": real[:CTX].half(), "imagined": real[CTX:].half()},
                       cache / f"{cid}.pt")
            seg = "orange" if video == "orange" else ("left" if side == "L" else "right")
            manifests[seg].append({"clip_id": cid, "source_video": f"data/interim/{video}/{video}_512.mp4",
                                   "pass_id": i, "side_in": side, "outcome": "through" if i % 2 else "hidden",
                                   "include": True, "n_frames": ALL * 2, "fps": 16, "source_fps": 16,
                                   "context_frames": [0, CTX * 2 - 1], "path": "x.mp4"})
        tracks[video] = (track, passes, SCENE)
    for seg, cs in manifests.items():
        (tmp_path / f"manifest_{seg}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cs))
    (tmp_path / "heldout.json").write_text(json.dumps({"videos": ["orange"]}))
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    monkeypatch.setattr(transformers.VJEPA2Model, "from_pretrained", staticmethod(lambda model_id: tiny_model()))
    monkeypatch.setattr(generalise, "load_tracking", lambda name: tracks[name])
    generalise.tracking.cache_clear()
    monkeypatch.setattr(generalise, "HELDOUT_CFG", tmp_path / "heldout.json")
    monkeypatch.setattr(posttrain, "CKPT_ROOT", tmp_path / "ck")
    return tmp_path


def test_scores_colour_direction_and_removal(world, monkeypatch):
    tmp = world
    posttrain.main(["train", "--run", "cv", "--epochs", "1", "--workers", "0",
                    "--manifest", str(tmp / "manifest_right.jsonl")])
    monkeypatch.setattr(plan, "found_weights", lambda name: tmp / "ck" / "cv" if name == "lossmix_both-after" else None)
    out = tmp / "ob"
    assert other_ball.main(["--manifest", str(tmp / "manifest_right.jsonl"), "--out", str(out)]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["colour_direction"]["colour_auroc_mean"] > 0.9            # the planted colour shift is found
    assert m["colour_direction"]["n_ball_tokens"]["new"] > 0
    for b in ("old", "new"):
        assert {"real", "pretrained", "pretrained (colour removed)", "C", "C (colour removed)"} <= set(m["scores"][b])
    rows = json.loads((out / "per_clip.json").read_text())
    assert sum(r["ball"] == "new" for r in rows) == 8 and (out / "colour.png").exists()
    d = torch.load(out / "colour_direction.pt")
    assert abs(float(d["ln"].norm()) - 1) < 1e-4


def test_removal_sets_the_coordinate():
    x = torch.randn(2, 3, 4, 4, D)
    d = other_ball.unit(torch.randn(D))
    assert torch.allclose(other_ball.remove(x, d, 0.7) @ d, torch.full(x.shape[:-1], 0.7), atol=1e-4)


def test_colour_direction_ignores_outcomes(world):
    """CLAUDE.md: nothing that reads the model may learn from outcomes. The direction reads only which ball."""
    def fit(flip):
        rows, is_new = [], []
        for seg in ("right", "orange"):
            for c in generalise.included(world / f"manifest_{seg}.jsonl"):
                if flip:
                    c = dict(c, outcome="hidden" if c["outcome"] == "through" else "through")
                enc = posttrain.load_cached(posttrain.cache_dir(""), c)
                track, passes, _ = generalise.tracking(generalise.stem(c))
                from robust_world.eval.ball import ball_labels, source_indices
                lab, centres = ball_labels(source_indices(c, passes, c["n_frames"], CTX * 2), track, G, 2)
                rows.append(other_ball.ball_rows(enc, lab, centres))
                is_new.append(seg == "orange")
        return other_ball.direction([r for rs, n in zip(rows, is_new) if n for r in rs],
                                    [r for rs, n in zip(rows, is_new) if not n for r in rs], "ln")
    assert torch.equal(fit(False), fit(True))

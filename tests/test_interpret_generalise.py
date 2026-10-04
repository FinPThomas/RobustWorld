"""interpret.py and generalise.py end to end on a tiny random V-JEPA 2 (no download, CPU)."""

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
import interpret  # noqa: E402
import posttrain  # noqa: E402

G, D, CTX, ALL = 4, 64, 3, 6          # 256 px frames in 64 px patches: a 4x4 token grid, like run.SIZE
SCENE = json.loads((ROOT / "configs" / "scenes" / "start.json").read_text())


def tiny_model():
    cfg = transformers.VJEPA2Config(hidden_size=D, num_hidden_layers=1, num_attention_heads=4, patch_size=64,
                                    pred_hidden_size=32, pred_num_hidden_layers=2, pred_num_attention_heads=4,
                                    crop_size=256, frames_per_clip=ALL * 2, mlp_ratio=2)
    torch.manual_seed(0)
    return transformers.VJEPA2Model(cfg).eval()


@pytest.fixture
def world(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    g = torch.Generator().manual_seed(0)
    clips, passes, track = [], {}, []
    for i in range(20):
        side = "L" if i >= 16 else "R"                       # four clips come from the other side
        outcome = "through" if i % 2 else "hidden"
        start = len(track)
        for k in range(ALL * 2):                             # ball moves across the frame
            track.append((1, 400 - 20 * k if side == "R" else 60 + 20 * k, 200.0, 10.0))
        passes[i] = {"id": i, "occlusion_start_frame": start + CTX * 2, "side_in": side}
        real = torch.randn(ALL, G, G, D, generator=g)
        torch.save({"real": real.half(), "context": real[:CTX].half(), "imagined": real[CTX:].half()},
                   cache / f"c{i}.pt")
        clips.append({"clip_id": f"c{i}", "source_video": "data/raw/videos/start.mp4", "pass_id": i,
                      "outcome": outcome, "include": True, "n_frames": ALL * 2, "fps": 16, "source_fps": 16,
                      "context_frames": [0, CTX * 2 - 1], "path": "x.mp4"})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(c) + "\n" for c in clips))
    fake_tracking = lambda name="start": (track, passes, SCENE)  # noqa: E731
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    monkeypatch.setattr(transformers.VJEPA2Model, "from_pretrained", staticmethod(lambda model_id: tiny_model()))
    monkeypatch.setattr(interpret, "load_tracking", fake_tracking)
    monkeypatch.setattr(generalise, "load_tracking", fake_tracking)
    generalise.tracking.cache_clear()
    monkeypatch.setattr(generalise, "SEEN", tmp_path / "seen.jsonl")
    monkeypatch.setattr(generalise, "HELDOUT_CFG", tmp_path / "none.json")
    import robust_world.eval.io as io
    monkeypatch.setattr(io, "read_video", lambda path: np.zeros((ALL * 2, 512, 512, 3), np.uint8))
    monkeypatch.setattr(posttrain, "CKPT_ROOT", tmp_path / "ck")
    return tmp_path, manifest


def test_split_score_and_interpret(world):
    tmp_path, manifest = world
    out = tmp_path / "gen"
    assert generalise.main(["split", "--manifest", str(manifest), "--out", str(out)]) == 0
    split = json.loads((out / "split.json").read_text())
    assert split["training_direction"] == "R" and split["n_seen"] == 16
    assert split["held_out"]["other side (from L)"]["n"] == 4

    seen = str(tmp_path / "seen.jsonl")
    common = ["--epochs", "1", "--workers", "0", "--manifest", seen, "--copy-gate", "--hypotheses", "2"]
    posttrain.main(["train", "--all", "--run", "all", *common])
    assert generalise.main(["score", "--manifest", str(manifest), "--run", str(tmp_path / "ck" / "all"),
                            "--out", str(out)]) == 0
    g = json.loads((out / "metrics.json").read_text())["groups"]["other side (from L)"]
    assert g["n"] == 4 and {"before", "after", "real"} <= set(g)

    posttrain.main(["train", "--run", "cv", *common])
    res_dir = tmp_path / "interp"
    interpret.main(["--run", str(tmp_path / "ck" / "cv"), "--manifest", seen, "--out", str(res_dir)])
    res = json.loads((res_dir / "interpret.json").read_text())
    assert set(res["patching"]["patch_in"]) >= {"embeddings", "layer 00", "layer 01", "output norm + proj", "heads"}
    assert 0 <= res["feature_change"]["share_on_plank"] <= 1
    assert (res_dir / "change_map.png").exists() and (res_dir / "layer_patching.png").exists()


def test_no_held_out_clips_means_waiting(world):
    tmp_path, manifest = world
    clips = [json.loads(line) for line in manifest.open()][:16]
    manifest.write_text("".join(json.dumps(c) + "\n" for c in clips))
    assert generalise.main(["split", "--manifest", str(manifest), "--out", str(tmp_path / "g")]) == generalise.WAITING
    assert len((tmp_path / "seen.jsonl").read_text().splitlines()) == 16

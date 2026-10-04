"""V-JEPA 2 post-training plumbing on a tiny random model (no download, CPU)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vjepa2"))
import posttrain  # noqa: E402

G, D, CTX, ALL = 4, 64, 3, 6          # 4x4 tokens per step, 3 context steps, 6 steps per clip


def tiny_model():
    cfg = transformers.VJEPA2Config(hidden_size=D, num_hidden_layers=1, num_attention_heads=4,
                                    pred_hidden_size=32, pred_num_hidden_layers=2, pred_num_attention_heads=4,
                                    crop_size=G * 16, frames_per_clip=ALL * 2, mlp_ratio=2)
    torch.manual_seed(0)
    return transformers.VJEPA2Model(cfg).eval()


def fake_clips(tmp_path, n=10):
    cache = tmp_path / "cache"
    cache.mkdir()
    clips = []
    g = torch.Generator().manual_seed(0)
    future = torch.randn(ALL - CTX, G, G, D, generator=g)                          # the same for every clip,
    for i in range(n):                                                              # so it can be learned
        outcome = "through" if i % 2 else "hidden"
        real = torch.randn(ALL, G, G, D, generator=g)
        real[CTX:] = future + 0.05 * torch.randn(ALL - CTX, G, G, D, generator=g)
        enc = {"real": real.half(), "context": real[:CTX].half(), "imagined": real[CTX:].half()}
        clip = {"clip_id": f"c{i}", "outcome": outcome, "include": True, "n_frames": ALL * 2,
                "context_frames": [0, CTX * 2 - 1]}
        torch.save(enc, cache / f"c{i}.pt")
        clips.append(clip)
    return clips, cache


def test_cache_must_match_clip_length(tmp_path):
    clips, cache = fake_clips(tmp_path, 1)
    assert posttrain.load_cached(cache, clips[0]) is not None
    recut = dict(clips[0], n_frames=ALL * 2 + 16, context_frames=[0, CTX * 2 + 7])     # re-cut to longer clips
    assert posttrain.load_cached(cache, recut) is None


def test_imagine_shapes():
    model = tiny_model()
    out = posttrain.imagine(model.predictor, torch.randn(2, CTX, G, G, D), ALL - CTX)
    assert out.shape == (2, ALL - CTX, G, G, D)


def test_training_lowers_held_out_l1_and_checkpoint_round_trips(tmp_path, monkeypatch):
    clips, cache = fake_clips(tmp_path)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    model = tiny_model()
    base = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    args = SimpleNamespace(model_id="tiny", lr=3e-3, weight_decay=0.0, batch_size=2, accum=1, epochs=15,
                           workers=0, seed=0, loss="l1")
    log, _ = posttrain.train_one(model, base, clips[:8], clips[8:], args, "cpu", "fold0")
    before = sum(log["held_out_l1_pretrained"].values())
    after = sum(log["held_out_l1_posttrained"].values())
    assert after < before
    assert log["train_loss"][-1] < log["train_loss"][0]

    path = tmp_path / "fold0.pt"
    torch.save({"predictor": model.predictor.state_dict(), "train_clip_ids": ["c0"]}, path)
    fresh = tiny_model()
    meta = posttrain.load_predictor(fresh, path)
    assert meta["train_clip_ids"] == ["c0"]
    ctx = torch.randn(1, CTX, G, G, D)
    with torch.no_grad():
        assert torch.allclose(posttrain.imagine(fresh.predictor, ctx, ALL - CTX),
                              posttrain.imagine(model.predictor.eval(), ctx, ALL - CTX))


def test_training_never_touches_the_eval_decoder_inputs(tmp_path, monkeypatch):
    """Post-training changes only the predictor: cached context/real features (what the frozen
    evaluation decoder is fitted on) are left as they were."""
    clips, cache = fake_clips(tmp_path, 4)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    before = {c["clip_id"]: torch.load(cache / f"{c['clip_id']}.pt") for c in clips}
    model = tiny_model()
    base = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    enc_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    args = SimpleNamespace(model_id="tiny", lr=1e-3, weight_decay=0.0, batch_size=2, accum=1, epochs=2,
                           workers=0, seed=0, loss="l1")
    _ = posttrain.train_one(model, base, clips[:3], clips[3:], args, "cpu", "fold0")
    for c in clips:
        after = torch.load(cache / f"{c['clip_id']}.pt")
        assert torch.equal(after["context"], before[c["clip_id"]]["context"])
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in enc_before.items())
    json.dumps(posttrain.by_outcome(clips, [1.0] * 4))


def test_commit_loss_trains_and_prefers_its_own_future(tmp_path, monkeypatch):
    clips, cache = fake_clips(tmp_path)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    model = tiny_model()
    base = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    args = SimpleNamespace(model_id="tiny", lr=3e-3, weight_decay=0.0, batch_size=2, accum=1, epochs=10,
                           workers=0, seed=0, loss="commit", motion_alpha=4.0, contrast_weight=0.1,
                           tau=0.02, negatives=3)
    log, _ = posttrain.train_one(model, base, clips[:8], clips[8:], args, "cpu", "fold0")
    assert set(log["train_parts"][0]) == {"l1", "wl1", "nce"}
    assert sum(log["held_out_l1_posttrained"].values()) < sum(log["held_out_l1_pretrained"].values())

    # The contrast term is low when a prediction matches its own future, high when it matches another's.
    t = torch.randn(2, ALL - CTX, G, G, D)
    w = torch.ones(2, ALL - CTX, G, G)
    right = posttrain.contrast(t.clone(), t, None, w, 0.02)
    swapped = posttrain.contrast(t.flip(0), t, None, w, 0.02)
    assert right < 0.01 < swapped


def test_motion_weights_flag_phantom_balls():
    last = torch.zeros(1, G, G, D)
    target = torch.zeros(1, ALL - CTX, G, G, D)                  # nothing moves in the real future
    pred = torch.zeros(1, ALL - CTX, G, G, D)
    pred[0, 1, 2, 3] = 5.0                                        # the prediction invents a moving ball
    w = posttrain.motion_weights(target, pred, last, alpha=4.0)
    assert w[0, 1, 2, 3] > w[0, 0, 0, 0] and w.min() >= 1


def test_rollout_and_codes_modes(tmp_path, monkeypatch):
    model = tiny_model()
    ctx = torch.randn(2, CTX, G, G, D)
    with torch.no_grad():
        first = posttrain.imagine(model.predictor, ctx, 1)
        steps = posttrain.rollout(model.predictor, ctx, ALL - CTX)
        assert steps.shape == (2, ALL - CTX, G, G, D)
        assert torch.allclose(steps[:, :1], first, atol=1e-5)       # step 1 sees only the real context

        book = torch.nn.functional.normalize(torch.randn(8, D), dim=-1)
        snapped = posttrain.imagine_mode(model.predictor, ctx, ALL - CTX, {"rollout": True, "codebook": book})
        assert all(any(torch.allclose(t, c) for c in book) for t in snapped.reshape(-1, D)[:20])

    clips, cache = fake_clips(tmp_path)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    centres = posttrain.fit_codebook(posttrain.Cached(clips, cache), k=8, seed=0, n_tokens=400, iters=5)
    assert centres.shape == (8, D) and torch.isfinite(centres).all()


def test_codes_rollout_trains(tmp_path, monkeypatch):
    clips, cache = fake_clips(tmp_path)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    model = tiny_model()
    base = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    args = SimpleNamespace(model_id="tiny", lr=3e-3, weight_decay=0.0, batch_size=2, accum=1, epochs=8,
                           workers=0, seed=0, loss="codes", codes=8, code_tau=0.05, rollout=True)
    log, mode = posttrain.train_one(model, base, clips[:8], clips[8:], args, "cpu", "fold0")
    assert mode["rollout"] and mode["codebook"].shape == (8, D)
    assert log["train_loss"][-1] < log["train_loss"][0]

    args.epochs = 0                                                 # "before": the pretrained predictor
    log0, _ = posttrain.train_one(model, base, clips[:8], clips[8:], args, "cpu", "fold0")
    assert all(torch.equal(v, base[k]) for k, v in model.predictor.state_dict().items())


def test_experiment_summary(tmp_path, monkeypatch):
    import experiments
    monkeypatch.setattr(experiments, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(experiments, "OUT", tmp_path / "out")
    outcomes = {"p_correct_balanced": 0.5, "balanced_accuracy": 0.5, "p_correct": {"through": 0.5},
                "outcome_auroc": 0.5}
    ball = {"hit_rate": 0.3, "error_px": 40.0, "phantom_rate": 0.1}
    for name in ("plain-before", "plain-after"):
        (tmp_path / "scores" / name).mkdir(parents=True)
        (tmp_path / "scores" / name / "metrics.json").write_text(json.dumps(
            {"outcomes_imagined": outcomes, "ball_imagined": ball, "outcomes_real": outcomes, "ball_real": ball}))
    experiments.summary(SimpleNamespace(variants=["plain", "codes"]))
    rows = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert [r["phase"] for r in rows] == ["before", "after", "-"]
    assert (tmp_path / "out" / "summary.png").exists()

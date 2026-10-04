"""V-JEPA 2 post-training plumbing on a tiny random model (no download, CPU)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    assert all(layer.gradient_checkpointing for layer in model.predictor.layer)     # saves GPU memory
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
            {"outcomes_imagined": outcomes, "ball_imagined": ball, "outcomes_real": outcomes, "ball_real": ball,
             "outcomes_imagined_one_ball": outcomes, "ball_imagined_argmax": ball, "future_cell_auroc_imagined": 0.6}))
    experiments.summary(SimpleNamespace(variants=["plain", "codes"]))
    rows = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert [r["phase"] for r in rows] == ["before", "after", "-"]
    assert rows[0]["one_ball_p_correct"] == 0.5 and rows[0]["cell_auroc"] == 0.6
    assert (tmp_path / "out" / "summary.png").exists()


def test_targets_in_prediction_space_and_fed_back_in_encoder_space(tmp_path):
    clips, cache = fake_clips(tmp_path, 1)
    ctx, tgt, last, _ = posttrain.Cached(clips, cache)[0]
    assert torch.allclose(tgt.std(-1).mean(), torch.tensor(1.0), atol=0.05)          # layer-normalised target
    like = torch.randn(2, G, G, D) * 3 + 1                                           # encoder-scale context step
    fed = posttrain.to_encoder_space(torch.randn(2, 1, G, G, D), like)
    assert torch.allclose(fed.mean(-1)[:, 0], like.mean(-1), atol=1e-4)
    assert torch.allclose(fed.std(-1)[:, 0], like.std(-1), atol=1e-3)


def test_scale_check_runs(tmp_path, monkeypatch):
    import numpy as np
    import scale_check
    clips, cache = fake_clips(tmp_path)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    data = []
    for i, c in enumerate(clips):
        lab = np.zeros((ALL, G, G), bool)
        lab[:CTX, 1, i % G] = True
        data.append({"clip": c, "enc": torch.load(cache / f"{c['clip_id']}.pt"), "lab": lab})
    args = SimpleNamespace(model_id="tiny", lr=3e-3, weight_decay=0.0, batch_size=2, accum=1, epochs=2,
                           workers=0, seed=0, loss="l1", rollout=False, grad_checkpoint=False)
    res = scale_check.check(tiny_model(), data[:8], data[8:], args, "cpu")
    assert set(res["before"]) == set(res["after"]) and res["relative_weight_change"] > 0
    assert abs(res["before"]["spread_target_ln"] - 1) < 0.05


def test_split_guard_refuses_leaks():
    a = {"clip_id": "a", "source_video": "v", "pass_id": 1}
    b = {"clip_id": "b", "source_video": "v", "pass_id": 2}
    posttrain.check_split([a], [b])
    with pytest.raises(ValueError):
        posttrain.check_split([a, b], [b])                                         # same clip
    with pytest.raises(ValueError):
        posttrain.check_split([a], [dict(b, clip_id="a2", pass_id=1)])            # re-cut of the same pass


def overfit_args(**kw):
    return SimpleNamespace(**{"model_id": "tiny", "lr": 1e-2, "weight_decay": 0.0, "batch_size": 2, "accum": 1,
                              "epochs": 25, "workers": 0, "seed": 0, "loss": "l1", "val_frac": 0.3,
                              "patience": 0, "keep_best": True} | kw)


def test_overfit_report():
    ok = posttrain.overfit_report([0.9, 0.8, 0.79], [0.88, 0.78, 0.77], 0.79, 3)
    assert not ok["flag"]
    rising = posttrain.overfit_report([0.8, 0.85, 0.9], [0.7, 0.6, 0.5], 0.8, 1)       # validation turns up
    assert rising["flag"] and rising["val_rise_from_best"] > 0.1
    memorising = posttrain.overfit_report([0.8, 0.8], [0.4, 0.4], 0.8, 1)             # train far below
    assert memorising["flag"] and memorising["val_over_train"] == 2.0


def test_best_validation_epoch_is_kept(tmp_path, monkeypatch):
    clips, cache = fake_clips(tmp_path, 14)
    g = torch.Generator().manual_seed(1)
    for c in clips:                                       # every clip gets its own random future:
        enc = torch.load(cache / f"{c['clip_id']}.pt")   # nothing carries over to validation clips
        enc["real"][CTX:] = torch.randn(ALL - CTX, G, G, D, generator=g).half()
        torch.save(enc, cache / f"{c['clip_id']}.pt")
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    model = tiny_model()
    base = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    args = overfit_args(lr=3e-2, epochs=8)
    log, mode = posttrain.train_one(model, base, clips[:12], clips[12:], args, "cpu", "fold0")
    assert log["n_val"] == 4 and len(log["val_l1"]) == 8 and "flag" in log["overfit"]
    fit, val = posttrain.split_validation(clips[:12], 0.3, 0)
    now = float(np.mean(posttrain.held_out_l1(model.predictor, posttrain.Cached(val, cache), "cpu",
                                                  lambda: torch.autocast("cpu", enabled=False), mode)))
    assert now == pytest.approx(min(log["val_l1"] + [log["val_l1_pretrained"]]), abs=1e-4)


def test_learnable_task_is_not_flagged_and_patience_stops(tmp_path, monkeypatch):
    clips, cache = fake_clips(tmp_path, 14)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    model = tiny_model()
    base = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    log, _ = posttrain.train_one(model, base, clips[:12], clips[12:], overfit_args(lr=3e-3, epochs=10),
                                 "cpu", "fold0")
    assert not log["overfit"]["flag"] and log["best_epoch"] > 0
    assert min(log["val_l1"]) < log["val_l1_pretrained"]


def test_train_all_writes_logs(tmp_path, monkeypatch):
    clips, cache = fake_clips(tmp_path, 25)
    monkeypatch.setattr(posttrain, "cache_dir", lambda model_id: cache)
    monkeypatch.setattr(posttrain, "training_clips", lambda manifest: clips)
    monkeypatch.setattr(posttrain, "CKPT_ROOT", tmp_path / "ck")
    monkeypatch.setattr(transformers.VJEPA2Model, "from_pretrained", staticmethod(lambda model_id: tiny_model()))
    posttrain.main(["train", "--run", "t", "--fold", "0", "--epochs", "2", "--workers", "0", "--lr", "3e-3"])
    log = json.loads((tmp_path / "ck" / "t" / "log.json").read_text())[0]
    rec = log["epochs"][-1]
    assert {"grad_norm_mean", "weight_change", "lr", "seconds", "val_l1", "train_clip_l1"} <= set(rec)
    assert rec["weight_change"] > 0 and log["final_weight_change"] >= 0
    info = json.loads((tmp_path / "ck" / "t" / "run_info.json").read_text())
    assert info["scale_check"]["target_spread_ln"] == pytest.approx(1.0, abs=0.05)
    assert info["folds"][0]["split"] == "fold0" and info["code_version"]


def test_grid_survives_failures_and_resumes(tmp_path, monkeypatch):
    import subprocess as sp
    import experiments
    monkeypatch.setattr(experiments, "SCORES", tmp_path / "scores")
    monkeypatch.setattr(experiments, "OUT", tmp_path / "out")
    calls, fail = [], {"codes-after"}
    metrics = {"outcomes_imagined": {}, "ball_imagined": {}, "outcomes_real": {}, "ball_real": {},
               "outcomes_imagined_one_ball": {"p_correct_balanced": 0.5}, "future_cell_auroc_imagined": 0.6}

    def fake_run(cmd, check=False, **kw):
        calls.append(cmd)
        if "ball_probe_cv.py" in cmd[1]:
            name = Path(cmd[-1]).name
            if name in fail:
                raise sp.CalledProcessError(1, cmd)
            (tmp_path / "scores" / name).mkdir(parents=True, exist_ok=True)
            (tmp_path / "scores" / name / "metrics.json").write_text(json.dumps(metrics))
        return sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    experiments.main("run --variants plain codes --epochs 2 --save-as night".split())
    grid = json.loads((tmp_path / "out" / "grid.json").read_text())
    assert grid["done"] == ["plain-before", "plain-after", "codes-before"] and "codes-after" in grid["failed"]
    saves = [c for c in calls if "save_results.py" in c[1]]
    assert len(saves) == 4 and all(grid["results_folder"] in c for c in saves)       # saved after every run

    calls.clear(), fail.clear()
    experiments.main("run --variants plain codes --epochs 2 --save-as night --resume".split())
    trained = [c[c.index("--run") + 1] for c in calls if "posttrain.py" in c[1]]
    assert trained == ["codes-after"]                                                 # only the failed one reruns
    grid2 = json.loads((tmp_path / "out" / "grid.json").read_text())
    assert grid2["results_folder"] == grid["results_folder"] and not grid2["failed"]

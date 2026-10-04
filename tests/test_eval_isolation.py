"""The evaluation decoder must never be able to learn the blockade.

Learning the blockade belongs to post-training the world model. The decoder that reads the
ball out of V-JEPA features is the measuring instrument and must be identical before and
after post-training, so it may only depend on real context-half features and same-frame
labels. These tests change everything else (outcomes, target-half labels, imagined
features, full-clip features) and require a bit-for-bit identical decoder.

Run: python -m pytest tests/test_eval_isolation.py   (needs torch)
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vjepa2"))
import eval_decoder  # noqa: E402

CTX, TGT, G, D = 8, 8, 4, 16


def fake_clip(rng, outcome):
    labels = np.zeros((CTX + TGT, G, G), bool)
    for s in range(CTX):                                   # ball rolls in on the near side
        labels[s, 1, G - 1 - s % G] = True
    if outcome == "through":
        labels[CTX + 3:, 2, 0] = True                      # comes out the far side
    enc = {"context": torch.from_numpy(rng.normal(size=(CTX, G, G, D)).astype(np.float32)),
           "real": torch.from_numpy(rng.normal(size=(CTX + TGT, G, G, D)).astype(np.float32)),
           "imagined": torch.from_numpy(rng.normal(size=(TGT, G, G, D)).astype(np.float32))}
    return {"outcome": outcome, "enc": enc, "labels": labels}


def fit_decoder(clips):
    return eval_decoder.fit([eval_decoder.examples_from(c["enc"], c["labels"]) for c in clips], seed=0, epochs=3)


def same(a, b):
    return all(torch.equal(a.state()[k], b.state()[k]) for k in a.state())


def test_decoder_ignores_outcomes_future_labels_and_predictions():
    rng = np.random.default_rng(0)
    clips = [fake_clip(rng, o) for o in ["through", "hidden"] * 6]
    reference = fit_decoder(clips)

    tampered = []
    rng2 = np.random.default_rng(1)
    for c in clips:
        labels = c["labels"].copy()
        labels[CTX:] = rng2.random(labels[CTX:].shape) > 0.5          # scramble target-half labels
        enc = dict(c["enc"])
        enc["imagined"] = torch.randn_like(enc["imagined"]) * 100      # different "world model"
        enc["real"] = torch.randn_like(enc["real"])                     # full-clip encoding (sees the future)
        tampered.append({"outcome": "hidden" if c["outcome"] == "through" else "through",
                         "enc": enc, "labels": labels})
    assert same(reference, fit_decoder(tampered)), "decoder depended on something other than the context half"


def test_decoder_does_depend_on_context_half():
    rng = np.random.default_rng(0)
    clips = [fake_clip(rng, o) for o in ["through", "hidden"] * 6]
    changed = [dict(c, enc=dict(c["enc"], context=c["enc"]["context"] + 1.0)) for c in clips]
    assert not same(fit_decoder(clips), fit_decoder(changed))


def test_fit_has_no_way_to_receive_outcomes_or_predictions():
    params = set(inspect.signature(eval_decoder.fit).parameters)
    assert params == {"examples", "seed", "epochs"}
    fields = set(eval_decoder.EvalExample.__dataclass_fields__)
    assert fields == {"context_features", "context_labels"}
    with pytest.raises(TypeError):
        eval_decoder.fit([{"context": None}])                            # raw dicts are refused


def test_no_other_readout_is_trained_on_vjepa_features():
    """Only eval_decoder.py may fit a readout; the old outcome/imagined probes must not come back."""
    root = Path(__file__).resolve().parents[1] / "models" / "vjepa2"
    banned = ["train_probe", "imag_probe", "LogisticRegression", ".fit(feats", "ctx_clf"]
    for f in root.glob("*.py"):
        if f.name == "eval_decoder.py":
            continue
        text = f.read_text()
        for b in banned:
            assert b not in text, f"{f.name} trains a readout ({b}); use eval_decoder instead"


def test_decoder_reads_tokens_regardless_of_scale():
    """Real (encoder) and imagined (predictor) tokens differ in scale; the decoder must read both alike."""
    rng = np.random.default_rng(0)
    dec = fit_decoder([fake_clip(rng, o) for o in ["through", "hidden"] * 3])
    t = torch.randn(5, D)
    assert torch.allclose(dec(t), dec(t * 5 + 2), atol=1e-5)

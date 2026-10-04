"""Small reproduction of the feature-scale mismatch, kept for reference (2026-10-04).

V-JEPA 2's predictor was pretrained to output layer-normalised features, but the Hugging Face
encoder returns features with its final layer norm's learned scale and offset applied (per-token
spread about 3.3 vs 0.65). Training against raw targets spends its effort stretching the scale,
and a decoder fitted on raw features reads every imagined cell as background (P(ball) ~0.02), so
before and after look identical. This script shows that, and that the fix (layer-normalised
targets; the decoder reading layer-normalised tokens, or predictions mapped back with
gamma * pred + beta) lets post-training show up.

It takes a few clips (default 16: 12 train, 4 held out), runs a short post-training on CPU or GPU
with posttrain.train_one, and prints, before and after:
  - per-token spread of real targets (raw / layer-normalised) and of the prediction
  - held-out L1 in target space
  - peak P(ball) on the first imagined steps from three readings of the same prediction:
    raw decoder on the raw prediction (the old, broken reading), layer-norm decoder (default
    now), and raw decoder on gamma * pred + beta (encoder-space alternative)
Decoders are fitted on the training clips' real context features only (eval_decoder rules).

    python models/vjepa2/posttrain.py encode            # once, if the cache is empty
    python models/vjepa2/scale_check.py --clips 16 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_decoder  # noqa: E402
from posttrain import MANIFEST, cache_dir, imagine, load_cached, target_space, train_one, training_clips  # noqa: E402


@torch.no_grad()
def readings(predictor, data: list[dict], decoders: dict, gamma, beta, device, steps: int = 3) -> dict:
    """Spreads, held-out L1 and peak P(ball) on the first `steps` imagined steps, averaged over clips."""
    predictor.eval()
    out = {"spread_target_raw": [], "spread_target_ln": [], "spread_pred": [], "held_out_l1": [],
           "peak_raw_decoder_raw_pred": [], "peak_ln_decoder": [], "peak_raw_decoder_encoder_space": []}
    for d in data:
        enc = d["enc"]
        cs = enc["context"].shape[0]
        real = enc["real"][cs:].float()
        pred = imagine(predictor, enc["context"][None].float().to(device), real.shape[0])[0].float().cpu()
        out["spread_target_raw"].append(float(real.std(-1).mean()))
        out["spread_target_ln"].append(float(target_space(real).std(-1).mean()))
        out["spread_pred"].append(float(pred.std(-1).mean()))
        out["held_out_l1"].append(float((pred - target_space(real)).abs().mean()))
        first = pred[:steps]
        out["peak_raw_decoder_raw_pred"].append(float(decoders["raw"](first).amax((1, 2)).mean()))
        out["peak_ln_decoder"].append(float(decoders["ln"](first).amax((1, 2)).mean()))
        out["peak_raw_decoder_encoder_space"].append(
            float(decoders["raw"](target_space(first) * gamma + beta).amax((1, 2)).mean()))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


def check(model, train: list[dict], test: list[dict], args, device) -> dict:
    """train/test: [{"clip", "enc", "lab"}]. Returns before/after readings and the training log."""
    examples = [eval_decoder.examples_from(d["enc"], d["lab"]) for d in train]
    decoders = {"ln": eval_decoder.fit(examples, seed=0), "raw": eval_decoder.fit_raw_features(examples, seed=0)}
    gamma = model.encoder.layernorm.weight.detach().float().cpu()
    beta = model.encoder.layernorm.bias.detach().float().cpu()
    pred = model.predictor
    base = {k: v.detach().clone() for k, v in pred.state_dict().items()}
    before = readings(pred, test, decoders, gamma, beta, device)
    log, _ = train_one(model, base, [d["clip"] for d in train], [d["clip"] for d in test], args, device, "check")
    after = readings(pred, test, decoders, gamma, beta, device)
    moved = sum(float((v.float() - base[k].float()).norm()) for k, v in pred.state_dict().items()) / \
        sum(float(v.float().norm()) for v in base.values())
    return {"before": before, "after": after, "relative_weight_change": round(moved, 5), "train_loss": log["train_loss"]}


def main(argv: list[str] | None = None) -> int:
    from transformers import VJEPA2Model
    from robust_world.eval.ball import ball_labels, load_tracking, source_indices
    from run import MODEL_ID, pick_device

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", type=int, default=16)
    p.add_argument("--held-out", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--out", type=Path, default=HERE.parents[1] / "outputs" / "vjepa2" / "scale_check.json")
    a = p.parse_args(argv)

    device = pick_device()
    model = VJEPA2Model.from_pretrained(a.model_id).to(device)
    track, passes, _ = load_tracking()
    clips = training_clips(a.manifest)
    random.Random(0).shuffle(clips)
    data = []
    for c in clips[:a.clips]:
        enc = load_cached(cache_dir(a.model_id), c)
        if enc is None:
            raise SystemExit(f"{c['clip_id']} is not encoded; run `posttrain.py encode` first")
        n_ctx = c["context_frames"][1] + 1
        lab, _ = ball_labels(source_indices(c, passes, c["n_frames"], n_ctx), track, enc["context"].shape[1],
                             model.config.tubelet_size)
        data.append({"clip": c, "enc": enc, "lab": lab})
    train, test = data[:-a.held_out], data[-a.held_out:]
    args = SimpleNamespace(model_id=a.model_id, lr=3e-5, weight_decay=0.04, batch_size=1, accum=4, epochs=a.epochs,
                           workers=0, seed=0, loss="l1", rollout=False, grad_checkpoint=device == "cuda")
    print(f"scale check on {device}: {len(train)} train / {len(test)} held-out clips, {a.epochs} epochs")
    res = check(model, train, test, args, device)
    for k in res["before"]:
        print(f"  {k:34s} before {res['before'][k]:8.4f}   after {res['after'][k]:8.4f}")
    print(f"  relative weight change {res['relative_weight_change']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

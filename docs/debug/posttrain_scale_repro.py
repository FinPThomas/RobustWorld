"""Why do post-trained results look identical to before? Measure the change at each stage:
weights -> imagined features -> evaluation decoder's ball maps -> outcome readout."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\finla\source\repos\RobustWorld")
sys.path.insert(0, str(REPO / "models" / "vjepa2"))
sys.path.insert(0, str(REPO / "src"))
import eval_decoder  # noqa: E402
import posttrain  # noqa: E402
from ball_probe import encode  # noqa: E402
from robust_world.eval.ball import (ball_labels, far_cells, included_clips, load_tracking,  # noqa: E402
                                    source_indices, video_of)
from robust_world.eval.io import read_video  # noqa: E402
from run import MODEL_ID  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
torch.manual_seed(0)
clips = included_clips()
rng = random.Random(0)
pick = []
for o in ("through", "bounce", "hidden"):
    pick += rng.sample([c for c in clips if c["outcome"] == o], 6 if o == "through" else 5)
# Held out: two through, one bounce, one hidden; the other 12 train.
test = [pick[0], pick[1], pick[6], pick[11]]
train = [c for c in pick if c not in test]
print("train", len(train), "test", [(c["clip_id"], c["outcome"]) for c in test], flush=True)

from transformers import AutoVideoProcessor, VJEPA2Model  # noqa: E402
model = VJEPA2Model.from_pretrained(MODEL_ID).eval()
proc = AutoVideoProcessor.from_pretrained(MODEL_ID)
mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
cache = posttrain.cache_dir(MODEL_ID)
cache.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for i, c in enumerate(pick):
    if posttrain.load_cached(cache, c) is None:
        enc = encode(model, read_video(REPO / c["path"]), c["context_frames"][1] + 1, mean, std, "cpu", imagine=True)
        torch.save(enc, cache / f"{c['clip_id']}.pt")
    print(f"\r  encoded {i + 1}/{len(pick)} ({time.time() - t0:.0f}s)", end="", flush=True)
print(flush=True)

n = sum(p.numel() for p in model.predictor.parameters())
print(f"predictor params {n / 1e6:.1f}M, requires_grad {all(p.requires_grad for p in model.predictor.parameters())}")
base = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}

# Pretrained imagined futures for the test clips, computed by the same function the scorer uses.
mode0 = {"rollout": False, "codebook": None}


@torch.no_grad()
def imagined(c, mode):
    enc = posttrain.load_cached(cache, c)
    cs = enc["context"].shape[0]
    return posttrain.imagine_mode(model.predictor, enc["context"][None].float(), enc["real"].shape[0] - cs, mode)[0].float(), enc


before = {c["clip_id"]: imagined(c, mode0)[0] for c in test}

ns = argparse.Namespace(model_id=MODEL_ID, epochs=EPOCHS, batch_size=2, accum=4, lr=3e-5, weight_decay=0.04,
                        workers=0, seed=0, loss="l1", rollout=False, codes=256, code_tau=0.05,
                        motion_alpha=4.0, contrast_weight=0.1, tau=0.02, negatives=8)
log, mode = posttrain.train_one(model, base, train, test, ns, "cpu", "debug")
print("train_loss per epoch", log["train_loss"])
print("held-out L1 pretrained", log.get("held_out_l1_pretrained"), "post-trained", log.get("held_out_l1_posttrained"))

# 1. weights
after_state = model.predictor.state_dict()
num = sum(float((after_state[k].float() - base[k].float()).norm() ** 2) for k in base) ** 0.5
den = sum(float(base[k].float().norm() ** 2) for k in base) ** 0.5
print(f"1. weights: relative change ||w - w0|| / ||w0|| = {num / den:.2e}")

# 2. imagined features
track, passes, scene = load_tracking(video_of(clips))
far = far_cells(scene)
dec = eval_decoder.fit([eval_decoder.examples_from(posttrain.load_cached(cache, c),
                                                   ball_labels(source_indices(c, passes, c["n_frames"], c["context_frames"][1] + 1),
                                                               track, 16, 2)[0]) for c in train], seed=0)
for c in test:
    a, enc = imagined(c, mode)
    b = before[c["clip_id"]]
    cs = enc["context"].shape[0]
    real = enc["real"][cs:].float()
    d_ba = (a - b).abs().mean().item()
    d_br, d_ar = (b - real).abs().mean().item(), (a - real).abs().mean().item()
    mb, ma, mr = dec(b).numpy(), dec(a).numpy(), dec(real).numpy()
    print(f"{c['clip_id']} ({c['outcome']}):")
    print(f"   2. imagined features: |after - before| {d_ba:.4f}  vs |before - real| {d_br:.4f}, |after - real| {d_ar:.4f}")
    print(f"   3. decoder max P(ball) per target step  before {mb.max((1, 2)).round(3)[:6]}... after {ma.max((1, 2)).round(3)[:6]}... real {mr.max((1, 2)).round(2)[:6]}...")
    print(f"   4. max P(ball beyond plank)  before {(mb * far).max():.4f}  after {(ma * far).max():.4f}  real {(mr * far).max():.3f}")

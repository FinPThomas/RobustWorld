# V-JEPA 2

[V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) predicts in
representation space, not pixels. So it is **scored** rather than shown in the video grid.
It's self-contained and doesn't use any other model folder.

**Surprise.**
1. The context half is encoded on its own, so the encoder never sees the future.
2. The predictor imagines the target half's features.
3. These are compared (L1) with the encoder's features of the real full clip, one value
   per 2-frame target step.

This is the V-JEPA intuitive-physics protocol. A copy-the-last-context-step baseline
gives the scale.

**Where V-JEPA puts the ball.** This uses the frozen evaluation decoder,
`eval_decoder.py`, which is the only readout allowed on V-JEPA features.
- It is a linear map from features to "ball in this cell", fitted only on *real* features
  of the context half (encoded alone).
- Its labels are the ball positions in those same frames.
- It never sees predictions, target-half labels or outcomes, so it **cannot learn the
  blockade**. That has to come from post-training the world model, and this decoder stays
  the same before and after.
- `tests/test_eval_isolation.py` enforces this (see `CLAUDE.md`).
- It still finds the ball beyond the plank in real frames: AUROC ~0.99.

To run it:
- `ball_probe.py`: videos for the 5-clip sample.
- `ball_probe_cv.py`: all clips, 5-fold, with run times.

Frames are resized to 256×256 rather than cropped, so the ball leaving through the
frame edge stays in view.

**Run it.** It runs locally: about 10 s per clip on an Apple M-series or A-series GPU,
and faster on CUDA.
```bash
pip install -e . -r models/vjepa2/requirements.txt
python models/vjepa2/run.py                              # surprise, all included clips (needs data/processed)
python models/vjepa2/run.py --sample data/eval/sample5   # the committed 5-clip sample
```

Encodings are cached in `outputs/vjepa2/cache/`, so reruns take seconds.

**Blocker figures.** `python models/vjepa2/blocker_figs.py` (after `ball_probe_cv.py`) asks whether
V-JEPA knows *where* the hidden blocker is. It fits nothing: it reads `per_clip.json` from
`ball_probe_cv.py` and places each pass by where its straight-line path from the context enters and
would leave the plank. P(blocked) is 1 − max P(ball beyond the plank) over the target.
- `blocker_map.png`: every pass's path, coloured by P(blocked), with the plank and the scene file's
  `blocker_polygon` outlined. Panels: true outcome, V-JEPA imagined, and a straight line plus the
  known blocker (a reference, not a model).
- `blocked_vs_crossing.png`: P(blocked) against the expected entry point and, separately, the
  expected exit point along the plank, with the blocker's span shaded.
- `crossing.json`: per-pass numbers and AUROCs.

For before/after, run it with `--label base`, then after post-training with
`--label fine-tuned --before outputs/vjepa2/blocker/crossing.json --out <new dir>`; the base curve
is drawn dashed.

**Predictive geometry.** `python models/vjepa2/geometry.py` renders, for each sample clip:
- the real and imagined token features, projected to colour with one shared PCA basis
- per-token change against V-JEPA's own output for an empty-scene clip
- `trajectories.png`: real versus imagined paths through feature space

**Outputs** go to `outputs/vjepa2/`:
- `summary.json`: mean surprise curves per outcome (label-free)
- `per_clip.json`: per-clip surprise
- `features.npz`

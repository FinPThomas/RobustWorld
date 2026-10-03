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

**Predicted outcome.**
1. A linear probe learns through-vs-hidden from features of the *real* target halves.
2. It is then applied to V-JEPA's *imagined* target halves, to read off which outcome
   the model expects.

Probes are 5-fold cross-validated. A context-only probe shows how much the first half
alone gives away.

Frames are resized to 256×256 rather than cropped, so the ball leaving through the
frame edge stays in view.

**Run it.** It runs locally: about 10 s per clip on an Apple M-series or A-series GPU,
and faster on CUDA.
```bash
pip install -e . -r models/vjepa2/requirements.txt
python models/vjepa2/run.py                              # all included clips (needs data/processed)
python models/vjepa2/run.py --sample data/eval/sample5   # the committed 5-clip sample
```

**Ball detector (the most interpretable view).** `python models/vjepa2/ball_probe.py`

1. The tracker labels which 32 px token cell holds the ball at each step.
2. Linear detectors are trained on 30 clips outside the eval sample:
   - one on V-JEPA's **real** features
   - one on its **imagined** (predictor) features
3. On the held-out sample this gives a "where is the ball" map for both the real and the
   imagined future.
4. It also gives two curves over time: P(ball visible) and P(ball beyond the plank).

The detectors are linear and calibrated, so what they find is in V-JEPA's features.

5. Against a hand-coded **kinematic baseline** (`kinematic.py`): it fits constant velocity to
   the tracked ball over the last 1/3 s of context, then extrapolates. The ball counts as hidden
   while its centre is under the plank or out of frame. It knows nothing about the blockade, so
   it always predicts the ball coming through. It is scored like V-JEPA:
   - future-cell AUROC
   - centre error (both at token-cell resolution)
   - visibility accuracy
   - whether the ball ends up beyond the plank

   These are written to `metrics.json` under `vjepa_vs_kinematic`. The blue line in `summary.png`
   and the blue diamond in the clip videos show the kinematic prediction.
Encodings are cached in `outputs/vjepa2/cache/`, so reruns take seconds.

**Predictive geometry.** `python models/vjepa2/geometry.py` renders, for each sample clip:
- the real and imagined token features, projected to colour with one shared PCA basis
- per-token change against V-JEPA's own output for an empty-scene clip
- `trajectories.png`: real versus imagined paths through feature space

**Outputs** go to `outputs/vjepa2/`:
- `summary.json`: mean surprise curves per outcome, and probe accuracies
- `per_clip.json`: per-clip surprise and V-JEPA's predicted P(hidden)
- `features.npz`

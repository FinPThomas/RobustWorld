# Findings: V-JEPA 2 encoder probing (2026-10-05, 254 right clips)

Written up from [summary.md](vjepa2/probing/summary.md), `baseline.json`, `blockade.json` and the
figures in this folder. Clips: 111 through, 77 hidden, 66 bounce; `cv_folds` (5 folds, seed 0).
Every probe was fitted like `eval_decoder` (real context half, same-frame ball cells), so nothing
here learnt from outcomes, target labels or predictions.

## 1. Locating the ball: the readout is not the bottleneck

- Any per-token or 3×3-neighbourhood probe finds the ball almost perfectly on real frames, at
  every layer from L8 up: context cell AUROC ≈ 1.000, far-side cell AUROC 0.996–0.999 on real
  target frames (cells no probe ever saw a ball in), median error 14–20 px on a 512 px frame.
- **Pooled (global) input fails**: far-side cell AUROC 0.33–0.58, hit rate ≈ 0. Never pool.
- Other factors barely matter (paired effects, `summary.md`): MLP vs linear +0.000 far-cell
  AUROC; layer-norm vs raw +0.001; 3×3 vs token +0.008. Sigmoid beats one-ball softmax for
  location (−0.023 far-cell AUROC, −0.035 hit rate with softmax).
- Layer: L8–L16 are marginally best, but `last` is within 0.001 and is the only layer the
  predictor outputs, so it is the right one to read imagined features from.
- **The current `eval_decoder` (last layer, linear, per token, layer-normed, sigmoid) is
  close to the best config**: far-cell AUROC 0.997 vs best 0.999, real-frame outcome AUROC
  0.991. No change is needed; changing it now would also break comparison with earlier runs.

Because the frozen decoder reads the outcome from *real* target frames at AUROC ≈ 0.99–1.00, the
ceiling for an imagined future is ≈ 1.0. Any shortfall after post-training is the predictor's.

## 2. The pretrained predictor knows nothing about the blockade

- Imagined target (pretrained predictor, last layer): outcome AUROC 0.23–0.53 across the 24
  last-layer probes, one-ball P(correct) 0.30–0.34 (chance for three outcomes is 0.33).
  With `eval_decoder`'s config: outcome AUROC 0.32, imagined far-cell AUROC 0.50.
- `blocked_vs_crossing.png`, right panel (best imagined probe, last-mlp-token-raw-softmax):
  imagined P(blocked) sits at 0.7–0.95 for *every* clip, whatever its outcome or where it crosses the plank. The pretrained predictor almost
  never puts a ball on the far side, so it "predicts blocked" everywhere. Any metric that
  rewards blocked-recall alone will look good for the wrong reason; keep the balanced and
  threshold-free scores as headlines (the plan already does).
- Reference points on these clips: tracker + straight line + plank rule ("continue") 0.54;
  tracker + blockade range fitted on training-fold outcomes 0.84 (balanced accuracy 0.54,
  bounce recall 0). The 0.98 TAPNext figure was on the old 88-clip set.

## 3. Label-free blockade analyses

**Real-frame P(blocked) vs crossing point** (`blocked_vs_crossing.png`, middle): AUROC 1.0. This
is a sanity check that the decoder reads the real outcome, not evidence of blockade knowledge.
It does show the outcome is *not* purely a function of where the straight path crosses the
plank: the fitted range is y ≈ 179–325 px (rows 5–10), but several hidden clips cross at
y ≈ 55–160 or 380–400 and bounces at 330–450. That is why the outcome-fitted rule only reaches
0.84 here (37 blocked clips called through). Inferred, not checked: either the straight-line
crossing estimate is off for curving or slow balls, or the blocker sat in different places
within the right segment.

**Static distinctiveness** (`static_distinctiveness.png`): ball-free context tokens on the plank
are slightly more distinct inside the blockade rows than outside (L16 0.0084 vs 0.0065; last
0.082 vs 0.072; profile peaks at rows 6–8). But the figure shows the brighter band runs across
the whole frame width, not just the plank, so it is most likely scene structure at that height
rather than the hidden blocker. Weak evidence at best; I would not cite it as "the blockade is
visible".

**Surprise by outcome** (`surprise_by_outcome.png`): the pretrained predictor's error is higher
for hidden than through clips at the plank's exit edge in rows 6–8 (+0.012 to +0.016). This is
confounded with path: hidden clips cross inside the range and through clips mostly outside, so
the difference partly says where the ball was, not what happened to it. It needs a
path-matched comparison (below) before it means anything.

## 4. What this means for the Colab plan (PR #3)

1. **Keep `eval_decoder` exactly as is.** The sweep shows it is near-optimal, so a post-trained
   score that stays at chance is the predictor's failure, not the readout's.
2. **Report imagined far-cell AUROC and hit rate alongside outcome scores.** They separate
   "the predictor's imagined ball became readable" from "the predictor learnt the blockade".
   Pretrained values (0.50 far cell for `eval_decoder`) are the floor.
3. **Make the P(blocked)-vs-crossing-point scatter for every before/after pair.** The
   pretrained panel is a flat band; learning the blockade looks like the imagined points
   dropping to ≈ 0 outside the range and staying high inside it, matching the middle panel.
   It is label-free (outcomes only colour points), so it is allowed under rule 5/6.
4. **Read the 0.84 tracker baseline as the bar on these clips, not 0.98.**

## 5. Suggested follow-up runs (all label-free or scoring-only)

- Path-matched surprise: surprise along each clip's own path, binned by crossing y, split by
  outcome, focusing on y ≈ 150–210 and 300–350 where both outcomes occur.
- Check whether the outcome-vs-crossing outliers cluster in time within `whole.mp4` (a moved
  blocker) or are tracker/straight-line errors; if the blocker moved, a single blockade range
  is the wrong target for both the baseline and the world model.

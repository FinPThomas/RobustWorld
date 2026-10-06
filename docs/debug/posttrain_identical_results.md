# Why post-trained V-JEPA 2 scores look identical to the pretrained ones

**Status:** cause found and checked locally; fix proposed, **not yet applied** to `posttrain.py` / `eval_decoder.py`.

## Cause: the predictor and the encoder output features on different scales

V-JEPA 2's predictor was pretrained to predict **layer-normalised** encoder features (original code:
`F.layer_norm(h, (h.size(-1),))` on the targets). The Hugging Face encoder's `last_hidden_state`
keeps the final layer norm's learned scale and offset, so it is on a different scale:

| features | per-token std | token norm |
|---|---|---|
| real encoder tokens (training targets and decoder input) | 3.27 | ~104 |
| predictor's imagined tokens | 0.65 | ~20 |

Consequences:
1. **Training mostly fights the scale gap.** L1(imagined, real) ≈ 2.0, but L1(imagined, LN(real)) ≈ 0.57.
   Loss went only 2.03 → 1.99 in 3 epochs; weights moved 0.1%.
2. **The evaluation decoder reads imagined tokens as empty.** It standardises with real-feature
   statistics, so on imagined tokens it gives P(ball) ≈ 0.02 in every cell, before and after
   (0.9–1.0 on real frames).
3. **The 0.5 thresholds pin the scores.** Ball found, "through" and "bounce" all need P > 0.5, so every
   clip is called "hidden" both times: balanced accuracy, hit rate and phantom rate don't move.

## Check of the fix (CPU, 16 right clips: 12 train / 4 held out, 3 epochs, lr 3e-5)

Changes: targets layer-normalised (`F.layer_norm`, no learned scale) in training, and the decoder fitted
on and applied to layer-normalised tokens (a fixed, label-free transform, so CLAUDE.md rules still hold).

| | raw targets (current) | layer-normalised targets |
|---|---|---|
| train loss, epochs 1→3 | 2.03 → 1.99 | 0.576 → 0.556 |
| held-out L1 before → after | 2.00–2.04 → 1.94–1.98 | 0.57–0.58 → 0.55–0.56 |
| decoder peak P(ball), first imagined step, after | ~0.03 | 0.14–0.47 |

Remaining issue: P(ball) still fades to ~0.03 after the first one or two imagined steps (the predictor
blurs the possible futures). Location information is there, though: the imagined map's top cell is on
the real ball in 10–12 of 12 steps for some clips. So threshold-free scores (future-cell AUROC, mean
P(true outcome)) will show progress before balanced accuracy or hit rate do.

## Proposed changes
- `posttrain.py`: layer-normalise the targets (and `last_real`, the codebook's training tokens) for
  `l1`, `commit` and `codes`.
- `eval_decoder.py`: layer-normalise every input token, real and imagined; keep
  `tests/test_eval_isolation.py` passing.
- `experiments.py` summary: report the threshold-free scores next to the thresholded ones.

## Reproduce
`python docs/debug/posttrain_scale_repro.py 3` (current behaviour) and
`python docs/debug/posttrain_scale_repro_layernorm.py 3` (with the fix). Each takes ~20 min on CPU
plus ~1 min per clip to encode the first time. Paths at the top of each script point at this machine.

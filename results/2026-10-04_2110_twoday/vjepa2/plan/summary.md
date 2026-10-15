## Results

Through vs blocked AUROC from the imagined future (one ball, frozen decoder), cross-validated (5 folds, seed 0). `auroc_ci`: 95% bootstrap interval over clips. `delta_vs_before_ci` / `p_no_gain`: paired bootstrap against the same variant's pretrained predictor on the same clips (p_no_gain = share of resamples with no gain). TAPNext + coded blockade fits its blockade on outcomes, so it is a reference, not an evaluation. Runs named -long / -frac use the variant picked as best on these same folds, so their numbers are slightly optimistic.

| method | phase | outcome_auroc | auroc_ci | delta_vs_before_ci | p_no_gain | one_ball_p_correct | balanced_accuracy | cell_auroc | far_cell_auroc | argmax_hit_rate | overfit_folds | best_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V-JEPA 2: real target frames (ceiling) | ceiling | 0.998 | [0.995, 1.0] | - | - | 0.673 | 0.725 | 0.971 | - | - | - | - |
| V-JEPA 2 pretrained | before | 0.422 | [0.348, 0.487] | - | - | 0.328 | 0.333 | 0.658 | - | 0.199 | - | - |
| TAPNext + straight line (no blockade) | baseline | 0.536 | [0.46, 0.609] | - | - | 0.333 | 0.333 | 0.596 | - | - | - | - |
| TAPNext + coded blockade (fitted on outcomes; reference) | baseline | 0.848 | [0.797, 0.894] | - | - | 0.52 | 0.525 | 0.554 | - | - | - | - |
| V-JEPA 2 plain-after | after | 0.884 | [0.841, 0.921] | [0.38, 0.542] | 0.0 | 0.376 | 0.431 | 0.803 | - | 0.483 | 0/5 | 10/10/9/10/10 |
| V-JEPA 2 plain-before | before | 0.422 | [0.348, 0.487] | - | - | 0.328 | 0.333 | 0.658 | - | 0.199 | 0/5 | 0/0/0/0/0 |
| V-JEPA 2 commit-after | after | 0.934 | [0.903, 0.961] | [0.434, 0.591] | 0.0 | 0.46 | 0.586 | 0.834 | 0.807 | 0.58 | 0/5 | 10/7/10/9/10 |
| V-JEPA 2 codes-after | after | 0.911 | [0.87, 0.944] | [0.383, 0.56] | 0.0 | 0.52 | 0.571 | 0.731 | 0.71 | 0.463 | 0/5 | 10/10/10/10/9 |
| V-JEPA 2 codes-before | before | 0.44 | [0.369, 0.519] | - | - | 0.33 | 0.329 | 0.571 | 0.504 | 0.134 | 0/5 | 0/0/0/0/0 |
| V-JEPA 2 rollout-before | before | 0.415 | [0.346, 0.481] | - | - | 0.333 | 0.333 | 0.675 | 0.548 | 0.154 | 0/5 | 0/0/0/0/0 |
| V-JEPA 2 gate-after | after | 0.885 | [0.842, 0.924] | [0.383, 0.544] | 0.0 | 0.371 | 0.383 | 0.803 | 0.769 | 0.478 | 0/5 | 10/10/9/10/10 |
| V-JEPA 2 hyp-after | after | 0.423 | [0.355, 0.488] | [-0.072, 0.075] | 0.521 | 0.328 | 0.333 | 0.665 | 0.544 | 0.23 | 1/5 | 10/3/7/2/1 |

### Generalisation of commit (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames | mirrored: before | mirrored: after | mirrored: real frames |
|---|---|---|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.355 | 0.98 | 0.726 | 0.53 | 0.996 |

other side (from L), by height at the plank: the blockade range fitted on the seen clips (176.3-329.4 px) gets 0.855 of these outcomes right. How well the "gets through" score follows that range (AUROC): real frames 0.842948717948718, before 0.6503496503496504, after 0.31410256410256415, mirrored after 0.49533799533799533. Figure: `vjepa2/plan/generalise/commit/height.png`.

### Generalisation of plain (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames | mirrored: before | mirrored: after | mirrored: real frames |
|---|---|---|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.39 | 0.98 | 0.726 | 0.523 | 0.996 |

other side (from L), by height at the plank: the blockade range fitted on the seen clips (176.3-329.4 px) gets 0.855 of these outcomes right. How well the "gets through" score follows that range (AUROC): real frames 0.842948717948718, before 0.6503496503496504, after 0.32517482517482516, mirrored after 0.5308857808857809. Figure: `vjepa2/plan/generalise/plain/height.png`.

### Interpretation: plain-after

- Feature change on the plank: 0.188 of the total (the plank is 0.172 of the frame), correlation 0.355.
- Patching in one group: highest AUROC from `layer 11` (0.426); before 0.422, after 0.885.
- Figures: `vjepa2/plan/interpret/plain-after/change_map.png`, `layer_patching.png`.

![summary](summary.png)

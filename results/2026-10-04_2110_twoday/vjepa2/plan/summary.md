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

### Generalisation of plain (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames |
|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.388 | 0.98 |

![summary](summary.png)

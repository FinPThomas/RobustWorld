## Results

Through vs blocked AUROC from the imagined future (one ball, frozen decoder), cross-validated (5 folds, seed 0). `auroc_ci`: 95% bootstrap interval over clips. `delta_vs_before_ci` / `p_no_gain`: paired bootstrap against the same variant's pretrained predictor on the same clips (p_no_gain = share of resamples with no gain). TAPNext + coded blockade fits its blockade on outcomes, so it is a reference, not an evaluation. Runs named -long / -frac use the variant picked as best on these same folds, so their numbers are slightly optimistic.

| method | phase | outcome_auroc | auroc_ci | delta_vs_before_ci | p_no_gain | one_ball_p_correct | balanced_accuracy | cell_auroc | argmax_hit_rate | overfit_folds | best_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V-JEPA 2: real target frames (ceiling) | ceiling | 0.998 | [0.995, 1.0] | - | - | 0.673 | 0.725 | 0.971 | - | - | - |
| V-JEPA 2 pretrained | before | 0.422 | [0.348, 0.487] | - | - | 0.328 | 0.333 | 0.658 | 0.199 | - | - |
| TAPNext + straight line (no blockade) | baseline | 0.536 | [0.46, 0.609] | - | - | 0.333 | 0.333 | 0.596 | - | - | - |
| TAPNext + coded blockade (fitted on outcomes; reference) | baseline | 0.848 | [0.797, 0.894] | - | - | 0.52 | 0.525 | 0.554 | - | - | - |
| V-JEPA 2 plain-before | before | 0.422 | [0.348, 0.487] | - | - | 0.328 | 0.333 | 0.658 | 0.199 | 0/5 | 0/0/0/0/0 |

![summary](summary.png)

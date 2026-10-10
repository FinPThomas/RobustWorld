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
| V-JEPA 2 both-before | before | 0.454 | [0.389, 0.516] | - | - | 0.337 | 0.396 | 0.666 | 0.56 | 0.177 | 0/5 | 0/0/0/0/0 |
| V-JEPA 2 both-before (ball from L) | before | 0.66 | [0.541, 0.777] | - | - | 0.352 | 0.333 | - | - | - | - | - |
| V-JEPA 2 both-before (ball from R) | before | 0.437 | [0.368, 0.511] | - | - | 0.329 | 0.333 | - | - | - | - | - |

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

## Table slope from ball tracks

s: slope term along the table (px/step², downhill = left; 1 step = 1/8 s). Share of S0: the source's s over the tracker's. Measured on open-table steps only; nothing fitted to read a model. See models/vjepa2/slope.py.

| source | clips | s (1-D) | 95% CI | share of S0 | s (2-D, -Gx) | 95% CI | G (x, y) | right of plank | left of plank | degrees |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 tracker (all steps) | 327 | 2.4193 | [1.1353, 3.4569] | 1.0 | 1.768 | [1.1513, 2.2867] | [-1.768, -0.1039] | 5.6029 | -7.1953 | None |
| S0t tracker (target steps) | 176 | 0.1014 | [-0.4539, 0.7696] | 0.04 | 0.4005 | [-0.0629, 0.9099] | [-0.4005, -0.8382] | 2.2894 | None | None |
| S1 decoder, real frames | 176 | 0.2466 | [-0.8097, 1.3953] | 0.1 | -0.0916 | [-0.8077, 0.7091] | [0.0916, -0.3688] | 2.1962 | None | None |
| S2 imagined, pretrained | 176 | 0.2957 | [-4.7895, 5.3487] | 0.12 | 1.0124 | [-1.7413, 3.8264] | [-1.0124, 1.5122] | 5.2198 | None | None |

Figures: `slope/acc_vs_speed.png` (downhill and uphill acceleration by speed), `slope/gradient_map.png` (G per region).


## Research scope 2026-10-09

From per_clip.json (frozen decoder) and blocker/*/crossing.json; nothing is fitted. See models/vjepa2/scope_report.py.

### Narrow gap and both sides: mean P(ball beyond the plank) by the height the ball reaches the plank

| run | side | wide gap (104-186) | narrow gap (315-367) | blocked middle (186-315) | AUROC at y 300-380 | AUROC all |
|---|---|---|---|---|---|---|
| real frames | R | 0.947 (n 82) | 0.911 (n 26) | 0.075 (n 106) | 0.997 (n 52) | 0.998 |
| pretrained | R | 0.175 (n 82) | 0.189 (n 26) | 0.185 (n 106) | 0.451 (n 52) | 0.422 |
| plain-after | R | 0.302 (n 82) | 0.255 (n 26) | 0.153 (n 106) | 0.787 (n 52) | 0.884 |
| commit-after | R | 0.602 (n 82) | 0.437 (n 26) | 0.133 (n 106) | 0.869 (n 52) | 0.934 |
| codes-after | R | 0.846 (n 82) | 0.486 (n 26) | 0.112 (n 106) | 0.868 (n 52) | 0.911 |
| real frames | L | 0.966 (n 25) | 0.992 (n 7) | 0.481 (n 40) | 1.0 (n 10) | 0.98 |
| both-before | L | 0.735 (n 25) | 0.699 (n 7) | 0.695 (n 40) | 1.0 (n 10) | 0.66 |
| real frames | R | 0.959 (n 82) | 0.919 (n 26) | 0.07 (n 106) | 0.996 (n 52) | 0.999 |
| both-before | R | 0.168 (n 82) | 0.179 (n 26) | 0.178 (n 106) | 0.501 (n 52) | 0.437 |

Figure: `scope/height_profile.png`. The slope (downhill) results are in `slope/` (slope.py).


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
| V-JEPA 2 commit_both-after | after | 0.843 | [0.8, 0.884] | [0.338, 0.451] | 0.0 | 0.451 | 0.486 | 0.846 | 0.831 | 0.551 | 0/5 | 10/10/9/8/10 |
| V-JEPA 2 commit_both-after (ball from L) | after | 0.767 | [0.636, 0.892] | [-0.015, 0.234] | 0.049 | 0.467 | 0.545 | - | - | - | - | - |
| V-JEPA 2 commit_both-after (ball from R) | after | 0.917 | [0.881, 0.95] | [0.405, 0.554] | 0.0 | 0.475 | 0.574 | - | - | - | - | - |
| V-JEPA 2 lossmix_e10-after | after | 0.949 | [0.921, 0.973] | [0.459, 0.599] | 0.0 | 0.487 | 0.613 | 0.851 | 0.835 | 0.637 | - | - |
| V-JEPA 2 lossmix_e20-after | after | 0.964 | [0.939, 0.985] | [0.472, 0.616] | 0.0 | 0.551 | 0.651 | 0.872 | 0.88 | 0.701 | 0/5 | 18/20/17/17/18 |

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

## Slope test: uphill vs downhill at matched speed, in ball sizes

s = (downhill - uphill acceleration along the motion) / 2 at matched speed, in ball diameters per step² (1 step = 1/8 s); friction cancels. Positions are mapped onto the table with the ball's size (camera calibration from real frames only). Clips: one fit per clip over its longest run of 5+ target steps on open table; open = the open-table clips (scripts/cut_open_clips.py, scored only), plank = the plank clips' target steps. See models/vjepa2/slope_test.py.

Calibration: radius = -0.0015 x + 0.0392 y + 13.49 px (R² 0.925, 13776 frames; radius [14.97, 29.28] px across the table).

| measure | s | 95% CI | downhill a | uphill a | n down | n up |
|---|---|---|---|---|---|---|
| real ball, whole video (285 stretches) | 0.0299 | [0.02748, 0.03238] | -0.0062 | -0.0669 | 396 | 138 |
| plank: real ball (tracker) | 0.0299 | [0.02147, 0.03956] | -0.01331 | -0.0589 | 26 | 38 |
| plank: decoder on real frames | 0.0254 | [0.00857, 0.04573] | -0.04056 | -0.07571 | 31 | 42 |
| plank: imagined, both-before | None | None | None | None | 0 | 0 |
| plank: imagined, commit_both-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_e10-after | None | None | 0.01223 | None | 2 | 0 |
| plank: imagined, lossmix_e20-after | None | [-0.0875, -0.02897] | -0.02791 | 0.0797 | 13 | 1 |

### Learning curve (C: both directions, loss-weighted sampling)

| epoch | s, open clips | 95% CI | s, plank clips | 95% CI | AUROC ball from R | AUROC ball from L |
|---|---|---|---|---|---|---|
| 0 | None | None | None | None | 0.437 | 0.66 |

Imagined paths count only target steps where the imagined ball is clearly visible (map peak >= 0.2) on open table, so n shows how often the model draws the ball at all. Figures: `slope_test/epochs.png`, `slope_test/real_acc_vs_speed.png`.


## Table slope from ball tracks

s: slope term along the table (px/step², downhill = left; 1 step = 1/8 s). Share of S0: the source's s over the tracker's. Measured on open-table steps only; nothing fitted to read a model. See models/vjepa2/slope.py.

| source | clips | s (1-D) | 95% CI | share of S0 | s (2-D, -Gx) | 95% CI | G (x, y) | right of plank | left of plank | degrees |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 tracker (all steps) | 327 | 2.4193 | [1.1353, 3.4569] | 1.0 | 1.768 | [1.1513, 2.2867] | [-1.768, -0.1039] | 5.6029 | -7.1953 | None |
| S0t tracker (target steps) | 176 | 0.1014 | [-0.4539, 0.7696] | 0.04 | 0.4005 | [-0.0629, 0.9099] | [-0.4005, -0.8382] | 2.2894 | None | None |
| S1 decoder, real frames | 176 | 0.2466 | [-0.8097, 1.3953] | 0.1 | -0.0916 | [-0.8077, 0.7091] | [0.0916, -0.3688] | 2.1962 | None | None |
| S2 imagined, pretrained | 176 | 0.2957 | [-4.7895, 5.3487] | 0.12 | 1.0124 | [-1.7413, 3.8264] | [-1.0124, 1.5122] | 5.2198 | None | None |
| S3 imagined, B right only @10 | 113 | 1.1506 | [-15.2029, 18.6548] | 0.48 | -0.2173 | [-7.1715, 7.2708] | [0.2173, -5.9285] | 42.7057 | -16.8352 | None |
| S3 imagined, B right only @20 | 114 | -13.2919 | [-26.4122, 4.3605] | -5.49 | -2.8377 | [-10.4806, 5.1896] | [2.8377, -4.0594] | 33.2964 | -67.5737 | None |
| S4 imagined, A both directions | 176 | -1.3353 | [-11.9186, 8.5185] | -0.55 | -0.1729 | [-4.2703, 4.1145] | [0.1729, 0.0307] | 34.209 | -27.9645 | None |

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
| lossmix_e10-after | R | 0.642 (n 82) | 0.539 (n 26) | 0.121 (n 106) | 0.908 (n 52) | 0.949 |
| lossmix_e20-after | R | 0.819 (n 82) | 0.786 (n 26) | 0.114 (n 106) | 0.923 (n 52) | 0.964 |
| real frames | L | 0.966 (n 25) | 0.992 (n 7) | 0.481 (n 40) | 1.0 (n 10) | 0.98 |
| both-before | L | 0.735 (n 25) | 0.699 (n 7) | 0.695 (n 40) | 1.0 (n 10) | 0.66 |
| real frames | R | 0.959 (n 82) | 0.919 (n 26) | 0.07 (n 106) | 0.996 (n 52) | 0.999 |
| both-before | R | 0.168 (n 82) | 0.179 (n 26) | 0.178 (n 106) | 0.501 (n 52) | 0.437 |
| commit_both-after | L | 0.933 (n 25) | 0.438 (n 7) | 0.559 (n 40) | 0.571 (n 10) | 0.767 |
| commit_both-after | R | 0.67 (n 82) | 0.445 (n 26) | 0.13 (n 106) | 0.92 (n 52) | 0.917 |

Figure: `scope/height_profile.png`. The slope (downhill) results are in `slope/` (slope.py).


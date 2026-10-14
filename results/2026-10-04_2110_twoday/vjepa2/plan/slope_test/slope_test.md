## Slope test: uphill vs downhill at matched speed, in ball sizes

s = (downhill - uphill acceleration along the motion) / 2 at matched speed, in ball diameters per step² (1 step = 1/8 s); friction cancels. Positions are mapped onto the table with the ball's size (camera calibration from real frames only). Clips: one fit per clip over its longest run of 5+ target steps on open table; open = the open-table clips (scripts/cut_open_clips.py, scored only), plank = the plank clips' target steps. See models/vjepa2/slope_test.py.

Calibration: radius = -0.0015 x + 0.0392 y + 13.49 px (R² 0.925, 13776 frames; radius [14.97, 29.28] px across the table).

| measure | s | 95% CI | downhill a | uphill a | n down | n up |
|---|---|---|---|---|---|---|
| real ball, whole video (285 stretches) | 0.0299 | [0.02748, 0.03238] | -0.0062 | -0.0669 | 396 | 138 |
| open: real ball (tracker) | 0.0334 | [0.0185, 0.0442] | 0.00767 | -0.05048 | 25 | 14 |
| open: decoder on real frames | 0.0289 | [0.01081, 0.04464] | 0.01929 | -0.0333 | 30 | 16 |
| open: imagined, both-before | None | None | None | None | 0 | 0 |
| open: imagined, lossmix_both-after | -0.0213 | [-0.051, 0.02165] | -0.07963 | -0.02239 | 17 | 27 |
| plank: real ball (tracker) | 0.0299 | [0.02033, 0.03969] | -0.01331 | -0.0589 | 26 | 38 |
| plank: decoder on real frames | 0.0254 | [0.00703, 0.05124] | -0.04056 | -0.07571 | 31 | 42 |
| plank: imagined, both-before | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both_e1-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both_e2-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both_e4-after | None | None | 0.01075 | None | 2 | 0 |
| plank: imagined, lossmix_both_e7-after | -0.0009 | [-0.02185, 0.01887] | -0.037 | -0.03526 | 7 | 3 |
| plank: imagined, lossmix_both_e10-after | 0.0019 | [-0.03632, 0.04067] | -0.00519 | 0.00281 | 9 | 12 |
| plank: imagined, lossmix_both_e15-after | 0.0188 | [-0.02007, 0.0582] | 0.00593 | -0.03708 | 10 | 16 |
| plank: imagined, lossmix_both-after | 0.0113 | [-0.03428, 0.0616] | 0.01538 | -0.02577 | 16 | 15 |
| plank: imagined, commit_both-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_e10-after | None | None | 0.01223 | None | 2 | 0 |
| plank: imagined, lossmix_e20-after | None | [-0.09343, -0.02459] | -0.02791 | 0.0797 | 13 | 1 |
| plank: imagined, lossmix_open_e10-after | -0.0817 | [-0.14308, -0.01028] | -0.12672 | -0.01679 | 5 | 20 |
| plank: imagined, lossmix_open-after | -0.0521 | [-0.13854, 0.00163] | -0.22222 | -0.04481 | 9 | 29 |

### Learning curve (C: both directions, loss-weighted sampling)

| epoch | s, open clips | 95% CI | s, plank clips | 95% CI | AUROC ball from R | AUROC ball from L |
|---|---|---|---|---|---|---|
| 0 | None | None | None | None | 0.437 | 0.66 |
| 1 | None | None | None | None | 0.401 | 0.523 |
| 2 | None | None | None | None | 0.603 | 0.526 |
| 4 | None | None | None | None | 0.796 | 0.593 |
| 7 | None | None | -0.0009 | [-0.02185, 0.01887] | 0.888 | 0.731 |
| 10 | None | None | 0.0019 | [-0.03632, 0.04067] | 0.914 | 0.85 |
| 15 | None | None | 0.0188 | [-0.02007, 0.0582] | 0.962 | 0.891 |
| 20 | -0.0213 | [-0.051, 0.02165] | 0.0113 | [-0.03428, 0.0616] | 0.961 | 0.893 |

Imagined paths count only target steps where the imagined ball is clearly visible (map peak >= 0.2) on open table, so n shows how often the model draws the ball at all. Figures: `slope_test/epochs.png`, `slope_test/real_acc_vs_speed.png`.

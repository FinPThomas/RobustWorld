## Slope test: uphill vs downhill at matched speed, in ball sizes

s = (downhill - uphill acceleration along the motion) / 2 at matched speed, in ball diameters per step² (1 step = 1/8 s); friction cancels. Positions are mapped onto the table with the ball's size (camera calibration from real frames only). Clips: one fit per clip over its longest run of 5+ target steps on open table; open = the open-table clips (scripts/cut_open_clips.py, scored only), plank = the plank clips' target steps. See models/vjepa2/slope_test.py.

Calibration: radius = -0.0015 x + 0.0392 y + 13.49 px (R² 0.925, 13776 frames; radius [14.97, 29.28] px across the table).

| measure | s | 95% CI | downhill a | uphill a | n down | n up |
|---|---|---|---|---|---|---|
| real ball, whole video (285 stretches) | 0.0299 | [0.02748, 0.03238] | -0.0062 | -0.0669 | 396 | 138 |
| open: real ball (tracker) | 0.0186 | [0.0081, 0.03172] | -0.00475 | -0.04432 | 24 | 9 |
| open: decoder on real frames | -0.131 | [-0.41979, 0.04953] | 0.02623 | 0.15171 | 10 | 12 |
| open: imagined, lossmix_both-after | 0.0067 | [-0.01524, 0.02441] | -0.05647 | -0.00224 | 8 | 5 |
| plank: real ball (tracker) | 0.0299 | [0.02166, 0.03914] | -0.01331 | -0.0589 | 26 | 38 |
| plank: decoder on real frames | 0.0254 | [0.00698, 0.04605] | -0.04056 | -0.07571 | 31 | 42 |
| plank: imagined, both-before | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both-after | 0.0115 | [-0.0386, 0.06648] | 0.0156 | -0.02565 | 16 | 15 |
| plank: imagined, commit_both-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_e10-after | None | None | 0.01223 | None | 2 | 0 |
| plank: imagined, lossmix_e20-after | None | [-0.09133, -0.03486] | -0.02791 | 0.0797 | 13 | 1 |

### Learning curve (C: both directions, loss-weighted sampling)

| epoch | s, open clips | 95% CI | s, plank clips | 95% CI | AUROC ball from R | AUROC ball from L |
|---|---|---|---|---|---|---|
| 0 | None | None | None | None | 0.437 | 0.66 |
| 20 | 0.0067 | [-0.01524, 0.02441] | 0.0115 | [-0.0386, 0.06648] | 0.961 | 0.893 |

Imagined paths count only target steps where the imagined ball is clearly visible (map peak >= 0.2) on open table, so n shows how often the model draws the ball at all. Figures: `slope_test/epochs.png`, `slope_test/real_acc_vs_speed.png`.

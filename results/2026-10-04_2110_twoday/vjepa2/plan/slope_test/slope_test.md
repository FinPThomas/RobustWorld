## Slope test: uphill vs downhill at matched speed, in ball sizes

s = (downhill - uphill acceleration along the motion) / 2 at matched speed, in ball diameters per step² (1 step = 1/8 s); friction cancels. Positions are mapped onto the table with the ball's size (camera calibration from real frames only). Clips: one fit per clip over its longest run of 5+ target steps on open table; open = the open-table clips (scripts/cut_open_clips.py, scored only), plank = the plank clips' target steps. See models/vjepa2/slope_test.py.

Calibration: radius = -0.0015 x + 0.0392 y + 13.49 px (R² 0.925, 13776 frames; radius [14.97, 29.28] px across the table).

| measure | s | 95% CI | downhill a | uphill a | n down | n up |
|---|---|---|---|---|---|---|
| real ball, whole video (285 stretches) | 0.0299 | [0.02748, 0.03238] | -0.0062 | -0.0669 | 396 | 138 |
| open: real ball (tracker) | 0.0186 | [0.0081, 0.03172] | -0.00475 | -0.04432 | 24 | 9 |
| open: decoder on real frames | -0.131 | [-0.41979, 0.04953] | 0.02623 | 0.15171 | 10 | 12 |
| open: imagined, both-before | None | None | None | None | 0 | 0 |
| open: imagined, lossmix_both_e1-after | None | None | 0.00278 | 0.00818 | 1 | 1 |
| open: imagined, lossmix_both_e2-after | None | None | 0.01839 | 0.01757 | 4 | 1 |
| open: imagined, lossmix_both_e4-after | None | [-0.00945, 0.0156] | -0.08829 | 0.00676 | 7 | 1 |
| open: imagined, lossmix_both_e7-after | 0.0267 | [0.00798, 0.04491] | -0.05618 | -0.02564 | 6 | 3 |
| open: imagined, lossmix_both_e10-after | None | [-0.38489, 0.03746] | -0.22064 | 0.00859 | 5 | 8 |
| open: imagined, lossmix_both_e15-after | 0.0038 | [-0.00318, 0.01014] | -0.06299 | -0.01137 | 9 | 5 |
| open: imagined, lossmix_both-after | 0.0067 | [-0.01456, 0.02508] | -0.05647 | -0.00224 | 8 | 5 |
| open: imagined, lossmix_open_e10-after | 0.0273 | [-0.1349, 0.04579] | -0.06117 | -0.05383 | 7 | 6 |
| open: imagined, lossmix_open-after | -0.2025 | [-0.53015, 0.02209] | -0.23612 | 0.00443 | 4 | 6 |
| plank: real ball (tracker) | 0.0299 | [0.02085, 0.03866] | -0.01331 | -0.0589 | 26 | 38 |
| plank: decoder on real frames | 0.0254 | [0.00733, 0.04846] | -0.04056 | -0.07571 | 31 | 42 |
| plank: imagined, both-before | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both_e1-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both_e2-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_both_e4-after | None | None | 0.01075 | None | 2 | 0 |
| plank: imagined, lossmix_both_e7-after | -0.0009 | [-0.02263, 0.01712] | -0.037 | -0.03526 | 7 | 3 |
| plank: imagined, lossmix_both_e10-after | 0.0019 | [-0.03577, 0.03721] | -0.00519 | 0.00281 | 9 | 12 |
| plank: imagined, lossmix_both_e15-after | 0.0188 | [-0.01782, 0.0574] | 0.00593 | -0.03708 | 10 | 16 |
| plank: imagined, lossmix_both-after | 0.0115 | [-0.03049, 0.06412] | 0.0156 | -0.02565 | 16 | 15 |
| plank: imagined, commit_both-after | None | None | None | None | 0 | 0 |
| plank: imagined, lossmix_e10-after | None | None | 0.01223 | None | 2 | 0 |
| plank: imagined, lossmix_e20-after | None | [-0.09437, -0.03439] | -0.02791 | 0.0797 | 13 | 1 |
| plank: imagined, lossmix_open_e10-after | -0.0817 | [-0.14353, -0.01759] | -0.12672 | -0.01679 | 5 | 20 |
| plank: imagined, lossmix_open-after | -0.0521 | [-0.14367, -0.0049] | -0.22222 | -0.04481 | 9 | 29 |

### Learning curve (C: both directions, loss-weighted sampling)

| epoch | s, open clips | 95% CI | s, plank clips | 95% CI | AUROC ball from R | AUROC ball from L |
|---|---|---|---|---|---|---|
| 0 | None | None | None | None | 0.437 | 0.66 |
| 1 | None | None | None | None | 0.401 | 0.523 |
| 2 | None | None | None | None | 0.603 | 0.526 |
| 4 | None | [-0.00945, 0.0156] | None | None | 0.796 | 0.593 |
| 7 | 0.0267 | [0.00798, 0.04491] | -0.0009 | [-0.02263, 0.01712] | 0.888 | 0.731 |
| 10 | None | [-0.38489, 0.03746] | 0.0019 | [-0.03577, 0.03721] | 0.914 | 0.85 |
| 15 | 0.0038 | [-0.00318, 0.01014] | 0.0188 | [-0.01782, 0.0574] | 0.962 | 0.891 |
| 20 | 0.0067 | [-0.01456, 0.02508] | 0.0115 | [-0.03049, 0.06412] | 0.961 | 0.893 |

Imagined paths count only target steps where the imagined ball is clearly visible (map peak >= 0.2) on open table, so n shows how often the model draws the ball at all. Figures: `slope_test/epochs.png`, `slope_test/real_acc_vs_speed.png`.

# Gripper experiments: summary

Pick-up rate. Simulation: PPO at forecast uncertainty u = 0 / 0.5 / 1 (and the finger opening it uses at u = 0 vs 1). Real passes: PPO driven by the pretrained world model's forecast (before), by C's (after), and by the tracker's own exit (perfect).

| experiment | why | sim u=0 | u=0.5 | u=1 | opening u=0 -> 1 | real: before | after (C) | perfect |
|---|---|---|---|---|---|---|---|---|
| pb_base | PyBullet, default reward (early bonus 0.5, wide-grip cost 0.3, miss -1); touching the ball before the grasp is a miss | 73% | 67% | 56% | 0.643 -> 0.657 | 60% | 53% | 76% |
| box_base | the same task in plain-Python kinematics | 72% | 61% | 46% | 1.0 -> 0.987 | 48% | 36% | 75% |
| box_early1 | bigger bonus for an early pick-up | 73% | 60% | 49% | 0.988 -> 0.992 | 45% | 37% | 76% |
| box_wide06 | wide grip costs more | 73% | 61% | 50% | 0.853 -> 0.86 | 44% | 38% | 76% |
| box_wide01 | wide grip nearly free | 72% | 60% | 47% | 1.0 -> 1.0 | 48% | 38% | 75% |
| box_miss2 | a miss costs twice as much | 71% | 60% | 47% | 1.0 -> 0.997 | 45% | 34% | 73% |
| box_sight025 | slower sight (4 frames): prediction matters more | (queued) | | | | | | |
| pb_early1_wide06 | PyBullet with early pick-ups and narrow grips rewarded more | (queued) | | | | | | |
| pb_sight025 | PyBullet, slower sight | (queued) | | | | | | |
| pb_base_4m | pb_base trained on to 4M steps per fold | (queued) | | | | | | |
| pb_narrow | fingers open at most 9 cm | (queued) | | | | | | |

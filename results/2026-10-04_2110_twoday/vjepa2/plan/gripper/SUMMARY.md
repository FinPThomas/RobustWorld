# Gripper experiments: summary

Pick-up rate. Simulation: PPO at forecast uncertainty u = 0 / 0.5 / 1 (and the finger opening it uses at u = 0 vs 1). Real passes: PPO driven by the pretrained world model's forecast (before), by C's (after), and by the tracker's own exit (perfect).

| experiment | why | sim u=0 | u=0.5 | u=1 | opening u=0 -> 1 | real: before | after (C) | perfect |
|---|---|---|---|---|---|---|---|---|
| pb_base | PyBullet, default reward (early bonus 0.5, wide-grip cost 0.3, miss -1); touching the ball before the grasp is a miss | 73% | 67% | 56% | 0.643 -> 0.657 | 60% | 53% | 76% |
| box_base | the same task in plain-Python kinematics | (queued) | | | | | | |
| box_early1 | bigger bonus for an early pick-up | (queued) | | | | | | |
| box_wide06 | wide grip costs more | (queued) | | | | | | |
| box_wide01 | wide grip nearly free | (queued) | | | | | | |
| box_miss2 | a miss costs twice as much | (queued) | | | | | | |
| box_sight025 | slower sight (4 frames): prediction matters more | (queued) | | | | | | |
| pb_early1_wide06 | PyBullet with early pick-ups and narrow grips rewarded more | (queued) | | | | | | |
| pb_sight025 | PyBullet, slower sight | (queued) | | | | | | |
| pb_base_4m | pb_base trained on to 4M steps per fold | (queued) | | | | | | |
| pb_narrow | fingers open at most 9 cm | (queued) | | | | | | |

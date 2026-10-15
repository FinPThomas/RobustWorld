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
| box_sight025 | slower sight (4 frames): prediction matters more | 69% | 52% | 34% | 0.996 -> 0.977 | 31% | 23% | 71% |
| pb_early1_wide06 | PyBullet with early pick-ups and narrow grips rewarded more | 75% | 70% | 56% | 0.427 -> 0.431 | 60% | 60% | 76% |
| pb_sight025 | PyBullet, slower sight | 70% | 62% | 43% | 0.423 -> 0.48 | 46% | 38% | 74% |
| box_base_honest | box_base's policies unchanged, real forecasts stated with each part's own sigma | 72% | 61% | 46% | 1.0 -> 0.987 | 48% | 38% | 75% |
| box_indep | trained with separate uncertainties for where / when / speed; honest sigmas | 69% | 59% | 48% | 1.0 -> 1.0 | 48% | 38% | 74% |
| box_sight025_indep | box_indep with slower sight (4 frames) | 65% | 52% | 32% | 0.976 -> 0.966 | 38% | 28% | 69% |
| pb_early1_wide06_indep | pb_early1_wide06 trained with separate uncertainties; honest sigmas | 71% | 66% | 55% | 0.413 -> 0.434 | 64% | 54% | 74% |
| pb_narrow | fingers open at most 9 cm | 68% | 55% | 48% | 0.893 -> 0.908 | 53% | 57% | 74% |
| box_slowy | fast sideways (0.35 m/s), slow towards the plank (0.05 m/s): commit to a depth early | 68% | 58% | 40% | 0.712 -> 0.772 | 52% | 41% | 70% |
| pb_slowy | box_slowy in PyBullet | 69% | 63% | 55% | 0.424 -> 0.456 | 57% | 56% | 72% |
| box_slowgrip | speed-limited gripper: 0.15 m/s, fingers close at 0.1 m/s | 66% | 48% | 29% | 0.969 -> 0.966 | 33% | 26% | 70% |
| pb_slowgrip | box_slowgrip in PyBullet | 70% | 61% | 50% | 0.328 -> 0.441 | 45% | 38% | 73% |
| pb_fingerslow | pb_early1_wide06 trained on with fingers closing at most 0.1 m/s (no snapping shut) | 68% | 62% | 50% | 0.367 -> 0.409 | 50% | 42% | 71% |
| box_lineup | line up on the forecast: sideways 3 cm/s once the ball is seen, fingers open at most 8 cm | 52% | 37% | 26% | 1.0 -> 1.0 | 28% | 21% | 60% |
| pb_narrow_fs | pb_narrow trained on with fingers closing at most 0.15 m/s | 71% | 53% | 42% | 0.859 -> 0.854 | 43% | 50% | 73% |
| pb_early1_wide06_3m | pb_early1_wide06 (best on the blockade test) trained on to 3M steps per fold | 73% | 68% | 55% | 0.411 -> 0.455 | 62% | 57% | 76% |
| pb_base_4m | pb_base trained on to 4M steps per fold | 73% | 68% | 58% | 0.433 -> 0.464 | 60% | 56% | 74% |

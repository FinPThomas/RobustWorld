# Blockade test across experiments

Mean reward over all 335 real passes (141 come out, 194 blocked); see each experiment's blockade.md. Generated from models/gripper/blockade_eval.py.

| experiment | no forecast | before | after (C) | perfect | C: set off when out / when blocked, picked up |
|---|---|---|---|---|---|
| box_base | -0.13 | -0.13 | +0.21 | +0.68 | 70% / 4%, 28% |
| box_base_honest | -0.13 | -0.13 | +0.22 | +0.68 | 70% / 4%, 30% |
| box_early1 | -0.13 | -0.13 | +0.26 | +0.81 | 70% / 4%, 28% |
| box_indep | -0.13 | -0.13 | +0.22 | +0.67 | 70% / 4%, 29% |
| box_miss2 | -0.13 | -0.13 | +0.20 | +0.66 | 70% / 4%, 26% |
| box_sight025 | -0.13 | -0.13 | +0.13 | +0.62 | 69% / 4%, 18% |
| box_sight025_indep | -0.13 | -0.13 | +0.15 | +0.61 | 69% / 4%, 21% |
| box_wide01 | -0.13 | -0.13 | +0.22 | +0.68 | 70% / 4%, 30% |
| box_wide06 | -0.13 | -0.13 | +0.22 | +0.68 | 70% / 4%, 30% |
| pb_base | -0.13 | -0.13 | +0.33 | +0.68 | 70% / 4%, 43% |
| pb_base_4m | -0.13 | -0.13 | +0.34 | +0.66 | 70% / 4%, 45% |
| pb_early1_wide06 | -0.13 | -0.13 | +0.45 | +0.78 | 70% / 4%, 50% |
| pb_early1_wide06_indep | -0.13 | -0.13 | +0.39 | +0.76 | 70% / 4%, 43% |
| pb_narrow | -0.13 | -0.13 | +0.36 | +0.67 | 70% / 4%, 48% |
| pb_sight025 | -0.13 | -0.13 | +0.22 | +0.63 | 70% / 4%, 30% |

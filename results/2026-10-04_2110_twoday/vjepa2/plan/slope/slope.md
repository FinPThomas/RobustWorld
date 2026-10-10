## Table slope from ball tracks

s: slope term along the table (px/step², downhill = left; 1 step = 1/8 s). Share of S0: the source's s over the tracker's. Measured on open-table steps only; nothing fitted to read a model. See models/vjepa2/slope.py.

| source | clips | s (1-D) | 95% CI | share of S0 | s (2-D, -Gx) | 95% CI | G (x, y) | right of plank | left of plank | degrees |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 tracker (all steps) | 327 | 2.4193 | [1.1353, 3.4569] | 1.0 | 1.768 | [1.1513, 2.2867] | [-1.768, -0.1039] | 5.6029 | -7.1953 | None |
| S0t tracker (target steps) | 176 | 0.1014 | [-0.4539, 0.7696] | 0.04 | 0.4005 | [-0.0629, 0.9099] | [-0.4005, -0.8382] | 2.2894 | None | None |
| S1 decoder, real frames | 176 | 0.2466 | [-0.8097, 1.3953] | 0.1 | -0.0916 | [-0.8077, 0.7091] | [0.0916, -0.3688] | 2.1962 | None | None |
| S2 imagined, pretrained | 176 | 0.2957 | [-4.7895, 5.3487] | 0.12 | 1.0124 | [-1.7413, 3.8264] | [-1.0124, 1.5122] | 5.2198 | None | None |

Figures: `slope/acc_vs_speed.png` (downhill and uphill acceleration by speed), `slope/gradient_map.png` (G per region).

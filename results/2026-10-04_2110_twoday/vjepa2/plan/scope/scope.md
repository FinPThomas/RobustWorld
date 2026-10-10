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

Figure: `scope/height_profile.png`. The slope (downhill) results are in `slope/` (slope.py).

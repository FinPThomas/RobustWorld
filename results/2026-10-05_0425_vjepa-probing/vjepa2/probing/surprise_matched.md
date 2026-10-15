# Predictor surprise, blocked vs through on matched paths

Pretrained predictor; each blocked clip paired with the through clip nearest in plank crossing (y) and speed. Positive = more surprise on the blocked clip.

| outcome | pairs | match distance (cells) | whole frame: mean diff (wins, p) | plank cells | per target step (whole frame) |
|---|---|---|---|---|---|
| hidden | 77 | 0.71 | +0.0032 (47/77, p=0.0147) | +0.0037 (48/77, p=0.0027) | -0.002 +0.020 +0.007 +0.011 +0.002 +0.012 +0.009 +0.005 -0.002 +0.001 -0.006 -0.019 |
| bounce | 63 | 1.5 | +0.0175 (59/63, p=0.0) | +0.0152 (59/63, p=0.0) | +0.013 +0.035 +0.011 +0.023 +0.019 +0.018 +0.018 +0.022 +0.020 +0.019 +0.007 +0.005 |

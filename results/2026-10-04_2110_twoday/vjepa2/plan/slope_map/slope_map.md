## Slope map (uniform pull over the whole table)

g = acceleration from the table after removing friction (a = g - mu * direction of motion), ball diameters per step²; x is left/right in the image (negative = pulled left = downhill), y is towards the camera. 95% intervals by bootstrap over clips. Figure: `slope_map/slope_map.png`.

| track | g_x | 95% CI | g_y | 95% CI | friction mu | stretches | clips |
|---|---|---|---|---|---|---|---|
| real ball (tracker) | -0.0267 | [-0.03, -0.0234] | -0.0086 | [-0.0106, -0.0069] | 0.0352 | 985 | 235 |
| probe on real frames (coarse) | -0.0309 | [-0.0388, -0.0225] | -0.0063 | [-0.0098, -0.0029] | 0.0269 | 632 | 176 |
| probe on real frames (sharp) | -0.0384 | [-0.0461, -0.0307] | -0.0068 | [-0.01, -0.0036] | 0.0285 | 621 | 169 |
| imagined, before post-training (sharp) | - | - | - | - | - | 0 | 0 |
| imagined, after post-training (C, 20 ep) (sharp) | -0.0214 | [-0.0439, 0.0002] | -0.0115 | [-0.0198, -0.0038] | 0.0257 | 292 | 95 |
| imagined, after (E: + open clips) (sharp) | 0.0069 | [-0.0336, 0.0583] | -0.0078 | [-0.0181, 0.0023] | 0.0505 | 77 | 36 |

### Downhill vs uphill on the same steps as the real ball

Acceleration along the motion (negative = slowing), median over stretches where both the track and the real ball are usable, split by the real ball's direction (left = downhill). s = (down - up) / 2; the real ball's s on the same stretches says what a track that follows the slope would give.

| track | stretches down / up | track: down | up | s | 95% CI | real ball: down | up | s | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| probe on real frames (sharp) | 202 / 216 | -0.0118 | -0.0395 | 0.0138 | [0.0048, 0.0232] | 0.0009 | -0.0486 | 0.0248 | [0.0202, 0.0312] |
| imagined, after post-training (C, 20 ep) | 70 / 65 | -0.0126 | -0.0504 | 0.0189 | [-0.0183, 0.0511] | 0.0033 | -0.0606 | 0.032 | [0.0203, 0.0422] |
| imagined, after (E: + open clips) | 2 / 28 | - | - | - | - | - | - | - | - |

## Slope map (uniform pull over the whole table)

g = acceleration from the table after removing friction (a = g - mu * direction of motion), ball diameters per step²; x is left/right in the image (negative = pulled left = downhill), y is towards the camera. 95% intervals by bootstrap over clips. Figure: `slope_map/slope_map.png`.

| track | g_x | 95% CI | g_y | 95% CI | friction mu | stretches | clips |
|---|---|---|---|---|---|---|---|
| real ball (tracker) | -0.0265 | [-0.0301, -0.0233] | -0.0089 | [-0.0113, -0.0069] | 0.0361 | 955 | 230 |
| probe on real frames (coarse) | -0.0157 | [-0.0394, 0.0292] | -0.016 | [-0.0334, -0.006] | 0.0201 | 355 | 114 |
| probe on real frames (sharp) | -0.0251 | [-0.0401, -0.0063] | -0.0138 | [-0.0217, -0.0078] | 0.0225 | 348 | 107 |
| imagined, before post-training (sharp) | - | - | - | - | - | 0 | 0 |
| imagined, after post-training (C, 20 ep) | 0.0069 | [-0.0152, 0.0314] | -0.0104 | [-0.0197, -0.0015] | 0.0071 | 144 | 56 |
| imagined, after (E: + open clips) (sharp) | 0.0126 | [-0.0274, 0.067] | -0.0097 | [-0.0244, 0.002] | 0.0513 | 128 | 54 |

### Downhill vs uphill on the same steps as the real ball

Acceleration along the motion (negative = slowing), median over stretches where both the track and the real ball are usable, split by the real ball's direction (left = downhill). s = (down - up) / 2; the real ball's s on the same stretches says what a track that follows the slope would give.

| track | stretches down / up | track: down | up | s | 95% CI | real ball: down | up | s | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| probe on real frames (sharp) | 87 / 129 | -0.0083 | -0.0412 | 0.0164 | [-0.0023, 0.0348] | -0.0045 | -0.0536 | 0.0246 | [0.0181, 0.0298] |
| imagined, after post-training (C, 20 ep) | 14 / 39 | -0.0195 | 0.0086 | -0.0141 | [-0.0687, 0.0378] | 0.0073 | -0.0497 | 0.0285 | [0.0144, 0.0408] |
| imagined, after (E: + open clips) | 12 / 36 | 0.0483 | -0.0022 | 0.0253 | [-0.0602, 0.0911] | 0.0005 | -0.0446 | 0.0226 | [0.0138, 0.0328] |

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

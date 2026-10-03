| method | clips | where_ball_goes_cell_auroc | outcome_auroc | outcome_accuracy | hidden_called_through | through_called_hidden |
|---|---|---|---|---|---|---|
| V-JEPA 2: real target frames, frozen decoder (upper bound) | 88 | 0.993 | 0.993 | 0.955 | 0 | 4 |
| V-JEPA 2 pretrained: imagined target (before post-training) | 88 | 0.675 | 0.398 | 0.273 | 0 | 64 |
| TAPNext + straight-line continuation | 88 | 0.777 | 0.609 | 0.75 | 21 | 1 |
| TAPNext + continuation + plank/blockade rules (blockade fitted on outcomes) | 88 | 0.791 | 0.98 | 0.966 | 1 | 2 |

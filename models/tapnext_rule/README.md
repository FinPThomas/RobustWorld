# TAPNext + rules (hand-engineered baseline)

There is no physics learning here. This is the baseline the world models should beat.

1. **Track.** BootsTAPNext follows 5 points on the ball through the context half only.
   The ball is rolling, so each surface point turns out of view; the centre is the mean
   of the visible points.
2. **Continue.** A straight line is fitted to the last 8 centres and extended at constant
   velocity.
3. **Rules.**
   - The ball is invisible inside the plank outline.
   - If the line crosses the plank's centre line within the blockade range, the ball
     stops there for good.
   - The ball is gone once it leaves the frame.
   - The blockade range along the plank is fitted on each fold's training clips. It
     comes out at about y = 178–247 px on every fold.
   - Variant `continue` drops the blockade rule.

**Scoring.** It uses the same ground truth, 32 px cell grid and 5 folds as the V-JEPA 2 ball
probe (`robust_world.eval.ball`). Compare them with `robustworld-eval ball-compare`.

**Pixel predictions.** For the eval sample, the ball is cut from the last context frame
and pasted along the predicted path behind the plank. The predictions are written to
`outputs/predictions/sample5/tapnext_{continue,blockade}`, so they show up in
`robustworld-eval grid`.

```bash
pip install -e . -r models/tapnext_rule/requirements.txt
python models/tapnext_rule/predict.py
```

**Speed.** Tracking takes about 0.76 s per frame on an Apple GPU, roughly 10 s per clip.
The continuation and rules take milliseconds.

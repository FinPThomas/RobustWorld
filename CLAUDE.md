# RobustWorld

The goal is to test whether a world model can learn an invisible blockade under a plank, purely
by post-training on video. A ball rolls under a plank; some passes come out the far side
("through") and some never do ("hidden").

- Clips pipeline: `robustworld` (`src/robust_world/pipeline.py`)
- Evaluation tools: `robustworld-eval` (`src/robust_world/eval/`)
- Models: one self-contained folder each under `models/`

## Evaluation rule: the evaluation must never be able to learn the blockade

Learning the blockade belongs to **post-training the world model**. Anything that reads a
model's output (decoders, probes, classifiers, retrieval keys) is a measuring instrument. It
must be identical before and after post-training, and it must carry no knowledge of outcomes.

1. **Never** train a readout on model predictions (e.g. V-JEPA "imagined" features, generated
   frames) against what really happened next.
2. **Never** train a readout on outcomes (through/hidden) or on target-half labels.
3. The only trained readout on V-JEPA features is `models/vjepa2/eval_decoder.py`:
   - linear;
   - fitted on real, context-half features only (encoded alone);
   - labelled with ball positions from the same frame.

   Build its data only with `eval_decoder.examples_from()`. Don't add parameters that let it
   see anything else.
4. `tests/test_eval_isolation.py` enforces rules 1–3: the decoder must be bit-for-bit unchanged
   when outcomes, target labels or predictions change. Run `python -m pytest tests/` after
   touching any evaluation code. Never weaken this test to make something pass.
5. Label-free measures (surprise/prediction error, nearest-neighbour in the model's own feature
   space) need no readout and are always fine.
6. Ground truth (tracker ball positions, outcomes) may be used to **score** predictions, never
   to **fit** anything that sits between a model's output and the score.
7. Baselines that learn from outcomes by design are allowed only if their name says so:
   - The TAPNext rule baseline fits its blockade range on training-fold outcomes.
   - Such baselines are reference models, not evaluation tools.

If a requested analysis would need a readout trained on predictions, outcomes or future labels,
stop and say so instead of building it.

## Other conventions

- Cross-validate with `robust_world.eval.ball.cv_folds` (5 folds, seed 0), so every model is
  scored on the same splits.
- `data/eval/sample5` is the committed 5-clip sample for Colab and video grids.
- `data/processed` and `outputs/` are regenerable and gitignored.

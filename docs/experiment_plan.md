# Experiment plan: can post-training teach V-JEPA 2 the hidden blockade?

Written 2026-10-04. This file is the guide for the next two days of experiments. It covers the plan, the
changes made so far and where the experiments stand. The **Status** section at the bottom is rewritten by
the Colab notebook after every step it finishes.

Run it with [`notebooks/colab_two_day_plan.ipynb`](../notebooks/colab_two_day_plan.ipynb). The runner is
`models/vjepa2/plan.py`.

## The question

A ball rolls under a plank. Some passes come out the far side ("through"). Others never reappear,
because a blocker hidden under the plank stops them ("hidden"). The blocker can't be seen. The only way
to know where it is comes from watching what happens to the ball.

**Claim to test:** post-training V-JEPA 2's predictor on our videos gives a *measurable* improvement in
predicting which passes are blocked, and the pretrained model can't do this. We then compare it with a
hand-coded tracker that is told where the blockade is. We check whether it generalises to the ball
coming from the other side and to a new ball. Finally, we look at what changed inside the model.

**What counts as success**
- The through-vs-blocked AUROC of the imagined future goes up after post-training. The paired bootstrap
  interval of (after − before) must exclude 0 on the same clips.
- The score gets as close as possible to the reference points: TAPNext with the coded blockade (0.98) and
  the real-frames ceiling (≈0.99).
- The gain holds up with longer training. It grows with more training clips. It carries over to held-out
  directions.

## Ground rules (from `CLAUDE.md`, unchanged)

- Only V-JEPA post-training may learn the blockade. Every V-JEPA score is read through the **frozen
  evaluation decoder** (`models/vjepa2/eval_decoder.py`). That decoder is linear and is fitted per fold, only
  on real context-half features with same-frame ball labels. Outcomes, predictions and future labels never
  reach it. `tests/test_eval_isolation.py` enforces this.
- The encoder stays frozen, and only the predictor is post-trained. This keeps the decoder bit-for-bit the
  same before and after.
- Cross-validation uses 5 folds with seed 0 (`robust_world.eval.ball.cv_folds`), the same splits for every
  model.
- TAPNext with the coded blockade fits its blockade on training-fold outcomes. It is a reference model,
  not an evaluation.

## Where things stand (2026-10-04, before this plan runs)

| method | through-vs-blocked AUROC | source |
|---|---|---|
| V-JEPA 2 pretrained, imagined future | 0.40 | `docs/results/ball_comparison.md` (old max-P metric) |
| TAPNext + straight line, no blockade | 0.61 | same |
| TAPNext + straight line + coded blockade | **0.98** | same |
| Real target frames through the frozen decoder (ceiling) | 0.99 | same |

No post-trained results are on GitHub yet: `results/README.md` is empty. The pretrained model hedges.
It imagines no ball beyond the plank at all and calls 64 of 64 through passes hidden.

## Changes so far

All the post-training work so far is from PR #1 (`claude/project-thread-qvh5by`):

1. Cached frozen-encoder features. Per-fold predictor post-training (`posttrain.py`), with the encoder
   frozen.
2. Training and scoring in V-JEPA's own layer-normalised space. This fixed a 5× scale mismatch between
   encoder features and predictor output. The γ·pred+β mapping into encoder space is kept as an
   alternative reading.
3. The model fits on a 16 GB T4: fp16, gradient checkpointing, batch 1 × 8 accumulation steps.
4. The objectives:
   - `plain`: V-JEPA latent L1.
   - `commit`: motion-weighted L1 plus an in-batch contrast term.
   - `codes`: a 256-entry k-means codebook, cross-entropy, outputs snapped to codes.
   - `rollout`: step-by-step prediction fed back in.
5. Threshold-free "one ball" metrics. Guards against leakage and overfitting: a validation split, keeping
   the best epoch, early stopping, an overfitting flag. Detailed per-epoch logs.
6. An unattended overnight grid that pushes results to `results/<date>_<time>_<name>/`.

This plan (PR for `claude/colab-two-day-plan-3gn90a`) adds:

7. **Architecture options** in `posttrain.py`. Both are label-free and both start as exactly the
   pretrained output:
   - `--copy-gate`: a per-token gate that can copy the last context frame. The static scene comes out
     clean, and the predictor's capacity goes to the ball.
   - `--hypotheses K`: K futures with a picker, trained winner-takes-all, so the model commits to
     "through" or "hidden" instead of averaging the two. The picker is part of the world model. Like L1,
     it learns only from the training clips' own video. The evaluation still reads whatever future it
     picks through the frozen decoder.
8. `--train-frac` for the data-efficiency runs, and `--resume`, which skips folds already trained with the
   same settings, so a Colab disconnect loses at most one fold.
9. `plan.py`: the stage runner. It keeps its state on Drive, rebuilds the report and pushes after every
   step.
10. `generalise.py`: held-out other-side and new-ball scoring.
11. `interpret.py`: change maps and layer patching.
12. A paired bootstrap of before vs after on the same clips.

## The plan

Each stage is one notebook cell. Every step can be repeated safely: finished steps are skipped, and a
step cut off by a disconnect runs again. The time column is a rough guess for a T4. The status table
below records the real minutes, so later stages can be budgeted from it.

| stage | steps | why | rough T4 time |
|---|---|---|---|
| 1 baselines | split, encode, pretrained, tapnext | The "before" and the references, all on the same clips and folds. | 1–2 h |
| 2 post-training | plain, commit, codes, rollout, codes_rollout (before + after, 10 epochs) | Does post-training learn the blockade at all, and which objective does it best? | 15–20 h (rollout runs are slowest) |
| 3 architecture | gate, hyp, gate_hyp_commit (10 epochs) | Can a cleaner, committed output beat the plain predictor? | 5–7 h |
| 4 robustness | best two × 30 epochs; best on 50% and 25% of clips | Does the gain last with longer training, and does it grow with data? | 8–10 h |
| 5 generalisation | train best on all seen clips; score other-side / new-ball clips | Has it learnt "an object at a place", not a one-way rule? Waits until those recordings exist. | 1–2 h |
| 6 interpretation | change maps, layer patching for the best run | Where in the image and in the network does the change sit? | ~1 h |

That adds up to roughly two days of GPU time. It will take about 4–5 Colab sessions on the free tier.
If time runs short, the order is the priority: stage 2's `plain`, `commit` and `codes` matter most.

### How each run is scored (same for every row)

- **Through-vs-blocked AUROC.** From the imagined future's share of the ball beyond the plank (one ball,
  frozen decoder), cross-validated. It comes with a 95% bootstrap interval. For post-trained runs it also
  comes with a paired bootstrap of (after − before) on the same clips, and `p_no_gain`, the share of
  resamples with no gain.
- Also reported:
  - balanced P(correct outcome);
  - balanced accuracy;
  - ball-cell AUROC;
  - the "most likely cell within 48 px" hit rate;
  - training health: folds flagged as overfitting, and the best epochs.
- The "best" variant for stages 4–6 is picked on the same folds it is reported on. So stage 4's numbers
  for it are slightly optimistic. Stage 5 (held out) is the clean check.

### Generalisation tests (stage 5)

- **Other side.** Record one session with the ball rolled from the other side. It is not trained on. The
  split is automatic: `passes.json` `side_in` differs from the main direction. For those clips, "far" and
  "near" are swapped.
- **New ball.** Add the video's name to `configs/heldout.json`, e.g. `{"videos": ["newball1"]}`.
- Pack the new clips into `robustworld_clips.zip` as usual. The next run of stage 5 (or of "Run
  everything left") picks them up.

### Interpretation (stage 6)

- **Change maps.** Where does the imagined future differ after post-training? This is shown for the
  features, and for the decoder's ball probability on through vs blocked clips. It is drawn over the
  frame with the plank outlined. The step reports what share of the change sits on the plank, and the
  per-cell correlation with it.
- **Layer patching.** Move one group of predictor weights at a time (embeddings, each layer, the output)
  from post-trained into pretrained, and the reverse. This shows where the blockade knowledge is stored.

### Separate track: encoder probing on Fin's computer

A separate session on Fin's own computer probes how best to read the encoded state. It runs overnight in
VS Code and doesn't need Colab. Its findings may change *how* we read results, but the frozen-decoder
rule above stays.

## What is saved, and where

After every step, `results/<date>_<time>_twoday/` on this branch is updated with:
- `vjepa2/plan/summary.md`, `summary.json`, `summary.png` and `status.md`: the report, plus the plan's
  state (`state.json`), including which variants were picked as best.
- `vjepa2/plan/scores/<run>/`: each run's metrics, per-clip readouts and figures, together with its
  training log and run settings (`train_log.json`, `train_run_info.json`).
- `vjepa2/plan/blocker/`, `baselines/tapnext/`, `generalise/`, `interpret/`: the blocker figures,
  TAPNext scores, held-out scores and interpretation outputs.
- `vjepa2/plan/logs/<step>.txt`: the full console output of every step, for debugging.

The status table in this file is updated at the same time. If a push fails, for example because the
network drops, everything goes up with the next step's push. Only model weights and the feature cache
stay on Drive, because they are too large for git. The clean-up cell pushes once more and deletes
nothing unless that push succeeds.

## Known limits

- Stages 1–4 read ball tracking for the `start` video only (`load_tracking`). Extra sessions *in the
  training direction* need that generalised before they can join training. Held-out sessions already
  work in stage 5.
- The TAPNext baseline runs on every included clip. Once held-out clips exist, its folds differ slightly
  from the V-JEPA runs.
- Drive needs about 8 GB: features of about 2 GB, plus about 0.5 GB per post-trained run. Before-runs'
  weights are deleted after scoring. This is working space only: every score, figure and log is on GitHub, and the
  notebook's last cell frees it (optionally keeping the best run's weights) once the plan is finished.

## Next ideas (not in this plan)

- A counterfactual sweep: paste the ball at different positions in one context, and plot P(blocked) along
  the plank.
- Discrete latent modes.
- An adversarial objective.
- Object-slot sparsity.
- An angled deflector session.
- Encoder LoRA is not recommended: it would change what the frozen decoder reads.

## Status

Rewritten by the notebook after every step. Don't edit between the markers.

<!-- status:start -->
_No steps run yet. Open `notebooks/colab_two_day_plan.ipynb` in Colab, run cell 1 (Google Drive, mounted and tested first) and cell 2 (Setup), then the stage cells._
<!-- status:end -->

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
9. `plan.py`: the stage runner. It works on Colab's local disk and backs up to Drive after every step and every
   10 minutes (`scripts/drive_sync.py`), so a Drive drop no longer interrupts a step; it rebuilds the report and
   pushes after every step.
10. `generalise.py`: held-out other-side and new-ball scoring.
11. `interpret.py`: change maps and layer patching.
12. A paired bootstrap of before vs after on the same clips.
13. (2026-10-06, after a Colab GPU timeout) Training saves its full state after every epoch
    (`<fold>.partial.pt`, backed up to Drive) and carries on from there, so a stop costs at most one epoch;
    batch 4 x 2 instead of 1 x 8 (same effective batch; batch 1 used 1.15 GB of the T4); and generalisation and
    interpretation of `plain` run straight after `plain-after`, before the other variants.

14. (2026-10-06) Kaggle: `notebooks/kaggle_plan.ipynb` runs the same plan on Kaggle's free GPU (about 30 h a week,
    12 h background sessions). Each session restores the plan's progress from the newest results folder on GitHub
    (`plan.py restore`), re-encodes the clips, runs steps until about 11 h (`--budget-hours`) and pushes after every
    step. Order is proof of concept first: plain, plain on the other side, commit, copy gate, interpretation of
    plain, multi-hypothesis, then the rest of the grid, then the long and smaller-data runs.
15. (2026-10-09) Research scope (`results/overview/RESEARCH_SCOPE.md` on the overview branch) replaces the rest
    of the queue (rollout, codes_rollout, gate_hyp_commit, 30-epoch and data-fraction runs are dropped). In order:
    `both-before` (pretrained, both directions, per-step ball positions: slope S0-S2), `lossmix_e20-after` (B:
    commit on right clips with `--sample-by-loss 0.25`, 20 epochs, its epoch-10 weights scored as
    `lossmix_e10-after`), `commit_both-after` (A: commit, `cv_folds` over all 337 clips, each side scored),
    `other-ball` (D, waits for its clips). `slope.py` (table slope from ball tracks, S0-S4) and
    `scope_report.py` (narrow gap, both sides) run in every report. Fold weights are also saved in float16 to
    `$ROBUSTWORLD_WEIGHTS` (`/kaggle/working/weights` on Kaggle).
16. (2026-10-10) The overnight run finished 15 (B best: 0.964; A lifted the left side to 0.77; slope.py
    inconclusive: over the predicted steps even real frames can't show the slope in pixels). Fin: show the slope
    is learnt, uphill vs downhill at matched speed, in ball sizes, with a learning curve. `slope_test.py` maps
    positions onto the table with the ball's size (radius linear in image position, fitted on the tracker's real
    frames: camera geometry only), measures the real slope over the whole video, and per run compares each
    through pass's imagined path with steady speed from its entry (downhill = ball from the right, uphill = from
    the left). New steps: `lossmix_both-after` (C: B's recipe on both directions, 20 epochs, weights saved after
    1, 2, 4, 7, 10, 15) and `lossmix_both-epochs` (scores those); `slope_test/epochs.png` plots slope and AUROC
    per side against epochs. Open-table clips (`scripts/cut_open_clips.py`, 92 windows chosen from the tracker
    alone where the ball rolls on open table across the context/target boundary; `robustworld_clips_open.zip`)
    are scored only (`ball_probe_cv.py --extra-manifest`, each by the fold holding out the clip that shares its
    frames) for every epoch; C waits until that zip is on the machine. Real slope from the tracker over the whole
    video: 0.031 ball diameters/step² [0.028, 0.033], the same in every speed band. Queued after C:
    `lossmix_open-after` (E: C with the open-table clips added to training via `posttrain.py --extra-train`,
    each kept out of the fold that scores it; scored at 10 and 20 epochs): does free rolling teach the slope
    better? Then `other-ball` (D, waits for its clips).

First result (2026-10-06, `plain-after`): through-vs-blocked AUROC 0.42 pretrained -> 0.88 post-trained (paired
gain 95% CI [0.38, 0.54]); TAPNext + coded blockade 0.85; real frames 0.998. The imagined ball still rarely shows
clearly past the plank (P(blocked) ~0.97 for every pass, unnormalised), so the gain is in where the predicted
ball mass lies, not yet a clean "ball comes out" picture.

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

- **Other side.** This is the `left` segment of `whole.mp4`: the ball rolled in from the left, from 1708 s
  on. Training and stages 1–4 use only the right segment (`manifest_right.jsonl`). To run the test, pack
  the left segment with `python scripts/pack_clips.py --manifest data/processed/clips/manifest_left.jsonl
  --out outputs/robustworld_clips_left.zip` and put the zip in `MyDrive/RobustWorld/`. Setup unpacks every
  `robustworld_clips*.zip` it finds there. For left clips, "far" and "near" are swapped.
- **New ball.** Add the video's name to `configs/heldout.json`, e.g. `{"videos": ["newball1"]}`.
- The next run of stage 5, or of "Run everything left", picks the new clips up.

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

- Stages 1–4 read ball tracking for one video at a time (`video_of`), currently `whole`. Extra sessions
  *in the training direction* need that generalised before they can join training. Held-out sessions
  already work in stage 5.
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
_Last update: 2026-10-10 19:08 UTC from Kaggle (Tesla T4), code `a70bd48`. Results: [results/2026-10-04_2110_twoday](../results/2026-10-04_2110_twoday/README.md)._

| stage | step | state | minutes | finished (UTC) |
|---|---|---|---|---|
| 1 | split: set aside held-out clips (other side, new ball) | done | 0.0 | 2026-10-04 21:10 |
| 1 | encode: encode every clip once (frozen encoder) | done | 27.1 | 2026-10-04 22:11 |
| 1 | pretrained: pretrained V-JEPA 2, frozen decoder | done | 17.7 | 2026-10-06 12:55 |
| 1 | tapnext: TAPNext + rules, with and without the coded blockade | done | 11.8 | 2026-10-06 13:07 |
| 2 | plain-before: plain: pretrained predictor, scored the plain way | done | 18.9 | 2026-10-06 13:26 |
| 2 | plain-after: plain: post-trained 10 epochs | done | 140.4 | 2026-10-06 15:46 |
| 2 | generalise-plain: plain: ball from the other side, before vs after | done | 31.0 | 2026-10-06 19:05 |
| 2 | commit-after: commit: post-trained 10 epochs | done | 126.1 | 2026-10-06 21:11 |
| 3 | gate-after: gate: post-trained 10 epochs | done | 123.8 | 2026-10-06 23:15 |
| 2 | interpret-plain: plain: change maps and layer patching | done | 183.7 | 2026-10-07 02:18 |
| 3 | hyp-after: hyp: post-trained 10 epochs | done | 90.9 | 2026-10-07 03:49 |
| 2 | generalise-commit: commit: ball from the other side, before vs after | done | 30.8 | 2026-10-07 09:55 |
| 2 | mirror-plain: plain: other side mirrored, and by height at the plank | done | 38.2 | 2026-10-08 19:57 |
| 2 | mirror-commit: commit: other side mirrored, and by height at the plank | done | 32.8 | 2026-10-08 20:29 |
| 2 | codes-before: codes: pretrained predictor, scored the codes way | done | 11.5 | 2026-10-07 04:01 |
| 2 | codes-after: codes: post-trained 10 epochs | done | 120.3 | 2026-10-07 11:55 |
| 2 | both-before: slope S0-S2: pretrained, both directions, per-step ball positions | done | 13.9 | 2026-10-10 00:46 |
| 2 | lossmix_e20-after: B. commit, right clips, loss-weighted sampling, 20 epochs (scored at 10 and 20) | done | 237.7 | 2026-10-10 04:44 |
| 2 | commit_both-after: A. commit trained on both directions, each side scored | done | 156.4 | 2026-10-10 07:20 |
| 2 | lossmix_both-after: C. both directions, loss-weighted sampling, 20 epochs (weights saved for the learning curve) | waiting (the open-table clips aren't on this machine (add robustworld_clips_open.zip to the dataset)) | 0.0 | 2026-10-10 19:08 |
| 2 | lossmix_both-epochs: C. learning curve: score the pretrained predictor and the weights saved after 1, 2, 4, 7, 10 and 15 epochs, plank and open-table clips | waiting (the open-table clips aren't on this machine (add robustworld_clips_open.zip to the dataset)) | 0.0 | 2026-10-10 19:08 |
| 2 | lossmix_open-after: E. C plus the open-table clips in training (slope from free rolling), 20 epochs, scored at 10 and 20 | to do |  |  |
| 5 | other-ball: D. the other ball, scored without training | waiting (the other ball's clips aren't added yet (configs/heldout.json and their clips zip)) | 0.0 | 2026-10-10 14:13 |

| method | phase | outcome_auroc | auroc_ci | delta_vs_before_ci | p_no_gain |
|---|---|---|---|---|---|
| V-JEPA 2: real target frames (ceiling) | ceiling | 0.998 | [0.995, 1.0] | - | - |
| V-JEPA 2 pretrained | before | 0.422 | [0.348, 0.487] | - | - |
| TAPNext + straight line (no blockade) | baseline | 0.536 | [0.46, 0.609] | - | - |
| TAPNext + coded blockade (fitted on outcomes; reference) | baseline | 0.848 | [0.797, 0.894] | - | - |
| V-JEPA 2 plain-after | after | 0.884 | [0.841, 0.921] | [0.38, 0.542] | 0.0 |
| V-JEPA 2 plain-before | before | 0.422 | [0.348, 0.487] | - | - |
| V-JEPA 2 commit-after | after | 0.934 | [0.903, 0.961] | [0.434, 0.591] | 0.0 |
| V-JEPA 2 codes-after | after | 0.911 | [0.87, 0.944] | [0.383, 0.56] | 0.0 |
| V-JEPA 2 codes-before | before | 0.44 | [0.369, 0.519] | - | - |
| V-JEPA 2 rollout-before | before | 0.415 | [0.346, 0.481] | - | - |
| V-JEPA 2 gate-after | after | 0.885 | [0.842, 0.924] | [0.383, 0.544] | 0.0 |
| V-JEPA 2 hyp-after | after | 0.423 | [0.355, 0.488] | [-0.072, 0.075] | 0.521 |
| V-JEPA 2 both-before | before | 0.454 | [0.389, 0.516] | - | - |
| V-JEPA 2 both-before (ball from L) | before | 0.66 | [0.541, 0.777] | - | - |
| V-JEPA 2 both-before (ball from R) | before | 0.437 | [0.368, 0.511] | - | - |
| V-JEPA 2 commit_both-after | after | 0.843 | [0.8, 0.884] | [0.338, 0.451] | 0.0 |
| V-JEPA 2 commit_both-after (ball from L) | after | 0.767 | [0.636, 0.892] | [-0.015, 0.234] | 0.049 |
| V-JEPA 2 commit_both-after (ball from R) | after | 0.917 | [0.881, 0.95] | [0.405, 0.554] | 0.0 |
| V-JEPA 2 lossmix_e10-after | after | 0.949 | [0.921, 0.973] | [0.459, 0.599] | 0.0 |
| V-JEPA 2 lossmix_e20-after | after | 0.964 | [0.939, 0.985] | [0.472, 0.616] | 0.0 |

### Generalisation of commit (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames | mirrored: before | mirrored: after | mirrored: real frames |
|---|---|---|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.355 | 0.98 | 0.726 | 0.53 | 0.996 |

other side (from L), by height at the plank: the blockade range fitted on the seen clips (176.3-329.4 px) gets 0.855 of these outcomes right. How well the "gets through" score follows that range (AUROC): real frames 0.842948717948718, before 0.6503496503496504, after 0.31410256410256415, mirrored after 0.49533799533799533. Figure: `vjepa2/plan/generalise/commit/height.png`.

### Generalisation of plain (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames | mirrored: before | mirrored: after | mirrored: real frames |
|---|---|---|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.39 | 0.98 | 0.726 | 0.523 | 0.996 |

other side (from L), by height at the plank: the blockade range fitted on the seen clips (176.3-329.4 px) gets 0.855 of these outcomes right. How well the "gets through" score follows that range (AUROC): real frames 0.842948717948718, before 0.6503496503496504, after 0.32517482517482516, mirrored after 0.5308857808857809. Figure: `vjepa2/plan/generalise/plain/height.png`.
<!-- status:end -->

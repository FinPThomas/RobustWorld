# Research scope: next Kaggle session (2026-10-09)

This scope is shorter than [NEXT_STEPS_PLAN.md](NEXT_STEPS_PLAN.md) and replaces its run list. It
assumes no new video except the other-ball recording Fin already has. Everything fits in **one
Kaggle session of about 7.5 GPU hours**. All scoring uses the frozen `eval_decoder`, and the tracker
and outcomes are used only to score (CLAUDE.md rules 1–6).

## Questions

1. Can training pick up the **narrow gap** better without new video?
2. Does training on **both directions** fix the backwards left-side result?
3. Did the model learn that the table **slopes downhill** (right to left)?
4. Does the learnt blockade carry over to **another ball**?

## Runs

| run | what changes vs `commit-after` (10 epochs, 5 folds, `cv_folds` seed 0) | GPU h | answers |
|---|---|---|---|
| **A. both directions** | Train and test on all 337 clips (254 right + 83 left). Left clips are split across the folds, and each side is scored separately. | ~2.8 | Q2, plus Q1 (it adds 7 left-side narrow-gap through passes) |
| **B. narrow-gap sampling** | Right clips only. Each epoch, sample training clips in proportion to their loss from the previous epoch (with a floor, so nothing drops out). Rare passes such as the narrow gap get more steps, and no outcome label is used. | ~2.3 | Q1 |
| **C. slope scoring** | No training. For pretrained, A and B, save the decoded imagined ball position at every step (eval_decoder argmax) and score it only on open-table steps. | ~0.8 | Q3 |
| **D. other ball** | No training. Encode the other-ball clips, check the real-frame ceiling, then score pretrained, A and B. | ~0.8 | Q4 |
| encode + save fp16 weights | Saved weights let later scoring skip retraining. | ~0.8 | |

Order: encode, A, B, C, D. Each step pushes its results before the next starts.

## How each question is scored

- **Narrow gap (Q1).** Use the mean imagined P(ball beyond plank) for through passes in the narrow
  gap (y 315–367). It is 0.42 now for commit, against 0.92 on real frames. Also report AUROC on
  passes inside y 300–380 only, plus the height-profile figure. Success means B or A beats commit
  on these numbers without losing overall AUROC (0.934).
- **Both directions (Q2).** Report the left-side AUROC (0.355 now) and the left-side
  height figure. Success means the narrow-gap score is high and the middle is low on the left side
  as well. Report through vs blocked only, because the left side has just 3 hidden clips.
- **Downhill (Q3).** On target steps where the real ball is on open table (more than 2 cells from
  the plank), fit a straight line to the predicted and real ball speed along the table over the
  steps. That gives each clip an acceleration, and the sign is what matters:

  | clips | real ball | what learning the slope looks like |
  |---|---|---|
  | right-entry approach (downhill) | speeds up | predicted acceleration > 0 |
  | left-entry approach (uphill) | slows down | predicted acceleration < 0 |
  | bounce-backs rolling right (uphill) | slow, may stop | predicted to slow down |

  The pretrained model should predict roughly constant speed. The checks are:
  - **sign agreement** with the real acceleration;
  - **correlation** with the real acceleration, per clip;
  - a **speed-over-steps plot** for each group.

  The strongest evidence is uphill left clips that slow down when the model has learnt the slope.
  Run A has seen uphill balls and B has not, so the A vs B difference separates "learnt the slope"
  from "learnt that balls go left". Real-frame decoded positions are the ceiling, and they show
  first that the slope is measurable at all.
- **Other ball (Q4).** Compare AUROC before vs after, against the real-frame ceiling. If the
  real-frame number is low, the frozen decoder misses the new ball, and we report that rather than
  refitting.

## Out of scope for now

Longer training, codes+commit, adversarial losses, the pixel decoder and the pick-up evaluation
all stay in NEXT_STEPS_PLAN.md. They come after this session, depending on its results.

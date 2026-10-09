# Research scope: next Kaggle session (updated 2026-10-09, 14:00)

This replaces the run list in [NEXT_STEPS_PLAN.md](NEXT_STEPS_PLAN.md) and the old PR #3 queue.
There is no new video except the other-ball recording. The session runs in priority order, so the
most important results come out first. All scoring uses the frozen `eval_decoder`, and tracker
positions and outcomes are used only to score (CLAUDE.md rules 1–6).

**Priorities (Fin):**
1. Table slope (high).
2. Both directions (medium).
3. Other ball (low).

From the old plan, two items fit in at no extra session: the narrow gap and training length.

## 1. Slope from the uphill vs downhill difference (high)

**Idea.** On open table, a rolling ball's acceleration along its direction of motion is

- downhill: a↓ = −f(v) + s
- uphill: a↑ = −f(v) − s

Here f(v) is rolling friction, which depends on speed, and s is the slope term (5/7·g·sin θ for a
solid ball). At the **same speed**, friction cancels:

- **s = (a↓ − a↑) / 2**
- **f = −(a↓ + a↑) / 2**

Use speed bins, and check that s comes out the same in every bin.

**Both directions happen on both sides of the plank** (the plank is at x ≈ 103–270):

| table region | downhill (moving left) | uphill (moving right) |
|---|---|---|
| right of plank (approach of right clips) | right-clip approaches | right-clip bounce-backs, left-clip through exits |
| left of plank (approach of left clips) | right-clip through exits, left-clip bounce-backs | left-clip approaches |

So the gradient can be estimated **separately for each side** ("before the plank" on each side),
and as a coarse **gradient map** over cells of about 64 px.

**2-D version.** Fit a = −f(|v|)·v̂ + **G** over all open-table track samples, with f one value per
speed bin and **G** a constant 2-D vector, the downhill direction and size. This gives the slope's
direction, including any tilt across the table, not just left–right.

**Measured on:**

| tracks | what it shows | GPU |
|---|---|---|
| S0. real tracker tracks, all 337 clips, context and target halves | the true slope (ground truth) | none, CPU only |
| S1. eval_decoder positions on real frames | the ceiling: whether the slope can be read at the decoder's resolution | scoring pass |
| S2. imagined positions, pretrained | expected near 0 (no slope knowledge) | scoring pass |
| S3. imagined positions, post-trained B (right clips only) | slope learnt from downhill approaches and uphill bounce-backs | after B |
| S4. imagined positions, post-trained A (both directions) | slope learnt with uphill approaches added | after A |

- **Headline:** the model's implied slope s (and **G**), as a share of the real one, before vs
  after, with bootstrap CIs over clips.
- **Plots:**
  - acceleration against speed for downhill and uphill, for real and imagined tracks (the gap
    between the two curves is 2s);
  - the **G** arrow map per region.
- Imagined tracks have 12 steps at 8 Hz, so each clip gives a noisy acceleration. The estimate pools
  all clips by regression. S1 shows how much of that noise comes from the decoder.
- **To convert s to degrees:** the plank is 449.6 px long in the 512 px frame, so **its real
  length in cm** gives the scale.

## 2. Narrow gap and training length (from the old plan, fused into run B)

**B: commit, right clips only, with loss-proportional sampling, 20 epochs** (cosine schedule over
20 epochs), scored at epoch 10 and epoch 20.

- Each epoch, training clips are sampled in proportion to their loss from the previous epoch,
  with a floor. Rare passes such as the narrow gap get more steps, and no outcome label is used.
- **B@10 vs commit-after:** does sampling help the narrow gap? The comparison is mildly
  confounded, because the learning rate is not yet at zero at epoch 10.
- **B@20 vs B@10:** does training longer help?
- **Narrow-gap measures:** the mean imagined P(ball beyond plank) for through passes at
  y 315–367 (0.42 now, real 0.92), and AUROC on passes at y 300–380.
- B's weights also feed S3.

## 3. Both directions (medium)

**A: commit, 10 epochs, `cv_folds` over all 337 clips**, with each side scored separately.

- Measures: the left-side AUROC (0.355 now) and the left-side height plot. Report through vs
  blocked only, because the left side has 3 hidden clips.
- A's weights feed S4. **The A vs B difference in implied slope is the key slope comparison.**

## 4. Other ball (low)

If the video is uploaded by then: encode it, check the real-frame ceiling, and score pretrained,
B and A as a held-out test. There is no training on it.

## Kaggle order (one session, ~9 GPU h of the 11 h budget)

| # | step | GPU h | priority |
|---|---|---|---|
| 1 | encode (while the CPU runs **S0 real-track slope**) | 0.5 | high |
| 2 | scoring pass, pretrained: per-step decoded positions on real and imagined frames (**S1, S2**), plus per-token errors for surprise maps | 0.4 | high |
| 3 | **B** train (20 epochs, save fp16 weights at 10 and 20), score both, **S3** | 4.2 | high |
| 4 | **A** train (both directions, save fp16 weights), score, **S4** | 3.0 | medium |
| 5 | other-ball held-out scoring | 0.8 | low |

- Each step pushes its results before the next starts, so slope results (steps 1–3) are on
  GitHub even if the session stops early.
- **Dropped from the old queue:** rollout, codes_rollout, gate_hyp_commit, the 30-epoch and
  data-fraction runs, the shift test and codes+commit.
- **Left for later:** adversarial losses, the pixel decoder and the pick-up evaluation.

## Code needed on PR #3 (owned by the Kaggle plan thread)

- `posttrain.py`: `--sample-by-loss`, `--save-at 10,20`, fp16 weight export.
- Scoring: save per-step decoded positions (real and imagined) and per-token errors.
- New `slope.py` (CPU): the S0–S4 estimators and plots.
- `plan.py`: replace the queue with the order above.

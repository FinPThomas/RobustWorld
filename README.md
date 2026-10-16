# RobustWorld: Learning Hidden Scene Physics from Home Video for Robot Control

> **Branch `final/robustworld-summary`.** Final, self-contained summary of the project. It holds only the
> code and results needed to reproduce the best world model and the best gripper policy. Exploratory runs
> remain on their original branches (`claude/colab-two-day-plan-3gn90a`, `claude/project-thread-yzzs6s`).

## Abstract

Robots operating in homes must act on properties of a scene that cannot be seen: an obstruction behind a
cover, a surface that is not level, an object that will or will not reappear. This work examines whether a
pretrained video world model can acquire such a property **purely by self-supervised post-training on a
small amount of home-recorded video**, and whether a robot policy benefits from the acquired knowledge.

- **Scene.** A ball rolls under a plank; a blocker hidden beneath the plank stops some passes, while others
  emerge on the far side. The table is also slightly tilted.
- **World model.** The predictor of V-JEPA 2 is post-trained on 337 passes. Its predictions are read by a
  frozen linear ball probe that never sees outcomes or predictions, so all blockade knowledge must reside
  in the world model.
- **Result.** Through-vs-blocked AUROC rises from **0.44 to 0.97** for passes from the training direction
  (0.66 to 0.87 from the opposite direction). The table slope is likely, but not conclusively, recovered.
- **Policy.** A simulated two-finger gripper conditioned on the forecast improves from **−0.08 to +0.69**
  mean reward per real pass (perfect forecast: +1.27), chiefly by declining to chase balls that never emerge.

![Through pass: real future vs pretrained vs post-trained prediction](docs/figures/rollout_through.gif)

*Figure 1. A pass that emerges on the far side. Left: the real future, read by the ball probe. Centre: the
pretrained V-JEPA 2 loses the ball under the plank. Right: the final post-trained model predicts it
emerging. Cyan marks where the probe locates the ball (opacity = probe confidence).*

## Contents

1. [Quick start](#quick-start)
2. [Problem statement and data](#1-problem-statement-and-data)
3. [Model choice and measurement](#2-model-choice-and-measurement)
4. [Post-training V-JEPA 2](#3-post-training-v-jepa-2)
5. [Robustness](#4-robustness)
6. [The table slope](#5-the-table-slope)
7. [From forecast to gripper policy](#6-from-forecast-to-gripper-policy)
8. [Conclusion, impact and outlook](#7-conclusion-impact-and-outlook)
9. [How to run](#8-how-to-run)
10. [References](#references)

## Quick start

```bash
pip install -e .                                   # clip pipeline and evaluation tools
robustworld data/raw/videos/whole.mp4              # video -> tracked, labelled 3 s clips
python -m pytest tests/                            # includes the evaluation-isolation test
```

Post-training runs on a GPU (Kaggle, 2x T4) through `notebooks/kaggle_plan.ipynb`; the gripper runs on CPU.
Full instructions, including the weights of the best run and prediction with V-JEPA 2, are in
[How to run](#8-how-to-run).

**Model names used throughout.**

| name | definition |
|---|---|
| Pretrained V-JEPA 2 | ViT-L V-JEPA 2 as released, no post-training |
| Two-way post-trained model | predictor post-trained 20 epochs on passes from both entry directions (motion-weighted loss, loss-weighted sampling) |
| **Final model** | the two-way post-trained model, fine-tuned a further 5 epochs with added open-table rolling clips |

## 1. Problem statement and data

### 1.1 Why this scene

The scene is designed so that success is unambiguous and automatically verifiable.

- **Invisible, binary property.** No single frame reveals the blocker; it can only be inferred from
  experience of many passes. Outcomes are verifiable from the video itself.
- **Object permanence.** The ball must persist in the model's prediction while fully occluded.
- **Distribution over outcomes.** Passes near the blocker's edges can go either way, so a useful
  prediction is a probability, not a single future. This is the regime in which deterministic
  fine-tuning tends to average futures into a blur.
- **Graded difficulty.** A slight table tilt adds a weaker, continuous effect (the slope) alongside the
  binary one (the blockade).

### 1.2 Data collection

- One fixed overhead phone recording of a hand-rolled ball, rolled from both sides of the plank.
- The blocker was not moved during the recording; its fitted extent is stable between the first and last
  14 minutes (178–320 px vs 194–325 px along the plank).
- **Dataset:** 254 passes entering from the right (training direction), 83 from the left, plus open-table
  clips in which the ball rolls freely (used for the slope).

### 1.3 Autonomous processing

The pipeline (`src/robust_world/pipeline.py`; full reference in [docs/PIPELINE.md](docs/PIPELINE.md))
requires no manual labelling.

- **Tracking:** a colour tracker locates the ball in every frame; a hand detector flags frames containing a hand.
- **Pass detection:** each approach to the plank becomes one candidate clip.
- **Clip format:** 3 s, 48 frames at 16 fps, 512x512. The context (first half) ends immediately before the
  ball touches the plank; the target (second half) is to be predicted.
- **Outcome labels:** fixed rules on the ball track (*through*, *bounce* or *hidden* within 1.5 s). Clips in
  which a hand touches the ball are excluded. Labels are used for scoring only.

![The three outcomes](docs/figures/data_outcomes.png)

*Figure 2. The three outcome classes on the last context frame. White: observed context track. Yellow
dashed: the target track the model must predict. "Bounce" and "Hidden" are both blocked passes.*

## 2. Model choice and measurement

### 2.1 Why a latent world model rather than a generative video model

Generative video models (Wan 2.1, Cosmos, LTX-Video; runners in `models/`) were evaluated zero-shot and
did not meet the requirements of the problem:

- **Objective mismatch.** Pixel generation spends capacity on texture and lighting that carry no
  information about the blocker; outputs were frequently blurry or physically implausible, and must be
  re-tracked before they can be scored.
- **Adaptation cost.** Billion-parameter diffusion models are expensive to fine-tune on a few hundred clips
  and prone to overfitting at that scale.
- **Inference cost.** Iterative denoising is slow; a robot control loop needs predictions at control
  frequency on embedded hardware.

V-JEPA 2 [1] predicts in a learned latent space (the Joint-Embedding Predictive Architecture [3, 4]):

- prediction targets abstract away unpredictable pixel detail, which suits multi-modal outcomes;
- the predictor can be post-trained alone with the encoder frozen, which is cheap and preserves the
  features read by the probe;
- a single forward pass produces the forecast, and the same architecture has already been used for
  zero-shot robot planning (V-JEPA 2-AC [1]). Self-supervised video pretraining of this kind has also
  been shown to yield intuitive-physics understanding [5].

**Trade-off.** A latent model gives no human-viewable frames; its predictions are only as legible as the
readout used to decode them. This motivates the measurement design below.

### 2.2 A ball probe that cannot learn the blockade

- A linear probe (`models/vjepa2/eval_decoder.py`) maps each V-JEPA token to the probability that the ball
  is in that patch. On real frames it locates the ball almost perfectly (cell AUROC ≈ 1.00; median error
  14–20 px on a 512 px frame).
- **Decoupling.** Any readout trained on predictions or outcomes could itself learn the blockade, and the
  experiment would then measure the readout rather than the world model. The probe is therefore fitted
  once, on real context-half features only, with ball positions from the same frame, and is identical
  before and after post-training.
- These rules are stated in [CLAUDE.md](CLAUDE.md) and enforced bit-for-bit by
  `tests/test_eval_isolation.py`. Ground-truth outcomes are used only to score, never to fit.

## 3. Post-training V-JEPA 2

### 3.1 Method

- **Frozen encoder, trained predictor.** The predictor learns to forecast target-half latents from the
  context (L1 loss in layer-normalised latent space). 5-fold cross-validation with fixed splits
  (`cv_folds`, seed 0), so every model is scored on identical held-out clips.
- **Trade-off of freezing the encoder.** It keeps the probe valid and training cheap (about 6 GPU-hours on
  free Kaggle T4s), at the cost of limiting learning to how the predictor uses existing features.
  Full or LoRA encoder fine-tuning would change the features the probe reads and invalidate the
  before/after comparison.

### 3.2 The core difficulty: multi-modal futures

- A plain regression loss is minimised by the *mean* future. Where a pass could go either way, the
  prediction hedges: the ball fades into a faint, spatially smeared trace, or the model defaults to the
  most frequent outcome for every pass.
- The pretrained model exhibits the extreme of this behaviour: it almost never predicts the ball beyond
  the plank (Figure 1, centre).

### 3.3 Techniques that produce committed, realistic predictions

- **Motion-weighted objective.** The latent loss is weighted towards tokens that change between frames,
  with a contrastive term, so the predictor commits to a distinct ball rather than a background-dominated
  average.
- **Loss-weighted sampling.** Each epoch, clips are drawn in proportion to their previous-epoch loss
  (with a floor). Hard passes near the blocker's edges receive more updates, without using any outcome label.
- **Longer schedule and both directions.** 20 epochs with a cosine schedule, on passes from both entry
  directions; learning plateaus at about epoch 15.
- **Open-table fine-tune (final model).** Five further epochs with free-rolling clips keep the predicted
  ball visible for longer on open table (87% of clips at 1.5 s, against 57% before this stage).
- **Multi-step rollout scoring.** Predictions are scored frame by frame up to 1.5 s ahead, not only at
  the final step.

Alternative objectives and architectures (discrete codebook targets, a copy-last-frame gate,
multi-hypothesis heads) were also evaluated; the configuration above performed best.

### 3.4 Results

![AUROC summary](docs/figures/auroc.png)

*Figure 3. Through-vs-blocked AUROC on held-out folds, from how much ball the predicted future places beyond
the plank (95% bootstrap intervals; dotted line = chance). (a) All 337 passes. The reference fits its
blockade range on outcomes and is shown for comparison only. (b) By entry direction.*

| model | AUROC, all passes | from the right | from the left |
|---|---|---|---|
| Real future frames through the probe (ceiling) | 0.998 | – | – |
| Pretrained V-JEPA 2 | 0.45 | 0.44 | 0.66 |
| Ball tracker + blockade fitted on outcomes (reference) | 0.85 | – | – |
| Two-way post-trained model | 0.91 | 0.96 | 0.89 |
| **Final model** | **0.93** | **0.97** | **0.87** |

![P(blocked) along the plank](docs/figures/p_blocked.png)

*Figure 4. Probability that a pass is blocked, binned by where the ball reaches the plank (all 337 passes;
mean and 95% bootstrap interval per bin). The pretrained model predicts "blocked" everywhere; the final
model recovers both gaps, with the narrow lower gap learnt less sharply than the wide one.*

![Blocked pass rollout](docs/figures/rollout_blocked.png)

*Figure 5. A blocked pass, predicted 0.25–1.5 s ahead. The final model holds the ball at the plank's edge,
matching the real outcome, rather than predicting it through. Prediction panels are drawn on the empty scene.*

## 4. Robustness

### 4.1 The opposite entry direction

- **Observation.** A model post-trained only on passes from the right becomes *worse* on passes from the
  left (AUROC 0.64 to 0.36), and mirroring the left clips does not recover performance.
- **Interpretation.** The single-direction model learns an image-space outcome map ("at these heights the
  ball ends up on the left") rather than a direction-independent representation of a solid object.
- **Resolution.** Post-training on both directions jointly yields 0.89 (two-way model) and 0.87 (final
  model) on held-out left-entry passes, with no loss on the right (Figure 3b).

### 4.2 A held-out object: the orange ball

- **Test.** The final model, unchanged, is scored on passes with an orange ball that never appears in
  training. A model that has learnt the *scene* should be insensitive to the ball's appearance.

> **Results pending (test in progress).** AUROC from the right / left: _to be added_.

### 4.3 Interpretability: isolating ball appearance

- **Aim.** Identify a latent direction that encodes ball appearance (colour), separate from the
  components encoding position, velocity and motion, so that the scene knowledge (blockade, slope) can be
  shown to be independent of the object's appearance and the appearance component can be suppressed.

> **Results pending (analysis in progress).**

### 4.4 Summary

- The blockade generalises across entry directions once both are observed in training.
- Cross-object generalisation (Section 4.2) will establish whether the learnt property belongs to the scene
  rather than to the training ball.

## 5. The table slope

### 5.1 Measurement

- The table slopes slightly downhill to the left. The slope is measured label-free on each predicted track:
  **s = (downhill − uphill acceleration) / 2** at matched speed, which cancels rolling friction.
- **Probe resolution sets the attainable value.** The frozen probe recovers only about half of the real
  pull even from real frames (0.014 vs 0.025). A model that has learnt the slope should therefore approach
  the probe-on-real value, not the tracker value.

### 5.2 Result

![Slope map](docs/figures/slope_map.png)

*Figure 6. Left-right acceleration from the table, by region, after removing friction (red = downhill to the
left; arrows = pull; n = number of track stretches; neighbouring cells pooled, weighted by stretches).
The pretrained model is omitted: its predicted ball is never visible on open table, so no stretches exist.*

| track | s (ball diameters / step²) | 95% interval |
|---|---|---|
| Real ball (tracker) | 0.025 | 0.021 to 0.031 |
| Probe on real frames (attainable value) | 0.014 | 0.006 to 0.025 |
| Two-way post-trained model | 0.019 | −0.018 to 0.051 |
| **Final model** | **0.024** | **−0.007 to 0.040** |

### 5.3 Interpretation

- The final model pulls the predicted ball downhill at or above the probe's attainable level, and the
  open-table fine-tune nearly doubles the number of measurable stretches.
- The interval still includes zero, so the slope is **likely learnt but not proven**.
- A conclusive result requires more open-table rolling or a steeper tilt: the slope is data-limited, not
  model-limited.

## 6. From forecast to gripper policy

### 6.1 Interface between world model and policy

The policy receives a compact forecast extracted from the world model's predicted future through the frozen probe:

- **whether** the ball emerges (P(comes out));
- **where** along the plank edge, **when**, and **how fast**;
- **how uncertain** the forecast is.

This interface is independent of the robot's embodiment and of the world model's internals.

### 6.2 Problem design

The task is constructed so that a better forecast must translate into better behaviour.

- **The gripper must commit before the ball is visible:** fingers close at most 0.15 m/s and open at most
  9 cm for a 4 cm ball, so it cannot react to every ball after it appears.
- **Grasp reward (simulation):**
  - lifted and held: +1, plus a bonus of up to +0.5 for an early pick-up, minus up to 0.3 for a wide opening;
  - miss: −1.
- **Decision reward (real passes):**
  - pick-up: +3, plus up to +0.5 for an early pick-up;
  - attempted and missed: −0.5;
  - ball emerges but the gripper stayed: −1;
  - wasted trip for a blocked ball: −1;
  - correctly holding back on a blocked ball: +0.5.
- **Decision rule:** the gripper sets off when the expected value of going (forecast probability times the
  simulated catch rate at that uncertainty) exceeds that of staying.

### 6.3 Simulation environment and training

- **PyBullet** two-finger gripper (`models/gripper/`), trained with **PPO** (Stable-Baselines3) on CPU,
  5 folds of 2.5 M steps.
- **Choice of simulator.** Only the grasp of a single rolling ball is modelled, so a lightweight
  rigid-body simulator gives fast, reproducible training on a laptop CPU. Heavier robotics stacks
  (Isaac Sim, MuJoCo with a full arm) add fidelity the task does not require.
- **Trained on synthetic forecasts, frozen, then evaluated on real ones.** During training, forecasts are
  the true exit plus noise of a stated magnitude. The policy is then frozen and scored on real passes with
  each world model's forecast. It never sees a world model's real forecast against a real outcome.
- **Trade-off.** A policy trained on synthetic noise must be calibrated to the uncertainty the world model
  actually reports. In exchange, one policy serves every world model, and the comparison between world
  models is not confounded by policy training.

### 6.4 Transferable components

The design separates what is specific to this robot from what transfers to other systems:

- **Transferable:** the post-trained world model, the probe, and the forecast interface (*arrive, where,
  when, how fast, how sure*). None depends on the gripper.
- **Embodiment-specific:** only the low-level policy that maps the forecast to motor commands, trained
  cheaply in simulation.
- **Integration with generalist policies.** The same forecast can condition an open-source pretrained
  manipulation policy (Octo [6], OpenVLA [7], π0 [8]) as an extra observation, so that scene-specific
  knowledge learnt from home video reaches any embodiment those policies support without retraining
  the world model.

### 6.5 Results

Mean reward per pass over all 335 real passes (141 emerge, 194 blocked):

| forecast given to the frozen policy | sets off (emerges / blocked) | picks up | mean reward |
|---|---|---|---|
| none, or pretrained V-JEPA 2 | 100% / 100% | 43% | −0.08 |
| **two-way post-trained model** | **72% / 4%** | 60% of attempts | **+0.69** |
| perfect forecast (tracker) | 100% / 0% | 73% | +1.27 |

| scenario | no post-training | post-trained | perfect forecast |
|---|---|---|---|
| **Blocked pass** | ![](docs/figures/gripper/blocked_no_posttraining.gif) | ![](docs/figures/gripper/blocked_posttrained.gif) | ![](docs/figures/gripper/blocked_perfect.gif) |
| **Emerges, uncertain forecast** | ![](docs/figures/gripper/uncertain_no_posttraining.gif) | ![](docs/figures/gripper/uncertain_posttrained.gif) | ![](docs/figures/gripper/uncertain_perfect.gif) |
| **Emerges, confident forecast** | ![](docs/figures/gripper/confident_no_posttraining.gif) | ![](docs/figures/gripper/confident_posttrained.gif) | ![](docs/figures/gripper/confident_perfect.gif) |

*Figure 7. The same frozen policy driven by three forecast sources on three real passes. The header of each
panel shows P(comes out), the decision threshold and the decision. Without post-training the gripper sets
off for every ball, including blocked ones; with the post-trained forecast it holds back on the blocked pass
and lines up on the emerging ones.*

Interactive version: [Gripper Uncertainty Response](https://claude.ai/artifact/1pygtzGBW2LqB9bt6HPB9d).

## 7. Conclusion, impact and outlook

### 7.1 Conclusion

- Self-supervised post-training of V-JEPA 2 on a few hundred home-recorded clips teaches it an invisible
  blockade (AUROC 0.44 to 0.97), measured through a readout that provably cannot have learnt it.
- The knowledge generalises across entry directions once both are observed, and the table slope is likely
  recovered.
- A frozen gripper policy converts the forecast into behaviour: mean reward rises from −0.08 to +0.69 per
  pass, chiefly by not pursuing balls that will not emerge; a perfect forecast reaches +1.27.

### 7.2 Impact and applications

- **Home and humanoid robotics.** Homes contain hidden, persistent state: a drawer that sticks, an object
  behind another, an uneven floor, a door that swings back. A robot that post-trains its world model on its
  own observations of one home can learn such properties without labels and plan around them.
- **Trend in the field.** World models are becoming the planning layer of robot foundation models: latent
  predictive models for planning [1, 2], generative world models for humanoids [9, 10], and surveys
  framing world models as the route to physical reasoning in manipulation [11, 12]. This work contributes a
  verifiable test of the property those systems rely on: that scene-specific physics can be acquired from
  passive video and measured independently of the readout.
- **Efficiency.** Latent prediction needs one forward pass per forecast, which suits on-board inference on
  power-limited humanoid hardware better than iterative video generation.

### 7.3 Handheld and egocentric video

- Large egocentric datasets (Ego4D [13]) and handheld robot-data collection tools (UMI [14]) show that
  human-held cameras are the scalable source of home video.
- Handheld video introduces camera motion. Latent models are better placed than pixel generators to
  absorb it, but the probe and the slope measurement would need to operate in a camera-stabilised or
  world-fixed frame.

### 7.4 Implications and limitations

- **Implication.** Scene-specific physical knowledge can be acquired by post-training on passive video
  and verified separately from the readout; a robot can benefit from it through a small, transferable
  forecast interface.
- **Limitations.** One scene and one blocker position; the predicted ball remains fainter than the real one
  in the narrow gap; the slope interval includes zero; the gripper is evaluated in simulation on real
  trajectories rather than on hardware.
- **Next steps.** A moved-blocker recording to test adaptation; cross-object testing (Section 4.2);
  conditioning an open-source generalist policy on the forecast; hardware evaluation.

## 8. How to run

### 8.1 Clips

- Place the videos in `data/raw/videos/` and run `robustworld data/raw/videos/` (see
  [docs/PIPELINE.md](docs/PIPELINE.md)).
- `scripts/cut_open_clips.py` cuts the open-table clips; `scripts/pack_clips.py` packages the clips for Kaggle.

### 8.2 Post-training the final model on Kaggle

1. Open `notebooks/kaggle_plan.ipynb` (GPU 2x T4; attach the `GITHUB_TOKEN` secret and the clips dataset).
2. Save & Run All. The notebook runs `models/vjepa2/plan.py`; to repeat only the best runs:

```bash
python models/vjepa2/plan.py run --only lossmix_both-after --redo     # two-way post-trained model (~5.6 h)
python models/vjepa2/plan.py run --only lossmix_topup-after --redo    # final model: + 5 epochs open table
```

### 8.3 Weights

- Fold weights (`fold0.pt`–`fold4.pt`) exceed GitHub's file limits and are kept in the Kaggle notebook
  outputs: `weights/lossmix_both-after/` (two-way model) and `weights/lossmix_topup-after/` (final model).
- Attach that output, or a Kaggle dataset named after the run (e.g. `lossmix-both-after`), as a notebook
  input; `plan.py` locates it automatically.

### 8.4 Predicting with V-JEPA 2

```bash
python models/vjepa2/ball_probe_cv.py --predictor-run <path>/weights/lossmix_topup-after \
    --manifest data/processed/clips/manifest_right.jsonl --out outputs/vjepa2/final
```

- Writes `per_clip.json` (predicted ball position and visibility per step, through-vs-blocked score) and
  `maps.npz` (probe maps per step).
- Saved scores for the pretrained, two-way and final models are in
  `results/2026-10-04_2110_twoday/vjepa2/plan/scores/` (`both-before`, `lossmix_both-after`,
  `lossmix_topup-after`).
- `python docs/figures/make_figures.py --clips data/processed/clips/whole` redraws every figure in this README.

### 8.5 Gripper (CPU)

The approved policies are committed in `models/gripper/runs/pb_narrow_fs/`.

```bash
pip install pybullet gymnasium stable-baselines3
cd models/gripper
export GRIPPER_ENV=pybullet GRIPPER_PARAMS='{"GAP_MAX": 0.09, "FINGER_SPEED": 0.15}'
PLAN=../../results/2026-10-04_2110_twoday/vjepa2/plan
python blockade_eval.py --plan $PLAN --tag pb_narrow_fs --v2                       # Section 6.5 table
python videos.py --plan $PLAN --policy runs/pb_narrow_fs --tag pb_narrow_fs --gui  # watch in PyBullet
```

Training (`train.py`) is described in [models/gripper/README.md](models/gripper/README.md).

## References

1. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
2. LeCun. *A Path Towards Autonomous Machine Intelligence.* 2022. [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
3. Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
4. Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA).* 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
5. Garrido et al. *Intuitive Physics Understanding Emerges from Self-Supervised Pretraining on Natural Videos.* 2025. [arXiv:2502.11831](https://arxiv.org/abs/2502.11831)
6. Octo Model Team. *Octo: An Open-Source Generalist Robot Policy.* 2024. [arXiv:2405.12213](https://arxiv.org/abs/2405.12213)
7. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
8. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
9. *Humanoid World Models: Open World Foundation Models for Humanoid Robotics.* 2025. [arXiv:2506.01182](https://arxiv.org/abs/2506.01182)
10. *Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report.* 2025. [arXiv:2510.07092](https://arxiv.org/abs/2510.07092)
11. *World Models for Robotic Manipulation: A Survey.* 2026. [arXiv:2606.00113](https://arxiv.org/abs/2606.00113)
12. *A Definition and Roadmap for World Models.* 2026. [arXiv:2607.06401](https://arxiv.org/abs/2607.06401)
13. Grauman et al. *Ego4D: Around the World in 3,000 Hours of Egocentric Video.* 2022. [arXiv:2110.07058](https://arxiv.org/abs/2110.07058)
14. Chi et al. *Universal Manipulation Interface.* 2024. [arXiv:2402.10329](https://arxiv.org/abs/2402.10329)

# Two-day plan: status

_Last update: 2026-10-07 12:29 UTC from Kaggle (Tesla T4), code `aaca21d`. Results: [results/2026-10-04_2110_twoday](../results/2026-10-04_2110_twoday/README.md)._

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
| 2 | codes-before: codes: pretrained predictor, scored the codes way | done | 11.5 | 2026-10-07 04:01 |
| 2 | codes-after: codes: post-trained 10 epochs | done | 120.3 | 2026-10-07 11:55 |
| 2 | rollout-before: rollout: pretrained predictor, scored the rollout way | done | 34.0 | 2026-10-07 12:29 |
| 2 | rollout-after: rollout: post-trained 10 epochs | to do |  |  |
| 2 | codes_rollout-before: codes_rollout: pretrained predictor, scored the codes_rollout way | to do |  |  |
| 2 | codes_rollout-after: codes_rollout: post-trained 10 epochs | to do |  |  |
| 3 | gate_hyp_commit-after: gate_hyp_commit: post-trained 10 epochs | to do |  |  |
| 4 | long-1: best variant, 30 epochs | failed (no stage 2/3 run has been scored yet: run those first) | 0.0 | 2026-10-04 21:24 |
| 4 | long-2: second-best variant, 30 epochs | to do |  |  |
| 4 | frac-50: best variant on 50% of the training clips | to do |  |  |
| 4 | frac-25: best variant on 25% of the training clips | to do |  |  |
| 5 | generalise: ball from the other side / new ball, before vs after | running |  |  |
| 6 | interpret: change maps and layer patching for the best run | failed (no stage 2/3 run has been scored yet: run those first) | 0.0 | 2026-10-04 21:32 |

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

### Generalisation of commit (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames |
|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.35 | 0.98 |

### Generalisation of plain (held out, never trained on)

| group | clips | AUROC before | AUROC after | AUROC real frames |
|---|---|---|---|---|
| other side (from L) | 83 | 0.643 | 0.388 | 0.98 |

# Two-day plan: status

_Last update: 2026-10-04 22:11 UTC from Colab (Tesla T4), code `9083c06`. Results: [results/2026-10-04_2110_twoday](../results/2026-10-04_2110_twoday/README.md)._

| stage | step | state | minutes | finished (UTC) |
|---|---|---|---|---|
| 1 | split: set aside held-out clips (other side, new ball) | done | 0.0 | 2026-10-04 21:10 |
| 1 | encode: encode every clip once (frozen encoder) | done | 27.1 | 2026-10-04 22:11 |
| 1 | pretrained: pretrained V-JEPA 2, frozen decoder | running |  |  |
| 1 | tapnext: TAPNext + rules, with and without the coded blockade | to do |  |  |
| 2 | plain-before: plain: pretrained predictor, scored the plain way | running |  |  |
| 2 | plain-after: plain: post-trained 10 epochs | to do |  |  |
| 2 | commit-after: commit: post-trained 10 epochs | to do |  |  |
| 2 | codes-before: codes: pretrained predictor, scored the codes way | to do |  |  |
| 2 | codes-after: codes: post-trained 10 epochs | to do |  |  |
| 2 | rollout-before: rollout: pretrained predictor, scored the rollout way | to do |  |  |
| 2 | rollout-after: rollout: post-trained 10 epochs | to do |  |  |
| 2 | codes_rollout-before: codes_rollout: pretrained predictor, scored the codes_rollout way | to do |  |  |
| 2 | codes_rollout-after: codes_rollout: post-trained 10 epochs | to do |  |  |
| 3 | gate-after: gate: post-trained 10 epochs | running |  |  |
| 3 | hyp-after: hyp: post-trained 10 epochs | to do |  |  |
| 3 | gate_hyp_commit-after: gate_hyp_commit: post-trained 10 epochs | to do |  |  |
| 4 | long-1: best variant, 30 epochs | failed (no stage 2/3 run has been scored yet: run those first) | 0.0 | 2026-10-04 21:24 |
| 4 | long-2: second-best variant, 30 epochs | to do |  |  |
| 4 | frac-50: best variant on 50% of the training clips | to do |  |  |
| 4 | frac-25: best variant on 25% of the training clips | to do |  |  |
| 5 | generalise: ball from the other side / new ball, before vs after | running |  |  |
| 6 | interpret: change maps and layer patching for the best run | failed (no stage 2/3 run has been scored yet: run those first) | 0.0 | 2026-10-04 21:32 |

| method | phase | outcome_auroc | auroc_ci | delta_vs_before_ci | p_no_gain |
|---|---|---|---|---|---|

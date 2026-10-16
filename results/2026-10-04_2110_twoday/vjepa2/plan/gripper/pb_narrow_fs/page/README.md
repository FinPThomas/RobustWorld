# Gripper Uncertainty Response: pb_narrow_fs (kept for the write-up)

Fin approved this page on 16 Oct 2026 as the gripper result for the write-up.
Live page: https://claude.ai/artifact/1pygtzGBW2LqB9bt6HPB9d (version 4, published 15 Oct 2026 ~23:40 UTC).

- Set-up: PyBullet, fingers open at most 9 cm (4 cm ball) and close at most 0.15 m/s; trained on from pb_narrow to 2.5M steps per fold (models/gripper/overnight.py, commit 6c65c83). Policies: models/gripper/runs/pb_narrow_fs/fold0-4.
- Blockade test v2 on 335 real passes (141 come out, 194 blocked), mean reward per pass:
  no post-training -0.08 (sets off for every ball, picks up 43% when it sets off),
  post-trained C +0.69 (sets off for 72% that come out and 4% of blocked, picks up 60%),
  perfect forecast +1.27 (picks up 73%). See ../blockade_v2.md.
- gripper_uncertainty.html is the published page (open with vid/ next to it). src/ has its pieces and the
  generators (gen3.py makes data.json, vids.py renders vid/; run from models/gripper with
  GRIPPER_ENV=pybullet GRIPPER_PARAMS='{"GAP_MAX": 0.09, "FINGER_SPEED": 0.15}').

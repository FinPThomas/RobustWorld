# Exit-and-catch gripper (PyBullet)

Does a better world model make a robot better at picking up the ball? A ball comes out from under the
plank (real exit positions, times and speeds from the tracker) and a two-finger gripper must grasp and
lift it within a catch zone (3 ball diameters past the plank). The policy is told the world model's
forecast of where, when and how fast the ball comes out, and how uncertain it is. Reward: a pick-up
earns more the earlier it happens and less the wider the fingers were open; a miss costs 1.

The policy is trained in simulation only (PPO, CPU), with forecasts that are the true exit plus noise of
the stated size, then frozen. It is scored on real passes with each world model's own forecast, before
and after post-training. It never learns from a world model's real forecasts against real outcomes
(CLAUDE.md rule 1).

    pip install pybullet gymnasium stable-baselines3          # CPU torch is enough
    python models/gripper/exits.py --per-clip <plan>/scores/both-before/per_clip.json   # once (exits.json is committed)
    python models/gripper/train.py                             # 5 folds x 1M steps, resumes, skips finished folds
    python models/gripper/evaluate.py --plan <plan>            # <plan>/gripper/gripper.md, gripper.png
    python models/gripper/videos.py --plan <plan>              # uncertainty.mp4, before_after.mp4
    python models/gripper/videos.py --plan <plan> --gui        # watch it live in PyBullet's window

`<plan>` is e.g. `results/2026-10-04_2110_twoday/vjepa2/plan`.

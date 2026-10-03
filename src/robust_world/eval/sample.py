"""Draw a reproducible random set of clips into data/eval/<name>/ (small enough to commit).

Clips are drawn at random within each outcome and balanced across outcomes (round-robin),
so a small sample still covers the blocked "hidden" case and not just the common "through".
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from ..paths import CLIPS_ROOT, REPO_ROOT, rel

EVAL_ROOT = REPO_ROOT / "data" / "eval"

# Shared text prompt. Describes the scene only, never the outcome, so the model has to infer
# from the context frames whether the ball passes under the plank or is blocked.
PROMPT = (
    "A fixed overhead camera looks straight down at a wooden table. A small blue and green "
    "striped ball rolls across the table towards a raised wooden plank. Realistic physics, "
    "natural lighting, the camera does not move."
)


def make_sample(name: str, n: int, seed: int, manifest: Path = CLIPS_ROOT / "manifest.jsonl") -> Path:
    records = [json.loads(line) for line in manifest.open()]
    rng = random.Random(seed)
    by_outcome = {}
    for r in records:
        if r.get("include"):
            by_outcome.setdefault(r["outcome"], []).append(r)
    for group in by_outcome.values():
        rng.shuffle(group)
    order = sorted(by_outcome, key=lambda o: -len(by_outcome[o]))
    picked = []
    while len(picked) < n and any(by_outcome.values()):
        for o in order:
            if by_outcome[o] and len(picked) < n:
                picked.append(by_outcome[o].pop())
    picked.sort(key=lambda r: r["clip_id"])

    out = EVAL_ROOT / name
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for r in picked:
        shutil.copy2(REPO_ROOT / r["path"], out / "clips" / f"{r['clip_id']}.mp4")
    first = picked[0]
    (out / "sample.json").write_text(json.dumps({
        "name": name,
        "seed": seed,
        "prompt": PROMPT,
        "fps": first["fps"],
        "context_frames": first["context_frames"],
        "target_frames": first["target_frames"],
        "clips": picked,
    }, indent=1))
    outcomes = ", ".join(f"{r['clip_id']} ({r['outcome']})" for r in picked)
    print(f"{n} clips (seed {seed}) -> {rel(out)}: {outcomes}")
    return out

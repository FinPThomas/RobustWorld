"""Is the pretrained predictor more surprised by blocked passes than by through passes on the same path?

Label-free: surprise is the per-token L1 between the pretrained predictor's imagined target and the
real target (both layer-normalised; probing.py's cache). Each hidden or bounce clip is paired with the
through clip whose straight-line path is most similar (where it crosses the plank and how fast it
moves, from the tracker's context positions). Nothing is fitted; outcomes only group the clips.
Writes outputs/vjepa2/probing/surprise_matched.{md,json}.

    python models/vjepa2/surprise_matched.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import probing  # noqa: E402
from robust_world.eval.ball import included_clips  # noqa: E402


def surprise(clip_id: str) -> np.ndarray:
    """[target steps, g, g] per-token L1 between imagined and real target."""
    a, b = (torch.from_numpy(probing.load(clip_id, k)).float() for k in ("imag", "tgt_last"))
    a, b = (torch.nn.functional.layer_norm(t, t.shape[-1:]) for t in (a, b))
    return (a - b).abs().mean(-1).numpy()


def main() -> int:
    from scipy.stats import wilcoxon
    clips = included_clips()
    rows, geo = probing.truth(clips)
    timing = json.loads((probing.OUT / "blockade_timing.json").read_text())["clips"]
    info = {t["clip_id"]: t for t in timing}
    plank = probing.cell_centres(np.array(geo["scene"]["occluder_polygon"], np.float32))
    s = {c["clip_id"]: surprise(c["clip_id"]) for c in clips}
    through = [c["clip_id"] for c in clips if c["outcome"] == "through" and info[c["clip_id"]]["straight_y"] is not None]
    feat = lambda cid: np.array([info[cid]["straight_y"] / 32, info[cid]["speed_px_per_frame"] / 4])  # ~1 per cell / per 4 px/frame  # noqa: E731
    out = {}
    lines = ["# Predictor surprise, blocked vs through on matched paths", "",
             "Pretrained predictor; each blocked clip paired with the through clip nearest in plank crossing (y) and speed. "
             "Positive = more surprise on the blocked clip.", "",
             "| outcome | pairs | match distance (cells) | whole frame: mean diff (wins, p) | plank cells | per target step (whole frame) |",
             "|---|---|---|---|---|---|"]
    for o in ("hidden", "bounce"):
        pairs = []
        for c in clips:
            cid = c["clip_id"]
            if c["outcome"] != o or info[cid]["straight_y"] is None:
                continue
            d = [float(np.linalg.norm(feat(cid) - feat(t))) for t in through]
            j = int(np.argmin(d))
            pairs.append((cid, through[j], d[j]))
        whole = np.array([s[a].mean() - s[b].mean() for a, b, _ in pairs])
        pl = np.array([s[a][:, plank].mean() - s[b][:, plank].mean() for a, b, _ in pairs])
        steps = np.mean([s[a].mean((1, 2)) - s[b].mean((1, 2)) for a, b, _ in pairs], 0)
        res = {"pairs": len(pairs), "match_distance_median": round(float(np.median([p[2] for p in pairs])), 2),
               "whole_mean_diff": round(float(whole.mean()), 4), "whole_wins": int((whole > 0).sum()),
               "whole_p": round(float(wilcoxon(whole).pvalue), 4), "plank_mean_diff": round(float(pl.mean()), 4),
               "plank_wins": int((pl > 0).sum()), "plank_p": round(float(wilcoxon(pl).pvalue), 4),
               "per_step_diff": [round(float(v), 4) for v in steps], "pairs_list": pairs}
        out[o] = res
        lines.append(f"| {o} | {res['pairs']} | {res['match_distance_median']} | {res['whole_mean_diff']:+.4f} "
                     f"({res['whole_wins']}/{res['pairs']}, p={res['whole_p']}) | {res['plank_mean_diff']:+.4f} "
                     f"({res['plank_wins']}/{res['pairs']}, p={res['plank_p']}) | "
                     + " ".join(f"{v:+.3f}" for v in res["per_step_diff"]) + " |")
    (probing.OUT / "surprise_matched.json").write_text(json.dumps(out, indent=1))
    (probing.OUT / "surprise_matched.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

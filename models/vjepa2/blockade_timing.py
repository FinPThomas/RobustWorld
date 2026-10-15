"""Did the blocker move during the right segment, and how good is the straight-line crossing estimate?

Reads the tracker and the probing baseline's straight-line crossings (outputs/vjepa2/probing/
baseline.json); fits nothing except the reference blockade interval (fit_blockade, on outcomes, as in
the TAPNext rule baseline). Writes outputs/vjepa2/probing/blockade_timing.{md,json}.

    python models/vjepa2/blockade_timing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import probing  # noqa: E402
from robust_world.eval.ball import included_clips  # noqa: E402


def main() -> int:
    clips = included_clips()
    rows, geo = probing.truth(clips)
    tr = probing.tapnext_rules()
    base = json.loads((probing.OUT / "baseline.json").read_text())
    k = rows[0]["ctx_steps"]
    poly = np.array(geo["scene"]["occluder_polygon"], np.float32)
    dist = lambda p: -cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True)  # noqa: E731

    recs = []
    for c, r in zip(clips, rows):
        seen = [p for p in r["centres"] if p]
        if c["outcome"] == "through":                       # midway between last near-side and first far-side sighting
            near = [p for p in r["centres"] if p and p[0] > 230]
            far = [p for p in r["centres"][k:] if p and p[0] < 180]
            actual = (near[-1][1] + far[0][1]) / 2 if near and far else None
        else:                                               # closest sighting to the plank
            actual = min(seen, key=dist)[1] if seen else None
        path = tr.continue_path({"centres": r["ctx_centres_px"]}, geo["scene"], 24)
        recs.append({"clip_id": c["clip_id"], "outcome": c["outcome"], "minute": round(c["source_frames"][0] / c["source_fps"] / 60, 2),
                     "straight_y": base["crossing_y"][c["clip_id"]], "actual_y": actual,
                     "speed_px_per_frame": round(float(np.hypot(*path["velocity"])), 1)})

    def fit(key, sel):
        sub = [{"crossing_y": r[key], "outcome": r["outcome"]} for r in recs if sel(r)]
        b = tr.fit_blockade(sub)
        acc = np.mean([(s["crossing_y"] is not None and b[0] <= s["crossing_y"] <= b[1]) == (s["outcome"] != "through") for s in sub])
        return {"range": [round(v, 1) for v in b], "in_sample_accuracy": round(float(acc), 3), "n": len(sub)}

    halves = {"first 14 min": lambda r: r["minute"] < 14, "last 14 min": lambda r: r["minute"] >= 14, "all": lambda r: True}
    fits = {key: {h: fit(key, s) for h, s in halves.items()} for key in ("straight_y", "actual_y")}
    lo, hi = fits["straight_y"]["all"]["range"]
    err = [r["straight_y"] - r["actual_y"] for r in recs if r["straight_y"] is not None and r["actual_y"] is not None]
    rate = {}
    for a in range(0, 28, 4):
        g = [r for r in recs if a <= r["minute"] < a + 4]
        mis = [(r["straight_y"] is not None and lo <= r["straight_y"] <= hi) != (r["outcome"] != "through") for r in g]
        rate[f"{a}-{a + 4} min"] = {"n": len(g), "misfit_rate": round(float(np.mean(mis)), 2)}
    out = {"fits": fits, "straight_minus_actual_px": {"median": round(float(np.median(err)), 1), "mae": round(float(np.mean(np.abs(err))), 1),
                                                      "p90_abs": round(float(np.percentile(np.abs(err), 90)), 1)},
           "misfit_rate_by_time": rate, "clips": recs}
    probing.OUT.mkdir(parents=True, exist_ok=True)
    (probing.OUT / "blockade_timing.json").write_text(json.dumps(out, indent=1))

    f = fits
    lines = ["# Did the blocker move? (right segment)", "",
             "| crossing used | first 14 min | last 14 min | all |", "|---|---|---|---|"]
    for key, label in (("straight_y", "straight line from context"), ("actual_y", "tracker (where it really met the plank)")):
        lines.append(f"| {label} | " + " | ".join(f"{f[key][h]['range'][0]:.0f}-{f[key][h]['range'][1]:.0f} px, acc {f[key][h]['in_sample_accuracy']}"
                                                   for h in halves) + " |")
    e = out["straight_minus_actual_px"]
    lines += ["", f"Straight-line crossing minus actual: median {e['median']} px, mean abs {e['mae']} px, 90th percentile {e['p90_abs']} px.", "",
              "Misfit rate (crossing in range but through, or out of range but blocked) by time: "
              + ", ".join(f"{k} {v['misfit_rate']} (n={v['n']})" for k, v in rate.items()) + ".", ""]
    (probing.OUT / "blockade_timing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

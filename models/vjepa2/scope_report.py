"""Scores for the 2026-10-09 research scope (results/overview/RESEARCH_SCOPE.md), from files already written.

Nothing is fitted or trained here. Every number reads the per-clip output of ball_probe_cv.py (frozen
evaluation decoder) and blocker_figs.py (where each pass reaches the plank); tracker positions and outcomes
are only used to group and score clips (CLAUDE.md rules 5-6).

    Q1 narrow gap   mean imagined P(ball beyond the plank) (one-ball, peak over the target) for through
                    passes by the height they reach the plank: wide gap y 104-186, narrow gap 315-367, and all
                    passes in the blocked middle 186-315; plus through-vs-blocked AUROC on passes at y 300-380.
    Q2 both sides   the same, for each direction separately (runs scored on both directions).
The table slope (downhill) is measured by slope.py.

Outputs: <plan>/scope/scope.json, scope.md, height_profile.png

    python models/vjepa2/scope_report.py --plan outputs/vjepa2/plan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

BANDS = {"wide gap (104-186)": (104, 186, "through"), "narrow gap (315-367)": (315, 367, "through"),
         "blocked middle (186-315)": (186, 315, None)}
NARROW_AUROC = (300, 380)
# Runs compared, in this order, when their scores exist.
RUNS = ["pretrained", "plain-after", "commit-after", "codes-after", "lossmix_e10-after", "lossmix_e20-after",
        "both-before", "commit_both-after", "lossmix_both-after", "lossmix_open-after"]


def auroc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.min(initial=1) == y.max(initial=0) or len(y) < 2:
        return None
    p, n = s[y == 1], s[y == 0]
    d = p[:, None] - n[None]
    return round(float((d > 0).mean() + 0.5 * (d == 0).mean()), 3)


def load(plan: Path, name: str):
    pc, cr = plan / "scores" / name / "per_clip.json", plan / "blocker" / name / "crossing.json"
    if not pc.exists():
        return None
    rows = json.loads(pc.read_text())
    ys = {}
    if cr.exists():
        ys = {p["clip_id"]: p["entry_xy"][1] for p in json.loads(cr.read_text())["passes"] if p.get("entry_xy")}
    for r in rows:
        r["entry_y"] = ys.get(r["clip_id"])
        r.setdefault("side_in", "R")
    return rows


def height_scores(rows: list[dict], key: str) -> dict:
    out = {}
    for band, (lo, hi, outcome) in BANDS.items():
        v = [max(r[key]) for r in rows if r["entry_y"] is not None and lo <= r["entry_y"] < hi
             and (outcome is None or r["outcome"] == outcome)]
        out[band] = {"mean": round(float(np.mean(v)), 3) if v else None, "n": len(v)}
    near = [r for r in rows if r["entry_y"] is not None and NARROW_AUROC[0] <= r["entry_y"] < NARROW_AUROC[1]]
    out[f"AUROC at y {NARROW_AUROC[0]}-{NARROW_AUROC[1]}"] = {
        "auroc": auroc([r["outcome"] == "through" for r in near], [max(r[key]) for r in near]), "n": len(near)}
    out["AUROC all"] = {"auroc": auroc([r["outcome"] == "through" for r in rows], [max(r[key]) for r in rows]),
                        "n": len(rows)}
    return out


def height_figure(rows_by_run: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sides = sorted({r["side_in"] for rows in rows_by_run.values() for r in rows})
    fig, axes = plt.subplots(1, len(sides), figsize=(5 * len(sides), 3.4), squeeze=False)
    edges = np.arange(60, 460, 30)
    for ax, side in zip(axes[0], sides):
        for run, rows in rows_by_run.items():
            sub = [r for r in rows if r["side_in"] == side and r["entry_y"] is not None]
            if not sub:
                continue
            for key, label, style in (("imagined_one_ball_far", run, "-"),) + (
                    (("real_one_ball_far", "real frames", "k--"),) if run == "pretrained" or run == "both-before" else ()):
                xs, ys = [], []
                for lo, hi in zip(edges, edges[1:]):
                    v = [max(r[key]) for r in sub if lo <= r["entry_y"] < hi]
                    if len(v) >= 2:
                        xs.append((lo + hi) / 2), ys.append(np.mean(v))
                if xs and not (label == "real frames" and any(line.get_label() == "real frames" for line in ax.lines)):
                    ax.plot(xs, ys, style, label=label, lw=1.2)
        for band, (lo, hi, _) in BANDS.items():
            if "gap" in band:
                ax.axvspan(lo, hi, color="tab:green", alpha=0.08)
        ax.set_title(f"ball from {side}", fontsize=9)
        ax.set_xlabel("height where the ball reaches the plank (px)", fontsize=8)
    axes[0][0].set_ylabel("P(ball beyond the plank), all passes", fontsize=8)
    if axes[0][0].lines:
        axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    args = p.parse_args(argv)
    runs = {n: rows for n in RUNS if (rows := load(args.plan, n)) is not None}
    if not any(n in runs for n in ("both-before", "commit_both-after", "lossmix_e10-after", "lossmix_e20-after",
                                       "lossmix_both-after")):
        return 0                                    # nothing from this scope yet
    out = args.plan / "scope"
    out.mkdir(parents=True, exist_ok=True)
    res = {"height": {}}
    md = ["## Research scope 2026-10-09", "",
          "From per_clip.json (frozen decoder) and blocker/*/crossing.json; nothing is fitted. See "
          "models/vjepa2/scope_report.py.", "",
          "### Narrow gap and both sides: mean P(ball beyond the plank) by the height the ball reaches the plank", "",
          "| run | side | " + " | ".join(BANDS) + f" | AUROC at y {NARROW_AUROC[0]}-{NARROW_AUROC[1]} | AUROC all |",
          "|---|---|" + "---|" * (len(BANDS) + 2)]
    for name, rows in runs.items():
        for side in sorted({r["side_in"] for r in rows}):
            sub = [r for r in rows if r["side_in"] == side]
            for key, label in ((("real_one_ball_far", "real frames"),) if name in ("pretrained", "both-before") else ()) + \
                    (("imagined_one_ball_far", name),):
                h = height_scores(sub, key)
                res["height"][f"{label} / {side}"] = h
                cells = [f"{h[b]['mean']} (n {h[b]['n']})" for b in BANDS]
                a1, a2 = h[f"AUROC at y {NARROW_AUROC[0]}-{NARROW_AUROC[1]}"], h["AUROC all"]
                md.append(f"| {label} | {side} | " + " | ".join(cells) + f" | {a1['auroc']} (n {a1['n']}) | {a2['auroc']} |")
    height_figure(runs, out / "height_profile.png")
    md += ["", "Figure: `scope/height_profile.png`. The slope (downhill) results are in `slope/` (slope.py)."]
    (out / "scope.json").write_text(json.dumps(res, indent=1))
    (out / "scope.md").write_text("\n".join(md) + "\n")
    print(f"-> {out}/ (scope.md, scope.json, height_profile.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

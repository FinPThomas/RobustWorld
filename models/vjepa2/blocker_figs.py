"""Does V-JEPA know where the hidden blocker is? Figures from ball_probe_cv.py's output.

Nothing is fitted here. The V-JEPA reading comes from per_clip.json, which ball_probe_cv.py
writes with the frozen evaluation decoder (cross-validated, context-half only; see CLAUDE.md).
Ground truth (outcomes, tracker positions, the scene file's blocker_polygon) is only used to
place and score passes.

Each pass's tracked ball over the last 1/3 s of context gives a constant-velocity line.
Followed straight, it enters the plank at one point and would leave it at another; both are
measured along the plank's long axis (px from the top-edge midpoint). Against them, P(blocked),
the chance the ball never shows up beyond the plank in the target half:
    true          1 for hidden passes, 0 for through
    real          1 - max P(ball beyond the plank) read from real target frames (upper bound)
    imagined      the same, read from V-JEPA's imagined target
    known blocker reference only: 1 if the straight line touches the scene file's
                  blocker_polygon (needs it set; it is ground truth, not a model)

Outputs (default outputs/vjepa2/blocker/):
    blocker_map.png          every pass's straight-line path, coloured by P(blocked), over a frame
                             with the plank and blocker outlined: true | imagined | known blocker
    blocked_vs_crossing.png  P(blocked) against the expected entry point (left) and the expected
                             exit point (right) along the plank, blocker span shaded. With the
                             blocker at the exit side, the right panel is the sharper step.
    crossing.json            per-pass numbers and AUROCs

Before/after post-training: run ball_probe_cv.py and this script for the base model with
--label base, then for the fine-tuned model with --label fine-tuned --before <base>/crossing.json;
the base curve is overlaid dashed.

    python models/vjepa2/ball_probe_cv.py
    python models/vjepa2/blocker_figs.py --label base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

FIT_S = 1 / 3        # seconds of context, ending at its last frame, for the straight-line fit
MIN_POINTS = 3       # visible tracker points needed for a fit
STEP = 0.25          # source frames per step when walking the straight line
MAX_FRAMES = 300     # how far ahead to walk before giving up on reaching the plank
N_BINS = 8           # bins along the plank for the mean curves


def line_fit(track, first: int, last: int, fps: float):
    """Least-squares constant velocity over visible source frames, ending at `last`.
    -> (x, y, vx, vy, radius) at `last`, velocity in px per source frame; None if too few points."""
    lo = max(first, last - int(round(FIT_S * fps)))
    pts = [(i, track[i][1], track[i][2], track[i][3]) for i in range(lo, last + 1) if track[i][0]]
    if len(pts) < MIN_POINTS:
        return None
    t, x, y, r = (np.array(v, float) for v in zip(*pts))
    vx, x0 = np.polyfit(t - last, x, 1)
    vy, y0 = np.polyfit(t - last, y, 1)
    return float(x0), float(y0), float(vx), float(vy), float(np.median(r))


def plank_axis(poly) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit vector) of the plank's long axis: top-edge midpoint towards bottom-edge midpoint."""
    poly = np.asarray(poly, float)
    ys = poly[:, 1]
    top, bot = poly[ys == ys.min()].mean(0), poly[ys == ys.max()].mean(0)
    d = bot - top
    return top, d / np.linalg.norm(d)


def plank_length(poly) -> float:
    poly = np.asarray(poly, float)
    ys = poly[:, 1]
    return float(np.linalg.norm(poly[ys == ys.max()].mean(0) - poly[ys == ys.min()].mean(0)))


def along(pt, axis) -> float:
    origin, u = axis
    return float(np.dot(np.asarray(pt, float) - origin, u))


def crossing(fit, poly):
    """Where the straight line from `fit` enters and would leave the plank polygon, as (x, y)
    points; None if it never reaches the plank."""
    if fit is None:
        return None
    x0, y0, vx, vy, _ = fit
    poly = np.asarray(poly, np.float32)
    entry = None
    for t in np.arange(0, MAX_FRAMES, STEP):
        p = (float(x0 + vx * t), float(y0 + vy * t))
        inside = cv2.pointPolygonTest(poly, p, False) >= 0
        if entry is None and inside:
            entry = p
        elif entry is not None and not inside:
            return entry, p
    return None


def hits_blocker(entry, exit_, radius: float, blocker) -> bool:
    """Does the ball (radius) touch the blocker anywhere on the straight line through the plank?"""
    poly = np.asarray(blocker, np.float32)
    return any(cv2.pointPolygonTest(poly, (float(entry[0] + (exit_[0] - entry[0]) * t),
                                           float(entry[1] + (exit_[1] - entry[1]) * t)), True) >= -radius
               for t in np.linspace(0, 1, 50))


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.size == 0 or y.min() == y.max():
        return None
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, p)), 3)


def pass_row(clip: dict, probe_row: dict, track, passes, occluder, blocker) -> dict | None:
    """One pass's crossing geometry and P(blocked) readings; None if its line never reaches the plank."""
    from robust_world.eval.ball import source_indices
    n_ctx = clip["context_frames"][1] + 1
    src = source_indices(clip, passes, clip["n_frames"], n_ctx)
    fit = line_fit(track, src[0], src[n_ctx - 1], clip["source_fps"])
    cr = crossing(fit, occluder)
    if cr is None:
        return None
    entry, exit_ = cr
    axis = plank_axis(occluder)
    row = {"clip_id": clip["clip_id"], "outcome": clip["outcome"], "true": int(clip["outcome"] == "hidden"),
           "real": round(1 - max(probe_row["real_far"]), 3),
           "imagined": round(1 - max(probe_row["imagined_far"]), 3),
           "entry_along": round(along(entry, axis), 1), "exit_along": round(along(exit_, axis), 1),
           "fit_xy": [round(fit[0], 1), round(fit[1], 1)],
           "entry_xy": [round(entry[0], 1), round(entry[1], 1)], "exit_xy": [round(exit_[0], 1), round(exit_[1], 1)]}
    if blocker:
        row["known_blocker"] = int(hits_blocker(entry, exit_, fit[4], blocker))
    return row


def summarise(rows: list[dict], occluder, blocker, label: str = "") -> dict:
    axis = plank_axis(occluder)
    span = [round(min(a), 1), round(max(a), 1)] if blocker and (a := [along(p, axis) for p in blocker]) else None
    y = [r["true"] for r in rows]
    out = {"label": label, "plank_length_px": round(plank_length(occluder), 1), "blocker_span_px": span,
           "n_passes": len(rows), "n_blocked": int(sum(y)),
           "auroc_real": auroc(y, [r["real"] for r in rows]),
           "auroc_imagined": auroc(y, [r["imagined"] for r in rows])}
    if blocker:
        out["auroc_known_blocker"] = auroc(y, [r["known_blocker"] for r in rows])
    out["passes"] = rows
    return out


def _binned(x, p, edges):
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    centres, means = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            centres.append((edges[b] + edges[b + 1]) / 2)
            means.append(float(np.mean(np.asarray(p)[m])))
    return centres, means


def write_figures(summary: dict, occluder, blocker, background: np.ndarray, out: Path,
                  before: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    rows, label = summary["passes"], summary["label"]
    if not rows:
        return
    name = f"V-JEPA imagined {label}".strip()
    readings = [("true", "true outcome"), ("imagined", name)]
    if blocker:
        readings.append(("known_blocker", "straight line + known blocker (reference)"))
    occ = np.asarray(occluder, float)
    cmap = plt.get_cmap("coolwarm")

    # 1. Paths over the frame, coloured by P(blocked).
    fig, axes = plt.subplots(1, len(readings), figsize=(4.4 * len(readings), 4.8), dpi=120, squeeze=False)
    for ax, (key, title) in zip(axes[0], readings):
        ax.imshow(background, alpha=0.45)
        lc = LineCollection([[r["fit_xy"], r["exit_xy"]] for r in rows], cmap=cmap, norm=plt.Normalize(0, 1),
                            linewidths=1.6, alpha=0.85)
        lc.set_array(np.array([r[key] for r in rows], float))
        ax.add_collection(lc)
        ax.add_patch(plt.Polygon(occ, fill=False, ec="black", lw=1.2, zorder=3))
        if blocker:
            ax.add_patch(plt.Polygon(np.asarray(blocker, float), fill=False, ec="gold", lw=2.5, zorder=4))
            ax.add_patch(plt.Polygon(np.asarray(blocker, float), fill=False, ec="black", lw=1.2, ls="--", zorder=5))
        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(lc, ax=axes[0].tolist(), fraction=0.02, label="P(blocked)")
    fig.suptitle("Straight-line path from each pass's context to where it would leave the plank"
                 + (" (blocker outlined in gold)" if blocker else ""), fontsize=10)
    fig.savefig(out / "blocker_map.png", bbox_inches="tight")
    plt.close(fig)

    # 2. P(blocked) against where the line enters and leaves the plank.
    edges = np.linspace(0, summary["plank_length_px"], N_BINS + 1)
    colours = {"true": "black", "imagined": "tab:red", "known_blocker": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120, sharey=True)
    rng = np.random.default_rng(0)
    for ax, pos, title in [(axes[0], "entry_along", "expected entry point along the plank (px)"),
                           (axes[1], "exit_along", "expected exit point along the plank (px)")]:
        if summary["blocker_span_px"]:
            ax.axvspan(*summary["blocker_span_px"], color="grey", alpha=0.2, label="blocker")
        x = np.array([r[pos] for r in rows])
        for key, title_ in readings:
            p = np.array([r[key] for r in rows], float)
            jitter = rng.uniform(-0.03, 0.03, len(p)) if key != "imagined" else 0
            ax.scatter(x, p + jitter, s=10, color=colours[key], alpha=0.35)
            cx, m = _binned(x, p, edges)
            style = ":s" if key == "known_blocker" else "-o"
            ax.plot(cx, m, style, color=colours[key], ms=3, lw=2.5 if key == "true" else 1.5,
                    zorder=4 if key == "true" else 3, label=title_)
        if before:
            px = np.array([r[pos] for r in before["passes"]])
            cx, m = _binned(px, [r["imagined"] for r in before["passes"]], edges)
            ax.plot(cx, m, "--", color="tab:red", alpha=0.6, label=f"V-JEPA imagined {before.get('label') or 'before'}")
        ax.set_xlim(0, summary["plank_length_px"])
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel(title)
    axes[0].set_ylabel("P(blocked)")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Where does V-JEPA expect the ball to be blocked? {summary['n_passes']} passes, "
                 f"{summary['n_blocked']} blocked. AUROC imagined {summary['auroc_imagined']} "
                 f"(real frames {summary['auroc_real']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "blocked_vs_crossing.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-clip", type=Path, default=REPO / "outputs" / "vjepa2" / "ball_probe_cv" / "per_clip.json")
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest.jsonl")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2" / "blocker")
    p.add_argument("--label", default="", help="name for this run, e.g. 'base' or 'fine-tuned'")
    p.add_argument("--before", type=Path, help="crossing.json from an earlier run, overlaid dashed")
    args = p.parse_args(argv)

    from robust_world.eval.ball import load_tracking
    from robust_world.eval.io import read_video

    track, passes, scene = load_tracking()
    occluder = scene["occluder_polygon"]
    blocker = scene.get("blocker_polygon") if isinstance(scene.get("blocker_polygon"), list) else None
    probe = {r["clip_id"]: r for r in json.loads(args.per_clip.read_text())}
    clips = [json.loads(line) for line in args.manifest.open()]
    clips = [c for c in clips if c["clip_id"] in probe]
    rows = [r for c in clips if (r := pass_row(c, probe[c["clip_id"]], track, passes, occluder, blocker))]
    summary = summarise(rows, occluder, blocker, args.label)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crossing.json").write_text(json.dumps(summary, indent=1))
    before = json.loads(args.before.read_text()) if args.before else None
    write_figures(summary, occluder, blocker, read_video(REPO / clips[0]["path"])[0], args.out, before)
    print(f"{summary['n_passes']} of {len(clips)} passes reach the plank on a straight line, "
          f"{summary['n_blocked']} blocked; P(blocked) AUROC: imagined {summary['auroc_imagined']}, "
          f"real frames {summary['auroc_real']}"
          + (f", known blocker {summary['auroc_known_blocker']}" if blocker else " (no blocker_polygon set)"))
    print(f"-> {args.out}/ (blocker_map.png, blocked_vs_crossing.png, crossing.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How the slope number is read (Fin, 2026-10-13: "so the slope is read from this? Can you visualise this").

Three panels, all from tracks already scored (nothing is fitted here):
  A  two real open-table rolls, one downhill and one uphill: speed along the motion every 1/8 s, from 5-step
     quadratic fits (the same stretches as slope_map.py). Downhill the ball keeps its speed; uphill it slows fast.
  B  every stretch where the real ball is on open table: its acceleration along the motion (negative = slowing),
     split by direction, for the tracker and for the frozen probe on the same real frames. Friction slows both
     directions alike, the slope helps one and hurts the other, so s = (median down - median up) / 2.
  C  s for each track (real ball, probe on real frames, imagined ball per run) with 95% intervals from
     slope_map.json's same-steps table; runs not yet scored on the aligned open-table clips are marked.

    python models/vjepa2/slope_explain.py --plan outputs/vjepa2/plan
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
sys.path.insert(0, str(HERE))

from slope_map import RUNS, WIN, stretches  # noqa: E402
from slope_test import Table, plank_distance  # noqa: E402

DOWN, UP = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
STEP_S = 1 / 8


def along(s: dict) -> tuple[float, float]:
    sp = float(np.linalg.norm(s["v"]))
    return sp, float(s["a"] @ s["v"] / sp)


def windows(pos: list, table, dist) -> list[tuple[int, dict]]:
    out = []
    for t in range(len(pos) - WIN + 1):
        s = stretches(pos[t:t + WIN], table, dist, 0)
        if s:
            out.append((t + WIN // 2, s[0]))
    return out


def per_stretch(rows: list[dict], key: str, table, dist) -> dict:
    """Along-motion acceleration of the track `key` and of the real ball on the same windows, by real direction."""
    out = {"track": {True: [], False: []}, "real": {True: [], False: []}}
    for r in rows:
        p, q = r.get(key) or [], r.get("true_pos") or []
        for t in range(min(len(p), len(q)) - WIN + 1):
            a, b = stretches(p[t:t + WIN], table, dist, 0), stretches(q[t:t + WIN], table, dist, 0)
            if a and b:
                d = bool(b[0]["v"][0] < 0)
                out["track"][d].append(along(a[0])[1])
                out["real"][d].append(along(b[0])[1])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    p.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = p.parse_args(argv)
    cal, smap = args.plan / "slope_test" / "slope_test.json", args.plan / "slope_map" / "slope_map.json"
    opn = args.plan / "scores" / "both-before" / "per_clip_open.json"
    if not (cal.exists() and smap.exists() and opn.exists()):
        print("slope_explain: needs slope_test.json, slope_map.json and both-before open-table scores")
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = Table(json.loads(cal.read_text())["calibration"])
    dist = plank_distance(json.loads(args.scene.read_text()))
    rows = json.loads(opn.read_text())
    plank = args.plan / "scores" / "both-before" / "per_clip.json"
    every = rows + (json.loads(plank.read_text()) if plank.exists() else [])   # same rows as slope_map's table
    same = json.loads(smap.read_text()).get("same_steps", {})
    key = "real_pos_nn" if any(r.get("real_pos_nn") for r in rows) else "real_pos"

    # A: a downhill and an uphill real roll, long and clean, starting at about the same speed
    cand = {True: [], False: []}
    for r in rows:
        w = windows(list(r.get("true_pos_context") or []) + list(r.get("true_pos") or []), table, dist)
        ks = [k for k, _ in w]
        if len(w) < 8 or ks[-1] - ks[0] != len(w) - 1:          # unbroken run of windows only
            continue
        dirs = [s["v"][0] < 0 for _, s in w]
        if all(dirs) or not any(dirs):
            cand[dirs[0]].append((r, w))
    best, gap = {}, None
    for dn in cand[True]:
        for up in cand[False]:
            g = abs(along(dn[1][0][1])[0] - along(up[1][0][1])[0]) - 0.002 * (len(dn[1]) + len(up[1]))
            if gap is None or g < gap:
                best, gap = {True: dn, False: up}, g
    st = per_stretch(every, key, table, dist)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25, 1.15]})
    fig.patch.set_facecolor("white")

    a = ax[0]
    for d, col, lab in ((True, DOWN, "rolling downhill (left)"), (False, UP, "rolling uphill (right)")):
        if d not in best:
            continue
        r, w = best[d]
        t = np.array([k for k, _ in w]) * STEP_S
        sp = np.array([along(s)[0] for _, s in w])
        a.plot(t - t[0], sp, color=col, lw=2, label=lab)
        a.plot(t - t[0], sp, "o", color=col, ms=5, mec="white", mew=1.5)
    a.set_xlabel("time (s)")
    a.set_ylabel("speed along the motion (ball diameters per 1/8 s)")
    a.set_title("A. Two real rolls on open table", loc="left", color=INK)
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.set_ylim(bottom=0)
    a.grid(axis="y", color=GRID)
    a.text(0.0, -0.2, "Each point: a quadratic fit over 5 steps (0.6 s).\nDownhill keeps its speed; uphill "
           "slows fast.", transform=a.transAxes, color=MUTED, fontsize=9, va="top")

    b = ax[1]
    rng = np.random.default_rng(0)
    groups = [("real ball\n(tracker)", st["real"]), ("probe on\nreal frames", st["track"])]
    for gi, (gl, g) in enumerate(groups):
        meds = {}
        for d, col, dx in ((True, DOWN, -0.18), (False, UP, 0.18)):
            v = np.array(g[d])
            if not len(v):
                continue
            x = gi + dx + rng.uniform(-0.09, 0.09, len(v))
            b.scatter(x, np.clip(v, -0.25, 0.25), s=9, color=col, alpha=0.35, lw=0)
            meds[d] = float(np.median(v))
            b.plot([gi + dx - 0.13, gi + dx + 0.13], [meds[d]] * 2, color=INK, lw=2.5)
        if len(meds) == 2:
            b.annotate("", xy=(gi + 0.36, meds[True]), xytext=(gi + 0.36, meds[False]),
                       arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
            b.text(gi + 0.4, 0.17 - 0.0 * gi, f"s = gap/2\n   = {(meds[True] - meds[False]) / 2:.3f}",
                   va="center", fontsize=9.5, color=INK, fontweight="bold")
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(range(len(groups)), [g for g, _ in groups])
    b.set_xlim(-0.5, len(groups) - 0.05)
    b.set_ylim(-0.25, 0.25)
    b.set_ylabel("acceleration along the motion (diam / step², below 0 = slowing)")
    b.set_title("B. Every 0.6 s stretch on open table", loc="left", color=INK)
    b.scatter([], [], color=DOWN, label="downhill"), b.scatter([], [], color=UP, label="uphill")
    b.legend(frameon=False, loc="lower left", fontsize=9)
    b.grid(axis="y", color=GRID)
    b.text(0.0, -0.2, "All clips, ball off the plank. Friction slows both directions alike; the slope helps\ndownhill "
           "and hurts uphill, so half the gap between the medians (black bars) is the slope's pull s.", transform=b.transAxes, color=MUTED,
           fontsize=9, va="top")

    c = ax[2]
    probe = next((v for k, v in same.items() if k.startswith("probe on real")), {})
    bars = [("real ball (tracker)", probe.get("s_real"), probe.get("s_real_ci")),
            ("probe on real frames", probe.get("s_track"), probe.get("s_track_ci"))]
    for name, lab in RUNS[1:]:
        if name == "lossmix_both_e30-after":           # F: dropped for the last Kaggle hours
            continue
        r = next((v for k, v in same.items() if k.startswith(lab)), None)
        short = lab.replace("imagined, after post-training ", "imagined after ").replace("imagined, after ", "imagined after ").replace(" aligned open clips", "")
        bars.append((short, r.get("s_track") if r else None, r.get("s_track_ci") if r else None))
    for i, (lab, s, ci) in enumerate(bars):
        y = len(bars) - 1 - i
        if s is None:
            c.text(0.001, y, "waiting for the next Kaggle run", va="center", fontsize=8.5,
                   color=MUTED, style="italic")
        else:
            c.barh(y, s, color=DOWN if i == 0 else ("#86b6ef" if i == 1 else "#1c5cab"), height=0.55)
            if ci:
                c.plot(ci, [y, y], color=INK, lw=1.2)
            c.text(max(s, ci[1] if ci else s) + 0.002, y, f"{s:.3f}", va="center", fontsize=9, color=INK)
    c.set_yticks(range(len(bars)), [b_[0] for b_ in bars][::-1], fontsize=9)
    c.set_ylim(-0.6, len(bars) - 0.4)
    c.axvline(0, color=MUTED, lw=0.8)
    c.set_xlim(-0.005, 0.05)
    c.set_xlabel("s: downhill pull (diam / step²), line = 95% interval")
    c.set_title("C. The slope number per track", loc="left", color=INK)
    c.grid(axis="x", color=GRID)
    c.text(0.0, -0.2, "A world model that learnt the slope should reach the probe-on-real value,\nsince the "
           "same probe reads its imagined ball.", transform=c.transAxes, color=MUTED, fontsize=9, va="top")
    out = args.plan / "slope_map" / "slope_explained.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"slope_explain: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

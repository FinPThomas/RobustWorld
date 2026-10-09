"""Did the model learn that the table slopes? Slope estimates from ball tracks (research scope 2026-10-09, §1).

Nothing here is fitted to a model's output in order to read it: every track is either the tracker's
(ground truth, S0) or the frozen evaluation decoder's per-step ball position (S1-S4, per_clip.json from
ball_probe_cv.py). The regression below is the measurement itself, applied identically to every track
source, like a speed-gun; it never sees outcomes (CLAUDE.md rules 5-6).

On open table a rolling ball's acceleration along its motion is a = -f(v) + s downhill and -f(v) - s uphill
(f: rolling friction, depends on speed; s: the slope term). At the same speed friction cancels:
s = (a_down - a_up) / 2. Two estimates per track source:

    1-D   samples moving mostly along the table (|vx| > 0.7 |v|), in speed bins; per bin
          s = (mean a_down - mean a_up) / 2 with "down" = moving left; pooled over bins (sample-weighted).
          Also per table region: right of the plank (x > 270) and left of it (x < 103).
    2-D   least squares over all samples: a = -f_b * v/|v| + G, with f_b one friction value per speed bin and
          G one constant 2-D vector (downhill direction and size); "G map": the same with one G per 128 px
          region (regions with too few samples are left out).

Samples are consecutive steps (1/8 s) where the real ball is on open table (over 2 cells from the plank):
v = (p[t+1] - p[t-1]) / 2, a = p[t+1] - 2 p[t] + p[t-1], in px per step (squared). The same steps are used
for every source of the same clips, so decoded and imagined tracks are compared on equal footing. 95%
intervals: bootstrap over clips.

Sources (where their scores exist):
    S0  tracker, context + target steps (all clips of the both-directions scoring pass): the true slope
    S0t tracker, target steps only (the same steps the model's tracks cover)
    S1  decoder on real target frames: the ceiling at the decoder's resolution
    S2  imagined, pretrained (expected about 0)
    S3  imagined, post-trained on right clips only (B, at epoch 10 and 20)
    S4  imagined, post-trained on both directions (A)

Outputs: <plan>/slope/slope.json, slope.md, acc_vs_speed.png, gradient_map.png

    python models/vjepa2/slope.py --plan outputs/vjepa2/plan [--plank-cm 60]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

OPEN_TABLE_PX = 64
ALONG = 0.7                      # |vx| / |v| above this counts as moving along the table
REGION_PX = 128
MIN_REGION_SAMPLES = 15
PLANK_X = (103, 270)             # the plank spans x 103-270 px (configs/scenes/whole.json)
PLANK_PX = 449.6                 # its length in the 512 px frame (blocker_figs.py)
STEP_S = 1 / 8                   # seconds per step (2 frames at 16 fps)
N_BOOT = 300
# (label, run, which track, steps): context+target tracker tracks need "true_pos_context".
SOURCES = [("S0 tracker (all steps)", "both-before", "true_all"),
           ("S0t tracker (target steps)", "both-before", "true"),
           ("S1 decoder, real frames", "both-before", "real"),
           ("S2 imagined, pretrained", "both-before", "imagined"),
           ("S3 imagined, B right only @10", "lossmix_e10-after", "imagined"),
           ("S3 imagined, B right only @20", "lossmix_e20-after", "imagined"),
           ("S4 imagined, A both directions", "commit_both-after", "imagined")]


def plank_distance(scene_path: Path):
    import cv2
    poly = np.array(json.loads(scene_path.read_text())["occluder_polygon"], np.float32)
    return lambda x, y: -cv2.pointPolygonTest(poly, (float(x), float(y)), True)


def tracks(r: dict, kind: str) -> tuple[list, list]:
    """(positions, open-table mask source) per step for one clip and one track kind."""
    true = r.get("true_pos") or []
    if kind == "true_all":
        ctx = r.get("true_pos_context") or []
        return ctx + true, ctx + true
    pos = {"true": true, "real": r.get("real_pos"), "imagined": r.get("imagined_pos")}[kind]
    return [None if p is None else p[:2] for p in pos], true


def samples(rows: list[dict], kind: str, dist) -> list[dict]:
    """Per clip: arrays of (position, velocity, acceleration) at open-table steps."""
    out = []
    for ci, r in enumerate(rows):
        pos, ref = tracks(r, kind)
        ok = [ref[t] is not None and pos[t] is not None and dist(*ref[t]) > OPEN_TABLE_PX for t in range(len(pos))]
        P, V, A = [], [], []
        for t in range(1, len(pos) - 1):
            if ok[t - 1] and ok[t] and ok[t + 1]:
                p0, p1, p2 = (np.asarray(pos[k], float) for k in (t - 1, t, t + 1))
                v, a = (p2 - p0) / 2, p2 - 2 * p1 + p0
                if np.linalg.norm(v) > 0.5:
                    P.append(p1), V.append(v), A.append(a)
        if P:
            out.append({"clip": ci, "side": r.get("side_in", "R"), "p": np.array(P), "v": np.array(V), "a": np.array(A)})
    return out


def stack(s: list[dict]):
    if not s:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    return (np.concatenate([x["p"] for x in s]), np.concatenate([x["v"] for x in s]),
            np.concatenate([x["a"] for x in s]))


def slope_1d(s: list[dict], edges: np.ndarray, region=None) -> dict:
    p, v, a = stack(s)
    speed = np.linalg.norm(v, axis=1) if len(v) else np.zeros(0)
    keep = (np.abs(v[:, 0]) > ALONG * speed) if len(v) else np.zeros(0, bool)
    if region is not None and len(p):
        keep &= region(p[:, 0])
    p, v, a, speed = p[keep], v[keep], a[keep], speed[keep]
    along = (a * v).sum(1) / np.maximum(speed, 1e-9)
    down = v[:, 0] < 0
    b = np.digitize(speed, edges)
    per_bin, num, den = [], 0.0, 0
    for k in range(1, len(edges)):
        d, u = along[(b == k) & down], along[(b == k) & ~down]
        if len(d) >= 3 and len(u) >= 3:
            sb = (d.mean() - u.mean()) / 2
            per_bin.append({"speed": [round(float(edges[k - 1]), 1), round(float(edges[k]), 1)], "s": round(float(sb), 4),
                            "a_down": round(float(d.mean()), 4), "a_up": round(float(u.mean()), 4),
                            "n_down": int(len(d)), "n_up": int(len(u))})
            num, den = num + sb * (len(d) + len(u)), den + len(d) + len(u)
    return {"s": num / den if den else None, "bins": per_bin, "n": int(len(along))}


def slope_2d(s: list[dict], edges: np.ndarray, regions: bool = False) -> dict | None:
    p, v, a = stack(s)
    if len(v) < 10:
        return None
    speed = np.linalg.norm(v, axis=1)
    vh = v / speed[:, None]
    b = np.clip(np.digitize(speed, edges) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    if regions:
        cell = (p // REGION_PX).astype(int)
        keys, inv, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        good = counts >= MIN_REGION_SAMPLES
        keep = good[inv]
        if not keep.any():
            return None
        vh, a, b, inv = vh[keep], a[keep], b[keep], inv[keep]
        idx = {k: i for i, k in enumerate(np.flatnonzero(good))}
        g_col = np.array([idx[k] for k in inv])
        ng = len(idx)
    else:
        g_col, ng = np.zeros(len(vh), int), 1
    n = len(vh)
    X = np.zeros((2 * n, nb + 2 * ng))
    y = np.concatenate([a[:, 0], a[:, 1]])
    X[np.arange(n), b] = -vh[:, 0]
    X[n + np.arange(n), b] = -vh[:, 1]
    X[np.arange(n), nb + 2 * g_col] = 1
    X[n + np.arange(n), nb + 2 * g_col + 1] = 1
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    G = coef[nb:].reshape(ng, 2)
    out = {"friction": [round(float(f), 4) for f in coef[:nb]], "n": int(n)}
    if regions:
        out["regions"] = [{"cell_px": [int(c) * REGION_PX for c in keys[k]], "G": [round(float(g), 4) for g in G[idx[k]]]}
                          for k in idx]
    else:
        out["G"] = [round(float(G[0, 0]), 4), round(float(G[0, 1]), 4)]
        out["s_left"] = round(float(-G[0, 0]), 4)          # the slope term along the table, downhill = left
    return out


def boot(s: list[dict], edges: np.ndarray, rng) -> dict:
    clips = sorted({x["clip"] for x in s})
    by = {}
    for x in s:
        by.setdefault(x["clip"], []).append(x)
    s1, s2 = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(clips, len(clips))
        ss = [x for c in pick for x in by[c]]
        r1, r2 = slope_1d(ss, edges)["s"], slope_2d(ss, edges)
        if r1 is not None:
            s1.append(r1)
        if r2 is not None:
            s2.append(r2["s_left"])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if v else None  # noqa: E731
    return {"s_1d_ci": ci(s1), "s_left_2d_ci": ci(s2)}


def degrees(s_px_step2: float | None, plank_cm: float | None) -> float | None:
    """Slope angle from s (= 5/7 g sin(theta) for a solid rolling ball), given the plank's real length."""
    if s_px_step2 is None or not plank_cm:
        return None
    m_per_px = plank_cm / 100 / PLANK_PX
    acc = s_px_step2 * m_per_px / STEP_S ** 2
    return round(math.degrees(math.asin(max(-1.0, min(1.0, acc / (5 / 7 * 9.81))))), 3)


def figures(res: dict, samples_by: dict, edges: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in res if n in samples_by]
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], names):
        for key, col, lab in (("a_down", "tab:blue", "downhill (moving left)"), ("a_up", "tab:red", "uphill (moving right)")):
            bins = res[n]["1d"]["bins"]
            ax.plot([np.mean(b["speed"]) if np.isfinite(b["speed"][1]) else b["speed"][0] for b in bins],
                    [b[key] for b in bins], "o-", color=col, label=lab, ms=3)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(n, fontsize=7)
        ax.set_xlabel("speed (px / step)", fontsize=7)
    axes[0][0].set_ylabel("acceleration along motion (px / step²)", fontsize=7)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Gap between the curves = 2 x slope term", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "acc_vs_speed.png", dpi=120)
    plt.close(fig)

    maps = [n for n in names if res[n].get("map")]
    if maps:
        fig, axes = plt.subplots(1, len(maps), figsize=(3.3 * len(maps), 3.4), squeeze=False)
        for ax, n in zip(axes[0], maps):
            regs = res[n]["map"]["regions"]
            xs = [r["cell_px"][0] + REGION_PX / 2 for r in regs]
            ys = [r["cell_px"][1] + REGION_PX / 2 for r in regs]
            ax.quiver(xs, ys, [r["G"][0] for r in regs], [r["G"][1] for r in regs], angles="xy", scale_units="xy",
                      scale=0.01, color="tab:blue")
            ax.axvspan(*PLANK_X, color="tab:brown", alpha=0.15)
            ax.set_xlim(0, 512), ax.set_ylim(512, 0)
            ax.set_title(n, fontsize=7)
            ax.set_aspect("equal")
        fig.suptitle("Downhill vector G per 128 px region (arrow 1 px = 0.01 px/step²; plank shaded)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "gradient_map.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    p.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    p.add_argument("--plank-cm", type=float, default=None,
                   help="the plank's real length, to give the slope in degrees (default: the scene file's plank_length_cm)")
    args = p.parse_args(argv)
    if args.plank_cm is None and args.scene.exists():     # set "plank_length_cm" in the scene file once measured
        args.plank_cm = json.loads(args.scene.read_text()).get("plank_length_cm")
    runs = {}
    for _, run, _ in SOURCES:
        f = args.plan / "scores" / run / "per_clip.json"
        if run not in runs and f.exists():
            rows = json.loads(f.read_text())
            if any("true_pos" in r for r in rows):
                runs[run] = rows
    if "both-before" not in runs:
        return 0                                  # no positions scored yet
    dist = plank_distance(args.scene)
    samples_by = {label: samples(runs[run], kind, dist) for label, run, kind in SOURCES if run in runs}
    _, v0, _ = stack(samples_by[SOURCES[0][0]])
    sp = np.linalg.norm(v0, axis=1)
    edges = np.unique(np.quantile(sp, np.linspace(0, 1, 6))) if len(sp) > 20 else np.array([0, 4, 8, 16, 1e9])
    edges[0], edges[-1] = 0.0, np.inf
    rng = np.random.default_rng(0)
    res = {}
    right, left = (lambda x: x > PLANK_X[1]), (lambda x: x < PLANK_X[0])
    for label, s in samples_by.items():
        r1 = slope_1d(s, edges)
        res[label] = {"n_clips": len({x["clip"] for x in s}), "1d": r1, "2d": slope_2d(s, edges),
                      "map": slope_2d(s, edges, regions=True),
                      "right_of_plank": slope_1d(s, edges, right)["s"], "left_of_plank": slope_1d(s, edges, left)["s"],
                      **boot(s, edges, rng)}
    s_true = res[SOURCES[0][0]]["1d"]["s"]
    md = ["## Table slope from ball tracks", "",
          "s: slope term along the table (px/step², downhill = left; 1 step = 1/8 s). Share of S0: the source's s over "
          "the tracker's. Measured on open-table steps only; nothing fitted to read a model. See models/vjepa2/slope.py.", "",
          "| source | clips | s (1-D) | 95% CI | share of S0 | s (2-D, -Gx) | 95% CI | G (x, y) | right of plank | "
          "left of plank | degrees |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    r4 = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    for label, r in res.items():
        s1 = r["1d"]["s"]
        share = round(s1 / s_true, 2) if s1 is not None and s_true else None
        g2 = r["2d"] or {}
        md.append(f"| {label} | {r['n_clips']} | {r4(s1)} | {r['s_1d_ci']} | {share} | {g2.get('s_left')} | "
                  f"{r['s_left_2d_ci']} | {g2.get('G')} | {r4(r['right_of_plank'])} | {r4(r['left_of_plank'])} | "
                  f"{degrees(s1, args.plank_cm)} |")
    md += ["", "Figures: `slope/acc_vs_speed.png` (downhill and uphill acceleration by speed), "
           "`slope/gradient_map.png` (G per region)."]
    out = args.plan / "slope"
    out.mkdir(parents=True, exist_ok=True)
    (out / "slope.json").write_text(json.dumps({"speed_bin_edges": [float(e) for e in edges[:-1]], "sources": res},
                                               indent=1, default=float))
    (out / "slope.md").write_text("\n".join(md) + "\n")
    figures(res, samples_by, edges, out)
    print(f"-> {out}/ (slope.md, slope.json, acc_vs_speed.png, gradient_map.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

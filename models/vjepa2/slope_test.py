"""Did the world model learn the table slope? Uphill vs downhill at matched speed, in ball sizes (Fin, 2026-10-10).

The camera looks down at about 45 degrees, so pixels near the camera cover less table than far ones and a ball
rolling towards or away from it seems to speed up or slow down. The ball's own size undoes that: a ball looks
smaller in proportion to its distance, and for points on a flat table 1/distance is a linear function of the
image position. So the tracker's radius r ~ a x + b y + c (fitted once on real frames, where the ball is on open
table), and (x, y) * r0 / r(x, y) are positions on the table itself (up to a fixed linear map), in pixels at the
typical depth r0. Speeds and accelerations below are in ball diameters (2 r0) per step (1/8 s).

This calibration is camera geometry from real frames only: it never sees outcomes, predictions or any model,
and it is the same for every run (CLAUDE.md rules 1-3, 5-6).

    real   the tracker over the whole video (every frame where the ball is visible on open table, no hand in
           shot): quadratic fits over 0.5 s windows give speed v and acceleration a along the motion. At the same
           speed friction cancels, so the slope term is s = (a_down - a_up) / 2 (down = rolling left).
    model  through passes, scored per run: from the last context steps (real frames) the ball's position and
           speed when it reaches the plank; the steady-speed position T steps later is p0 + v0 T. Each target
           step where the ball is (or is imagined) visible beyond the plank gives the gap e = (p - p0 - v0 T) along
           the motion; a per clip is the least-squares fit e = a T^2 / 2. Downhill passes (ball from the right)
           against uphill ones (from the left) in speed bins: s = (a_down - a_up) / 2. Sources: the tracker
           (true positions), the frozen decoder on real frames and on each run's imagined frames.
    epochs the model's s and the through-vs-blocked AUROC (per side) for the run saved after 1, 2, 4, 7, 10, 15
           and 20 epochs (C: both directions, loss-weighted sampling), against the real s.

Outputs: <plan>/slope_test/slope_test.json, slope_test.md, epochs.png, real_acc_vs_speed.png

    python models/vjepa2/slope_test.py --plan outputs/vjepa2/plan
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

OPEN_PX = 48                     # the ball centre is this far from the plank outline (open table)
EDGE_PX = 24                     # and this far inside the frame
STEP_S = 1 / 8
WINDOW_S = 0.5                   # real-video fits
HAND_PAD_S = 0.5                 # frames this close to a hand in shot are left out (pushes, catches)
MIN_PEAK = 0.5                   # decoded/imagined positions count when the ball map peaks this high
N_BOOT = 500
SPEED_BINS = 3
# Learning curve: (epoch, run). Epoch 0 is the pretrained predictor scored on both directions.
EPOCH_RUNS = [(0, "both-before")] + [(e, f"lossmix_both_e{e}-after") for e in (1, 2, 4, 7, 10, 15)] + \
             [(20, "lossmix_both-after")]
OTHER_RUNS = ["commit_both-after", "lossmix_e10-after", "lossmix_e20-after"]


# ------------------------------------------------------------------------------------------- calibration

def plank_distance(scene: dict):
    import cv2
    poly = np.array(scene["occluder_polygon"], np.float32)
    return lambda x, y: -cv2.pointPolygonTest(poly, (float(x), float(y)), True)


def open_table(x, y, dist) -> bool:
    return EDGE_PX < x < 512 - EDGE_PX and EDGE_PX < y < 512 - EDGE_PX and dist(x, y) > OPEN_PX


def fit_radius(rows: list[dict], dist) -> dict:
    """Robust least squares r = a x + b y + c over visible open-table frames (3 rounds, dropping > 2.5 MAD)."""
    pts = np.array([(r["x"], r["y"], r["radius"]) for r in rows if r["visible"] and r.get("radius")
                    and open_table(r["x"], r["y"], dist)], float)
    if len(pts) < 50:
        return {"ok": False, "n": int(len(pts))}
    keep = np.ones(len(pts), bool)
    X = np.c_[pts[:, :2], np.ones(len(pts))]
    for _ in range(3):
        coef = np.linalg.lstsq(X[keep], pts[keep, 2], rcond=None)[0]
        res = pts[:, 2] - X @ coef
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-9
        keep = np.abs(res) < 2.5 * 1.4826 * mad
    pred = X[keep] @ coef
    r2 = 1 - ((pts[keep, 2] - pred) ** 2).sum() / max(((pts[keep, 2] - pts[keep, 2].mean()) ** 2).sum(), 1e-9)
    r0 = float(np.median(pts[keep, 2]))
    return {"ok": True, "coef": [float(c) for c in coef], "r0": r0, "r2": round(float(r2), 3), "n": int(keep.sum()),
            "radius_range_px": [round(float(np.percentile(pred, 2)), 2), round(float(np.percentile(pred, 98)), 2)]}


class Table:
    """Image px -> table position in ball diameters (identity in px / (2 r0) when no calibration is possible)."""

    def __init__(self, cal: dict, fallback_r0: float = 8.0):
        self.cal = cal
        self.r0 = cal["r0"] if cal.get("ok") else fallback_r0

    def __call__(self, p) -> np.ndarray:
        p = np.asarray(p, float)[..., :2]
        if not self.cal.get("ok"):
            return p / (2 * self.r0)
        a, b, c = self.cal["coef"]
        r = np.maximum(a * p[..., 0] + b * p[..., 1] + c, 0.25 * self.r0)
        return p * (self.r0 / r)[..., None] / (2 * self.r0)


# ------------------------------------------------------------------------------------------- real slope

def real_samples(rows: list[dict], fps: float, table: Table, dist, hand_px: float) -> tuple[list[dict], dict]:
    """Per stretch of consecutive usable frames: speed and acceleration from quadratic fits over WINDOW_S."""
    n = len(rows)
    hand = np.array([(r.get("fg_area") or 0) >= hand_px for r in rows])
    note = {}
    if hand.mean() > 0.8:                       # the foreground measure doesn't separate hands here: don't use it
        hand[:] = False
        note["hand_filter"] = "off (over 80% of frames flagged)"
    pad = int(round(HAND_PAD_S * fps))
    near_hand = np.convolve(hand.astype(float), np.ones(2 * pad + 1), "same") > 0
    good = np.array([bool(r["visible"]) and r["x"] is not None and open_table(r["x"], r["y"], dist) for r in rows]) & ~near_hand
    w = max(5, int(round(WINDOW_S * fps)))
    t_steps = 1 / (fps * STEP_S)                # steps per frame
    out, seg, i = [], 0, 0
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < n and good[j]:
            j += 1
        if j - i >= w:
            q = table(np.array([(rows[k]["x"], rows[k]["y"]) for k in range(i, j)]))
            P, V, A = [], [], []
            for s in range(0, j - i - w + 1, max(1, w // 2)):
                T = (np.arange(w) - (w - 1) / 2) * t_steps
                c = np.polyfit(T, q[s:s + w], 2)            # [3, 2]: a/2, v, p
                P.append(c[2]), V.append(c[1]), A.append(2 * c[0])
            out.append({"clip": seg, "p": np.array(P), "v": np.array(V), "a": np.array(A)})
            seg += 1
        i = j
    note.update({"stretches": seg, "frames_used": int(good.sum()), "window_frames": w})
    return out, note


def slope_binned(v: np.ndarray, a: np.ndarray, down: np.ndarray, edges: np.ndarray) -> dict:
    """s = (mean a_down - mean a_up) / 2 per speed bin (accelerations along the motion), pooled by count."""
    speed = np.linalg.norm(v, axis=1)
    b = np.clip(np.digitize(speed, edges) - 1, 0, len(edges) - 2)
    bins, num, den = [], 0.0, 0
    for k in range(len(edges) - 1):
        d, u = a[(b == k) & down], a[(b == k) & ~down]
        if len(d) >= 3 and len(u) >= 3:
            sb = (d.mean() - u.mean()) / 2
            bins.append({"speed": [round(float(edges[k]), 3), round(float(edges[k + 1]), 3)], "s": round(float(sb), 5),
                         "a_down": round(float(d.mean()), 5), "a_up": round(float(u.mean()), 5),
                         "n_down": int(len(d)), "n_up": int(len(u))})
            num, den = num + sb * (len(d) + len(u)), den + len(d) + len(u)
    return {"s": num / den if den else None, "bins": bins}


def along(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (a * v).sum(1) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)


def real_slope(samples: list[dict], rng) -> dict:
    if not samples:
        return {"s": None}
    v, a = np.concatenate([x["v"] for x in samples]), np.concatenate([x["a"] for x in samples])
    seg = np.concatenate([np.full(len(x["v"]), x["clip"]) for x in samples])
    keep = (np.abs(v[:, 0]) > 0.7 * np.linalg.norm(v, axis=1)) & (np.linalg.norm(v, axis=1) > 0.05)
    v, a, seg = v[keep], a[keep], seg[keep]
    acc, down = along(v, a), v[:, 0] < 0
    edges = speed_edges(np.linalg.norm(v, axis=1)[down], np.linalg.norm(v, axis=1)[~down])
    res = slope_binned(v, acc, down, edges)
    boots, ids = [], np.unique(seg)
    by = {i: np.flatnonzero(seg == i) for i in ids}
    for _ in range(N_BOOT):
        idx = np.concatenate([by[i] for i in rng.choice(ids, len(ids))])
        s = slope_binned(v[idx], acc[idx], down[idx], edges)["s"]
        if s is not None:
            boots.append(s)
    return {**res, "ci": ci(boots), "n_down": int(down.sum()), "n_up": int((~down).sum()),
            "speed_edges": [float(e) for e in edges]}


def speed_edges(sd: np.ndarray, su: np.ndarray) -> np.ndarray:
    """SPEED_BINS bins over the speeds both directions share (matched speeds only)."""
    if len(sd) < 3 or len(su) < 3:
        return np.array([0.0, np.inf])
    lo, hi = max(np.percentile(sd, 2), np.percentile(su, 2)), min(np.percentile(sd, 98), np.percentile(su, 98))
    if hi <= lo:
        return np.array([0.0, np.inf])
    both = np.concatenate([sd, su])
    both = both[(both >= lo) & (both <= hi)]
    e = np.unique(np.quantile(both, np.linspace(0, 1, SPEED_BINS + 1)))
    return e if len(e) >= 2 else np.array([lo, hi])


def ci(v) -> list | None:
    return [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)] if len(v) > 10 else None


# ------------------------------------------------------------------------------------------- model test

def entry(r: dict, table: Table, dist):
    """(step index, table position, table velocity) of the real ball at its last open-table context steps."""
    c = r.get("true_pos_context") or []
    idx = [k for k, p in enumerate(c) if p is not None and dist(*p[:2]) > OPEN_PX / 2]
    if len(idx) < 2:
        return None
    run = [idx[-1]]
    for k in reversed(idx[:-1]):
        if k != run[-1] - 1 or len(run) == 4:
            break
        run.append(k)
    if len(run) < 2:
        return None
    run = run[::-1]
    q = table(np.array([c[k][:2] for k in run]))
    slope_, icpt = np.polyfit(np.array(run, float), q, 1)
    return run[-1], slope_ * run[-1] + icpt, slope_


def clip_accel(r: dict, src: dict, key: str, table: Table, dist) -> dict | None:
    """Least-squares a in e = a T^2 / 2 over the target steps where the source has the ball on open table."""
    if r["outcome"] != "through":
        return None
    e0 = entry(r, table, dist)
    if e0 is None:
        return None
    k, p0, v0 = e0
    speed = float(np.linalg.norm(v0))
    if speed < 1e-3:
        return None
    vh = v0 / speed
    num = den = 0.0
    n = 0
    for j, p in enumerate(src.get(key) or []):
        if p is None or (len(p) > 2 and p[2] < MIN_PEAK) or dist(*p[:2]) < OPEN_PX / 2:
            continue
        T = len(r.get("true_pos_context") or []) + j - k
        e = float((table(p) - p0 - v0 * T) @ vh)
        num, den, n = num + e * T * T, den + 0.5 * T ** 4, n + 1
    if not n:
        return None
    return {"side": r.get("side_in", "R"), "speed": speed, "a": num / den, "n": n}


def model_slope(per: list[dict], edges: np.ndarray, rng) -> dict:
    if not per:
        return {"s": None, "n_down": 0, "n_up": 0}
    a = np.array([x["a"] for x in per])
    sp = np.array([x["speed"] for x in per])
    down = np.array([x["side"] == "R" for x in per])        # rolled in from the right = downhill (leftwards)
    v = np.c_[sp, np.zeros_like(sp)]
    res = slope_binned(v, a, down, edges)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(a), len(a))
        s = slope_binned(v[i], a[i], down[i], edges)["s"]
        if s is not None:
            boots.append(s)
    return {**res, "ci": ci(boots), "n_down": int(down.sum()), "n_up": int((~down).sum()),
            "a_down": round(float(a[down].mean()), 5) if down.any() else None,
            "a_up": round(float(a[~down].mean()), 5) if (~down).any() else None}


def auroc_side(rows: list[dict], side: str):
    from scope_report import auroc
    sub = [r for r in rows if r.get("side_in", "R") == side]
    return auroc([r["outcome"] == "through" for r in sub], [max(r["imagined_one_ball_far"]) for r in sub]) if sub else None


# ------------------------------------------------------------------------------------------- report

def figures(res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [x for x in res["epochs"] if x["model"]["s"] is not None or x["auroc_R"] is not None]
    if ep:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
        e = [x["epoch"] for x in ep]
        s = [x["model"]["s"] for x in ep]
        lo = [x["model"]["ci"][0] if x["model"].get("ci") else np.nan for x in ep]
        hi = [x["model"]["ci"][1] if x["model"].get("ci") else np.nan for x in ep]
        a1.plot(e, [np.nan if v is None else v for v in s], "o-", color="tab:blue", label="imagined")
        a1.fill_between(e, lo, hi, color="tab:blue", alpha=0.15)
        for key, style, lab in (("tracker", "k--", "real ball (same clips)"), ("decoder_real", "k:", "decoder on real frames")):
            if res["clips"].get(key, {}).get("s") is not None:
                a1.axhline(res["clips"][key]["s"], ls=style[1:], color="k", lw=1, label=lab)
        if res["real"].get("s") is not None:
            a1.axhline(res["real"]["s"], color="tab:green", lw=1, label="real ball, whole video")
        a1.axhline(0, color="grey", lw=0.5)
        a1.set_xlabel("epochs of post-training", fontsize=8)
        a1.set_ylabel("slope term s (ball diameters / step²)", fontsize=8)
        a1.set_title("Downhill minus uphill, at matched speed (/2)", fontsize=9)
        a1.legend(fontsize=6)
        for side, col in (("R", "tab:red"), ("L", "tab:purple")):
            a2.plot(e, [np.nan if x[f"auroc_{side}"] is None else x[f"auroc_{side}"] for x in ep], "o-", color=col,
                    label=f"ball from {side}")
        a2.axhline(0.5, color="grey", lw=0.5)
        a2.set_xlabel("epochs of post-training", fontsize=8)
        a2.set_ylabel("through vs blocked AUROC", fontsize=8)
        a2.set_title("Blockade", fontsize=9)
        a2.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "epochs.png", dpi=120)
        plt.close(fig)
    bins = res["real"].get("bins") or []
    if bins:
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        mid = [np.mean(b["speed"]) if np.isfinite(b["speed"][1]) else b["speed"][0] for b in bins]
        ax.plot(mid, [b["a_down"] for b in bins], "o-", color="tab:blue", label="downhill (rolling left)")
        ax.plot(mid, [b["a_up"] for b in bins], "o-", color="tab:red", label="uphill (rolling right)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("speed (ball diameters / step)", fontsize=8)
        ax.set_ylabel("acceleration along motion (diam. / step²)", fontsize=8)
        ax.set_title("Real ball, whole video: gap = 2 x slope term", fontsize=8)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "real_acc_vs_speed.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    p.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    p.add_argument("--track", type=Path, default=REPO / "data" / "interim" / "whole" / "track.csv")
    args = p.parse_args(argv)
    from robust_world.scene import load_scene
    scene = load_scene(args.scene)
    dist = plank_distance(scene)
    rng = np.random.default_rng(0)

    rows, fps = [], None
    if args.track.exists():
        from robust_world import track as trk
        rows = trk.load(args.track)
        if len(rows) > 1:
            fps = 1 / max(rows[1]["t"] - rows[0]["t"], 1e-6)
    cal = fit_radius(rows, dist) if rows else {"ok": False, "n": 0}
    table = Table(cal)
    real = {"s": None}
    if rows and fps:
        samples, note = real_samples(rows, fps, table, dist, scene["hand_min_px"])
        real = {**real_slope(samples, rng), **note}

    runs = {}
    for name in ["both-before"] + [r for _, r in EPOCH_RUNS[1:]] + OTHER_RUNS:
        f = args.plan / "scores" / name / "per_clip.json"
        if f.exists():
            rr = json.loads(f.read_text())
            if any("true_pos" in r for r in rr):
                runs[name] = rr
    if "both-before" not in runs and not rows:
        return 0
    base = {r["clip_id"]: r for r in runs.get("both-before", [])}

    def per(rows_: list[dict], key: str) -> list[dict]:
        out = []
        for r in rows_:
            b = base.get(r["clip_id"], r)
            x = clip_accel(b, r, key, table, dist)
            if x is not None:
                out.append(x)
        return out

    tracker = per(list(base.values()), "true_pos")
    edges = speed_edges(np.array([x["speed"] for x in tracker if x["side"] == "R"]),
                        np.array([x["speed"] for x in tracker if x["side"] == "L"]))
    clips = {"tracker": model_slope(tracker, edges, rng),
             "decoder_real": model_slope(per(list(base.values()), "real_pos"), edges, rng)}
    by_run = {n: model_slope(per(rr, "imagined_pos"), edges, rng) for n, rr in runs.items()}
    epochs = [{"epoch": e, "run": n, "model": by_run[n], "auroc_R": auroc_side(runs[n], "R"),
               "auroc_L": auroc_side(runs[n], "L")} for e, n in EPOCH_RUNS if n in runs]
    res = {"calibration": cal, "real": real, "clips": clips, "runs": by_run, "epochs": epochs,
           "entry_speed_edges": [float(e) for e in edges]}

    out = args.plan / "slope_test"
    out.mkdir(parents=True, exist_ok=True)
    f5 = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    unit = "ball diameters" if cal.get("ok") else "ball diameters (no radius calibration: pixels / 16)"
    md = ["## Slope test: uphill vs downhill at matched speed, in ball sizes", "",
          f"s = (downhill - uphill acceleration along the motion) / 2 at matched speed, in {unit} per step² "
          "(1 step = 1/8 s); friction cancels. " + ("Positions are mapped onto the table with the ball's size (camera "
          "calibration from real frames only). " if cal.get("ok") else "No tracker file here, so no camera calibration "
          "and no whole-video measure. ") + "See models/vjepa2/slope_test.py.", ""]
    if cal.get("ok"):
        md.append(f"Calibration: radius = {cal['coef'][0]:.4f} x + {cal['coef'][1]:.4f} y + {cal['coef'][2]:.2f} px "
                  f"(R² {cal['r2']}, {cal['n']} frames; radius {cal['radius_range_px']} px across the table).")
    md += ["", "| measure | s | 95% CI | downhill a | uphill a | n down | n up |", "|---|---|---|---|---|---|---|"]
    if real.get("s") is not None:
        b = real["bins"]
        md.append(f"| real ball, whole video ({real.get('stretches')} stretches) | {f5(real['s'])} | {real['ci']} | "
                  f"{f5(np.mean([x['a_down'] for x in b]))} | {f5(np.mean([x['a_up'] for x in b]))} | "
                  f"{real['n_down']} | {real['n_up']} |")
    for label, r in (("real ball, through passes (tracker)", clips["tracker"]),
                     ("decoder on real frames, through passes", clips["decoder_real"]),
                     *((f"imagined: {n}", r) for n, r in by_run.items())):
        md.append(f"| {label} | {f5(r['s'])} | {r.get('ci')} | {r.get('a_down')} | {r.get('a_up')} | "
                  f"{r['n_down']} | {r['n_up']} |")
    if epochs:
        md += ["", "### Learning curve (C: both directions, loss-weighted sampling)", "",
               "| epoch | s | 95% CI | n down / up | AUROC ball from R | AUROC ball from L |", "|---|---|---|---|---|---|"]
        for x in epochs:
            m = x["model"]
            md.append(f"| {x['epoch']} | {f5(m['s'])} | {m.get('ci')} | {m['n_down']} / {m['n_up']} | "
                      f"{x['auroc_R']} | {x['auroc_L']} |")
    md += ["", "Model passes count only target steps where the imagined ball is clearly visible (map peak "
           f">= {MIN_PEAK}) beyond the plank, so n shows how often the model draws the ball at all. Figures: "
           "`slope_test/epochs.png`, `slope_test/real_acc_vs_speed.png`."]
    (out / "slope_test.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "slope_test.md").write_text("\n".join(md) + "\n")
    figures(res, out)
    print(f"-> {out}/ (slope_test.md, epochs.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

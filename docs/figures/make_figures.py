"""Redraw the README figures from saved results (no model is run and nothing is fitted).

    python docs/figures/make_figures.py --clips data/processed/clips/whole

Reads scores, crossing and slope-map JSON from results/2026-10-04_2110_twoday/vjepa2/plan and, for the frame
figures, four clips (whole_0000, whole_0007, whole_0206 and the gripper videos). Writes PNG and GIF files next to
this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLAN = REPO / "results" / "2026-10-04_2110_twoday" / "vjepa2" / "plan"
SCENE = json.loads((REPO / "configs" / "scenes" / "whole.json").read_text())
PLANK = np.asarray(SCENE["occluder_polygon"], float)

PRE, TWO, FINAL = "both-before", "lossmix_both-after", "lossmix_topup-after"
NAMES = {PRE: "Pretrained V-JEPA 2",
         TWO: "Post-trained, both directions",
         FINAL: "Post-trained, both directions + open table"}
COL = {"real": "#1a1a1a", PRE: "#8c8c8c", TWO: "#e08a2c", FINAL: "#c0392b", "ref": "#2c7fb8"}
GAPS = [(104, 186), (315, 367)]          # through gaps along the plank (px), from the real outcomes

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 150, "font.family": "DejaVu Sans"})


def save(fig, name: str) -> None:
    fig.savefig(HERE / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data: the three outcomes
def fig_outcomes(clips: Path) -> None:
    rows = {r["clip_id"]: r for r in json.loads((PLAN / "scores" / PRE / "per_clip.json").read_text())}
    picks = [("whole_0206", "Through"), ("whole_0000", "Bounce"), ("whole_0007", "Hidden (blocked)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (cid, title) in zip(axes, picks):
        frames = iio.imread(clips / "right" / f"{cid}.mp4")
        r = rows[cid]
        ax.imshow(frames[23])
        ctx = np.array([p for p in r["true_pos_context"] if p], float)
        fut = np.array([p for p in r["true_pos"] if p], float)
        ax.plot(ctx[:, 0], ctx[:, 1], "-", color="white", lw=2.5, label="context (seen)")
        if len(fut):
            ax.plot(fut[:, 0], fut[:, 1], "--", color="#ffd23f", lw=2.5, label="target (to predict)")
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
    axes[0].legend(loc="lower left", framealpha=0.85)
    save(fig, "data_outcomes.png")


# ---------------------------------------------------------------- rollouts: real vs imagined
def _heat(ax, m) -> None:
    """Probe map as a colour overlay that is transparent where the probe sees no ball."""
    m = np.clip(np.asarray(m, float), 0, 1)
    rgba = np.zeros(m.shape + (4,))
    rgba[..., :3] = (0.0, 0.85, 1.0)                 # cyan: stands out on wood and on grey
    rgba[..., 3] = np.clip(m * 1.3, 0, 0.92)
    ax.imshow(rgba, extent=(0, 512, 512, 0), interpolation="bilinear")


def _grey(img) -> np.ndarray:
    """Greyscale, slightly dimmed, so the cyan probe map is the only colour in a prediction panel."""
    g = (np.asarray(img, float) @ [0.299, 0.587, 0.114]) * 0.8
    return np.repeat(g[..., None], 3, axis=2).astype(np.uint8)


def _background(frames) -> np.ndarray:
    """The empty scene (median over the clip, so the moving ball drops out)."""
    return np.median(frames, axis=0).astype(np.uint8)


def _maps(run: str):
    return np.load(PLAN / "scores" / run / "maps.npz")


def fig_rollout(clips: Path, cid: str, side: str, title: str, name: str) -> None:
    real = _maps(PRE)[f"{cid}/real"].astype(float)
    rows = [("Real future\n(probe on real frames)", real, True),
            (NAMES[PRE], _maps(PRE)[f"{cid}/imagined"].astype(float), False),
            ("Post-trained\n(final model)", _maps(FINAL)[f"{cid}/imagined"].astype(float), False)]
    frames = iio.imread(clips / side / f"{cid}.mp4")
    cs = len(frames) // 2
    bg = _background(frames)
    steps = [1, 3, 5, 7, 11]
    fig, axes = plt.subplots(3, len(steps) + 1, figsize=(2.35 * (len(steps) + 1), 7.4),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for r, (lab, m, is_real) in enumerate(rows):
        for c in range(len(steps) + 1):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.7")
            if c == 0:
                ax.imshow(frames[cs - 1])
                if r == 0:
                    ax.set_title("Last seen frame", fontsize=12)
                ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=12, fontweight="bold",
                              labelpad=10)
                continue
            k = steps[c - 1]
            ax.imshow(_grey(frames[min(cs + 2 * k, len(frames) - 1)] if is_real else bg))
            _heat(ax, m[k])
            if r == 0:
                ax.set_title(f"+{(k + 1) / 8:.2f} s", fontsize=12)
    # row labels sit left of the first image column; free that column for rows 2 and 3
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
    save(fig, name)


def gif_rollout(clips: Path, cid: str, side: str, name: str) -> None:
    real = _maps(PRE)[f"{cid}/real"].astype(float)
    pre = _maps(PRE)[f"{cid}/imagined"].astype(float)
    post = _maps(FINAL)[f"{cid}/imagined"].astype(float)
    frames = iio.imread(clips / side / f"{cid}.mp4")
    cs = len(frames) // 2
    bg = _background(frames)
    out = []
    for k in [-1] + list(range(12)):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.1, wspace=0.05)
        for ax, lab, m in zip(axes, ["Real future", NAMES[PRE], "Post-trained (final)"], [real, pre, post]):
            ax.axis("off")
            ax.set_title(lab, fontsize=13, fontweight="bold")
            if k < 0:
                ax.imshow(frames[cs - 1])
                continue
            ax.imshow(_grey(frames[min(cs + 2 * k, len(frames) - 1)] if m is real else bg))
            _heat(ax, m[k])
        fig.text(0.5, 0.025, "last seen frame" if k < 0 else f"prediction +{(k + 1) / 8:.2f} s",
                 ha="center", fontsize=13)
        fig.canvas.draw()
        im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        out.append(im.resize((im.width * 4 // 5, im.height * 4 // 5), Image.LANCZOS).quantize(colors=64))
        plt.close(fig)
    dur = [1200] + [330] * 11 + [1500]
    out[0].save(HERE / name, save_all=True, append_images=out[1:], duration=dur, loop=0, optimize=True)
    print("wrote", name)


# ---------------------------------------------------------------- P(blocked) along the plank
def _binned(x, p, edges, rng, n_boot=500):
    cx, m, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = p[(x >= a) & (x < b)]
        if len(sel) < 4:
            continue
        boots = [rng.choice(sel, len(sel)).mean() for _ in range(n_boot)]
        cx.append((a + b) / 2)
        m.append(sel.mean())
        lo.append(np.percentile(boots, 2.5))
        hi.append(np.percentile(boots, 97.5))
    return map(np.array, (cx, m, lo, hi))


def fig_pblocked() -> None:
    runs = {r: json.loads((PLAN / "blocker" / r / "crossing.json").read_text()) for r in (PRE, FINAL)}
    length = runs[FINAL]["plank_length_px"]
    edges = np.linspace(60, 440, 12)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for a, b in GAPS:
        ax.axvspan(a, b, color="#2ca25f", alpha=0.10, lw=0)
    ax.text(np.mean(GAPS[0]), 1.06, "gap", ha="center", color="#2ca25f", fontsize=11)
    ax.text(np.mean(GAPS[1]), 1.06, "gap", ha="center", color="#2ca25f", fontsize=11)
    ax.text(250, 1.06, "hidden blocker", ha="center", color="0.35", fontsize=11)
    series = [("true", runs[FINAL], "Real outcome", COL["real"], 2.8),
              ("imagined", runs[PRE], NAMES[PRE], COL[PRE], 2.0),
              ("imagined", runs[FINAL], "Post-trained (final model)",
               COL[FINAL], 2.4)]
    for key, run, lab, col, lw in series:
        x = np.array([r["entry_along"] for r in run["passes"]])
        p = np.array([r[key] for r in run["passes"]], float)
        cx, m, lo, hi = _binned(x, p, edges, rng)
        ax.fill_between(cx, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(cx, m, "-o", color=col, lw=lw, ms=5, label=lab)
    ax.set_xlim(60, min(440, length))
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("Where the ball reaches the plank (px along the plank)")
    ax.set_ylabel("P(blocked)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=3)
    ax.set_title("Predicted vs real blocking along the plank", fontsize=13, fontweight="bold")
    save(fig, "p_blocked.png")


# ---------------------------------------------------------------- AUROC summary
def fig_auroc() -> None:
    overall = [("Real future frames\n(probe ceiling)", 0.998, 0.995, 1.0, COL["real"]),
               ("Ball tracker + blockade\nfitted on outcomes (reference)", 0.848, 0.797, 0.894, COL["ref"]),
               (NAMES[PRE], 0.454, 0.389, 0.516, COL[PRE]),
               ("Post-trained,\nboth directions", 0.912, 0.881, 0.941, COL[TWO]),
               ("Post-trained, both\ndirections + open table", 0.925, 0.895, 0.954, COL[FINAL])]
    side = {"from the right": [(0.437, 0.368, 0.511), (0.961, 0.941, 0.98), (0.971, 0.952, 0.987)],
            "from the left": [(0.660, 0.541, 0.777), (0.893, 0.817, 0.961), (0.873, 0.784, 0.949)]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    y = np.arange(len(overall))[::-1]
    for yi, (lab, v, lo, hi, col) in zip(y, overall):
        a.barh(yi, v, color=col, height=0.62)
        a.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color="k", capsize=4, lw=1.2)
        a.text(hi + 0.015, yi, f"{v:.2f}", va="center", fontsize=11)
    a.set_yticks(y, [o[0] for o in overall])
    a.axvline(0.5, color="0.5", ls=":", lw=1.2)
    a.set_xlim(0, 1.1)
    a.set_xlabel("Through-vs-blocked AUROC (all 337 passes)")
    a.set_title("(a) Learning the hidden blockade", loc="left")
    w = 0.26
    models = [(NAMES[PRE], COL[PRE]), ("Post-trained, both directions", COL[TWO]),
              ("Post-trained, + open table", COL[FINAL])]
    for i, (lab, col) in enumerate(models):
        for j, (s, vals) in enumerate(side.items()):
            v, lo, hi = vals[i]
            b.bar(j + (i - 1) * w, v, w * 0.92, color=col, label=lab if j == 0 else None)
            b.errorbar(j + (i - 1) * w, v, yerr=[[v - lo], [hi - v]], color="k", capsize=3, lw=1.1)
    b.set_xticks([0, 1], ["Ball from the right", "Ball from the left"])
    b.axhline(0.5, color="0.5", ls=":", lw=1.2)
    b.set_ylim(0, 1.05)
    b.set_ylabel("AUROC (held-out folds)")
    b.set_title("(b) Both entry directions", loc="left")
    fig.legend(*b.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.06), frameon=False,
               ncol=3)
    save(fig, "auroc.png")


# ---------------------------------------------------------------- slope map, pooled regions
REGIONS = {"left of plank": lambda cx, cy: cx <= 1,
           "right, upper": lambda cx, cy: cx >= 3 and cy <= 1,
           "right, middle": lambda cx, cy: cx >= 3 and 2 <= cy <= 3,
           "right, lower": lambda cx, cy: cx >= 3 and cy >= 4}
BOXES = {"left of plank": (0, 0, 192, 192), "right, upper": (288, 0, 192, 192),
         "right, middle": (288, 192, 192, 192), "right, lower": (288, 384, 192, 128)}


def _pool(cells: dict) -> dict:
    out = {}
    for name, inside in REGIONS.items():
        g, n = np.zeros(2), 0
        for key, v in cells.items():
            cx, cy = map(int, key.split(","))
            if inside(cx, cy):
                g += v["n"] * np.asarray(v["g"])
                n += v["n"]
        if n >= 12:
            out[name] = (g / n, n)
    return out


def fig_slope() -> None:
    d = json.loads((PLAN / "slope_map" / "slope_map.json").read_text())
    same = d["same_steps"]
    panels = [("Real ball\n(tracker)", "real ball (tracker)", same["probe on real frames (sharp)"]["s_real"],
               same["probe on real frames (sharp)"]["s_real_ci"]),
              ("Probe on real frames\n(measurement ceiling)", "probe on real frames (sharp)",
               same["probe on real frames (sharp)"]["s_track"], same["probe on real frames (sharp)"]["s_track_ci"])]
    g_key = next(k for k in d if k.startswith("imagined, after (G"))
    gs = next(v for k, v in same.items() if k.startswith("imagined, after (G"))
    panels.append(("Imagined ball\n(post-trained, final model)", g_key, gs["s_track"], gs["s_track_ci"]))
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.05, vmax=0.05)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), gridspec_kw={"wspace": 0.06})
    for ax, (title, key, s, ci) in zip(axes, panels):
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal")
        ax.fill(PLANK[:, 0], PLANK[:, 1], color="#b9a27f", zorder=1)
        ax.text(PLANK[:, 0].mean() + 10, 490, "plank", ha="center", fontsize=11, color="0.3")
        for name, (g, n) in _pool(d[key]["cells"]).items():
            x0, y0, w, h = BOXES[name]
            ax.add_patch(plt.Rectangle((x0 + 3, y0 + 3), w - 6, h - 6, color=plt.cm.RdBu(norm(g[0])), alpha=0.85,
                                       zorder=0, lw=0))
            cx, cy = x0 + w / 2, y0 + h / 2
            sc = 1800
            gx = np.clip(g[0], -0.045, 0.045)
            ax.annotate("", (cx + sc * gx, cy), (cx, cy), zorder=3,
                        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "k", "mutation_scale": 18})
            ax.text(x0 + 10, y0 + h - 12, f"n = {n}", fontsize=10, color="0.15", zorder=2)
        ax.set_title(f"{title}\ns = {s:.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("0.6")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015)
    cb.set_label("left-right pull (ball diameters / step²)\nred = downhill to the left", fontsize=11)
    fig.suptitle("Table slope recovered from rolling: real ball vs the model's imagined ball", fontsize=14,
                 fontweight="bold", y=1.02)
    save(fig, "slope_map.png")


# ---------------------------------------------------------------- gripper GIFs (3 forecasts x 3 scenarios)
def gifs_gripper() -> None:
    vid = PLAN / "gripper" / "pb_narrow_fs" / "page" / "vid"
    out = HERE / "gripper"
    out.mkdir(exist_ok=True)
    for scen in ("blocked", "uncertain", "confident"):
        frames = iio.imread(vid / f"{scen}.mp4")
        w = (frames.shape[2] - 8) // 3
        for i, src in enumerate(("no_posttraining", "posttrained", "perfect")):
            x0 = i * (w + 4)
            ims = [Image.fromarray(f[:, x0:x0 + w]).quantize(colors=128) for f in frames]
            dur = [125] * (len(ims) - 1) + [1500]
            ims[0].save(out / f"{scen}_{src}.gif", save_all=True, append_images=ims[1:], duration=dur, loop=0,
                        optimize=True)
            print("wrote", f"gripper/{scen}_{src}.gif")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=REPO / "data" / "processed" / "clips" / "whole")
    a = ap.parse_args()
    fig_outcomes(a.clips)
    fig_rollout(a.clips, "whole_0206", "right", "Through pass: the post-trained model predicts the ball emerging",
                "rollout_through.png")
    fig_rollout(a.clips, "whole_0007", "right", "Blocked pass: the post-trained model keeps the ball hidden",
                "rollout_blocked.png")
    gif_rollout(a.clips, "whole_0206", "right", "rollout_through.gif")
    gif_rollout(a.clips, "whole_0007", "right", "rollout_blocked.gif")
    fig_pblocked()
    fig_auroc()
    fig_slope()
    gifs_gripper()


if __name__ == "__main__":
    main()

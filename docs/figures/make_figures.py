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

# hidden blocker as placed (Fin, 16 Oct): centred on the seam between the plank's two boards, about half
# the plank's width, along the plank between its middle and far dark markers
BLOCKER = np.array([[175, 265], [232, 265], [220, 360], [164, 360]], float)

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


# ---------------------------------------------------------------- data: the three outcomes, and a data GIF
from matplotlib import patheffects as pe  # noqa: E402

FONT = str(Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf")
OUTLINE = [pe.Stroke(linewidth=5, foreground="black", alpha=0.55), pe.Normal()]


def _blocker(ax, label: bool = True) -> None:
    """Dashed outline of the hidden blocker (never visible in the video)."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(BLOCKER, closed=True, fill=False, ec="#ff2d2d", lw=2.2, ls=(0, (4, 2))))
    if label:
        ax.text(BLOCKER[:, 0].min() - 8, BLOCKER[:, 1].mean(), "hidden\nblocker", ha="right", va="center",
                fontsize=10, color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#b00000")])


def _blocker_pil(im: Image.Image, scale: float) -> Image.Image:
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    pts = [tuple(q) for q in (BLOCKER * scale)]
    for a, b in zip(pts, pts[1:] + pts[:1]):          # dashed edges
        n = int(np.hypot(b[0] - a[0], b[1] - a[1]) // 6) + 1
        for k in range(0, n, 2):
            t0, t1 = k / n, (k + 1) / n
            d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                    (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=(255, 45, 45), width=3)
    return im


def fig_outcomes(clips: Path) -> None:
    rows = {r["clip_id"]: r for r in json.loads((PLAN / "scores" / PRE / "per_clip.json").read_text())}
    picks = [("whole_0206", "Through", "the ball emerges on the far side"),
             ("whole_0000", "Bounce", "the ball rebounds off the hidden blocker"),
             ("whole_0061", "Hidden", "the ball stops under the plank")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), gridspec_kw={"wspace": 0.04})
    for ax, (cid, title, sub) in zip(axes, picks):
        frames = iio.imread(clips / "right" / f"{cid}.mp4")
        r = rows[cid]
        ax.imshow(frames[23][40:470, 60:512], extent=(60, 512, 470, 40))
        ctx = np.array([q for q in r["true_pos_context"] if q], float)
        fut = np.array([q for q in r["true_pos"] if q], float)
        ax.plot(ctx[:, 0], ctx[:, 1], "-", color="white", lw=3, path_effects=OUTLINE)
        ax.plot(*ctx[0], "o", color="white", ms=8, path_effects=OUTLINE)
        _blocker(ax, label=title == "Bounce")
        if title == "Hidden":            # never seen again: the ball runs under the plank and stops there
            end = ctx[-1] + 2.2 * (ctx[-1] - ctx[-2])
            ax.annotate("", end, ctx[-1], arrowprops={"arrowstyle": "-|>", "color": "#ffd23f", "lw": 3,
                                                      "ls": (0, (2, 1.5)), "mutation_scale": 22})
            ax.text(end[0] + 22, end[1] - 26, "unseen", ha="left", fontsize=11, color="#ffd23f",
                    fontweight="bold", path_effects=OUTLINE)
        elif len(fut):
            ax.plot(fut[:, 0], fut[:, 1], "-", color="#ffd23f", lw=3, path_effects=OUTLINE)
            ax.annotate("", fut[-1], fut[max(len(fut) - 3, 0)], arrowprops={"arrowstyle": "-|>", "color": "#ffd23f",
                                                                        "lw": 3, "mutation_scale": 22})
        ax.set_xlim(60, 512)
        ax.set_ylim(470, 40)
        ax.set_title(title, fontweight="bold", fontsize=15)
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=12)
        ax.axis("off")
    handles = [plt.Line2D([], [], color="white", lw=3, path_effects=OUTLINE, label="context: observed (first 1.5 s)"),
               plt.Line2D([], [], color="#ffd23f", lw=3, path_effects=OUTLINE, label="target: to be predicted (next 1.5 s)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=12)
    save(fig, "data_outcomes.png")


def _caption(img: Image.Image, text: str, sub: str = "", h: int = 58) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    out = Image.new("RGB", (img.width, img.height + h), (24, 28, 34))
    out.paste(img, (0, h))
    d = ImageDraw.Draw(out)
    d.text((12, 7), text, fill=(255, 255, 255), font=ImageFont.truetype(FONT, 21))
    if sub:
        d.text((12, 33), sub, fill=(255, 210, 63), font=ImageFont.truetype(FONT, 16))
    return out


def _clean_orange(path: Path) -> list[np.ndarray]:
    """Four ball-only stills of the orange-ball recording from its check sheet, with the sheet's grid and
    labels painted out."""
    import cv2
    sheet = cv2.imread(str(path))
    out = []
    for idx in (0, 3, 4, 7):
        r, c = divmod(idx, 4)
        t = sheet[r * 512:(r + 1) * 512, c * 512:(c + 1) * 512].copy()
        m = np.zeros(t.shape[:2], np.uint8)
        for k in range(0, 513, 32):
            m[:, max(k - 1, 0):k + 2] = 255
            m[max(k - 1, 0):k + 2, :] = 255
        f = cv2.inpaint(t, m, 5, cv2.INPAINT_TELEA)[20:470, 30:508]
        out.append(cv2.resize(f, (360, 360))[..., ::-1])
    return out


def gif_data(clips: Path, orange: Path | None) -> None:
    parts = [("right", "whole_0206", "Ball from the right", "outcome: through"),
             ("right", "whole_0000", "Ball from the right", "outcome: bounce (blocked)"),
             ("right", "whole_0061", "Ball from the right", "outcome: hidden (blocked)"),
             ("left", "whole_0290", "Ball from the left", "outcome: through")]
    frames, dur = [], []
    for side, cid, title, sub in parts:
        v = iio.imread(clips / side / f"{cid}.mp4")
        for i in range(0, len(v), 3):
            im = _blocker_pil(Image.fromarray(v[i]).resize((360, 360), Image.LANCZOS), 360 / 512)
            frames.append(_caption(im, title, sub))
            dur.append(190)
        dur[-1] = 900
    if orange is not None and orange.exists():
        for im in _clean_orange(orange):
            frames.append(_caption(Image.fromarray(im), "Orange ball (held out)",
                                   "never seen in training; stills"))
            dur.append(700)
    frames = [f.quantize(colors=160, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(HERE / "data.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    print("wrote data.gif")


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
                _blocker(ax)
                continue
            ax.imshow(_grey(frames[min(cs + 2 * k, len(frames) - 1)] if m is real else bg))
            _heat(ax, m[k])
            _blocker(ax)
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
    fig, ax = plt.subplots(figsize=(13, 5.2))
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


# ---------------------------------------------------------------- slope map, kernel-smoothed
CELL = 96          # slope_map.py cell size (px)
SIGMA = 130        # smoothing kernel (px); the same for every panel


def _smooth(cells: dict, grid: np.ndarray):
    """Stretch-weighted Gaussian smoothing of the per-cell pulls: g(p) = sum n_i K(p, c_i) g_i / sum n_i K(p, c_i).
    Returns the smoothed x-pull and the effective number of stretches behind each point."""
    c = np.array([[(int(k.split(",")[0]) + 0.5) * CELL, (int(k.split(",")[1]) + 0.5) * CELL] for k in cells])
    n = np.array([v["n"] for v in cells.values()], float)
    g = np.array([v["g"][0] for v in cells.values()], float)
    k = np.exp(-((grid[..., None, :] - c) ** 2).sum(-1) / (2 * SIGMA ** 2)) * n
    w = k.sum(-1)
    return (k * g).sum(-1) / np.maximum(w, 1e-9), w


def fig_slope() -> None:
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.path import Path as MPath
    d = json.loads((PLAN / "slope_map" / "slope_map.json").read_text())
    same = d["same_steps"]
    real = same["probe on real frames (sharp)"]
    g_key = next(k for k in d if k.startswith("imagined, after (G"))
    gs = next(v for k, v in same.items() if k.startswith("imagined, after (G"))
    panels = [("Real ball (tracker)", "real ball (tracker)", real["s_real"], real["s_real_ci"]),
              ("Probe on real frames", "probe on real frames (sharp)", real["s_track"], real["s_track_ci"]),
              ("Predicted ball (final model)", g_key, gs["s_track"], gs["s_track_ci"])]
    xs = np.linspace(0, 512, 129)
    grid = np.stack(np.meshgrid(xs, xs), -1)
    plank = MPath(PLANK).contains_points(grid.reshape(-1, 2), radius=-8).reshape(grid.shape[:2])
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.05, vmax=0.05)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), gridspec_kw={"wspace": 0.05})
    for ax, (title, key, s, ci) in zip(axes, panels):
        gx, w = _smooth(d[key]["cells"], grid)
        img = np.ma.masked_where((w < 20) | plank, gx)
        ax.imshow(img, cmap="RdBu", norm=norm, extent=(0, 512, 512, 0), interpolation="bilinear")
        ax.fill(PLANK[:, 0], PLANK[:, 1], color="#b9a27f", zorder=2)
        ax.text(PLANK[:, 0].mean() + 10, 495, "plank", ha="center", fontsize=11, color="0.3")
        ag = np.stack(np.meshgrid(np.arange(48, 512, 80), np.arange(48, 512, 80)), -1).astype(float)
        ax_, aw = _smooth(d[key]["cells"], ag)
        inside = MPath(PLANK).contains_points(ag.reshape(-1, 2), radius=12).reshape(ag.shape[:2])
        ok = (aw >= 40) & ~inside
        ax.quiver(ag[..., 0][ok], ag[..., 1][ok], np.clip(ax_[ok], -0.05, 0.05), np.zeros(ok.sum()),
                  angles="xy", scale_units="xy", scale=1 / 900, width=0.008, color="k", zorder=3)
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\ns = {s:.3f}  (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("0.6")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015)
    cb.set_label("left-right pull (ball diameters / step²)\nred = downhill, to the left", fontsize=11)
    save(fig, "slope_map.png")


# ---------------------------------------------------------------- post-training variants
def fig_variants() -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 0.8, 1.1],
                                                                         "wspace": 0.55})
    v = [("Pretrained", 0.422, COL[PRE]), ("Multi-hypothesis", 0.423, "#bdbdbd"), ("Latent L1", 0.884, "#9ecae1"),
         ("Copy gate", 0.885, "#9ecae1"), ("Discrete codes", 0.911, "#6baed6"), ("Motion-weighted\n+ contrast", 0.934,
                                                                                "#2171b5")]
    y = np.arange(len(v))[::-1]
    a.barh(y, [x[1] for x in v], color=[x[2] for x in v], height=0.65)
    for yi, (_, val, _) in zip(y, v):
        a.text(val + 0.015, yi, f"{val:.2f}", va="center", fontsize=11)
    a.set_yticks(y, [x[0] for x in v])
    a.set_xlim(0, 1.1)
    a.axvline(0.5, color="0.5", ls=":", lw=1.2)
    a.set_xlabel("AUROC (10 epochs, right-entry)")
    a.set_title("(a) Objective", loc="left")
    sch = [("10\nepochs", 0.934), ("+ loss-\nweighted\nsampling", 0.949), ("+ 20\nepochs", 0.964)]
    b.bar(range(3), [x[1] for x in sch], color=["#6baed6", "#3182bd", "#08519c"], width=0.65)
    for i, (_, val) in enumerate(sch):
        b.text(i, val + 0.004, f"{val:.3f}", ha="center", fontsize=11)
    b.set_xticks(range(3), [x[0] for x in sch], fontsize=10)
    b.set_ylim(0.9, 0.98)
    b.set_ylabel("AUROC (right-entry)")
    b.set_title("(b) Schedule", loc="left")
    ep = [0, 1, 2, 4, 7, 10, 15, 20]
    c.plot(ep, [0.437, 0.401, 0.603, 0.796, 0.888, 0.914, 0.962, 0.961], "-o", color=COL[TWO], lw=2.2,
           label="ball from the right")
    c.plot(ep, [0.660, 0.523, 0.526, 0.593, 0.731, 0.850, 0.891, 0.893], "--s", color=COL[TWO], lw=2.2,
           label="ball from the left", alpha=0.75)
    c.axhline(0.5, color="0.5", ls=":", lw=1.2)
    c.set_xlabel("epoch")
    c.set_ylabel("AUROC")
    c.set_ylim(0.35, 1.0)
    c.legend(frameon=False, loc="lower right")
    c.set_title("(c) Learning curve, both directions", loc="left")
    save(fig, "variants.png")


# ---------------------------------------------------------------- gripper: what uncertainty looks like
def fig_uncertainty() -> None:
    vid = PLAN / "gripper" / "pb_narrow_fs" / "page" / "vid"
    data = json.loads((vid.parent / "src" / "data.json").read_text())
    unc, conf = iio.imread(vid / "uncertain.mp4")[0], iio.imread(vid / "confident.mp4")[0]
    w = (unc.shape[1] - 8) // 3
    crop = lambda f, i: f[95:250, i * (w + 4) + 150:i * (w + 4) + w - 90]   # noqa: E731
    sc = {s["name"]: s["sources"] for s in data["scenarios"]}
    panels = [(crop(unc, 0), "No post-training", sc["uncertain"]["before"]),
              (crop(unc, 1), "Post-trained, uncertain pass", sc["uncertain"]["after (C)"]),
              (crop(conf, 1), "Post-trained, confident pass", sc["confident"]["after (C)"]),
              (crop(unc, 2), "Perfect forecast", sc["uncertain"]["perfect"])]
    fig = plt.figure(figsize=(13, 7.0))
    gs_ = fig.add_gridspec(2, 4, height_ratios=[1, 1.3], hspace=0.42, wspace=0.05)
    for i, (im, title, src) in enumerate(panels):
        ax = fig.add_subplot(gs_[0, i])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"{title}\nP(emerges) {src['p']:.2f}, u = {src['u']:.2f}", fontsize=11.5)
        sig = src["forecast"].get("sig_s", 0.0) if src["u"] > 0 else 0.0
        ax.text(0.5, -0.06, f"exit radius {100 * sig:.1f} cm\ntiming ±{src['forecast'].get('sig_t', 0) if src['u'] > 0 else 0:.2f} s",
                transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax = fig.add_subplot(gs_[1, 1:3])
    us = sorted(float(k) for k in data["sim_pickup_by_u"])
    ax.plot(us, [data["sim_pickup_by_u"][str(u) if str(u) in data["sim_pickup_by_u"] else f"{u:.1f}"] for u in us],
            "-o", color="#e08a2c", lw=2.4, label="frozen policy in simulation")
    for title, src, col in [("no post-training", sc["uncertain"]["before"], COL[PRE]),
                            ("post-trained, uncertain", sc["uncertain"]["after (C)"], COL[TWO]),
                            ("post-trained, confident", sc["confident"]["after (C)"], COL[FINAL]),
                            ("perfect", sc["uncertain"]["perfect"], COL["real"])]:
        ax.axvline(src["u"], color=col, ls="--", lw=1.6, label=f"{title} (u = {src['u']:.2f})")
    ax.set_xlabel("forecast uncertainty u (0 = exact, 1 = no information)")
    ax.set_ylabel("simulated pick-up rate")
    ax.set_ylim(0.35, 0.78)
    ax.set_title("Pick-up rate falls as the forecast becomes less certain", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10.5)
    save(fig, "gripper_uncertainty.png")


# ---------------------------------------------------------------- gripper GIFs (3 forecasts x 3 scenarios)
def _footer(img: Image.Image) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    out = Image.new("RGB", (img.width, img.height + 30), (24, 28, 34))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.ellipse((10, img.height + 8, 24, img.height + 22), fill=(255, 140, 0))
    d.text((32, img.height + 6), "forecast exit region (larger = less certain)", fill=(230, 230, 230),
           font=ImageFont.truetype(FONT, 15))
    return out


def gifs_gripper() -> None:
    vid = PLAN / "gripper" / "pb_narrow_fs" / "page" / "vid"
    out = HERE / "gripper"
    out.mkdir(exist_ok=True)
    for scen in ("blocked", "uncertain", "confident"):
        frames = iio.imread(vid / f"{scen}.mp4")
        w = (frames.shape[2] - 8) // 3
        for i, src in enumerate(("no_posttraining", "posttrained", "perfect")):
            x0 = i * (w + 4)
            ims = [_footer(Image.fromarray(f[:, x0:x0 + w])).quantize(colors=128) for f in frames]
            dur = [125] * (len(ims) - 1) + [1500]
            ims[0].save(out / f"{scen}_{src}.gif", save_all=True, append_images=ims[1:], duration=dur, loop=0,
                        optimize=True)
            print("wrote", f"gripper/{scen}_{src}.gif")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=REPO / "data" / "processed" / "clips" / "whole")
    ap.add_argument("--orange", type=Path, default=None,
                    help="frames.jpg check sheet of the orange-ball recording (results/other_ball_check/)")
    a = ap.parse_args()
    fig_outcomes(a.clips)
    gif_data(a.clips, a.orange)
    fig_variants()
    fig_uncertainty()
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

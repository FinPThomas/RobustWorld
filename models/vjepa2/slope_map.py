"""Map of the table slope, read from ball tracks: real ball vs the world model's imagined ball (Fin, 2026-10-11).

A ball rolling on a tilted table speeds up in the downhill direction wherever it is, and friction slows it along its
own motion. So every short stretch of track gives a ~ g(cell) - mu * v/|v|: g is the slope's pull in the table
cell the ball is in, mu the rolling friction (one number per track source). Both are fitted by least squares on the
track alone, so the map needs nothing but positions over time:

    tracker   the real ball (tracker positions, context and target steps of every clip): the reference
    decoder   the frozen eval_decoder on real frames (target steps): what the measuring instrument can see
    imagined  the same frozen decoder on each run's imagined target steps (before and after post-training)

Positions are put on the table with the ball-size calibration of slope_test.py (camera geometry from real frames
only), in ball diameters; one step = 1/8 s. Each stretch is a quadratic fit over 5 consecutive steps with the ball on
open table (decoded: map peak >= 0.2). Decoded tracks use the sharper nearest-real-patch positions (nn_locate.py,
"sharp") where a run was scored with them, else the decoder's cell-level positions ("coarse"). Nothing here is fitted between a model's output and a score: g and mu are
a physical summary of each track source on its own, and the real ball's map is only shown next to it (CLAUDE.md
rules 1-3, 5-6). Uniform-slope g (one vector for the whole table) gets a 95% interval by bootstrap over clips.

Outputs: <plan>/slope_map/slope_map.png, slope_map.json, slope_map.md

    python models/vjepa2/slope_map.py --plan outputs/vjepa2/plan
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

from slope_test import MIN_PEAK, Table, open_table, plank_distance  # noqa: E402

WIN = 5                          # steps per local quadratic fit
MIN_SPEED = 0.05                 # ball diameters / step: slower stretches have no clear friction direction
CELL_PX = 96                     # map cells in image px
MIN_CELL = 12                    # stretches needed for a cell's arrow
N_BOOT = 300
RUNS = [("both-before", "imagined, before post-training"),
        ("lossmix_both-after", "imagined, after post-training (C, 20 ep)"),
        ("lossmix_open-after", "imagined, after (E: + open clips)"),
        ("lossmix_both_e30-after", "imagined, after (F, 30 ep)")]


def stretches(pos: list, table: Table, dist, clip: int) -> list[dict]:
    """Quadratic fits over every WIN consecutive usable steps: image position, table velocity and acceleration."""
    ok = [p is not None and (len(p) < 3 or p[2] >= MIN_PEAK) and open_table(p[0], p[1], dist) for p in pos]
    T = np.arange(WIN) - (WIN - 1) / 2
    out = []
    for t in range(len(pos) - WIN + 1):
        if not all(ok[t:t + WIN]):
            continue
        px = np.array([pos[k][:2] for k in range(t, t + WIN)], float)
        c = np.polyfit(T, table(px), 2)
        v = c[1]
        if np.linalg.norm(v) > MIN_SPEED:
            out.append({"clip": clip, "xy": px[WIN // 2], "v": v, "a": 2 * c[0]})
    return out


def cell_of(xy: np.ndarray) -> np.ndarray:
    return (np.asarray(xy) // CELL_PX).astype(int)


def fit(samples: list[dict], per_cell: bool) -> dict | None:
    """Least squares a = g (per cell, or one for the table) - mu * v_hat; returns g per cell and mu."""
    if len(samples) < 2 * MIN_CELL:
        return None
    a = np.array([s["a"] for s in samples])
    vh = np.array([s["v"] / np.linalg.norm(s["v"]) for s in samples])
    keys = [tuple(cell_of(s["xy"])) if per_cell else (0, 0) for s in samples]
    cells = sorted(set(keys))
    idx = {c: i for i, c in enumerate(cells)}
    n, k = len(samples), len(cells)
    X = np.zeros((2 * n, 2 * k + 1))
    for i, c in enumerate(keys):
        X[2 * i, 2 * idx[c]] = 1
        X[2 * i + 1, 2 * idx[c] + 1] = 1
        X[2 * i:2 * i + 2, -1] = -vh[i]
    sol = np.linalg.lstsq(X, a.reshape(-1), rcond=None)[0]
    count = {c: keys.count(c) for c in cells}
    return {"mu": float(sol[-1]), "n": n,
            "cells": {c: {"g": sol[2 * idx[c]:2 * idx[c] + 2].tolist(), "n": count[c]} for c in cells}}


def uniform(samples: list[dict], rng) -> dict | None:
    res = fit(samples, per_cell=False)
    if res is None:
        return None
    g = res["cells"][(0, 0)]["g"]
    ids = np.unique([s["clip"] for s in samples])
    by = {i: [s for s in samples if s["clip"] == i] for i in ids}
    boots = []
    for _ in range(N_BOOT):
        r = fit([s for i in rng.choice(ids, len(ids)) for s in by[i]], per_cell=False)
        if r is not None:
            boots.append(r["cells"][(0, 0)]["g"])
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5], axis=0).T.round(4).tolist() if len(boots) > 20 else None
    return {"g": [round(x, 4) for x in g], "g_ci": ci, "mu": round(res["mu"], 4), "n": res["n"],
            "n_clips": int(len(ids))}


def gather(rows: list[dict], key: str, table: Table, dist, ids: dict, context: bool = False) -> list[dict]:
    out = []
    for r in rows:
        pos = list(r.get(key) or [])
        if context:
            pos = list(r.get("true_pos_context") or []) + pos
        out += stretches(pos, table, dist, ids[r["clip_id"]])
    return out


def draw(panels: list[tuple[str, dict | None, dict | None]], poly: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9))
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.08, vmax=0.08)
    for ax, (title, cellmap, uni) in zip(np.atleast_1d(axes), panels):
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal")
        ax.fill(poly[:, 0], poly[:, 1], color="0.8", zorder=0)
        ax.text(poly[:, 0].mean(), 470, "plank", ha="center", fontsize=6, color="0.4")
        if cellmap:
            for c, v in cellmap["cells"].items():
                if v["n"] < MIN_CELL:
                    continue
                x0, y0 = c[0] * CELL_PX, c[1] * CELL_PX
                gx, gy = v["g"]
                ax.add_patch(plt.Rectangle((x0, y0), CELL_PX, CELL_PX, color=plt.cm.RdBu(norm(gx)), alpha=0.75,
                                           zorder=1))
                cx, cy = x0 + CELL_PX / 2, y0 + CELL_PX / 2
                sc = 600                       # px per (ball diameter / step²)
                ax.annotate("", (cx + sc * gx, cy + sc * gy), (cx, cy), zorder=3,
                            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "k"})
                ax.text(x0 + 3, y0 + CELL_PX - 4, str(v["n"]), fontsize=5, color="0.25", zorder=2)
        sub = "no usable stretches"
        if uni:
            sub = f"whole table g_x = {uni['g'][0]:+.3f}"
            if uni["g_ci"]:
                sub += f" [{uni['g_ci'][0][0]:+.3f}, {uni['g_ci'][0][1]:+.3f}]"
            sub += f"\n{uni['n']} stretches, {uni['n_clips']} clips"
        ax.set_title(f"{title}\n{sub}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("slope pull left(-) / right(+), ball diameters / step²", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("Slope map: acceleration the ball gets from the table (friction removed). "
                 "Arrows = pull; number = stretches", fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    p.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = p.parse_args(argv)
    cal_file = args.plan / "slope_test" / "slope_test.json"
    if not cal_file.exists():
        print("slope_map: needs slope_test/slope_test.json (the camera calibration); run slope_test.py first")
        return 0
    table = Table(json.loads(cal_file.read_text())["calibration"])
    scene = json.loads(args.scene.read_text())
    dist = plank_distance(scene)
    rng = np.random.default_rng(0)
    runs = {}
    for name, _ in RUNS:
        for f in ("per_clip.json", "per_clip_open.json"):
            q = args.plan / "scores" / name / f
            if q.exists():
                runs.setdefault(name, []).extend(json.loads(q.read_text()))
    if "both-before" not in runs:
        print("slope_map: no both-before scores yet")
        return 0
    base = {r["clip_id"]: r for r in runs["both-before"]}
    ids = {c: i for i, c in enumerate(sorted(base))}
    def sharp(rows: list[dict], key: str) -> str:      # nearest-real-patch positions (nn_locate.py) when scored
        return key + "_nn" if any(r.get(key + "_nn") for r in rows) else key

    rows0 = list(base.values())
    sources = [("real ball (tracker)", gather(rows0, "true_pos", table, dist, ids, context=True)),
               ("probe on real frames (coarse)", gather(rows0, "real_pos", table, dist, ids))]
    if sharp(rows0, "real_pos") != "real_pos":
        sources.append(("probe on real frames (sharp)", gather(rows0, "real_pos_nn", table, dist, ids)))
    for n, lab in RUNS:
        if n in runs:
            k = sharp(runs[n], "imagined_pos")
            sources.append((lab + (" (sharp)" if k.endswith("_nn") else ""), gather(runs[n], k, table, dist, ids)))
    res, panels = {}, []
    for lab, s in sources:
        cm, uni = fit(s, per_cell=True), uniform(s, rng)
        panels.append((lab, cm, uni))
        res[lab] = {"uniform": uni, "cells": None if cm is None else
                    {f"{c[0]},{c[1]}": {"g": [round(x, 4) for x in v["g"]], "n": v["n"]}
                     for c, v in cm["cells"].items()}}
    out = args.plan / "slope_map"
    out.mkdir(parents=True, exist_ok=True)
    draw(panels, np.array(scene["occluder_polygon"], float), out / "slope_map.png")
    (out / "slope_map.json").write_text(json.dumps(res, indent=1))
    md = ["## Slope map (uniform pull over the whole table)", "",
          "g = acceleration from the table after removing friction (a = g - mu * direction of motion), ball "
          "diameters per step²; x is left/right in the image (negative = pulled left = downhill), y is towards the "
          "camera. 95% intervals by bootstrap over clips. Figure: `slope_map/slope_map.png`.", "",
          "| track | g_x | 95% CI | g_y | 95% CI | friction mu | stretches | clips |", "|---|---|---|---|---|---|---|---|"]
    for lab, r in res.items():
        u = r["uniform"]
        if u is None:
            md.append(f"| {lab} | - | - | - | - | - | 0 | 0 |")
            continue
        ci = u["g_ci"] or [None, None]
        md.append(f"| {lab} | {u['g'][0]} | {ci[0]} | {u['g'][1]} | {ci[1]} | {u['mu']} | {u['n']} | {u['n_clips']} |")
    (out / "slope_map.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

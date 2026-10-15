"""Real vs imagined ball tracks for a few clips, and how often the imagined ball is clear per step (Fin, 2026-10-13).

Per clip: the real ball (tracker) through context and target, the frozen probe on real target frames, and the
probe on each run's imagined target steps (marker opacity = the probe map's peak, so a fading ball fades). Plank
clips (one per outcome and side) and open-table clips (downhill and uphill). Reads per_clip.json /
per_clip_open.json only; nothing is fitted.

Outputs: <plan>/track_viz/tracks.png, visibility.png

    python models/vjepa2/track_viz.py --plan outputs/vjepa2/plan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = [("both-before", "before post-training", "tab:gray"), ("lossmix_both-after", "after: C", "tab:orange"),
        ("lossmix_open-after", "after: E (+ open clips)", "tab:red"),
        ("lossmix_both_e30-after", "after: F (30 ep)", "tab:purple"),
        ("lossmix_topup-after", "after: G (C + aligned open clips)", "tab:brown")]


def load(plan: Path, f: str) -> dict:
    return {n: {r["clip_id"]: r for r in json.loads((plan / "scores" / n / f).read_text())}
            for n, *_ in RUNS if (plan / "scores" / n / f).exists()}


def pos(r: dict, key: str) -> list:
    p = r.get(key + "_nn") if any(r.get(key + "_nn") or []) else r.get(key)
    return p or []


def pick(runs: dict, open_set: bool) -> list[str]:
    """Clips with the most clear imagined steps in the latest after-run: one per (side, outcome), or per
    direction for open-table clips."""
    name = next(n for n, *_ in reversed(RUNS) if n in runs)
    rows = list(runs[name].values())
    score = lambda r: sum(1 for p in r["imagined_pos"] if p and p[2] >= 0.2)  # noqa: E731
    groups = {}
    for r in rows:
        key = r.get("side_in") if open_set else (r.get("side_in"), r["outcome"])
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups, key=str):
        best = sorted(groups[key], key=score, reverse=True)
        out += [r["clip_id"] for r in best[:2 if open_set else 1]]
    return out[:6]


def draw_tracks(plan: Path, scene: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    poly = np.array(scene["occluder_polygon"], float)
    sets = [(load(plan, "per_clip.json"), False), (load(plan, "per_clip_open.json"), True)]
    clips = [(runs, cid, op) for runs, op in sets if runs for cid in pick(runs, op)]
    if not clips:
        return
    cols = 6
    rows_n = int(np.ceil(len(clips) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.6 * cols, 2.9 * rows_n))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (runs, cid, op) in zip(np.atleast_1d(axes).ravel(), clips):
        ax.axis("on")
        base = runs["both-before"][cid] if "both-before" in runs else next(iter(runs.values()))[cid]
        ax.fill(poly[:, 0], poly[:, 1], color="0.85", zorder=0)
        ctx = np.array([c for c in base.get("true_pos_context") or [] if c] or np.zeros((0, 2)))
        tgt = np.array([c for c in base["true_pos"] if c] or np.zeros((0, 2)))
        if len(ctx):
            ax.plot(ctx[:, 0], ctx[:, 1], "o-", color="k", ms=2.5, lw=0.8, label="real, context")
        if len(tgt):
            ax.plot(tgt[:, 0], tgt[:, 1], "o-", color="tab:green", ms=3, lw=1, label="real, future")
        rp = [p for p in pos(base, "real_pos") if p and p[2] >= 0.2]
        if rp:
            ax.scatter([p[0] for p in rp], [p[1] for p in rp], s=10, marker="x", color="tab:blue", lw=0.8,
                       label="probe, real future frames", zorder=3)
        for n, lab, col in RUNS:
            if n not in runs or cid not in runs[n]:
                continue
            ip = [p for p in pos(runs[n][cid], "imagined_pos") if p]
            if not ip:
                continue
            a = np.clip([p[2] for p in ip], 0.05, 1)
            ax.scatter([p[0] for p in ip], [p[1] for p in ip], s=14, color=col, alpha=a, label=f"imagined, {lab}",
                       zorder=4)
            vis = [p for p in ip if p[2] >= 0.2]
            if len(vis) > 1:
                ax.plot([p[0] for p in vis], [p[1] for p in vis], "-", color=col, lw=0.7, alpha=0.6)
        title = f"{cid}\n{'open table' if op else base['outcome']}, from {base.get('side_in')}"
        ax.set_title(title, fontsize=7)
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    h, lab = np.atleast_1d(axes).ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle("Where the ball goes: real vs imagined (probe positions; faint = the imagined ball is fading)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / "tracks.png", dpi=130)
    plt.close(fig)


def draw_visibility(plan: Path, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, f, title in zip(axes, ("per_clip.json", "per_clip_open.json"), ("plank clips", "open-table clips")):
        runs = load(plan, f)
        for n, lab, col in RUNS:
            if n not in runs:
                continue
            rows = list(runs[n].values())
            vis = np.array([[bool(p and p[2] >= 0.2) for p in r["imagined_pos"]] for r in rows], float)
            res[f"{n} {title}"] = vis.mean(0).round(3).tolist()
            ax.plot(np.arange(1, vis.shape[1] + 1) / 8, vis.mean(0), "o-", color=col, ms=3, label=f"imagined, {lab}")
        if "both-before" in runs:
            rows = list(runs["both-before"].values())
            real = np.array([[bool(p and p[2] >= 0.2) for p in r["real_pos"]] for r in rows], float)
            ax.plot(np.arange(1, real.shape[1] + 1) / 8, real.mean(0), "--", color="tab:blue",
                    label="probe on the real frames")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("seconds into the future", fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("share of clips with a clear ball (peak >= 0.2)", fontsize=8)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "visibility.png", dpi=130)
    plt.close(fig)
    return res


def clip_paths() -> dict:
    out = {}
    for man in (REPO / "data" / "processed" / "clips").glob("manifest*.jsonl"):
        for line in man.open():
            c = json.loads(line)
            out[c["clip_id"]] = REPO / c["path"]
    return out


def draw_frames(plan: Path, out: Path) -> list[str]:
    """Per picked clip: the real frames (last context, then future steps) with the probe's map on them, and the
    imagined maps before and after post-training on the same frames (dimmed, for reference). Needs the clips
    (on Kaggle) and maps.npz from ball_probe_cv.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPO / "src"))
    from robust_world.eval.io import read_video
    maps = {n: np.load(plan / "scores" / n / "maps.npz") for n, *_ in RUNS
            if (plan / "scores" / n / "maps.npz").exists()}
    after = [n for n, *_ in RUNS[1:] if n in maps]
    if not after:
        return []
    shown = ([("both-before", "imagined, before")] if "both-before" in maps else []) + \
        [(n, f"imagined, {lab}") for n, lab, _ in RUNS if n == after[-1]]
    real = maps["both-before"] if "both-before" in maps else maps[after[-1]]   # real-frame maps: same in every run
    paths, written = clip_paths(), []
    steps = [0, 2, 4, 6, 8, 11]                      # target steps shown (1/8 s each)
    for f, tag in (("per_clip.json", "plank"), ("per_clip_open.json", "open")):
        runs = load(plan, f)
        if not runs:
            continue
        for cid in pick(runs, tag == "open"):
            if cid not in paths or not paths[cid].exists() or f"{cid}/real" not in real:
                continue
            frames = read_video(paths[cid])
            cs = len(frames) // 2
            fig, axes = plt.subplots(1 + len(shown), len(steps) + 1, figsize=(2.0 * (len(steps) + 1),
                                                                               2.1 * (1 + len(shown))))
            rows = [("real frame + probe", real[f"{cid}/real"], 1.0)] + \
                   [(lab, maps[n][f"{cid}/imagined"], 0.45) for n, lab in shown if f"{cid}/imagined" in maps[n]]
            for r, (lab, m, bright) in enumerate(rows):
                for c in range(len(steps) + 1):
                    ax = axes[r, c]
                    ax.axis("off")
                    if c == 0:
                        ax.imshow(frames[cs - 1])
                        if r == 0:
                            ax.set_title("last context frame", fontsize=7)
                        ax.text(-20, 256, lab, rotation=90, va="center", ha="right", fontsize=7)
                        continue
                    k = steps[c - 1]
                    ax.imshow((frames[min(cs + 2 * k, len(frames) - 1)] * bright).astype(np.uint8))
                    ax.imshow(m[k].astype(float), cmap="magma", alpha=0.55, vmin=0, vmax=1,
                              extent=(0, 512, 512, 0), interpolation="bilinear")
                    ax.text(6, 30, f"peak {float(m[k].max()):.2f}", color="w", fontsize=6)
                    if r == 0:
                        ax.set_title(f"+{(k + 1) / 8:.2f} s", fontsize=7)
            base = runs["both-before"][cid] if "both-before" in runs else next(iter(runs.values()))[cid]
            fig.suptitle(f"{cid}: {base['outcome']}, ball from {base.get('side_in')}. Rows 2+: what the model "
                         "imagines, drawn on the real frame (dimmed) for reference", fontsize=8)
            fig.tight_layout()
            name = f"frames_{tag}_{cid}.png"
            fig.savefig(out / name, dpi=90)
            plt.close(fig)
            written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    p.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = p.parse_args(argv)
    if not (args.plan / "scores" / "both-before" / "per_clip.json").exists():
        print("track_viz: no both-before scores yet")
        return 0
    out = args.plan / "track_viz"
    out.mkdir(parents=True, exist_ok=True)
    draw_tracks(args.plan, json.loads(args.scene.read_text()), out)
    (out / "visibility.json").write_text(json.dumps(draw_visibility(args.plan, out), indent=1))
    draw_frames(args.plan, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

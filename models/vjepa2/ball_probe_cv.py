"""Cross-validated V-JEPA 2 ball probe over every included clip, with per-clip run times.

Same probes as ball_probe.py (linear, calibrated, trained on real and on imagined tokens),
but scored on all clips with k-fold cross-validation, so each clip is read by probes that
never saw it. Answers, across the whole dataset:

  - real features: is the ball found? (sanity)
  - imagined features: is the ball's true future location recoverable?
  - outcome: does V-JEPA's imagined future put the ball beyond the plank more for
    "through" clips than for "hidden" ones? (AUROC of max imagined P(beyond plank))

Also times every stage per clip on this device (encodings are cached, so only clips
encoded in this run are timed).

Outputs (outputs/vjepa2/ball_probe_cv/): curves.png, outcome.png, metrics.json, per_clip.json

    python models/vjepa2/ball_probe_cv.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from ball_probe import CLIP_SIZE, encode, train_probe  # noqa: E402
from robust_world.eval.ball import ball_labels, load_tracking, source_indices  # noqa: E402
from robust_world.eval.io import read_video  # noqa: E402
from robust_world.passes import side_fn  # noqa: E402
from run import MODEL_ID, SIZE, pick_device  # noqa: E402


def plots(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"through": "tab:blue", "hidden": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120, sharey=True)
    for ax, key, title in [(axes[0], "visible", "P(ball visible)"), (axes[1], "far", "P(ball beyond the plank)")]:
        for outcome, col in colours.items():
            grp = [r for r in rows if r["outcome"] == outcome]
            if not grp:
                continue
            t = np.arange(len(grp[0][f"real_{key}"])) * 2 + grp[0]["first_target_frame"]
            for src, style in (("real", "-"), ("imagined", "--")):
                m = np.array([r[f"{src}_{key}"] for r in grp])
                mu, se = m.mean(0), m.std(0) / np.sqrt(len(m))
                ax.plot(t, mu, style, color=col, lw=2, label=f"{outcome} ({len(grp)}): {src}")
                ax.fill_between(t, mu - se, mu + se, color=col, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("clip frame (target half)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8)
    fig.suptitle("V-JEPA 2 ball probe, cross-validated over all clips (mean ± s.e.)")
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    rng = np.random.default_rng(0)
    for i, src in enumerate(("real", "imagined")):
        for outcome, col in colours.items():
            v = [max(r[f"{src}_far"]) for r in rows if r["outcome"] == outcome]
            x = i * 2 + (0 if outcome == "through" else 0.7) + rng.uniform(-0.15, 0.15, len(v))
            ax.scatter(x, v, s=14, color=col, alpha=0.7, label=outcome if i == 0 else None)
    ax.set_xticks([0.35, 2.35])
    ax.set_xticklabels(["real features", "V-JEPA imagined"])
    ax.set_ylabel("max P(ball beyond the plank) over target")
    ax.set_title("Does the imagined future put the ball beyond the plank?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "outcome.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest.jsonl")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2" / "ball_probe_cv")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoVideoProcessor, VJEPA2Model

    device = pick_device()
    t_load = time.perf_counter()
    model = VJEPA2Model.from_pretrained(args.model_id).to(device).eval()
    proc = AutoVideoProcessor.from_pretrained(args.model_id)
    t_load = time.perf_counter() - t_load
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    track, passes, scene = load_tracking()
    side = side_fn(scene["occluder_polygon"])
    cache = REPO / "outputs" / "vjepa2" / "cache" / args.model_id.replace("/", "--")
    cache.mkdir(parents=True, exist_ok=True)

    clips = [json.loads(line) for line in args.manifest.open()]
    clips = [c for c in clips if c.get("include") and c["outcome"] in ("through", "hidden")]
    print(f"V-JEPA 2 ball probe (cross-validated) on {device}: {len(clips)} clips, model load {t_load:.1f}s")

    data, timings = [], []
    for i, c in enumerate(clips):
        path = cache / f"{c['clip_id']}.pt"
        t0 = time.perf_counter()
        frames = read_video(REPO / c["path"])
        t_read = time.perf_counter() - t0
        n_ctx = c["context_frames"][1] + 1
        if path.exists():
            enc = torch.load(path)
        else:
            enc = encode(model, frames, n_ctx, mean, std, device, imagine=True)
            torch.save(enc, path)
            timings.append({"read_video": t_read, **enc["timing"],
                            "total": t_read + sum(enc["timing"].values())})
        lab, _ = ball_labels(source_indices(c, passes, len(frames), n_ctx), track, grid, tub)
        data.append({"clip": c, "enc": enc, "lab": lab, "cs": enc["context"].shape[0]})
        print(f"\r  {i + 1}/{len(clips)} clips ({len(timings)} newly encoded)", end="", flush=True)
    print()

    cell = CLIP_SIZE / grid
    occluder = np.array(scene["occluder_polygon"], np.float32)
    far_cells = np.array([[side((x + 0.5) * cell, (y + 0.5) * cell) == "L" and
                           cv2.pointPolygonTest(occluder, ((x + 0.5) * cell, (y + 0.5) * cell), True) < -cell
                           for x in range(grid)] for y in range(grid)])

    y_out = np.array([d["clip"]["outcome"] == "through" for d in data], int)
    rows = [None] * len(data)
    fut_lab, fut_imag, fut_hold, real_lab, real_map = [], [], [], [], []
    D = model.config.hidden_size
    for fold, (tr, te) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(y_out, y_out)):
        real_probe = train_probe(torch.cat([data[i]["enc"]["real"].reshape(-1, D) for i in tr]).float(),
                                 torch.cat([torch.from_numpy(data[i]["lab"].reshape(-1)).float() for i in tr]),
                                 seed=args.seed)
        imag_probe = train_probe(torch.cat([data[i]["enc"]["imagined"].reshape(-1, D) for i in tr]).float(),
                                 torch.cat([torch.from_numpy(data[i]["lab"][data[i]["cs"]:].reshape(-1)).float()
                                            for i in tr]), seed=args.seed)
        for i in te:
            d = data[i]
            cs = d["cs"]
            m_real = real_probe(d["enc"]["real"]).numpy()
            m_ctx = real_probe(d["enc"]["context"]).numpy()
            m_imag = imag_probe(d["enc"]["imagined"]).numpy()
            real_lab.append(d["lab"].reshape(-1)), real_map.append(m_real.reshape(-1))
            fut_lab.append(d["lab"][cs:].reshape(-1)), fut_imag.append(m_imag.reshape(-1))
            fut_hold.append(np.repeat(m_ctx[-1:], len(m_imag), 0).reshape(-1))
            rows[i] = {"clip_id": d["clip"]["clip_id"], "outcome": d["clip"]["outcome"], "fold": fold,
                       "first_target_frame": cs * tub,
                       "real_visible": m_real[cs:].max((1, 2)).round(3).tolist(),
                       "real_far": (m_real[cs:] * far_cells).max((1, 2)).round(3).tolist(),
                       "imagined_visible": m_imag.max((1, 2)).round(3).tolist(),
                       "imagined_far": (m_imag * far_cells).max((1, 2)).round(3).tolist()}
        print(f"\r  fold {fold + 1}/{args.folds} scored", end="", flush=True)
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    rl, rm = np.concatenate(real_lab), np.concatenate(real_map)
    fl, fi, fh = np.concatenate(fut_lab), np.concatenate(fut_imag), np.concatenate(fut_hold)
    score_imag = np.array([max(r["imagined_far"]) for r in rows])
    score_real = np.array([max(r["real_far"]) for r in rows])
    t_arr = {k: np.array([t[k] for t in timings]) for k in (timings[0] if timings else {})}
    metrics = {
        "model_id": args.model_id, "device": device, "n_clips": len(rows),
        "n_through": int(y_out.sum()), "n_hidden": int(len(y_out) - y_out.sum()), "folds": args.folds,
        "real_cell_auroc": round(float(roc_auc_score(rl, rm)), 3),
        "future_cell_auroc_imagined": round(float(roc_auc_score(fl, fi)), 3),
        "future_cell_auroc_hold_last_context": round(float(roc_auc_score(fl, fh)), 3),
        "outcome_auroc_real": round(float(roc_auc_score(y_out, score_real)), 3),
        "outcome_auroc_imagined": round(float(roc_auc_score(y_out, score_imag)), 3),
        "imagined_far_peak_mean": {o: round(float(score_imag[y_out == (o == "through")].mean()), 3)
                                   for o in ("through", "hidden")},
        "timing_seconds_per_clip": {k: {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
                                        "n": int(len(v))} for k, v in t_arr.items()},
        "model_load_seconds": round(t_load, 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (args.out / "per_clip.json").write_text(json.dumps(rows, indent=1))
    plots(rows, args.out)

    print(f"  real features: ball cell AUROC {metrics['real_cell_auroc']}")
    print(f"  imagined future: ball cell AUROC {metrics['future_cell_auroc_imagined']} "
          f"(hold last context: {metrics['future_cell_auroc_hold_last_context']})")
    print(f"  outcome from max P(beyond plank): real {metrics['outcome_auroc_real']}, "
          f"imagined {metrics['outcome_auroc_imagined']}; imagined peak mean {metrics['imagined_far_peak_mean']}")
    if timings:
        tt = metrics["timing_seconds_per_clip"]
        print("  seconds per clip on " + device + ": " +
              ", ".join(f"{k} {v['mean']:.2f}" for k, v in tt.items()) + f"  (n={tt['total']['n']})")
    print(f"-> {args.out}/ (curves.png, outcome.png, metrics.json, per_clip.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

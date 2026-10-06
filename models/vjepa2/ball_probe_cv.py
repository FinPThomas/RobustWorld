"""Cross-validated V-JEPA 2 ball readout over every included clip, with per-clip run times.

Uses the frozen evaluation decoder (eval_decoder.py), fitted per fold on the training clips'
context halves only, so each clip is read by a decoder that never saw it, and no decoder
ever sees predictions, target-half labels or outcomes. Answers, across the whole dataset:

  - real frames: is the ball found, including beyond the plank? (sanity / upper bound)
  - imagined future: where does V-JEPA put the ball, and does it put it beyond the plank
    more for "through" clips than blocked ones (hidden, or bounce: back out on the near side)?
    Any such difference comes from the world model, since the decoder has no way to know
    about the blockade.

Also times every stage per clip on this device (encodings are cached, so only clips
encoded in this run are timed).

With --predictor-run (checkpoints from posttrain.py), each fold's clips are imagined by the
predictor post-trained on that fold's training clips; the encoder, the cached features and the
decoder are the same as for the pretrained model.

Outputs (outputs/vjepa2/ball_probe_cv/, or .../<run>/ with --predictor-run): curves.png,
outcome.png, metrics.json, per_clip.json

    python models/vjepa2/ball_probe_cv.py
    python models/vjepa2/ball_probe_cv.py --predictor-run checkpoints/vjepa2/l1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
import eval_decoder  # noqa: E402
from ball_probe import encode  # noqa: E402
from posttrain import cache_dir, imagine_mode, load_cached, load_predictor, target_space  # noqa: E402
from robust_world.eval.ball import (OUTCOMES, ball_labels, ball_track_metrics, cv_folds,  # noqa: E402
                                    far_cells, load_tracking, near_cells, one_ball_readouts, outcome_metrics,
                                    roc_auc_score_safe, source_indices, video_of)
from robust_world.eval.io import read_video  # noqa: E402
from run import MODEL_ID, SIZE, pick_device  # noqa: E402


@lru_cache(maxsize=None)
def normalisation(model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """The processor's pixel mean and std (only needed to encode a clip that isn't cached yet)."""
    from transformers import AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(model_id)
    return np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)


def ram_gb() -> tuple[float, float] | None:
    """(free, total) GB of RAM from /proc/meminfo (Linux, e.g. Colab), else None."""
    try:
        info = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
        return info["MemAvailable"] / 2**20, info["MemTotal"] / 2**20
    except (OSError, KeyError, ValueError, IndexError):
        return None


def plots(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"through": "tab:blue", "hidden": "tab:orange", "bounce": "tab:green"}
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
    fig.suptitle("V-JEPA 2 read by the frozen evaluation decoder, cross-validated (mean ± s.e.)")
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    rng = np.random.default_rng(0)
    for i, src in enumerate(("real", "imagined")):
        for outcome, col in colours.items():
            v = [max(r[f"{src}_far"]) for r in rows if r["outcome"] == outcome]
            x = i * 2 + 0.5 * list(colours).index(outcome) + rng.uniform(-0.15, 0.15, len(v))
            ax.scatter(x, v, s=14, color=col, alpha=0.7, label=outcome if i == 0 else None)
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(["real features", "V-JEPA imagined"])
    ax.set_ylabel("max P(ball beyond the plank) over target")
    ax.set_title("Does the imagined future put the ball beyond the plank?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "outcome.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest_right.jsonl")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2" / "ball_probe_cv")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--predictor-run", type=Path, help="posttrain.py run folder with fold{k}.pt checkpoints")
    args = p.parse_args(argv)
    if args.predictor_run and args.out == p.get_default("out"):
        args.out = args.out / args.predictor_run.name

    from sklearn.metrics import roc_auc_score
    from transformers import VJEPA2Model

    device = pick_device()
    t_load = time.perf_counter()
    model = VJEPA2Model.from_pretrained(args.model_id).to(device).eval()
    t_load = time.perf_counter() - t_load
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    cache = cache_dir(args.model_id)
    cache.mkdir(parents=True, exist_ok=True)

    clips = [json.loads(line) for line in args.manifest.open()]
    clips = [c for c in clips if c.get("include") and c["outcome"] in OUTCOMES]
    track, passes, scene = load_tracking(video_of(clips))
    print(f"V-JEPA 2 ball probe (cross-validated) on {device}: {len(clips)} clips, model load {t_load:.1f}s")

    data, timings = [], []
    for i, c in enumerate(clips):
        path = cache / f"{c['clip_id']}.pt"
        t0 = time.perf_counter()
        frames = read_video(REPO / c["path"])
        t_read = time.perf_counter() - t0
        n_ctx = c["context_frames"][1] + 1
        enc = load_cached(cache, c)
        if enc is None:
            enc = encode(model, frames, n_ctx, *normalisation(args.model_id), device, imagine=True)
            torch.save(enc, path)
            timings.append({"read_video": t_read, **enc["timing"],
                            "total": t_read + sum(enc["timing"].values())})
        lab, centres = ball_labels(source_indices(c, passes, len(frames), n_ctx), track, grid, tub)
        # Keep only the context half in memory (what the decoder is fitted on); a clip's full encoding is read
        # back from the cache when it is scored. Holding every clip's real + imagined features as well needs
        # about 25 MB per clip, which ran Colab's ~12 GB of RAM out at 254 clips.
        data.append({"clip": c, "ctx": enc["context"].clone(), "lab": lab, "centres": centres, "cs": enc["context"].shape[0]})
        del enc, frames
        print(f"\r  {i + 1}/{len(clips)} clips ({len(timings)} newly encoded)", end="", flush=True)
    print()
    held = sum(d["ctx"].numel() * d["ctx"].element_size() for d in data) / 2**30
    # Fitting the decoder makes a float32 copy of ~80% of these (2x their float16 size) plus a temporary for its
    # spread, and a fold's test clips are read back one at a time: about 3.5x what is held now, at the peak.
    ram, need = ram_gb(), 3.5 * held
    print(f"context features in memory: {held:.1f} GB; fitting needs about {need:.1f} GB more; RAM: "
          + (f"{ram[0]:.1f} GB free of {ram[1]:.1f} GB" if ram else "unknown"), flush=True)
    if ram and ram[0] < need:
        raise SystemExit(f"not enough RAM to fit the decoder ({ram[0]:.1f} GB free, about {need:.1f} GB needed): "
                         "use a runtime with more RAM (Runtime > Change runtime type > High-RAM) or fewer clips")

    far, near = far_cells(scene, grid), near_cells(scene, grid)

    y_out = np.array([d["clip"]["outcome"] == "through" for d in data], int)
    rows = [None] * len(data)
    fut_lab, fut_imag, fut_hold, fut_alt, real_lab, real_map = [], [], [], [], [], []
    # Far-side cells only (beyond the plank, where no context frame ever had the ball): can the
    # decoder find the ball where it comes out? Read from real target frames (ceiling) and imagined.
    far_lab, far_real, far_imag = [], [], []
    far_mask = np.asarray(far, bool)
    track_maps = {"real": [], "imagined": [], "imagined_encoder_space": []}
    # Alternative to layer-normalising the decoder's input: map predictions into encoder space with
    # the encoder's own final layer norm (gamma * prediction + beta) and read them with the decoder
    # fitted on raw features. Both decoders are fitted on the same real context features only.
    gamma = model.encoder.layernorm.weight.detach().float().cpu()
    beta = model.encoder.layernorm.bias.detach().float().cpu()
    track_centres = []
    D = model.config.hidden_size
    for fold, (tr, te) in enumerate(cv_folds([d["clip"] for d in data], args.folds, args.seed)):
        examples = [eval_decoder.examples_from({"context": data[i]["ctx"]}, data[i]["lab"]) for i in tr]
        decoder = eval_decoder.fit(examples, seed=args.seed)
        decoder_raw = eval_decoder.fit_raw_features(examples, seed=args.seed)
        if args.predictor_run:
            meta = load_predictor(model, args.predictor_run / f"fold{fold}.pt")
            test_ids = {data[i]["clip"]["clip_id"] for i in te}
            if test_ids & set(meta["train_clip_ids"]):
                raise SystemExit(f"fold{fold}.pt was trained on clips this fold scores; "
                                 "train it with the same manifest, --folds and --seed")
        del examples
        for i in te:
            d = data[i]
            cs = d["cs"]
            d["enc"] = load_cached(cache, d["clip"])
            m_real = decoder(d["enc"]["real"]).numpy()
            m_ctx = decoder(d["enc"]["context"]).numpy()
            if args.predictor_run:
                with torch.no_grad():
                    imag = imagine_mode(model.predictor, d["enc"]["context"][None].float().to(device),
                                        d["enc"]["real"].shape[0] - cs, meta)[0].float().cpu()
            else:
                imag = d["enc"]["imagined"]
            m_imag = decoder(imag).numpy()
            m_alt = decoder_raw(target_space(imag.float()) * gamma + beta).numpy()
            fut_alt.append(m_alt.reshape(-1))
            track_maps["imagined_encoder_space"].append(m_alt)
            real_lab.append(d["lab"].reshape(-1)), real_map.append(m_real.reshape(-1))
            fut_lab.append(d["lab"][cs:].reshape(-1)), fut_imag.append(m_imag.reshape(-1))
            far_lab.append(d["lab"][cs:][:, far_mask].reshape(-1))
            far_real.append(m_real[cs:][:, far_mask].reshape(-1)), far_imag.append(m_imag[:, far_mask].reshape(-1))
            fut_hold.append(np.repeat(m_ctx[-1:], len(m_imag), 0).reshape(-1))
            track_maps["real"].append(m_real[cs:]), track_maps["imagined"].append(m_imag)
            track_centres.append(d["centres"][cs:])
            rows[i] = {"clip_id": d["clip"]["clip_id"], "outcome": d["clip"]["outcome"], "fold": fold,
                       "first_target_frame": cs * tub,
                       "real_visible": m_real[cs:].max((1, 2)).round(3).tolist(),
                       "real_far": (m_real[cs:] * far).max((1, 2)).round(3).tolist(),
                       "real_near": (m_real[cs:] * near).max((1, 2)).round(3).tolist(),
                       "imagined_visible": m_imag.max((1, 2)).round(3).tolist(),
                       "imagined_far": (m_imag * far).max((1, 2)).round(3).tolist(),
                       "imagined_near": (m_imag * near).max((1, 2)).round(3).tolist(),
                       **{f"{k}_one_ball_{side}": v for k, m in (("real", m_real[cs:]), ("imagined", m_imag),
                                                                 ("imagined_encoder_space", m_alt))
                          for side, v in one_ball_readouts(m, far, near).items()}}
            del d["enc"]                      # scored: free it
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
        "predictor_run": str(args.predictor_run) if args.predictor_run else "pretrained",
        **{f"n_{o}": sum(r["outcome"] == o for r in rows) for o in OUTCOMES}, "folds": args.folds,
        "real_cell_auroc": round(float(roc_auc_score(rl, rm)), 3),
        "future_cell_auroc_imagined": round(float(roc_auc_score(fl, fi)), 3),
        "future_cell_auroc_hold_last_context": round(float(roc_auc_score(fl, fh)), 3),
        "future_cell_auroc_imagined_encoder_space": round(float(roc_auc_score(fl, np.concatenate(fut_alt))), 3),
        "far_cell_auroc_real": round(float(roc_auc_score_safe(np.concatenate(far_lab), np.concatenate(far_real))), 3),
        "far_cell_auroc_imagined": round(float(roc_auc_score_safe(np.concatenate(far_lab), np.concatenate(far_imag))), 3),
        "outcome_auroc_real": round(float(roc_auc_score(y_out, score_real)), 3),
        "outcome_auroc_imagined": round(float(roc_auc_score(y_out, score_imag)), 3),
        "imagined_far_peak_mean": {o: round(float(score_imag[[r["outcome"] == o for r in rows]].mean()), 3)
                                   for o in OUTCOMES if any(r["outcome"] == o for r in rows)},
        **{f"outcomes_{kind}": outcome_metrics([{"outcome": r["outcome"], "far": r[f"{kind}_far"],
                                                 "near": r[f"{kind}_near"]} for r in rows])
           for kind in ("real", "imagined")},
        **{f"ball_{kind}": ball_track_metrics(track_maps[kind], track_centres) for kind in ("real", "imagined")},
        # Threshold-free, assuming one ball: outcome from the share of the ball beyond / back on the
        # near side, and position from the most likely cell at every step (no 0.5 cut-off).
        **{f"outcomes_{kind}_one_ball": outcome_metrics([{"outcome": r["outcome"], "far": r[f"{kind}_one_ball_far"],
                                                          "near": r[f"{kind}_one_ball_near"]} for r in rows])
           for kind in ("real", "imagined", "imagined_encoder_space")},
        **{f"ball_{kind}_argmax": ball_track_metrics(track_maps[kind], track_centres, confident=0.0)
           for kind in ("real", "imagined", "imagined_encoder_space")},
        "timing_seconds_per_clip": {k: {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
                                        "n": int(len(v))} for k, v in t_arr.items()},
        "model_load_seconds": round(t_load, 1),
    }
    old_metrics = args.out / "metrics.json"
    if not timings and old_metrics.exists():          # everything was cached: keep the last measured timings
        prev = json.loads(old_metrics.read_text())
        metrics["timing_seconds_per_clip"] = prev.get("timing_seconds_per_clip", {})
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (args.out / "per_clip.json").write_text(json.dumps(rows, indent=1))
    plots(rows, args.out)

    print(f"  real features: ball cell AUROC {metrics['real_cell_auroc']}")
    print(f"  imagined future: ball cell AUROC {metrics['future_cell_auroc_imagined']} "
          f"(hold last context: {metrics['future_cell_auroc_hold_last_context']}); far-side cells: "
          f"imagined {metrics['far_cell_auroc_imagined']}, real {metrics['far_cell_auroc_real']}")
    print(f"  outcome from max P(beyond plank): real {metrics['outcome_auroc_real']}, "
          f"imagined {metrics['outcome_auroc_imagined']}; imagined peak mean {metrics['imagined_far_peak_mean']}")
    oi, bi = metrics["outcomes_imagined"], metrics["ball_imagined"]
    print(f"  imagined: P(correct outcome) {oi.get('p_correct')} (balanced {oi.get('p_correct_balanced')}), "
          f"balanced accuracy {oi.get('balanced_accuracy')}; ball hit rate {bi['hit_rate']}, "
          f"error {bi['error_px']} px, phantom rate {bi['phantom_rate']}")
    ob, ab = metrics["outcomes_imagined_one_ball"], metrics["ball_imagined_argmax"]
    oe, ae = metrics["outcomes_imagined_encoder_space_one_ball"], metrics["ball_imagined_encoder_space_argmax"]
    print(f"  threshold-free (one ball): through-vs-blocked AUROC {ob['outcome_auroc']}, P(correct) balanced "
          f"{ob.get('p_correct_balanced')}, argmax ball within 48 px {ab['hit_rate']} (median {ab['error_px']} px)")
    print(f"  encoder-space alternative (gamma*pred+beta, raw decoder): cell AUROC "
          f"{metrics['future_cell_auroc_imagined_encoder_space']}, through-vs-blocked AUROC {oe['outcome_auroc']}, "
          f"argmax ball within 48 px {ae['hit_rate']}")
    if "bounce_auroc" in metrics["outcomes_imagined"]:
        print(f"  bounce vs hidden (ball back on the near side): real {metrics['outcomes_real']['bounce_auroc']}, "
              f"imagined {metrics['outcomes_imagined']['bounce_auroc']}; three-way accuracy imagined "
              f"{metrics['outcomes_imagined']['outcome3_accuracy']}")
    if timings:
        tt = metrics["timing_seconds_per_clip"]
        print("  seconds per clip on " + device + ": " +
              ", ".join(f"{k} {v['mean']:.2f}" for k, v in tt.items()) + f"  (n={tt['total']['n']})")
    print(f"-> {args.out}/ (curves.png, outcome.png, metrics.json, per_clip.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

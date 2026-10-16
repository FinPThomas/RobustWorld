"""The other ball (scope D), scored without training, and where ball colour lives in V-JEPA 2's features.

1. Scores. Every clip of the new-ball video(s) in configs/heldout.json is read with the frozen evaluation
   decoder (generalise.py's: real context halves of the seen clips, same-frame ball positions), for real
   frames, the pretrained predictor, C (lossmix_both-after, the five fold predictors averaged; none was
   trained on this ball) and G (lossmix_topup-after) when its weights are here.

2. Colour direction. One direction in the frozen encoder's feature space: mean token at the ball's cell
   for the new ball minus the same for the training ball, on REAL CONTEXT-HALF frames only. Its labels are
   which ball is in the video, never outcomes, target halves or predictions (CLAUDE.md), and the encoder is
   frozen, so the direction is identical before and after post-training. It is fitted on four folds of
   clips and checked on the fifth (cv_folds):
     colour     AUROC of the projection for new vs training ball on held-out clips (1 = pure colour axis)
     motion     within each ball, correlation of the projection with the ball's x, y and speed
                (about 0 = it carries no position or motion)
     decoder    cosine between the direction and the evaluation decoder's position read-out
   and in imagination: the projection of predicted tokens at the cells where the ball really is (scoring
   only), before vs after post-training: does the model keep the ball's colour in what it imagines?

3. "Ignore the colour". The direction is removed from every context token (its coordinate set to the
   training clips' average) before the predictor imagines, for both balls; the unchanged decoder scores the
   result. If the training ball's scores hold and the new ball's move towards them, the colour sits on its
   own axis and the motion forecast doesn't need it.

    python models/vjepa2/other_ball.py --out outputs/vjepa2/plan/other_ball
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
import generalise  # noqa: E402
from robust_world.eval.ball import (ball_labels, ball_track_metrics, cv_folds, one_ball_readouts,  # noqa: E402
                                    outcome_metrics, roc_auc_score_safe, source_indices)

OUT = REPO / "outputs" / "vjepa2" / "plan" / "other_ball"
RUNS = {"C": "lossmix_both-after", "G": "lossmix_topup-after"}


def ball_rows(enc: dict, lab: np.ndarray, centres: list) -> list[dict]:
    """Context steps with the ball in view: raw and layer-normed mean token at its cells, centre and speed."""
    import torch

    import eval_decoder
    ctx = enc["context"]
    rows = []
    for t in range(ctx.shape[0]):
        cells = np.asarray(lab[t], bool)
        if not cells.any() or centres[t] is None:
            continue
        tok = ctx[t][torch.from_numpy(cells)].float()
        prev = centres[t - 1] if t else None
        speed = float(np.hypot(centres[t][0] - prev[0], centres[t][1] - prev[1])) if prev is not None else None
        rows.append({"raw": tok.mean(0), "ln": eval_decoder.token_space(tok).mean(0),
                     "x": float(centres[t][0]), "y": float(centres[t][1]), "speed": speed})
    return rows


def unit(v):
    return v / (v.norm() + 1e-8)


def direction(rows_new: list, rows_old: list, key: str):
    import torch
    return unit(torch.stack([r[key] for r in rows_new]).mean(0) - torch.stack([r[key] for r in rows_old]).mean(0))


def corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)


def colour_report(per_clip_rows: list[list[dict]], is_new: np.ndarray, decoder) -> tuple[dict, dict]:
    """Cross-validated colour direction checks. -> (report, the directions fitted on every clip)."""
    import torch
    folds = cv_folds([{"outcome": "new" if n else "old"} for n in is_new])
    proj, auc = {k: [] for k in ("new", "old")}, []
    for tr, te in folds:
        d = direction([r for i in tr if is_new[i] for r in per_clip_rows[i]],
                      [r for i in tr if not is_new[i] for r in per_clip_rows[i]], "ln")
        y, s = [], []
        for i in te:
            for r in per_clip_rows[i]:
                p = float(r["ln"] @ d)
                y.append(int(is_new[i]))
                s.append(p)
                proj["new" if is_new[i] else "old"].append({**{k: r[k] for k in ("x", "y", "speed")}, "p": p})
        auc.append(roc_auc_score_safe(np.array(y), np.array(s)))
    d_ln = direction([r for i in np.flatnonzero(is_new) for r in per_clip_rows[i]],
                     [r for i in np.flatnonzero(~is_new) for r in per_clip_rows[i]], "ln")
    d_raw = direction([r for i in np.flatnonzero(is_new) for r in per_clip_rows[i]],
                      [r for i in np.flatnonzero(~is_new) for r in per_clip_rows[i]], "raw")
    w = unit((decoder.linear.weight[0] / decoder.sd).detach())   # the decoder's read-out in layer-normed space
    report = {"colour_auroc_heldout_folds": [round(float(a), 3) for a in auc],
              "colour_auroc_mean": round(float(np.mean(auc)), 3),
              "motion_correlation": {b: {k: corr([r[k] for r in proj[b]], [r["p"] for r in proj[b]])
                                         for k in ("x", "y", "speed")} for b in proj},
              "cos_with_decoder_readout": round(float(d_ln @ w), 3),
              "n_ball_tokens": {"new": len(proj["new"]), "old": len(proj["old"])}}
    return report, {"ln": d_ln, "raw": d_raw, "decoder_readout": w,
                    "projections": proj}


def remove(ctx, d, level: float):
    """Set every token's coordinate along d to `level` (raw encoder space)."""
    return ctx - ((ctx.float() @ d) - level)[..., None].to(ctx.dtype) * d.to(ctx.dtype)


def figure(dirs: dict, scores: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    pr = dirs["projections"]
    ax[0].hist([r["p"] for r in pr["old"]], 40, alpha=.6, label="training ball", color="tab:blue", density=True)
    ax[0].hist([r["p"] for r in pr["new"]], 40, alpha=.6, label="new ball", color="tab:orange", density=True)
    ax[0].set_title("Ball token along the colour direction\n(real context frames, held-out folds)", fontsize=9)
    ax[0].legend(fontsize=8)
    for b, col in (("old", "tab:blue"), ("new", "tab:orange")):
        ok = [r for r in pr[b] if r["speed"] is not None]
        ax[1].scatter([r["speed"] for r in ok], [r["p"] for r in ok], s=4, alpha=.4, color=col)
    ax[1].set_xlabel("ball speed (px per step)", fontsize=8)
    ax[1].set_title("Colour coordinate vs speed (flat = no motion in it)", fontsize=9)
    names = [k for k in scores if scores[k] is not None]
    x = np.arange(len(names))
    for j, (b, col) in enumerate((("old", "tab:blue"), ("new", "tab:orange"))):
        ax[2].bar(x + (j - .5) * .38, [scores[n][b] or 0 for n in names], .36, color=col,
                  label="training ball" if b == "old" else "new ball")
    ax[2].set_xticks(x, names, fontsize=7, rotation=20)
    ax[2].set_ylim(0, 1)
    ax[2].axhline(.5, color="grey", lw=.6)
    ax[2].set_title("Through-vs-blocked AUROC (imagined)", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import VJEPA2Model

    import eval_decoder
    from plan import found_weights
    from posttrain import cache_dir, imagine, imagine_mode, load_cached, load_predictor
    from run import MODEL_ID, SIZE, pick_device

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=generalise.MANIFEST)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--old-scored", type=int, default=80,
                   help="training-ball clips scored for the comparison (spread over the list; all are used for "
                        "the colour direction)")
    args = p.parse_args(argv)

    new_videos = generalise.holdout_videos(None)
    right = generalise.included(args.manifest)
    others = generalise.other_segments(args.manifest)
    seen, groups, _ = generalise.split(right, new_videos, others)
    new = [c for g, cs in groups.items() if g.startswith("new ball") for c in cs]
    old = seen + [c for g, cs in groups.items() if not g.startswith("new ball") for c in cs]  # training ball, both sides
    if not new:
        print("no new-ball clips: list the video in configs/heldout.json and add its clips")
        return generalise.WAITING
    device = pick_device()
    model = VJEPA2Model.from_pretrained(MODEL_ID)
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    model.encoder = None
    model.to(device).eval()
    cache = cache_dir(MODEL_ID)

    def item(c):
        enc = load_cached(cache, c)
        if enc is None:
            raise SystemExit(f"{c['clip_id']} is not encoded; run posttrain.py encode on its manifest")
        track, passes, scene = generalise.tracking(generalise.stem(c))
        n_ctx = c["context_frames"][1] + 1
        lab, centres = ball_labels(source_indices(c, passes, c["n_frames"], n_ctx), track, grid, tub)
        return enc, lab, centres, scene

    print(f"== {len(new)} new-ball clips, {len(old)} training-ball clips ({len(seen)} seen)", flush=True)
    examples = []
    for c in seen:
        enc, lab, _, _ = item(c)
        examples.append(eval_decoder.examples_from(enc, lab))
    decoder = eval_decoder.fit(examples, seed=0)
    del examples

    clips = new + old
    is_new = np.array([True] * len(new) + [False] * len(old))
    rows = [ball_rows(*item(c)[:3]) for c in clips]
    report, dirs = colour_report(rows, is_new, decoder)
    print(f"colour direction: AUROC {report['colour_auroc_mean']} on held-out folds; motion correlation "
          f"{report['motion_correlation']}; cos with decoder {report['cos_with_decoder_readout']}", flush=True)
    d_raw = dirs["raw"].to(device)
    # Average coordinate over the training ball's context tokens (every cell, not just the ball's).
    acc, n = 0.0, 0
    for c in old:
        x = item(c)[0]["context"].float().reshape(-1, d_raw.numel())
        acc, n = acc + float((x.to(device) @ d_raw).sum()), n + len(x)
    level = acc / n

    base = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}
    runs = {"pretrained": None}
    for k, name in RUNS.items():
        w = found_weights(name)
        if w is not None:
            runs[k] = [(load_predictor(model, f), {k2: v.detach().clone() for k2, v in model.predictor.state_dict().items()})
                       for f in sorted(w.glob("fold?.pt"))]
        else:
            print(f"== {k} ({name}): weights aren't here; skipped", flush=True)
    model.predictor.load_state_dict(base)

    def predict(run, ctx, steps, c, all_folds):
        """Imagined target for this clip; post-trained runs average the folds not trained on it (new ball), or
        use the one fold that held it out (training ball)."""
        if runs[run] is None:
            model.predictor.load_state_dict(base)
            return imagine(model.predictor, ctx, steps)[0].float().cpu()
        outs = []
        for meta, st in runs[run]:
            if c["clip_id"] in set(meta.get("train_clip_ids", [])):
                continue
            model.predictor.load_state_dict(st)
            outs.append(imagine_mode(model.predictor, ctx, steps, meta)[0].float().cpu())
            if not all_folds:
                break
        return torch.stack(outs).mean(0) if outs else None

    scored = new + [old[j] for j in np.unique(np.linspace(0, len(old) - 1, min(args.old_scored, len(old))).astype(int))]
    per_clip, tables = [], {}
    phases = ["real"] + [f"{r}{s}" for r in runs for s in ("", " (colour removed)")]
    for i, c in enumerate(scored):
        b = "new" if c in new else "old"
        enc, lab, centres, scene = item(c)
        far, near = generalise.sides_for(c, scene, grid)
        cs = enc["context"].shape[0]
        steps = enc["real"].shape[0] - cs
        ctx = enc["context"][None].float().to(device)
        ctx_rm = remove(ctx, d_raw, level)
        maps = {"real": decoder(enc["real"][cs:]).numpy()}
        tgt = torch.from_numpy(np.asarray(lab[cs:], bool))
        colour_kept = {"real": float(eval_decoder.token_space(enc["real"][cs:].float())[tgt]
                                     .mean(0) @ dirs["ln"]) if lab[cs:].any() else None}
        with torch.no_grad():
            for r in runs:
                pred = predict(r, ctx, steps, c, b == "new")
                if pred is None:
                    continue
                maps[r] = decoder(pred).numpy()
                maps[f"{r} (colour removed)"] = decoder(predict(r, ctx_rm, steps, c, b == "new")).numpy()
                if lab[cs:].any():
                    colour_kept[r] = float(eval_decoder.token_space(pred)[tgt].mean(0) @ dirs["ln"])
        rec = {"clip_id": c["clip_id"], "ball": b, "outcome": c["outcome"],
               "colour_at_real_ball_cells": colour_kept}
        for k, m in maps.items():
            ro = one_ball_readouts(m, far, near)
            tables.setdefault((rec["ball"], k), []).append(({"outcome": c["outcome"], **ro}, m, lab[cs:], centres[cs:]))
            rec[f"{k}_far"] = ro["far"]
        per_clip.append(rec)
        print(f"\r  scored {i + 1}/{len(scored)} clips", end="", flush=True)
    print()

    results = {}
    for (b, k), items in tables.items():
        rs, ms, ls, cents = zip(*items)
        fl = np.concatenate([x.reshape(-1) for x in ls])
        results.setdefault(b, {})[k] = {"n": len(rs), **outcome_metrics(list(rs)),
                                        "cell_auroc": round(float(roc_auc_score_safe(fl, np.concatenate(
                                            [m.reshape(-1) for m in ms]))), 3),
                                        "argmax_hit_rate": ball_track_metrics(list(ms), list(cents), 0.0)["hit_rate"]}
    for b in results:
        kept = {}
        for k in ["real", *runs]:
            v = [r["colour_at_real_ball_cells"].get(k) for r in per_clip if r["ball"] == b]
            v = [x for x in v if x is not None]
            kept[k] = round(float(np.mean(v)), 3) if v else None
        results[b]["colour_coordinate_at_real_ball_cells"] = kept
    out = {"new_ball_videos": new_videos, "colour_direction": report, "colour_removed_level": round(level, 4),
           "scores": results}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=1))
    (args.out / "per_clip.json").write_text(json.dumps(per_clip, indent=1))
    torch.save({k: v.cpu() for k, v in dirs.items() if k != "projections"}, args.out / "colour_direction.pt")
    (args.out / "projections.json").write_text(json.dumps(dirs["projections"]))
    auc = {k: {b: results.get(b, {}).get(k, {}).get("outcome_auroc") for b in ("old", "new")}
           for k in phases if any(k in results.get(b, {}) for b in results)}
    figure(dirs, auc, args.out / "colour.png")
    for k, v in auc.items():
        print(f"{k:28s} through-vs-blocked AUROC: training ball {v['old']}, new ball {v['new']}")
    print(f"-> {args.out}/ (metrics.json, per_clip.json, colour_direction.pt, colour.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

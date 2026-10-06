"""What did post-training change, and where is it stored? Label-free reads of one post-trained run.

Nothing here is fitted except the frozen evaluation decoder, built exactly as ball_probe_cv.py
builds it (per fold, real context halves of that fold's training clips, same-frame ball labels,
eval_decoder.examples_from). Outcomes and the scene's plank/blocker outlines are only used to
group, score and draw (CLAUDE.md rule 6).

1. Change maps. For every held-out clip (its own fold's predictor), the imagined future after vs
   before post-training:
     - feature change: mean |after - before| per token cell over the target half (layer-normalised);
     - ball change: the decoder's one-ball P(ball) after minus before, per cell, for through clips
       and for blocked (hidden or bounce) clips.
   Drawn over a context frame with the plank (and blocker, if the scene sets one) outlined, with
   the share of the change that falls on the plank against the plank's share of the frame, and the
   per-cell correlation between change and the plank.
2. Layer patching. The predictor's parameters in groups (embeddings, each transformer layer, the
   output norm + projection, and the added heads if any). "Patch in": the pretrained predictor with
   only that group post-trained. "Revert": the post-trained predictor with only that group put
   back. The through-vs-blocked AUROC (one ball, frozen decoder) of each shows where the blockade
   knowledge sits.

    python models/vjepa2/interpret.py --run checkpoints/vjepa2/plan/codes-after
Outputs (default outputs/vjepa2/plan/interpret/<run>/): change_map.png, layer_patching.png, interpret.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.ball import (CLIP_SIZE, OUTCOMES, ball_labels, cv_folds, far_cells,  # noqa: E402
                                    load_tracking, near_cells, one_ball_readouts, outcome_metrics,
                                    source_indices, video_of)


def group_of(key: str) -> str:
    m = re.match(r"layer\.(\d+)\.", key)
    if m:
        return f"layer {int(m.group(1)):02d}"
    return "embeddings" if key.startswith("embeddings") else "output norm + proj"


def one_ball(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(m.sum(axis=(1, 2), keepdims=True), 1e-12)


def plank_mask(scene: dict, grid: int) -> np.ndarray:
    import cv2
    poly = np.array(scene["occluder_polygon"], np.float32)
    cell = CLIP_SIZE / grid
    return np.array([[cv2.pointPolygonTest(poly, ((x + 0.5) * cell, (y + 0.5) * cell), False) >= 0
                      for x in range(grid)] for y in range(grid)])


def placement(change: np.ndarray, mask: np.ndarray) -> dict:
    """How much of a per-cell change map sits on the plank, and its correlation with the plank."""
    c = np.abs(change)
    share = float(c[mask].sum() / max(c.sum(), 1e-12))
    r = float(np.corrcoef(c.reshape(-1), mask.reshape(-1).astype(float))[0, 1]) if c.std() > 0 else float("nan")
    return {"share_on_plank": round(share, 3), "plank_share_of_frame": round(float(mask.mean()), 3),
            "correlation_with_plank": round(r, 3)}


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import VJEPA2Model

    import eval_decoder
    from posttrain import (MANIFEST, cache_dir, imagine, imagine_mode, load_cached, load_predictor,
                           target_space, training_clips)
    from robust_world.eval.io import read_video
    from run import MODEL_ID, SIZE, pick_device

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="posttrain.py run folder with fold{k}.pt")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out = args.out or REPO / "outputs" / "vjepa2" / "plan" / "interpret" / args.run.name
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    model = VJEPA2Model.from_pretrained(args.model_id)
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    model.encoder = None
    model.to(device).eval()
    base = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}
    groups = sorted({group_of(k) for k in base})
    clips = training_clips(args.manifest)
    track, passes, scene = load_tracking(video_of(clips))
    far, near, plank = far_cells(scene, grid), near_cells(scene, grid), plank_mask(scene, grid)
    cache = cache_dir(args.model_id)

    data = []
    for c in clips:
        enc = load_cached(cache, c)
        n_ctx = c["context_frames"][1] + 1
        lab, _ = ball_labels(source_indices(c, passes, c["n_frames"], n_ctx), track, grid, tub)
        # only the context half (and the clip's length): every clip's full features would not fit in Colab's RAM
        data.append((c, {"context": enc["context"].clone(), "steps": enc["real"].shape[0]}, lab))
        del enc

    feat_change = {o: [] for o in OUTCOMES}
    ball_change = {o: [] for o in OUTCOMES}
    patched = {"patch_in": {g: [] for g in groups}, "revert": {g: [] for g in groups}}
    ends = {"before": [], "after": []}

    def readout(decoder, imag, outcome):
        return {"outcome": outcome, **one_ball_readouts(decoder(imag).numpy(), far, near)}

    for fold, (tr, te) in enumerate(cv_folds(clips, args.folds, args.seed)):
        decoder = eval_decoder.fit([eval_decoder.examples_from(data[i][1], data[i][2]) for i in tr], seed=args.seed)
        meta = load_predictor(model, args.run / f"fold{fold}.pt")
        if {data[i][0]["clip_id"] for i in te} & set(meta["train_clip_ids"]):
            raise SystemExit(f"fold{fold}.pt was trained on clips this fold reads; use the same manifest and folds")
        after = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}
        heads = meta.get("heads")
        if heads is not None and "heads" not in patched["patch_in"]:
            for kind in patched:
                patched[kind]["heads"] = []
        variants = [("patch_in", g, {k: (after if group_of(k) == g else base)[k] for k in base}) for g in groups]
        variants += [("revert", g, {k: (base if group_of(k) == g else after)[k] for k in base}) for g in groups]
        if heads is not None:
            variants += [("patch_in", "heads", base), ("revert", "heads", after)]
        for i in te:
            c, enc, _ = data[i]
            cs = enc["context"].shape[0]
            ctx = enc["context"][None].float().to(device)
            steps = enc["steps"] - cs
            with torch.no_grad():
                model.predictor.load_state_dict(base)
                im_b = imagine(model.predictor, ctx, steps)[0].float().cpu()
                model.predictor.load_state_dict(after)
                im_a = imagine_mode(model.predictor, ctx, steps, meta)[0].float().cpu()
            feat_change[c["outcome"]].append((target_space(im_a) - target_space(im_b)).abs().mean(-1).mean(0).numpy())
            ball_change[c["outcome"]].append((one_ball(decoder(im_a).numpy()) - one_ball(decoder(im_b).numpy())).mean(0))
            ends["before"].append(readout(decoder, im_b, c["outcome"]))
            ends["after"].append(readout(decoder, im_a, c["outcome"]))
        for kind, g, state in variants:
            model.predictor.load_state_dict(state)
            # the heads go with the post-trained side: patched in only for "heads", reverted only for "heads"
            use_heads = heads is not None and ((kind == "patch_in") == (g == "heads"))
            mode = {**meta, "heads": heads if use_heads else None}
            for i in te:
                c, enc, _ = data[i]
                cs = enc["context"].shape[0]
                with torch.no_grad():
                    im = imagine_mode(model.predictor, enc["context"][None].float().to(device),
                                      enc["steps"] - cs, mode)[0].float().cpu()
                patched[kind][g].append(readout(decoder, im, c["outcome"]))
        print(f"\r  fold {fold + 1}/{args.folds} read", end="", flush=True)
    print()

    auroc = lambda rows: outcome_metrics(rows)["outcome_auroc"]  # noqa: E731
    present = [o for o in OUTCOMES if feat_change[o]]
    blocked = [o for o in present if o != "through"]
    fc_all = np.mean([m for o in present for m in feat_change[o]], 0)
    bc_through = np.mean(ball_change["through"], 0) if ball_change["through"] else np.zeros((grid, grid))
    bc_blocked = (np.mean([m for o in blocked for m in ball_change[o]], 0) if blocked else np.zeros((grid, grid)))
    result = {
        "run": str(args.run), "auroc_before": auroc(ends["before"]), "auroc_after": auroc(ends["after"]),
        "feature_change": placement(fc_all, plank),
        "ball_change_through": placement(bc_through, plank), "ball_change_blocked": placement(bc_blocked, plank),
        "ball_beyond_plank_change": {o: round(float(np.mean([(m * far).sum() for m in ball_change[o]])), 4)
                                     for o in present},
        "patching": {kind: {g: auroc(rows) for g, rows in by.items()} for kind, by in patched.items()},
    }
    (out / "interpret.json").write_text(json.dumps(result, indent=1))

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c0 = data[0][0]
    frame = read_video(REPO / c0["path"])[c0["context_frames"][1]]
    poly = np.array(scene["occluder_polygon"] + scene["occluder_polygon"][:1])
    blocker = scene.get("blocker_polygon") or c0.get("blocker_polygon")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=120)
    lim = max(np.abs(bc_through).max(), np.abs(bc_blocked).max(), 1e-6)
    panels = [(fc_all, "feature change |after - before|", "magma", None),
              (bc_through, "P(ball) after - before: through clips", "RdBu_r", lim),
              (bc_blocked, "P(ball) after - before: blocked clips", "RdBu_r", lim)]
    for ax, (m, title, cmap, v) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame, cmap="gray",
                  extent=(0, CLIP_SIZE, CLIP_SIZE, 0))
        im = ax.imshow(m, cmap=cmap, alpha=0.6, extent=(0, CLIP_SIZE, CLIP_SIZE, 0),
                       vmin=-v if v else None, vmax=v)
        ax.plot(poly[:, 0], poly[:, 1], color="white", lw=1.5)
        if blocker:
            b = np.array(blocker + blocker[:1])
            ax.plot(b[:, 0], b[:, 1], color="#f1c40f", lw=1.5, ls="--")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{args.run.name}: what post-training changed in the imagined future (plank outlined"
                 + (", blocker dashed" if blocker else "") + ")", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "change_map.png")
    plt.close(fig)

    names = list(result["patching"]["patch_in"])
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(names) + 2), 4), dpi=120)
    x = np.arange(len(names))
    for kind, colour, label in (("patch_in", "#c0392b", "pretrained + only this group post-trained"),
                                ("revert", "#2c3e50", "post-trained with only this group reverted")):
        ax.plot(x, [result["patching"][kind][g] for g in names], "o-", color=colour, lw=2, ms=6, label=label)
    ax.axhline(result["auroc_before"], color="#9aa5b1", ls="--", lw=1, label="pretrained")
    ax.axhline(result["auroc_after"], color="#c0392b", ls="--", lw=1, label="post-trained")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("through vs blocked AUROC (one ball)")
    ax.set_title(f"{args.run.name}: where is the blockade stored?", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "layer_patching.png")
    plt.close(fig)
    print(f"  AUROC before {result['auroc_before']} -> after {result['auroc_after']}; feature change on plank "
          f"{result['feature_change']['share_on_plank']} (plank is {result['feature_change']['plank_share_of_frame']} "
          f"of the frame)")
    print(f"-> {out}/ (change_map.png, layer_patching.png, interpret.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

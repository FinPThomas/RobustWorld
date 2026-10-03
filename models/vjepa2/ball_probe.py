"""Where does V-JEPA 2 imagine the ball? Read with the frozen evaluation decoder.

1. Label: the tracker knows where the ball is in every frame, so each 32x32 px token cell
   at each 2-frame step is labelled "ball here" or not.
2. Decoder: eval_decoder.fit, the only readout allowed on V-JEPA features. Linear, fitted on
   training clips' context halves only (encoded alone, same-frame labels). It never sees
   predictions, target-half labels or outcomes, so it cannot learn the blockade; that is
   for post-training the world model, and this decoder stays the same before and after.
3. Check: on the held-out eval clips, does it find the ball in real frames (incl. beyond
   the plank, which it never saw during fitting)?
4. Read the imagination: the same decoder on the predictor's imagined target features.
   Reference: the last context step's ball map held still.

Outputs (outputs/vjepa2/ball_probe/):
    <clip>.mp4        frame | real ball map | context/imagined ball map; green = real ball,
                      red = imagined ball (argmax cell, when the decoder is confident)
    summary.png       per clip, over the target: P(ball visible) and P(ball beyond the plank),
                      real vs imagined
    metrics.json      decoder quality on held-out real frames and per-clip readouts

    python models/vjepa2/ball_probe.py --train-clips 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.io import read_video  # noqa: E402
from robust_world.media import write_video  # noqa: E402
from robust_world.eval.ball import ball_labels, load_tracking, source_indices  # noqa: E402
from robust_world.passes import side_fn  # noqa: E402
import eval_decoder  # noqa: E402
from run import MODEL_ID, SIZE, pick_device, to_pixels  # noqa: E402

CLIP_SIZE = 512
PANEL = 256
CONFIDENT = 0.5


def cached_encode(cache: Path, clip_id: str, *args, **kwargs) -> dict:
    path = cache / f"{clip_id}.pt"
    if path.exists():
        return torch.load(path)
    enc = encode(*args, **kwargs)
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(enc, path)
    return enc


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def encode(model, frames, n_ctx, mean, std, device, imagine: bool) -> dict:
    """Real (and optionally context-only + imagined) token grids, plus seconds spent per stage."""
    import time
    cfg = model.config
    g = SIZE // cfg.patch_size
    per_step = g * g
    ctx_steps, all_steps = n_ctx // cfg.tubelet_size, len(frames) // cfg.tubelet_size
    timing = {}
    _sync(device)
    t = time.perf_counter()
    full = model(to_pixels(frames, mean, std, device), skip_predictor=True).last_hidden_state
    _sync(device)
    timing["encode_full"], t = time.perf_counter() - t, time.perf_counter()
    out = {"real": full.reshape(all_steps, g, g, -1).half().cpu()}
    if imagine:
        ctx = model(to_pixels(frames[:n_ctx], mean, std, device), skip_predictor=True).last_hidden_state
        _sync(device)
        timing["encode_context"], t = time.perf_counter() - t, time.perf_counter()
        pred = model.predictor(
            encoder_hidden_states=ctx,
            context_mask=[torch.arange(ctx_steps * per_step, device=device).unsqueeze(0)],
            target_mask=[torch.arange(ctx_steps * per_step, all_steps * per_step, device=device).unsqueeze(0)],
        ).last_hidden_state
        _sync(device)
        timing["predict"] = time.perf_counter() - t
        out["context"] = ctx.reshape(ctx_steps, g, g, -1).half().cpu()
        out["imagined"] = pred.reshape(all_steps - ctx_steps, g, g, -1).half().cpu()
    out["timing"] = timing
    return out


def heat(p: np.ndarray) -> np.ndarray:
    u8 = (np.clip(p, 0, 1) * 255).astype(np.uint8)
    im = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def argmax_cell(p: np.ndarray):
    if p.max() < CONFIDENT:
        return None
    y, x = np.unravel_index(p.argmax(), p.shape)
    c = PANEL / p.shape[0]
    return int((x + 0.5) * c), int((y + 0.5) * c)


def put(img, text, org=(6, 18), colour=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def clip_video(clip, frames, maps_real, maps_seen, centres, ctx_steps, tubelet) -> list[np.ndarray]:
    out = []
    for k, frame in enumerate(frames):
        s = min(k // tubelet, len(maps_real) - 1)
        target = s >= ctx_steps
        f = cv2.resize(frame, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
        if centres[s] is not None:
            cx, cy = centres[s]
            cv2.circle(f, (int(cx * PANEL / CLIP_SIZE), int(cy * PANEL / CLIP_SIZE)), 11, (60, 220, 60), 2, cv2.LINE_AA)
        guess = argmax_cell(maps_seen[s])
        if target and guess:
            cv2.drawMarker(f, guess, (240, 60, 60), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        a, b = heat(maps_real[s]), heat(maps_seen[s])
        put(f, "frame")
        put(a, "ball map: real")
        put(b, "ball map: IMAGINED" if target else "ball map: context (given)",
            colour=(255, 210, 120) if target else (255, 255, 255))
        row = np.hstack([f, a, b])
        bar = np.zeros((26, row.shape[1], 3), np.uint8)
        put(bar, f"{clip['clip_id']}  true: {clip['outcome']}  frame {k}  "
                 f"{'TARGET (imagined)' if target else 'CONTEXT'}", (6, 18),
            (255, 210, 120) if target else (200, 200, 200))
        out.append(np.vstack([bar, row]))
    return out


def summary_plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(rows), figsize=(3.2 * len(rows), 5.2), dpi=120, sharey=True, squeeze=False)
    for j, r in enumerate(rows):
        t = np.arange(len(r["real_visible"])) * 2 + r["first_target_frame"]
        for i, (key, title) in enumerate([("visible", "P(ball visible)"), ("far", "P(ball beyond\nthe plank)")]):
            ax = axes[i, j]
            ax.plot(t, r[f"real_{key}"], "-o", color="tab:green", ms=3, label="real")
            ax.plot(t, r[f"imagined_{key}"], "--x", color="tab:red", ms=4, label="V-JEPA imagined")
            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(f"{r['clip_id']}\ntrue: {r['outcome']}", fontsize=9)
            if j == 0:
                ax.set_ylabel(title)
            if i == 1:
                ax.set_xlabel("clip frame")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Frozen evaluation decoder on V-JEPA 2 features: real vs imagined target", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, default=REPO / "data" / "eval" / "sample5")
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest.jsonl")
    p.add_argument("--train-clips", type=int, default=30)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2" / "ball_probe")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from transformers import AutoVideoProcessor, VJEPA2Model

    device = pick_device()
    model = VJEPA2Model.from_pretrained(args.model_id).to(device).eval()
    proc = AutoVideoProcessor.from_pretrained(args.model_id)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    track, passes, scene = load_tracking()
    side = side_fn(scene["occluder_polygon"])

    sample = json.loads((args.sample / "sample.json").read_text())
    eval_ids = {c["clip_id"] for c in sample["clips"]}
    pool = [json.loads(line) for line in args.manifest.open()]
    pool = [r for r in pool if r.get("include") and r["clip_id"] not in eval_ids]
    train = random.Random(args.seed).sample(pool, min(args.train_clips, len(pool)))
    n_ctx = sample["context_frames"][1] + 1
    print(f"V-JEPA 2 ball probe on {device}: {len(train)} training clips, {len(eval_ids)} held-out eval clips")

    cache = REPO / "outputs" / "vjepa2" / "cache" / args.model_id.replace("/", "--")
    examples = []
    for i, c in enumerate(train):
        frames = read_video(REPO / c["path"])
        lab, _ = ball_labels(source_indices(c, passes, len(frames), n_ctx), track, grid, tub)
        enc = cached_encode(cache, c["clip_id"], model, frames, n_ctx, mean, std, device, imagine=True)
        examples.append(eval_decoder.examples_from(enc, lab))
        print(f"\r  encoded training clip {i + 1}/{len(train)}", end="", flush=True)
    print()
    decoder = eval_decoder.fit(examples, seed=args.seed)
    print(f"  fitted the evaluation decoder on {len(examples)} clips' context halves only")

    args.out.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_auc_score
    occluder = np.array(scene["occluder_polygon"], np.float32)
    cell_px = CLIP_SIZE / grid
    # Far side = beyond the plank and clear of it (cell centre at least one cell outside the polygon).
    far_cells = np.array([[side((xx + 0.5) * cell_px, (yy + 0.5) * cell_px) == "L" and
                           cv2.pointPolygonTest(occluder, ((xx + 0.5) * cell_px, (yy + 0.5) * cell_px), True) < -cell_px
                           for xx in range(grid)] for yy in range(grid)])
    rows, hits, total, aucs = [], 0, 0, []
    future_lab, future_imag, future_hold = [], [], []
    for c in sample["clips"]:
        frames = read_video(args.sample / "clips" / f"{c['clip_id']}.mp4")
        full_rec = next(r for r in pool + [json.loads(line) for line in args.manifest.open()] if r["clip_id"] == c["clip_id"])
        lab, centres = ball_labels(source_indices(full_rec, passes, len(frames), n_ctx), track, grid, tub)
        enc = cached_encode(cache, c["clip_id"], model, frames, n_ctx, mean, std, device, imagine=True)
        cs = enc["context"].shape[0]
        maps_real = decoder(enc["real"]).numpy()
        maps_ctx = decoder(enc["context"]).numpy()
        maps_imag = decoder(enc["imagined"]).numpy()
        maps_seen = np.concatenate([maps_ctx, maps_imag])
        future_lab.append(lab[cs:].reshape(-1))
        future_imag.append(maps_imag.reshape(-1))
        future_hold.append(np.repeat(maps_ctx[-1:], len(maps_imag), 0).reshape(-1))

        # Decoder quality on held-out real frames: is the most ball-like cell on the ball?
        for s in range(len(lab)):
            if lab[s].any():
                total += 1
                yy, xx = np.unravel_index(maps_real[s].argmax(), maps_real[s].shape)
                hits += bool(lab[s, max(yy - 1, 0):yy + 2, max(xx - 1, 0):xx + 2].any())
        if lab.any():
            aucs.append(roc_auc_score(lab.reshape(-1), maps_real.reshape(-1)))

        # Readouts over the target: chance the ball is visible, and on the far side of the plank.
        def readout(m):
            # Calibrated decoder: background cells sit near 0, so the most confident cell is the
            # readout. visible = anywhere; far = beyond the plank and clear of it.
            vis = m.max(axis=(1, 2))
            far = (m * far_cells).max(axis=(1, 2))
            return vis.round(3).tolist(), far.round(3).tolist()
        rv, rf = readout(maps_real[cs:])
        iv, ifar = readout(maps_seen[cs:])
        rows.append({"clip_id": c["clip_id"], "outcome": c["outcome"], "first_target_frame": cs * tub,
                     "real_visible": rv, "real_far": rf, "imagined_visible": iv, "imagined_far": ifar})
        write_video(args.out / f"{c['clip_id']}.mp4",
                    clip_video(c, frames, maps_real, maps_seen, centres, cs, tub), 16)
        print(f"  {c['clip_id']} ({c['outcome']}): last target step - P(visible) real {rv[-1]:.2f} / imagined {iv[-1]:.2f}; "
              f"P(beyond plank) real {rf[-1]:.2f} / imagined {ifar[-1]:.2f}")

    summary_plot(rows, args.out / "summary.png")
    fl, fi, fh = np.concatenate(future_lab), np.concatenate(future_imag), np.concatenate(future_hold)
    imag_auc = roc_auc_score(fl, fi) if 0 < fl.sum() < len(fl) else None
    hold_auc = roc_auc_score(fl, fh) if 0 < fl.sum() < len(fl) else None
    metrics = {"model_id": args.model_id, "train_clips": [c["clip_id"] for c in train],
               "heldout_argmax_hit_rate": round(hits / max(total, 1), 3), "heldout_steps_with_ball": total,
               "heldout_cell_auroc": round(float(np.mean(aucs)), 3) if aucs else None,
               "future_cell_auroc_imagined": round(float(imag_auc), 3) if imag_auc else None,
               "future_cell_auroc_hold_last_context": round(float(hold_auc), 3) if hold_auc else None,
               "clips": rows}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"  held-out real frames: decoder's top cell on the ball in {hits}/{total} steps "
          f"({metrics['heldout_argmax_hit_rate']:.0%}), cell AUROC {metrics['heldout_cell_auroc']}")
    print(f"  where the ball really goes, read from V-JEPA's imagined tokens: cell AUROC "
          f"{metrics['future_cell_auroc_imagined']} (holding the last context map still: "
          f"{metrics['future_cell_auroc_hold_last_context']})")
    print(f"-> {args.out}/ (per-clip .mp4, summary.png, metrics.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

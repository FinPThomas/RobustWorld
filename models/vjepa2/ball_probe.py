"""Read the ball out of V-JEPA 2's features, then out of its imagined future.

1. Label: the tracker knows where the ball is in every frame, so each 32x32 px token cell
   at each 2-frame step is labelled "ball here" or not (hidden under the plank = no ball).
2. Probes: single linear layers (1024 -> 1), trained on clips outside the eval sample.
   Being linear, anything they find is in V-JEPA's features, not in the probe.
     - real probe: on the encoder's tokens of real clips (and the context half alone)
     - imagined probe: on the predictor's imagined target tokens (context encoded alone,
       no peeking), labelled with where the ball really was. The predictor's outputs are
       smoothed averages that sit outside the encoder's distribution, so the real probe
       reads them as a flat ~0.5; this probe is fitted to their own space instead.
3. Check: on the held-out eval clips, does the real probe find the ball in real tokens?
4. Read the imagination: on held-out clips, does the imagined probe recover where the
   ball really goes, better than holding the last context step's ball map still?
5. Against kinematics: the hand-coded baseline in kinematic.py extrapolates the tracked
   ball at constant velocity, hidden while under the plank. Labelled into cells the same
   way, it is scored with the same metrics, plus centre error and whether the ball ends
   up beyond the plank.

Outputs (outputs/vjepa2/ball_probe/):
    <clip>.mp4        frame | real ball map | context/imagined ball map; green = real ball,
                      red = imagined ball (argmax cell, when the probe is confident),
                      blue = kinematic prediction
    summary.png       per clip, over the target: P(ball visible) and P(ball beyond the plank),
                      real vs imagined (most confident cell, from calibrated probes) vs kinematic
    metrics.json      probe quality on held-out real tokens, V-JEPA vs kinematic, per-clip readouts

    python models/vjepa2/ball_probe.py --train-clips 30
"""

from __future__ import annotations

import argparse
import csv
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
from robust_world.passes import side_fn  # noqa: E402
from kinematic import extrapolate  # noqa: E402
from run import MODEL_ID, SIZE, pick_device, to_pixels  # noqa: E402

CLIP_SIZE = 512
PANEL = 256
CONFIDENT = 0.5


def load_tracking(name: str = "start"):
    interim = REPO / "data" / "interim" / name
    track = []
    for r in csv.DictReader((interim / "track.csv").open()):
        track.append((int(r["visible"]), float(r["x"] or 0), float(r["y"] or 0), float(r["radius"] or 0)))
    passes = {p["id"]: p for p in json.loads((interim / "passes.json").read_text())["passes"]}
    scene = json.loads((REPO / "configs" / "scenes" / f"{name}.json").read_text())
    return track, passes, scene


def source_indices(clip: dict, passes: dict, n_frames: int, n_ctx: int) -> list[int]:
    """Clip frame k -> source frame, exactly as robust_world.clips cut it."""
    mid = passes[clip["pass_id"]]["occlusion_start_frame"] - 1
    step = clip["source_fps"] / clip["fps"]
    return [int(round(mid + (k - (n_ctx - 1)) * step)) for k in range(n_frames)]


def ball_labels(src: list[int], track, grid: int, tubelet: int) -> tuple[np.ndarray, list]:
    """[steps, grid, grid] bool cells covered by the visible ball, and its centre per step (or None)."""
    steps = len(src) // tubelet
    cell = CLIP_SIZE / grid
    labels = np.zeros((steps, grid, grid), bool)
    centres = []
    for s in range(steps):
        pts = [track[i] for i in src[s * tubelet:(s + 1) * tubelet] if track[i][0]]
        for _, x, y, r in pts:
            r = max(r, 8.0)
            x0, x1 = int((x - r) // cell), int((x + r) // cell)
            y0, y1 = int((y - r) // cell), int((y + r) // cell)
            labels[s, max(y0, 0):min(y1, grid - 1) + 1, max(x0, 0):min(x1, grid - 1) + 1] = True
        centres.append((np.mean([p[1] for p in pts]), np.mean([p[2] for p in pts])) if pts else None)
    return labels, centres


def cached_encode(cache: Path, clip_id: str, *args, **kwargs) -> dict:
    path = cache / f"{clip_id}.pt"
    if path.exists():
        return torch.load(path)
    enc = encode(*args, **kwargs)
    cache.mkdir(parents=True, exist_ok=True)
    torch.save(enc, path)
    return enc


@torch.no_grad()
def encode(model, frames, n_ctx, mean, std, device, imagine: bool) -> dict:
    cfg = model.config
    g = SIZE // cfg.patch_size
    per_step = g * g
    ctx_steps, all_steps = n_ctx // cfg.tubelet_size, len(frames) // cfg.tubelet_size
    full = model(to_pixels(frames, mean, std, device), skip_predictor=True).last_hidden_state
    out = {"real": full.reshape(all_steps, g, g, -1).half().cpu()}
    if imagine:
        ctx = model(to_pixels(frames[:n_ctx], mean, std, device), skip_predictor=True).last_hidden_state
        pred = model.predictor(
            encoder_hidden_states=ctx,
            context_mask=[torch.arange(ctx_steps * per_step, device=device).unsqueeze(0)],
            target_mask=[torch.arange(ctx_steps * per_step, all_steps * per_step, device=device).unsqueeze(0)],
        ).last_hidden_state
        out["context"] = ctx.reshape(ctx_steps, g, g, -1).half().cpu()
        out["imagined"] = pred.reshape(all_steps - ctx_steps, g, g, -1).half().cpu()
    return out


def train_probe(x: torch.Tensor, y: torch.Tensor, epochs: int = 20, seed: int = 0) -> torch.nn.Module:
    """Linear probe with standardised inputs. Plain (unweighted) BCE keeps its outputs calibrated,
    so P(ball here) can be read as a probability."""
    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-4
    lin = torch.nn.Linear(x.shape[1], 1)
    rate = y.mean().clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():                      # start at the base rate (~1.5% of cells), not 50%
        lin.weight.zero_()
        lin.bias.fill_(float(torch.log(rate / (1 - rate))))
    opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(4096):
            opt.zero_grad()
            loss = loss_fn(lin((x[idx] - mu) / sd).squeeze(1), y[idx])
            loss.backward()
            opt.step()

    class Probe(torch.nn.Module):
        def forward(self, t):                     # [..., D] -> probabilities [...]
            with torch.no_grad():
                return torch.sigmoid(lin((t.float() - mu) / sd).squeeze(-1))
    return Probe()


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


def clip_video(clip, frames, maps_real, maps_seen, centres, kin_centres, ctx_steps, tubelet) -> list[np.ndarray]:
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
        if target and kin_centres[s] is not None:
            kx, ky = kin_centres[s]
            cv2.drawMarker(f, (int(kx * PANEL / CLIP_SIZE), int(ky * PANEL / CLIP_SIZE)), (60, 140, 255),
                           cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
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
            ax.plot(t, r[f"kinematic_{key}"], ":d", color="tab:blue", ms=3, label="kinematic")
            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(f"{r['clip_id']}\ntrue: {r['outcome']}", fontsize=9)
            if j == 0:
                ax.set_ylabel(title)
            if i == 1:
                ax.set_xlabel("clip frame")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Linear ball probe on V-JEPA 2 features: real vs imagined target vs kinematic baseline",
                 fontsize=10)
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
    xs, ys, xi, yi = [], [], [], []
    for i, c in enumerate(train):
        frames = read_video(REPO / c["path"])
        lab, _ = ball_labels(source_indices(c, passes, len(frames), n_ctx), track, grid, tub)
        enc = cached_encode(cache, c["clip_id"], model, frames, n_ctx, mean, std, device, imagine=True)
        cs = enc["context"].shape[0]
        xs.append(enc["real"].reshape(-1, model.config.hidden_size))
        ys.append(torch.from_numpy(lab.reshape(-1)).float())
        xi.append(enc["imagined"].reshape(-1, model.config.hidden_size))
        yi.append(torch.from_numpy(lab[cs:].reshape(-1)).float())
        print(f"\r  encoded training clip {i + 1}/{len(train)}", end="", flush=True)
    print()
    real_probe = train_probe(torch.cat(xs).float(), torch.cat(ys), seed=args.seed)
    imag_probe = train_probe(torch.cat(xi).float(), torch.cat(yi), seed=args.seed)
    print(f"  trained real probe on {sum(len(v) for v in ys)} tokens, "
          f"imagined probe on {sum(len(v) for v in yi)} imagined tokens")

    args.out.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_auc_score
    occluder = np.array(scene["occluder_polygon"], np.float32)
    cell_px = CLIP_SIZE / grid
    # Far side = beyond the plank and clear of it (cell centre at least one cell outside the polygon).
    far_cells = np.array([[side((xx + 0.5) * cell_px, (yy + 0.5) * cell_px) == "L" and
                           cv2.pointPolygonTest(occluder, ((xx + 0.5) * cell_px, (yy + 0.5) * cell_px), True) < -cell_px
                           for xx in range(grid)] for yy in range(grid)])
    rows, hits, total, aucs = [], 0, 0, []
    future_lab, future_imag, future_hold, future_kin = [], [], [], []
    # Per target step with the real ball visible: predicted centre error (clip px), or a miss.
    # Per target step: does the prediction say visible when the ball really is?
    compare = {m: {"errors": [], "misses": 0, "visible_agree": 0} for m in ("vjepa", "kinematic")}
    n_target_steps = 0
    for c in sample["clips"]:
        frames = read_video(args.sample / "clips" / f"{c['clip_id']}.mp4")
        full_rec = next(r for r in pool + [json.loads(line) for line in args.manifest.open()] if r["clip_id"] == c["clip_id"])
        src = source_indices(full_rec, passes, len(frames), n_ctx)
        lab, centres = ball_labels(src, track, grid, tub)
        kin_track, fit = extrapolate(track, src, n_ctx, scene["occluder_polygon"], full_rec["source_fps"], CLIP_SIZE)
        kin_lab, kin_centres = ball_labels(src, kin_track, grid, tub)
        enc = cached_encode(cache, c["clip_id"], model, frames, n_ctx, mean, std, device, imagine=True)
        cs = enc["context"].shape[0]
        maps_real = real_probe(enc["real"]).numpy()
        maps_ctx = real_probe(enc["context"]).numpy()
        maps_imag = imag_probe(enc["imagined"]).numpy()
        maps_seen = np.concatenate([maps_ctx, maps_imag])
        future_lab.append(lab[cs:].reshape(-1))
        future_imag.append(maps_imag.reshape(-1))
        future_hold.append(np.repeat(maps_ctx[-1:], len(maps_imag), 0).reshape(-1))
        future_kin.append(kin_lab[cs:].reshape(-1).astype(np.float32))

        for s in range(cs, len(lab)):
            n_target_steps += 1
            m = maps_imag[s - cs]
            if m.max() >= CONFIDENT:
                yy, xx = np.unravel_index(m.argmax(), m.shape)
                guesses = {"vjepa": ((xx + 0.5) * cell_px, (yy + 0.5) * cell_px)}
            else:
                guesses = {"vjepa": None}
            # Snapped to its cell centre, the same resolution as V-JEPA's argmax cell.
            k = kin_centres[s]
            guesses["kinematic"] = None if k is None else ((k[0] // cell_px + 0.5) * cell_px,
                                                          (k[1] // cell_px + 0.5) * cell_px)
            for name, g in guesses.items():
                compare[name]["visible_agree"] += (g is not None) == (centres[s] is not None)
                if centres[s] is not None:
                    if g is None:
                        compare[name]["misses"] += 1
                    else:
                        compare[name]["errors"].append(float(np.hypot(g[0] - centres[s][0], g[1] - centres[s][1])))

        # Probe quality on held-out real tokens: is the most ball-like cell on the ball?
        for s in range(len(lab)):
            if lab[s].any():
                total += 1
                yy, xx = np.unravel_index(maps_real[s].argmax(), maps_real[s].shape)
                hits += bool(lab[s, max(yy - 1, 0):yy + 2, max(xx - 1, 0):xx + 2].any())
        if lab.any():
            aucs.append(roc_auc_score(lab.reshape(-1), maps_real.reshape(-1)))

        # Readouts over the target: chance the ball is visible, and on the far side of the plank.
        def readout(m):
            # Calibrated probes: background cells sit near 0, so the most confident cell is the
            # readout. visible = anywhere; far = beyond the plank and clear of it.
            vis = m.max(axis=(1, 2))
            far = (m * far_cells).max(axis=(1, 2))
            return vis.round(3).tolist(), far.round(3).tolist()
        rv, rf = readout(maps_real[cs:])
        iv, ifar = readout(maps_seen[cs:])
        kv, kf = readout(kin_lab[cs:].astype(np.float32))
        beyond = {"true": bool((lab[cs:] & far_cells).any()), "vjepa": max(ifar) >= CONFIDENT,
                  "kinematic": bool((kin_lab[cs:] & far_cells).any())}
        rows.append({"clip_id": c["clip_id"], "outcome": c["outcome"], "first_target_frame": cs * tub,
                     "real_visible": rv, "real_far": rf, "imagined_visible": iv, "imagined_far": ifar,
                     "kinematic_visible": kv, "kinematic_far": kf, "beyond_plank": beyond,
                     "kinematic_fit": None if fit is None else
                     {"x": round(fit[0], 1), "y": round(fit[1], 1), "vx_px_per_frame": round(fit[2], 2),
                      "vy_px_per_frame": round(fit[3], 2), "radius": round(fit[4], 1)}})
        write_video(args.out / f"{c['clip_id']}.mp4",
                    clip_video(c, frames, maps_real, maps_seen, centres, kin_centres, cs, tub), 16)
        print(f"  {c['clip_id']} ({c['outcome']}): last target step - P(visible) real {rv[-1]:.2f} / imagined {iv[-1]:.2f}"
              f" / kinematic {kv[-1]:.0f}; P(beyond plank) real {rf[-1]:.2f} / imagined {ifar[-1]:.2f}"
              f" / kinematic {kf[-1]:.0f}")

    summary_plot(rows, args.out / "summary.png")
    fl, fi, fh, fk = (np.concatenate(v) for v in (future_lab, future_imag, future_hold, future_kin))
    both = 0 < fl.sum() < len(fl)
    imag_auc = roc_auc_score(fl, fi) if both else None
    hold_auc = roc_auc_score(fl, fh) if both else None
    kin_auc = roc_auc_score(fl, fk) if both else None
    vs = {}
    for name, cmp in compare.items():
        errs = cmp["errors"]
        vs[name] = {
            "centre_error_px_mean": round(float(np.mean(errs)), 1) if errs else None,
            "centre_error_px_median": round(float(np.median(errs)), 1) if errs else None,
            "ball_steps_located": f"{len(errs)}/{len(errs) + cmp['misses']}",
            "visibility_accuracy": round(cmp["visible_agree"] / max(n_target_steps, 1), 3),
            "beyond_plank_accuracy": round(float(np.mean([r["beyond_plank"][name] == r["beyond_plank"]["true"]
                                                          for r in rows])), 3),
        }
    metrics = {"model_id": args.model_id, "train_clips": [c["clip_id"] for c in train],
               "heldout_argmax_hit_rate": round(hits / max(total, 1), 3), "heldout_steps_with_ball": total,
               "heldout_cell_auroc": round(float(np.mean(aucs)), 3) if aucs else None,
               "future_cell_auroc_imagined": round(float(imag_auc), 3) if imag_auc else None,
               "future_cell_auroc_hold_last_context": round(float(hold_auc), 3) if hold_auc else None,
               "future_cell_auroc_kinematic": round(float(kin_auc), 3) if kin_auc else None,
               "vjepa_vs_kinematic": vs, "clips": rows}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"  held-out real tokens: probe's top cell on the ball in {hits}/{total} steps "
          f"({metrics['heldout_argmax_hit_rate']:.0%}), cell AUROC {metrics['heldout_cell_auroc']}")
    print(f"  where the ball really goes, read from V-JEPA's imagined tokens: cell AUROC "
          f"{metrics['future_cell_auroc_imagined']} (holding the last context map still: "
          f"{metrics['future_cell_auroc_hold_last_context']}; kinematic: {metrics['future_cell_auroc_kinematic']})")
    print(f"  {'':10s} {'centre err px (mean/median)':>28s} {'ball located':>13s} {'visible acc':>12s} {'beyond-plank acc':>17s}")
    for name, v in vs.items():
        print(f"  {name:10s} {str(v['centre_error_px_mean']) + ' / ' + str(v['centre_error_px_median']):>28s} "
              f"{v['ball_steps_located']:>13s} {v['visibility_accuracy']:>12.2f} {v['beyond_plank_accuracy']:>17.2f}")
    print(f"-> {args.out}/ (per-clip .mp4, summary.png, metrics.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

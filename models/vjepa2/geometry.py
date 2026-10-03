"""Visualise V-JEPA 2's predictive geometry: what its imagined future looks like in feature space.

For each clip, the context half is encoded alone and the predictor imagines the target
half's 16x16 token grid at each 2-frame step; the real full clip is encoded for comparison.

Per-clip video (one row, time runs along the clip):
    Frame            the real clip; during the target, green = real "ball" location and
                     red = imagined (token change-map centroids)
    Real / Imagined features
                     tokens projected to RGB with one PCA basis shared by all clips and
                     both real and imagined, so colours are comparable
    Real / Imagined change
                     distance of each token from V-JEPA's own output for an empty-scene
                     version of the clip (the per-pixel median frame, held still) at the same
                     time step; the imagined side is compared with the predictor's output
                     from that empty scene. V-JEPA tokens drift with the time step even
                     where nothing moves, so this baseline is what isolates the ball.

Imagined panels are contrast-stretched on their own: the predictor regresses towards the
mean and its raw outputs are much flatter than real features.

trajectories.png: every clip's per-step features (empty-scene output removed) projected to 2D.
Solid = real path through context and target; dashed = imagined target branching off at
the end of the context.

    python models/vjepa2/geometry.py --sample data/eval/sample5
"""

from __future__ import annotations

import argparse
import json
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
from run import MODEL_ID, SIZE, pick_device, to_pixels  # noqa: E402

PANEL = 256
FPS = 16


@torch.no_grad()
def encode(model, frames, n_ctx, mean, std, device) -> dict:
    """Token grids [steps, g, g, D]: real clip, context alone, imagined target, and the same three
    for an empty-scene clip (median frame held still) to subtract time-step effects."""
    cfg = model.config
    g = SIZE // cfg.patch_size
    per_step = g * g
    ctx_steps, all_steps = n_ctx // cfg.tubelet_size, len(frames) // cfg.tubelet_size
    ctx_mask = [torch.arange(ctx_steps * per_step, device=device).unsqueeze(0)]
    tgt_mask = [torch.arange(ctx_steps * per_step, all_steps * per_step, device=device).unsqueeze(0)]

    def run(clip_frames):
        full = model(to_pixels(clip_frames, mean, std, device), skip_predictor=True).last_hidden_state
        ctx = model(to_pixels(clip_frames[:n_ctx], mean, std, device), skip_predictor=True).last_hidden_state
        pred = model.predictor(encoder_hidden_states=ctx, context_mask=ctx_mask,
                               target_mask=tgt_mask).last_hidden_state
        return full, ctx, pred

    def grid(t, steps):
        return t.reshape(steps, g, g, -1).float().cpu().numpy()

    full, ctx, pred = run(frames)
    empty = np.median(np.stack(frames), axis=0).astype(np.uint8)
    e_full, e_ctx, e_pred = run([empty] * len(frames))
    return {"real": grid(full, all_steps), "context": grid(ctx, ctx_steps),
            "imagined": grid(pred, all_steps - ctx_steps),
            "empty_real": grid(e_full, all_steps), "empty_context": grid(e_ctx, ctx_steps),
            "empty_imagined": grid(e_pred, all_steps - ctx_steps), "ctx_steps": ctx_steps}


def pca_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:k]


def to_rgb(tokens: np.ndarray, mu, basis, lo, hi) -> np.ndarray:
    z = (tokens.reshape(-1, tokens.shape[-1]) - mu) @ basis.T
    z = np.clip((z - lo) / (hi - lo), 0, 1).reshape(*tokens.shape[:-1], 3)
    return (z * 255).astype(np.uint8)


def heat(m: np.ndarray, scale: float) -> np.ndarray:
    u8 = (np.clip(m / scale, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def centroid(m: np.ndarray, frac: float = 0.5):
    """Weighted centre of the strongest cells, in panel pixels; None if nothing stands out."""
    w = np.where(m >= frac * m.max(), m, 0)
    if w.sum() <= 0:
        return None
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    s = PANEL / m.shape[0]
    return ((xs * w).sum() / w.sum() + 0.5) * s, ((ys * w).sum() / w.sum() + 0.5) * s


def up(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def clip_video(clip: dict, frames, enc: dict, mu, basis, lo, hi, tubelet: int) -> list[np.ndarray]:
    real, ctx, imag, cs = enc["real"], enc["context"], enc["imagined"], enc["ctx_steps"]
    change_real = np.linalg.norm(real - enc["empty_real"], axis=-1)
    change_seen = np.concatenate([np.linalg.norm(ctx - enc["empty_context"], axis=-1),
                                  np.linalg.norm(imag - enc["empty_imagined"], axis=-1)])
    scale_real = np.percentile(change_real, 99.5)
    scale_imag = np.percentile(change_seen[cs:], 99.5)
    rgb_real = to_rgb(real, mu, basis, lo, hi)
    rgb_ctx = to_rgb(ctx, mu, basis, lo, hi)
    zi = (imag.reshape(-1, imag.shape[-1]) - mu) @ basis.T
    rgb_imag = to_rgb(imag, mu, basis, np.percentile(zi, 1, 0), np.percentile(zi, 99, 0))
    rgb_seen = np.concatenate([rgb_ctx, rgb_imag])

    out = []
    for f_idx, frame in enumerate(frames):
        s = min(f_idx // tubelet, len(real) - 1)
        target = s >= cs
        f = cv2.resize(frame, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
        if target:
            for m, colour in ((change_real[s], (60, 220, 60)), (change_seen[s], (240, 60, 60))):
                c = centroid(m)
                if c:
                    cv2.circle(f, (int(c[0]), int(c[1])), 10, colour, 2, cv2.LINE_AA)
        panels = [f, up(rgb_real[s]), up(rgb_seen[s]), up(heat(change_real[s], scale_real)),
                  up(heat(change_seen[s], scale_imag if target else scale_real))]
        names = ["frame", "real features", "imagined features" if target else "context features (given)",
                 "real change", "imagined change" if target else "context change (given)"]
        for p, n in zip(panels, names):
            label(p, n)
        row = np.hstack(panels)
        bar = np.zeros((26, row.shape[1], 3), np.uint8)
        phase = "TARGET: imagined vs real" if target else "CONTEXT: given to the model"
        cv2.putText(bar, f"{clip['clip_id']}  true outcome: {clip['outcome']}   frame {f_idx}   {phase}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 120) if target else (200, 200, 200), 1,
                    cv2.LINE_AA)
        out.append(np.vstack([bar, row]))
    return out


def trajectories(clips, encs, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def steps(t, empty):                       # [steps, g, g, D] -> [steps, 16*D] (4x4 pooled, empty scene removed)
        d = t - empty
        n, g = d.shape[0], d.shape[1]
        k = g // 4
        return d.reshape(n, 4, k, 4, k, -1).mean((2, 4)).reshape(n, -1)

    real, imag = [], []
    for e in encs:
        real.append(steps(e["real"], e["empty_real"]))
        imag.append(steps(e["imagined"], e["empty_imagined"]))
    mu, basis = pca_basis(np.concatenate(real), 2)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
    colours = plt.cm.tab10.colors
    for i, (c, r, m, e) in enumerate(zip(clips, real, imag, encs)):
        cs = e["ctx_steps"]
        pr, pm = (r - mu) @ basis.T, (m - mu) @ basis.T
        col = colours[i % 10]
        ax.plot(pr[:cs + 1, 0], pr[:cs + 1, 1], "-", color=col, alpha=0.45, lw=1.5)
        ax.plot(pr[cs:, 0], pr[cs:, 1], "-o", color=col, lw=2, ms=3, label=f"{c['clip_id']} ({c['outcome']}) real")
        branch = np.vstack([pr[cs - 1:cs], pm])
        ax.plot(branch[:, 0], branch[:, 1], "--x", color=col, lw=1.5, ms=4)
        ax.plot(*pr[0], "s", color=col, ms=6)
    ax.plot([], [], "k-", label="real (faint = context)")
    ax.plot([], [], "k--x", label="imagined target")
    ax.set_title("V-JEPA 2 predictive geometry: per-step features minus empty-scene output (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, default=REPO / "data" / "eval" / "sample5")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "vjepa2" / "geometry")
    p.add_argument("--model-id", default=MODEL_ID)
    args = p.parse_args(argv)

    from transformers import AutoVideoProcessor, VJEPA2Model

    device = pick_device()
    sample = json.loads((args.sample / "sample.json").read_text())
    model = VJEPA2Model.from_pretrained(args.model_id).to(device).eval()
    proc = AutoVideoProcessor.from_pretrained(args.model_id)
    mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    n_ctx = sample["context_frames"][1] + 1
    print(f"V-JEPA 2 geometry ({args.model_id}) on {device}; {len(sample['clips'])} clips")

    clips, frames_all, encs = sample["clips"], [], []
    for c in clips:
        frames = read_video(args.sample / "clips" / f"{c['clip_id']}.mp4")
        frames_all.append(frames)
        encs.append(encode(model, frames, n_ctx, mean, std, device))
        print(f"  encoded {c['clip_id']}")

    # One PCA colour basis for every clip, real and imagined alike.
    tokens = np.concatenate([e["real"].reshape(-1, e["real"].shape[-1]) for e in encs])
    mu, basis = pca_basis(tokens[np.random.default_rng(0).choice(len(tokens), min(len(tokens), 20000), False)], 3)
    z = (tokens - mu) @ basis.T
    lo, hi = np.percentile(z, 1, 0), np.percentile(z, 99, 0)

    args.out.mkdir(parents=True, exist_ok=True)
    videos = []
    for c, frames, e in zip(clips, frames_all, encs):
        v = clip_video(c, frames, e, mu, basis, lo, hi, model.config.tubelet_size)
        write_video(args.out / f"{c['clip_id']}.mp4", v, FPS)
        videos.append(v)
    write_video(args.out / "all.mp4", [np.vstack(fs) for fs in zip(*videos)], FPS)
    trajectories(clips, encs, args.out / "trajectories.png")
    print(f"-> {args.out}/ (per-clip .mp4, all.mp4, trajectories.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

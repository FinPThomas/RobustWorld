"""Side-by-side comparison videos: ground truth plus any number of models.

Layout (one column per source, two rows):
    row "Input":  what each source was conditioned on. Ground truth shows the whole
                  context; a model shows only the frames it was given (frames it
                  wasn't given are dimmed).
    row "Output": ground-truth target vs each model's prediction.

Timeline: the context plays in the input row while the output row waits, then the
input row freezes on its last frame while the outputs play, then a short hold.
Adding models only adds columns, so the same code serves larger comparisons later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..media import write_video
from .io import FPS, load_prediction, load_sample, read_clip, resize

HOLD = 8                     # frames the final state is held before the video loops
LABEL_W = 90                 # left gutter for row labels
HEADER_H = 30
TITLE_H = 28
BG = (24, 24, 24)
WHITE = (240, 240, 240)
DIM = 0.3


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    # Drop shadow at the same thickness: a thicker outline is spaced wider and leaves ghost letters.
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fit(s: str, width: int, scale: float = 0.55, min_scale: float = 0.38) -> tuple[str, float]:
    """Shrink, then truncate, a label so it fits in `width` pixels."""
    while scale > min_scale and cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        scale -= 0.02
    while len(s) > 4 and cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        s = s[:-2].rstrip() + "…"
        s = s.replace("……", "…")
    return s.replace("…", "..."), scale


def _placeholder(cell: int, msg: str) -> np.ndarray:
    im = np.full((cell, cell, 3), 40, np.uint8)
    size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    _text(im, msg, ((cell - size[0]) // 2, cell // 2), color=(160, 160, 160))
    return im


def clip_grid(sample: dict, clip: dict, columns: list[tuple[str, Path | None]], cell: int = 256) -> list[np.ndarray]:
    """Frames of one clip's comparison video. columns: (label, prediction dir); None = ground truth."""
    context, target = read_clip(sample, clip["clip_id"])
    context, target = resize(context, cell), resize(target, cell)
    n_ctx, n_tgt = len(context), len(target)

    cols = []  # per column: (label, input frame per context step or None, output frames or None)
    for label, pred_dir in columns:
        if pred_dir is None:
            cols.append((label, list(context), target, None))
            continue
        pred = load_prediction(pred_dir, clip["clip_id"])
        if pred is None:
            cols.append((label, [None] * n_ctx, None, "no prediction"))
            continue
        given = dict(zip(pred["input_frame_indices"], resize(pred["inputs"], cell)))
        inputs = [given.get(t) for t in range(n_ctx)]
        cols.append((pred.get("label", label), inputs, resize(pred["frames"], cell)[:n_tgt], None))

    width = LABEL_W + len(cols) * cell
    height = TITLE_H + HEADER_H + 2 * cell
    base = np.full((height, width, 3), BG, np.uint8)
    title = f"{clip['clip_id']}   true outcome: {clip.get('outcome', '?')}"
    _text(base, title, (8, 19), 0.55)
    for c, (label, *_rest) in enumerate(cols):
        fitted, scale = _fit(label, cell - 12)
        _text(base, fitted, (LABEL_W + c * cell + 6, TITLE_H + 21), scale)
    _text(base, "Input", (10, TITLE_H + HEADER_H + cell // 2), 0.6)
    _text(base, "Output", (10, TITLE_H + HEADER_H + cell + cell // 2), 0.6)

    waiting = _placeholder(cell, "...")
    frames = []
    total = n_ctx + n_tgt + HOLD
    for step in range(total):
        im = base.copy()
        t_ctx = min(step, n_ctx - 1)
        t_tgt = step - n_ctx
        for c, (label, inputs, outputs, missing) in enumerate(cols):
            x = LABEL_W + c * cell
            y_in, y_out = TITLE_H + HEADER_H, TITLE_H + HEADER_H + cell
            # Input row: the frame this source was given at time t, else the dimmed ground truth.
            f = inputs[t_ctx]
            if f is None:
                last_given = [i for i in range(t_ctx + 1) if inputs[i] is not None]
                if step >= n_ctx and last_given:
                    f = inputs[last_given[-1]]
                else:
                    f = (context[t_ctx] * DIM).astype(np.uint8)
                    _text(f, "not given", (8, cell - 10), 0.45, (170, 170, 170))
            im[y_in:y_in + cell, x:x + cell] = f
            # Output row: wait during the context, then play the target / prediction.
            if missing:
                im[y_out:y_out + cell, x:x + cell] = _placeholder(cell, missing)
            elif t_tgt < 0:
                im[y_out:y_out + cell, x:x + cell] = waiting
            else:
                im[y_out:y_out + cell, x:x + cell] = outputs[min(t_tgt, len(outputs) - 1)]
        phase = (f"context  frame {step}/{n_ctx - 1}" if step < n_ctx
                 else f"prediction  frame {min(t_tgt, n_tgt - 1) + n_ctx}/{n_ctx + n_tgt - 1}")
        _text(im, phase, (width - 260, 19), 0.5, (255, 210, 120))
        frames.append(im)
    return frames


def build(sample_dir: Path, pred_root: Path, models: list[str], out_dir: Path,
          cell: int = 256) -> list[Path]:
    """One grid video per clip plus all.mp4 (every clip stacked) and index.html."""
    sample = load_sample(sample_dir)
    columns = [("Ground truth", None)] + [(m, Path(pred_root) / m) for m in models]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, blocks = [], []
    for clip in sample["clips"]:
        frames = clip_grid(sample, clip, columns, cell)
        p = out_dir / f"{clip['clip_id']}.mp4"
        write_video(p, frames, FPS)
        paths.append(p)
        blocks.append([cv2.resize(f, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA) for f in frames])
    # Stack every clip; even-size the canvas for H.264.
    stacked = [np.vstack([b[i] for b in blocks]) for i in range(len(blocks[0]))]
    h, w = stacked[0].shape[:2]
    stacked = [np.pad(f, ((0, h % 2), (0, w % 2), (0, 0))) for f in stacked]
    write_video(out_dir / "all.mp4", stacked, FPS)
    videos = "\n".join(f'<figure><video src="{p.name}" autoplay loop muted playsinline></video>'
                       f"<figcaption>{p.stem}</figcaption></figure>" for p in paths)
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Model comparison</title>"
        "<style>body{background:#111;color:#ddd;font:14px sans-serif;margin:16px}"
        "video{max-width:100%}figure{margin:0 0 24px}</style>"
        f"<h1>{sample['name']}: {', '.join(['ground truth'] + models)}</h1>{videos}")
    print(f"grid: {len(paths)} clips x {len(columns)} columns -> {out_dir}/ (per-clip .mp4, all.mp4, index.html)")
    return paths

"""Eval samples and the prediction file format shared by every model.

Sample (data/eval/<sample>/, committed so Colab can use it):
    sample.json        {"name", "prompt", "context_frames", "target_frames", "clips": [manifest records]}
    clips/<id>.mp4     the full clip (context + ground-truth target)

Prediction (<pred_root>/<sample>/<model>/):
    <id>.mp4           predicted target: exactly len(target_frames) frames, 512x512, 16 fps
    <id>_input.mp4     the frames the model was conditioned on, resized to 512x512
    <id>.json          {"model", "label", "input_frame_indices": clip frame index of each input frame, ...settings}
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..media import iter_frames, write_video

FPS = 16


def load_sample(sample_dir: Path) -> dict:
    sample_dir = Path(sample_dir)
    sample = json.loads((sample_dir / "sample.json").read_text())
    sample["dir"] = sample_dir
    return sample


def read_video(path: Path) -> list[np.ndarray]:
    return [f.copy() for f in iter_frames(Path(path))]


def read_clip(sample: dict, clip_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(context frames, ground-truth target frames) as RGB uint8 arrays."""
    frames = read_video(sample["dir"] / "clips" / f"{clip_id}.mp4")
    c0, c1 = sample["context_frames"]
    t0, t1 = sample["target_frames"]
    return frames[c0:c1 + 1], frames[t0:t1 + 1]


def resize(frames: list[np.ndarray], size: int | tuple[int, int]) -> list[np.ndarray]:
    w, h = (size, size) if isinstance(size, int) else size
    if frames and frames[0].shape[:2] == (h, w):
        return list(frames)
    interp = cv2.INTER_AREA if frames and frames[0].shape[1] > w else cv2.INTER_CUBIC
    return [cv2.resize(f, (w, h), interpolation=interp) for f in frames]


def to_uint8(frames) -> list[np.ndarray]:
    """Accept float [0,1] or uint8 arrays / PIL images; return uint8 RGB arrays."""
    out = []
    for f in frames:
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        out.append(a[..., :3])
    return out


def write_prediction(out_dir: Path, clip_id: str, prediction, inputs, input_frame_indices: list[int],
                     meta: dict, size: int = 512) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = resize(to_uint8(prediction), size)
    inp = resize(to_uint8(inputs), size)
    assert len(inp) == len(input_frame_indices), "one clip frame index per input frame"
    write_video(out_dir / f"{clip_id}.mp4", pred, FPS)
    write_video(out_dir / f"{clip_id}_input.mp4", inp, FPS)
    (out_dir / f"{clip_id}.json").write_text(json.dumps(
        {"clip_id": clip_id, "input_frame_indices": list(input_frame_indices), **meta}, indent=1))


def load_prediction(model_dir: Path, clip_id: str) -> dict | None:
    model_dir = Path(model_dir)
    meta_path = model_dir / f"{clip_id}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    meta["frames"] = read_video(model_dir / f"{clip_id}.mp4")
    meta["inputs"] = read_video(model_dir / f"{clip_id}_input.mp4")
    return meta

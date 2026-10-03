"""ffmpeg helpers (via the binary bundled with imageio-ffmpeg)."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Iterator

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# imageio-ffmpeg warns whenever -vf rescales the output; that's intended here.
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def probe(path: Path) -> dict:
    """Width, height, fps and duration from ffmpeg's stream header."""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True).stderr
    w, h = map(int, re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", out).groups())
    fps = float(re.search(r"([\d.]+) fps", out).group(1))
    hh, mm, ss = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return {"width": w, "height": h, "fps": fps,
            "duration": int(hh) * 3600 + int(mm) * 60 + float(ss)}


def downsize(src: Path, dst: Path, size: int = 512, mode: str = "crop",
             fps: float | None = None, crf: int = 18) -> dict:
    """Square, scale and strip audio. Keeps the source frame rate unless fps is given.

    mode decides how a non-square source is squared: centre "crop", "pad" or "stretch".
    A keyframe every second keeps later frame-accurate seeks fast.
    """
    info = probe(src)
    w, h = info["width"], info["height"]
    filters = []
    if w != h and mode == "crop":
        filters.append(f"crop={min(w, h)}:{min(w, h)}")
    elif w != h and mode == "pad":
        filters.append(f"pad={max(w, h)}:{max(w, h)}:(ow-iw)/2:(oh-ih)/2")
    filters.append(f"scale={size}:{size}:flags=lanczos")
    if fps:
        filters.append(f"fps={fps}")
    filters.append("setsar=1")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".partial.mp4")
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-stats", "-y",
                    "-i", str(src), "-vf", ",".join(filters), "-an",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-g", str(round(fps or info["fps"])),
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(dst)
    return info


def iter_frames(path: Path, start_frame: int = 0, fps: float | None = None,
                vf: str | None = None) -> Iterator[np.ndarray]:
    """Yield RGB frames. start_frame seeks frame-accurately (needs fps)."""
    input_params = []
    if start_frame:
        # Input-side -ss is frame-accurate when decoding; back off half a frame so start_frame is included.
        input_params = ["-ss", f"{(start_frame - 0.5) / fps:.4f}"]
    reader = imageio_ffmpeg.read_frames(str(path), input_params=input_params,
                                        output_params=["-vf", vf] if vf else None)
    w, h = next(reader)["size"]  # size after any -vf filters
    try:
        for buf in reader:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        reader.close()


def read_frames(path: Path, first: int, count: int, fps: float) -> list[np.ndarray]:
    frames = []
    for f in iter_frames(path, first, fps):
        frames.append(f)
        if len(frames) == count:
            break
    return frames


def write_video(path: Path, frames: list[np.ndarray], fps: float, crf: int = 18) -> None:
    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"])
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()

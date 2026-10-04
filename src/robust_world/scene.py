"""Per-video scene calibration, ball detection and hand/foreground masks.

Assumes a fixed overhead camera, a blue ball and an occluder whose outline is
given in configs/scenes/<video>.json (pixel coords in the downsized frame).
Anything not set in that file falls back to DEFAULTS.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .media import iter_frames

DEFAULTS = {
    "ball_hsv_lo": [90, 70, 40],     # OpenCV HSV (H 0-179): blue
    "ball_hsv_hi": [140, 255, 255],
    "ball_min_area": 60,             # px; smaller blue blobs are noise
    # Optional second colour range for a striped ball. When set, the ball is the blue core plus
    # touching stripe pixels, so its centre and radius cover the whole ball, not one blue band.
    "ball_stripe_hsv_lo": None,
    "ball_stripe_hsv_hi": None,
    "background_breaks_s": "auto",   # or a list of times where the static scene changed
    "ignore_regions": [],            # polygons of clutter (e.g. furniture) never treated as a hand
    "blocker_polygon": None,         # outline of the hidden blocker under the occluder; "none" if
                                     # there is no blocker; unset (None) if not recorded
    "hand_min_px": 2500,             # foreground pixels in a frame that count as a hand in shot
}
# Settings that change tracking output (track.csv, backgrounds.npz); editing any reruns tracking.
TRACK_KEYS = ["occluder_polygon", "ball_hsv_lo", "ball_hsv_hi", "ball_min_area", "ball_stripe_hsv_lo",
              "ball_stripe_hsv_hi", "background_breaks_s", "ignore_regions"]
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
BALL_PX_DILATE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

BG_SAMPLES = 150        # frames sampled per scene segment for its median background
FG_THRESH = 35          # per-channel abs difference from the brightness-matched background
GAIN_SIGMA = 20         # px; soft lighting changes (body shadows) at this scale are absorbed
BALL_EXCLUDE = 1.6      # foreground within this many ball radii is the ball itself, not a hand

BREAK_WINDOW_S = 10     # seconds compared either side of a candidate scene change
BREAK_THRESH = 6.0      # mean grey-level change (64x64 thumbnails) that counts as a scene change
BREAK_MIN_GAP_S = 30


def load_scene(path: Path) -> dict:
    scene = {**DEFAULTS, **json.loads(path.read_text())}
    scene["occluder"] = np.array(scene["occluder_polygon"], dtype=np.float32)
    scene["ignore"] = [np.array(r, dtype=np.int32) for r in scene["ignore_regions"]]
    scene["blocker"] = blocker_array(scene)
    if scene["blocker"] is not None:
        outside = [p for p in scene["blocker_polygon"]
                   if cv2.pointPolygonTest(scene["occluder"], (float(p[0]), float(p[1])), True) < -2]
        if outside:
            print(f"  warning: blocker_polygon points {outside} lie outside occluder_polygon in {path.name}")
    return scene


def blocker_array(scene: dict) -> np.ndarray | None:
    """The blocker outline as a float32 array, or None when there is no blocker or it isn't recorded."""
    b = scene.get("blocker_polygon")
    return np.array(b, dtype=np.float32) if isinstance(b, list) and b else None


def write_scene_template(path: Path, video: Path, calibration_png: Path) -> None:
    """Save a gridded reference frame and a scene file to fill in by hand."""
    frame = next(iter_frames(video))
    im = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = im.shape[:2]
    for x in range(0, w, 32):
        cv2.line(im, (x, 0), (x, h - 1), (0, 0, 255) if x % 64 == 0 else (0, 0, 120), 1)
        if x % 64 == 0:
            cv2.putText(im, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for y in range(0, h, 32):
        cv2.line(im, (0, y), (w - 1, y), (255, 0, 0) if y % 64 == 0 else (120, 0, 0), 1)
        if y % 64 == 0:
            cv2.putText(im, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    calibration_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(calibration_png), im)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{\n'
        f'  "_comment": "Scene for {path.stem}.mp4. Set occluder_polygon from {calibration_png.name} (x, y px), then remove TODO. '
        'Set blocker_polygon to the hidden blocker\'s outline, or \\"none\\" if there is no blocker.",\n'
        '  "TODO": true,\n'
        '  "occluder_polygon": [[0, 0], [0, 0], [0, 0], [0, 0]],\n'
        '  "blocker_polygon": null\n'
        '}\n')


def detect_ball(rgb: np.ndarray, scene: dict) -> dict:
    """Largest blue blob (with its stripes, if configured), and its signed distance to the
    occluder edge (>0 = overlapping)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, tuple(scene["ball_hsv_lo"]), tuple(scene["ball_hsv_hi"]))
    striped = bool(scene.get("ball_stripe_hsv_lo"))
    mask = blue
    if striped:
        mask = blue | cv2.inRange(hsv, tuple(scene["ball_stripe_hsv_lo"]), tuple(scene["ball_stripe_hsv_hi"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if not striped:
        blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= scene["ball_min_area"]]
    else:
        # A blob is the ball only if enough of it is blue; stripe colours alone are other objects.
        blue_px = np.bincount(labels[blue > 0], minlength=n)
        blobs = [i for i in range(1, n) if blue_px[i] >= scene["ball_min_area"]]
    if not blobs:
        return {"visible": 0, "n_blobs": 0}
    i = max(blobs, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    x, y = cents[i]
    # Bounding-box radius stays sensible when the ball is half-hidden by the occluder or frame edge.
    radius = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / 2
    occ_dist = cv2.pointPolygonTest(scene["occluder"], (float(x), float(y)), True) + radius
    return {"visible": 1, "x": round(float(x), 1), "y": round(float(y), 1),
            "area": int(stats[i, cv2.CC_STAT_AREA]), "radius": round(float(radius), 1),
            "occ_dist": round(float(occ_dist), 1), "n_blobs": len(blobs)}


def detect_scene_breaks(video: Path) -> list[float]:
    """Times (s) where the static scene changes persistently, e.g. a prop or chair moved.

    Compares the median 64x64 thumbnail of the BREAK_WINDOW_S seconds before and
    after each second; brief events (hands, the ball) don't survive the median.
    """
    thumbs = np.stack([f.astype(np.float32).mean(2)
                       for f in iter_frames(video, vf="fps=1,scale=64:64")])
    win = BREAK_WINDOW_S
    score = np.zeros(len(thumbs))
    for t in range(win, len(thumbs) - win):
        score[t] = np.abs(np.median(thumbs[t - win:t], 0) - np.median(thumbs[t:t + win], 0)).mean()
    breaks: list[int] = []
    for t in np.argsort(-score):
        if score[t] < BREAK_THRESH:
            break
        if all(abs(t - b) >= BREAK_MIN_GAP_S for b in breaks):
            breaks.append(int(t))
    return sorted(float(b) for b in breaks)


def segment_backgrounds(video: Path, starts: list[int], total: int) -> np.ndarray:
    """Median 'empty scene' frame for each segment [starts[i], starts[i+1])."""
    bounds = starts + [total]
    wanted = {}
    for seg, (a, b) in enumerate(zip(bounds, bounds[1:])):
        for idx in range(a, b, max(1, (b - a) // BG_SAMPLES)):
            wanted[idx] = seg
    samples = [[] for _ in starts]
    for idx, f in enumerate(iter_frames(video)):
        if idx in wanted:
            samples[wanted[idx]].append(f.copy())
    return np.stack([cv2.GaussianBlur(np.median(np.stack(s), 0).astype(np.uint8), (7, 7), 0)
                     for s in samples])


def background_for(frame: int, bgs: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return bgs[np.searchsorted(starts, frame, side="right") - 1]


def _local_mean(grey: np.ndarray) -> np.ndarray:
    # Blur at 1/8 resolution: same result as a GAIN_SIGMA blur at full size, far cheaper.
    h, w = grey.shape
    small = cv2.resize(grey, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), GAIN_SIGMA / 8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def foreground_mask(rgb: np.ndarray, bg: np.ndarray, ball: dict, scene: dict) -> np.ndarray:
    """Pixels that differ from the background, excluding the ball and ignore_regions (uint8 0/1).

    The background is rescaled by the local brightness ratio first, so soft body
    shadows and lighting drift cancel and only sharp, differently coloured
    objects (hands, arms) remain.
    """
    f = cv2.GaussianBlur(rgb, (7, 7), 0).astype(np.float32) + 1
    b = bg.astype(np.float32) + 1
    gain = _local_mean(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)) / _local_mean(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
    r, g, bl = cv2.split(cv2.absdiff(f, b * gain[..., None]))
    diff = cv2.max(cv2.max(r, g), bl)
    mask = cv2.morphologyEx((diff > FG_THRESH).astype(np.uint8), cv2.MORPH_OPEN, KERNEL)
    if ball["visible"]:
        cv2.circle(mask, (int(ball["x"]), int(ball["y"])), int(ball["radius"] * BALL_EXCLUDE), 0, -1)
    if scene.get("ball_stripe_hsv_lo"):
        # Ball-coloured pixels are never a hand. This also removes the edge of a ball half out
        # of frame, which the circle above misses because its centre and radius are off.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ball_px = (cv2.inRange(hsv, tuple(scene["ball_hsv_lo"]), tuple(scene["ball_hsv_hi"]))
                   | cv2.inRange(hsv, tuple(scene["ball_stripe_hsv_lo"]), tuple(scene["ball_stripe_hsv_hi"])))
        mask[cv2.dilate(ball_px, BALL_PX_DILATE) > 0] = 0
    if scene["ignore"]:
        cv2.fillPoly(mask, scene["ignore"], 0)
    return mask


def fill_mask(m: np.ndarray) -> np.ndarray:
    """Turn fragmented hand edges into a solid, slightly padded region."""
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c for c in contours if cv2.contourArea(c) > 200], -1, 1, -1)
    return cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

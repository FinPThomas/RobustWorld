"""Hand-coded kinematic baseline: where the ball goes if it just keeps rolling.

From the tracker's positions over the end of the context (source fps), fit a constant
velocity and extrapolate through the target. The ball counts as not visible while its
centre is under the occluder or outside the frame. Nothing else is known: no blockade,
no bounce, no friction, so on a blocked pass it predicts the ball coming straight through.

Output is a track in the tracker's own format, so it's labelled into token cells exactly
like the real ball.
"""

from __future__ import annotations

import cv2
import numpy as np

FIT_S = 1 / 3          # seconds of context, ending at the last context frame, used for the fit
MIN_POINTS = 3         # visible tracker points needed for a fit


def fit_velocity(track, first: int, last: int, fps: float):
    """Least-squares constant-velocity fit over visible source frames first..last.
    -> (x, y, vx, vy, radius) at frame `last`, velocity in px per source frame; None if too few points."""
    lo = max(first, last - int(round(FIT_S * fps)))
    pts = [(i, track[i][1], track[i][2], track[i][3]) for i in range(lo, last + 1) if track[i][0]]
    if len(pts) < MIN_POINTS:
        return None
    t, x, y, r = (np.array(v, float) for v in zip(*pts))
    vx, x0 = np.polyfit(t - last, x, 1)
    vy, y0 = np.polyfit(t - last, y, 1)
    return float(x0), float(y0), float(vx), float(vy), float(np.median(r))


def extrapolate(track, src: list[int], n_ctx: int, occluder, fps: float, frame_size: int = 512):
    """Copy of `track` with the target's source frames replaced by the constant-velocity prediction.
    Context frames keep their real positions. Returns (track, fit) where fit is None if no fit."""
    fit = fit_velocity(track, src[0], src[n_ctx - 1], fps)
    out = list(track)
    poly = np.asarray(occluder, np.float32)
    first, last = src[n_ctx], src[-1]
    for i in range(first, last + 1):
        if fit is None:
            out[i] = (0, 0.0, 0.0, 0.0)
            continue
        x0, y0, vx, vy, r = fit
        dt = i - src[n_ctx - 1]
        x, y = x0 + vx * dt, y0 + vy * dt
        in_frame = 0 <= x < frame_size and 0 <= y < frame_size
        under = cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
        out[i] = (int(in_frame and not under), float(x), float(y), r)
    return out, fit

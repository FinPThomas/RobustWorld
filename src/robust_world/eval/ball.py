"""Ball-level ground truth and scoring shared by every model (V-JEPA probes, trackers, ...).

Ground truth comes from the colour tracker (data/interim/<video>/track.csv), mapped onto
each clip's frames exactly as robust_world.clips cut them. Predictions are compared on a
GRID x GRID cell map per 2-frame step, so feature-space models and trackers share a scale.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ..passes import side_fn
from ..paths import REPO_ROOT

CLIP_SIZE = 512
GRID = 16          # 32 px cells, matching V-JEPA 2's 16x16 tokens at 256 px
STEP = 2           # frames per step (V-JEPA 2 tubelet)


def included_clips(manifest: Path = REPO_ROOT / "data" / "processed" / "clips" / "manifest.jsonl") -> list[dict]:
    """Included clips whose outcome is through or hidden, in manifest order (fixes the CV folds)."""
    clips = [json.loads(line) for line in manifest.open()]
    return [c for c in clips if c.get("include") and c["outcome"] in ("through", "hidden")]


def cv_folds(clips: list[dict], folds: int = 5, seed: int = 0):
    from sklearn.model_selection import StratifiedKFold
    y = np.array([c["outcome"] == "through" for c in clips], int)
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(y, y))


def load_tracking(name: str = "start"):
    interim = REPO_ROOT / "data" / "interim" / name
    track = []
    for r in csv.DictReader((interim / "track.csv").open()):
        track.append((int(r["visible"]), float(r["x"] or 0), float(r["y"] or 0), float(r["radius"] or 0)))
    passes = {p["id"]: p for p in json.loads((interim / "passes.json").read_text())["passes"]}
    scene = json.loads((REPO_ROOT / "configs" / "scenes" / f"{name}.json").read_text())
    return track, passes, scene


def source_indices(clip: dict, passes: dict, n_frames: int, n_ctx: int) -> list[int]:
    """Clip frame k -> source frame, exactly as robust_world.clips cut it."""
    mid = passes[clip["pass_id"]]["occlusion_start_frame"] - 1
    step = clip["source_fps"] / clip["fps"]
    return [int(round(mid + (k - (n_ctx - 1)) * step)) for k in range(n_frames)]


def ball_labels(src: list[int], track, grid: int = GRID, tubelet: int = STEP) -> tuple[np.ndarray, list]:
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


def far_cells(scene: dict, grid: int = GRID) -> np.ndarray:
    """Cells beyond the occluder (far side) and clear of it by at least one cell."""
    side = side_fn(scene["occluder_polygon"])
    occluder = np.array(scene["occluder_polygon"], np.float32)
    cell = CLIP_SIZE / grid
    return np.array([[side((x + 0.5) * cell, (y + 0.5) * cell) == "L" and
                      cv2.pointPolygonTest(occluder, ((x + 0.5) * cell, (y + 0.5) * cell), True) < -cell
                      for x in range(grid)] for y in range(grid)])


def readouts(maps: np.ndarray, far: np.ndarray) -> dict:
    """Per-step P(ball visible) and P(ball beyond the occluder) from [steps, grid, grid] probabilities."""
    return {"visible": maps.max(axis=(1, 2)).round(3).tolist(),
            "far": (maps * far).max(axis=(1, 2)).round(3).tolist()}


def summary_metrics(rows: list[dict], future_labels: list[np.ndarray], future_maps: list[np.ndarray]) -> dict:
    """Dataset-level scores shared across models: cell AUROC for where the ball goes, and
    through-vs-hidden AUROC / accuracy from the peak P(ball beyond the occluder)."""
    from sklearn.metrics import roc_auc_score
    fl, fm = np.concatenate([x.reshape(-1) for x in future_labels]), np.concatenate([x.reshape(-1) for x in future_maps])
    y = np.array([r["outcome"] == "through" for r in rows], int)
    score = np.array([max(r["far"]) for r in rows])
    return {
        "n_clips": len(rows), "n_through": int(y.sum()), "n_hidden": int(len(y) - y.sum()),
        "future_cell_auroc": round(float(roc_auc_score(fl, fm)), 3),
        "outcome_auroc": round(float(roc_auc_score(y, score)), 3),
        "outcome_accuracy": round(float(((score > 0.5) == y).mean()), 3),
        "peak_far_mean": {"through": round(float(score[y == 1].mean()), 3),
                          "hidden": round(float(score[y == 0].mean()), 3)},
    }

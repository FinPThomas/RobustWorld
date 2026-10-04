"""Hand-engineered baseline: BootsTAPNext point tracking + straight-line continuation + scene rules.

No learning about physics. For each clip:
1. Track. The ball is found by colour in the context half; 5 points on it (centre plus 4
   around it) are tracked with BootsTAPNext over the context frames only. A rolling ball turns
   any single surface point out of view, so the centre is the mean of the visible points.
2. Continue. A straight line is fitted to the last 8 tracked centres and extended through
   the target frames at constant velocity.
3. Rules:
   - inside the plank outline the ball is invisible;
   - if the line crosses the plank's centre line inside the blockade, the ball stops there
     (hidden for good). The blockade's extent along the plank is not visible, so it is
     fitted on the training clips of each cross-validation fold (never the clip scored);
   - outside the frame the ball is gone.
   Variant "continue" skips the blockade rule (pure continuation).

Scored on the same ground truth, cell grid and 5 folds as the V-JEPA 2 ball probe
(robust_world.eval.ball). For an eval sample it also renders pixel predictions (ball pasted
onto the last context frame with the ball removed, hidden behind the plank) so it can sit
in the comparison grid next to the video models.

    python models/tapnext_rule/predict.py                       # all clips, cross-validated
    python models/tapnext_rule/predict.py --render data/eval/sample5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.ball import (CLIP_SIZE, GRID, STEP, ball_labels, cv_folds, far_cells, near_cells,  # noqa: E402
                                    included_clips, load_tracking, readouts, source_indices,
                                    summary_metrics, video_of)
from robust_world.eval.io import read_video, write_prediction  # noqa: E402
from robust_world.scene import detect_ball, load_scene  # noqa: E402

TAPNET_DIR = REPO / "third_party" / "tapnet"
CHECKPOINT = REPO / "checkpoints" / "tapnext" / "bootstapnext_ckpt.npz"
CHECKPOINT_URL = "https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz"
TAP_SIZE = 256
FIT_FRAMES = 8
SIGMA = 24.0            # px spread of the predicted ball on the cell map (about one ball radius)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_tapnext(device: str):
    if not TAPNET_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/google-deepmind/tapnet.git",
                        str(TAPNET_DIR)], check=True)
    if not CHECKPOINT.exists():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)
    sys.path.insert(0, str(TAPNET_DIR))
    from tapnet.tapnext.tapnext_torch import TAPNext
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint
    model = TAPNext(image_size=(TAP_SIZE, TAP_SIZE))
    return restore_model_from_jax_checkpoint(model, str(CHECKPOINT)).to(device).eval()


@torch.no_grad()
def track_context(model, context: list[np.ndarray], scene: dict, device: str) -> dict:
    """Ball centre per context frame (512 px coords, None if lost) from 5 tracked points."""
    det = [detect_ball(f, scene) for f in context]
    areas = [d["area"] for d in det if d["visible"]]
    if not areas:
        return {"centres": [None] * len(context), "radius": 20.0, "query_frame": None}
    full = 0.6 * max(areas)
    q = next(i for i, d in enumerate(det) if d["visible"] and d["area"] >= full)
    cx, cy, r = det[q]["x"], det[q]["y"], det[q]["radius"]
    s = TAP_SIZE / CLIP_SIZE
    offsets = [(0, 0), (0.45, 0), (-0.45, 0), (0, 0.45), (0, -0.45)]
    queries = torch.tensor([[[0.0, (cy + dy * r) * s, (cx + dx * r) * s] for dx, dy in offsets]],
                           dtype=torch.float32, device=device)                        # (t, y, x)
    video = np.stack([cv2.resize(f, (TAP_SIZE, TAP_SIZE), interpolation=cv2.INTER_AREA) for f in context])
    video = torch.from_numpy(video.astype(np.float32) / 255 * 2 - 1)[None].to(device)
    centres = [None] * len(context)
    state = None
    for f in range(q, len(context)):
        if state is None:
            tracks, _, vis_logits, state = model(video=video[:, f:f + 1], query_points=queries)
        else:
            tracks, _, vis_logits, state = model(video=video[:, f:f + 1], state=state)
        yx = tracks[0, 0].float().cpu().numpy() / s                                    # [points, (y, x)]
        vis = vis_logits[0, 0, :, 0].float().cpu().numpy() > 0
        pts = yx[vis] if vis.any() else yx
        offset = np.array([[dy * r, dx * r] for dx, dy in offsets])[vis if vis.any() else slice(None)]
        y, x = (pts - offset).mean(0)                                                   # back to the centre
        centres[f] = (float(x), float(y))
    return {"centres": centres, "radius": float(r), "query_frame": q}


def centre_line(scene: dict):
    poly = scene["occluder_polygon"]
    ys = [p[1] for p in poly]
    top = [p for p in poly if p[1] == min(ys)]
    bot = [p for p in poly if p[1] == max(ys)]
    return (np.array([np.mean([p[0] for p in top]), min(ys)]), np.array([np.mean([p[0] for p in bot]), max(ys)]))


def continue_path(track: dict, scene: dict, n_target: int) -> dict:
    """Constant-velocity extension of the last tracked centres, and where it meets the plank's centre line."""
    pts = [(k, c) for k, c in enumerate(track["centres"]) if c is not None][-FIT_FRAMES:]
    n_ctx = len(track["centres"])
    if len(pts) < 2:
        last = pts[-1][1] if pts else (CLIP_SIZE / 2, CLIP_SIZE / 2)
        return {"anchor": np.array(last), "velocity": np.zeros(2), "crossing_y": None, "crossing_t": None,
                "positions": [np.array(last)] * n_target}
    t = np.array([k for k, _ in pts], float)
    xy = np.array([c for _, c in pts])
    v = np.polyfit(t, xy, 1)[0]                                    # px / frame, both axes
    anchor = np.polyval(np.polyfit(t, xy, 1), n_ctx - 1) if len(pts) > 2 else xy[-1]
    a, b = centre_line(scene)
    # anchor + v*s = a + (b - a)*u  ->  solve for s (frames ahead) and u (along the centre line)
    m = np.array([v, a - b]).T
    crossing_y = crossing_t = None
    if abs(np.linalg.det(m)) > 1e-6:
        s_, u = np.linalg.solve(m, a - anchor)
        if s_ > 0 and 0 <= u <= 1:
            crossing_t, crossing_y = float(s_), float(anchor[1] + v[1] * s_)
    return {"anchor": anchor, "velocity": v, "crossing_y": crossing_y, "crossing_t": crossing_t,
            "positions": [anchor + v * (k + 1) for k in range(n_target)]}


def apply_rules(path: dict, radius: float, scene: dict, blockade: tuple[float, float] | None) -> list[dict]:
    """Per target frame: centre, and whether the ball is visible under the scene rules."""
    occluder = np.array(scene["occluder_polygon"], np.float32)
    blocked = (blockade is not None and path["crossing_y"] is not None
               and blockade[0] <= path["crossing_y"] <= blockade[1])
    out = []
    for k, p in enumerate(path["positions"]):
        if blocked and (k + 1) >= path["crossing_t"]:
            p = path["anchor"] + path["velocity"] * path["crossing_t"]           # stopped at the blockade
        inside = cv2.pointPolygonTest(occluder, (float(p[0]), float(p[1])), True) > -0.3 * radius
        in_frame = -radius < p[0] < CLIP_SIZE + radius and -radius < p[1] < CLIP_SIZE + radius
        out.append({"x": float(p[0]), "y": float(p[1]), "visible": bool(in_frame and not inside),
                    "blocked": bool(blocked and (k + 1) >= path["crossing_t"])})
    return out


def fit_blockade(train: list[dict]) -> tuple[float, float] | None:
    """Interval of centre-line crossing heights that best separates blocked (hidden or bounce) from
    through on training clips."""
    ys = sorted({r["crossing_y"] for r in train if r["crossing_y"] is not None})
    if not ys:
        return None
    best, best_acc = None, -1
    cands = [ys[0] - 1] + [(a + b) / 2 for a, b in zip(ys, ys[1:])] + [ys[-1] + 1]
    for i, lo in enumerate(cands):
        for hi in cands[i + 1:]:
            pred = [r["crossing_y"] is not None and lo <= r["crossing_y"] <= hi for r in train]
            acc = np.mean([p == (r["outcome"] != "through") for p, r in zip(pred, train)])
            if acc > best_acc:
                best, best_acc = (lo, hi), acc
    return best


def cell_maps(frames: list[dict], n_ctx: int) -> np.ndarray:
    """[steps, GRID, GRID] soft ball maps from per-frame predictions (max over each 2-frame step)."""
    cell = CLIP_SIZE / GRID
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    cx, cy = (xx + 0.5) * cell, (yy + 0.5) * cell
    maps = np.zeros((len(frames) // STEP, GRID, GRID))
    for k, f in enumerate(frames):
        if f["visible"]:
            m = np.exp(-((cx - f["x"]) ** 2 + (cy - f["y"]) ** 2) / (2 * SIGMA ** 2))
            maps[k // STEP] = np.maximum(maps[k // STEP], m)
    return maps


def render(context: list[np.ndarray], track: dict, frames: list[dict], scene: dict,
           scene_cfg: dict) -> list[np.ndarray]:
    """Pixel prediction: the ball sprite from the last context frame, moved along the predicted path,
    drawn behind the plank, on that frame with the ball painted out."""
    last = context[-1]
    det = detect_ball(last, scene_cfg)
    r = int(round(det["radius"] if det["visible"] else track["radius"]))
    bg = last.copy()
    sprite = None
    if det["visible"]:
        x0, y0 = int(det["x"]), int(det["y"])
        mask = np.zeros(last.shape[:2], np.uint8)
        cv2.circle(mask, (x0, y0), int(r * 1.5), 255, -1)
        bg = cv2.inpaint(last, mask, 5, cv2.INPAINT_TELEA)
        pad = np.pad(last, ((r, r), (r, r), (0, 0)), mode="edge")
        sprite = pad[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1].copy()
    occluder = np.zeros(last.shape[:2], np.uint8)
    cv2.fillPoly(occluder, [np.array(scene["occluder_polygon"], np.int32)], 1)
    disc = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(disc, (r, r), r, 1, -1)
    out = []
    for f in frames:
        im = bg.copy()
        if sprite is not None and not f.get("blocked"):
            x, y = int(round(f["x"])), int(round(f["y"]))
            ys, xs = slice(max(y - r, 0), min(y + r + 1, CLIP_SIZE)), slice(max(x - r, 0), min(x + r + 1, CLIP_SIZE))
            sy, sx = slice(ys.start - (y - r), ys.stop - (y - r)), slice(xs.start - (x - r), xs.stop - (x - r))
            if ys.stop > ys.start and xs.stop > xs.start:
                m = (disc[sy, sx] & (1 - occluder[ys, xs])).astype(bool)          # behind the plank
                im[ys, xs][m] = sprite[sy, sx][m]
        out.append(im)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "tapnext_rule")
    p.add_argument("--render", type=Path, default=REPO / "data" / "eval" / "sample5",
                   help="eval sample to render pixel predictions for")
    p.add_argument("--pred-root", type=Path, default=REPO / "outputs" / "predictions")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = pick_device()
    t0 = time.perf_counter()
    model = load_tapnext(device)
    t_load = time.perf_counter() - t0
    clips = included_clips()
    video = video_of(clips)
    scene_cfg = load_scene(REPO / "configs" / "scenes" / f"{video}.json")
    track_gt, passes, scene_json = load_tracking(video)
    far, near = far_cells(scene_json), near_cells(scene_json)
    n_ctx = clips[0]["context_frames"][1] + 1
    n_target = clips[0]["target_frames"][1] - clips[0]["target_frames"][0] + 1
    print(f"TAPNext + rules on {device}: {len(clips)} clips, model load {t_load:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "tracks.json"
    tracks = json.loads(cache.read_text()) if cache.exists() else {}
    timings, records = [], []
    for i, c in enumerate(clips):
        frames = read_video(REPO / c["path"])
        if c["clip_id"] not in tracks:
            sync(device)
            t0 = time.perf_counter()
            tracks[c["clip_id"]] = track_context(model, frames[:n_ctx], scene_cfg, device)
            sync(device)
            timings.append(time.perf_counter() - t0)
        tr = tracks[c["clip_id"]]
        path = continue_path(tr, scene_json, n_target)
        lab, _ = ball_labels(source_indices(c, passes, len(frames), n_ctx), track_gt)
        records.append({"clip": c, "track": tr, "path": path, "labels": lab,
                        "crossing_y": path["crossing_y"], "outcome": c["outcome"]})
        print(f"\r  tracked {i + 1}/{len(clips)}", end="", flush=True)
    print()
    cache.write_text(json.dumps(tracks))

    results = {}
    for variant in ("continue", "blockade"):
        rows, fut_lab, fut_map = [None] * len(records), [None] * len(records), [None] * len(records)
        blockades = []
        for tr_idx, te_idx in cv_folds(clips, args.folds, args.seed):
            blockade = fit_blockade([records[i] for i in tr_idx]) if variant == "blockade" else None
            blockades.append(blockade)
            for i in te_idx:
                rec = records[i]
                frames_pred = apply_rules(rec["path"], rec["track"]["radius"], scene_json, blockade)
                maps = cell_maps(frames_pred, n_ctx)
                ro = readouts(maps, far, near)
                rows[i] = {"clip_id": rec["clip"]["clip_id"], "outcome": rec["outcome"],
                           "crossing_y": rec["crossing_y"], **ro}
                fut_lab[i], fut_map[i] = rec["labels"][n_ctx // STEP:], maps
        m = summary_metrics(rows, fut_lab, fut_map)
        m["blockade_per_fold"] = [list(map(lambda v: round(v, 1), b)) if b else None for b in blockades]
        results[variant] = {"metrics": m, "rows": rows}
        print(f"  {variant:9s} where the ball goes: cell AUROC {m['future_cell_auroc']}; "
              f"outcome AUROC {m['outcome_auroc']}, accuracy {m['outcome_accuracy']}; "
              f"peak P(beyond plank) {m['peak_far_mean']}")
    if timings:
        print(f"  tracking time per clip on {device}: mean {np.mean(timings):.2f}s (n={len(timings)}), "
              f"model load {t_load:.1f}s")
    (args.out / "metrics.json").write_text(json.dumps({
        "device": device, "model_load_seconds": round(t_load, 1),
        "track_seconds_per_clip": round(float(np.mean(timings)), 2) if timings else None,
        **{v: r["metrics"] for v, r in results.items()}}, indent=1))
    (args.out / "per_clip.json").write_text(json.dumps({v: r["rows"] for v, r in results.items()}, indent=1))

    # Pixel predictions for the eval sample (blockade fitted on every clip outside the sample).
    if args.render:
        sample = json.loads((args.render / "sample.json").read_text())
        ids = {c["clip_id"] for c in sample["clips"]}
        blockade = fit_blockade([r for r in records if r["clip"]["clip_id"] not in ids])
        for c in sample["clips"]:
            frames = read_video(args.render / "clips" / f"{c['clip_id']}.mp4")
            context = frames[:n_ctx]
            tr = tracks.get(c["clip_id"]) or track_context(model, context, scene_cfg, device)
            path = continue_path(tr, scene_json, n_target)
            for variant, b in (("tapnext_continue", None), ("tapnext_blockade", blockade)):
                pred = apply_rules(path, tr["radius"], scene_json, b)
                label = "TAPNext + continuation" + (" + blockade" if b else "")
                write_prediction(args.pred_root / sample["name"] / variant, c["clip_id"],
                                 render(context, tr, pred, scene_json, scene_cfg), context, list(range(n_ctx)),
                                 {"model": variant, "label": label, "blockade": b,
                                  "crossing_y": path["crossing_y"],
                                  "velocity_px_per_frame": [round(float(v), 2) for v in path["velocity"]]})
        print(f"  rendered {len(sample['clips'])} sample clips -> {args.pred_root / sample['name']}/tapnext_*"
              f" (blockade {tuple(round(v, 1) for v in blockade) if blockade else None})")
    print(f"-> {args.out}/ (metrics.json, per_clip.json, tracks.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

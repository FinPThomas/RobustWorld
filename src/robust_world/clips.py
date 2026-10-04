"""Stage: cut one context/target clip per pass, centred just before the ball goes under.

Clip layout (OUT_FPS, 2 * CTX_FRAMES frames):
    frames 0..CTX-1     context: ball rolling in; the last one is the final frame before
                        the ball's outline touches the occluder
    frames CTX..2CTX-1  target: what the world model should predict (through / bounce / hidden)

Every outcome is kept, including the ball never re-emerging (blocked under the occluder).
A clip is excluded only when the context half is unusable (window runs off the video,
another ball in it, ball didn't roll in from the frame edge) or a hand touches the ball
after the throw: it must roll untouched for the last FREE_ROLL context frames and the
whole target. Other hands are kept and saved as per-frame masks (<clip>_hand.npz,
bool [frames, H, W]) for masking out of the loss or inpainting later.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .media import read_frames, write_video
from .paths import CLIPS_ROOT, rel
from .scene import BALL_EXCLUDE, background_for, fill_mask, foreground_mask

OUT_FPS = 16
CTX_FRAMES = 16                    # context half; the target half is the same length
HAND_MIN_PX = 2500                 # foreground pixels in a frame that count as a hand in shot
FREE_ROLL = 8                      # context frames before the midpoint the ball must roll untouched
TOUCH_RADII = BALL_EXCLUDE + 1.5   # hand pixels within this many ball radii count as touching it
SHEET_FRAMES = [0, 5, 10, 15, 16, 21, 26, 31]


def contact_sheet(frames: list[np.ndarray], masks: np.ndarray | None, label: str) -> np.ndarray:
    tiles = []
    for k in SHEET_FRAMES:
        t = frames[k].copy()
        if masks is not None and masks[k].any():
            t[masks[k] > 0] = (0.5 * t[masks[k] > 0] + [127, 0, 127]).astype(np.uint8)
        t = cv2.resize(t, (192, 192), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(k), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(t)
        if k == CTX_FRAMES - 1:
            tiles.append(np.full((192, 6, 3), (255, 0, 0), np.uint8))  # context | target divider
    sheet = np.hstack(tiles)
    bar = np.zeros((22, sheet.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, sheet])


def run(name: str, video: Path, scene: dict, track: list[dict], passes: list[dict], fps: float,
        bg_npz: Path, out_dir: Path) -> list[dict]:
    bgz = np.load(bg_npz)
    bgs, bg_starts = bgz["backgrounds"], bgz["segment_starts"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sheets").mkdir(exist_ok=True)
    step = fps / OUT_FPS
    n = 2 * CTX_FRAMES

    records, sheets = [], []
    for ps in passes:
        mid = ps["occlusion_start_frame"] - 1
        src_idx = [int(round(mid + (k - (CTX_FRAMES - 1)) * step)) for k in range(n)]
        first, last = src_idx[0], src_idx[-1]
        clip_id = f"{name}_{ps['id']:04d}"
        base = {"clip_id": clip_id, "source_video": rel(video), "pass_id": ps["id"]}
        if first < 0 or last >= len(track):
            records.append({**base, "include": False, "reasons": ["window outside video"]})
            continue

        raw = read_frames(video, first, last - first + 1, fps)
        frames = [raw[i - first] for i in src_idx]

        masks = np.zeros((n, *frames[0].shape[:2]), np.uint8)
        touch = []
        for k, i in enumerate(src_idx):
            ball = track[i]
            m = foreground_mask(frames[k], background_for(i, bgs, bg_starts), ball, scene)
            if m.sum() < HAND_MIN_PX:
                continue
            masks[k] = fill_mask(m)
            if ball["visible"]:
                ring = np.zeros_like(m)
                cv2.circle(ring, (int(ball["x"]), int(ball["y"])), int(ball["radius"] * TOUCH_RADII), 1, -1)
                if (ring & m).any():
                    touch.append(k)
        hand_frames = [k for k in range(n) if masks[k].any()]

        def to_clip(src):
            if src is None or src < first or src > last:
                return None
            return int(np.argmin([abs(i - src) for i in src_idx]))

        reasons = []
        if ps["prev_ball_frame"] is not None and ps["prev_ball_frame"] >= first:
            reasons.append("other ball in context")
        if not ps["entry_at_edge"]:
            reasons.append("ball did not roll in from frame edge")
        if any(k >= CTX_FRAMES - FREE_ROLL for k in touch):
            reasons.append("hand touches ball after release")

        reappear = to_clip(ps["reappear_frame"])
        clip_path = out_dir / f"{clip_id}.mp4"
        write_video(clip_path, frames, OUT_FPS)
        mask_path = None
        if hand_frames:
            mask_path = out_dir / f"{clip_id}_hand.npz"
            np.savez_compressed(mask_path, masks=masks.astype(bool))

        rec = {
            **base,
            "include": not reasons,
            "reasons": reasons,
            "path": rel(clip_path),
            "hand_mask_path": rel(mask_path) if mask_path else None,
            "fps": OUT_FPS,
            "n_frames": n,
            "context_frames": [0, CTX_FRAMES - 1],
            "target_frames": [CTX_FRAMES, n - 1],
            "outcome": "hidden" if reappear is None else ps["outcome"],
            "pass_outcome": ps["outcome"],
            "ball_at_start": bool(track[first]["visible"]),
            "entry_frame": to_clip(ps["entry_frame"]),
            "occlusion_start_frame": to_clip(ps["occlusion_start_frame"]),
            "hidden_frame": to_clip(ps["hidden_frame"]),
            "reappear_frame": reappear,
            "exit_frame": to_clip(ps["exit_frame"]),
            "hand_in_context": any(k < CTX_FRAMES for k in hand_frames),
            "hand_in_target": any(k >= CTX_FRAMES for k in hand_frames),
            "hand_touch_frames": touch,
            "other_ball_in_target": ps["next_ball_frame"] is not None and ps["next_ball_frame"] <= last,
            "blocker_polygon": scene.get("blocker_polygon"),
            "source_frames": [first, last],
            "source_fps": fps,
        }
        records.append(rec)

        label = (f"{clip_id}  {rec['outcome']}  ball@0={'y' if rec['ball_at_start'] else 'n'}"
                 f"{'  hand' if hand_frames else ''}{'  EXCLUDED: ' + ', '.join(reasons) if reasons else ''}")
        sheet = contact_sheet(frames, masks if hand_frames else None, label)
        cv2.imwrite(str(out_dir / "sheets" / f"{clip_id}.png"), sheet[:, :, ::-1])
        sheets.append(cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        print(f"\r  {len(records)}/{len(passes)} clips", end="", flush=True)
    print()

    with (out_dir / "manifest.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    if sheets:
        cv2.imwrite(str(out_dir / "review.jpg"), np.vstack(sheets)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])

    kept = [r for r in records if r["include"]]
    print(f"  {len(kept)} clips included, {len(records) - len(kept)} excluded")
    print("  outcomes: " + ", ".join(f"{sum(r['outcome'] == o for r in kept)} {o}"
                                     for o in ("through", "hidden", "bounce")))
    print(f"  ball visible in first frame: {sum(r['ball_at_start'] for r in kept)}/{len(kept)}; "
          f"hand masked in context {sum(r['hand_in_context'] for r in kept)}, "
          f"in target {sum(r['hand_in_target'] for r in kept)}")
    reasons = {}
    for r in records:
        for reason in r["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    for reason, c in reasons.items():
        print(f"  excluded: {reason} ({c})")
    print(f"  -> {rel(out_dir)}/ (clips, manifest.jsonl, sheets/, review.jpg)")
    return records


def merge_manifests(root: Path = CLIPS_ROOT) -> Path:
    """Concatenate every per-video manifest into <root>/manifest.jsonl."""
    out = root / "manifest.jsonl"
    with out.open("w") as f:
        for m in sorted(root.glob("*/manifest.jsonl")):
            f.write(m.read_text())
    return out

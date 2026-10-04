"""Stage: cut one context/target clip per pass, centred just before the ball goes under.

Clip layout (OUT_FPS, 2 * CTX_FRAMES frames):
    frames 0..CTX-1     context: ball rolling in; the last one is the final frame before
                        the ball's outline touches the occluder
    frames CTX..2CTX-1  target: what the world model should predict (through / bounce / hidden)

Every outcome is kept: through, bounce (back out on the entry side) and hidden (blocked
under the occluder). Each clip goes into one set:
    main      usable as is (include: true)
    hand      usable except that a hand is in shot somewhere in the clip. Set aside in
              <segment>/hand/ with per-frame masks (<clip>_hand.npz, bool [frames, H, W])
              for masking or inpainting later.
    excluded  the context half is unusable (window runs off the video or its segment,
              another ball in it, ball didn't roll in from the frame edge or came from the
              wrong side for the segment) or a hand touches the ball after the throw: it must
              roll untouched for the last FREE_ROLL context frames and the whole target.

Clips are split by the scene's "segments" (time ranges of the recording, e.g. ball thrown
from the right early on and from the left later). A clip's whole window must lie inside
one segment, so segments never share frames. Each segment has its own folder and manifest.
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
CTX_FRAMES = 24                    # context half (1.5 s); the target half is the same length
FREE_ROLL = 8                      # context frames before the midpoint the ball must roll untouched
TOUCH_RADII = BALL_EXCLUDE + 1.5   # hand pixels within this many ball radii count as touching it
SHEET_FRAMES = [0, 8, 16, 23, 24, 31, 39, 47]


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


def segments_of(scene: dict) -> list[dict]:
    """The scene's time segments; the whole video is one segment "all" if none are set."""
    return scene.get("segments") or [{"name": "all", "side_in": None, "start_s": 0, "end_s": None}]


def run(name: str, video: Path, scene: dict, track: list[dict], passes: list[dict], fps: float,
        bg_npz: Path, out_dir: Path) -> list[dict]:
    """Cut clips for every segment into out_dir/<segment>/. Returns all records."""
    records = []
    for seg in segments_of(scene):
        lo = int(round(seg["start_s"] * fps))
        hi = len(track) - 1 if seg.get("end_s") is None else int(round(seg["end_s"] * fps))
        inside = [ps for ps in passes if lo <= ps["occlusion_start_frame"] <= hi]
        print(f"  segment {seg['name']}: {seg['start_s']}s-{seg.get('end_s') or 'end'}s, "
              f"{len(inside)} passes" + (f", ball rolled in from {seg['side_in']}" if seg.get("side_in") else ""))
        records += run_segment(name, seg, (lo, hi), video, scene, track, inside, fps, bg_npz, out_dir / seg["name"])
    return records


def run_segment(name: str, seg: dict, bounds: tuple[int, int], video: Path, scene: dict, track: list[dict],
                passes: list[dict], fps: float, bg_npz: Path, out_dir: Path) -> list[dict]:
    bgz = np.load(bg_npz)
    bgs, bg_starts = bgz["backgrounds"], bgz["segment_starts"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sheets").mkdir(exist_ok=True)
    (out_dir / "hand").mkdir(exist_ok=True)
    step = fps / OUT_FPS
    n = 2 * CTX_FRAMES

    records, sheets = [], []
    for ps in passes:
        mid = ps["occlusion_start_frame"] - 1
        src_idx = [int(round(mid + (k - (CTX_FRAMES - 1)) * step)) for k in range(n)]
        first, last = src_idx[0], src_idx[-1]
        clip_id = f"{name}_{ps['id']:04d}"
        base = {"clip_id": clip_id, "source_video": rel(video), "segment": seg["name"], "pass_id": ps["id"],
                "side_in": ps["side_in"]}
        if first < 0 or last >= len(track):
            records.append({**base, "include": False, "set": "excluded", "reasons": ["window outside video"]})
            continue
        if first < bounds[0] or last > bounds[1]:
            records.append({**base, "include": False, "set": "excluded", "reasons": ["window outside segment"]})
            continue
        raw = read_frames(video, first, last - first + 1, fps)
        frames = [raw[i - first] for i in src_idx]

        masks = np.zeros((n, *frames[0].shape[:2]), np.uint8)
        touch = []
        for k, i in enumerate(src_idx):
            ball = track[i]
            m = foreground_mask(frames[k], background_for(i, bgs, bg_starts), ball, scene)
            if m.sum() < scene["hand_min_px"]:
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
        if seg.get("side_in") and ps["side_in"] != seg["side_in"]:
            reasons.append(f"rolled in from {ps['side_in']}, segment is {seg['side_in']}")
        if any(k >= CTX_FRAMES - FREE_ROLL for k in touch):
            reasons.append("hand touches ball after release")
        clip_set = "excluded" if reasons else "hand" if hand_frames else "main"
        if clip_set == "hand":
            reasons = ["hand in shot"]

        reappear = to_clip(ps["reappear_frame"])
        clip_dir = out_dir / "hand" if clip_set == "hand" else out_dir
        clip_path = clip_dir / f"{clip_id}.mp4"
        write_video(clip_path, frames, OUT_FPS)
        mask_path = None
        if hand_frames:
            mask_path = clip_dir / f"{clip_id}_hand.npz"
            np.savez_compressed(mask_path, masks=masks.astype(bool))

        rec = {
            **base,
            "include": clip_set == "main",
            "set": clip_set,
            "reasons": reasons,
            "path": rel(clip_path),
            "hand_mask_path": rel(mask_path) if mask_path else None,
            "fps": OUT_FPS,
            "n_frames": n,
            "context_frames": [0, CTX_FRAMES - 1],
            "target_frames": [CTX_FRAMES, n - 1],
            "outcome": "hidden" if reappear is None else ps["outcome"],
            "pass_outcome": ps["outcome"],
            "touch_only": ps.get("touch_only", False),
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
                 f"{'  hand' if hand_frames else ''}"
                 f"{'  ' + clip_set.upper() + ': ' + ', '.join(reasons) if reasons else ''}")
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

    for name_, members in (("main", [r for r in records if r["set"] == "main"]),
                           ("hand (set aside)", [r for r in records if r["set"] == "hand"])):
        print(f"  {name_}: {len(members)} clips; " + ", ".join(
            f"{sum(r['outcome'] == o for r in members)} {o}" for o in ("through", "bounce", "hidden")))
    reasons = {}
    for r in records:
        if r["set"] == "excluded":
            for reason in r["reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
    print(f"  excluded: {sum(r['set'] == 'excluded' for r in records)} clips"
          + "".join(f"\n    {reason} ({c})" for reason, c in reasons.items()))
    print(f"  -> {rel(out_dir)}/ (clips, hand/, manifest.jsonl, sheets/, review.jpg)")
    return records


def merge_manifests(root: Path = CLIPS_ROOT) -> list[Path]:
    """Concatenate every video's manifest for each segment into <root>/manifest_<segment>.jsonl.
    Segments are never merged with each other."""
    outs = []
    for seg in sorted({m.parent.name for m in root.glob("*/*/manifest.jsonl")}):
        out = root / f"manifest_{seg}.jsonl"
        with out.open("w") as f:
            for m in sorted(root.glob(f"*/{seg}/manifest.jsonl")):
                f.write(m.read_text())
        outs.append(out)
    return outs

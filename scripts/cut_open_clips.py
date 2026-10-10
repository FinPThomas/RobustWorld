"""Cut "open table" clips: 3 s windows where the ball rolls on open table across the context/target boundary.

For the slope test (models/vjepa2/slope_test.py): the plank clips reach the plank right at the boundary, so
the model predicts almost no open-table rolling there. These windows are chosen from the tracker alone
(track.csv), never from outcomes or any model: the ball is on open table (away from the plank outline and the
frame edge) in the last three context steps and in a run of at least 6 of the 12 target steps, and no hand is
in shot anywhere in the window. Windows lie inside one segment; consecutive windows' targets don't overlap.
Same layout as the plank clips (16 fps, 24 + 24 frames, 512 px, same encoder settings).

The clips are scored only (ball_probe_cv.py --extra-manifest), never trained on or used to fit a decoder.
"side_in" is the side the ball rolls from (R = rolling left = downhill).

    python scripts/cut_open_clips.py               # -> data/processed/clips/whole/open/, manifest_open.jsonl,
                                                   #    outputs/robustworld_clips_open.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from robust_world import track as trk  # noqa: E402
from robust_world.media import read_frames, write_video  # noqa: E402
from robust_world.scene import load_scene  # noqa: E402

OUT_FPS, N_FRAMES, CTX = 16, 48, 24
OPEN_PX, EDGE_PX = 48, 24             # as slope_test.py
MIN_TARGET_RUN = 6                    # target steps in a row with the ball on open table
CTX_OPEN = 3                          # last context steps with the ball on open table


def windows(rows: list[dict], scene: dict, fps: float) -> list[tuple[int, str]]:
    poly = np.array(scene["occluder_polygon"], np.float32)
    ok = np.array([bool(r["visible"]) and r["x"] is not None and EDGE_PX < r["x"] < 512 - EDGE_PX
                   and EDGE_PX < r["y"] < 512 - EDGE_PX
                   and -cv2.pointPolygonTest(poly, (float(r["x"]), float(r["y"])), True) > OPEN_PX for r in rows])
    hand = np.array([(r.get("fg_area") or 0) >= scene["hand_min_px"] for r in rows])
    bounds = [(int(s["start_s"] * fps), int(s["end_s"] * fps) if s.get("end_s") else len(rows) - 1, s["name"])
              for s in scene.get("segments", [{"start_s": 0, "name": "all"}])]
    step = fps / OUT_FPS
    out, s = [], 0
    while True:
        idx = [int(round(s + k * step)) for k in range(N_FRAMES)]
        if idx[-1] >= len(rows):
            break
        seg = [name for a, b, name in bounds if a <= idx[0] and idx[-1] <= b]
        steps = [ok[idx[2 * k]] and ok[idx[2 * k + 1]] for k in range(N_FRAMES // 2)]
        tgt, run, best = steps[CTX // 2:], 0, 0
        for o in tgt:
            run = run + 1 if o else 0
            best = max(best, run)
        if seg and not hand[idx[0]:idx[-1] + 1].any() and all(steps[CTX // 2 - CTX_OPEN:CTX // 2]) \
                and best >= MIN_TARGET_RUN:
            out.append((s, seg[0]))
            s += int(round(CTX * step))      # the next window's target starts after this one's
        else:
            s += 4
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="whole")
    p.add_argument("--zip", type=Path, default=REPO / "outputs" / "robustworld_clips_open.zip")
    args = p.parse_args(argv)
    interim = REPO / "data" / "interim" / args.video
    src = interim / f"{args.video}_512.mp4"
    rows = trk.load(interim / "track.csv")
    fps = round(1 / (rows[1]["t"] - rows[0]["t"]), 3)
    scene = load_scene(REPO / "configs" / "scenes" / f"{args.video}.json")
    wins = windows(rows, scene, fps)
    out_dir = REPO / "data" / "processed" / "clips" / args.video / "open"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = fps / OUT_FPS
    records = []
    for n, (s, seg) in enumerate(wins):
        idx = [int(round(s + k * step)) for k in range(N_FRAMES)]
        raw = read_frames(src, idx[0], idx[-1] - idx[0] + 1, fps)
        frames = [raw[i - idx[0]] for i in idx]
        clip_id = f"{args.video}_open_{n:04d}"
        path = out_dir / f"{clip_id}.mp4"
        write_video(path, frames, OUT_FPS)
        x0, x1 = rows[idx[CTX - 6]]["x"], rows[idx[CTX - 1]]["x"]
        records.append({"clip_id": clip_id, "source_video": f"data/interim/{args.video}/{args.video}_512.mp4",
                        "segment": seg, "pass_id": None, "side_in": "R" if x1 < x0 else "L", "include": True,
                        "set": "open", "reasons": [], "path": path.relative_to(REPO).as_posix(),
                        "hand_mask_path": None, "fps": OUT_FPS, "n_frames": N_FRAMES,
                        "context_frames": [0, CTX - 1], "target_frames": [CTX, N_FRAMES - 1], "outcome": "open",
                        "source_frames": [idx[0], idx[-1]], "source_fps": fps})
        print(f"\r  {n + 1}/{len(wins)} clips", end="", flush=True)
    print()
    man = REPO / "data" / "processed" / "clips" / "manifest_open.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in records))
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(man, man.relative_to(REPO).as_posix())
        for r in records:
            z.write(REPO / r["path"], r["path"])
    sides = {k: sum(r["side_in"] == k for r in records) for k in "RL"}
    print(f"{len(records)} open-table clips (rolling left / downhill {sides['R']}, right / uphill {sides['L']}) "
          f"-> {man.relative_to(REPO)}, {args.zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

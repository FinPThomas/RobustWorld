"""Raw capture video -> 3 s world-model clips (512x512, 16 fps, context | target).

Stages, each cached and rerun only when its inputs change (or with --from):
    downsize  data/raw/videos/<name>.mp4  -> data/interim/<name>/<name>_512.mp4  (square, no audio)
    track     -> data/interim/<name>/track.csv, backgrounds.npz  (needs configs/scenes/<name>.json)
    passes    -> data/interim/<name>/passes.json
    clips     -> data/processed/clips/<name>/<segment>/  (+ merged data/processed/clips/manifest_<segment>.jsonl)

A new video needs a scene file giving the occluder outline. On first run the
pipeline downsizes the video, writes a gridded calibration.png and a template
configs/scenes/<name>.json, then stops so the outline can be filled in.

Usage:
    robustworld data/raw/videos/start.mp4
    robustworld data/raw/videos/            # every .mp4 in the folder
    robustworld start.mp4 --from track      # force a rerun from a stage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import clips, passes, track
from .media import downsize
from .paths import VideoPaths, rel
from .scene import TRACK_KEYS, load_scene, write_scene_template

STAGES = ["downsize", "track", "passes", "clips"]


def stale(outputs: list[Path], inputs: list[Path]) -> bool:
    if not all(o.exists() for o in outputs):
        return True
    newest_in = max(i.stat().st_mtime for i in inputs)
    return min(o.stat().st_mtime for o in outputs) < newest_in


def tracking_settings(scene: dict) -> dict:
    """The scene settings tracking depends on. Others (segments, hand_min_px) only affect clips."""
    return {k: scene.get(k) for k in TRACK_KEYS}


def process(paths: VideoPaths, force_from: str | None = None) -> bool:
    """Run all stages for one video. Returns False if it stopped for calibration."""
    forced = set(STAGES[STAGES.index(force_from):]) if force_from else set()
    print(f"== {paths.name}")

    if "downsize" in forced or stale([paths.video], [paths.raw]):
        print(f"[downsize] {rel(paths.raw)} -> {paths.size}x{paths.size}")
        t0 = time.time()
        info = downsize(paths.raw, paths.video, size=paths.size)
        mb_in, mb_out = paths.raw.stat().st_size / 1e6, paths.video.stat().st_size / 1e6
        print(f"\n  {info['width']}x{info['height']} @ {info['fps']:g} fps, {info['duration']:.0f}s; "
              f"{mb_in:,.0f} MB -> {mb_out:,.0f} MB in {time.time() - t0:.0f}s")
        forced |= {"track", "passes", "clips"}

    if not paths.scene.exists() or json.loads(paths.scene.read_text()).get("TODO"):
        if not paths.scene.exists():
            write_scene_template(paths.scene, paths.video, paths.calibration)
        print(f"[calibrate] set occluder_polygon in {rel(paths.scene)} using {rel(paths.calibration)}, "
              f"delete the TODO line, then rerun.")
        return False
    scene = load_scene(paths.scene)

    settings = tracking_settings(scene)
    tracked_with = paths.interim / "track_scene.json"
    if ("track" in forced or stale([paths.track, paths.backgrounds, tracked_with], [paths.video])
            or json.loads(tracked_with.read_text()) != settings):
        print("[track]")
        track.run(paths.video, scene, paths.track, paths.backgrounds)
        tracked_with.write_text(json.dumps(settings, indent=1))
        forced |= {"passes", "clips"}
    rows = track.load(paths.track)
    fps = float(np.load(paths.backgrounds)["fps"])

    if "passes" in forced or stale([paths.passes], [paths.track]):
        print("[passes]")
        passes.run(rows, fps, scene, paths.size, paths.passes)
        forced.add("clips")
    found = json.loads(paths.passes.read_text())["passes"]

    manifests = [paths.clips / seg["name"] / "manifest.jsonl" for seg in clips.segments_of(scene)]
    if "clips" in forced or stale(manifests, [paths.passes, paths.scene]):
        print("[clips]")
        clips.run(paths.name, paths.video, scene, rows, found, fps, paths.backgrounds, paths.clips)
    else:
        print(f"[clips] up to date: {rel(paths.clips)}")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robustworld", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="raw .mp4 files or folders of them")
    p.add_argument("--from", dest="force_from", choices=STAGES, help="force rerun from this stage onward")
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args(argv)

    videos = []
    for i in args.inputs:
        videos += sorted(i.glob("*.mp4")) if i.is_dir() else [i]
    if not videos:
        p.error("no .mp4 files found")

    pending = [v.name for v in videos if not process(VideoPaths(v.resolve(), args.size), args.force_from)]
    print()
    for merged in clips.merge_manifests():
        n = sum(1 for _ in merged.open())
        print(f"merged manifest: {rel(merged)} ({n} clips across all videos)")
    if pending:
        print(f"awaiting calibration: {', '.join(pending)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

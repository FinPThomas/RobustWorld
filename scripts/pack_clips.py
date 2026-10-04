"""Zip the processed clips and tracking for training on another machine (e.g. Colab).

Packs one segment's merged manifest (default: right, rolled in from the right) with every
included clip in it, that segment's per-video manifests, and each video's track.csv and
passes.json (needed for the ball labels the evaluation scores with). Clips set aside (hand in
shot) or excluded, and other segments (left is held out), are not packed. Scene files are in the
repo already. Unzip at the repo root on the other machine.

    python scripts/pack_clips.py                     # -> outputs/robustworld_clips.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "clips" / "manifest_right.jsonl")
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "robustworld_clips.zip")
    args = p.parse_args(argv)

    clips = [json.loads(line) for line in args.manifest.open()]
    kept = [c for c in clips if c.get("include")]
    segment = args.manifest.stem.removeprefix("manifest_")
    files = {args.manifest} | set(args.manifest.parent.glob(f"*/{segment}/manifest.jsonl"))
    for c in kept:
        files.add(REPO / c["path"])
        if c.get("hand_mask_path"):
            files.add(REPO / c["hand_mask_path"])
    for video in {Path(c["source_video"]).parent.name for c in kept}:
        for name in ("track.csv", "passes.json"):
            files.add(REPO / "data" / "interim" / video / name)
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} files, e.g. {missing[0]}; rerun `robustworld` first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, f.relative_to(REPO))
    mb = args.out.stat().st_size / 1e6
    print(f"{len(kept)} included clips of {len(clips)} ({segment}) -> {args.out.relative_to(REPO)} ({mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Copy new and changed files from one folder to another (Colab's local disk <-> Google Drive).

The plan works on Colab's local disk, which never drops, and keeps Drive as the backup: Setup copies
Drive -> local once per runtime, and plan.py copies local -> Drive after every step and every few
minutes. A file is copied when it is missing there or newer (never an older one over a newer one); it is
written to "<name>.part" and renamed, so a copy cut off by a Drive drop never leaves a half file.
Files are never deleted, except a fold's per-epoch training state ("*.partial.pt") once that fold is
finished and its state is gone from the source. The encoded-feature cache is left out (plan.py re-encodes it on Colab's
disk in a few minutes), which keeps Drive reads, and so Drive drops, to a minimum.

    python scripts/drive_sync.py <from> <to> [<excluded relative path> ...]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SKIP = (".part", ".tmp")
NOT_BACKED_UP = ("vjepa2/cache",)       # under outputs/: ~2 GB of features, quicker to re-encode than to read from Drive
MIRRORED = ".partial.pt"                # per-epoch training state (~0.4 GB): removed from the copy once the fold is saved


def sync(src: Path, dst: Path, exclude: tuple[str, ...] = ()) -> int:
    """-> number of files copied, leaving out paths under `exclude` (relative to src, e.g. "vjepa2/cache").
    Raises OSError if either side fails (e.g. Drive dropped)."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        return 0
    copied = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.name.endswith(SKIP) or f.name.startswith(".write_test_"):
            continue
        rel = f.relative_to(src)
        if any(rel.as_posix().startswith(e.rstrip("/") + "/") for e in exclude):
            continue
        d = dst / rel
        s = f.stat()
        if d.exists():                       # copy only what is newer here (never older over newer)
            t = d.stat()
            newer = s.st_mtime > t.st_mtime + 1
            if not newer and not (t.st_size != s.st_size and s.st_mtime >= t.st_mtime - 1):
                continue
        d.parent.mkdir(parents=True, exist_ok=True)
        part = d.with_name(d.name + ".part")
        shutil.copy2(f, part)
        part.replace(d)
        copied += 1
    for d in dst.rglob(f"*{MIRRORED}") if dst.is_dir() else ():
        if not (src / d.relative_to(dst)).exists():
            d.unlink(missing_ok=True)
    return copied


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    n = sync(Path(argv[0]), Path(argv[1]), tuple(argv[2:]))
    print(f"copied {n} files {argv[0]} -> {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""scripts/drive_sync.py: the Colab disk <-> Drive backup copies only newer files and never half files."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import drive_sync  # noqa: E402


def test_copies_new_and_newer_never_older_over_newer(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    (local / "a").mkdir(parents=True)
    (local / "a" / "state.json").write_text("new")
    (local / "a" / "half.pt.tmp").write_text("x")              # being written: skipped
    assert drive_sync.sync(local, drive) == 1
    assert (drive / "a" / "state.json").read_text() == "new" and not (drive / "a" / "half.pt.tmp").exists()
    assert drive_sync.sync(local, drive) == 0                   # nothing changed: nothing copied

    old = drive / "a" / "state.json"                            # an older backup must not overwrite newer work
    old.write_text("older backup!")
    t = (local / "a" / "state.json").stat().st_mtime - 100
    os.utime(old, (t, t))
    assert drive_sync.sync(drive, local) == 0 and (local / "a" / "state.json").read_text() == "new"
    assert drive_sync.sync(local, drive) == 1 and old.read_text() == "new"
    assert not list(drive.rglob("*.part"))

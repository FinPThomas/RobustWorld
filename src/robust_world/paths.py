"""Where each pipeline stage reads and writes, per source video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO_ROOT / "configs" / "scenes"
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
CLIPS_ROOT = REPO_ROOT / "data" / "processed" / "clips"


def rel(p: Path) -> str:
    """Repo-relative path for logs and manifests, with forward slashes so manifests written on
    Windows still work on Linux (e.g. Colab)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


@dataclass(frozen=True)
class VideoPaths:
    raw: Path
    size: int = 512

    @property
    def name(self) -> str:
        return self.raw.stem

    @property
    def scene(self) -> Path:
        return SCENES_DIR / f"{self.name}.json"

    @property
    def interim(self) -> Path:
        return INTERIM_ROOT / self.name

    @property
    def video(self) -> Path:
        return self.interim / f"{self.name}_{self.size}.mp4"

    @property
    def calibration(self) -> Path:
        return self.interim / "calibration.png"

    @property
    def track(self) -> Path:
        return self.interim / "track.csv"

    @property
    def backgrounds(self) -> Path:
        return self.interim / "backgrounds.npz"

    @property
    def passes(self) -> Path:
        return self.interim / "passes.json"

    @property
    def clips(self) -> Path:
        return CLIPS_ROOT / self.name

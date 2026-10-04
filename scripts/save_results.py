"""Save a run's results into the repo under results/<date>_<time>_<name>/, never overwriting an
earlier run, and optionally commit and push them to GitHub (used at the end of the Colab notebook).

What is saved: every .json, .md and .png under outputs/vjepa2/ (scores, per-clip readouts,
summary table and figures; not the feature cache) and each run's checkpoints/vjepa2/<run>/log.json
(training curves; not the weights). A README.md in the folder records when, which commit, the grid's
config and the headline table. results/README.md is the index, rebuilt from the folders.

    python scripts/save_results.py --name grid                       # save locally
    GITHUB_TOKEN=... python scripts/save_results.py --name grid --push --branch <branch>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = "results"
MAX_BYTES = 5_000_000                   # skip anything bigger (keeps the repo small)
GITHUB = "github.com/FinPThomas/RobustWorld.git"


def git(*args, cwd=REPO, check=True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def collect(dest: Path, repo: Path = REPO) -> list[Path]:
    """Copy result files (not caches or weights) from outputs/ and checkpoints/ into dest."""
    saved = []
    sources = [(p, p.relative_to(repo / "outputs")) for p in (repo / "outputs" / "vjepa2").rglob("*")
               if p.suffix in (".json", ".md", ".png") and "cache" not in p.parts]
    sources += [(p, p.relative_to(repo)) for p in (repo / "checkpoints" / "vjepa2").glob("*/log.json")]
    for src, rel in sources:
        if src.is_file() and src.stat().st_size <= MAX_BYTES:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)
            saved.append(rel)
    return saved


def describe(dest: Path, run_id: str, name: str, note: str, commit: str) -> None:
    cfg_path = dest / "vjepa2" / "experiments" / "config.json"
    summary = dest / "vjepa2" / "experiments" / "summary.md"
    lines = [f"# {run_id}", "", f"- Name: {name}", f"- Saved: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"- Code commit: `{commit}`"]
    if note:
        lines.append(f"- Note: {note}")
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        lines.append(f"- Grid: variants {', '.join(cfg['variants'])}; {cfg['epochs']} epochs"
                     + (f"; extra `{' '.join(cfg['extra'])}`" if cfg["extra"] else ""))
    if summary.exists():
        lines += ["", "## Headline", "", summary.read_text().strip(), "",
                  "![summary](vjepa2/experiments/summary.png)"]
    figs = sorted(p.relative_to(dest) for p in dest.rglob("*.png"))
    if figs:
        lines += ["", "## Figures", ""] + [f"- [{p}]({p})" for p in figs]
    (dest / "README.md").write_text("\n".join(lines) + "\n")


def headline(folder: Path) -> str:
    """One line for the index: best balanced P(correct) after post-training, if there is a summary."""
    path = folder / "vjepa2" / "experiments" / "summary.json"
    if not path.exists():
        return ""
    rows = [r for r in json.loads(path.read_text()) if r.get("phase") == "after" and r.get("p_correct") is not None]
    if not rows:
        return ""
    best = max(rows, key=lambda r: r["p_correct"])
    return f"best after: {best['variant']} P(correct) {best['p_correct']}, hit rate {best['ball_hit_rate']}"


def write_index(root: Path) -> None:
    runs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    lines = ["# Results", "",
             "One folder per saved run, named `<date>_<time>_<name>` (UTC); folders are never overwritten.",
             "Each holds a README.md (when, which commit, config, headline table), the scores and",
             "figures from `outputs/vjepa2/`, and training logs. Save a run with",
             "`python scripts/save_results.py --name <name> [--note ...] [--push --branch <branch>]`;",
             "the Colab notebook does this at the end. This index is rebuilt from the folders.", "",
             "| run | headline |", "|---|---|"]
    lines += [f"| [{p.name}]({p.name}/README.md) | {headline(p)} |" for p in runs]
    (root / "README.md").write_text("\n".join(lines) + "\n")


def new_id(root: Path, name: str) -> str:
    base = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d_%H%M}_{name}"
    run_id, n = base, 2
    while (root / run_id).exists():
        run_id, n = f"{base}_{n}", n + 1
    return run_id


def save(name: str, note: str = "", repo: Path = REPO) -> Path:
    root = repo / RESULTS
    root.mkdir(exist_ok=True)
    dest = root / new_id(root, name)
    dest.mkdir()
    saved = collect(dest, repo)
    if not saved:
        dest.rmdir()
        raise SystemExit("nothing to save: no results under outputs/vjepa2/")
    describe(dest, dest.name, name, note, git("rev-parse", "--short", "HEAD", cwd=repo, check=False) or "unknown")
    write_index(root)
    print(f"saved {len(saved)} files -> {dest.relative_to(repo)}")
    return dest


def push(dest: Path, branch: str, url: str, token: str = "", tries: int = 4) -> None:
    """Commit dest to `branch` from a fresh shallow clone, so local code edits are never pushed.
    Retries if someone else pushed in between."""
    for attempt in range(1, tries + 1):
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            git("clone", "--depth", "1", "--branch", branch, url, str(clone), cwd=tmp)
            root = clone / RESULTS
            root.mkdir(exist_ok=True)
            target = root / dest.name
            if target.exists():
                raise SystemExit(f"{RESULTS}/{dest.name} is already on {branch}; not overwriting")
            shutil.copytree(dest, target)
            write_index(root)
            git("add", RESULTS, cwd=clone)
            git("-c", "user.name=RobustWorld Colab", "-c", "user.email=colab@robustworld.local",
                "commit", "-q", "-m", f"Results: {dest.name}", cwd=clone)
            r = subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], cwd=clone, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"pushed {RESULTS}/{dest.name} to {branch}")
                return
            err = r.stderr.replace(token, "***") if token else r.stderr
            print(f"push attempt {attempt} failed: {err.strip()[-300:]}")
            time.sleep(2 ** attempt)
    raise SystemExit("could not push; the results are still saved locally")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="run", help="short label, e.g. grid or codes-20ep")
    p.add_argument("--note", default="", help="one line saying what was tried")
    p.add_argument("--push", action="store_true", help="commit and push to GitHub (needs GITHUB_TOKEN)")
    p.add_argument("--branch", default=None, help="branch to push to (default: the current one)")
    args = p.parse_args(argv)
    if args.name != "".join(c for c in args.name if c.isalnum() or c in "-_"):
        raise SystemExit("--name: letters, digits, - and _ only")
    dest = save(args.name, args.note)
    if args.push:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise SystemExit("--push needs GITHUB_TOKEN (a token with write access to the repo)")
        push(dest, args.branch or git("rev-parse", "--abbrev-ref", "HEAD"),
             f"https://x-access-token:{token}@{GITHUB}", token)
    return 0


if __name__ == "__main__":
    sys.exit(main())

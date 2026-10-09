"""scripts/save_results.py: each save gets its own folder, nothing is overwritten, push works."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import save_results  # noqa: E402


def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    exp = repo / "outputs" / "vjepa2" / "experiments"
    exp.mkdir(parents=True)
    (exp / "summary.json").write_text(json.dumps([{"variant": "codes", "phase": "after", "one_ball_p_correct": 0.7,
                                                   "argmax_hit_rate": 0.5}]))
    (exp / "summary.md").write_text("| variant |\n|---|\n| codes |\n")
    (exp / "summary.png").write_bytes(b"png")
    (exp / "config.json").write_text(json.dumps({"variants": ["codes"], "epochs": 2, "extra": []}))
    (repo / "outputs" / "vjepa2" / "cache").mkdir()
    (repo / "outputs" / "vjepa2" / "cache" / "c0.json").write_text("{}")         # never saved
    ck = repo / "checkpoints" / "vjepa2" / "codes-after"
    ck.mkdir(parents=True)
    (ck / "log.json").write_text("[]")
    (ck / "fold0.pt").write_bytes(b"weights")                                      # never saved
    return repo


def test_saves_never_overwrite(tmp_path):
    repo = fake_repo(tmp_path)
    a = save_results.save("grid", "first try", repo)
    b = save_results.save("grid", "", repo)
    assert a != b and a.exists() and b.exists()
    assert (a / "vjepa2" / "experiments" / "summary.png").exists()
    assert (a / "checkpoints" / "vjepa2" / "codes-after" / "log.json").exists()
    assert not list(a.rglob("*.pt")) and not (a / "vjepa2" / "cache").exists()
    assert "first try" in (a / "README.md").read_text()
    index = (repo / "results" / "README.md").read_text()
    assert a.name in index and b.name in index and "codes P(correct, one ball) 0.7" in index


def test_push_to_branch(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    run = lambda *a, cwd=tmp_path: subprocess.run(a, cwd=cwd, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q", "--bare", str(remote))
    run("git", "init", "-q", "-b", "exp", str(work))
    (work / "f.txt").write_text("x")
    run("git", "add", ".", cwd=work)
    run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=work)
    run("git", "push", "-q", str(remote), "exp", cwd=work)

    dest = save_results.save("grid", "", fake_repo(tmp_path))
    save_results.push(dest, "exp", f"file://{remote}")
    log = subprocess.run(["git", "--git-dir", str(remote), "log", "--oneline", "exp"], capture_output=True, text=True)
    assert f"Results: {dest.name}" in log.stdout
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "exp"],
                           capture_output=True, text=True).stdout
    assert f"results/{dest.name}/README.md" in files and "results/README.md" in files


def test_grid_folder_updates_in_place(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    run = lambda *a, cwd=tmp_path: subprocess.run(a, cwd=cwd, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q", "--bare", str(remote))
    run("git", "init", "-q", "-b", "exp", str(work))
    (work / "f.txt").write_text("x")
    run("git", "add", ".", cwd=work)
    run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=work)
    run("git", "push", "-q", str(remote), "exp", cwd=work)

    repo = fake_repo(tmp_path)
    a = save_results.save("night", "", repo, into="2026-10-04_2200_night")
    save_results.push(a, "exp", f"file://{remote}", update=True)
    (repo / "outputs" / "vjepa2" / "experiments" / "summary.md").write_text("| more runs |\n")
    b = save_results.save("night", "", repo, into="2026-10-04_2200_night")
    save_results.push(b, "exp", f"file://{remote}", update=True)
    assert a == b and len([p for p in (repo / "results").iterdir() if p.is_dir()]) == 1
    show = subprocess.run(["git", "--git-dir", str(remote), "show",
                           "exp:results/2026-10-04_2200_night/vjepa2/experiments/summary.md"],
                          capture_output=True, text=True).stdout
    assert "more runs" in show


def test_source_folder_and_status_block(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    run = lambda *a, cwd=tmp_path: subprocess.run(a, cwd=cwd, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q", "--bare", str(remote))
    run("git", "init", "-q", "-b", "exp", str(work))
    (work / "docs").mkdir()
    (work / "docs" / "plan.md").write_text("# Plan\nkeep me\n<!-- status:start -->\nold\n<!-- status:end -->\ntail\n")
    run("git", "add", ".", cwd=work)
    run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=work)
    run("git", "push", "-q", str(remote), "exp", cwd=work)

    repo = fake_repo(tmp_path)
    plan = repo / "outputs" / "vjepa2" / "plan"
    plan.mkdir()
    (plan / "summary.json").write_text(json.dumps([{"variant": "gate", "phase": "after", "one_ball_p_correct": 0.8}]))
    dest = save_results.save("twoday", "", repo, into="2026-10-05_0900_twoday", source="outputs/vjepa2/plan")
    assert (dest / "vjepa2" / "plan" / "summary.json").exists()
    assert not (dest / "vjepa2" / "experiments").exists() and not (dest / "checkpoints").exists()
    save_results.push(dest, "exp", f"file://{remote}", update=True, status=("docs/plan.md", "| step | done |"))
    doc = subprocess.run(["git", "--git-dir", str(remote), "show", "exp:docs/plan.md"],
                         capture_output=True, text=True).stdout
    assert "keep me" in doc and "tail" in doc and "| step | done |" in doc and "old" not in doc
    assert "gate P(correct, one ball) 0.8" in (repo / "results" / "README.md").read_text()

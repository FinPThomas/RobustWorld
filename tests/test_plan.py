"""models/vjepa2/plan.py: steps run in order, finished steps are skipped, failures and missing
recordings don't stop the plan, and the report scores every method the same way."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vjepa2"))
import plan  # noqa: E402


def per_clip(n: int, skill: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        outcome = "through" if i % 2 else "hidden"
        p = float(np.clip(0.3 + skill * (outcome == "through") + 0.2 * rng.standard_normal(), 0, 1))
        row = {"clip_id": f"c{i}", "outcome": outcome}
        for k in ("real", "imagined"):
            row[f"{k}_one_ball_far"], row[f"{k}_one_ball_near"] = [0.0, p], [0.5, 0.1]
        rows.append(row)
    return rows


@pytest.fixture
def fake(tmp_path, monkeypatch):
    for name, value in {"REPO": tmp_path, "PLAN": tmp_path / "plan", "SCORES": tmp_path / "plan" / "scores",
                        "CKPT": tmp_path / "ck", "STATE": tmp_path / "plan" / "state.json",
                        "MANIFEST": tmp_path / "manifest.jsonl", "SEEN": tmp_path / "seen.jsonl"}.items():
        monkeypatch.setattr(plan, name, value)
    monkeypatch.setattr(plan, "save_and_push", lambda state, block, args: None)
    calls, fail = [], set()

    def fake_sh(cmd, log=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if "ball_probe_cv.py" in cmd[1]:
            out = Path(cmd[cmd.index("--out") + 1])
            if out.name in fail:
                return 1
            out.mkdir(parents=True, exist_ok=True)
            skill = 0.0 if out.name in ("pretrained",) or out.name.endswith("-before") else 0.5
            (out / "per_clip.json").write_text(json.dumps(per_clip(20, skill, len(out.name))))
            (out / "metrics.json").write_text(json.dumps({"future_cell_auroc_imagined": 0.7, "real_cell_auroc": 0.99}))
        if cmd[1].endswith("predict.py"):
            d = tmp_path / "outputs" / "tapnext_rule"
            d.mkdir(parents=True, exist_ok=True)
            rows = [{"clip_id": r["clip_id"], "outcome": r["outcome"], "far": r["imagined_one_ball_far"],
                     "near": r["imagined_one_ball_near"]} for r in per_clip(20, 0.6, 1)]
            (d / "per_clip.json").write_text(json.dumps({"continue": rows, "blockade": rows}))
            (d / "metrics.json").write_text(json.dumps({"continue": {}, "blockade": {"future_cell_auroc": 0.8}}))
        return 0

    monkeypatch.setattr(plan, "sh", fake_sh)
    import generalise

    def fake_split(argv):
        (tmp_path / "plan" / "generalise").mkdir(parents=True, exist_ok=True)
        (tmp_path / "plan" / "generalise" / "split.json").write_text(json.dumps({"held_out": {}}))
        return 3
    monkeypatch.setattr(generalise, "main", fake_split)
    return calls, fail


def test_plan_runs_resumes_and_waits(fake, monkeypatch):
    calls, fail = fake
    fail.add("rollout-after")
    plan.main(["run", "--stage", "all"])
    state = json.loads(plan.STATE.read_text())["steps"]
    assert state["plain-after"]["state"] == "done" and state["rollout-after"]["state"] == "failed"
    assert state["generalise"]["state"] == "waiting"                 # no other-side clips yet
    assert "commit-before" not in state and "gate-before" not in state   # same as plain-before
    trained = [c[c.index("--run") + 1] for c in calls if "posttrain.py" in c[1] and "train" in c]
    assert "plan/plain-after" in trained and all("--resume" in c for c in calls if "train" in c)
    best = json.loads(plan.STATE.read_text())["choices"]["best"]
    assert len(best) == 2 and f"plan/{best[0]}-long" in trained and f"plan/{best[0]}-frac25" in trained
    assert any("interpret.py" in c[1] for c in calls)
    order = [" ".join(c) for c in calls]                              # results first: plain is interpreted
    first_interp = next(i for i, c in enumerate(order) if "interpret.py" in c and "plain-after" in c)
    first = lambda run: next(i for i, c in enumerate(order) if run in c)  # noqa: E731
    assert first("plan/plain-after") < first("plan/commit-after") < first("plan/gate-after") < first_interp
    assert first_interp < first("plan/codes-after") < first("plan/codes_rollout-after") < first("-long")
    assert state["generalise-plain"]["state"] == "waiting"

    calls.clear(), fail.clear()
    plan.main(["run", "--stage", "all"])                              # second session: only what's left
    trained = [c[c.index("--run") + 1] for c in calls if "posttrain.py" in c[1] and "train" in c]
    assert trained == ["plan/rollout-after"]

    summary = json.loads((plan.PLAN / "summary.json").read_text())
    after = next(r for r in summary if r["method"] == "V-JEPA 2 plain-after")
    assert after["outcome_auroc"] > 0.9 and after["delta_vs_before_ci"][0] > 0 and after["p_no_gain"] < 0.05
    assert any(r["variant"] == "tapnext_blockade" for r in summary)
    status = (plan.PLAN / "status.md").read_text()
    assert "| 2 | plain-after" in status and "waiting" in status
    assert (plan.PLAN / "summary.png").exists()


def test_interrupted_step_runs_again(fake):
    calls, _ = fake
    plan.save_state({"results_folder": "x", "started": "x", "choices": {},
                     "steps": {"encode": {"state": "running"}, "split": {"state": "done"}}})
    plan.main(["run", "--stage", "1", "--only", "encode", "split"])
    assert sum("encode" in c for c in calls) == 1
    assert json.loads(plan.STATE.read_text())["steps"]["encode"]["state"] == "done"


def test_bootstrap_detects_a_real_gain_only():
    y = np.array([0, 1] * 30)
    s0 = np.random.default_rng(0).random(60)
    assert plan.bootstrap_auroc(y, y + 0.1 * s0, s0)["p_no_gain"] < 0.01
    assert plan.bootstrap_auroc(y, s0, s0)["p_no_gain"] == 1.0


def test_stops_cleanly_when_storage_fails(fake, monkeypatch):
    calls, _ = fake
    monkeypatch.setattr(plan, "storage_ok", lambda: "Transport endpoint is not connected")
    with pytest.raises(SystemExit):
        plan.main(["run", "--stage", "1"])
    assert not calls
    seq = iter([None, None, "Transport endpoint is not connected"])      # Drive drops after the first step
    monkeypatch.setattr(plan, "storage_ok", lambda: next(seq, "gone"))
    plan.main(["run", "--stage", "1", "--only", "encode", "pretrained"])
    steps = json.loads(plan.STATE.read_text())["steps"]
    assert steps["encode"]["state"] == "done" and "pretrained" not in steps


def test_everything_important_is_in_the_results_folder(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import save_results
    src = tmp_path / "outputs" / "vjepa2" / "plan"
    for rel in ("state.json", "status.md", "summary.md", "summary.json", "summary.png", "logs/codes-after.txt",
                "scores/codes-after/metrics.json", "scores/codes-after/per_clip.json", "scores/codes-after/train_log.json",
                "scores/codes-after/curves.png", "blocker/codes-after/crossing.json", "baselines/tapnext/metrics.json",
                "generalise/metrics.json", "generalise/split.json", "interpret/codes-after/interpret.json",
                "interpret/codes-after/change_map.png"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x")
    (src / "cache").mkdir()
    (src / "cache" / "big.pt").write_text("x")
    saved = {str(p) for p in save_results.collect(tmp_path / "dest", tmp_path, "outputs/vjepa2/plan")}
    assert len(saved) == 16 and "vjepa2/plan/logs/codes-after.txt" in saved
    assert not any("cache" in p or p.endswith(".pt") for p in saved)


def test_report_push_failure_is_reported(fake, monkeypatch):
    monkeypatch.setattr(plan, "save_and_push", lambda state, block, args: False)
    assert plan.main(["report", "--push"]) == 1
    monkeypatch.setattr(plan, "save_and_push", lambda state, block, args: True)
    assert plan.main(["report", "--push"]) == 0


def test_drive_drop_inside_a_step_stops_without_marking_it_failed(fake, monkeypatch):
    calls, _ = fake
    monkeypatch.setattr(plan, "DRIVE_RETRY_WAIT", 0)
    real_sh = plan.sh

    def dropping_sh(cmd, log=None):                # encode dies with the error Colab's Drive mount gives
        if "encode" in [str(c) for c in cmd]:
            calls.append([str(c) for c in cmd])
            plan.TAIL.append("OSError: [Errno 107] Transport endpoint is not connected\n")
            return 1
        return real_sh(cmd, log)
    monkeypatch.setattr(plan, "sh", dropping_sh)
    plan.main(["run", "--stage", "1"])
    steps = json.loads(plan.STATE.read_text())["steps"]
    assert steps["encode"]["state"] == "running"   # not failed, not done: redone next time
    assert "pretrained" not in steps and "tapnext" not in steps
    assert sum("encode" in c for c in calls) == 2  # tried once more first


def test_cut_off_earlier_step_runs_first(fake):
    plan.save_state({"results_folder": "x", "started": "x", "choices": {}, "steps": {"encode": {"state": "running"}}})
    plan.main(["run", "--stage", "2", "--only", "plain-before"])   # --only: just that step
    assert json.loads(plan.STATE.read_text())["steps"]["encode"]["state"] == "running"
    plan.main(["run", "--stage", "2"])
    assert json.loads(plan.STATE.read_text())["steps"]["encode"]["state"] == "done"


def test_backs_up_to_drive_after_every_step_and_survives_a_drop(fake, tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    monkeypatch.setenv("ROBUSTWORLD_BACKUP", str(drive))
    monkeypatch.setattr(plan, "PLAN", tmp_path / "outputs" / "plan")      # as in the repo: the plan under outputs/
    monkeypatch.setattr(plan, "STATE", tmp_path / "outputs" / "plan" / "state.json")
    import drive_sync
    real = drive_sync.sync
    calls = {"n": 0}

    def flaky(src, dst, exclude=()):               # Drive is down for the first backup
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError(107, "Transport endpoint is not connected")
        return real(src, dst, exclude)
    monkeypatch.setattr(drive_sync, "sync", flaky)
    plan.main(["run", "--stage", "1", "--only", "encode", "pretrained"])
    steps = json.loads(plan.STATE.read_text())["steps"]
    assert steps["encode"]["state"] == "done" and steps["pretrained"]["state"] == "done"
    assert json.loads((drive / "outputs" / "plan" / "state.json").read_text())["steps"]["pretrained"]["state"] == "done"


def test_budget_stops_before_a_step_that_would_not_finish(fake, monkeypatch):
    calls, _ = fake
    monkeypatch.setattr(plan, "T_START", plan.time.time() - 3600 * 9.5)   # 9.5 h into an 11 h session
    plan.main(["run", "--stage", "all", "--budget-hours", "11"])
    steps = json.loads(plan.STATE.read_text())["steps"]
    assert steps["pretrained"]["state"] == "done" and "plain-after" not in steps   # 2.8 h wouldn't fit


def test_restore_brings_back_progress_saved_on_github(fake):
    saved = plan.REPO / "results" / "2026-10-04_2110_twoday" / "vjepa2" / "plan"
    (saved / "scores" / "plain-after").mkdir(parents=True)
    (saved / "state.json").write_text(json.dumps({"results_folder": "2026-10-04_2110_twoday", "started": "x",
                                                  "choices": {}, "steps": {"plain-after": {"state": "done"}}}))
    (saved / "scores" / "plain-after" / "per_clip.json").write_text("[]")
    assert plan.main(["restore"]) == 0
    assert json.loads(plan.STATE.read_text())["steps"]["plain-after"]["state"] == "done"
    assert (plan.PLAN / "scores" / "plain-after" / "per_clip.json").exists()

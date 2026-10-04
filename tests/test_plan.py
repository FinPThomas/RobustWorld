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

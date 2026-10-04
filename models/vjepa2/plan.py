"""The two-day experiment plan (docs/experiment_plan.md), stage by stage, resumable.

Every step is one unit of work: encode, a baseline, one training run and its scoring, an analysis.
Progress is kept in outputs/vjepa2/plan/state.json (keep outputs/ and checkpoints/ on Google
Drive). A finished step is never redone; a step cut off by a disconnect, or one that failed, runs
again next time, and training resumes at the first fold not yet saved. After every step the
report is rebuilt and saved to results/<date>_<time>_twoday/; with --push it is committed to
GitHub together with the status block in docs/experiment_plan.md.

    python models/vjepa2/plan.py status
    python models/vjepa2/plan.py run --stage 2 --push --branch <branch>
    python models/vjepa2/plan.py run --stage all --push --branch <branch>      # everything left
    python models/vjepa2/plan.py report

Stages (each one Colab cell):
    1  baselines        split, encode, pretrained V-JEPA, TAPNext with and without the coded blockade
    2  post-training    plain / commit / codes / rollout / codes_rollout, before and after
    3  architecture     copy gate, multi-hypothesis, both with the commit loss
    4  robustness       the two best variants trained longer; the best on 25% and 50% of the clips
    5  generalisation   ball from the other side, new ball (waits until those clips exist)
    6  interpretation   change maps and layer patching for the best run, blocker figures

Scoring is always ball_probe_cv.py's frozen evaluation decoder (CLAUDE.md). "Best" is picked by the
cross-validated through-vs-blocked AUROC of stage 2 and 3 runs; that choice uses the same folds it
is reported on, so stage 4's numbers for it are slightly optimistic (said in the report).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PLAN = REPO / "outputs" / "vjepa2" / "plan"
SCORES = PLAN / "scores"
CKPT = REPO / "checkpoints" / "vjepa2" / "plan"
STATE = PLAN / "state.json"
DOC = "docs/experiment_plan.md"
MANIFEST = REPO / "data" / "processed" / "clips" / "manifest.jsonl"
SEEN = REPO / "data" / "processed" / "clips" / "manifest_seen.jsonl"
PY = sys.executable

EPOCHS, LONG_EPOCHS = 10, 30
T4 = ["--batch-size", "1", "--accum", "8"]          # effective batch 8, fits a 16 GB T4
VARIANTS = {                                        # name: (posttrain.py flags, the "before" it is compared with)
    "plain": ([], "plain"),
    "commit": (["--loss", "commit"], "plain"),
    "codes": (["--loss", "codes"], "codes"),
    "rollout": (["--rollout"], "rollout"),
    "codes_rollout": (["--loss", "codes", "--rollout"], "codes_rollout"),
    "gate": (["--copy-gate"], "plain"),
    "hyp": (["--hypotheses", "4"], "plain"),
    "gate_hyp_commit": (["--copy-gate", "--hypotheses", "4", "--loss", "commit"], "plain"),
}
STAGE2 = ["plain", "commit", "codes", "rollout", "codes_rollout"]
STAGE3 = ["gate", "hyp", "gate_hyp_commit"]
WAITING = 3


class Waiting(Exception):
    """The step can't run yet (e.g. no held-out recordings); it is retried next time."""


# --------------------------------------------------------------------------------------------- state

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M")
    return {"results_folder": f"{stamp}_twoday", "started": stamp, "steps": {}, "choices": {}}


def save_state(state: dict) -> None:
    PLAN.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)                                  # never a half-written state on Drive


def commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def gpu() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "no GPU"
    except FileNotFoundError:
        return "no GPU"


def sh(cmd: list[str], log: Path | None = None) -> int:
    """Run a command, streaming its output to the cell (and to `log`). -> exit code."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    with (log.open("a") if log else open(os.devnull, "w")) as f:
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
    return proc.wait()


def must(cmd: list[str], log: Path | None = None) -> None:
    code = sh(cmd, log)
    if code == WAITING:
        raise Waiting("waiting for data")
    if code:
        raise RuntimeError(f"exit code {code}: {' '.join(str(c) for c in cmd[:3])}")


def manifest() -> Path:
    return SEEN if SEEN.exists() else MANIFEST


# --------------------------------------------------------------------------------------------- steps

@dataclass
class Step:
    stage: int
    name: str
    title: str
    fn: Callable[[argparse.Namespace, dict, Path], None]


def blocker_figs(name: str, before: str | None, log: Path) -> None:
    """Blocker figure for a scored run (a plotting failure never fails the step)."""
    cmd = [PY, HERE / "blocker_figs.py", "--per-clip", SCORES / name / "per_clip.json", "--manifest", manifest(),
           "--label", name, "--out", PLAN / "blocker" / name]
    if before and (PLAN / "blocker" / before / "crossing.json").exists():
        cmd += ["--before", PLAN / "blocker" / before / "crossing.json"]
    if sh(cmd, log):
        print(f"!! blocker figure for {name} failed; carrying on", flush=True)


def train_and_score(name: str, flags: list[str], epochs: int, before: str | None, args, log: Path) -> None:
    """posttrain.py (resumes at the first fold not saved), then the frozen-decoder score."""
    run = f"plan/{name}"
    must([PY, HERE / "posttrain.py", "train", "--run", run, "--epochs", str(epochs), "--manifest", manifest(),
          "--resume", *flags, *T4, *args.extra], log)
    must([PY, HERE / "ball_probe_cv.py", "--predictor-run", CKPT / name, "--manifest", manifest(),
          "--out", SCORES / name], log)
    for f in ("log.json", "run_info.json"):
        if (CKPT / name / f).exists():
            shutil.copy2(CKPT / name / f, SCORES / name / f"train_{f}")
    blocker_figs(name, before, log)
    if epochs == 0:                                  # "before" weights are the pretrained ones: not kept
        for pt in (CKPT / name).glob("*.pt"):
            pt.unlink()


def variant_steps(stage: int, variants: list[str]) -> list[Step]:
    steps, befores = [], set()
    for v in variants:
        flags, before = VARIANTS[v]
        if before == v and before not in befores:
            befores.add(before)
            steps.append(Step(stage, f"{v}-before", f"{v}: pretrained predictor, scored the {v} way",
                              lambda a, s, log, v=v, f=flags: train_and_score(f"{v}-before", f, 0, None, a, log)))
        steps.append(Step(stage, f"{v}-after", f"{v}: post-trained {EPOCHS} epochs",
                          lambda a, s, log, v=v, f=flags, b=before:
                          train_and_score(f"{v}-after", f, EPOCHS, f"{b}-before", a, log)))
    return steps


def step_split(args, state, log):
    import generalise
    code = generalise.main(["split", "--manifest", str(MANIFEST)])
    state["choices"]["held_out"] = json.loads((PLAN / "generalise" / "split.json").read_text())["held_out"]
    print(f"stages 1-4 use {manifest().name}" + ("" if code else "; held-out clips set aside"))


def step_encode(args, state, log):
    must([PY, HERE / "posttrain.py", "encode", "--manifest", MANIFEST], log)


def step_pretrained(args, state, log):
    must([PY, HERE / "ball_probe_cv.py", "--manifest", manifest(), "--out", SCORES / "pretrained"], log)
    blocker_figs("pretrained", None, log)


def step_tapnext(args, state, log):
    must([PY, "-m", "pip", "install", "-q", "-r", REPO / "models" / "tapnext_rule" / "requirements.txt"], log)
    must([PY, REPO / "models" / "tapnext_rule" / "predict.py"], log)
    dest = PLAN / "baselines" / "tapnext"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.json", "per_clip.json"):
        shutil.copy2(REPO / "outputs" / "tapnext_rule" / f, dest / f)


def best_variants(state: dict, n: int = 2) -> list[str]:
    """The n stage 2-3 variants with the highest after-training AUROC; frozen once chosen."""
    if "best" in state["choices"]:
        return state["choices"]["best"][:n]
    rows = [r for r in report_rows() if r["phase"] == "after" and r["variant"] in STAGE2 + STAGE3]
    if not rows:
        raise RuntimeError("no stage 2/3 run has been scored yet: run those first")
    ranked = sorted(rows, key=lambda r: -(r["outcome_auroc"] if r["outcome_auroc"] == r["outcome_auroc"] else -1))
    state["choices"]["best"] = [r["variant"] for r in ranked[:2]]
    state["choices"]["best_reason"] = {r["variant"]: r["outcome_auroc"] for r in ranked}
    save_state(state)
    return state["choices"]["best"][:n]


def step_long(i: int):
    def fn(args, state, log):
        v = best_variants(state)[i]
        flags, before = VARIANTS[v]
        train_and_score(f"{v}-long", flags + ["--patience", "6"], LONG_EPOCHS, f"{before}-before", args, log)
    return fn


def step_frac(frac: float):
    def fn(args, state, log):
        v = best_variants(state)[0]
        flags, before = VARIANTS[v]
        train_and_score(f"{v}-frac{int(frac * 100)}", flags + ["--train-frac", str(frac)], EPOCHS,
                        f"{before}-before", args, log)
    return fn


def step_generalise(args, state, log):
    if not state["choices"].get("held_out"):
        step_split(args, state, log)                # recordings may have arrived since stage 1
        if not state["choices"].get("held_out"):
            raise Waiting("no clips from the other side or a new ball yet")
    v = best_variants(state)[0]
    must([PY, HERE / "posttrain.py", "train", "--all", "--run", f"plan/{v}-all", "--epochs", str(EPOCHS),
          "--manifest", SEEN, "--resume", *VARIANTS[v][0], *T4, *args.extra], log)
    must([PY, HERE / "generalise.py", "score", "--run", CKPT / f"{v}-all", "--manifest", MANIFEST], log)


def step_interpret(args, state, log):
    v = best_variants(state)[0]
    must([PY, HERE / "interpret.py", "--run", CKPT / f"{v}-after", "--manifest", manifest()], log)


STEPS: list[Step] = [
    Step(1, "split", "set aside held-out clips (other side, new ball)", step_split),
    Step(1, "encode", "encode every clip once (frozen encoder)", step_encode),
    Step(1, "pretrained", "pretrained V-JEPA 2, frozen decoder", step_pretrained),
    Step(1, "tapnext", "TAPNext + rules, with and without the coded blockade", step_tapnext),
    *variant_steps(2, STAGE2),
    *variant_steps(3, STAGE3),
    Step(4, "long-1", f"best variant, {LONG_EPOCHS} epochs", step_long(0)),
    Step(4, "long-2", f"second-best variant, {LONG_EPOCHS} epochs", step_long(1)),
    Step(4, "frac-50", "best variant on 50% of the training clips", step_frac(0.5)),
    Step(4, "frac-25", "best variant on 25% of the training clips", step_frac(0.25)),
    Step(5, "generalise", "ball from the other side / new ball, before vs after", step_generalise),
    Step(6, "interpret", "change maps and layer patching for the best run", step_interpret),
]


# --------------------------------------------------------------------------------------------- report

def auroc_rows(sp: np.ndarray, sn: np.ndarray) -> np.ndarray:
    """AUROC per row from positive [R, P] and negative [R, N] scores (ties count half)."""
    d = sp[:, :, None] - sn[:, None, :]
    return (d > 0).mean((1, 2)) + 0.5 * (d == 0).mean((1, 2))


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, s0: np.ndarray | None = None, n: int = 1000, seed: int = 0):
    """95% interval of AUROC(y, s) over clips resampled with replacement, stratified by class; with
    s0 (the same clips' "before" scores), also the paired difference and P(difference <= 0)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return None
    ip, ineg = rng.choice(pos, (n, len(pos))), rng.choice(neg, (n, len(neg)))
    a = auroc_rows(s[ip], s[ineg])
    out = {"ci": [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]}
    if s0 is not None:
        d = a - auroc_rows(s0[ip], s0[ineg])
        out |= {"delta_ci": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                "p_no_gain": round(float(np.mean(d <= 0)), 4)}
    return out


def clip_rows(per_clip: list[dict], prefix: str) -> list[dict]:
    return [{"clip_id": r["clip_id"], "outcome": r["outcome"], "far": r[f"{prefix}_far"], "near": r[f"{prefix}_near"]}
            for r in per_clip]


def score_row(method: str, variant: str, phase: str, rows: list[dict], extra: dict, before_rows=None) -> dict:
    from robust_world.eval.ball import outcome_metrics
    m = outcome_metrics(rows)
    y = np.array([r["outcome"] == "through" for r in rows], int)
    s = np.array([max(r["far"]) for r in rows])
    s0 = None
    if before_rows is not None:
        by_id = {r["clip_id"]: max(r["far"]) for r in before_rows}
        if all(r.get("clip_id") in by_id for r in rows):
            s0 = np.array([by_id[r["clip_id"]] for r in rows])
    boot = bootstrap_auroc(y, s, s0) or {}
    return {"method": method, "variant": variant, "phase": phase, "n": len(rows),
            "outcome_auroc": m["outcome_auroc"], "auroc_ci": boot.get("ci"),
            "delta_vs_before_ci": boot.get("delta_ci"), "p_no_gain": boot.get("p_no_gain"),
            "one_ball_p_correct": m.get("p_correct_balanced"), "balanced_accuracy": m.get("balanced_accuracy"),
            **extra}


def health(name: str) -> dict:
    path = SCORES / name / "train_log.json"
    logs = [g for g in (json.loads(path.read_text()) if path.exists() else []) if "overfit" in g]
    if not logs:
        return {}
    return {"overfit_folds": f"{sum(g['overfit']['flag'] for g in logs)}/{len(logs)}",
            "best_epochs": "/".join(str(g["best_epoch"]) for g in logs)}


def report_rows() -> list[dict]:
    """Every scored method, scored the same way: through-vs-blocked AUROC from per-clip readouts
    (V-JEPA: one-ball share of the ball beyond the plank, frozen decoder), with bootstrap intervals."""
    rows = []
    pre = SCORES / "pretrained"
    if (pre / "per_clip.json").exists():
        pc, m = json.loads((pre / "per_clip.json").read_text()), json.loads((pre / "metrics.json").read_text())
        rows.append(score_row("V-JEPA 2: real target frames (ceiling)", "real frames", "ceiling",
                              clip_rows(pc, "real_one_ball"), {"cell_auroc": m.get("real_cell_auroc")}))
        rows.append(score_row("V-JEPA 2 pretrained", "pretrained", "before", clip_rows(pc, "imagined_one_ball"),
                              {"cell_auroc": m.get("future_cell_auroc_imagined"),
                               "argmax_hit_rate": m.get("ball_imagined_argmax", {}).get("hit_rate")}))
    tap = PLAN / "baselines" / "tapnext"
    if (tap / "per_clip.json").exists():
        pc, m = json.loads((tap / "per_clip.json").read_text()), json.loads((tap / "metrics.json").read_text())
        for v, label in (("continue", "TAPNext + straight line (no blockade)"),
                         ("blockade", "TAPNext + coded blockade (fitted on outcomes; reference)")):
            rows.append(score_row(label, f"tapnext_{v}", "baseline", pc[v], {"cell_auroc": m[v].get("future_cell_auroc")}))
    names = sorted(p.name for p in SCORES.glob("*") if (p / "per_clip.json").exists() and p.name != "pretrained")
    order = {v: i for i, v in enumerate(STAGE2 + STAGE3)}
    names.sort(key=lambda n: (order.get(n.rsplit("-", 1)[0], 99), n))
    for name in names:
        variant, phase = name.rsplit("-", 1)
        before = f"{VARIANTS.get(variant, ([], variant))[1]}-before"
        pc, m = json.loads((SCORES / name / "per_clip.json").read_text()), json.loads((SCORES / name / "metrics.json").read_text())
        before_rows = (clip_rows(json.loads((SCORES / before / "per_clip.json").read_text()), "imagined_one_ball")
                       if phase != "before" and (SCORES / before / "per_clip.json").exists() else None)
        rows.append(score_row(f"V-JEPA 2 {name}", variant, phase, clip_rows(pc, "imagined_one_ball"),
                              {"cell_auroc": m.get("future_cell_auroc_imagined"),
                               "argmax_hit_rate": m.get("ball_imagined_argmax", {}).get("hit_rate"), **health(name)},
                              before_rows))
    return rows


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, list):
        return f"[{v[0]}, {v[1]}]"
    return str(v)


def report(state: dict) -> str:
    """Rebuild summary.md / summary.json / summary.png and return the status block for the outline."""
    rows = report_rows()
    PLAN.mkdir(parents=True, exist_ok=True)
    (PLAN / "summary.json").write_text(json.dumps(rows, indent=1))
    head = ["method", "phase", "outcome_auroc", "auroc_ci", "delta_vs_before_ci", "p_no_gain", "one_ball_p_correct",
            "balanced_accuracy", "cell_auroc", "argmax_hit_rate", "overfit_folds", "best_epochs"]
    table = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    table += ["| " + " | ".join(fmt(r.get(h)) for h in head) + " |" for r in rows]
    gen = PLAN / "generalise" / "metrics.json"
    gen_md = []
    if gen.exists():
        g = json.loads(gen.read_text())
        gen_md = ["", "### Generalisation (held out, never trained on)", "",
                  "| group | clips | AUROC before | AUROC after | AUROC real frames |", "|---|---|---|---|---|"]
        gen_md += [f"| {k} | {v['n']} | {v['before']['outcome_auroc']} | {v['after']['outcome_auroc']} | "
                   f"{v['real']['outcome_auroc']} |" for k, v in g["groups"].items()]
    interp = sorted((PLAN / "interpret").glob("*/interpret.json"))
    int_md = []
    for path in interp:
        r = json.loads(path.read_text())
        pin = r["patching"]["patch_in"]
        top = max(pin, key=lambda k: pin[k] if pin[k] == pin[k] else -1)
        int_md += ["", f"### Interpretation: {path.parent.name}", "",
                   f"- Feature change on the plank: {r['feature_change']['share_on_plank']} of the total "
                   f"(the plank is {r['feature_change']['plank_share_of_frame']} of the frame), correlation "
                   f"{r['feature_change']['correlation_with_plank']}.",
                   f"- Patching in one group: highest AUROC from `{top}` ({pin[top]}); before {r['auroc_before']}, "
                   f"after {r['auroc_after']}.",
                   f"- Figures: `vjepa2/plan/interpret/{path.parent.name}/change_map.png`, `layer_patching.png`."]
    figure(rows)
    md = ["## Results", "",
          "Through vs blocked AUROC from the imagined future (one ball, frozen decoder), cross-validated "
          "(5 folds, seed 0). `auroc_ci`: 95% bootstrap interval over clips. `delta_vs_before_ci` / `p_no_gain`: "
          "paired bootstrap against the same variant's pretrained predictor on the same clips (p_no_gain = share "
          "of resamples with no gain). TAPNext + coded blockade fits its blockade on outcomes, so it is a "
          "reference, not an evaluation. Runs named -long / -frac use the variant picked as best on these same "
          "folds, so their numbers are slightly optimistic.", "", *table, *gen_md, *int_md, "",
          "![summary](summary.png)"]
    (PLAN / "summary.md").write_text("\n".join(md) + "\n")

    lines = [f"_Last update: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC from Colab ({gpu()}), "
             f"code `{commit()}`. Results: [results/{state['results_folder']}](../results/{state['results_folder']}/README.md)._",
             "", "| stage | step | state | minutes | finished (UTC) |", "|---|---|---|---|---|"]
    for s in STEPS:
        st = state["steps"].get(s.name, {})
        lines.append(f"| {s.stage} | {s.name}: {s.title} | {st.get('state', 'to do')}"
                     + (f" ({st['error']})" if st.get("error") else "")
                     + f" | {st.get('minutes', '')} | {st.get('finished', '')} |")
    if state["choices"].get("best"):
        lines += ["", f"Best variants (stage 4 onwards): {', '.join(state['choices']['best'])}."]
    keep = ["method", "phase", "outcome_auroc", "auroc_ci", "delta_vs_before_ci", "p_no_gain"]
    lines += ["", "| " + " | ".join(keep) + " |", "|" + "---|" * len(keep)]
    lines += ["| " + " | ".join(fmt(r.get(h)) for h in keep) + " |" for r in rows]
    lines += gen_md
    block = "\n".join(lines)
    (PLAN / "status.md").write_text("# Two-day plan: status\n\n" + block + "\n")
    return block


def figure(rows: list[dict]) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"before": "#9aa5b1", "after": "#c0392b", "long": "#922b21", "baseline": "#2471a3", "ceiling": "#1e8449"}
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.4), dpi=120)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = colours.get(r["phase"], "#c0392b")
        ax.barh(yi, r["outcome_auroc"] or 0, color=c, height=0.6)
        if r.get("auroc_ci"):
            ax.plot(r["auroc_ci"], [yi, yi], color="#222", lw=1.2)
    ax.axvline(0.5, color="#888", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["method"].replace("V-JEPA 2 ", "") for r in rows], fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("through vs blocked AUROC (95% bootstrap interval); 0.5 = chance")
    ax.set_title("Imagined future vs TAPNext, cross-validated", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLAN / "summary.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- commands

def save_and_push(state: dict, block: str, args) -> None:
    import save_results
    try:
        dest = save_results.save("twoday", f"two-day plan, {sum(s.get('state') == 'done' for s in state['steps'].values())} "
                                 f"steps done", into=state["results_folder"], source="outputs/vjepa2/plan")
    except SystemExit as e:
        print(e)
        return
    if not args.push:
        return
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("!! --push needs the GITHUB_TOKEN secret; results are saved locally only", flush=True)
        return
    branch = args.branch or save_results.git("rev-parse", "--abbrev-ref", "HEAD")
    try:
        save_results.push(dest, branch, f"https://x-access-token:{token}@{save_results.GITHUB}", token,
                          update=True, status=(DOC, block))
    except (SystemExit, subprocess.CalledProcessError) as e:
        print(f"!! push failed ({str(e).replace(token, '***')[-300:]}); results stay on Drive, the plan carries on",
              flush=True)


def storage_ok() -> str | None:
    """None if the plan's folder (on Drive in Colab) can be written and read back, else the reason."""
    probe = PLAN / f".write_test_{os.getpid()}"
    try:
        PLAN.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"ok")
        ok = probe.read_bytes() == b"ok"
        probe.unlink()
        return None if ok else "read back different bytes"
    except OSError as e:
        return str(e)


def run(args) -> None:
    if Path("/content").exists() and not (REPO / "outputs").is_symlink():
        print("!! outputs/ is not on Google Drive: progress is lost if Colab stops. Run the Drive and Setup "
              "cells first.", flush=True)
    problem = storage_ok()
    if problem:
        raise SystemExit(f"!! can't write to {PLAN} ({problem}): rerun the Google Drive cell, then this one")
    state = load_state()
    stages = list(range(1, 7)) if args.stage == "all" else [int(args.stage)]
    todo = [s for s in STEPS if s.stage in stages and (not args.only or s.name in args.only)]
    for s in todo:
        st = state["steps"].get(s.name, {})
        if st.get("state") == "done" and not args.redo:
            print(f"== {s.name}: done already ({st.get('finished')}), skipping", flush=True)
            continue
        problem = storage_ok()
        if problem:                                   # e.g. Drive dropped: stop cleanly, nothing is marked failed
            print(f"!! storage stopped working ({problem}); stopping. Rerun the Google Drive cell, then this "
                  f"cell: it carries on from {s.name}.", flush=True)
            break
        if st.get("state") == "running":
            print(f"== {s.name}: was cut off last time, running it again", flush=True)
        print(f"\n== stage {s.stage}, {s.name}: {s.title}", flush=True)
        state["steps"][s.name] = {"state": "running", "started": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M}",
                                  "code": commit()}
        save_state(state)
        (PLAN / "logs").mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            s.fn(args, state, PLAN / "logs" / f"{s.name}.txt")
            result, error = "done", None
        except Waiting as e:
            result, error = "waiting", str(e)
        except Exception as e:  # noqa: BLE001     one failed step must not stop the plan
            result, error = "failed", str(e)[-200:]
        state["steps"][s.name] |= {"state": result, "minutes": round((time.perf_counter() - t0) / 60, 1),
                                   "finished": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M}"}
        state["steps"][s.name].pop("error", None)
        if error:
            state["steps"][s.name]["error"] = error
            print(f"!! {s.name} {result}: {error}; carrying on with the next step", flush=True)
        save_state(state)
        try:
            block = report(state)
        except Exception as e:  # noqa: BLE001
            print(f"!! report failed: {e!r}", flush=True)
            continue
        save_and_push(state, block, args)
    status(args)


def status(args) -> None:
    state = load_state()
    print(f"results folder: results/{state['results_folder']}")
    for s in STEPS:
        st = state["steps"].get(s.name, {})
        print(f"  stage {s.stage}  {s.name:22s} {st.get('state', 'to do'):8s} {st.get('minutes', ''):>6} min  "
              f"{st.get('error', '')}")
    left = [s for s in STEPS if state["steps"].get(s.name, {}).get("state") != "done"]
    print(f"{len(STEPS) - len(left)}/{len(STEPS)} steps done" +
          (f"; next: stage {left[0].stage}, {left[0].name}" if left else "; the plan is finished"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "status", "report"])
    p.add_argument("--stage", default="all", help="1-6 or all")
    p.add_argument("--only", nargs="+", help="just these steps (names from `status`)")
    p.add_argument("--redo", action="store_true", help="run steps again even if done")
    p.add_argument("--push", action="store_true", help="commit results and the status block to GitHub after each step")
    p.add_argument("--branch", default=None)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more posttrain.py flags for every run")
    args = p.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "status":
        status(args)
    else:
        state = load_state()
        block = report(state)
        print(block)
        if args.push:
            save_and_push(state, block, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

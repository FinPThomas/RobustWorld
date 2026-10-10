"""The two-day experiment plan (docs/experiment_plan.md), stage by stage, resumable.

Every step is one unit of work: encode, a baseline, one training run and its scoring, an analysis.
Progress is kept in outputs/vjepa2/plan/state.json. On Colab, outputs/ and checkpoints/ live on the
local disk and are backed up to Google Drive ($ROBUSTWORLD_BACKUP, set by the notebook) after every
step and every 10 minutes, so a Drive drop never interrupts a step. A finished step is never redone; a step cut off by a disconnect, or one that failed, runs
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
import threading
import time
from collections import deque
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
CLIPS = REPO / "data" / "processed" / "clips"
# Training clips: the right-entry segment (whole.mp4 pipeline); older single-video data has manifest.jsonl.
# Other segments (manifest_left.jsonl) are held out for stage 5.
MANIFEST = CLIPS / "manifest_right.jsonl" if (CLIPS / "manifest_right.jsonl").exists() else CLIPS / "manifest.jsonl"
SEEN = PLAN / "manifest_seen.jsonl"          # written by the split step; backed up to Drive, so it survives restarts
BOTH = PLAN / "manifest_both.jsonl"          # both directions: the training clips and the other side's (2026-10-09 scope)
OPEN = CLIPS / "manifest_open.jsonl"          # open-table clips (scripts/cut_open_clips.py): slope test, scored only
PY = sys.executable

EPOCHS, LONG_EPOCHS = 10, 30
# Effective batch 8. Batch 1 x 8 used 1.15 GB of the T4's 16 GB (plain-after), so 4 x 2 fits and keeps the GPU busier.
T4 = ["--batch-size", "4", "--accum", "2"]
VARIANTS = {                                        # name: (posttrain.py flags, the "before" it is compared with)
    "plain": ([], "plain"),
    "commit": (["--loss", "commit"], "plain"),
    "codes": (["--loss", "codes"], "codes"),
    "rollout": (["--rollout"], "rollout"),
    "codes_rollout": (["--loss", "codes", "--rollout"], "codes_rollout"),
    "gate": (["--copy-gate"], "plain"),
    "hyp": (["--hypotheses", "4"], "plain"),
    "gate_hyp_commit": (["--copy-gate", "--hypotheses", "4", "--loss", "commit"], "plain"),
    # Research scope 2026-10-09 (results/overview/RESEARCH_SCOPE.md): both directions, and loss-weighted sampling
    "both": ([], "both"),
    "commit_both": (["--loss", "commit"], "both"),
    # B: right clips, loss-weighted sampling, 20 epochs; its epoch-10 weights are scored as lossmix_e10
    "lossmix_e20": (["--loss", "commit", "--sample-by-loss", "0.25", "--patience", "0"], "plain"),
    "lossmix_e10": (["--loss", "commit", "--sample-by-loss", "0.25", "--patience", "0"], "plain"),
    # C (Fin, 2026-10-10): B's recipe on both directions, saved along the way for the learning curve (slope_test.py)
    "lossmix_both": (["--loss", "commit", "--sample-by-loss", "0.25", "--patience", "0"], "both"),
}
C_EPOCHS = [1, 2, 4, 7, 10, 15]                     # C's weights are also saved and scored after these epochs
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


TAIL: deque[str] = deque(maxlen=200)     # the last lines commands printed, to tell a Drive drop from a real failure
# What a Colab Google Drive mount raises when it drops (FUSE): not the step's fault, so it is never recorded as failed.
DRIVE_ERRORS = ("Transport endpoint is not connected", "Errno 107", "Errno 5]", "Input/output error",
                "Stale file handle", "Software caused connection abort")
DRIVE_RETRY_WAIT = 60                     # seconds to wait before trying a step once more after a Drive drop


def sh(cmd: list[str], log: Path | None = None) -> int:
    """Run a command, streaming its output to the cell (and to `log`, while it can be written). -> exit code."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        f = log.open("a") if log else None
    except OSError:
        f = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        TAIL.append(line)
        if f:
            try:
                f.write(line)
            except OSError:
                f = None
    if f:
        try:
            f.close()
        except OSError:
            pass
    return proc.wait()


def drive_dropped(error: BaseException) -> str | None:
    """The Drive problem behind a failed step, or None if the step failed on its own."""
    text = f"{error!r} {error} " + "".join(TAIL)
    hit = next((m for m in DRIVE_ERRORS if m in text), None)
    return storage_ok() or hit


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


def blocker_figs(name: str, before: str | None, log: Path, man: Path | None = None) -> None:
    """Blocker figure for a scored run (a plotting failure never fails the step)."""
    cmd = [PY, HERE / "blocker_figs.py", "--per-clip", SCORES / name / "per_clip.json", "--manifest", man or manifest(),
           "--label", name, "--out", PLAN / "blocker" / name]
    if before and (PLAN / "blocker" / before / "crossing.json").exists():
        cmd += ["--before", PLAN / "blocker" / before / "crossing.json"]
    if sh(cmd, log):
        print(f"!! blocker figure for {name} failed; carrying on", flush=True)


def export_weights(name: str) -> None:
    """Copy a run's fold weights in float16 to $ROBUSTWORLD_WEIGHTS/<name>/ (on Kaggle: /kaggle/working, which is
    kept as the notebook's output), so later scoring can load them instead of training again."""
    dest = os.environ.get("ROBUSTWORLD_WEIGHTS") or ("/kaggle/working/weights" if Path("/kaggle/working").exists() else None)
    if not dest:
        return
    import torch
    out = Path(dest) / name
    out.mkdir(parents=True, exist_ok=True)
    for pt in sorted((CKPT / name).glob("fold*.pt")):
        if pt.name.endswith(".partial.pt"):
            continue
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        ck["predictor"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["predictor"].items()}
        torch.save(ck, out / pt.name)
    print(f"== {name}: fold weights (float16) saved to {out}", flush=True)


def train_and_score(name: str, flags: list[str], epochs: int, before: str | None, args, log: Path,
                    man: Path | None = None, extra: Path | None = None) -> None:
    """posttrain.py (resumes at the first fold not saved), then the frozen-decoder score (`extra`: a manifest of
    clips also imagined and located, scored only)."""
    run = f"plan/{name}"
    man = man or manifest()
    must([PY, HERE / "posttrain.py", "train", "--run", run, "--epochs", str(epochs), "--manifest", man,
          "--resume", *flags, *T4, *args.extra], log)
    must([PY, HERE / "ball_probe_cv.py", "--predictor-run", CKPT / name, "--manifest", man,
          "--out", SCORES / name, *(["--extra-manifest", extra] if extra else [])], log)
    for f in ("log.json", "run_info.json"):
        if (CKPT / name / f).exists():
            shutil.copy2(CKPT / name / f, SCORES / name / f"train_{f}")
    blocker_figs(name, before, log, man)
    if epochs:
        export_weights(name)
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
    """Every clip once: the training manifest and any other segment's (held out for stage 5)."""
    for m in [MANIFEST, *sorted(p for p in MANIFEST.parent.glob("manifest_*.jsonl") if p != MANIFEST)]:
        must([PY, HERE / "posttrain.py", "encode", "--manifest", m], log)


def cache_complete() -> bool:
    """Every included clip of every manifest has encoded features on this disk (they aren't backed up to Drive)."""
    from robust_world.eval.ball import OUTCOMES
    cached = {p.stem for p in (REPO / "outputs" / "vjepa2" / "cache").glob("*/*.pt")}
    for m in [MANIFEST, *MANIFEST.parent.glob("manifest_*.jsonl")]:
        if m.exists():
            for line in m.open():
                c = json.loads(line)
                if c.get("include") and c["outcome"] in OUTCOMES and c["clip_id"] not in cached:
                    return False
    return True


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


def generalise_variant(v: str, args, state, log) -> None:
    out = PLAN / "generalise" / v
    if (out / "metrics.json").exists() and "mirror_after" in json.dumps(json.loads((out / "metrics.json").read_text())):
        print(f"== generalisation for {v} is done already ({out})", flush=True)
        return
    if not state["choices"].get("held_out"):
        step_split(args, state, log)                # recordings may have arrived since stage 1
        if not state["choices"].get("held_out"):
            raise Waiting("no clips from the other side or a new ball yet")
    step_encode(args, state, log)                   # held-out clips that arrived later (cached ones are skipped)
    must([PY, HERE / "posttrain.py", "train", "--all", "--run", f"plan/{v}-all", "--epochs", str(EPOCHS),
          "--manifest", SEEN, "--resume", *VARIANTS[v][0], *T4, *args.extra], log)
    must([PY, HERE / "generalise.py", "score", "--run", CKPT / f"{v}-all", "--manifest", MANIFEST, "--out", out], log)
    for f in ("log.json", "run_info.json"):
        if (CKPT / f"{v}-all" / f).exists():
            shutil.copy2(CKPT / f"{v}-all" / f, out / f"train_{f}")


def interpret_variant(v: str, args, log) -> None:
    if (PLAN / "interpret" / f"{v}-after" / "interpret.json").exists():
        print(f"== interpretation of {v}-after is done already", flush=True)
        return
    if len(list((CKPT / f"{v}-after").glob("fold*.pt"))) < 5:   # weights aren't on GitHub: a new machine retrains
        print(f"== the {v}-after weights aren't on this machine: training them again (same settings and folds)",
              flush=True)
        must([PY, HERE / "posttrain.py", "train", "--run", f"plan/{v}-after", "--epochs", str(EPOCHS),
              "--manifest", manifest(), "--resume", *VARIANTS[v][0], *T4, *args.extra], log)
    must([PY, HERE / "interpret.py", "--run", CKPT / f"{v}-after", "--manifest", manifest()], log)


def both_manifest() -> Path:
    """Every included clip of both directions: the training manifest's and the other segments' (left side)."""
    import generalise
    if not MANIFEST.exists():
        raise Waiting(f"no clips on this machine ({MANIFEST.name})")
    clips = generalise.included(MANIFEST) + generalise.other_segments(MANIFEST)
    if not any(generalise.side_in(c) != generalise.side_in(clips[0]) for c in clips):
        raise Waiting("only one direction of clips on this machine (add the left segment's zip)")
    BOTH.parent.mkdir(parents=True, exist_ok=True)
    BOTH.write_text("".join(json.dumps(c) + "\n" for c in clips))
    return BOTH


def step_lossmix(args, state, log):
    """B: 20 epochs (cosine over 20), the epoch-10 weights saved on the way as lossmix_e10-after; both scored."""
    flags = VARIANTS["lossmix_e20"][0] + ["--save-at", "10:plan/lossmix_e10-after"]
    train_and_score("lossmix_e20-after", flags, 2 * EPOCHS, "plain-before", args, log)
    must([PY, HERE / "ball_probe_cv.py", "--predictor-run", CKPT / "lossmix_e10-after", "--manifest", manifest(),
          "--out", SCORES / "lossmix_e10-after"], log)
    blocker_figs("lossmix_e10-after", "plain-before", log)
    export_weights("lossmix_e10-after")


def open_manifest() -> Path:
    if not OPEN.exists():
        raise Waiting("the open-table clips aren't on this machine (add robustworld_clips_open.zip to the dataset)")
    return OPEN


def step_lossmix_both(args, state, log):
    """C: commit + loss-weighted sampling on both directions, 20 epochs, weights saved after each of C_EPOCHS."""
    extra = open_manifest()                             # wait for the open-table clips rather than train without them
    flags = VARIANTS["lossmix_both"][0] + ["--save-at", *(f"{e}:plan/lossmix_both_e{e}-after" for e in C_EPOCHS)]
    train_and_score("lossmix_both-after", flags, 2 * EPOCHS, "both-before", args, log, both_manifest(), extra)


def step_lossmix_both_epochs(args, state, log):
    """Score C's saved epochs (learning curve) and the pretrained predictor (epoch 0) on the open-table clips too;
    each epoch's weights are deleted once scored (disk)."""
    man, extra = both_manifest(), open_manifest()
    if not (SCORES / "both-before" / "per_clip_open.json").exists():
        must([PY, HERE / "ball_probe_cv.py", "--manifest", man, "--out", SCORES / "both-before",
              "--extra-manifest", extra], log)
    for e in C_EPOCHS:
        name = f"lossmix_both_e{e}-after"
        if (SCORES / name / "per_clip.json").exists():
            print(f"== {name} is scored already", flush=True)
            continue
        if len(list((CKPT / name).glob("fold*.pt"))) < 5:
            raise RuntimeError(f"{name}: the saved weights aren't on this machine (C was trained in another session); "
                               "train C again with --save-at, or score fewer epochs")
        must([PY, HERE / "ball_probe_cv.py", "--predictor-run", CKPT / name, "--manifest", man,
              "--out", SCORES / name, "--extra-manifest", extra], log)
        shutil.rmtree(CKPT / name, ignore_errors=True)


def step_other_ball(args, state, log):
    """Other-ball clips, scored without training (scope D). Waits until they are packed and listed."""
    raise Waiting("the other ball's clips aren't added yet (configs/heldout.json and their clips zip)")


def step_generalise(args, state, log):
    generalise_variant(best_variants(state)[0], args, state, log)


def step_interpret(args, state, log):
    interpret_variant(best_variants(state)[0], args, log)


STEPS: list[Step] = [
    Step(1, "split", "set aside held-out clips (other side, new ball)", step_split),
    Step(1, "encode", "encode every clip once (frozen encoder)", step_encode),
    Step(1, "pretrained", "pretrained V-JEPA 2, frozen decoder", step_pretrained),
    Step(1, "tapnext", "TAPNext + rules, with and without the coded blockade", step_tapnext),
    # Proof of concept first (Fin, 2026-10-06): the first post-trained run, then the cheapest steps that answer
    # the other questions (does it carry over to the other side? can the output be made cleaner? what changed?),
    # then the rest of the grid, then the demanding runs. Stages 5 and 6 repeat generalisation and
    # interpretation for the best variant if that isn't plain.
    *variant_steps(2, STAGE2[:1]),
    Step(2, "generalise-plain", "plain: ball from the other side, before vs after",
         lambda a, s, log: generalise_variant("plain", a, s, log)),
    *variant_steps(2, ["commit"]),
    *variant_steps(3, ["gate"]),
    Step(2, "interpret-plain", "plain: change maps and layer patching", lambda a, s, log: interpret_variant("plain", a, log)),
    *variant_steps(3, ["hyp"]),
    # commit came out best in the first Kaggle session (2026-10-07), so check that it carries over too
    Step(2, "generalise-commit", "commit: ball from the other side, before vs after",
         lambda a, s, log: generalise_variant("commit", a, s, log)),
    # Fin, 2026-10-07: did it learn a rule for one direction, or where the blockade is? Rescores the other
    # side with the clips mirrored and against the height the ball reaches the plank (generalise.py).
    Step(2, "mirror-plain", "plain: other side mirrored, and by height at the plank",
         lambda a, s, log: generalise_variant("plain", a, s, log)),
    Step(2, "mirror-commit", "commit: other side mirrored, and by height at the plank",
         lambda a, s, log: generalise_variant("commit", a, s, log)),
    # Research scope (Fin, 2026-10-09; results/overview/RESEARCH_SCOPE.md) replaces the rest of the old queue
    # (rollout, codes_rollout, gate_hyp_commit, 30-epoch and data-fraction runs, stages 5-6 for the best run).
    # In Fin's priority order: the table slope (scoring pass with per-step ball positions, then B and A, whose
    # imagined tracks slope.py reads), the narrow gap and training length (B at epoch 10 and 20), both
    # directions (A), the other ball (D). slope.py and scope_report.py run in the report after every step.
    *variant_steps(2, ["codes"]),
    Step(2, "both-before", "slope S0-S2: pretrained, both directions, per-step ball positions",
         lambda a, s, log: train_and_score("both-before", [], 0, None, a, log, both_manifest())),
    Step(2, "lossmix_e20-after", "B. commit, right clips, loss-weighted sampling, 20 epochs (scored at 10 and 20)",
         step_lossmix),
    Step(2, "commit_both-after", "A. commit trained on both directions, each side scored",
         lambda a, s, log: train_and_score("commit_both-after", VARIANTS["commit_both"][0], EPOCHS, "both-before",
                                           a, log, both_manifest())),
    # Fin, 2026-10-10: show the slope is learnt (uphill vs downhill at matched speed, in ball sizes) and how it and
    # the blockade grow with training: C, scored after 1, 2, 4, 7, 10, 15 and 20 epochs (slope_test.py).
    Step(2, "lossmix_both-after", "C. both directions, loss-weighted sampling, 20 epochs (weights saved for the "
         "learning curve)", step_lossmix_both),
    Step(2, "lossmix_both-epochs", "C. learning curve: score the pretrained predictor and the weights saved after "
         "1, 2, 4, 7, 10 and 15 epochs, plank and open-table clips",
         step_lossmix_both_epochs),
    Step(5, "other-ball", "D. the other ball, scored without training", step_other_ball),
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
                              clip_rows(pc, "real_one_ball"), {"cell_auroc": m.get("real_cell_auroc"),
                                                              "far_cell_auroc": m.get("far_cell_auroc_real")}))
        rows.append(score_row("V-JEPA 2 pretrained", "pretrained", "before", clip_rows(pc, "imagined_one_ball"),
                              {"cell_auroc": m.get("future_cell_auroc_imagined"), "far_cell_auroc": m.get("far_cell_auroc_imagined"),
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
                              {"cell_auroc": m.get("future_cell_auroc_imagined"), "far_cell_auroc": m.get("far_cell_auroc_imagined"),
                               "argmax_hit_rate": m.get("ball_imagined_argmax", {}).get("hit_rate"), **health(name)},
                              before_rows))
        sides = sorted({r.get("side_in") for r in pc if r.get("side_in")})
        for side in sides if len(sides) > 1 else []:        # both directions: each side on its own as well
            sub = [r for r in pc if r.get("side_in") == side]
            rows.append(score_row(f"V-JEPA 2 {name} (ball from {side})", variant, phase,
                                  clip_rows(sub, "imagined_one_ball"), {}, before_rows))
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
            "balanced_accuracy", "cell_auroc", "far_cell_auroc", "argmax_hit_rate", "overfit_folds", "best_epochs"]
    table = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    table += ["| " + " | ".join(fmt(r.get(h)) for h in head) + " |" for r in rows]
    gen_md = []
    for gen in sorted((PLAN / "generalise").glob("*/metrics.json")):
        g = json.loads(gen.read_text())
        gen_md += ["", f"### Generalisation of {gen.parent.name} (held out, never trained on)", "",
                   "| group | clips | AUROC before | AUROC after | AUROC real frames | mirrored: before | "
                   "mirrored: after | mirrored: real frames |", "|---|---|---|---|---|---|---|---|"]
        a = lambda v, k: v.get(k, {}).get("outcome_auroc", "-")  # noqa: E731
        gen_md += [f"| {k} | {v['n']} | {a(v, 'before')} | {a(v, 'after')} | {a(v, 'real')} | "
                   f"{a(v, 'mirror_before')} | {a(v, 'mirror_after')} | {a(v, 'mirror_real')} |"
                   for k, v in g["groups"].items()]
        for k, v in g["groups"].items():
            h = v.get("height") or {}
            if h.get("range"):
                vs = h["auroc_vs_height_rule"]
                gen_md += ["", f"{k}, by height at the plank: the blockade range fitted on the seen clips "
                           f"({h['range'][0]}-{h['range'][1]} px) gets {h['accuracy_on_real_outcomes']} of these "
                           f"outcomes right. How well the \"gets through\" score follows that range (AUROC): real "
                           f"frames {vs.get('real')}, before {vs.get('before')}, after {vs.get('after')}, mirrored "
                           f"after {vs.get('mirror_after')}. Figure: `vjepa2/plan/generalise/{gen.parent.name}/height.png`."]
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
    for mod in ("slope", "scope_report", "slope_test"):
        try:
            __import__(mod).main(["--plan", str(PLAN)])
        except Exception as e:                        # a figure never stops the plan
            print(f"!! {mod} report failed: {e!r}", flush=True)
    md = ["## Results", "",
          "Through vs blocked AUROC from the imagined future (one ball, frozen decoder), cross-validated "
          "(5 folds, seed 0). `auroc_ci`: 95% bootstrap interval over clips. `delta_vs_before_ci` / `p_no_gain`: "
          "paired bootstrap against the same variant's pretrained predictor on the same clips (p_no_gain = share "
          "of resamples with no gain). TAPNext + coded blockade fits its blockade on outcomes, so it is a "
          "reference, not an evaluation. Runs named -long / -frac use the variant picked as best on these same "
          "folds, so their numbers are slightly optimistic.", "", *table, *gen_md, *int_md, "",
          "![summary](summary.png)"]
    for extra in (PLAN / "slope_test" / "slope_test.md", PLAN / "slope" / "slope.md", PLAN / "scope" / "scope.md"):
        if extra.exists():
            md += ["", extra.read_text()]
    (PLAN / "summary.md").write_text("\n".join(md) + "\n")

    lines = [f"_Last update: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC from {'Kaggle' if Path('/kaggle').exists() else 'Colab'} ({gpu()}), "
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

def save_and_push(state: dict, block: str, args) -> bool:
    """Save the results folder and, with --push, commit it and the status block. -> True once pushed."""
    import save_results
    try:
        dest = save_results.save("twoday", f"two-day plan, {sum(s.get('state') == 'done' for s in state['steps'].values())} "
                                 f"steps done", into=state["results_folder"], source="outputs/vjepa2/plan")
    except SystemExit as e:
        print(e)
        return False
    if not args.push:
        return False
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("!! --push needs the GITHUB_TOKEN secret; results are saved locally only", flush=True)
        return False
    branch = args.branch or save_results.git("rev-parse", "--abbrev-ref", "HEAD")
    try:
        save_results.push(dest, branch, f"https://x-access-token:{token}@{save_results.GITHUB}", token,
                          update=True, status=(DOC, block))
    except (SystemExit, subprocess.CalledProcessError) as e:
        print(f"!! push failed ({str(e).replace(token, '***')[-300:]}); results stay on Drive and go up with the "
              "next step's push (or `plan.py report --push`)", flush=True)
        return False
    state["last_push"] = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M}"
    save_state(state)
    return True


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


MIN_FREE_GB = 5                              # a step writes at most ~1 GB (a run's checkpoints); the whole plan ~8 GB
BACKUP_EVERY = 600                           # seconds between backups to Drive while a step runs
_backup_lock = threading.Lock()


def _backup(dest: str, why: str, wait: float) -> None:
    import drive_sync
    if not _backup_lock.acquire(timeout=wait or 0.01):
        return                                        # one is already running (Drive may be slow): skip this one
    try:
        n = sum(drive_sync.sync(REPO / d, Path(dest) / d, drive_sync.NOT_BACKED_UP) for d in ("outputs", "checkpoints"))
        if n and why:
            print(f"backed up {n} files to Drive ({why})", flush=True)
    except OSError as e:
        print(f"!! backup to Google Drive failed ({e}); the work is safe on Colab's disk, GitHub pushes go on, and "
              "the next backup tries again.", flush=True)
    finally:
        _backup_lock.release()


def backup(why: str = "", wait: float = 0) -> None:
    """Copy new work in outputs/ and checkpoints/ to the Drive backup ($ROBUSTWORLD_BACKUP), in the background so
    a slow or dropped Drive never holds up a step or a GitHub push; `wait`: seconds to wait for it at most."""
    dest = os.environ.get("ROBUSTWORLD_BACKUP")
    if not dest:
        return
    t = threading.Thread(target=_backup, args=(dest, why, wait), daemon=True)
    t.start()
    if wait:
        t.join(2 * wait)


def backup_every(stop: threading.Event) -> None:
    while not stop.wait(BACKUP_EVERY):
        backup()


def run(args) -> None:
    if Path("/content").exists() and not (REPO / "outputs").is_symlink() and not os.environ.get("ROBUSTWORLD_BACKUP"):
        print("!! outputs/ is not backed up to Google Drive: progress is lost if Colab stops. Run the Drive and "
              "Setup cells first.", flush=True)
    stop = threading.Event()
    threading.Thread(target=backup_every, args=(stop,), daemon=True).start()
    try:
        _run(args)
    finally:
        stop.set()
        backup("end of run", wait=600)


T_START = time.time()


def step_hours(name: str) -> float:
    """A generous guess of a step's run time on a T4 (plain-after took 2.3 h), for --budget-hours."""
    special = {"commit_both-after": 3.2,          # a third more clips than the other runs
               "lossmix_e20-after": 4.4,          # 20 epochs, then two scoring passes
               "lossmix_both-after": 5.6,         # 20 epochs on both directions (commit_both: 2.6 h for 10)
               "lossmix_both-epochs": 1.9}        # seven scoring passes
    if name in special:
        return special[name]
    if name.startswith("long"):
        return 7.0
    if name.endswith("-after") or name.startswith("interpret"):
        return 2.8                                  # interpret may first retrain the weights it reads
    if name.startswith("frac"):
        return 1.5
    if name.startswith(("generalise", "mirror", "other-ball")):
        return 1.0
    return 0.7


def restore() -> int:
    """A new machine (Kaggle, a fresh Colab without Drive): bring back the plan's progress from the newest
    results/<date>_<time>_twoday/ folder on GitHub (state, scores, logs; weights aren't on GitHub)."""
    if STATE.exists():
        print(f"{STATE} exists: nothing to restore")
        return 0
    found = sorted((REPO / "results").glob("*_twoday/vjepa2/plan/state.json"))
    if not found:
        print("no saved plan on GitHub: starting from the beginning")
        return 0
    src = found[-1].parent
    shutil.copytree(src, PLAN, dirs_exist_ok=True)
    state = load_state()
    done = [k for k, v in state["steps"].items() if v.get("state") == "done"]
    print(f"restored from {src.relative_to(REPO)}: {len(done)} steps done ({', '.join(done)})")
    return 0


def _run(args) -> None:
    problem = storage_ok()
    if problem:
        raise SystemExit(f"!! can't write to {PLAN} ({problem}): rerun the Google Drive cell, then this one")
    state = load_state()
    if state["steps"].get("split", {}).get("state") == "done" and not SEEN.exists():
        print("== the seen-clips manifest is missing: recreating it", flush=True)
        step_split(args, state, None)
    if state["steps"].get("encode", {}).get("state") == "done" and not cache_complete():
        print("== the encoded features aren't on this disk (a new runtime): encoding again", flush=True)
        step_encode(args, state, PLAN / "logs" / "encode.txt")
    stages = list(range(1, 7)) if args.stage == "all" else [int(args.stage)]
    todo = [s for s in STEPS if s.stage in stages and (not args.only or s.name in args.only)]
    cut_off = [s for s in STEPS if s.stage < min(stages) and state["steps"].get(s.name, {}).get("state") == "running"]
    if cut_off and not args.only:                     # e.g. encode stopped by a Drive drop: later stages need it
        print(f"== finishing first what was cut off: {', '.join(s.name for s in cut_off)}", flush=True)
        todo = cut_off + todo
    for s in todo:
        st = state["steps"].get(s.name, {})
        if st.get("state") == "done" and not args.redo:
            print(f"== {s.name}: done already ({st.get('finished')}), skipping", flush=True)
            continue
        if args.budget_hours:                         # e.g. Kaggle's 12 h sessions: don't start what can't finish
            used, need = (time.time() - T_START) / 3600, step_hours(s.name)
            if used + need > args.budget_hours:
                print(f"== {s.name} needs about {need:.1f} h and this session has {args.budget_hours - used:.1f} h "
                      "left: stopping here. Run again (a new session) to carry on from this step.", flush=True)
                break
        free = shutil.disk_usage(REPO).free / 2**30
        if free < MIN_FREE_GB:                        # Colab's disk: stop before a write fails half-way
            print(f"!! only {free:.1f} GB free on this disk (a step needs up to {MIN_FREE_GB} GB); stopping before "
                  f"{s.name}. Free space (e.g. Runtime > Disconnect and delete runtime, then cells 1, 2 and this one: "
                  "Setup brings the work back from Drive).", flush=True)
            break
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
        dropped = None
        for attempt in (1, 2):
            TAIL.clear()
            try:
                s.fn(args, state, PLAN / "logs" / f"{s.name}.txt")
                result, error, dropped = "done", None, None
            except Waiting as e:
                result, error, dropped = "waiting", str(e), None
            except Exception as e:  # noqa: BLE001     one failed step must not stop the plan
                result, error = "failed", str(e)[-200:]
                dropped = drive_dropped(e)
            if not dropped or attempt == 2:
                break
            print(f"!! Google Drive dropped during {s.name} ({dropped}); waiting {DRIVE_RETRY_WAIT} s and trying "
                  "it once more (finished parts are kept)", flush=True)
            time.sleep(DRIVE_RETRY_WAIT)
            if storage_ok():
                break
        if dropped:                                   # stop cleanly; the step stays "running", so it is redone
            print(f"!! Google Drive stopped working during {s.name} ({dropped}). Stopping: nothing is marked failed. "
                  f"Rerun cell 1 (Google Drive), then cell 2 (Setup), then this cell: it carries on from {s.name}.",
                  flush=True)
            break
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
            block = None
        if block is not None:
            save_and_push(state, block, args)
        backup(f"after {s.name}")
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
    p.add_argument("command", choices=["run", "status", "report", "restore"])
    p.add_argument("--stage", default="all", help="1-6 or all")
    p.add_argument("--only", nargs="+", help="just these steps (names from `status`)")
    p.add_argument("--redo", action="store_true", help="run steps again even if done")
    p.add_argument("--push", action="store_true", help="commit results and the status block to GitHub after each step")
    p.add_argument("--branch", default=None)
    p.add_argument("--budget-hours", type=float, default=None,
                   help="don't start a step that wouldn't finish within this many hours of the run starting")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more posttrain.py flags for every run")
    args = p.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "status":
        status(args)
    elif args.command == "restore":
        return restore()
    else:
        state = load_state()
        block = report(state)
        print(block)
        if args.push and not save_and_push(state, block, args):
            return 1                                  # the clean-up cell relies on this
    return 0


if __name__ == "__main__":
    sys.exit(main())

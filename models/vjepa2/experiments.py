"""The post-training grid: plain V-JEPA, discrete codes, step-by-step rollout, and both,
each scored before (pretrained predictor, 0 epochs) and after post-training.

Every run is scored by ball_probe_cv.py with the same frozen evaluation decoder and folds, so the
only thing that differs between "before" and "after" is the predictor's weights.

    variant        posttrain.py flags
    plain          (V-JEPA latent L1, all steps at once)
    codes          --loss codes
    rollout        --rollout
    codes_rollout  --loss codes --rollout

Headline metrics (robust_world.eval.ball), from V-JEPA's imagined future:
    P(correct)         mean probability of the true outcome (through / bounce / hidden), averaged
                       over outcomes so each counts equally however many clips it has
    balanced accuracy  share of clips whose most likely outcome is right, averaged over outcomes
    ball hit rate      share of target steps with a real ball where the predicted ball is within
                       1.5 cells (48 px) of it; plus median error and phantom-ball rate

    python models/vjepa2/experiments.py run --epochs 10              # train + score all 8 runs
    python models/vjepa2/experiments.py run --variants plain codes   # a subset
    python models/vjepa2/experiments.py summary                      # table + figure from scores
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VARIANTS = {"plain": [], "codes": ["--loss", "codes"], "rollout": ["--rollout"],
            "codes_rollout": ["--loss", "codes", "--rollout"]}
SCORES = REPO / "outputs" / "vjepa2" / "ball_probe_cv"
OUT = REPO / "outputs" / "vjepa2" / "experiments"


def runs(variants):
    return [(v, phase, f"{v}-{phase}") for v in variants for phase in ("before", "after")]


def run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps({"variants": args.variants, "epochs": args.epochs,
                                                 "extra": args.extra, "argv": sys.argv}, indent=1))
    for v, phase, name in runs(args.variants):
        epochs = 0 if phase == "before" else args.epochs
        print(f"== {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / "posttrain.py"), "train", "--run", name,
                        "--epochs", str(epochs), *VARIANTS[v], *args.extra], check=True)
        subprocess.run([sys.executable, str(HERE / "ball_probe_cv.py"),
                        "--predictor-run", str(REPO / "checkpoints" / "vjepa2" / name)], check=True)
    summary(args)


def headline(outcomes: dict, ball: dict) -> dict:
    return {"p_correct": outcomes.get("p_correct_balanced"), "balanced_accuracy": outcomes.get("balanced_accuracy"),
            **{f"p_correct_{k}": v for k, v in outcomes.get("p_correct", {}).items()},
            "ball_hit_rate": ball["hit_rate"], "ball_error_px": ball["error_px"], "phantom_rate": ball["phantom_rate"],
            "through_vs_blocked_auroc": outcomes.get("outcome_auroc")}


def summary(args) -> None:
    rows, last = [], None
    for v, phase, name in runs(args.variants):
        path = SCORES / name / "metrics.json"
        if path.exists():
            last = json.loads(path.read_text())
            rows.append({"variant": v, "phase": phase, **headline(last["outcomes_imagined"], last["ball_imagined"])})
    if last is None:
        raise SystemExit(f"no scores under {SCORES}; run `experiments.py run` first")
    # the decoder on the real future frames: the ceiling any imagined future can reach
    rows.append({"variant": "real frames (ceiling)", "phase": "-", **headline(last["outcomes_real"], last["ball_real"])})
    OUT.mkdir(parents=True, exist_ok=True)
    head = list(dict.fromkeys(k for r in rows for k in r))
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(r.get(h, "-")) for h in head) + " |" for r in rows]
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))
    figure(rows)
    print(f"-> {OUT}/ (summary.md, summary.json, summary.png)")


def figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = [r for r in rows if r["phase"] in ("before", "after")]
    variants = list(dict.fromkeys(r["variant"] for r in grid))
    real = next(r for r in rows if r["phase"] == "-")
    metrics = [("p_correct", "P(correct outcome), balanced"), ("balanced_accuracy", "balanced accuracy"),
               ("ball_hit_rate", "ball hit rate (within 48 px)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8), dpi=120)
    x = np.arange(len(variants))
    for ax, (key, title) in zip(axes, metrics):
        for i, (phase, colour) in enumerate((("before", "#9aa5b1"), ("after", "#c0392b"))):
            vals = [next((r[key] for r in grid if r["variant"] == v and r["phase"] == phase), None) or 0
                    for v in variants]
            ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=colour, label=phase)
        if real.get(key) is not None:
            ax.axhline(real[key], color="black", ls="--", lw=1, label="real frames")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("V-JEPA 2 imagined future, read by the frozen decoder: before vs after post-training", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "summary.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["run", "summary"])
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="more posttrain.py flags for every run, e.g. --extra --fold 0")
    args = p.parse_args(argv)
    run(args) if args.stage == "run" else summary(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

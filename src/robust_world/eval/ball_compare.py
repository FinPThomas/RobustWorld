"""Compare models on the shared ball-level measures (robust_world.eval.ball), all cross-validated
on the same folds. V-JEPA 2 is read by its frozen evaluation decoder (models/vjepa2/eval_decoder.py),
which can't learn the blockade; any outcome knowledge in its row comes from the world model.
The TAPNext rule baseline, by design, fits its blockade range on training-fold outcomes.

  future cell AUROC   where the ball really is in the target half, scored per 32 px cell
  outcome AUROC/acc   through vs hidden, from the peak P(ball beyond the occluder)
  curves              mean P(ball visible) and P(ball beyond the occluder) over the target,
                      split by true outcome

Each entry points at a per-clip JSON of rows with per-step "visible" and "far" lists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..paths import REPO_ROOT, rel

OUT = REPO_ROOT / "outputs" / "ball_eval"


def _vjepa(kind: str):
    rows = json.loads((REPO_ROOT / "outputs/vjepa2/ball_probe_cv/per_clip.json").read_text())
    return [{"clip_id": r["clip_id"], "outcome": r["outcome"], "visible": r[f"{kind}_visible"],
             "far": r[f"{kind}_far"]} for r in rows]


def _tapnext(variant: str):
    return json.loads((REPO_ROOT / "outputs/tapnext_rule/per_clip.json").read_text())[variant]


def _metrics(name: str) -> dict:
    if name.startswith("vjepa"):
        m = json.loads((REPO_ROOT / "outputs/vjepa2/ball_probe_cv/metrics.json").read_text())
        kind = name.split(":")[1]
        return {"future_cell_auroc": m["real_cell_auroc"] if kind == "real" else m["future_cell_auroc_imagined"]}
    m = json.loads((REPO_ROOT / "outputs/tapnext_rule/metrics.json").read_text())
    return {"future_cell_auroc": m[name.split(":")[1]]["future_cell_auroc"]}


METHODS = [
    ("V-JEPA 2: real target frames, frozen decoder (upper bound)", "vjepa:real", lambda: _vjepa("real"), "tab:green", "-"),
    ("V-JEPA 2 pretrained: imagined target (before post-training)", "vjepa:imagined", lambda: _vjepa("imagined"), "tab:red", "--"),
    ("TAPNext + straight-line continuation", "tapnext:continue", lambda: _tapnext("continue"), "tab:gray", ":"),
    ("TAPNext + continuation + plank/blockade rules (blockade fitted on outcomes)", "tapnext:blockade", lambda: _tapnext("blockade"), "tab:blue", "-."),
]


def compare(out: Path = OUT) -> Path:
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    table, curves = [], {}
    for label, key, load, colour, style in METHODS:
        try:
            rows = load()
        except FileNotFoundError:
            continue
        y = np.array([r["outcome"] == "through" for r in rows], int)
        peak = np.array([max(r["far"]) for r in rows])
        table.append({"method": label, "clips": len(rows),
                      "where_ball_goes_cell_auroc": _metrics(key)["future_cell_auroc"],
                      "outcome_auroc": round(float(roc_auc_score(y, peak)), 3),
                      "outcome_accuracy": round(float(((peak > 0.5) == y).mean()), 3),
                      "hidden_called_through": int(((peak > 0.5) & (y == 0)).sum()),
                      "through_called_hidden": int(((peak <= 0.5) & (y == 1)).sum())})
        curves[label] = (rows, colour, style)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=120, sharex=True, sharey=True)
    for j, outcome in enumerate(("through", "hidden")):
        for i, key in enumerate(("visible", "far")):
            ax = axes[i, j]
            for label, (rows, colour, style) in curves.items():
                m = np.array([r[key] for r in rows if r["outcome"] == outcome])
                t = np.arange(m.shape[1]) * 2 + 16
                ax.plot(t, m.mean(0), style, color=colour, lw=2, label=label)
                ax.fill_between(t, m.mean(0) - m.std(0) / np.sqrt(len(m)), m.mean(0) + m.std(0) / np.sqrt(len(m)),
                                color=colour, alpha=0.12)
            n = sum(r["outcome"] == outcome for r in next(iter(curves.values()))[0])
            ax.set_title(f"true outcome: {outcome} ({n} clips)")
            ax.set_ylim(-0.02, 1.02)
            if j == 0:
                ax.set_ylabel("P(ball visible)" if key == "visible" else "P(ball beyond the plank)")
            if i == 1:
                ax.set_xlabel("clip frame (target half)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Ball-level comparison, cross-validated over all clips (mean ± s.e.)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png")
    plt.close(fig)

    head = ["method", "clips", "where_ball_goes_cell_auroc", "outcome_auroc", "outcome_accuracy",
            "hidden_called_through", "through_called_hidden"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in table]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    (out / "comparison.json").write_text(json.dumps(table, indent=1))
    print("\n".join(md))
    print(f"-> {rel(out)}/ (comparison.png, comparison.md, comparison.json)")
    return out

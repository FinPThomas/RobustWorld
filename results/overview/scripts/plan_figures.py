import json, sys, numpy as np, matplotlib
# Regenerates figures/plan_*.png. Usage: python plan_figures.py <PR #3 results/2026-10-04_2110_twoday/vjepa2/plan> <out dir>
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
C = {"real frames": "#7a7974", "pretrained": "#2a78d6", "plain": "#eb6834", "commit": "#1baf7a", "codes": "#4a3aa7", "gate": "#eda100"}
OC = {"through": "#1baf7a", "hidden": "#4a3aa7", "bounce": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#9a9993", "figure.facecolor": "white"})
J = lambda p: json.load(open(D / p))
GAP_A, GAP_B, RANGE = (104, 186), (315, 367), (176.3, 329.4)

# 1. training curves
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for v in ["plain", "commit", "codes", "gate"]:
    L = J(f"scores/{v}-after/train_log.json")
    V = np.array([f["val_l1"] for f in L])
    rel = (V - V[:, :1]) / (V[:, :1] - V[:, -1:]) * -1 * 100  # % of total drop achieved
    for r in rel: axs[0].plot(range(1, 11), r, color=C[v], lw=0.6, alpha=0.35)
    axs[0].plot(range(1, 11), rel.mean(0), color=C[v], lw=2, label=v)
L = J("scores/commit-after/train_log.json")
N = np.array([[e["parts"]["nce"] for e in f["epochs"]] for f in L])
for r in N: axs[1].plot(range(1, 11), r, color=C["commit"], lw=0.6, alpha=0.35)
axs[1].plot(range(1, 11), N.mean(0), color=C["commit"], lw=2, label="commit: contrastive (nce) part")
lr = [e["lr"] / 3e-5 for e in L[0]["epochs"]]
ax2 = axs[0]
axs[0].set(title="Validation error: share of the 10-epoch drop reached", xlabel="epoch", ylabel="% of drop from epoch 1 to 10")
axs[0].legend(frameon=False, fontsize=8)
axs[1].set(title="Commit: contrastive loss is still falling", xlabel="epoch", ylabel="train nce loss")
axs[1].legend(frameon=False, fontsize=8)
for e, l in zip(range(1, 11), lr):
    pass
fig.text(0.01, 0.01, "Learning rate follows a cosine schedule from 3e-5 to ~0 at epoch 10, so a flat end of the curve is partly forced by the schedule.", fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "plan_training_curves.png", dpi=150); plt.close(fig)

# 2. right side: imagined far-side ball vs crossing height
X = {p["clip_id"]: p for p in J("blocker/commit-after/crossing.json")["passes"]}
bins = np.arange(70, 450, 20)
fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
ys = np.array([X[c]["entry_along"] for c in X]); outs = np.array([X[c]["outcome"] for c in X])
for o in ["through", "hidden", "bounce"]:
    axs[0].hist(ys[outs == o], bins=bins, color=OC[o], alpha=0.85, label=o, histtype="stepfilled", edgecolor="white", linewidth=1)
axs[0].set(ylabel="passes", title="Right-entry (training-side) passes by height where they reach the plank")
axs[0].legend(frameon=False, fontsize=8, ncol=3)
def prof(v, key):
    R = J(f"scores/{v}/per_clip.json"); y = np.array([X[r["clip_id"]]["entry_along"] for r in R]); s = np.array([max(r[key]) for r in R])
    idx = np.digitize(y, bins); xs, ms = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 3: xs.append(bins[i - 1] + 10); ms.append(s[m].mean())
    return xs, ms
xs, ms = prof("commit-after", "real_one_ball_far"); axs[1].plot(xs, ms, color=C["real frames"], lw=2, ls="--", label="real frames (ceiling)")
for v, lab in [("pretrained", "pretrained"), ("plain-after", "plain"), ("commit-after", "commit"), ("codes-after", "codes")]:
    xs, ms = prof(v, "imagined_one_ball_far"); axs[1].plot(xs, ms, color=C[lab], lw=2, marker="o", ms=4, label=lab)
for ax in axs:
    for g, n in [(GAP_A, "wide gap"), (GAP_B, "narrow gap")]:
        ax.axvspan(*g, color="#1baf7a", alpha=0.07, lw=0)
axs[1].text(np.mean(GAP_A), 1.02, "wide gap (83 through)", ha="center", fontsize=8, color="#52514e")
axs[1].text(np.mean(GAP_B), 1.02, "narrow gap (28 through)", ha="center", fontsize=8, color="#52514e")
axs[1].set(ylim=(0, 1.08), xlabel="height where the ball reaches the plank (px, 512 px frame)", ylabel="P(ball beyond plank), peak over the future",
           title="Imagined ball beyond the plank, by height (each point: mean of passes in a 20 px band)")
axs[1].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "plan_height_profile_right.png", dpi=150); plt.close(fig)

# 3. left clips
G = J("generalise/commit/per_clip.json")
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, (k, t) in zip(axs, [("real", "real future frames"), ("before", "pretrained predictor"), ("after", "commit, after post-training")]):
    ax.axvspan(*RANGE, color="#7a7974", alpha=0.10, lw=0)
    for o in ["bounce", "hidden", "through"]:
        R = [r for r in G if r["outcome"] == o]
        ax.scatter([r["entry_y"] for r in R], [max(r[f"{k}_far"]) for r in R], s=14, color=OC[o], label=o, edgecolor="white", linewidth=0.5)
    ax.set(title=t, xlabel="height at the plank (px)")
axs[0].set_ylabel("P(ball beyond plank)")
axs[0].text(np.mean(RANGE), 0.02, "blockade range fitted\non right-entry clips", ha="center", fontsize=7, color="#52514e")
axs[0].legend(frameon=False, fontsize=7, loc="lower left")
fig.suptitle("Left-entry clips (held out, uphill): the post-trained score is lowest in the narrow gap, where the ball really gets through", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "plan_left_height.png", dpi=150); plt.close(fig)

# 4. far score by step for through passes
fig, ax = plt.subplots(figsize=(6.5, 3.4))
R = [r for r in J("scores/commit-after/per_clip.json") if r["outcome"] == "through"]
ax.plot(range(1, 13), np.mean([r["real_one_ball_far"] for r in R], 0), color=C["real frames"], lw=2, ls="--", label="real frames")
for v, lab in [("pretrained", "pretrained"), ("plain-after", "plain"), ("commit-after", "commit"), ("codes-after", "codes")]:
    R = [r for r in J(f"scores/{v}/per_clip.json") if r["outcome"] == "through"]
    ax.plot(range(1, 13), np.mean([r["imagined_one_ball_far"] for r in R], 0), color=C[lab], lw=2, label=lab)
ax.set(xlabel="predicted step (each 2 frames, 1/8 s)", ylabel="mean P(ball beyond plank)", ylim=(0, 0.85),
       title="Through passes only: the imagined ball fades in the last steps")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "plan_through_by_step.png", dpi=150); plt.close(fig)
print("ok")

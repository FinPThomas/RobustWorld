"""See what a gripper policy does with a forecast: top-down views for both simulators (env.py, box_env.py).

For real passes (each ball comes out at its real exit, as in evaluate.py's replay), the same frozen policy
is driven by three forecasts: the world model's before post-training, after (C), and the tracker's own
exit (perfect). Each panel shows, seen from above, with the plank edge at the top:

  orange   the forecast: exit point with a one-sigma circle (where), the path its speed predicts, and a
           timeline bar with its exit time +- one sigma (when) against the real exit time (white tick)
  blue     the ball: hollow while the gripper cannot see it yet (not out, or within the sight lag)
  grey     the gripper's two fingers, at their current opening; its path so far as a line
  green/red the grasp point, and the outcome

1. passes.png: one row per pass, the whole episode drawn as paths (static, quick to read).
2. passes.mp4: the same passes played step by step, side by side.

Scoring only: the forecasts are the evaluation's (evaluate.py, fixed rules), nothing is fitted.

    python models/gripper/viz.py --plan results/2026-10-04_2110_twoday/vjepa2/plan --tag box_base
    python models/gripper/viz.py --plan <plan> --tag box_indep --fold 2 --passes 8
--tag reads the experiment's settings (simulator, overrides) from <plan>/gripper/<tag>/gripper.json and its
policies from runs/<tag>; --env/--params/--policy override them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FPS = 8                                         # half real speed (one policy step = 1/16 s)
X0, X1, Y0, Y1 = 0.0, 0.42, -0.035, 0.205     # metres shown: along the edge; past it (plank above y = 0)
PX = 1250                                       # pixels per metre
BAR = 30                                        # timeline bar height
COLS = ("before post-training", "after (C)", "perfect forecast")


def setup(args) -> None:
    """Set the experiment's simulator and overrides before env/evaluate are imported (they read them once)."""
    g = args.plan / "gripper" / args.tag / "gripper.json" if args.tag else None
    exp = json.loads(g.read_text()).get("experiment", {}) if g and g.exists() else {}
    os.environ["GRIPPER_ENV"] = args.env or exp.get("env", "pybullet")
    os.environ["GRIPPER_PARAMS"] = args.params or exp.get("params", "{}")
    sys.path.insert(0, str(HERE))


def record(E, env, policy, options: dict, seed: int = 0) -> dict:
    """Run one episode and keep what the views need at every policy step."""
    obs, _ = env.reset(seed=seed, options=options)
    steps = []

    def keep():
        out = env.t >= env.true["t"]
        steps.append({"t": env.t, "g": env.grip_xy().tolist(), "gap": float(env.gap()),
                      "b": env.ball_pos()[:2].tolist() if out else None, "seen": bool(env.visible())})
    keep()
    sim = getattr(env, "_sim", None)
    if sim is not None:            # PyBullet: also keep the grasp (fingers closing, lift) one frame at a time
        def sim_kept(n=E.SUBSTEPS):
            for k in range(0, n, E.SUBSTEPS):
                sim(min(E.SUBSTEPS, n - k))
                if env.grasping:
                    keep()
        env._sim = sim_kept
    while True:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        keep()
        if term or trunc:
            break
    if sim is not None:
        env._sim = sim
    elif info.get("opening") is not None:     # box: the grasp is solved at once; draw the fingers closing in time
        g, w0, t = np.array(steps[-1]["g"]), steps[-1]["gap"], steps[-1]["t"]
        w1 = 2 * E.BALL_R if info["caught"] else E.GAP_MIN
        for k in range(1, int(np.ceil((w0 - w1) / 2 / E.FINGER_SPEED / E.DT)) + 3):
            tk = t + k * E.DT
            b = env.ball_at(tk)
            steps.append({"t": tk, "g": g.tolist(), "gap": max(w1, w0 - 2 * E.FINGER_SPEED * k * E.DT),
                          "b": (g.tolist() if info["caught"] and k * E.DT * 2 * E.FINGER_SPEED >= w0 - w1
                                else (b.tolist() if b is not None else None)), "seen": True})
    return {"fc": dict(env.fc), "true": dict(env.true), "steps": steps, "result": info}


# ---- drawing (cv2, BGR) ---------------------------------------------------------------------------------
def _p(x, y):
    return int(round((x - X0) * PX)), int(round((y - Y0) * PX)) + BAR


def panel(E, rec: dict, i: int, label: str):
    import cv2
    w, h = int((X1 - X0) * PX), int((Y1 - Y0) * PX) + BAR
    im = np.full((h, w, 3), (52, 48, 44), np.uint8)
    cv2.rectangle(im, _p(X0, Y0), _p(X1, 0.0), (45, 80, 120), -1)                     # plank
    cv2.rectangle(im, _p(X0, 0.0), _p(X1, E.ZONE), (70, 64, 58), -1)                   # catch zone
    cv2.line(im, _p(X0, E.ZONE), _p(X1, E.ZONE), (60, 60, 200), 1)
    fc, tr, st = rec["fc"], rec["true"], rec["steps"]
    s = st[min(i, len(st) - 1)]
    # forecast: exit, one-sigma circle along the edge, predicted path for 0.6 s
    orange = (0, 150, 255)
    c = _p(fc["x"], 0.0)
    cv2.circle(im, c, max(int(fc["sig_s"] * PX), 3), orange, 1, cv2.LINE_AA)
    cv2.circle(im, c, 4, orange, -1, cv2.LINE_AA)
    cv2.arrowedLine(im, c, _p(fc["x"] + fc["vx"] * 0.6, fc["vy"] * 0.6), orange, 1, cv2.LINE_AA, tipLength=0.08)
    # timeline: 0..MAX_T, forecast exit time +- sigma, real exit time, now
    tx = lambda t: int(np.clip(t / E.MAX_T, 0, 1) * (w - 1))  # noqa: E731
    cv2.rectangle(im, (tx(fc["t"] - fc["sig_t"]), 6), (tx(fc["t"] + fc["sig_t"]), BAR - 6), (0, 90, 160), -1)
    cv2.line(im, (tx(fc["t"]), 4), (tx(fc["t"]), BAR - 4), orange, 2)
    if np.isfinite(tr["t"]):                                                          # a blocked ball never comes out
        cv2.line(im, (tx(tr["t"]), 2), (tx(tr["t"]), BAR - 2), (255, 255, 255), 2)
    cv2.line(im, (tx(s["t"]), 0), (tx(s["t"]), BAR), (120, 220, 120), 1)
    # paths so far
    gp = np.array([_p(*x["g"]) for x in st[: i + 1]], np.int32)
    cv2.polylines(im, [gp], False, (150, 150, 150), 1, cv2.LINE_AA)
    bp = np.array([_p(*x["b"]) for x in st[: i + 1] if x["b"]], np.int32)
    if len(bp) > 1:
        cv2.polylines(im, [bp], False, (200, 120, 40), 1, cv2.LINE_AA)
    # ball (hollow until the gripper can see it) and fingers at their opening
    if s["b"]:
        cv2.circle(im, _p(*s["b"]), int(E.BALL_R * PX), (230, 140, 40), -1 if s["seen"] else 2, cv2.LINE_AA)
    gx, gy = s["g"]
    for side in (-1, 1):
        x = gx + side * (s["gap"] / 2 + 0.006)
        cv2.rectangle(im, _p(x - 0.006, gy - 0.02), _p(x + 0.006, gy + 0.02), (210, 210, 210), -1)
    cv2.putText(im, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if i >= len(st) - 1:
        ok = rec["result"].get("caught") or rec["result"].get("good")
        cv2.drawMarker(im, _p(gx, gy), (80, 220, 80) if ok else (80, 80, 230), cv2.MARKER_TILTED_CROSS, 18, 2)
        txt = rec["result"].get("label") or ("PICKED UP" if ok else f"missed: {rec['result'].get('why')}")
        cv2.putText(im, txt, (w - 10 - 12 * len(txt), BAR + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (80, 220, 80) if ok else (80, 80, 230), 2, cv2.LINE_AA)
    return im


def video(E, rows: list, path: Path) -> None:
    """rows: [(clip_id, [rec per column])]; plays each pass's columns side by side."""
    import cv2
    from robust_world.media import write_video
    frames = []
    for clip_id, recs in rows:
        n = max(len(r["steps"]) for r in recs) + FPS          # hold the end for a second
        for i in range(n):
            f = np.concatenate([panel(E, r, i, f"{clip_id}  {c}") for r, c in zip(recs, COLS)], axis=1)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    write_video(path, frames, FPS)


def sheet(E, rows: list, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.4 * len(rows) + 0.6), squeeze=False)
    for (clip_id, recs), ax_row in zip(rows, axes):
        for rec, col, ax in zip(recs, COLS, ax_row):
            fc, st, res = rec["fc"], rec["steps"], rec["result"]
            ax.add_patch(Rectangle((X0, Y0), X1 - X0, -Y0, color="#785a3c"))
            ax.add_patch(Rectangle((X0, 0), X1 - X0, E.ZONE, color="#efe9e1"))
            ax.add_patch(Circle((fc["x"], 0), max(fc["sig_s"], 0.003), fill=False, color="tab:orange"))
            ax.annotate("", (fc["x"] + fc["vx"] * 0.6, fc["vy"] * 0.6), (fc["x"], 0),
                        arrowprops={"arrowstyle": "->", "color": "tab:orange", "ls": "--"})
            b = np.array([x["b"] for x in st if x["b"]])
            if len(b):
                ax.plot(b[:, 0], b[:, 1], "-", color="tab:blue", lw=1.5, label="ball")
            g = np.array([x["g"] for x in st])
            ax.plot(g[:, 0], g[:, 1], "-", color="0.35", lw=1, label="gripper")
            ok = res.get("caught") or res.get("good")
            ax.plot(*g[-1], "X", ms=10, color="tab:green" if ok else "tab:red")
            ax.plot([g[-1][0] - st[-1]["gap"] / 2, g[-1][0] + st[-1]["gap"] / 2], [g[-1][1]] * 2, "-",
                    color="tab:green" if ok else "tab:red", lw=3, alpha=0.6)
            ax.set(xlim=(X0, X1), ylim=(Y1, Y0), aspect="equal", xticks=[], yticks=[])
            ax.set_title(f"{clip_id} {col}: {res.get('label') or ('picked up' if ok else res.get('why'))}\n"
                         f"forecast t = {fc['t']:.2f} +- {fc['sig_t']:.2f} s, real {rec['true']['t']:.2f} s",
                         fontsize=8, color="tab:green" if ok else "tab:red")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--tag", default="", help="experiment: settings from its gripper.json, policies from runs/<tag>")
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--env", choices=("pybullet", "box"), default=None)
    ap.add_argument("--params", default=None, help="GRIPPER_PARAMS JSON")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    setup(args)
    sys.path.insert(0, str(REPO / "src"))
    import evaluate as V
    from exits import Edge
    E = V.E
    exits = E.load_exits()
    pols = V.load_policies(args.policy or (HERE / "runs" / (args.tag or "ppo")))
    pol = pols.get(args.fold) or E.trust_forecast
    poly = json.loads(args.scene.read_text())["occluder_polygon"]
    edges = {s: Edge(poly, s) for s in "RL"}
    per = {n: {r["clip_id"]: r for r in json.loads((args.plan / "scores" / n / "per_clip.json").read_text())}
           for n in (V.BEFORE, V.AFTER)}
    # the fold's held-out passes, in order, that the after model gives a forecast for (else all panels match)
    held = [e for e in exits["exits"] if e["fold"] == args.fold]
    shown = [e for e in held if V.wm_forecast(per[V.AFTER][e["clip_id"]], edges[e["side"]], exits["prior"])]
    shown = shown[: args.passes]
    env = E.CatchEnv(dict(exits, exits=held))
    rows = []
    for e in shown:
        fcs = [V.wm_forecast(per[n][e["clip_id"]], edges[e["side"]], exits["prior"]) or V.prior_forecast(exits["prior"])
               for n in (V.BEFORE, V.AFTER)] + [V.true_forecast(e)]
        rows.append((e["clip_id"], [record(E, env, pol, {"exit": e, "forecast": f}) for f in fcs]))
    env.close()
    dest = args.plan / "gripper" / args.tag if args.tag else args.plan / "gripper"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{args.tag or 'policy'} ({os.environ['GRIPPER_ENV']}, {os.environ['GRIPPER_PARAMS']}), fold {args.fold}"
    sheet(E, rows, dest / "passes.png", f"{name}: held-out real passes that the after model forecasts")
    print("wrote", dest / "passes.png", flush=True)
    if not args.no_video:
        video(E, rows, dest / "passes.mp4")
        print("wrote", dest / "passes.mp4", flush=True)


if __name__ == "__main__":
    main()

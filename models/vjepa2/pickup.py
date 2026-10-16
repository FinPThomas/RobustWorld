"""Pick-up test P1 (NEXT_STEPS_PLAN.md section 9): a scripted gripper on a catch line, driven by each
model's forecast, with the ball replayed from the tracker's real track.

Nothing here is trained. The gripper is a fixed rule and the forecast comes from per_clip.json
(ball_probe_cv.py: the frozen evaluation decoder read on the imagined target half of the CV fold that
never trained on the clip). The tracker only scores (CLAUDE.md rules 5-6). The threshold theta is swept,
not chosen, so the headline number (pick-up AUC) is threshold-free like the outcome AUROC.

Set-up (pixels of the 512 px clip frame, 1 step = 2 frames = 1/8 s):
    catch line   the plank's far edge (configs/scenes/whole.json) moved LINE_PX further out; for a
                 ball from the right the far edge is the plank's left edge, and vice versa
    decision     the end of the context half (frame 24); the gripper then has the 12 target steps (1.5 s)
    gripper      starts on the catch line at frame height y = 256, slides along the line at SPEED px/s
                 (no acceleration limit), catches if it is within CATCH_PX of the ball when the ball
                 crosses the line
    truth        the real ball "arrives" if its tracked centre crosses the line during the target half;
                 crossing point and time are interpolated between the last step before and the first after

Drivers (each one decides go / stay at the decision point and where to go):
    world model  P(arrives) = max over steps of imagined_far (the score behind the outcome AUROC);
                 where/when = the first imagined step whose decoded ball (peak >= CONF) is past the line.
                 Rule: go if P(arrives) >= theta and the forecast says where; otherwise stay.
    straight     the last two tracked context positions extrapolated at constant speed (no blockade, no
                 slope); goes whenever that line reaches the catch line within the target half
    reactive     no forecast: starts when the real ball first shows beyond the plank's far edge and keeps
                 heading to where it currently is
    oracle       the real future

Metrics, per driver and gripper speed:
    success      share of arriving balls caught
    wasted       share of non-arriving balls (hidden, bounce) the gripper moved for
    pick-up AUC  area under success vs wasted as theta sweeps (oracle 1.0); for single-point drivers the
                 trapezoid through (0,0), the point and (1, its success)
    at 0.5       success and wasted at theta = 0.5

Outputs: <plan>/pickup/pickup.json, pickup.md, pickup.png

    python models/vjepa2/pickup.py --plan results/2026-10-04_2110_twoday/vjepa2/plan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

LINE_PX = 64                     # catch line: 2 cells beyond the plank's far edge
CATCH_PX = 40                    # gripper half-opening, about one ball diameter mid-table
CONF = 0.5                       # decoded ball counts where its peak cell is at least this
REST_Y = 256
STEP_S = 1 / 8
N_STEPS = 12
SPEEDS = [150, 300, 600, 1200, None]      # px/s along the line; None = instant
MAIN_SPEED = 300
N_BOOT = 500
BEFORE, AFTER = "both-before", "lossmix_both-after"


def step_time(k: float) -> float:
    """Time after the decision point of target step k's centroid (frames 2k, 2k+1 at 16 fps)."""
    return (k + 0.25) * STEP_S


class Line:
    """The catch line for balls entering from one side: signed distance past it and position along it."""

    def __init__(self, polygon: list, side_in: str):
        p = np.array(polygon, float)
        top, bot = p[:, 1].min(), p[:, 1].max()
        top_pts, bot_pts = p[p[:, 1] == top], p[p[:, 1] == bot]
        if side_in == "R":           # far edge = the plank's left edge
            a, b = top_pts[top_pts[:, 0].argmin()], bot_pts[bot_pts[:, 0].argmin()]
        else:                        # far edge = the right edge, whose top corner is the step at y = 30
            a, b = p[p[:, 0].argmax()], bot_pts[bot_pts[:, 0].argmax()]
        self.dir = (b - a) / np.linalg.norm(b - a)
        n = np.array([self.dir[1], -self.dir[0]])
        self.normal = n if (n[0] < 0) == (side_in == "R") else -n      # points to the far side
        self.origin = a + LINE_PX * self.normal
        self.edge = a

    def past(self, xy) -> float:
        return float(np.dot(np.asarray(xy[:2], float) - self.origin, self.normal))

    def past_edge(self, xy) -> float:
        return float(np.dot(np.asarray(xy[:2], float) - self.edge, self.normal))

    def along(self, xy) -> float:
        return float(np.dot(np.asarray(xy[:2], float) - self.origin, self.dir))

    def rest(self) -> float:
        t = (REST_Y - self.origin[1]) / self.dir[1]
        return float(t)


def crossing(points: list, line: Line, conf: float | None = None):
    """(s, t) where a per-step track first crosses the line, or None. Points may be None or [x, y(, peak)]."""
    prev = None
    for k, p in enumerate(points):
        if p is None or (conf is not None and p[2] < conf):
            continue
        d = line.past(p)
        if d >= 0:
            if prev is not None and prev[2] < 0:
                kp, pp, dp = prev
                f = -dp / (d - dp)
                s = line.along(pp) + f * (line.along(p) - line.along(pp))
                return s, step_time(kp + f * (k - kp))
            return line.along(p), step_time(k)
        prev = (k, p, d)
    return None


def straight_line(ctx: list, line: Line):
    """Constant-speed extrapolation of the last two tracked context steps to the catch line."""
    seen = [(k, p) for k, p in enumerate(ctx) if p is not None]
    if len(seen) < 2:
        return None
    (k0, p0), (k1, p1) = seen[-2:]
    v = (np.array(p1[:2]) - np.array(p0[:2])) / (k1 - k0)
    d1, dv = line.past(p1), float(np.dot(v, line.normal))
    if dv <= 0:
        return None
    k = (-d1) / dv                                   # steps after the last context step
    end = len(ctx) - 1
    t = (k1 + k - end - 0.75) * STEP_S                # back to target-step time (step 0 at 0.25)
    if t < 0 or t > step_time(N_STEPS - 1):
        return None
    return line.along(np.array(p1[:2]) + k * v), t


def gripper_at(s0: float, target: float, t: float, speed) -> float:
    if speed is None:
        return target
    move = min(speed * t, abs(target - s0))
    return s0 + np.sign(target - s0) * move


def reactive(true_pos: list, line: Line, t_cross: float, speed) -> float:
    """Gripper position at t_cross when it only sees the real ball once it is beyond the plank."""
    g, t_prev, goal = line.rest(), None, None
    for k, p in enumerate(true_pos):
        t = step_time(k)
        if t > t_cross:
            break
        if goal is not None:
            g = gripper_at(g, goal, t - t_prev, speed)
        if p is not None and line.past_edge(p) >= 0:
            goal = line.along(p)
        t_prev = t if goal is not None else None
    if goal is not None and t_prev is not None:
        g = gripper_at(g, goal, t_cross - t_prev, speed)
    return g


def caught(plan, truth, s0: float, speed) -> bool:
    if plan is None or truth is None:
        return False
    s_true, t_true = truth
    return abs(gripper_at(s0, plan[0], t_true, speed) - s_true) <= CATCH_PX


def curve(score: np.ndarray, success: np.ndarray, arrives: np.ndarray):
    """(wasted, success) points as theta sweeps; success counts only when the driver goes."""
    pts = [(0.0, 0.0)]
    for th in sorted(set(score.tolist()), reverse=True):
        go = score >= th
        pts.append((float((go & ~arrives).sum() / max((~arrives).sum(), 1)),
                    float((go & success & arrives).sum() / max(arrives.sum(), 1))))
    last = pts[-1]
    if last[0] < 1.0:
        pts.append((1.0, last[1]))
    return pts


def auc(pts) -> float:
    x, y = np.array(pts).T
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def evaluate(rows: list, polygon: list, driver: str, speed) -> dict:
    """Per-clip arrays for one driver at one gripper speed."""
    lines = {s: Line(polygon, s) for s in "RL"}
    score, success, arrives = [], [], []
    for r in rows:
        line = lines[r.get("side_in", "R")]
        truth = crossing(r["true_pos"], line)
        s0 = line.rest()
        if driver == "oracle":
            sc, plan = float(truth is not None), truth
        elif driver == "straight":
            plan = straight_line(r.get("true_pos_context") or [], line)
            sc = float(plan is not None)
        elif driver == "reactive":
            sc, plan = 1.0, None
        else:
            plan = crossing(r["imagined_pos"], line, CONF)
            sc = max(r["imagined_far"]) if plan is not None else 0.0
        if driver == "reactive":
            ok = truth is not None and abs(reactive(r["true_pos"], line, truth[1], speed) - truth[0]) <= CATCH_PX
        else:
            ok = caught(plan, truth, s0, speed)
        score.append(sc), success.append(ok), arrives.append(truth is not None)
    return {"score": np.array(score), "success": np.array(success), "arrives": np.array(arrives)}


def summarise(e: dict, reactive_driver: bool = False) -> dict:
    s, ok, a = e["score"], e["success"], e["arrives"]
    if reactive_driver:      # never moves for a ball that does not come out on the far side
        succ = float(ok[a].mean()) if a.any() else float("nan")
        pts = [(0.0, 0.0), (0.0, succ), (1.0, succ)]
        return {"pickup_auc": round(auc(pts), 3), "success_05": round(succ, 3), "wasted_05": 0.0}
    go = s >= 0.5
    return {"pickup_auc": round(auc(curve(s, ok, a)), 3),
            "success_05": round(float((go & ok)[a].mean()), 3),
            "wasted_05": round(float(go[~a].mean()), 3)}


def boot_delta(e0: dict, e1: dict, rng) -> list:
    n = len(e0["score"])
    ds = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        sub = lambda e: {k: v[i] for k, v in e.items()}      # noqa: E731
        ds.append(auc(curve(**sub(e1))) - auc(curve(**sub(e0))))
    return [round(float(np.percentile(ds, 2.5)), 3), round(float(np.percentile(ds, 97.5)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", type=Path, default=REPO / "outputs" / "vjepa2" / "plan")
    ap.add_argument("--scene", type=Path, default=REPO / "configs" / "scenes" / "whole.json")
    args = ap.parse_args()
    polygon = json.loads(args.scene.read_text())["occluder_polygon"]
    scores = args.plan / "scores"
    runs = {}
    for d in sorted(scores.iterdir()):
        f = d / "per_clip.json"
        if f.exists():
            rows = json.loads(f.read_text())
            if rows and "imagined_pos" in rows[0] and "true_pos_context" in rows[0]:
                runs[d.name] = rows
    if BEFORE not in runs or AFTER not in runs:
        raise SystemExit(f"need {BEFORE} and {AFTER} scores with per-step positions in {scores}")
    base = runs[BEFORE]
    ids = [r["clip_id"] for r in base]
    for name in list(runs):           # every driver is scored on the same clips
        by_id = {r["clip_id"]: r for r in runs[name]}
        if not all(i in by_id for i in ids):
            del runs[name]
            continue
        runs[name] = [by_id[i] for i in ids]

    out = {"settings": {"line_px": LINE_PX, "catch_px": CATCH_PX, "conf": CONF, "speeds_px_s": SPEEDS,
                        "clips": len(ids)}, "by_speed": {}, "by_side": {}}
    drivers = ["oracle", "reactive", "straight", *runs]
    rng = np.random.default_rng(0)
    evals = {}
    for v in SPEEDS:
        key = "instant" if v is None else str(v)
        out["by_speed"][key] = {}
        for drv in drivers:
            rows = runs.get(drv, base)
            e = evaluate(rows, polygon, drv, v)
            evals[(drv, key)] = e
            out["by_speed"][key][drv] = summarise(e, drv == "reactive")
    main_key = str(MAIN_SPEED)
    out["after_minus_before_auc_ci"] = boot_delta(evals[(BEFORE, main_key)], evals[(AFTER, main_key)], rng)
    a = evals[("oracle", main_key)]["arrives"]
    out["arriving_clips"] = {o: int(sum(1 for r, x in zip(base, a) if x and r["outcome"] == o))
                             for o in ("through", "hidden", "bounce")}
    out["outcome_counts"] = {o: sum(1 for r in base if r["outcome"] == o) for o in ("through", "hidden", "bounce")}
    for side in "RL":
        idx = np.array([r.get("side_in", "R") == side for r in base])
        out["by_side"][side] = {drv: summarise({k: v[idx] for k, v in evals[(drv, main_key)].items()},
                                               drv == "reactive")
                                for drv in drivers}
    # placement error where the forecast and the truth both cross
    place = {}
    for drv in [BEFORE, AFTER]:
        errs_s, errs_t = [], []
        for r in runs[drv]:
            line = Line(polygon, r.get("side_in", "R"))
            t, p = crossing(r["true_pos"], line), crossing(r["imagined_pos"], line, CONF)
            if t and p:
                errs_s.append(abs(p[0] - t[0])), errs_t.append(abs(p[1] - t[1]))
        place[drv] = {"n": len(errs_s), "median_px": round(float(np.median(errs_s)), 1) if errs_s else None,
                      "median_s": round(float(np.median(errs_t)), 3) if errs_t else None}
    out["placement"] = place

    dest = args.plan / "pickup"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pickup.json").write_text(json.dumps(out, indent=1))
    write_md(out, dest / "pickup.md", drivers)
    plot(evals, drivers, dest / "pickup.png")
    print((dest / "pickup.md").read_text())


def write_md(out: dict, path: Path, drivers: list) -> None:
    keys = list(out["by_speed"])
    lines = ["# Pick-up test P1: scripted gripper, replayed real tracks", "",
             f"Clips: {out['settings']['clips']} ({out['outcome_counts']}); arriving at the catch line in "
             f"the target half: {out['arriving_clips']}.",
             f"Catch line {LINE_PX} px beyond the plank, gripper half-opening {CATCH_PX} px. Nothing trained; "
             "theta swept. Generated by models/vjepa2/pickup.py.", "",
             "## Pick-up AUC by gripper speed (px/s along the line)", "",
             "| driver | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for d in drivers:
        lines.append(f"| {d} | " + " | ".join(f"{out['by_speed'][k][d]['pickup_auc']:.3f}" for k in keys) + " |")
    lines += ["", f"After minus before ({AFTER} vs {BEFORE}) at {MAIN_SPEED} px/s, 95% bootstrap CI: "
              f"{out['after_minus_before_auc_ci']}", "",
              f"## At theta = 0.5, {MAIN_SPEED} px/s", "", "| driver | caught | wasted moves |", "|---|---|---|"]
    for d in drivers:
        m = out["by_speed"][str(MAIN_SPEED)][d]
        lines.append(f"| {d} | {m['success_05']:.0%} | {m['wasted_05']:.0%} |")
    lines += ["", f"## By side, pick-up AUC at {MAIN_SPEED} px/s", "", "| driver | right entry | left entry |",
              "|---|---|---|"]
    for d in drivers:
        lines.append(f"| {d} | {out['by_side']['R'][d]['pickup_auc']:.3f} | {out['by_side']['L'][d]['pickup_auc']:.3f} |")
    lines += ["", "## Placement (clips where both the forecast and the real ball cross)", ""]
    for d, p in out["placement"].items():
        lines.append(f"- {d}: n = {p['n']}, median error {p['median_px']} px along the line, {p['median_s']} s")
    path.write_text("\n".join(lines) + "\n")


def plot(evals: dict, drivers: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = {"oracle": "black", "reactive": "tab:orange", "straight": "tab:green", BEFORE: "tab:red",
              AFTER: "tab:blue"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    key = str(MAIN_SPEED)
    for d in ["oracle", "straight", BEFORE, AFTER]:
        e = evals[(d, key)]
        x, y = np.array(curve(e["score"], e["success"], e["arrives"])).T
        a1.plot(x, y, label=d, color=colour[d], drawstyle="steps-post" if d in (BEFORE, AFTER) else "default", marker="o" if d == "straight" else None)
    a1.set_xlabel("wasted moves (share of hidden + bounce passes)")
    a1.set_ylabel("balls caught (share of arriving passes)")
    a1.set_title(f"Gripper at {MAIN_SPEED} px/s, theta swept")
    a1.legend(fontsize=8)
    speeds = [k for k in ("150", "300", "600", "1200", "instant")]
    epoch_runs = sorted([d for d in drivers if d.startswith("lossmix_both_e")],
                        key=lambda n: int(n.split("_e")[1].split("-")[0]))
    for d in ["oracle", "reactive", "straight", BEFORE, *epoch_runs, AFTER]:
        if (d, "300") not in evals:
            continue
        ys = [summarise(evals[(d, k)], d == "reactive")["pickup_auc"] for k in speeds]
        main_line = d in colour
        a2.plot(range(len(speeds)), ys, "-" if main_line else ":", marker="o", label=d,
                color=colour.get(d, str(0.8 - 0.6 * epoch_runs.index(d) / max(len(epoch_runs) - 1, 1))
                                 if d in epoch_runs else None))
    a2.set_xticks(range(len(speeds)), speeds)
    a2.set_xlabel("gripper speed (px/s along the catch line)")
    a2.set_ylabel("pick-up AUC")
    a2.set_title("Speed sweep (grey dotted: C at epochs 1-15)")
    a2.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()

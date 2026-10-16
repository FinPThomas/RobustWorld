import json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import evaluate as V
from evaluate import E
from exits import Edge
import traces as T
import blockade_eval as B
REPO = "../.."
plan = f"{REPO}/results/2026-10-04_2110_twoday/vjepa2/plan"
tag = "pb_narrow_fs"
B.PICK = 3.0
exits = E.load_exits(); prior = exits["prior"]; by_id = {e["clip_id"]: e for e in exits["exits"]}
poly = json.load(open(f"{REPO}/configs/scenes/whole.json"))["occluder_polygon"]; edges = {s: Edge(poly, s) for s in "RL"}
rows = {k: {r["clip_id"]: r for r in json.load(open(f"{plan}/scores/{v}/per_clip.json"))} for k, v in (("before", V.BEFORE), ("after (C)", V.AFTER))}
sim = json.load(open(f"{plan}/gripper/{tag}/gripper.json"))["sim"]["PPO policy"]
us = np.array(sorted(float(u) for u in sim)); cs = np.array([sim[str(u)]["caught"] for u in us])
pols = V.load_policies(T.HERE / "runs" / tag)
summary = json.load(open(f"{plan}/gripper/{tag}/blockade_v2.json"))
env = E.CatchEnv(exits)
scen = [("blocked", "whole_0277"), ("uncertain", "whole_0320"), ("confident", "whole_0014")]
out = {"tag": tag, "sim_pickup_by_u": dict(zip(us.tolist(), cs.tolist())), "summary": summary["real"],
       "settings": {k: getattr(E, k) for k in ("ZONE", "GAP_MIN", "GAP_MAX", "BALL_R", "DT", "EARLY")}, "scenarios": []}
for name, cid in scen:
    blocked = cid not in by_id
    side = rows["after (C)"][cid].get("side_in", "R")
    e = by_id.get(cid)
    sc = {"name": name, "clip": cid, "blocked": blocked, "true_exit": None if blocked else {k: round(float(v), 4) for k, v in E.exit_to_m(e).items()} if hasattr(E, "exit_to_m") else None, "sources": {}}
    for src in ("before", "after (C)", "perfect"):
        if src == "perfect":
            p, fc = (0.0, V.prior_forecast(prior)) if blocked else (1.0, V.true_forecast(e))
        elif src == "before":          # variant 2: its "comes out" score never varies, so it counts as no forecast
            p, fc = B.BASE, V.prior_forecast(prior)
        else:
            p = B.peak_far(rows[src][cid])
            fc = V.wm_forecast(rows[src][cid], edges[side], prior) or V.prior_forecast(prior)
        c = float(np.interp(fc["u"], us, cs))
        go, stay = B.go_value(p, c)
        d = {"p": round(p, 3), "u": round(float(fc["u"]), 3), "c": round(c, 3), "go_value": round(go, 3), "stay_value": round(stay, 3),
             "threshold": round(1.5 / (c * (B.PICK + E.EARLY / 2) - (1 - c) * B.TRY_MISS + 2.5), 3), "set_off": bool(go > stay),
             "forecast": {k: round(float(fc[k]), 4) for k in ("x", "t", "vx", "vy", "sig_s", "sig_t") if k in fc}}
        if blocked:
            d["reward"] = -B.WASTED if d["set_off"] else B.HOLD
        elif d["set_off"]:
            ep = T.episode(env, pols[e["fold"]], {"exit": e, "forecast": fc}, seed=0)
            d["episode"] = ep
            d["reward"] = round(B.PICK + E.EARLY * (1 - ep["result"]["depth"]), 3) if ep["result"]["caught"] else -B.TRY_MISS
        else:
            d["reward"] = -B.LET_GO
        sc["sources"][src] = d
        print(name, cid, src, {k: d[k] for k in ("p", "u", "c", "set_off", "reward")}, d.get("episode", {}).get("result"))
    out["scenarios"].append(sc)
clips = json.load(open(f"{plan}/gripper/{tag}/blockade_v2_clips.json"))
out["strip"] = [[r["p"], int(r["blocked"]), int(r["set_off"])] for r in clips if r["source"] == "after (C)"]
out["pick"] = B.PICK
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))

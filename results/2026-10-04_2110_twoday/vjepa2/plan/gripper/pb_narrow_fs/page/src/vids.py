import json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, cv2, pybullet as p
import evaluate as V
from evaluate import E
import videos as VD
from exits import Edge
sys.path.insert(0, "../../src")
from robust_world.media import write_video
data = json.load(open(sys.argv[1])); outdir = sys.argv[2]
exits = E.load_exits(); prior = exits["prior"]; by_id = {e["clip_id"]: e for e in exits["exits"]}
pols = V.load_policies(VD.HERE / "runs" / data["tag"])
env = E.CatchEnv(exits)
NAMES = {"before": "No post-training", "after (C)": "Post-trained", "perfect": "Perfect forecast"}
STAY = lambda env, obs: np.array([0.0, 0.0, -1.0, 0.0])
def run(sc, src, v, nmax):
    e = by_id.get(sc["clip"])
    fc = dict(v["forecast"], u=v["u"])
    for k in ("x", "t", "vx", "vy"): fc.setdefault(k, 0.0)
    fc.setdefault("sig_s", 0.0); fc.setdefault("sig_t", 0.0)
    if "sig_v" not in fc: fc["sig_v"] = E.SIG_V * v["u"]   # as evaluate.to_m (gen stores no sig_v)
    pol = pols[e["fold"] if e else 0] if v["set_off"] else STAY
    obs, _ = env.reset(seed=0, options={"exit": e if e else None, "forecast": fc})
    m = VD.marker(env) if v["set_off"] else None
    frames, info = [], {}
    while len(frames) < nmax:
        frames.append(env.render())
        obs, r, term, trunc, info = env.step(pol(env, obs))
        if term or trunc: break
    frames.append(env.render())
    if m is not None: p.removeBody(m, physicsClientId=env.cid)
    if sc["blocked"]: verdict, good = ("wasted trip" if v["set_off"] else "held back: correct"), not v["set_off"]
    elif not v["set_off"]: verdict, good = "stayed: ball let go", False
    else: verdict, good = ("PICKED UP" if info.get("caught") else f"missed ({info.get('why')})"), bool(info.get("caught"))
    out = []
    for i, f in enumerate(frames):
        f = np.ascontiguousarray(f)
        cv2.rectangle(f, (0, 0), (480, 54), (25, 30, 36), -1)
        cv2.putText(f, NAMES[src], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(f, f"P(comes out) {v['p']:.2f}  bar {v['threshold']:.2f}  -> {'SETS OFF' if v['set_off'] else 'STAYS'}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        out.append(f)
    return out, verdict, good
for sc in data["scenarios"]:
    clips = []
    for src in ("before", "after (C)", "perfect"):
        fr, verdict, good = run(sc, src, sc["sources"][src], 30)
        clips.append((fr, verdict, good))
    n = max(len(c[0]) for c in clips) + 12
    panels = []
    for fr, verdict, good in clips:
        fr = fr + [fr[-1].copy() for _ in range(n - len(fr))]
        for f in fr[-12:]:
            cv2.putText(f, verdict, (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 200, 80) if good else (60, 80, 230)[::-1], 2, cv2.LINE_AA)
        panels.append(fr)
    frames = [np.concatenate([np.pad(pn[i], ((0, 0), (0, 4 if k < 2 else 0), (0, 0)), constant_values=255) for k, pn in enumerate(panels)], axis=1) for i in range(n)]
    write_video(f"{outdir}/{sc['name']}.mp4", frames, fps=8)
    cv2.imwrite(f"{outdir}/{sc['name']}.jpg", cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    print(sc["name"], n, [c[1] for c in clips])

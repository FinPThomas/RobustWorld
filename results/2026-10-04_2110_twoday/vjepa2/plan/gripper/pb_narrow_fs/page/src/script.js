const D = window.__DATA__;
const SRC = [["before","No post-training","none"],["after (C)","Post-trained","post"],["perfect","Perfect forecast","perfect"]];
const NS = "http://www.w3.org/2000/svg";
const el = (t, a = {}, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
const pct = v => Math.round(v * 100) + "%";
const sgn = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const cvar = k => `var(--${k})`;

// summary
const sum = document.getElementById("sum");
for (const [k, name, c] of SRC) {
  const m = D.summary[k];
  const d = document.createElement("div"); d.className = "col"; d.style.setProperty("--c", cvar(c));
  d.innerHTML = `<span class="label k-${c}">${name}</span><span class="big">${sgn(m.mean_reward)}</span><span class="muted" style="font-size:.85rem">mean reward per pass</span>
  <dl><dt>sets off when the ball comes out</dt><dd>${pct(m.set_off_through)}</dd><dt>sets off when it's blocked</dt><dd>${pct(m.set_off_blocked)}</dd><dt>picks up, when it sets off for a ball that comes out</dt><dd>${pct(m.picked_up / m.set_off_through)}</dd><dt>picks up (of all balls that come out)</dt><dd>${pct(m.picked_up)}</dd></dl>`;
  sum.appendChild(d);
}

// bar vs vagueness
const us = Object.keys(D.sim_pickup_by_u).map(Number).sort((a, b) => a - b);
const cs = us.map(u => D.sim_pickup_by_u[u.toFixed(1)] ?? D.sim_pickup_by_u[String(u)]);
const bars = cs.map(c => 1.5 / (c * (D.pick + D.settings.EARLY / 2) - (1 - c) * 0.5 + 2.5));
document.getElementById("crange").textContent = `${pct(cs[0])} → ${pct(cs[cs.length - 1])}`;
document.getElementById("brange").textContent = `${bars[0].toFixed(2)} to ${bars[bars.length - 1].toFixed(2)}`;
{
  const s = document.getElementById("barchart"), L = 46, R = 340, T = 16, B = 160;
  const y0 = 0.28, y1 = 0.40, X = u => L + u * (R - L), Y = v => B - (v - y0) / (y1 - y0) * (B - T);
  for (let v = 0.28; v <= 0.4001; v += 0.02) { el("line", { x1: L, x2: R, y1: Y(v), y2: Y(v), stroke: "var(--line)" }, s); el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, s).textContent = v.toFixed(2); }
  for (const u of [0, 0.5, 1]) el("text", { x: X(u), y: B + 16, "text-anchor": "middle" }, s).textContent = u;
  el("text", { x: (L + R) / 2, y: B + 32, "text-anchor": "middle" }, s).textContent = "forecast vagueness u (0 sharp, 1 no idea)";
  el("text", { x: L, y: 10 }, s).textContent = "bar on P(comes out) to set off";
  el("path", { d: us.map((u, i) => `${i ? "L" : "M"}${X(u)},${Y(bars[i])}`).join(""), fill: "none", stroke: "var(--post)", "stroke-width": 2 }, s);
  us.forEach((u, i) => el("circle", { cx: X(u), cy: Y(bars[i]), r: 3, fill: "var(--post)" }, s));
}

// strip chart
{
  const s = document.getElementById("strip"), L = 110, R = 880, X = p => L + p * (R - L);
  const rowsY = { 0: 50, 1: 120 };
  const lo = Math.min(...bars), hi = Math.max(...bars);
  el("rect", { x: X(lo), y: 18, width: X(hi) - X(lo), height: 140, fill: "var(--post)", opacity: 0.14 }, s);
  el("line", { x1: X(lo), x2: X(lo), y1: 18, y2: 158, stroke: "var(--post)", "stroke-dasharray": "3 3" }, s);
  el("text", { x: X(hi) + 6, y: 28 }, s).textContent = `bar ${lo.toFixed(2)}–${hi.toFixed(2)}`;
  for (const p of [0, 0.25, 0.5, 0.75, 1]) { el("line", { x1: X(p), x2: X(p), y1: 158, y2: 163, stroke: "var(--muted)" }, s); el("text", { x: X(p), y: 177, "text-anchor": "middle" }, s).textContent = p; }
  el("line", { x1: L, x2: R, y1: 158, y2: 158, stroke: "var(--line)" }, s);
  const cnt = { 0: [0, 0], 1: [0, 0] };
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (const [p, b, go] of D.strip) {
    cnt[b][0]++; cnt[b][1] += go;
    el("circle", { cx: X(p), cy: rowsY[b] + (rnd() - 0.5) * 34, r: 3.2, fill: go ? (b ? "var(--bad)" : "var(--post)") : "none", stroke: b ? "var(--bad)" : "var(--post)", "stroke-width": 1, opacity: 0.85 }, s);
  }
  el("text", { x: L - 10, y: rowsY[0] - 4, "text-anchor": "end" }, s).textContent = "comes out";
  el("text", { x: L - 10, y: rowsY[0] + 10, "text-anchor": "end" }, s).textContent = `${cnt[0][1]} of ${cnt[0][0]} go`;
  el("text", { x: L - 10, y: rowsY[1] - 4, "text-anchor": "end" }, s).textContent = "blocked";
  el("text", { x: L - 10, y: rowsY[1] + 10, "text-anchor": "end" }, s).textContent = `${cnt[1][1]} of ${cnt[1][0]} go`;
}

// scenarios
const Z = D.settings.ZONE, FW = 0.012, FL = 0.04, BR = D.settings.BALL_R;
const XD = [0.03, 0.25], YD = [-0.035, Z + 0.015], W = 260, H = W * (YD[1] - YD[0]) / (XD[1] - XD[0]);
const px = x => (x - XD[0]) / (XD[1] - XD[0]) * W, py = y => (y - YD[0]) / (YD[1] - YD[0]) * H;
const TEXT = {
  blocked: ["A blocked ball", "This ball is stopped by the hidden blocker and never comes out. The right move is to stay."],
  uncertain: ["An uncertain ball", "This ball comes out, but the post-trained model is only about half sure it will (51%), and vague on where."],
  confident: ["A ball the model is sure about", "This ball comes out, and the post-trained model is sure it will and gives a sharp exit."]
};
const views = [];
function truePath(te) { const pts = []; for (let t = 0; t <= 2.5; t += 0.02) { const y = te.vy * (t - te.t); if (y < 0) continue; if (y > Z - BR) break; pts.push([t, te.x + te.vx * (t - te.t), y]); } return pts; }
function drawTD(card, sc, v, c) {
  const box = document.createElement("div"); box.className = "td"; card.appendChild(box);
  const s = el("svg", { viewBox: `0 0 ${W} ${H.toFixed(1)}`, role: "img", "aria-label": "top-down view of the catch" }, box);
  el("rect", { x: 0, y: 0, width: W, height: py(0), class: "plank" }, s);
  el("rect", { x: 0, y: py(0), width: W, height: py(Z) - py(0), class: "zone" }, s);
  el("text", { x: 4, y: 11 }, s).textContent = "plank";
  el("text", { x: 4, y: py(Z) - 4 }, s).textContent = "edge of catch zone";
  const f = v.forecast, showF = v.p > 0.05 && !(sc.blocked && v.p === 0);
  if (showF && !sc.blocked) {
    const sg = Math.max(f.sig_s || 0, 0.002);
    el("rect", { x: px(f.x - sg), y: py(0) - 5, width: Math.max(px(f.x + sg) - px(f.x - sg), 2), height: 10, class: "fband" }, s);
    el("line", { x1: px(f.x), x2: px(f.x), y1: py(0) - 7, y2: py(0) + 7, class: "fline" }, s);
  }
  const ep = v.episode, parts = { trail: el("path", { class: "trail" }, s), btrail: el("path", { class: "btrail" }, s) };
  let ballPts = [];
  if (ep) ballPts = ep.steps.filter(st => st[5] >= -0.02).map(st => [st[0], st[4], st[5]]);
  else if (!sc.blocked) ballPts = truePath(sc.true_exit);
  parts.f1 = el("rect", { width: px(FW + XD[0]), height: py(FL + YD[0]) - py(YD[0]), class: "finger", rx: 1 }, s);
  parts.f2 = el("rect", { width: px(FW + XD[0]), height: py(FL + YD[0]) - py(YD[0]), class: "finger", rx: 1 }, s);
  parts.ball = el("circle", { r: px(BR + XD[0]), class: "ballc" }, s);
  const start = [0, 0.1366, 0.06, 0.045];
  const gSteps = ep ? ep.steps.map(st => [st[0], st[1], st[2], st[3]]) : [start];
  if (!ep) el("text", { x: px(start[1]), y: py(start[2] + FL / 2) + 14, "text-anchor": "middle" }, s).textContent = "stays put";
  const tEnd = Math.max(ep ? ep.steps[ep.steps.length - 1][0] : 0, ballPts.length ? ballPts[ballPts.length - 1][0] : 0);
  views.push({ parts, gSteps, ballPts, tEnd, render(t) {
    const g = gSteps.filter(st => st[0] <= t + 1e-9); const cur = g.length ? g[g.length - 1] : gSteps[0];
    parts.trail.setAttribute("d", g.length > 1 ? g.map((st, i) => `${i ? "L" : "M"}${px(st[1])},${py(st[2])}`).join("") : "");
    const hw = cur[3] / 2;
    parts.f1.setAttribute("x", px(cur[1] - hw - FW)); parts.f2.setAttribute("x", px(cur[1] + hw));
    for (const f of [parts.f1, parts.f2]) f.setAttribute("y", py(cur[2] - FL / 2));
    const b = ballPts.filter(p => p[0] <= t + 1e-9);
    parts.btrail.setAttribute("d", b.length > 1 ? b.map((p, i) => `${i ? "L" : "M"}${px(p[1])},${py(p[2])}`).join("") : "");
    if (b.length) { const q = b[b.length - 1]; parts.ball.setAttribute("cx", px(q[1])); parts.ball.setAttribute("cy", py(q[2])); parts.ball.removeAttribute("visibility"); }
    else parts.ball.setAttribute("visibility", "hidden");
  } });
}
const host = document.getElementById("scens");
for (const sc of D.scenarios) {
  const [title, blurb] = TEXT[sc.name];
  const wrap = document.createElement("div"); wrap.className = "scen";
  wrap.innerHTML = `<div class="scen-head"><h2 style="font-size:1.35rem">${title}</h2><span class="tag">${sc.clip} · ${sc.blocked ? "blocked" : "comes out"}</span></div><p class="note">${blurb}</p>`;
  const vid = document.createElement("video");
  Object.assign(vid, { src: `vid/${sc.name}.mp4`, poster: `vid/${sc.name}.jpg`, muted: true, loop: true, autoplay: !matchMedia("(prefers-reduced-motion: reduce)").matches, controls: true, playsInline: true });
  vid.className = "sim"; vid.setAttribute("aria-label", `PyBullet replay of ${sc.clip}`);
  const vwrap = document.createElement("div"); vwrap.className = "vwrap"; vwrap.appendChild(vid);
  const cap = document.createElement("span"); cap.className = "mono muted"; cap.textContent = "PyBullet replay, half speed. Left to right: no post-training, post-trained, perfect forecast. The orange disc is the forecast exit, ± its uncertainty."; vwrap.appendChild(cap);
  wrap.appendChild(vwrap);
  const trio = document.createElement("div"); trio.className = "trio"; wrap.appendChild(trio);
  for (const [k, name, c] of SRC) {
    const v = sc.sources[k];
    const card = document.createElement("div"); card.className = "card"; card.style.setProperty("--c", cvar(c));
    const u = v.u >= 0.999 ? "no idea (falls back to the average exit)" : v.u < 0.01 ? "sharp" : `vague (u ${v.u.toFixed(2)})`;
    let out, cls;
    if (sc.blocked) { out = v.set_off ? "Wasted trip" : "Correctly held back"; cls = v.set_off ? "bad" : "good"; }
    else if (!v.set_off) { out = "Ball let go"; cls = "bad"; }
    else if (v.episode && v.episode.result.caught) { out = `Picked up ${Math.round(v.episode.result.depth * 100)}% of the way into the zone`; cls = "good"; }
    else { out = "Set off but missed"; cls = "bad"; }
    card.innerHTML = `<h3>${name}</h3>
      <div class="row"><span>P(ball comes out)</span><span class="mono">${v.p.toFixed(2)}</span></div>
      <div class="meter"><i style="width:${Math.max(v.p * 100, 1)}%"></i><b style="left:${v.threshold * 100}%" title="bar ${v.threshold}"></b></div>
      <div class="row"><span>bar to set off</span><span class="mono">${v.threshold.toFixed(2)}</span></div>
      <div class="row"><span>where and when</span><span class="mono" style="text-align:right">${u}</span></div>
      <div class="decision ${v.set_off ? "go" : "stay"}" style="color:${v.set_off ? "var(--c)" : ""}">${v.set_off ? "Sets off" : "Stays"}</div>`;
    drawTD(card, sc, v, c);
    const o = document.createElement("div"); o.className = `out ${cls}`; o.innerHTML = `${out} <span class="mono">(${sgn(v.reward)})</span>`; card.appendChild(o);
    trio.appendChild(card);
  }
  host.appendChild(wrap);
}
const notes = document.createElement("p"); notes.className = "note";
notes.textContent = "With a narrow grip and slow-closing fingers the gripper has to guess: it lines up on the forecast before the ball appears and cannot snap onto a ball that comes out somewhere else. Without post-training it sets off for every ball on the average guess, wastes every blocked trip and picks up 43% of the balls that come out. The post-trained forecast places the exit about 3 times closer, so when it sets off it picks up 60% (perfect: 73%), and it stays for 96% of blocked balls. In the sure pass above, the average guess knocks the ball away while the post-trained gripper is in place.";
host.appendChild(notes);

const tMax = Math.max(...views.map(v => v.tEnd));
const clock = document.getElementById("clock");
views.forEach(v => v.render(tMax));
let raf = null;
document.getElementById("play").addEventListener("click", () => {
  if (raf) cancelAnimationFrame(raf);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches, dur = reduce ? 1 : (tMax * 1500), t0 = performance.now();
  const step = now => { const t = Math.min((now - t0) / dur, 1) * tMax; views.forEach(v => v.render(t)); clock.textContent = `t = ${t.toFixed(2)} s`; if (t < tMax) raf = requestAnimationFrame(step); else { raf = null; clock.textContent = "t = end"; } };
  raf = requestAnimationFrame(step);
});

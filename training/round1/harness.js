// Round-1 harness: run a reader copy (hw_exp.js / hw_exp2.js) + model on the owner's exercises in Chromium.
// node harness.js read   <reader.js> <model.json> <items.json> <ids.json|all> <out.json>
// node harness.js groups <reader.js> <model.json> <items.json> <ids.json|all> <out.json>
const { chromium } = require('playwright'); const fs = require('fs');
const [mode, readerP, modelP, itemsP, idsP, outP] = process.argv.slice(2);
const src = fs.readFileSync(readerP, 'utf8').replace('/*@model:hw*/null', fs.readFileSync(modelP, 'utf8'));
let items = JSON.parse(fs.readFileSync(itemsP));
if (idsP !== 'all') { const ids = new Set(JSON.parse(fs.readFileSync(idsP))); items = items.filter(i => ids.has(i.id)); }
const target = s => s.replace(/[\[\]>\s]/g, '');
function tokens(text) { const out = []; for (let i = 0; i < text.length; i++) { const c = text[i]; if (/\s/.test(c)) continue; if (c === '^') { let j = i + 1; while (j < text.length && /\d/.test(text[j])) { out.push('^' + text[j]); j++; } i = j - 1; continue; } out.push(c); } return out; }
function barMask(src) { const m = []; let depth = 0; for (let i = 0; i < src.length; i++) { const c = src[i]; if (c === '[') { depth++; continue; } if (c === ']') { depth--; continue; } if (c === '>' || /\s/.test(c)) continue;
  if (c === '^') { let j = i + 1; while (j < src.length && /\d/.test(src[j])) { m.push(false); j++; } i = j - 1; continue; } m.push(c === '/' && depth > 0); } return m; }
const X = items.map(it => { const T = tokens(target(it.promptText)), m = barMask(it.promptText);
  const vars = [...new Set(T.filter((t, i) => /^[a-z]$/.test(t) || t === '÷' || (t === '/' && !m[i])))];
  return { id: it.id, tpl: it.template, tgt: target(it.promptText), T, bar: m, vars, s: it.strokes.map(st => st.map(q => [q[0], q[1]])), old: it.readText }; });
(async () => {
  const b = await chromium.launch(); const p = await b.newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.setContent('<!doctype html><meta charset=utf8><body><script>' + src + ';window.MODEL=' + fs.readFileSync(modelP, 'utf8') + ';</script>');
  const PAR = process.env.PARAMS ? JSON.parse(process.env.PARAMS) : [null];
  const all = [];
  for (const par of PAR) {
  const res = await p.evaluate(({ X, mode, par }) => {
    if (par) { window.__P5 = par.P5; window.__P6 = par.P6; }
    const HW = window.__hw;
    // forced alignment of target tokens to runs of strokes in WRITING (time) order, scored by the model
    const M = window.MODEL, C = M.classes, L = M.layers.map(l => { const bin = atob(l.w), W = new Float32Array(bin.length); for (let i = 0; i < bin.length; i++) { let v = bin.charCodeAt(i); if (v > 127) v -= 256; W[i] = v * l.s; } return { W, b: l.b, n: l.in, o: l.out }; });
    function probs(st) { let h = HW.pre(st); L.forEach((l, li) => { const o = new Float32Array(l.o); for (let j = 0; j < l.o; j++) { let s = l.b[j]; const off = j * l.n; for (let i = 0; i < l.n; i++) s += l.W[off + i] * h[i]; o[j] = li < L.length - 1 ? (s > 0 ? s : 0) : s; } h = o; });
      let mx = -1e9; for (const v of h) mx = Math.max(mx, v); let z = 0; const e = Array.from(h, v => { const q = Math.exp(v - mx); z += q; return q; }); return e.map(v => v / z); }
    const bb = pts => { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } return { x0, x1, y0, y1, w: x1 - x0, h: y1 - y0 }; };
    function falign(x, HW) {
      const S = x.s.filter(s => s.length), m = S.length, T = x.T, n = T.length, B = S.map(bb);
      const hs = B.map(b => b.h).filter(h => h > 0).sort((a, b) => a - b); const cH = Math.max(8, hs[Math.floor(hs.length * 0.75)] || 20);
      const flat = b => b.w > 0.25 * cH && b.h < 0.4 * Math.max(b.w, 1), tiny = b => Math.max(b.w, b.h) < 0.3 * cH;
      const INF = 1e9, cost = (i, j, k) => {   // token i takes strokes j..k-1
        const t = T[i].replace('^', ''), run = B.slice(j, k), cnt = k - j;
        if (x.bar[i]) return cnt === 1 && flat(run[0]) && run[0].w > 0.5 * cH ? 0 : INF;
        if (t === '-') return cnt === 1 && flat(run[0]) ? 0 : INF;
        if (t === '=') return cnt === 2 && flat(run[0]) && flat(run[1]) && Math.min(run[0].x1, run[1].x1) - Math.max(run[0].x0, run[1].x0) > 0.3 * Math.min(run[0].w, run[1].w) ? 0 : INF;
        if (t === '.') return cnt === 1 && tiny(run[0]) ? 0 : INF;
        if (cnt > 3) return INF;
        const NS = { '+': [2, 2], '(': [1, 1], ')': [1, 1], '/': [1, 1], '1': [1, 2], 't': [2, 2], '÷': [3, 3], 'c': [1, 2], '0': [1, 2], 's': [1, 2], 'z': [1, 2], '2': [1, 2], '3': [1, 2], '6': [1, 2], '8': [1, 2], '9': [1, 2], 'a': [1, 2], 'd': [1, 2], 'b': [1, 2], 'p': [1, 2], 'y': [1, 2], 'n': [1, 2], 'm': [1, 2], 'x': [1, 2], '7': [1, 2], '4': [1, 2], '5': [1, 2], 'k': [1, 3] }[t] || [1, 3];
        const nTiny = run.filter(b => Math.max(b.w, b.h) < 0.15 * cH).length;   // a stray dot may ride along
        if (cnt - nTiny < NS[0] || cnt - nTiny > NS[1]) return INF;
        const u = bb(run.flatMap(b => [[b.x0, b.y0], [b.x1, b.y1]]));
        if (u.w > 1.8 * Math.max(cH, u.h) + 4) return INF;                      // too wide for one symbol
        for (let q = 1; q < cnt; q++) { const a = run[q]; const gap = Math.max(0, Math.max(a.x0, u.x0) - Math.min(a.x1, u.x1)); if (a.x0 > u.x1 + 0.3 * cH) return INF; }
        const ci = C.indexOf(t); if (ci < 0) return INF;
        return -Math.log(Math.max(1e-6, probs(S.slice(j, k))[ci])) + 0.3 * (cnt - 1);
      };
      const D = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(INF)), P = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(-1)); D[0][0] = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { if (D[i][j] >= INF) continue; for (let k = j + 1; k <= Math.min(m, j + 4); k++) { const c = cost(i, j, k); if (c >= INF) continue; if (D[i][j] + c < D[i + 1][k]) { D[i + 1][k] = D[i][j] + c; P[i + 1][k] = j; } } }
      if (D[n][m] >= INF) return { id: x.id, ok: false };
      const seg = []; let k = m; for (let i = n; i > 0; i--) { const j = P[i][k]; seg.unshift([j, k]); k = j; }
      const per = seg.map(([j, k], i) => { const u = bb(B.slice(j, k).flatMap(b => [[b.x0, b.y0], [b.x1, b.y1]])); return { t: T[i], j, k, c: cost(i, j, k), cx: (u.x0 + u.x1) / 2, x0: u.x0, x1: u.x1 }; });
      // spatial check independent of the model: outside fractions, symbols must go left to right
      const inFrac = i => x.bar.slice(Math.max(0, i - 4), i + 5).some(Boolean);
      for (let i = 1; i < per.length; i++) if (!inFrac(i) && !inFrac(i - 1) && per[i].cx < per[i - 1].cx - 0.1 * cH) return { id: x.id, ok: false, why: 'order' };
      for (let i = 1; i < per.length; i++) if (!inFrac(i) && !inFrac(i - 1) && per[i].x0 < per[i - 1].x0 - 0.2 * cH) return { id: x.id, ok: false, why: 'order' };
      return { id: x.id, ok: true, total: D[n][m], maxc: Math.max(...per.map(q => q.c)), per, S };
    }
    return X.map(x => {
      const inp = x.s.map(st => ({ t: 'pen', p: st.flatMap(q => [q[0], q[1], 0.5]) }));
      if (mode === 'falign') return falign(x, HW);
      if (mode === 'read') { const r = HW.readAnswer(inp, 'alg', x.vars) || { text: '', p: 0, alts: [] }; return { id: x.id, text: r.text, p: r.p, alts: r.alts || [] }; }
      // groups in reading order: row items by x0; a fraction = numerator groups, bar, denominator groups
      const { groups, cH } = HW.segment(inp);
      const allowed = [...new Set([...'0123456789()+', ...x.vars])];
      const { rest, fr } = HW.fractions(groups, cH, allowed);
      const used = groups.filter(g => !rest.includes(g));
      const units = [...rest.map(g => ({ x0: g.x0, list: [g] })), ...fr.map(f => {
        const mine = used.filter(g => g.cx >= f.x0 - 0.3 * cH && g.cx <= f.x1 + 0.3 * cH);
        const flat = mine.filter(g => g.strokes.length === 1 && g.h < 0.35 * g.w).sort((a, b) => Math.abs(a.cy - f.cy) - Math.abs(b.cy - f.cy));
        const bar = flat[0]; const up = mine.filter(g => g !== bar && g.cy < f.cy).sort((a, b) => a.x0 - b.x0), dn = mine.filter(g => g !== bar && g.cy > f.cy).sort((a, b) => a.x0 - b.x0);
        return { x0: f.x0, list: [...up, ...(bar ? [bar] : []), ...dn], isFrac: true }; })].sort((a, b) => a.x0 - b.x0);
      const G = units.flatMap(u => u.list).map(g => ({ s: g.strokes.map(q => q.pts), dash: g.strokes.every(q => q.h < 0.4 * Math.max(q.w, 1) && q.w > 0.25 * cH), tiny: Math.max(g.w, g.h) < 0.3 * cH, w: g.w, h: g.h }));
      return { id: x.id, G, cH }; });
  }, { X, mode, par });
  const outOne = mode === 'falign' ? res.map((r, i) => Object.assign(r, { T: X[i].T, bar: X[i].bar, tpl: X[i].tpl })) : mode === 'read' ? res.map((r, i) => Object.assign(r, { tgt: X[i].tgt, T: X[i].T, bar: X[i].bar, tpl: X[i].tpl, old: X[i].old })) : res.map((r, i) => Object.assign(r, { T: X[i].T, bar: X[i].bar, tpl: X[i].tpl }));
  all.push(outOne); }
  const out = PAR.length > 1 ? all : all[0];
  fs.writeFileSync(outP, JSON.stringify(out)); console.log(mode, out.length, errs.length ? errs.slice(0, 3) : 'no errors'); await b.close();
})();

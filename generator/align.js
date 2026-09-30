// Forced alignment of TRAIN exercises: target tokens <-> runs of strokes. Strict checks: stroke-count limits per symbol,
// geometry for - = . bar, classifier (v3) probability floor, left-to-right order outside fractions.
// Tries time order first, then time order with "delayed strokes" moved next to the strokes they overlap.
const fs = require('fs'); const { open } = require('./page.js'); const { trainItems } = require('./common.js');
(async () => {
  const items = trainItems(); const { b, p } = await open();
  const X = await p.evaluate(src => src.map(it => { const parts = it.promptText.split('>'); let T = [], bar = []; parts.forEach((q, k) => { if (k) { T.push('>'); bar.push(false); } T.push(...tokens(target(q))); bar.push(...barMask(q)); });
    if (T.filter(t => t !== '>').join('') !== tokens(target(it.promptText)).join('')) throw new Error('tok'); return { id: it.id, T, bar }; }), items.map(i => ({ promptText: i.promptText })));
  X.forEach((x, i) => { x.id = items[i].id; x.s = items[i].strokes.map((st, k) => [k, st.map(q => [q[0], q[1]])]).filter(z => z[1].length); });
  const res = await p.evaluate((X) => {
    const C = M.classes, L = M.layers.map(l => { const bin = atob(l.w), W = new Float32Array(bin.length); for (let i = 0; i < bin.length; i++) { let v = bin.charCodeAt(i); if (v > 127) v -= 256; W[i] = v * l.s; } return { W, b: l.b, n: l.in, o: l.out }; });
    const cache = new Map();
    function probs(st, key) { if (cache.has(key)) return cache.get(key); let h = HW.pre(st); L.forEach((l, li) => { const o = new Float32Array(l.o); for (let j = 0; j < l.o; j++) { let s = l.b[j]; const off = j * l.n; for (let i = 0; i < l.n; i++) s += l.W[off + i] * h[i]; o[j] = li < L.length - 1 ? (s > 0 ? s : 0) : s; } h = o; });
      let mx = -1e9; for (const v of h) mx = Math.max(mx, v); let z = 0; const e = Array.from(h, v => { const q = Math.exp(v - mx); z += q; return q; }); const r = e.map(v => v / z); cache.set(key, r); return r; }
    const bb = pts => { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } return { x0, x1, y0, y1, w: x1 - x0, h: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 }; };
    const NS = { '+': [1, 2], '(': [1, 2], ')': [1, 2], '/': [1, 2], '1': [1, 2], 't': [1, 2], '÷': [2, 4], 'c': [1, 1], '0': [1, 2], 's': [1, 2], 'z': [1, 2], '2': [1, 2], '3': [1, 2], '6': [1, 2], '8': [1, 2], '9': [1, 2], 'a': [1, 2], 'd': [1, 2], 'b': [1, 2], 'p': [1, 2], 'y': [1, 2], 'n': [1, 2], 'm': [1, 2], 'x': [1, 2], '7': [1, 2], '4': [1, 2], '5': [1, 2], 'k': [1, 3] };
    function run(x, order) {
      cache.clear();
      const S = order.map(k => x.s[k][1]), m = S.length, T = x.T, n = T.length, B = S.map(bb);
      const hs = B.map(b => b.h).filter(h => h > 0).sort((a, b) => a - b); const cH = Math.max(8, hs[Math.floor(hs.length * 0.75)] || 20);
      const straight = (P, b) => { const a = P[0], e = P[P.length - 1], dx = e[0] - a[0], dy = e[1] - a[1], Ln = Math.hypot(dx, dy) || 1; let dev = 0; for (const [u, v] of P) dev = Math.max(dev, Math.abs((u - a[0]) * dy - (v - a[1]) * dx) / Ln); return Ln >= 0.8 * b.w && Math.abs(dy) < 0.62 * Math.abs(dx) && dev <= Math.max(1.5, 0.18 * Ln); };
      const flat = q => { const b = B[q]; return b.w > 0.22 * cH && (b.h < 0.4 * Math.max(b.w, 1) || straight(S[q], b)); };
      const tiny = q => Math.max(B[q].w, B[q].h) < 0.3 * cH;
      const INF = 1e9, why = {};
      const cost = (i, j, k) => {
        const t = T[i].replace('^', ''), cnt = k - j, idx = [...Array(cnt).keys()].map(q => q + j), run = idx.map(q => B[q]);
        if (t === '>') { if (cnt > 2) return INF; return run.some(b => b.w > 0.5 * cH && b.h < 0.4 * b.w) && run.every(b => b.w > 0.1 * cH || b.h > 0.1 * cH) ? 0.5 : INF; }
        if (x.bar[i]) return cnt === 1 && flat(j) && B[j].w > 0.5 * cH ? 0 : INF;
        if (t === '-') return cnt === 1 && flat(j) && B[j].w < 2.2 * cH ? 0 : INF;
        if (t === '=') { if (cnt !== 2) return INF; const [a, c] = run; const dash = b => b.w > 0.12 * cH && b.h < 0.75 * b.w && b.h < 0.35 * cH; if (!dash(a) || !dash(c)) return INF;
          const ov = Math.min(a.x1, c.x1) - Math.max(a.x0, c.x0); if (ov < 0.3 * Math.min(a.w, c.w)) return INF; const dv = Math.abs(a.cy - c.cy); return dv > 0.08 * cH && dv < 0.8 * cH ? 0 : INF; }
        if (t === '÷') { if (cnt < 2 || cnt > 4) return INF; const dash = run.filter(b => b.w > 0.12 * cH && b.h < 0.65 * b.w && b.h < 0.45 * cH), dots = run.filter(b => Math.max(b.w, b.h) < 0.3 * cH && !dash.includes(b));
          if (dash.length > 2 || dots.length < (dash.length ? 1 : 2) || dash.length + dots.length !== cnt) return INF;
          if (!dash.length) { const [a, c] = dots; return Math.abs(a.cx - c.cx) < 0.35 * cH && Math.abs(a.cy - c.cy) > 0.2 * cH && Math.abs(a.cy - c.cy) < 1.0 * cH ? 0 : INF; }   // the owner's "÷" is often ":"
          const d = bb(dash.flatMap(b => [[b.x0, b.y0], [b.x1, b.y1]]));
          return dots.every(o => o.cx > d.x0 - 0.2 * cH && o.cx < d.x1 + 0.2 * cH && Math.abs(o.cy - d.cy) > 0.1 * cH && Math.abs(o.cy - d.cy) < 0.9 * cH) ? 0 : INF; }
        if (t === '.') return cnt === 1 && tiny(j) ? 0 : INF;
        if (cnt > (t === '÷' ? 4 : 3)) return INF;
        const ns = NS[t] || [1, 3]; if (cnt < ns[0] || cnt > ns[1]) return INF;
        const u = bb(run.flatMap(b => [[b.x0, b.y0], [b.x1, b.y1]]));
        if (u.w > 1.8 * Math.max(cH, u.h) + 4) return INF;
        if (t !== '÷' && idx.every(flat)) return INF;
        for (let q = 1; q < cnt; q++) { const a = run[q]; let ovx = Math.min(a.x1, u.x1) - Math.max(a.x0, u.x0); const ru = bb(run.slice(0, q).flatMap(b => [[b.x0, b.y0], [b.x1, b.y1]])); ovx = Math.min(a.x1, ru.x1) - Math.max(a.x0, ru.x0); if (ovx < -0.25 * cH) return INF; }
        const ci = C.indexOf(t); if (ci < 0) return INF;
        const pr = probs(S.slice(j, k), j + ':' + k)[ci];
        return -Math.log(Math.max(1e-6, pr)) + 0.3 * (cnt - 1) + ((t === '+' || t === 't') && cnt === 1 ? 2 : 0) + ('()/'.includes(t) && cnt === 2 ? 2 : 0);
      };
      const D = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(INF)), P = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(-1)); D[0][0] = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j <= m; j++) { if (D[i][j] >= INF) continue; if (T[i] === '>' && D[i][j] < D[i + 1][j]) { D[i + 1][j] = D[i][j]; P[i + 1][j] = j; } if (j === m) continue; for (let k = j + 1; k <= Math.min(m, j + 4); k++) { const c = cost(i, j, k); if (c >= INF) continue; if (D[i][j] + c < D[i + 1][k]) { D[i + 1][k] = D[i][j] + c; P[i + 1][k] = j; } } }
      if (D[n][m] >= INF) { let best = 0; for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) if (D[i][j] < INF) best = Math.max(best, i); return { ok: false, why: 'nopath@' + best + ':' + T[best] }; }
      const seg = []; let k = m; for (let i = n; i > 0; i--) { const j = P[i][k]; seg.unshift([j, k]); k = j; }
      const per = seg.map(([j, k], i) => { const u = bb(B.slice(j, k).flatMap(b => [[b.x0, b.y0], [b.x1, b.y1]])); const c = cost(i, j, k);
        return { t: T[i], bar: x.bar[i], arrowGone: j === k, strokes: order.slice(j, k).map(q => x.s[q][0]), c, x0: u.x0, x1: u.x1, y0: u.y0, y1: u.y1 }; });
      const arrow = per.filter(q => q.t === '>').map(q => q.strokes); for (let i = per.length - 1; i >= 0; i--) if (per[i].t === '>') per.splice(i, 1);
      const inFrac = i => per.slice(Math.max(0, i - 6), i + 7).some(q => q.bar);
      for (let i = 1; i < per.length; i++) if (!inFrac(i) && !inFrac(i - 1) && ((per[i].x0 + per[i].x1) / 2 < (per[i - 1].x0 + per[i - 1].x1) / 2 - 0.1 * cH || per[i].x0 < per[i - 1].x0 - 0.2 * cH)) return { ok: false, why: 'order', at: [i, per[i - 1].t, per[i].t, per[i - 1].x0, per[i - 1].x1, per[i].x0, per[i].x1, cH] };
      const maxc = Math.max(...per.map(q => q.c));
      if (maxc > -Math.log(0.03) + 0.6) return { ok: false, why: 'lowprob:' + per.find(q => q.c === maxc).t };
      return { ok: true, total: D[n][m], maxc, per, cH, arrow };
    }
    function reorder(x) { // delayed strokes: a stroke far left of the previous one goes right after the latest stroke it overlaps
      const B = x.s.map(z => bb(z[1])); const hs = B.map(b => b.h).sort((a, b) => a - b); const cH = Math.max(8, hs[Math.floor(hs.length * 0.75)] || 20);
      const out = []; for (let q = 0; q < x.s.length; q++) { const b = B[q]; const prev = out.length ? B[out[out.length - 1]] : null;
        if (prev && b.cx < prev.x0 - 0.5 * cH) { let at = -1, bo = 0; for (let r = out.length - 1; r >= 0; r--) { const o = B[out[r]], ov = Math.min(o.x1, b.x1) - Math.max(o.x0, b.x0); if (ov > bo * 1.0001 && ov > 0) { bo = ov; at = r; } }
          if (at >= 0) { out.splice(at + 1, 0, q); continue; } }
        out.push(q); }
      return out; }
    return X.map(x => { const o1 = [...x.s.keys()]; let r = run(x, o1); r.mode = 'time';
      if (!r.ok) { const o2 = reorder(x); if (o2.join() !== o1.join()) { const r2 = run(x, o2); if (r2.ok) { r = r2; r.mode = 'reorder'; } else r.why2 = r2.why; } }
      if (!r.ok) { const o3 = [...x.s.keys()].sort((a, c) => { const A = bb(x.s[a][1]), Cc = bb(x.s[c][1]); return A.cx - Cc.cx; }); const r3 = run(x, o3); if (r3.ok) { r = r3; r.mode = 'spatial'; } else r.why3 = r3.why; }
      r.id = x.id; r.T = x.T.filter(t => t !== '>'); r.barMask = x.bar.filter((b, i) => x.T[i] !== '>'); return r; });
  }, X);
  const ok = res.filter(r => r.ok); const why = {}; res.filter(r => !r.ok).forEach(r => { const k = r.why.replace(/@\d+/, ''); why[k] = (why[k] || 0) + 1; });
  console.log('aligned', ok.length, '/', res.length, 'time', ok.filter(r => r.mode === 'time').length, 'reorder', ok.filter(r => r.mode === 'reorder').length, 'spatial', ok.filter(r => r.mode === 'spatial').length);
  console.log(JSON.stringify(why));
  fs.writeFileSync('align_train.json', JSON.stringify(res)); await b.close();
})();

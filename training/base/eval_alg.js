// Evaluate a handwriting model THROUGH the app's own reader (src/js/hw.js) on the frozen algebra set.
// usage: node eval_alg.js <model.json> <testset_frozen dir> [digits_strokes.json] [--json out.json]
// Uses playwright's chromium (same canvas renderer family as the Android WebView).
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path');
const [modelP, dir, digitsP] = process.argv.slice(2);
const outJ = process.argv.includes('--json') ? process.argv[process.argv.indexOf('--json') + 1] : null;
const hwSrc = fs.readFileSync(process.env.HWJS || path.join(__dirname, '../../src/js/hw.js'), 'utf8')  // HWJS=exp/hw_exp.js for the reader experiment
  .replace('/*@model:hw*/null', fs.readFileSync(modelP, 'utf8'));
const LET = 'abcdkmnptysz';
const ALLOWED = [...'0123456789+x/()', ...LET, '÷'];

// expected app text for a linear CROHME caption (null = outside what the answer reader supports)
function expText(e) {
  const t = e.tex.split(' '); let s = '', i = 0;
  while (i < t.length) {
    const w = t[i];
    if (w === '_') return null;
    if (w === '^') { if (t[i + 1] !== '{') return null; let j = i + 2, g = ''; while (t[j] !== '}') { g += t[j]; j++; } if (!/^\d+$/.test(g)) return null; s += '^' + g; i = j + 1; continue; }
    if (w === '{' || w === '}') { i++; continue; }
    if (!/^[0-9+\-=().]$/.test(w) && !LET.includes(w) && w !== 'x') return null;
    s += w; i++;
  }
  return /[a-z]/.test(s) ? s : null;   // algebra only
}

(async () => {
  const b = await chromium.launch(); const p = await b.newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.setContent('<!doctype html><meta charset=utf8><body><script>' + hwSrc + '</script>');
  const load = f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
  const res = {};
  // ---- per-symbol, classifier symbols ----
  for (const [name, file] of [['real', 'real_symbols.json'], ['synth', 'synth_symbols.json']]) {
    const S = load(file).map(s => ({ c: s.c === '×' ? 'x' : s.c, s: s.s })).filter(s => ALLOWED.includes(s.c) || s.c === 'l' || s.c === 'X');
    res['sym_' + name] = await p.evaluate(({ S, ALLOWED }) => {
      const per = {}, conf = {};
      for (const it of S) { const r = __hw.classify(it.s, ALLOWED); const ok = r.ch === it.c; const k = it.c; per[k] = per[k] || [0, 0]; per[k][1]++; if (ok) per[k][0]++; else { const q = it.c + '→' + r.ch; conf[q] = (conf[q] || 0) + 1; } }
      const acc = Object.fromEntries(Object.entries(per).map(([k, [a, n]]) => [k, +(a / n).toFixed(3) + ' (' + n + ')']));
      let a = 0, n = 0; for (const [k, [x, y]] of Object.entries(per)) { if (k === 'l' || k === 'X') continue; a += x; n += y; }
      return { overall: +(a / n).toFixed(4), n, acc, conf, topConf: Object.entries(conf).sort((a, b) => b[1] - a[1]).slice(0, 25) };
    }, { S, ALLOWED });
  }
  // ---- whole expressions through readAnswer(…,'alg',vars) ----
  const real = load('real_exprs.json').map(e => ({ text: expText(e), s: e.s })).filter(e => e.text);
  const synth = load('synth_exprs.json');
  const pen = fs.existsSync(path.join(dir, 'synth_exprs_pen.json')) ? load('synth_exprs_pen.json') : [];
  for (const [name, E] of [['real', real], ['synth', synth], ['pen', pen]]) {
    if (!E.length) continue;
    res['expr_' + name] = await p.evaluate(E => {
      let ok = 0, unsure = 0, fx = 0; const bad = []; const sub = {}; const rec = [];
      // guard: a read may earn a ✗ only if it is a well-formed answer with the same letters as the correct answer
      // and the same shape (same count of =, fraction, power, parentheses), and not much shorter/longer
      const shape = t => [(t.match(/=/g) || []).length, t.includes('/'), t.includes('^'), (t.match(/[()]/g) || []).length, [...new Set(t.replace(/[^a-z]/g, ''))].sort().join('')].join('|');
      const wellFormed = t => { if (!t || /[+\-=^\/.]$/.test(t) || /^[+=^\/.)]/.test(t) || /[+\-=^\/.]{2,}/.test(t) || /\^\D/.test(t) || /\.\D|\D\./.test(t)) return false; let d = 0; for (const c of t) { if (c === '(') d++; if (c === ')') d--; if (d < 0) return false; } if (d) return false; if (/\(\)|\([+=^\/.]|[+\-=^\/.(]\)/.test(t)) return false; return true; };
      const guard = (read, exp) => wellFormed(read) && shape(read) === shape(exp) && Math.abs(read.length - exp.length) <= 1;
      for (const e of E) {
        const vars = [...new Set(e.text.replace(/[^a-z]/g, ''))];
        const strokes = e.s.map(st => ({ t: 'pen', p: st.flatMap(([x, y]) => [x, y, .5]) }));
        const r = __hw.readAnswer(strokes, 'alg', vars) || { text: '', sure: false, alts: [], p: 0 };
        const v = r.text === e.text ? 1 : (!r.sure || (r.alts || []).includes(e.text)) ? 0 : 2;
        rec.push([r.p || 0, r.text === e.text ? 1 : (r.alts || []).includes(e.text) ? 0 : 2, guard(r.text, e.text)]);
        const tags = ['all']; if (/^[^a-z]*x[^a-wyz]*$/.test(e.text)) tags.push('x-only'); if (e.text.includes('/')) tags.push('fraction'); if (/-/.test(e.text)) tags.push('minus'); if (/\^/.test(e.text)) tags.push('power'); if (/[()]/.test(e.text)) tags.push('parens'); if (/\./.test(e.text)) tags.push('decimal'); if (e.text.includes('=')) tags.push('equals');
        tags.forEach(t => { sub[t] = sub[t] || { n: 0, exact: 0, sureWrong: 0 }; sub[t].n++; if (v === 1) sub[t].exact++; if (v === 2) sub[t].sureWrong++; });
        if (v === 1) ok++;
        else if (v === 0) unsure++;
        else { fx++; if (bad.length < 20) bad.push(e.text + ' → ' + r.text + ' p' + r.p.toFixed(2)); }
      }
      const N = E.length; for (const k in sub) { sub[k].exact = +(sub[k].exact / sub[k].n).toFixed(3); sub[k].sureWrong = +(sub[k].sureWrong / sub[k].n).toFixed(3); }
      const sweep = {}; for (const t of [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.01]) { let sw = 0, un = 0, ex = 0; for (const [pp, c] of rec) { if (pp < t || c === 0) un++; else if (c === 2) sw++; else ex++; } sweep[t] = { sureWrong: +(sw / N).toFixed(3), unsure: +(un / N).toFixed(3), sureRight: +(ex / N).toFixed(3) };
        let gw = 0, gu = 0; for (const [pp, c, gd] of rec) { if (pp < t || c === 0 || !gd) gu++; else if (c === 2) gw++; } sweep[t].guardedSureWrong = +(gw / N).toFixed(4); sweep[t].guardedUnsure = +(gu / N).toFixed(3); }
      return { n: N, sweep, rec, exact: +(ok / N).toFixed(3), unsure: +(unsure / N).toFixed(3), sureWrong: +(fx / N).toFixed(3), sub, examples: bad };
    }, E);
  }
  // ---- old reference: MNIST-derived pen digits (as test_hw.js) ----
  if (digitsP && fs.existsSync(digitsP)) {
    const D = JSON.parse(fs.readFileSync(digitsP));
    res.digits_ref = await p.evaluate(D => {
      const mk = (st, dx, dy) => st.map(s => ({ t: 'pen', p: s.flatMap(([x, y]) => [x + dx, y + dy, .5]) }));
      let ok = 0; for (const d of D) { const r = __hw.readAnswer(mk(d.s, 100, 100), 'num'); if (r && r.text === String(d.d)) ok++; }
      let ok2 = 0, n2 = 0; for (let i = 0; i + 2 < D.length; i += 3) { const a = D[i], b = D[i + 1], c = D[i + 2]; const w = x => Math.max(...x.s.flat(2).filter((v, j) => j % 2 == 0));
        const s = [...mk(a.s, 100, 100), ...mk(b.s, 100 + w(a) + 6, 100 + (i % 7) - 3), ...mk(c.s, 100 + w(a) + w(b) + 12, 100 + (i % 5) - 2)]; const r = __hw.readAnswer(s, 'num'); n2++; if (r && r.text === `${a.d}${b.d}${c.d}`) ok2++; }
      return { single: +(ok / D.length).toFixed(4), triple: +(ok2 / n2).toFixed(4), n: D.length };
    }, D);
  }
  res.errors = errs;
  console.log(JSON.stringify(res, null, 1));
  if (outJ) fs.writeFileSync(outJ, JSON.stringify(res, null, 1));
  await b.close();
})();

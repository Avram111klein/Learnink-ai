// Run the trainer page's reader (makeHW + model v3, readAnswer("alg", vars)) on items; score like the page's judge().
// node realism.js synth <file.jsonl.gz> out.json   |   node realism.js real - out.json   (real = owner's TRAIN items)
const fs = require('fs'), zlib = require('zlib'); const { open } = require('./page.js'); const { trainItems } = require('./common.js');
(async () => {
  const [mode, file, outP] = process.argv.slice(2);
  const items = mode === 'real' ? trainItems() : zlib.gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(JSON.parse);
  const { b, p } = await open(); const res = [];
  for (let i = 0; i < items.length; i += 50) {
    const chunk = items.slice(i, i + 50).map(x => ({ id: x.id, src: x.promptText, s: x.strokes.filter(st => st.length).map(st => st.map(q => [q[0], q[1]])) }));
    res.push(...await p.evaluate(C => C.map(x => { const a = readerArgs(x.src); const inp = x.s.map(st => ({ t: 'pen', p: st.flatMap(q => [q[0], q[1], 0.5]) }));
      const r = HW.readAnswer(inp, a.atype, a.vars); const J = judge({ src: x.src }, r);
      return { id: x.id, tgt: J.tgt, read: J.readText, exact: J.exact, n: J.perTok.length, ok: J.nOk, errs: J.errs.map(e => e.key + '→' + e.to), extra: J.extra }; }), chunk));
  }
  await b.close();
  const N = res.reduce((a, r) => a + r.n, 0), OK = res.reduce((a, r) => a + r.ok, 0);
  const types = { sub: 0, del: 0, ins: 0, pow: 0 }, conf = {};
  for (const r of res) { for (const e of r.errs) { const to = e.split('→')[1]; types[to === '∅' ? 'del' : e.startsWith('^→') ? 'pow' : 'sub']++; conf[e] = (conf[e] || 0) + 1; } types.ins += r.extra.length; }
  const S = { n: res.length, tokens: N, symbolAcc: +(OK / N).toFixed(4), exact: +(res.filter(r => r.exact).length / res.length).toFixed(4),
    errPerTok: Object.fromEntries(Object.entries(types).map(([k, v]) => [k, +(v / N).toFixed(4)])),
    errShare: Object.fromEntries(Object.entries(types).map(([k, v]) => [k, +(v / Math.max(1, Object.values(types).reduce((a, b) => a + b, 0))).toFixed(3)])),
    topConf: Object.entries(conf).sort((a, b) => b[1] - a[1]).slice(0, 15) };
  fs.writeFileSync(outP, JSON.stringify({ summary: S, items: res })); console.log(JSON.stringify(S));
})();

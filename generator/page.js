// Shared: a headless Chromium page holding the trainer's reader (makeHW from trainer/index.html, verbatim) + model v3
// and the trainer's helper functions (tokens/target/barMask/align/judge/readerArgs), extracted from index.html.
const { chromium } = require('/home/claude/.npm-global/lib/node_modules/playwright'); const fs = require('fs');
const SP = '__WORKDIR__';
function extract() {
  const h = fs.readFileSync(SP + '/trainer/index.html', 'utf8');
  const a = h.indexOf('function makeHW(M){'), b = h.indexOf('\n}\n', h.indexOf('return {readAnswer,readExercises', a)) + 3;
  let src = h.slice(a, b);
  const fn = name => { const re = new RegExp('^(function ' + name + '\\(|const ' + name + '=)', 'm'); const m = re.exec(h); if (!m) throw new Error(name);
    let i = m.index, end = h.indexOf('\n', i); // single- or multi-line: take lines until next top-level decl
    let j = end + 1; while (j < h.length && !/^(function |const |let |\/\*|\/\/|<\/script>)/.test(h.slice(j, j + 12))) j = h.indexOf('\n', j) + 1;
    return h.slice(i, j); };
  for (const n of ['target', 'tokens', 'barMask', 'baseTok', 'readerArgs', 'align', 'judge']) src += '\n' + fn(n);
  return src;
}
async function open() {
  const b = await chromium.launch(); const p = await b.newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message));
  const model = fs.readFileSync(SP + '/trainer/model/current.json', 'utf8');
  await p.setContent('<!doctype html><meta charset=utf8><body><script>' + extract() + '\nwindow.M=' + model + ';window.HW=makeHW(window.M);</script>');
  const ok = await p.evaluate(() => typeof HW.readAnswer === 'function' && typeof judge === 'function');
  if (!ok || errs.length) throw new Error('page setup failed ' + errs.join(';'));
  return { b, p, errs };
}
module.exports = { open, SP };

const fs = require('fs'); const { SP } = require('./page.js');
// owner's real items; TRAIN only (split.json), non-discarded, non-warmup
function loadItems() {
  const dir = SP + '/trainer_data2/samples/__WRITER__/ex/';
  const all = []; for (const f of fs.readdirSync(dir)) if (f.endsWith('.json')) all.push(...JSON.parse(fs.readFileSync(dir + f)).items);
  return all;
}
function split() { return JSON.parse(fs.readFileSync(SP + '/hwdata/round2/split.json')); }
function trainItems() {
  const sp = split(), tr = new Set(sp.train), test = new Set([...sp.testA, ...sp.testB]);
  const it = loadItems().filter(x => tr.has(x.id) && !x.discarded && !x.warmup);
  for (const x of it) if (test.has(x.id) || !tr.has(x.id)) throw new Error('TEST leak ' + x.id);
  return it;
}
module.exports = { loadItems, split, trainItems };

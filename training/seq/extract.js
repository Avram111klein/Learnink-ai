// aligned token groups → seq features (node, reader/seq.js) + pixel probs (Chromium, reader/hw_exp.js classify, no `allowed`)
// node extract.js <repo> <items.json> <align.json> <out.json>
const fs=require('fs'), {chromium}=require('playwright');
const [repo,itemsP,alP,outP]=process.argv.slice(2);
const SEQ=require(repo+'/reader/seq.js');
const I=Object.fromEntries(JSON.parse(fs.readFileSync(itemsP)).map(x=>[x.id,x]));
const A=JSON.parse(fs.readFileSync(alP)).filter(a=>a.ok);
const CL=JSON.parse(fs.readFileSync(repo+'/models/v3_r5.json')).classes;
const rows=[];
for(const a of A){ const x=I[a.id], S=x.strokes.filter(s=>s.length), ctx=SEQ.context(S);
  a.per.forEach((p,i)=>{ if(a.bar[i]) return; const t=p.t.replace('^',''); if(!CL.includes(t)) return;   // '=', '-', '.', fraction bars: not one of the 28
    const g=S.slice(p.j,p.k); rows.push({id:a.id,sess:x.sessionId,i,t,sup:p.t.startsWith('^'),ns:g.length,f:Array.from(SEQ.features(g,ctx)),g:g.map(s=>s.map(q=>[q[0],q[1]]))}); }); }
(async()=>{ const b=await chromium.launch(), pg=await b.newPage(); const errs=[]; pg.on('pageerror',e=>errs.push(e.message));
  await pg.setContent('<!doctype html><meta charset=utf8><body><script>'+fs.readFileSync(repo+'/reader/hw_exp.js','utf8').replace('/*@model:hw*/null',fs.readFileSync(repo+'/models/v3_r5.json','utf8'))+'</script>');
  const P=await pg.evaluate(G=>G.map(g=>{ const r=window.__hw.classify(g); return [r.ch,r.p]; }),rows.map(r=>r.g));
  rows.forEach((r,k)=>{ r.pix=P[k][0]; r.pixp=P[k][1]; delete r.g; });
  fs.writeFileSync(outP,JSON.stringify({classes:CL,dim:SEQ.DIM,rows})); console.log(outP,rows.length,errs.length?errs:'no errors'); await b.close(); })();

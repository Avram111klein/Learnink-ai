// Shadow measurement: pixel (reader/hw_exp.js classify) vs sequence (reader/seq.js) per aligned symbol, in Chromium.
// No combination of the two engines. Writes per-symbol rows; stats are computed by measure.py.
// node measure.js <repo> <seq_model.json> <items.json> <align.json> <out.json>
const fs=require('fs'), {chromium}=require('playwright');
const [repo,modelP,itemsP,alP,outP]=process.argv.slice(2);
const I=Object.fromEntries(JSON.parse(fs.readFileSync(itemsP)).map(x=>[x.id,x]));
const CL=JSON.parse(fs.readFileSync(repo+'/models/v3_r5.json')).classes;
const X=[];   // one entry per aligned item: full strokes + token spans (28-class tokens only)
for(const a of JSON.parse(fs.readFileSync(alP)).filter(a=>a.ok)){ const S=I[a.id].strokes.filter(s=>s.length);
  X.push({id:a.id,S,toks:a.per.map((p,i)=>({i,t:p.t.replace('^',''),j:p.j,k:p.k,bar:a.bar[i]})).filter(q=>!q.bar&&CL.includes(q.t))}); }
const src=fs.readFileSync(repo+'/reader/hw_exp.js','utf8').replace('/*@model:hw*/null',fs.readFileSync(repo+'/models/v3_r5.json','utf8'))
  +'\n'+fs.readFileSync(repo+'/reader/seq.js','utf8').replace('/*@model:seq*/null',fs.readFileSync(modelP,'utf8'));
(async()=>{ const b=await chromium.launch(), pg=await b.newPage(); const errs=[]; pg.on('pageerror',e=>errs.push(e.message));
  await pg.setContent('<!doctype html><meta charset=utf8><body><script>'+src+'</script>');
  const res=await pg.evaluate(X=>{ const HW=window.__hw, SEQ=window.__seq, rows=[]; let tPix=0,tSeq=0,n=0;
    for(const x of X){ const ctx=SEQ.context(x.S);
      for(const q of x.toks){ const g=x.S.slice(q.j,q.k);
        let a=performance.now(); const rp=HW.classify(g.map(s=>s.map(p=>[p[0],p[1]]))); tPix+=performance.now()-a;
        a=performance.now(); const rs=SEQ.classify(g,ctx); tSeq+=performance.now()-a; n++;
        rows.push({id:x.id,i:q.i,t:q.t,ns:g.length,pix:rp.ch,pixp:rp.p,seq:rs.ch,seqp:rs.p,seqalt:rs.alt}); } }
    // backward compatibility: points without time → null
    const nul=[SEQ.classify([[[1,2],[3,4]]]),SEQ.classify([[[1,2,0.5],[3,4,0.5]]]),SEQ.classify(X[0].S.slice(0,1).map(s=>s.map(p=>[p[0],p[1]])))];
    // timing of the sequence engine alone, repeated (warm)
    let a=performance.now(), m=0; for(let r=0;r<5;r++) for(const x of X){ const ctx=SEQ.context(x.S); for(const q of x.toks){ SEQ.classify(x.S.slice(q.j,q.k),ctx); m++; } }
    return {rows,msPix:tPix/n,msSeq:tSeq/n,msSeqWarm:(performance.now()-a)/m,nullChecks:nul}; },X);
  res.errs=errs; fs.writeFileSync(outP,JSON.stringify(res)); console.log(outP,res.rows.length,'ms/sym pix',res.msPix.toFixed(3),'seq',res.msSeq.toFixed(3),'seq warm',res.msSeqWarm.toFixed(3),'null',JSON.stringify(res.nullChecks),errs.length?errs:'no errors'); await b.close(); })();

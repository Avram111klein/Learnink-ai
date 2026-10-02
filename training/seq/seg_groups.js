// The character groups readAnswer really forms (segment + merge + fractions, reader/hw_fuse.js with FUSE off = hw_exp.js),
// with the reader's final label per group and the pixel distribution over the allowed set; labelled by an alignment:
// a group whose strokes are exactly one aligned token's strokes → that token; otherwise "∅" (fragment / several symbols);
// groups equal to a token outside the 28 classes ('=', '-', '.', fraction bar) → skipped. Items without alignment → skipped.
// node seg_groups.js <repo> <items.json> <align.json> <out.json>
const fs=require('fs'), {chromium}=require('playwright');
const [repo,itemsP,alP,outP]=process.argv.slice(2);
const HWM=fs.readFileSync(repo+'/models/v3_r5.json','utf8'), CL=JSON.parse(HWM).classes;
const A=Object.fromEntries(JSON.parse(fs.readFileSync(alP)).filter(a=>a.ok).map(a=>[a.id,a]));
const tg=s=>s.replace(/[\[\]>\s]/g,'');
function tokens(t){ const o=[]; for(let i=0;i<t.length;i++){ const c=t[i]; if(/\s/.test(c)) continue; if(c==='^'){ let j=i+1; while(j<t.length&&/\d/.test(t[j])){ o.push('^'+t[j]); j++; } i=j-1; continue; } o.push(c); } return o; }
function barMask(src){ const m=[]; let d=0; for(let i=0;i<src.length;i++){ const c=src[i]; if(c==='['){d++;continue;} if(c===']'){d--;continue;} if(c==='>'||/\s/.test(c)) continue;
  if(c==='^'){ let j=i+1; while(j<src.length&&/\d/.test(src[j])){ m.push(false); j++; } i=j-1; continue; } m.push(c==='/'&&d>0); } return m; }
const X=JSON.parse(fs.readFileSync(itemsP)).filter(it=>A[it.id]).map(it=>{ const T=tokens(tg(it.promptText)), m=barMask(it.promptText);
  return {id:it.id,sess:it.sessionId,vars:[...new Set(T.filter((t,i)=>/^[a-z]$/.test(t)||t==='÷'||(t==='/'&&!m[i])))],S:it.strokes}; });
(async()=>{ const b=await chromium.launch(), pg=await b.newPage(); const errs=[]; pg.on('pageerror',e=>errs.push(e.message));
  await pg.setContent('<!doctype html><meta charset=utf8><body><script>'+fs.readFileSync(repo+'/reader/seq.js','utf8')+'\n'+fs.readFileSync(repo+'/reader/hw_fuse.js','utf8').replace('/*@model:hw*/null',HWM)+'</script>');
  const G=await pg.evaluate(X=>{ const HW=window.__hw; HW.setFuse(false); const out=[];
    for(const x of X){ const inp=x.S.map(st=>({t:'pen',p:st.flatMap(q=>[q[0],q[1],0.5]),tp:st})), allowed=[...new Set([..."0123456789()+",...x.vars])];
      HW.readAnswer(inp,'alg',x.vars); const L=HW.lastLayout(), ix=new Map(inp.map((s,k)=>[s,k]));
      for(const g of L.groups){ const idx=g.strokes.map(s=>ix.get(s.src)).sort((a,b)=>a-b);
        const d=HW.fuseGroup(idx.map(k=>x.S[k]),x.S,allowed);   // pixel distribution (classify with the same allowed set)
        out.push({id:x.id,idx,ch:g.ch==null?null:g.ch,p:g.p==null?null:g.p,dist:d.dist}); } }
    return out; },X);
  const rows=[]; const tok=Object.fromEntries(Object.values(A).map(a=>[a.id,new Map(a.per.map((p,i)=>[Array.from({length:p.k-p.j},(_,q)=>p.j+q).join('.'),{t:p.t.replace('^',''),bar:a.bar[i]}]))]));
  const sessOf=Object.fromEntries(X.map(x=>[x.id,x.sess]));
  let skipped=0;
  for(const g of G){ const m=tok[g.id].get(g.idx.join('.')); let t='∅';
    if(m){ if(m.bar||!CL.includes(m.t)){ skipped++; continue; } t=m.t; }
    rows.push(Object.assign(g,{t,sess:sessOf[g.id]})); }
  fs.writeFileSync(outP,JSON.stringify({rows,classes:CL}));
  const n0=rows.filter(r=>r.t==='∅').length; console.log(outP,'items',X.length,'groups',G.length,'kept',rows.length,'symbol',rows.length-n0,'∅',n0,'skipped(=,-,.,bar)',skipped,errs.length?errs:'no errors');
  await b.close(); })();

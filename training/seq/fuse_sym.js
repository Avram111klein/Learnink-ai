// Per-symbol measurement of the fused reader (reader/hw_fuse.js + reader/seq.js), base vs fusion, two kinds of groups:
//  "seg":     the groups readAnswer itself forms (segment/merge/fractions), final label after readAnswer, with FUSE off
//             (= hw_exp.js) and on; kept when the group's strokes are exactly one aligned 28-class token.
//  "aligned": the alignment's own token groups, classify(group, allowed) vs fuse (HW.fuseGroup), same allowed set.
// node fuse_sym.js <repo> <seq_model.json> <items.json> <align.json> <W json> <out.json>
const fs=require('fs'), {chromium}=require('playwright');
const [repo,seqP,itemsP,alP,Wj,outP]=process.argv.slice(2); const W=JSON.parse(Wj);
const HWM=fs.readFileSync(repo+'/models/v3_r5.json','utf8'), CL=JSON.parse(HWM).classes;
const A=Object.fromEntries(JSON.parse(fs.readFileSync(alP)).filter(a=>a.ok).map(a=>[a.id,a]));
const tg=s=>s.replace(/[\[\]>\s]/g,'');
function tokens(t){ const o=[]; for(let i=0;i<t.length;i++){ const c=t[i]; if(/\s/.test(c)) continue; if(c==='^'){ let j=i+1; while(j<t.length&&/\d/.test(t[j])){ o.push('^'+t[j]); j++; } i=j-1; continue; } o.push(c); } return o; }
function barMask(src){ const m=[]; let d=0; for(let i=0;i<src.length;i++){ const c=src[i]; if(c==='['){d++;continue;} if(c===']'){d--;continue;} if(c==='>'||/\s/.test(c)) continue;
  if(c==='^'){ let j=i+1; while(j<src.length&&/\d/.test(src[j])){ m.push(false); j++; } i=j-1; continue; } m.push(c==='/'&&d>0); } return m; }
const X=JSON.parse(fs.readFileSync(itemsP)).filter(it=>A[it.id]).map(it=>{ const T=tokens(tg(it.promptText)), m=barMask(it.promptText), a=A[it.id];
  return {id:it.id,vars:[...new Set(T.filter((t,i)=>/^[a-z]$/.test(t)||t==='÷'||(t==='/'&&!m[i])))],S:it.strokes,
    toks:a.per.map((p,i)=>({t:p.t.replace('^',''),idx:Array.from({length:p.k-p.j},(_,q)=>p.j+q),bar:a.bar[i]})).filter(q=>!q.bar&&CL.includes(q.t))}; });
(async()=>{ const b=await chromium.launch(), pg=await b.newPage(); const errs=[]; pg.on('pageerror',e=>errs.push(e.message));
  await pg.setContent('<!doctype html><meta charset=utf8><body><script>'+fs.readFileSync(repo+'/reader/seq.js','utf8').replace('/*@model:seq*/null',fs.readFileSync(seqP,'utf8'))+'\n'+fs.readFileSync(repo+'/reader/hw_fuse.js','utf8').replace('/*@model:hw*/null',HWM)+'</script>');
  const res=await pg.evaluate(({X,W})=>{ const HW=window.__hw, rows=[]; let tSeq=0, nSeq=0;
    for(const x of X){ const inp=x.S.map(st=>({t:'pen',p:st.flatMap(q=>[q[0],q[1],0.5]),tp:st})), allowed=[...new Set([..."0123456789()+",...x.vars])];
      const lab={}; for(const mode of ['base','fuse']){ HW.setFuse(mode==='fuse',{W}); HW.readAnswer(inp,'alg',x.vars); const L=HW.lastLayout(), ix=new Map(inp.map((s,k)=>[s,k]));
        lab[mode]=new Map(L.groups.map(g=>[g.strokes.map(s=>ix.get(s.src)).sort((a,b)=>a-b).join('.'),g.ch])); }
      HW.setFuse(false);
      for(const q of x.toks){ const key=q.idx.join('.');
        if(lab.base.has(key)) rows.push({kind:'seg',id:x.id,t:q.t,base:lab.base.get(key),fuse:lab.fuse.get(key)});
        HW.setFuse(false); const a=performance.now(); const r=HW.fuseGroup(q.idx.map(k=>x.S[k]),x.S,allowed); tSeq+=performance.now()-a; nSeq++;
        rows.push({kind:'aligned',id:x.id,t:q.t,base:r.base.ch,fuse:r.fused.ch,bp:r.base.p,fp:r.fused.p}); } }
    return {rows,msFuseGroup:tSeq/nSeq}; },{X,W});
  res.errs=errs; res.W=W; fs.writeFileSync(outP,JSON.stringify(res));
  console.log(outP,'seg',res.rows.filter(r=>r.kind==='seg').length,'aligned',res.rows.filter(r=>r.kind==='aligned').length,'ms/group(pixel+seq)',res.msFuseGroup.toFixed(2),errs.length?errs:'no errors'); await b.close(); })();

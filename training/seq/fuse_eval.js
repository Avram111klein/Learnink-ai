// End-to-end check of reader/hw_fuse.js against reader/hw_exp.js with readAnswer, called exactly as the trainer page does
// ({t:"pen",p:[x,y,0.5,…]}, atype "alg", vars from the prompt) plus the timed points in .tp.
// Two pages: BASE = hw_exp.js; FUSE = seq.js + hw_fuse.js (run with FUSE off and on).
// Also proves segmentation is untouched: groups (as sets of input-stroke indices) after segment() and the fraction
// split are compared between BASE and FUSE-on, for every item.
// node fuse_eval.js <repo> <seq_model.json> <items.json> <out.json> [paramsJSON]
const fs=require('fs'), {chromium}=require('playwright');
const [repo,seqP,itemsP,outP,parP]=process.argv.slice(2); const PAR=parP?JSON.parse(parP):{};
const HWM=fs.readFileSync(repo+'/models/v3_r5.json','utf8');
const target=s=>s.replace(/[\[\]>\s]/g,'');
function tokens(t){ const o=[]; for(let i=0;i<t.length;i++){ const c=t[i]; if(/\s/.test(c)) continue; if(c==='^'){ let j=i+1; while(j<t.length&&/\d/.test(t[j])){ o.push('^'+t[j]); j++; } i=j-1; continue; } o.push(c); } return o; }
function barMask(src){ const m=[]; let d=0; for(let i=0;i<src.length;i++){ const c=src[i]; if(c==='['){d++;continue;} if(c===']'){d--;continue;} if(c==='>'||/\s/.test(c)) continue;
  if(c==='^'){ let j=i+1; while(j<src.length&&/\d/.test(src[j])){ m.push(false); j++; } i=j-1; continue; } m.push(c==='/'&&d>0); } return m; }
const X=JSON.parse(fs.readFileSync(itemsP)).map(it=>{ const T=tokens(target(it.promptText)), m=barMask(it.promptText);
  return {id:it.id, vars:[...new Set(T.filter((t,i)=>/^[a-z]$/.test(t)||t==='÷'||(t==='/'&&!m[i])))], S:it.strokes, stored:it.readText}; });
const PAGE=`(${function(X,mode,PAR){ const HW=window.__hw, out=[];
  const sig=(inp,allowed)=>{ const idx=new Map(); inp.forEach((s,k)=>{ const pts=[]; for(let i=0;i<s.p.length;i+=3) pts.push([s.p[i],s.p[i+1]]); const q=JSON.stringify(pts); idx.set(q,(idx.get(q)||[]).concat(k)); });
    let bad=0; const key=g=>g.strokes.map(s=>{ const v=idx.get(JSON.stringify(s.pts)); if(!v){ bad++; return -1; } return v[0]; }).sort((a,b)=>a-b).join('.');
    // FUSE page: the groups formed INSIDE readAnswer (with fusion active); base page: segment()+fractions() called directly
    const Lx=HW.lastLayout?HW.lastLayout():null; let groups,cH,rest,fr;
    if(Lx){ ({groups,cH,rest,fr}=Lx); } else { ({groups,cH}=HW.segment(inp)); ({rest,fr}=HW.fractions(groups,cH,allowed)); }
    return {G:groups.map(key).sort(),rest:rest.map(key).sort(),nfr:fr.length,cH,bad}; };
  if(mode==='on') HW.setFuse(true,PAR); else if(mode==='off'&&HW.setFuse) HW.setFuse(false);
  let ms=0;
  for(const x of X){ const inp=x.S.map(st=>({t:'pen',p:st.flatMap(q=>[q[0],q[1],0.5]),tp:st}));
    const allowed=[...new Set([..."0123456789()+",...x.vars])];
    const a=performance.now(); const r=HW.readAnswer(inp,'alg',x.vars); ms+=performance.now()-a;
    out.push({id:x.id,r,sig:sig(inp,allowed)}); }
  return {out,ms:ms/X.length,info:HW.fuseInfo?HW.fuseInfo():null}; }})`;
(async()=>{ const b=await chromium.launch(); const errs=[]; const res={};
  const seqSrc=fs.readFileSync(repo+'/reader/seq.js','utf8').replace('/*@model:seq*/null',fs.readFileSync(seqP,'utf8'));
  const pages={base:fs.readFileSync(repo+'/reader/hw_exp.js','utf8').replace('/*@model:hw*/null',HWM),
               fuse:seqSrc+'\n'+fs.readFileSync(repo+'/reader/hw_fuse.js','utf8').replace('/*@model:hw*/null',HWM)};
  for(const [name,src,mode] of [['base',pages.base,'base'],['off',pages.fuse,'off'],['on',pages.fuse,'on']]){
    const pg=await b.newPage(); pg.on('pageerror',e=>errs.push(name+': '+e.message));
    await pg.setContent('<!doctype html><meta charset=utf8><body><script>'+src+'</script>');
    res[name]=await pg.evaluate(`${PAGE}(${JSON.stringify(X)},${JSON.stringify(mode)},${JSON.stringify(PAR)})`); await pg.close(); }
  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const offIdentical=res.base.out.every((o,i)=>same(o.r,res.off.out[i].r));
  const segIdentical=res.base.out.every((o,i)=>same(o.sig,res.on.out[i].sig)), unresolved=res.base.out.reduce((n,o)=>n+o.sig.bad,0), nGroups=res.base.out.reduce((n,o)=>n+o.sig.G.length,0);
  const storedMatch=res.base.out.filter((o,i)=>(o.r?o.r.text:'')===(X[i].stored||'')).length;
  const rows=X.map((x,i)=>({id:x.id,base:res.base.out[i].r,fuse:res.on.out[i].r}));
  fs.writeFileSync(outP,JSON.stringify({rows,offIdentical,segIdentical,unresolved,nGroups,storedMatch,n:X.length,ms:{base:res.base.ms,off:res.off.ms,on:res.on.ms},info:res.on.info,errs}));
  console.log(outP,'n',X.length,'FUSE-off identical to base:',offIdentical,'| groups identical (segment+fractions) base vs FUSE-on:',segIdentical,'groups',nGroups,'unresolved strokes',unresolved,
    '| base text == stored readText:',storedMatch+'/'+X.length,'| ms/answer',JSON.stringify(res.ms||{base:res.base.ms,off:res.off.ms,on:res.on.ms}),'| info',JSON.stringify(res.on.info),errs.length?errs.slice(0,3):'no errors');
  await b.close(); })();

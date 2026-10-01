// augmented seq features for TRAIN groups: random affine (rotate/shear/scale), small px jitter, time warp (global pace
// + per-stroke), pressure scale. Recomputed with reader/seq.js features() so train and runtime share code.
// node aug.js <repo> <items.json> <align.json> <K> <seed> <out.json>
const fs=require('fs'); const [repo,itemsP,alP,K,seed,outP]=process.argv.slice(2);
const SEQ=require(repo+'/reader/seq.js'); const CL=JSON.parse(fs.readFileSync(repo+'/models/v3_r5.json')).classes;
let s0=+seed; const rnd=()=>{ s0=(s0*1664525+1013904223)>>>0; return s0/4294967296; }, nrm=()=>Math.sqrt(-2*Math.log(rnd()+1e-12))*Math.cos(6.2832*rnd());
const I=Object.fromEntries(JSON.parse(fs.readFileSync(itemsP)).map(x=>[x.id,x]));
const rows=[];
for(const a of JSON.parse(fs.readFileSync(alP)).filter(a=>a.ok)){ const x=I[a.id], S=x.strokes.filter(s=>s.length), ctx=SEQ.context(S);
  a.per.forEach((p,i)=>{ if(a.bar[i]) return; const t=p.t.replace('^',''); if(!CL.includes(t)) return; const g=S.slice(p.j,p.k);
    for(let k=0;k<+K;k++){ const th=nrm()*0.08, sh=nrm()*0.12, sx=Math.exp(nrm()*0.1), sy=Math.exp(nrm()*0.1), pace=Math.exp(nrm()*0.25), pr=Math.exp(nrm()*0.1);
      const c=Math.cos(th), s=Math.sin(th); let tOff=0;
      const G=g.map((st,si)=>{ const ps=pace*Math.exp(nrm()*0.1); if(si>0) tOff+=nrm()*30; const t0=st[0][2];
        return st.map(q=>{ const X=q[0]*sx+q[1]*sh, Y=q[1]*sy; return [X*c-Y*s+nrm()*0.4, X*s+Y*c+nrm()*0.4, Math.max(0,(t0+tOff)*pace+(q[2]-t0)*ps), q[3]*pr]; }); });
      // keep stroke start times ordered
      for(let si=1;si<G.length;si++){ const e=G[si-1][G[si-1].length-1][2], d=G[si][0][2]-e; if(d<0) G[si].forEach(q=>q[2]-=d); }
      rows.push({id:a.id,sess:x.sessionId,t,f:Array.from(SEQ.features(G,ctx))}); } }); }
fs.writeFileSync(outP,JSON.stringify({rows})); console.log(outP,rows.length);

// Sequence features (reader/seq.js) for a list of groups {id, idx:[stroke indices], t, sess}: the group's timed strokes in
// writing-time order, pace context from the whole answer. K>0 adds K augmented copies per group (affine, jitter, time warp,
// pressure scale; context stays the original answer's).  node featurize.js <repo> <items.json> <groups.json> <K> <seed> <out.json>
const fs=require('fs'); const [repo,itemsP,gP,K,seed,outP]=process.argv.slice(2); const SEQ=require(repo+'/reader/seq.js');
let s0=+seed; const rnd=()=>{ s0=(s0*1664525+1013904223)>>>0; return s0/4294967296; }, nrm=()=>Math.sqrt(-2*Math.log(rnd()+1e-12))*Math.cos(6.2832*rnd());
const I=Object.fromEntries(JSON.parse(fs.readFileSync(itemsP)).map(x=>[x.id,x]));
function aug(g){ const th=nrm()*0.08, sh=nrm()*0.12, sx=Math.exp(nrm()*0.1), sy=Math.exp(nrm()*0.1), pace=Math.exp(nrm()*0.25), pr=Math.exp(nrm()*0.1);
  const c=Math.cos(th), s=Math.sin(th); let tOff=0;
  const G=g.map((st,si)=>{ const ps=pace*Math.exp(nrm()*0.1); if(si>0) tOff+=nrm()*30; const t0=st[0][2];
    return st.map(q=>{ const X=q[0]*sx+q[1]*sh, Y=q[1]*sy; return [X*c-Y*s+nrm()*0.4, X*s+Y*c+nrm()*0.4, Math.max(0,(t0+tOff)*pace+(q[2]-t0)*ps), q[3]*pr]; }); });
  for(let si=1;si<G.length;si++){ const e=G[si-1][G[si-1].length-1][2], d=G[si][0][2]-e; if(d<0) G[si].forEach(q=>q[2]-=d); }
  return G; }
const out=[];
for(const r of JSON.parse(fs.readFileSync(gP)).rows){ const S=I[r.id].strokes, ctx=SEQ.context(S), g=r.idx.map(k=>S[k]).sort((a,b)=>a[0][2]-b[0][2]);
  out.push({id:r.id,sess:r.sess,t:r.t,aug:0,f:Array.from(SEQ.features(g,ctx))});
  for(let k=0;k<+K;k++) out.push({id:r.id,sess:r.sess,t:r.t,aug:1,f:Array.from(SEQ.features(aug(g),ctx))}); }
fs.writeFileSync(outP,JSON.stringify({rows:out})); console.log(outP,out.length);

/* ---------- stroke-sequence symbol reader (shadow engine, stage 2) ----------
   A second classifier for ONE character group, next to the pixel engine (hw_exp.js). It reads HOW the symbol was
   written: direction, stroke order, speed relative to the writer's own pace, pressure, pen lifts. It does not
   segment; it gets the strokes of a group that hw_exp.js segment() (or an alignment) already formed.
   Same 28 classes and the same result shape as HW.classify(): {ch,p,alt,ap} plus `probs` (one per class).
   Points must carry time: [x,y,tMs,pressure*100] (the trainer format). Points without time ([x,y] or [x,y,p])
   → classify() returns null and the caller simply keeps the pixel result.
   Runs offline in the browser and in node (features() is pure JS, so training uses the exact same code).        */
const SEQ=(()=>{
  const M=/*@model:seq*/null;
  const N=32;                                   // points per group after arc-length resampling
  const timed=st=>Array.isArray(st)&&st.length>0&&st.every(q=>Array.isArray(q)&&q.length>=4&&Number.isFinite(q[2]));
  const med=a=>{ if(!a.length) return 0; const s=Float64Array.from(a).sort(); return s[s.length>>1]; };
  // per-point speed (px/ms) from REAL time differences; equal t values are widened to the nearest distinct neighbours
  function speeds(st){ const n=st.length, v=new Float64Array(n); if(n<2) return v;
    for(let i=0;i<n;i++){ let a=Math.max(0,i-1), b=Math.min(n-1,i+1);
      while(st[b][2]-st[a][2]<=0&&(a>0||b<n-1)){ if(a>0) a--; if(b<n-1) b++; }
      const dt=st[b][2]-st[a][2]; let d=0; for(let k=a;k<b;k++) d+=Math.hypot(st[k+1][0]-st[k][0],st[k+1][1]-st[k][1]);
      v[i]=dt>0?d/dt:0; } return v; }
  // pressure per point; the LAST point of every stroke is ignored (old trainer stored 50 there) → copy its neighbour
  const press=st=>st.map((q,i)=>i===st.length-1&&i>0?st[i-1][3]:q[3]);
  /* writer's pace in this exercise: median pen speed, median pressure, median gap between strokes */
  function context(strokes){ const S=(strokes||[]).filter(s=>s&&s.length); if(!S.length||!S.every(timed)) return null;
    const V=[], P=[], G=[];
    S.forEach((st,k)=>{ const v=speeds(st); for(let i=0;i<v.length;i++) if(v[i]>0) V.push(v[i]); press(st).forEach(p=>P.push(p));
      if(k>0) G.push(Math.max(0,st[0][2]-S[k-1][S[k-1].length-1][2])); });
    return {v:med(V)||0.3, p:med(P)||0, g:med(G)||300}; }
  /* fixed-length feature vector of one group (strokes in writing order). F per point × N + globals */
  const F=10, NG=8, DIM=N*F+NG;
  function features(group,ctx){
    const S=(group||[]).filter(s=>s&&s.length); if(!S.length||!S.every(timed)) return null;
    ctx=ctx||context(S);
    let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9; for(const st of S) for(const [x,y] of st){ if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; }
    const w=x1-x0, h=y1-y0, sc=Math.max(w,h,1), cx=(x0+x1)/2, cy=(y0+y1)/2;
    // arc length per stroke; every stroke gets ≥2 resampled points, the rest by length
    const len=S.map(st=>{ let d=0; for(let i=1;i<st.length;i++) d+=Math.hypot(st[i][0]-st[i-1][0],st[i][1]-st[i-1][1]); return d; });
    const tot=len.reduce((a,b)=>a+b,0)||1, ns=S.length, base=Math.min(2,Math.floor(N/ns));
    let cnt=len.map(l=>base+Math.floor((N-base*ns)*l/tot)); let left=N-cnt.reduce((a,b)=>a+b,0);
    const ord=len.map((l,i)=>i).sort((a,b)=>len[b]-len[a]); for(let q=0;left>0;q++,left--) cnt[ord[q%ns]]++;
    const out=new Float32Array(DIM); let o=0, prevEnd=null; const lv=Math.log(ctx.v||0.3);
    S.forEach((st,k)=>{ const n=cnt[k]; if(!n){ prevEnd=st[st.length-1][2]; return; }
      const v=speeds(st), p=press(st), cum=[0]; for(let i=1;i<st.length;i++) cum.push(cum[i-1]+Math.hypot(st[i][0]-st[i-1][0],st[i][1]-st[i-1][1]));
      const L=cum[cum.length-1], R=[]; let j=0;
      for(let r=0;r<n;r++){ const s=n>1?L*r/(n-1):0; while(j<st.length-2&&cum[j+1]<s) j++;
        const seg=cum[j+1]!==undefined?cum[j+1]-cum[j]:0, u=seg>0?Math.min(1,Math.max(0,(s-cum[j])/seg)):0, b=Math.min(j+1,st.length-1);
        R.push([st[j][0]+(st[b][0]-st[j][0])*u, st[j][1]+(st[b][1]-st[j][1])*u, v[j]+(v[b]-v[j])*u, p[j]+(p[b]-p[j])*u]); }
      const gap=prevEnd==null?0:Math.log1p(Math.max(0,st[0][2]-prevEnd)/(ctx.g||300)); prevEnd=st[st.length-1][2];
      for(let r=0;r<n;r++){ const a=R[Math.max(0,r-1)], b=R[Math.min(n-1,r+1)], dx=b[0]-a[0], dy=b[1]-a[1], dl=Math.hypot(dx,dy);
        // turning: angle between incoming and outgoing direction (0 at the ends)
        let ts=0, tc=1; if(r>0&&r<n-1){ const ax=R[r][0]-a[0], ay=R[r][1]-a[1], bx=b[0]-R[r][0], by=b[1]-R[r][1], la=Math.hypot(ax,ay), lb=Math.hypot(bx,by);
          if(la>1e-6&&lb>1e-6){ ts=(ax*by-ay*bx)/(la*lb); tc=(ax*bx+ay*by)/(la*lb); } }
        out[o++]=(R[r][0]-cx)/sc*2; out[o++]=(R[r][1]-cy)/sc*2;
        out[o++]=dl>1e-6?dx/dl:0; out[o++]=dl>1e-6?dy/dl:0; out[o++]=ts; out[o++]=tc;
        out[o++]=R[r][2]>0?Math.max(-3,Math.min(3,Math.log(R[r][2])-lv)):-3;          // speed relative to this writer's pace
        out[o++]=ctx.p>0?Math.max(-2,Math.min(2,R[r][3]/ctx.p-1)):0;                     // pressure relative to this writer
        out[o++]=r===0?1:0; out[o++]=r===0?gap:0; } });                                   // pen-down flag, pause before it
    const dur=S[S.length-1][S[S.length-1].length-1][2]-S[0][0][2];
    out[o++]=Math.min(ns,4)/4; out[o++]=ns===1?1:0; out[o++]=ns===2?1:0; out[o++]=ns>=3?1:0;
    out[o++]=Math.log((w+1)/(h+1)); out[o++]=Math.log1p(Math.max(0,dur)/(ctx.g||300));
    out[o++]=Math.log(tot/sc+1e-3); out[o++]=Math.log1p(len.filter(l=>l<0.1*sc).length);   // ink length / size, tiny strokes
    return out; }
  /* model: int8 MLP {classes, mean, std, layers:[{in,out,s,w(base64 int8 row-major out×in),b}]} */
  let L=null, C=null, MU=null, SD=null;
  function load(m){ if(!m){ L=null; return false; } C=m.classes; MU=Float32Array.from(m.mean); SD=Float32Array.from(m.std);
    L=m.layers.map(l=>{ const bin=atob(l.w), W=new Float32Array(bin.length); for(let i=0;i<bin.length;i++){ let v=bin.charCodeAt(i); if(v>127) v-=256; W[i]=v*l.s; } return {W,b:Float32Array.from(l.b),n:l.in,o:l.out}; });
    return true; }
  if(M) load(M);
  function forward(x){ let h=new Float32Array(x.length); for(let i=0;i<x.length;i++) h[i]=(x[i]-MU[i])/SD[i];
    L.forEach((l,li)=>{ const o=new Float32Array(l.o); for(let j=0;j<l.o;j++){ let s=l.b[j]; const off=j*l.n; for(let i=0;i<l.n;i++) s+=l.W[off+i]*h[i]; o[j]=li<L.length-1?(s>0?s:0):s; } h=o; });
    let mx=-1e9; for(const v of h) mx=Math.max(mx,v); let z=0; const p=Array.from(h,v=>{ const e=Math.exp(v-mx); z+=e; return e; }); return p.map(v=>v/z); }
  /* same result shape as HW.classify(strokes,allowed); null when there is no time or no model */
  function classify(group,ctx,allowed){ if(!L) return null; const f=features(group,ctx); if(!f) return null;
    const p=forward(f); let best=-1,bp=0,alt=-1,ap=0,tot=0;
    C.forEach((c,i)=>{ if(allowed&&!allowed.includes(c)) return; tot+=p[i]; if(p[i]>bp){ alt=best; ap=bp; bp=p[i]; best=i; } else if(p[i]>ap){ ap=p[i]; alt=i; } });
    if(best<0) return null;
    return {ch:C[best], p:tot>0?bp/tot:0, alt:alt>=0?C[alt]:null, ap:tot>0?ap/tot:0, probs:p}; }
  return {classify,features,context,load,timed,N,DIM,classes:()=>C};
})();
if(typeof window!=="undefined") window.__seq=SEQ;
if(typeof module!=="undefined"&&module.exports) module.exports=SEQ;

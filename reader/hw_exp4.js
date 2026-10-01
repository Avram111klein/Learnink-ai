/* READER v4 – hw_exp.js + V4a (÷ assembly) V4b (first-pass guard for "(" ")" and superscripts) V4c (classifier-scored merge) V4d (baseline-relative powers). */
/* EXPERIMENT ONLY – copy of src/js/hw.js with reader-rule changes (tagged EXP1..EXP4). Not used by the app. */
/* ---------- on-device handwriting recognition (no internet) ---------- */
const HW=(()=>{
  const M=/*@model:hw*/null;
  const L=M.layers.map(l=>{ const bin=atob(l.w), W=new Float32Array(bin.length); for(let i=0;i<bin.length;i++){ let v=bin.charCodeAt(i); if(v>127) v-=256; W[i]=v*l.s; } return {W,b:Float32Array.from(l.b),n:l.in,o:l.out}; });
  const C=M.classes, LETS=[..."abcdkmnpty"], CX=C.concat(LETS.filter(l=>!C.includes(l))), cv=document.createElement("canvas"); cv.width=cv.height=112; const cx=cv.getContext("2d",{willReadFrequently:true});
  function pre(strokes){
    let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9; for(const st of strokes) for(const [x,y] of st){ if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; }
    const w=Math.max(x1-x0,1e-3), h=Math.max(y1-y0,1e-3), s=20/Math.max(w,h), ox=(28-w*s)/2, oy=(28-h*s)/2, F=4, lw=2.4;
    cx.setTransform(1,0,0,1,0,0); cx.fillStyle="#000"; cx.fillRect(0,0,112,112); cx.strokeStyle="#fff"; cx.fillStyle="#fff"; cx.lineWidth=lw*F; cx.lineCap="round"; cx.lineJoin="round";
    for(const st of strokes){ const P=st.map(([x,y])=>[((x-x0)*s+ox)*F,((y-y0)*s+oy)*F]);
      if(P.length===1){ cx.beginPath(); cx.arc(P[0][0],P[0][1],lw*F/2,0,7); cx.fill(); }
      else { cx.beginPath(); cx.moveTo(P[0][0],P[0][1]); for(let i=1;i<P.length;i++) cx.lineTo(P[i][0],P[i][1]); cx.stroke(); } }
    const d=cx.getImageData(0,0,112,112).data, a=new Float32Array(784); let m=0,mx=0,my=0;
    for(let y=0;y<28;y++) for(let x=0;x<28;x++){ let t=0; for(let dy=0;dy<4;dy++){ const row=((y*4+dy)*112+x*4)*4; t+=d[row]+d[row+4]+d[row+8]+d[row+12]; } const v=t/(16*255); a[y*28+x]=v; m+=v; mx+=v*x; my+=v*y; }
    if(m<=0) return a; const sx=Math.round(14-mx/m), sy=Math.round(14-my/m), b=new Float32Array(784);
    for(let y=0;y<28;y++){ const ty=y+sy; if(ty<0||ty>27) continue; for(let x=0;x<28;x++){ const tx=x+sx; if(tx<0||tx>27) continue; b[ty*28+tx]=a[y*28+x]; } }
    return b; }
  function forward(a){ let h=a; L.forEach((l,li)=>{ const o=new Float32Array(l.o); for(let j=0;j<l.o;j++){ let s=l.b[j]; const off=j*l.n; for(let i=0;i<l.n;i++) s+=l.W[off+i]*h[i]; o[j]=li<L.length-1?(s>0?s:0):s; } h=o; });
    let mx=-1e9; for(const v of h) mx=Math.max(mx,v); let z=0; const p=h.map(v=>{ const e=Math.exp(v-mx); z+=e; return e; }); return p.map(v=>v/z); }
  let PERS=null, PSET=new Set();
  function setPersonal(cal){ PERS=[]; PSET=new Set(); for(const [ch,list] of Object.entries(cal||{})){ const ci=CX.indexOf(ch); if(ci>=0&&list.length) PSET.add(ci); if(ci<0) continue; for(const strokes of list){ const v=pre(strokes); let n=0; for(const x of v) n+=x*x; PERS.push({ci,v,n:Math.sqrt(n)||1}); } } if(!PERS.length) PERS=null; }
  function blend(p,v){ let nv=0; for(const x of v) nv+=x*x; nv=Math.sqrt(nv)||1; const sims=[];
    for(const s of PERS){ let d=0; for(let i=0;i<784;i++) d+=v[i]*s.v[i]; sims.push([d/(nv*s.n),s.ci]); }
    sims.sort((a,b)=>b[0]-a[0]); const top=sims.slice(0,5).filter(x=>x[0]>0.72); if(!top.length) return p;
    const vote=new Float32Array(p.length); let tot=0; for(const [s,ci] of top){ const w=(s-0.7)*(s-0.7); vote[ci]+=w; tot+=w; }
    const conf=Math.min(0.6,(top[0][0]-0.72)*2.2); return p.map((x,i)=>x*(1-conf)+conf*vote[i]/tot); }
  function classify(strokes,allowed){ const v0=pre(strokes); let p=Array.from(forward(v0)); while(p.length<CX.length) p.push(0); if(PERS) p=Array.from(blend(p,v0)); let best=-1,bp=0,tot=0;
    let alt=-1,ap=0;
    // a letter the student has not calibrated: the network's "x" (a letter-like shape) stands in for it
    if(allowed){ const xi=C.indexOf("x"); for(const l of allowed){ const li=CX.indexOf(l); if(li>=C.length&&!PSET.has(li)&&xi>=0) p[li]=Math.max(p[li],p[xi]*0.9); } }
    // a calibrated letter the item expects: the network never saw letters, so the student's own samples decide
    if(allowed&&PERS&&allowed.some(l=>{ const li=CX.indexOf(l); return li>=C.length&&PSET.has(li); })){ let nv=0; for(const x of v0) nv+=x*x; nv=Math.sqrt(nv)||1; let bL=0, bLi=-1, bD=0;
      for(const q of PERS){ let d=0; for(let i=0;i<784;i++) d+=v0[i]*q.v[i]; d/=nv*q.n; if(q.ci>=C.length){ if(allowed.includes(CX[q.ci])&&d>bL){ bL=d; bLi=q.ci; } } else if(d>bD) bD=d; }
      if(bLi>=0&&bL>=0.74&&bL>=bD+0.02){ const w=Math.min(0.95,0.6+(bL-0.74)*2.5); p=p.map((x,i)=>i===bLi?w:x*(1-w)); }
      if(window.__dbgL) window.__dbgL.push([bL.toFixed(3),bD.toFixed(3)]); }
    CX.forEach((c,i)=>{ if(allowed&&!allowed.includes(c)) return; tot+=p[i]; if(p[i]>bp){ alt=best; ap=bp; bp=p[i]; best=i; } else if(p[i]>ap){ ap=p[i]; alt=i; } });
    return {ch:CX[best], p:tot>0?bp/tot:0, alt:alt>=0?CX[alt]:null, ap:tot>0?ap/tot:0}; }

  /* ---- layout: strokes → character groups → rows, fractions, exponents ---- */
  const box=pts=>{ let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9; for(const [x,y] of pts){ if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; } return {x0,y0,x1,y1,w:x1-x0,h:y1-y0,cx:(x0+x1)/2,cy:(y0+y1)/2}; };
  const toS=s=>{ const pts=[]; for(let i=0;i<s.p.length;i+=3) pts.push([s.p[i],s.p[i+1]]); return Object.assign({pts},box(pts)); };
  // a dash: level, or a straight stroke tilted up to ~32° (quick "=" and "-" are often slanted)
  const isFlat=(s,cH)=>{ if(s.w<=0.28*cH) return false; if(s.h<0.34*Math.max(s.w,1)) return true; if(s._dash!==undefined) return s._dash;
    const P=s.pts, a=P[0], b=P[P.length-1], dx=b[0]-a[0], dy=b[1]-a[1], L=Math.hypot(dx,dy); let dev=0;
    if(L>=0.85*s.w&&Math.abs(dy)<0.62*Math.abs(dx)) for(const [x,y] of P) dev=Math.max(dev,Math.abs((x-a[0])*dy-(y-a[1])*dx)/L); else dev=1e9;
    return s._dash=(dev<=Math.max(1.2,0.16*L)&&s.h<0.62*s.w); };
  const isTiny=(s,cH)=>Math.max(s.w,s.h)<0.3*cH;
  const isDashy=(s,cH)=>isFlat(s,cH)||(s.h<0.5*s.w&&s.w>0.15*cH);   // a short second line of a quick "="
  function regroup(g){ const b=box(g.strokes.flatMap(s=>[[s.x0,s.y0],[s.x1,s.y1]])); Object.assign(g,b); return g; }
  // ---- reader v4 parameters (defaults = tuned on synthetic dev + owner TRAIN only) ----
  const P8=()=>(typeof window!=="undefined"&&window.__P8)||P8D;
  const P8D={arrow:1, div:1, g:1, ph:0.6, pw:0.55, pb:0.05, ovP:0.8, touch:0.04, supH:0.75, m:1, conf:0.8, mTouch:0.06, sup:1, pw2:0, off:0.3, cut:0.65};
  let SEGALLOW=null;   // symbols the current answer may contain (set by readAnswer)
  function bowOf(s){ const P=s.pts, a=P[0], b=P[P.length-1], L=Math.hypot(b[0]-a[0],b[1]-a[1])||1; let m=0; for(const [x,y] of P) m=Math.max(m,Math.abs((x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0]))/L); return m/Math.max(1,s.h); }
  function parenLike(s,cH){ const q=P8(); return s.pts.length>=4&&s.h>=q.ph*cH&&s.w<=q.pw*s.h&&bowOf(s)>=q.pb; }
  function sDist(a,b){ let m=1e18; for(const [x,y] of a.pts) for(const [u,v] of b.pts){ const d=(x-u)*(x-u)+(y-v)*(y-v); if(d<m) m=d; } return Math.sqrt(m); }
  function gDist(g,s){ let m=1e18; for(const q of g.strokes) m=Math.min(m,sDist(q,s)); return m; }
  // V4b: is s a superscript-like neighbour of group g (small, sitting high), or the other way round?
  function supPair(A,B,q){ const lo=A.h<B.h?A:B, hi=lo===A?B:A; return lo.h<q.supH*hi.h&&lo.y1<hi.cy&&lo.x1>hi.x1-0.05*hi.w; }
  function segment(strokes){
    const S=strokes.map(toS).sort((a,b)=>a.x0-b.x0);
    const hs=S.filter(s=>s.h>=s.w*0.5).map(s=>s.h).sort((a,b)=>a-b); let cH=hs.length?hs[Math.floor(hs.length*0.75)]:Math.max(20,...S.map(s=>s.w*0.6)); cH=Math.max(8,cH);
    const groups=[];
    for(const s of S){ const flat=isFlat(s,cH), tiny=isTiny(s,cH), longBar=flat&&s.w>1.5*cH; let hit=null;
      if(!longBar) for(const g of groups){ if(g.longBar) continue;
        const pad=cH*0.12, ov=Math.min(g.x1,s.x1)+pad-(Math.max(g.x0,s.x0)-pad), minw=Math.min(g.w,s.w)+2*pad;
        const vgap=Math.max(0,Math.max(g.y0,s.y0)-Math.min(g.y1,s.y1)), gFlat=g.strokes.every(q=>isFlat(q,cH)), gTiny=g.strokes.every(q=>isTiny(q,cH));
        const eqPair=g.strokes.length===1&&(flat&&isDashy(g.strokes[0],cH)||gFlat&&isDashy(s,cH));
        const lim = (flat&&gFlat||eqPair)?0.65*cH : (tiny&&gTiny)?0.8*cH : (flat||gFlat)?0.12*cH : 0.45*cH;
        let okG=ov/minw>=0.5&&vgap<=lim;
        if(okG&&P8().g&&!flat&&!tiny&&!gFlat&&!gTiny){ const q=P8(), pp=parenLike(s,cH)||g.strokes.length===1&&parenLike(g.strokes[0],cH), sp=supPair(g,s,q);   // V4b
          if((pp||sp)&&ov/(Math.max(g.w,s.w)+2*pad)<q.ovP&&gDist(g,s)>q.touch*cH) okG=false; }
        if(okG){ hit=g; break; } }
      if(hit){ hit.strokes.push(s); regroup(hit); } else groups.push(regroup({strokes:[s],longBar})); }
    const gh=groups.filter(g=>!g.strokes.every(q=>isFlat(q,cH)||isTiny(q,cH))).map(g=>g.h).sort((a,b)=>a-b);
    if(gh.length) cH=Math.max(8,gh[Math.floor(gh.length*0.6)]);
    mergePass(groups,cH);
    if(P8().div&&(!SEGALLOW||SEGALLOW.includes("÷"))) divPass(groups,cH);
    if(P8().arrow&&SEGALLOW) arrowPass(groups,cH);
    return {groups,cH}; }
  // V4e: a copied grey arrow "⟶" (a level shaft with a ">" head at its right end) is not part of the answer
  function chevron(g){ if(g.strokes.length>2) return false; const P=g.strokes.flatMap(s=>s.pts); let ri=0; P.forEach((p,i)=>{ if(p[0]>P[ri][0]) ri=i; });
    const R=P[ri], ends=g.strokes.length===1?[P[0],P[P.length-1]]:g.strokes.map(s=>s.pts[0][0]<s.pts[s.pts.length-1][0]?s.pts[0]:s.pts[s.pts.length-1]);
    return ends.every(e=>e[0]<R[0]-0.5*g.w)&&R[1]>g.y0+0.2*g.h&&R[1]<g.y1-0.2*g.h&&g.w<1.5*g.h+2&&g.h<1.5*g.w+2; }
  function arrowPass(groups,cH){
    const isShaft=q=>isFlat(q,cH)&&q.w>=0.8*cH;
    const headOf=(f,parts)=>{ if(!parts.length||parts.length>2) return false; const h=regroup({strokes:parts});
      return h.x0>f.x0+0.4*f.w&&h.x0<f.x1+0.15*cH&&h.y0<f.cy&&h.y1>f.cy&&h.h<1.2*cH&&chevron(h); };
    for(const g of groups.slice()){ if(!groups.includes(g)) continue;
      const f=g.strokes.find(isShaft); if(!f) continue;
      if(g.strokes.length>1&&headOf(f,g.strokes.filter(q=>q!==f))){ groups.splice(groups.indexOf(g),1); continue; }
      if(g.strokes.length===1){ const h=groups.find(o=>o!==g&&headOf(f,o.strokes)); if(h){ groups.splice(groups.indexOf(h),1); groups.splice(groups.indexOf(g),1); } } } }
  // V4a: "÷" = a level dash (possibly drawn twice) with dots above/below it, or – this writer's habit – just two
  // dots stacked vertically (a ":" written where a "÷" is expected). Only when the answer may contain "÷".
  function divPass(groups,cH){
    const tinyG=g=>g.strokes.every(q=>isTiny(q,cH));
    const barG=g=>!g.longBar&&g.strokes.length<=2&&g.strokes.every(q=>isFlat(q,cH))&&(g.strokes.length===1||Math.abs(g.strokes[0].cy-g.strokes[1].cy)<0.25*cH)&&g.w<1.3*cH;
    for(const f of groups.filter(barG)){
      if(!groups.includes(f)) continue;
      const dots=groups.filter(g=>g!==f&&tinyG(g)&&g.cx>f.x0-0.3*f.w&&g.cx<f.x1+0.3*f.w&&Math.abs(g.cy-f.cy)<0.9*cH);
      const nd=dots.reduce((a,g)=>a+g.strokes.length,0); if(nd<1||nd>3) continue;
      for(const g of dots){ f.strokes.push(...g.strokes); groups.splice(groups.indexOf(g),1); }
      regroup(f); f.ch="÷"; f.p=nd>=2?0.9:0.7; }
    for(const g of groups){ if(g.ch||!tinyG(g)||g.strokes.length!==2) continue;
      const [a,b]=g.strokes; if(Math.abs(a.cx-b.cx)<0.35*cH&&Math.abs(a.cy-b.cy)>0.25*cH){ g.ch="÷"; g.p=0.75; } } }
  function minDist(A,B){ let m=1e18; for(const s of A.strokes) for(const q of B.strokes){ if(s.x0>q.x1+40||q.x0>s.x1+40) continue;
      for(let i=0;i<s.pts.length;i++){ const [x,y]=s.pts[i]; for(let j=0;j<q.pts.length;j++){ const dx=x-q.pts[j][0], dy=y-q.pts[j][1], d=dx*dx+dy*dy; if(d<m) m=d; } } } return Math.sqrt(m); }
  function cls(g){ if(!g._c) g._c=classify(g.strokes.map(s=>s.pts)); return g._c; }
  // join pieces of one character that were drawn as separate strokes (the bar of a 7, the arm of a 4…)
  function mergePass(groups,cH){
    for(let guard=0;guard<60;guard++){ let did=false; groups.sort((a,b)=>a.x0-b.x0);
      outer: for(let i=0;i<groups.length-1;i++){ const A=groups[i];
        for(let j=i+1;j<groups.length&&groups[j].x0<A.x1+0.4*cH;j++){ const B=groups[j]; if(A.longBar||B.longBar) continue;
          if(Math.min(A.y1,B.y1)-Math.max(A.y0,B.y0)< -0.15*cH) continue;
          const W=Math.max(A.x1,B.x1)-Math.min(A.x0,B.x0), Hh=Math.max(A.y1,B.y1)-Math.min(A.y0,B.y0);
          if(W>1.15*Math.max(cH,Hh)) continue;
          const bothTiny=A.strokes.every(q=>isTiny(q,cH))&&B.strokes.every(q=>isTiny(q,cH)); if(bothTiny) continue;
          const dAB=minDist(A,B); if(dAB>0.22*cH) continue;
          const pa=cls(A), pb=cls(B), pm=classify([...A.strokes,...B.strokes].map(s=>s.pts));
          if(P8().m){ const q=P8();   // V4c: two parts that each read confidently as a symbol and do not touch stay apart
            if(pa.p>=q.conf&&pb.p>=q.conf&&dAB>q.mTouch*cH&&!(pm.p>=0.98&&pm.ch!==pa.ch&&pm.ch!==pb.ch)) continue;
            if(q.sup&&supPair(A,B,q)&&/\d/.test((A.h<B.h?pa:pb).ch)&&(A.h<B.h?pa:pb).p>=0.6&&dAB>q.mTouch*cH) continue;
            const pA=A.strokes.length===1&&parenLike(A.strokes[0],cH), pB=B.strokes.length===1&&parenLike(B.strokes[0],cH);
            if((pA&&/[()1]/.test(pa.ch)||pB&&/[()1]/.test(pb.ch))&&dAB>q.mTouch*cH) continue; }
          const frag=x=>x.h<0.6*cH||x.strokes.every(s=>isFlat(s,cH));
          const xov=Math.min(A.x1,B.x1)-Math.max(A.x0,B.x0), narrow=Math.max(1,Math.min(A.w,B.w));
          if(pm.p>=0.55&&((frag(A)||frag(B))&&pm.p>=0.6*Math.min(pa.p,pb.p)||xov>=0.3*narrow&&pm.p>Math.max(pa.p,pb.p))){ A.strokes.push(...B.strokes); regroup(A); delete A._c; delete A.ch; groups.splice(j,1); did=true; break outer; } } }
      if(!did) break; } }
  // EXP4: confidence of a dash from its straightness and length (1 = long, straight, level)
  function barConf(s,cH){ const P=s.pts, a=P[0], b=P[P.length-1], dx=b[0]-a[0], dy=b[1]-a[1], L=Math.hypot(dx,dy)||1; let dev=0;
    for(const [x,y] of P) dev=Math.max(dev,Math.abs((x-a[0])*dy-(y-a[1])*dx)/L);
    const straight=Math.max(0,1-dev/(0.25*L)), level=Math.max(0,1-Math.abs(dy)/(0.6*Math.abs(dx)+1e-6)), len=Math.min(1,s.w/(0.45*cH));
    return Math.max(0.3,Math.min(0.995,0.5+0.5*Math.min(straight,level,len))); }
  function eqPar(all){ const [a,b]=all, ov=(Math.min(a.x1,b.x1)-Math.max(a.x0,b.x0))/Math.max(1,Math.min(a.w,b.w)); return Math.max(0.5,Math.min(1,0.6+0.5*ov)); }
  function label(g,cH,allowed){
    if(g.ch) return g;
    const all=g.strokes;
    if(all.every(s=>isTiny(s,cH))){ g.ch=all.length>=2?":":"."; g.p=Math.min(0.995,0.6+0.4*(1-Math.max(...all.map(s=>Math.max(s.w,s.h)))/(0.3*cH))); return g; }   // EXP4
    if(all.every(s=>isFlat(s,cH))){ g.ch=all.length>=2?"=":"-"; g.p=Math.min(...all.map(s=>barConf(s,cH)))*(all.length>=2?eqPar(all):1); return g; }   // EXP4
    if(all.length===2&&all.every(s=>isDashy(s,cH))&&all.some(s=>isFlat(s,cH))&&Math.abs(all[0].cy-all[1].cy)>=Math.max(all[0].h,all[1].h)*0.6){ g.ch="="; g.p=0.9*Math.min(...all.map(s=>barConf(s,cH)))*eqPar(all); return g; }
    const r=classify(all.map(s=>s.pts),allowed); g.ch=r.ch; g.p=r.p; g.alt=r.alt; g.ap=r.ap; return g; }
  // pull fractions out: a flat line with writing right above and below it
  function fractions(groups,cH,allowed){
    const out=[]; const used=new Set();
    for(const f of groups){ if(used.has(f)) continue; if(!f.strokes.every(s=>isFlat(s,cH))||f.strokes.length!==1) continue;
      const inX=g=>g!==f&&!used.has(g)&&g.cx>=f.x0-0.25*cH&&g.cx<=f.x1+0.25*cH;
      const up=groups.filter(g=>inX(g)&&g.cy<f.cy&&f.cy-g.y1<0.9*cH&&g.y1<=f.cy+0.1*cH), dn=groups.filter(g=>inX(g)&&g.cy>f.cy&&g.y0-f.cy<0.9*cH&&g.y0>=f.cy-0.1*cH);
      if(!up.length||!dn.length) continue;
      if(up.some(g=>f.cy-g.y0>2.2*cH)) continue;          // a column exercise, not a fraction
      [f,...up,...dn].forEach(g=>used.add(g));
      const n=rowText(up,cH,allowed), d=rowText(dn,cH,allowed);
      out.push({frac:true,n,d,x0:Math.min(f.x0,...up.map(g=>g.x0),...dn.map(g=>g.x0)),x1:Math.max(f.x1,...up.map(g=>g.x1),...dn.map(g=>g.x1)),y0:Math.min(...up.map(g=>g.y0)),y1:Math.max(...dn.map(g=>g.y1)),cx:f.cx,cy:f.cy,h:cH,w:f.w,p:Math.min(n.p,d.p)}); }
    return {rest:groups.filter(g=>!used.has(g)),fr:out}; }
  // EXP3: "+" is a symmetric cross between two operands; "t" has its bar in the upper part, a longer stem, and is not
  // squeezed between operands like an operator. Ambiguous → keep the network's pick but lower its confidence.
  function plusOrT(g,items,cH){ const i=items.indexOf(g), L=items[i-1], R=items[i+1];
    const S=g.strokes; if(S.length<2){ return; }
    const bar=S.reduce((a,b)=>(b.w/(b.h+1))>(a.w/(a.h+1))?b:a), stem=S.find(s=>s!==bar)||bar;
    const barY=(bar.cy-g.y0)/Math.max(1,g.h), sym=Math.abs((stem.cx-bar.x0)/Math.max(1,bar.w)-0.5), tall=stem.h/Math.max(1,bar.w);
    let tS=0; if(barY<0.4) tS++; if(tall>1.35) tS++; if(sym>0.2) tS++; if(!R||!L) tS++;                  // no operand on one side → not an operator
    if(g.h>0.85*cH) tS++; if(L&&R&&g.h<0.75*cH) tS--;
    const want=tS>=3?"t":tS<=1?"+":null;
    if(want&&want!==g.ch){ g.alt=g.ch; g.ap=g.p*0.5; g.ch=want; g.p=Math.min(g.p,0.8); }
    else if(!want){ g.p=Math.min(g.p,0.6); } }
  function rowText(gs,cH,allowed){
    const items=[...gs].sort((a,b)=>a.x0-b.x0); let s="", p=1, prev=null; const vars=[];
    const bots=items.filter(g=>!g.frac&&g.h>0.6*cH&&!g.strokes.every(q=>isFlat(q,cH))).map(g=>g.y1).sort((a,b)=>a-b), baseY=bots.length?bots[Math.floor(bots.length/2)]:Math.max(...items.map(g=>g.y1||0));   // V4d row baseline
    for(const g of items){
      if(g.frac){ if(prev&&/\d$/.test(s)) s+=" "; const st=s.length, t=`${g.n.text}/${g.d.text}`; s+=t; p=Math.min(p,g.p);
        g.n.alts.forEach(a=>vars.push({st,len:t.length,r:`${a}/${g.d.text}`})); g.d.alts.forEach(a=>vars.push({st,len:t.length,r:`${g.n.text}/${a}`})); prev=g; continue; }
      label(g,cH,allowed);
      if((g.ch==="+"||g.ch==="t")&&allowed&&allowed.includes("+")&&allowed.includes("t")) plusOrT(g,items,cH);   // EXP3
      const q8=P8(), oldExp=(prev&&!prev.frac&&/[\da-z)]/.test(prev.ch||"")&&/\d/.test(g.ch)&&g.h<1.1*prev.h&&g.cy<prev.y0+0.25*prev.h&&g.y1<prev.cy+0.1*prev.h&&allowed&&allowed.some(c=>/[a-z]/.test(c)));
      const exp = q8.pw2===2 ? oldExp||(prev&&!prev.frac&&/[\da-z)]/.test(prev.ch||"")&&/\d/.test(g.ch)&&g.h<0.9*prev.h&&(baseY-g.y1)>q8.off*cH&&g.y1<prev.cy+0.15*prev.h&&g.cy<prev.cy&&allowed&&allowed.some(c=>/[a-z]/.test(c))) : q8.pw2 ? (prev&&!prev.frac&&/[\da-z)]/.test(prev.ch||"")&&/\d/.test(g.ch)&&g.h<0.9*prev.h&&(baseY-g.y1)>q8.off*cH&&g.y1<prev.cy+0.15*prev.h&&g.cy<prev.cy&&allowed&&allowed.some(c=>/[a-z]/.test(c)))   // V4d
        : (prev&&!prev.frac&&/[\da-z)]/.test(prev.ch||"")&&/\d/.test(g.ch)&&g.h<1.1*prev.h&&g.cy<prev.y0+0.25*prev.h&&g.y1<prev.cy+0.1*prev.h&&allowed&&allowed.some(c=>/[a-z]/.test(c)));   // EXP2
      if(exp&&!/\^\d*$/.test(s)) s+="^";
      if(g.alt&&g.ap>=0.05) vars.push({st:s.length,len:g.ch.length,r:g.alt});
      s+=g.ch; p=Math.min(p,g.p); if(!exp) prev=g; }
    return {text:s,p,alts:vars.map(v=>s.slice(0,v.st)+v.r+s.slice(v.st+v.len))}; }
  function rowsOf(items,cH){ const rows=[]; for(const g of [...items].sort((a,b)=>a.cy-b.cy)){ const r=rows.find(r=>Math.abs(r.cy-g.cy)<0.7*cH); if(r){ r.items.push(g); r.cy=(r.cy*(r.items.length-1)+g.cy)/r.items.length; } else rows.push({cy:g.cy,items:[g]}); } return rows; }
  const ANS={num:"0123456789",eq:"0123456789x",frac:"0123456789/",factor:"0123456789x",poly:"0123456789x()+",alg:"0123456789()+",algf:"0123456789()+"};
  const EXP=[..."0123456789+x/()"];
  /* an answer written after a printed exercise */
  function readAnswer(strokes,atype,vars){
    if(!strokes.length) return null;
    SEGALLOW=[...new Set([...(ANS[atype]||ANS.num),...(vars||[])])];
    const {groups,cH}=segment(strokes), allowed=[...new Set([...(ANS[atype]||ANS.num),...(vars||[])])], keepMinus=atype==="alg"||atype==="poly";
    const {rest,fr}=fractions(groups,cH,allowed);   // EXP1: numerator/denominator may hold letters, + ( )
    const r=rowText([...rest,...fr],cH,allowed);
    const clean=x=>{ let t=x.replace(/:/g,"").replace(/^=+/,"");
      t=t.replace(/\.(?!\d)/g,"").replace(/(^|[^\d])\./g,"$1");        // stray dots
      t=t.replace(/=+/g,"="); if(!keepMinus) t=t.replace(/(^|=)-/g,"$1\u0000").replace(/-/g,"").replace(/\u0000/g,"-");   // a minus only at the start
      return /[\da-z]/.test(t)?t:""; };
    const t=clean(r.text), alts=[...new Set(r.alts.map(clean))].filter(a=>a&&a!==t);
    // V4f: v4 reads more answers as well-formed, so its answer confidence is re-mapped (calibrated on the synthetic dev
    // set + owner TRAIN only): a raw min-symbol confidence of P8().cut becomes 0.5, the "sure" line.
    const cut=P8().cut||0.5, pc=r.p<cut?r.p*0.5/cut:0.5+(r.p-cut)*0.5/(1-cut);
    return {text:t, p:pc, sure:pc>=0.5, alts}; }
  /* exercises the student wrote by hand: horizontal "23+45=68", column form, equations */
  function readExercises(strokes){
    const out=[]; if(!strokes.length) return out;
    SEGALLOW=null; const {groups,cH}=segment(strokes); let pool=groups.slice();
    // column exercises: a long line with two or more rows above it
    for(const bar of groups.filter(g=>g.longBar)){
      const inX=g=>g!==bar&&g.cx>=bar.x0-0.6*cH&&g.cx<=bar.x1+0.3*cH;
      const up=pool.filter(g=>inX(g)&&g.cy<bar.cy&&bar.cy-g.cy<3.6*cH), dn=pool.filter(g=>inX(g)&&g.cy>bar.cy&&g.cy-bar.cy<1.8*cH);
      const rows=rowsOf(up,cH); if(rows.length<2||!dn.length) continue;
      let op=null, nums=[], p=1;
      for(const r of rows){ const its=r.items.sort((a,b)=>a.x0-b.x0); its.forEach(g=>label(g,cH,EXP)); let digits=its;
        const f=its[0]; if(["+","-","x","/"].includes(f.ch)&&its.length>1){ op=f.ch; digits=its.slice(1); }
        const t=rowText(digits,cH,[..."0123456789"]); nums.push(t.text); p=Math.min(p,t.p); }
      if(!op) op="+"; const ans=rowText(dn,cH,[..."0123456789"]);
      out.push({type:"arith",expr:nums.join(op==="x"?"*":op),answer:ans.text,alts:ans.alts.map(a=>({answer:a})),p:Math.min(p,ans.p),x:Math.max(...dn.map(g=>g.x1)),y:(Math.min(...dn.map(g=>g.y0))+Math.max(...dn.map(g=>g.y1)))/2});
      pool=pool.filter(g=>g!==bar&&!up.includes(g)&&!dn.includes(g)); }
    // horizontal lines
    const {rest,fr}=fractions(pool,cH,[..."0123456789"]);
    const rows=rowsOf([...rest,...fr],cH);
    const parsed=rows.map(r=>{ const t=rowText(r.items,cH,EXP); return {t,items:r.items,cy:r.cy}; });
    for(let i=0;i<parsed.length;i++){ const R=parsed[i], s=R.t.text; if(!s.includes("=")) continue; if(/^x=/.test(s)) continue;
      const k=s.lastIndexOf("="), left=s.slice(0,k), right=s.slice(k+1);
      const ansItems=R.items.filter(g=>g.cx>((R.items.find(q=>q.ch==="="&&!q.frac)||{cx:-1}).cx));
      const next=parsed[i+1];
      if(/x/.test(left.replace(/(\d|\))x(?=\d|\()/g,"$1*"))||(/x/.test(left)&&next&&/^x=/.test(next.t.text))){
        // equation with unknown x; the answer is on the next line ("x=5") or missing
        const a=next&&/^x=/.test(next.t.text)?next.t.text.slice(2):null; const tgt=a!=null?next.items:R.items;
        out.push({type:"eq",expr:s.replace(/x(?=\d)/g,"x*"),answer:a,p:Math.min(R.t.p,a!=null?next.t.p:1),x:Math.max(...tgt.map(g=>g.x1)),y:a!=null?next.cy:R.cy}); if(a!=null) i++; continue; }
      const alts=R.t.alts.filter(a=>a.includes("=")).map(a=>{ const q=a.lastIndexOf("="); return {expr:a.slice(0,q).replace(/x/g,"*"),answer:a.slice(q+1)||null}; });
      out.push({type:"arith",expr:left.replace(/x/g,"*"),answer:right||null,alts,p:R.t.p,x:Math.max(...R.items.map(g=>g.x1)),y:R.cy}); }
    return out; }
  return {readAnswer,readExercises,classify,pre,segment,rowsOf,rowText,fractions,setPersonal,knows:ch=>PSET.has(CX.indexOf(ch))};
})();
window.__hw=HW;


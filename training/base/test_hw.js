const { chromium } = require('playwright'); const fs=require('fs');
(async()=>{const b=await chromium.launch();const p=await b.newPage();
const errs=[];p.on('pageerror',e=>errs.push(e.message));
const helpers=`function evalExpr(){return NaN}`;
fs.writeFileSync('h.html','<!doctype html><meta charset=utf8><body><script>'+fs.readFileSync('hw_built.js','utf8')+';window.HW=HW;</script>');
await p.goto('file://'+process.cwd()+'/h.html');
const D=JSON.parse(fs.readFileSync('digits_strokes.json'));
const res=await p.evaluate((D)=>{
  const mk=(st,dx,dy,sc=1)=>st.map(s=>({t:'pen',p:s.flatMap(([x,y])=>[x*sc+dx,y*sc+dy,.5])}));
  // single digits
  let ok=0; const conf={}; for(const d of D){ const r=HW.readAnswer(mk(d.s,100,100),'num'); if(r&&r.text===String(d.d)) ok++; else { const k=d.d+'->'+(r&&r.text); conf[k]=(conf[k]||0)+1; } }
  // two/three-digit numbers
  let ok2=0,n2=0; for(let i=0;i+2<D.length;i+=3){ const a=D[i],b=D[i+1],c=D[i+2]; const w=x=>Math.max(...x.s.flat(2).filter((v,j)=>j%2==0));
    const s=[...mk(a.s,100,100), ...mk(b.s,100+w(a)+6,100+ (i%7)-3), ...mk(c.s,100+w(a)+w(b)+12,100+(i%5)-2)]; const r=HW.readAnswer(s,'num'); n2++; if(r&&r.text===`${a.d}${b.d}${c.d}`) ok2++; }
  return {single:ok/D.length, triple:ok2/n2, conf:Object.entries(conf).sort((a,b)=>b[1]-a[1]).slice(0,12)};
},D);
console.log(JSON.stringify(res));
const par=JSON.parse(fs.readFileSync('parity.json'));
console.log(errs.join('\n')||'no errors');await b.close();})();

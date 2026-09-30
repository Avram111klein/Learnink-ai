const { chromium } = require('playwright'); const fs=require('fs');
(async()=>{const b=await chromium.launch();const p=await b.newPage();await p.goto('file://'+process.cwd()+'/h.html');
const D=JSON.parse(fs.readFileSync('digits_strokes.json'));
console.log(await p.evaluate((D)=>{ const by={}; D.forEach(d=>(by[d.d]=by[d.d]||[]).push(d.s));
 const mk=(st,dx,dy)=>st.map(s=>({t:'pen',p:s.flatMap(([x,y])=>[x+dx,y+dy,.5])}));
 let ok=0,uns=0,fx=0,N=300; const fails=[]; let seed=5; const rnd=n=>{seed=(seed*16807)%2147483647; return seed%n;};
 for(let t=0;t<N;t++){ const num=String(rnd(990)+10); let x=100, strokes=[]; for(const ch of num){ const s=by[ch][rnd(by[ch].length)]; strokes.push(...mk(s,x,100+rnd(5))); x+=Math.max(...s.flat(2).filter((v,i)=>i%2==0))+4+rnd(6); }
   const r=HW.readAnswer(strokes,'num'); if(r.text===num) ok++; else if(!r.sure||r.alts.includes(num)||r.text.replace(/1/g,"")===num.replace(/1/g,"")) uns++; else { fx++; if(fails.length<25) fails.push(num+"->"+r.text+" p"+r.p.toFixed(2)+" alts"+r.alts.join(",")); } }
 return `correct ${ok/N}, unsure ${uns/N}, false-wrong ${fx/N}\n`+fails.join("\n"); },D)); await b.close();})();

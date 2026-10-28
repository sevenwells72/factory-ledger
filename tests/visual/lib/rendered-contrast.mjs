// Raster-backed contrast supplement for image/gradient/translucent backgrounds.
// The existing CSS-only audit deliberately reports those backgrounds as
// indeterminate. Read foregrounds and text rectangles before hiding only glyph
// paint, then sample the actual browser-composited backing pixels. No app logic
// or layout is changed; screenshot-only styles are removed immediately.
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {PNG}=require('playwright-core/lib/utilsBundle');
export async function renderedContrast(page, {viewportOnly=false}={}) {
  const records=await page.evaluate(({viewportOnly})=>{
    const output=[];
    const swatch=document.createElement('canvas');swatch.width=1;swatch.height=1;
    const paint=swatch.getContext('2d',{willReadFrequently:true});
    function add(el,text,rects,pseudo){
      const closed=el.closest('details:not([open])');
      if(closed&&!closed.querySelector(':scope > summary')?.contains(el))return;
      const cs=getComputedStyle(el,pseudo),normal=getComputedStyle(el);
      if(normal.visibility!=='visible'||normal.display==='none'||normal.clipPath==='inset(50%)')return;
      // WCAG 1.4.3 explicitly exempts inactive user-interface controls.
      if(el.closest(':disabled,[aria-disabled="true"]'))return;
      let opacity=1;
      for(let n=el;n;n=n.parentElement)opacity*=Number(getComputedStyle(n).opacity);
      if(opacity===0)return;
      if(pseudo)opacity*=Number(cs.opacity);
      const fontPx=parseFloat(cs.fontSize),weight=Number(cs.fontWeight)||400;
      const foreground=el instanceof SVGElement?cs.fill:cs.color;
      // Computed color-mix can be color(srgb ...), with channels in 0..1.
      // Let the browser convert every supported CSS color to screenshot sRGB.
      paint.clearRect(0,0,1,1);paint.fillStyle=foreground;paint.fillRect(0,0,1,1);
      const rgba=[...paint.getImageData(0,0,1,1).data];
      const fg=[...rgba.slice(0,3),rgba[3]/255*opacity];
      const boxes=rects.filter(r=>r.width>0&&r.height>0).map(r=>({x:r.x+(viewportOnly?0:scrollX),y:r.y+(viewportOnly?0:scrollY),w:r.width,h:r.height}));
      if(!boxes.length)return;
      const points=[];
      for(const r of boxes)for(const fx of [.15,.35,.5,.65,.85])for(const fy of [.25,.5,.75]){
        const x=r.x+r.w*fx,y=r.y+r.h*fy;
        if(viewportOnly){const hit=document.elementFromPoint(x,y);if(!hit||!(el.contains(hit)||(normal.pointerEvents==='none'&&hit.contains(el))))continue;}
        points.push([x,y]);
      }
      if(!points.length)return;
      output.push({tag:el.tagName,id:el.id,text:text.trim().slice(0,80),fontPx,weight,foreground,fg,target:fontPx>=24||(fontPx>=18.66&&weight>=700)?3:4.5,boxes,points});
    }
    const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
    for(let node;node=walker.nextNode();){
      if(!node.textContent.trim()||/^(SCRIPT|STYLE|OPTION)$/.test(node.parentElement.tagName))continue;
      const range=document.createRange();range.selectNodeContents(node);
      add(node.parentElement,node.textContent,[...range.getClientRects()]);
    }
    for(const el of document.querySelectorAll('input,textarea,select')){
      if(['checkbox','radio','file','hidden'].includes(el.type))continue;
      const placeholder=!el.value&&el.placeholder;
      const text=placeholder||el.selectedOptions?.[0]?.textContent||el.value;
      if(!text||el.type==='password'||!el.getClientRects().length)continue;
      const cs=getComputedStyle(el),r=el.getBoundingClientRect(),pad=parseFloat(cs.paddingLeft)+parseFloat(cs.borderLeftWidth);
      const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');ctx.font=cs.font;
      const height=parseFloat(cs.fontSize)*1.15;
      add(el,text,[{x:r.x+pad,y:r.y+(r.height-height)/2,width:Math.min(ctx.measureText(text).width,r.width-pad*2-4),height}],placeholder?'::placeholder':null);
    }
    return output;
  },{viewportOnly});
  const style=await page.addStyleTag({content:'* { transition: none !important; animation: none !important; } * { -webkit-text-fill-color: transparent !important; text-shadow: none !important; } *::placeholder { -webkit-text-fill-color: transparent !important; color: transparent !important; } svg text, svg tspan { fill: transparent !important; stroke: transparent !important; }'});
  let png;
  try{png=PNG.sync.read(await page.screenshot({fullPage:!viewportOnly,animations:'disabled'}));}finally{
    await style.evaluate(el=>{el.textContent='* { transition: none !important; animation: none !important; }';});
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    await style.evaluate(el=>el.remove());
  }
  const scale=await page.evaluate(()=>devicePixelRatio);
  const lum=rgb=>rgb.slice(0,3).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((sum,v,i)=>sum+v*[.2126,.7152,.0722][i],0);
  let checked=0;const failures=[];let minimum=Infinity;
  for(const rec of records){
    let worst=Infinity,worstBg=null;
    for(const [px,py] of rec.points){
      const x=Math.floor(px*scale),y=Math.floor(py*scale);
      if(x<0||y<0||x>=png.width||y>=png.height)continue;
      const index=(y*png.width+x)*4,bg=[png.data[index],png.data[index+1],png.data[index+2]];
      const fg=rec.fg.slice(0,3).map((v,i)=>v*rec.fg[3]+bg[i]*(1-rec.fg[3]));
      const a=lum(fg),b=lum(bg),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
      if(ratio<worst){worst=ratio;worstBg=bg;}
    }
    if(!Number.isFinite(worst))continue;
    checked++;minimum=Math.min(minimum,worst);
    if(worst+.005<rec.target){const {boxes,points,...small}=rec;failures.push({...small,ratio:Number(worst.toFixed(2)),sampledBackground:worstBg});}
  }
  failures.sort((a,b)=>a.ratio-b.ratio);
  return {method:'Actual composited backing pixels at 15 points per text rectangle; glyph paint hidden without changing layout or backgrounds.',checked,failures:failures.length,minimumRatio:Number(minimum.toFixed(2)),worst:failures};
}

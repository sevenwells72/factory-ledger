import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium,webkit} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {protect,ORIGIN} from './safety.mjs';
import {SCREENS,PAGE_URLS,gaps,setup,cancel} from './screens.mjs';
import {inspect,keyboard} from './checks.mjs';
import {CHECK_SOURCE} from './check-source.mjs';
const ROOT=path.resolve(import.meta.dirname,'../../..');
const OUT=path.join(ROOT,'docs/design/audit/live-evidence');
await fs.mkdir(OUT,{recursive:true});
const filters=Object.fromEntries(process.argv.slice(2).map(x=>x.split('=')));
const selected=SCREENS.filter(s=>!filters.screens||filters.screens.split(',').includes(s.id));
const variants=[['390x844',390,844],['844x390',844,390],['768x1024',768,1024],['1280x800',1280,800]].flatMap(([size,width,height])=>['light','dark'].map(theme=>({key:size+'-'+theme,width,height,theme})));
const engines=Object.entries({chromium,webkit}).filter(([name])=>!filters.engine||filters.engine===name);
async function theme(p,t){
 return p.evaluate(t=>{const e=document.documentElement;const c=()=>getComputedStyle(e).getPropertyValue('--bg-primary')||getComputedStyle(document.body).backgroundColor;e.dataset.theme='dark';const dark=c();e.dataset.theme='light';const light=c();e.dataset.theme=t;return {responded:light!==dark,dark,light};},t);
}
async function worker(name,engine){
 const b=await engine.launch();
 const requests=[],responses=[],dialogs=[];
 const c=await b.newContext({viewport:{width:1280,height:800},timezoneId:'America/New_York',locale:'en-US',serviceWorkers:'block',reducedMotion:'reduce'});
 await protect(c,requests);let p=null,lastPage=null;
 for(const s of selected){
  const outfile=path.join(OUT,`${name}-${s.id}.json`);if(filters.resume==='true'){try{const old=JSON.parse(await fs.readFile(outfile));if(old.harnessRevision===2&&old.records.length===8&&!old.records.some(r=>r.error))continue;}catch{}}
  const requestStart=requests.length,responseStart=responses.length,dialogStart=dialogs.length;
  const changed=lastPage!==s.page;
  if(changed){
    if(p){await p.emulateMedia({media:'screen'});await p.close();}p=await c.newPage();lastPage=s.page;p.setDefaultTimeout(10000);
    p.on('dialog',async d=>{dialogs.push({type:d.type(),message:d.message().slice(0,200)});await d.dismiss();});
    p.on('response',r=>{const u=new URL(r.url());responses.push({host:u.hostname,path:u.pathname,status:r.status()});});
  }
  const result={harnessRevision:2,id:s.id,name:s.name,engine:name,browserVersion:b.version(),url:ORIGIN+PAGE_URLS[s.page],started:new Date().toISOString(),records:[]};
  let gap=gaps[s.id]||null;
  try{
   await p.setViewportSize({width:1280,height:800});await p.emulateMedia({media:'screen'});
   if(changed){await p.goto(result.url,{waitUntil:'domcontentloaded',timeout:30000});await p.waitForTimeout(6000);}
   if(s.page==='index'){
     for(const sel of ['#note-cancel-btn','#er-cancel-btn','#supply-request-cancel-btn','#lot-panel-close','#order-back-btn']){const l=p.locator(sel).first();if(await l.isVisible().catch(()=>false))await l.click();}
     await p.locator('.tab[data-tab=operations]').click();
     await p.fill('#global-search','');
     if(await p.locator('#navLinks').evaluate(e=>e.classList.contains('open')))await p.locator('#navToggle').evaluate(e=>e.click());
   }
   if(!gap)await setup(p,s);
   if(s.region&&!gap && !(await p.locator(s.region).first().isVisible()))throw Error('Requested region is not visible: '+s.region);
  }catch(e){gap=String(e.message).split('\n')[0];}
  // Unavailable states get contextual screenshots, never passing checks for their parent page.
  for(const v of variants){
   if(filters.variant&&!v.key.startsWith(filters.variant))continue;
   const record={variant:v.key,width:v.width,height:v.height,theme:v.theme,gap};
   try{
    await p.setViewportSize({width:v.width,height:v.height});await p.emulateMedia({colorScheme:v.theme,media:s.print?'print':'screen'});record.palette=await theme(p,v.theme);await p.waitForTimeout(420);
    if(s.id==='S-01'&&!gap){const nav=p.locator('#navToggle');if(v.width<=768 && await nav.isVisible() && !(await p.locator('#navLinks').evaluate(e=>e.classList.contains('open'))))await nav.click();}
    const region=gap?null:s.region;
    if(region){await p.locator(s.id==='S-69'?'#graphContainer':region).first().evaluate(e=>e.scrollIntoView({block:'start',inline:'nearest'}));await p.evaluate(()=>{const bottom=Math.max(0,...[...document.querySelectorAll('.site-nav,.app-header,.tab-bar')].map(e=>e.getBoundingClientRect().bottom));window.scrollBy(0,-Math.min(bottom+12,innerHeight*.8));});}
    else await p.evaluate(()=>window.scrollTo(0,0));
    if(s.id==='S-69'&&!gap){try{await p.getByRole('button',{name:'Fit',exact:true}).click();await p.waitForTimeout(350);await p.locator('#graphSvg g[cursor=pointer]').first().hover();await p.locator('#tooltip').waitFor({state:'visible'});}catch{record.gap='Tooltip not pointer-reachable at this viewport after using Fit; contextual graph screenshot only.';}}
    await p.waitForTimeout(200);
    const dir=path.join(OUT,'screenshots',name,v.key);await fs.mkdir(dir,{recursive:true});
    const shot=path.join(dir,s.id+'.png');await p.screenshot({path:shot,fullPage:false,timeout:20000});record.screenshot=path.relative(OUT,shot);
    if(!gap&&!record.gap){
     await p.addScriptTag({content:CHECK_SOURCE});
     record.touch=await p.evaluate(sel=>window.__FL_AUDIT.touchTargets(sel),region||null);
     record.dom=await p.evaluate(inspect,region||null);
     let axe=new AxeBuilder({page:p}).withRules(['color-contrast']);if(region)axe=axe.include(region);
     const a=await axe.analyze();record.axe={violations:a.violations.map(x=>({id:x.id,nodes:x.nodes.map(n=>({target:n.target,summary:n.failureSummary,checks:n.any.map(c=>c.data)}))})),incomplete:a.incomplete.map(x=>({id:x.id,count:x.nodes.length})),passes:a.passes.map(x=>({id:x.id,count:x.nodes.length}))};
     // Keyboard check once per size/theme on every region containing inputs; all controls are only focused.
     record.keyboard=await keyboard(p,region,{isForm:/Form|Dialog/.test(s.name)});
    }
   }catch(e){record.error=String(e.message).split('\n')[0];}
   result.records.push(record);
  }
  if(s.print){await p.emulateMedia({media:'screen'});await p.waitForTimeout(150);}
  await cancel(p,s).catch(()=>{});
  result.guards=await p.evaluate(()=>window.__auditGuard).catch(()=>[]);
  result.requests=requests.slice(requestStart);result.responses=responses.slice(responseStart);result.dialogs=dialogs.slice(dialogStart);result.finished=new Date().toISOString();
  await fs.writeFile(outfile,JSON.stringify(result,null,2));
  console.log(name,s.id,gap?'GAP: '+gap:'captured',result.records.filter(r=>r.error).map(r=>r.variant+':'+r.error).join('; '));

 }
 await c.close();await b.close();
}
for(const [name,engine] of engines)await worker(name,engine);

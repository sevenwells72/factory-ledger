// Visual-only wallpaper review on real pages, with every API request stubbed locally.
// WALLPAPER_SOURCE_ROOT may point to a frozen pre-change copy of dashboard/.
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium,webkit} from 'playwright';
import {startStaticServer} from './lib/server.mjs';
import {CHECK_SOURCE} from './lib/checks.mjs';
import {renderedContrast} from './lib/rendered-contrast.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const source=process.env.WALLPAPER_SOURCE_ROOT||root;
const phase=process.env.WALLPAPER_PHASE||'after';
const output=path.join(root,'audits/screenshots/wallpaper-refresh');
await fs.mkdir(output,{recursive:true});
const server=await startStaticServer(source);
const browserType=process.env.WALLPAPER_BROWSER==='webkit'?webkit:chromium;
const browser=await browserType.launch({headless:true});
const widths=process.env.WALLPAPER_WIDTH?[Number(process.env.WALLPAPER_WIDTH)]:[1440,390];
const stylesheetHashes={};
for(const stylesheet of ['wallpaper.css','ledger-forms.css']){
  const data=await fs.readFile(path.join(source,'dashboard',stylesheet)).catch(()=>null);
  if(data)stylesheetHashes[stylesheet]=createHash('sha256').update(data).digest('hex');
}
const report={phase,sourceRoot:source,stylesheetHashes,captures:[]};
const actor={id:7,name:'Pilot Worker',role:'owner'};
try {
  for(const width of widths) for(const theme of ['light','dark']) {
    const context=await browser.newContext({viewport:{width,height:width===390?844:1000},deviceScaleFactor:1,colorScheme:theme,reducedMotion:'reduce',timezoneId:'America/New_York'});
    await context.addInitScript(theme=>{
      sessionStorage.setItem('fl-ledger-language','en');
      sessionStorage.setItem('fl-key-session-token','fls_wallpaper_visual_fixture');
      localStorage.setItem('dashboard-theme',theme);
      document.addEventListener('DOMContentLoaded',()=>document.documentElement.dataset.theme=theme);
    },theme);
    const unknown=[],remote=[],errors=[];
    await context.route('**/*',async route=>{
      const req=route.request(),url=new URL(req.url()),p=url.pathname;
      const send=json=>route.fulfill({json});
      if(url.origin!==server.origin){remote.push(req.url());return route.abort();}
      if(p.startsWith('/dashboard/'))return route.continue();
      if(p==='/auth/session/config')return send({key_sessions_enabled:true});
      if(p==='/auth/dashboard-key')return send({dashboard_key:'fixture-read-key'});
      if(p==='/auth/whoami')return send({key_kind:'session',actor,permissions:{create_order:true,create_customer:true}});
      if(p==='/exceptions')return send({exceptions:[],count:0});
      if(p==='/production/runs')return send({runs:[],count:0});
      if(p==='/transactions/history')return send({transactions:[],has_more:false});
      if(p==='/assistant/session')return send({session_id:'0c101ab1-8c56-42a2-8f29-c048330030c3'});
      if(p==='/assistant/resume')return send({turns:[],drafts:[]});
      if(p==='/resolve'){
        const kind=req.postDataJSON().kind;
        const item=kind==='customer'?{id:90,name:'Sunrise Bakery'}:kind==='supplier'?{id:91,name:'Green Valley Farms'}:{id:13,name:'Classic Granola 20 LB Case'};
        return send({outcome:'match',match:{...item,label:item.name}});
      }
      unknown.push({method:req.method(),path:p});return route.fulfill({status:404,json:{detail:'Unknown local fixture route'}});
    });
    for(const [name,file,ready] of [['create-order','office.html','#office-form'],['record-entry','entry.html','#entry-form'],['fl-assistant','fl-assistant.html','#chat-panel'],['exceptions','exceptions.html','main h1'],['runs','runs.html','#week-board'],['history','history.html','#history-form']]){
      const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
      await page.goto(server.origin+'/dashboard/'+file,{waitUntil:'networkidle'});
      await page.locator(ready).waitFor({state:'visible'});
      await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
      await page.waitForTimeout(500);
      await page.addScriptTag({content:CHECK_SOURCE});
      const result=await page.evaluate(()=>({
        title:document.title,width:innerWidth,theme:document.documentElement.dataset.theme,
        contrast:window.__FL_AUDIT.contrast(),touch:window.__FL_AUDIT.touchTargets(),overflow:window.__FL_AUDIT.horizontalOverflow(),
        wallpaper:{image:getComputedStyle(document.body).backgroundImage,attachment:getComputedStyle(document.body).backgroundAttachment,size:getComputedStyle(document.body).backgroundSize},
      }));
      result.renderedContrast=await renderedContrast(page);
      const filename=`${phase}-${name}-${width}-${theme}.png`;
      await page.screenshot({path:path.join(output,filename),fullPage:true});
      report.captures.push({name,file:filename,...result});
      await fs.writeFile(path.join(output,`${phase}-page-checks.json`),JSON.stringify({...report,unknown,remote,errors},null,2)+'\n');
      console.log(filename,JSON.stringify({contrast:result.contrast.failures,renderedContrast:result.renderedContrast.failures,overflow:result.overflow,touch:result.touch.failures}));
      await page.close();
    }
    await context.close();
    if(unknown.length||remote.length||errors.length)throw Error(JSON.stringify({unknown,remote,errors}));
  }
  if(phase.startsWith('after'))for(const capture of report.captures){
    assert.equal(capture.overflow.overflowPx,0,`${capture.file}: horizontal overflow`);
    assert.equal(capture.renderedContrast.failures,0,`${capture.file}: rendered contrast failures`);
    assert.ok(capture.wallpaper.image.includes('wallpaper.webp'),`${capture.file}: wallpaper loaded`);
    assert.equal(capture.wallpaper.size,'cover',`${capture.file}: wallpaper covers page`);
    assert.equal(capture.wallpaper.attachment,capture.width<=768?'scroll':'fixed',`${capture.file}: responsive background attachment`);
  }
}finally{await browser.close();await server.close();}

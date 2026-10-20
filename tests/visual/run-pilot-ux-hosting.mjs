// Hosting-only smoke: real static files with local API fixtures, no database/model/network writes.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {startStaticServer} from './lib/server.mjs';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const output=process.env.PILOT_UX_HOSTING_OUT||'/tmp/fl-pilotux-hosting';
await fs.mkdir(output,{recursive:true});
const layouts=[{name:'published-root',directory:path.join(root,'dashboard'),prefix:'/'},{name:'dashboard-prefix',directory:root,prefix:'/dashboard/'}];
const report={local_only:true,fixture_only:true,openai_calls:0,legacy_alias_note:'Only the known seven initial /dash/ asset preloads may return 404; every final /dashboard/ asset must return 200. Published-root and ordinary /dashboard/ pages allow no missing assets.',scenarios:[],screenshots:[]};
const aliasAssets=new Set(['dashboard-runtime.js','ledger-forms.css','fl-assistant.css','key-session.js','assistant-display.js','fl-assistant.js','receipt-shell.js']);
const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||'chrome'});
const actor={id:901,name:'Hosting fixture',role:'office'},permissions={create_order:true,create_customer:true,create_expected_receipt:true,mark_order_ready:true};
let checks=0;
try{
 for(const layout of layouts){
  const server=await startStaticServer(layout.directory);
  try{
   for(const sessions of [false,true])for(const language of ['en','es'])for(const width of [1440,390]){
    const scenario={layout:layout.name,sessions,width,language,checks:0,passed:false};report.scenarios.push(scenario);
    const context=await browser.newContext({viewport:{width,height:width===390?844:1000}});
    const remote=[],unexpected=[],pageErrors=[],missingAssets=[],assetResponses=[],requests=[];let headerOnly=false;
    await context.addInitScript(({language,sessions})=>{if(!['http:','https:'].includes(location.protocol))return;sessionStorage.setItem('fl-ledger-language',language);localStorage.setItem('fl-assistant-language',language);if(sessions)sessionStorage.setItem('fl-key-session-token','fls_hosting_fixture');},{language,sessions});
    await context.route('**/*',async route=>{
     const request=route.request(),url=new URL(request.url());
     if(url.origin!==server.origin){remote.push(url.href);return route.abort();}
     // Isolate the shared header on report pages; their full app behavior has separate runners.
     if(headerOnly&&request.resourceType()==='script'&&!['dashboard-runtime.js','receipt-shell.js'].includes(url.pathname.split('/').pop()))return route.fulfill({status:200,contentType:'text/javascript',body:'/* shared-header-only fixture */'});
     if(headerOnly&&request.resourceType()==='document')return route.fulfill({status:200,contentType:'text/html',headers:{'Content-Security-Policy':"script-src 'self'"},body:await fs.readFile(path.join(root,'dashboard',url.pathname.split('/').pop()))});
     if(url.pathname==='/dash/fl-assistant'&&layout.name==='dashboard-prefix')return route.fulfill({status:200,contentType:'text/html',body:await fs.readFile(path.join(root,'dashboard/fl-assistant.html'))});
     if(!url.pathname.startsWith('/auth/')&&!url.pathname.startsWith('/assistant/'))return route.continue();
     requests.push({path:url.pathname,method:request.method()});
     const json=(body,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
     if(url.pathname==='/auth/session/config')return json({key_sessions_enabled:sessions,pin_login_enabled:false,assistant_enabled:false});
     if(url.pathname==='/auth/dashboard-key')return json({dashboard_key:'hosting_read_fixture'});
     if(url.pathname==='/auth/whoami')return json({key_kind:sessions?'session':'actor',actor,permissions});
     if(url.pathname==='/auth/session')return json({key_kind:'session',session_token:'fls_hosting_fixture',actor});
     if(url.pathname==='/auth/session/logout')return json({success:true});
     if(['/assistant/session','/assistant/resume'].includes(url.pathname))return json({detail:{error_code:'ASSISTANT_DISABLED',message:'FL Assistant is not enabled / FL Assistant no está habilitado.'}},503);
     unexpected.push(url.pathname);return json({detail:{error_code:'UNEXPECTED_FIXTURE_REQUEST'}},501);
    });
    context.on('page',page=>{page.on('pageerror',error=>pageErrors.push(error.message));page.on('response',response=>{const url=new URL(response.url());if(url.origin===server.origin)assetResponses.push({path:url.pathname,status:response.status()});if(url.origin===server.origin&&response.status()>=400&&!url.pathname.startsWith('/assistant/'))missingAssets.push({path:url.pathname,status:response.status()});});});
    const page=await context.newPage();page.setDefaultTimeout(8000);
    const check=()=>{checks++;scenario.checks++;};
    const screenshot=async name=>{const file=path.join(output,`${layout.name}-${sessions?'on':'off'}-${width}-${language}-${name}.png`);await page.screenshot({path:file,fullPage:true});report.screenshots.push(file);};
    async function offline(file='fl-assistant.html'){
     if(file!==null)await page.goto(server.origin+(file.startsWith('/')?file:layout.prefix+file));
     await page.waitForFunction(()=>window.FLDashboard&&window.FLKeySessions&&window.FLReceipts);
     await page.locator('#auth-loading').waitFor({state:'hidden'});
     if(await page.locator('#sign-in-panel').isVisible()){
      await page.locator('#actor-key').fill('hosting_person_fixture');await page.locator('#sign-in-form button').click();
     }
     await page.locator('.card.handoff a[href*="office.html"]').waitFor();
     assert.equal(await page.locator('.card.handoff a').count(),2);
     assert.equal(await page.locator('#send').isDisabled(),true);
     assert.ok(requests.some(r=>r.path==='/assistant/session'||r.path==='/assistant/resume'));
     check();
    }
    async function header(){
     const menu=page.locator('#header-menu'),hasMenu=await menu.count(),wasOpen=hasMenu?await menu.evaluate(node=>node.open):false;if(hasMenu)await menu.evaluate(node=>node.open=true);
     const tools=page.locator('.receipt-tools a[data-fl-write-ui]');assert.equal(await tools.count(),3);
     for(const tool of await tools.all()){
      assert.equal(await tool.isVisible(),sessions);
      if(sessions)assert.ok(await tool.evaluate(node=>{const style=getComputedStyle(node),rect=node.getBoundingClientRect();return parseFloat(style.borderTopWidth)>=1&&parseFloat(style.paddingTop)>=6&&parseFloat(style.borderRadius)>=4&&rect.height>=44;}));
     }
     assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);if(hasMenu)await menu.evaluate((node,open)=>node.open=open,wasOpen);
     check();
    }
    try{
     await offline();await header();
     for(const [file,form] of [['office.html','#office-form'],['entry.html','#entry-form']]){
      const link=page.locator(`.card.handoff a[href*="${file}"]`),target=new URL(await link.getAttribute('href'),page.url());
      assert.equal(target.pathname,layout.prefix+file);await link.click();
      await page.waitForFunction(()=>window.FLLedger&&window.FLReceipts);await page.evaluate(()=>FLLedger.ready);
      if(sessions)await page.locator(form).waitFor();else assert.equal(await page.locator(form).count(),0);
      assert.equal(new URL(page.url()).pathname,layout.prefix+file);await header();check();
      const home=page.locator('header a').filter({hasText:'Factory Ledger'}).first();assert.equal(new URL(await home.getAttribute('href'),page.url()).pathname,layout.prefix+'fl-assistant.html');
      await screenshot(file.replace('.html',''));
      await home.click();assert.equal(new URL(page.url()).pathname,layout.prefix+'fl-assistant.html');await offline(null);check();
     }
     if(layout.name==='dashboard-prefix'){
      const beforeAlias=assetResponses.length;await offline('/dash/fl-assistant');await header();assert.equal(await page.evaluate(()=>document.baseURI),server.origin+'/dashboard/');assert.equal(new URL(await page.locator('.card.handoff a[href*="entry.html"]').getAttribute('href'),page.url()).pathname,'/dashboard/entry.html');
      for(const asset of aliasAssets)assert.ok(assetResponses.slice(beforeAlias).some(response=>response.path==='/dashboard/'+asset&&response.status===200),'Final alias asset loaded: '+asset);
      scenario.legacy_alias_preloads=missingAssets.filter(response=>response.path.startsWith('/dash/')&&aliasAssets.has(response.path.slice('/dash/'.length))&&response.status===404);check();
     }
     await screenshot('offline');
     if(sessions){
      headerOnly=true;
      for(const file of ['index.html','history.html','runs.html','process-flow.html','sankey.html','traceability.html','shift-summary.html','receipt.html','exceptions.html']){
       await page.goto(server.origin+layout.prefix+file);await page.waitForFunction(()=>window.FLDashboard&&window.FLReceipts);await page.evaluate(()=>FLDashboard.ready);await header();
      }
      headerOnly=false;
     }
     assert.deepEqual(remote,[]);assert.deepEqual(unexpected,[]);assert.deepEqual(pageErrors,[]);assert.deepEqual(missingAssets.filter(response=>!scenario.legacy_alias_preloads?.includes(response)),[]);assert.equal(requests.some(r=>r.path==='/assistant/turn'),false);check();scenario.passed=true;
    }catch(error){scenario.error=error.stack||String(error);await screenshot('FAIL').catch(()=>{});throw error;}
    finally{await context.close();await fs.writeFile(path.join(output,'results.json'),JSON.stringify({...report,checks},null,2));}
   }
  }finally{await server.close();}
 }
}finally{await browser.close();}
console.log(JSON.stringify({scenarios:report.scenarios.length,checks,passed:report.scenarios.filter(s=>s.passed).length,screenshots:report.screenshots.length,results:path.join(output,'results.json')}));

// All API traffic is intercepted. FL_KEY_TEST_ROOT can point at an archived baseline tree.
// Exercises the actual page loaders and Retry controls, including both bilingual pages.
import assert from 'node:assert/strict';
import path from 'node:path';
import {chromium} from 'playwright';
import {startStaticServer} from './lib/server.mjs';
import {API_HOST,buildTokenTable,installApiStub} from './lib/stub.mjs';
const root=path.resolve(process.env.FL_KEY_TEST_ROOT||'.');
const message='Access unavailable — refresh the page or contact the administrator';
const spanish='Acceso no disponible — actualiza la página o contacta al administrador';
const cases=[
 ['today','index.html','#today-tile','.today-tile-retry'],
 ['recent','index.html?section=recent','#recent-entries-feed','.recent-entries-retry'],
 ['orders','index.html?section=orders','#orders-error','#orders-error button'],
 ['expected','index.html?section=expected','#er-error','#er-error button'],
 ['supply inventory','index.html?section=supplies','#supplies-inventory-error','#supplies-inventory-error button'],
 ['supply requests','index.html?section=supplies','#supply-requests-error','#supply-requests-error button'],
 ['calendar','index.html','.mini-calendar-access','.mini-calendar-access button','en','.reference-calendar summary'],
 ['production lines','process-flow.html','#error-banner','#error-banner button'],
 ['material flow','sankey.html','#banner','#banner button'],
 ['traceability','traceability.html','#statusBar','#statusBar button'],
 ['history','history.html','#history-status','#history-form button'],
 ['runs','runs.html','#load-status','#retry-load'],
 ...['en','es'].flatMap(lang=>[
  ['receipt '+lang,'receipt.html?number=RCV-261014-001','#notice','#notice button',lang],
  ['shift summary '+lang,'shift-summary.html','#notice','#read-only',lang,'#read-only'],
 ]),
];
const server=await startStaticServer(path.join(root,'dashboard'));
const browser=await chromium.launch({channel:'chrome',headless:true});
let passed=0;const failures=[];
async function check(spec,mode){
 const [name,url,region,retry,lang='en',start]=spec,context=await browser.newContext({viewport:{width:1440,height:1000}});
 let broken=true,keyReads=0,refused=0,success=0;const errors=[],hosted=[];
 const page=await context.newPage();page.setDefaultTimeout(5000);page.on('pageerror',e=>errors.push(e.message));
 try{
  // External chart libraries are unnecessary for the empty-data recovery fixture.
  await context.route('**/*',route=>new URL(route.request().url()).origin===server.origin?route.continue():route.abort());
  await installApiStub(context,buildTokenTable(),{fail:[],overrides:{},status:{},unmatched:new Set()});
  await context.addInitScript(lang=>{sessionStorage.setItem('fl-ledger-language',lang);localStorage.setItem('fl-shift-language',lang);},lang);
  await context.route(`**://${API_HOST}/**`,async route=>{
   const req=route.request(),u=new URL(req.url());
   assert.equal(req.method(),'GET','error/retry coverage must remain read-only');
   if(u.pathname==='/auth/session/config')return route.fulfill({json:{key_sessions_enabled:false}});
   if(u.pathname==='/auth/dashboard-key'){
    keyReads++;
    if(broken&&mode==='key503')return route.fulfill({status:503,json:{detail:'Dashboard key unavailable (HTTP 503) private-detail'}});
    return route.fulfill({json:{dashboard_key:broken?'fixture-refused':'fixture-recovered'}});
   }
   if(req.headers()['x-api-key']){
    if(broken){refused++;return route.fulfill({status:Number(mode.slice(-3))||403,json:{detail:'Invalid API key private-detail'}});}
    // A Retry must discard an earlier wrong key, not just replay it until its TTL.
    assert.equal(req.headers()['x-api-key'],'fixture-recovered');success++;
   }
   if(u.pathname==='/transactions/history')return route.fulfill({json:{transactions:[],count:0}});
   if(u.pathname==='/production/runs')return route.fulfill({json:{runs:[]}});
   if(u.pathname==='/auth/whoami')return route.fulfill({json:{actor:null,key_kind:'dashboard'}});
   if(u.pathname.startsWith('/receipts/'))return route.fulfill({json:{receipt_number:'RCV-261014-001',status:'posted',transactions:[],lots:[]}});
   if(u.pathname==='/reports/shift-summary')return route.fulfill({json:{title:'Recovered shift',receipts_today:0,sections:[],date:u.searchParams.get('date'),confirmation:{status:'unconfirmed',can_confirm:false,reason:'Read only'}}});
   return route.fallback();
  });
  page.on('request',r=>{const u=new URL(r.url());if(u.hostname.endsWith('railway.app')&&u.hostname!==API_HOST)hosted.push(u.href);});
  await page.goto(server.origin+'/'+url);if(start)await page.locator(start).click();
  await page.waitForLoadState('networkidle');
  const box=page.locator(region).first(),expected=lang==='es'?spanish:message;
  assert.ok(await box.isVisible(),`${name}: failure notice visible`);
  assert.ok((await box.innerText()).includes(expected),`${name}: expected plain access message; got ${(await box.innerText()).slice(0,150)}`);
  assert.doesNotMatch(await page.locator('body').innerText(),/HTTP\s*\d{3}|Invalid API key|private-detail|Dashboard key unavailable|permission to make this change|ledger API did not answer|"detail"/i);
  assert.ok(keyReads>0,'real key bootstrap was attempted');
  if(mode==='key503')assert.equal(refused,0,'failed key bootstrap never sends a keyed request');else assert.ok(refused>0,'wrong credential reaches a keyed GET');
  assert.deepEqual(errors,[],'no page errors during refusal');
  const readsBefore=keyReads;broken=false;
  const recovered=page.waitForResponse(r=>Boolean(r.request().headers()['x-api-key'])&&r.status()===200);
  await page.locator(retry).first().click();
  await recovered;
  await page.waitForFunction(({region,expected})=>{const el=document.querySelector(region);return !el||el.hidden||getComputedStyle(el).display==='none'||!el.innerText.includes(expected);},{region,expected});
  assert.ok(success>0,'Retry loads real successful response');
  assert.ok(keyReads>readsBefore,'Retry fetches current key immediately');
  assert.ok(!(await box.isVisible())||!(await box.innerText()).includes(expected),'target notice clears after recovery');
  assert.deepEqual(errors,[],'no page errors during recovery');assert.deepEqual(hosted,[]);
  passed++;console.log(`PASS ${name} ${mode}: plain message, no details, Retry recovery`);
 }catch(error){if(process.env.FL_KEY_DEBUG){console.log({name,mode,keyReads,refused,success,errors});await page.screenshot({path:'.venv-test/key-debug-'+name.replaceAll(' ','-')+'.png',fullPage:true});console.log(await page.locator(region).allTextContents());}failures.push(`${name} ${mode}: ${error.message}`);console.log(`FAIL ${failures.at(-1)}`);}
 finally{await context.close();}
}
const selected=cases.filter(c=>!process.env.FL_KEY_CASE||process.env.FL_KEY_CASE.split(',').includes(c[0]));
try{for(const spec of selected)await Promise.all(['key503','wrong403','wrong401'].map(mode=>check(spec,mode)));}
finally{await browser.close();await server.close();}
console.log(`${passed}/${selected.length*3} dashboard access cases passed; ${failures.length} failed; no real API requests.`);
if(failures.length)process.exitCode=1;

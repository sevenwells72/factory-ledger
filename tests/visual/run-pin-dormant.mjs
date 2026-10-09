// Compare actual browser behavior with the unchanged pre-A11 main assets.
// All API/CDN traffic is fulfilled locally. Never display credentials or PINs.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {chromium} from 'playwright';
import {startStaticServer} from './lib/server.mjs';
import {installApiStub, buildTokenTable} from './lib/stub.mjs';

const baseline = process.env.PIN_PARITY_BASE || 'ce7bb57';
const reference = await fs.mkdtemp(path.join(os.tmpdir(), 'fl-a11-parity-'));
const files = execFileSync('git',['ls-tree','-r','--name-only',baseline,'dashboard'],{encoding:'utf8'}).trim().split('\n');
for (const file of files) {
  const out = path.join(reference, file.slice('dashboard/'.length));
  await fs.mkdir(path.dirname(out),{recursive:true});
  await fs.writeFile(out,execFileSync('git',['show',baseline+':'+file]));
}
// The baseline's already-public dashboard key stays in memory only.
const legacy = await fs.readFile(path.join(reference,'dashboard.js'),'utf8');
const key = legacy.match(/const SALES_API_KEY = '([^']+)'/)[1];
const oldServer = await startStaticServer(reference), newServer = await startStaticServer(path.resolve('dashboard'));
const browser = await chromium.launch();
const pages = files.filter(f=>f.endsWith('.html')).map(f=>f.slice('dashboard/'.length));
const tokens = buildTokenTable(new Date('2026-10-09T16:00:00Z'));
let checks=0;
async function inspect(server, name, width, configState=false) {
  const context = await browser.newContext({viewport:{width,height:900},timezoneId:'America/New_York'});
  const requests=[], errors=[];
  await context.addInitScript(() => {
    const NativeDate=Date, now=Date.parse('2026-10-09T16:00:00Z');
    window.Date=class extends NativeDate {constructor(...args){super(...(args.length?args:[now]));} static now(){return now;}};
    localStorage.setItem('fl-actor-key','synthetic-existing-key');
    sessionStorage.setItem('fl-session-v1','synthetic-existing-session');
  });
  // Deny every unstubbed external URL, including production, before installing fixtures.
  await context.route('**/*', route => new URL(route.request().url()).origin===server.origin ? route.continue() : route.abort());
  await installApiStub(context,tokens,{});
  await context.route('**/auth/config', route => configState==='failure' ? route.fulfill({status:503,body:'Unavailable'}) :
    route.fulfill({json:configState?{pin_login_enabled:true}:{pin_login_enabled:false,dashboard_key:key}}));
  await context.route('https://cdnjs.cloudflare.com/ajax/libs/d3/**',async route=>route.fulfill({contentType:'application/javascript',body:await fs.readFile(path.resolve('node_modules/d3/dist/d3.min.js'))}));
  await context.route('https://cdnjs.cloudflare.com/ajax/libs/d3-sankey/**',async route=>route.fulfill({contentType:'application/javascript',body:await fs.readFile(path.resolve('node_modules/d3-sankey/dist/d3-sankey.min.js'))}));
  context.on('request', req=>{
    const u=new URL(req.url());
    if(u.hostname!=='fastapi-production-b73a.up.railway.app')return;
    const h=req.headers();
    // Compare credential identity by boolean only; never save the credential.
    requests.push({url:u.pathname+u.search,method:req.method(),dashboardKey:h['x-api-key']===key,
      hasKey:!!h['x-api-key'],hasMarker:!!h['x-fl-client'],contentType:h['content-type']||null});
  });
  const page=await context.newPage();page.on('pageerror', e=>errors.push(e.name));
  await page.goto(server.origin+'/'+name);await page.waitForLoadState('networkidle');
  if (!configState) {
    assert.equal(await page.locator('.fl-login,.fl-person-bar').count(),0);checks++;
    assert.ok(await page.evaluate(()=>localStorage.getItem('fl-actor-key')==='synthetic-existing-key' && sessionStorage.getItem('fl-session-v1')==='synthetic-existing-session'));checks++;
  }
  const result={requests:requests.sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b))),
    text:await page.locator('body').innerText(),errors};
  if(configState===true){assert.equal(await page.locator('.fl-person-bar').count(),1);checks++;}
  if(configState==='failure'){assert.equal(requests.length,0);checks++;}
  await context.close();return result;
}
try {
  for(const width of [390,1440])for(const name of pages){
    const before=await inspect(oldServer,name,width), after=await inspect(newServer,name,width);
    // A custom message suppresses credentials even if a future change regresses redaction.
    assert.ok(JSON.stringify(after)===JSON.stringify(before),`Dormant browser parity: ${name} at ${width}px`);checks++;
  }
  await inspect(newServer,'index.html',390,true);
  await inspect(newServer,'index.html',390,'failure');
  // A separately hosted admin URL has no visible forms and redirects while off.
  const context=await browser.newContext();
  await context.route('**/*', route => new URL(route.request().url()).origin===newServer.origin ? route.continue() : route.abort());
  await context.route('**/auth/config',route=>route.fulfill({json:{pin_login_enabled:false,dashboard_key:key}}));
  const page=await context.newPage();await page.goto(newServer.origin+'/pin-management.html');
  await page.waitForURL('**/index.html');assert.equal(await page.locator('#own-pin,#admin-pin').count(),0);checks++;
  await context.close();
  console.log(JSON.stringify({baseline,pages:pages.length,widths:[390,1440],checks,production_requests:0}));
} finally {
  await browser.close();await oldServer.close();await newServer.close();await fs.rm(reference,{recursive:true,force:true});
}

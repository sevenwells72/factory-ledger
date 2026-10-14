// Virtual Netlify origins; every API request is intercepted and forwarded ONLY to local FL.
import assert from 'node:assert/strict';import fs from 'node:fs/promises';import path from 'node:path';import {chromium} from 'playwright';
const base=process.env.D12_BASE||'http://127.0.0.1:8125';assert.ok(['localhost','127.0.0.1'].includes(new URL(base).hostname));
const production='https://fastapi-production-b73a.up.railway.app',staging='https://fastapi-staging-production-dd7b.up.railway.app';
const pages=['index.html','history.html','runs.html','process-flow.html','traceability.html','sankey.html','entry.html','office.html','exceptions.html','receipt.html','shift-summary.html','fl-assistant.html','scheduler/seven-wells-production-board.html'];
const browser=await chromium.launch({channel:'chrome',headless:true});
try{
 for(const host of ['deploy-preview-107--cns-factory-ledger.netlify.app','feature--cns-factory-ledger.netlify.app','other-site.netlify.app','cns-factory-ledger.netlify.app']){
  const origin='https://'+host,expected=host==='cns-factory-ledger.netlify.app'?production:staging,context=await browser.newContext(),calls=[],blocked=[];
  await context.addInitScript(production=>{window.FL_SESSION_API_BASE=production;},production);
  await context.route('**/*',async route=>{
   const url=new URL(route.request().url());
   if(url.origin===origin){const file=path.resolve('.'+url.pathname);if(!file.startsWith(path.resolve('dashboard')+path.sep))return route.abort();try{const body=await fs.readFile(file);const ext=path.extname(file);return route.fulfill({body,contentType:ext==='.html'?'text/html':ext==='.js'?'text/javascript':ext==='.css'?'text/css':'image/svg+xml'});}catch{return route.abort();}}
   if([production,staging].includes(url.origin)){
    calls.push({origin:url.origin,path:url.pathname});if(url.origin!==expected){blocked.push(url.href);return route.abort();}
    // Override the target before any network call: never contact hosted FL, even in canonical-host simulation.
    const response=await route.fetch({url:base+url.pathname+url.search,maxRedirects:0});return route.fulfill({response});
   }
   // Local requests are route.fetch forwarding; external assets are unnecessary for destination assertions.
   return route.abort();
  });
  const page=await context.newPage();
  for(const name of pages){const before=calls.length;await page.goto(origin+'/dashboard/'+name+'?api='+production);await page.evaluate(()=>FLDashboard.ready);assert.equal(await page.evaluate(()=>FLDashboard.base),expected);
   if(await page.evaluate(()=>!!window.FL?.fetchWithTimeout))await page.evaluate(production=>FL.fetchWithTimeout(production+'/health').then(r=>r.status),production);
   assert.ok(calls.slice(before).some(c=>c.path==='/auth/session/config'),name+' config requested');
  }
  assert.deepEqual(blocked,[],'preview never targets production');assert.ok(calls.length>=pages.length);assert.ok(calls.every(c=>c.origin===expected));await context.close();
 }
 console.log('Netlify host routing passed: 13 pages × 4 hosts, unsafe overrides ignored on previews, zero hosted API network access.');
}finally{await browser.close();}

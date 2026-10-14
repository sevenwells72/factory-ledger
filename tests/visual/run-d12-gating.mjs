import assert from 'node:assert/strict';import fs from 'node:fs/promises';import {chromium} from 'playwright';
const base=process.env.D12_BASE||'http://127.0.0.1:8125',on=process.env.D12_GATE_ON==='1';assert.ok(['localhost','127.0.0.1'].includes(new URL(base).hostname));
const f=JSON.parse(await fs.readFile('.venv-test/d12-fixture.json','utf8'));
async function api(path,body){const r=await fetch(base+path,{method:body?'POST':'GET',headers:{'X-API-Key':'d12-local-floor','Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});assert.ok(r.ok,path);return r.json();}
const prepared=await api('/receive/prepare',{product_id:f.products['Rolled Oats'],supplier_id:f.supplier,cases:1,case_size_lb:1,bol_reference:'D12 gate fixture',supplier_lot_code:'GATE-001',client_source:'dashboard'});
const receipt=await api('/tickets/'+prepared.ticket+'/commit',{payload_hash:prepared.payload_hash,acknowledged_warnings:prepared.warnings.map(w=>w.code)});
const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:1280,height:900}}),calls=[];
page.on('request',r=>calls.push({path:new URL(r.url()).pathname,method:r.method()}));await page.route('**/*',r=>new URL(r.request().url()).origin===base?r.continue():r.abort());
const noNewUI=async()=>{assert.equal(await page.locator('[data-fl-write-ui]:not(html):visible').count(),0);assert.equal(await page.locator('.fl-session-bar:visible,#sign-in-form:visible,#identity:visible').count(),0);assert.equal(await page.getByRole('link',{name:/Record entry|Record shipment|Exceptions|Excepciones|Open exceptions/}).count(),0);assert.equal(await page.locator('.receipt-lookup').isVisible(),true);};
async function ready(){assert.equal(await page.evaluate(()=>FLDashboard.ready),on);}
try{
 await page.goto(base+'/dashboard/index.html?section=orders');await ready();
 if(on){await page.getByRole('link',{name:'Record entry / Registrar',exact:true}).waitFor();await page.locator('.fl-session-bar').waitFor();}else await noNewUI();
 const order=page.locator('.order-row[data-order-id="'+f.order+'"]');await order.waitFor();await order.locator('.order-expand-toggle').click();await page.locator('.so-open-detail[data-order-id="'+f.order+'"]').click();await page.locator('.order-detail-header').waitFor();
 const shipment=page.getByRole('link',{name:'Record shipment — BOL + photo',exact:true});assert.equal(await shipment.isVisible(),on);
 await page.getByRole('button',{name:'Preview all remaining lines',exact:true}).waitFor();
 if(!on){const preview=page.waitForResponse(r=>new URL(r.url()).pathname==='/sales/orders/'+f.order+'/ship/preview');await page.getByRole('button',{name:'Preview all remaining lines',exact:true}).click();assert.equal((await preview).status(),200,'legacy preview still works');await noNewUI();}
 await page.setViewportSize({width:390,height:844});
 for(const name of ['entry','exceptions']){await page.goto(base+'/dashboard/'+name+'.html');await ready();if(on){await page.locator('#sign-in-form').waitFor();if(name==='entry')await page.locator('#entry-form').waitFor();}else{await noNewUI();assert.equal(await page.locator('#entry-form,#resolution form').count(),0);}}
 // Receipt lookup and row identities work without signing in when cutover is off.
 await page.locator('.receipt-lookup input').fill(receipt.receipt_number);await page.locator('.receipt-lookup button').click();await page.getByRole('heading',{name:receipt.receipt_number,exact:true}).waitFor();if(!on)await noNewUI();
 await page.goto(base+'/dashboard/history.html');await page.locator('summary').filter({hasText:receipt.receipt_number}).click();await page.getByRole('link',{name:receipt.receipt_number,exact:true}).waitFor();if(!on)await noNewUI();
 if(!on){
  assert.equal(calls.some(r=>r.path==='/auth/session'||r.path==='/auth/whoami'||r.path==='/exceptions'),false);assert.deepEqual(calls.filter(r=>r.method==='POST').map(r=>r.path),['/sales/orders/'+f.order+'/ship/preview']);
  // Neither slow nor unavailable configuration flashes forms or blocks a read-only receipt.
  let release;const pending=new Promise(r=>release=r);await page.route('**/auth/session/config',async r=>{await pending;return r.fulfill({status:503,json:{detail:{message:'Unavailable'}}});});
  await page.goto(base+'/dashboard/entry.html');await noNewUI();assert.equal(await page.locator('#entry-form').count(),0);release();await ready();await noNewUI();
  await page.locator('.receipt-lookup input').fill(receipt.receipt_number);await page.locator('.receipt-lookup button').click();await page.getByRole('heading',{name:receipt.receipt_number,exact:true}).waitFor();await noNewUI();
 }
 await fs.mkdir('.venv-test/d12-gating',{recursive:true});await page.screenshot({path:'.venv-test/d12-gating/'+(on?'on':'off')+'-390.png',fullPage:true});
 console.log('D12 cutover '+(on?'on: entry, shipment, Exceptions and sign-in visible':'off: no new write/sign-in UI; legacy shipment preview and receipt lookup/rows work; unavailable config stays read-only')+' passed.');
}finally{await browser.close();}

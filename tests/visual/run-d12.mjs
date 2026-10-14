import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from 'playwright';
const base=process.env.D12_BASE||'http://127.0.0.1:8122';
assert.ok(['127.0.0.1','localhost'].includes(new URL(base).hostname),'Use local fixture only');
const fixture=JSON.parse(await fs.readFile('.venv-test/d12-fixture.json','utf8'));
const out='.venv-test/d12-visual';await fs.mkdir(out,{recursive:true});
async function api(path,body,key='d12-local-floor'){const response=await fetch(base+path,{method:body?'POST':'GET',headers:{'X-API-Key':key,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await response.json();assert.ok(response.ok,JSON.stringify(data));return data;}
const prepared=await api('/receive/prepare',{product_id:fixture.products['Oat Bran'],supplier_id:fixture.supplier,cases:1,case_size_lb:10,bol_reference:'D12 browser',supplier_lot_code:'UNKNOWN',client_source:'dashboard'});
const receipt=await api('/tickets/'+prepared.ticket+'/commit',{payload_hash:prepared.payload_hash,acknowledged_warnings:prepared.warnings.map(w=>w.code)});
const heldDraft=await api('/adjust/prepare',{lot_id:fixture.lots['Oat Bran'].id,delta_lb:501,reason_code:'physical_count',client_source:'dashboard'});
const hold=await api('/tickets/'+heldDraft.ticket+'/commit',{payload_hash:heldDraft.payload_hash,acknowledged_warnings:heldDraft.warnings.map(w=>w.code)});
assert.equal(hold.held,true);
const browser=await chromium.launch({channel:'chrome',headless:true});
const context=await browser.newContext({viewport:{width:390,height:844}});const page=await context.newPage();const requests=[],errors=[];
page.on('request',request=>{if(new URL(request.url()).origin===base)requests.push({path:new URL(request.url()).pathname,method:request.method(),key:request.headers()['x-api-key'],body:request.postData()});});page.on('pageerror',error=>errors.push(error.message));
await page.route('**/*',route=>new URL(route.request().url()).origin===base?route.continue():route.abort());
async function signIn(role){await page.locator('#actor-key').fill('d12-local-'+role);await page.locator('#sign-in-form button').click();await page.getByText('Signed in as '+(role==='floor'?'Arturo D12':'Owner D12'),{exact:true}).waitFor();assert.equal(await page.locator('#actor-key').inputValue(),'');assert.equal(await page.evaluate(()=>JSON.stringify(sessionStorage).includes('d12-local-')),false);}
try {
 await page.goto(base+'/dashboard/exceptions.html');await signIn('floor');await page.getByRole('heading',{name:/UNIDENTIFIED_LOT/}).waitFor();
 assert.ok(requests.some(r=>r.path==='/exceptions'&&r.key?.startsWith('fls_')));
 await page.locator('.receipt-lookup input').fill(receipt.receipt_number);await page.locator('.receipt-lookup button').click();await page.getByRole('heading',{name:receipt.receipt_number,exact:true}).waitFor();assert.ok(requests.some(r=>r.path==='/receipts/'+receipt.receipt_number&&r.key?.startsWith('fls_')));
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:out+'/d1-receipt-390.png',fullPage:true});
 await page.goto(base+'/dashboard/exceptions.html');await page.getByRole('heading',{name:/UNIDENTIFIED_LOT/}).waitFor();
 const unidentified=page.locator('.panel').filter({has:page.getByRole('heading',{name:/UNIDENTIFIED_LOT/})});await unidentified.getByRole('button',{name:'Resolve',exact:true}).first().click();await page.locator('[name=supplier_lot_code]').fill('SUP-IDENTIFIED');await page.locator('[name=note]').fill('Read from supplier tag');await page.getByRole('button',{name:'Review resolution'}).click();await page.getByRole('heading',{name:'Draft — review before recording'}).waitFor();
 assert.ok(requests.some(r=>/^\/lots\/\d+\/supplier-lot\/prepare$/.test(r.path)&&r.method==='POST'&&r.key?.startsWith('fls_')));assert.ok(!requests.some(r=>r.method==='PATCH'));
 await page.getByRole('button',{name:'Sign out',exact:true}).click();await page.locator('#sign-in-form').waitFor();assert.equal(await page.evaluate(()=>sessionStorage.getItem('fl-key-session-token')),null);
 await signIn('owner');await page.getByRole('heading',{name:/LARGE_CORRECTION/}).waitFor();
 const large=page.locator('.panel').filter({has:page.getByRole('heading',{name:/LARGE_CORRECTION/})});await large.getByRole('button',{name:'Resolve',exact:true}).first().click();await page.locator('[name=note]').fill('Owner checked physical count');await page.getByRole('button',{name:'Review resolution'}).click();await page.getByRole('button',{name:'Confirm resolution'}).click();await page.locator('#notice.good').waitFor();assert.ok(requests.some(r=>/^\/exceptions\/\d+\/approve$/.test(r.path)&&r.key?.startsWith('fls_')));
 await page.getByRole('button',{name:'ES',exact:true}).click();await page.getByRole('heading',{name:'Excepciones',exact:true}).waitFor();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:out+'/d1-exceptions-es-390.png',fullPage:true});
 assert.deepEqual(errors,[]);console.log('D1 real HTTP browser passed: session headers, receipt lookup, exceptions, supplier-lot ticket draft, owner approval, logout, EN/ES and 390px.');
} finally {await browser.close();}

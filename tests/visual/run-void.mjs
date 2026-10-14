import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from 'playwright';
const base=process.env.D12_BASE||'http://127.0.0.1:8128',on=process.env.D12_GATE_ON!=='0';
assert.ok(['localhost','127.0.0.1'].includes(new URL(base).hostname));
const f=JSON.parse(await fs.readFile('.venv-test/d12-fixture.json','utf8')),out='.venv-test/void-visual';
await fs.mkdir(out,{recursive:true});
async function api(path,body,role='floor'){
 const r=await fetch(base+path,{method:body?'POST':'GET',headers:{'X-API-Key':'d12-local-'+role,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});
 assert.ok(r.ok,path+' returned '+r.status);return r.json();
}
async function source(role){
 const suffix=role+'-'+Date.now();
 const p=await api('/receive/prepare',{product_id:f.products['Oat Bran'],supplier_id:f.supplier,cases:1,case_size_lb:10,lot_code:'VOID-'+suffix,supplier_lot_code:'SUP-'+suffix,bol_reference:'VOID-'+suffix},role);
 return api('/tickets/'+p.ticket+'/commit',{payload_hash:p.payload_hash,acknowledged_warnings:p.warnings.map(w=>w.code)},role);
}
const own=await source('floor'),other=await source('owner');
const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:390,height:844}});
const requests=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>requests.push({path:new URL(r.url()).pathname,method:r.method(),body:r.method()==='POST'?r.postDataJSON():null}));
await page.route('**/*',r=>new URL(r.request().url()).origin===base?r.continue():r.abort());
async function visit(receipt){await page.goto(base+'/dashboard/receipt.html?number='+receipt.receipt_number);await page.locator('[data-state]').waitFor();}
async function signIn(role){if(await page.locator('#sign-out').isVisible())await page.locator('#sign-out').click();await page.locator('#actor-key').fill('d12-local-'+role);await page.locator('#sign-in-form button').click();await page.locator('#sign-out').waitFor();}
async function voidEntry(original,fail=false){
 await page.locator('[data-void="'+original.transaction_id+'"]').click();
 await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();
 assert.deepEqual(requests.findLast(r=>r.path.endsWith('/prepare')).body,{reason:'data_entry_error',client_source:'dashboard'});
 await page.locator('#language').click();assert.ok((await page.locator('#resolution').textContent()).includes('Borrador'));
 await page.locator('#language').click();
 if(fail){
  const route='**/tickets/*/commit';await page.route(route,r=>r.fulfill({status:409,contentType:'application/json',body:JSON.stringify({detail:{error_code:'TEST_REFUSAL',message:'Synthetic commit refusal',message_es:'Rechazo de prueba'}})}));
  await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();await page.locator('#notice.bad').waitFor();
  assert.match(await page.locator('#notice').textContent(),/^NOT recorded — Synthetic commit refusal/);assert.equal(await page.locator('#resolution .good').count(),0);
  await page.unroute(route);
 }
 await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();await page.locator('#resolution .good a').waitFor();
 const number=await page.locator('#resolution .good a').textContent();assert.match(number,/^VD-/);
 await page.locator('[data-state=voided]').waitFor();assert.equal(await page.locator('[data-void]').count(),0);
 assert.ok(await page.locator('#content a').filter({hasText:number}).count());
 const detail=await api('/receipts/'+original.receipt_number);assert.equal(detail.void_receipt_number,number);
 const vd=await api('/receipts/'+number);assert.equal(vd.response.source_receipt_number,original.receipt_number);
 const commits=requests.filter(r=>r.path.endsWith('/commit'));if(fail)assert.equal(commits.at(-1).path,commits.at(-2).path);
 const replay=await api(commits.at(-1).path,commits.at(-1).body,original===other?'owner':'floor');
 assert.equal(replay.receipt_number,number);assert.equal(replay.replayed,true);
 await page.screenshot({path:out+'/'+(original===other?'owner':'floor')+'-voided-390.png',fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.locator('#resolution .good a').click();await page.locator('[data-state=recorded]').waitFor();
 assert.ok((await page.locator('#content').textContent()).includes(original.receipt_number));
 return number;
}
try{
 await visit(own);
 assert.equal(await page.evaluate(()=>FLDashboard.ready),on);
 assert.equal(await page.locator('[data-void]').count(),0);
 if(on){
  await signIn('office');await page.locator('[data-state]').waitFor();assert.equal(await page.locator('[data-void]').count(),0);
  await signIn('floor');await page.locator('[data-void]').waitFor();
  await visit(other);assert.equal(await page.locator('[data-void]').count(),0);
  await visit(own);
  const floor=await voidEntry(own,true);
  await signIn('owner');await visit(other);const owner=await voidEntry(other);
  await visit(own);assert.equal(await page.locator('[data-void]').count(),0);
  await page.locator('#sign-out').click();assert.equal(await page.locator('[data-void]').count(),0);
  await fs.writeFile(out+'/receipts.json',JSON.stringify({floor,owner},null,2));
 }else{
  assert.equal(await page.locator('#sign-in-form').isVisible(),false);
  assert.equal(requests.some(r=>r.method==='POST'),false);
  assert.equal(requests.some(r=>r.path==='/auth/whoami'),false);
 }
 assert.deepEqual(errors,[]);
 assert.equal(requests.some(r=>r.method==='POST'&&r.path.startsWith('/void/')&&!r.path.endsWith('/prepare')),false);
 console.log('Void browser '+(on?'ON: floor + owner, failure/retry/replay, VD/original backlinks, EN/ES, 390px':'OFF: read only, no void or sign-in controls')+' passed');
}finally{await browser.close();}

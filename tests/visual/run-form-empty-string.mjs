// Oct 29 2026 regression: a blank lot-code row on the Make draft reached FL as "" and showed
// "NOT recorded — String should have at least 1 character". Real local FastAPI (d12_fixture.py) + Chrome, 390 px.
import assert from 'node:assert/strict';import fs from 'node:fs/promises';import {chromium} from 'playwright';
const base=process.env.D12_BASE||'http://127.0.0.1:8122';assert.ok(['localhost','127.0.0.1'].includes(new URL(base).hostname),'Use local fixture only');
const fixture=JSON.parse(await fs.readFile('.venv-test/d12-fixture.json','utf8')),out='.venv-test/form-empty-string';await fs.mkdir(out,{recursive:true});
async function api(path,body,key='d12-local-floor'){const response=await fetch(base+path,{method:body?'POST':'GET',headers:{'X-API-Key':key,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await response.json();assert.ok(response.ok,JSON.stringify(data));return data;}
// A second Oat Bran lot, so 15 batches (150 lb) draw from two lots: the multi-lot Make Arturo entered.
{const draft=await api('/receive/prepare',{client_source:'dashboard',product_id:fixture.products['Oat Bran'],supplier_id:fixture.supplier,cases:10,case_size_lb:10,bol_reference:'Empty-string second lot',supplier_lot_code:'EMPTY-SUP-002'});
 await api('/tickets/'+draft.ticket+'/commit',{payload_hash:draft.payload_hash,acknowledged_warnings:draft.warnings.map(w=>w.code)});}
const browser=await chromium.launch({channel:'chrome',headless:true}),context=await browser.newContext({viewport:{width:390,height:844}}),page=await context.newPage(),requests=[],errors=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(new URL(r.url()).origin===base)requests.push({path:new URL(r.url()).pathname,method:r.method(),body:r.headers()['content-type']?.includes('application/json')?r.postDataJSON():null});});
await page.route('**/*',r=>new URL(r.request().url()).origin===base?r.continue():r.abort());
const posts=path=>requests.filter(r=>r.method==='POST'&&r.path===path).length;
async function choose(name,query,scope=page){const picker=scope.locator('[data-picker="'+name+'"]');await picker.locator('input').fill(query);await picker.getByRole('button',{name:'Search',exact:true}).click();await picker.locator('.picker-options button').first().click();}
async function prepare(){await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();}
async function record(){for(const box of await page.locator('#resolution [data-warning]').all())await box.check();await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();const done=page.locator('#resolution p.good').filter({hasText:'Recorded'});await done.waitFor({timeout:90000});return done.locator('a').textContent();}
const rows=()=>page.locator('#resolution fieldset.lot-confirmation-row');
const codeOf=async row=>(await row.locator('legend').textContent()).split(' · ').at(-1);
async function check(){await page.locator('#resolution').getByRole('button',{name:'Check confirmations → new draft',exact:true}).click();}
async function recordable(){await page.waitForFunction(()=>{const b=Array.from(document.querySelectorAll('#resolution button')).find(b=>b.textContent==='Record');return b&&!b.disabled;});}
async function noRawPydantic(){assert.equal((await page.locator('body').textContent()).includes('String should have'),false,'raw Pydantic text shown');}
try{
 await page.goto(base+'/dashboard/entry.html');await page.locator('#actor-key').fill('d12-local-floor');await page.locator('#sign-in-form button[type=submit]').click();await page.getByText('Signed in as Arturo D12',{exact:true}).waitFor();
 // Make (formula cards since PR #173): 15 batches over two Oat Bran lots; output lot code and note left blank.
 const lotCodes=Object.fromEntries(((await api('/entry/lots?product_id='+fixture.products['Oat Bran'])).lots||[]).map(l=>[String(l.lot_id||l.id),l.lot_code]));
 await page.locator('#entry-action').selectOption('make');await choose('product','D1290');await page.locator('[name=batches]').fill('15');const sku=page.locator('[name=confirmed_sku]');if(await sku.count())await sku.check();await prepare();
 const first=requests.filter(r=>r.path==='/make/prepare').at(-1).body;assert.equal('lot_code' in first,false,'blank output lot omitted');assert.equal('note' in first,false,'blank note omitted');
 const bags=page.locator('#resolution .formula-bag');await bags.first().waitFor();assert.equal(await bags.count(),2,'two bags for the split ingredient');
 const bagCode=async row=>lotCodes[await row.locator('[name=bag_lot]').inputValue()];const [codeA,codeB]=[await bagCode(bags.nth(0)),await bagCode(bags.nth(1))];assert.ok(codeA&&codeB);
 await bags.nth(0).locator('[name=bag_method]').selectOption('full_code');await bags.nth(0).locator('[name=bag_code]').fill(codeA);
 await page.getByRole('button',{name:'Check bags and pounds → new draft',exact:true}).click();await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();
 const partial=requests.filter(r=>r.path==='/make/prepare').at(-1).body.lot_confirmations||[];assert.ok(partial.every(c=>c.value.trim()!==''),'blank bag code never sent as ""');
 await page.getByText(new RegExp('Confirm D12 Oat Bran lot '+codeB)).first().waitFor();await noRawPydantic();
 await page.screenshot({path:out+'/make-blank-bag-390.png',fullPage:true});
 const again=page.locator('#resolution .formula-bag');await again.nth(1).locator('[name=bag_method]').selectOption('full_code');await again.nth(1).locator('[name=bag_code]').fill(codeB);
 if(!(await again.nth(0).locator('[name=bag_code]').inputValue()))await again.nth(0).locator('[name=bag_code]').fill(codeA);
 await page.getByRole('button',{name:'Check bags and pounds → new draft',exact:true}).click();await recordable();
 const sent=requests.filter(r=>r.path==='/make/prepare').at(-1).body.lot_confirmations;assert.deepEqual(sent.map(c=>c.value).sort(),[codeA,codeB].sort());
 const made=await record();assert.match(made,/^[A-Z]+-\d{6}-\d+$/);
 // Pack: the same shared draft card.
 await page.getByRole('button',{name:'Record another entry',exact:true}).click();await page.locator('#entry-action').selectOption('pack');await choose('source','D1290');await choose('target','D1270');await page.locator('[name=cases]').fill('1');await prepare();
 const packBefore=posts('/pack/prepare');await check();await page.locator('#resolution [data-blank-confirmation]').first().waitFor();assert.equal(posts('/pack/prepare'),packBefore);await noRawPydantic();
 for(const row of await rows().all()){await row.locator('select').selectOption('full_code');await row.locator('input').fill(await codeOf(row));}await check();await recordable();const packed=await record();assert.match(packed,/^[A-Z]+-\d{6}-\d+$/);
 // Ship as Owner: sales-order mode first, then standalone, where a blank or spaces-only reference is refused by name.
 await page.getByRole('button',{name:'Sign out',exact:true}).click();await page.locator('#actor-key').fill('d12-local-owner');await page.locator('#sign-in-form button[type=submit]').click();await page.getByText('Signed in as Owner D12',{exact:true}).waitFor();
 // Sales-order Ship (the default mode): the hidden standalone reference must not block Prepare draft.
 await page.locator('#entry-action').selectOption('ship');assert.equal(await page.locator('[name=ship_mode]').inputValue(),'order');await choose('order','SO-D12-BROWSER');
 await page.locator('[data-shipment-line-id] input[type=number]').first().fill('1');await page.locator('[name=bol_reference]').fill('Empty-string order BOL');
 const orderShip='/sales/orders/'+fixture.order+'/ship/prepare';await prepare();assert.equal(posts(orderShip),1,'sales-order Ship sends one prepare request');
 assert.equal(requests.filter(r=>r.path===orderShip).at(-1).body.order_reference,undefined,'order mode sends no standalone reference');await noRawPydantic();
 await page.screenshot({path:out+'/ship-sales-order-draft-390.png',fullPage:true});await page.locator('#resolution').getByRole('button',{name:'Cancel',exact:true}).click();
 await page.locator('[name=ship_mode]').selectOption('standalone');await choose('customer','D12 Test Customer');await choose('ship_product','D1270');await page.locator('[name=quantity_lb]').fill('1');await page.locator('[name=bol_reference]').fill('Empty-string BOL');
 await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.getByText('Enter the shipment reference.',{exact:false}).first().waitFor();
 await page.locator('[name=order_reference]').fill('   ');await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.getByText('Enter the shipment reference.',{exact:false}).first().waitFor();
 assert.equal(posts('/ship/prepare'),0,'blank reference never sent');await noRawPydantic();await page.screenshot({path:out+'/ship-blank-reference-390.png',fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.deepEqual(errors,[]);
 console.log(`Form empty-string browser run passed: multi-lot Make ${made} (blank bag code never sent as \"\", plain blocker; blank optional fields omitted), Pack ${packed}, sales-order Ship draft prepared (1 request), standalone Ship blank reference refused by name — 390 px, real local FastAPI.`);
}catch(error){await page.screenshot({path:out+'/failure.png',fullPage:true});console.error('Visible result:',await page.locator('#notice').textContent().catch(()=>''));console.error('Page errors:',errors);throw error;}finally{await browser.close();}

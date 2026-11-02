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
async function record(){for(const box of await page.locator('#resolution [data-warning]').all())await box.check();await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();await page.locator('#resolution .good').waitFor();return page.locator('#resolution .good a').textContent();}
const rows=()=>page.locator('#resolution fieldset.lot-confirmation-row');
const codeOf=async row=>(await row.locator('legend').textContent()).split(' · ').at(-1);
async function check(){await page.locator('#resolution').getByRole('button',{name:'Check confirmations → new draft',exact:true}).click();}
async function recordable(){await page.waitForFunction(()=>{const b=Array.from(document.querySelectorAll('#resolution button')).find(b=>b.textContent==='Record');return b&&!b.disabled;});}
async function noRawPydantic(){assert.equal((await page.locator('body').textContent()).includes('String should have'),false,'raw Pydantic text shown');}
try{
 await page.goto(base+'/dashboard/entry.html');await page.locator('#actor-key').fill('d12-local-floor');await page.locator('#sign-in-form button[type=submit]').click();await page.getByText('Signed in as Arturo D12',{exact:true}).waitFor();
 // Make: 15 batches over two Oat Bran lots; output lot code and note left blank.
 await page.locator('#entry-action').selectOption('make');await choose('product','D1290');await page.locator('[name=batches]').fill('15');await page.locator('[name=confirmed_sku]').check();await prepare();
 const first=requests.filter(r=>r.path==='/make/prepare').at(-1).body;assert.equal('lot_code' in first,false,'blank output lot omitted');assert.equal('note' in first,false,'blank note omitted');
 assert.equal(await rows().count(),2,'two lot rows for the split ingredient');const [codeA,codeB]=[await codeOf(rows().nth(0)),await codeOf(rows().nth(1))];
 await rows().nth(0).locator('select').selectOption('full_code');await rows().nth(0).locator('input').fill(codeA);
 const before=posts('/make/prepare');await check();
 const message=page.locator('#resolution [data-blank-confirmation]');await message.waitFor();assert.equal(await message.count(),1);
 assert.equal(await message.textContent(),'Type the code from the lot label for D12 Oat Bran · '+codeB+'.');assert.equal(posts('/make/prepare'),before,'blank row never sent');await noRawPydantic();
 await page.screenshot({path:out+'/make-blank-lot-390.png',fullPage:true});
 await rows().nth(1).locator('select').selectOption('full_code');await rows().nth(1).locator('input').fill(codeB);await check();await recordable();
 const sent=requests.filter(r=>r.path==='/make/prepare').at(-1).body.lot_confirmations;assert.deepEqual(sent.map(c=>c.value),[codeA,codeB]);
 const made=await record();assert.match(made,/^[A-Z]+-\d{6}-\d+$/);
 // Pack: the same shared draft card.
 await page.getByRole('button',{name:'Record another entry',exact:true}).click();await page.locator('#entry-action').selectOption('pack');await choose('source','D1290');await choose('target','D1270');await page.locator('[name=cases]').fill('1');await prepare();
 const packBefore=posts('/pack/prepare');await check();await page.locator('#resolution [data-blank-confirmation]').first().waitFor();assert.equal(posts('/pack/prepare'),packBefore);await noRawPydantic();
 for(const row of await rows().all()){await row.locator('select').selectOption('full_code');await row.locator('input').fill(await codeOf(row));}await check();await recordable();const packed=await record();assert.match(packed,/^[A-Z]+-\d{6}-\d+$/);
 // Standalone ship: the reference FL requires is required in the form; spaces alone are refused by name.
 await page.getByRole('button',{name:'Sign out',exact:true}).click();await page.locator('#actor-key').fill('d12-local-owner');await page.locator('#sign-in-form button[type=submit]').click();await page.getByText('Signed in as Owner D12',{exact:true}).waitFor();
 await page.locator('#entry-action').selectOption('ship');await page.locator('[name=ship_mode]').selectOption('standalone');await choose('customer','D12 Test Customer');await choose('ship_product','D1270');await page.locator('[name=quantity_lb]').fill('1');await page.locator('[name=bol_reference]').fill('Empty-string BOL');
 assert.equal(await page.locator('[name=order_reference]').evaluate(input=>input.required),true);
 await page.getByRole('button',{name:'Prepare draft',exact:true}).click();assert.equal(await page.locator('[name=order_reference]').evaluate(input=>input.validity.valueMissing),true);
 await page.locator('[name=order_reference]').fill('   ');await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.getByText('Enter the shipment reference.',{exact:false}).first().waitFor();
 assert.equal(posts('/ship/prepare'),0,'blank reference never sent');await noRawPydantic();await page.screenshot({path:out+'/ship-blank-reference-390.png',fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);assert.deepEqual(errors,[]);
 console.log(`Form empty-string browser run passed: multi-lot Make ${made} (blank lot row blocked and named, blank optional fields omitted), Pack ${packed}, standalone Ship reference required — 390 px, real local FastAPI.`);
}catch(error){await page.screenshot({path:out+'/failure.png',fullPage:true});console.error('Visible result:',await page.locator('#notice').textContent().catch(()=>''));console.error('Page errors:',errors);throw error;}finally{await browser.close();}

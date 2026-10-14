// PR #107 review fix round 1: one real-route browser regression per fix (0–6). Local fixture only.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from 'playwright';
const base=process.env.D12_BASE||'http://127.0.0.1:8122';
assert.ok(['127.0.0.1','localhost'].includes(new URL(base).hostname),'Use local fixture only');
const fixture=JSON.parse(await fs.readFile('.venv-test/d12-fixture.json','utf8'));
const out='.venv-test/d12-visual';await fs.mkdir(out,{recursive:true});
async function api(path,body,key='d12-local-floor'){const response=await fetch(base+path,{method:body?'POST':'GET',headers:{'X-API-Key':key,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await response.json();assert.ok(response.ok,path+' '+JSON.stringify(data));return data;}
async function ticket(path,body,commit={}){const draft=await api(path,{...body,client_source:'dashboard'});return api('/tickets/'+draft.ticket+'/commit',{payload_hash:draft.payload_hash,acknowledged_warnings:draft.warnings.map(w=>w.code),...commit});}
const open=async kind=>(await api('/exceptions?status=open&limit=1000',undefined,'d12-local-owner')).exceptions.filter(e=>e.kind===kind);

// Seed through the real API: an unidentified lot, two held corrections, a SHORTAGE with a source receipt, a voided receipt.
await ticket('/receive/prepare',{product_id:fixture.products['Rolled Oats'],supplier_id:fixture.supplier,cases:1,case_size_lb:10,bol_reference:'D12 fixes',supplier_lot_code:'UNKNOWN'});
for(const delta of [501,502])assert.equal((await ticket('/adjust/prepare',{lot_id:fixture.lots['Oat Bran'].id,delta_lb:delta,reason_code:'physical_count'})).held,true);
const oat=fixture.lots['Oat Bran'];const make=await ticket('/make/prepare',{product_id:fixture.products['Test Batch'],batches:11,lot_confirmations:[{lot_id:oat.id,method:'last4',value:oat.code.slice(-4)}]});
const shortage=(await open('SHORTAGE')).find(e=>e.receipt_number===make.receipt_number);assert.ok(shortage,'make must open a SHORTAGE tied to its receipt');
const doomed=await ticket('/adjust/prepare',{lot_id:fixture.lots['Chips 4000 CT'].id,delta_lb:-1,reason_code:'damage_disposal'});
const voidResult=await api('/void/'+doomed.transaction_id,{reason:'D12 fix-round void'},'d12-local-master');

const browser=await chromium.launch({channel:'chrome',headless:true});const errors=[];
async function newPage(width){const context=await browser.newContext({viewport:{width,height:900}}),page=await context.newPage();page.requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(new URL(r.url()).origin===base)page.requests.push({path:new URL(r.url()).pathname,method:r.method(),body:r.postData(),session:r.headers()['x-api-key']?.startsWith('fls_')});});
 await page.route('**/*',r=>new URL(r.request().url()).origin===base?r.continue():r.abort());return page;}
async function signIn(page,role){await page.locator('#actor-key').fill('d12-local-'+role);await page.locator('#sign-in-form button').click();await page.locator('#identity').filter({hasText:/^(Signed in as|Sesión de) /}).waitFor();}
const section=(page,kind)=>page.locator('section.panel').filter({has:page.getByRole('heading',{name:new RegExp('^'+kind+' ')})});
async function resolve(page,kind,id){const s=section(page,kind);await s.waitFor();const row=id?s.locator('[data-exception-id="'+id+'"]'):s.locator('.exception-row').first();const selectedId=await row.getAttribute('data-exception-id');await row.getByRole('button').click();await page.locator('#resolution h2',{hasText:new RegExp(' '+kind+' #'+selectedId+'$')}).waitFor();}
const noGreen=async page=>assert.equal(await page.locator('.good').count(),0,'no green state');
try{
 // Fix 2 + fix 0 + fix 1 — floor, 390 px, English.
 {const page=await newPage(390);await page.goto(base+'/dashboard/exceptions.html');await signIn(page,'floor');
  await section(page,'LARGE_CORRECTION').waitFor();assert.equal(await section(page,'LARGE_CORRECTION').getByRole('button').count(),0,'floor has no actions on owner holds');
  await resolve(page,'UNIDENTIFIED_LOT');const options=await page.locator('#resolution select').first().locator('option').allTextContents();
  assert.deepEqual(options,['Identify supplier lot'],'fix 2: floor sees no owner write-off');
  // Fix 0: the live A5 route, ticket commit, then FL's resolved_exception_ids.
  const unidentified=(await open('UNIDENTIFIED_LOT'))[0];
  await page.locator('[name=supplier_lot_code]').fill('SUP-FIX-001');await page.locator('[name=note]').fill('Read from supplier tag');await page.getByRole('button',{name:'Review resolution'}).click();
  await page.getByRole('heading',{name:'Draft — review before recording'}).waitFor();await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();
  const recorded=page.locator('#resolution .good');await recorded.waitFor();assert.match(await recorded.textContent(),/^Recorded — receipt LOT-\d{6}-\d{3}$/);
  const resolvedNote=page.locator('#resolution [data-resolved]');await resolvedNote.waitFor();assert.equal(await resolvedNote.getAttribute('data-resolved'),String(unidentified.id));assert.match(await resolvedNote.textContent(),new RegExp('Resolved exception #'+unidentified.id));
  assert.ok(page.requests.some(r=>r.path==='/lots/'+unidentified.lot.id+'/supplier-lot/prepare'&&r.method==='POST'));assert.ok(!page.requests.some(r=>/\/exceptions\/\d+\/resolve$/.test(r.path)),'fix 0: no second resolve call');
  assert.equal((await api('/exceptions/'+unidentified.id,undefined,'d12-local-owner')).status,'resolved');
  // Counted resolution records its own XR and ADJ, never the source make receipt.
  await resolve(page,'SHORTAGE',shortage.id);assert.deepEqual(await page.locator('#resolution select').first().locator('option').evaluateAll(opts=>opts.map(o=>o.value)),['counted','missing_movement','voided']);await page.locator('#resolution select').first().selectOption('counted');assert.equal(await page.locator('[name=note]').evaluate(n=>n.required),true);
  await page.locator('[name=counted_lb]').fill('0');await page.locator('[name=note]').fill('Counted the lot: empty');await page.getByRole('button',{name:'Review resolution'}).click();
  const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/exceptions/'+shortage.id+'/resolve'&&r.request().method()==='POST');await page.getByRole('button',{name:'Confirm resolution'}).click();const resolved=await (await response).json();
  const notice=page.locator('#notice.good');await notice.waitFor();assert.match(resolved.resolution_receipt_number,/^XR-/);assert.equal(await notice.locator('a').first().textContent(),resolved.resolution_receipt_number);assert.ok(!(await notice.textContent()).includes(shortage.receipt_number));
  assert.equal(resolved.status,'resolved');assert.ok(resolved.resolution_posting_receipt_number.startsWith('ADJ-'));assert.equal(await notice.locator('a').nth(1).textContent(),resolved.resolution_posting_receipt_number);
  await page.locator('[data-exception-id="'+shortage.id+'"]').waitFor({state:'detached'});
  const decisionRequest=page.requests.find(r=>r.path==='/exceptions/'+shortage.id+'/resolve');assert.equal(decisionRequest.method,'POST');assert.ok(decisionRequest.session);assert.deepEqual(JSON.parse(decisionRequest.body),{resolution_kind:'counted',note:'Counted the lot: empty',counted_lb:0});
  await notice.locator('a').first().click();await page.getByRole('heading',{name:resolved.resolution_receipt_number,exact:true}).waitFor();await page.locator('#content [data-state=recorded]').waitFor();
  await page.screenshot({path:out+'/fix-floor-exceptions-390.png',fullPage:true});await page.context().close();}

 // Fix 3 — owner, desktop, Spanish: approve with a blank note records; reject without a note is blocked by the form.
 {const page=await newPage(1280);await page.goto(base+'/dashboard/exceptions.html');await signIn(page,'owner');await page.locator('#language').click();
  await resolve(page,'LARGE_CORRECTION');assert.deepEqual(await page.locator('#resolution select').first().locator('option').allTextContents(),['Aprobar como propietario','Rechazar como propietario']);
  await page.locator('#resolution select').first().selectOption('declined');await page.getByRole('button',{name:'Revisar resolución'}).click();
  assert.equal(await page.locator('[name=note]:invalid').count(),1,'fix 3: reject requires a note');assert.equal(await page.getByRole('button',{name:'Confirmar resolución'}).count(),0);
  await page.locator('#resolution select').first().selectOption('approved');assert.equal(await page.locator('[name=note]').evaluate(n=>n.required),false);await page.getByText('Nota (opcional)').waitFor();
  await page.getByRole('button',{name:'Revisar resolución'}).click();await page.getByRole('button',{name:'Confirmar resolución'}).click();
  const good=page.locator('#notice.good');await good.waitFor();assert.match(await good.textContent(),/^Decisión registrada — recibo XR-/);assert.match(await good.locator('a').nth(1).textContent(),/^ADJ-/);
  const approve=page.requests.find(r=>/\/exceptions\/\d+\/approve$/.test(r.path));assert.equal(approve.body,'{}','fix 3: blank note omitted');await page.locator('[data-exception-id="'+approve.path.split('/')[2]+'"]').waitFor({state:'detached'});
  await page.screenshot({path:out+'/fix-owner-approve-es-1280.png',fullPage:true});
  await resolve(page,'LARGE_CORRECTION');await page.locator('#resolution select').first().selectOption('declined');await page.locator('[name=note]').fill('Reject synthetic incorrect count');await page.getByRole('button',{name:'Revisar resolución'}).click();
  const response=page.waitForResponse(r=>/\/exceptions\/\d+\/reject$/.test(new URL(r.url()).pathname));await page.getByRole('button',{name:'Confirmar resolución'}).click();const rejected=await (await response).json();await page.locator('#notice.good').waitFor();assert.equal(await page.locator('#notice a').first().textContent(),rejected.resolution_receipt_number);assert.match(rejected.resolution_receipt_number,/^XR-/);assert.equal(rejected.posted,false);assert.equal(await page.locator('#notice a').count(),1);await page.getByText('Rechazada; no se registró movimiento de inventario.',{exact:true}).waitFor();await page.context().close();}

 // Fix 4 — a voided receipt shows VOIDED, never green, in both languages.
 {const page=await newPage(390);await page.goto(base+'/dashboard/receipt.html?number='+doomed.receipt_number);await signIn(page,'floor');const state=page.locator('#content [data-state]');await state.waitFor();
  assert.equal(await state.getAttribute('data-state'),'voided');assert.equal(await state.getAttribute('class'),'voided');assert.match(await state.textContent(),/^VOIDED — /);await noGreen(page);
  const backlink=page.locator('#content a').filter({hasText:voidResult.receipt_number});await backlink.waitFor();assert.match(await backlink.getAttribute('href'),new RegExp('number='+voidResult.receipt_number));await backlink.click();await page.getByRole('heading',{name:voidResult.receipt_number,exact:true}).waitFor();await page.goBack();await state.waitFor();
  await page.locator('#language').click();await page.getByText(/^ANULADO — /).waitFor();await noGreen(page);
  await page.goto(base+'/dashboard/receipt.html?number='+make.receipt_number);await page.locator('#content [data-state=recorded].good').waitFor();
  await page.screenshot({path:out+'/fix-voided-receipt-390.png',fullPage:true});await page.context().close();}

 // Fix 6 then fix 5 — entry form, 390 px.
 {const page=await newPage(390);await page.goto(base+'/dashboard/entry.html?action=adjust');await signIn(page,'floor');
  const picker=page.locator('[data-picker=lot]');await picker.locator('input').fill(fixture.lots['Rolled Oats'].code);await picker.getByRole('button',{name:'Search',exact:true}).click();await picker.locator('.picker-options button').first().click();
  await page.locator('[name=delta_lb]').fill('-1');await page.locator('[name=reason_code]').selectOption('damage_disposal');await page.locator('[name=note]').fill('half-filled');
  await page.locator('#language').click();await page.getByText('Lote a ajustar',{exact:true}).waitFor();
  assert.equal(await page.locator('[name=delta_lb]').inputValue(),'-1','fix 6: values survive the switch');assert.equal(await page.locator('[name=note]').inputValue(),'half-filled');assert.equal(await page.locator('[name=reason_code]').inputValue(),'damage_disposal');
  assert.equal(await page.locator('[name=reason_code] option:checked').textContent(),'Daño o desecho');assert.notEqual(await picker.locator('input').inputValue(),'');
  await page.getByRole('button',{name:'Preparar borrador',exact:true}).click();await page.getByRole('heading',{name:'Borrador — revisa antes de registrar'}).waitFor();
  await page.locator('#language').click();await page.getByRole('heading',{name:'Draft — review before recording'}).waitFor();await page.locator('#resolution dt',{hasText:'Change (lb)'}).first().waitFor();
  assert.equal(await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).count(),1,'fix 6: open draft survives the switch');
  // Fix 5: FL commits but answers after the 20 s timeout; the same-ticket retry replays the receipt and the red banner goes.
  await page.route('**/tickets/*/commit',async r=>{const response=await r.fetch();await new Promise(z=>setTimeout(z,21000));try{await r.fulfill({response});}catch{}},{times:1});
  await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();const bad=page.locator('#notice.bad');await bad.waitFor({timeout:30000});const message=await bad.textContent();
  assert.equal(message,'No answer from FL within 20 seconds — this is NOT confirmed. Press Record again to check: it reuses the same ticket and cannot record twice.');assert.ok(!/signal|abort/i.test(message));await noGreen(page);
  await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();await page.locator('#resolution .good').waitFor();
  assert.equal(await page.locator('#notice').isHidden(),true,'fix 5: red banner cleared after the successful retry');
  assert.equal(page.requests.filter(r=>/\/tickets\/.+\/commit$/.test(r.path)).length,2);
  await page.screenshot({path:out+'/fix-entry-retry-390.png',fullPage:true});await page.context().close();}
 assert.deepEqual(errors,[]);console.log('D12 fix round passed: 0 supplier-lot round trip, 1 XR resolve/reject receipt links, 2 server-only allowed actions, 3 optional approve note, 4 VOIDED receipt, 5 plain timeout + cleared banner, 6 EN/ES keeps form and draft.');
}finally{await browser.close();}

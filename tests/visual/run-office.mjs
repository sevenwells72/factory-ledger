import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from 'playwright';
import office from '../../dashboard/office-logic.js';
const base=process.env.D12_BASE||'http://127.0.0.1:8122',on=process.env.D12_GATE_ON!=='0';
assert.ok(['localhost','127.0.0.1'].includes(new URL(base).hostname));
const fixture=JSON.parse(await fs.readFile('.venv-test/d12-fixture.json','utf8')),out='.venv-test/office-visual';await fs.mkdir(out,{recursive:true});
const browser=await chromiumium();
async function chromiumium(){return chromium.launch({channel:'chrome',headless:true});}
const page=await browser.newPage({viewport:{width:390,height:844}}),calls=[],errors=[],receipts={};
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>calls.push({path:new URL(r.url()).pathname,method:r.method(),body:r.headers()['content-type']?.includes('application/json')?r.postDataJSON():null}));
await page.route('**/*',r=>new URL(r.request().url()).origin===base?r.continue():r.abort());
async function api(path){const response=await fetch(base+path,{headers:{'X-API-Key':'d12-local-owner'}});assert.ok(response.ok,path);return response.json();}
async function choose(name,query,scope=page){const box=scope.locator('[data-picker="'+name+'"]');await box.locator('input').fill(query);await box.getByRole('button',{name:'Search',exact:true}).click();await box.locator('.picker-options button').first().click();}
async function signIn(role){if(await page.locator('#sign-out').isVisible())await page.locator('#sign-out').click();await page.locator('#actor-key').fill('d12-local-'+role);await page.locator('#sign-in-form button').click();await page.locator('#office-action').waitFor();}
async function action(value){await page.locator('#office-action').selectOption(value);await page.locator('#office-form').waitFor();}
async function prepare(){const before=calls.filter(c=>c.path.endsWith('/commit')).length;await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();assert.equal(calls.filter(c=>c.path.endsWith('/commit')).length,before);}
async function record(name){for(const box of await page.locator('#resolution [data-warning]').all())await box.check();await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();await page.locator('#resolution .good a').waitFor();const receipt=await page.locator('#resolution .good a').textContent();receipts[name]=receipt;return api('/receipts/'+receipt);}
async function line(product,quantity,unit,price=''){const row=page.locator('[data-lines]>fieldset').last();await choose('product',product,row);await row.locator('[name=quantity]').fill(quantity);await row.locator('[name=unit]').selectOption(unit);if(price!=='')await row.locator('[name=unit_price]').fill(price);}
async function selectOrder(id){const order=await api('/sales/orders/'+id);const loaded=page.waitForResponse(r=>r.url().endsWith('/office-options'));await choose('order',order.order_number);await loaded;}
async function newOrder(po,duplicate=false){await action('create_order');await choose('customer','D12 Test Customer');await page.locator('[name=customer_po]').fill(po);
 await line('D2POUCH','24','pouches','30');await page.getByRole('button',{name:'Add line',exact:true}).click();await line('D2PALLET','1','each','12.50');
 await page.getByRole('button',{name:'ES',exact:true}).click();assert.equal(await page.locator('[name=customer_po]').inputValue(),po);await page.getByRole('button',{name:'EN',exact:true}).click();
 await prepare();assert.ok(calls.some(c=>c.path==='/resolve'&&c.body?.kind==='unit'&&c.body.query==='pouches'));
 if(duplicate){await page.locator('#resolution [data-warning=DUPLICATE_CUSTOMER_PO]').waitFor();await page.getByRole('button',{name:'ES',exact:true}).click();await page.locator('#resolution').getByRole('button',{name:'Registrar',exact:true}).click();await page.locator('#notice.bad').waitFor();assert.equal(await page.locator('#notice').textContent(),'NO registrado — Confirme las advertencias del borrador antes de registrar.');assert.equal(await page.locator('#resolution .good').count(),0);await page.getByRole('button',{name:'EN',exact:true}).click();}
 await page.screenshot({path:out+'/order-draft-390.png',fullPage:true});
 return record(duplicate?'duplicate':'order');
}

async function missingCustomer(name,similar=''){
 const picker=page.locator('[data-picker=customer]');await picker.locator('input').fill(name);await picker.getByRole('button',{name:'Search',exact:true}).click();if(similar)await picker.getByRole('button',{name:similar,exact:true}).waitFor();await picker.getByRole('button',{name:"Add '"+name+"' as a new customer",exact:true}).click();
 return page.locator('[data-customer-editor]');
}
async function similarCustomerThenOrder(){
 await action('create_order');const name='Sunrise Bakery',po='000NEW-CUSTOMER';
 await page.locator('[name=customer_po]').fill(po);await page.locator('[name=requested_ship_date]').fill('2026-10-30');await page.locator('[name=notes]').fill('Keep this order note');await line('D1270','2','cases','20');
 const editor=await missingCustomer(name,'Sunshine Bakery');await editor.locator('[name=contact_name]').fill('Pat New');await editor.locator('[name=email]').fill('pat@example.test');await editor.locator('[name=phone]').fill('555-0123');await editor.locator('[name=address]').fill('1 Example Road');await editor.locator('[name=notes]').fill('Call before delivery');
 await page.getByRole('button',{name:'ES',exact:true}).click();assert.equal(await editor.locator('[name=name]').inputValue(),name);await page.getByRole('button',{name:'EN',exact:true}).click();
 const before=calls.filter(c=>c.path.endsWith('/commit')).length;await editor.getByRole('button',{name:'Prepare customer',exact:true}).click();
 const customerDraft=page.locator('[data-customer-draft]');await customerDraft.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();assert.equal(calls.filter(c=>c.path.endsWith('/commit')).length,before);
 assert.equal(await page.locator('#office-form').getByRole('button',{name:'Prepare draft',exact:true}).isDisabled(),true);assert.equal(await page.locator('[data-picker=customer] input').isDisabled(),true);assert.equal(await page.locator('[data-picker=customer] [data-new-customer]').isDisabled(),true);
 await page.screenshot({path:out+'/new-customer-draft-390.png',fullPage:true});
 await customerDraft.getByRole('button',{name:'Record',exact:true}).click();await customerDraft.locator('.good a').waitFor();const number=await customerDraft.locator('.good a').textContent();assert.match(number,/^CUS-/);receipts.newCustomer=number;
 const customerReceipt=await api('/receipts/'+number);assert.equal(customerReceipt.action,'create_customer');assert.equal(customerReceipt.response.customer.name,name);
 assert.equal(await page.locator('[data-picker=customer] input').inputValue(),name);assert.equal(await page.locator('[name=customer_po]').inputValue(),po);assert.equal(await page.locator('[name=requested_ship_date]').inputValue(),'2026-10-30');assert.equal(await page.locator('[name=notes]').inputValue(),'Keep this order note');assert.equal(await page.locator('[name=quantity]').inputValue(),'2');
 await page.screenshot({path:out+'/new-customer-receipt-390.png',fullPage:true});
 await prepare();const order=await record('newCustomerOrder');assert.equal(order.response.customer_id,customerReceipt.response.customer_id);assert.equal(order.response.customer_po,po);assert.equal(order.response.customer,name);
 const customerRequest=calls.filter(c=>c.path==='/customers/prepare').at(-1);assert.equal(customerRequest.body.name,name);assert.equal(customerRequest.body.contact_name,'Pat New');assert.equal(customerRequest.body.notes,'Call before delivery');
}
async function duplicateCustomer(){
 await action('create_order');await page.locator('[name=customer_po]').fill('000KEEP');const editor=await missingCustomer('ZXQWV 576291 Missing');await editor.locator('[name=name]').fill('  d12   test customer  ');
 const before=calls.filter(c=>c.path.endsWith('/commit')).length;await editor.getByRole('button',{name:'Prepare customer',exact:true}).click();const customerDraft=page.locator('[data-customer-draft]');
 await page.getByRole('button',{name:'Use existing customer: D12 Test Customer',exact:true}).waitFor();assert.equal(await customerDraft.getByRole('button',{name:'Record',exact:true}).isDisabled(),true);assert.equal(calls.filter(c=>c.path.endsWith('/commit')).length,before);
 await page.getByRole('button',{name:'Use existing customer: D12 Test Customer',exact:true}).click();assert.equal(await page.locator('[data-picker=customer] input').inputValue(),'D12 Test Customer');assert.equal(await page.locator('[name=customer_po]').inputValue(),'000KEEP');assert.equal(await page.getByRole('button',{name:'Prepare draft',exact:true}).isDisabled(),false);
}

async function ordersRow(id){await page.goto(base+'/dashboard/index.html?section=orders');await page.evaluate(()=>FLDashboard.ready);const row=page.locator('.order-row[data-order-id="'+id+'"]');await row.waitFor();return row;}
async function readyChanges(id){
 const initialMark=(await api('/sales/orders?state=open&limit=200')).orders.find(order=>order.order_id===id),markedReady=!initialMark.ready;
 for(const [kind,ready,note] of [['checkbox',markedReady,initialMark.note||''],['note',markedReady,'Pallet 3 reviewed'],['detail',!markedReady,'Pallet 3 reviewed']]){
  const row=await ordersRow(id),before=calls.filter(c=>c.path.endsWith('/commit')).length;
  const legacyResponse=!on?page.waitForResponse(r=>/^\/sales-orders\/[^/]+\/ready$/.test(new URL(r.url()).pathname)&&r.request().method()==='POST'):null;
  if(kind==='checkbox')await row.locator('.order-ready-checkbox').click();
  else{await row.locator('.order-expand-toggle').click();if(kind==='note'){await page.locator('#ready-note-'+id).fill(note);await page.locator('.order-ready-note-save[data-order-id="'+id+'"]').click();}else{await page.locator('.so-open-detail[data-order-id="'+id+'"]').click();await page.locator('.so-detail-ready-checkbox').click();}}
  if(on){
   await page.waitForURL('**/office.html?*');await page.locator('#office-action').waitFor();assert.equal(await page.locator('#office-action').inputValue(),'mark_order_ready');
   await page.waitForFunction(({ready,note})=>document.querySelector('[name=ready]')?.checked===ready&&document.querySelector('[name=note]')?.value===note,{ready,note});
   assert.equal(calls.filter(c=>c.path.endsWith('/commit')).length,before,'Changing a control only opens a draft');
   await prepare();const receipt=await record('ready-'+kind);assert.equal(receipt.action,'mark_order_ready');assert.equal(receipt.response.ready,ready);assert.equal(receipt.response.note,note||null);
  }else{
   assert.equal((await legacyResponse).status(),200);
   await page.waitForFunction(({id,ready,kind})=>{const control=document.querySelector(kind==='detail'?'.so-detail-ready-checkbox':'.order-row[data-order-id="'+id+'"] .order-ready-checkbox');return control?.checked===ready&&!control.disabled;},{id,ready,kind});
   const mark=(await api('/sales/orders?state=open&limit=200')).orders.find(o=>o.order_id===id);assert.equal(mark.ready,ready);assert.equal(mark.note,note||null);assert.ok(!page.url().includes('office.html'));
  }
 }
}
try{
 await page.goto(base+'/dashboard/office.html');assert.equal(await page.evaluate(()=>FLDashboard.ready),on);
 if(!on){
   for(const [key] of office.actions){await page.goto(base+'/dashboard/office.html?action='+key);await page.evaluate(()=>FLDashboard.ready);assert.equal(await page.locator('#office-form,#office-action').count(),0);assert.equal(await page.locator('[data-fl-write-ui]:not(html):visible').count(),0);}
   assert.equal(calls.some(c=>c.method==='POST'),false);assert.equal(calls.some(c=>c.path==='/auth/whoami'),false);
   await readyChanges(fixture.order);assert.equal(calls.filter(c=>c.method==='POST').length,3);assert.ok(calls.filter(c=>c.method==='POST').every(c=>/^\/sales-orders\/[^/]+\/ready$/.test(c.path)));
 }else{
   await signIn('office');
   await similarCustomerThenOrder();await duplicateCustomer();
   const created=await newOrder('000D2-PO');const id=created.order.id;
   assert.equal(created.response.customer_po,'000D2-PO');assert.equal(created.response.total_lb,15);
   assert.equal(created.response.lines.find(l=>l.product_id===fixture.pallet).quantity_lb,0);
   await newOrder('000D2-PO',true);
   await action('create_order');await choose('customer','D12 Test Customer');await page.locator('[name=no_po]').check();await line('D1270','1','cases');await prepare();await page.locator('[data-warning=NO_PO]').waitFor();await record('noPO');
   await action('add_order_lines');await selectOrder(id);await line('D1270','1','cases');await prepare();const added=await record('add');
   const lineId=added.response.lines_added[0].line_id;
   await action('update_order_line');await selectOrder(id);await page.locator('[name=line_id]').selectOption(String(lineId));await page.locator('[name=quantity_lb]').fill('6');await page.locator('[name=unit_price]').fill('0');await prepare();assert.equal(await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).isDisabled(),true);await page.getByRole('button',{name:'Edit entry',exact:true}).click();await page.locator('[name=quantity_lb]').fill('10');await prepare();await record('lineUpdate');
   await action('cancel_order_line');await selectOrder(id);await page.locator('[name=line_id]').selectOption(String(lineId));await prepare();await record('lineCancel');
   await action('update_order_header');await selectOrder(id);await page.locator('[name=customer_po]').fill('000UPDATED');await page.locator('[name=notes]').fill('Pilot office edit');await page.locator('[name=notes_es]').fill('Edición de oficina');await prepare();await record('header');
   await action('update_order_status');await selectOrder(id);await page.locator('[name=status]').selectOption('in_production');await prepare();await record('status');
   await readyChanges(id);await signIn('floor');assert.deepEqual(await page.locator('#office-action option').evaluateAll(options=>options.map(o=>o.value)),['mark_order_ready']);await selectOrder(id);await page.waitForFunction(()=>document.querySelector('[name=note]')?.value==='Pallet 3 reviewed');await signIn('office');
   await action('close_order');await selectOrder(id);assert.equal(await page.locator('[name=reason] option[value=shipped_not_recorded]').count(),0);
   await signIn('owner');await action('close_order');await selectOrder(id);await page.locator('[name=reason]').selectOption('shipped_not_recorded');await page.locator('[name=note]').fill('Local legacy close fixture');await prepare();await record('close');
   await action('reopen_order');await selectOrder(id);await page.locator('[name=note]').fill('Resume local fixture');await prepare();await record('reopen');
   await action('cancel_order');await selectOrder(id);await page.locator('[name=reason]').selectOption('customer_cancelled');await page.locator('[name=note]').fill('Local fixture cancelled');await prepare();await record('cancel');
   await action('create_expected_receipt');await choose('supplier','D12 Test Supplier');await choose('product','D1201');await page.locator('[name=expected_qty]').fill('40');await page.locator('[name=reference_number]').fill('000ER');await prepare();const er=await record('erCreate'),erId=er.response.expected_receipt_id;
   for(const kind of ['update_expected_receipt','cancel_expected_receipt']){await action(kind);await page.locator('[name=expected_receipt_id]').fill(String(erId));const loaded=page.waitForResponse(r=>r.url().endsWith('/expected-receipts/'+erId));await page.getByRole('button',{name:'Load expected delivery',exact:true}).click();await loaded;if(kind==='update_expected_receipt'){await page.locator('[name=expected_qty]').fill('45');await page.locator('[name=reference_number]').fill('');}await prepare();await record(kind);}
   for(const [kind,docKind,party,query] of [['approve_so','sales','customer','D12 Test Customer'],['approve_er','purchase','supplier','D12 Test Supplier']]){
     await action(kind);await page.locator('[name=document_id]').fill(String(fixture.documents[docKind]));await page.getByRole('button',{name:'Load extracted document',exact:true}).click();await page.getByText(docKind+'.png',{exact:true}).waitFor();await choose(party,query);await choose('product','D1270',page.locator('[data-lines]>fieldset').first());await prepare();await record(kind);
   }
   // Each successful form used a ticket's own receipt; no direct office write route.
   assert.equal(Object.keys(receipts).length,21);
   const writes=calls.filter(c=>c.method!=='GET'&&!c.path.startsWith('/auth/'));
   assert.ok(writes.every(c=>c.path==='/resolve'||c.path.endsWith('/prepare')||/^\/tickets\/[^/]+\/commit$/.test(c.path)),JSON.stringify(writes.map(c=>c.path)));
   await action('create_order');await choose('customer','D12 Test Customer');await line('D2POUCH','13','pouches');await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.locator('#notice.bad').waitFor();assert.equal(await page.locator('#resolution .good').count(),0);
   await page.getByRole('button',{name:'ES',exact:true}).click();assert.equal(await page.locator('[name=quantity]').inputValue(),'13');
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   await page.screenshot({path:out+'/office-es-390.png',fullPage:true});
 }
 await page.evaluate(()=>sessionStorage.setItem('fl-ledger-language','en'));
 await page.goto(base+'/dashboard/receipt.html?number=RCV-000000-999');await page.locator('#notice.neutral').waitFor();assert.equal(await page.locator('#notice').textContent(),'No receipt with this number');assert.equal(await page.locator('#notice.bad').count(),0);
 await page.locator('#language').click();assert.equal(await page.locator('#notice').textContent(),'No hay recibo con este número');
 assert.deepEqual(errors,[]);await fs.writeFile(out+'/receipts.json',JSON.stringify(receipts,null,2));console.log('Office '+(on?'all 15 forms, Sunrise Bakery added beside Sunshine suggestion then order with CUS/ORD receipts, normalized duplicate choice, ready checkbox/note/detail tickets, floor permission, duplicate/No PO, pouch rules, service line, document tickets, EN/ES and 390px':'all ticket forms hidden, legacy ready checkbox/note/detail writes')+'; neutral missing receipt passed.');
}catch(error){await page.screenshot({path:out+'/failure.png',fullPage:true});if(await page.locator('#notice').count())console.error('Visible notice:',await page.locator('#notice').textContent());throw error;}finally{await browser.close();}

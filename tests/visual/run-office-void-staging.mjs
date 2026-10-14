import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {chromium} from 'playwright';
const f=JSON.parse(await fs.readFile('.venv-test/office-void-staging-fixture.json','utf8')),base=f.base;
assert.equal(base,'http://127.0.0.1:8130');assert.ok(f.marker.startsWith('STG-D2OFF-'));
const out='docs/validation/office-void-staging';await fs.mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'chrome',headless:true}),page=await browser.newPage({viewport:{width:390,height:844}});
page.setDefaultTimeout(60000);const results=[],requests=[],errors=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{const p=new URL(r.url()).pathname;if(r.method()==='POST'&&!p.startsWith('/auth/')){
 assert.ok(r.headers()['x-api-key']?.startsWith('fls_'));requests.push(p.replace(/\/tickets\/[^/]+\//,'/tickets/{ticket}/'));
}});
await page.route('**/*',r=>new URL(r.request().url()).origin===base?r.continue():r.abort());
async function signIn(role){
 const key=process.env['D2_STAGING_'+role.toUpperCase()+'_KEY'];assert.ok(key);
 await page.locator('#actor-key').fill(key);await page.locator('#sign-in-form button').click();
 await page.getByText('Signed in as '+f.marker+' '+role,{exact:true}).waitFor();
 assert.equal(await page.locator('#actor-key').inputValue(),'');
 assert.equal(await page.evaluate(k=>JSON.stringify(sessionStorage).includes(k),key),false);
}
async function choose(name,query,scope=page){
 const p=scope.locator('[data-picker="'+name+'"]');await p.locator('input').fill(query);
 await p.getByRole('button',{name:'Search',exact:true}).click();await p.locator('.picker-options button').first().click();
}
async function prepare(){await page.getByRole('button',{name:'Prepare draft',exact:true}).click();await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();}
async function record(action){
 for(const b of await page.locator('#resolution [data-warning]').all())await b.check();
 const response=page.waitForResponse(r=>/\/tickets\/[^/]+\/commit$/.test(new URL(r.url()).pathname)&&r.request().method()==='POST');
 await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();
 const r=await response,data=await r.json();await page.locator('#resolution .good a').waitFor();
 results.push({action,receipt_number:data.receipt_number,ticket_id:data.ticket_id,order_id:data.order_id,transaction_id:data.transaction_id,lot_id:data.lot_id});
 const replay=await page.evaluate(async({url,body})=>FLLedger.request(new URL(url).pathname,body),{url:r.url(),body:r.request().postDataJSON()});
 assert.equal(replay.receipt_number,data.receipt_number);assert.equal(replay.replayed,true);return data;
}
async function line(label,quantity,unit,price){
 const row=page.locator('[data-lines]>fieldset').last();await choose('product',f.products[label].odoo_code,row);
 await row.locator('[name=quantity]').fill(quantity);await row.locator('[name=unit]').selectOption(unit);await row.locator('[name=unit_price]').fill(price);
}
async function signOut(){await page.locator('#sign-out').click();await page.locator('#sign-in-form').waitFor();assert.equal(await page.evaluate(()=>FLLedger.session.token),'');}
try{
 await page.goto(base+'/dashboard/office.html');await signIn('office');await page.locator('#office-action').waitFor();
 await choose('customer',f.customer.name);await page.locator('[name=customer_po]').fill('000-'+f.marker);
 await line('Cases','2','cases','30');await page.getByRole('button',{name:'Add line',exact:true}).click();await line('Pallet Charge','3','each','12.50');
 await prepare();await page.screenshot({path:out+'/order-draft-390.png',fullPage:true});const order=await record('create_order');
 assert.equal(order.customer_po,'000-'+f.marker);assert.equal(order.total_lb,50);
 assert.equal(Number(order.lines.reduce((n,l)=>n+Number(l.amount),0)),97.5);
 assert.equal(order.lines.find(l=>l.product_id===f.products['Pallet Charge'].id).quantity_lb,0);
 await page.screenshot({path:out+'/order-recorded-390.png',fullPage:true});await signOut();
 await page.goto(base+'/dashboard/entry.html');await signIn('floor');
 await choose('product',f.products.Ingredient.odoo_code);await choose('supplier',f.supplier.name);
 await page.locator('[name=cases]').fill('1');await page.locator('[name=case_size_lb]').fill('10');
 await page.locator('[name=bol_reference]').fill(f.marker+'-BOL');await page.locator('[name=supplier_lot_code]').fill(f.marker+'-SUP');
 await prepare();const received=await record('receive');
 await page.locator('#resolution .good a').click();await page.locator('[data-void="'+received.transaction_id+'"]').click();
 await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();
 assert.ok((await page.locator('#resolution').textContent()).includes('data_entry_error'));
 await page.screenshot({path:out+'/void-draft-390.png',fullPage:true});
 const vd=await record('void');assert.match(vd.receipt_number,/^VD-/);assert.equal(vd.source_receipt_number,received.receipt_number);
 await page.locator('[data-state=voided]').waitFor();assert.equal(await page.locator('[data-void]').count(),0);
 assert.ok(await page.locator('#content a').filter({hasText:vd.receipt_number}).count());
 await page.screenshot({path:out+'/voided-original-390.png',fullPage:true});
 await page.locator('#language').click();assert.match(await page.locator('[data-state=voided]').textContent(),/^ANULADO/);
 await page.screenshot({path:out+'/voided-original-es-390.png',fullPage:true});await page.locator('#language').click();
 await page.locator('#resolution .good a').click();await page.locator('[data-state=recorded]').waitFor();
 assert.ok((await page.locator('#content').textContent()).includes(received.receipt_number));
 await page.screenshot({path:out+'/void-receipt-390.png',fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 assert.deepEqual(errors,[]);await signOut();
 assert.equal(requests.some(p=>!p.endsWith('/prepare')&&!p.endsWith('/commit')&&p!='/resolve'),false);
 await fs.writeFile(out+'/evidence.json',JSON.stringify({mode:'Branch UI and HTTP against guarded staging DB; lifespan off; no hosted deployment',
 staging_origin:f.staging_origin,marker:f.marker,checked_at:new Date().toISOString(),results,requests,
 checks:['Office personal key exchanged for session','PO leading zeroes retained','Two cases 50 lb plus three pallet charges: $97.50','ORD receipt + replay',
 'Floor own same-day receive','Fixed data_entry_error prepare/draft/Record','VD receipt + replay','Original VOIDED and VD backlink','VD source receipt',
 'EN/ES 390px no overflow','Temporary sessions signed out','No production access']},null,2)+'\n');
 console.log('Staging branch acceptance passed: '+results.map(r=>r.receipt_number).join(', '));
}catch(error){await page.screenshot({path:'.venv-test/office-void-staging-failure.png',fullPage:true});throw Error('Staging browser failed; local screenshot saved. Credentials suppressed.');}finally{await browser.close();}

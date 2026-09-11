// Browser contract tests for the list. All requests are intercepted; no live writes.
// node tests/visual/run-sales-orders-interactions.mjs [dashboard-root] [output-dir]
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { API_HOST, buildTokenTable, installApiStub, loadFixture } from './lib/stub.mjs';

const root=path.resolve(process.argv[2]||'dashboard');
const out=path.resolve(process.argv[3]||'docs/design/audit/screenshots/sales-orders-interactions');
await fs.mkdir(out,{recursive:true});
const server=await startStaticServer(root);
const browser=await chromium.launch();
const results=[];

async function installStatefulList(context){
 const tokens=buildTokenTable();
 const base={fail:[],overrides:{},status:{},unmatched:new Set()};
 await installApiStub(context,tokens,base);
 const {orders}=await loadFixture('sales-orders-list.json',tokens);
 const requests=[], writes=[];
 let failList=false, failNextExit=false;
 await context.route(`**://${API_HOST}/**`,async route=>{
  const request=route.request(),url=new URL(request.url()),p=url.pathname;
  if(p==='/sales/orders'&&request.method()==='GET'){
   requests.push({kind:'list',query:Object.fromEntries(url.searchParams)});
   if(failList)return route.fulfill({status:503,json:{error:'HTTP 503: database_connection failed',internal_trace:'raw_json_secret'}});
   const rows=orders.filter(o=>(!url.searchParams.has('state')||o.state===url.searchParams.get('state'))&&(!url.searchParams.has('fulfillment')||o.fulfillment===url.searchParams.get('fulfillment'))&&(!url.searchParams.has('overdue_only')||(o.state==='open'&&o.overdue&&o.fulfillment!=='shipped'))&&(!url.searchParams.has('customer')||o.customer.toLowerCase().includes(url.searchParams.get('customer').toLowerCase())));
   return route.fulfill({json:{orders:rows,count:rows.length}});
  }
  if(p==='/sales/orders/counts'){
   requests.push({kind:'counts'});
   const open=orders.filter(o=>o.state==='open');
   return route.fulfill({json:{open:open.length,ready_to_ship:open.filter(o=>o.ready).length,overdue:open.filter(o=>o.overdue&&o.fulfillment!=='shipped').length,shipped:open.filter(o=>o.fulfillment==='shipped').length,closed:orders.filter(o=>o.state==='closed').length,cancelled:orders.filter(o=>o.state==='cancelled').length}});
  }
  const exit=p.match(/^\/sales\/orders\/(\d+)\/(close|cancel|reopen)$/);
  if(exit&&request.method()==='POST'){
   const body=request.postDataJSON(),order=orders.find(o=>o.order_id===Number(exit[1])),action=exit[2];
   writes.push({action,id:Number(exit[1]),body});
   assert.equal(Object.hasOwn(body,'changed_by'),false,'The client must not invent attribution');
   if(failNextExit){failNextExit=false;return route.fulfill({status:503,json:{error:'HTTP 503: internal_database_error',trace:'raw_json_secret'}});}
   if(action==='cancel'&&order.fulfillment!=='unshipped')return route.fulfill({status:409,json:{detail:{error_code:'ORDER_ALREADY_SHIPPED',message:'POST /sales/orders/105/close short_closed',suggested_action:'close',suggested_reason:'short_closed'}}});
   const reservations=[{id:901,quantity_lb:1250},{id:902,quantity_lb:250}];
   if(body.mode==='preview')return route.fulfill({json:{mode:'preview',order_id:order.order_id,resulting_state:action==='close'?'closed':action==='cancel'?'cancelled':'open',reservations_to_release:action==='reopen'?[]:reservations}});
   assert.equal(body.mode,'commit');
   order.state=action==='close'?'closed':action==='cancel'?'cancelled':'open';
   order.state_reason=action==='reopen'?null:body.reason;
   return route.fulfill({json:{...order,mode:'commit',reservations_released:action==='reopen'?[]:reservations}});
  }
  const ready=p.match(/^\/sales-orders\/([^/]+)\/ready$/);
  if(ready&&request.method()==='POST'){
   const body=request.postDataJSON(),order=orders.find(o=>o.order_number===decodeURIComponent(ready[1]));
   writes.push({action:'ready',id:order.order_id,body});
   Object.assign(order,{ready:body.ready,ready_by:'floor',ready_at:tokens.get('{{NOW}}'),note:body.note});
   return route.fulfill({json:order});
  }
  const lookup=p.match(/^\/sales\/orders\/(SO-[^/]+)$/);
  if(lookup){const order=orders.find(o=>o.order_number===decodeURIComponent(lookup[1]));return route.fulfill({status:order?200:404,json:order||{error:'unknown order'}});}
  return route.fallback();
 });
 return {requests,writes,orders,setListFailure:value=>{failList=value;},setExitFailure:()=>{failNextExit=true;}};
}

async function loaded(page){
 await page.waitForFunction(()=>{const c=document.querySelector('#orders-table-container');return c&&c.getAttribute('aria-busy')!=='true'&&(c.querySelector('.order-row,.orders-empty')||document.querySelector('#orders-error')?.textContent.trim());});
}
async function switchTab(page,name){await page.getByRole('tab',{name:new RegExp(`^${name}\\b`)}).click();await loaded(page);}
async function ids(page){return page.locator('#orders-table-container .order-row').evaluateAll(rows=>rows.map(r=>Number(r.dataset.orderId)).sort((a,b)=>a-b));}
async function openAction(page,id,action){
 const expand=page.locator(`.order-row[data-order-id="${id}"] .order-expand-toggle`);
 if(await expand.getAttribute('aria-expanded')!=='true')await expand.click();
 await page.locator(`.so-exit-action[data-order-id="${id}"][data-action="${action}"]`).click();
 const dialog=page.getByRole('dialog');await dialog.waitFor({state:'visible'});return dialog;
}
async function checkExplanation(page,trigger,mode){
 const id=await trigger.getAttribute('data-explain');
 assert(id,'Explanation trigger requires data-explain');
 assert((await trigger.getAttribute('aria-describedby')||'').split(/\s+/).includes(id));
 const popover=page.locator(`[id="${id}"]`);
 if(mode==='tap')await trigger.tap();else if(mode==='focus')await trigger.focus();else await trigger.hover();
 await popover.waitFor({state:'visible'});
 assert((await popover.innerText()).trim().length>20,'Explanation must contain meaningful record text');
 assert.equal(await page.locator('[data-explain]').evaluateAll(nodes=>nodes.filter(n=>{const p=document.getElementById(n.dataset.explain);return p&&p.getBoundingClientRect().height>0&&getComputedStyle(p).visibility!=='hidden';}).length),1,'Only one explanation may be open');
 await page.keyboard.press('Escape');await popover.waitFor({state:'hidden'});
}

try{
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},hasTouch:width===390,isMobile:width===390,colorScheme:'light'});
  await context.addInitScript(()=>localStorage.setItem('dashboard-theme','light'));
  const api=await installStatefulList(context),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.origin,{waitUntil:'domcontentloaded'});
  await page.locator(width===390?'.mobile-nav > a[data-section="orders"]':'.tab[data-tab="orders"]').click();
  await loaded(page);
  assert.deepEqual(await ids(page),[101,102,103,104,105,106,107,108,109]);
  // The independent mini calendar also fetches /sales/orders during
  // startup. Compare list/count refreshes caused by list-tab interactions.
  const initialListReads=api.requests.filter(r=>r.kind==='list').length;
  const initialCountReads=api.requests.filter(r=>r.kind==='counts').length;
  for(const [name,count] of [['Open',9],['Ready to ship',4],['Overdue',3],['Shipped',1],['Closed',1],['Cancelled',1]])assert.match(await page.getByRole('tab',{name:new RegExp(`^${name}\\b`)}).innerText(),new RegExp(`\\b${count}\\b`));
  for(const [name,expected] of [['Ready to ship',[102,104,107,109]],['Overdue',[101,102,103]],['Shipped',[109]],['Closed',[110]],['Cancelled',[111]],['Open',[101,102,103,104,105,106,107,108,109]]]){
   await switchTab(page,name);assert.deepEqual(await ids(page),expected,`${name} membership`);
   const q=api.requests.filter(r=>r.kind==='list').at(-1).query;
   assert.equal(q.state,['Closed','Cancelled'].includes(name)?name.toLowerCase():'open');
   assert.equal(q.overdue_only,name==='Overdue'?'true':undefined);
  }
  assert.equal(api.requests.filter(r=>r.kind==='list').length-initialListReads,api.requests.filter(r=>r.kind==='counts').length-initialCountReads,'Counts refresh with every list-tab fetch');
  await page.locator('#orders-customer-search').fill('Wexford');
  await page.locator('#orders-customer-search').press('Enter');await loaded(page);
  await page.waitForFunction(()=>document.querySelectorAll('#orders-table-container .order-row').length===1);
  assert.deepEqual(await ids(page),[101]);
  await page.locator('#orders-customer-search').fill('');await page.locator('#orders-customer-search').press('Enter');await loaded(page);
  await page.waitForFunction(()=>document.querySelectorAll('#orders-table-container .order-row').length===9);
  assert.equal(await page.getByLabel('Hide ready to ship',{exact:true}).count(),0,'Ready membership is controlled by the Ready to ship tab');
  assert.equal(await page.locator('#section-orders > .section-header .table-tools').count(),1,'Sort and Resize belong in the table header');
  const orderTrigger=page.locator('.order-row[data-order-id="101"] > td:nth-child(2) [data-explain]').first();
  await checkExplanation(page,orderTrigger,width===390?'tap':'hover');
  if(width===1440)await checkExplanation(page,orderTrigger,'focus');
  const healthTrigger=page.locator('.order-row[data-order-id="101"] > td:nth-child(6) [data-explain]').first();
  if(width===390)await healthTrigger.tap();else await healthTrigger.focus();
  const healthId=await healthTrigger.getAttribute('data-explain'),health=page.locator(`[id="${healthId}"]`);
  await health.waitFor({state:'visible'});assert.match(await health.innerText(),/2,250 lb/);
  const summary=health.locator('summary');assert.equal(await summary.count(),1);await summary.click();assert.match(await health.innerText(),/CQ-25/);assert.match(await health.innerText(),/1,250/);
  await page.keyboard.press('Escape');
  const ready=page.locator('.order-row[data-order-id="101"] .order-ready-checkbox');await ready.check();await loaded(page);
  await page.waitForFunction(()=>document.querySelector('.order-row[data-order-id="101"] .order-ready-checkbox')?.checked);
  assert.equal(api.writes.at(-1).action,'ready');assert.equal(api.writes.at(-1).body.ready,true);
  if(width===390){
   const tabBar=page.locator('[role="tablist"]').filter({has:page.locator('[data-orders-tab]')});
   assert(await tabBar.evaluate(el=>el.scrollWidth>el.clientWidth),'Phone tabs must have horizontal scroll');
   await page.screenshot({path:path.join(out,`${width}-interactions.png`)});
   assert.deepEqual(errors,[]);results.push({width,tabs:true,counts:true,refinements:true,popovers:true,healthDetail:true,readyToggle:true,pageErrors:errors});await context.close();continue;
  }
  let dialog=await openAction(page,101,'close');
  assert.equal(await dialog.getByRole('button',{name:'Close order',exact:true}).isVisible(),false);
  await dialog.getByRole('button',{name:'Preview close',exact:true}).click();await dialog.locator('.so-exit-release-summary').waitFor({state:'visible'});
  assert.equal(await dialog.locator('.so-exit-release-summary').innerText(),'Will release 2 reservations (1,500 lb)');
  await dialog.locator('#so-exit-note').fill('Receipt confirmed by customer.');
  assert.equal(await dialog.getByRole('button',{name:'Close order',exact:true}).isVisible(),false,'Editing invalidates a preview');
  await dialog.getByRole('button',{name:'Preview close',exact:true}).click();await dialog.getByRole('button',{name:'Close order',exact:true}).click();await dialog.waitFor({state:'hidden'});await loaded(page);
  const closeWrites=api.writes.filter(w=>w.id===101&&w.action==='close');
  assert.deepEqual(closeWrites.map(w=>w.body.mode),['preview','preview','commit']);assert.equal(closeWrites.at(-1).body.note,'Receipt confirmed by customer.');assert(!(await ids(page)).includes(101));
  dialog=await openAction(page,103,'cancel');await dialog.locator('#so-exit-reason').selectOption('other');
  let before=api.writes.length;await dialog.getByRole('button',{name:'Preview cancel',exact:true}).click();assert.equal(api.writes.length,before,'Other requires a note before requesting preview');
  await dialog.locator('#so-exit-note').fill('Customer asked us to withdraw.');
  await dialog.getByRole('button',{name:'Preview cancel',exact:true}).click();await dialog.locator('.so-exit-preview').waitFor({state:'visible'});
  await dialog.locator('#so-exit-reason').selectOption('duplicate');
  assert.equal(await dialog.getByRole('button',{name:'Cancel order',exact:true}).isVisible(),false);
  before=api.writes.length;await dialog.getByRole('button',{name:'Preview cancel',exact:true}).click();assert.equal(api.writes.length,before,'Duplicate requires a related SO');
  await dialog.locator('#so-exit-related').fill('SO-1422');await dialog.getByRole('button',{name:'Preview cancel',exact:true}).click();await dialog.locator('.so-exit-preview').waitFor({state:'visible'});
  assert.equal(api.writes.at(-1).body.related_so_id,102);
  await dialog.locator('#so-exit-reason').selectOption('superseded');assert.equal(await dialog.getByRole('button',{name:'Cancel order',exact:true}).isVisible(),false);
  await dialog.getByRole('button',{name:'Preview cancel',exact:true}).click();await dialog.getByRole('button',{name:'Cancel order',exact:true}).click();await dialog.waitFor({state:'hidden'});await loaded(page);
  assert.equal(api.writes.at(-1).body.reason,'superseded');assert.equal(api.writes.at(-1).body.related_so_id,102);
  dialog=await openAction(page,105,'cancel');await dialog.getByRole('button',{name:'Preview cancel',exact:true}).click();
  await dialog.getByRole('button',{name:/Preview close instead/}).click();await dialog.locator('.so-exit-preview').waitFor({state:'visible'});
  assert.equal(api.writes.at(-1).action,'close');assert.equal(api.writes.at(-1).body.reason,'short_closed');assert.equal(api.writes.at(-1).body.mode,'preview');
  assert(!/POST|short_closed|error_code|HTTP/.test(await dialog.innerText()));await dialog.getByRole('button',{name:'Back',exact:true}).click();
  await switchTab(page,'Closed');dialog=await openAction(page,110,'reopen');await dialog.getByRole('button',{name:'Preview reopen',exact:true}).click();await dialog.getByRole('button',{name:'Reopen order',exact:true}).click();await dialog.waitFor({state:'hidden'});await loaded(page);
  assert.deepEqual(api.writes.filter(w=>w.id===110).map(w=>w.body.mode),['preview','commit']);assert(!(await ids(page)).includes(110));
  await switchTab(page,'Cancelled');dialog=await openAction(page,111,'reopen');await dialog.getByRole('button',{name:'Preview reopen',exact:true}).click();await dialog.getByRole('button',{name:'Reopen order',exact:true}).click();await dialog.waitFor({state:'hidden'});await loaded(page);
  assert.deepEqual(api.writes.filter(w=>w.id===111).map(w=>w.body.mode),['preview','commit']);
  await switchTab(page,'Open');dialog=await openAction(page,104,'close');api.setExitFailure();await dialog.getByRole('button',{name:'Preview close',exact:true}).click();await dialog.locator('.so-exit-error').waitFor({state:'visible'});
  assert.match(await dialog.locator('.so-exit-error').innerText(),/temporarily unavailable/i);assert(!/HTTP|503|internal_database|raw_json/.test(await dialog.innerText()));await dialog.getByRole('button',{name:'Back',exact:true}).click();
  api.setListFailure(true);await page.locator('#refresh-btn').click();await page.locator('#orders-error').waitFor({state:'visible'});
  assert(!/HTTP|503|database_connection|raw_json/.test(await page.locator('#orders-error').innerText()));api.setListFailure(false);await page.locator('#refresh-btn').click();await loaded(page);
  await page.screenshot({path:path.join(out,`${width}-interactions.png`)});
  assert.deepEqual(errors,[]);results.push({width,tabs:true,counts:true,refinements:true,popovers:true,healthDetail:true,readyToggle:true,previewBeforeCommit:true,previewInvalidation:true,cancelValidation:true,relatedOrderResolution:true,conflictCloseOffer:true,reopenBothStates:true,plainLanguageErrors:true,pageErrors:errors,writes:api.writes});await context.close();
 }
}finally{await browser.close();await server.close();}
await fs.writeFile(path.join(out,'results.json'),JSON.stringify(results,null,2)+'\n');console.log(JSON.stringify(results,null,2));

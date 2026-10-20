// Read-only integration browser harness. All API requests are intercepted;
// upload/extraction/match are synthetic, and no production writes can occur.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';
import { installApiStub, buildTokenTable, loadFixture, API_HOST } from './lib/stub.mjs';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const output = process.env.BROWSER_OUTPUT || path.join(root, 'test-results/intake-integration');
await fs.mkdir(output, {recursive:true});
const server = await startStaticServer(path.join(root,'dashboard'));
const browser = await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL || 'chrome'});
const results=[];
const product={product_id:9,name:'Fixture Granola 10 LB',odoo_code:'1614',label_type:'house',prior_sales:true};
const line={vendor_description:'Fixture granola 10 lb case',description:'Fixture granola 10 lb case',customer_item_code:'CQ-77',quantity:3,unit:'CASE',unit_price:32.5,match_source:'exact',confidence:1,product,candidates:[],lb_per_unit:10,lb_source:'case_size',expected_qty_lb:30,case_size_lb:10,case_size_source:'product',quantity_lb:30};
try {
 for(const width of [1440,390]) for(const theme of ['light','dark']) {
  const context=await browser.newContext({viewport:{width,height:900},colorScheme:theme});
  const page=await context.newPage();
  const errors=[], requests=[], blocked=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(theme=>{localStorage.setItem('dashboard-theme',theme);window.__opened=[];window.open=(...args)=>{window.__opened.push(args);return null;};},theme);
  const tokens=buildTokenTable();const state={unmatched:new Set()};
  // Block all nonlocal/non-API traffic, then install the API fixture handler.
  await context.route('**/*',route=>{
   const u=new URL(route.request().url());
   if(u.origin===server.origin)return route.continue();
   blocked.push(u.hostname);return route.abort();
  });
  await installApiStub(page,tokens,state);
  await page.route(`https://${API_HOST}/**`,async route=>{
   const req=route.request();const u=new URL(req.url());requests.push({method:req.method(),path:u.pathname});
   const reply=data=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)});
   if(u.pathname==='/auth/session/config')return reply({key_sessions_enabled:false});
   if(u.pathname==='/customers')return reply({customers:[{id:4,name:'Fixture Customer'}]});
   if(u.pathname==='/suppliers')return reply({suppliers:[{id:4,name:'Fixture Supplier'}]});
   if(u.pathname==='/sales/orders' && req.method()==='GET'){
    const data=await loadFixture('sales-orders-list.json',tokens);
    data.orders.forEach(o=>{o.customer_po='PO-'+o.order_id;if(o.ready){o.ready_by='floor';o.ready_at='2026-09-08T17:17:00Z';}});
    const selected=u.searchParams.get('state');data.orders=data.orders.filter(o=>!selected||o.state===selected);return reply(data);
   }
   if(u.pathname.match(/^\/purchase-documents\/\d+\/url$/))return reply({url:server.origin+'/fixture-document.pdf'});
   if(['/expected-receipts/extract','/sales/orders/extract'].includes(u.pathname)){
    assert.match(req.headers()['content-type'],/^multipart\/form-data/);
    assert.ok(req.postDataBuffer().length>0);
    return reply({document_id:u.pathname.startsWith('/sales')?902:901,already_seen:false});
   }
   if(u.pathname.match(/^\/purchase-documents\/90[12]\/extract$/))return reply({document_id:u.pathname.includes('902')?902:901,storage_path:'fixture.pdf',extraction:{supplier_name:'Fixture Supplier',customer_name:'Fixture Customer',reference_number:'ER-FIXTURE',po_number:'SO-FIXTURE',document_date:'2026-09-08',expected_delivery_date:'2026-09-10',requested_ship_date:'2026-09-10',lines:[line,line,line]}});
   if(u.pathname==='/expected-receipts/match')return reply({supplier:{match:{supplier_id:4,name:'Fixture Supplier'},candidates:[]},duplicate_warning:null,lines:[line,line,line]});
   if(u.pathname==='/sales/orders/match')return reply({customer:{match:{customer_id:4,name:'Fixture Customer'},candidates:[]},duplicate_warning:null,prior_sales_product_ids:[9],lines:[line,line,line]});
   if(req.method()!=='GET')throw Error('Unexpected write: '+u.pathname);
   return route.fallback();
  });
  await page.goto(server.origin+'/?section=orders');
  // The Sales Orders board now uses State tabs, eight columns and a readiness
  // checkbox. The former status select/paperclip/ready badge were retired.
  await page.locator('[data-orders-tab="open"]').click();
  const table=page.locator('#orders-list-view .orders-table').first();
  await table.locator('th .table-sort').first().waitFor({state:'attached'});
  const original=await table.locator('.order-row').evaluateAll(rows=>rows.map(row=>({id:row.dataset.orderId,state:row.dataset.state})));
  assert.ok(original.length>0);
  assert.ok(original.every(row=>row.state==='open'));
  assert.equal(await table.locator('thead th').count(),8);
  // Sorting must retain the complete order and its associated expansion row.
  await page.locator('#orders-table-tools .table-tools select').first().selectOption({label:'Customer'});
  for(const row of original){
   assert.equal(await table.locator(`.order-row[data-order-id="${row.id}"]`).evaluate(r=>r.nextElementSibling.dataset.orderId),row.id);
  }
  assert.ok(await page.locator('#orders-list-view').isVisible());
  assert.equal(await page.locator('#order-detail-view').isVisible(),false);
  assert.ok(await table.locator('.order-link').first().evaluate(el=>el.tagName==='BUTTON'));
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
  const cardGeometry=await table.locator('.order-row').evaluateAll(rows=>rows.map(r=>{
   const ready=r.querySelector('.order-ready-checkbox').getBoundingClientRect(),row=r.getBoundingClientRect();
   return {id:r.dataset.orderId,readyContained:ready.left>=row.left&&ready.right<=row.right&&ready.top>=row.top&&ready.bottom<=row.bottom};
  }));
  assert.ok(cardGeometry.every(row=>row.readyContained),JSON.stringify(cardGeometry));
  await page.evaluate(()=>window.scrollTo(0,0));
  const toolbar=await page.locator('#orders-list-view .orders-toolbar').evaluate(e=>({height:e.getBoundingClientRect().height,width:e.getBoundingClientRect().width}));
  await page.screenshot({path:path.join(output,`orders-${width}-${theme}.png`)});
  await table.locator('.order-row').filter({has:page.locator('.order-ready-checkbox:checked')}).first().screenshot({path:path.join(output,`ready-card-${width}-${theme}.png`)});
  const measurements=[];
  for(const kind of ['er','so']){
   if(kind==='er')await page.goto(server.origin+'/?section=expected');
   else await page.goto(server.origin+'/?section=orders');
   // Section ids are handled by the existing shell; explicit tab control if needed.
   const open=page.locator(`#${kind}-new-btn`);
   if(!await open.isVisible()){
    const target=kind==='er'?'expected':'orders';
    await page.locator(`[data-tab="${target}"]`).first().click({force:true});
   }
   await open.click();await page.locator(`#${kind}-modal-overlay`).waitFor({state:'visible'});
   if(width===390)assert.equal(await page.locator('.mobile-nav').isVisible(),false);
   const upload=page.locator(`#${kind}-file-input`);
   await upload.setInputFiles({name:'fixture-po.pdf',mimeType:'application/pdf',buffer:Buffer.from('%PDF-1.4\n% fixture upload; extraction is stubbed\n')});
   await page.locator(`#${kind}-review-body .er-line-card`).first().waitFor();
   assert.equal(await page.locator(`#${kind}-review-body .er-line-card`).count(),3);
   const approve=page.locator(`#${kind}-review-approve`);
   assert.equal(await approve.isEnabled(),true);
   const modal=page.locator(`#${kind}-modal-overlay .er-modal`);
   await modal.locator('.note-modal-body').evaluate(el=>{el.scrollTop=el.scrollHeight;});
   const measure=await modal.evaluate(el=>{
    const body=el.querySelector('.note-modal-body'),footer=el.querySelector('.er-review-footer'),button=footer.querySelector('.er-approve-btn');
    const box=el.getBoundingClientRect(),action=button.getBoundingClientRect();
    return {modalWidth:box.width,modalRight:box.right,bodyWidth:body.clientWidth,bodyScrollWidth:body.scrollWidth,footerPosition:getComputedStyle(footer).position,actionTop:action.top,actionBottom:action.bottom,actionRight:action.right,actionHeight:action.height,font:getComputedStyle(button).fontSize,pageWidth:document.documentElement.scrollWidth,viewport:innerWidth};
   });
   measurements.push({kind,...measure});
   assert.ok(measure.bodyScrollWidth<=measure.bodyWidth+1,JSON.stringify(measure));
   assert.ok(measure.pageWidth<=width,JSON.stringify(measure));
   assert.ok(measure.actionBottom<=900 && measure.actionTop>=0,JSON.stringify(measure));
   assert.ok(measure.actionRight<=width,JSON.stringify(measure));
   assert.equal(measure.footerPosition,'sticky');
   const overlap=await modal.locator('.er-line-qtyrow').evaluateAll(rows=>rows.some(row=>{const cells=[...row.querySelectorAll('.er-qcell')];return cells.some(cell=>[...cell.children].some(child=>{const a=child.getBoundingClientRect();return cells.some(other=>{if(other===cell)return false;const b=other.getBoundingClientRect();return Math.min(a.right,b.right)-Math.max(a.left,b.left)>1 && Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1;});}));}));
   assert.equal(overlap,false,'quantity labels and source tags must not overlap adjacent cells');
   await page.screenshot({path:path.join(output,`${kind}-${width}-${theme}.png`)});
   await modal.locator('.note-modal-body').evaluate(el=>{el.scrollTop=0;});
   await page.screenshot({path:path.join(output,`${kind}-${width}-${theme}-top.png`)});
   // Review edits are real UI transitions; no approval request is sent.
   await page.locator(`#${kind}-review-${kind==='er'?'reference':'po'}`).fill(kind.toUpperCase()+'-EDITED');
   await page.locator(`#${kind}-modal-close`).click();
  }
  await page.goto(server.origin+'/history.html?day=2026-09-08&type=ship');
  await page.locator('#history-status').filter({hasNotText:'Loading'}).waitFor();
  await page.screenshot({path:path.join(output,`history-${width}-${theme}.png`)});
  const historyStyle=await page.locator('#history-form').evaluate(e=>({padding:getComputedStyle(e.closest('main')).paddingLeft,controlBackground:getComputedStyle(e.querySelector('select')).backgroundColor}));
  assert.deepEqual(errors,[]);
  assert.ok(!requests.some(r=>r.path.endsWith('/approve')));
  results.push({width,theme,rowsSorted:original.length,listContract:'State tabs, eight columns, readiness checkbox and attached expansion rows',cardGeometry,toolbar,historyStyle,measurements,errors,unmatched:[...state.unmatched],blocked:[...new Set(blocked)]});
  await context.close();
 }
 await fs.writeFile(path.join(output,'results.json'),JSON.stringify(results,null,2));
 console.log(JSON.stringify(results,null,2));
} finally {await browser.close();await server.close();}

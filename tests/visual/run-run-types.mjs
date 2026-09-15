import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
const server=await startStaticServer(path.resolve('dashboard'));
const browser=await chromium.launch();
const out=path.resolve(process.env.RUNS_AUDIT_OUT||'work/run-types-audit');
await fs.mkdir(out,{recursive:true});
try {
 for(const width of [1440,390]) for(const theme of ['light','dark']) {
  const context=await browser.newContext({viewport:{width,height:950}});
  await context.addInitScript(({theme})=>{
   localStorage.setItem('dashboard-theme',theme);
   const D=Date;window.Date=class extends D {constructor(...a){super(...(a.length?a:['2026-09-15T12:00:00Z']));}static now(){return new D('2026-09-15T12:00:00Z').valueOf();}};
  },{theme});
  const recipe={id:107,name:'Batch Classic Granola #9',odoo_code:'90002',type:'batch',default_batch_lb:323,active:true};
  const coconut={id:108,name:'Sweetened Coconut Batch',odoo_code:'90003',type:'batch',default_batch_lb:360,active:true};
  const fg={id:10,name:'Classic Granola 25 lb',type:'finished',case_size_lb:25,active:true};
  const runs=[{id:1,product_id:10,product_name:fg.name,run_type:'pack',planned_qty:4,planned_unit:'cases',planned_qty_lb:100,planned_date:'2026-10-01',status:'planned',coverage:[{sales_order_line_id:11,qty_lb:100}]}];
  const writes=[],errors=[];
  await context.route('**/*',async route=>{
   const req=route.request(),url=new URL(req.url()),p=url.pathname;
   if(url.origin===server.origin)return route.continue();
   const send=data=>route.fulfill({contentType:'application/json',body:JSON.stringify(data)});
   if(p==='/products/search')return send({products:[recipe,coconut,fg]});
   if(p==='/production/runs'&&req.method()==='GET')return send({runs:runs.filter(r=>!url.searchParams.has('from')||(r.planned_date>=url.searchParams.get('from')&&r.planned_date<=url.searchParams.get('to')))});
   if(p==='/sales/orders')return send({orders:[{state:'open',order_id:5,pallet_lines:[{product_id:10}]}]});
   if(p==='/sales/orders/5')return send({state:'open',order_number:'SO-5',lines:[{line_id:11,product_id:10,line_status:'pending',readiness:{remaining_lb:1000}},{line_id:12,product_id:999,line_status:'pending',readiness:{remaining_lb:1000}}]});
   if(p.startsWith('/production/runs')&&req.method()!=='GET') {
    const body=req.postDataJSON();writes.push({method:req.method(),body});
    if(req.method()==='POST') {
     const prod=[recipe,coconut,fg].find(x=>x.id===body.product_id);
     const r={...body,id:runs.length+1,product_name:prod.name,pan_yield_lb_used:body.planned_unit==='pans'?prod.default_batch_lb:null,planned_qty_lb:body.planned_qty*(body.planned_unit==='pans'?prod.default_batch_lb:1),status:'planned',coverage:[],coverage_product_ids:[10]};
     r.expected_lb=r.planned_qty_lb;runs.push(r);return send({run:r});
    }
    const r=runs.find(x=>x.id===Number(p.split('/')[3]));
    if(req.method()==='PATCH')Object.assign(r,body);
    if(req.method()==='PUT')r.coverage=body.coverage.map(c=>({...c,mixed_bake_pack:true}));
    return send({run:r});
   }
   errors.push(req.method()+' '+p);await route.abort();
  });
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.origin+'/runs.html');await page.locator('#week-board').waitFor();
  await page.locator('#new-run').click();
  assert.match(await page.locator('#run-type').innerText(),/does not consume WIP yet/);
  assert.equal(await page.locator('#planned-line').count(),0);
  await page.locator('#run-type').selectOption('bake');
  await page.locator('#product-search').fill('batch');await page.locator('[data-product]').first().waitFor();
  assert.equal(await page.locator('[data-product]').count(),2);
  await page.locator('[data-product]').first().click();await page.locator('#planned-qty').fill('12');
  assert.equal(await page.locator('#planned-unit').inputValue(),'pans');
  assert.match(await page.locator('#case-reason').innerText(),/Expected 3,876 lb/);
  await page.evaluate(()=>window.SOList.closeExplanation());
  await page.screenshot({path:path.join(out,`bake-form-${width}-${theme}.png`)});
  await page.locator('#planned-qty').fill('1.5');await page.locator('#save-run').click();assert.equal(writes.length,0);
  await page.locator('#planned-qty').fill('12');await page.locator('#save-run').click();await page.locator('[data-open="2"]').waitFor();
  assert.equal(writes[0].body.run_type,'bake');assert.equal(writes[0].body.line_id,undefined);
  assert.match(await page.locator('[data-run-id="2"]').innerText(),/12 pans/);
  assert.match(await page.locator('[data-run-id="2"]').innerText(),/Expected 3,876 lb/);
  await page.screenshot({path:path.join(out,`bake-board-${width}-${theme}.png`)});
  await page.locator('[data-open="2"]').click();await page.locator('#edit-run').click();
  recipe.default_batch_lb=400; // Refresh metadata for a subsequent edit.
  await page.locator('#planned-notes').fill('Keep saved conversion');await page.locator('#save-run').click();await page.locator('[data-open="2"]').waitFor();
  assert.equal(writes.at(-1).body.planned_qty,undefined);assert.equal(writes.at(-1).body.run_type,undefined);assert.equal(writes.at(-1).body.line_id,undefined);
  await page.locator('[data-open="2"]').click();await page.locator('#edit-run').click();
  await page.waitForFunction(()=>document.querySelector('#case-reason').textContent.includes('3,876'));
  assert.match(await page.locator('#case-reason').innerText(),/3,876/);
  await page.locator('#planned-qty').fill('13');await page.waitForFunction(()=>document.querySelector('#case-reason').textContent.includes('5,200'));assert.match(await page.locator('#case-reason').innerText(),/5,200/);
  await page.locator('#dismiss-dialog').click();
  await page.locator('[data-open="2"]').click();await page.locator('#edit-coverage').click();await page.locator('[data-cover="11"]').waitFor();
  assert.equal(await page.locator('[data-cover]').count(),1,'Only the routed finished SKU is offered');
  await page.locator('[data-cover="11"]').fill('100');assert.match(await page.locator('#coverage-warning').innerText(),/both bake and pack coverage/);
  await page.screenshot({path:path.join(out,`mixed-coverage-${width}-${theme}.png`)});
  await page.locator('#save-run').click();await page.locator('[data-open="2"]').waitFor();
  assert.equal(writes.at(-1).method,'PUT');await page.locator('[data-run-id="2"] .error').waitFor();assert.match(await page.locator('[data-run-id="2"]').innerText(),/both bake and pack coverage/);
  await page.locator('#new-run').click();await page.locator('#run-type').selectOption('coconut');
  await page.locator('#product-search').fill('coconut');await page.locator('[data-product]').last().waitFor();await page.locator('[data-product]').last().click();
  await page.locator('#planned-qty').fill('12');assert.match(await page.locator('#case-reason').innerText(),/4,320 lb/);
  await page.locator('#run-type').selectOption('other');assert.equal(await page.locator('#planned-unit option[value="pans"]').count(),0);
  assert.deepEqual(errors,[]);await context.close();
 }
 console.log('Run-type UI: bake/coconut/pack/other, whole pans, saved yields, routed coverage, cross-week warning, daily board; 1440/390 light/dark passed.');
} finally {await browser.close();await server.close();}

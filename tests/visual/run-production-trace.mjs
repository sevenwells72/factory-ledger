// Exercise real dashboard disclosures with intercepted data; never contacts production.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { API_HOST, buildTokenTable, installApiStub } from './lib/stub.mjs';
const out = path.resolve(process.argv[2] || 'work/production-trace');
await fs.mkdir(out, {recursive: true});
const server = await startStaticServer(path.resolve('dashboard'));
const browser = await chromium.launch();
const tokens = buildTokenTable();
const day = tokens.get('{{TODAY}}');
const ingredient = {product_id:3, product_name:'Oats <img src=x onerror=alert(1)>', lot_code:'LOT<&>', supplier_lot_code:'SUP<&>', quantity:18.1234, unit:'lb', product_type:'ingredient', lot_id:33};
const batch = {product_id:1, product_name:'Batch Classic Granola', lot_code:'BATCH-1', quantity:25, unit:'lb', product_type:'batch', lot_id:11};
const bag = {product_id:4, product_name:'Packaging bags', lot_code:'BAG-1', quantity:2, unit:'each', product_type:'ingredient', lot_id:44};
try {
 for (const width of [1440,390]) for (const theme of ['light','dark']) {
  const context = await browser.newContext({viewport:{width,height:1000}});
  await context.addInitScript(t=>localStorage.setItem('dashboard-theme',t),theme);
  await installApiStub(context,tokens,{fail:[],overrides:{},status:{},unmatched:new Set()});
  let failed=false;
  const requests=[];
  await context.route(`**://${API_HOST}/dashboard/api/production**`, async route=>{
   const url=new URL(route.request().url());
   assert.equal(route.request().method(),'GET');
   if(url.pathname.endsWith('/trace')) {
    requests.push(Object.fromEntries(url.searchParams));
    if(!failed){failed=true;return route.fulfill({status:503,json:{error:'private technical error'}});}
    const kind=url.searchParams.get('kind') || 'make';
    const consumed=kind==='pack'?[batch,bag]:[ingredient];
    return route.fulfill({json:{kind,transactions:[{transaction_id:kind==='pack'?202:101,consumed}],subtotals:consumed}});
   }
   return route.fulfill({json:{days:[{date:day,day_name:'Today',batches:[{product_id:1,product_name:'Batch Classic Granola',batch_count:1}],finished_goods:[{product_id:2,product_name:'Classic Granola 25 LB',sku:'GR25',pack_format:'25lb',unit_count:1}]}]}});
  });
  const page=await context.newPage();
  const errors=[];page.on('pageerror', e=>errors.push(e.message));
  await page.goto(server.origin);
  const card=page.locator(`[data-production-date="${day}"]`);
  await card.click();
  const made=page.locator('.production-trace[data-kind="make"]');
  const summary=made.locator(':scope > summary');
  await summary.focus();await page.keyboard.press('Enter');
  await made.getByRole('button',{name:'Retry'}).click();
  await made.getByText('Ingredient subtotals',{exact:true}).waitFor();
  assert((await made.innerText()).includes('SUP<&>'));
  assert((await made.innerText()).includes('18.1234 lb'));
  assert.equal(await made.locator('img').count(),0);
  await summary.focus();await page.keyboard.press('Space');
  await page.waitForFunction(()=>!document.querySelector('.production-trace[data-kind="make"]').open);
  await page.keyboard.press('Space');
  const packed=page.locator('.production-trace[data-kind="pack"]');
  await packed.locator(':scope > summary').click();
  await packed.getByText('Consumed subtotals',{exact:true}).waitFor();
  assert((await packed.innerText()).includes('Packaging bags · Lot BAG-1 · 2 each'));
  const nested=packed.locator('.production-trace');
  await nested.locator(':scope > summary').focus();await page.keyboard.press('Enter');
  await nested.getByText('Ingredient subtotals',{exact:true}).waitFor();
  assert(requests.some(r=>r.lot_id==='11'&&!r.start_date));
  assert(requests.filter(r=>r.product_id).every(r=>r.start_date===day&&r.end_date===day));
  const geometry=await page.locator('.production-trace > summary').evaluateAll(nodes=>nodes.map(n=>({height:n.getBoundingClientRect().height,width:n.getBoundingClientRect().width})));
  assert(geometry.every(g=>g.height>=44&&g.width>=44));
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.deepEqual(errors,[]);
  await page.locator('#production-calendar-detail').screenshot({path:path.join(out,`${width}-${theme}.png`)});
  await context.close();
 }
 console.log('PASS: four viewport/theme combinations, keyboard expand/collapse, nested prior-date lot trace, retry, escaped text, units, touch targets, no overflow or page errors.');
} finally {await browser.close();await server.close();}

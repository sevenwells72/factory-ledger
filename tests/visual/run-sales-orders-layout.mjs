// Regression: orthogonal columns fit desktop widths, compact rows, sticky sort
// controls, narrow-tablet scrolling, and mobile cards.
// node tests/visual/run-sales-orders-layout.mjs [dashboard-root] [output-dir]
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { installApiStub, buildTokenTable, loadFixture, API_HOST } from './lib/stub.mjs';
import { SCREENS } from './lib/screens.mjs';
const root = path.resolve(process.argv[2] || 'dashboard');
const out = path.resolve(process.argv[3] || 'docs/design/audit/screenshots/sales-orders-layout');
await fs.mkdir(out, { recursive:true });
const server = await startStaticServer(root);
const browser = await chromium.launch();
const results = [];
try {
 for (const state of ['open','closed','cancelled']) for (const width of (state==='open'?[1440,1385,1280,1200,1101,1024,390]:[1440,1200,390])) for (const theme of ['light','dark']) {
  const context = await browser.newContext({viewport:{width,height:900},colorScheme:theme});
  await context.addInitScript(t=>localStorage.setItem('dashboard-theme',t),theme);
  const tokens = buildTokenTable();
  await installApiStub(context,tokens,{fail:[],overrides:{},status:{}});
  // Production-length identifiers, attachment, partial quantity, and long
  // customers exercise content widths hidden by the short base identifiers.
  const orders = await loadFixture('sales-orders-list.json',tokens);
  orders.orders.forEach((o,i)=>{o.order_number=`SO-260908-${String(i+1).padStart(3,'0')}`;if(o.ready)o.ready_by='floor';});
  orders.orders[0].source_document_id=3001;
  orders.orders[0].customer='International Gourmet Foods Inc';
  await context.route(`https://${API_HOST}/sales/orders?*`,route=>{
   const selected=new URL(route.request().url()).searchParams.get('state');
   const rows=orders.orders.filter(o=>!selected||o.state===selected);
   return route.fulfill({json:{orders:rows,count:rows.length}});
  });
  const page = await context.newPage();
  await page.goto(server.origin);
  await SCREENS.find(s=>s.id==='S-25').setup(page);
  if(state!=='open'){
   await page.getByRole('tab',{name:new RegExp(`^${state}\\b`,'i')}).click();
   await page.locator(`.order-row[data-state="${state}"]`).first().waitFor({state:'visible'});
  }
  const controls = page.locator('#section-orders .table-tools');
  const sort = controls.locator('select');
  // The table enhancement observer runs after the tab's rows are attached.
  await sort.waitFor({state:'visible'});
  await sort.locator('option').filter({hasText:/ship by/i}).first().waitFor({state:'attached'});
  const shipByOption = await sort.locator('option').evaluateAll(options=>options.find(o=>/ship by/i.test(o.textContent))?.value);
  if(shipByOption===undefined)throw new Error('Ship by sorting option is missing');
  await sort.selectOption(shipByOption);
  await page.evaluate(()=>window.scrollTo(0,600));
  await page.waitForTimeout(300);
  const result = await page.evaluate(() => {
   const table=document.querySelector('#orders-table-container table');
   const tools=document.querySelector('#section-orders .table-tools');
   const desktop=innerWidth>768, fitted=innerWidth>1100;
   const stack=['.site-nav','.app-header','.tab-bar'].map(s=>document.querySelector(s)).filter(e=>['sticky','fixed'].includes(getComputedStyle(e).position)&&e.getBoundingClientRect().height).reduce((n,e)=>Math.max(n,e.getBoundingClientRect().bottom),0);
   const r=tools.getBoundingClientRect();
   const scroller=getComputedStyle(table).overflowX==='auto'?table:table.parentElement;
   const bounds=scroller.getBoundingClientRect(), t=table.getBoundingClientRect();
   const hint=getComputedStyle(tools,'::after').content;
   const rows=[...table.querySelectorAll('.order-row')];
   const cells=rows.flatMap(row=>[...row.children]).filter(e=>e.tagName==='TD'&&e.getBoundingClientRect().width);
   const headers=[...table.querySelectorAll('thead th')].map(e=>e.textContent.replace(/[↑↓↕]/g,'').trim().replace(/\s+/g,' '));
   const expected=['Ready to ship','SO #','Customer','Ship by','Fulfillment','Health','Left to ship','Pallets'];
   const expanded=rows.filter(row=>!row.classList.contains('expanded'));
   const farCells=rows.map(row=>row.children[7]).filter(Boolean);
   const controlsVisible=!desktop||r.top>=stack-1;
   return {width:innerWidth,desktop,fitted,stack,controlsTop:r.top,tableWidth:t.width,containerWidth:bounds.width,scrollWidth:scroller.scrollWidth,headers,
    controlsVisible,
    controlsInHeader:Boolean(tools.closest('#section-orders > .section-header')),
    legacyHideReadyRemoved:!document.querySelector('#orders-hide-ready'),
    eightOrthogonalColumns:headers.length===8&&headers.every((h,i)=>h.toLowerCase()===expected[i].toLowerCase())&&rows.every(row=>row.children.length===8),
    expanderInOrderCell:rows.every(row=>row.children[1].querySelector('.order-expand-toggle')),
    allColumnsFit:!fitted||(t.right<=bounds.right+1&&t.left>=bounds.left-1&&scroller.scrollWidth<=scroller.clientWidth+1&&cells.every(e=>{const b=e.getBoundingClientRect();return b.left>=bounds.left-1&&b.right<=bounds.right+1&&e.scrollWidth<=e.clientWidth+1;})),
    compactDesktopRows:innerWidth<1200||expanded.every(row=>row.getBoundingClientRect().height<=56.5),
    tallestRow:Math.max(...expanded.map(row=>row.getBoundingClientRect().height)),
    scrollAffordance:!desktop||(fitted?!hint.includes('Scroll sideways'):hint.includes('Scroll sideways')&&scroller.scrollWidth>scroller.clientWidth),
    pageContained:document.documentElement.scrollWidth<=innerWidth,
    mobileCards:desktop||rows.every(row=>getComputedStyle(row).display==='grid'),
    visibleLastColumn:!fitted||farCells.every(e=>e.getBoundingClientRect().right<=bounds.right+1)};
  });
  await page.screenshot({path:path.join(out,`${state}-${width}-${theme}-left.png`)});
  const lastColumnReachable = await page.evaluate(()=>{
   const table=document.querySelector('#orders-table-container table');
   const scroller=getComputedStyle(table).overflowX==='auto'?table:table.parentElement;
   if(innerWidth>768&&innerWidth<=1100)scroller.scrollLeft=scroller.scrollWidth;
   const r=scroller.getBoundingClientRect();
   return [...table.querySelectorAll('.order-row > td:last-child')].filter(e=>e.getBoundingClientRect().width).every(e=>{
    const b=e.getBoundingClientRect();return b.left>=r.left-1&&b.right<=r.right+1&&e.scrollWidth<=e.clientWidth+1;
   });
  });
  await page.screenshot({path:path.join(out,`${state}-${width}-${theme}-right.png`)});
  results.push({state,theme,...result,lastColumnReachable});
  await context.close();
 }
} finally { await browser.close();await server.close(); }
await fs.writeFile(path.join(out,'results.json'),JSON.stringify(results,null,2));
console.log(JSON.stringify(results,null,2));
const checks=['controlsVisible','controlsInHeader','legacyHideReadyRemoved','eightOrthogonalColumns','expanderInOrderCell','allColumnsFit','compactDesktopRows','scrollAffordance','pageContained','mobileCards','visibleLastColumn','lastColumnReachable'];
if(results.some(r=>checks.some(k=>!r[k])))process.exitCode=1;

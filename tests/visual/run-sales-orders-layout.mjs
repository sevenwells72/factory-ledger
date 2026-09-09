// Regression: all columns fit at desktop widths, with content-sized Status,
// wrapped Blockers and sticky sorting. Narrow tablets scroll; phones use cards.
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
 for (const width of [1440,1385,1280,1101,1024,390]) for (const theme of ['light','dark']) {
  const context = await browser.newContext({viewport:{width,height:900},colorScheme:theme});
  await context.addInitScript(t=>localStorage.setItem('dashboard-theme',t),theme);
  const tokens = buildTokenTable();
  await installApiStub(context,tokens,{fail:[],overrides:{},status:{}});
  // Production-length order numbers, an attachment, long customer names and
  // ready stamps exercise minimum widths that the short SO-1421 fixture missed.
  const orders = await loadFixture('sales-orders.json',tokens);
  orders.orders.forEach((o,i)=>{o.order_number=`SO-260908-${String(i+1).padStart(3,'0')}`;if(o.ready)o.ready_by='floor';});
  orders.orders[0].source_document_id=3001;
  orders.orders[0].customer='International Gourmet Foods Inc';
  await context.route(`https://${API_HOST}/sales/orders?*`,route=>route.fulfill({json:orders}));
  const page = await context.newPage();
  await page.goto(server.origin);
  await SCREENS.find(s=>s.id==='S-25').setup(page);
  const controls = page.locator('#orders-table-container .table-tools');
  await controls.locator('select').selectOption('3');
  await page.evaluate(()=>window.scrollTo(0,600));
  await page.waitForTimeout(300);
  const result = await page.evaluate(() => {
   const table=document.querySelector('#orders-table-container table');
   const tools=document.querySelector('#orders-table-container .table-tools');
   const desktop=innerWidth>768, fitted=innerWidth>1100;
   const stack=['.site-nav','.app-header','.tab-bar'].map(s=>document.querySelector(s)).filter(e=>['sticky','fixed'].includes(getComputedStyle(e).position)&&e.getBoundingClientRect().height).reduce((n,e)=>Math.max(n,e.getBoundingClientRect().bottom),0);
   const r=tools.getBoundingClientRect();
   const scroller=getComputedStyle(table).overflowX==='auto'?table:table.parentElement;
   const bounds=scroller.getBoundingClientRect(), t=table.getBoundingClientRect();
   const hint=getComputedStyle(tools,'::after').content;
   const cells=[...table.querySelectorAll('.order-row > td')].filter(e=>e.getBoundingClientRect().width);
   const statuses=[...table.querySelectorAll('.order-row > td:nth-child(7)')];
   const widestBadge=Math.max(...statuses.flatMap(c=>[...c.children].map(e=>e.getBoundingClientRect().width)));
   const statusWidth=statuses[0].getBoundingClientRect().width;
   const chips=[...table.querySelectorAll('.order-blockers-cell .readiness-chip-label')];
   return {width:innerWidth,desktop,fitted,stack,controlsTop:r.top,tableWidth:t.width,containerWidth:bounds.width,scrollWidth:scroller.scrollWidth,statusWidth,widestBadge,
    controlsVisible: !desktop || r.top>=stack-1,
    allColumnsFit:!fitted || (t.right<=bounds.right+1&&t.left>=bounds.left-1&&scroller.scrollWidth<=scroller.clientWidth+1&&cells.every(e=>{const b=e.getBoundingClientRect();return b.left>=bounds.left-1&&b.right<=bounds.right+1&&e.scrollWidth<=e.clientWidth+1;})),
    statusContentSized:!fitted || statusWidth<=widestBadge+14,
    blockersAtMostTwoLines:!fitted || chips.every(e=>e.getBoundingClientRect().height<=2*parseFloat(getComputedStyle(e).lineHeight)+1),
    scrollAffordance: !desktop || (fitted ? !hint.includes('Scroll sideways') : hint.includes('Scroll sideways')&&scroller.scrollWidth>scroller.clientWidth),
    pageContained:document.documentElement.scrollWidth<=innerWidth,
    mobileCards:desktop || getComputedStyle(document.querySelector('.order-row')).display==='grid'};
  });
  await page.screenshot({path:path.join(out,`${width}-${theme}-left.png`)});
  const reachable = await page.evaluate(()=>{
   const table=document.querySelector('#orders-table-container table');
   const scroller=getComputedStyle(table).overflowX==='auto'?table:table.parentElement;
   // Fitted tables must show Blockers BEFORE any horizontal scrolling.
   if(innerWidth>768&&innerWidth<=1100)scroller.scrollLeft=scroller.scrollWidth;
   const r=scroller.getBoundingClientRect();
   return [...document.querySelectorAll('.order-row > .order-blockers-cell')].filter(e=>e.getBoundingClientRect().width).every(e=>{
    const b=e.getBoundingClientRect();return b.left>=r.left-1&&b.right<=r.right+1&&[...e.querySelectorAll('.readiness-chip')].every(c=>c.scrollWidth<=c.clientWidth+1);
   });
  });
  await page.screenshot({path:path.join(out,`${width}-${theme}-right.png`)});
  results.push({theme,...result,blockersReachable:reachable});
  await context.close();
 }
} finally { await browser.close();await server.close(); }
await fs.writeFile(path.join(out,'results.json'),JSON.stringify(results,null,2));
console.log(JSON.stringify(results,null,2));
const checks=['controlsVisible','allColumnsFit','statusContentSized','blockersAtMostTwoLines','scrollAffordance','pageContained','mobileCards','blockersReachable'];
if(results.some(r=>checks.some(k=>!r[k])))process.exitCode=1;

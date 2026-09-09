// Focused regression for desktop table overflow and scrolled sort controls.
// node tests/visual/run-sales-orders-layout.mjs [dashboard-root] [output-dir]
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { installApiStub, buildTokenTable } from './lib/stub.mjs';
import { SCREENS } from './lib/screens.mjs';
const root = path.resolve(process.argv[2] || 'dashboard');
const out = path.resolve(process.argv[3] || 'docs/design/audit/screenshots/sales-orders-layout');
await fs.mkdir(out, { recursive:true });
const server = await startStaticServer(root);
const browser = await chromium.launch();
const results = [];
try {
 for (const width of [1385,1440,390]) for (const theme of ['light','dark']) {
  const context = await browser.newContext({viewport:{width,height:900},colorScheme:theme});
  await context.addInitScript(t=>localStorage.setItem('dashboard-theme',t),theme);
  await installApiStub(context,buildTokenTable(),{fail:[],overrides:{},status:{}});
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
   const desktop=innerWidth>1100;
   const stack=['.site-nav','.app-header','.tab-bar'].map(s=>document.querySelector(s)).filter(e=>['sticky','fixed'].includes(getComputedStyle(e).position)&&e.getBoundingClientRect().height).reduce((n,e)=>Math.max(n,e.getBoundingClientRect().bottom),0);
   const r=tools.getBoundingClientRect();
   const scroller=getComputedStyle(table).overflowX==='auto'?table:table.parentElement;
   const hint=getComputedStyle(tools,'::after').content;
   return {desktop,stack,controlsTop:r.top,controlsBottom:r.bottom,tableWidth:table.clientWidth,scrollWidth:scroller.scrollWidth,containerWidth:scroller.clientWidth,
    controlsVisible: !desktop || r.top>=stack-1,
    scrollAffordance:!desktop || (hint.includes('Scroll sideways')&&scroller.scrollWidth>scroller.clientWidth),
    pageContained:document.documentElement.scrollWidth<=innerWidth,
    mobileCards:desktop || getComputedStyle(document.querySelector('.order-row')).display==='grid'};
  });
  await page.screenshot({path:path.join(out,`${width}-${theme}-left.png`)});
  const reachable = await page.evaluate(()=>{
   const table=document.querySelector('#orders-table-container table');
   const scroller=getComputedStyle(table).overflowX==='auto'?table:table.parentElement;
   scroller.scrollLeft=scroller.scrollWidth;
   const r=scroller.getBoundingClientRect();
   return [...document.querySelectorAll('.order-row > .order-blockers-cell')].filter(e=>e.getBoundingClientRect().width).every(e=>{
    const b=e.getBoundingClientRect();return b.left>=r.left-1&&b.right<=r.right+1&&[...e.querySelectorAll('.readiness-chip')].every(c=>c.scrollWidth<=c.clientWidth+1);
   });
  });
  await page.screenshot({path:path.join(out,`${width}-${theme}-right.png`)});
  results.push({width,theme,...result,blockersReachable:reachable});
  await context.close();
 }
} finally { await browser.close();await server.close(); }
await fs.writeFile(path.join(out,'results.json'),JSON.stringify(results,null,2));
console.log(JSON.stringify(results,null,2));
if(results.some(r=>!r.controlsVisible||!r.scrollAffordance||!r.pageContained||!r.mobileCards||!r.blockersReachable))process.exitCode=1;

// Real browser -> real local HTTP server -> application routes -> local ledger.
// No context.route/page.route interception. The server observes actual headers.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import net from 'node:net';
import path from 'node:path';
import {chromium} from 'playwright';

const out=path.resolve(process.argv[2] || 'work/lot-history-browser');
await fs.mkdir(out,{recursive:true});
const socket=net.createServer();
await new Promise(resolve=>socket.listen(0,'127.0.0.1',resolve));
const port=socket.address().port;
await new Promise(resolve=>socket.close(resolve));
const origin=`http://127.0.0.1:${port}`;
const proc=spawn('.venv-test/bin/python',['tests/visual/lot_history_server.py',String(port)],{
  env:{...process.env,TEST_DATABASE_URL:'postgresql://localhost:5432/factory_ledger_test',PYTHONWARNINGS:'ignore'},stdio:['ignore','pipe','pipe']});
let serverLog='';proc.stdout.on('data',d=>serverLog+=d);proc.stderr.on('data',d=>serverLog+=d);
let browser;
try {
  let info;
  for(let i=0;i<100;i++) {
    try {info=await (await fetch(origin+'/__lot_history_test')).json();break;} catch {}
    if(proc.exitCode!=null)throw new Error(serverLog);
    await new Promise(r=>setTimeout(r,100));
  }
  assert(info,'local application started');
  browser=await chromium.launch();
  for(const width of [1440,820,390]) for(const theme of ['light','dark']) {
    const context=await browser.newContext({viewport:{width,height:1000},hasTouch:true,acceptDownloads:true});
    await context.addInitScript(t=>{
      localStorage.setItem('dashboard-theme',t);
      localStorage.setItem('fl-person',JSON.stringify({id:'Michael'}));
    },theme);
    const page=await context.newPage();
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin+'/dashboard/index.html?section=operations',{waitUntil:'domcontentloaded'});
    const history=page.locator('#ingredients-panels .lot-history-action').filter({hasText:'Lot history'}).first();
    await history.waitFor({state:'attached'});
    // Expand the normal inventory panel with one tap, then follow its link.
    const parentId=await history.locator('xpath=ancestor::div[contains(@class,"collapsible-body")]').getAttribute('id');
    const header=page.locator(`[data-panel="${parentId}"]`);
    if(!(await header.getAttribute('class')).includes('expanded'))await header.tap();
    const box=await history.boundingBox();assert(box.height>=44 && box.width>=44);
    await history.tap();
    await page.getByRole('heading',{name:'Lot history — Oats'}).waitFor();
    const rows=page.locator('#historyContent tbody tr');
    assert.equal(await rows.count(),2,'found and fully consumed received lots');
    assert((await rows.first().innerText()).includes('Found inventory'));
    const old=page.getByRole('link',{name:'LH-BROWSER-OATS',exact:true});
    assert((await old.locator('xpath=ancestor::tr').innerText()).includes('0 lb'));
    await page.screenshot({path:path.join(out,`history-${width}-${theme}.png`),fullPage:true});
    await old.tap();
    await page.locator('#traceDetail .audit-section').first().waitFor();
    const text=await page.locator('#traceDetail').innerText();
    for(const tid of info.makes)assert(text.includes(`make #${tid}`),'separate identical makes');
    assert(text.includes(`pack #${info.packed}`) && text.includes('Audit Customer') && text.includes('SO-AUDIT'));
    assert(text.includes('LH-BROWSER-FINISHED') && text.includes('60 lb'));
    assert.equal(await page.locator('#traceDetail script').count(),0);
    assert(text.includes('Supplier <script>bad</script>'),'receipt values stay literal text');
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:path.join(out,`forward-${width}-${theme}.png`),fullPage:true});
    // Finished-product entry must open both directions.
    await page.goto(origin+'/dashboard/index.html?section=operations',{waitUntil:'domcontentloaded'});
    const finished=page.locator(`#finished-goods-panels .lot-history-action[href*="product_id=${info.finished[0]}"]`);
    await finished.waitFor({state:'attached'});
    const panel=finished.locator('xpath=ancestor::div[contains(@class,"collapsible-body")]');
    if(!(await panel.getAttribute('class')).includes('expanded')) {
      const id=await panel.getAttribute('id');await page.locator(`[data-panel="${id}"]`).tap();
    }
    // The existing lot detail fetch also must carry the key on real HTTP.
    await finished.locator('xpath=ancestor::tr').locator('td.num').first().tap();
    await page.locator('#finished-goods-panels .lot-link[data-lot="LH-BROWSER-FINISHED"]').tap();
    await page.locator('#lot-panel-body .loading-indicator').waitFor({state:'detached'});
    const lotPanelText=await page.locator('#lot-panel-body').innerText();
    assert(!/HTTP 40[13]|failed|error|not defined/i.test(lotPanelText),lotPanelText);
    await page.locator('#lot-panel-close').tap();
    await finished.tap();
    await page.getByRole('link',{name:'LH-BROWSER-FINISHED',exact:true}).tap();
    await page.locator('#traceDetail .audit-section').first().waitFor();
    const backward=await page.locator('#traceDetail').innerText();
    assert(backward.includes('Backward — recorded inputs') && backward.includes('Forward — shipments'));
    assert(backward.includes('Estimated share') && backward.includes('not physically verified'));
    assert(backward.includes('37.5 lb'),'75/100 of each 50 lb make');
    for(const tid of info.makes)assert(backward.includes(`make #${tid}`));
    const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'Download Text Report'}).tap()]);
    const exported=await fs.readFile(await download.path(),'utf8');
    assert(exported.includes('37.5 lb') && exported.includes('Audit Customer'));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    if(width===1440 && theme==='light') {
      await fs.writeFile(path.join(out,'audit-report.txt'),exported);
      await page.pdf({path:path.join(out,'audit-report.pdf'),format:'A4',printBackground:true});
    }
    await page.screenshot({path:path.join(out,`backward-${width}-${theme}.png`),fullPage:true});
    assert.deepEqual(errors,[],'no uncaught browser errors');
    await context.close();
  }
  const observed=(await (await fetch(origin+'/__lot_history_test')).json()).requests;
  for(const required of ['/dashboard/api/inventory/ingredients','/dashboard/api/inventory/finished-goods',
    `/dashboard/api/inventory/${info.oats[0]}/lot-history`,`/dashboard/api/inventory/${info.finished[0]}/lot-history`,
    '/dashboard/api/lot-report/LH-BROWSER-OATS','/dashboard/api/lot-report/LH-BROWSER-FINISHED','/audit/integrity','/dashboard/api/lot/LH-BROWSER-FINISHED']) {
    assert(observed.some(r=>r.path===required && r.status===200),`real server saw ${required}`);
  }
  assert(observed.length>20);
  assert(observed.every(r=>r.key_ok),`missing key: ${JSON.stringify(observed.filter(r=>!r.key_ok))}`);
  console.log(`PASS: ${observed.length} actual HTTP requests carried X-API-Key; touch workflows, full trace, text/PDF and responsive layouts at 1440/820/390 in both themes.`);
} finally {
  await browser?.close();
  const exited=new Promise(resolve=>proc.once('exit',resolve));
  if(proc.exitCode===null){proc.kill('SIGTERM');await exited;}
  await fs.writeFile(path.join(out,'server.log'),serverLog);
}

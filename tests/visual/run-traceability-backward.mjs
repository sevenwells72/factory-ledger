// Render the versioned audit shape and reject stale/incomplete API responses.
// Separate from the non-stubbed default-suite HTTP/header test.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {startStaticServer} from './lib/server.mjs';
import {API_HOST,FIXTURE_DASHBOARD_KEY,buildTokenTable,installApiStub} from './lib/stub.mjs';
const out=path.resolve(process.argv[2] || 'work/traceability-backward');
await fs.mkdir(out,{recursive:true});
const fixture=JSON.parse(await fs.readFile('tests/visual/fixtures/trace-BB110927.json','utf8'));
Object.assign(fixture,{report_version:1,complete:true,origin:fixture.input_tree});
const server=await startStaticServer(path.resolve('dashboard')),browser=await chromium.launch();
async function open(width,theme,payload,delay=0) {
  const context=await browser.newContext({viewport:{width,height:1000},acceptDownloads:true});
  await context.addInitScript(t=>localStorage.setItem('dashboard-theme',t),theme);
  await installApiStub(context,buildTokenTable(),{fail:[],overrides:{},status:{},unmatched:new Set()});
  const requests=[];
  context.on('request',req=>{const u=new URL(req.url());if(u.host===API_HOST)requests.push(u.pathname+u.search);});
  await context.route(`**://${API_HOST}/dashboard/api/lot-report/**`,async route=>{
    assert.equal(await route.request().headerValue('X-API-Key'),FIXTURE_DASHBOARD_KEY);
    if(delay)await new Promise(r=>setTimeout(r,delay));
    await route.fulfill({json:payload});
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.origin+'/traceability.html?lot=BB110927&direction=backward&product_id=145');
  return {context,page,errors,requests,setPayload:value=>{payload=value;}};
}
try {
  for(const width of [1440,390])for(const theme of ['light','dark']) {
    const {context,page,errors,requests}=await open(width,theme,fixture,width===1440 && theme==='light' ? 16000 : 500);
    await page.getByText(/Building complete report for BB110927/).waitFor();
    await page.locator('#traceDetail .audit-section').first().waitFor();
    const panel=await page.locator('#traceDetail').innerText();
    for(const word of ['make #2947','3,855 lb','Trace gaps: 2','Nation Harvest','National Harvest','not physically verified'])assert(panel.includes(word),word);
    const rows=page.locator('#traceDetail .audit-section').first().locator('tbody tr');
    assert.equal(await rows.filter({hasText:'make #2947'}).count(),7);
    assert(!(await page.locator('#traceDetail').innerText()).includes('undefined'));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    assert.deepEqual(errors,[]);
    assert(!requests.some(p=>p.startsWith('/trace/') || /transaction_type=/.test(p)),'no legacy trace/history-scan calls');
    await page.locator('#detailPanel').screenshot({path:path.join(out,`detail-${width}-${theme}.png`)});
    await context.close();
  }
  {
    const {context,page,setPayload}=await open(1440,'light',fixture);
    await page.locator('#traceDetail .audit-section').first().waitFor();
    setPayload({ingredients:[]});
    await page.evaluate(()=>runTrace());
    assert((await page.locator('#statusBar').innerText()).includes('complete audit report format'));
    assert.equal(await page.locator('#graphSvg').locator('*').count(),0,'failed next report cannot retain previous graph');
    assert.equal(await page.locator('#traceInfo').innerText(),'');
    assert.equal(await page.locator('#detailPanel').isVisible(),false);
    await context.close();
  }
  for(const [payload,expected] of [[{ingredients:[]},'complete audit report format'],[{...fixture,complete:false,incomplete_reason:'Result cap reached'},'Result cap reached']]) {
    const {context,page,requests,errors}=await open(1440,'light',payload);
    await page.locator('#statusBar.error').waitFor();
    assert((await page.locator('#statusBar').innerText()).includes(expected));
    assert.equal(await page.locator('#detailPanel').isVisible(),false);
    assert(!requests.some(p=>p.startsWith('/trace/') || /transaction_type=/.test(p)));
    assert.deepEqual(errors,[]);
    await context.close();
  }
  console.log('PASS: complete backward reports, visible building state, seven inputs for 2947, two gaps, responsive themes; stale/incomplete responses show errors with no legacy fallback.');
} finally {await browser.close();await server.close();}

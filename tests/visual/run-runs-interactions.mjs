import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { CHECK_SOURCE } from './lib/checks.mjs';

const SCREENS = {runs:{path:'/runs.html',region:'#runs-screen'},dialog:{region:'#run-dialog'}};
const out = path.resolve('docs/design/audit/pr-screenshots/feat-runs-screen');
await fs.mkdir(out, {recursive:true});
const server = await startStaticServer(path.resolve('dashboard'));
const browser = await chromium.launch();
const fixture=JSON.parse(await fs.readFile(new URL('./fixtures/production-runs.json',import.meta.url),'utf8'));
const variants = [1440,390].flatMap(width => ['light','dark'].map(theme => ({width,theme})));
try {
  if (process.argv.includes('--before')) {
    for (const {width,theme} of variants) {
      const page = await browser.newPage({viewport:{width,height:900}});
      const response = await page.goto(server.origin + '/runs.html');
      assert.equal(response.status(),404);
      await page.screenshot({path:path.join(out,`before-${width}-${theme}.png`)});
      await page.close();
    }
    await fs.writeFile(path.join(out,'before.md'),'# Before implementation\n\nBase: 7c183e0. `/runs.html` does not exist (404). Eight STATUS checks: N/A, not a passing baseline. Captures at 1440/390 in light/dark record the missing screen. Existing screen files are unchanged.\n');
    console.log('Before: four captures; runs screen absent, STATUS checks N/A.');
  } else { await auditRuns(); }
} finally { await browser.close(); await server.close(); }

// Registration is local to this runner; no existing screen entry changes.

async function auditRuns() {
  const pass=process.argv.find(x=>x.startsWith('--pass='))?.split('=')[1]||'1';
  const records=[], checks=[], errors=[];
  const ruleMethods={'STATUS-002':'nominalBadges','STATUS-004':'alarmsPerRow','STATUS-005':'explainHooks','STATUS-006':'numberFormat','STATUS-007':'orphanPlaceholders','STATUS-008':'rowHeights','STATUS-010':'devVocabulary','STATUS-011':'repeatedSentences'};
  async function capture(page,name,width,theme,region='#runs-screen') {
    if(name!=='quantity-explanation')await page.evaluate(()=>window.SOList.closeExplanation());
    await page.addScriptTag({content:CHECK_SOURCE});
    const rules=await page.evaluate(({region,width,methods})=>Object.fromEntries(Object.entries(methods).map(([rule,method])=>[rule,window.__FL_AUDIT[method](region,{width})])),{region,width,methods:ruleMethods});
    records.push({name,width,theme,rules});
    for(const [rule,result] of Object.entries(rules)) if(result.failures)errors.push(`${name}/${width}/${theme} ${rule}: ${JSON.stringify(result.worst)}`);
    const target=path.join(out,`after-pass${pass}-${name}-${width}-${theme}.png`);
    await page.screenshot({path:target});
    assert.ok((await fs.stat(target)).size<1000000,'Evidence must be smaller than 1 MB');
  }
  for(const {width,theme} of variants) {
    const context=await browser.newContext({viewport:{width,height:900},timezoneId:'Pacific/Honolulu',colorScheme:theme});
    await context.addInitScript(({theme})=>{
      localStorage.setItem('dashboard-theme',theme);
      // Sunday on the browser's clock, Monday in the factory: date must use factory time.
      const RealDate=Date; window.Date=class extends RealDate {constructor(...args){super(...(args.length?args:['2026-09-14T05:30:00Z']));}static now(){return new RealDate('2026-09-14T05:30:00Z').valueOf();}};
    },{theme});
    const state=structuredClone(fixture), writes=[], reads=[], unexpected=[];
    let nextError=null, listDelay=0, searchDelay=0;
    const respond=(route,data,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
    await context.route('**/*',async route=>{
      const req=route.request(),url=new URL(req.url()),method=req.method(),pathname=url.pathname;
      if(url.origin===server.origin)return route.continue();
      if(url.hostname!=='fastapi-production-b73a.up.railway.app'){unexpected.push(req.url());return route.abort();}
      assert.equal(req.headers()['x-api-key'],'dashboard-key-2026');
      if(nextError&&nextError.method===method&&pathname.includes(nextError.path)) {const e=nextError;nextError=null;return respond(route,{detail:{error_code:e.code,sales_order_line_id:e.line,message:'API JSON payload undefined qty_lb'}},e.status||409);}
      if(method==='GET')reads.push(pathname+url.search);
      if(pathname==='/products/search'){
        const delay=searchDelay;searchDelay=0;if(delay)await new Promise(resolve=>setTimeout(resolve,delay));
        return respond(route,{products:state.products,count:state.products.length});
      }
      if(pathname==='/sales/orders'&&method==='GET'){
        assert.equal(url.searchParams.get('state'),'open');assert.equal(url.searchParams.get('limit'),'200');
        const orders=state.orders.filter(o=>!url.searchParams.get('customer')||o.customer.toLowerCase().includes(url.searchParams.get('customer').toLowerCase()));
        return respond(route,{orders,count:orders.length});
      }
      if(/^\/sales\/orders\/\d+$/.test(pathname)&&method==='GET')return respond(route,state.details[pathname.split('/').pop()]);
      if(pathname==='/production/runs'&&method==='GET'){
        const delay=listDelay;listDelay=0;if(delay)await new Promise(resolve=>setTimeout(resolve,delay));
        const runs=state.runs.filter(r=>(!url.searchParams.get('from')||r.planned_date>=url.searchParams.get('from'))&&(!url.searchParams.get('to')||r.planned_date<=url.searchParams.get('to'))&&(!url.searchParams.get('status')||r.status===url.searchParams.get('status'))&&(!url.searchParams.get('product')||r.product_id===Number(url.searchParams.get('product'))));
        return respond(route,{runs,count:runs.length});
      }
      if(pathname.startsWith('/production/runs')) {
        const id=Number(pathname.split('/')[3]),run=state.runs.find(r=>r.id===id),action=pathname.split('/')[4];
        if(method==='GET'&&action==='evidence')return respond(route,state.evidence[id]);
        const body=req.postDataJSON();writes.push({method,pathname,body});
        if(method==='POST'&&pathname==='/production/runs') {
          const product=state.products.find(p=>p.id===body.product_id);
          const newRun={id:Math.max(...state.runs.map(r=>r.id))+1,...body,product_name:product.name,sku:product.odoo_code,planned_qty_lb:body.planned_qty*(body.planned_unit==='cases'?product.case_size_lb:1),case_size_lb_used:body.planned_unit==='cases'?product.case_size_lb:null,status:'planned',covered_lb:0,coverage:[],line_id:body.line_id??1,line_name:'Line A'};
          state.runs.push(newRun);return respond(route,{run_id:newRun.id,run:newRun},201);
        }
        if(method==='PATCH'&&run&&!action) {
          Object.assign(run,body);
          if(body.planned_qty!=null||body.planned_unit!=null)run.planned_qty_lb=run.planned_qty*(run.planned_unit==='cases'?25:1);
        }else if(method==='POST'&&action==='cancel') {run.status='cancelled';run.notes=body.reason;}
        else if(method==='POST'&&action==='complete')run.status='done';
        else if(method==='PUT'&&action==='coverage') {run.coverage=body.coverage.map(c=>({...c,order_number:c.sales_order_line_id===9901?'SO-199':'SO-'+Math.floor(c.sales_order_line_id/10)}));run.covered_lb=run.coverage.reduce((s,c)=>s+c.qty_lb,0);}
        else {unexpected.push(method+' '+pathname);return route.abort();}
        return respond(route,{run_id:id,run,evidence:state.evidence[id]});
      }
      unexpected.push(method+' '+pathname);return route.abort();
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(`${width}/${theme} browser error: ${e.message}`));
    try {
      await page.goto(server.origin+'/runs.html');
      await page.locator('.run-row').first().waitFor();
      assert.equal(await page.locator('.run-row').count(),5);
      if(width>=1200)assert.ok(await page.locator('.run-row').evaluateAll(rows=>rows.every(r=>r.getBoundingClientRect().height<=56)));
      assert.ok(reads.includes('/production/runs?from=2026-09-14&to=2026-09-20'));
      assert.equal(await page.locator('[data-run-id="1"] .run-quantity').innerText(),'40 cases');
      assert.equal(await page.locator('[data-run-id="2"] .run-quantity').innerText(),'13,500 lb');
      await capture(page,'week',width,theme);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'No page overflow');
      for(const el of await page.locator('.run-row [data-explain]').all()) {
        assert.equal(await el.getAttribute('aria-describedby'),await el.getAttribute('data-explain'));
      }
      const quantity=page.locator('[data-run-id="1"] .run-quantity');
      await quantity.click(); const explainId=await quantity.getAttribute('data-explain');
      assert.ok((await page.locator('#'+explainId).innerText()).includes('1,000 lb'));
      await capture(page,'quantity-explanation',width,theme);
      await page.keyboard.press('Escape'); await quantity.focus(); await page.keyboard.press('Escape');assert.equal(await page.locator('#'+explainId).isVisible(),false);
      for(const tab of ['Planned','In progress','Done','Cancelled']) {
        await page.getByRole('tab',{name:new RegExp('^'+tab)}).click();
        assert.equal(await page.locator('.run-row').count(),tab==='Planned'?2:1);
        await capture(page,tab.toLowerCase().replaceAll(' ','-'),width,theme);
      }
      await page.getByRole('tab',{name:/^This week/}).click();
      listDelay=350;await page.getByRole('button',{name:'Previous week',exact:true}).click();
      await page.getByRole('button',{name:'Next week',exact:true}).click();
      await page.locator('.run-row').first().waitFor();await page.waitForTimeout(450);assert.equal(await page.locator('.run-row').count(),5);
      nextError={method:'GET',path:'/production/runs',status:503};await page.locator('#current-week').click();await page.locator('#retry-load').waitFor();
      await capture(page,'load-error',width,theme);await page.locator('#retry-load').click();await page.locator('.run-row').first().waitFor();
      const open=async id=>{await page.locator(`[data-open="${id}"]`).click();};
      const close=async()=>{await page.locator('#close-dialog').click();};
      const save=async()=>{await page.locator('#save-run').click();await page.locator('#run-dialog').waitFor({state:'hidden'});await page.locator('.run-row').first().waitFor();};
      await open(4);assert.equal(await page.locator('#edit-run').count(),0);await close();
      await open(5);assert.equal(await page.locator('#complete-run').count(),0);await close();
      await page.locator('#new-run').click();await page.locator('#product-search').fill('granola');await page.locator('[data-product]').first().waitFor();
      assert.equal(await page.locator('[data-product]').count(),2);
      await page.locator('[data-product="1"]').click();assert.equal(await page.locator('#planned-unit option[value="cases"]').evaluate(el=>el.disabled),true);
      await capture(page,'new-bulk',width,theme,'#run-dialog');
      await page.locator('#product-search').fill('classic');await page.locator('[data-product]').first().waitFor();await page.locator('[data-product="0"]').click();
      await page.locator('#planned-qty').fill('60');await page.locator('#planned-unit').selectOption('cases');await page.locator('#planned-notes').fill('Afternoon packing.');
      nextError={method:'POST',path:'/production/runs',code:'CASE_WEIGHT_REQUIRED'};
      await page.locator('#save-run').click();await page.getByText('This product has no case weight. Choose pounds to plan this run.').waitFor();
      await capture(page,'case-error',width,theme,'#run-dialog');assert.equal(await page.locator('#planned-qty').inputValue(),'60');
      await save();assert.equal(state.runs.at(-1).planned_qty_lb,1500);assert.equal(writes.at(-1).method,'POST');
      await open(6);await page.locator('#edit-run').click();assert.equal(await page.locator('#planned-line').inputValue(),'1');
      await page.locator('#planned-status').selectOption('in_progress');await page.locator('#planned-notes').fill('Started.');
      await capture(page,'edit',width,theme,'#run-dialog');await save();assert.equal(state.runs.at(-1).status,'in_progress');assert.equal(writes.at(-1).body.planned_qty,undefined,'Unchanged cases must not be reconverted');
      await open(6);await page.locator('#cancel-run').click();await page.locator('#cancel-reason').fill('Moved to tomorrow.');
      await capture(page,'cancel',width,theme,'#run-dialog');await save();assert.equal(state.runs.at(-1).status,'cancelled');
      await open(1);await page.locator('#edit-coverage').click();await page.locator('[data-cover="1101"]').waitFor();
      assert.equal(await page.locator('[data-cover]').count(),3,'Only open physical matching lines, plus retained link');
      assert.ok((await page.locator('[data-line-id="1101"]').innerText()).includes('800 lb remaining'));
      assert.equal(await page.locator('[data-cover="9901"]').inputValue(),'100');
      await page.locator('[data-cover="1101"]').fill('800');await page.locator('[data-cover="1201"]').fill('800');
      await page.locator('#save-run').click();await page.getByText('Coverage exceeds the planned pounds.',{exact:false}).waitFor();
      await capture(page,'overcoverage',width,theme,'#run-dialog');
      await page.locator('[data-cover="1101"]').fill('801');await page.locator('[data-cover="1201"]').fill('0');
      await page.locator('#save-run').click();await page.locator('.line-error').filter({hasText:'Coverage is greater'}).waitFor();
      await capture(page,'line-error',width,theme,'#run-dialog');
      await page.locator('[data-cover="1101"]').fill('700');
      nextError={method:'PUT',path:'/coverage',code:'COVERAGE_EXCEEDS_REMAINING',line:1101};await page.locator('#save-run').click();await page.locator('.line-error').filter({hasText:'Coverage is greater'}).waitFor();
      assert.equal(await page.locator('[data-cover="1101"]').inputValue(),'700');await save();
      assert.deepEqual(writes.at(-1).body.coverage,[{sales_order_line_id:1101,qty_lb:700},{sales_order_line_id:9901,qty_lb:100}]);
      await open(1);await page.locator('#edit-coverage').click();await page.locator('[data-cover="1101"]').waitFor();
      await page.locator('#coverage-customer').fill('South');await page.locator('#coverage-reload').click();await page.locator('[data-cover="1201"]').waitFor();
      assert.equal(await page.locator('[data-cover="1101"]').inputValue(),'700','Narrowing preserves existing coverage');
      await capture(page,'coverage',width,theme,'#run-dialog');
      for(const input of await page.locator('[data-cover]').all())await input.fill('0');await save();assert.deepEqual(writes.at(-1).body,{coverage:[]});
      for(const [id,kind,label] of [[1,'partial','Confirm anyway'],[2,'none','Confirm anyway'],[3,'looks-complete','Looks complete — Confirm']]) {
        await open(id);await page.locator('#complete-run').click();await page.getByRole('button',{name:label,exact:true}).waitFor();
        await capture(page,'evidence-'+kind,width,theme,'#run-dialog');
        assert.equal(await page.getByText('Completing this run creates no inventory, changes no sales order, and does not set Ready to Ship.',{exact:true}).count(),1);
        await save();assert.equal(state.runs.find(r=>r.id===id).status,'done');
      }
      assert.equal(unexpected.length,0,unexpected.join('\n'));
      assert.ok(writes.every(w=>w.pathname.startsWith('/production/runs')),'Never write to SO or inventory');
      for(const method of ['POST','PATCH','PUT'])assert.ok(writes.some(w=>w.method===method));
      assert.ok(reads.some(r=>r==='/sales/orders/101'));
      checks.push(`${width}-${theme}: PASS — factory date, week races, tabs, quantities, hooks, errors, create/edit/cancel, full replacement/clear/preserve, all completion states, authentication, no SO writes.`);
    } catch(e) {errors.push(`${width}-${theme} interaction: ${e.stack}`);await page.screenshot({path:path.join(out,`failure-pass${pass}-${width}-${theme}.png`)});}
    finally {await context.close();}
  }
  const totals=Object.fromEntries(Object.keys(ruleMethods).map(rule=>[rule,records.reduce((sum,r)=>sum+(r.rules[rule].failures||0),0)]));
  const table='| Rule | Before | After failures |\n|---|---|---:|\n'+Object.entries(totals).map(([r,total])=>`| ${r} | N/A — absent page | ${total} |`).join('\n');
  await fs.writeFile(path.join(out,`review-pass${pass}.md`),`# Production Runs review pass ${pass}\n\n${table}\n\n${checks.join('\n\n')}\n\n${records.length} captures checked with the existing shared STATUS implementation. Baseline is an absent page; zero is not claimed for it.\n\n${errors.length?'## Failures\n\n'+errors.map(e=>'```\n'+e+'\n```').join('\n\n'):'No STATUS or interaction failures.'}\n`);
  console.log(table+'\n'+checks.join('\n')+'\n'+errors.join('\n'));
  if(errors.length)process.exitCode=1;
}

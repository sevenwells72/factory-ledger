// PR review checks on actual local pages and FL handlers; no model or hosted traffic.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const python=process.env.PILOT_UX_PYTHON || path.join(root,'.venv-test/bin/python');
const pg=process.env.PILOT_UX_PG_BIN || '/opt/homebrew/opt/postgresql@17/bin';
const pgPort=String(process.env.PILOT_UX_PG_PORT || '5432');
assert.match(pgPort,/^\d+$/);
const out=process.env.PILOT_UX_REVIEW_OUT || '/tmp/fl-pilotux-review';
const env={...process.env};
for(const name of Object.keys(env))if(['DATABASE_URL','OPENAI_API_KEY','ANTHROPIC_API_KEY','PGHOSTADDR','PGSERVICE','PGSERVICEFILE','PGOPTIONS'].includes(name)||name.startsWith('SUPABASE_'))delete env[name];
await fs.mkdir(out,{recursive:true});
const report={created_at:new Date().toISOString(),local_only:true,actual_handlers:true,openai_calls:0,records:0,width:390,results:[],databases_dropped:[]};
const db=`fl_pilotux_review_${process.pid}`;
const pgRun=(command,args)=>execFileSync(path.join(pg,command),['-h','127.0.0.1','-p',pgPort,...args],{cwd:root,env,stdio:'pipe'});
let child,browser,diagnostic='';
const save=()=>fs.writeFile(path.join(out,'results.json'),JSON.stringify(report,null,2));
pgRun('createdb',['-E','UTF8','-T','template0',db]);
try{
  pgRun('psql',['-d',db,'-v','ON_ERROR_STOP=1','-q','-c','CREATE EXTENSION pg_trgm']);
  pgRun('psql',['-d',db,'-v','ON_ERROR_STOP=1','-q','-f',path.join(root,'tests/schema/schema.sql')]);
  // Synthetic pouch packaging and already-extracted documents; extraction is never run.
  pgRun('psql',['-d',db,'-v','ON_ERROR_STOP=1','-q','-c',`
    INSERT INTO products(id,name,odoo_code,type,uom,pack_format,case_size_lb,bags_per_case,units_per_case,retail_bag_oz,active)
      VALUES(501,'Pilot Review Pouches','PR142POUCH','finished','lb','bagged',7.5,12,12,10,true);
    INSERT INTO purchase_documents(id,storage_path,original_filename,mime_type,file_sha256,byte_size,status,document_kind,extraction)
      VALUES(9101,'pilot-review/sales','pilot-review-sales.png','image/png','pilot-review-sales',10,'extracted','sales',
        '{"customer_name":"Setton Farms","po_number":"PRACTICE-REVIEW","lines":[{"description":"Practice cases","quantity":3,"unit":"cases","quantity_lb":75},{"description":"Practice boxes","quantity":4,"unit":"boxes","quantity_lb":100}]}'),
      (9102,'pilot-review/purchase','pilot-review-purchase.png','image/png','pilot-review-purchase',10,'extracted','purchase',
        '{"supplier_name":"Grain Supply","lines":[{"description":"Practice cases","quantity":3,"unit":"cases","expected_qty_lb":75},{"description":"Practice boxes","quantity":4,"unit":"boxes","expected_qty_lb":100}]}');
  `]);
  child=spawn(python,['-u',path.join(root,'tests/visual/pilot_ux_page_bridge.py')],{cwd:root,env:{...env,TEST_DATABASE_URL:`postgresql://127.0.0.1:${pgPort}/${db}`,KEY_SESSIONS_ENABLED:'1'},stdio:['ignore','pipe','pipe']});
  child.stderr.on('data',c=>{diagnostic+=c;});
  const lines=readline.createInterface({input:child.stdout});
  const ready=await Promise.race([new Promise((resolve,reject)=>{lines.once('line',line=>{try{resolve(JSON.parse(line));}catch{reject(Error(line));}});child.once('exit',code=>reject(Error(`Bridge exited ${code}: ${diagnostic}`)));}),new Promise((_,reject)=>{const timer=setTimeout(()=>reject(Error('Bridge startup timed out')),30000);timer.unref();})]);
  assert.equal(ready.openai_forbidden,true);
  const origin=`http://127.0.0.1:${ready.port}`;
  const fixture=await(await fetch(origin+'/test/fixture')).json();
  browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL || 'chrome'});
  for(const initialLanguage of ['en','es']){
    const context=await browser.newContext({viewport:{width:390,height:844},timezoneId:'America/New_York',colorScheme:'light'});
    await context.addInitScript(language=>sessionStorage.setItem('fl-ledger-language',language),initialLanguage);
    const page=await context.newPage();page.setDefaultTimeout(12000);
    const errors=[],remote=[],commits=[];
    page.on('pageerror',error=>errors.push(error.message));
    page.on('request',request=>{if(request.method()==='POST'&&/\/commit$/.test(new URL(request.url()).pathname))commits.push(request.url());});
    await page.route('**/*',route=>{if(new URL(route.request().url()).origin!==origin){remote.push(route.request().url());return route.abort();}return route.continue();});
    const language=()=>page.evaluate(()=>FLLedger.lang);
    const setLanguage=async expected=>{if(await language()!==expected)await page.locator('#language').click();assert.equal(await language(),expected);};
    const open=async file=>{await page.goto(origin+'/dashboard/'+file);await page.evaluate(()=>FLLedger.ready);};
    const login=async(role,file)=>{await open(file);if(!(await page.locator('#actor-key').isVisible())){await page.locator('#sign-out').click();await page.locator('#actor-key').waitFor();}await page.locator('#actor-key').fill(fixture.actors[role].key);await page.locator('#sign-in-form button').click();await page.locator('#identity').filter({hasText:fixture.actors[role].name}).waitFor();if(file==='office.html')await page.waitForFunction(()=>document.querySelector('[data-office-start]'));};
    const picker=async(name,query,match,scope=page)=>{const field=scope.locator(`[data-picker="${name}"]`).first();await field.locator('input').first().fill(query);await field.locator('.picker-options button').filter({hasText:match}).first().click();};
    const form=()=>page.locator('#entry-form,#office-form');
    const fill=async(name,value)=>form().locator(`[name="${name}"]`).first().fill(String(value));
    const prepare=async()=>{await form().getByRole('button',{name:/^(Prepare draft|Preparar borrador)$/}).click();await page.locator('#resolution').getByRole('heading',{name:/^(Draft — review before recording|Borrador — revisa antes de registrar)$/}).waitFor();};
    const cancel=()=>page.locator('#resolution').getByRole('button',{name:/^(Cancel|Cancelar)$/}).click();
    const check=async(name,fn)=>{const row={name,initial_language:initialLanguage,width:390};try{await fn();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'No phone overflow');row.passed=true;}catch(error){row.passed=false;row.error=error.stack||String(error);row.notice=await page.locator('#notice').textContent().catch(()=>null);row.screenshot=path.join(out,`FAIL-${initialLanguage}-${report.results.length}.png`);await page.screenshot({path:row.screenshot,fullPage:true});}report.results.push(row);await save();if(!row.passed)throw Error(`${name}: ${row.error}\nNotice: ${row.notice}\nBridge: ${diagnostic}`);};
    const reasons=[['physical_count','Physical count','Conteo físico'],['missing_receipt','Missing receipt','Recepción no registrada'],['missing_production','Missing production','Producción no registrada'],['wrong_lot','Wrong lot used','Lote equivocado'],['damage_disposal','Damage/disposal','Daño o desecho'],['unrecorded_usage','Unrecorded usage','Uso no registrado'],['data_entry_error','Data-entry error','Error de captura'],['unknown','Unknown','Desconocido']];
    try{
      await login('floor','entry.html');
      for(const action of ['adjust','found'])for(const [code,en,es] of reasons.filter(([code])=>action==='adjust'||['physical_count','missing_receipt','missing_production','unknown'].includes(code))){
        await check(`${action}: ${code} form and draft agree, including live language changes`,async()=>{
          await open('entry.html');await setLanguage(initialLanguage);await page.locator('#entry-action').selectOption(action);
          const select=form().locator('[name=reason_code]');
          for(const lang of ['en','es',initialLanguage]){await setLanguage(lang);assert.equal(await select.locator(`option[value="${code}"]`).textContent(),lang==='es'?es:en);}
          await select.selectOption(code);
          if(action==='adjust'){await picker('lot','oats','26-10-18-OATS-001');await fill('delta_lb',code==='missing_receipt'?1:-1);await fill('note','PRACTICE — review reason label');}
          else{await picker('product','Oats','Oats');await picker('known_lot','oats','26-10-18-OATS-001');await fill('quantity',1);await fill('notes','PRACTICE — review reason label');await fill('found_location','Practice shelf');await fill('supplier_lot_code','26-10-18-OATS-001');}
          await prepare();
          for(const lang of [initialLanguage,initialLanguage==='en'?'es':'en',initialLanguage]){await setLanguage(lang);const reasonValues=await page.locator('#resolution .draft-data dt').evaluateAll(nodes=>nodes.filter(node=>['Reason','Motivo'].includes(node.textContent)).map(node=>node.nextElementSibling.textContent));assert.ok(reasonValues.length>0);for(const value of reasonValues)assert.ok(value.startsWith(lang==='es'?es:en),`Every reason field must use its canonical ${lang} label: ${value}`);}
          await cancel();
        });
      }
      await login('office','office.html');
      await check('Office order Cases and Boxes stay distinct on form and draft in both languages',async()=>{
        await setLanguage(initialLanguage);await page.locator('#office-action').selectOption('create_order');await picker('customer','Setton','Setton Farms');await form().locator('[name=no_po]').check();
        const rows=page.locator('[data-lines]>*');await picker('product','Granola Classic 25','Granola Classic 25 LB',rows.first());await rows.first().locator('[name=quantity]').fill('2');await rows.first().locator('[name=unit]').selectOption('cases');
        await page.getByRole('button',{name:/^(Add line|Agregar línea)$/}).click();await picker('product','Granola Fruit Nut','Granola Fruit Nut 25 LB',rows.nth(1));await rows.nth(1).locator('[name=quantity]').fill('3');await rows.nth(1).locator('[name=unit]').selectOption('boxes');
        for(const lang of ['en','es',initialLanguage]){await setLanguage(lang);const unit=rows.first().locator('[name=unit]');assert.equal(await unit.locator('option[value=cases]').textContent(),lang==='es'?'Cajones':'Cases');assert.equal(await unit.locator('option[value=boxes]').textContent(),lang==='es'?'Cajas':'Boxes');}
        await prepare();for(const lang of [initialLanguage,initialLanguage==='en'?'es':'en',initialLanguage]){await setLanguage(lang);const data=page.locator('#resolution .draft-data');assert.ok(await data.getByText(lang==='es'?'Cajones':'Cases',{exact:true}).count()>0);assert.ok(await data.getByText(lang==='es'?'Cajas':'Boxes',{exact:true}).count()>0);assert.equal(await data.getByText(lang==='es'?'Cases':'Cajones',{exact:true}).count(),0);assert.equal(await data.getByText(lang==='es'?'Boxes':'Cajas',{exact:true}).count(),0);}await cancel();
      });
      await check('Expected delivery missing-input help and calculated total switch language without losing numbers',async()=>{
        await setLanguage(initialLanguage);await page.locator('#office-action').selectOption('create_expected_receipt');await form().locator('[name=expected_unit]').selectOption('cases');
        for(const lang of ['en','es',initialLanguage]){await setLanguage(lang);await form().getByText(lang==='es'?'Ingresa cajones y lb por cajón para calcular el total.':'Enter cases and lb per case to calculate the total.',{exact:true}).waitFor();}
        await fill('expected_cases',10);await fill('expected_case_size',25);
        for(const lang of ['en','es',initialLanguage]){await setLanguage(lang);await form().getByText(lang==='es'?'Total esperado: 250 lb':'Expected total: 250 lb',{exact:true}).waitFor();assert.equal(await form().locator('[name=expected_cases]').inputValue(),'10');assert.equal(await form().locator('[name=expected_case_size]').inputValue(),'25');}
        await picker('supplier','Grain Supply','Grain Supply');await picker('product','Sunflower Seeds','Sunflower Seeds');await prepare();assert.match(await page.locator('#resolution .draft-data').textContent(),/250/);await cancel();
      });
      await check('Real catalog pouch conversion switches both form and draft wording',async()=>{
        await setLanguage(initialLanguage);await page.locator('#office-action').selectOption('create_order');await picker('customer','Setton','Setton Farms');await form().locator('[name=no_po]').check();const row=page.locator('[data-lines]>*').first();await picker('product','Pilot Review Pouches','Pilot Review Pouches',row);await row.locator('[name=quantity]').fill('24');await row.locator('[name=unit]').selectOption('pouches');await prepare();
        for(const lang of ['en','es',initialLanguage]){await setLanguage(lang);const text=lang==='es'?'24 bolsitas → 2 cajones':'24 pouches → 2 cases';assert.equal(await page.locator('#resolution > p.neutral').textContent(),text);assert.equal(await row.locator('p.neutral span').textContent(),text);}await cancel();
      });
      for(const [action,id] of [['approve_so',9101],['approve_er',9102]])await check(`${action} shows distinct translated units on already-extracted document lines`,async()=>{
        await setLanguage(initialLanguage);await page.locator('#office-action').selectOption(action);await fill('document_id',id);await form().getByRole('button',{name:/^(Load extracted document|Cargar documento extraído)$/}).click();await page.locator('[data-lines]>*').nth(1).waitFor();
        for(const lang of ['en','es',initialLanguage]){await setLanguage(lang);const extracted=await page.locator('[data-lines] p').filter({hasText:/^(Extracted|Extraído):/}).allTextContents();assert.deepEqual(extracted,lang==='es'?['Extraído: 3 Cajones','Extraído: 4 Cajas']:['Extracted: 3 Cases','Extracted: 4 Boxes']);}
      });
      assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);assert.deepEqual(commits,[]);
    }finally{await context.close();}
  }
  const verified=await(await fetch(origin+'/test/verify')).json();assert.equal(verified.receipts.length,0);report.verified=verified;
}finally{
  if(browser)await browser.close();
  if(child){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{child.kill('SIGKILL');resolve();},3000).unref();});}
  pgRun('dropdb',['--if-exists',db]);report.databases_dropped.push(db);await save();
}
console.log(JSON.stringify({checks:report.results.length,passed:report.results.filter(row=>row.passed).length,failed:report.results.filter(row=>!row.passed).length,records:report.records,results:path.join(out,'results.json')}));

// Actual local photo reclassification and in-place EN/ES checks; no model or hosted traffic.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const { chromium }=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const python=process.env.PILOT_UX_PYTHON || path.join(root,'.venv-test/bin/python');
const pg=process.env.PILOT_UX_PG_BIN || '/opt/homebrew/opt/postgresql@17/bin';
const pgPort=String(process.env.PILOT_UX_PG_PORT || '5432');assert.match(pgPort,/^\d+$/);
const out=process.env.PILOT_PHOTO_LANGUAGE_OUT || '/tmp/fl-pilotux-photo-language';
const env={...process.env};
for(const name of Object.keys(env))if(['DATABASE_URL','OPENAI_API_KEY','ANTHROPIC_API_KEY','PGHOSTADDR','PGSERVICE','PGSERVICEFILE','PGOPTIONS'].includes(name)||/SUPABASE|RAILWAY/.test(name))delete env[name];
const languages=['en','es'],plan=['sign-in','form-photo-kind','draft-photo-kind-receipt','pack-live-language','make-live-language','integrity'];
const report={created_at:new Date().toISOString(),local_only:true,actual_handlers:true,openai_calls:0,width:390,expected_checks:languages.length*plan.length+1,results:[],receipts:[],photos:[],databases_dropped:[]};
const db=`fl_pilotux_photo_language_${process.pid}`;
const pgRun=(command,args)=>execFileSync(path.join(pg,command),['-h','127.0.0.1','-p',pgPort,...args],{cwd:root,env,stdio:'pipe'});
const image=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+iDbcAAAAASUVORK5CYII=','base64');
let child,browser,diagnostic='',databaseCreated=false;
await fs.mkdir(out,{recursive:true});
const save=()=>fs.writeFile(path.join(out,'results.json'),JSON.stringify(report,null,2));
async function check(language,name,fn,page){
 const row={language,name,width:390};
 try{await fn();row.passed=true;}catch(error){row.passed=false;row.error=error.stack||String(error);if(page){row.notice=await page.locator('#notice').textContent().catch(()=>null);row.screenshot=path.join(out,`FAIL-${language}-${name}.png`);await page.screenshot({path:row.screenshot,fullPage:true}).catch(()=>{});}}
 report.results.push(row);await save();if(!row.passed)throw Error(`${language}/${name}: ${row.error}\nNotice: ${row.notice}\nBridge: ${diagnostic}`);
}
try{
 pgRun('createdb',['-E','UTF8','-T','template0',db]);databaseCreated=true;
 pgRun('psql',['-d',db,'-v','ON_ERROR_STOP=1','-q','-c','CREATE EXTENSION pg_trgm']);
 pgRun('psql',['-d',db,'-v','ON_ERROR_STOP=1','-q','-f',path.join(root,'tests/schema/schema.sql')]);
 child=spawn(python,['-u',path.join(root,'tests/visual/pilot_ux_page_bridge.py')],{cwd:root,env:{...env,TEST_DATABASE_URL:`postgresql://127.0.0.1:${pgPort}/${db}`,KEY_SESSIONS_ENABLED:'1'},stdio:['ignore','pipe','pipe']});
 child.stderr.on('data',chunk=>{diagnostic+=chunk;});
 const lines=readline.createInterface({input:child.stdout});
 const ready=await Promise.race([new Promise((resolve,reject)=>{lines.once('line',line=>{try{resolve(JSON.parse(line));}catch{reject(Error(line));}});child.once('exit',code=>reject(Error(`Bridge exited ${code}: ${diagnostic}`)));}),new Promise((_,reject)=>{setTimeout(()=>reject(Error('Bridge startup timed out')),30000).unref();})]);
 assert.equal(ready.openai_forbidden,true);
 const origin=`http://127.0.0.1:${ready.port}`,fixture=await(await fetch(origin+'/test/fixture')).json();
 browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL || 'chrome'});
 for(const initialLanguage of languages){
  const context=await browser.newContext({viewport:{width:390,height:844},timezoneId:'America/New_York',colorScheme:'light'});
  await context.addInitScript(language=>sessionStorage.setItem('fl-ledger-language',language),initialLanguage);
  const page=await context.newPage();page.setDefaultTimeout(12000);const errors=[],remote=[],requests=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(new URL(request.url()).origin===origin)requests.push({method:request.method(),path:new URL(request.url()).pathname});});
  await page.route('**/*',route=>{if(new URL(route.request().url()).origin!==origin){remote.push(route.request().url());return route.abort();}return route.continue();});
  const language=()=>page.evaluate(()=>FLLedger.lang);
  const setLanguage=async expected=>{if(await language()!==expected)await page.locator('#language').click();assert.equal(await language(),expected);};
  const open=async file=>{await page.goto(origin+'/dashboard/'+file);await page.evaluate(()=>FLLedger.ready);};
  const action=async value=>{await open('entry.html');await setLanguage(initialLanguage);await page.locator('#entry-action').selectOption(value);};
  const form=()=>page.locator('#entry-form'),draft=()=>page.locator('#resolution');
  const fill=(name,value)=>form().locator(`[name="${name}"]`).fill(String(value));
  const picker=async(name,query,match)=>{const field=form().locator(`[data-picker="${name}"]`).first();await field.locator('input').first().fill(query);await field.locator('.picker-options button').filter({hasText:match}).first().click();};
  const prepare=async()=>{await form().getByRole('button',{name:/^(Prepare draft|Preparar borrador)$/}).click();await draft().getByRole('heading',{name:/^(Draft — review before recording|Borrador — revisa antes de registrar)$/}).waitFor();};
  const noPhotos=async()=>{assert.equal(await form().locator('input[type=file]').count(),0);assert.equal(await draft().locator('input[type=file]').count(),0);assert.equal(await draft().getByText(/^(Attach photo|Adjuntar foto)$/).count(),0);};
  const upload=async(trigger,scope,kind)=>{const response=page.waitForResponse(response=>new URL(response.url()).pathname==='/attachments'&&response.request().method()==='POST');await trigger();const reply=await response;assert.equal(reply.status(),200);const attached=await reply.json();assert.equal(attached.kind,kind);assert.ok(attached.attachment_id);await scope.getByText(/^(Photo attached|Foto adjunta)/).first().waitFor();await page.waitForFunction(()=>![...document.querySelectorAll('select[name$="photo_kind"]')].some(node=>node.disabled));report.photos.push({language:initialLanguage,...attached});return attached;};
  let first,retyped,finalPhoto;
  const firstKind=initialLanguage==='en'?'lot_tag':'bol',formKind=firstKind==='lot_tag'?'bol':'lot_tag',draftKind=firstKind;
  try{
   await check(initialLanguage,'sign-in',async()=>{await open('entry.html');await page.locator('#actor-key').fill(fixture.actors.floor.key);await page.locator('#sign-in-form button').click();await page.locator('#identity').filter({hasText:fixture.actors.floor.name}).waitFor();await form().waitFor();},page);
   await check(initialLanguage,'form-photo-kind',async()=>{
    await action('receive');await picker('product','Sunflower Seeds','Sunflower Seeds');await picker('supplier','Grain Supply','Grain Supply');await fill('cases',1);await fill('case_size_lb',10);await fill('bol_reference','PRACTICE photo language '+initialLanguage);await fill('supplier_lot_code','PRACTICE-PHOTO-'+initialLanguage);await fill('note','PRACTICE — same image, corrected evidence type');
    const type=form().locator('[name=entry_photo_kind]');await type.selectOption(firstKind);
    first=await upload(()=>form().locator('[name=entry_photo]').setInputFiles({name:`same-image-${initialLanguage}.png`,mimeType:'image/png',buffer:image}),form(),firstKind);
    retyped=await upload(()=>type.selectOption(formKind),form(),formKind);assert.notEqual(retyped.attachment_id,first.attachment_id);assert.equal(retyped.sha256,first.sha256,'Reclassification retains the exact image bytes');
    for(const lang of [initialLanguage==='en'?'es':'en',initialLanguage]){await setLanguage(lang);await form().getByText(lang==='es'?'Foto adjunta — incluida con el borrador y el recibo.':'Photo attached — included with the draft and receipt.',{exact:true}).waitFor();assert.equal(await type.inputValue(),formKind);assert.equal(await form().locator('[name=note]').inputValue(),'PRACTICE — same image, corrected evidence type');const product=form().locator('[data-picker=product]');const label=await product.locator('input').inputValue();await product.locator('[role=status]').getByText((lang==='es'?'Seleccionado: ':'Selected: ')+label,{exact:true}).waitFor();}
    assert.equal(requests.filter(row=>/\/commit$/.test(row.path)).length,0);
   },page);
   await check(initialLanguage,'draft-photo-kind-receipt',async()=>{
    await prepare();const type=draft().locator('[name=draft_photo_kind]');assert.equal(await type.inputValue(),formKind);
    finalPhoto=await upload(()=>type.selectOption(draftKind),draft(),draftKind);assert.notEqual(finalPhoto.attachment_id,retyped.attachment_id);assert.equal(finalPhoto.sha256,first.sha256,'The draft keeps the same image while changing its classification');
    for(const lang of [initialLanguage==='en'?'es':'en',initialLanguage]){await setLanguage(lang);await draft().getByText(lang==='es'?'Foto adjunta':'Photo attached',{exact:true}).waitFor();assert.equal(await type.inputValue(),draftKind);}
    for(const input of await draft().locator('[data-warning]').all())await input.check();await draft().getByRole('button',{name:/^(Record|Registrar)$/}).click();await draft().locator('.good a').waitFor();const number=await draft().locator('.good a').textContent();assert.match(number,/^RCV-\d{6}-\d{3,}$/);
    const response=await fetch(origin+'/receipts/'+number,{headers:{'X-API-Key':fixture.actors.floor.key}});assert.equal(response.status,200);const receipt=await response.json();assert.equal(receipt.attachments.length,1);assert.equal(receipt.attachments[0].kind,draftKind);assert.equal(Number(receipt.attachments[0].attachment_id||receipt.attachments[0].id),Number(finalPhoto.attachment_id));report.receipts.push({language:initialLanguage,number,attachment:receipt.attachments[0]});
    await open('receipt.html?number='+number);await page.locator('#content').getByRole('heading',{name:number,exact:true}).waitFor();assert.match(await page.locator('#content').textContent(),/Sunflower Seeds/);
   },page);
   await check(initialLanguage,'pack-live-language',async()=>{
    await action('pack');await picker('source','Classic #9','Batch Classic Granola #9');const recent=form().locator('[data-recent-lot]').first();await recent.waitFor();const code=(await recent.textContent()).split(' · ')[0];await recent.click();await picker('target','Granola Classic 25','Granola Classic 25 LB');await fill('cases',1);
    for(const lang of [initialLanguage,initialLanguage==='en'?'es':'en',initialLanguage]){await setLanguage(lang);await form().getByText((lang==='es'?'Lote seleccionado: ':'Selected batch: ')+code,{exact:true}).waitFor();assert.equal(await form().locator('[name=cases]').inputValue(),'1');}
    await prepare();await noPhotos();assert.match(await draft().textContent(),new RegExp(code));assert.equal(await draft().locator('.record-bar button.primary').isDisabled(),true);await draft().getByRole('button',{name:/^(Cancel|Cancelar)$/}).click();
   },page);
   await check(initialLanguage,'make-live-language',async()=>{
    await action('make');await picker('product','Classic #9','Batch Classic Granola #9');await fill('batches',1);await fill('note','PRACTICE — retain tag input');await form().locator('[name=confirmed_sku]').check();await prepare();await noPhotos();
    const row=draft().locator('[data-make-ingredient="11"]'),code='26-10-18-OATS-001';await row.locator('[data-make-type-code]').first().click();const method=row.locator('[data-make-code-method]').first(),input=row.locator('[data-make-code-value]').first();await method.selectOption('full_code');await input.fill(code);const before=requests.filter(row=>/\/prepare$/.test(row.path)).length;
    for(const lang of [initialLanguage,initialLanguage==='en'?'es':'en',initialLanguage]){await setLanguage(lang);assert.equal(await input.inputValue(),code);assert.equal(await method.inputValue(),'full_code');assert.equal(await input.isVisible(),true);assert.match(await row.locator('[data-make-type-code]').first().textContent(),lang==='es'?/Escribir código/:/Type code instead/);assert.match(await draft().locator('[data-make-shortage="13"]').textContent(),lang==='es'?/Solo hay 0 lb.*faltante/:/Only 0 lb.*shortage/);assert.match(await draft().locator('.make-meta').textContent(),lang==='es'?/Vence/:/Expires/);assert.equal(await draft().locator('.record-bar button.primary').isDisabled(),true);await noPhotos();}
    assert.equal(requests.filter(row=>/\/prepare$/.test(row.path)).length,before,'Language switches never reprepare or discard typed evidence');await draft().getByRole('button',{name:/^(Cancel|Cancelar)$/}).click();
   },page);
   await check(initialLanguage,'integrity',async()=>{assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);assert.equal(requests.filter(row=>row.method==='POST'&&/\/commit$/.test(row.path)).length,1);assert.equal(requests.filter(row=>row.method==='POST'&&['/receive','/make','/pack','/adjust','/inventory/found','/ship','/assistant/turn'].includes(row.path)).length,0);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);},page);
  }finally{await context.close();}
 }
 await check('both','database-proof',async()=>{const verified=await(await fetch(origin+'/test/verify')).json();report.verified=verified;assert.equal(verified.receipts.length,2);assert.equal(verified.attachments,6);assert.equal(verified.confirmations,0);assert.equal(report.receipts.length,2);assert.equal(report.photos.length,6);});
}catch(error){report.fatal=error.stack||String(error);process.exitCode=1;
}finally{
 if(browser)await browser.close().catch(error=>{report.cleanup_error=String(error);process.exitCode=1;});
 if(child){child.kill('SIGTERM');await new Promise(resolve=>{child.once('exit',resolve);setTimeout(()=>{child.kill('SIGKILL');resolve();},3000).unref();});}
 if(databaseCreated){try{pgRun('dropdb',['--if-exists',db]);report.databases_dropped.push(db);}catch(error){report.cleanup_error=String(error);process.exitCode=1;}}
 report.passed=report.results.filter(row=>row.passed).length;report.failed=report.results.filter(row=>!row.passed).length;report.complete=report.results.length===report.expected_checks&&report.failed===0&&!report.fatal&&!report.cleanup_error;
 if(!report.complete){process.exitCode=1;report.incomplete_reason=report.fatal||report.cleanup_error||`Completed ${report.results.length}/${report.expected_checks} planned checks; ${report.failed} failed.`;}
 await save();console.log(JSON.stringify({checks:report.results.length,expected:report.expected_checks,passed:report.passed,failed:report.failed,complete:report.complete,receipts:report.receipts.length,photos:report.photos.length,reason:report.incomplete_reason,results:path.join(out,'results.json')}));
}

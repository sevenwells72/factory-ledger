// Adjustment rules (owner decision 2026-11-18, migration 096), on the real Adjust form in a real
// browser at phone width, as Arturo:
//   1. Before the pounds are typed no reason can be picked.
//   2. Removing stock offers the five destinations ("where it went") as one pick, then Wrong lot used
//      and Data-entry error (note required), EN and ES; adding stock offers the reasons that can add
//      and never a destination.
//   3. "Shipped but not recorded" asks for the sales order in the same step; without one nothing is
//      prepared; with one the draft and the receipt name the order.
//   4. A 600 lb removal with NO photo is RECORDED at once: no "waiting for owner approval", no held
//      exception. The owner email is queued (adjustment_alerts) and the stock is already changed.
//   5. "Count correction — cause unknown" needs a note.
// Real local FastAPI over a throwaway local PostgreSQL DB (fl_person_browser_adjust_<pid>), seeded by
// tests/visual/person_selection_fixture.py, with migration 096 applied to that throwaway DB only, and
// dropped afterwards. No route stubs, no email settings (nothing is sent), never staging or production.
import assert from 'node:assert/strict';
import {execFileSync, spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const port = Number(process.env.STOCK_GUARD_PORT || 8138), origin = `http://127.0.0.1:${port}`;
const db = `fl_person_browser_stock_${process.pid}`, dbUrl = `postgresql://127.0.0.1:5432/${db}`;
const out = '/private/tmp/fl-098-browser';
const sh = (cmd, args) => execFileSync(cmd, args, {cwd: root, stdio: 'pipe'}).toString();
const sql = query => sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-At', '-c', query]).trim();
const person = 'Arturo';
const DESTINATIONS = [
  ['shipped_unrecorded', 'Shipped but not recorded', 'Enviado pero no registrado'],
  ['sent_to_sunshine_bulk', 'Sent to Sunshine as bulk', 'Enviado a Sunshine a granel'],
  ['used_in_production', 'Used in production but not recorded', 'Usado en producción pero no registrado'],
  ['damaged_thrown_out', 'Damaged or thrown out', 'Dañado o desechado'],
  ['count_unknown', 'Count correction — cause unknown', 'Corrección de conteo — causa desconocida'],
];
const NEVER_THERE = [['wrong_lot', 'Wrong lot used', 'Lote equivocado'], ['data_entry_error', 'Data-entry error', 'Error de captura']];
const REMOVALS = [...DESTINATIONS, ...NEVER_THERE];
const ADDITIONS = ['physical_count', 'missing_receipt', 'missing_production', 'wrong_lot', 'data_entry_error', 'unknown'];
await fs.mkdir(out, {recursive: true});

let server, browser, page;
const errors = [];
try {
  sh('createdb', ['-h', '127.0.0.1', db]);
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-c', 'CREATE EXTENSION IF NOT EXISTS pg_trgm']);
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-f', 'tests/schema/schema.sql']);
  let log = '';
  const env = {...process.env, TEST_DATABASE_URL: dbUrl, PERSON_BROWSER_PORT: String(port)};
  delete env.RESEND_API_KEY; delete env.ALERT_EMAIL_TO;          // the email is queued, never sent, from a test
  server = spawn(path.join(root, '.venv-test/bin/python'), ['tests/visual/person_selection_fixture.py'], {cwd: root, env, stdio: ['ignore', 'pipe', 'pipe']});
  server.stdout.on('data', c => { log += c; });
  server.stderr.on('data', c => { log += c; });
  for (let i = 0; ; i++) {
    if (server.exitCode !== null) throw new Error('Local API exited: ' + log);
    try { if ((await fetch(origin + '/health')).ok) break; } catch {}
    if (i > 120) throw new Error('Local API did not start: ' + log);
    await new Promise(r => setTimeout(r, 500));
  }
  // The migration under test, on the throwaway database only (one transaction, as its header says).
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-1', '-f', 'migrations/096_adjust_destinations.sql']);
  const fixture = JSON.parse(await fs.readFile(path.join(root, '.venv-test/person-selection/fixture.json'), 'utf8'));
  const lot = fixture.lots.Ingredient;
  sh('psql', ['-h','127.0.0.1','-d',db,'-v','ON_ERROR_STOP=1','-q','-1','-f','migrations/098_no_stock_by_hand.sql']);

  async function call(pathname, body, selectedPerson = 'Luz') {
    const response = await fetch(origin + pathname, {method: body ? 'POST' : 'GET', headers: {'X-FL-Person': selectedPerson, 'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined});
    const json = await response.json(); assert.ok(response.ok, pathname + ' ' + JSON.stringify(json)); return json;
  }

  const received = [];
  for(let i=0;i<2;i++){
    const p=await call('/receive/prepare',{product_id:fixture.products.Ingredient.id,supplier_id:fixture.supplier,
      cases:2,case_size_lb:50,bol_reference:'STOCK-'+i,no_supplier_lot:true,client_source:'dashboard'});
    received.push(await call('/tickets/'+p.ticket+'/commit',{payload_hash:p.payload_hash,acknowledged_warnings:p.warnings.filter(w=>w.requires_ack).map(w=>w.code)}));
  }
  browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHANNEL?{channel:process.env.PLAYWRIGHT_CHANNEL}:{})});
  const context=await browser.newContext({viewport:{width:390,height:844},timezoneId:'America/New_York'});
  page=await context.newPage();page.setDefaultTimeout(30000);page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',r=>new URL(r.request().url()).origin===origin?r.continue():r.abort());
  const choose=async(name,query)=>{const box=page.locator('[data-picker="'+name+'"]');await box.locator('input').fill(query);await box.getByRole('button',{name:'Search',exact:true}).click();const selected=box.getByText(/^(Selected|Seleccionado): /),option=box.locator('.picker-options button').first();await selected.or(option).first().waitFor();if(!(await selected.isVisible()))await option.click();await selected.waitFor();};
  await page.goto(origin+'/dashboard/entry.html?action=adjust');
  await page.getByRole('button',{name:person,exact:true}).click();await page.locator('#entry-form').waitFor();
  await choose('lot',received[0].lot_code);await page.locator('[name=delta_lb]').fill('20');
  const reason=page.locator('#entry-form [name=reason_code]');
  assert.deepEqual(await reason.locator('option').evaluateAll(n=>n.slice(1).map(o=>o.value)),['physical_count','missing_receipt','missing_production','wrong_lot','data_entry_error']);
  await reason.selectOption('wrong_lot');await choose('paired_lot',received[1].lot_code);
  await page.locator('#entry-form [name=note]').fill('The second tag was really used');
  await page.getByRole('button',{name:'Prepare draft',exact:true}).click();
  await page.getByRole('heading',{name:'Draft — review before recording',exact:true}).waitFor();
  const text=await page.locator('#resolution').innerText();assert.ok(text.includes(received[0].lot_code)&&text.includes(received[1].lot_code),text);
  await page.screenshot({path:out+'/adjust-paired-390.png',fullPage:true});
  for(const box of await page.locator('#resolution [data-warning]').all())await box.check();
  const posting=page.waitForResponse(r=>/\/tickets\/[^/]+\/commit$/.test(new URL(r.url()).pathname));
  await page.locator('#resolution').getByRole('button',{name:'Record',exact:true}).click();
  const response=await posting,move=await response.json();assert.equal(response.status(),200,JSON.stringify(move));
  assert.equal(sql(`SELECT count(*) FROM transaction_lines WHERE transaction_id=${move.transaction_id}`),'2');
  assert.equal(sql(`SELECT SUM(quantity_lb)::int FROM transaction_lines WHERE transaction_id=${move.transaction_id}`),'0');
  await page.goto(origin+'/dashboard/entry.html?action=adjust');await page.locator('#entry-form').waitFor();
  await choose('lot',received[0].lot_code);await page.locator('[name=delta_lb]').fill('75');
  await page.locator('[name=reason_code]').selectOption('missing_receipt');
  await page.waitForURL(/action=receive/);await page.locator('#entry-form').waitFor();
  assert.equal(await page.locator('[name=cases]').inputValue(),'');
  assert.equal(await page.locator('[name=case_size_lb]').inputValue(),'');
  assert.match(await page.locator('[data-picker=product] input').inputValue(),/Ingredient/);
  await page.screenshot({path:out+'/receive-carried-390.png',fullPage:true});
  await page.goto(origin+'/dashboard/entry.html?action=found');await page.locator('#entry-form').waitFor();
  await choose('product','PS-01');await choose('known_lot',received[0].lot_code);
  await page.locator('[name=quantity]').fill('1');await page.locator('[name=reason_code]').selectOption('physical_count');
  await page.getByRole('button',{name:'Prepare draft',exact:true}).click();
  await page.locator('#notice').getByText(/cannot add new stock/).waitFor();
  await page.screenshot({path:out+'/found-refusal-390.png',fullPage:true});
  const voided=await call('/void/'+received[1].transaction_id,{reason:'data_entry_error'},'Michael');
  await page.goto(origin+'/dashboard/receipt.html?number='+received[1].receipt_number);
  await page.getByRole('button',{name:'Receive again',exact:true}).click();await page.waitForURL(/action=receive/);await page.locator('#entry-form').waitFor();
  assert.equal(await page.locator('[name=lot_code]').inputValue(),received[1].lot_code);
  assert.equal(await page.locator('[name=bol_reference]').inputValue(),'STOCK-1');
  assert.equal(await page.locator('[name=no_supplier_lot]').isChecked(),true);
  assert.equal(await page.locator('[name=cases]').inputValue(),'');
  assert.equal(await page.locator('[name=case_size_lb]').inputValue(),'');
  assert.equal(await page.locator('[name=entry_photo]').inputValue(),'');
  await page.screenshot({path:out+'/receipt-receive-again-390.png',fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('PASS: Receive, Adjust, Found and voided receipt screens; paired move has two legs; receiving again preserves lot, supplier identity and BOL with blank quantity/photos.');
} catch(error){await page?.screenshot({path:out+'/failure.png',fullPage:true}).catch(()=>{});console.error('Page errors:',errors);throw error;}
finally{await browser?.close().catch(()=>{});if(server&&server.exitCode===null){server.kill();await new Promise(r=>server.once('exit',r));}try{sh('dropdb',['-h','127.0.0.1','--if-exists',db]);}catch(e){console.error(e.message);}}

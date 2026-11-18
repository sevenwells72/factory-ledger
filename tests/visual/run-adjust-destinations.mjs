// Adjustment rules (owner decision 2026-11-18, migration 095), on the real Adjust form in a real
// browser at phone width, as Arturo:
//   1. Before the pounds are typed no reason can be picked.
//   2. Removing stock offers exactly the five destinations ("where it went") as one pick, EN and ES;
//      adding stock offers the reasons that can add and never a destination.
//   3. "Shipped but not recorded" asks for the sales order in the same step; without one nothing is
//      prepared; with one the draft and the receipt name the order.
//   4. A 600 lb removal with NO photo is RECORDED at once: no "waiting for owner approval", no held
//      exception. The owner email is queued (adjustment_alerts) and the stock is already changed.
//   5. "Count correction — cause unknown" needs a note.
// Real local FastAPI over a throwaway local PostgreSQL DB (fl_person_browser_adjust_<pid>), seeded by
// tests/visual/person_selection_fixture.py, with migration 095 applied to that throwaway DB only, and
// dropped afterwards. No route stubs, no email settings (nothing is sent), never staging or production.
import assert from 'node:assert/strict';
import {execFileSync, spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const port = Number(process.env.ADJUST_DEST_PORT || 8137), origin = `http://127.0.0.1:${port}`;
const db = `fl_person_browser_adjust_${process.pid}`, dbUrl = `postgresql://127.0.0.1:5432/${db}`;
const out = path.join(root, '.venv-test/adjust-destinations');
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
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-1', '-f', 'migrations/095_adjust_destinations.sql']);
  const fixture = JSON.parse(await fs.readFile(path.join(root, '.venv-test/person-selection/fixture.json'), 'utf8'));
  const lot = fixture.lots.Ingredient;

  async function call(pathname, body) {
    const response = await fetch(origin + pathname, {method: body ? 'POST' : 'GET', headers: {'X-FL-Person': 'Luz', 'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined});
    const json = await response.json(); assert.ok(response.ok, pathname + ' ' + JSON.stringify(json)); return json;
  }
  const preparedOrder = await call('/sales/orders/prepare', {client_source: 'dashboard', customer_id: fixture.customer, customer_po: 'ADJ-DEST-' + process.pid, lines: [{product_id: fixture.products.Cases.id, quantity: 2, unit: 'cases'}]});
  const order = await call('/tickets/' + preparedOrder.ticket + '/commit', {payload_hash: preparedOrder.payload_hash, acknowledged_warnings: preparedOrder.warnings.filter(w => w.requires_ack).map(w => w.code)});
  const orderNumber = order.order_number || order.so_number || sql('SELECT order_number FROM sales_orders ORDER BY id DESC LIMIT 1');

  browser = await chromium.launch({headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? {channel: process.env.PLAYWRIGHT_CHANNEL} : {})});
  const context = await browser.newContext({viewport: {width: 390, height: 844}, timezoneId: 'America/New_York'});
  page = await context.newPage();
  page.setDefaultTimeout(30000);
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', r => new URL(r.request().url()).origin === origin ? r.continue() : r.abort());
  const choose = async (picker, query) => {
    const box = page.locator('[data-picker="' + picker + '"]');
    await box.locator('input').fill(query);
    await box.getByRole('button', {name: 'Search', exact: true}).click();
    const selected = box.getByText(/^(Selected|Seleccionado): /), option = box.locator('.picker-options button').first();
    await selected.or(option).first().waitFor();
    if (!(await selected.isVisible())) await option.click();
    await selected.waitFor();
  };
  const reason = page.locator('#entry-form [name=reason_code]'), delta = page.locator('#entry-form [name=delta_lb]');
  const offered = () => reason.locator('option').evaluateAll(nodes => nodes.slice(1).map(o => [o.value, o.textContent]));
  const orderBox = page.locator('[data-destination-order]');

  await page.goto(origin + '/dashboard/entry.html?action=adjust');
  // A fresh browser always starts on the four-name screen (the form is in the page but hidden).
  await page.getByRole('button', {name: person, exact: true}).click();
  await page.locator('#entry-form').waitFor();

  // 1. Nothing typed: no reason to pick yet.
  assert.deepEqual(await offered(), []);
  assert.equal(await reason.locator('option').first().textContent(), 'Enter the change in lb first');

  // 2. Removing stock: exactly the five destinations, in order.
  await choose('lot', lot.code);
  await delta.fill('-600');
  assert.deepEqual(await offered(), DESTINATIONS.map(([code, en]) => [code, en]));
  assert.equal(await page.locator('[data-adjust-hint]').textContent(), 'Removing stock: choose where it went.');
  assert.ok(await orderBox.isHidden());
  await page.screenshot({path: out + '/removal-reasons-en-390.png', fullPage: true});
  await page.locator('#language').click();
  assert.deepEqual(await offered(), DESTINATIONS.map(([code, , es]) => [code, es]));
  assert.equal(await page.locator('[data-adjust-hint]').textContent(), 'Para quitar inventario: elige a dónde se fue.');
  await page.screenshot({path: out + '/removal-reasons-es-390.png', fullPage: true});
  await page.locator('#language').click();
  // Adding stock: the reasons that can add, never a destination.
  await delta.fill('30');
  assert.deepEqual((await offered()).map(([code]) => code), ADDITIONS);
  await delta.fill('-600');

  // 5. "Count correction — cause unknown" makes the note required.
  await reason.selectOption('count_unknown');
  assert.equal(await page.locator('#entry-form [name=note]').getAttribute('required'), '');
  await reason.selectOption('damaged_thrown_out');
  assert.equal(await page.locator('#entry-form [name=note]').getAttribute('required'), null);

  // 3. Shipped but not recorded: the sales order is asked for in the same step.
  await reason.selectOption('shipped_unrecorded');
  assert.ok(await orderBox.isVisible());
  await page.getByRole('button', {name: 'Prepare draft', exact: true}).click();
  await page.locator('#notice').getByText(/Sales order it shipped on/).waitFor();
  assert.equal(sql("SELECT count(*) FROM write_tickets WHERE action='adjust'"), '0');
  await choose('destination_order', orderNumber);
  await page.screenshot({path: out + '/shipped-order-picked-390.png', fullPage: true});
  await page.getByRole('button', {name: 'Prepare draft', exact: true}).click();
  await page.getByRole('heading', {name: 'Draft — review before recording', exact: true}).waitFor();
  const draftText = await page.locator('#resolution').innerText();
  assert.match(draftText, /Where it went/);
  assert.ok(draftText.includes('Shipped but not recorded — ' + orderNumber), draftText);
  assert.match(draftText, /recorded right away and the owner gets an email\. A photo is optional\./);
  await page.screenshot({path: out + '/draft-over-500-390.png', fullPage: true});

  // 4. Record with NO photo: recorded at once, never held.
  for (const box of await page.locator('#resolution [data-warning]').all()) await box.check();
  const committed = page.waitForResponse(r => /\/tickets\/[^/]+\/commit$/.test(new URL(r.url()).pathname));
  await page.locator('#resolution').getByRole('button', {name: 'Record', exact: true}).click();
  const response = await committed, receipt = await response.json();
  assert.equal(response.status(), 200, JSON.stringify(receipt));
  assert.match(receipt.receipt_number, /^ADJ-\d{6}-\d{3,}$/);
  assert.notEqual(receipt.held, true);
  assert.equal(receipt.destination.sales_order.order_number, orderNumber);
  assert.equal(receipt.correction_review.owner_email, true);
  assert.equal(receipt.correction_review.photo_required, false);
  await page.locator('#resolution .good a').waitFor();
  const recordedText = await page.locator('#resolution').innerText();
  assert.ok(recordedText.includes(receipt.receipt_number), recordedText);
  assert.doesNotMatch(recordedText, /owner approval|NOT recorded/i);
  await page.screenshot({path: out + '/recorded-over-500-390.png', fullPage: true});

  assert.equal(sql(`SELECT COALESCE(SUM(quantity_lb),0)::int FROM transaction_lines WHERE lot_id=${lot.id}`), '400');
  assert.equal(sql("SELECT count(*) FROM exceptions WHERE kind='LARGE_CORRECTION'"), '0');
  assert.equal(sql("SELECT status FROM write_tickets WHERE action='adjust'"), 'committed');
  assert.equal(sql('SELECT status || \'|\' || lb::int || \'|\' || action FROM adjustment_alerts'), 'pending|-600|adjust');
  assert.equal(sql(`SELECT reason_code || '|' || operator_id FROM transactions WHERE id=${receipt.transaction_id}`), 'shipped_unrecorded|' + person);
  assert.ok(sql(`SELECT notes FROM transactions WHERE id=${receipt.transaction_id}`).includes(orderNumber));
  // The order itself is untouched: naming it explains the removal only.
  assert.equal(sql('SELECT state FROM sales_orders ORDER BY id DESC LIMIT 1'), 'open');
  assert.equal(sql('SELECT count(*) FROM shipments'), '0');

  assert.deepEqual(errors, []);
  console.log('PASS: Adjust form offers the five destinations on a removal, names the sales order for shipped stock, and records 600 lb at once with no photo and no hold (' + receipt.receipt_number + ')');
} catch (error) {
  await page?.screenshot({path: out + '/failure.png', fullPage: true}).catch(() => {});
  console.error('Page errors:', errors);
  throw error;
} finally {
  await browser?.close().catch(() => {});
  if (server && server.exitCode === null) { server.kill(); await new Promise(r => server.once('exit', r)); }
  try { sh('dropdb', ['-h', '127.0.0.1', '--if-exists', db]); } catch (e) { console.error('dropdb failed for ' + db + ': ' + e.message); }
}

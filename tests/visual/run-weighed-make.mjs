// Weighed batch output on the Make screen, in a real browser at 390 × 844. Sweetened coconut,
// 360 lb pan (recipe as on fl-staging product 125, 2026-11-26):
//   1. 12 pans weighed at 4,365 lb: the draft says so, the batch lot gets 4,365 lb and the
//      ingredients leave at recipe × 12 (2,400 / 60 / 84 / 30 / 1,320 lb; water skipped).
//   2. 1 pan with the field blank: 360 lb, exactly as before, no weighed key sent.
//   3. A typo (43,650 lb for 12 pans) is refused before anything is recorded.
// Real local FastAPI over a throwaway local PostgreSQL DB (fl_person_browser_weighed_<pid>),
// seeded by tests/visual/person_selection_fixture.py with COCONUT_BROWSER=1 and dropped
// afterwards. No route stubs, never staging or production.
import assert from 'node:assert/strict';
import {execFileSync, spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const port = Number(process.env.WEIGHED_MAKE_PORT || 8151), origin = `http://127.0.0.1:${port}`;
const db = `fl_person_browser_weighed_${process.pid}`;
const pgUser = process.env.PGUSER ? process.env.PGUSER + '@' : '';
const dbUrl = `postgresql://${pgUser}127.0.0.1:5432/${db}`;
const out = path.join(root, '.venv-test/weighed-make-browser');
const sh = (cmd, args) => execFileSync(cmd, args, {cwd: root, stdio: 'pipe'});
const sql = q => execFileSync('psql', ['-h', '127.0.0.1', '-d', db, '-X', '-A', '-t', '-c', q], {encoding: 'utf8'}).trim();
await fs.mkdir(out, {recursive: true});

let server, browser, page;
const errors = [], posts = [], results = {};
try {
  sh('createdb', ['-h', '127.0.0.1', db]);
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-c', 'CREATE EXTENSION IF NOT EXISTS pg_trgm']);
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-q', '-f', 'tests/schema/schema.sql']);
  let log = '';
  const env = {...process.env, TEST_DATABASE_URL: dbUrl, PERSON_BROWSER_PORT: String(port), COCONUT_BROWSER: '1'};
  server = spawn(path.join(root, '.venv-test/bin/python'), ['tests/visual/person_selection_fixture.py'], {cwd: root, env, stdio: ['ignore', 'pipe', 'pipe']});
  server.stdout.on('data', c => { log += c; });
  server.stderr.on('data', c => { log += c; });
  for (let i = 0; ; i++) {
    if (server.exitCode !== null) throw new Error('Local API exited: ' + log);
    try { if ((await fetch(origin + '/health')).ok) break; } catch {}
    if (i > 120) throw new Error('Local API did not start: ' + log);
    await new Promise(r => setTimeout(r, 500));
  }
  const fixture = JSON.parse(await fs.readFile(path.join(root, '.venv-test/person-selection/fixture.json'), 'utf8'));
  const P = fixture.coconut.products;

  browser = await chromium.launch({headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? {channel: process.env.PLAYWRIGHT_CHANNEL} : {})});
  const context = await browser.newContext({viewport: {width: 390, height: 844}, timezoneId: 'America/New_York', isMobile: true, hasTouch: true});
  page = await context.newPage();
  page.setDefaultTimeout(30000);
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => { if (r.method() === 'POST') posts.push({path: new URL(r.url()).pathname, body: r.postDataJSON?.()}); });
  await page.route('**/*', r => new URL(r.request().url()).origin === origin ? r.continue() : r.abort());

  async function clickResponse(prefix, action) {
    const response = page.waitForResponse(r => new URL(r.url()).pathname.startsWith(prefix) && r.request().method() === 'POST');
    await action(); const r = await response; const body = await r.json();
    assert.ok(r.ok(), JSON.stringify(body)); return body;
  }
  const record = () => page.getByRole('button', {name: 'Record', exact: true});
  async function tapAll() {
    const open = page.locator('#resolution [data-lot-confirm]:not(.confirmed):not([disabled])');
    for (let n = await open.count(); n > 0; n = await open.count()) await open.first().click();
  }
  async function startMake(pans, weighed) {
    await page.goto(origin + '/dashboard/entry.html?action=make');
    if (await page.getByRole('button', {name: 'Arturo', exact: true}).isVisible().catch(() => false))
      await page.getByRole('button', {name: 'Arturo', exact: true}).click();
    await page.locator('#entry-form').waitFor();
    if (await page.locator('#entry-action').inputValue() !== 'make') await page.locator('#entry-action').selectOption('make');
    const picker = page.locator('[data-picker="product"]');
    await picker.locator('input').fill('CB-90003');
    await picker.locator('.picker-options button').first().click();
    await page.locator('[name=batches]').fill(String(pans));
    if (weighed !== undefined) await page.locator('[name=weighed_output_lb]').fill(String(weighed));
    await page.locator('[name=confirmed_sku]').check();
    if (weighed === 4365) await page.locator('#entry-form').screenshot({path: `${out}/00-form-weighed-390.png`});
    return clickResponse('/make/prepare', () => page.getByRole('button', {name: 'Show ingredients and lots', exact: true}).click());
  }
  async function recordIt() {
    await tapAll();
    assert.equal(await record().isDisabled(), false, 'every lot tapped: Record on');
    return clickResponse('/tickets/', () => record().click());
  }
  const lines = tx => Object.fromEntries(sql(`SELECT product_id||':'||quantity_lb::float FROM transaction_lines WHERE transaction_id=${tx} ORDER BY id`)
    .split('\n').map(r => r.split(':').map(Number)));

  // 1. 12 pans weighed at 4,365 lb.
  let prepared = await startMake(12, 4365);
  assert.equal(posts.at(-1).body.weighed_output_lb, 4365);
  assert.deepEqual([prepared.draft.total_output_lb, prepared.draft.standard_output_lb, prepared.draft.weighed_difference_pct], [4365, 4320, 1]);
  await page.locator('#resolution [data-formula-summary]').getByText(/12 pans × 360 lb = 4,320 lb\. Weighed output: 4,365 lb \(booked on the batch lot\)\./).waitFor();
  assert.equal(prepared.warnings.some(w => w.code === 'WEIGHED_OUTPUT_OFF_STANDARD'), false, '+1% needs no acknowledgement');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'no side scroll');
  await page.locator('#resolution').screenshot({path: `${out}/01-draft-weighed-390.png`});
  let result = await recordIt();
  await page.screenshot({path: `${out}/02-recorded-weighed-390.png`});
  const weighedTx = result.transaction_id;
  assert.deepEqual(lines(weighedTx), {
    [P['Coconut Sweetened Fancy']]: 4365, [P['Coconut Fancy Desiccated']]: -2400, [P['Corn Starch']]: -60,
    [P['Glycol']]: -84, [P['Salt']]: -30, [P['Sugar 6X']]: -1320});
  assert.match(sql(`SELECT notes FROM transactions WHERE id=${weighedTx}`), /weighed output 4,365 lb \(standard 4,320 lb, \+1%\)/);
  results.weighed = {receipt: result.receipt_number, lines: lines(weighedTx)};

  // 2. Same product, 1 pan, field left blank: 360 lb as before.
  prepared = await startMake(1);
  assert.equal('weighed_output_lb' in posts.at(-1).body, false, 'blank field is not sent');
  assert.equal(prepared.draft.total_output_lb, 360);
  assert.equal('weighed_output_lb' in prepared.draft.formula_entry, false);
  result = await recordIt();
  assert.equal(lines(result.transaction_id)[P['Coconut Sweetened Fancy']], 360);
  results.blank = {receipt: result.receipt_number, output_lb: 360};

  // 3. Typo: 43,650 lb for 12 pans is refused; nothing recorded.
  const before = sql(`SELECT count(*) FROM transactions WHERE type='make'`);
  prepared = await startMake(12, 43650);
  assert.ok(prepared.blockers.some(b => b.code === 'WEIGHED_OUTPUT_IMPLAUSIBLE'), JSON.stringify(prepared.blockers));
  await page.locator('#resolution').getByText(/Weighed output 43,650 lb is \d+% over the expected 4,320 lb/).first().waitFor();
  assert.equal(await record().isDisabled().catch(() => true), true);
  await page.locator('#resolution').screenshot({path: `${out}/03-typo-refused-390.png`});
  assert.equal(sql(`SELECT count(*) FROM transactions WHERE type='make'`), before);
  results.typo = 'refused, nothing recorded';
  assert.deepEqual(errors, []);
  await fs.writeFile(`${out}/result.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
  console.log(`PASS: Make screen books a weighed 4,365 lb on 12 coconut pans with recipe x 12 ingredients (${results.weighed.receipt}), keeps 360 lb when blank (${results.blank.receipt}), and refuses a 43,650 lb typo`);
} finally {
  await browser?.close().catch(() => {});
  if (server && server.exitCode === null) { server.kill(); await new Promise(r => server.once('exit', r)); }
  try { sh('dropdb', ['-h', '127.0.0.1', '--if-exists', db]); } catch (e) { console.error('dropdb failed for ' + db + ': ' + e.message); }
}

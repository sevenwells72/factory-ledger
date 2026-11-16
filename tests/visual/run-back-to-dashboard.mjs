// Every page needs an obvious way home. Checks, in a real browser at phone width:
//   1. Office name screen (no one chosen yet) keeps exactly the four names.
//   2. Office page header: the "Dashboard / Panel" tool-bar button exists, is first and
//      visible, and goes home; the FL logo/title links home; "Today's entries" never
//      shows a stray backslash.
//   3. Office → New order and Office → Close order "Recorded — receipt …" screens:
//      "Back to dashboard" sits right after "Another office action", relabels in ES,
//      and goes home.
// Real local FastAPI over a throwaway local PostgreSQL DB (fl_person_browser_home_<pid>),
// seeded by tests/visual/person_selection_fixture.py and dropped afterwards. No route
// stubs, never staging or production.
import assert from 'node:assert/strict';
import {execFileSync, spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const port = Number(process.env.HOME_NAV_PORT || 8133), origin = `http://127.0.0.1:${port}`, base = origin + '/dashboard';
const db = `fl_person_browser_home_${process.pid}`, dbUrl = `postgresql://127.0.0.1:5432/${db}`;
const out = path.join(root, '.venv-test/back-to-dashboard');
const sh = (cmd, args) => execFileSync(cmd, args, {cwd: root, stdio: 'pipe'});
const home = '/dashboard/index.html';
await fs.mkdir(out, {recursive: true});

function findKey(value, key) {
  if (!value || typeof value !== 'object') return undefined;
  if (typeof value[key] === 'string') return value[key];
  for (const v of Object.values(value)) { const found = findKey(v, key); if (found) return found; }
  return undefined;
}

let server, browser, page;
const errors = [];
try {
  sh('createdb', ['-h', '127.0.0.1', db]);
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-c', 'CREATE EXTENSION IF NOT EXISTS pg_trgm']);
  sh('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-f', 'tests/schema/schema.sql']);
  let log = '';
  server = spawn(path.join(root, '.venv-test/bin/python'), ['tests/visual/person_selection_fixture.py'], {
    cwd: root, env: {...process.env, TEST_DATABASE_URL: dbUrl, PERSON_BROWSER_PORT: String(port)}, stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stdout.on('data', c => { log += c; });
  server.stderr.on('data', c => { log += c; });
  for (let i = 0; ; i++) {
    if (server.exitCode !== null) throw new Error('Local API exited: ' + log);
    try { if ((await fetch(origin + '/health')).ok) break; } catch {}
    if (i > 120) throw new Error('Local API did not start: ' + log);
    await new Promise(r => setTimeout(r, 500));
  }

  browser = await chromium.launch({headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? {channel: process.env.PLAYWRIGHT_CHANNEL} : {})});
  const context = await browser.newContext({viewport: {width: 390, height: 844}, timezoneId: 'America/New_York'});
  page = await context.newPage();
  page.setDefaultTimeout(30000);
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', r => new URL(r.request().url()).origin === origin ? r.continue() : r.abort());
  const atHome = async () => { await page.waitForURL(u => new URL(u).pathname === home); await page.locator('html[data-fl-shell]').waitFor({state: 'attached'}); };
  const choose = async (picker, query, scope = page) => {
    const box = scope.locator('[data-picker="' + picker + '"]');
    await box.locator('input').fill(query);
    await box.getByRole('button', {name: 'Search', exact: true}).click();
    await box.locator('.picker-options button').first().click();
  };
  async function recordOfficeAction(label) {
    await page.getByRole('button', {name: 'Prepare draft', exact: true}).click();
    await page.getByRole('heading', {name: 'Draft — review before recording', exact: true}).waitFor();
    for (const box of await page.locator('#resolution [data-warning]').all()) await box.check();
    const committed = page.waitForResponse(r => /\/tickets\/[^/]+\/commit$/.test(new URL(r.url()).pathname));
    await page.locator('#resolution').getByRole('button', {name: 'Record', exact: true}).click();
    const result = await (await committed).json();
    await page.locator('#resolution .good a').waitFor();
    assert.match(await page.locator('#resolution .good').textContent(), /^Recorded — receipt [A-Z]+-\d{6}-\d+/);
    const another = page.locator('#resolution').getByRole('button', {name: 'Another office action', exact: true});
    const back = page.locator('#resolution a[data-fl-dashboard]');
    assert.equal(await back.count(), 1, label + ': exactly one Back to dashboard');
    assert.equal(await back.textContent(), 'Back to dashboard');
    assert.equal(await back.isVisible(), true);
    assert.equal(await another.evaluate(b => b.nextElementSibling?.dataset.flDashboard), '', label + ': Back to dashboard sits next to Another office action');
    await page.getByRole('button', {name: 'ES', exact: true}).click();
    assert.equal(await back.textContent(), 'Volver al panel');
    await page.getByRole('button', {name: 'EN', exact: true}).click();
    await page.screenshot({path: out + '/' + label + '-recorded-390.png', fullPage: true});
    await back.click();
    await atHome();
    return result;
  }

  // 1. Name screen on the Office page: it stays exactly the four names (owner rule, #180);
  //    the way home is the header button right after a name is tapped.
  await page.goto(base + '/office.html');
  await page.locator('.fl-person-option', {hasText: 'Michael'}).waitFor();
  assert.equal(await page.locator('.fl-person-overlay a').count(), 0, 'name screen keeps only the four names');
  await page.screenshot({path: out + '/office-name-screen-390.png'});

  // 2. Office header after choosing a name.
  await page.locator('.fl-person-option', {hasText: 'Michael'}).click();
  const button = page.locator('.receipt-tools > a[data-fl-dashboard]');
  await button.waitFor();
  assert.equal(await button.textContent(), 'Dashboard / Panel');
  assert.equal(await page.locator('.receipt-tools > a').first().getAttribute('data-fl-dashboard'), '', 'Dashboard is the first tool-bar link');
  assert.equal(new URL(await button.evaluate(a => a.href)).pathname, home);
  assert.equal(new URL(await page.locator('.ledger-brand').evaluate(a => a.href)).pathname, home, 'FL logo and title link home');
  const today = page.locator('#today-entries-link');
  for (const lang of ['ES', 'EN']) await page.getByRole('button', {name: lang, exact: true}).click();
  assert.equal(await today.textContent(), 'Today’s entries');
  assert.doesNotMatch(await today.getAttribute('data-en'), /\\/);
  await page.screenshot({path: out + '/office-header-390.png'});
  await button.click();
  await atHome();

  // 3a. Office → New order.
  await page.goto(base + '/office.html');
  await page.locator('#office-action').selectOption('create_order');
  await page.locator('#office-form').waitFor();
  await choose('customer', 'Person Browser Customer');
  await page.locator('[name=customer_po]').fill('HOME-NAV-' + process.pid);
  const row = page.locator('[data-lines]>fieldset').last();
  await choose('product', 'PS-70', row);
  await row.locator('[name=quantity]').fill('10');
  await row.locator('[name=unit]').selectOption('lb');
  const created = await recordOfficeAction('new-order');
  const orderNumber = findKey(created, 'order_number');
  assert.match(orderNumber || '', /^SO-/, 'new order number: ' + JSON.stringify(created).slice(0, 400));

  // 3b. Office → Close order (the screen from the original report).
  await page.goto(base + '/office.html');
  await page.locator('#office-action').selectOption('close_order');
  await page.locator('#office-form').waitFor();
  const loaded = page.waitForResponse(r => r.url().endsWith('/office-options'));
  await choose('order', orderNumber);
  await loaded;
  await page.locator('[name=reason]').selectOption('shipped_not_recorded');
  await page.locator('[name=note]').fill('Back-to-dashboard browser check');
  await recordOfficeAction('close-order');

  // 4. FL Assistant header (its own layout): visible Dashboard button that fits the header row.
  await page.goto(base + '/fl-assistant.html');
  const assistantHome = page.locator('.header-actions a[data-fl-dashboard]');
  await assistantHome.waitFor();
  // The local API is not production, so the page adds the STAGING banner, which on this
  // page's flex layout takes a column of its own. Production shows no banner; measure that.
  await page.addStyleTag({content: '#fl-environment-banner{display:none!important}'});
  const boxes = await page.locator('.header-actions > :visible').evaluateAll(nodes => nodes.map(n => n.getBoundingClientRect()).map(r => ({left: r.left, right: r.right, top: r.top})));
  for (let i = 1; i < boxes.length; i++) assert.ok(boxes[i].left >= boxes[i - 1].right - 1 || boxes[i].top !== boxes[i - 1].top, 'Assistant header controls do not overlap');
  assert.ok(boxes.every(b => b.right <= 390), 'Assistant header controls stay on screen');
  await page.screenshot({path: out + '/assistant-header-390.png'});
  await assistantHome.click();
  await atHome();

  assert.deepEqual(errors, []);
  console.log('PASS: Office Dashboard button, Assistant header, New order and Close order "Back to dashboard" all reach ' + home);
} catch (error) {
  await page?.screenshot({path: out + '/failure.png', fullPage: true}).catch(() => {});
  console.error('Page errors:', errors);
  throw error;
} finally {
  await browser?.close();
  if (server && server.exitCode === null) { server.kill(); await new Promise(r => server.once('exit', r)); }
  try { sh('dropdb', ['-h', '127.0.0.1', '--if-exists', db]); } catch (e) { console.error('dropdb failed for ' + db + ': ' + e.message); }
}

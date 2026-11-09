// Archive feature, end to end in a real browser against the LOCAL app started by
// tests/visual/archive_fixture.py. Every request the dashboard sends to the
// production API host is answered by the local server; any other non-local
// request is aborted, so nothing leaves the machine.
// Writes screenshots to docs/screenshots/archive/.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const local = process.env.ARCHIVE_BROWSER_BASE || 'http://127.0.0.1:8157';
assert.ok(['127.0.0.1', 'localhost'].includes(new URL(local).hostname), 'local server only');
const PROD = 'https://fastapi-production-b73a.up.railway.app';
const out = path.resolve('docs/screenshots/archive');
await fs.mkdir(out, { recursive: true });
const fixture = JSON.parse(await fs.readFile(path.resolve('.venv-test/archive-browser/fixture.json'), 'utf8'));
const michael = { 'X-FL-Person': 'Michael', 'Content-Type': 'application/json' };
const api = async (p, init = {}) => {
  const r = await fetch(local + p, { ...init, headers: { ...michael, ...(init.headers || {}) } });
  return { status: r.status, body: await r.json().catch(() => ({})) };
};

const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'light' });
await context.addInitScript(() => {
  try { localStorage.setItem('fl-person', JSON.stringify({ id: 'Michael', display_name: 'Michael' })); } catch (_) {}
});
const leaked = [];
await context.route('**/*', async route => {
  const url = new URL(route.request().url());
  if (url.origin === PROD) {
    const response = await route.fetch({ url: local + url.pathname + url.search });
    return route.fulfill({ response });
  }
  if (['127.0.0.1', 'localhost'].includes(url.hostname) || url.protocol === 'data:') return route.continue();
  if (/fonts\.(googleapis|gstatic)\.com$/.test(url.hostname)) return route.abort();
  leaked.push(url.href);
  return route.abort();
});
const page = await context.newPage();
page.setDefaultTimeout(20000);
const errors = [];
page.on('pageerror', e => errors.push(e.message));

async function expandAll(scope) {
  const collapsed = page.locator(scope.split(', ').map(x => x + ' .collapsible-header:not(.expanded)').join(', '));
  for (let guard = 0; guard < 50 && await collapsed.count(); guard++) await collapsed.first().click();
}
async function openOnHand() {
  await page.goto(local + '/dashboard/index.html?section=operations');
  await page.locator('#ingredients-panels .inv-table').first().waitFor({ state: 'attached' });
  await page.locator('#batch-inventory .inv-table').first().waitFor({ state: 'attached' });
  await page.locator('#finished-goods-panels .inv-table').first().waitFor({ state: 'attached' });
  // Open every collapsed panel so the screenshot shows the rows.
  await expandAll('#section-finished-goods, #section-ingredients');
}
// Page coordinates (boundingBox() is relative to the scrolled viewport).
const pageBox = sel => page.locator(sel).evaluate(el => {
  const r = el.getBoundingClientRect();
  return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height };
});
async function shotOnHand(name) {
  const box = await pageBox('#section-finished-goods');
  const end = await pageBox('#section-ingredients');
  await page.screenshot({ path: path.join(out, name), fullPage: true,
    clip: { x: box.x - 8, y: box.y - 8, width: box.width + 16, height: end.y + end.height - box.y + 16 } });
}
const onHandText = () => page.locator('#section-finished-goods, #section-batches, #section-ingredients').allInnerTexts().then(t => t.join('\n'));

// ── Before ──
await openOnHand();
const before = await onHandText();
assert.match(before, /BS Ingredients/);
assert.match(before, /6x7 OZ Retail Cases \(BS Line\)/);
assert.doesNotMatch(before, /6x8 OZ Retail Cases/, '6x8 cases were already archived');
await shotOnHand('01-on-hand-before.png');

// ── Block: an ingredient an active recipe still uses ──
const blocked = await api('/admin/products/archive', { method: 'POST', body: JSON.stringify({ product_ids: [fixture.ids['BS Oats GF']] }) });
assert.equal(blocked.status, 409);
assert.match(blocked.body.detail.message, /Batch BS Dark Chocolate Granola 350/);

// ── Archive the four BS batches through the UI (bulk + confirm) ──
const bsBatches = ['Batch BS Almond Butter Granola 350', 'Batch BS Dark Chocolate Granola 350',
  'Batch BS Hazelnut Butter Granola 350', 'Batch BS Peanut Butter Banana Granola'];
for (const name of bsBatches) {
  await page.locator(`#batch-inventory .archive-select[data-archive-name="${name}"]`).check();
}
await page.locator('[data-archive-bulk="batches"]').click();
const dialog = page.locator('.archive-confirm');
await dialog.waitFor();
assert.match(await dialog.innerText(), /Archive 4 items\? It will be hidden but history is kept\./);
await page.screenshot({ path: path.join(out, '03-archive-confirm.png') });
const archived = page.waitForResponse(r => r.url().endsWith('/admin/products/archive'));
await dialog.locator('[data-archive-ok]').click();
assert.equal((await archived).status(), 200);
await page.waitForFunction(() => !document.querySelector('#batch-inventory').innerText.includes('Batch BS'));

// ── The rest of the BS items with the one-off data script (as for production) ──
const db = process.env.TEST_DATABASE_URL;
assert.match(db, /^postgresql:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/fl_archive_browser_/);
const dry = execFileSync('.venv-test/bin/python', ['scripts/archive_bs_items.py'], { env: { ...process.env, DATABASE_URL: db }, encoding: 'utf8' });
const count = Number(/(\d+) to archive/.exec(dry)[1]);
assert.equal(count, 26, dry);
assert.doesNotMatch(dry, /Sunflower/);
const applied = execFileSync('.venv-test/bin/python', ['scripts/archive_bs_items.py', '--apply', '--expect', String(count)], { env: { ...process.env, DATABASE_URL: db }, encoding: 'utf8' });
assert.match(applied, /ARCHIVED 26 items \(archived_by=Michael\)/);

// ── After ──
await openOnHand();
const after = await onHandText();
assert.doesNotMatch(after, /\bBS\b/, 'no BS item or BS section on the on-hand page');
assert.match(after, /Granola General Ingredients/);
const sunflower = page.locator('#ingredients-panels tr.expandable', { hasText: 'Sunflower Seeds' });
assert.equal(await sunflower.count(), 1, 'Sunflower Seeds listed once, in the regular group');
assert.match(await sunflower.innerText(), /2,114/);
await shotOnHand('02-on-hand-after.png');

// ── Show archived on ──
await page.locator('[data-archive-toggle="ingredients"]').check();
await page.locator('#ingredients-panels .archived-row').first().waitFor();
await expandAll('#section-ingredients');
const ing = await pageBox('#section-ingredients');
await page.screenshot({ path: path.join(out, '04-ingredients-show-archived.png'), fullPage: true,
  clip: { x: ing.x - 8, y: ing.y - 8, width: ing.width + 16, height: Math.min(ing.height + 16, 1600) } });
assert.match(await page.locator('#ingredients-panels').innerText(), /BS Ingredients[\s\S]*Archived/);
await page.locator('[data-archive-toggle="ingredients"]').uncheck();

// ── Pickers and alerts ──
const make = await api('/entry/make-products?query=BS');
assert.equal(make.status, 200);
assert.equal(JSON.stringify(make.body).includes('Batch BS'), false, 'Make picker hides archived batches');
const resolved = await api('/resolve', { method: 'POST', body: JSON.stringify({ kind: 'product', query: 'BS Oats GF' }) });
assert.equal(resolved.status, 200, JSON.stringify(resolved.body));
const offered = [resolved.body.match, ...(resolved.body.candidates || [])].filter(Boolean).map(c => c.id);
assert.ok(!offered.includes(fixture.ids['BS Oats GF']), 'Receive/Pack/Ship resolver does not offer an archived ingredient');
const live = await api('/resolve', { method: 'POST', body: JSON.stringify({ kind: 'product', query: 'Sunflower Seeds' }) });
assert.ok([live.body.match, ...(live.body.candidates || [])].filter(Boolean).some(c => c.id === fixture.ids['Sunflower Seeds']), 'Sunflower Seeds still offered');
const search = await api('/dashboard/api/search?q=BS%20Oats');
assert.deepEqual(search.body.products.map(p => [p.name, p.archived]), [['BS Oats GF', true]]);

// ── Old Made record still displays ──
await page.goto(local + `/dashboard/history.html?day=2026-07-29&type=make&transaction=${fixture.made_transaction}`);
const card = page.locator(`#transaction-${fixture.made_transaction}`);
await card.waitFor();
const cardText = await card.innerText();
for (const name of ['Batch BS Dark Chocolate Granola 350', 'BS Oats GF', 'BS Cacao Nibs']) assert.match(cardText, new RegExp(name));
await card.screenshot({ path: path.join(out, '05-old-made-record.png') });
const lot = await api('/dashboard/api/lot/' + fixture.made_lot);
assert.equal(lot.status, 200);
assert.match(JSON.stringify(lot.body), /Batch BS Dark Chocolate Granola 350/);

// ── Unarchive restores ──
await openOnHand();
await page.locator('[data-archive-toggle="batches"]').check();
const restoreBtn = page.locator('#batch-inventory [data-archive-restore][data-archive-name="Batch BS Almond Butter Granola 350"]');
await restoreBtn.waitFor();
await restoreBtn.click();
await page.locator('.archive-confirm').waitFor();
const restored = page.waitForResponse(r => r.url().endsWith('/admin/products/unarchive'));
await page.locator('.archive-confirm [data-archive-ok]').click();
assert.equal((await restored).status(), 200);
await page.locator('[data-archive-toggle="batches"]').uncheck();
await page.waitForFunction(() => document.querySelector('#batch-inventory').innerText.includes('Batch BS Almond Butter Granola 350'));
assert.equal(await page.locator('#batch-inventory tr.archived-row').count(), 0);
await page.locator('#section-batches').screenshot({ path: path.join(out, '06-unarchive-restored.png') });
// Put it back so the fixture ends fully archived.
assert.equal((await api('/admin/products/archive', { method: 'POST', body: JSON.stringify({ product_ids: [fixture.ids['Batch BS Almond Butter Granola 350']] }) })).status, 200);

// ── Only Michael sees the controls / may archive ──
const luz = await fetch(local + '/admin/products/archive', { method: 'POST', headers: { 'X-FL-Person': 'Luz', 'Content-Type': 'application/json' }, body: JSON.stringify({ product_ids: [fixture.ids['Oats']] }) });
assert.equal(luz.status, 403);

assert.deepEqual(errors, [], 'no page errors');
assert.deepEqual(leaked, [], 'no request left the machine');
await browser.close();
console.log('ARCHIVE BROWSER CHECK: PASS');

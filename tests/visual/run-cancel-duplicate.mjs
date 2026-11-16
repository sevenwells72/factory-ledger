// Cancel as duplicate + Office order picker fixes, in a real browser at 390 × 844 against a
// LOCAL server (tests/visual/person_selection_fixture.py on a dedicated local DB; other origins aborted).
// Proves: an exact SO code selects on Search and on Enter (shows "Selected: CODE"); a missing
// order is marked on the picker; the order list opens one panel with both orders side by side,
// the suggested original and a pre-filled note; self / cancelled originals are blocked; one
// Confirm runs the ordinary cancel_order ticket (prepare -> commit) and names both codes.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
const base = process.env.PERSON_BROWSER_ORIGIN || 'http://127.0.0.1:8149';
assert.ok(['localhost', '127.0.0.1'].includes(new URL(base).hostname));
const fixture = JSON.parse(await fs.readFile('.venv-test/person-selection/fixture.json', 'utf8'));
const out = '.venv-test/cancel-duplicate-visual'; await fs.mkdir(out, { recursive: true });
const person = 'Luz';  // writes carry only the tapped name, exactly as the dashboard sends them

async function call(path, body) {
  const response = await fetch(base + path, { method: body ? 'POST' : 'GET', headers: { 'X-FL-Person': person, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  const json = await response.json(); assert.ok(response.ok, path + ' ' + JSON.stringify(json)); return json;
}
async function createOrder(po) {
  const prepared = await call('/sales/orders/prepare', { client_source: 'dashboard', customer_id: fixture.customer, customer_po: po, lines: [{ product_id: fixture.products.Cases.id, quantity: 2, unit: 'cases' }] });
  const ack = prepared.warnings.filter(w => w.requires_ack).map(w => w.code);
  return call('/tickets/' + prepared.ticket + '/commit', { payload_hash: prepared.payload_hash, acknowledged_warnings: ack });
}

const po = 'DUP-' + Date.now();
const original = await createOrder(po), duplicate = await createOrder(po);
const [orig, dup] = [original.order_number, duplicate.order_number];

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } }), calls = [], errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('request', r => calls.push({ path: new URL(r.url()).pathname, method: r.method() }));
await page.route('**/*', r => new URL(r.request().url()).origin === base ? r.continue() : r.abort());
async function signIn() { await page.getByRole('button', { name: person, exact: true }).click(); await page.locator('#office-action').waitFor(); }
async function action(value) { await page.locator('#office-action').selectOption(value); await page.locator('#office-form,#duplicate-cancel').first().waitFor(); }
const picker = name => page.locator(`[data-picker="${name}"]`);

try {
  await page.goto(base + '/dashboard/office.html'); await signIn();

  // 1. Exact code + Search selects without clicking a result.
  await action('cancel_order');
  await picker('order').locator('input').fill(dup);
  const loaded = page.waitForResponse(r => r.url().endsWith('/office-options'));
  await picker('order').getByRole('button', { name: 'Search', exact: true }).click(); await loaded;
  await picker('order').getByText('Selected: ' + dup, { exact: true }).waitFor();
  assert.equal(await picker('order').locator('.picker-options button').count(), 0);

  // 2. Enter runs Search; the exact code selects.
  await action('close_order'); await action('cancel_order');
  await picker('order').locator('input').fill(dup.toLowerCase()); await picker('order').locator('input').press('Enter');
  await picker('order').getByText('Selected: ' + dup, { exact: true }).waitFor();

  // 3. Nothing chosen: the picker itself is marked, not only the banner.
  await action('close_order'); await action('cancel_order');
  await picker('order').locator('input').fill(dup.slice(0, -1));
  await page.locator('[name=reason]').selectOption('customer_cancelled'); await page.locator('[name=note]').fill('picker check');
  await page.getByRole('button', { name: 'Prepare draft', exact: true }).click();
  await page.locator('[data-picker="order"].picker-missing').waitFor();
  assert.match(await picker('order').locator('[role=status]').textContent(), /^Choose the sales order: tap a search result, or type its full code/);
  assert.equal(await picker('order').locator('input').getAttribute('aria-invalid'), 'true');
  assert.equal(await page.locator('#resolution .panel').count(), 0, 'nothing was prepared');
  await page.screenshot({ path: out + '/picker-missing-390.png', fullPage: true });

  // 4. From the order list: Cancel as duplicate opens the panel with both sides.
  await page.goto(base + '/dashboard/index.html?section=orders'); await page.evaluate(() => FLDashboard.ready);
  const row = page.locator(`.order-row[data-order-id="${duplicate.order_id}"]`); await row.waitFor();
  await row.locator('.order-expand-toggle').click();
  await page.locator(`.so-exit-action[data-action="cancel_duplicate"][data-order-id="${duplicate.order_id}"]`).click();
  await page.waitForURL('**/office.html?*'); await page.locator('#duplicate-cancel').waitFor();
  await page.locator('[data-side=original]').getByText(orig, { exact: true }).waitFor();
  assert.ok((await page.locator('[data-side=duplicate]').textContent()).includes(dup));
  assert.ok((await page.locator('[data-side=duplicate]').textContent()).includes(po));
  assert.ok((await page.locator('[data-side=original]').textContent()).includes('by ' + person), 'created by the tapped name');
  await page.locator('[data-suggested]').getByText(/same customer PO and same items and quantities/).waitFor();
  assert.equal(await page.locator('#duplicate-cancel [name=note]').inputValue(), 'Duplicate of ' + orig);
  assert.equal(await page.locator('[data-reason=duplicate]').textContent(), 'Reason: Duplicate');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: out + '/panel-suggested-390.png', fullPage: true });

  // 5. Changing the original to the duplicate itself is blocked; back to the real one.
  await picker('original').locator('input').fill(dup); await picker('original').locator('input').press('Enter');
  await page.locator('[data-problem=RELATED_SO_INVALID]').waitFor();
  assert.equal(await page.locator('[data-duplicate-confirm]').isDisabled(), true);
  await picker('original').locator('input').fill(orig); await picker('original').locator('input').press('Enter');
  await page.locator('[data-side=original]').getByText(orig, { exact: true }).waitFor();
  assert.equal(await page.locator('[data-problem=RELATED_SO_INVALID]').count(), 0);

  // 6. One Confirm: prepare + commit, success names both codes.
  const before = calls.length;
  await page.getByRole('button', { name: `Confirm: cancel ${dup} as a duplicate of ${orig}`, exact: true }).click();
  await page.locator('[data-duplicate-result] .good').waitFor();
  const success = await page.locator('[data-duplicate-result] .good').textContent();
  assert.ok(success.includes(`Cancelled ${dup} as a duplicate of ${orig}.`), success);
  assert.match(success, /Receipt ORD-\d{6}-\d{3,}/);
  const writes = calls.slice(before).filter(c => c.method === 'POST');
  assert.deepEqual(writes.map(c => c.path.replace(/\/tickets\/[^/]+\/commit$/, '/tickets/:t/commit')), [`/sales/orders/${duplicate.order_id}/cancel/prepare`, '/tickets/:t/commit']);
  const saved = await call('/sales/orders/' + duplicate.order_id);
  assert.equal(saved.state, 'cancelled'); assert.equal(saved.state_reason, 'duplicate'); assert.equal(saved.related_so_id, original.order_id);
  assert.equal((await call('/sales/orders/' + original.order_id)).state, 'open');
  await page.screenshot({ path: out + '/panel-success-390.png', fullPage: true });

  // 7. The original cannot now be cancelled as a duplicate of the cancelled copy.
  await page.goto(base + `/dashboard/office.html?action=cancel_duplicate&order=${original.order_id}`); await page.locator('#duplicate-cancel').waitFor();
  await page.locator('[data-side=duplicate]').getByText(orig, { exact: true }).waitFor();
  await page.getByText(/^No likely original found/).waitFor();
  await picker('original').locator('input').fill(dup); await picker('original').locator('input').press('Enter');
  await page.locator('[data-problem=RELATED_SO_CANCELLED]').waitFor();
  assert.equal(await page.locator('[data-duplicate-confirm]').isDisabled(), true);
  await page.screenshot({ path: out + '/panel-cancelled-original-390.png', fullPage: true });

  // Every write went through /resolve or a ticket; no direct cancel route.
  const posts = calls.filter(c => c.method === 'POST' && !c.path.startsWith('/auth/'));
  assert.ok(posts.every(c => c.path === '/resolve' || c.path.endsWith('/prepare') || /^\/tickets\/[^/]+\/commit$/.test(c.path)), JSON.stringify(posts));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, original: orig, duplicate: dup }));
} catch (error) { await page.screenshot({ path: out + '/failure.png', fullPage: true }); if (await page.locator('#notice').count()) console.error('Visible notice:', await page.locator('#notice').textContent()); throw error; }
finally { await browser.close(); }

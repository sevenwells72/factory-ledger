// Cancel order with only a reason (Michael, 2026-11-17), in a real browser at 390 × 844 against a
// LOCAL server (tests/visual/person_selection_fixture.py on a dedicated local DB; other origins aborted).
// Proves, for EVERY cancel reason (Duplicate and Replaced included), on both entry paths:
//   - Office → Cancel order (order chosen in the picker), and
//   - order detail → Cancel order (office.html?action=cancel_order&order=<id>, order preselected),
// that Reason note and Related order carry no `required`, that Prepare draft → Record succeeds with both
// left blank, and that the saved order has the reason, no note and no related order. Then a voluntary
// related order + note are saved, and a cancelled order cannot be cancelled again.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
const base = process.env.PERSON_BROWSER_ORIGIN || 'http://127.0.0.1:8149';
assert.ok(['localhost', '127.0.0.1'].includes(new URL(base).hostname));
const fixture = JSON.parse(await fs.readFile('.venv-test/person-selection/fixture.json', 'utf8'));
const out = '.venv-test/cancel-optional-visual'; await fs.mkdir(out, { recursive: true });
const person = 'Luz';
const REASONS = ['customer_cancelled', 'cns_declined', 'duplicate', 'superseded', 'other'];

async function call(path, body) {
  const response = await fetch(base + path, { method: body ? 'POST' : 'GET', headers: { 'X-FL-Person': person, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  const json = await response.json(); assert.ok(response.ok, path + ' ' + JSON.stringify(json)); return json;
}
async function createOrder() {
  const prepared = await call('/sales/orders/prepare', { client_source: 'dashboard', customer_id: fixture.customer, customer_po: 'OPT-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6), lines: [{ product_id: fixture.products.Cases.id, quantity: 2, unit: 'cases' }] });
  const ack = prepared.warnings.filter(w => w.requires_ack).map(w => w.code);
  return call('/tickets/' + prepared.ticket + '/commit', { payload_hash: prepared.payload_hash, acknowledged_warnings: ack });
}

const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } }), calls = [], errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('request', r => calls.push({ path: new URL(r.url()).pathname, method: r.method() }));
await page.route('**/*', r => new URL(r.request().url()).origin === base ? r.continue() : r.abort());
const picker = name => page.locator(`[data-picker="${name}"]`);
async function signInIfAsked() {
  const who = page.getByRole('button', { name: person, exact: true });
  await who.or(page.locator('#office-form')).first().waitFor();
  if (await who.isVisible()) await who.click();
  await page.locator('#office-form').waitFor();
}
async function prepareAndRecord() {
  await page.getByRole('button', { name: 'Prepare draft', exact: true }).click();
  await page.getByRole('heading', { name: 'Draft — review before recording', exact: true }).waitFor();
  assert.equal(await page.locator('#office-form .picker-missing,[aria-invalid=true]').count(), 0, 'no field marked missing');
  await page.locator('#resolution').getByRole('button', { name: 'Record', exact: true }).click();
  await page.locator('#resolution .good a').waitFor();
  return page.locator('#resolution .good a').textContent();
}
async function assertOptionalFields() {
  assert.equal(await page.locator('[name=reason]').getAttribute('required'), '');
  assert.equal(await page.locator('[name=note]').getAttribute('required'), null);
  assert.equal(await picker('related').locator('input').getAttribute('required'), null);
}
const results = [];

try {
  await page.goto(base + '/dashboard/office.html'); await signInIfAsked();

  for (const reason of REASONS) {
    // Path 1: Office → Office action → Cancel order, order chosen in the picker.
    const viaOffice = await createOrder();
    await page.goto(base + '/dashboard/office.html'); await signInIfAsked();
    await page.locator('#office-action').selectOption('cancel_order'); await page.locator('#office-form').waitFor();
    await picker('order').locator('input').fill(viaOffice.order_number); await picker('order').locator('input').press('Enter');
    await picker('order').getByText('Selected: ' + viaOffice.order_number, { exact: true }).waitFor();
    await page.locator('[name=reason]').selectOption(reason);
    await assertOptionalFields();
    const receipt1 = await prepareAndRecord();
    assert.match(receipt1, /^ORD-\d{6}-\d{3,}$/);
    let saved = await call('/sales/orders/' + viaOffice.order_id);
    assert.deepEqual([saved.state, saved.state_reason, saved.state_note ?? null, saved.related_so_id ?? null], ['cancelled', reason, null, null]);
    results.push({ path: 'office', reason, receipt: receipt1 });

    // Path 2: order detail → Cancel order lands on Office with the order already selected.
    const viaDetail = await createOrder();
    await page.goto(base + `/dashboard/office.html?action=cancel_order&order=${viaDetail.order_id}`); await signInIfAsked();
    await picker('order').getByText('Selected: ' + viaDetail.order_number, { exact: true }).waitFor();
    await page.locator('[name=reason]').selectOption(reason);
    await assertOptionalFields();
    const receipt2 = await prepareAndRecord();
    saved = await call('/sales/orders/' + viaDetail.order_id);
    assert.deepEqual([saved.state, saved.state_reason, saved.state_note ?? null, saved.related_so_id ?? null], ['cancelled', reason, null, null]);
    results.push({ path: 'detail', reason, receipt: receipt2 });
    if (reason === 'duplicate') await page.screenshot({ path: out + '/duplicate-blank-recorded-390.png', fullPage: true });

    // Already cancelled: the draft is refused, nothing recorded.
    if (reason === 'superseded') {
      await page.goto(base + `/dashboard/office.html?action=cancel_order&order=${viaDetail.order_id}`); await signInIfAsked();
      await picker('order').getByText('Selected: ' + viaDetail.order_number, { exact: true }).waitFor();
      await page.locator('[name=reason]').selectOption('customer_cancelled');
      const commits = calls.filter(c => c.path.endsWith('/commit')).length;
      await page.getByRole('button', { name: 'Prepare draft', exact: true }).click();
      await page.getByText(/already 'cancelled'|only an open order can be cancelled|ORDER_NOT_OPEN/).first().waitFor();
      assert.equal(await page.locator('#resolution .good a').count(), 0);
      assert.equal(calls.filter(c => c.path.endsWith('/commit')).length, commits);
    }
  }

  // A voluntary related order and note are still saved.
  const kept = await createOrder(), replaced = await createOrder();
  await page.goto(base + `/dashboard/office.html?action=cancel_order&order=${replaced.order_id}`); await signInIfAsked();
  await picker('order').getByText('Selected: ' + replaced.order_number, { exact: true }).waitFor();
  await page.locator('[name=reason]').selectOption('superseded');
  await page.locator('[name=note]').fill('Replaced by ' + kept.order_number);
  await picker('related').locator('input').fill(kept.order_number); await picker('related').locator('input').press('Enter');
  await picker('related').getByText('Selected: ' + kept.order_number, { exact: true }).waitFor();
  await prepareAndRecord();
  const saved = await call('/sales/orders/' + replaced.order_id);
  assert.deepEqual([saved.state_reason, saved.state_note, saved.related_so_id], ['superseded', 'Replaced by ' + kept.order_number, kept.order_id]);
  assert.equal((await call('/sales/orders/' + kept.order_id)).state, 'open');

  // Every write went through /resolve or a ticket; never the direct cancel route.
  const posts = calls.filter(c => c.method === 'POST' && !c.path.startsWith('/auth/'));
  assert.ok(posts.every(c => c.path === '/resolve' || c.path.endsWith('/prepare') || /^\/tickets\/[^/]+\/commit$/.test(c.path)), JSON.stringify(posts));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, cancels: results.length, results }));
} catch (error) { await page.screenshot({ path: out + '/failure.png', fullPage: true }); if (await page.locator('#notice').count()) console.error('Visible notice:', await page.locator('#notice').textContent()); throw error; }
finally { await browser.close(); }

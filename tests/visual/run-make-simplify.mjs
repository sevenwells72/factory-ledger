// Make-entry simplification PR-1 in a real browser at 390 × 844 against a LOCAL server
// (tests/visual/d12_fixture.py with FORMULA_BROWSER=1 MAKE_SIMPLIFY_BROWSER=1):
// tap-to-confirm lots, entries kept through Edit / an expired draft / a lost answer,
// a real two-lot split, shortage posting, double-tap and retry never posting twice,
// and an extra-kosher Make held for the owner.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
const base = process.env.D12_BASE || 'http://127.0.0.1:8139';
assert.ok(['localhost', '127.0.0.1'].includes(new URL(base).hostname));
const data = JSON.parse(await fs.readFile('.venv-test/d12-fixture.json', 'utf8'));
assert.ok(data.formula && data.make_simplify, 'Start d12_fixture.py with FORMULA_BROWSER=1 MAKE_SIMPLIFY_BROWSER=1');
const { products: P, lots: L } = data.formula;
const out = '.venv-test/make-simplify-browser'; await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
const errors = [], posts = [];
page.on('pageerror', error => errors.push(error.message));
page.on('request', r => { if (r.method() === 'POST') posts.push({ path: new URL(r.url()).pathname, body: r.postDataJSON?.() }); });
await page.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
const commits = () => posts.filter(p => p.path.endsWith('/commit'));
async function clickResponse(path, action, ok = true) {
  const response = page.waitForResponse(r => new URL(r.url()).pathname.startsWith(path) && r.request().method() === 'POST');
  await action(); const r = await response; const body = await r.json();
  if (ok) assert.ok(r.ok(), JSON.stringify(body)); return body;
}
const status = () => page.locator('#resolution [data-entry-status]');
const record = () => page.getByRole('button', { name: 'Record', exact: true });
const ingredient = name => page.locator(`#resolution .formula-ingredient[data-product-id="${P[name]}"]`);
const noSideScroll = async () => assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
async function tapAll() {
  const open = page.locator('#resolution [data-lot-confirm]:not(.confirmed):not([disabled])');
  for (let n = await open.count(); n > 0; n = await open.count()) await open.first().click();
}
async function startMake(code, pans) {
  await page.locator('#entry-action').selectOption('make');
  const picker = page.locator('[data-picker="product"]');
  await picker.locator('input').fill(code);
  await picker.getByRole('button', { name: 'Search', exact: true }).click();
  await picker.locator('.picker-options button').first().click();
  await page.locator('[name=batches]').fill(String(pans));
  await page.locator('[name=confirmed_sku]').check();
  return clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Show ingredients and lots', exact: true }).click());
}
const results = {};
try {
  await page.goto(base + '/dashboard/entry.html');
  await page.locator('#actor-key').fill('d12-local-floor');
  await page.locator('#sign-in-form button[type=submit]').click();
  await page.getByText('Signed in as Arturo D12', { exact: true }).waitFor();

  // 1. Product + pans → one suggested lot per ingredient, nothing confirmed, NOT RECORDED YET.
  let prepared = await startMake('FB-90', 2);
  const attempt = posts.at(-1).body.entry_attempt_id;
  assert.match(attempt, /^[A-Za-z0-9-]{8,64}$/);
  await status().getByText(/^NOT RECORDED YET/).waitFor();
  assert.deepEqual(prepared.draft.formula_entry.ingredients.find(i => i.product_id === P['Plain Oats']).bags.map(b => [b.lot_id, b.quantity_lb]), [[L.oats_old.id, 150]]);
  assert.equal(await page.locator('#resolution [name=bag_code]').count(), 0, 'no typed codes');
  assert.equal(await record().isDisabled(), true);
  await page.locator('#resolution [data-pending-lots]').getByText(/lots still need a tap/).waitFor();
  await ingredient('Plain Oats').locator('[data-bag-short]').getByText(/only 50 lb on the books/).waitFor();
  await noSideScroll();
  await page.screenshot({ path: `${out}/01-draft-390.png` });
  // The sticky Record bar is on screen on a phone.
  const bar = await page.locator('#resolution .record-bar.sticky').boundingBox();
  assert.ok(bar && bar.y + bar.height <= 844 + 1, JSON.stringify(bar));

  // 2. Real two-lot use with known pounds: 50 lb old + 100 lb new (intentional list choice).
  await ingredient('Plain Oats').locator('summary').click();
  await ingredient('Plain Oats').locator('[name=bag_lb]').first().fill('50');
  await ingredient('Plain Oats').getByRole('button', { name: 'Add another lot', exact: true }).click();
  const second = ingredient('Plain Oats').locator('.formula-bag').nth(1);
  await second.locator(`[data-lot-id="${L.oats_new.id}"]`).click();
  await second.locator('[name=bag_lb]').fill('100');
  prepared = await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Check lots and pounds → new draft', exact: true }).click());
  assert.deepEqual(posts.at(-1).body.entry_attempt_id, attempt);
  await tapAll();
  assert.equal(await record().isDisabled(), false, 'every lot tapped: Record on');
  await page.screenshot({ path: `${out}/02-two-lots-tapped-390.png`, fullPage: true });

  // 3. Edit entry → form → prepare again: lots, pounds and taps are kept.
  await page.getByRole('button', { name: 'Edit entry', exact: true }).click();
  assert.equal(await page.locator('[name=batches]').inputValue(), '2');
  prepared = await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Show ingredients and lots', exact: true }).click());
  const carried = posts.at(-1).body;
  assert.deepEqual(carried.ingredient_lines.find(l => l.ingredient_product_id === P['Plain Oats']).bags, [{ lot_id: L.oats_old.id, quantity_lb: 50 }, { lot_id: L.oats_new.id, quantity_lb: 100 }]);
  assert.equal(carried.entry_attempt_id, attempt);
  assert.ok(carried.lot_confirmations.every(c => c.method === 'selected'));
  assert.equal(await page.locator('#resolution [data-lot-confirm]:not(.confirmed)').count(), 0, 'taps survived');
  assert.equal(await record().isDisabled(), false);

  // 4. Server refusal at Record (expired draft): inline NOT RECORDED, entries kept, Check again.
  await page.route('**/tickets/*/commit', route => route.fulfill({ status: 409, contentType: 'application/json',
    body: JSON.stringify({ detail: { error_code: 'TICKET_EXPIRED', message: 'Ticket expired; prepare again.' } }) }), { times: 1 });
  await record().click();
  await status().getByText(/^NOT RECORDED — Ticket expired/).waitFor();
  assert.equal(await ingredient('Plain Oats').locator('[name=bag_lb]').nth(1).inputValue(), '100');
  await page.screenshot({ path: `${out}/03-not-recorded-expired-390.png` });
  prepared = await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Check again (keeps my entries)', exact: true }).click());
  assert.equal(prepared.can_commit, true, JSON.stringify(prepared.blockers));

  // 5. Lost answer: FL records, the phone never hears back → NOT CONFIRMED; Record again replays.
  await page.route('**/tickets/*/commit', async route => { await route.fetch(); await route.abort('connectionreset'); }, { times: 1 });
  for (const box of await page.locator('#resolution [data-warning]').all()) await box.check();
  await record().click();
  await status().getByText(/^NOT CONFIRMED — FL did not answer/).waitFor();
  assert.equal(await record().isDisabled(), false);
  await page.screenshot({ path: `${out}/04-not-confirmed-390.png` });
  const receipt = await clickResponse('/tickets/', () => record().click());
  assert.equal(receipt.replayed, true, 'the retry returns the saved Make');
  await status().getByText(/^RECORDED ✓ MK-.*nothing was recorded twice/).waitFor();
  assert.equal(new Set(commits().map(c => c.path)).size, 2, 'expired ticket + one live ticket (retried), never a third');
  const used = {}; for (const row of receipt.input_plan) used[row.lot_id] = row.quantity_lb;
  assert.equal(used[L.oats_old.id], 50); assert.equal(used[L.oats_new.id], 100);
  assert.ok(receipt.input_plan.every(r => r.confirmation?.method === 'selected'));
  assert.deepEqual(receipt.shortages.map(s => [s.lot_id, s.short_lb]), [[L.honey.id, 10]], 'short honey still posts and is flagged');
  results.receipt = receipt.receipt_number;
  await page.screenshot({ path: `${out}/05-recorded-390.png` });

  // 6. Shortage on the only suggested lot + double tap on Record: one commit.
  await page.getByRole('button', { name: 'Record another entry', exact: true }).click();
  prepared = await startMake('FB-90', 1);
  assert.notEqual(posts.at(-1).body.entry_attempt_id, attempt, 'a new form is a new Make');
  assert.ok(prepared.warnings.some(w => w.code === 'WILL_CREATE_SHORTAGE' && w.requires_ack === false));
  await page.locator('#resolution [data-shortage-warning]').first().waitFor();
  await tapAll();
  const before = commits().length;
  await record().dblclick();
  await status().getByText(/^RECORDED ✓ MK-/).waitFor();
  assert.equal(commits().length - before, 1, 'double tap posts once');

  // 7. Extra-kosher Make: tapped lots are accepted, but nothing posts before the owner approves.
  await page.getByRole('button', { name: 'Record another entry', exact: true }).click();
  prepared = await startMake('FB-XK', 1);
  assert.equal(prepared.draft.kosher_approval?.required, true);
  await tapAll();
  const held = await clickResponse('/tickets/', () => record().click());
  assert.equal(held.posted, false);
  await status().getByText(/^NOT RECORDED YET — waiting for owner approval/).waitFor();
  await page.screenshot({ path: `${out}/06-extra-kosher-held-390.png` });

  // 8. Spanish draft at 390: no sideways scroll.
  await page.goto(base + '/dashboard/entry.html?action=make');
  await page.locator('#language').click();
  await page.locator('[data-picker="product"] input').fill('FB-90');
  await page.locator('[data-picker="product"]').getByRole('button', { name: 'Buscar', exact: true }).click();
  await page.locator('[data-picker="product"] .picker-options button').first().click();
  await page.locator('[name=batches]').fill('1'); await page.locator('[name=confirmed_sku]').check();
  await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Ver ingredientes y lotes', exact: true }).click());
  await status().getByText(/^AÚN NO REGISTRADO/).waitFor();
  await noSideScroll();
  await page.screenshot({ path: `${out}/07-draft-es-390.png`, fullPage: true });
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, ...results, commits: commits().length, screenshots: out }));
} catch (error) {
  await page.screenshot({ path: `${out}/failure.png`, fullPage: true }).catch(() => {});
  console.error(error); process.exitCode = 1;
} finally { await browser.close(); }

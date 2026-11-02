// Formula-driven Make in a real browser against a LOCAL server (tests/visual/d12_fixture.py,
// FORMULA_BROWSER=1): pans math, FIFO bags, bag split, reason on 0 lb, oat bran → oats on a
// non-SS recipe, > 10% owner flag, short honey still saves, nothing posts before Record.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
const base = process.env.D12_BASE || 'http://127.0.0.1:8137';
assert.ok(['localhost', '127.0.0.1'].includes(new URL(base).hostname));
const fixture = JSON.parse(await fs.readFile('.venv-test/d12-fixture.json', 'utf8')).formula;
assert.ok(fixture, 'Start d12_fixture.py with FORMULA_BROWSER=1');
const { products: P, lots: L } = fixture;
const codes = Object.fromEntries(Object.values(L).map(lot => [String(lot.id), lot.code]));
const out = '.venv-test/formula-browser'; await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
const errors = [], posts = [];
page.on('pageerror', error => errors.push(error.message));
page.on('request', r => { if (r.method() === 'POST') posts.push(new URL(r.url()).pathname); });
await page.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
async function clickResponse(path, action) {
  const response = page.waitForResponse(r => new URL(r.url()).pathname === path && r.request().method() === 'POST');
  await action(); const r = await response; const data = await r.json();
  assert.ok(r.ok(), JSON.stringify(data)); return data;
}
const ingredient = name => page.locator(`#resolution .formula-ingredient[data-product-id="${P[name]}"]`);
const bagValues = async name => {
  const rows = ingredient(name).locator('.formula-bag'); const result = [];
  for (const row of await rows.all()) result.push([await row.locator('[name=bag_lot]').inputValue(), await row.locator('[name=bag_lb]').inputValue()]);
  return result;
};
async function typeCodes() {
  for (const row of await page.locator('#resolution .formula-bag').all()) {
    await row.locator('[name=bag_method]').selectOption('full_code');
    await row.locator('[name=bag_code]').fill(codes[await row.locator('[name=bag_lot]').inputValue()]);
  }
}
const update = () => clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Check bags and pounds → new draft', exact: true }).click());
const results = {};
try {
  await page.goto(base + '/dashboard/entry.html');
  await page.locator('#actor-key').fill('d12-local-floor');
  await page.locator('#sign-in-form button[type=submit]').click();
  await page.getByText('Signed in as Arturo D12', { exact: true }).waitFor();
  await page.locator('#entry-action').selectOption('make');
  await page.getByText('Number of pans', { exact: true }).waitFor();
  const picker = page.locator('[data-picker="product"]');
  await picker.locator('input').fill('FB-90');
  await picker.getByRole('button', { name: 'Search', exact: true }).click();
  await picker.locator('.picker-options button').first().click();
  await page.locator('[name=batches]').fill('2');
  await page.locator('[name=confirmed_sku]').check();
  let prepared = await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Prepare draft', exact: true }).click());
  const block = prepared.draft.formula_entry;
  // Pans math: 2 × 101 lb; recipe × pans per ingredient.
  assert.deepEqual([block.pans, block.pan_lb, block.output_lb], [2, 101, 202]);
  await page.getByText('2 pans × 101 lb = 202 lb', { exact: false }).waitFor();
  // FIFO spill pre-filled, nothing confirmed, Record disabled.
  assert.deepEqual(await bagValues('Plain Oats'), [[String(L.oats_old.id), '50'], [String(L.oats_new.id), '100']]);
  // Oat bran (none in stock) → rolled oats pound for pound on a non-SS recipe.
  assert.equal(prepared.draft.auto_substitutions[0].quantity_lb, 10);
  assert.deepEqual(await bagValues('Rolled Oats'), [[String(L.rolled.id), '10']]);
  await ingredient('Rolled Oats').getByText('Includes rolled oats for missing oat bran (pound for pound).').waitFor();
  assert.equal(await ingredient('Oat Bran').count(), 0);
  // Short honey: 40 lb needed, 30 on hand — warned, not blocked.
  assert.deepEqual(await bagValues('Honey'), [[String(L.honey.id), '40']]);
  await page.locator('#resolution [data-shortage-warning]').first().waitFor();
  assert.equal(await page.getByRole('button', { name: 'Record', exact: true }).isDisabled(), true);
  await page.screenshot({ path: `${out}/draft-fifo-390.png`, fullPage: true });

  // Bag split + edit: 50 old + 80 new + 40 spare = 170 lb (recipe 150, +13.3%).
  await ingredient('Plain Oats').locator('.formula-bag').nth(1).locator('[name=bag_lb]').fill('80');
  await ingredient('Plain Oats').getByRole('button', { name: 'Add another bag', exact: true }).click();
  const spare = ingredient('Plain Oats').locator('.formula-bag').nth(2);
  assert.equal(await spare.locator('[name=bag_lot]').inputValue(), String(L.oats_spare.id));
  await spare.locator('[name=bag_lb]').fill('40');
  await ingredient('Plain Oats').getByText(/^Using 170 lb \(\+13\.3%\) — more than 10% off the recipe/).waitFor();
  // 0 lb salt without a reason is refused on the draft.
  await ingredient('Salt').getByRole('button', { name: 'Not used (0 lb)', exact: true }).click();
  await typeCodes();
  prepared = await update();
  assert.equal(prepared.can_commit, false);
  assert.ok(prepared.blockers.some(b => b.code === 'MAKE_REASON_REQUIRED'), JSON.stringify(prepared.blockers));
  await page.locator('#resolution .bad').getByText('Choose a reason for Formula Browser Salt', { exact: false }).first().waitFor();
  // Codes typed earlier survive the new draft (pounds changed, bags did not).
  assert.equal(await ingredient('Plain Oats').locator('[data-bag-state]').first().textContent(), '✓ Confirmed');
  await ingredient('Salt').locator('[name=ingredient_reason]').selectOption('not_used');
  prepared = await update();
  assert.equal(prepared.can_commit, true, JSON.stringify(prepared.blockers));
  const off = prepared.warnings.find(w => w.code === 'INGREDIENT_OFF_RECIPE');
  assert.ok(off && off.requires_ack && /weekly review/.test(off.message));
  assert.equal(posts.filter(p => p.includes('/commit')).length, 0, 'nothing recorded before Record');
  for (const box of await page.locator('#resolution [data-warning]').all()) await box.check();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: `${out}/draft-edited-en-390.png`, fullPage: true });
  await page.locator('#language').click();
  await ingredient('Plain Oats').getByText('Agregar otra bolsa', { exact: true }).waitFor();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: `${out}/draft-edited-es-390.png`, fullPage: true });
  await page.locator('#language').click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: `${out}/draft-edited-en-desktop.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });

  const receipt = await clickResponse('/tickets/' + prepared.ticket + '/commit', () => page.getByRole('button', { name: 'Record', exact: true }).click());
  assert.match(receipt.receipt_number, /^MK-/);
  const used = {}; for (const row of receipt.input_plan) used[row.lot_id] = row.quantity_lb;
  assert.deepEqual(used, { [L.oats_old.id]: 50, [L.oats_new.id]: 80, [L.oats_spare.id]: 40, [L.rolled.id]: 10, [L.honey.id]: 40 });
  const review = Object.fromEntries(receipt.ingredient_review.map(r => [r.product_id, r]));
  assert.equal(review[P['Plain Oats']].kind, 'off_recipe'); assert.equal(review[P['Plain Oats']].difference_pct, 13.3);
  assert.equal(review[P.Salt].kind, 'not_used'); assert.equal(review[P.Salt].reason_code, 'not_used');
  assert.equal(receipt.shortages.reduce((n, s) => n + s.short_lb, 0), 10);
  await page.locator('#resolution .good').getByText('Recorded', { exact: false }).waitFor();
  await page.screenshot({ path: `${out}/receipt-en-390.png`, fullPage: true });
  results.receipt = receipt.receipt_number;
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, ...results, screenshots: out }));
} catch (error) {
  await page.screenshot({ path: `${out}/failure.png`, fullPage: true }).catch(() => {});
  console.error(error); process.exitCode = 1;
} finally { await browser.close(); }

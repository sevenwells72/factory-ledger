// Actual local prepare/commit and receipt UI: partial bran, no bran, then short oats.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';
const base = process.env.D12_BASE || 'http://127.0.0.1:8137';
assert.ok(['localhost', '127.0.0.1'].includes(new URL(base).hostname));
const fixture = JSON.parse(await fs.readFile('.venv-test/d12-fixture.json', 'utf8')).oatbran;
assert.ok(fixture, 'Start d12_fixture.py with OATBRAN_BROWSER=1');
const out = '.venv-test/oatbran-browser'; await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
const errors = [], results = [];
page.on('pageerror', error => errors.push(error.message));
await page.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
async function clickResponse(path, action) {
  const response = page.waitForResponse(r => new URL(r.url()).pathname === path && r.request().method() === 'POST');
  await action(); const r = await response; const data = await r.json();
  assert.ok(r.ok(), JSON.stringify(data)); return data;
}
try {
  await page.goto(base + '/dashboard/entry.html');
  await page.locator('#actor-key').fill('d12-local-floor');
  await page.locator('#sign-in-form button').click();
  await page.getByText('Signed in as Arturo D12', { exact: true }).waitFor();
  for (const scenario of [
    { code: '90011', bran: 10, extra: 19, oats: 204, short: 0, name: 'partial-bran' },
    { code: '90016', bran: 0, extra: 28, oats: 213, short: 0, name: 'no-bran' },
    { code: '90013', bran: 0, extra: 28, oats: 213, short: 130, name: 'short-oats' },
  ]) {
    await page.locator('#entry-action').selectOption('make');
    const picker = page.locator('[data-picker="product"]');
    await picker.locator('input').fill(scenario.code);
    await picker.getByRole('button', { name: 'Search', exact: true }).click();
    await picker.locator('.picker-options button').first().click();
    await page.locator('[name=batches]').fill('1');
    await page.locator('[name=confirmed_sku]').check();
    let prepared = await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Prepare draft', exact: true }).click());
    const substitution = prepared.draft.auto_substitutions[0];
    assert.equal(substitution.automatic, true);
    assert.equal(substitution.quantity_lb, scenario.extra);
    assert.equal(substitution.available_bran_lb, scenario.bran);
    assert.equal(prepared.draft.input_plan.filter(lot => lot.product_id === fixture.products['Oat Bran']).reduce((n, lot) => n + lot.quantity_lb, 0), scenario.bran);
    assert.equal(prepared.draft.input_plan.filter(lot => lot.product_id === fixture.products['Rolled Oats']).reduce((n, lot) => n + lot.quantity_lb, 0), scenario.oats);
    await page.locator('#resolution .auto-substitution').getByText(substitution.message_en, { exact: true }).waitFor();
    await page.locator('#language').click();
    await page.locator('#resolution .auto-substitution').getByText(substitution.message_es, { exact: true }).waitFor();
    await page.locator('#language').click();
    const fields = page.locator('#resolution fieldset');
    for (const field of await fields.all()) await field.locator('input').fill('-001');
    prepared = await clickResponse('/make/prepare', () => page.getByRole('button', { name: 'Check confirmations → new draft', exact: true }).click());
    assert.equal(prepared.can_commit, true, JSON.stringify(prepared.blockers));
    assert.ok(!prepared.warnings.some(warning => /substitut/i.test(warning.code) && warning.requires_ack));
    for (const box of await page.locator('#resolution [data-warning]').all()) await box.check();
    await page.screenshot({ path: `${out}/${scenario.name}-preview-390.png`, fullPage: true });
    if (scenario.name === 'no-bran') {
      await page.locator('#language').click();
      await page.screenshot({ path: `${out}/no-bran-preview-es-390.png`, fullPage: true });
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.screenshot({ path: `${out}/no-bran-preview-es-desktop.png`, fullPage: true });
      await page.locator('#language').click();
      await page.screenshot({ path: `${out}/no-bran-preview-en-desktop.png`, fullPage: true });
      await page.setViewportSize({ width: 390, height: 844 });
    }
    const receipt = await clickResponse('/tickets/' + prepared.ticket + '/commit', () => page.getByRole('button', { name: 'Record', exact: true }).click());
    assert.match(receipt.receipt_number, /^MK-/);
    assert.equal(receipt.auto_substitutions[0].quantity_lb, scenario.extra);
    assert.equal((receipt.shortages || []).reduce((n, row) => n + row.short_lb, 0), scenario.short);
    for (const shortage of receipt.shortages || []) assert.equal(shortage.product_id, fixture.products['Rolled Oats']);
    await page.locator('#resolution .auto-substitution').getByText(receipt.auto_substitutions[0].message_en, { exact: true }).waitFor();
    await page.locator('#language').click();
    await page.locator('#resolution .auto-substitution').getByText(receipt.auto_substitutions[0].message_es, { exact: true }).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: `${out}/${scenario.name}-receipt-es-390.png`, fullPage: true });
    await page.locator('#language').click();
    if (scenario.name === 'no-bran') {
      await page.screenshot({ path: `${out}/no-bran-receipt-en-390.png`, fullPage: true });
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.screenshot({ path: `${out}/no-bran-receipt-en-desktop.png`, fullPage: true });
      await page.locator('#language').click();
      await page.screenshot({ path: `${out}/no-bran-receipt-es-desktop.png`, fullPage: true });
      await page.locator('#language').click();
      await page.setViewportSize({ width: 390, height: 844 });
    }
    results.push({ scenario: scenario.name, receipt_number: receipt.receipt_number, extra_oats_lb: scenario.extra, short_lb: scenario.short });
  }
  await page.locator('#resolution .good a').click();
  await page.locator('#content .auto-substitution').getByText(/used.*28 lb/).waitFor();
  await page.locator('#language').click();
  await page.locator('#content .auto-substitution').getByText(/28 lb/).first().waitFor();
  assert.deepEqual(errors, []);
  await fs.writeFile(out + '/results.json', JSON.stringify(results, null, 2));
  console.log('Oat bran browser: 3 real makes passed; partial/no bran, short oats exception, actual planned lots, EN/ES preview/receipt, receipt lookup and 390px.');
} catch (error) {
  await page.screenshot({ path: out + '/failure.png', fullPage: true });
  throw error;
} finally { await browser.close(); }

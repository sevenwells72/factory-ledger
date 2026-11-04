// Fresh browser contexts, real HTTP headers and real backend: no auth/network stubs.
// Local mode records synthetic Make/Pack/Ship/Receive; --live-readonly checks the
// hosted sign-in/forms/owner screens without adding synthetic production stock.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const live = process.argv.includes('--live-readonly');
const staging = process.argv.includes('--staging-record');
assert.ok(!(live && staging), 'Choose either hosted read-only or staging-record mode');
const base = process.env.PERSON_BROWSER_BASE || 'http://127.0.0.1:8149/dashboard';
assert.ok(live || staging || ['localhost', '127.0.0.1'].includes(new URL(base).hostname), 'Record mode requires localhost or explicit staging mode');
if (staging) assert.equal(base, 'https://fastapi-staging-production-dd7b.up.railway.app/dashboard');
const root = path.resolve('.venv-test/person-selection');
await fs.mkdir(root, { recursive: true });
const fixture = live ? null : JSON.parse(await fs.readFile(path.join(root, staging ? 'staging-fixture.json' : 'fixture.json'), 'utf8'));
const out = path.join(root, live ? 'live' : staging ? 'staging' : 'local');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
const names = ['Michael', 'Luz', 'Arturo', 'Miriam'];
const results = [];
const forbidden = /\b(?:personal (?:actor )?key|enter (?:your )?(?:PIN|password)|approve this browser|approval code|pairing code|ask Michael|session expired|sign in again)\b/i;
let activePage;

async function noPrompt(page) {
  assert.equal(await page.locator('.fl-person-overlay:visible').count(), 0);
  assert.doesNotMatch(await page.locator('body').innerText(), forbidden);
  assert.equal(await page.locator('input[type=password]:visible').count(), 0);
}
async function responseFrom(page, route, action) {
  const pending = page.waitForResponse(r => new URL(r.url()).pathname === route && r.request().method() === 'POST');
  await action();
  const response = await pending, data = await response.json();
  assert.ok(response.ok(), route + ': ' + JSON.stringify(data));
  await noPrompt(page);
  return data;
}
async function choose(page, picker, query) {
  const box = page.locator('[data-picker="' + picker + '"]');
  await box.locator('input').fill(query);
  await box.getByRole('button', { name: 'Search', exact: true }).click();
  await box.locator('.picker-options button').first().click();
}
async function prepare(page, action) {
  let draft = await responseFrom(page, '/' + action + '/prepare', () => page.getByRole('button', {
    name: action === 'make' ? 'Show ingredients and lots' : 'Prepare draft', exact: true,
  }).click());
  await page.getByRole('heading', { name: 'Draft — review before recording', exact: true }).waitFor();
  if (action === 'make') {
    for (const button of await page.locator('#resolution [data-lot-confirm]:not(.confirmed):not([disabled])').all()) await button.click();
  } else {
    const rows = page.locator('#resolution .lot-confirmation-row');
    for (let i = 0; i < await rows.count(); i++) {
      const row = rows.nth(i), lot = draft.draft.input_plan[i];
      assert.ok(lot?.lot_code, 'Every displayed physical lot has an actual code');
      await row.locator('select').selectOption('full_code');
      await row.locator('input').fill(lot.lot_code);
    }
    if (await rows.count()) draft = await responseFrom(page, '/' + action + '/prepare', () =>
      page.getByRole('button', { name: 'Check confirmations → new draft', exact: true }).click());
  }
  for (const box of await page.locator('#resolution [data-warning]').all()) await box.check();
  return draft;
}
async function record(page, action) {
  const draft = await prepare(page, action);
  const response = await responseFrom(page, '/tickets/' + draft.ticket + '/commit', () =>
    page.locator('#resolution').getByRole('button', { name: 'Record', exact: true }).click());
  assert.ok(response.receipt_number, JSON.stringify(response));
  await page.locator('#resolution').getByText('Recorded — receipt', { exact: false }).first().waitFor();
  return { action, receipt: response.receipt_number, transaction_id: response.transaction_id,
    ticket_id: draft.ticket_id, lot_code: response.lot_code };
}
async function ownerScreens(page, receive, label) {
  await page.goto(base + '/entry.html?action=adjust');
  await page.locator('[data-picker=lot] input').waitFor();
  await noPrompt(page);
  await page.screenshot({ path: path.join(out, label + '-adjust.png'), fullPage: true });
  let receipt = receive?.receipt;
  if (!receipt) {
    const recent = await page.evaluate(() => FLLedger.request('/receipts?limit=100&summary=true', undefined, {read:true}));
    receipt = recent.receipts?.find(r => r.status === 'committed' && (!r.effective_status || r.effective_status === 'posted') && ['receive', 'make', 'pack', 'adjust', 'found'].includes(r.action))?.receipt_number;
  }
  if (receipt) {
    await page.goto(base + '/receipt.html?number=' + encodeURIComponent(receipt));
    await page.locator('[data-void]').first().waitFor();
    await noPrompt(page);
    await page.screenshot({ path: path.join(out, label + '-void.png'), fullPage: true });
  }
  assert.ok(receipt, 'Owner void screen must be checked on a real receipt; none found');
  return { adjustment: true, void: true };
}

try {
  for (const [viewportName, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
    for (const name of names) {
      const context = await browser.newContext({ viewport });
      const page = await context.newPage(); activePage = page;
      page.setDefaultTimeout(30000);
      const requests = [], errors = [], authFailures = [], dialogs = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss(); });
      page.on('request', request => {
        const url = new URL(request.url()), headers = request.headers();
        if (!/\.(?:html|js|css|png|svg|ico|json|woff2?)$/.test(url.pathname)) requests.push({
          method: request.method(), path: url.pathname.replace(/\/tickets\/[^/]+\//, '/tickets/{ticket}/'),
          person: headers['x-fl-person'] || null, key: !!headers['x-api-key'],
        });
      });
      page.on('response', response => { if ([401, 403].includes(response.status())) authFailures.push(new URL(response.url()).pathname); });
      const label = viewportName + '-' + name.toLowerCase();
      await page.goto(base + '/index.html');
      const overlay = page.locator('.fl-person-overlay');
      await overlay.waitFor();
      assert.deepEqual(await overlay.getByRole('button').allTextContents(), names);
      assert.equal((await overlay.innerText()).replace(/\s+/g, ' ').trim(), names.join(' '));
      assert.equal(await page.locator('input:visible').count(), 0, 'Fresh sign-in exposes only the four buttons');
      assert.equal(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight), true,
        'The four-button sign-in screen cannot scroll into the underlying dashboard');
      await page.screenshot({ path: path.join(out, label + '-signin.png'), fullPage: true });
      await overlay.getByRole('button', { name, exact: true }).click();
      await noPrompt(page);
      await page.goto(base + '/entry.html');
      await page.locator('#entry-action').waitFor();
      await noPrompt(page);
      const records = [];
      if (live) {
        for (const action of ['make', 'pack', 'ship', 'receive']) {
          await page.locator('#entry-action').selectOption(action);
          await page.locator('#entry-form').waitFor();
          await noPrompt(page);
        }
      } else {
        await choose(page, 'product', fixture.products.Ingredient.code);
        await choose(page, 'supplier', fixture.supplier_name || 'Person Browser Supplier');
        await page.locator('[name=cases]').fill('1');
        await page.locator('[name=case_size_lb]').fill('10');
        await page.locator('[name=bol_reference]').fill((fixture.marker || 'LOCAL') + '-' + label);
        await page.locator('[name=supplier_lot_code]').fill((fixture.marker || 'LOCAL') + '-SUP-' + label);
        records.push(await record(page, 'receive'));
        await page.locator('#entry-action').selectOption('make');
        await choose(page, 'product', fixture.products.Batch.code);
        await page.locator('[name=batches]').fill('1');
        await page.locator('[name=confirmed_sku]').check();
        records.push(await record(page, 'make'));
        await page.locator('#entry-action').selectOption('pack');
        await choose(page, 'source', fixture.products.Batch.code);
        await choose(page, 'target', fixture.products.Cases.code);
        await page.locator('[name=cases]').fill('1');
        records.push(await record(page, 'pack'));
        await page.locator('#entry-action').selectOption('ship');
        await page.locator('[name=ship_mode]').selectOption('standalone');
        await choose(page, 'customer', fixture.customer_name || 'Person Browser Customer');
        await choose(page, 'ship_product', fixture.products.Cases.code);
        await page.locator('[name=quantity_lb]').fill('5');
        await page.locator('[name=order_reference]').fill((fixture.marker || 'LOCAL') + '-' + label);
        await page.locator('[name=bol_reference]').fill((fixture.marker || 'LOCAL') + '-BOL-' + label);
        records.push(await record(page, 'ship'));
      }
      const owner = name === 'Michael' ? await ownerScreens(page, records[0], label) : null;
      await page.reload();
      await noPrompt(page);
      assert.deepEqual(errors, [], label + ': page errors');
      assert.deepEqual(authFailures, [], label + ': refused real requests');
      assert.deepEqual(dialogs, [], label + ': unexpected browser prompts');
      for (const request of requests.filter(r => r.method === 'POST' && !r.path.startsWith('/auth/'))) {
        assert.equal(request.person, name, JSON.stringify(request));
        assert.equal(request.key, false, 'Human writes must not carry a browser/session/person key');
      }
      assert.equal(requests.some(r => /\/auth\/(?:browsers|session|pin|devices)/.test(r.path)), false,
        'Name selection must not invoke approval/session/PIN APIs');
      if (live) assert.equal(requests.some(r => r.method === 'POST'), false, 'Live read-only mode must not post');
      const result = { name, viewport: viewportName, freshContext: true, records, owner, errors, authFailures,
        promptCount: dialogs.length, requests };
      results.push(result);
      await fs.writeFile(path.join(out, 'results.json'), JSON.stringify({ liveReadOnly: live, stagingRecords: staging, base, results }, null, 2));
      console.log(label + ': PASS (' + (live ? 'live forms' : records.length + ' actual records') + ')');
      await context.close();
    }
  }
  assert.equal(results.length, 8);
  console.log(JSON.stringify({ ok: true, liveReadOnly: live, contexts: results.length, records: results.flatMap(r => r.records).length, out }));
} catch (error) {
  await activePage?.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }).catch(() => {});
  await fs.writeFile(path.join(out, 'failure.json'), JSON.stringify({ message: error.message, results }, null, 2));
  throw error;
} finally { await browser.close(); }

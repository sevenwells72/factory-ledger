// Traceability Backward trace from the server's input_tree, with intercepted data;
// never contacts production. Fixture: the real 2026-10-06 Granola Fruit Nut 25 LB
// lot (product 142) — pack 2594 -> Fruit Nut Batch make 2593 (50 lb recorded for
// 769.04 lb made) -> Classic #9 make 2510 -> suppliers; walnuts/raisins came in
// on placeholder "found" receipts. Also checks the one-level fallback for an API
// response without input_tree.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { API_HOST, FIXTURE_DASHBOARD_KEY, buildTokenTable, installApiStub } from './lib/stub.mjs';

const out = path.resolve(process.argv[2] || 'work/traceability-backward');
await fs.mkdir(out, { recursive: true });
const fixture = JSON.parse(await fs.readFile(
  path.resolve('tests/visual/fixtures/trace-batch-fruit-nut-2026-10-06.json'), 'utf8'));
const server = await startStaticServer(path.resolve('dashboard'));
const browser = await chromium.launch(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {});
const tokens = buildTokenTable();
const MISMATCH = 'Recorded ingredients: 50 lb for 769.04 lb made';

async function open(width, theme, payload) {
  const context = await browser.newContext({ viewport: { width, height: 1000 }, acceptDownloads: true });
  await context.addInitScript(t => localStorage.setItem('dashboard-theme', t), theme);
  await installApiStub(context, tokens, { fail: [], overrides: {}, status: {}, unmatched: new Set() });
  const scans = [];
  context.on('request', req => {
    const url = new URL(req.url());
    if (url.host === API_HOST && url.pathname === '/transactions/history' && url.searchParams.get('transaction_type')) {
      scans.push(url.searchParams.get('transaction_type'));
    }
  });
  await context.route(`**://${API_HOST}/trace/batch/**`, async route => {
    const url = new URL(route.request().url());
    assert.equal(route.request().method(), 'GET');
    assert.equal(await route.request().headerValue('X-API-Key'), FIXTURE_DASHBOARD_KEY);
    assert.equal(decodeURIComponent(url.pathname), '/trace/batch/OCT 06 2026');
    assert.equal(url.searchParams.get('product_id'), '142');
    return route.fulfill({ json: payload });
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${server.origin}/traceability.html?lot=${encodeURIComponent('OCT 06 2026')}&direction=backward&product_id=142`);
  await page.locator('#traceDetail table').waitFor({ timeout: 15000 });
  return { context, page, errors, scans };
}

const svgTexts = page => page.locator('#graphSvg text').allTextContents();

try {
  for (const width of [1440, 390]) for (const theme of ['light', 'dark']) {
    const { context, page, errors, scans } = await open(width, theme, fixture);
    const texts = await svgTexts(page);
    for (const expected of ['OCT 06 2026', 'Granola Fruit Nut Batch', 'SEP 28 2026', 'Batch Classic Granola #9',
                            '26-02-04-FOUN-001', 'Refrig-IT', 'A1 Baker', 'QUALI PACK', 'Dutch Gold',
                            'Found inventory', 'No supplier receipt', 'Fauzi and Company']) {
      assert(texts.some(t => t.includes(expected)), `graph shows ${expected}`);
    }
    assert(!texts.some(t => /Supplier unknown/i.test(t)), 'no "Supplier unknown" for made lots');
    assert(!texts.some(t => t.trim().toLowerCase() === 'found'), 'placeholder "found" is never a supplier node');

    const badges = page.locator('#graphSvg .mismatch-badge');
    assert.equal(await badges.count(), 1, 'only the Fruit Nut Batch make is flagged');
    assert.equal((await badges.first().textContent()).replace('⚠ ', ''),
                 'Recorded ingredients:50 lb for 769.04 lb made');

    const panel = (await page.locator('#traceDetail').innerText()).replace(/ /g, ' ');
    assert(panel.includes(`Txn #2593`) && panel.includes(MISMATCH), 'detail panel lists the input check');
    assert(panel.includes('Found inventory — no supplier receipt | Recorded as "found" on Txn #69'));
    const rows = await page.locator('#traceDetail tbody tr').evaluateAll(trs => trs.map(tr => tr.cells[0].textContent));
    assert.equal(rows.filter(r => r === 'Data Gap').length, 3, 'coconut 25120, raisins and walnuts');
    assert.equal(rows.filter(r => r === 'Supplier').length, 8);
    assert.equal(rows.filter(r => r === 'Batch').length, 3, 'case lot, Fruit Nut Batch, Classic #9');
    assert((await page.locator('#traceInfo').innerText()).includes('1 make(s) with recorded ingredients outside 80–120% of output'));

    if (width === 1440 && theme === 'light') {
      const [download] = await Promise.all([page.waitForEvent('download'),
        page.getByRole('button', { name: 'Download Text Report' }).click()]);
      const text = await fs.readFile(await download.path(), 'utf8');
      assert(text.includes('INPUT CHECKS (display only; nothing was changed)'));
      assert(text.includes(`Txn #2593 OCT 06 2026 Granola Fruit Nut Batch: ${MISMATCH}`));
      assert(text.includes(`WARNING: ${MISMATCH}`));
      await fs.writeFile(path.join(out, 'audit-report.txt'), text);
    }

    assert.deepEqual(scans, [], 'no browser-side receipt/shipment scans in tree mode');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    await page.locator('#graphContainer').screenshot({ path: path.join(out, `graph-${width}-${theme}.png`) });
    await page.locator('#detailPanel').screenshot({ path: path.join(out, `detail-${width}-${theme}.png`) });
    await context.close();
  }

  // Synthetic (test only): the packed lot and the Fruit Nut Batch were also topped up by
  // receipts. The makes stay; each receipt is an extra origin, and found/unnamed ones are gaps.
  {
    const topped = structuredClone(fixture);
    topped.input_tree.also_received = [
      { kind: 'found_inventory', transaction_id: 9001, recorded_shipper: 'FOUND <b>INVENTORY</b>', quantity_lb: 410,
        received_date: '2026-10-20', received_time: '10:00 AM ET' },
      { kind: 'unknown', reason: 'receipt_without_supplier', transaction_id: 9002, recorded_shipper: 'Unknown',
        quantity_lb: 20, received_date: '2026-10-21', received_time: '10:00 AM ET' }];
    topped.input_tree.transactions[0].inputs[0].origin.also_received = [
      { kind: 'received', transaction_id: 9003, supplier_name: 'Top-Up Supplier', bol_reference: 'BOL-TOPUP',
        quantity_lb: 30, received_date: '2026-10-22', received_time: '10:00 AM ET' }];
    for (const width of [1440, 390]) {
      const { context, page, errors, scans } = await open(width, 'light', topped);
      const texts = await svgTexts(page);
      assert(texts.some(t => t.includes('Top-Up Supplier')), 'a named receipt into a made lot is a supplier node');
      assert(texts.some(t => t.includes('410 lb received')) && texts.some(t => t.includes('20 lb received')));
      assert(texts.some(t => t.includes('Granola Fruit Nut Batch')), 'the makes stay the origin');
      const rows = await page.locator('#traceDetail tbody tr').evaluateAll(trs => trs.map(tr => tr.cells[0].textContent));
      assert.equal(rows.filter(r => r === 'Data Gap').length, 5, 'three real gaps plus the found and unnamed receipts');
      assert.equal(rows.filter(r => r === 'Supplier').length, 9);
      assert.equal(rows.filter(r => r === 'Batch').length, 3);
      const panel = (await page.locator('#traceDetail').innerText()).replace(/\u00a0/g, ' ');
      assert(panel.includes('Found inventory added to this lot — no supplier receipt | 410 lb received into this lot | Recorded as "FOUND <b>INVENTORY</b>" on Txn #9001'));
      assert(panel.includes('Receipt without a supplier name | 20 lb received into this lot | Txn #9002'));
      assert(panel.includes('30 lb received into this lot'));
      assert.equal(await page.locator('#traceDetail b, #graphSvg b').count(), 0, 'recorded shipper text is escaped');
      assert.deepEqual(scans, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.deepEqual(errors, []);
      await page.locator('#graphContainer').screenshot({ path: path.join(out, `also-received-graph-${width}.png`) });
      await page.locator('#detailPanel').screenshot({ path: path.join(out, `also-received-detail-${width}.png`) });
      await context.close();
    }
  }

  // An API response without input_tree (deploy window) keeps the one-level trace.
  const { input_tree, ...legacy } = fixture;
  assert(input_tree);
  const { context, page, errors } = await open(1440, 'light', legacy);
  const texts = await svgTexts(page);
  assert(texts.some(t => t.includes('Granola Fruit Nut Batch')));
  assert.equal(await page.locator('#graphSvg .mismatch-badge').count(), 0);
  assert.deepEqual(errors, []);
  await context.close();

  console.log('PASS: four viewport/theme combinations, nested makes to suppliers, placeholder receipts as gaps, '
    + 'recorded-ingredients badge, panel and text report, receipts into made lots as extra origins, no browser-side scans, no overflow or page errors, legacy fallback.');
} finally { await browser.close(); await server.close(); }

// Production Detail ingredient table: real 2026-11-06 data, stubbed API, never contacts production.
//
//   node tests/visual/run-production-detail.mjs [dashboardDir] [outDir] [label]
//
// label "before" only captures screenshots (for the pre-change dashboard).
// Any other label also asserts the Ingredient | Quantity | Lot Information table,
// the Summary / Full Traceability toggle and the untouched Packed section.
// PW_CHANNEL=chrome uses the installed Google Chrome instead of Playwright's download.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { API_HOST, buildTokenTable, installApiStub } from './lib/stub.mjs';

const root = path.resolve(process.argv[2] || 'dashboard');
const out = path.resolve(process.argv[3] || 'docs/screenshots/production-detail');
const label = process.argv[4] || 'after';
const checking = label !== 'before';
const fixture = JSON.parse(await fs.readFile(new URL('./fixtures/production-detail-2026-11-06.json', import.meta.url), 'utf8'));
const DAY = '2026-11-06';
const GRANOLA = 114;   // Batch SS Chocolate Chip Granola #2 (Made #2965)
const COCONUT = 125;   // Batch Coconut Sweetened Fancy (Made #2966)

await fs.mkdir(out, { recursive: true });
const server = await startStaticServer(root);
const browser = await chromium.launch(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {});
const tokens = buildTokenTable();

function traceFor(params) {
  if (params.get('lot_id')) return fixture.trace[`lot:${params.get('lot_id')}`];
  return fixture.trace[`${params.get('kind') || 'make'}:${params.get('product_id')}`];
}

async function openRow(page, productId) {
  const row = page.locator(`.production-trace[data-kind="make"][data-product-id="${productId}"]`);
  if (!(await row.evaluate(node => node.open))) await row.locator(':scope > summary').click();
  await row.locator('.production-trace-panel h6').first().waitFor();
  return row;
}

async function rowsOf(row) {
  return row.locator('.ingredient-table tbody tr').evaluateAll(trs => trs.map(tr => ({
    ingredient: tr.querySelector('.ingredient-name')?.textContent.trim() || '',
    quantity: tr.querySelector('.ingredient-qty')?.textContent.trim() || '',
    lots: [...tr.querySelectorAll('.ingredient-lot')].map(lot => lot.textContent.replace(/\s+/g, ' ').trim()),
  })));
}

try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 } });
    await context.clock.setFixedTime(new Date(`${DAY}T17:00:00Z`));
    await context.addInitScript(() => localStorage.setItem('dashboard-theme', 'dark'));
    await installApiStub(context, tokens, { fail: [], overrides: {}, status: {}, unmatched: new Set() });
    await context.route(`**://${API_HOST}/dashboard/api/production**`, async route => {
      const url = new URL(route.request().url());
      assert.equal(route.request().method(), 'GET');
      if (url.pathname.endsWith('/trace')) {
        const data = traceFor(url.searchParams);
        assert(data, `no fixture for ${url.search}`);
        return route.fulfill({ json: data });
      }
      return route.fulfill({ json: { days: [fixture.day] } });
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(server.origin);
    await page.locator(`[data-production-date="${DAY}"]`).click();
    const detail = page.locator('#production-calendar-detail');
    await detail.waitFor();

    const views = checking ? ['summary', 'full'] : [null];
    for (const view of views) {
      if (view) await detail.locator(`.production-view-toggle button[data-view="${view}"]`).click();
      for (const [name, id] of [['granola', GRANOLA], ['coconut', COCONUT]]) {
        const row = await openRow(page, id);
        await row.scrollIntoViewIfNeeded();
        const file = [label, name, view, width].filter(Boolean).join('-') + '.png';
        await row.screenshot({ path: path.join(out, file) });
      }
    }

    if (checking) {
      const granola = await openRow(page, GRANOLA);
      const coconut = await openRow(page, COCONUT);
      const headers = await granola.locator('.ingredient-table thead th').allTextContents();
      assert.deepEqual(headers.map(h => h.trim()), ['Ingredient', 'Quantity', 'Lot Information']);
      const rows = await rowsOf(granola);
      const chips = rows.find(r => r.ingredient.startsWith('Chocolate Chips'));
      assert.equal(chips.quantity, '304 lb');
      assert.deepEqual(chips.lots, ['26-09-16-ABAK-001 · Supplier 260901 · 80 lb',
                                    '26-10-06-ABAK-001 · Supplier 260914 · 224 lb']);
      assert.equal(rows.find(r => r.ingredient === 'Oats').quantity, '1,832 lb');
      assert.deepEqual(rows.find(r => r.ingredient === 'Honey').lots, ['26-10-27-FOUND-006']);
      assert.deepEqual(rows.find(r => r.ingredient.startsWith('Almonds')).lots, ['26-11-05-NATI-001 · Supplier 236268101']);
      assert.equal(rows.length, 7, 'one row per ingredient');
      const coconutRows = await rowsOf(coconut);
      assert.equal(coconutRows.find(r => r.ingredient === 'Coconut Fancy Desiccated').quantity, '600 lb');
      assert.equal(coconutRows.find(r => r.ingredient === 'Salt').quantity, '7.5 lb');
      for (const row of [granola, coconut]) {
        const text = await row.innerText();
        assert(!/Ingredient subtotals/i.test(text), 'duplicate subtotal section removed');
        assert(/Made #\d+/.test(text));
      }
      // Quantity sits directly beside Ingredient: it starts where the ingredient column ends.
      const geo = await granola.locator('.ingredient-table thead th').evaluateAll(ths => ths.map(th => th.getBoundingClientRect()));
      assert(geo[1].left - geo[0].right >= 0 && geo[1].left - geo[0].right < 12, 'Quantity directly beside Ingredient');
      if (width === 1440) {
        assert(geo[1].right <= geo[2].left + 1);
        const table = geo[2].right - geo[0].left;
        assert(geo[0].width / table > 0.29 && geo[0].width / table < 0.35, 'ingredient column ~30–34%');
        // 13%, or 6.25rem when the panel is narrow: the family cards sit side by side, so the table is ~520px here.
        const qty = table > 760 ? table * 0.13 : 100;
        assert(Math.abs(geo[1].width - qty) < 2, 'quantity column 13%, or 6.25rem in a narrow panel');
      }

      // Summary hides lots, Full shows them, the choice is remembered, data is unchanged.
      await detail.locator('.production-view-toggle button[data-view="summary"]').click();
      assert.equal(await granola.locator('.ingredient-table th.ingredient-lots').isVisible(), false);
      assert.equal(await granola.locator('.ingredient-lot').first().isVisible(), false);
      assert.deepEqual((await rowsOf(granola)).map(r => r.quantity), rows.map(r => r.quantity));
      assert.equal(await page.evaluate(() => localStorage.getItem('fl-production-detail-view')), 'summary');
      await detail.locator('.production-view-toggle button[data-view="full"]').click();
      assert.equal(await granola.locator('.ingredient-lot').first().isVisible(), true);
      assert.equal(await page.evaluate(() => localStorage.getItem('fl-production-detail-view')), 'full');
      await page.reload();
      await page.locator(`[data-production-date="${DAY}"]`).click();
      assert.equal(await detail.locator('.production-view-toggle button[aria-pressed="true"]').getAttribute('data-view'), 'full');

      // Expand/collapse still works, and Packed (with its nested batch) is unchanged.
      const again = await openRow(page, GRANOLA);
      await again.locator(':scope > summary').click();
      assert.equal(await again.evaluate(node => node.open), false);
      const packed = page.locator('.production-trace[data-kind="pack"]').first();
      await packed.locator(':scope > summary').click();
      await packed.getByText('Consumed subtotals', { exact: true }).waitFor();
      assert.equal(await packed.locator('.ingredient-table').count(), 0, 'Packed keeps its own display');
      const nested = packed.locator('.production-trace').first();
      await nested.locator(':scope > summary').click();
      await nested.getByText(/^Ingredients in full batch/).waitFor();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'no horizontal page scroll');
      assert.deepEqual(errors, []);
    }
    await context.close();
  }
  console.log(checking
    ? 'PASS: table columns/order/widths, Made #2965 304 lb = 80 + 224, coconut, Summary/Full toggle + memory, expand/collapse, Packed unchanged, no overflow at 1440/390.'
    : `Captured ${label} screenshots to ${out}`);
} finally {
  await browser.close();
  await server.close();
}

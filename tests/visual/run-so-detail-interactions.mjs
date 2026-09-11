// Real Sales Order detail flows against a stateful, in-memory API. All other
// remote traffic is blocked; these checks never write to the live ledger.
// node tests/visual/run-so-detail-interactions.mjs [dashboard-root] [output-dir]
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { API_HOST, buildTokenTable, installApiStub, loadFixture } from './lib/stub.mjs';

const root = path.resolve(process.argv[2] || 'dashboard');
const out = path.resolve(process.argv[3] || 'docs/design/audit/screenshots/so-detail-interactions');
await fs.mkdir(out, { recursive: true });
const server = await startStaticServer(root);
const browser = await chromium.launch();
const results = [];

async function installStatefulDetail(context) {
  await context.route('**/*', route => new URL(route.request().url()).origin === server.origin
    ? route.continue() : route.abort());
  const tokens = buildTokenTable();
  await installApiStub(context, tokens, { fail: [], overrides: {}, status: {}, unmatched: new Set() });
  const { orders } = await loadFixture('sales-orders-list.json', tokens);
  const details = await loadFixture('sales-order-detail.json', tokens);
  const reads = [], writes = [];
  const offPageOrder = orders.find(order => order.order_id === 102);
  Object.assign(offPageOrder, { note: 'Keep the staged pallets together.', ready_by: 'packing-lead' });
  let releaseReady = null, holdReady = false;
  // These exceptional line/exit facts belong only to the interaction scenario;
  // the full visual audit keeps its documented baseline fixture quantities.
  details[101].lines[0].quantity_shipped_lb = 999;
  details[101].lines[0].readiness.shipped_recorded_lb = 999;
  details[111].lines[1].line_status = 'cancelled';
  details[111].ordered_lb = details[111].lines[0].quantity_lb;
  details[111].remaining_effective_lb = details[111].lines[0].readiness.remaining_lb;
  Object.assign(orders.find(order => order.order_id === 111), {
    ordered_lb: details[111].ordered_lb, remaining_effective_lb: details[111].remaining_effective_lb,
  });
  details[111].related_so_id = 102;
  details[111].state_reason = 'duplicate';
  await context.route(`**://${API_HOST}/**`, async route => {
    const request = route.request(), url = new URL(request.url()), pathname = url.pathname;
    if (pathname === '/sales/orders' && request.method() === 'GET') {
      reads.push({ kind: 'list' });
      const rows = orders.filter(order =>
        (!url.searchParams.has('state') || order.state === url.searchParams.get('state')) &&
        (!url.searchParams.has('fulfillment') || order.fulfillment === url.searchParams.get('fulfillment')) &&
        (!url.searchParams.has('overdue_only') || (order.state === 'open' && order.overdue && order.fulfillment !== 'shipped')) &&
        (!url.searchParams.has('customer') || order.customer.toLowerCase().includes(url.searchParams.get('customer').toLowerCase())) &&
        // The target exists beyond the first loaded page. A focused customer
        // lookup can retrieve its flag metadata, while initial lists cannot.
        (url.searchParams.has('customer') || order.order_id !== 102));
      return route.fulfill({ json: { orders: rows, count: rows.length } });
    }
    if (pathname === '/sales/orders/counts') {
      reads.push({ kind: 'counts' });
      const open = orders.filter(order => order.state === 'open');
      return route.fulfill({ json: {
        open: open.length, ready_to_ship: open.filter(order => order.ready).length,
        overdue: open.filter(order => order.overdue && order.fulfillment !== 'shipped').length,
        shipped: open.filter(order => order.fulfillment === 'shipped').length,
        closed: orders.filter(order => order.state === 'closed').length,
        cancelled: orders.filter(order => order.state === 'cancelled').length,
      } });
    }
    const detail = pathname.match(/^\/sales\/orders\/(\d+|SO-[^/]+)$/);
    if (detail && request.method() === 'GET') {
      const order = orders.find(order => String(order.order_id) === detail[1] || order.order_number === decodeURIComponent(detail[1]));
      reads.push({ kind: 'detail', id: order?.order_id });
      return route.fulfill({ status: order ? 200 : 404, json: order ? details[order.order_id] : { error: 'Order not found' } });
    }
    const exit = pathname.match(/^\/sales\/orders\/(\d+)\/(close|cancel|reopen)$/);
    if (exit && request.method() === 'POST') {
      const id = Number(exit[1]), action = exit[2], body = request.postDataJSON();
      const order = orders.find(order => order.order_id === id);
      writes.push({ id, action, body });
      assert.equal(Object.hasOwn(body, 'changed_by'), false, 'Client attribution must not be invented');
      if (action === 'cancel' && order.fulfillment !== 'unshipped') {
        return route.fulfill({ status: 409, json: { detail: {
          error_code: 'ORDER_ALREADY_SHIPPED', message: 'POST /sales/orders/105/close RAW_DATABASE_SECRET',
          suggested_action: 'close', suggested_reason: 'short_closed',
        } } });
      }
      const reservations = [{ id: 8101, quantity_lb: 1250.5 }, { id: 8102, quantity_lb: 249 }];
      const nextState = { close: 'closed', cancel: 'cancelled', reopen: 'open' }[action];
      if (body.mode === 'preview') return route.fulfill({ json: {
        mode: 'preview', order_id: id, resulting_state: nextState,
        reservations_to_release: action === 'reopen' ? [] : reservations,
      } });
      assert.equal(body.mode, 'commit');
      const updated = {
        state: nextState, state_reason: action === 'reopen' ? null : body.reason,
        state_note: action === 'reopen' ? null : body.note || null,
        state_changed_at: tokens.get('{{NOW}}'), state_changed_by: 'dashboard',
        related_so_id: body.related_so_id || null,
        health: { level: 'quiet', reasons: [], info: [], info_detail: [] },
      };
      Object.assign(order, updated);
      Object.assign(details[id], updated);
      return route.fulfill({ json: { ...updated, mode: 'commit', order_id: id, reservations_released: action === 'reopen' ? [] : reservations } });
    }
    const ready = pathname.match(/^\/sales-orders\/([^/]+)\/ready$/);
    if (ready && request.method() === 'POST') {
      const order = orders.find(order => order.order_number === decodeURIComponent(ready[1]));
      const body = request.postDataJSON();
      writes.push({ id: order.order_id, action: 'ready', body });
      assert.equal(order.state, 'open', 'Closed/cancelled Ready writes must be gated before a request');
      if (holdReady) {
        holdReady = false;
        await new Promise(resolve => { releaseReady = resolve; });
      }
      Object.assign(order, { ready: body.ready, ready_by: 'floor', ready_at: body.ready ? tokens.get('{{NOW}}') : null, note: body.note });
      details[order.order_id].floor_ready = body.ready;
      return route.fulfill({ json: { so_number: order.order_number, ready: order.ready, ready_at: order.ready_at, ready_by: order.ready_by, note: order.note } });
    }
    return route.fallback();
  });
  return { reads, writes, orders, details, holdNextReady: () => { holdReady = true; }, releaseReady: () => { releaseReady?.(); releaseReady = null; } };
}

async function listLoaded(page) {
  await page.waitForFunction(() => {
    const node = document.querySelector('#orders-table-container');
    return node && node.getAttribute('aria-busy') !== 'true' && node.querySelector('.order-row,.orders-empty');
  });
}
async function navigateDetail(page, id, tab = 'Open') {
  if (await page.locator('#order-back-btn').isVisible()) await page.locator('#order-back-btn').click();
  await page.getByRole('tab', { name: new RegExp(`^${tab}\\b`) }).click();
  await listLoaded(page);
  const expand = page.locator(`.order-row[data-order-id="${id}"] .order-expand-toggle`);
  if (await expand.getAttribute('aria-expanded') !== 'true') await expand.click();
  await page.locator(`.so-open-detail[data-order-id="${id}"]`).click();
  await page.locator('#so-detail-content').waitFor({ state: 'visible' });
  await page.waitForFunction(orderId => document.querySelector('#so-detail-content .so-detail-exit-action')?.dataset.orderId === String(orderId), id);
}
async function preview(page, action) {
  await page.getByRole('dialog').getByRole('button', { name: `Preview ${action}`, exact: true }).click();
  await page.locator('.so-exit-dialog').evaluate(async node => {
    while (node.getAttribute('aria-busy') === 'true') await new Promise(resolve => setTimeout(resolve, 10));
  });
}
async function commit(page, action) {
  await page.getByRole('dialog').getByRole('button', { name: `${action} order`, exact: true }).click();
  await page.locator('.so-exit-dialog').waitFor({ state: 'detached' });
  await page.locator('#so-detail-content').waitFor({ state: 'visible' });
}
async function openExit(page, action) {
  await page.locator(`#so-detail-content .so-detail-exit-action[data-action="${action}"]`).click();
  assert.equal(await page.locator('.so-exit-dialog[open]').count(), 1, 'Detail reuses the single shared exit dialog');
}
async function waitForWrites(api, count) {
  for (let tries = 0; api.writes.length < count && tries < 100; tries++) await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(api.writes.length, count);
}

try {
  for (const width of [1440, 390]) for (const theme of ['light', 'dark']) {
    const variant = `${width}-${theme}`;
    const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: theme, hasTouch: width === 390 });
    await context.addInitScript(value => localStorage.setItem('dashboard-theme', value), theme);
    const api = await installStatefulDetail(context);
    const page = await context.newPage(), pageErrors = [];
    page.setDefaultTimeout(10000);
    page.on('pageerror', error => pageErrors.push(error.message));
    const check = async (name, run) => { await run(); results.push({ variant, check: name, passed: true }); console.log(`PASS ${variant}: ${name}`); };
    try {
      await page.goto(server.origin, { waitUntil: 'domcontentloaded' });
      await page.locator(width === 390 ? '.mobile-nav > a[data-section="orders"]' : '.tab[data-tab="orders"]').click();
      await listLoaded(page);
      await navigateDetail(page, 101);

      await check('Four independent dimensions; keyboard explanations and single-popover dismissal', async () => {
        const dimensions = page.locator('#so-detail-content [data-dimension]');
        assert.deepEqual(await dimensions.evaluateAll(nodes => nodes.map(node => node.dataset.dimension)), ['state', 'ready', 'fulfillment', 'health']);
        for (const dimension of ['state', 'ready', 'fulfillment', 'health']) {
          const trigger = page.locator(`#so-detail-content [data-dimension="${dimension}"] [data-explain]`).first();
          const id = await trigger.getAttribute('data-explain');
          assert((await trigger.getAttribute('aria-describedby') || '').split(/\s+/).includes(id));
          assert(await trigger.evaluate(node => node.tabIndex >= 0));
          // Reach the trigger by a real keyboard Tab, then dismiss with Escape.
          await trigger.focus();
          await page.keyboard.press('Shift+Tab');
          await page.keyboard.press('Tab');
          assert(await trigger.evaluate(node => node === document.activeElement));
          const panel = page.locator(`[id="${id}"]`);
          await panel.waitFor({ state: 'visible' });
          assert((await panel.innerText()).trim().length > 20);
          assert.equal(await page.locator('.so-explanation:visible').count(), 1);
          if (dimension === 'fulfillment') assert.match(await panel.innerText(), /Ledger shows 0 of 4,500 lb shipped/i);
          if (dimension === 'health') {
            assert.match(await panel.innerText(), /Short 2,250 lb/);
            await panel.locator('summary').click();
            assert.match(await panel.innerText(), /Classic Granola 25 LB/);
            assert.match(await panel.innerText(), /500 lb/);
          }
          await page.keyboard.press('Escape');
          await panel.waitFor({ state: 'hidden' });
        }
        const triggers = page.locator('#so-detail-content [data-dimension] [data-explain]');
        await triggers.first().focus();
        await triggers.last().focus();
        assert.equal(await page.locator('.so-explanation:visible').count(), 1);
        await page.locator('.order-detail-header h2').click();
        assert.equal(await page.locator('.so-explanation:visible').count(), 0);
      });

      await check('Line explanations use matching line IDs and effective ledger pounds through SOList.number', async () => {
        const row = page.locator('.so-detail-lines-table tr[data-line-id="1010"]');
        const cells = row.locator('td');
        assert.equal(await cells.count(), 8);
        const cellText = await cells.allInnerTexts();
        assert.match(cellText[1], /1,000/);
        assert.equal(cellText[2].trim(), '0', 'Recorded 999 lb must not replace effective 0 lb');
        assert.match(cellText[3], /1,000/);
        assert.match(cellText[4], /500/);
        assert.match(cellText[5], /500/);
        const trigger = row.locator('.so-line-health[data-explain]');
        await trigger.focus();
        const panel = page.locator(`[id="${await trigger.getAttribute('data-explain')}"]`);
        assert.match(await panel.innerText(), /Classic Granola 25 LB/);
        assert.match(await panel.innerText(), /500 lb/);
        assert.doesNotMatch(await panel.innerText(), /Classic Granola 10 LB/);
        await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(() => SOList.number(1499.5)), '1,500');
        const copy = await page.locator('#so-detail-content').textContent();
        assert.equal((copy.match(/allocations not enforced/gi) || []).length, 1);
        assert.doesNotMatch(copy, /unless the API returns|\bdispatch_ready\b|RAW_DATABASE_SECRET/);
      });

      await check('Open Ready checkbox persists, blocks in-flight duplicates, and refreshes counts', async () => {
        let ready = page.locator('.so-detail-ready-checkbox');
        assert.equal(await ready.isDisabled(), false);
        assert.equal(await ready.isChecked(), false);
        api.holdNextReady();
        const writesBefore = api.writes.length, countsBefore = api.reads.filter(read => read.kind === 'counts').length;
        await ready.check();
        await waitForWrites(api, writesBefore + 1);
        assert.equal(await ready.isDisabled(), true);
        await ready.dispatchEvent('change');
        assert.equal(api.writes.length, writesBefore + 1);
        api.releaseReady();
        await page.waitForFunction(() => { const input = document.querySelector('.so-detail-ready-checkbox'); return input?.checked && !input.disabled; });
        assert.equal(api.details[101].floor_ready, true);
        assert.equal(api.writes.at(-1).body.ready, true);
        assert(api.reads.filter(read => read.kind === 'counts').length > countsBefore);
      });

      await check('Shared Close preview invalidates on edits; commit refreshes detail and list counts and focus', async () => {
        await openExit(page, 'close');
        assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
        await preview(page, 'close');
        assert.equal(await page.locator('.so-exit-release-summary').innerText(), 'Will release 2 reservations (1,500 lb)');
        assert.equal(api.writes.filter(write => write.action === 'close' && write.body.mode === 'commit').length, 0);
        await page.getByRole('dialog').getByLabel('Note (optional)', { exact: true }).fill('Customer confirmed receipt.');
        assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
        await preview(page, 'close');
        await page.getByRole('dialog').getByLabel('Reason', { exact: true }).selectOption('shipped_not_recorded');
        assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
        await preview(page, 'close');
        const readsBefore = Object.fromEntries(['detail', 'list', 'counts'].map(kind => [kind, api.reads.filter(read => read.kind === kind).length]));
        await commit(page, 'Close');
        assert.match(await page.locator('[data-dimension="state"]').innerText(), /Closed/);
        assert.equal(await page.locator('.so-detail-ready-checkbox').isDisabled(), true);
        assert.equal(await page.locator('.so-detail-exit-action[data-action="reopen"]').count(), 1);
        assert(await page.locator('.order-detail-header').evaluate(node => node.contains(document.activeElement)), 'Focus returns to an available detail header action');
        for (const kind of ['detail', 'list', 'counts']) assert(api.reads.filter(read => read.kind === kind).length > readsBefore[kind], `${kind} refresh after exit`);
        assert.match(await page.locator('[data-orders-tab="open"]').textContent(), /\b8\b/);
        assert.match(await page.locator('[data-orders-tab="closed"]').textContent(), /\b2\b/);
        assert.deepEqual(api.writes.filter(write => write.id === 101 && write.action === 'close').map(write => write.body.mode), ['preview', 'preview', 'preview', 'commit']);
      });

      await check('Partial shipment 409 offers Close and previews the suggested reason before commit', async () => {
        await navigateDetail(page, 105);
        await openExit(page, 'cancel');
        await preview(page, 'cancel');
        assert.match(await page.locator('.so-exit-error').innerText(), /already shipped/);
        assert.doesNotMatch(await page.getByRole('dialog').innerText(), /RAW_DATABASE_SECRET|POST|short_closed/);
        await page.getByRole('button', { name: 'Preview close instead · Short-closed', exact: true }).click();
        await page.getByRole('button', { name: 'Close order', exact: true }).waitFor();
        assert.equal(await page.getByRole('dialog').getByLabel('Reason', { exact: true }).inputValue(), 'short_closed');
        assert.deepEqual(api.writes.filter(write => write.id === 105).map(write => [write.action, write.body.mode]), [['cancel', 'preview'], ['close', 'preview']]);
        await commit(page, 'Close');
        assert.equal(api.details[105].state, 'closed');
        assert.equal(api.details[105].state_reason, 'short_closed');
      });

      await check('Open/shipped Ready remains editable while header edits are locked', async () => {
        await navigateDetail(page, 109, 'Shipped');
        assert.equal(await page.locator('.so-detail-ready-checkbox').isDisabled(), false);
        assert.equal(await page.locator('.so-detail-ready-checkbox').isChecked(), true);
        assert.match(await page.locator('.order-edit-locked').innerText(), /shipped/i);
        await page.locator('.so-detail-ready-checkbox').uncheck();
        await page.waitForFunction(() => { const input = document.querySelector('.so-detail-ready-checkbox'); return input && !input.checked && !input.disabled; });
        assert.equal(api.writes.at(-1).id, 109);
        assert.equal(api.writes.at(-1).body.ready, false);
      });

      await check('Closed/cancelled Ready is gated; Reopen uses preview/commit; cancelled lines and related SO stay reachable', async () => {
        for (const [id, tab] of [[110, 'Closed'], [111, 'Cancelled']]) {
          await navigateDetail(page, id, tab);
          const ready = page.locator('.so-detail-ready-checkbox');
          assert.equal(await ready.isDisabled(), true);
          const writesBefore = api.writes.length;
          await ready.dispatchEvent('change');
          assert.equal(api.writes.length, writesBefore);
          assert.match(await page.locator('.order-detail-header').innerText(), /floor/);
          if (id === 111) {
            const cancelled = page.locator('.so-detail-lines-table tr[data-line-id="1111"]');
            assert.equal(await cancelled.isVisible(), true);
            assert.match(await cancelled.locator('.so-line-status').innerText(), /Cancelled/);
            const related = page.locator('#so-detail-content [data-related-so-id="102"]');
            await related.click();
            await page.waitForFunction(() => document.querySelector('.so-detail-exit-action')?.dataset.orderId === '102');
            await navigateDetail(page, 111, 'Cancelled');
          }
          await openExit(page, 'reopen');
          await preview(page, 'reopen');
          assert.match(await page.locator('.so-exit-description').innerText(), /reservations are not restored/);
          await commit(page, 'Reopen');
          assert.equal(api.details[id].state, 'open');
          assert.equal(await page.locator('.so-detail-ready-checkbox').isDisabled(), false);
          assert.deepEqual(api.writes.filter(write => write.id === id).map(write => write.body.mode), ['preview', 'commit']);
        }
      });

      await check('Direct detail links preserve an off-page Ready note and provenance', async () => {
        const direct = new URL(server.origin);
        direct.searchParams.set('searchRecord', JSON.stringify({ type: 'order', id: 102 }));
        await page.goto(direct.href, { waitUntil: 'domcontentloaded' });
        await page.waitForFunction(() => document.querySelector('.so-detail-exit-action')?.dataset.orderId === '102');
        const before = api.writes.length;
        await page.locator('.so-detail-ready-checkbox').uncheck();
        await waitForWrites(api, before + 1);
        await page.waitForFunction(() => { const input = document.querySelector('.so-detail-ready-checkbox'); return input && !input.checked && !input.disabled; });
        assert.equal(api.writes.at(-1).body.note, 'Keep the staged pallets together.', 'A detail response without Ready metadata must not erase a note on an unloaded list page');
        await page.locator('.so-ready-explanation').focus();
        const panelId = await page.locator('.so-ready-explanation').getAttribute('data-explain');
        assert.match(await page.locator(`[id="${panelId}"]`).innerText(), /Keep the staged pallets together/);
        await page.keyboard.press('Escape');
        // Reopened orders omit exited provenance, so inspect a fresh exited
        // record with the same related-order metadata for native link behavior.
        api.details[111].state = 'cancelled';
        api.orders.find(order => order.order_id === 111).state = 'cancelled';
        api.details[111].related_so_id = 102;
        await navigateDetail(page, 111, 'Cancelled');
        const related = page.locator('.so-related-order[data-related-so-id="102"]');
        await related.waitFor({ state: 'visible' });
        const href = new URL(await related.getAttribute('href'), server.origin);
        assert.deepEqual(JSON.parse(href.searchParams.get('searchRecord')), { type: 'order', id: 102 });
      });

      await check('Detail page stays in the viewport without browser errors', async () => {
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        assert.deepEqual(pageErrors, []);
        assert.equal(api.writes.some(write => Object.hasOwn(write.body, 'changed_by')), false);
        await page.screenshot({ path: path.join(out, `${variant}.png`), fullPage: true });
      });
    } catch (error) {
      results.push({ variant, passed: false, error: error.stack });
      process.exitCode = 1;
      console.error(`${variant}:`, error);
    } finally { api.releaseReady(); await context.close(); }
  }
} catch (error) {
  results.push({ passed: false, error: error.stack });
  process.exitCode = 1;
  console.error(error);
} finally {
  await fs.writeFile(path.join(out, 'results.json'), JSON.stringify(results, null, 2) + '\n');
  await browser.close();
  await server.close();
}
console.log(`${results.filter(result => result.passed).length} Sales Order detail interaction checks passed.`);

// Pilot Record / Office acceptance: real page DOM, deterministic local API fixtures.
// No database, credential file, model transport, or remote HTTP request is used.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.env.PILOT_VISUAL_OUT || path.join(root, '.venv-test/pilot-screens');
const modes = process.env.PILOT_SESSIONS === undefined ? [false, true] : [process.env.PILOT_SESSIONS === '1'];
await fs.mkdir(out, { recursive: true });
const server = await startStaticServer(root);
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome' });
const actor = { id: 7, name: 'Pilot Worker', role: 'owner' };
const at = '2026-10-19T19:37:00-04:00';
const token = 'fls_pilot_browser_fixture';
const products = {
  oats: { id: 11, name: 'Rolled Oats' }, batch: { id: 12, name: 'Classic Granola Bulk Batch' },
  finished: { id: 13, name: 'Classic Granola 20 LB Case' }, honey: { id: 14, name: 'Honey' }
};
const lotRows = [
  { lot_id: 101, lot_code: 'OPEN261014-P1561', available_lb: 100 },
  { lot_id: 102, lot_code: 'PACK261019-P1561', available_lb: 80 },
  { lot_id: 103, lot_code: 'PACK261019-P1562', available_lb: 20 }
];
const makePlan = [
  { lot_id: 201, lot_code: 'OATS-001', product_id: 11, product_name: 'Rolled Oats', quantity_lb: 20 },
  { lot_id: 202, lot_code: 'OATS-002', product_id: 11, product_name: 'Rolled Oats', quantity_lb: 10 },
  { lot_id: 301, lot_code: 'HONEY-001', product_id: 14, product_name: 'Honey', quantity_lb: 5 }
];
const makeIngredients = [
  { ingredient_id: 11, ingredient_name: 'Rolled Oats', needed_lb: 30, available_lb: 1240,
    lots: Array.from({ length: 62 }, (_, i) => ({ lot_code: `OATS-${String(i + 1).padStart(3, '0')}`, available_lb: 20 })) },
  { ingredient_id: 14, ingredient_name: 'Honey', needed_lb: 5, available_lb: 50,
    lots: [{ lot_code: 'HONEY-001', available_lb: 25 }, { lot_code: 'HONEY-002', available_lb: 25 }] }
];
const results = [];

function receipt(number, action, draft, response = {}, extra = {}) {
  return { receipt_number: number, action, status: 'committed', effective_status: 'posted', actor,
    happened_at: at, entered_at: at, draft, response: { success: true, receipt_number: number, ...response },
    transactions: [], ...extra };
}
function fixtureReceipts() {
  const shipped = lotRows.slice(1).map((lot, i) => ({ ...lot, quantity_lb: i ? 10 : 30 }));
  const shipDraft = { customer: 'Pilot Bakery', bol_reference: 'PILOT-BOL', photo: 'pending',
    lines: [{ product: products.finished.name, quantity_lb: 40, lots: shipped }] };
  return new Map([
    ['RCV-261019-001', receipt('RCV-261019-001', 'receive', { product_name: products.oats.name, note: 'Receive pilot note' })],
    ['MK-261019-001', receipt('MK-261019-001', 'make', { product_name: products.batch.name, note: 'Make pilot note' })],
    ['PK-261019-001', receipt('PK-261019-001', 'pack', { target_product: products.finished.name, note: 'Pack pilot note' })],
    ['VD-261019-001', receipt('VD-261019-001', 'void', { note: 'Void pilot note' })],
    ['FND-261019-001', receipt('FND-261019-001', 'found', { product_name: products.oats.name, found_location: 'Dry storage rack 3', notes: 'Found pilot note' })],
    ['CUS-261019-001', receipt('CUS-261019-001', 'create_customer', { customer: { name: 'New Pilot Bakery' }, happened_at: '2026-10-19T00:00:00-04:00' }, { customer: { id: 92, name: 'New Pilot Bakery' } })],
    ['ORD-261019-001', receipt('ORD-261019-001', 'create_order', { customer: 'Pilot Bakery', notes: 'Deliver to loading door 2', happened_at: '2026-10-19T00:00:00-04:00' }, { notes: 'Deliver to loading door 2' })],
    ['SHP-261019-001', receipt('SHP-261019-001', 'ship_order', shipDraft,
      { proof_status: 'photo_pending', proof_due_at: '2026-10-19T23:59:00-04:00' },
      { shipment: { id: 72, proof_status: 'complete', proof_due_at: '2026-10-19T23:59:00-04:00' }, attachments: [{ id: 81, kind: 'bol' }] })],
    ['SHP-261019-002', receipt('SHP-261019-002', 'ship_order', shipDraft,
      { proof_status: 'photo_pending', proof_due_at: '2026-10-19T23:59:00-04:00' },
      { shipment: { id: 73, proof_status: 'photo_pending', proof_due_at: '2026-10-19T23:59:00-04:00' } })],
    ['RCV-261019-099', receipt('RCV-261019-099', 'receive', { product_name: products.oats.name }, {},
      { transactions: [{ id: 99, type: 'receive', effective_status: 'posted', allowed_actions: ['void'], lines: [{ product_name: products.oats.name, quantity_lb: 20 }] }] })]
  ]);
}

try {
  for (const enabled of modes) for (const width of [390, 1440]) {
    const context = await browser.newContext({ viewport: { width, height: width === 390 ? 844 : 1000 }, timezoneId: 'America/New_York' });
    const state = { errors: [], unknown: [], remote: [], prepares: [], commits: [], tickets: new Map(), receipts: fixtureReceipts(), next: 0 };
    await context.addInitScript(({ enabled, token }) => {
      sessionStorage.setItem('fl-ledger-language', 'en');
      if (enabled) sessionStorage.setItem('fl-key-session-token', token);
    }, { enabled, token });
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    page.on('pageerror', error => state.errors.push(error.message));
    await page.route('**/*', async route => {
      const req = route.request(), url = new URL(req.url()), pathname = url.pathname;
      const send = json => route.fulfill({ json });
      if (url.origin !== server.origin) { state.remote.push(req.url()); return route.abort(); }
      if (pathname.startsWith('/dashboard/')) return route.continue();
      if (pathname === '/auth/session/config') return send({ key_sessions_enabled: enabled });
      if (pathname === '/auth/dashboard-key') return send({ dashboard_key: 'pilot-fixture-read-key' });
      if (pathname === '/auth/whoami') return send({ key_kind: 'session', actor, permissions: { create_order: true, create_customer: true } });
      if (pathname === '/auth/session') return send({ key_kind: 'session', actor, session_token: token });
      if (pathname === '/auth/session/logout') return send({ signed_out: true });
      if (pathname.startsWith('/receipts/')) {
        const found = state.receipts.get(decodeURIComponent(pathname.split('/').at(-1)));
        assert.ok(found, pathname); return send(found);
      }
      if (pathname === '/resolve') {
        const body = req.postDataJSON();
        if (body.kind === 'unit') return send({ outcome: 'match', draft: { product_id: body.context.product_id, quantity: body.quantity, unit: body.query } });
        if (body.query === 'New Pilot Bakery') return send({ outcome: 'not_found', candidates: [] });
        const item = body.kind === 'supplier' ? { id: 91, name: 'Pilot Supplier' }
          : body.kind === 'customer' ? { id: 90, name: 'Pilot Bakery' }
          : body.kind === 'order' ? { id: 70, name: 'SO-PILOT-001' }
          : Object.values(products).find(p => p.name === body.query) || products.oats;
        return send({ outcome: 'match', match: { ...item, label: item.name } });
      }
      if (pathname === '/sales/orders/70') return send({ order_id: 70, order_number: 'SO-PILOT-001', customer: 'Pilot Bakery',
        lines: [{ line_id: 701, product_id: 13, product_name: products.finished.name, product: products.finished.name, remaining_lb: 40, quantity_lb: 40, is_service: false }] });
      if (pathname === '/entry/lots') return send({ lots: [] });
      if (pathname === '/ship/lots') return send({ product_id: Number(url.searchParams.get('product_id')), product: products.finished.name,
        case_size_lb: 20, lots: lotRows, suggested_lots: [{ ...lotRows[0], quantity_lb: Number(url.searchParams.get('quantity_lb')) || 40 }] });
      if (pathname === '/attachments') return send({ attachment_id: 81, kind: 'bol' });
      if (pathname.endsWith('/prepare')) {
        assert.equal(req.headers()['x-api-key'], token);
        const body = req.postDataJSON(); state.prepares.push({ path: pathname, body });
        const action = pathname === '/sales/orders/70/ship/prepare' ? 'ship_order'
          : pathname === '/sales/orders/prepare' ? 'create_order' : pathname === '/customers/prepare' ? 'create_customer'
          : pathname.startsWith('/void/') ? 'void' : pathname === '/inventory/found/prepare' ? 'found' : pathname.split('/')[1];
        let plan = [], draft = { happened_at: at, note: body.note };
        if (action === 'receive') Object.assign(draft, { product_name: products.oats.name, cases: body.cases, case_size_lb: body.case_size_lb });
        if (action === 'found') Object.assign(draft, { product_name: products.oats.name, quantity: body.quantity, found_location: body.found_location, notes: body.notes });
        if (action === 'make') { plan = structuredClone(makePlan); Object.assign(draft, { product_name: products.batch.name, batches: 1, total_output_lb: 35, ingredients: makeIngredients, lot_confirmation_group: { ingredients: makeIngredients.map(ingredient => ({ product_id: ingredient.ingredient_id, product_name: ingredient.ingredient_name, needed_lb: ingredient.needed_lb, candidates: ingredient.lots.map((lot, index) => ({ ...lot, lot_id: (ingredient.ingredient_id === 11 ? 201 : 301) + index, received_at: '2026-10-01T12:00:00Z' })) })) } }); }
        if (action === 'pack') Object.assign(draft, { source_product: products.batch.name, target_product: products.finished.name, target_product_id: 13, cases: body.cases, total_lb: 40 });
        if (action === 'void') Object.assign(draft, { voided_transaction_id: 99, reason: 'data_entry_error' });
        if (action === 'create_customer') Object.assign(draft, { customer: { id: 92, name: body.name, notes: body.notes } });
        if (action === 'create_order') Object.assign(draft, { customer: 'Pilot Bakery', notes: body.notes, lines: [{ product: products.finished.name, quantity_lb: 40 }] });
        if (action === 'ship' || action === 'ship_order') {
          const requested = action === 'ship_order' ? body.lines?.[0]?.lot_allocations : body.lot_allocations;
          const selected = requested || [{ lot_id: 101, quantity_lb: 40 }];
          plan = selected.map(p => ({ ...p, lot_code: lotRows.find(l => l.lot_id === p.lot_id).lot_code }));
          Object.assign(draft, { customer: 'Pilot Bakery', photo: 'pending', bol_reference: body.bol_reference,
            lines: [{ product: products.finished.name, quantity_lb: 40, lots: plan }] });
        }
        const confirmed = plan.every(lot => body.lot_confirmations?.some(c => c.lot_id === lot.lot_id && (action === 'make' ? ['tap', 'full_code', 'scan'] : ['full_code']).includes(c.method) && c.value === lot.lot_code));
        draft.input_plan = plan.map(lot => ({ ...lot, confirmed, ...(confirmed ? { confirmation: body.lot_confirmations.find(item => item.lot_id === lot.lot_id) } : {}) }));
        const ticket = `pilot-${++state.next}`;
        const prepared = { action, ticket, payload_hash: ticket + '-hash', actor, draft, can_commit: confirmed,
          blockers: confirmed ? [] : [{ code: 'LOT_NOT_CONFIRMED', message: 'Confirm every physical lot tag.' }], warnings: [], expires_at: '2026-10-19T20:00:00-04:00' };
        state.tickets.set(ticket, { prepared, body }); return send(prepared);
      }
      if (/^\/tickets\/[^/]+\/commit$/.test(pathname)) {
        const saved = state.tickets.get(pathname.split('/')[2]); assert.ok(saved); assert.equal(saved.prepared.can_commit, true);
        const body = req.postDataJSON(); state.commits.push({ path: pathname, body });
        const { draft, action } = saved.prepared;
        const prefix = ({ receive: 'RCV', make: 'MK', pack: 'PK', found: 'FND', void: 'VD', create_customer: 'CUS', create_order: 'ORD' })[action] || 'SHP';
        const number = `${prefix}-261019-${String(100 + state.commits.length).padStart(3, '0')}`;
        const result = { success: true, receipt_number: number, note: saved.body.note, found_location: saved.body.found_location,
          notes: saved.body.notes, ...(action === 'make' ? { input_plan: draft.input_plan } : {}), customer: action === 'create_customer' ? { id: 92, name: saved.body.name } : 'Pilot Bakery' };
        const shipped = prefix === 'SHP';
        if (shipped) Object.assign(result, { proof_status: body.attachment_ids?.includes(81) ? 'complete' : 'photo_pending', lots_used: draft.lines[0].lots });
        state.receipts.set(number, receipt(number, action, draft, result, shipped ? {
          shipment: { proof_status: result.proof_status, proof_due_at: '2026-10-19T23:59:00-04:00' },
          attachments: body.attachment_ids?.includes(81) ? [{ id: 81, kind: 'bol' }] : []
        } : {})); return send(result);
      }
      state.unknown.push(`${req.method()} ${pathname}`);
      return route.fulfill({ status: 500, json: { detail: 'Unexpected local pilot fixture request' } });
    });
    const check = name => results.push({ name, sessions: enabled ? 1 : 0, width, passed: true });
    const screenshot = async name => {
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${name}: horizontal overflow`);
      await page.screenshot({ path: path.join(out, `${enabled ? 'on' : 'off'}-${width}-${name}.png`), fullPage: true });
    };
    const open = async (file, query = '') => { await page.goto(`${server.origin}/dashboard/${file}${query}`); await page.evaluate(() => FLLedger.ready); };
    const pick = async (name, query, scope = page) => {
      const field = scope.locator(`[data-picker="${name}"]`); await field.locator('input').fill(query);
      await field.getByRole('button', { name: 'Search', exact: true }).click();
      await field.getByRole('button', { name: query, exact: true }).click();
    };
    const prepare = async () => {
      const before = state.commits.length;
      await page.evaluate(() => {
        window.pilotSubmitEvents = [];
        document.querySelector('#entry-form,#office-form')?.addEventListener('submit', event => window.pilotSubmitEvents.push({ target: event.target.id, prevented: event.defaultPrevented }));
        document.addEventListener('click', event => window.pilotSubmitEvents.push({ click: event.target.textContent, form: event.target.form?.id, type: event.target.type }), { once: true, capture: true });
      });
      const prepareButton = page.getByRole('button', { name: 'Prepare draft', exact: true });
      // Settle the focused number input before clicking after a full-page capture.
      await prepareButton.scrollIntoViewIfNeeded(); await prepareButton.focus(); await prepareButton.click();
      try { await page.getByRole('heading', { name: 'Draft — review before recording', exact: true }).waitFor(); }
      catch (error) {
        await page.screenshot({ path: path.join(out, `failed-prepare-${enabled}-${width}.png`), fullPage: true });
        throw Error(`${error.message}\n${JSON.stringify({ errors: state.errors, latest: state.prepares.at(-1),
          notice: await page.locator('#notice').textContent(), events: await page.evaluate(() => window.pilotSubmitEvents),
          form: await page.locator('#entry-form,#office-form').evaluate(node => ({ hidden: node.hidden, valid: node.checkValidity(), buttons: [...node.querySelectorAll('button')].map(b => ({ label: b.textContent, type: b.type, form: b.form?.id })) })),
          invalid: await page.locator(':invalid').evaluateAll(nodes => nodes.map(n => ({ name: n.name, value: n.value, message: n.validationMessage }))) })}`);
      }
      assert.equal(state.commits.length, before, 'Preparing cannot record');
    };
    const openReceipt = async number => { await open('receipt.html', `?number=${number}`); await page.getByRole('heading', { name: number, exact: true }).waitFor(); };
    const record = async (scope = page.locator('#resolution')) => {
      await scope.getByRole('button', { name: 'Record', exact: true }).click();
      const link = scope.locator('.good a'); await link.waitFor(); return await link.textContent();
    };
    const confirm = async () => {
      const taps = page.locator('#resolution [data-make-lot-confirm]');
      if (await taps.count()) {
        const ids = await taps.evaluateAll(nodes => nodes.filter(node => node.getAttribute('aria-pressed') !== 'true').map(node => node.dataset.makeLotConfirm));
        for (const id of ids) await page.locator(`#resolution [data-make-lot-confirm="${id}"]`).click();
        await page.waitForFunction(() => { const button = document.querySelector('#resolution .record-bar button'); return button && !button.disabled; });
        return;
      }
      const fields = page.locator('#resolution fieldset').filter({ hasText: 'Confirm physical lot:' });
      for (const field of await fields.all()) {
        const code = (await field.locator('legend').textContent()).replace('Confirm physical lot: ', '').split(' · ').at(-1);
        await field.locator('select').selectOption('full_code'); await field.locator('input').fill(code);
      }
      await page.getByRole('button', { name: 'Check confirmations → new draft', exact: true }).click();
      await page.waitForFunction(() => { const button = document.querySelector('#resolution .record-bar button'); return button && !button.disabled; });
    };

    // Stored drafts can contain pending-photo text after the current shipment is complete.
    await openReceipt('SHP-261019-001');
    let content = await page.locator('#content').innerText();
    assert.match(content, /Photo attached ✓/); assert.doesNotMatch(content, /Photo pending|Photo due|11:59 PM/);
    assert.match(content, /PACK261019-P1561/); assert.match(content, /PACK261019-P1562/);
    await screenshot('ship-complete-receipt'); check('Current attached proof replaces stale pending text and shows shipped lots');
    await openReceipt('SHP-261019-002'); content = await page.locator('#content').innerText();
    assert.match(content, /Photo pending|pending/); assert.match(content, /11:59 PM/); check('Missing photo keeps the genuine deadline');
    for (const [number, expected] of [
      ['RCV-261019-001', 'Receive pilot note'], ['MK-261019-001', 'Make pilot note'], ['PK-261019-001', 'Pack pilot note'],
      ['VD-261019-001', 'Void pilot note'], ['FND-261019-001', 'Dry storage rack 3'], ['ORD-261019-001', 'Deliver to loading door 2']
    ]) { await openReceipt(number); assert.ok((await page.locator('#content').innerText()).includes(expected)); await screenshot(number.slice(0, 3).toLowerCase() + '-receipt'); check(`${number.slice(0, 3)} receipt retains note or found location`); }
    for (const number of ['CUS-261019-001', 'ORD-261019-001']) {
      await openReceipt(number); content = await page.locator('#content').innerText();
      assert.match(content, /7:37 PM/); assert.doesNotMatch(content, /12:00 AM/); check(`${number.slice(0, 3)} receipt shows actual New York time`);
    }
    if (!enabled) {
      for (const file of ['entry.html', 'office.html']) { await open(file); assert.equal(await page.locator('#entry-form,#office-form').count(), 0); }
      assert.equal(state.prepares.length, 0); assert.equal(state.commits.length, 0); check('Session-off Record and Office gating is unchanged');
    } else {
      for (const action of ['receive', 'pack', 'found']) {
        await open('entry.html', `?action=${action}`); await page.locator('#entry-form').waitFor();
        if (action === 'pack') { await pick('source', products.batch.name); await pick('target', products.finished.name); await page.locator('[name=cases]').fill('2'); }
        else await pick('product', products.oats.name);
        if (action === 'receive') {
          await pick('supplier', 'Pilot Supplier'); await page.locator('[name=cases]').fill('2'); await page.locator('[name=case_size_lb]').fill('20');
          await page.locator('[name=bol_reference]').fill('PILOT-RECEIVE');
        }
        if (action === 'found') {
          await page.locator('[name=quantity]').fill('10'); await page.locator('[name=reason_code]').selectOption('physical_count');
          await page.locator('[name=found_location]').fill('Dry storage rack 3'); await page.locator('[name=notes]').fill('Found pilot note');
        } else {
          assert.equal(await page.locator('[name=note]').getAttribute('required'), null);
          await page.locator('[name=note]').fill(`${action === 'pack' ? 'Pack' : 'Receive'} pilot note`);
        }
        await screenshot(action + '-form'); await prepare();
        const text = await page.locator('#resolution').innerText();
        assert.ok(text.includes(action === 'found' ? 'Dry storage rack 3' : `${action === 'pack' ? 'Pack' : 'Receive'} pilot note`));
        if (action === 'pack') assert.ok(text.includes(products.finished.name), 'Pack draft must name the finished product');
        await screenshot(action + '-draft'); const number = await record(); await openReceipt(number);
        assert.ok((await page.locator('#content').innerText()).includes(action === 'found' ? 'Dry storage rack 3' : `${action === 'pack' ? 'Pack' : 'Receive'} pilot note`));
        check(`${action} form, request, draft and receipt preserve requested fields`);
      }
      await open('entry.html', '?action=make'); await pick('product', products.batch.name);
      await page.locator('[name=batches]').fill('1'); await page.locator('[name=confirmed_sku]').check(); await page.locator('[name=note]').fill('Make pilot note');
      await prepare(); assert.equal(state.prepares.at(-1).body.note, 'Make pilot note');
      assert.match(await page.locator('#resolution').innerText(), /Make pilot note/);
      const rows = page.locator('#resolution [data-make-ingredient]'), tags = page.locator('#resolution [data-make-lot-confirm]');
      assert.equal(await rows.count(), 2); assert.equal(await tags.count(), 3);
      for (const button of await tags.all()) assert.equal(await button.getAttribute('aria-pressed'), 'false');
      for (const input of await page.locator('[data-make-code-value]').all()) assert.equal(await input.inputValue(), '');
      assert.equal(await page.locator('#resolution .record-bar button').isDisabled(), true);
      const extraLot = page.locator('[data-make-lot-choice="262"] .make-lot-code');
      await screenshot('make-lots-suggested');
      assert.equal(await extraLot.isVisible(), false);
      assert.ok((await page.locator('[data-make-ingredient="11"] .make-selected-lots').innerText()).includes('OATS-001'));
      const before = state.prepares.length; await page.locator('[data-make-change="11"]').click();
      assert.equal(await page.locator('[data-make-ingredient="11"] [data-make-lot-choice]').count(), 6);
      assert.equal(await extraLot.isVisible(), false);
      const moreLots = page.locator('[data-make-more-lots="11"]');
      while (await moreLots.isVisible()) await moreLots.click();
      assert.equal(await page.locator('[data-make-ingredient="11"] [data-make-lot-choice]').count(), 62);
      assert.equal(await extraLot.isVisible(), true); assert.equal(state.prepares.length, before);
      for (const button of await tags.all()) assert.equal(await button.getAttribute('aria-pressed'), 'false');
      await screenshot('make-lots-draft');
      await page.locator('[data-make-lot-confirm="201"]').click();
      assert.equal(await page.locator('[data-make-progress]').textContent(), '0 of 2 checked', 'A split ingredient needs every FIFO tag');
      assert.equal(await page.locator('#resolution .record-bar button').isDisabled(), true);
      await confirm();
      assert.equal(state.prepares.at(-1).body.lot_confirmations.length, 3);
      assert.ok(state.prepares.at(-1).body.lot_confirmations.every(item => item.method === 'tap'));
      const made = await record(); await openReceipt(made); assert.match(await page.locator('#content').innerText(), /Make pilot note/);
      check('Make has compact ingredient rows, Change-only lot choices and an explicit check for every FIFO lot');

      // Default remains FIFO; selecting later lots changes exactly the posted allocation request.
      for (const shipMode of ['standalone', 'order']) {
        await open('entry.html', '?action=ship');
        await page.locator('[name=ship_mode]').selectOption(shipMode);
        if (shipMode === 'standalone') {
          await pick('customer', 'Pilot Bakery'); await pick('ship_product', products.finished.name); await page.locator('[name=quantity_lb]').fill('40');
        } else { await pick('order', 'SO-PILOT-001'); await page.locator('[name=ship_all]').check(); }
        await page.locator('[name=bol_reference]').fill('PILOT-BOL');
        const lots = page.locator('[data-ship-lots]').first(); await lots.locator('[name=shipment_lot_mode]').waitFor();
        assert.equal(await lots.locator('[name=shipment_lot_mode]').inputValue(), 'fifo');
        await prepare();
        let request = state.prepares.at(-1).body;
        assert.equal(request.lot_allocations, undefined); assert.ok(!request.lines?.some(line => line.lot_allocations));
        assert.match(await page.locator('#resolution').innerText(), /OPEN261014-P1561/);
        await page.getByRole('button', { name: 'Edit entry', exact: true }).click();
        await lots.locator('[name=shipment_lot_mode]').selectOption('manual');
        await lots.locator('[name=shipment_lot_id]').first().selectOption('102');
        await lots.locator('[name=shipment_lot_quantity]').first().fill('30');
        await lots.getByRole('button', { name: 'Add another shipped lot', exact: true }).click();
        await lots.locator('[name=shipment_lot_id]').last().selectOption('103');
        await lots.locator('[name=shipment_lot_quantity]').last().fill('10');
        assert.match(await lots.locator('[data-selected-ship-lot]').first().innerText(), /PACK261019-P1561.*80 lb/);
        assert.match(await lots.locator('[data-selected-ship-lot]').last().innerText(), /PACK261019-P1562.*20 lb/);
        assert.match(await lots.innerText(), /80/); assert.match(await lots.innerText(), /20/);
        await screenshot('ship-' + shipMode + '-lots'); await prepare(); request = state.prepares.at(-1).body;
        assert.deepEqual(shipMode === 'standalone' ? request.lot_allocations : request.lines[0].lot_allocations,
          [{ lot_id: 102, quantity_lb: 30 }, { lot_id: 103, quantity_lb: 10 }]);
        if (shipMode === 'order') assert.equal(request.ship_all, false);
        const draftText = await page.locator('#resolution').innerText();
        assert.match(draftText, /PACK261019-P1561/); assert.match(draftText, /PACK261019-P1562/); assert.doesNotMatch(draftText, /OPEN261014-P1561/);
        assert.equal(await page.locator('#resolution .record-bar button').isDisabled(), true); await confirm();
        await page.locator('#resolution input[type=file]').setInputFiles({ name: 'bol.png', mimeType: 'image/png', buffer: Buffer.from('pilot-bol-fixture') });
        await page.locator('#resolution').getByText('Photo attached', { exact: true }).waitFor();
        const number = await record(); assert.deepEqual(state.commits.at(-1).body.attachment_ids, [81]); await openReceipt(number);
        const text = await page.locator('#content').innerText(); assert.match(text, /PACK261019-P1561/); assert.match(text, /PACK261019-P1562/); assert.match(text, /Photo attached ✓/); assert.doesNotMatch(text, /Photo pending|Photo due/);
        await screenshot('ship-' + shipMode + '-receipt'); check(`${shipMode} shipment defaults FIFO and records worker-selected split lots with attached proof`);
      }

      await openReceipt('RCV-261019-099'); const note = page.locator('[data-void-note="99"]');
      await note.fill('Void pilot note'); await page.getByRole('button', { name: 'Prepare void for entry #99', exact: true }).click();
      await page.getByRole('heading', { name: 'Draft — review before recording', exact: true }).waitFor();
      assert.equal(state.prepares.at(-1).body.note, 'Void pilot note'); assert.match(await page.locator('#resolution').innerText(), /Void pilot note/);
      await screenshot('void-draft'); const voided = await record(); await openReceipt(voided); assert.match(await page.locator('#content').innerText(), /Void pilot note/);
      check('Void note travels from optional field through draft and receipt');

      await open('office.html'); await page.locator('#office-form').waitFor();
      const customerPicker = page.locator('[data-picker=customer]'); await customerPicker.locator('input').fill('New Pilot Bakery');
      await customerPicker.getByRole('button', { name: 'Search', exact: true }).click();
      await customerPicker.getByRole('button', { name: "Add 'New Pilot Bakery' as a new customer", exact: true }).click();
      const editor = page.locator('[data-customer-editor]'); await editor.locator('[name=notes]').fill('Customer pilot note');
      await editor.getByRole('button', { name: 'Prepare customer', exact: true }).click();
      const customerDraft = page.locator('[data-customer-draft]'); await customerDraft.getByRole('heading', { name: 'Draft — review before recording', exact: true }).waitFor();
      const customerNumber = await record(customerDraft); await openReceipt(customerNumber);
      assert.match(await page.locator('#content').innerText(), /7:37 PM/); await screenshot('customer-receipt'); check('Created customer receipt preserves actual time');
      await open('office.html'); await page.locator('#office-form').waitFor(); await pick('customer', 'Pilot Bakery');
      await page.locator('[name=customer_po]').fill('PILOT-PO'); await page.locator('[name=notes]').fill('Deliver to loading door 2');
      const line = page.locator('[data-lines]>fieldset').first(); await pick('product', products.finished.name, line); await line.locator('[name=quantity]').fill('2');
      await prepare(); assert.equal(state.prepares.at(-1).body.notes, 'Deliver to loading door 2'); await screenshot('order-draft');
      const ordered = await record(); await openReceipt(ordered); content = await page.locator('#content').innerText();
      assert.match(content, /Deliver to loading door 2/); assert.match(content, /7:37 PM/); assert.doesNotMatch(content, /12:00 AM/);
      await screenshot('order-receipt'); check('Created order receipt shows note and actual time');
    }
    assert.deepEqual(state.errors, []); assert.deepEqual(state.unknown, []); assert.deepEqual(state.remote, []);
    await context.close();
  }
  await fs.writeFile(path.join(out, 'results.json'), JSON.stringify({ results, passed: results.length, network: 'All API calls stubbed; off-origin requests refused; no database or OpenAI calls.' }, null, 2));
  console.log(`Pilot screens: ${results.length} passed; 390px and 1440px; modes ${modes.map(Number).join('/')}; no remote calls or databases.`);
} finally { await browser.close(); await server.close(); }

// Deterministic Fix A browser acceptance. All ledger/auth traffic is fixture-only.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const pilot = process.env.PILOT_SESSIONS === '1';
const mode = pilot ? 'sessions-on' : 'sessions-off';
const out = process.env.FIX_A_VISUAL_OUT || path.join(root, 'docs/validation/fl-assistant-round-a', mode);
await fs.mkdir(out, { recursive: true });
const server = await startStaticServer(root);
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
const sid = '0c101ab1-8c56-42a2-8f29-c048330030c3';
const actor = { id: 4, name: 'Arturo Garcia', role: 'floor' };
const token = 'fls_fixA_browser_fixture';
const receiptNumber = 'RCV-261015-610';
const id = suffix => '616eac35-0b6e-4288-b40f-920639a8' + String(suffix).padStart(4, '0');
const prepared = (action, draft, extra = {}) => ({ action, actor, can_commit: true, blockers: [], warnings: [], draft, ...extra });
const card = (suffix, action, draft, extra) => ({ kind: 'draft', id: id(suffix), prepared: prepared(action, draft, extra) });
const received = { product_name: 'Sunflower Seeds', quantity: 50, happened_at: '2026-10-15T11:19:00-04:00' };
const adjusted = { product_name: 'Sunflower Seeds', delta_lb: -5, current_quantity_lb: 2000, new_balance_lb: 1995, reason_code: 'physical_count', happened_at: '2026-10-15T11:19:00-04:00', notes: 'missing_receipt; UNIDENTIFIED', identity_status: 'UNIDENTIFIED' };
const inventory = { kind: 'read', tool: 'inventory_lookup', result: { results: [{ product: { name: 'Batch Classic Granola #9' }, total_on_hand: 17678,
  lots: [3000, 3000, 3000, 3000, 3000, 2678].map((quantity, index) => ({ lot: `26-10-15-CLAS-${String(index + 1).padStart(3, '0')}`, qty_on_hand: quantity, uom: '10 lb case' })) }] } };
const casesInventory = { kind: 'read', tool: 'inventory_lookup', result: { results: [{ product_name: 'Classic Granola retail', unit: 'cases', quantity: 12, lots: [{ lot_code: '26-10-15-RETL-001', quantity: 12, unit: 'cases' }] }] } };
const finishedInventory = { kind: 'read', tool: 'inventory_lookup', result: { results: [{ product: 'Granola Classic 20 LB', total_on_hand: 1600, unit_count: 80, lots: [{ lot: 'FIN-001', qty_on_hand: 1600, unit: 'case' }] }] } };
const correctionReason = { kind: 'refusal', status: 422, result: { detail: [{ loc: ['body', 'reason_code'], type: 'missing', msg: 'Field required' }] },
  reasons: [{ code: 'physical_count', label_en: 'Physical count', label_es: 'Conteo físico' }, { code: 'data_entry_error', label_en: 'Data-entry error', label_es: 'Error de captura' }] };
const entries = Array.from({ length: 610 }, (_, index) => ({ receipt_number: `RCV-261015-${String(index + 1).padStart(3, '0')}`, action: 'receive', actor,
  happened_at: new Date(Date.parse('2026-10-15T04:00:00Z') + index * 60000).toISOString(), entered_at: new Date(Date.parse('2026-10-15T04:00:00Z') + index * 60000).toISOString(),
  summary: 'Received 50 lb Sunflower Seeds', summary_es: 'Recibido 50 lb Sunflower Seeds' }));
const receipt = { success: true, receipt_number: receiptNumber, action: 'receive', status: 'committed', effective_status: 'posted', actor,
  happened_at: received.happened_at, entered_at: received.happened_at, draft: received,
  transactions: [{ id: 71, type: 'receive', effective_status: 'posted', allowed_actions: ['void'], lines: [{ product_name: 'Sunflower Seeds', quantity_lb: 50, lot_code: '26-10-15-SUNF-001' }] }] };
const results = [], screenshots = [];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; }
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: width === 390 ? 844 : 1000 }, timezoneId: 'America/New_York', colorScheme: 'light' });
    const state = { turns: [], records: [], pages: [], gates: new Map(), authGate: null, receiptMode: 'normal', receiptRows: null, heldRoutes: [], assets: [], unknown: [], errors: [] };
    await context.addInitScript(({ pilot, token, sid }) => {
      localStorage.setItem('fl-assistant-language', 'en');
      localStorage.setItem('fl-assistant-chat:4', sid);
      if (pilot) sessionStorage.setItem('fl-key-session-token', token);
      window.__signInWasVisible = false;
      new MutationObserver(() => {
        const panel = document.getElementById('sign-in-panel');
        if (panel && !panel.hidden && getComputedStyle(panel).display !== 'none') window.__signInWasVisible = true;
      }).observe(document, { subtree: true, childList: true, attributes: true, attributeFilter: ['hidden', 'class', 'style'] });
    }, { pilot, token, sid });
    const page = await context.newPage();
    page.setDefaultTimeout(6000);
    await page.clock.install({ time: new Date('2026-10-15T15:19:00Z') });
    page.on('pageerror', error => state.errors.push(error.message));
    page.on('response', response => {
      const url = new URL(response.url());
      if (url.pathname.startsWith('/dashboard/')) state.assets.push({ path: url.pathname, status: response.status() });
    });
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url()), pathname = url.pathname;
      if (pathname === '/dash/fl-assistant') return route.fulfill({ contentType: 'text/html', body: await fs.readFile(path.join(root, 'dashboard/fl-assistant.html'), 'utf8') });
      if (pathname.startsWith('/dashboard/')) return route.continue();
      if (pathname === '/auth/session/config') { if (state.authGate) await state.authGate.promise; return route.fulfill({ json: { key_sessions_enabled: pilot } }); }
      if (pathname === '/auth/whoami') { if (state.authGate) await state.authGate.promise; await pause(60); return route.fulfill({ json: { key_kind: pilot ? 'session' : 'actor', actor } }); }
      if (pathname === '/auth/session') return route.fulfill({ json: { key_kind: 'session', actor, session_token: token } });
      if (pathname === '/auth/dashboard-key') return route.fulfill({ json: { dashboard_key: 'fixture-scoped-dashboard-key' } });
      if (pathname === '/auth/session/logout') return route.fulfill({ json: { signed_out: true } });
      if (pathname.startsWith('/assistant/') || pathname === '/receipts') assert.equal(request.headers()['x-api-key'], pilot ? token : 'fixture-personal-key');
      const gate = state.gates.get(pathname);
      if (gate) { state.gates.delete(pathname); await gate.promise; }
      if (pathname === '/assistant/session') return route.fulfill({ json: { session_id: sid } });
      if (pathname === '/assistant/resume') { if (state.authGate) await state.authGate.promise; await pause(60); return route.fulfill({ json: { session_id: sid, turns: [], drafts: [] } }); }
      if (pathname === '/assistant/turn') {
        const body = request.postDataJSON(); state.turns.push(body);
        let cards;
        if (body.text === 'inventory') cards = [inventory, structuredClone(inventory), casesInventory];
        else if (body.text === 'finished inventory') cards = [finishedInventory];
        else if (body.text === 'Show my entries please') cards = [{ kind: 'read', tool: 'today_entries', result: { receipts: [] } }];
        else if (body.text === 'draft details') cards = [card(1, 'adjust', adjusted), card(2, 'found', { product_name: 'Cocoa Powder', quantity: 25, reason_code: 'missing_receipt', identity_status: 'UNIDENTIFIED' }), card(3, 'pack', { product_name: 'Classic Granola retail', cases: 12 }), card(4, 'receive', {}, { request: { product_id: 11007, quantity: 50 } })];
        else if (body.text === 'extra kosher receive') cards = [card(5, 'receive', received, { can_commit: false, blockers: [{ code: 'EXTRA_KOSHER_RECEIVE_BLOCKED', message: 'Extra-kosher receiving is not allowed.', message_es: 'No se permite recibir este producto extra kosher.' }] })];
        else if (body.text === 'choose reason') cards = [correctionReason];
        else if (body.text?.includes('correction reason is')) cards = [card(6, 'adjust', adjusted)];
        else if (body.text === 'possible duplicate') cards = [card(7, 'receive', received, { warnings: [{ code: 'POSSIBLE_DUPLICATE', message: 'A similar entry exists.', message_es: 'Existe una entrada similar.', requires_ack: true }] })];
        else if (body.text === 'confirm output') cards = [card(8, 'make', { product_name: 'Batch Classic Granola #9', total_output_lb: 400 }, { can_commit: false, blockers: [{ code: 'SKU_CONFIRMATION_REQUIRED', message: 'Confirm this output product.' }] })];
        else if (body.text?.startsWith('void ')) cards = [{ kind: 'refusal', status: 403, result: { detail: { message: 'Open the receipt screen to void this entry.' } } }];
        else cards = [card(9, 'receive', received)];
        return route.fulfill({ json: { cards } });
      }
      if (pathname === '/assistant/confirm-sku') return route.fulfill({ json: { card: card(10, 'make', { product_name: 'Batch Classic Granola #9', total_output_lb: 400 }) } });
      if (pathname === '/assistant/record') { const body = request.postDataJSON(); state.records.push(body); const receive = body.draft_id === id(9); return route.fulfill({ json: { kind: 'receipt', result: { success: true, receipt_number: receive ? 'RCV-261015-001' : 'MK-261015-001', product_name: receive ? 'Sunflower Seeds' : 'Batch Classic Granola #9', transaction_id: 17 } } }); }
      if (pathname === '/assistant/cancel') return route.fulfill({ json: { cancelled: true } });
      if (pathname === '/receipts') {
        const offset = Number(url.searchParams.get('offset') || 0), limit = Number(url.searchParams.get('limit'));
        state.pages.push({ offset, limit, summary: url.searchParams.get('summary'), date: url.searchParams.get('date') });
        if (state.receiptMode === 'timeout') { state.heldRoutes.push(route); return; }
        if (state.receiptMode === 'permission') return route.fulfill({ status: 403, json: { detail: { message: 'You do not have permission to view these entries.' } } });
        if (state.receiptMode === 'server') return route.fulfill({ status: 503, json: { detail: 'Today entries are unavailable. Please try again.' } });
        if (state.receiptMode === 'network') return route.abort('failed');
        const visible = state.receiptRows === null ? entries : entries.slice(0, state.receiptRows);
        const rows = visible.slice(offset, offset + limit).map(row => state.receiptRows === 6 ? { ...row, actor: { id: 8, name: 'Rosa Perez', role: 'floor' } } : row);
        return route.fulfill({ json: { receipts: rows, has_more: offset + rows.length < visible.length, next_offset: offset + rows.length < visible.length ? offset + rows.length : null } });
      }
      if (pathname === '/receipts/' + receiptNumber) return route.fulfill({ json: receipt });
      if (pathname === '/favicon.ico') return route.fulfill({ status: 204 });
      state.unknown.push(pathname); return route.abort('blockedbyclient');
    });
    const take = async (name, locator) => {
      if (locator) await locator.scrollIntoViewIfNeeded();
      const screenshot = path.join(out, `${String(width)}-${name}.png`);
      await page.screenshot({ path: screenshot }); screenshots.push(screenshot); return screenshot;
    };
    const login = async () => {
      if (!pilot) {
        await page.locator('#sign-in-panel').waitFor({ state: 'visible' });
        await page.locator('#actor-key').fill('fixture-personal-key');
        await page.locator('#sign-in-form button').click();
      }
      await page.locator('#chat-panel').waitFor({ state: 'visible' });
      await page.waitForFunction(() => !document.getElementById('send').disabled);
    };
    const reset = async (url = '/dashboard/fl-assistant.html') => {
      state.receiptMode = 'normal'; state.receiptRows = null; state.authGate = null; state.assets = []; state.pages = []; state.turns = []; state.records = [];
      await page.goto(server.origin + url, { waitUntil: 'domcontentloaded' }); await login();
    };
    const send = async text => {
      await page.locator('#message').fill(text); await page.locator('#send').click();
      await page.waitForFunction(() => !document.getElementById('send').disabled);
    };
    const test = async (item, name, body) => {
      const start = Date.now(), before = screenshots.length;
      try { await reset(); await body(); results.push({ item, name, mode, width, status: 'passed', elapsed_ms: Date.now() - start, screenshots: screenshots.slice(before) }); }
      catch (error) { await take(`${String(item).padStart(2, '0')}-FAIL`).catch(() => {}); results.push({ item, name, mode, width, status: 'failed', error: error.stack || String(error), elapsed_ms: Date.now() - start, screenshots: screenshots.slice(before) }); }
    };
    await test(1, 'All today entries newest first, fast pagination, 15 second error', async () => {
      const started = Date.now();
      await page.locator(width === 390 ? '#today-entries-mobile' : '#today-entries').click();
      await page.waitForFunction(() => document.querySelectorAll('.today-entry').length === 610);
      const elapsed = Date.now() - started;
      assert.ok(elapsed < 5000, `610 entries took ${elapsed} ms`);
      assert.deepEqual(state.pages.map(p => p.offset), [0, 500]); assert.ok(state.pages.every(p => p.summary === 'true' && p.limit === 500 && p.date === '2026-10-15'));
      const rows = await page.locator('.today-entry').allTextContents();
      assert.match(rows[0], /RCV-261015-610/); assert.match(rows.at(-1), /RCV-261015-001/);
      const values = (await page.locator('.today-entry').first().textContent()).split(' — ');
      assert.match(values[0], /^\d{1,2}:\d{2} [AP]M$/); assert.equal(values[1], 'Received 50 lb Sunflower Seeds'); assert.equal(values[2], receiptNumber); assert.equal(values[3], actor.name);
      assert.ok(await page.locator('.today-entry').first().evaluate(row => row.scrollWidth <= row.clientWidth));
      await take('01-today-entries', page.locator('.read-summary').last());
      state.receiptMode = 'timeout';
      await page.locator(width === 390 ? '#today-entries-mobile' : '#today-entries').evaluate(button => button.click());
      await page.waitForFunction(() => document.getElementById('send').disabled);
      await page.clock.fastForward(15100);
      await page.getByText('Today’s entries could not load within 15 seconds. Please try again.', { exact: true }).waitFor();
      assert.equal(await page.locator('#send').isDisabled(), false); assert.equal(await page.getByRole('button', { name: 'Try again', exact: true }).count(), 1);
      await take('01-today-timeout', page.locator('.card').last());
      for (const route of state.heldRoutes.splice(0)) await route.abort().catch(() => {});
    });
    await test(2, 'Plain inventory sentence, readable table, correct units and result dedupe', async () => {
      await send('inventory');
      assert.equal(await page.locator('.assistant-message .card').count(), 2);
      const first = page.locator('.assistant-message .card').first();
      assert.match(await first.innerText(), /Batch Classic Granola #9: 17,678 lb in 6 lots/);
      assert.deepEqual(await first.locator('th').allTextContents(), ['Lot', 'Quantity', 'Unit']);
      assert.deepEqual(await first.locator('tbody tr').first().locator('td').allTextContents(), ['26-10-15-CLAS-001', '3,000', 'lb']);
      assert.equal(await page.locator('#messages pre,#messages details').count(), 0); assert.doesNotMatch(await page.locator('#messages').innerText(), /qty_on_hand|total_on_hand|10 lb case|\{"/);
      const cases = page.locator('.assistant-message .card').last();
      assert.match(await cases.innerText(), /12 cases/); assert.equal(await cases.locator('tbody td').last().textContent(), 'cases');
      await take('02-inventory', first); await take('02-case-units', cases);
    });
    await test(3, 'Plain complete draft fields, current-day time, adjustment before and after', async () => {
      await send('draft details');
      const cards = page.locator('.assistant-message .card'); assert.equal(await cards.count(), 4);
      const text = await page.locator('#messages').innerText();
      assert.match(text, /Today 11:19 AM/); assert.match(text, /Physical count/); assert.match(text, /Missing receipt/); assert.match(text, /Lot identity needs to be checked/); assert.match(text, /2,000 → 1,995 lb/); assert.match(text, /12 cases/);
      assert.doesNotMatch(text, /physical_count|missing_receipt|UNIDENTIFIED|2026-10-15T/);
      for (const draft of await cards.all()) { const labels = await draft.locator('dt').allTextContents(); assert.ok(labels.includes('Product') && labels.includes('Quantity')); }
      assert.match(await cards.last().innerText(), /Product #11007/); assert.match(await cards.last().innerText(), /50 lb/);
      await take('03-adjust-draft', cards.first()); await take('03-complete-fallback-draft', cards.last());
    });
    await test(4, 'Refused entry is a clear refusal with no Record control', async () => {
      await send('extra kosher receive'); const refused = page.locator('.card.refusal');
      assert.equal(await refused.count(), 1); assert.match(await refused.innerText(), /CANNOT RECORD/); assert.match(await refused.innerText(), /Extra-kosher receiving is not allowed/);
      assert.equal(await refused.getByRole('button', { name: 'Record', exact: true }).count(), 0); assert.doesNotMatch(await refused.innerText(), /DRAFT · NOT RECORDED/);
      await take('04-refusal', refused);
    });
    await test(5, 'Correction-reason button submits immediately', async () => {
      await send('choose reason'); const before = state.turns.length;
      await page.getByRole('button', { name: 'Physical count', exact: true }).click();
      await page.waitForFunction(() => !document.getElementById('send').disabled);
      assert.equal(state.turns.length, before + 1); assert.equal(state.turns.at(-1).text, 'The correction reason is physical_count.'); assert.equal(await page.locator('#message').inputValue(), '');
      assert.equal(await page.locator('.user-message').last().textContent(), 'The correction reason is Physical count.');
      await take('05-reason-submitted', page.locator('[data-draft-id]').last());
    });
    await test(6, 'Duplicate warning has the complete explicit checkbox label', async () => {
      await send('possible duplicate'); const label = 'Yes, this is a separate entry, record it anyway.';
      const checkbox = page.getByRole('checkbox', { name: label, exact: true }), record = page.getByRole('button', { name: 'Record', exact: true });
      assert.equal(await checkbox.count(), 1); assert.equal(await record.isDisabled(), true); await checkbox.check(); assert.equal(await record.isDisabled(), false);
      await take('06-duplicate-label', page.locator('.card').last());
    });
    await test(7, 'Send, Confirm and Record show a spinner and Working feedback', async () => {
      const gate = deferred(); state.gates.set('/assistant/turn', gate);
      await page.locator('#message').fill('confirm output'); await page.locator('#send').click();
      await page.locator('#send.is-working').waitFor(); assert.equal(await page.locator('#send').getAttribute('aria-busy'), 'true'); assert.equal(await page.locator('#working-status').isVisible(), true);
      assert.match(await page.locator('#send').textContent(), /Working/); assert.equal(await page.locator('#send').evaluate(button => getComputedStyle(button, '::before').animationName), 'fl-spin');
      await take('07-send-working'); gate.resolve(); await page.waitForFunction(() => !document.getElementById('send').disabled);
      const confirmGate = deferred(); state.gates.set('/assistant/confirm-sku', confirmGate);
      const confirm = page.getByRole('button', { name: 'Confirm this output SKU', exact: true }); await confirm.click();
      await page.locator('.card button.is-working').waitFor(); assert.match(await page.locator('.card button.is-working').textContent(), /Working/);
      await take('07-confirm-working', page.locator('.card').last()); confirmGate.resolve(); await page.waitForFunction(() => !document.getElementById('send').disabled);
      const recordGate = deferred(); state.gates.set('/assistant/record', recordGate);
      await page.getByRole('button', { name: 'Record', exact: true }).last().click();
      await page.locator('.card button.is-working').waitFor(); assert.equal(await page.locator('.card button.is-working').getAttribute('aria-busy'), 'true'); assert.equal(await page.locator('#working-status').isVisible(), true);
      await take('07-record-working', page.locator('.card').last()); recordGate.resolve(); await page.locator('.receipt-number').waitFor(); assert.equal(state.records.length, 1);
    });
    await test(8, 'Header controls and receipt tools stay within desktop and iPhone widths', async () => {
      await page.evaluate(() => { document.documentElement.dataset.flWriteUi = 'on'; window.scrollTo(0, 0); });
      for (const language of ['en', 'es']) {
        if (language === 'es') await page.locator('#language').click();
        const boxes = await page.evaluate(() => {
          const header = document.querySelector('.topbar').getBoundingClientRect(), tools = document.querySelector('.receipt-tools').getBoundingClientRect(), chat = document.querySelector('#chat-panel').getBoundingClientRect();
          return { width: innerWidth, scrollWidth: document.documentElement.scrollWidth, headerBottom: header.bottom, toolsBottom: tools.bottom, chatTop: chat.top,
            controls: [...document.querySelectorAll('.header-actions button,.receipt-tools input,.receipt-tools button,.receipt-tools>a')].filter(node => getComputedStyle(node).display !== 'none').map(node => { const b = node.getBoundingClientRect(), range = document.createRange(); range.selectNodeContents(node); return { text: node.textContent, left: b.left, right: b.right, bottom: b.bottom, width: b.width, lines: node.matches('.header-actions button') ? range.getClientRects().length : 1 }; }) };
        });
        assert.equal(boxes.width, boxes.scrollWidth); assert.ok(boxes.toolsBottom <= boxes.headerBottom); assert.ok(boxes.chatTop >= boxes.toolsBottom);
        assert.ok(boxes.controls.every(c => c.width > 0 && c.left >= 0 && c.right <= width && c.bottom <= boxes.headerBottom && c.lines === 1), JSON.stringify(boxes));
        await take(`08-header-${language}`);
      }
    });
    await test(9, 'Short Assistant URL loads every absolute asset', async () => {
      await reset('/dash/fl-assistant');
      for (const name of ['dashboard-runtime.js', 'ledger-forms.css', 'receipt-shell.js', 'assistant-display.js', 'fl-assistant.js', 'fl-assistant.css', 'key-session.js']) {
        assert.ok(state.assets.some(asset => asset.path === '/dashboard/' + name && asset.status === 200), `${name} was not served from /dashboard/`);
      }
      assert.equal(new URL(page.url()).pathname, '/dash/fl-assistant'); await take('09-short-url-assets');
    });
    await test(10, 'Session restoration does not flash the sign-in screen', async () => {
      const gate = deferred(); state.authGate = gate;
      await page.goto(server.origin + '/dashboard/fl-assistant.html', { waitUntil: 'domcontentloaded' });
      await page.locator('#auth-loading').waitFor({ state: 'visible' });
      assert.equal(await page.locator('#sign-in-panel').isVisible(), false); assert.equal(await page.evaluate(() => window.__signInWasVisible), false);
      await take('10-checking-session'); gate.resolve(); state.authGate = null; await login();
      if (pilot) assert.equal(await page.evaluate(() => window.__signInWasVisible), false);
      assert.equal(await page.locator('#auth-loading').isVisible(), false); assert.equal(await page.locator('#sign-in-panel').isVisible(), false);
      await take('10-restored-session');
    });
    await test(11, 'Void hand-off opens the requested receipt screen', async () => {
      await send('void ' + receiptNumber);
      const link = page.getByRole('link', { name: `Open ${receiptNumber} to void`, exact: true });
      const target = new URL(await link.getAttribute('href'), page.url()); assert.equal(target.pathname, '/dashboard/receipt.html'); assert.equal(target.searchParams.get('number'), receiptNumber);
      await link.evaluate(node => node.scrollIntoView({ block: 'center' }));
      await take('11-void-handoff', link);
      await link.click(); await page.getByRole('heading', { name: receiptNumber, exact: true }).waitFor();
      assert.equal(new URL(page.url()).searchParams.get('number'), receiptNumber);
      if (pilot) {
        const prepareVoid = page.getByRole('button', { name: 'Prepare void for entry #71', exact: true });
        await prepareVoid.waitFor(); await prepareVoid.evaluate(node => node.scrollIntoView({ block: 'center' }));
      }
      await take('11-receipt-void-screen'); assert.equal(state.records.length, 0);
    });
    await test(12, 'Legacy own-entry tool stays distinct from six team entries in EN and ES', async () => {
      state.receiptRows = 6;
      await send('Show my entries please');
      const own = page.locator('.assistant-message .card').first();
      assert.equal(await own.locator('.read-summary').textContent(), "You've recorded 0 entries today. Team total: see Today's entries.");
      assert.equal(state.turns.at(-1).text, 'Show my entries please');
      await page.locator(width === 390 ? '#today-entries-mobile' : '#today-entries').click();
      await page.waitForFunction(() => !document.getElementById('send').disabled);
      const team = page.locator('.assistant-message .card').last();
      assert.equal(await team.locator('.read-summary').textContent(), '6 entries today.');
      assert.equal(await team.locator('.today-entry').count(), 6);
      assert.equal(await own.locator('.read-summary').textContent(), "You've recorded 0 entries today. Team total: see Today's entries.");
      await take('12-own-and-team-entries', team);
      await page.locator('#language').click();
      assert.equal(await own.locator('.read-summary').textContent(), 'Has registrado 0 entradas hoy. Total del equipo: consulta las entradas de hoy.');
      assert.equal(await team.locator('.read-summary').textContent(), '6 entradas hoy.');
      await take('12-own-and-team-entries-es', team);
    });
    await test(13, 'Finished inventory shows pounds alongside cases without changing weight units', async () => {
      await send('finished inventory');
      const stock = page.locator('.assistant-message .card').last();
      assert.equal(await stock.locator('.read-summary').textContent(), 'Granola Classic 20 LB: 1,600 lb (80 cases) in 1 lot.');
      assert.deepEqual(await stock.locator('tbody tr').first().locator('td').allTextContents(), ['FIN-001', '1,600', 'lb']);
      await take('13-finished-lb-cases', stock);
      await page.locator('#language').click();
      assert.equal(await stock.locator('.read-summary').textContent(), 'Granola Classic 20 LB: 1,600 lb (80 cajas) en 1 lote.');
    });
    await test(14, 'Immediate Today permission, server and network failures keep their own message', async () => {
      for (const [kind, message] of [
        ['permission', 'You do not have permission to view these entries.'],
        ['server', 'Today entries are unavailable. Please try again.'],
        ['network', 'Connection failed. Please try again.']
      ]) {
        state.receiptMode = kind;
        await page.locator(width === 390 ? '#today-entries-mobile' : '#today-entries').click();
        await page.waitForFunction(() => !document.getElementById('send').disabled);
        const error = page.locator('.assistant-message .card').last();
        assert.ok((await error.innerText()).includes(message));
        assert.doesNotMatch(await error.innerText(), /15 seconds|15 segundos/);
        assert.equal(await error.getByRole('button', { name: 'Try again', exact: true }).count(), 1);
        await take(`14-today-${kind}`, error);
      }
      state.receiptMode = 'normal';
      await page.getByRole('button', { name: 'Try again', exact: true }).last().click();
      await page.waitForFunction(() => !document.getElementById('send').disabled);
      assert.equal(await page.locator('.assistant-message .card').last().locator('.today-entry').count(), 610);
    });
    await test(15, 'Recorded card uses the confirmed receipt number as its identifier in EN and ES', async () => {
      await send('receive seeds');
      await page.getByRole('button', { name: 'Record', exact: true }).click();
      const recorded = page.locator('.card.receipt');
      await recorded.locator('.receipt-number').waitFor();
      assert.equal(await recorded.locator('.receipt-number').textContent(), 'RCV-261015-001');
      assert.doesNotMatch(await recorded.innerText(), /Transaction|Transacción/);
      assert.equal(await recorded.locator('dd').allTextContents().then(values => values.includes('17')), false);
      assert.equal(state.records.length, 1);
      await take('15-recorded-receipt-number', recorded);
      await page.locator('#language').click();
      assert.equal(await recorded.locator('.receipt-number').textContent(), 'RCV-261015-001');
      assert.doesNotMatch(await recorded.innerText(), /Transaction|Transacción/);
    });
    assert.deepEqual(state.unknown, [], 'Unstubbed network requests'); assert.deepEqual(state.errors, [], 'Browser JavaScript errors');
    await context.close();
  }
} finally {
  await browser.close(); await server.close();
  const failed = results.filter(result => result.status === 'failed');
  const summary = { mode, passed: results.length - failed.length, failed: failed.length, cases: results.length, screenshot_count: screenshots.length, results };
  await fs.writeFile(path.join(out, 'results.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ mode, passed: summary.passed, failed: summary.failed, cases: summary.cases, screenshot_count: summary.screenshot_count, out }));
  for (const failure of failed) console.error(`Item ${failure.item}, ${failure.width}px: ${failure.error}`);
  if (failed.length) process.exitCode = 1;
}

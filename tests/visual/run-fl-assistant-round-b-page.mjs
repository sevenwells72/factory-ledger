// Real dashboard -> HTTP -> branch TestClient -> disposable PostgreSQL.
// FIXB_REAL_MODEL=1 uses actual OpenAI selection; default queues only selection.
// No endpoint response, display data, or page asset is intercepted in Chromium.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const python = process.env.FIXB_PYTHON || path.join(root, '.venv-test/bin/python');
const pg = process.env.FIXB_PG_BIN || '/opt/homebrew/opt/postgresql@17/bin';
const realModel = process.env.FIXB_REAL_MODEL === '1';
const bugQueriesOnly = process.env.FIXB_PAGE_BUG_QUERIES_ONLY === '1';
const selectedItems = process.env.FIXB_PAGE_ITEMS ? new Set(process.env.FIXB_PAGE_ITEMS.split(',').map(Number)) : null;
const evidenceRoot = process.env.FIXB_PAGE_OUT || path.join(root, 'docs/validation/fl-assistant-round-b', realModel ? 'real-model' : 'real-page');
await fs.mkdir(evidenceRoot, { recursive: true });
const localEnv = { ...process.env };
for (const name of ['PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE', 'PGOPTIONS']) delete localEnv[name];
const modes = process.env.PILOT_SESSIONS === undefined ? [0, 1] : [Number(process.env.PILOT_SESSIONS)];
const widths = process.env.FIXB_PAGE_WIDTHS ? process.env.FIXB_PAGE_WIDTHS.split(',').map(Number) : [1440, 390];
const evidence = { runner: 'tests/visual/run-fl-assistant-round-b-page.mjs', local_only: true, actual_page: true, model_tool_selection_stubbed: !realModel, created_at: new Date().toISOString(), results: [], supplemental_screenshots: [], modes: [] };
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
const resolve = (kind, query, context = {}) => ['resolve', { kind, query, context }];
let failed = 0;
try {
  for (const mode of modes) {
    const modeName = `sessions-${mode ? 'on' : 'off'}`;
    const db = `fl_fixB2_page_${process.pid}_${mode}_${realModel ? 'real' : 'queued'}`;
    const run = (command, args) => execFileSync(path.join(pg, command), args, { cwd: root, env: localEnv, stdio: 'pipe' });
    let child;
    run('createdb', ['-h', '127.0.0.1', db]);
    try {
      run('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-c', 'CREATE EXTENSION pg_trgm']);
      run('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-f', path.join(root, 'tests/schema/schema.sql')]);
      child = spawn(python, ['-u', path.join(root, 'tests/visual/assistant_round_b_page_bridge.py')], {
        cwd: root, env: { ...localEnv, TEST_DATABASE_URL: `postgresql://127.0.0.1:5432/${db}`, KEY_SESSIONS_ENABLED: String(mode) }, stdio: ['ignore', 'pipe', 'pipe'],
      });
      let diagnostic = '';
      child.stderr.on('data', chunk => { diagnostic += chunk; process.stderr.write(chunk); });
      const lines = readline.createInterface({ input: child.stdout });
      const ready = await Promise.race([
        new Promise((resolve, reject) => {
          lines.once('line', line => { try { resolve(JSON.parse(line)); } catch { reject(new Error('Invalid fixture startup output')); } });
          child.once('exit', code => reject(new Error(`Fixture exited ${code}: ${diagnostic}`)));
        }),
        new Promise((_, reject) => { const timer = setTimeout(() => reject(new Error('Fixture startup timed out')), 30000); timer.unref(); }),
      ]);
      const origin = `http://127.0.0.1:${ready.port}`;
      const fixture = await (await fetch(origin + '/test/fixture')).json();
      const before = await (await fetch(origin + '/test/verify')).json();
      for (const width of widths) {
        const out = path.join(evidenceRoot, modeName);
        await fs.mkdir(out, { recursive: true });
        const context = await browser.newContext({ viewport: { width, height: width === 390 ? 844 : 1000 }, timezoneId: 'America/New_York', colorScheme: 'light' });
        await context.addInitScript(() => localStorage.setItem('fl-assistant-language', 'en'));
        const page = await context.newPage();
        page.setDefaultTimeout(realModel ? 110000 : 10000);
        const errors = [], apiFailures = [], assetResponses = [], turns = [], records = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('request', request => {
          const pathname = new URL(request.url()).pathname;
          if (pathname === '/assistant/turn') turns.push(request.postDataJSON());
          if (pathname === '/assistant/record') records.push(request.postDataJSON());
        });
        page.on('response', response => {
          const pathname = new URL(response.url()).pathname;
          if (pathname.startsWith('/dashboard/')) assetResponses.push({ path: pathname, status: response.status() });
          if (response.status() >= 400) apiFailures.push({ path: pathname, status: response.status() });
        });
        await page.goto(origin + '/dash/fl-assistant');
        const login = async () => {
          if (!mode || !(await page.evaluate(() => !!sessionStorage.getItem('fl-key-session-token')))) {
            await page.locator('#sign-in-panel').waitFor({ state: 'visible' });
            await page.locator('#actor-key').fill('fixb-page-floor');
            await page.locator('#sign-in-form button').click();
          }
          await page.locator('#chat-panel').waitFor({ state: 'visible' });
          await page.waitForFunction(() => !document.getElementById('send').disabled);
        };
        await login();
        const queue = async calls => {
          const response = await fetch(origin + '/test/model', { method: 'POST', body: JSON.stringify(calls), headers: { 'Content-Type': 'application/json' } });
          assert.equal(response.status, 200, 'Model queue must start empty');
        };
        const send = async (text, calls) => {
          await queue(calls);
          const count = await page.locator('.assistant-message').count();
          await page.locator('#message').fill(text);
          await page.locator('#send').click();
          await page.waitForFunction(() => !document.getElementById('send').disabled, null, { timeout: realModel ? 180000 : 15000 });
          assert.ok(await page.locator('.assistant-message').count() > count, 'An assistant answer must render');
          return page.locator('.assistant-message .card').last();
        };
        const fresh = async () => {
          // Fixture reset: the mobile page has no visible desktop rail button.
          // Preserve authentication preferences, clear only the conversation ID,
          // then enter a genuine new assistant session through the page itself.
          await page.evaluate(() => { for (const key of Object.keys(localStorage)) if (key.startsWith('fl-assistant-chat:')) localStorage.removeItem(key); });
          await page.reload({ waitUntil: 'domcontentloaded' });
          await login();
          assert.equal(await page.locator('.assistant-message').count(), 0);
        };
        const take = async name => {
          const screenshot = path.join(out, `${width}-${name}.png`);
          await page.screenshot({ path: screenshot, fullPage: false });
          return path.relative(root, screenshot);
        };
        const check = async (item, language, label, callback, recordsAllowed = false) => {
          if (bugQueriesOnly && item < 10) return;
          if (selectedItems && !selectedItems.has(item)) return;
          await fresh();
          const start = Date.now(), priorRecords = records.length, priorFailures = apiFailures.length;
          const row = { item, language, label, width, sessions_enabled: Boolean(mode), real_model: realModel };
          try {
            await callback(language);
            if (!recordsAllowed) assert.equal(records.length, priorRecords, 'Read/prepare walkthrough must not post a record');
            assert.equal(await page.getByRole('button', { name: /^(Try again|Reintentar)$/ }).count(), 0, 'Valid answers should not ask the worker to retry');
            assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'Page must not overflow at the viewport width');
            row.status = 'passed';
          } catch (error) {
            row.status = 'failed'; row.error = error.stack || String(error); failed++;
          }
          row.elapsed_ms = Date.now() - start;
          row.api_failures = apiFailures.slice(priorFailures);
          row.screenshot = await take(`${String(item).padStart(2, '0')}-${label}-${language}${row.status === 'failed' ? '-FAIL' : ''}`);
          row.visible_text = await page.locator('#messages').innerText();
          evidence.results.push(row);
          console.log(`${modeName} ${width}px ${language} ${label}: ${row.status}`);
        };
        for (const lang of ['en', 'es']) {
          const es = lang === 'es';
          await check(1, lang, 'open-orders', async () => {
            const card = await send(es ? '¿Qué pedidos abiertos hay?' : 'What orders are open?', [['open_orders', { customer: null, product: null }]]);
            const text = await card.innerText();
            for (const order of fixture.orders) { assert.ok(text.includes(order.order_number)); assert.ok(text.includes(order.customer)); }
            assert.match(text, /Granola Classic 25 LB/); assert.match(text, /Granola Kookies & Kreme 25 LB/);
            assert.match(text, es ? /pedidos abiertos/i : /open orders/i);
            assert.ok(await card.locator('table, .order-list .order-entry').count() > 0, 'Orders must appear in readable rows');
            assert.doesNotMatch(text, /FL returned this information|FL devolvió esta información|\{"/);
            for (const [index, order] of (await card.locator('.order-entry').all()).entries()) {
              await order.evaluate(element => element.scrollIntoView({ block: 'center' }));
              evidence.supplemental_screenshots.push(await take(`01-open-order-${index + 1}-${lang}`));
            }
          });
          await check(2, lang, 'lot-quantity', async () => {
            const card = await send(es ? '¿Cuánto hay en el lote SUNFLOWER-ABCD?' : 'How much is in lot SUNFLOWER-ABCD?', [['lot_lookup', { lot_code: 'SUNFLOWER-ABCD' }]]);
            const text = await card.innerText();
            assert.match(text, /SUNFLOWER-ABCD/); assert.match(text, /100 lb/);
            assert.doesNotMatch(text, /FL returned this information|FL devolvió esta información/);
          });
          await check(3, lang, 'receive-supplier', async () => {
            const card = await send(es ? 'Recibí 10 lb de semillas de girasol' : 'Received 10 lb of Sunflower Seeds', [resolve('product', es ? 'semillas de girasol' : 'Sunflower Seeds'), ['prepare_receive', { product_id: 283 }]]);
            const text = await card.innerText();
            assert.match(text, es ? /proveedor/i : /supplier/i);
            const suppliers = card.locator('.choice-button');
            assert.ok(await suppliers.count() > 0 && await suppliers.count() <= 6, `Show at most six suppliers, got ${await suppliers.count()}`);
            assert.ok(await card.locator('input[type="search"]').count() > 0, 'Supplier choices need a search field');
            assert.doesNotMatch(text, /SUPPLIER_REQUIRED|supplier_id/);
            evidence.supplemental_screenshots.push(await take(`03-receive-supplier-initial-${lang}`));
            await card.locator('.supplier-search').evaluate(element => element.scrollIntoView({ block: 'center' }));
            evidence.supplemental_screenshots.push(await take(`03-receive-supplier-search-${lang}`));
            await card.locator('input[type="search"]').fill('Supplier 57');
            await card.locator('.supplier-search button').click();
            await page.waitForFunction(() => !document.getElementById('send').disabled);
            const searched = page.locator('.assistant-message .card').last();
            assert.deepEqual(await searched.locator('.supplier-choices .choice-button').allTextContents(), ['Supplier 57']);
          });
          await check(4, lang, 'make-lot-confirmation', async () => {
            let card = await send(es ? 'Hicimos 1 tanda del producto 90002' : 'Made 1 batch of product 90002', [resolve('product', '90002', { action: 'make' }), ['prepare_make', { product_id: 285, batches: 1 }]]);
            if (await card.locator('.choice-button').count()) {
              await queue([['prepare_make', { product_id: 285, batches: 1 }]]);
              await card.locator('.choice-button').first().click();
              await page.waitForFunction(() => !document.getElementById('send').disabled);
              card = page.locator('.assistant-message .card').last();
            }
            const instruction = es ? 'Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.' : 'Type the last 4 characters from the tag on each bag you used.';
            assert.equal((await card.innerText()).split(instruction).length - 1, 1, 'Make has exactly one instruction line');
            assert.equal(await card.getByRole('button', { name: es ? 'Otro lote' : 'Different lot', exact: true }).count(), 2);
            assert.equal(await card.locator('.lot-confirmation input').count(), 2);
            for (const input of await card.locator('.lot-confirmation input').all()) assert.equal(await input.inputValue(), '', 'No physical tag is auto-confirmed');
            assert.equal(await card.getByRole('button', { name: es ? 'Registrar' : 'Record', exact: true }).isDisabled(), true);
            assert.doesNotMatch(await card.innerText(), /Confirmed by FL|Confirmado por FL/);
            evidence.supplemental_screenshots.push(await take(`04-make-instruction-${lang}`));
            await card.locator('.ingredient-confirmation').last().evaluate(element => element.scrollIntoView({ block: 'center' }));
            evidence.supplemental_screenshots.push(await take(`04-make-final-ingredient-${lang}`));
          });
          await check(5, lang, 'out-of-scope', async () => {
            const card = await send(es ? '¿Cuánto es siete más cinco?' : 'Tell me a joke', [['unsupported_request', {}]]);
            assert.match(await card.innerText(), es ? /Solo puedo registrar o consultar elementos de Factory Ledger/ : /I can only record or look up Factory Ledger items/);
            assert.doesNotMatch(await card.innerText(), /ChatGPT|try again|int[eé]ntalo de nuevo/i);
          });
          await check(6, lang, 'unknown-lot', async () => {
            const card = await send(es ? '¿Cuánto hay en el lote NO-SUCH-LOT?' : 'How much is in lot NO-SUCH-LOT?', [['lot_lookup', { lot_code: 'NO-SUCH-LOT' }]]);
            assert.match(await card.innerText(), es ? /No hay (?:un )?lote (?:llamado )?NO-SUCH-LOT|No existe (?:un )?lote (?:llamado )?NO-SUCH-LOT/ : /No lot called NO-SUCH-LOT/);
            assert.doesNotMatch(await card.innerText(), /try again|int[eé]ntalo de nuevo|reintentar/i);
          });
        }
        await check(7, 'es', 'automatic-spanish', async () => {
          // The user never clicks ES anywhere in this runner. Fresh Spanish text
          // must choose Spanish content even when the page started in English.
          const card = await send('¿Qué pedidos abiertos hay para Sunshine?', [['open_orders', { customer: 'Sunshine', product: null }]]);
          const text = await card.innerText();
          assert.match(text, /1 pedido abierto/); assert.doesNotMatch(text, /1 pedidos abiertos|There (?:are|is)|open order/);
          assert.ok(text.includes(fixture.orders[0].order_number));
        });
        await check(8, 'en', 'different-lot-trace', async () => {
          let card = await send('Made 1 batch of product 90002', [resolve('product', '90002', { action: 'make' }), ['prepare_make', { product_id: 285, batches: 1 }]]);
          if (await card.locator('.choice-button').count()) {
            await queue([['prepare_make', { product_id: 285, batches: 1 }]]);
            await card.locator('.choice-button').first().click();
            await page.waitForFunction(() => !document.getElementById('send').disabled);
            card = page.locator('.assistant-message .card').last();
          }
          const groups = card.locator('.ingredient-confirmation');
          assert.equal(await groups.count(), 2);
          await groups.first().getByRole('button', { name: 'Different lot', exact: true }).click();
          assert.equal(await groups.first().locator('input').inputValue(), '');
          assert.equal(await card.getByRole('button', { name: 'Record', exact: true }).isDisabled(), true);
          await groups.first().locator('input').fill('PHYSICAL-TAG-7654');
          assert.equal(await card.getByRole('button', { name: 'Record', exact: true }).isDisabled(), true, 'Every other ingredient still needs a physical tag');
          await groups.nth(1).locator('input').fill('EFGH');
          for (const warning of await card.locator('.warning input[type="checkbox"]').all()) await warning.check();
          await card.getByRole('button', { name: 'Record', exact: true }).click();
          await page.waitForFunction(() => !document.getElementById('send').disabled);
          assert.match(await card.innerText(), /RECORDED IN FACTORY LEDGER/);
          const sent = records.at(-1);
          assert.deepEqual(sent.lot_confirmations.map(row => ({ lot_id: row.lot_id, method: row.method, value: row.value })), [
            { lot_id: fixture.lots['283'].id, method: 'different_lot', value: 'PHYSICAL-TAG-7654' },
            { lot_id: fixture.lots['284'].id, method: 'last4', value: 'EFGH' },
          ]);
          const proof = await (await fetch(origin + '/test/verify')).json();
          assert.equal(proof.sunflower_original_stock, 100, 'Different lot never substitutes the suggested FIFO stock');
          assert.ok(proof.exceptions.some(row => row.detail.physical_tag === 'PHYSICAL-TAG-7654'), 'The physical tag remains on a follow-up exception');
        }, true);
        for (const lang of ['en', 'es']) await check(9, lang, 'correct-receive-weight', async () => {
          const es = lang === 'es';
          const first = await send(es
            ? 'Recibí 1 caja de semillas de girasol, 10 lb total, de Farm Direct, lote del proveedor SUP-CORR, BOL BOL-CORR-ES'
            : 'Received 1 case of Sunflower Seeds, 10 lb total, from Farm Direct, supplier lot SUP-CORR, BOL BOL-CORR-EN',
          [resolve('product', es ? 'semillas de girasol' : 'Sunflower Seeds'), resolve('supplier', 'Farm Direct'), ['prepare_receive', { product_id: 283, supplier_id: 4, cases: 1, case_size_lb: 10, bol_reference: 'BOL-CORR-' + lang, supplier_lot_code: 'SUP-CORR' }]]);
          assert.match(await first.innerText(), /10 lb/);
          const originalId = await first.getAttribute('data-draft-id');
          assert.ok(originalId, 'Initial receive creates a real draft');
          const original = page.locator(`[data-draft-id="${originalId}"]`);
          const corrected = await send(es ? 'eran 15 lb total' : 'it was 15 lb total', []);
          assert.match(await corrected.innerText(), /15 lb/);
          assert.match(await original.innerText(), es ? /Reemplazado por el borrador corregido/ : /Replaced by the corrected draft/);
          assert.equal(await original.getByRole('button').count(), 0, 'The wrong-weight draft has no stale controls');
          assert.equal(turns.at(-1).session_id, turns.at(-2).session_id, 'Correction stays in the same chat');
          await corrected.getByText('15 lb', { exact: true }).evaluate(element => element.scrollIntoView({ block: 'center' }));
          evidence.supplemental_screenshots.push(await take(`09-corrected-receive-quantity-${lang}`));
        });
        if (width === 1440 && mode === 0) for (const lang of ['en', 'es']) {
          const es = lang === 'es';
          await check(10, lang, 'kookies-order-filter', async () => {
            const card = await send(es ? '¿Cuáles pedidos abiertos tienen Kookies & Kreme?' : 'Which open orders have Kookies & Kreme?', [['open_orders', { customer: null, product: 'Kookies & Kreme' }]]);
            const text = await card.innerText();
            assert.ok(text.includes(fixture.orders[1].order_number));
            assert.match(text, /Setton Farms/); assert.match(text, /Kookies & Kreme/);
            assert.doesNotMatch(text, /Sunshine Granola|UNIPRO/);
            assert.match(text, es ? /1 pedido abierto/ : /1 open order/);
            assert.doesNotMatch(text, /1 open orders|1 pedidos abiertos/);
          });
          await check(11, lang, 'exact-classic-inventory', async () => {
            const card = await send(es ? '¿Cuánto hay de Batch Classic Granola #9?' : 'How much Batch Classic Granola #9?', [['inventory_lookup', { q: 'Batch Classic Granola #9' }]]);
            const text = await card.innerText();
            const made = evidence.results.filter(row => row.item === 8 && row.sessions_enabled === Boolean(mode) && row.status === 'passed').length;
            assert.match(text, /Batch Classic Granola #9/); assert.ok(text.includes(`${100 + 10 * made} lb`), 'Exact product stock reflects only explicitly recorded fixture makes');
            assert.doesNotMatch(text, /Chocolate Chip|Fruit Nut|Kookies/);
            assert.equal(await page.locator('.assistant-message .card').count(), 1);
            assert.equal(await card.locator('.read-summary').count(), 1, 'Exact inventory has one product section');
          });
        }
        assert.deepEqual(errors, [], 'No browser JavaScript errors');
        assert.ok(assetResponses.length >= 7 && assetResponses.every(row => row.status === 200), 'All real dashboard assets loaded');
        evidence.modes.push({ sessions_enabled: Boolean(mode), width, javascript_errors: errors, assets: assetResponses, turns: turns.length, records: records.length });
        await context.close();
      }
      const after = await (await fetch(origin + '/test/verify')).json();
      const recorded = evidence.results.filter(row => row.sessions_enabled === Boolean(mode) && row.item === 8 && row.status === 'passed').length;
      assert.equal(after.transactions, before.transactions + recorded, 'Only explicit completed trace scenarios write ledger stock');
      assert.equal(after.committed, recorded, 'No lot is skipped or auto-recorded');
      assert.equal(after.supplier_count, 66, 'Display deduplication does not mutate suppliers');
      evidence.modes.push({ sessions_enabled: Boolean(mode), database: { initial_transactions: before.transactions, final_transactions: after.transactions, committed_drafts: after.committed, suppliers: after.supplier_count }, model_requests: after.model_requests });
    } finally {
      if (child && child.exitCode === null) { child.kill('SIGTERM'); await new Promise(resolve => child.once('exit', resolve)); }
      run('dropdb', ['-h', '127.0.0.1', '--if-exists', '--force', db]);
    }
  }
} finally {
  await browser.close();
  evidence.databases_dropped = true;
  evidence.passed = evidence.results.filter(row => row.status === 'passed').length;
  evidence.failed = failed;
  await fs.writeFile(path.join(evidenceRoot, 'results.json'), JSON.stringify(evidence, null, 2) + '\n');
}
assert.equal(failed, 0, `${failed} actual-page acceptance checks failed; see ${evidenceRoot}/results.json`);
console.log(`Round B actual page: ${evidence.results.length} checks passed; disposable databases dropped.`);

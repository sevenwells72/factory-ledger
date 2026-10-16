// Actual dashboard -> HTTP -> branch handlers -> disposable local PostgreSQL.
// Default stubs only OpenAI tool selection; no page/API/display responses are mocked.
// Browser checks NEVER Record. Separate API acceptance tests record with physical tags.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const { chromium, webkit } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const python = process.env.FIXC_PYTHON || path.join(root, '.venv-test/bin/python');
const pg = process.env.FIXC_PG_BIN || '/opt/homebrew/opt/postgresql@17/bin';
const realModel = process.env.FIXC_REAL_MODEL === '1';
const widths = (process.env.FIXC_WIDTHS || '390').split(',').map(Number);
const modes = process.env.PILOT_SESSIONS === undefined ? [0, 1] : [Number(process.env.PILOT_SESSIONS)];
const engines = (process.env.FIXC_ENGINES || 'chromium').split(',');
const evidenceRoot = process.env.FIXC_OUT || path.join(root, 'docs/validation/fl-assistant-round-c/transcripts');
await fs.mkdir(evidenceRoot, { recursive: true });
const localEnv = { ...process.env };
for (const name of ['PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE', 'PGOPTIONS']) delete localEnv[name];
const evidence = { runner: 'tests/visual/run-fl-assistant-round-c.mjs', actual_page: true, actual_api: true, local_only: true,
  model_tool_selection_stubbed: !realModel, actual_microphone_or_iphone: false, browser_record_calls: 0,
  coverage_limit: 'Text transcripts use real page/API/DB. Default model tool selection is queued; speech recognition and physical iPhone Safari are not exercised.',
  results: [], modes: [], created_at: new Date().toISOString() };
let failed = 0;
const resolve = (kind, query, action) => ['resolve', { kind, query, context: action ? { action } : {} }];
for (const engine of engines) {
  const browser = await ({ chromium, webkit }[engine]).launch({ headless: true, ...(engine === 'chromium' && process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
  try {
    for (const mode of modes) {
      const db = `${process.env.TEST_DB_PREFIX || 'fl_fixC_'}page_${process.pid}_${mode}_${engine}`;
      const run = (command, args) => execFileSync(path.join(pg, command), args, { cwd: root, env: localEnv, stdio: 'pipe' });
      let child;
      const existing = run('psql', ['-h', '127.0.0.1', '-d', 'postgres', '-tAc', `SELECT datname FROM pg_database WHERE datname='${db}'`]).toString().trim();
      assert.equal(existing, '', 'Refuse to reuse an existing database');
      run('createdb', ['-h', '127.0.0.1', db]);
      try {
        run('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-c', 'CREATE EXTENSION pg_trgm']);
        run('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-f', path.join(root, 'tests/schema/schema.sql')]);
        child = spawn(python, ['-u', path.join(root, 'tests/visual/assistant_round_c_page_bridge.py')], {
          cwd: root, env: { ...localEnv, TEST_DATABASE_URL: `postgresql://127.0.0.1:5432/${db}`, KEY_SESSIONS_ENABLED: String(mode) }, stdio: ['ignore', 'pipe', 'pipe'],
        });
        let diagnostic = '';
        child.stderr.on('data', chunk => { diagnostic += chunk; process.stderr.write(chunk); });
        const lines = readline.createInterface({ input: child.stdout });
        const ready = await Promise.race([
          new Promise((resolve, reject) => { lines.once('line', line => { try { resolve(JSON.parse(line)); } catch { reject(new Error('Invalid bridge startup output')); } }); child.once('exit', code => reject(new Error(`Bridge exited ${code}: ${diagnostic}`))); }),
          new Promise((_, reject) => { const timer = setTimeout(() => reject(new Error('Bridge startup timed out')), 30000); timer.unref(); }),
        ]);
        const origin = `http://127.0.0.1:${ready.port}`;
        const fixture = await (await fetch(origin + '/test/fixture')).json(), p = fixture.products;
        const before = await (await fetch(origin + '/test/verify')).json();
        for (const width of widths) {
          const context = await browser.newContext({ viewport: { width, height: 844 }, timezoneId: 'America/New_York', colorScheme: 'light', isMobile: true, hasTouch: true });
          await context.addInitScript(() => localStorage.setItem('fl-assistant-language', 'en'));
          const page = await context.newPage(), errors = [], records = [];
          page.setDefaultTimeout(realModel ? 120000 : 10000);
          page.on('pageerror', error => errors.push(error.message));
          page.on('request', request => { if (new URL(request.url()).pathname === '/assistant/record') records.push(request.postDataJSON()); });
          const login = async () => {
            await page.locator('#sign-in-panel').waitFor({ state: 'visible' });
            await page.locator('#actor-key').fill('fixc-page-worker');
            await page.locator('#sign-in-form button').click();
            await page.locator('#chat-panel').waitFor({ state: 'visible' });
            await page.waitForFunction(() => !document.getElementById('send').disabled);
          };
          await page.goto(origin + '/dash/fl-assistant'); await login();
          const queue = async calls => { const response = await fetch(origin + '/test/model', { method: 'POST', body: JSON.stringify(calls), headers: { 'Content-Type': 'application/json' } }); assert.equal(response.status, 200, 'Model queue must start empty'); };
          const send = async (text, calls) => {
            await queue(calls); const count = await page.locator('.assistant-message').count();
            await page.locator('#message').fill(text); await page.locator('#send').click();
            await page.waitForFunction(() => !document.getElementById('send').disabled);
            assert.ok(await page.locator('.assistant-message').count() > count, 'Assistant must answer');
            return page.locator('.assistant-message .card').last();
          };
          const fresh = async () => {
            await page.evaluate(() => { for (const key of Object.keys(localStorage)) if (key.startsWith('fl-assistant-chat:')) localStorage.removeItem(key); });
            await page.reload({ waitUntil: 'domcontentloaded' });
            if (!mode || !(await page.evaluate(() => !!sessionStorage.getItem('fl-key-session-token')))) await login();
            await page.locator('#chat-panel').waitFor({ state: 'visible' });
            await page.waitForFunction(() => !document.getElementById('send').disabled);
          };
          const check = async (label, lang, callback) => {
            await fresh(); const row = { label, language: lang, width, engine, sessions_enabled: Boolean(mode) };
            try {
              await callback();
              assert.equal(records.length, 0, 'Transcript browser runner stops before Record');
              assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth || document.body.scrollWidth > innerWidth), false, 'No sideways page scroll');
              assert.doesNotMatch(await page.locator('#messages').innerText(), /Couldn't look that up|Please clarify:|Internal server error|FL returned an unreadable response|order_id, line ids|Happened at Today 12:00 AM/);
              row.status = 'passed';
            } catch (error) { row.status = 'failed'; row.error = error.stack || String(error); failed++; }
            const out = path.join(evidenceRoot, `sessions-${mode}-${engine}`); await fs.mkdir(out, { recursive: true });
            const screenshot = path.join(out, `${width}-${label}-${lang}${row.status === 'failed' ? '-FAIL' : ''}.png`);
            await page.screenshot({ path: screenshot, fullPage: false }); row.screenshot = path.relative(root, screenshot);
            row.visible_text = await page.locator('#messages').innerText(); evidence.results.push(row);
            console.log(`${engine} sessions=${mode} ${width}px ${lang} ${label}: ${row.status}`);
          };
          const traceDraft = async card => {
            assert.ok(await card.locator('.lot-confirmation input').count() > 0);
            for (const input of await card.locator('.lot-confirmation input').all()) assert.equal(await input.inputValue(), '');
            assert.equal(await card.getByRole('button', { name: /^(Record|Registrar)$/ }).isDisabled(), true);
          };
          for (const lang of ['en', 'es']) {
            const es = lang === 'es';
            await check('classic-inventory', lang, async () => {
              const card = await send(es ? '¿Cuánto tenemos de classic número nueve?' : 'How much classic number nine do we have?', [['inventory_lookup', { q: es ? 'classic número nueve' : 'classic number nine' }]]);
              assert.match(await card.innerText(), /Batch Classic Granola #9/); assert.match(await card.innerText(), /1,?000 lb/);
            });
            for (const [label, query, product, batches] of [['classic-make', es ? 'classic número nueve' : 'classic number nine', p.classic, 2], ['ss-make', es ? 'SS número uno' : 'SS number one', p.ss1, 1]]) await check(label, lang, async () => {
              const text = label === 'classic-make' ? (es ? 'Hicimos dos tandas de classic número nueve.' : 'Made two batches of classic number nine.') : (es ? 'Hicimos una tanda de SS número uno.' : 'Made one batch of SS number one.');
              let card = await send(text, [resolve('product', query, 'make'), ['prepare_make', { product_id: product.id, batches }]]);
              if (await card.locator('.choice-button').count()) {
                await card.locator('.choice-button').filter({ hasText: product.name }).first().click();
                await page.waitForFunction(() => !document.getElementById('send').disabled);
                card = page.locator('.assistant-message .card').last();
              }
              assert.match(await card.innerText(), new RegExp(product.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))); await traceDraft(card);
            });
            await check('receive-green-supply', lang, async () => {
              let card = await send(es ? 'Recibimos cuatro bolsas de semillas de girasol, cincuenta libras cada una, de Green Supply, POL 5521.' : 'We received four bags of sunflower seeds, 50 pounds each, from Green Supply, POL 5521.', [resolve('product', es ? 'semillas de girasol' : 'sunflower seeds', 'receive'), resolve('supplier', 'Green Supply', 'receive')]);
              const names = await card.locator('.choice-button').allTextContents(); assert.ok(names.some(name => name.includes('Grain Supply')));
              assert.doesNotMatch(await card.locator('p').first().innerText(), /Grain Supply/, 'Names appear only on choice buttons');
              await queue([['prepare_receive', { product_id: p.seeds.id, supplier_id: fixture.suppliers[0].id, cases: 4, case_size_lb: 50, bol_reference: '5521', supplier_lot_code: 'SUP-C-5521' }]]);
              await card.getByRole('button', { name: 'Grain Supply', exact: true }).click(); await page.waitForFunction(() => !document.getElementById('send').disabled);
              card = page.locator('.assistant-message .card').last(); assert.match(await card.innerText(), /200 lb/); assert.ok(await card.getByRole('button', { name: /^(Record|Registrar)$/ }).count());
            });
            await check('pack-spoken-lot', lang, async () => {
              const card = await send(es ? 'Empacamos cuarenta cajas de fruit and nut granola del lote uno dos tres cuatro.' : 'Packed 40 cases of fruit and nut granola from batch lot one two three four.', [resolve('product', 'fruit and nut granola', 'pack'), ['packing_context', { target_product_id: p.fruit.id, lot_code: es ? 'uno dos tres cuatro' : 'one two three four' }], ['prepare_pack', { source_product_id: p.fruit_batch.id, target_product_id: p.fruit.id, cases: 40 }]]);
              assert.match(await card.innerText(), /200 lb/); assert.match(await card.innerText(), /1234/); await traceDraft(card);
            });
            await check('customer-only-order', lang, async () => {
              let card = await send(es ? 'Nuevo pedido de fake bakery. Cinco cajas de rainbow sprinkles.' : 'New order from fake bakery. Five cases of rainbow sprinkles.', [resolve('customer', 'fake bakery', 'order')]);
              const choices = await card.locator('.choice-button').allTextContents(); assert.ok(choices.length > 0); assert.ok(choices.some(name => name.includes('Lake Bakery'))); assert.ok(choices.every(name => !/A-1 Bakery Supply|Bakers Depot/.test(name)));
              await queue([resolve('product', 'rainbow sprinkles', 'order'), ['prepare_create_order', { customer_id: fixture.customers[0].id, lines: [{ product_id: p.sprinkles.id, quantity: 5, unit: 'cases' }] }]]);
              await card.getByRole('button', { name: 'Lake Bakery', exact: true }).click(); await page.waitForFunction(() => !document.getElementById('send').disabled);
              card = page.locator('.assistant-message .card').last(); assert.match(await card.innerText(), /Rainbow Sprinkles/); assert.ok(await card.getByRole('button', { name: /^(Record|Registrar)$/ }).count());
            });
            await check('today-exact-transcript', lang, async () => {
              const card = await send(es ? 'Abre las entradas de hoy. ¿Está todo correcto?' : "Open today's entries. Is everything there correct?", []);
              assert.match(await card.innerText(), es ? /hoy|equipo/i : /today|team/i); assert.doesNotMatch(await card.innerText(), /try again|int[eé]ntalo/i);
            });
            await check('today-short-query', lang, async () => {
              const card = await send(es ? '¿Qué registré hoy?' : 'What did I record today?', []);
              assert.match(await card.innerText(), es ? /hoy|equipo/i : /today|team/i);
            });
            await check('spoken-letter-code', lang, async () => {
              const card = await send(es ? '¿Cuánto hay en el lote P quince ochenta y cinco?' : 'How much is in lot P fifteen eighty-five?', [['lot_lookup', { lot_code: es ? 'P quince ochenta y cinco' : 'P fifteen eighty-five' }]]);
              assert.match(await card.innerText(), /P1585/); assert.match(await card.innerText(), /1,?000 lb/);
            });
          }
          assert.deepEqual(errors, [], 'No JavaScript errors'); evidence.browser_record_calls += records.length;
          evidence.modes.push({ sessions_enabled: Boolean(mode), engine, width, javascript_errors: errors });
          await context.close();
        }
        const after = await (await fetch(origin + '/test/verify')).json();
        assert.equal(after.transactions, before.transactions, 'Browser stops before all ledger writes');
        assert.equal(after.committed, 0); assert.equal(after.orders, 0);
        evidence.modes.push({ sessions_enabled: Boolean(mode), engine, database: { initial_transactions: before.transactions, final_transactions: after.transactions, committed: after.committed }, model_requests: after.model_requests });
      } finally {
        if (child && child.exitCode === null) { child.kill('SIGTERM'); await new Promise(resolve => child.once('exit', resolve)); }
        run('dropdb', ['-h', '127.0.0.1', '--if-exists', '--force', db]);
        evidence.databases_dropped = true;
        evidence.passed = evidence.results.filter(row => row.status === 'passed').length; evidence.failed = failed;
        await fs.writeFile(path.join(evidenceRoot, 'results.json'), JSON.stringify(evidence, null, 2) + '\n');
      }
    }
  } finally { await browser.close(); }
}
assert.equal(failed, 0, `${failed} real-page transcript checks failed; inspect ${evidenceRoot}/results.json`);
console.log(`Round C transcripts: ${evidence.results.length} checks passed; zero Record calls; disposable databases dropped.`);

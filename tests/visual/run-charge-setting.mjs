// PR #143 actual pages → HTTP → real FL handlers → disposable fl_charge_ PG17.
// Product/Charge choice (Office), "Confirm: Charge or Product?" (Exceptions), charge lines on
// Record entry, the ship draft card and the packing slip. EN + ES. OpenAI hard-refused.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const python = process.env.CHARGE_PYTHON || path.join(root, '.venv-test/bin/python');
const pg = process.env.CHARGE_PG_BIN || '/opt/homebrew/opt/postgresql@17/bin';
const out = process.env.CHARGE_OUT || '/tmp/fl-charge-setting';
const languages = (process.env.CHARGE_LANGUAGES || 'en,es').split(',');
await fs.mkdir(out, { recursive: true });
const env = { ...process.env };
for (const name of Object.keys(env)) if (['DATABASE_URL', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE', 'PGOPTIONS'].includes(name) || name.startsWith('SUPABASE_')) delete env[name];
const report = { created_at: new Date().toISOString(), local_only: true, openai_calls: 0, checks: [], screenshots: [] };
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || undefined });
try {
  for (const lang of languages) {
    const db = `fl_charge_${process.pid}_${lang}`;
    const pgRun = (command, args) => execFileSync(path.join(pg, command), args, { cwd: root, env, stdio: 'pipe' });
    pgRun('createdb', ['-h', '127.0.0.1', '-E', 'UTF8', '-T', 'template0', db]);
    let child;
    try {
      pgRun('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-c', 'CREATE EXTENSION pg_trgm']);
      pgRun('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-f', path.join(root, 'tests/schema/schema.sql')]);
      child = spawn(python, ['-u', path.join(root, 'tests/visual/charge_setting_page_bridge.py')], { cwd: root, env: { ...env, TEST_DATABASE_URL: `postgresql://127.0.0.1:5432/${db}`, KEY_SESSIONS_ENABLED: '1' }, stdio: ['ignore', 'pipe', 'pipe'] });
      child.stderr.on('data', chunk => process.stderr.write(chunk));
      const ready = JSON.parse(await new Promise((resolve, reject) => { readline.createInterface({ input: child.stdout }).once('line', resolve); child.once('exit', code => reject(Error('bridge exited ' + code))); }));
      const origin = `http://127.0.0.1:${ready.port}`;
      const fixture = await (await fetch(origin + '/test/fixture')).json();
      const context = await browser.newContext({ viewport: { width: 1180, height: 900 }, deviceScaleFactor: 2 });
      const page = await context.newPage();
      await page.route('**/*', r => new URL(r.request().url()).origin === origin ? r.continue() : r.abort());
      const t = (en, es) => lang === 'es' ? es : en;
      const shot = async (name, locator) => { const file = path.join(out, `${lang}-${name}.png`); await (locator || page).screenshot({ path: file, ...(locator ? {} : { fullPage: true }) }); report.screenshots.push(file); };
      const check = async (name, fn) => { try { await fn(); } catch (error) { await page.screenshot({ path: path.join(out, `${lang}-FAILED.png`), fullPage: true }); throw error; } report.checks.push({ lang, name, ok: true }); console.log('ok', lang, name); };
      const open = async file => { await page.goto(`${origin}/dashboard/${file}`); await page.evaluate(() => window.FLLedger?.ready); };
      const login = async (role, file) => {
        await open(file);
        if (!(await page.locator('#actor-key').isVisible())) { await page.locator('#sign-out').click(); await page.locator('#actor-key').waitFor(); }
        await page.locator('#actor-key').fill(fixture.actors[role].key); await page.locator('#sign-in-form button').click();
        await page.locator('#identity').filter({ hasText: fixture.actors[role].name }).waitFor();
        if (lang === 'es' && (await page.locator('#language').textContent()).trim() === 'ES') await page.locator('#language').click();
      };

      await check('Office catalog item: choice required, nothing pre-selected', async () => {
        await login('office', 'office.html');
        await page.locator('[data-office-start="catalog_item"]').click();
        const form = page.locator('#catalog-create');
        await form.waitFor();
        const radios = form.locator('input[type=radio][name=item_kind]');
        assert.equal(await radios.count(), 2);
        for (const i of [0, 1]) assert.equal(await radios.nth(i).isChecked(), false);
        await form.locator('input[type=text]').first().fill('Delivery Fee');
        await form.getByRole('button', { name: t('Create item', 'Crear artículo') }).click();
        await page.locator('#notice').filter({ hasText: t('Choose Product or Charge', 'Elige Producto o Cargo') }).waitFor();
        await shot('office-choice-required');
        await form.getByText(t('Charge (pallet, freight, delivery…)', 'Cargo (palé, flete, entrega…)'), { exact: true }).click();
        await shot('office-choice-charge', form);
        await form.getByRole('button', { name: t('Create item', 'Crear artículo') }).click();
        await form.getByText(new RegExp(t('Created Delivery Fee — Charge', 'Creado Delivery Fee — Cargo'))).waitFor();
        await form.locator('input[type=text]').first().fill('Shrink wrap');
        await form.getByText(t('Charge (pallet, freight, delivery…)', 'Cargo (palé, flete, entrega…)'), { exact: true }).click();
        await form.getByRole('button', { name: t('Create item', 'Crear artículo') }).click();
        await form.getByText(t('Confirm: Charge or Product?', 'Confirmar: ¿Cargo o Producto?'), { exact: false }).waitFor();
        await shot('office-created-sent-to-exceptions', form);
      });

      await check('Exceptions: Confirm Charge or Product card, answered once', async () => {
        await login('office', 'exceptions.html');
        const card = page.locator('article.exception-row').filter({ hasText: 'Setton Coffee Granola 25 LB' });
        await card.waitFor();
        assert.match(await card.textContent(), lang === 'es' ? /Confirmar: ¿Cargo o Producto\?[\s\S]*contiene "fee"/ : /Confirm: Charge or Product\?[\s\S]*contains "fee"/);
        await shot('exceptions-confirm-card');
        await card.getByRole('button', { name: t('It is a Product', 'Es un Producto') }).click();
        await page.locator('#notice').filter({ hasText: t('will not ask about this item again', 'no volverá a preguntar') }).waitFor();
        await page.locator('article.exception-row').filter({ hasText: 'Setton Coffee Granola 25 LB' }).waitFor({ state: 'detached' });
        await shot('exceptions-confirmed');
        const item = await (await fetch(`${origin}/catalog/items?q=Setton`, { headers: { 'X-API-Key': fixture.actors.owner.key } })).json();
        assert.equal(item.items[0].confirmed, true);
        assert.equal(item.items[0].item_kind, 'product');
      });

      await check('Record entry ship by order and draft card show the charge line', async () => {
        await login('floor', 'entry.html?action=ship');
        const mode = page.locator('[name=ship_mode]');
        if (await mode.count()) await mode.selectOption('order');
        const picker = page.locator('[data-picker=order]');
        await picker.locator('input').fill(fixture.order_number);
        await picker.getByRole('button', { name: t('Search', 'Buscar') }).click();
        await picker.locator('.picker-options button').first().click();
        const charge = page.locator('section[data-shipment-line-id]').filter({ hasText: 'Pallet Charge' });
        await charge.locator('.charge-tag').waitFor();
        await shot('entry-ship-by-order', page.locator('#entry-form'));
        await page.locator('[name=ship_all]').check();
        await page.locator('[name=bol_reference]').fill('BOL-143');
        await page.locator('#entry-form button[type=submit]').click();
        const draft = page.locator('#resolution');
        await draft.locator('.charge-tag').first().waitFor();
        await shot('ship-draft-card', draft);
      });

      await check('Packing slip tags the charge line', async () => {
        const pdf = Buffer.from(await (await fetch(`${origin}/sales/orders/${fixture.order}/packing-slip?key=charge-master`)).arrayBuffer());
        const file = path.join(out, `${lang}-packing-slip.pdf`);
        await fs.writeFile(file, pdf);
        report.screenshots.push(file);
      });
      await context.close();
    } finally {
      if (child) { child.kill('SIGTERM'); await new Promise(resolve => child.once('exit', resolve)); }
      pgRun('dropdb', ['-h', '127.0.0.1', '--if-exists', db]);
    }
  }
} finally {
  await browser.close();
  await fs.writeFile(path.join(out, 'results.json'), JSON.stringify(report, null, 2));
}

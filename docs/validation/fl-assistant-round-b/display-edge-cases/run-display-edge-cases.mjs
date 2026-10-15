// Fixture-renderer evidence only: actual dashboard assets, read cards generated
// by assistant_replies.read_card. No database, hosted API, or model is used.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const out = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(out, '../../../..');
const { startStaticServer } = await import(path.join(root, 'tests/visual/lib/server.mjs'));
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const python = process.env.FIXB_PYTHON || path.join(root, '.venv-test/bin/python');
const source = String.raw`
import json,assistant_replies
inventory={'query':'Sunflower Seeds','results':[{'product':'Sunflower Seeds','total_on_hand':100,
 'lots':[{'lot':'SUNFLOWER-VERIFIED','qty_on_hand':100}],
 'unverified_lots':[{'lot':'UNVERIFIED-PHYSICAL-TAG-7654','quantity_lb':-4,'unverified':True}]}]}
receipt={'receipt_number':'ADJ-261015-240','action':'adjust','status':'committed','effective_status':'voided',
 'actor':{'name':'Round B Worker'},'happened_at':'2026-10-15T12:00:00-04:00',
 'draft':{'product_name':'Sunflower Seeds','delta_lb':5},
 'transactions':[{'type':'adjust','effective_status':'voided','lines':[{'product_name':'Sunflower Seeds','lot_code':'SUNFLOWER-VERIFIED','quantity_lb':5}]}]}
print(json.dumps({lang:{'voided':assistant_replies.read_card('receipt_lookup',receipt,lang),
 'unverified':assistant_replies.read_card('inventory_lookup',inventory,lang)} for lang in ['en','es']}))
`;
const fixtures = JSON.parse(execFileSync(python, ['-c', source], { cwd: root, encoding: 'utf8' }));
const server = await startStaticServer(root);
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
const sid = '0c101ab1-8c56-42a2-8f29-c048330030c3';
const actor = { id: 4, name: 'Round B Worker', role: 'floor' };
const results = [];
try {
  for (const width of [1440, 390]) for (const language of ['en', 'es']) for (const kind of ['voided', 'unverified']) {
    const context = await browser.newContext({ viewport: { width, height: width === 390 ? 844 : 1000 }, timezoneId: 'America/New_York' });
    await context.addInitScript(() => localStorage.setItem('fl-assistant-language', 'en'));
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/dash/fl-assistant') return route.fulfill({ contentType: 'text/html', body: await fs.readFile(path.join(root, 'dashboard/fl-assistant.html'), 'utf8') });
      if (url.pathname.startsWith('/dashboard/')) return route.continue();
      if (url.pathname === '/auth/session/config') return route.fulfill({ json: { key_sessions_enabled: false } });
      if (url.pathname === '/auth/whoami') return route.fulfill({ json: { actor, key_kind: 'actor' } });
      if (url.pathname === '/auth/dashboard-key') return route.fulfill({ json: { dashboard_key: 'fixture-read-key' } });
      if (url.pathname === '/assistant/session') return route.fulfill({ json: { session_id: sid, actor } });
      if (url.pathname === '/assistant/resume') return route.fulfill({ json: { session_id: sid, turns: [], drafts: [], actor } });
      if (url.pathname === '/assistant/turn') return route.fulfill({ json: { cards: [fixtures[language][kind]] } });
      if (url.pathname === '/favicon.ico') return route.fulfill({ status: 204 });
      throw new Error('Unexpected fixture route: ' + url.pathname);
    });
    await page.goto(server.origin + '/dash/fl-assistant');
    await page.locator('#sign-in-panel').waitFor({ state: 'visible' });
    await page.locator('#actor-key').fill('fixture-personal-key');
    await page.locator('#sign-in-form button').click();
    await page.locator('#chat-panel').waitFor({ state: 'visible' });
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    const text = kind === 'voided' ? (language === 'es' ? 'Consulta el recibo ADJ-261015-240' : 'Look up receipt ADJ-261015-240') : (language === 'es' ? '¿Cuántas semillas de girasol hay?' : 'How much Sunflower Seeds do we have?');
    await page.locator('#message').fill(text); await page.locator('#send').click();
    await page.waitForFunction(() => !document.getElementById('send').disabled);
    const card = page.locator('.assistant-message .card').last(), visible = await card.innerText();
    if (kind === 'voided') {
      assert.match(visible, /ADJ-261015-240/); assert.match(visible, language === 'es' ? /ANULADO/ : /VOIDED/);
    } else {
      assert.match(visible, /Sunflower Seeds: 100 lb/);
      assert.match(visible, language === 'es' ? /sin verificar, pendiente de seguimiento/ : /unverified, awaiting follow-up/);
      assert.equal(await card.locator('tbody tr').count(), 1); assert.match(await card.locator('tbody').innerText(), /SUNFLOWER-VERIFIED/);
      assert.doesNotMatch(visible, /-4|96 lb|UNVERIFIED-PHYSICAL/);
    }
    assert.doesNotMatch(visible, /try again|int[eé]ntalo de nuevo/i); assert.deepEqual(errors, []);
    await card.evaluate(element => element.scrollIntoView({ block: 'center' }));
    const screenshot = `${width}-${kind}-${language}.png`;
    await page.screenshot({ path: path.join(out, screenshot), fullPage: false });
    results.push({ kind, language, width, status: 'passed', screenshot, visible_text: visible });
    await context.close();
  }
} finally { await browser.close(); await server.close(); }
await fs.writeFile(path.join(out, 'results.json'), JSON.stringify({ runner: 'run-display-edge-cases.mjs', fixture_renderer_only: true, cards_from: 'assistant_replies.read_card', database_used: false, model_used: false, passed: results.length, failed: 0, results }, null, 2) + '\n');
console.log(`Display edge cases: ${results.length} renderer-only checks passed; no database or model used.`);

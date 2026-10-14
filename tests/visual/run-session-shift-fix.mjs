import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const server = await startStaticServer(root);
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome' });
const TOKEN = 'fls_browser-fixture', actor = { id: 7, name: 'Arturo', role: 'office' };
try {
  for (const enabled of [false, true]) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await context.addInitScript(({ enabled, token }) => {
      window.FL_SESSION_API_BASE = location.origin;
      if (enabled && !sessionStorage.getItem('fixture-seeded')) {
        sessionStorage.setItem('fl-key-session-token', token); sessionStorage.setItem('fixture-seeded', 'yes');
      }
      const original = window.setTimeout;
      window.fixtureTimeouts = [];
      window.setTimeout = (fn, ms, ...rest) => { window.fixtureTimeouts.push(ms); return original(fn, ms, ...rest); };
    }, { enabled, token: TOKEN });
    const page = await context.newPage(), calls = [], errors = [];
    let expired = false, exchanges = 0, logouts = 0;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url()), p = url.pathname;
      assert.equal(url.origin, server.origin, 'all browser traffic stays on the disposable local server');
      if (p.startsWith('/dashboard/')) return route.continue();
      const key = request.headers()['x-api-key']; calls.push({ path: p, key, method: request.method() });
      let body = {};
      if (p === '/auth/session/config') body = { key_sessions_enabled: enabled };
      else if (p === '/auth/whoami') body = { key_kind: enabled ? 'session' : 'actor', actor };
      else if (p === '/auth/session') { exchanges++; body = { key_kind: 'session', session_token: TOKEN, actor }; }
      else if (p === '/auth/session/logout') { logouts++; body = { signed_out: true }; }
      else if (['/sales/orders', '/sales/orders/counts', '/expected-receipts', '/suppliers'].includes(p)) {
        // Model the staging cutover refusal: a shared read key cannot pass.
        if (enabled && key !== TOKEN) return route.fulfill({ status: 403, json: { detail: 'Personal session required' } });
        if (p === '/sales/orders') body = { orders: [] };
        if (p === '/sales/orders/counts') body = { open: 0, overdue: 0, closed: 0, cancelled: 0 };
        if (p === '/expected-receipts') body = { expected_receipts: [{ id: 3, product_name: 'Fixture oats', supplier_name: 'Fixture Supplier', expected_qty: 20, received_qty: 0, remaining: 20, status: 'open', expected_date: '2026-10-15' }] };
        if (p === '/suppliers') body = { suppliers: [{ id: 4, name: 'Fixture Supplier' }] };
      } else if (p.startsWith('/reports/shift-summary')) {
        if (expired) return route.fulfill({ status: 401, json: { detail: { error_code: 'SESSION_EXPIRED', message: 'Sign in again.' } } });
        if (p.endsWith('/confirm')) body = { message: 'Shift confirmed.' };
        else body = { date: url.searchParams.get('date'), title: 'Shared shift for Arturo', actor, receipts_today: 0, summary_hash: 'a'.repeat(64), sections: [], confirmation: { status: 'unconfirmed', can_confirm: true } };
      }
      return route.fulfill({ json: body });
    });
    await page.goto(server.origin + '/dashboard/index.html?section=orders');
    await page.locator('#orders-table-container [class*=empty]').waitFor();
    assert.equal(await page.locator('#orders-error').isVisible(), false);
    await page.goto(server.origin + '/dashboard/index.html?section=expected');
    await page.getByText('Fixture oats', { exact: true }).waitFor();
    await page.goto(server.origin + '/dashboard/index.html?section=notes');
    await page.locator('#notes-add-btn').click();
    await page.locator('#note-entity-type').selectOption('supplier');
    await page.locator('#note-entity-id').fill('Fixture');
    await page.getByRole('button', { name: 'Fixture Supplier — supplier #4', exact: true }).waitFor();
    const readKey = calls.find(call => call.path === '/sales/orders').key;
    assert.ok(readKey); assert.equal(readKey === TOKEN, enabled);
    for (const endpoint of ['/sales/orders', '/expected-receipts', '/suppliers']) {
      const reads = calls.filter(call => call.path === endpoint);
      assert.ok(reads.length, endpoint + ' is exercised by its real screen');
      assert.ok(reads.every(call => call.key === readKey));
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(server.origin + '/dashboard/shift-summary.html');
    if (enabled) {
      await page.getByText('Shared shift for Arturo', { exact: true }).waitFor();
      assert.equal(await page.locator('#sign-in-panel').isVisible(), false);
      assert.equal(exchanges, 0, 'navigation reuses the existing sign-in without another personal key');
    } else {
      await page.locator('#sign-in-panel').waitFor();
      await page.locator('#actor-key').fill('legacy-browser-fixture');
      await page.locator('#sign-in-form button[type=submit]').click();
      await page.getByText('Shared shift for Arturo', { exact: true }).waitFor();
      assert.equal(exchanges, 0, 'flag-off personal sign-in keeps its existing behavior');
    }
    await page.locator('#confirm-match').click();
    await page.getByText('Shift confirmed.', { exact: true }).waitFor();
    const confirm = calls.find(call => call.path === '/reports/shift-summary/confirm');
    assert.equal(confirm.key, enabled ? TOKEN : 'legacy-browser-fixture');
    assert.ok(await page.evaluate(() => fixtureTimeouts.includes(60000)));
    assert.equal(await page.evaluate(() => fixtureTimeouts.includes(20000)), false);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    if (enabled) {
      expired = true; await page.locator('#refresh').click();
      await page.getByText('Sign in again.', { exact: true }).waitFor();
      assert.equal(await page.locator('#summary-panel').isVisible(), false);
      assert.equal(await page.evaluate(() => sessionStorage.getItem('fl-key-session-token')), null);
      expired = false; await page.locator('#actor-key').fill('fixture-personal');
      await page.locator('#sign-in-form button[type=submit]').click();
      await page.getByText('Shared shift for Arturo', { exact: true }).waitFor();
      assert.equal(exchanges, 1);
      await page.locator('#sign-out').click(); await page.locator('#sign-in-panel').waitFor();
      assert.equal(logouts, 1);
      assert.equal(await page.evaluate(() => sessionStorage.getItem('fl-key-session-token')), null);
    }
    assert.deepEqual(errors, []); await context.close();
  }
  console.log('Session/Shift browser: flag off/on, real Sales Orders/Expected Receipts/supplier reads, shared sign-in reuse, session confirmation, 60-second timer, expiry, re-sign-in, logout and 390px layout passed.');
} finally { await browser.close(); await server.close(); }

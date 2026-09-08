// Every request the dashboard makes to the Railway API is answered from a
// fixture file. Nothing in this harness touches the network or the database.
//
// Fixtures carry date tokens rather than hard-coded dates, so "today", the
// rolling 5-day calendar, and the overdue flags are correct on any run day
// while the payload itself stays byte-stable within a run — which is what
// LAYOUT-020's before/after comparison depends on.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const FIXTURE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'fixtures');
export const API_HOST = 'fastapi-production-b73a.up.railway.app';

const WEEKDAY = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

// The dashboard's plant timezone. Anchoring the tokens here keeps the fixture
// dates and the app's own `plantToday()` in agreement.
const PLANT_TZ = 'America/New_York';

function plantTodayParts(now) {
  const iso = now.toLocaleDateString('en-CA', { timeZone: PLANT_TZ });
  return new Date(`${iso}T12:00:00Z`);
}

function shift(base, days) {
  const d = new Date(base.getTime());
  d.setUTCDate(d.getUTCDate() + days);
  return d;
}

const isoDay = d => d.toISOString().slice(0, 10);

export function buildTokenTable(now = new Date()) {
  const base = plantTodayParts(now);
  const table = new Map();
  table.set('{{NOW}}', now.toISOString());
  table.set('{{NOW+36H}}', new Date(now.getTime() + 36 * 3600 * 1000).toISOString());
  for (let i = -120; i <= 120; i++) {
    const d = shift(base, i);
    const key = i === 0 ? '{{TODAY}}' : `{{TODAY${i > 0 ? '+' : ''}${i}}}`;
    table.set(key, isoDay(d));
    const nameKey = i === 0 ? '{{TODAY_NAME}}' : `{{TODAY${i > 0 ? '+' : ''}${i}_NAME}}`;
    table.set(nameKey, WEEKDAY[d.getUTCDay()]);
  }
  return table;
}

function applyTokens(text, tokens) {
  return text.replace(/\{\{[A-Z0-9_+\-]+\}\}/g, m => (tokens.has(m) ? tokens.get(m) : m));
}

const cache = new Map();

export async function loadFixture(name, tokens) {
  const key = `${name}`;
  if (!cache.has(key)) {
    cache.set(key, await fs.readFile(path.join(FIXTURE_DIR, name), 'utf8'));
  }
  return JSON.parse(applyTokens(cache.get(key), tokens));
}

// Ordered rules — first match wins. `pick` may return a fixture name, or an
// object {name, select} where `select` pulls one keyed record out of the file.
const ROUTES = [
  { test: u => u.pathname === '/dashboard/api/production', fixture: 'production.json' },
  { test: u => u.pathname === '/dashboard/api/inventory/finished-goods', fixture: 'finished-goods.json' },
  { test: u => u.pathname === '/dashboard/api/inventory/batches', fixture: 'batches.json' },
  { test: u => u.pathname === '/dashboard/api/inventory/ingredients', fixture: 'ingredients.json' },
  { test: u => u.pathname === '/dashboard/api/activity/shipments', fixture: 'shipments.json' },
  { test: u => u.pathname === '/dashboard/api/activity/receipts', fixture: 'receipts.json' },
  { test: u => u.pathname === '/dashboard/api/activity/daily-entries', fixture: 'daily-entries.json' },
  { test: u => u.pathname.startsWith('/dashboard/api/lot/'), fixture: 'lot.json' },
  { test: u => /^\/dashboard\/api\/product\/\d+\/lots$/.test(u.pathname), fixture: 'product-lots.json' },
  { test: u => u.pathname === '/dashboard/api/search', fixture: 'search.json' },
  { test: u => u.pathname === '/dashboard/api/notes', fixture: 'notes.json' },
  { test: u => u.pathname === '/production/today-tile', fixture: 'today-tile.json' },
  { test: u => u.pathname === '/audit/integrity', fixture: 'audit-integrity.json' },
  { test: u => u.pathname === '/ledger/recent', fixture: 'ledger-recent.json' },
  { test: u => u.pathname === '/sales/orders/fulfillment-check', fixture: 'fulfillment-check.json' },
  { test: u => u.pathname === '/sales/orders', fixture: 'sales-orders.json' },
  {
    test: u => /^\/sales\/orders\/\d+\/allocations$/.test(u.pathname),
    fixture: 'allocations.json',
    select: u => u.pathname.split('/')[3],
  },
  {
    test: u => /^\/sales\/orders\/\d+\/ship\/preview$/.test(u.pathname),
    fixture: 'ship-preview.json',
  },
  {
    test: u => /^\/sales\/orders\/\d+$/.test(u.pathname),
    fixture: 'sales-order-detail.json',
    select: u => u.pathname.split('/')[3],
  },
  { test: u => u.pathname === '/expected-receipts', fixture: 'expected-receipts.json' },
  { test: u => u.pathname === '/suppliers', fixture: 'suppliers.json' },
  { test: u => u.pathname === '/supplies/inventory', fixture: 'supplies-inventory.json' },
  {
    test: u => /^\/supplies\/inventory\/\d+\/lots$/.test(u.pathname),
    fixture: 'supplies-lots.json',
    select: u => u.pathname.split('/')[3],
  },
  { test: u => u.pathname === '/supply-requests', fixture: 'supply-requests.json' },
  { test: u => u.pathname === '/products/search', fixture: 'products-search.json' },
  { test: u => u.pathname === '/inventory/current', fixture: 'inventory-current.json' },
  { test: u => u.pathname.startsWith('/trace/batch/'), fixture: 'trace-batch.json' },
  { test: u => u.pathname.startsWith('/trace/ingredient/'), fixture: 'trace-ingredient.json' },
  { test: u => u.pathname.startsWith('/lots/by-code/'), fixture: 'lot-by-code.json' },
  {
    test: u => u.pathname === '/transactions/history',
    pick: u => ({
      make: 'transactions-make.json',
      pack: 'transactions-pack.json',
      ship: 'transactions-ship.json',
      receive: 'transactions-receive.json',
    }[u.searchParams.get('transaction_type')] || 'transactions-all.json'),
  },
];

function matchRoute(url) {
  for (const rule of ROUTES) {
    if (rule.test(url)) return rule;
  }
  return null;
}

/**
 * Installs the API stub on a Playwright BrowserContext.
 *
 * `state` is mutable so a screen recipe can swap fixtures or force failures
 * for one capture without tearing the context down:
 *   state.overrides = { '/dashboard/api/notes': 'notes-empty.json' }
 *   state.fail      = ['/production/today-tile']   // -> 503
 *   state.status    = { '/dashboard/api/lot/': 409 }
 */
export async function installApiStub(context, tokens, state) {
  await context.route(`**://${API_HOST}/**`, async route => {
    const request = route.request();
    const url = new URL(request.url());

    const forced = (state.fail || []).find(p => url.pathname.startsWith(p));
    if (forced) {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Service Unavailable (visual-audit stub)' }),
      });
      return;
    }

    // Writes are acknowledged, never applied — this harness never mutates.
    if (request.method() !== 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, message: 'Stubbed write (visual-audit).' }),
      });
      return;
    }

    const overrideKey = Object.keys(state.overrides || {}).find(p => url.pathname.startsWith(p));
    const statusKey = Object.keys(state.status || {}).find(p => url.pathname.startsWith(p));

    let fixtureName;
    let select = null;
    if (overrideKey) {
      fixtureName = state.overrides[overrideKey];
    } else {
      const rule = matchRoute(url);
      if (!rule) {
        state.unmatched.add(url.pathname);
        await route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ error: `No fixture for ${url.pathname}` }),
        });
        return;
      }
      fixtureName = rule.fixture || rule.pick(url);
      select = rule.select ? rule.select(url) : null;
    }

    let payload = await loadFixture(fixtureName, tokens);
    if (select != null) {
      payload = payload[select] ?? payload[Object.keys(payload)[0]];
    }

    await route.fulfill({
      status: statusKey ? state.status[statusKey] : 200,
      contentType: 'application/json',
      body: JSON.stringify(payload),
    });
  });
}

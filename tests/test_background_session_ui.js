const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ROOT = path.resolve(__dirname, '..');
const IDLE = 180 * 60000, TOKEN = 'fls_background-fixture';
const actor = { id: 7, name: 'Arturo', role: 'office' };
const dashboard = fs.readFileSync(path.join(ROOT, 'dashboard/dashboard.js'), 'utf8');
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
function fn(source, name) {
  const start = source.search(new RegExp('  (?:async )?function ' + name + '\\('));
  assert.ok(start >= 0, name);
  const end = source.indexOf('\n  }', start) + 4;
  return source.slice(start, end);
}
function element(id) {
  return { id, hidden: false, disabled: false, value: '', textContent: '', innerHTML: '', dataset: {}, offsetHeight: 0,
    append() {}, setAttribute(k, v) { this[k] = v; }, focus() {}, classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {}, querySelector() { return element('retry'); }, showModal() {}, close() {} };
}
function harness({ enabled = true } = {}) {
  const calls = [], intervals = [], elements = new Map(), data = new Map([[ 'fl-key-session-token', TOKEN ]]);
  let now = 0, lastSeen = 0, expiresAt = IDLE;
  const storage = { getItem: k => data.get(k) || null, setItem: (k, v) => data.set(k, v), removeItem: k => data.delete(k) };
  const get = id => { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); };
  const context = vm.createContext({ URL, Headers, Response, AbortController, Date, Intl, console,
    setTimeout: () => 1, clearTimeout() {}, setInterval: (callback, ms) => { const timer = { callback, ms }; intervals.push(timer); return timer; }, clearInterval() {},
    location: { href: 'https://fl.local/dashboard/index.html?section=recent', origin: 'https://fl.local', hostname: 'fl.local' },
    sessionStorage: storage,
    document: { readyState: 'complete', hidden: false, body: element('body'), createElement: element, getElementById: get, querySelector: () => null },
    FLDashboard: { base: 'https://fl.local', config: Promise.resolve({ key_sessions_enabled: enabled }), ready: Promise.resolve(enabled), route: url => String(url), key: async () => 'shared-fixture' },
    SALES_API_BASE: 'https://fl.local',
    state: { currentTab: 'recent', recentEntries: { loaded: false, pollTimer: null } },
    renderRecentEntries: () => { get('recent-entries-feed').innerHTML = ''; }, escHtml: String,
    fetch: async (url, options = {}) => {
      const pathname = new URL(url).pathname, headers = new Headers(options.headers), key = headers.get('X-API-Key');
      const call = { pathname, headers, method: options.method || 'GET', expiresBefore: expiresAt, lastSeenBefore: lastSeen }; calls.push(call);
      let body = {}, status = 200;
      // This test server models FL's idle policy; assertions cover the real client's headers and custody.
      if (key === TOKEN) {
        if (now >= expiresAt) { status = 401; body = { detail: { error_code: 'SESSION_EXPIRED', message: 'Sign in again.' } }; }
        else {
          if (headers.get('X-FL-Background') !== '1') { lastSeen = now; expiresAt = now + IDLE; }
          body = pathname === '/auth/whoami' ? { key_kind: 'session', actor } : { events: [] };
        }
      } else if (pathname === '/ledger/recent') body = { events: [] };
      call.status = status; call.expiresAfter = expiresAt; call.lastSeenAfter = lastSeen;
      return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    }
  });
  context.window = context;
  for (const name of ['key-session.js', 'fetch-timeout.js']) vm.runInContext(fs.readFileSync(path.join(ROOT, 'dashboard', name), 'utf8'), context, { filename: name });
  const backgroundDeclaration = dashboard.match(/^  const BACKGROUND_READ = .*;$/m);
  assert.ok(backgroundDeclaration, 'the dashboard must declare its automatic-read options');
  vm.runInContext(backgroundDeclaration[0] + '\n' +
    ['fetchSalesAPI', 'refreshRecentEntries', 'startRecentEntriesPolling', 'stopRecentEntriesPolling', 'openRecentEntries'].map(name => fn(dashboard, name)).join('\n'), context);
  return { context, calls, intervals, data, get, setNow: value => { now = value; }, expiry: () => expiresAt, lastSeen: () => lastSeen };
}

test('restoration and every minute of polling preserve expiry; the 180-minute poll expires and clears sign-in', async () => {
  const h = harness(); await settle();
  const who = h.calls.find(c => c.pathname === '/auth/whoami');
  assert.equal(who.headers.get('X-FL-Background'), '1'); assert.equal(h.expiry(), IDLE);
  h.context.openRecentEntries({ headers: { 'X-FL-Background': '1' } }); await settle();
  assert.equal(h.intervals.length, 1); assert.equal(h.intervals[0].ms, 60000);
  for (let minute = 1; minute < 180; minute++) {
    h.setNow(minute * 60000); h.intervals[0].callback(); await settle();
    assert.equal(h.calls.at(-1).headers.get('X-FL-Background'), '1', `minute ${minute}`);
    assert.equal(h.calls.at(-1).headers.get('X-API-Key'), TOKEN); assert.equal(h.calls.at(-1).status, 200);
    assert.equal(h.expiry(), IDLE); assert.equal(h.lastSeen(), 0);
  }
  h.setNow(IDLE); h.intervals[0].callback(); await settle();
  assert.equal(h.calls.at(-1).status, 401); assert.equal(h.expiry(), IDLE);
  assert.equal(h.data.has('fl-key-session-token'), false);
  assert.equal(h.get('recent-entries-status').textContent, 'Recent entries could not be loaded.');
});

test('manual Refresh and a user write slide the personal idle expiry, unlike the following poll', async () => {
  const h = harness(); await settle(); h.setNow(60000);
  await h.context.refreshRecentEntries();
  assert.equal(h.calls.at(-1).headers.get('X-FL-Background'), null); assert.equal(h.expiry(), IDLE + 60000);
  h.setNow(120000); await h.context.fetchSalesAPI('/notes', { method: 'POST', body: '{}' });
  assert.equal(h.calls.at(-1).headers.get('X-FL-Background'), null); assert.equal(h.calls.at(-1).method, 'POST');
  assert.equal(h.expiry(), IDLE + 120000); assert.equal(h.lastSeen(), 120000);
  h.context.startRecentEntriesPolling(); h.setNow(180000); h.intervals[0].callback(); await settle();
  assert.equal(h.expiry(), IDLE + 120000); assert.equal(h.lastSeen(), 120000);
});

test('flag-off polling and manual Refresh retain the shared read credential and distinct activity markers', async () => {
  const h = harness({ enabled: false }); await settle(); h.context.startRecentEntriesPolling();
  h.intervals[0].callback(); await settle();
  assert.equal(h.calls.at(-1).headers.get('X-API-Key'), 'shared-fixture'); assert.equal(h.calls.at(-1).headers.get('X-FL-Background'), '1');
  await h.context.refreshRecentEntries(); assert.equal(h.calls.at(-1).headers.get('X-API-Key'), 'shared-fixture');
  assert.equal(h.calls.at(-1).headers.get('X-FL-Background'), null);
});

test('initial Refresh All passes the background marker to every loader while a manual Refresh passes none', async () => {
  const calls = [], get = () => element('refresh');
  const names = ['refreshTodayTile','refreshProductionCalendar','refreshFinishedGoods','refreshBatchInventory','refreshIngredients','refreshShipments','refreshReceipts','refreshDailyEntries','refreshNotes','refreshOrders','refreshExpectedReceipts','refreshSuppliesInventory','refreshSupplyRequests','refreshRecentEntries','refreshHealthBadge','refreshDispatchAttention'];
  const globals = { document: { getElementById: get }, state: { currentTab: 'recent' }, Date, Promise };
  for (const name of names) globals[name] = options => { calls.push({ name, options }); };
  const context = vm.createContext(globals); vm.runInContext(fn(dashboard, 'refreshAll'), context);
  await context.refreshAll({ headers: { 'X-FL-Background': '1' } });
  assert.equal(calls.length, names.length); assert.ok(calls.every(c => new Headers(c.options.headers).get('X-FL-Background') === '1'));
  calls.length = 0; await context.refreshAll(); assert.equal(calls.length, names.length);
  assert.ok(calls.every(c => new Headers(c.options.headers).get('X-FL-Background') === null));
});

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');
const TOKEN = 'fls_fixture-restored';
const PERSON = { id: 7, name: 'Arturo', role: 'floor' };
const settle = async () => { for (let i = 0; i < 10; i++) await new Promise(resolve => setImmediate(resolve)); };
function element(id) {
  return { id, hidden: false, disabled: false, value: '', textContent: '', innerHTML: '', dataset: {}, offsetHeight: 0, listeners: {},
    append() {}, setAttribute(k, v) { this[k] = v; }, focus() {}, classList: { toggle() {} },
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    fire(type) { for (const fn of this.listeners[type] || []) fn({ preventDefault() {} }); },
    showModal() { this.open = true; }, close() { this.open = false; this.fire('close'); } };
}
function harness({ enabled = true, saved = TOKEN, whoDeferred = false } = {}) {
  const data = new Map(saved ? [['fl-key-session-token', saved]] : []), calls = [], timers = [], els = new Map();
  let summaryDeferred = false, exchangeDeferred = false, exchangeToken = TOKEN, exchangePerson = PERSON;
  const storage = { getItem: k => data.get(k) || null, setItem: (k, v) => data.set(k, v), removeItem: k => data.delete(k) };
  const get = id => { if (!els.has(id)) els.set(id, element(id)); return els.get(id); };
  const summary = () => ({ date: get('date').value, title: 'Arturo shared shift', receipts_today: 0, summary_hash: 'a'.repeat(64), actor: PERSON,
    sections: [], confirmation: { status: 'unconfirmed', can_confirm: true } });
  const context = vm.createContext({ URL, URLSearchParams, Headers, Response, AbortController, Intl, Date, console,
    setTimeout: (fn, ms) => { const timer = { fn, ms }; timers.push(timer); return timer; }, clearTimeout: () => {},
    location: { href: 'https://fl.local/dashboard/shift-summary.html', protocol: 'https:', hostname: 'fl.local', origin: 'https://fl.local', search: '' },
    sessionStorage: storage, localStorage: storage,
    document: { readyState: 'complete', body: element('body'), documentElement: { lang: 'en', dataset: {}, style: { setProperty() {} } },
      createElement: tag => element(tag), getElementById: get, querySelector: () => null, querySelectorAll: () => [] },
    fetch: (url, init = {}) => new Promise(resolve => {
      const pathname = new URL(url).pathname;
      const call = { url: String(url), init, pathname, resolve: (body, status = 200) => resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })) };
      calls.push(call);
      if (pathname === '/auth/whoami') { if (!whoDeferred) call.resolve({ key_kind: 'session', actor: PERSON }); }
      else if (pathname === '/auth/session') { if (!exchangeDeferred) call.resolve({ key_kind: 'session', session_token: exchangeToken, actor: exchangePerson }); }
      else if (pathname === '/auth/session/logout') call.resolve({ signed_out: true });
      else if (pathname.endsWith('/confirm')) call.resolve({ message: 'Shift confirmed.' });
      else if (pathname === '/reports/shift-summary' && !summaryDeferred) call.resolve(summary());
      else if (pathname === '/auth/session/config') call.resolve({ key_sessions_enabled: enabled });
    }) });
  context.window = context;
  context.FLDashboard = { base: 'https://fl.local', apiBase: () => 'https://fl.local', config: Promise.resolve({ key_sessions_enabled: enabled }), ready: Promise.resolve(enabled), writesEnabled: enabled, key: async base => { assert.equal(base, 'https://fl.local'); return 'shared-fixture'; } };
  function run(name) { vm.runInContext(fs.readFileSync(path.join(ROOT, 'dashboard', name), 'utf8'), context, { filename: name }); }
  run('key-session.js');
  return { context, run, calls, timers, data, get, summary, deferSummary: () => { summaryDeferred = true; }, deferExchange: () => { exchangeDeferred = true; }, exchangeAs: (token, person) => { exchangeToken = token; exchangePerson = person; } };
}

for (const endpoint of ['/sales-orders', '/expected-receipts', '/suppliers']) test(`signed-in ${endpoint} reads await and use the restored personal session`, async () => {
  const h = harness({ whoDeferred: true }), transport = h.context.FLKeySessions.installDashboard('https://fl.local');
  const prepared = transport.prepare('https://fl.local' + endpoint, { headers: { 'X-API-Key': 'shared-fixture' } });
  await settle();
  h.calls.find(c => c.pathname === '/auth/whoami').resolve({ key_kind: 'session', actor: PERSON });
  assert.equal(new Headers((await prepared).headers).get('X-API-Key'), TOKEN);
});

test('flag-off and signed-out reads remain available, and personal credentials stay on the API origin', async () => {
  for (const options of [{ enabled: false }, { saved: '' }, {}]) {
    const h = harness(options), transport = h.context.FLKeySessions.installDashboard('https://fl.local');
    const url = Object.keys(options).length ? 'https://fl.local/sales-orders' : 'https://external.example/sales-orders';
    const prepared = await transport.prepare(url, { headers: { 'X-API-Key': 'shared-fixture' } });
    assert.equal(new Headers(prepared.headers).get('X-API-Key'), 'shared-fixture');
  }
});

test('an expired signed-in read clears shared token custody', async () => {
  const h = harness(), transport = h.context.FLKeySessions.installDashboard('https://fl.local');
  await transport.prepare('https://fl.local/suppliers', {});
  await settle();
  await transport.observe(new Response(JSON.stringify({ detail: { error_code: 'SESSION_EXPIRED' } }), { status: 401 }));
  assert.equal(h.data.has('fl-key-session-token'), false);
});

test('Shift Summary restores shared sign-in and confirms using its session without asking for a key', async () => {
  const h = harness(); h.run('shift-summary.js'); await settle();
  assert.equal(h.calls.filter(c => c.pathname === '/reports/shift-summary').length, 1);
  assert.equal(h.calls.find(c => c.pathname === '/reports/shift-summary').init.headers['X-API-Key'], TOKEN);
  assert.equal(h.get('sign-in-panel').hidden, true);
  assert.equal(h.get('title').textContent, 'Arturo shared shift');
  assert.equal(h.calls.find(c => c.pathname === '/auth/whoami').init.headers['X-FL-Background'], '1');
  assert.equal(h.calls.find(c => c.pathname === '/reports/shift-summary').init.headers['X-FL-Background'], '1');
  h.get('refresh').fire('click'); await settle();
  assert.equal(h.calls.at(-1).init.headers['X-FL-Background'], undefined);
  h.get('confirm-match').fire('click'); await settle();
  assert.equal(h.calls.find(c => c.pathname.endsWith('/confirm')).init.headers['X-API-Key'], TOKEN);
  assert.equal(h.calls.filter(c => c.pathname === '/auth/session').length, 0);
});

test('Shift Summary exchanges a new personal key once and signs out the shared session', async () => {
  const h = harness({ saved: '' }); h.run('shift-summary.js'); await settle();
  h.get('actor-key').value = 'fixture-personal'; h.get('sign-in-form').fire('submit'); await settle();
  const exchange = h.calls.find(c => c.pathname === '/auth/session');
  assert.ok(exchange, 'sign-in exchanges the key through FLKeySessions');
  assert.equal(JSON.parse(exchange.init.body).actor_key, 'fixture-personal');
  assert.equal(h.get('actor-key').value, '');
  assert.equal(exchange.init.headers['X-FL-Background'], undefined);
  assert.equal(h.calls.at(-1).init.headers['X-FL-Background'], undefined);
  assert.equal(h.data.get('fl-key-session-token'), TOKEN);
  h.get('sign-out').fire('click'); await settle();
  assert.equal(h.calls.at(-1).pathname, '/auth/session/logout');
  assert.equal(h.calls.at(-1).init.headers['X-API-Key'], TOKEN);
  assert.equal(h.data.has('fl-key-session-token'), false);
  assert.equal(h.get('summary-panel').hidden, true);
});

test('Shift Summary requests allow sixty seconds before aborting', async () => {
  const h = harness({ enabled: false, saved: '' }); h.run('shift-summary.js'); await settle();
  h.get('actor-key').value = 'legacy-fixture'; h.get('sign-in-form').fire('submit'); await settle();
  assert.ok(h.timers.some(timer => timer.ms === 60000), 'the summary has a 60-second request timeout');
  assert.equal(h.timers.some(timer => timer.ms === 20000), false);
});

test('expired Shift Summary reads clear the person and cannot leave a confirmable stale shift', async () => {
  const h = harness(); h.run('shift-summary.js'); await settle();
  h.deferSummary(); h.get('refresh').fire('click'); await settle();
  h.calls.at(-1).resolve({ detail: { error_code: 'SESSION_EXPIRED', message: 'Sign in again.' } }, 401); await settle();
  assert.equal(h.data.has('fl-key-session-token'), false);
  assert.equal(h.get('confirm-bar').hidden, true);
  assert.equal(h.get('summary-panel').hidden, true);
  assert.equal(h.get('identity').textContent.includes('Arturo'), false);
});

test('sign-out while a shared summary is loading never restores the old person or document', async () => {
  const h = harness(); h.deferSummary(); h.run('shift-summary.js'); await settle();
  const old = h.calls.find(c => c.pathname === '/reports/shift-summary');
  assert.ok(old, 'the shared session automatically loads the shift');
  h.get('sign-out').fire('click'); await settle();
  old.resolve(h.summary()); await settle();
  assert.equal(h.get('title').textContent, '');
  assert.equal(h.get('confirm-bar').hidden, true);
  assert.equal(h.get('identity').textContent.includes('Arturo'), false);
});

test('the real Shift Summary page loads the shared session library before its controller', () => {
  const html = fs.readFileSync(path.join(ROOT, 'dashboard/shift-summary.html'), 'utf8');
  assert.match(html, /key-session\.js\?v=/);
  assert.ok(html.indexOf('key-session.js') < html.indexOf('shift-summary.js'));
});

test('a delayed expired response for the previous person cannot clear the new shared sign-in', async () => {
  const h = harness(); h.deferSummary(); h.run('shift-summary.js'); await settle();
  const old = h.calls.find(c => c.pathname === '/reports/shift-summary');
  h.get('sign-out').fire('click'); await settle();
  h.exchangeAs('fls_fixture-beatriz', { id: 8, name: 'Beatriz', role: 'office' });
  h.get('actor-key').value = 'fixture-beatriz'; h.get('sign-in-form').fire('submit'); await settle();
  old.resolve({ detail: { error_code: 'SESSION_EXPIRED', message: 'Old session expired.' } }, 401); await settle();
  assert.equal(h.data.get('fl-key-session-token'), 'fls_fixture-beatriz');
  assert.equal(h.get('identity').textContent, 'Signed in as Beatriz · office');
  assert.equal(h.get('notice').textContent, '');
  assert.equal(h.calls.at(-1).init.headers['X-API-Key'], 'fls_fixture-beatriz');
});

test('choosing read-only cancels custody of a personal exchange still in flight', async () => {
  const h = harness({ saved: '' }); h.deferExchange(); h.run('shift-summary.js'); await settle();
  h.get('actor-key').value = 'fixture-personal'; h.get('sign-in-form').fire('submit'); await settle();
  const exchange = h.calls.find(c => c.pathname === '/auth/session');
  h.get('read-only').fire('click'); await settle();
  exchange.resolve({ key_kind: 'session', session_token: TOKEN, actor: PERSON }); await settle();
  assert.equal(h.data.has('fl-key-session-token'), false);
  const readOnlyKey = h.calls.find(c => c.pathname === '/auth/whoami').init.headers['X-API-Key'];
  assert.equal(readOnlyKey, 'shared-fixture', 'read-only uses the key FL serves'); assert.notEqual(readOnlyKey, TOKEN); assert.notEqual(readOnlyKey, 'fixture-personal');
  assert.equal(h.calls.at(-1).init.headers['X-API-Key'], readOnlyKey);
});

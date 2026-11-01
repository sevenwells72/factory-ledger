// The Notes list read sends a key before the API requires one (docs/security/notes-endpoints.md).
// Runs the real refreshNotes/fetchAPI with key-session.js and fetch-timeout.js against a fake API.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ROOT = path.resolve(__dirname, '..');
const dashboardRuntime = require('../dashboard/dashboard-runtime.js');
const TOKEN = 'fls_notes-fixture', SHARED = 'shared-fixture';
const dashboard = fs.readFileSync(path.join(ROOT, 'dashboard/dashboard.js'), 'utf8');
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
function fn(source, name) {
  const start = source.search(new RegExp('  (?:async )?function ' + name + '\\('));
  assert.ok(start >= 0, name);
  return source.slice(start, source.indexOf('\n  }', start) + 4);
}
function element(id) {
  return { id, hidden: false, value: '', textContent: '', innerHTML: '', dataset: {}, append() {}, after() {},
    setAttribute() {}, focus() {}, classList: { add() {}, remove() {}, toggle() {} }, addEventListener() {} };
}
function harness({ enabled = true, signedIn = false, notesStatus = 200, keyFails = false } = {}) {
  const calls = [], rendered = [], loadErrors = [], data = new Map(signedIn ? [['fl-key-session-token', TOKEN]] : []);
  const storage = { getItem: k => data.get(k) || null, setItem: (k, v) => data.set(k, v), removeItem: k => data.delete(k) };
  const elements = new Map(), get = id => { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); };
  const runtime = { ...dashboardRuntime, base: 'https://fl.local', config: Promise.resolve({ key_sessions_enabled: enabled }),
    ready: Promise.resolve(enabled), route: url => String(url),
    key: async () => { if (keyFails) throw dashboardRuntime.accessError(); return SHARED; } };
  const context = vm.createContext({ URL, Headers, Response, AbortController, Date, console,
    setTimeout: () => 1, clearTimeout() {},
    location: { href: 'https://fl.local/dashboard/index.html', origin: 'https://fl.local', hostname: 'fl.local' },
    sessionStorage: storage,
    document: { readyState: 'complete', body: element('body'), createElement: element, getElementById: get, querySelector: () => null },
    FLDashboard: runtime, API_BASE: 'https://fl.local/dashboard/api',
    state: { notesFilter: 'all', notesShowDone: false, notesData: [], notesLoaded: false },
    hideError() {}, renderNotes: () => rendered.push(true),
    showLoadError: (id, prefix, error) => loadErrors.push({ id, error }),
    fetch: async (url, options = {}) => {
      const target = new URL(url), headers = new Headers(options.headers);
      calls.push({ pathname: target.pathname, search: target.search, headers });
      if (target.pathname === '/auth/whoami') {
        const ok = headers.get('X-API-Key') === TOKEN;
        return new Response(JSON.stringify(ok ? { key_kind: 'session', actor: { id: 7, name: 'Arturo', role: 'office' } } : { detail: 'Invalid API key' }), { status: ok ? 200 : 403 });
      }
      const body = notesStatus === 200 ? { notes: [{ id: 1, category: 'todo', title: 'Order oats' }] } : { detail: 'Invalid API key' };
      return new Response(JSON.stringify(body), { status: notesStatus, headers: { 'Content-Type': 'application/json' } });
    } });
  context.window = context;
  for (const name of ['key-session.js', 'fetch-timeout.js']) vm.runInContext(fs.readFileSync(path.join(ROOT, 'dashboard', name), 'utf8'), context, { filename: name });
  vm.runInContext(['fetchAPI', 'refreshNotes'].map(name => fn(dashboard, name)).join('\n'), context);
  const notesCalls = () => calls.filter(c => c.pathname === '/dashboard/api/notes');
  return { context, calls, notesCalls, rendered, loadErrors };
}

test('signed out, the Notes list sends the shared dashboard key and keeps the background marker', async () => {
  const h = harness(); await settle();
  await h.context.refreshNotes({ headers: { 'X-FL-Background': '1' } });
  const [call] = h.notesCalls();
  assert.equal(call.headers.get('X-API-Key'), SHARED);
  assert.equal(call.headers.get('X-FL-Background'), '1');
  assert.equal(call.search, '?status=open');
  assert.equal(h.context.state.notesLoaded, true); assert.equal(h.rendered.length, 1);
});

test('signed in, the Notes list is read with the personal session, not the shared key', async () => {
  const h = harness({ signedIn: true }); await settle();
  await h.context.refreshNotes();
  const [call] = h.notesCalls();
  assert.equal(call.headers.get('X-API-Key'), TOKEN);
  assert.equal(call.headers.get('X-FL-Background'), null);
  assert.equal(h.context.state.notesData.length, 1);
});

test('with key sessions off, the Notes list still sends the shared dashboard key', async () => {
  const h = harness({ enabled: false }); await settle();
  await h.context.refreshNotes();
  assert.equal(h.notesCalls()[0].headers.get('X-API-Key'), SHARED);
});

test('on first load, a refused key shows the access message and renders no notes', async () => {
  const h = harness({ notesStatus: 403 }); await settle();
  await h.context.refreshNotes();
  assert.equal(h.loadErrors.length, 1);
  assert.equal(h.loadErrors[0].id, 'notes-error');
  assert.equal(h.loadErrors[0].error.accessUnavailable, true);
  assert.equal(h.context.state.notesLoaded, false); assert.equal(h.rendered.length, 0);
});

test('when the dashboard key cannot be fetched, no unauthenticated Notes request is sent', async () => {
  const h = harness({ keyFails: true }); await settle();
  await h.context.refreshNotes();
  assert.equal(h.notesCalls().length, 0);
  assert.equal(h.loadErrors.length, 1); assert.equal(h.loadErrors[0].error.accessUnavailable, true);
});

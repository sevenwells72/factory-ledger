'use strict';
// Owner-approved browser + one-tap name (dashboard/person-select.js). Pure client
// logic with fake fetch/storage/timers: every test asserts what is SENT (or
// refused) and what is REMEMBERED, including storage, race and missing-script failures.
const test = require('node:test');
const assert = require('node:assert/strict');
const lib = require('../dashboard/person-select.js');
const runtime = require('../dashboard/dashboard-runtime.js');

const TOKEN_A = 'flb_' + 'A'.repeat(43), TOKEN_B = 'flb_' + 'B'.repeat(43);
const PEOPLE = [
  { id: 1, display_name: 'Michael', role_label: 'Owner', available: true },
  { id: 3, display_name: 'Luz', role_label: 'Office', available: true },
  { id: 2, display_name: 'Arturo', role_label: 'Floor Lead', available: true },
  { id: 4, display_name: 'Miriam', role_label: 'Office', available: true },
];
const CONFIG = { person_select_enabled: true, person_select_available: true, key_sessions_enabled: true };

function memoryStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data, failSet: false, failRemove: false, failGet: false, lieOnSet: false,
    getItem(k) { if (this.failGet) throw new Error('blocked'); return data.has(k) ? data.get(k) : null; },
    setItem(k, v) { if (this.failSet) throw new Error('quota'); if (!this.lieOnSet) data.set(k, String(v)); },
    removeItem(k) { if (this.failRemove) throw new Error('blocked'); data.delete(k); },
  };
}
function fakeTimers() { return { setTimeout: () => 0, clearTimeout: () => {} }; }
function server(routes) {
  const calls = [];
  const fetch = async (url, options = {}) => {
    const path = url.replace('https://fl.test', '');
    calls.push({ path, method: options.method || 'GET', key: options.headers?.['X-API-Key'] });
    const handler = routes[path];
    const result = typeof handler === 'function' ? await handler(options) : handler;
    if (result instanceof Error) throw result;
    return { ok: result.status < 400, status: result.status, json: async () => result.body };
  };
  return { fetch, calls };
}
function events() {
  const listeners = {};
  return { addEventListener(type, fn) { (listeners[type] ||= []).push(fn); }, dispatchEvent(e) { (listeners[e.type] || []).forEach(fn => fn(e)); return true; },
           fire(type, e) { (listeners[type] || []).forEach(fn => fn(e)); }, listeners };
}
const approvedStatus = (people = PEOPLE) => ({ status: 200, body: { status: 'approved', label: 'Tablet', people } });
async function client({ storage = memoryStorage({ 'fl-browser-token': TOKEN_A }), routes = {}, config = CONFIG, ev = events() } = {}) {
  const s = server({ '/auth/browsers/status': approvedStatus(), ...routes });
  const c = lib.create({ base: 'https://fl.test', fetch: s.fetch, storage, config: Promise.resolve(config), events: ev, timers: fakeTimers() });
  await c.init();
  return { c, s, storage, ev };
}

test('mode off: no headers, so callers keep their existing credential path', async () => {
  const { c } = await client({ config: { person_select_enabled: false, key_sessions_enabled: true } });
  assert.equal(c.mode, false);
  assert.equal(c.headers({ write: true }), null);
});

test('mode on but FL reports unavailable: every write is refused, no key/PIN fallback', async () => {
  for (const problem of ['PERSON_SELECT_UNAVAILABLE', 'PERSON_CHOICES_MISCONFIGURED']) {
    const { c } = await client({ config: { ...CONFIG, person_select_available: false, person_select_problem: problem } });
    assert.equal(c.mode, true);
    assert.equal(c.state.status, 'unavailable');
    assert.throws(() => c.headers({ write: true }), e => e.code === 'UNAVAILABLE');
    assert.throws(() => c.headers({ write: false }), e => e.code === 'UNAVAILABLE');
  }
});

test('untrusted browser asks for approval, polls, then shows exactly the four names', async () => {
  let approved = false;
  const storage = memoryStorage();
  const { c, s } = await client({ storage, routes: {
    '/auth/browsers/request': { status: 201, body: { browser_token: TOKEN_A, pairing_code: 'ABCD-EFGH', pairing_expires_at: 'x' } },
    '/auth/browsers/status': () => approved ? approvedStatus() : { status: 200, body: { status: 'pending' } } } });
  assert.equal(c.state.status, 'untrusted');
  assert.throws(() => c.headers({ write: true }), e => e.code === 'BROWSER_NOT_APPROVED');
  const pairing = await c.requestApproval();
  assert.equal(pairing.code, 'ABCD-EFGH');
  assert.equal(storage.data.get('fl-browser-token'), TOKEN_A);
  await c.refresh();
  assert.equal(c.state.status, 'pending');
  approved = true;
  await c.refresh();
  assert.deepEqual(c.state.people.map(p => p.display_name), ['Michael', 'Luz', 'Arturo', 'Miriam']);
  assert.ok(s.calls.every(call => call.path !== '/auth/browsers/status' || call.key === TOKEN_A));
});

test('a write needs a chosen name; the name persists across a browser restart and never expires', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const first = await client({ storage });
  assert.throws(() => first.c.headers({ write: true }), e => e.code === 'PERSON_REQUIRED');
  assert.deepEqual(first.c.headers({ write: false }), { 'X-API-Key': TOKEN_A });
  first.c.select({ id: 2 });
  assert.deepEqual(first.c.headers({ write: true }), { 'X-API-Key': TOKEN_A, 'X-FL-Person': '2' });
  const restarted = await client({ storage });          // new page / browser restart, same storage
  assert.equal(restarted.c.person.display_name, 'Arturo');
  assert.deepEqual(restarted.c.headers({ write: true }), { 'X-API-Key': TOKEN_A, 'X-FL-Person': '2' });
});

test('an unavailable (deactivated) name cannot be chosen and a saved one is dropped, never substituted', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const { c } = await client({ storage });
  c.select({ id: 3 });
  const people = PEOPLE.map(p => p.id === 3 ? { ...p, available: false } : p);
  const later = await client({ storage, routes: { '/auth/browsers/status': approvedStatus(people) } });
  assert.equal(later.c.person, null);
  assert.throws(() => later.c.select({ id: 3 }), e => e.code === 'PERSON_NOT_AVAILABLE');
  assert.throws(() => later.c.headers({ write: true }), e => e.code === 'PERSON_REQUIRED');
});

test('a name saved under an earlier browser approval is not restored after re-approval', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const { c } = await client({ storage });
  c.select({ id: 1 });
  storage.data.set('fl-browser-token', TOKEN_B);          // approved again: new token
  const again = await client({ storage });
  assert.equal(again.c.person, null);
});

test('stale person/token values in storage are rejected (wrong shape, wrong token tag, non-integer id)', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A,
    'fl-person': JSON.stringify({ id: '2', display_name: 'Arturo', role_label: 'Floor Lead', browser: TOKEN_A.slice(4, 16) }) });
  const { c } = await client({ storage });
  assert.equal(c.person, null);
});

test('Change user in another tab: this tab adopts it and refuses the pending write once', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const ev = events();
  const { c } = await client({ storage, ev });
  c.select({ id: 2 });
  const changes = [];
  ev.addEventListener('fl-person-change', e => changes.push(e.detail.person?.id ?? null));
  // The other tab writes Luz directly to shared storage.
  storage.data.set('fl-person', JSON.stringify({ id: 3, display_name: 'Luz', role_label: 'Office', browser: TOKEN_A.slice(4, 16) }));
  assert.throws(() => c.headers({ write: true }), e => e.code === 'PERSON_CHANGED');
  assert.deepEqual(c.headers({ write: true }), { 'X-API-Key': TOKEN_A, 'X-FL-Person': '3' });
  assert.deepEqual(changes, [3]);
  // A storage event (normal browser path) also adopts the change.
  storage.data.delete('fl-person');
  ev.fire('storage', { key: 'fl-person' });
  assert.equal(c.person, null);
});

test('R6: failing to SAVE a name leaves nobody selected and stays unavailable on retry', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const { c } = await client({ storage });
  storage.failSet = true;
  assert.throws(() => c.select({ id: 2 }), e => e.code === 'UNAVAILABLE');
  assert.equal(c.person, null);
  storage.failSet = false;                                 // storage recovers; the page must still not guess
  assert.throws(() => c.headers({ write: true }), e => e.code === 'UNAVAILABLE');
  assert.throws(() => c.select({ id: 2 }), e => e.code === 'UNAVAILABLE');
});

test('R6: failing to CLEAR a name on Change user never restores the previous person on a later retry', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const { c } = await client({ storage });
  c.select({ id: 2 });
  storage.failRemove = true;
  assert.throws(() => c.changeUser(), e => e.code === 'UNAVAILABLE');
  assert.equal(c.person, null);
  for (let attempt = 0; attempt < 3; attempt++) {          // first, second, third retry
    assert.throws(() => c.headers({ write: true }), e => e.code === 'UNAVAILABLE');
    await c.refresh();
    assert.equal(c.person, null, 'attempt ' + attempt);
  }
  // Even a fresh page that CAN read the stale saved name does not get it silently:
  // the stale entry is still there, so the owner/user sees the picker only after a real clear.
  assert.equal(c.broken, true);
});

test('R6: storage that silently drops writes (read-back mismatch) fails closed', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const { c } = await client({ storage });
  storage.lieOnSet = true;
  assert.throws(() => c.select({ id: 2 }), e => e.code === 'UNAVAILABLE');
  assert.equal(c.person, null);
  assert.equal(c.state.status, 'unavailable');
});

test('R6: unreadable storage at start fails closed', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  storage.failGet = true;
  const { c } = await client({ storage });
  assert.throws(() => c.headers({ write: true }), e => e.code === 'UNAVAILABLE');
});

test('R7: a status answer for an OLD token is ignored after re-approval starts', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  let release;
  const slow = new Promise(resolve => { release = resolve; });
  const routes = {
    '/auth/browsers/status': async options => options.headers['X-API-Key'] === TOKEN_A ? (await slow, approvedStatus()) : { status: 200, body: { status: 'pending' } },
    '/auth/browsers/request': { status: 201, body: { browser_token: TOKEN_B, pairing_code: 'WXYZ-2345', pairing_expires_at: 'x' } } };
  const s = server(routes);
  const c = lib.create({ base: 'https://fl.test', fetch: s.fetch, storage, config: Promise.resolve(CONFIG), events: events(), timers: fakeTimers() });
  const starting = c.init();                                // refresh for TOKEN_A in flight
  await new Promise(r => setImmediate(r));
  await c.requestApproval();                                // now TOKEN_B, pending
  release();
  await starting;
  assert.equal(c.state.status, 'pending');                  // NOT "approved" from TOKEN_A's late answer
  assert.throws(() => c.headers({ write: true }), e => e.code === 'BROWSER_NOT_APPROVED');
});

test('R7: an older refresh finishing after a newer one does not overwrite it', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  let n = 0, releaseFirst;
  const first = new Promise(r => { releaseFirst = r; });
  const routes = { '/auth/browsers/status': async () => {
    n += 1;
    if (n === 2) { await first; return approvedStatus(); }             // older, slow, still "approved"
    if (n === 3) return { status: 401, body: { detail: { error_code: 'BROWSER_REVOKED' } } };  // newer: revoked
    return approvedStatus();
  } };
  const { c } = await client({ storage, routes });
  const older = c.refresh(), newer = c.refresh();
  await newer;
  releaseFirst();
  await older;
  assert.equal(c.state.status, 'revoked');
  assert.equal(storage.data.get('fl-browser-token'), undefined);
});

test('revoked browser reported by FL on any request clears the token and the name', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A });
  const { c } = await client({ storage });
  c.select({ id: 2 });
  c.observe(401, { detail: { error_code: 'BROWSER_REVOKED' } });
  assert.equal(c.state.status, 'revoked');
  assert.equal(c.person, null);
  assert.equal(storage.data.has('fl-browser-token'), false);
  assert.throws(() => c.headers({ write: true }), e => e.code === 'BROWSER_NOT_APPROVED');
});

test('R8: selector mode with the script missing gives every transport a refusing stand-in', () => {
  const stand = runtime.personClient({});                   // window without FLPersonSelect
  assert.throws(() => stand.headers({ write: true }), e => e.code === 'PERSON_SELECT_UNAVAILABLE');
  assert.equal(stand.headers({ write: false }), null);       // reads keep the read-only dashboard key
  assert.equal(stand.broken, true);
  const real = { headers: () => ({ ok: 1 }) };
  assert.equal(runtime.personClient({ FLPersonSelect: real }), real);
  assert.equal(runtime.personSelect({ person_select_enabled: 'true' }), false);   // exact true only
  assert.equal(runtime.personSelect({ person_select_enabled: true }), true);
});

test('R10: restoring the saved name on page open is not a change of user; real changes are', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A,
    'fl-person': JSON.stringify({ id: 3, display_name: 'Luz', role_label: 'Office', browser: TOKEN_A.slice(4, 16) }) });
  const ev = events(), changes = [];
  ev.addEventListener('fl-person-change', e => changes.push([e.detail.previous?.id ?? null, e.detail.person?.id ?? null]));
  const { c } = await client({ storage, ev });
  assert.equal(c.person.display_name, 'Luz');
  await c.refresh();                                    // later polls/refreshes with the same name: still nothing
  assert.deepEqual(changes, []);
  c.select({ id: 2 });                                  // a real Change user
  assert.deepEqual(changes, [[3, 2]]);
  storage.data.set('fl-person', JSON.stringify({ id: 4, display_name: 'Miriam', role_label: 'Office', browser: TOKEN_A.slice(4, 16) }));
  ev.fire('storage', { key: 'fl-person' });             // another tab
  assert.deepEqual(changes, [[3, 2], [2, 4]]);
});

test('R10: a name that became unavailable while the page was closed is dropped quietly on open', async () => {
  const storage = memoryStorage({ 'fl-browser-token': TOKEN_A,
    'fl-person': JSON.stringify({ id: 3, display_name: 'Luz', role_label: 'Office', browser: TOKEN_A.slice(4, 16) }) });
  const ev = events(), changes = [];
  ev.addEventListener('fl-person-change', e => changes.push(e.detail.person?.id ?? null));
  const people = PEOPLE.map(p => p.id === 3 ? { ...p, available: false } : p);
  const { c } = await client({ storage, ev, routes: { '/auth/browsers/status': approvedStatus(people) } });
  assert.equal(c.person, null);
  assert.deepEqual(changes, []);
  assert.equal(storage.data.has('fl-person'), false);
});

test('R11: unreadable settings mean "not available", never selection-off or legacy', async () => {
  const s = server({});
  const c = lib.create({ base: 'https://fl.test', fetch: s.fetch, storage: memoryStorage({ 'fl-browser-token': TOKEN_A }),
                         config: Promise.reject(new Error('503')), events: events(), timers: fakeTimers() });
  await c.init();
  assert.equal(c.mode, true);
  assert.equal(c.state.status, 'unavailable');
  assert.equal(c.state.problem, 'CONFIG_UNKNOWN');
  assert.throws(() => c.headers({ write: true }), e => e.code === 'UNAVAILABLE');
  assert.equal(s.calls.length, 0, 'no second config/status request');
});

test('R11: key-session clients share the page config by default; a failed config restores nothing', async () => {
  const sessions = require('../dashboard/key-session.js');
  const calls = [];
  const previous = globalThis.FLDashboard;
  globalThis.FLDashboard = { config: Promise.reject(new Error('503')) };
  globalThis.FLDashboard.config.catch(() => {});
  try {
    const storage = memoryStorage({ 'fl-key-session-token': 'fls_' + 'L'.repeat(43) });
    const client = sessions.create({ base: 'https://fl.test', storage,
      fetch: async url => { calls.push(url); return { ok: true, status: 200, json: async () => ({ key_sessions_enabled: true, actor: { name: 'Luz' }, key_kind: 'session' }) }; } });
    await assert.rejects(client.init());
    assert.deepEqual(calls, [], 'no independent config fetch and no whoami with the old token');
    assert.equal(client.actor, null);
    assert.equal(client.token, '');
  } finally { globalThis.FLDashboard = previous; }
});

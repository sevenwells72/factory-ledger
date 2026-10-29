const test = require('node:test'); const assert = require('node:assert/strict'); const fs = require('node:fs'); const path = require('node:path');
const sessions = require('../dashboard/key-session.js');
// Phone PIN sign-in in the shared client: what is stored where, and which credential goes to which route.

const SECRET = 'd'.repeat(43);
function store(initial = {}) { const map = new Map(Object.entries(initial)); return { map, getItem: k => map.has(k) ? map.get(k) : null, setItem: (k, v) => map.set(k, String(v)), removeItem: k => map.delete(k) }; }
function server(config, { pinStatus = 200, pinBody } = {}) {
  const calls = [];
  const fetch = async (url, options = {}) => {
    const route = new URL(url).pathname, body = options.body ? JSON.parse(options.body) : undefined;
    calls.push({ route, body, key: options.headers?.['X-API-Key'] });
    const reply = (status, json) => ({ ok: status < 400, status, json: async () => json });
    if (route === '/auth/session/config') return reply(200, config);
    if (route === '/auth/whoami') return reply(200, { key_kind: 'session', actor: { id: 2, name: 'Arturo', role: 'floor' } });
    if (route === '/auth/session') return reply(200, { key_kind: 'session', session_token: 'fls_key', actor: { id: 2, name: 'Arturo', role: 'floor' } });
    if (route === '/auth/pin') return reply(pinStatus, pinBody || { key_kind: 'session', session_token: 'fls_pin', actor: { id: 2, name: 'Arturo', role: 'floor' } });
    if (route === '/auth/device/enroll') return reply(200, { key_kind: 'session', session_token: 'fls_phone', device_secret: SECRET, device: { id: 7, label: "Arturo's phone" }, actor: { id: 2, name: 'Arturo', role: 'floor' } });
    if (route === '/auth/session/logout' || route === '/auth/device/forget') return reply(200, {});
    return reply(404, {});
  };
  return { calls, fetch };
}
const ON = { key_sessions_enabled: true, pin_sign_in: true };
const enrolled = () => store({ [sessions.DEVICE]: JSON.stringify({ secret: SECRET, actor: { id: 2, name: 'Arturo', role: 'floor' }, label: "Arturo's phone" }) });
const client = (srv, storage, deviceStorage, config = ON) => sessions.create({ base: 'https://fl.local', fetch: srv.fetch, storage, deviceStorage, config });

test('six digits on an enrolled phone are a PIN; a personal key is still a key', async () => {
  const srv = server(ON), tab = store(), phone = enrolled(), c = client(srv, tab, phone);
  await c.init();
  assert.equal(c.pinEnabled, true); assert.equal(c.device.actor.name, 'Arturo'); assert.equal('secret' in c.device, false);
  await c.signIn('482916');
  assert.deepEqual(srv.calls.at(-1), { route: '/auth/pin', body: { device_secret: SECRET, pin: '482916' }, key: undefined });
  // The 12-hour phone session survives closing the browser; the per-tab store stays empty.
  assert.equal(phone.getItem(sessions.PIN_STORE), 'fls_pin'); assert.equal(tab.getItem(sessions.STORE), null); assert.equal(c.phoneSession, true);
  await c.signIn('actor-key-long-value');
  assert.equal(srv.calls.at(-1).route, '/auth/session'); assert.equal(tab.getItem(sessions.STORE), 'fls_key'); assert.equal(phone.getItem(sessions.PIN_STORE), null);
});

test('a reopened browser restores the phone session from the phone store', async () => {
  const srv = server(ON), phone = enrolled(); phone.setItem(sessions.PIN_STORE, 'fls_pin');
  const c = client(srv, store(), phone); await c.init();
  assert.equal(c.actor.name, 'Arturo'); assert.equal(c.token, 'fls_pin');
  assert.deepEqual(srv.calls.map(x => [x.route, x.key]), [['/auth/whoami', 'fls_pin']]);
});

test('enrolment keeps only the device secret and the phone token; the PIN is never stored', async () => {
  const srv = server(ON), tab = store(), phone = store(), c = client(srv, tab, phone);
  await c.init(); await c.signIn('actor-key-long-value');
  await c.enroll('482916', "Arturo's phone", 'actor-key-long-value');
  assert.deepEqual(srv.calls.at(-1), { route: '/auth/device/enroll', body: { actor_key: 'actor-key-long-value', pin: '482916', label: "Arturo's phone" }, key: 'fls_key' });
  assert.equal(JSON.parse(phone.getItem(sessions.DEVICE)).secret, SECRET);
  assert.equal(phone.getItem(sessions.PIN_STORE), 'fls_phone'); assert.equal(tab.getItem(sessions.STORE), null);
  const everything = [...tab.map.values(), ...phone.map.values()].join(' ');
  assert.doesNotMatch(everything, /482916|actor-key-long-value/);
});

test('signing out keeps the phone enrolled; an unrecognised phone forgets itself', async () => {
  const srv = server(ON), phone = enrolled(), c = client(srv, store(), phone);
  await c.init(); await c.signIn('482916'); await c.signOut();
  assert.equal(c.actor, null); assert.equal(phone.getItem(sessions.PIN_STORE), null); assert.ok(phone.getItem(sessions.DEVICE));
  const gone = server(ON, { pinStatus: 401, pinBody: { detail: { error_code: 'DEVICE_NOT_RECOGNIZED', message: 'This phone is not set up for a PIN.' } } });
  const d = client(gone, store(), phone); await d.init();
  await assert.rejects(d.signIn('482916'), error => error.code === 'DEVICE_NOT_RECOGNIZED');
  assert.equal(phone.getItem(sessions.DEVICE), null); assert.equal(d.device, null);
});

test('a wrong PIN keeps the phone and reports the server message', async () => {
  const srv = server(ON, { pinStatus: 401, pinBody: { detail: { error_code: 'PIN_INVALID', message: 'Wrong PIN. / PIN incorrecto.', attempts_left: 4 } } });
  const phone = enrolled(), c = client(srv, store(), phone); await c.init();
  await assert.rejects(c.signIn('000001'), error => error.code === 'PIN_INVALID' && error.detail.attempts_left === 4 && /Wrong PIN/.test(error.message));
  assert.ok(phone.getItem(sessions.DEVICE)); assert.equal(c.actor, null);
});

test('with PINs switched off a phone token is dropped and digits go to the key exchange', async () => {
  const srv = server({ key_sessions_enabled: true, pin_sign_in: false }), phone = enrolled(); phone.setItem(sessions.PIN_STORE, 'fls_pin');
  const c = client(srv, store(), phone, { key_sessions_enabled: true, pin_sign_in: false }); await c.init();
  assert.equal(c.pinEnabled, false); assert.equal(c.token, ''); assert.equal(phone.getItem(sessions.PIN_STORE), null); assert.ok(phone.getItem(sessions.DEVICE));
  await c.signIn('482916'); assert.equal(srv.calls.at(-1).route, '/auth/session');
});

test('without browser storage (or key sessions off) nothing about PINs happens', async () => {
  const srv = server(ON), c = sessions.create({ base: 'https://fl.local', fetch: srv.fetch, storage: store(), deviceStorage: null, config: ON });
  await c.init(); assert.equal(c.pinEnabled, false);
  const off = client(server({ key_sessions_enabled: false, pin_sign_in: true }), store(), enrolled(), { key_sessions_enabled: false, pin_sign_in: true });
  await off.init(); assert.equal(off.pinEnabled, false); assert.equal(off.enabled, false);
});

test('pages show PIN links only while the server has PINs on', () => {
  const read = name => fs.readFileSync(path.join(__dirname, '../dashboard', name), 'utf8');
  assert.match(read('ledger-forms.css'), /:root:not\(\[data-fl-pin="on"\]\) \[data-fl-pin-ui\]\{display:none!important\}/);
  assert.match(read('dashboard-runtime.js'), /dataset\.flPin=api\.pinEnabled\?'on':'off'/);
  assert.match(read('receipt-shell.js'), /a\.dataset\.flWriteUi='';a\.dataset\.flPinUi='';a\.href=page\('pin\.html'\)/);
  for (const page of ['pin.html', 'staff-access.html']) {
    const html = read(page);
    assert.match(html, /<main data-fl-write-ui>/); assert.match(html, /key-session\.js\?v=5/); assert.match(html, /id="actor-key" type="password"/);
  }
  // The owner screen shows no secrets: it only renders labels, status and times from /auth/devices.
  assert.doesNotMatch(read('staff-access.js'), /secret|pin_hash|key_hash/);
});

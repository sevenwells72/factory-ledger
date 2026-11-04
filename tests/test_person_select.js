'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const lib = require('../dashboard/person-select.js');
const runtime = require('../dashboard/dashboard-runtime.js');
function store(initial = {}) {
  const values = new Map(Object.entries(initial));
  return { values, getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v), removeItem: k => values.delete(k) };
}
function events() {
  const listeners = {};
  return { addEventListener(type, fn) { (listeners[type] ||= []).push(fn); }, dispatchEvent(e) { for (const fn of listeners[e.type] || []) fn(e); },
    fire(type, e) { for (const fn of listeners[type] || []) fn(e); } };
}
async function client(storage = store(), ev = events(), options = {}) {
  const c = lib.create({ storage, events: ev, ...options }); await c.init(); return c;
}
test('fresh browser offers exactly four names and no authentication request', async () => {
  let calls = 0;
  const c = await client(store(), events(), { fetch() { calls++; throw Error('No network needed'); }, config: { person_select_enabled: false } });
  assert.equal(c.mode, true); assert.equal(c.state.status, 'approved');
  assert.deepEqual(c.state.people.map(p => p.display_name), ['Michael', 'Luz', 'Arturo', 'Miriam']);
  for (const name of c.state.people) {
    c.select(name); assert.deepEqual(c.headers({ write: true }), { 'X-FL-Person': name.display_name });
  }
  assert.equal(calls, 0);
});
test('a tapped name persists across page/browser restarts without token or expiry', async () => {
  const storage = store(); const a = await client(storage);
  a.select({ id: 'Arturo' }); const b = await client(storage);
  assert.equal(b.person.display_name, 'Arturo');
  assert.deepEqual(b.headers({ write: true }), { 'X-FL-Person': 'Arturo' });
  assert.deepEqual([...storage.values.keys()], ['fl-person']);
});
test('old stored choices migrate by display name without needing their browser token', async () => {
  const c = await client(store({ 'fl-person': JSON.stringify({ id: 7, display_name: 'Luz', browser: 'old-approval' }), 'fl-browser-token': 'obsolete' }));
  assert.equal(c.person.display_name, 'Luz'); assert.deepEqual(c.headers({ write: true }), { 'X-FL-Person': 'Luz' });
});
test('unknown stored names cannot be mistaken for one of the four operators', async () => {
  const c = await client(store({ 'fl-person': JSON.stringify({ id: 1, display_name: 'Someone else' }) }));
  assert.equal(c.person, null); assert.throws(() => c.headers({ write: true }), e => e.code === 'PERSON_REQUIRED');
  assert.equal(c.headers(), null);
});
test('selection works even when browser storage refuses reads or writes', async () => {
  const blocked = { getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); }, removeItem() { throw Error('blocked'); } };
  const c = await client(blocked); c.select({ id: 'Michael' });
  assert.deepEqual(c.headers({ write: true }), { 'X-FL-Person': 'Michael' });
  c.changeUser(); c.select({ id: 'Miriam' });
  assert.deepEqual(c.headers({ write: true }), { 'X-FL-Person': 'Miriam' });
});
test('Change user clears the saved choice and immediately accepts the next tap', async () => {
  const storage = store(), c = await client(storage);
  c.select({ id: 'Michael' }); c.changeUser();
  assert.equal(storage.getItem('fl-person'), null); assert.equal(c.person, null);
  c.select({ id: 'Arturo' }); assert.equal(c.person.display_name, 'Arturo');
});
test('another tab changes the operator without a recheck prompt', async () => {
  const storage = store(), ev = events(), c = await client(storage, ev);
  c.select({ id: 'Arturo' });
  storage.setItem('fl-person', JSON.stringify({ display_name: 'Luz' })); ev.fire('storage', { key: 'fl-person' });
  assert.deepEqual(c.headers({ write: true }), { 'X-FL-Person': 'Luz' });
});
test('expired legacy credentials and browser revocation responses never log a name out', async () => {
  const c = await client(); c.select({ id: 'Michael' });
  for (const error_code of ['SESSION_EXPIRED', 'BROWSER_REVOKED', 'BROWSER_NOT_APPROVED']) c.observe(401, { detail: { error_code } });
  await c.refresh(); assert.equal(c.person.display_name, 'Michael');
});
test('restoring a saved name does not emit a user-change reset; a new tap does', async () => {
  const ev = events(), changes = [];
  ev.addEventListener('fl-person-change', e => changes.push(e.detail.person?.display_name));
  const c = await client(store({ 'fl-person': JSON.stringify({ display_name: 'Luz' }) }), ev);
  assert.deepEqual(changes, []); c.select({ id: 'Miriam' }); assert.deepEqual(changes, ['Miriam']);
});
test('missing selector script refuses unattributed writes', () => {
  const missing = runtime.personClient({}); assert.throws(() => missing.headers({ write: true }), e => e.code === 'PERSON_SELECT_UNAVAILABLE');
});
test('sign-in UI contains exactly four buttons and no title, instructions, roles or close button', async () => {
  class Element {
    constructor(tag) { this.tagName = tag; this.children = []; this.dataset = {}; }
    append(...nodes) { this.children.push(...nodes); } setAttribute() {}
  }
  const body = new Element('body');
  const document = { body, documentElement: { dataset: {}, lang: 'en' }, createElement: tag => new Element(tag), querySelector: () => null };
  const c = await client(); lib.mount(c, { document });
  const overlay = body.children.find(n => n.className === 'fl-person-overlay'), buttons = overlay.children[0].children;
  assert.deepEqual(buttons.map(b => b.textContent), ['Michael', 'Luz', 'Arturo', 'Miriam']);
  assert.ok(buttons.every(b => b.tagName === 'button')); assert.equal(overlay.children.length, 1);
  buttons[0].onclick(); assert.equal(overlay.hidden, true); assert.equal(document.documentElement.dataset.flPersonPicker, 'off');
});

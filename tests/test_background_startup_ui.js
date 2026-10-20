const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');
const settle = async () => { for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve)); };
function element(tag = 'div') {
  return { tag, isConnected: true, children: [], listeners: {}, dataset: {}, style: {}, value: '', hidden: false,
    append(...children) { children.forEach(child => child.parentElement = this); this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this[name] = value; },
    addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); },
    async fire(name, event = {}) { for (const callback of this.listeners[name] || []) await callback({ preventDefault() {}, target: this, ...event }); },
    querySelector(selector) { return this.children.find(child => child.tag === selector) || element(selector); },
    querySelectorAll() { return []; }, focus() {}, scrollIntoView() {}, remove() {} };
}
function harness(search = '') {
  const els = new Map(), events = new Map(), calls = [];
  const get = id => { if (!els.has(id)) els.set(id, element(id)); return els.get(id); };
  const ui = { bind() {}, ready: Promise.resolve(true), session: { token: 'fls_test-startup', actor: { id: 7 } },
    problem: value => Error(String(value)),
    setText: (node, value) => { node.textContent = String(value); },
    t: en => en, el: (tag, value, parent) => { const node = element(tag); if (value !== undefined) node.textContent = String(value); parent?.append(node); return node; },
    request: async (pathname, body, options = {}) => {
      calls.push({ pathname, body, options });
      if (pathname.startsWith('/receipts/')) return { receipt_number: 'RCV-TEST', status: 'committed', transactions: [] };
      if (pathname.startsWith('/exceptions')) return { exceptions: [], count: 0 };
      if (pathname === '/resolve') return { outcome: 'match', match: { id: 22, label: 'Selected order' } };
      return { order_id: 17, order_number: 'SO-TEST', lines: [] };
    },
    exceptionQueue: require('../dashboard/ledger-ui.js').exceptionQueue, openingStock: require('../dashboard/ledger-ui.js').openingStock, groups: () => [], receiptState: () => ({ state: 'recorded', tone: 'good' }), voidChoices: () => [], draftHtml: () => '',
    notice() {}, failure(error) { throw error; }, relabel() {}, translate() {} };
  const context = vm.createContext({ setTimeout, clearTimeout, URL, URLSearchParams, FLLedger: ui, FLReceipts: { page: name => name },
    FLDashboard: { writesEnabled: true }, location: { search, href: 'https://fl.local/entry.html' + search }, history: { replaceState() {} },
    document: { getElementById: get }, console });
  context.window = context;
  context.addEventListener = (name, callback) => { if (!events.has(name)) events.set(name, []); events.get(name).push(callback); };
  return { calls, ui, get, context,
    run: name => vm.runInContext(fs.readFileSync(path.join(ROOT, 'dashboard', name), 'utf8'), context, { filename: name }),
    fire: async name => { for (const callback of events.get(name) || []) await callback({ type: name }); } };
}
function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }

test('receipt initial restoration is background; explicit sign-in reload is user activity', async () => {
  const h = harness('?number=RCV-TEST'); h.run('receipt.js'); await settle();
  assert.equal(h.calls[0].options.background, true);
  assert.equal(h.calls[0].options.read, true);
  await h.fire('ledger-auth'); await settle();
  assert.equal(h.calls.at(-1).options.background, false);
});

test('exceptions restore in background while Refresh and sign-in remain user activity', async () => {
  const h = harness(); h.run('exceptions.js'); await settle();
  assert.equal(h.calls[0].options.background, true);
  await h.get('content').children[0].onclick({ type: 'click' });
  assert.equal(h.calls.at(-1).options.background, false);
  await h.fire('ledger-auth');
  assert.equal(h.calls.at(-1).options.background, false);
});

test('shipping preload and chained detail are background; selecting another order is user activity', async () => {
  const h = harness('?action=ship&order=17'); h.run('entry.js'); await settle();
  assert.equal(h.calls.length, 2);
  assert.ok(h.calls.every(call => call.options.background === true));
  const picker = descendants(h.get('content')).find(node => node.dataset.picker === 'order');
  const search = picker.children.find(node => node.tag === 'button');
  await search.onclick(); await settle();
  const candidate = descendants(picker).find(node => node.tag === 'button' && node.textContent === 'Selected order');
  candidate.onclick(); await settle();
  assert.equal(h.calls.at(-1).pathname, '/sales/orders/22');
  assert.equal(h.calls.at(-1).options.background, false);
});

for (const savedChat of [true, false]) test(`Assistant automatic ${savedChat ? 'resume' : 'chat creation'} is background; New chat is user activity`, async () => {
  const h = harness(), data = new Map(savedChat ? [['fl-assistant-chat:7', 'test-chat']] : []);
  const storage = { getItem: key => data.get(key) || null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
  const actor = { id: 7, name: 'Test Floor', role: 'floor' };
  const pilot = { enabled: true, actor, token: 'fls_test-startup', init: async () => true };
  const calls = [];
  Object.assign(h.context, { AbortController, setTimeout, clearTimeout, navigator: {}, localStorage: storage, sessionStorage: storage,
    FLKeySessions: { create: () => pilot },
    fetch: async (url, init) => { calls.push({ pathname: new URL(url).pathname, init }); return { ok: true, json: async () => ({ session_id: 'test-chat', turns: [], drafts: [] }) }; } });
  h.context.FLDashboard.base = 'https://fl.local';
  h.context.document.documentElement = {};
  h.context.document.querySelectorAll = () => [];
  h.context.document.addEventListener = () => {};
  h.context.location.reload = () => {};
  h.run('fl-assistant.js'); await settle();
  assert.equal(calls[0].pathname, savedChat ? '/assistant/resume' : '/assistant/session');
  assert.equal(calls[0].init.headers['X-FL-Background'], '1');
  assert.equal(calls[0].init.headers['X-API-Key'], pilot.token);
  await h.get('new-chat').fire('click'); await settle();
  assert.equal(calls.at(-1).pathname, '/assistant/session');
  assert.equal(calls.at(-1).init.headers['X-FL-Background'], undefined);
});

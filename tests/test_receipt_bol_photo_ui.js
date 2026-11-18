// Receipt page: the "BOL photo" button for photos the receiving Form holds in Google Drive
// (receiving_form.py, migration 095). The photo is streamed through the API for a named person.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };
function element(tag = 'div') {
  return { tag, children: [], listeners: {}, dataset: {}, style: {}, value: '', hidden: false, open: false,
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); },
    showModal() { this.open = true; }, close() { this.open = false; (this.listeners.close || []).forEach(f => f()); },
    remove() { this.removed = true; }, scrollIntoView() {} };
}
const descendants = node => [node, ...node.children.flatMap(descendants)];

function harness({ receipt, photos, photoType = 'image/jpeg' }) {
  const els = new Map(), calls = [], opened = [], revoked = [], failures = [];
  const get = id => { if (!els.has(id)) els.set(id, element(id)); return els.get(id); };
  const body = element('body');
  const ui = { bind() {}, session: { token: 'fls_test', actor: { id: 7, name: 'Arturo' } }, lang: 'en',
    t: en => en,
    el: (tag, value, parent) => { const node = element(tag); if (value !== undefined) node.textContent = String(value); parent?.append(node); return node; },
    request: async (pathname, requestBody, options = {}) => {
      calls.push({ pathname, options });
      if (pathname.endsWith('/bol-photos')) { if (photos instanceof Error) throw photos; return { receipt_number: receipt.receipt_number, photos }; }
      if (pathname.includes('/bol-photos/')) return { type: photoType, size: 9 };
      return receipt;
    },
    receiveLotHtml: () => '', receiptData: value => value, draftHtml: () => '', receiptState: () => ({ state: 'recorded', tone: 'good' }),
    voidChoices: () => [], notice() {}, failure(error) { failures.push(error); }, relabel() {}, translate() {}, text: value => String(value) };
  const context = vm.createContext({ setTimeout: () => 0, clearTimeout() {}, URLSearchParams,
    URL: { createObjectURL: () => 'blob:photo', revokeObjectURL: url => revoked.push(url) },
    FLLedger: ui, FLReceipts: { page: name => name, link: name => name }, FLDashboard: { writesEnabled: true },
    location: { search: '?number=' + receipt.receipt_number },
    document: { getElementById: get, querySelector: () => null, body }, console });
  context.window = context;
  context.open = (url, target, features) => opened.push({ url, target, features });
  context.addEventListener = () => {};
  vm.runInContext(fs.readFileSync(path.join(ROOT, 'dashboard', 'receipt.js'), 'utf8'), context, { filename: 'receipt.js' });
  const buttons = () => descendants(get('content')).filter(node => node.tag === 'button' && node.dataset.bolPhoto);
  return { calls, opened, revoked, failures, body, buttons, get };
}

const RECEIVE = { receipt_number: 'RCV-261112-001', action: 'receive', status: 'committed', transactions: [{ id: 5 }] };

test('a receive receipt with a linked photo gets a "BOL photo" button that streams it through the API', async () => {
  const h = harness({ receipt: RECEIVE, photos: [{ photo_id: 41, form_row_number: 152, bol_number: '632054' }] });
  await settle();
  const asked = h.calls.find(call => call.pathname.endsWith('/bol-photos'));
  assert.equal(asked.pathname, '/receipts/RCV-261112-001/bol-photos');
  assert.deepEqual({ read: asked.options.read, background: asked.options.background }, { read: true, background: true });
  const [button] = h.buttons();
  assert.equal(button.textContent, 'BOL photo');
  assert.ok(descendants(h.get('content')).some(node => node.tag === 'h3' && node.textContent === 'From the receiving form'));
  await button.onclick();
  const file = h.calls.at(-1);
  assert.equal(file.pathname, '/receipts/RCV-261112-001/bol-photos/41/file');
  assert.deepEqual({ read: file.options.read, binary: file.options.binary }, { read: true, binary: true });
  const dialog = h.body.children.find(node => node.tag === 'dialog');
  assert.equal(dialog.open, true);
  assert.equal(dialog.children.find(node => node.tag === 'img').src, 'blob:photo');
  dialog.close();                                   // nothing is kept: the picture's address is released
  assert.deepEqual(h.revoked, ['blob:photo']);
  assert.equal(button.disabled, false);
  assert.deepEqual(h.failures, []);
});

test('two photos are numbered; a PDF opens in its own tab instead of the picture box', async () => {
  const h = harness({ receipt: RECEIVE, photos: [{ photo_id: 41 }, { photo_id: 42 }], photoType: 'application/pdf' });
  await settle();
  assert.deepEqual(h.buttons().map(button => button.textContent), ['BOL photo 1', 'BOL photo 2']);
  await h.buttons()[1].onclick();
  assert.equal(h.calls.at(-1).pathname, '/receipts/RCV-261112-001/bol-photos/42/file');
  assert.deepEqual(h.opened, [{ url: 'blob:photo', target: '_blank', features: 'noopener' }]);
  assert.equal(h.body.children.length, 0);
});

test('no button when nothing is linked, when the request is refused, or on a receipt that is not a receive', async () => {
  for (const photos of [[], Object.assign(Error('refused'), { status: 403 }), [{ photo_id: 'not-a-number' }]]) {
    const h = harness({ receipt: RECEIVE, photos });
    await settle();
    assert.equal(h.buttons().length, 0);
    assert.deepEqual(h.failures, []);              // a refusal is quiet: the receipt itself still shows
  }
  const pack = harness({ receipt: { ...RECEIVE, receipt_number: 'PK-261112-001', action: 'pack' }, photos: [{ photo_id: 41 }] });
  await settle();
  assert.equal(pack.buttons().length, 0);
  assert.equal(pack.calls.some(call => call.pathname.includes('bol-photos')), false);
});

test('the page never asks Drive directly and never shows a Drive address', () => {
  const source = fs.readFileSync(path.join(ROOT, 'dashboard', 'receipt.js'), 'utf8');
  assert.doesNotMatch(source, /drive\.google\.com|googleapis\.com/);
});

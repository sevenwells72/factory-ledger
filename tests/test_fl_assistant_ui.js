const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ui = require('../dashboard/fl-assistant.js');

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
async function resumedCard(card, language = 'en', savedDrafts = []) {
  const elements = new Map(), calls = [], bodies = [];
  const element = tag => ({ tag, children: [], dataset: {}, listeners: {}, value: '', textContent: '', className: '',
    append(...children) { this.children.push(...children); }, replaceChildren(...children) { this.children = children; },
    setAttribute(name, value) { this[name] = value; }, addEventListener(name, callback) { this.listeners[name] = callback; },
    focus() {}, remove() {} });
  const get = id => { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); };
  const storage = new Map([['fl-assistant-language', language], ['fl-assistant-chat:7', 'saved-chat']]);
  const store = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) };
  const context = vm.createContext({ URL, crypto: require('node:crypto').webcrypto, AbortController, setTimeout, clearTimeout, console, navigator: {},
    localStorage: store, sessionStorage: store, location: { origin: 'https://fl.local' },
    document: { getElementById: get, createElement: element, documentElement: {}, addEventListener() {},
      querySelectorAll: selector => selector === '[data-en]'
        ? [...elements.values()].flatMap(descendants).filter(node => node.dataset.en) : [] },
    FLKeySessions: { create: () => ({ enabled: true, actor: { id: 7, name: 'Arturo', role: 'floor' },
      token: 'fixture-session', init: async () => true }) },
    fetch: async (url, options) => {
      calls.push(new URL(url).pathname);
      if (new URL(url).pathname === '/assistant/turn') { bodies.push(JSON.parse(options.body)); return { ok: true, json: async () => ({ cards: [{ kind: 'missing_facts', fields: ['cases'] }] }) }; }
      assert.equal(new URL(url).pathname, '/assistant/resume');
      return { ok: true, json: async () => ({ turns: [{ cards: [card] }], drafts: savedDrafts }) };
    }, addEventListener() {} });
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/assistant-display.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/fl-assistant.js'), 'utf8'), context);
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  const rendered = descendants(get('messages')).find(node => node.tag === 'article');
  assert.ok(rendered, get('notice').textContent);
  return { get, card: rendered, calls, bodies };
}

for (const language of ['en', 'es']) test(`missing facts card lists plain missing details neutrally in ${language}`, async () => {
  const h = await resumedCard({ kind: 'missing_facts', fields: ['cases', 'case_size_lb', 'bol_reference'] }, language);
  const nodes = descendants(h.card), text = nodes.map(node => node.textContent).join(' ');
  const expected = language === 'en'
    ? ['Please add the missing details', 'number of cases', 'case size (lb)', 'BOL number']
    : ['Agrega los datos que faltan', 'número de cajas', 'peso por caja (lb)', 'número de BOL'];
  for (const phrase of expected) assert.ok(text.includes(phrase), `${phrase}: ${text}`);
  assert.equal(nodes.some(node => /error-box|receipt/.test(node.className)), false);
  assert.equal(nodes.some(node => node.tag === 'button'), false);
  assert.doesNotMatch(text, /case_size_lb|bol_reference|Field required|VALIDATION_ERROR|NOT recorded|NO registrado|didn’t understand|No entendí/);
  await h.get('language').listeners.click();
  const switched = nodes.map(node => node.textContent).join(' ');
  assert.ok(switched.includes(language === 'en' ? 'número de cajas' : 'number of cases'));
  assert.deepEqual(h.calls, ['/assistant/resume']);
});

for (const language of ['en', 'es']) test(`missing correction reason keeps reason buttons in ${language}`, async () => {
  const h = await resumedCard({ kind: 'refusal', status: 422,
    result: { detail: [{ loc: ['body', 'reason_code'], msg: 'Field required', type: 'missing' }] },
    reasons: [{ code: 'damaged', label_en: 'Damaged', label_es: 'Dañado' }] }, language);
  const nodes = descendants(h.card), choices = nodes.filter(node => node.tag === 'button');
  assert.equal(choices.length, 1);
  assert.equal(choices[0].textContent, language === 'en' ? 'Damaged' : 'Dañado');
  assert.equal(nodes.some(node => /error-box|receipt/.test(node.className)), false);
  assert.doesNotMatch(nodes.map(node => node.textContent).join(' '), /Field required|reason_code|didn’t understand|No entendí/);
  await choices[0].listeners.click();
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.get('message').value, '');
  assert.equal(h.bodies.length, 1);
  assert.equal(h.bodies[0].text, language === 'en' ? 'The correction reason is damaged.' : 'El motivo de corrección es damaged.');
  assert.deepEqual(h.calls, ['/assistant/resume', '/assistant/turn']);
});
test('only a complete real receipt shape may become green', () => {
  for (const value of [null, {}, { receipt_number: 'recorded' }, { receipt_number: 'MK-261008-001', success: false }, { text: 'RCV-261008-001' }]) assert.equal(ui.validReceipt(value), false);
  assert.equal(ui.validReceipt({ receipt_number: 'RCV-261008-001', success: true }), true);
});
test('A5 recoverable lot confirmation blockers allow manual evidence; other blockers stay blocked', () => {
  assert.equal(ui.canRecord({ can_commit: true }), true);
  assert.equal(ui.canRecord({ can_commit: false, blockers: [{ code: 'LOT_NOT_CONFIRMED' }] }), true);
  assert.equal(ui.canRecord({ can_commit: false, blockers: [{ code: 'LOT_NOT_CONFIRMED' }, { code: 'INSUFFICIENT_STOCK' }] }), false);
  assert.equal(ui.canRecord({ can_commit: false, blockers: [] }), false);
});
test('permission refusals show the plain FL explanation', () => {
  assert.equal(ui.errorText({ result: { detail: { error_code: 'ROLE_NOT_ALLOWED', action: 'make', role: 'office', message: 'Not allowed' } } }), 'Not allowed');
  const validation = ui.errorText({ detail: [{ loc: ['body', 'cases'], msg: 'Field required' }] });
  assert.equal(validation, ui.neutralMessages.misunderstood.join(' / '));
  assert.ok(!validation.includes('Field required') && !validation.includes('cases:'));
});

test('neutral outcomes have fixed English and Spanish text with no receipt assertion', () => {
  for (const [kind, messages] of Object.entries(ui.neutralMessages)) {
    assert.equal(messages.length, 2, kind);
    assert.notEqual(messages[0], messages[1], kind);
    assert.ok(messages.every(text => !/NOT recorded|NO registrado|Field required|VALIDATION_ERROR/.test(text)), kind);
    assert.equal(ui.validReceipt({ kind }), false);
  }
  assert.ok(ui.neutralMessages.unsupported_request.every(text => text.includes('ChatGPT')));
});

test('A3b hold is an explicit pending outcome and cannot become a receipt', () => {
  const held = { held: true, status: 'awaiting_approval', exception_id: 17 };
  assert.equal(ui.awaitingApproval(held), true);
  assert.equal(ui.validReceipt(held), false);
  for (const value of [null, {}, { held: true }, { status: 'awaiting_approval' }, { held: false, status: 'awaiting_approval' }]) assert.equal(ui.awaitingApproval(value), false);
});

test('A6 SHP receipts are real receipts while BOL blockers always prevent Record', () => {
  assert.equal(ui.validReceipt({ success: true, receipt_number: 'SHP-261013-001', shipment_id: 4 }), true);
  assert.equal(ui.validReceipt({ success: false, receipt_number: 'SHP-261013-001' }), false);
  for (const can_commit of [false, true]) {
    assert.equal(ui.canRecord({ can_commit, blockers: [{ code: 'BOL_REQUIRED' }] }), false);
    assert.equal(ui.canRecord({ can_commit, blockers: [{ code: 'LOT_NOT_CONFIRMED' }, { code: 'BOL_REQUIRED' }] }), false);
  }
});
test('A6 private evidence needs a positive integer ID and an explicit evidence kind', () => {
  for (const kind of ui.evidenceKinds) assert.equal(ui.validAttachment({ id: 7, kind }), true);
  for (const value of [null, {}, { id: '7', kind: 'bol' }, { id: '7e3381ea-acfb-4cd7-92c0-4ceda5d7c2e3', kind: 'bol' },
    { id: 0, kind: 'bol' }, { id: 7.5, kind: 'bol' }, { id: 7 }, { id: 7, kind: 'public_photo' }]) assert.equal(ui.validAttachment(value), false);
});

for (const language of ['en', 'es']) for (const partial of [false, true]) for (const recorded of [false, true]) {
  test(`automatic oat replacement is plain ${language} in Assistant ${recorded ? 'receipt' : 'draft'}, ${partial ? 'partial' : 'no'} bran`, async () => {
    const substitution = {
      automatic: true, quantity_lb: partial ? 18 : 28,
      message_en: partial ? `Only 10 lb oat bran in stock — ${recorded ? 'used' : 'will use'} 18 lb extra oats instead.`
        : `No oat bran in stock — ${recorded ? 'used' : 'will use'} 28 lb extra oats instead.`,
      message_es: partial ? `Solo hay 10 lb de salvado de avena — ${recorded ? 'se usaron' : 'se usarán'} 18 lb de avena extra.`
        : `No hay salvado de avena — ${recorded ? 'se usaron' : 'se usarán'} 28 lb de avena extra.`
    };
    const card = { kind: 'draft', id: 'oatbran-test', prepared: { action: 'make', actor: { name: 'Arturo' },
      can_commit: true, blockers: [], warnings: [], draft: { product_name: 'SS #1', auto_substitutions: [substitution] } } };
    const saved = recorded ? [{ id: card.id, status: 'committed', card,
      result: { success: true, receipt_number: 'MK-261015-001', auto_substitutions: [substitution] } }] : [];
    const h = await resumedCard(card, language, saved);
    const notices = descendants(h.card).filter(node => node.className === 'auto-substitution');
    assert.equal(notices.length, 1, 'a visible notice is required outside FL details');
    assert.equal(notices[0].textContent, substitution[`message_${language}`]);
    assert.equal(notices[0].role, 'status');
    await h.get('language').listeners.click();
    assert.equal(notices[0].textContent, substitution[language === 'en' ? 'message_es' : 'message_en']);
    assert.equal(descendants(notices[0]).some(node => node.tag === 'input'), false, 'no approval checkbox');
    assert.deepEqual(h.calls, ['/assistant/resume']);
  });
}

for (const language of ['en', 'es']) test(`blocked receive becomes a clear refusal with no draft controls in ${language}`, async () => {
  const h = await resumedCard({kind:'draft',id:'refused',prepared:{action:'receive',can_commit:false,draft:{happened_at:'2026-10-15T15:19:00Z'},blockers:[{code:'EXTRA_KOSHER_RECEIVE_BLOCKED',message:'Extra-kosher batches are made, never received.',message_es:'Los lotes extra kosher se producen; no se reciben.'}]}},language);
  const nodes=descendants(h.card), text=nodes.map(n=>n.textContent).join(' ');
  assert.match(h.card.className,/refusal/);
  assert.match(text,language==='es'?/rechazada/:/refused/);
  assert.equal(nodes.some(n=>n.tag==='button'),false);
  assert.doesNotMatch(text,/DRAFT|BORRADOR|EXTRA_KOSHER_RECEIVE_BLOCKED/);
});
test('void hand-off only recognizes explicit void requests with a receipt', () => {
  assert.equal(ui.voidTarget('Void receipt ADJ-261015-242, data-entry error'),'ADJ-261015-242');
  assert.equal(ui.voidTarget('Anular RCV-261015-001'),'RCV-261015-001');
  assert.equal(ui.voidTarget('Look up ADJ-261015-242'),null);
  assert.equal(ui.voidTarget('Void all entries'),null);
});

test('plain Today questions go directly to ledger reads without interpreting write messages', () => {
  for (const text of ['What did I enter today?', 'Show all today’s entries', "today's entries", 'List entries for today', '¿Qué registré hoy?', 'Mostrar todas las entradas de hoy']) assert.equal(ui.isTodayQuery(text), true, text);
  for (const text of ['Received 50 lb today', 'What inventory do we have today?', 'Adjust today’s entries by 5 lb', 'Void ADJ-261015-001']) assert.equal(ui.isTodayQuery(text), false, text);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ui = require('../dashboard/fl-assistant.js');

function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
async function resumedCard(card, language = 'en', options = {}) {
  const elements = new Map(), calls = [], bodies = [], requests = [];
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
    FLKeySessions: { create: () => ({ enabled: true, actor: { id: 7, name: 'Arturo', role: options.role || 'floor' },
      token: 'fixture-session', init: async () => true }) },
    fetch: async (url, fetchOptions) => {
      calls.push(new URL(url).pathname);
      const requestPath = new URL(url).pathname;
      const body = JSON.parse(fetchOptions.body || '{}');
      requests.push({ path: requestPath, body });
      if (requestPath === '/assistant/turn') bodies.push(body);
      const response = requestPath === '/assistant/resume'
        ? { turns: [{ text: options.turnText || '', cards: Array.isArray(card) ? card : [card] }], drafts: options.savedDrafts || [] }
        : options.onRequest
          ? await options.onRequest(requestPath, body)
          : requestPath === '/assistant/turn'
            ? { cards: [{ kind: 'missing_facts', fields: ['cases'] }] }
            : undefined;
      assert.ok(response, `Unexpected request: ${url}`);
      return { ok: true, json: async () => response };
    }, addEventListener() {} });
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/assistant-display.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../dashboard/fl-assistant.js'), 'utf8'), context);
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  const rendered = descendants(get('messages')).find(node => node.tag === 'article');
  assert.ok(rendered, get('notice').textContent);
  return { get, card: rendered, calls, bodies, requests };
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
  assert.equal(ui.neutralMessages.unsupported_request[0], 'I can only record or look up Factory Ledger items.');
  assert.ok(ui.neutralMessages.unsupported_request.every(text => !text.includes('ChatGPT')));
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
    const h = await resumedCard(card, language, { savedDrafts: saved });
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

for (const language of ['en', 'es']) test(`customer draft waits for Record and CUS receipt continues only on click in ${language}`, async () => {
  const prepared = { action: 'create_customer', can_commit: true, blockers: [], actor: { name: 'Office' },
    draft: { summary: 'Create customer New Market', summary_es: 'Crear cliente New Market',
      customer: { name: 'New Market', contact_name: 'A Person', email: 'orders@example.test', phone: '555-1234', address: '10 Main St' } } };
  const receipt = { success: true, receipt_number: 'CUS-261015-001', customer_id: 42, customer: { id: 42, name: 'New Market' } };
  const h = await resumedCard({ kind: 'draft', id: 'customer-draft', prepared }, language, { role: 'office',
    onRequest: async (path, body) => {
      if (path === '/assistant/record') { assert.equal(body.draft_id, 'customer-draft'); return { kind: 'receipt', result: receipt }; }
      assert.equal(path, '/assistant/turn');
      assert.match(body.text, /CUS-261015-001/);
      return { cards: [{ kind: 'missing_facts', fields: ['lines'] }] };
    } });
  let nodes = descendants(h.card), text = nodes.map(node => node.textContent).join(' ');
  for (const phrase of ['New Market', 'A Person', 'orders@example.test', '555-1234', '10 Main St']) assert.ok(text.includes(phrase), phrase);
  assert.ok(text.includes(language === 'en' ? 'Create customer' : 'Crear cliente'));
  assert.deepEqual(h.calls, ['/assistant/resume']);
  assert.equal(h.card.className, 'card');
  const record = nodes.find(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Record' : 'Registrar'));
  assert.equal(record.disabled, false);
  await record.listeners.click();
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(h.calls, ['/assistant/resume', '/assistant/record']);
  assert.equal(h.card.className, 'card receipt');
  nodes = descendants(h.card);
  assert.ok(nodes.some(node => node.textContent === 'CUS-261015-001'));
  const next = nodes.find(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Continue with an order' : 'Continuar con un pedido'));
  assert.ok(next);
  await next.listeners.click();
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(h.calls, ['/assistant/resume', '/assistant/record', '/assistant/turn']);
  assert.ok(h.calls.every(path => path.startsWith('/assistant/')));
});

for (const role of ['office', 'owner', 'floor']) test(`no-match customer offers New customer only for office/owner: ${role}`, async () => {
  const h = await resumedCard({ kind: 'choices', id: 'choice', resolution_kind: 'customer', query: 'New Market LLC',
    result: { outcome: 'none', candidates: [] } }, 'en', { role,
    onRequest: async (path, body) => {
      assert.equal(path, '/assistant/turn');
      assert.equal(body.text, 'Create a new customer named New Market LLC first, then continue the order.');
      return { cards: [{ kind: 'missing_facts', fields: ['name'] }] };
    } });
  const create = descendants(h.card).find(node => node.tag === 'button' && node.textContent === "Add 'New Market LLC' as a new customer");
  assert.equal(Boolean(create), role !== 'floor');
  if (create) {
    await create.listeners.click();
    for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(h.calls, ['/assistant/resume', '/assistant/turn']);
  }
});

test('customer duplicate prevents Record and offers the existing active customer', async () => {
  const h = await resumedCard({ kind: 'draft', id: 'duplicate', prepared: { action: 'create_customer', can_commit: false,
    blockers: [{ code: 'CUSTOMER_ALREADY_EXISTS', message: 'Use existing customer New Market', message_es: 'Use el cliente existente New Market', existing_customer: { id: 42, name: 'New Market', active: true } }],
    draft: { customer: { name: 'new market' } } } }, 'en', { role: 'office' });
  const buttons = descendants(h.card).filter(node => node.tag === 'button');
  assert.equal(buttons.find(node => node.textContent === 'Record').disabled, true);
  assert.ok(buttons.some(node => node.textContent === 'Use existing customer: New Market'));
  await h.get('language').listeners.click();
  assert.ok(descendants(h.card).some(node => node.textContent === 'Use el cliente existente New Market'));
  assert.deepEqual(h.calls, ['/assistant/resume']);
});

test('CUS and ORD receipt numbers are accepted; a failed customer result never is', () => {
  for (const prefix of ['CUS', 'ORD']) {
    assert.equal(ui.validReceipt({ success: true, receipt_number: `${prefix}-261015-001` }), true);
    assert.equal(ui.validReceipt({ success: false, receipt_number: `${prefix}-261015-001` }), false);
  }
});


test('order draft shows physical quantity, pounds, PO and service charge details before Record', async () => {
  const h = await resumedCard({ kind: 'draft', id: 'order', prepared: { action: 'create_order', can_commit: true, blockers: [],
    draft: { summary: 'Create order for New Market: 2 lines, 50 lb, PO ABC-42', customer: 'New Market', customer_po: 'ABC-42',
      requested_ship_date: '2026-10-20', lines: [
        { product: 'Classic Granola', quantity: 2, unit: 'cases', quantity_lb: 50, unit_price: 30, amount: 60 },
        { product: 'Pallet Charge', quantity: 3, unit: 'each', quantity_lb: 0, unit_price: 12.5, amount: 37.5, is_service: true }] } } }, 'en', { role: 'office' });
  const nodes = descendants(h.card), text = nodes.map(node => node.textContent).join(' ');
  for (const phrase of ['Create order', 'New Market', 'ABC-42', '2026-10-20', 'Classic Granola', '2 cases', '50', 'Pallet Charge', '3 each', '12.5', '37.5']) assert.ok(text.includes(phrase), phrase);
  assert.equal(nodes.find(node => node.tag === 'button' && node.textContent === 'Record').disabled, false);
  assert.deepEqual(h.calls, ['/assistant/resume']);
});

for (const outcome of ['ambiguous', 'match']) for (const language of ['en', 'es']) {
  test(`customer suggestions retain explicit add-new choice: ${outcome} ${language}`, async () => {
    const name = 'Sunrise Bakery';
    const h = await resumedCard({ kind: 'choices', id: 'similar-choice', resolution_kind: 'customer', query: name,
      result: { outcome, offset: 0, candidates: [{ id: 42, name: 'Sunshine Bakery' }] } }, language, { role: 'office',
      onRequest: async (path, body) => {
        assert.equal(path, '/assistant/turn');
        assert.equal(body.text, language === 'en' ? `Create a new customer named ${name} first, then continue the order.`
          : `Crea primero un cliente nuevo llamado ${name} y luego continúa el pedido.`);
        return { cards: [{ kind: 'missing_facts', fields: ['name'] }] };
      } });
    const buttons = descendants(h.card).filter(node => node.tag === 'button');
    assert.ok(buttons.some(node => node.textContent === 'Sunshine Bakery'));
    const add = buttons.find(node => node.textContent === (language === 'en'
      ? `Add '${name}' as a new customer` : `Agregar '${name}' como cliente nuevo`));
    assert.ok(add, 'The typed name must remain available alongside the similar customer');
    assert.deepEqual(h.calls, ['/assistant/resume']);
    await add.listeners.click();
    for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(h.calls, ['/assistant/resume', '/assistant/turn']);
  });
}

test('customer add-new choice appears only on the first result page', async () => {
  const h = await resumedCard({ kind: 'choices', id: 'later-choice', resolution_kind: 'customer', query: 'Sunrise Bakery',
    result: { outcome: 'none', offset: 8, candidates: [] } }, 'en', { role: 'office' });
  assert.equal(descendants(h.card).some(node => node.tag === 'button' && /new customer/i.test(node.textContent)), false);
});

test('supplier display merges only near-duplicate labels and keeps the best six original records', () => {
  const rows = [{id: 1, name: 'Grain Supply, LLC'}, {id: 2, name: 'Grain Supply'}, {id: 3, name: 'A & B Co.'}, {id: 4, name: 'A and B'},
    ...Array.from({length: 64}, (_, i) => ({id: i + 10, name: 'Supplier ' + i}))];
  const selected = ui.supplierChoices(rows);
  assert.equal(selected.length, 6);
  assert.deepEqual(selected.map(row => row.id), [1, 3, 10, 11, 12, 13]);
  assert.equal(rows.length, 68, 'display grouping never mutates the source records');
});

for (const language of ['en', 'es']) {
  test(`Round B missing receive retains weight and shows six searchable suppliers in ${language}`, async () => {
    const h = await resumedCard({kind: 'missing_facts', language, fields: ['supplier_id', 'cases'], choice_id: 'supplier-choice', supplier_search: true,
      summary: language === 'en' ? 'I kept the quantity: 10 lb. Please add the missing details.' : 'Conservo la cantidad: 10 lb. Agrega los datos que faltan.',
      supplier_choices: Array.from({length: 66}, (_, i) => ({id: i + 1, name: 'Supplier ' + i}))}, 'en');
    const nodes = descendants(h.card), copy = nodes.map(node => node.textContent).join(' ');
    assert.match(copy, /10 lb/);
    assert.match(copy, language === 'en' ? /supplier/ : /proveedor/);
    assert.equal(nodes.filter(node => node.className === 'choice-button').length, 6);
    assert.equal(h.get('language').textContent, language === 'es' ? 'EN' : 'ES', 'reply language applies without pressing ES');
    const form = nodes.find(node => node.className === 'supplier-search');
    const input = descendants(form).find(node => node.tag === 'input');
    input.value = 'Grain'; form.listeners.submit({preventDefault() {}});
    for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
    assert.equal(h.bodies[0].text, 'supplier: Grain');
    const bubble = descendants(h.get('messages')).find(node => node.className === 'message user-message');
    assert.equal(bubble.textContent, language === 'es' ? 'Buscar proveedor: Grain' : 'Supplier search: Grain');
    assert.equal(h.bodies[0].choice_id, undefined, 'search narrows names without selecting a supplier');
  });

  test(`Round B lot misses and scope responses stay plain in ${language}`, async () => {
    const missing = language === 'en' ? 'No lot called MISSING-123.' : 'No hay un lote llamado MISSING-123.';
    const h = await resumedCard({kind: 'lookup_failed', language, error_code: 'LOT_NOT_FOUND', message: missing}, 'en');
    assert.equal(descendants(h.card).filter(node => node.tag === 'p')[0].textContent, missing);
    assert.doesNotMatch(descendants(h.card).map(node => node.textContent).join(' '), /try again|Inténtalo|ChatGPT/);
    const scope = await resumedCard({kind: 'unsupported_request', language}, 'en');
    assert.equal(descendants(scope.card).find(node => node.tag === 'p').textContent, ui.neutralMessages.unsupported_request[language === 'es' ? 1 : 0]);
  });

  test(`Round B make has one instruction and one Different lot button per ingredient in ${language}`, async () => {
    const lots = [{product_id: 1, product_name: 'Seeds', lot_id: 11, lot_code: 'SEED-ABCD', quantity_lb: 2, confirmed: false},
      {product_id: 1, product_name: 'Seeds', lot_id: 12, lot_code: 'SEED-EFGH', quantity_lb: 2, confirmed: false},
      {product_id: 2, product_name: 'Oats', lot_id: 21, lot_code: 'OAT-IJKL', quantity_lb: 6, confirmed: false}];
    let recorded;
    const h = await resumedCard({kind: 'draft', id: 'make', language, prepared: {action: 'make', actor: {name: 'Arturo'}, can_commit: false,
      blockers: [{code: 'LOT_NOT_CONFIRMED', message: 'Type the last 4 characters from the tag on each bag you used.'}],
      draft: {product_name: 'Classic', total_lb: 10, input_plan: lots, lot_confirmation_group: {
        instruction_en: 'Type the last 4 characters from the tag on each bag you used.', instruction_es: 'Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.',
        ingredients: [{product_id: 1, product_name: 'Seeds', needed_lb: 4}, {product_id: 2, product_name: 'Oats', needed_lb: 6}]}}}}, 'en', {onRequest: async (path, body) => {
          assert.equal(path, '/assistant/record'); recorded = body;
          return {kind: 'receipt', result: {success: true, receipt_number: 'MK-261015-999'}};
        }});
    const nodes = descendants(h.card), instruction = nodes.filter(node => node.className === 'lot-instruction');
    assert.equal(instruction.length, 1);
    assert.equal(nodes.filter(node => node.textContent === instruction[0].textContent).length, 1, 'no repeated blocker or per-lot instruction');
    const different = nodes.filter(node => node.className === 'different-lot');
    assert.equal(different.length, 2);
    assert.ok(different.every(node => node.textContent === (language === 'en' ? 'Different lot' : 'Otro lote')));
    const inputs = nodes.filter(node => node.tag === 'input' && node.type === 'text');
    assert.equal(inputs.length, 3);
    const record = nodes.find(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Record' : 'Registrar'));
    assert.equal(record.disabled, true);
    inputs[0].value = 'ABCD'; inputs[0].listeners.input(); assert.equal(record.disabled, true);
    different[0].listeners.click();
    assert.equal(inputs[0].value, '', 'choosing Different lot clears prior evidence');
    assert.equal(record.disabled, true, 'Different lot never skips physical evidence');
    const suggestions = nodes.filter(node => node.className === 'suggested-lot-code');
    const methods = nodes.filter(node => node.tag === 'select');
    assert.deepEqual(suggestions.map(node => !!node.hidden), [true, true, false], 'hide only this ingredient suggestions');
    assert.deepEqual(methods.map(node => !!node.hidden), [true, true, false], 'hide only this ingredient method pickers');
    different[0].listeners.click();
    assert.ok(suggestions.every(node => !node.hidden)); assert.ok(methods.every(node => !node.hidden));
    assert.equal(inputs[0].value, '', 'returning to the suggested lot clears physical evidence');
    different[0].listeners.click();
    for (const [i, input] of inputs.entries()) { input.value = ['REAL-0001', 'REAL-0002', 'IJKL'][i]; input.listeners.input(); }
    assert.equal(record.disabled, false);
    record.listeners.click();
    for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(recorded.lot_confirmations.map(c => [c.lot_id, c.method, c.value]), [[11, 'different_lot', 'REAL-0001'], [12, 'different_lot', 'REAL-0002'], [21, 'last4', 'IJKL']]);
  });

  test(`Round B receipt preserves visible physical lot follow-up in ${language}`, async () => {
    const card = {kind: 'draft', id: 'receipt-lots', prepared: {action: 'make', actor: {name: 'Arturo'}, can_commit: true, draft: {product_name: 'Classic'}}};
    const result = {success: true, receipt_number: 'MK-261015-888', lot_exceptions: [{product_name: 'Seeds', physical_tag: 'BAG-4321', quantity_lb: 4, exception_id: 45}]};
    const h = await resumedCard(card, language, {savedDrafts: [{id: card.id, card, status: 'committed', result}]});
    const warning = descendants(h.card).find(node => node.className === 'warning lot-exception');
    assert.ok(warning);
    const copy = descendants(warning).map(node => node.textContent).join(' ');
    assert.match(copy, /BAG-4321/);
    assert.match(copy, language === 'en' ? /awaiting follow-up/ : /pendiente de seguimiento/);
    assert.match(copy, language === 'en' ? /not confirmed or used/ : /no se confirmó ni se usó/);
  });
}


for (const language of ['en', 'es']) test(`a corrected receive replaces the stale visible draft and its Record control in ${language}`, async () => {
  const prepared = total => ({action: 'receive', actor: {name: 'Arturo'}, can_commit: true, draft: {product_name: 'Seeds', total_lb: total}});
  const h = await resumedCard([{kind: 'draft', id: 'old-weight', language, prepared: prepared(10)},
    {kind: 'draft', id: 'new-weight', language, supersedes_draft_id: 'old-weight', prepared: prepared(15)}], 'en');
  const cards = descendants(h.get('messages')).filter(node => node.tag === 'article');
  assert.equal(cards.length, 2);
  const oldNodes = descendants(cards[0]), newNodes = descendants(cards[1]);
  assert.match(oldNodes.map(node => node.textContent).join(' '), language === 'en' ? /Replaced by the corrected draft below/ : /Reemplazado por el borrador corregido abajo/);
  assert.equal(oldNodes.some(node => node.tag === 'button'), false, 'the old quantity cannot still offer Record');
  assert.match(newNodes.map(node => node.textContent).join(' '), /15 lb/);
  assert.equal(newNodes.filter(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Record' : 'Registrar')).length, 1);
  assert.deepEqual(h.calls, ['/assistant/resume'], 'restoring corrected drafts never records either entry');
});


test('legacy supplier cards merge Baker, Bakery and Bakery Supply display names without changing IDs', () => {
  const candidates = [{id: 8, name: 'A1 Baker'}, {id: 9, name: 'A1 Bakery, LLC'}, {id: 10, name: 'A1 Bakery Supply'}, {id: 11, name: 'Other Grain Supply'}];
  assert.deepEqual(ui.supplierChoices(candidates), [candidates[0], candidates[3]]);
  assert.deepEqual(candidates.map(row => row.id), [8, 9, 10, 11]);
});


for (const language of ['en', 'es']) test(`an unsuccessful corrected receive retires the wrong draft without claiming success in ${language}`, async () => {
  const h = await resumedCard([{kind: 'draft', id: 'wrong', language, prepared: {action: 'receive', actor: {name: 'Arturo'}, can_commit: true, draft: {product_name: 'Seeds', total_lb: 10}}},
    {kind: 'missing_facts', language, supersedes_draft_id: 'wrong', fields: ['case_size_lb']}], 'en');
  const cards = descendants(h.get('messages')).filter(node => node.tag === 'article');
  const previous = descendants(cards[0]);
  assert.match(previous.map(node => node.textContent).join(' '), language === 'en' ? /Previous draft cancelled after your correction/ : /Borrador anterior cancelado después de tu corrección/);
  assert.doesNotMatch(previous.map(node => node.textContent).join(' '), /corrected draft below|borrador corregido abajo/);
  assert.equal(previous.some(node => node.tag === 'button'), false);
  assert.equal(cards.some(node => node.className === 'card receipt'), false);
});


for (const language of ['en', 'es']) for (const prefix of ['supplier', 'proveedor']) test(`saved ${prefix} searches show worker copy in ${language}`, async () => {
  const h = await resumedCard({kind: 'choices', id: 'suppliers', language, supplier_search: true, result: {candidates: []}}, 'en', {turnText: prefix + ': Supplier 57'});
  const bubble = descendants(h.get('messages')).find(node => node.className === 'message user-message');
  assert.equal(bubble.textContent, language === 'es' ? 'Buscar proveedor: Supplier 57' : 'Supplier search: Supplier 57');
  assert.deepEqual(h.calls, ['/assistant/resume']);
});

for (const language of ['en', 'es']) test(`catalog choices appear once as buttons with no model enumeration in ${language}`, async () => {
  const h = await resumedCard({kind: 'choices', id: 'voice-choice', resolution_kind: 'product', query: 'Classic number nine', result: {ask: 'Please clarify: 1) Classic #9 2) Classic Chocolate Chip #9', candidates: [{id: 1, name: 'Classic #9'}, {id: 2, name: 'Classic Chocolate Chip #9'}]}}, language);
  const nodes = descendants(h.card), prose = nodes.filter(node => node.tag !== 'button').map(node => node.textContent || '').join(' ');
  assert.match(prose, language === 'en' ? /Did you mean one of these/ : /¿Quiso decir una de estas opciones/);
  assert.doesNotMatch(prose, /Classic #9|Classic Chocolate Chip|Please clarify|Which one|¿Cuál/);
  assert.equal(nodes.filter(node => node.tag === 'button' && node.textContent === 'Classic #9').length, 1);
});

for (const language of ['en', 'es']) test(`zero catalog candidates have a useful named response in ${language}`, async () => {
  const h = await resumedCard({kind: 'choices', id: 'no-choice', resolution_kind: 'product', query: 'Invisible flour', result: {candidates: []}}, language);
  const nodes = descendants(h.card), text = nodes.map(node => node.textContent || '').join(' ');
  assert.match(text, /Invisible flour/);
  assert.match(text, language === 'en' ? /Check the name or add more details/ : /Revisa el nombre o agrega más detalles/);
  assert.doesNotMatch(text, /Which one|¿Cuál|try again|Inténtalo de nuevo/);
  assert.equal(nodes.filter(node => node.tag === 'button').length, 0);
});

for (const language of ['en', 'es']) test(`order draft omits assignment notes and unspecified midnight in ${language}`, async () => {
  const h = await resumedCard({kind: 'draft', id: 'order-wording', prepared: {action: 'create_order', can_commit: true, blockers: [], draft: {product_name: 'Rainbow Sprinkles', cases: 5, note: 'order_id, line ids and the order number are assigned at commit.', happened_at: '2026-10-15T04:00:00Z'}}}, language, {role: 'office'});
  const text = descendants(h.card).map(node => node.textContent || '').join(' ');
  assert.doesNotMatch(text, /order_id|line ids|assigned at commit|12:00 AM|Happened at|Ocurrió el/);
  assert.match(text, /Rainbow Sprinkles/);
});

for (const language of ['en', 'es']) test(`fuzzy catalog fallback identifies the unmatched name without listing button names in ${language}`, async () => {
  const h = await resumedCard({kind: 'choices', id: 'fuzzy-choice', resolution_kind: 'supplier', query: 'Green Supply', result: {outcome: 'none', ask: 'Please clarify: 1) Grain Supply', candidates: [{id:1, name:'Grain Supply'}]}}, language);
  const prose = descendants(h.card).filter(node => node.tag === 'p').map(node => node.textContent).join(' ');
  assert.match(prose, /Green Supply/); assert.doesNotMatch(prose, /Grain Supply|Please clarify/);
  assert.match(prose, language === 'en' ? /Did you mean one of these/ : /¿Quiso decir una de estas opciones/);
  await h.get('language').listeners.click();
  assert.match(descendants(h.card).filter(node => node.tag === 'p').map(node => node.textContent).join(' '), language === 'en' ? /No encontré.*¿Quiso decir una de estas opciones/ : /I couldn’t find.*Did you mean one of these/);
});

for (const language of ['en', 'es']) test(`Round D batch choices use plain prompt and Different batch in ${language}`, async () => {
  const h = await resumedCard({ kind: 'choices', id: 'bulk-batch-choice', packing_lots: true, resolution_kind: 'lot',
    result: { candidates: [{ id: 123, name: 'B26-1015-004', label: 'Batch Coconut Sweetened Flake · B26-1015-004 · 1,000 lb' }] } }, language);
  const nodes = descendants(h.card), text = nodes.map(node => node.textContent).join(' ');
  assert.match(text, language === 'en' ? /Which batch did these come from/ : /¿De qué lote a granel salieron/);
  assert.doesNotMatch(text, /source product|producto de origen|SKU/);
  const different = nodes.find(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Different batch' : 'Otro lote a granel'));
  assert.ok(different);
  await different.listeners.click();
  for (let i = 0; i < 6; i++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.bodies[0].selected_id, 'different_batch');
});
for (const language of ['en', 'es']) test(`Round D missing batch and missing order never request a product identifier in ${language}`, async () => {
  const h = await resumedCard({ kind: 'missing_facts', fields: ['source_product_id'], packing_batch_needed: true }, language);
  const text = descendants(h.card).map(node => node.textContent).join(' ');
  assert.match(text, language === 'en' ? /Which batch/ : /¿De qué lote/);
  assert.doesNotMatch(text, /source product|producto de origen|SKU/);
  const order = await resumedCard({ kind: 'choices', id: 'order-choice', resolution_kind: 'order', order_not_found: true, query: 'SO-261015-999',
    result: { candidates: [{ id: 123, name: 'SO-261015-002', label: 'SO-261015-002 · Setton Farms' }] } }, language);
  const copy = descendants(order.card).map(node => node.textContent).join(' ');
  assert.match(copy, language === 'en' ? /I couldn't find order SO-261015-999/ : /No encontré el pedido SO-261015-999/);
  assert.doesNotMatch(copy, /SKU|product number|número de producto/);
});

for (const language of ['en', 'es']) for (const [pounds, size, quantity] of [
  [50, 50, ['1 case (50 lb)', '1 caja (50 lb)']],
  [37.5, 25, ['1.5 cases (37.5 lb)', '1.5 cajas (37.5 lb)']],
  [12.5, null, ['12.5 lb', '12.5 lb']]
]) test(`Round D missing shipping quantity shows ${quantity[0]} and offers all remaining in ${language}`, async () => {
  const h = await resumedCard({kind:'missing_facts', fields:['lines'], shipping_quantity_needed:true,
    order:{order_number:'SO-261015-001', remaining_lb:pounds, items:[
      {remaining_lb:pounds, case_size_lb:size}, {remaining_lb:500, case_size_lb:5, is_service:true}
    ]}}, language);
  const nodes=descendants(h.card), text=nodes.map(node=>node.textContent).join(' ');
  assert.match(text, language==='en' ? /How much of order SO-261015-001 should we ship/ : /¿Cuánto del pedido SO-261015-001 vamos a enviar/);
  assert.ok(text.includes(quantity[language==='en'?0:1]), text);
  assert.doesNotMatch(text, /100 cases|100 cajas|500 lb/);
  assert.doesNotMatch(text, /already fully shipped|completamente enviado|SKU/);
  await h.get('language').listeners.click();
  const switched=nodes.map(node=>node.textContent).join(' ');
  assert.ok(switched.includes(quantity[language==='en'?1:0]), switched);
  await h.get('language').listeners.click();
  const all=nodes.find(node=>node.tag==='button' && node.textContent===(language==='en'?'Ship all remaining':'Enviar todo lo pendiente'));
  assert.ok(all); await all.listeners.click();
  for (let i=0;i<6;i++) await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.bodies[0].text, language==='en'?'Ship all remaining on order SO-261015-001.':'Envía todo lo pendiente del pedido SO-261015-001.');
});

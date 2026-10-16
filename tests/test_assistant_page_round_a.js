const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ui = require('../dashboard/fl-assistant.js');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'dashboard/fl-assistant.html'), 'utf8');
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const descendants = node => [node, ...node.children.flatMap(descendants)];
const text = node => descendants(node).map(child => child.textContent || '').join(' ');
const response = data => ({ ok: true, json: async () => data });
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };

function page({ cards = [], drafts = [], language = 'en', request, init, enabled = true, turnText = '' } = {}) {
  const elements = new Map(), calls = [], timers = new Map(), scheduled = [], prompts = [];
  let nextTimer = 0, now = Date.parse('2026-10-16T02:19:00Z'); // Oct 15 at the plant, Oct 16 in UTC.
  class ClockDate extends Date { constructor(...args) { super(...(args.length ? args : [now])); } static now() { return now; } }
  const element = tag => ({ tag, children: [], dataset: {}, listeners: {}, value: '', textContent: '', className: '', hidden: false,
    append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } },
    replaceChildren(...children) { this.children = []; this.append(...children); },
    setAttribute(name, value) { this[name] = String(value); }, getAttribute(name) { return this[name] ?? null; },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    focus() {}, remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); },
    requestSubmit() { this.listeners.submit?.({ preventDefault() {}, target: this }); },
    querySelector(selector) { return descendants(this).find(child => child.tag === selector); } });
  const get = id => { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); };
  for (const match of html.matchAll(/<([a-z-]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const node = get(match[3]); node.tag = match[1]; node.hidden = /(?:^|\s)hidden(?:\s|$)/.test(match[2]);
  }
  for (const prompt of ['What did I enter today?', 'Look up inventory for ']) { const node = element('button'); node.dataset.prompt = prompt; prompts.push(node); }
  get('messages').append(get('welcome'));
  const storage = new Map([['fl-assistant-language', language], ['fl-assistant-chat:7', 'saved-chat']]);
  const store = { getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) };
  const actor = { id: 7, name: 'Arturo', role: 'floor' };
  const pilot = { enabled, actor: enabled ? actor : null, token: 'fixture-session', init: () => init || Promise.resolve(enabled),
    clear() { this.actor = null; this.token = ''; this.cleared = true; } };
  const context = vm.createContext({ URL, crypto: require('node:crypto').webcrypto, AbortController, Error, Date: ClockDate, console, navigator: {},
    setTimeout(callback, delay) { const id = ++nextTimer; const timer = { id, callback, delay }; timers.set(id, timer); scheduled.push(timer); return id; },
    clearTimeout(id) { timers.delete(id); }, localStorage: store, sessionStorage: store,
    location: { origin: 'https://fl.local', reload() {} },
    document: { getElementById: get, createElement: element, documentElement: {}, body: element('body'), addEventListener() {},
      querySelectorAll: selector => selector === '[data-prompt]' ? prompts : selector === '[data-en]'
        ? [...new Set([...elements.values()].flatMap(descendants))].filter(node => node.dataset.en) : [] },
    FLKeySessions: { create: () => pilot },
    fetch: async (url, options) => {
      const call = { url: new URL(url), options, body: options.body ? JSON.parse(options.body) : null }; calls.push(call);
      if (call.url.pathname === '/assistant/resume') return response({ turns: [{ text: turnText, cards }], drafts });
      if (request) return request(call, { advance(ms) { now += ms; } });
      throw new Error('Unexpected request: ' + url);
    }, addEventListener() {} });
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(root, 'dashboard/assistant-display.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(root, 'dashboard/fl-assistant.js'), 'utf8'), context);
  return { get, calls, timers, scheduled, prompts, pilot, ready: flush, text: () => text(get('messages')),
    cards: () => descendants(get('messages')).filter(node => node.tag === 'article'),
    click(node) { node.listeners.click({ target: node }); },
    submit(value) { get('message').value = value; get('composer').requestSubmit(); },
    fire(delay) { const timer = [...timers.values()].find(item => item.delay === delay); assert.ok(timer, `missing ${delay}ms timer`); timers.delete(timer.id); timer.callback(); } };
}

const draft = (updates = {}) => ({ kind: 'draft', id: 'draft-1', prepared: { action: 'receive', actor: { name: 'Arturo' },
  can_commit: true, blockers: [], warnings: [], draft: { product_name: 'Sunflower Seeds', total_lb: 50 }, ...updates } });

function assertWorking(h, control = h.get('send')) {
  assert.equal(h.get('working-status').hidden, false);
  assert.equal(h.get('send').getAttribute('aria-busy'), 'true');
  assert.equal(h.get('send').textContent, 'Working…');
  assert.match(control.className, /is-working/);
  assert.equal(control.getAttribute('aria-busy'), 'true');
  assert.equal(control.textContent, 'Working…');
}

test('Today button reads every actor across 500/101 pages, directly, newest first on the plant date', async () => {
  const receipts = Array.from({ length: 601 }, (_, i) => ({ receipt_number: `RCV-261015-${String(i + 1).padStart(3, '0')}`,
    action: 'receive', entered_at: new Date(Date.parse('2026-10-15T15:00:00Z') + i * 1000).toISOString(),
    actor: { name: i % 2 ? 'Michael' : 'Arturo' }, draft: { product_name: 'Sunflower Seeds', total_lb: 50 } })).reverse();
  const h = page({ request: async ({ url }) => {
    assert.equal(url.pathname, '/receipts');
    const offset = Number(url.searchParams.get('offset'));
    return response({ receipts: receipts.slice(offset, offset + 500), has_more: offset === 0, next_offset: offset === 0 ? 500 : null });
  } });
  await h.ready(); h.click(h.get('today-entries')); await flush();
  const reads = h.calls.filter(call => call.url.pathname === '/receipts');
  assert.deepEqual(reads.map(call => call.url.searchParams.get('offset')), ['0', '500']);
  for (const call of reads) {
    assert.equal(call.url.searchParams.get('date'), '2026-10-15');
    assert.equal(call.url.searchParams.get('summary'), 'true');
    assert.equal(call.url.searchParams.get('limit'), '500');
    assert.equal(call.url.searchParams.has('actor'), false);
    assert.equal(call.options.method, 'GET');
  }
  assert.equal(h.calls.some(call => call.url.pathname === '/assistant/turn'), false);
  const [card] = h.cards(), rows = descendants(card).filter(node => node.className === 'today-entry');
  assert.equal(rows.length, 601);
  assert.equal(rows[0].children[0].textContent, '11:10 AM — Received 50 lb Sunflower Seeds — RCV-261015-601 — Arturo');
  assert.equal(descendants(card).some(node => node.tag === 'table'), false, 'Today entries remain plain lines');
  assert.match(text(card), /You recorded 301 entries today; the team recorded 601\./);
  assert.match(text(card), /Michael/);
  assert.equal(h.get('send').getAttribute('aria-busy'), 'false');
});

test('Today pagination shares one 15 second deadline instead of restarting it on every page', async () => {
  const h = page({ request: async ({ url }, clock) => {
    if (url.searchParams.get('offset') === '0') { clock.advance(10000); return response({ receipts: [], has_more: true, next_offset: 500 }); }
    return response({ receipts: [], has_more: false });
  } });
  await h.ready(); h.click(h.get('today-entries')); await flush();
  assert.deepEqual(h.scheduled.filter(timer => timer.delay <= 15000).map(timer => timer.delay), [15000, 5000]);
  assert.match(h.text(), /You recorded 0 entries today; the team recorded 0\./);
  assert.equal(h.timers.size, 0);
});

test('Today aborts at 15 seconds, restores controls and shows a retry action', async () => {
  const h = page({ request: ({ options }) => new Promise((resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('aborted')))) });
  await h.ready(); h.click(h.get('today-entries')); assertWorking(h);
  h.fire(15000); await flush();
  assert.match(h.text(), /Today’s entries could not load within 15 seconds\. Please try again\./);
  assert.ok(descendants(h.cards()[0]).some(node => node.tag === 'button' && node.textContent === 'Try again'));
  assert.equal(h.get('send').disabled, false);
  assert.equal(h.get('working-status').hidden, true);
});

test('invalid Today pagination cannot loop forever', async () => {
  const h = page({ request: async () => response({ receipts: [], has_more: true, next_offset: 0 }) });
  await h.ready(); h.click(h.get('today-entries')); await flush();
  assert.equal(h.calls.filter(call => call.url.pathname === '/receipts').length, 1);
  assert.match(h.text(), /FL could not load the next page of today’s entries\. Please try again\./);
  assert.doesNotMatch(h.text(), /15 seconds/);
});

for (const language of ['en', 'es']) test(`Today HTTP and connection failures retain their own plain messages in ${language}`, async () => {
  for (const [status, message] of [[403, 'You do not have permission to view these entries.'], [503, 'FL is temporarily unavailable. Please try again.']]) {
    const h = page({ language, request: async () => ({ ok: false, status, json: async () => ({ detail: { message } }) }) });
    await h.ready(); h.click(h.get('today-entries')); await flush();
    assert.ok(h.text().includes(message), h.text());
    assert.doesNotMatch(h.text(), /15 seconds|15 segundos/);
    assert.equal(h.get('send').disabled, false);
  }
  const h = page({ language, request: async () => { throw new Error('Failed to fetch'); } });
  await h.ready(); h.click(h.get('today-entries')); await flush();
  assert.match(h.text(), language === 'en' ? /Connection failed\. Please try again\./ : /La conexión falló\. Inténtalo de nuevo\./);
  assert.doesNotMatch(h.text(), /15 seconds|15 segundos|Failed to fetch/);
});

test('Today session expiry returns to sign-in with the session message', async () => {
  const h = page({ request: async () => ({ ok: false, status: 401, json: async () => ({ detail: { error_code: 'SESSION_EXPIRED', message: 'Your session expired.' } }) }) });
  await h.ready(); h.click(h.get('today-entries')); await flush();
  assert.equal(h.pilot.cleared, true);
  assert.equal(h.get('sign-in-panel').hidden, false);
  assert.equal(h.get('chat-panel').hidden, true);
  assert.match(h.get('notice').textContent, /Your session ended\. Sign in again/);
  assert.doesNotMatch(h.text(), /15 seconds|15 segundos/);
});

test('Today deadline exhaustion between pages uses the slowness message', async () => {
  const h = page({ request: async (_call, clock) => { clock.advance(15000); return response({ receipts: [], has_more: true, next_offset: 500 }); } });
  await h.ready(); h.click(h.get('today-entries')); await flush();
  assert.equal(h.calls.filter(call => call.url.pathname === '/receipts').length, 1);
  assert.match(h.text(), /Today’s entries could not load within 15 seconds\./);
});

test('Review my day suggestion takes the direct Today read path', async () => {
  const h = page({ request: async ({ url }) => { assert.equal(url.pathname, '/receipts'); return response({ receipts: [] }); } });
  await h.ready(); h.click(h.prompts[0]); await flush();
  assert.deepEqual(h.calls.map(call => call.url.pathname), ['/assistant/resume', '/receipts']);
});

test('exact Today questions in English and Spanish use the direct reader', () => {
  for (const query of ['What did I enter today?', 'Show all today’s entries', "Today's entries", '¿Qué registré hoy?', 'Entradas de hoy']) assert.equal(ui.isTodayQuery(query), true, query);
  for (const query of ['Receive 50 lb oats today', 'Void RCV-261015-001', 'What did I enter yesterday?']) assert.equal(ui.isTodayQuery(query), false, query);
});

test('composer Today query reads directly without spending an assistant turn', async () => {
  const h = page({ request: async ({ url }) => { assert.equal(url.pathname, '/receipts'); return response({ receipts: [] }); } });
  await h.ready(); h.submit('What did I enter today?'); await flush();
  assert.deepEqual(h.calls.map(call => call.url.pathname), ['/assistant/resume', '/receipts']);
  assert.equal(h.get('message').value, '');
});

test('Send displays working feedback for the whole pending request', async () => {
  const pending = deferred();
  const h = page({ request: ({ url }) => { assert.equal(url.pathname, '/assistant/turn'); return pending.promise; } });
  await h.ready(); h.submit('Receive oats'); assertWorking(h);
  assert.equal(h.get('send').disabled, true);
  pending.resolve(response({ cards: [{ kind: 'missing_facts', fields: ['cases'] }] })); await flush();
  assert.equal(h.get('working-status').hidden, true);
  assert.equal(h.get('send').textContent, 'Send ↑');
  assert.equal(h.get('send').getAttribute('aria-busy'), 'false');
});

test('Confirm shows working feedback while FL prepares the new SKU draft', async () => {
  const pending = deferred();
  const h = page({ cards: [draft({ action: 'make', can_commit: false, blockers: [{ code: 'SKU_CONFIRMATION_REQUIRED', message: 'Confirm Batch Classic Granola #9 as the output SKU and prepare again.' }] })],
    request: ({ url }) => { assert.equal(url.pathname, '/assistant/confirm-sku'); return pending.promise; } });
  await h.ready(); const confirm = descendants(h.cards()[0]).find(node => node.tag === 'button' && /Confirm this output SKU/.test(node.textContent));
  assert.ok(confirm); h.click(confirm); assertWorking(h, confirm);
  pending.resolve(response({ card: draft() })); await flush();
  assert.equal(h.get('working-status').hidden, true);
  assert.match(h.text(), /Output SKU confirmed/);
});

for (const language of ['en', 'es']) test(`duplicate acknowledgment has a plain label and gates Record in ${language}`, async () => {
  const h = page({ language, cards: [draft({ warnings: [{ code: 'POSSIBLE_DUPLICATE', requires_ack: true, message: 'A matching receive was already posted.', message_es: 'Ya se registró una recepción igual.' }] })] });
  await h.ready(); const card = h.cards()[0], nodes = descendants(card);
  const label = nodes.find(node => node.tag === 'label' && node.children.some(child => child.type === 'checkbox'));
  assert.ok(label);
  assert.match(text(label), language === 'en' ? /Yes, this is a separate entry, record it anyway\./ : /Sí, esta es una entrada diferente; regístrala de todos modos\./);
  const record = nodes.find(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Record' : 'Registrar'));
  assert.equal(record.disabled, true);
  const checkbox = label.children.find(node => node.type === 'checkbox'); checkbox.checked = true; checkbox.listeners.input();
  assert.equal(record.disabled, false);
});

test('Record shows working feedback until a real numbered receipt arrives', async () => {
  const pending = deferred();
  const h = page({ cards: [draft()], request: ({ url, body }) => { assert.equal(url.pathname, '/assistant/record'); assert.equal(body.draft_id, 'draft-1'); return pending.promise; } });
  await h.ready(); const record = descendants(h.cards()[0]).find(node => node.tag === 'button' && node.textContent === 'Record');
  h.click(record); assertWorking(h, record); assert.equal(record.disabled, true);
  pending.resolve(response({ kind: 'receipt', result: { success: true, receipt_number: 'RCV-261015-001', transaction_id: 7, product_name: 'Sunflower Seeds', total_lb: 50 } })); await flush();
  assert.equal(h.get('working-status').hidden, true);
  assert.match(h.cards()[0].className, /receipt/);
  assert.match(h.text(), /RCV-261015-001/);
  assert.doesNotMatch(h.text(), /Transaction|Transacción/);
  assert.equal(descendants(h.cards()[0]).some(node => node.tag === 'dd' && node.textContent === '7'), false);
});

for (const language of ['en', 'es']) test(`restored recorded card identifies the receipt instead of an internal transaction in ${language}`, async () => {
  const card = draft();
  const h = page({ language, cards: [card], drafts: [{ id: card.id, card, status: 'committed', result: { success: true,
    receipt_number: 'RCV-261015-007', transaction_id: 7, product_name: 'Sunflower Seeds', total_lb: 50 } }] });
  await h.ready();
  assert.equal(h.cards()[0].children.find(node => node.tag === 'h3').textContent, 'RCV-261015-007');
  assert.doesNotMatch(h.text(), /Transaction|Transacción/);
  assert.equal(descendants(h.cards()[0]).some(node => node.tag === 'dd' && node.textContent === '7'), false);
});

for (const [action, lines] of [
  ['ship_order', [{ product_name: 'Classic Granola', quantity_lb: 50, cases: 2, lots: [{ lot_code: 'LOT-CLASSIC' }] }]],
  ['ship_bulk', [{ product_name: 'Classic Granola', quantity_lb: 50, cases: 2, lots: [{ lot_code: 'LOT-CLASSIC' }] },
    { product: 'Chocolate Granola', quantity_lb: 25, cases: 1, lots: [{ lot_code: 'LOT-CHOCOLATE' }] },
    { product: 'Freight', is_service: true }]]
]) test(`${action} identifies each product once and preserves individual line details`, async () => {
  const h = page({ cards: [draft({ action, draft: { lines } })] });
  await h.ready();
  const rows = descendants(h.cards()[0]).filter(node => node.className === 'ship-line');
  assert.equal(rows.length, lines.length);
  for (const [i, line] of lines.entries()) {
    const name = line.product_name || line.product;
    assert.equal(h.text().split(name).length - 1, 1, name);
    assert.equal(rows[i].children[0].textContent, name);
    if (line.is_service) assert.match(text(rows[i]), /Service charge/);
    else {
      assert.ok(text(rows[i]).includes(String(line.quantity_lb)));
      assert.ok(text(rows[i]).includes(String(line.cases)));
      assert.ok(text(rows[i]).includes(line.lots[0].lot_code));
    }
  }
});

test('session restoration keeps sign-in hidden until the signed-in chat is ready', async () => {
  const init = deferred(), h = page({ init: init.promise });
  assert.equal(h.get('sign-in-panel').hidden, true);
  assert.equal(h.get('auth-loading').hidden, false);
  assert.equal(h.get('chat-panel').hidden, true);
  init.resolve(true); await h.ready();
  assert.equal(h.get('sign-in-panel').hidden, true);
  assert.equal(h.get('auth-loading').hidden, true);
  assert.equal(h.get('chat-panel').hidden, false);
  assert.match(h.get('identity').textContent, /Arturo/);
});

test('with sessions off the loading state resolves to the personal-key form', async () => {
  const init = deferred(), h = page({ init: init.promise, enabled: false });
  assert.equal(h.get('sign-in-panel').hidden, true);
  init.resolve(false); await h.ready();
  assert.equal(h.get('auth-loading').hidden, true);
  assert.equal(h.get('sign-in-panel').hidden, false);
  assert.equal(h.get('chat-panel').hidden, true);
});

test('read cards start with a sentence and table, dedupe identical results, and expose no JSON', async () => {
  const card = { kind: 'read', tool: 'inventory_lookup', result: { results: [{ product: 'Batch Classic Granola #9', total_on_hand: 17678,
    lots: [{ lot: 'LOT-1', qty_on_hand: 17678, unit: '10 lb case' }] }] } };
  const h = page({ cards: [card, { ...card, id: 'same-again' }] }); await h.ready();
  assert.equal(h.cards().length, 1);
  const nodes = descendants(h.cards()[0]);
  assert.equal(nodes.find(node => node.className === 'read-summary').textContent, 'Batch Classic Granola #9: 17,678 lb in 1 lot.');
  assert.equal(nodes.filter(node => node.tag === 'table').length, 1);
  assert.equal(nodes.some(node => node.tag === 'pre' || node.tag === 'details'), false);
  assert.doesNotMatch(h.text(), /qty_on_hand|10 lb case|total_on_hand/);
});

test('adjust draft renders product, quantity, before/after and plain reason without JSON', async () => {
  const h = page({ cards: [draft({ action: 'adjust', draft: { product_name: 'Oats', delta_lb: -5, current_quantity_lb: 2000,
    new_balance_lb: 1995, reason_code: 'physical_count', identity_status: 'UNIDENTIFIED', happened_at: '2026-10-16T02:19:00Z' } })] });
  await h.ready(); const content = h.text();
  for (const phrase of ['Oats', '-5 lb', '2,000 → 1,995 lb', 'Physical count', 'Today 10:19 PM', 'Lot identity needs to be checked']) assert.ok(content.includes(phrase), phrase);
  assert.doesNotMatch(content, /physical_count|UNIDENTIFIED|T02:19/);
});

for (const language of ['en', 'es']) test(`receive page shows actionable supplier lot guidance in ${language}`, async () => {
  const h = page({ language, cards: [draft({ draft: { product_name: 'Oats', total_lb: 50,
    lot_code: '26-10-15-SUPP-001', identity_status: 'UNIDENTIFIED' } })] });
  await h.ready();
  assert.ok(h.text().includes(language === 'en' ? 'No supplier lot number given — add it if the bag has one' :
    'No se indicó el número de lote del proveedor — agrégalo si aparece en la bolsa'));
  assert.doesNotMatch(h.text(), /Lot identity needs to be checked|Hay que verificar la identidad del lote|UNIDENTIFIED/);
  assert.equal(descendants(h.cards()[0]).filter(node => node.tag === 'button' && node.textContent === (language === 'en' ? 'Record' : 'Registrar')).length, 1);
});

test('extra-kosher receive refusal has a refusal card and no disabled Record control', async () => {
  const h = page({ cards: [draft({ can_commit: false, blockers: [{ code: 'EXTRA_KOSHER_RECEIVE_BLOCKED', message: 'Extra-kosher batches are made, never received.' }] })] });
  await h.ready(); assert.match(h.cards()[0].className, /refusal/);
  assert.equal(descendants(h.cards()[0]).some(node => node.tag === 'button'), false);
  assert.match(h.text(), /Nothing was recorded/);
});

test('void refusal provides a direct handoff to that receipt screen', async () => {
  const h = page({ turnText: 'Void ADJ-261015-242, data-entry error', cards: [{ kind: 'unsupported_request' }] });
  await h.ready(); const link = descendants(h.get('messages')).find(node => node.tag === 'a' && /to void/.test(node.textContent));
  assert.ok(link); const target = new URL(link.href, 'https://fl.local');
  assert.equal(target.pathname, '/dashboard/receipt.html'); assert.equal(target.searchParams.get('number'), 'ADJ-261015-242');
});

test('every Assistant asset resolves to a real absolute dashboard file from either page URL', () => {
  const assets = [...html.matchAll(/<(?:script|link)\b[^>]*(?:src|href)="([^"]+)"/g)].map(match => match[1]);
  for (const name of ['dashboard-runtime.js', 'ledger-forms.css', 'receipt-shell.js']) assert.ok(assets.some(asset => asset.includes('/' + name)));
  for (const asset of assets) {
    assert.match(asset, /^\/dashboard\//);
    for (const pageUrl of ['https://fl.local/dash/fl-assistant', 'https://fl.local/dashboard/fl-assistant.html']) {
      const resolved = new URL(asset, pageUrl); assert.equal(resolved.pathname.startsWith('/dashboard/'), true);
      assert.equal(fs.existsSync(path.join(root, resolved.pathname)), true, asset);
    }
  }
});

test('header controls and receipt tools retain responsive containment rules', () => {
  const css = fs.readFileSync(path.join(root, 'dashboard/fl-assistant.css'), 'utf8');
  assert.match(css, /\.topbar\{[^}]*height:auto;[^}]*display:grid;[^}]*grid-template-columns:minmax\(0,1fr\) auto/);
  assert.match(css, /\.header-actions button\{[^}]*white-space:nowrap/);
  assert.match(css, /\.topbar \.receipt-tools\{[^}]*grid-column:1\/-1;[^}]*min-width:0/);
  assert.match(css, /@media\(max-width:600px\)\{[\s\S]*\.topbar \.receipt-lookup input\{width:100%/);
});

for (const language of ['en', 'es']) test(`voice Today phrases read a truthful own/team summary and newest first in ${language}`, async () => {
  const prompts = language === 'en' ? ["Open today's entries", 'Is everything there correct?', 'What did I record today?', "Show today's entries", "Open today's entries. Is everything there correct?"] : ['¿Qué registré hoy?', 'Muestra las entradas de hoy', 'Abre las entradas de hoy', '¿Está todo correcto?'];
  for (const prompt of prompts) {
    const h = page({ language, request: async ({ url }) => {
      assert.equal(url.pathname, '/receipts');
      return response({ receipts: [
        {receipt_number: 'RCV-261015-001', actor: {id: 7, name: 'Arturo'}, action: 'receive', entered_at: '2026-10-15T15:00:00Z'},
        {receipt_number: 'RCV-261015-002', actor: {id: 8, name: 'Michael'}, action: 'receive', entered_at: '2026-10-15T16:00:00Z'}], has_more: false });
    } });
    await h.ready(); h.submit(prompt); await flush();
    assert.match(h.text(), language === 'en' ? /You recorded 1 entry today; the team recorded 2\./ : /Registraste 1 entrada hoy; el equipo registró 2\./);
    const rows = descendants(h.cards()[0]).filter(node => node.className === 'today-entry');
    assert.match(text(rows[0]), /RCV-261015-002/);
    assert.equal(h.calls.some(call => call.url.pathname === '/assistant/turn'), false);
    assert.doesNotMatch(text(h.cards()[0]), /try again|todo.*correcto|everything.*correct/i);
  }
});

for (const language of ['en', 'es']) test(`unreadable and internal HTTP errors stay plain in ${language}`, async () => {
  for (const request of [async () => ({ok: false, status: 500, json: async () => { throw new SyntaxError('Unexpected token I'); }}), async () => ({ok: false, status: 500, json: async () => ({detail: 'Internal server error'})})]) {
    const h = page({language, request}); await h.ready(); h.click(h.get('today-entries')); await flush();
    assert.match(h.text(), language === 'en' ? /Factory Ledger is temporarily unavailable — try again in a minute/ : /Factory Ledger no está disponible temporalmente — inténtalo de nuevo en un minuto/);
    assert.doesNotMatch(h.text(), /Internal server error|unreadable response|Unexpected token/);
  }
});

for (const language of ['en', 'es']) test(`spoken Today language follows the message when the UI started in ${language}`, async () => {
  const h = page({language, request: async () => response({receipts: [], has_more: false})});
  await h.ready(); h.submit(language === 'en' ? '¿Qué registré hoy?' : 'What did I record today?'); await flush();
  assert.match(text(h.cards()[0]), language === 'en' ? /Registraste 0 entradas hoy; el equipo registró 0/ : /You recorded 0 entries today; the team recorded 0/);
});

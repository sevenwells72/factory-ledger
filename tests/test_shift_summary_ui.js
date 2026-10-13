const test = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../dashboard/shift-summary.js');

const doc = {
  date: '2026-10-07', lang: 'en', receipts_today: 2, summary_hash: 'a'.repeat(64),
  title: 'SHIFT SUMMARY — Tue Oct 7 — Arturo', title_en: 'SHIFT SUMMARY — Tue Oct 7 — Arturo', title_es: 'RESUMEN DEL TURNO — mar 7 oct — Arturo',
  actor: { id: 7, name: 'Arturo', role: 'floor' },
  confirmation: { status: 'unconfirmed', latest: null, can_confirm: true, reason: null, reason_en: null, reason_es: null },
  sections: [
    { key: 'received', label: 'RECEIVED', label_en: 'RECEIVED', label_es: 'RECIBIDO', empty: 'none', empty_en: 'none', empty_es: 'ninguno', count: 1, rows: [
      { receipt_number: 'RCV-261007-002', action: 'receive', effective_status: 'posted', late_entry: false, hours_late: 0, entered_on_later_day: false,
        actor: { id: 7, name: 'Arturo', role: 'floor' }, entered_at: '2026-10-07T08:43:10-04:00',
        line: 'RCV-261007-002  Oats <Rolled>  2,000 lb  lot 26-10-07-QUAL-001  08:41', line_en: 'RCV-261007-002  Oats <Rolled>  2,000 lb  lot 26-10-07-QUAL-001  08:41',
        flags: [{ code: 'UNIDENTIFIED_LOT', severity: 'warn', message: 'UNIDENTIFIED — resolve by Oct 14', message_en: 'UNIDENTIFIED — resolve by Oct 14', refs: {} }] } ] },
    { key: 'made', label: 'MADE', label_en: 'MADE', label_es: 'PRODUCIDO', empty: 'none', empty_en: 'none', empty_es: 'ninguno', count: 0, rows: [] },
    { key: 'not_entered_yet', label: 'NOT ENTERED YET?', label_en: 'NOT ENTERED YET?', label_es: '¿FALTA REGISTRAR?', empty: 'nothing expected', empty_en: 'nothing expected', empty_es: 'nada pendiente', count: 1,
      rows: [{ kind: 'expected_receipt', overdue: true, line: 'expected receipt ER-1 (Graham crumbs, due Oct 5, overdue) — no RCV receipt', line_en: 'expected receipt ER-1 (Graham crumbs, due Oct 5, overdue) — no RCV receipt' }] },
    { key: 'lot_balances_touched', label: 'LOT BALANCES TOUCHED TODAY', label_en: 'LOT BALANCES TOUCHED TODAY', label_es: 'LOTES MOVIDOS HOY', empty: 'none', empty_en: 'none', empty_es: 'ninguno', count: 1,
      rows: [{ lot: { id: 1 }, negative: true, short: true, receipts: ['MK-261007-002'], line: 'oats lot 26-09-30-OATS-002 0 → −12 lb (short)', line_en: 'oats lot 26-09-30-OATS-002 0 → −12 lb (short)' }] },
    { key: 'open_exceptions', label: 'OPEN EXCEPTIONS (yours)', label_en: 'OPEN EXCEPTIONS (yours)', label_es: 'EXCEPCIONES ABIERTAS (tuyas)', empty: 'none', empty_en: 'none', empty_es: 'ninguna', count: 1,
      rollup: '1 shortage (due Oct 9)', rollup_en: '1 shortage (due Oct 9)', rollup_es: '1 faltante (vence 9 oct)',
      rows: [{ exception_id: 41, overdue: false, due_at: '2026-10-09T23:59:00-04:00', summary: 'SHORT 12 lb oats lot X — resolve by Oct 9', summary_en: 'SHORT 12 lb oats lot X — resolve by Oct 9' }] },
    { key: 'late_entries', label: 'LATE ENTRIES TODAY', label_en: 'LATE ENTRIES TODAY', label_es: 'ENTRADAS TARDÍAS HOY', empty: 'none', empty_en: 'none', empty_es: 'ninguna', count: 0, rows: [] },
  ],
};

test('api base: explicit override, FL-served origin, netlify/file fall back to production', () => {
  assert.equal(ui.apiBase({ protocol: 'https:', hostname: 'cns-factory-ledger.netlify.app', origin: 'https://cns-factory-ledger.netlify.app' }, ''), ui.PRODUCTION_API);
  assert.equal(ui.apiBase({ protocol: 'file:', hostname: '', origin: 'null' }, ''), ui.PRODUCTION_API);
  assert.equal(ui.apiBase({ protocol: 'https:', hostname: 'fastapi-staging-production-dd7b.up.railway.app', origin: 'https://fastapi-staging-production-dd7b.up.railway.app' }, ''), 'https://fastapi-staging-production-dd7b.up.railway.app');
  assert.equal(ui.apiBase({ protocol: 'https:', hostname: 'cns-factory-ledger.netlify.app', origin: 'x' }, '?api=https://staging.example/'), 'https://staging.example');
  assert.equal(ui.apiBase({ protocol: 'https:', hostname: 'cns-factory-ledger.netlify.app', origin: 'x' }, '?api=javascript:alert(1)'), ui.PRODUCTION_API);
});

test('sections render in document order with labels, counts, empty text and escaped lines', () => {
  const html = ui.summaryHtml(doc, 'en');
  const keys = [...html.matchAll(/data-key="([a-z_]+)"/g)].map(m => m[1]);
  assert.deepEqual(keys, ['received', 'made', 'not_entered_yet', 'lot_balances_touched', 'open_exceptions', 'late_entries']);
  assert.ok(html.includes('<span>RECEIVED</span><span class="count">1</span>'));
  assert.ok(html.includes('<div class="section-empty">none</div>'));
  assert.ok(html.includes('Oats &lt;Rolled&gt;'), 'HTML in a product name is escaped');
  assert.ok(!html.includes('<Rolled>'));
  assert.ok(html.includes('<span class="receipt-number">RCV-261007-002</span><span class="line">Oats &lt;Rolled&gt;  2,000 lb'), 'receipt number is split out of the line');
  assert.ok(html.includes('<span class="flag">⚠ UNIDENTIFIED — resolve by Oct 14</span>'));
  assert.ok(html.includes('<div class="section-rollup">1 shortage (due Oct 9)</div>'));
  assert.ok(html.includes('class="row lot-line short"'));
  assert.ok(html.includes('class="row late"') && html.includes('overdue) — no RCV receipt'));
  assert.ok(html.includes('#41 · due 2026-10-09 23:59'));
});

test('row meta carries actor, entry clock and lateness; voided rows are struck through', () => {
  const row = { ...doc.sections[0].rows[0], late_entry: true, hours_late: 52.5, entered_on_later_day: true, effective_status: 'voided', flags: [] };
  const en = ui.receiptRowHtml(row, 'en');
  assert.ok(en.includes('class="row late voided"'));
  assert.ok(en.includes('Arturo · entered 08:43 · late 52.5 h · entered on a later day'));
  const es = ui.receiptRowHtml(row, 'es');
  assert.ok(es.includes('Arturo · registrado 08:43 · tardía 52.5 h · registrado otro día'));
});

test('confirm body echoes the hash and the receipt count shown, in the page language', () => {
  assert.deepEqual(ui.confirmBody(doc, 'discrepancy', 'oats missing', 'es'),
    { date: '2026-10-07', outcome: 'discrepancy', notes: 'oats missing', summary_hash: 'a'.repeat(64), receipts_seen: 2, lang: 'es' });
  assert.deepEqual(ui.confirmBody(doc, 'match'), { date: '2026-10-07', outcome: 'match', notes: '', summary_hash: 'a'.repeat(64), receipts_seen: 2, lang: 'en' });
});

test('stale 409 hands back the fresh summary; other errors do not', () => {
  const fresh = { ...doc, summary_hash: 'b'.repeat(64) };
  const stale = { detail: { error_code: 'SHIFT_SUMMARY_STALE', message: 'changed', summary: fresh }, success: false,
    error_detail: { code: 'SHIFT_SUMMARY_STALE', message: 'changed', message_es: 'cambió' }, httpStatus: 409 };
  assert.equal(ui.staleSummary(stale), fresh);
  assert.equal(ui.staleSummary({ detail: { error_code: 'NOTES_REQUIRED', message: 'x' } }), null);
  assert.equal(ui.staleSummary({ detail: { error_code: 'SHIFT_SUMMARY_STALE', message: 'x' } }), null);
});

test('error text prefers the Spanish copy in Spanish and falls back to the English message', () => {
  const error = { detail: { error_code: 'ACTOR_REQUIRED', message: 'A shift is confirmed by a person', message_es: 'El turno lo confirma una persona' },
    error_detail: { code: 'ACTOR_REQUIRED', message: 'A shift is confirmed by a person', message_es: 'El turno lo confirma una persona' } };
  assert.equal(ui.errorText(error, 'es'), 'El turno lo confirma una persona');
  assert.equal(ui.errorText(error, 'en'), 'A shift is confirmed by a person');
  assert.equal(ui.errorText({ error_detail: { code: 'HTTP_500', message: 'boom' } }, 'es'), 'boom');
  assert.equal(ui.errorText({ detail: [{ loc: ['body', 'receipts_seen'], msg: 'required' }] }, 'en'), 'receipts_seen: required');
  assert.equal(ui.errorText(undefined, 'es'), 'La solicitud falló');
});

test('banner reflects the latest confirmation, with notes, in both languages', () => {
  assert.equal(ui.bannerText(doc, 'en'), null);
  const confirmed = { ...doc, confirmation: { status: 'match', can_confirm: true, latest: { actor: { name: 'Arturo' }, confirmed_at: '2026-10-07T17:05:02-04:00', notes: '' } } };
  assert.equal(ui.bannerText(confirmed, 'en'), 'Confirmed by Arturo at 2026-10-07 17:05');
  const missing = { ...doc, confirmation: { status: 'discrepancy', can_confirm: true, latest: { actor: { name: 'Arturo' }, confirmed_at: '2026-10-07T17:05:02-04:00', notes: 'oats' } } };
  assert.equal(ui.bannerText(missing, 'es'), 'Marcado como "falta algo" por Arturo el 2026-10-07 17:05 — oats');
});

// ---------------------------------------------------------------------------
// Fix round 2026-10-13 (review items 8–9): the browser event handlers themselves,
// driven through a minimal fake DOM (ids, hidden/value/textContent, listeners) and a
// fetch stub whose responses the test releases in any order.
// ---------------------------------------------------------------------------
const path = require('node:path');
const MODULE = process.env.SHIFT_SUMMARY_JS || path.resolve(__dirname, '../dashboard/shift-summary.js');
const IDS = ['language', 'sign-out', 'identity', 'sign-in-panel', 'sign-in-form', 'actor-key', 'read-only', 'summary-panel', 'date', 'yesterday',
  'today', 'refresh', 'load-status', 'status-banner', 'title', 'receipts-count', 'sections', 'confirm-bar', 'confirm-question', 'confirm-reason',
  'confirm-match', 'confirm-notes', 'confirm-missing', 'notes-form', 'notes-label', 'notes', 'notes-cancel', 'notes-send', 'notice'];

function element(id) {
  return { id, hidden: false, disabled: false, value: '', textContent: '', innerHTML: '', className: '', placeholder: '', max: '', dataset: {}, offsetHeight: 0, listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    fire(type, event) { (this.listeners[type] || []).forEach(fn => fn({ preventDefault() {}, ...(event || {}) })); },
    setAttribute(name, value) { this[name] = value; }, focus() {}, classList: { toggle() {} } };
}

function mount() {
  const els = Object.fromEntries(IDS.map(id => [id, element(id)]));
  els['confirm-bar'].offsetHeight = 225;
  for (const id of ['sign-out', 'summary-panel', 'status-banner', 'confirm-bar', 'confirm-reason', 'notes-form', 'notice']) els[id].hidden = true;   // as in the HTML
  const style = {};
  const calls = [];
  global.document = { documentElement: { lang: 'en', style: { setProperty: (k, v) => { style[k] = v; } } }, getElementById: id => els[id] || null, querySelectorAll: () => [] };
  global.fetch = (url, init) => new Promise((resolve, reject) => {
    const call = { url: String(url), init, body: init && init.body ? JSON.parse(init.body) : null, done: false,
      resolve: data => { call.done = true; resolve({ ok: true, status: 200, json: async () => data }); },
      reject: (status, data) => { call.done = true; resolve({ ok: false, status, json: async () => data }); },
      fail: () => { call.done = true; reject(new TypeError('Failed to fetch')); } };           // network error / timeout
    if (init && init.signal) init.signal.addEventListener('abort', () => call.resolve({}));
    calls.push(call);
  });
  const store = {};
  global.localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; } };
  global.location = { protocol: 'https:', hostname: 'fl.local', origin: 'https://fl.local', search: '' };
  delete require.cache[MODULE];
  require(MODULE);
  const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(r => setImmediate(r)); };
  const finish = async () => { calls.filter(c => !c.done).forEach(c => c.resolve({})); await settle(); };
  return { els, calls, style, settle, finish };
}

const plantToday = () => {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(new Date());
  const get = type => parts.find(p => p.type === type).value;
  return `${get('year')}-${get('month')}-${get('day')}`;
};
const dayBefore = iso => { const d = new Date(iso + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() - 1); return d.toISOString().slice(0, 10); };
const forActor = (name, id, date, hash) => ({ ...doc, date, summary_hash: hash, title: `SHIFT SUMMARY — ${date} — ${name}`, title_en: `SHIFT SUMMARY — ${date} — ${name}`, actor: { id, name, role: 'floor' } });

test('a response for a previous sign-in never renders under the next person', async () => {
  const { els, calls, settle, finish } = mount();
  const today = plantToday();
  const docA = forActor('Arturo', 7, today, 'a'.repeat(64)), docB = forActor('Beatriz', 8, today, 'b'.repeat(64));
  // A signs in; before FL answers, A signs out and B signs in.
  els['actor-key'].value = 'key-A'; els['sign-in-form'].fire('submit'); await settle();
  assert.equal(calls.length, 1); assert.match(calls[0].url, /\/auth\/whoami$/); assert.equal(calls[0].init.headers['X-API-Key'], 'key-A');
  els['sign-out'].fire('click'); await settle();
  els['actor-key'].value = 'key-B'; els['sign-in-form'].fire('submit'); await settle();
  assert.equal(calls.length, 2);
  calls[0].resolve({ actor: docA.actor }); await settle();            // A's delayed whoami
  assert.equal(calls.length, 2, 'no summary is loaded for the signed-out person');
  assert.equal(els['summary-panel'].hidden, true); assert.ok(!/Arturo/.test(els['identity'].textContent));
  calls[1].resolve({ actor: docB.actor }); await settle();
  assert.equal(els['identity'].textContent, 'Signed in as Beatriz · floor');
  assert.equal(calls.length, 3); assert.equal(calls[2].init.headers['X-API-Key'], 'key-B');
  calls[2].resolve(docB); await settle();
  assert.equal(els['title'].textContent, docB.title); assert.equal(els['confirm-bar'].hidden, false);
  // B refreshes; before FL answers, B signs out and A signs in; B's old summary then arrives.
  els['refresh'].fire('click'); await settle();
  assert.equal(calls.length, 4);
  els['sign-out'].fire('click'); await settle();
  assert.equal(els['title'].textContent, '' ); assert.equal(els['sections'].innerHTML, ''); assert.equal(els['confirm-bar'].hidden, true);
  els['actor-key'].value = 'key-A'; els['sign-in-form'].fire('submit'); await settle();
  calls[4].resolve({ actor: docA.actor }); await settle();
  assert.equal(calls.length, 6);
  calls[3].resolve(docB); await settle();                             // B's stale summary
  assert.equal(els['title'].textContent, '', 'a stale summary never renders'); assert.equal(els['sections'].innerHTML, '');
  assert.equal(els['confirm-bar'].hidden, true, 'no confirm buttons for a document that is not this person\'s');
  calls[5].resolve(docA); await settle();
  assert.equal(els['title'].textContent, docA.title); assert.equal(els['confirm-bar'].hidden, false);
  // A confirm goes out with A's key and A's hash, never B's.
  els['confirm-match'].fire('click'); await settle();
  assert.equal(calls[6].init.headers['X-API-Key'], 'key-A'); assert.deepEqual(calls[6].body, { date: today, outcome: 'match', notes: '', summary_hash: docA.summary_hash, receipts_seen: 2, lang: 'en' });
  await finish();
});

test('notes are bound to the person and the day they were opened for; text is sent verbatim', async () => {
  const { els, calls, settle, finish } = mount();
  const today = plantToday(), yesterday = dayBefore(today);
  const docToday = forActor('Arturo', 7, today, 'c'.repeat(64)), docYesterday = forActor('Arturo', 7, yesterday, 'd'.repeat(64));
  els['actor-key'].value = 'key-A'; els['sign-in-form'].fire('submit'); await settle();
  calls[0].resolve({ actor: docToday.actor }); await settle(); calls[1].resolve(docToday); await settle();
  assert.equal(els['title'].textContent, docToday.title);
  els['confirm-missing'].fire('click'); await settle();
  assert.equal(els['notes-form'].hidden, false); assert.equal(els['notes-label'].textContent, 'What is missing? (required)');
  // Changing the day closes the notes that were opened for today.
  els['date'].value = yesterday; els['date'].fire('change'); await settle();
  assert.equal(els['notes-form'].hidden, true); assert.equal(calls.length, 3); assert.match(calls[2].url, new RegExp(`date=${yesterday}`));
  calls[2].resolve(docYesterday); await settle();
  // Notes opened for yesterday are posted for yesterday, verbatim.
  els['confirm-missing'].fire('click'); await settle();
  els['notes'].value = '  oats never arrived\n'; els['notes-form'].fire('submit'); await settle();
  assert.equal(calls.length, 4); assert.match(calls[3].url, /\/confirm$/);
  assert.deepEqual(calls[3].body, { date: yesterday, outcome: 'discrepancy', notes: '  oats never arrived\n', summary_hash: docYesterday.summary_hash, receipts_seen: 2, lang: 'en' });
  calls[3].resolve({ message: 'Shift recorded as: something is missing.' }); await settle();
  assert.equal(els['notes-form'].hidden, true); assert.equal(calls.length, 5);
  calls[4].resolve(docYesterday); await settle();
  assert.equal(els['notice'].textContent, 'Shift recorded as: something is missing.', 'FL\'s message survives the reload');
  // Blank notes for "something is missing" never post.
  els['confirm-missing'].fire('click'); await settle();
  els['notes'].value = ' \n\t'; els['notes-form'].fire('submit'); await settle();
  assert.equal(calls.length, 5); assert.equal(els['notice'].textContent, 'Say what is missing so the owner can act on it.');
  // A form left open across a day change (Today shortcut) is closed; a stale submit posts nothing.
  els['notes'].value = 'late text';
  els['today'].fire('click'); await settle();
  assert.equal(els['notes-form'].hidden, true); assert.equal(calls.length, 6);
  els['notes-form'].fire('submit'); await settle();
  assert.equal(calls.length, 6, 'no POST without an open, matching notes context');
  calls[5].resolve(docToday); await settle();
  // Sign-out while notes are open: the next person starts with no pending notes.
  els['confirm-notes'].fire('click'); await settle(); els['notes'].value = 'mine';
  els['sign-out'].fire('click'); await settle();
  assert.equal(els['notes-form'].hidden, true); assert.equal(els['notes'].value, '');
  await finish();
});

test('a failed date change leaves nothing to confirm until the selected day loads', async () => {
  // Fix round 2 (Codex comment 6150241244): load today → pick yesterday → GET fails. The old
  // document must not stay confirmable under a picker that says another day.
  const { els, calls, settle, finish } = mount();
  const today = plantToday(), yesterday = dayBefore(today);
  const docToday = forActor('Arturo', 7, today, 'f'.repeat(64)), docYesterday = forActor('Arturo', 7, yesterday, '0'.repeat(64));
  els['actor-key'].value = 'key-A'; els['sign-in-form'].fire('submit'); await settle();
  calls[0].resolve({ actor: docToday.actor }); await settle(); calls[1].resolve(docToday); await settle();
  assert.equal(els['title'].textContent, docToday.title); assert.equal(els['confirm-match'].disabled, false);
  // While the other day is loading the buttons are disabled and a click posts nothing.
  els['date'].value = yesterday; els['date'].fire('change'); await settle();
  assert.equal(calls.length, 3); assert.match(calls[2].url, new RegExp(`date=${yesterday}`));
  assert.equal(els['confirm-match'].disabled, true, 'confirm is locked while a day loads');
  els['confirm-match'].fire('click'); await settle();
  assert.equal(calls.length, 3, 'no POST while the selected day is loading');
  // The load fails: the old day's document is gone, nothing can be confirmed, the error is shown.
  calls[2].reject(500, { success: false, error_detail: { code: 'HTTP_500', message: 'Internal error' } }); await settle();
  assert.equal(els['notice'].textContent, 'Internal error'); assert.equal(els['notice'].hidden, false);
  assert.equal(els['confirm-bar'].hidden, true, 'no confirm bar without a document for the selected day');
  assert.equal(els['title'].textContent, ''); assert.equal(els['sections'].innerHTML, ''); assert.equal(els['receipts-count'].textContent, '');
  assert.equal(els['status-banner'].hidden, true); assert.equal(els['notes-form'].hidden, true);
  els['confirm-match'].fire('click'); await settle(); els['confirm-missing'].fire('click'); await settle(); els['notes-form'].fire('submit'); await settle();
  assert.equal(calls.length, 3, 'no POST after a failed load — the wrong day can never be confirmed');
  assert.equal(els['notes-form'].hidden, true, 'notes cannot open without a document');
  // Refresh retries the selected day; a successful load re-enables confirm for that day only.
  els['refresh'].fire('click'); await settle();
  assert.equal(calls.length, 4); assert.match(calls[3].url, new RegExp(`date=${yesterday}`));
  calls[3].resolve(docYesterday); await settle();
  assert.equal(els['title'].textContent, docYesterday.title); assert.equal(els['confirm-bar'].hidden, false); assert.equal(els['confirm-match'].disabled, false);
  els['confirm-match'].fire('click'); await settle();
  assert.equal(calls.length, 5); assert.match(calls[4].url, /\/confirm$/);
  assert.deepEqual(calls[4].body, { date: yesterday, outcome: 'match', notes: '', summary_hash: docYesterday.summary_hash, receipts_seen: 2, lang: 'en' });
  calls[4].resolve({ message: 'Shift confirmed.' }); await settle(); calls[5].resolve(docYesterday); await settle();
  // The network path (timeout / connection) behaves the same as an HTTP error.
  els['today'].fire('click'); await settle();
  assert.equal(calls.length, 7);
  calls[6].fail(); await settle();
  assert.equal(els['confirm-bar'].hidden, true); assert.equal(els['title'].textContent, '');
  assert.equal(els['notice'].textContent, 'Connection failed or timed out.');
  els['confirm-match'].fire('click'); await settle();
  assert.equal(calls.length, 7, 'no POST after a timed-out load');
  // The language toggle still retries the selected day while signed in (no document is needed).
  els['language'].fire('click'); await settle();
  assert.equal(calls.length, 8); assert.match(calls[7].url, new RegExp(`date=${today}.*lang=es|lang=es.*date=${today}`));
  await finish();
});

test('the measured footer height drives the page padding so nothing hides under the bar', async () => {
  const { els, calls, style, settle, finish } = mount();
  const today = plantToday();
  assert.equal(style['--footer-h'], '0px', 'hidden bar reserves nothing');
  els['actor-key'].value = 'key-A'; els['sign-in-form'].fire('submit'); await settle();
  calls[0].resolve({ actor: { id: 7, name: 'Arturo', role: 'floor' } }); await settle(); calls[1].resolve(forActor('Arturo', 7, today, 'e'.repeat(64))); await settle();
  assert.equal(style['--footer-h'], '225px');
  els['confirm-bar'].offsetHeight = 331; els['confirm-notes'].fire('click'); await settle();
  assert.equal(style['--footer-h'], '331px', 'open notes grow the reserved space');
  els['confirm-bar'].offsetHeight = 225; els['notes-cancel'].fire('click'); await settle();
  assert.equal(style['--footer-h'], '225px');
  els['sign-out'].fire('click'); await settle();
  assert.equal(style['--footer-h'], '0px');
  await finish();
});

test('other_receipts block renders only when FL lists any, with its own label', () => {
  assert.equal(ui.otherReceiptsHtml({ count: 0, rows: [] }, 'en'), '');
  assert.equal(ui.otherReceiptsHtml(undefined, 'en'), '');
  const other = { label: 'OTHER ENTRIES (orders, expected receipts, lot moves)', label_en: 'OTHER ENTRIES (orders, expected receipts, lot moves)', count: 1,
    rows: [{ receipt_number: 'ORD-261007-001', action: 'create_order', kind: 'order', actor: { name: 'Lucía' }, entered_at: '2026-10-07T09:00:00-04:00', late_entry: false,
      line: 'ORD-261007-001  order created  SO-261007-001 Setton <Farms>  08:59', line_en: 'ORD-261007-001  order created  SO-261007-001 Setton <Farms>  08:59' }] };
  const html = ui.otherReceiptsHtml(other, 'en');
  assert.ok(html.includes('data-key="other_receipts"') && html.includes('<span>OTHER ENTRIES (orders, expected receipts, lot moves)</span><span class="count">1</span>'));
  assert.ok(html.includes('Setton &lt;Farms&gt;') && html.includes('Lucía · entered 09:00'));
  const whole = ui.summaryHtml({ ...doc, other_receipts: other }, 'en');
  assert.ok(whole.endsWith('</section>') && whole.indexOf('data-key="other_receipts"') > whole.indexOf('data-key="late_entries"'));
});

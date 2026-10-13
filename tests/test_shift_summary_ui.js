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

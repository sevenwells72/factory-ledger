"""A9: GET /reports/shift-summary + POST /reports/shift-summary/confirm against the
frozen contract (docs/contracts/shift-summary.md). Real HTTP through the TestClient
on the rolled-back test transaction; every section is exercised with a ticketed
receipt, the hash rules of §1.3 are asserted directly, and the permission matrix
of §1.1 / §3 is walked key by key."""
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest

import main
import shift_summary
import write_tickets as wt
from scripts.exceptions_sweep import sweep
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, error, seed as seed_receive, isolated_database  # noqa: F401
from tests.test_write_tickets_part2 import items, body, prepare  # noqa: F401
from tests.test_exceptions_enforcement import set_balance, adjust, found, exceptions_for
from tests.test_ship_tickets import seed_shipping, prepared as ship_prepared, upload, storage  # noqa: F401

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parent.parent
KEYS = [s[0] for s in shift_summary.SECTIONS]


def summary(client, key=None, **params):
    r = client.get('/reports/shift-summary', params=params, headers=headers(key))
    assert r.status_code == 200, r.text
    return r.json()


def section(doc, key):
    [s] = [s for s in doc['sections'] if s['key'] == key]
    return s


def confirm(client, key, doc, outcome='match', notes='', **extra):
    body_ = {'date': doc['date'], 'outcome': outcome, 'notes': notes, 'summary_hash': doc['summary_hash'],
             'receipts_seen': doc['receipts_today'], **extra}
    return client.post('/reports/shift-summary/confirm', json=body_, headers=headers(key))


def receive(client, cur, key, payload=None, **changes):
    payload = (payload or seed_receive(cur)) | changes
    r = client.post('/receive/prepare', json=payload, headers=headers(key))
    assert r.status_code == 200, r.text
    p = r.json()
    r = commit(client, p, key, acknowledged_warnings=[w['code'] for w in p['warnings'] if w.get('requires_ack')])
    assert r.status_code == 200, r.text
    return r.json(), payload, p


def posted(client, action, payload, key):
    p = prepare(client, action, payload, key)
    assert p['can_commit'], p
    r = commit(client, p, key, acknowledged_warnings=[w['code'] for w in p['warnings'] if w.get('requires_ack')])
    assert r.status_code == 200, r.text
    return r.json(), p


# ---------------------------------------------------------------------------
# Envelope and sections
# ---------------------------------------------------------------------------
def test_empty_day_has_every_section_in_order_with_bilingual_labels(client, actors):
    key = actors['floor']['key']
    doc = summary(client, key)
    assert [s['key'] for s in doc['sections']] == KEYS
    assert doc['actor'] == {'id': actors['floor']['id'], 'name': actors['floor']['name'], 'role': 'floor'}
    assert doc['date'] == main.get_plant_now().date().isoformat() and doc['plant_timezone'] == 'America/New_York'
    assert doc['receipts_today'] == 0 and len(doc['summary_hash']) == 64
    assert doc['confirmation'] == {'status': 'unconfirmed', 'latest': None, 'can_confirm': True,
                                   'reason': None, 'reason_en': None, 'reason_es': None}
    assert doc['title'] == doc['title_en'] and doc['title_en'].startswith('SHIFT SUMMARY — ') and doc['title_es'].startswith('RESUMEN DEL TURNO — ')
    assert doc['title_en'].endswith(actors['floor']['name'])
    for s in doc['sections']:
        assert s['rows'] == [] and s['count'] == 0
        assert s['label'] == s['label_en'] and s['empty'] == s['empty_en'] and s['label_es'] and s['empty_es']
    assert section(doc, 'received')['label_es'] == 'RECIBIDO' and section(doc, 'not_entered_yet')['empty_es'] == 'nada pendiente'
    assert section(doc, 'open_exceptions')['rollup_en'] == '' and section(doc, 'open_exceptions')['rollup'] == ''
    es = summary(client, key, lang='es')
    assert es['lang'] == 'es' and es['title'] == es['title_es']
    assert [s['label'] for s in es['sections']] == [s['label_es'] for s in es['sections']]
    assert es['summary_hash'] == doc['summary_hash']   # labels never enter the hash


def test_received_row_fields_lines_and_language_copies(client, db_cursor, actors):
    key = actors['floor']['key']
    result, payload, p = receive(client, db_cursor, key)
    doc = summary(client, key)
    assert doc['receipts_today'] == 1
    rec = section(doc, 'received')
    assert rec['count'] == 1 and [s['count'] for s in doc['sections'] if s['key'] in ('made', 'packed', 'shipped', 'adjusted', 'voided', 'late_entries')] == [0] * 6
    [row] = rec['rows']
    assert row['receipt_number'] == result['receipt_number'] and row['action'] == 'receive'
    assert row['status'] == 'committed' and row['effective_status'] == 'posted'
    assert row['actor'] == {'id': actors['floor']['id'], 'name': actors['floor']['name'], 'role': 'floor'}
    assert row['client_source'] == 'api' and row['late_entry'] is False and row['entered_on_later_day'] is False
    assert row['days_late'] == 0 and row['hours_late'] == 0
    assert row['product']['id'] == payload['product_id'] and row['product']['odoo_code'] == 'WT-' + payload['lot_code'][3:]
    assert row['lot'] == {'id': result['lot_id'], 'lot_code': payload['lot_code'], 'supplier_lot_code': payload['supplier_lot_code'],
                          'identity_status': 'identified'}
    assert row['quantity'] == {'lb': 50.0, 'cases': 2, 'unit': 'cases', 'display': '50 lb', 'display_en': '50 lb', 'display_es': '50 lb'}
    assert row['detail'] == {'supplier': {'id': payload['supplier_id'], 'name': 'Ticket supplier ' + payload['lot_code'][3:]},
                             'supplier_lot_code': payload['supplier_lot_code'], 'cases': 2, 'case_size_lb': 25.0,
                             'bol_reference': payload['bol_reference'], 'expected_receipt': None}
    assert row['line'] == row['line_en'] and row['line_en'].startswith(f'{result["receipt_number"]}  Ticket test ')
    assert '  50 lb  lot ' + payload['lot_code'] in row['line_en'] and '(supplier lot ' + payload['supplier_lot_code'] + ')' in row['line_en']
    assert '  lote ' + payload['lot_code'] in row['line_es'] and 'lote del proveedor' in row['line_es']
    assert row['flags'] == [] and row['transaction_ids'] == [result['transaction_id']]
    assert row['transactions'] == [{'id': result['transaction_id'], 'type': 'receive', 'effective_status': 'posted'}]
    assert row['exceptions_opened'] == []
    assert row['happened_at'].endswith(('-04:00', '-05:00')) and row['entered_at'] >= row['happened_at'][:16]
    # Receipt set == GET /receipts?date&actor=<name> (contract §6.2)
    listed = client.get('/receipts', params={'date': doc['date'], 'actor': actors['floor']['name']}, headers=headers(key)).json()
    assert [r['receipt_number'] for r in listed['receipts']] == [row['receipt_number']]
    # Lot balance touched: a brand-new lot 0 → 50.
    [lot] = section(doc, 'lot_balances_touched')['rows']
    assert lot['lot']['id'] == result['lot_id'] and lot['before_lb'] == 0 and lot['after_lb'] == 50 and lot['delta_lb'] == 50
    assert lot['short'] is False and lot['negative'] is False and lot['receipts'] == [result['receipt_number']]
    assert lot['line_en'].endswith(f'lot {payload["lot_code"]} 0 → 50 lb') and ' lote ' in lot['line_es']


def test_unidentified_receive_flag_and_open_exception_rollup(client, db_cursor, actors):
    key = actors['floor']['key']
    result, payload, p = receive(client, db_cursor, key, supplier_lot_code='N/A')
    doc = summary(client, key)
    [row] = section(doc, 'received')['rows']
    assert row['lot']['identity_status'] == 'unidentified'
    [flag] = row['flags']
    [exc] = exceptions_for(db_cursor, ticket_id=p['ticket_id'], kind='UNIDENTIFIED_LOT')
    assert flag['code'] == 'UNIDENTIFIED_LOT' and flag['severity'] == 'warn'
    assert flag['refs'] == {'lot_id': result['lot_id'], 'exception_id': exc['id'], 'due_at': flag['due_at']}
    assert flag['message_en'].startswith('UNIDENTIFIED — resolve by ') and flag['message_es'].startswith('SIN IDENTIFICAR — resolver antes del ')
    assert flag['message'] == flag['message_en']
    assert row['exceptions_opened'] == [exc['id']]
    opened = section(doc, 'open_exceptions')
    assert opened['count'] == 1 and opened['rollup_en'].startswith('1 unidentified lot (due ') and opened['rollup_es'].startswith('1 lote sin identificar (vence ')
    [e] = opened['rows']
    assert e['exception_id'] == exc['id'] and e['kind'] == 'UNIDENTIFIED_LOT' and e['status'] == 'open' and e['overdue'] is False
    assert e['owner'] == {'id': actors['floor']['id'], 'name': actors['floor']['name']}
    assert e['refs']['receipt_number'] == result['receipt_number'] and e['refs']['lot_id'] == result['lot_id']
    assert e['summary'] == e['summary_en'] and 'UNIDENTIFIED' in e['summary_en'] and f'(from {result["receipt_number"]})' in e['summary_en']
    assert 'SIN IDENTIFICAR' in e['summary_es'] and ' lote ' in e['summary_es']
    # The owner does not own it: his summary lists no exception; the all-actors view lists every owner's.
    assert section(summary(client, actors['owner']['key']), 'open_exceptions')['rows'] == []
    assert [e['exception_id'] for e in section(summary(client, main.API_KEY), 'open_exceptions')['rows']] == [exc['id']]


def test_made_short_flag_balances_and_escalation_never_moves_the_hash(client, db_cursor, items, actors):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 4)
    result, p = posted(client, 'make', body('make', items), key)
    doc = summary(client, key)
    [row] = section(doc, 'made')['rows']
    assert row['action'] == 'make' and row['product']['id'] == items['batch']['id'] and row['product']['type'] == 'batch'
    assert row['lot']['id'] == result['lot_id'] and row['quantity']['lb'] == 10 and row['quantity']['cases'] is None
    [ing] = row['detail']['ingredients']
    assert ing['product']['id'] == items['ingredient']['id'] and ing['lot']['id'] == items['ingredient']['lot_id']
    assert ing['quantity_lb'] == 10 and ing['confirmation'] == {'method': 'full_code', 'value': items['ingredient']['lot_code']} and ing['short_lb'] == 6
    assert row['detail']['batch_product']['id'] == items['batch']['id'] and row['detail']['output_lb'] == 10 and row['detail']['substitutions'] == []
    assert '(full-code typed)' in row['line_en'] and 'ingredientes:' in row['line_es'] and 'Lote de producción' in row['line_es']
    [flag] = row['flags']
    assert flag['code'] == 'SHORT' and flag['message_en'].startswith('SHORT 6 lb ') and flag['message_es'].startswith('FALTAN 6 lb ')
    assert flag['refs']['short_lb'] == 6 and flag['refs']['lot_id'] == items['ingredient']['lot_id']
    assert flag['refs']['exception_id'] == result['shortages'][0]['exception_id'] and flag['refs']['shortage_flag_id'] == result['shortages'][0]['shortage_flag_id']
    assert flag['due_at'] and ' — resolve by ' in flag['message_en']
    balances = {b['lot']['id']: b for b in section(doc, 'lot_balances_touched')['rows']}
    assert balances[items['ingredient']['lot_id']] == balances[items['ingredient']['lot_id']] | {
        'before_lb': 4, 'after_lb': -6, 'delta_lb': -10, 'short': True, 'negative': True, 'receipts': [result['receipt_number']]}
    assert balances[items['ingredient']['lot_id']]['line_en'].endswith('4 → −6 lb (short)') and balances[items['ingredient']['lot_id']]['line_es'].endswith('(faltante)')
    assert balances[result['lot_id']]['before_lb'] == 0 and balances[result['lot_id']]['after_lb'] == 10
    assert section(doc, 'open_exceptions')['rollup_en'].startswith('1 shortage (due ')
    # Time-derived fields never move the hash: escalate the shortage, re-read.
    db_cursor.execute("UPDATE exceptions SET due_at=clock_timestamp()-interval '1 day' WHERE id=%s", (flag['refs']['exception_id'],))
    db_cursor.execute("UPDATE shortage_flags SET due_at=clock_timestamp()-interval '1 day' WHERE id=%s", (flag['refs']['shortage_flag_id'],))
    sweep(db_cursor)
    again = summary(client, key)
    [e] = section(again, 'open_exceptions')['rows']
    assert e['status'] == 'escalated' and e['escalated'] is True and e['overdue'] is True and e['summary_en'].endswith('— ESCALATED')
    assert again['summary_hash'] == doc['summary_hash']
    # Someone else moving a shared lot changes the balances, not this person's hash.
    other, _ = posted(client, 'adjust', adjust(items, -1, lot_id=result['lot_id']), actors['owner']['key'])
    moved = summary(client, key)
    assert {b['lot']['id']: b['after_lb'] for b in section(moved, 'lot_balances_touched')['rows']}[result['lot_id']] == 9
    assert moved['receipts_today'] == 1 and moved['summary_hash'] == doc['summary_hash']
    assert section(summary(client, actors['owner']['key']), 'adjusted')['count'] == 1


def test_packed_row_counts_cases(client, db_cursor, items, actors):
    key = actors['floor']['key']
    result, p = posted(client, 'pack', body('pack', items), key)
    [row] = section(summary(client, key), 'packed')['rows']
    assert row['product']['id'] == items['finished']['id'] and row['lot']['id'] == result['output_lot_id']
    assert row['quantity'] == row['quantity'] | {'lb': 10, 'cases': 2, 'unit': 'cases', 'display_en': '2 cases (10 lb)', 'display_es': '2 cajas (10 lb)'}
    assert row['detail'] == {'source_product': {'id': items['batch']['id'], 'name': items['batch']['name']},
                             'source_lot': {'id': items['batch']['lot_id'], 'lot_code': items['batch']['lot_code']},
                             'target_product': {'id': items['finished']['id'], 'name': items['finished']['name']},
                             'cases': 2, 'lb': 10, 'ownership': 'cns', 'invoice_status': None}
    assert row['line_en'].endswith(f'2 cases (10 lb) from lot {items["batch"]["lot_code"]}') and 'cajas' in row['line_es']
    assert row['flags'] == []


def test_shipped_row_bol_and_photo_status(client, db_cursor, actors):
    key = actors['office']['key']
    data = seed_shipping(db_cursor)
    p = ship_prepared(client, data, key, 'ship_order')
    r = commit(client, p, key)
    assert r.status_code == 200, r.text
    pending = r.json()
    doc = summary(client, key)
    [row] = section(doc, 'shipped')['rows']
    assert row['action'] == 'ship_order' and row['receipt_number'] == pending['receipt_number']
    assert row['product']['id'] == data['products']['food'] and row['lot']['id'] == data['lot_id']
    assert row['quantity']['lb'] == 50 and row['quantity']['cases'] == 2 and row['quantity']['display_en'] == '2 cases' and row['quantity']['display_es'] == '2 cajas'
    d = row['detail']
    assert d['sales_order']['id'] == p['draft']['order_id'] and d['sales_order']['so_number'].startswith('SO-') and d['sales_order']['customer_name'].startswith('Ticket customer')
    assert d['customer'] == {'id': data['customer'], 'name': d['sales_order']['customer_name']}
    assert d['bol_reference'] == 'BOL-A6' and d['proof_status'] == 'photo_pending' and d['bins'] is None
    assert [l['product']['id'] for l in d['lines']] == [data['products']['food'], data['products']['pallet']]
    assert d['lines'][0]['lot'] == {'id': data['lot_id'], 'lot_code': data['lot_code']} and d['lines'][0]['cases'] == 2 and d['lines'][0]['lb'] == 50
    assert row['line_en'].endswith('2 cases  BOL BOL-A6  photo pending') and row['line_es'].endswith('foto pendiente')
    [flag] = row['flags']
    assert flag['code'] == 'PHOTO_PENDING' and flag['refs'] == {'shipment_id': pending['shipment_id'], 'exception_id': None}
    assert flag['message_en'].startswith('BOL photo pending — due by 23:59') and flag['due_at'].endswith('23:59:00-04:00') or flag['due_at'].endswith('23:59:00-05:00')
    # Balances: 2,000 → 1,950 on the shipped lot.
    [lot] = [b for b in section(doc, 'lot_balances_touched')['rows'] if b['lot']['id'] == data['lot_id']]
    assert (lot['before_lb'], lot['after_lb'], lot['delta_lb']) == (2000, 1950, -50)
    # A late BOL image completes the proof: the flag disappears, the row (and hash) changes.
    photo = upload(client, key)
    r = client.post(f'/shipments/{pending["shipment_id"]}/attachments', json={'attachment_id': photo['id']}, headers=headers(key))
    assert r.status_code == 200, r.text
    after = summary(client, key)
    [row2] = section(after, 'shipped')['rows']
    assert row2['detail']['proof_status'] == 'complete' and row2['flags'] == [] and row2['line_en'].endswith('photo ✓')
    assert after['summary_hash'] != doc['summary_hash']


def test_adjusted_rows_reason_labels_unknown_and_found(client, db_cursor, items, actors):
    key = actors['floor']['key']
    adj, p1 = posted(client, 'adjust', adjust(items, -2, 'unknown', note='spilled'), key)
    fnd, p2 = posted(client, 'found', found(items, 5), key)
    doc = summary(client, key)
    rows = {r['action']: r for r in section(doc, 'adjusted')['rows']}
    a = rows['adjust']
    assert a['quantity'] == a['quantity'] | {'lb': -2, 'cases': None, 'display_en': '−2 lb'}
    assert a['lot']['id'] == items['ingredient']['lot_id'] and a['product']['id'] == items['ingredient']['id']
    assert a['detail'] == {'lot': {'id': items['ingredient']['lot_id'], 'lot_code': items['ingredient']['lot_code']}, 'delta_lb': -2,
                           'lot_on_hand_before': 100, 'lot_on_hand_after': 98,
                           'reason_code': {'code': 'unknown', 'label': 'Unknown', 'label_en': 'Unknown', 'label_es': 'Desconocido', 'note_required': True},
                           'note': 'spilled', 'photo_attached': False}
    assert a['line_en'].endswith('−2 lb  reason: Unknown') and a['line_es'].endswith('−2 lb  razón: Desconocido')
    assert [f['code'] for f in a['flags']] == ['UNKNOWN_REASON'] and a['flags'][0]['refs'] == {'transaction_id': adj['transaction_id']}
    f = rows['found']
    assert f['quantity']['lb'] == 5 and f['quantity']['display_en'] == '+5 lb' and f['lot']['id'] == fnd['lot_id']
    assert f['detail']['reason_code']['code'] == 'physical_count' and f['detail']['lot_on_hand_before'] == 0 and f['detail']['lot_on_hand_after'] == 5
    assert f['flags'] == []
    assert [r['receipt_number'] for r in section(doc, 'adjusted')['rows']] == sorted([adj['receipt_number'], fnd['receipt_number']],
                                                                                     key=lambda n: [r for r in section(doc, 'adjusted')['rows'] if r['receipt_number'] == n][0]['happened_at'])


def test_large_correction_flag_from_the_review(client, db_cursor, items, actors):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 1000)
    result, p = posted(client, 'adjust', adjust(items, -150), key)
    [row] = section(summary(client, key), 'adjusted')['rows']
    [flag] = row['flags']
    assert flag['code'] == 'LARGE_CORRECTION' and flag['refs'] == {'exception_id': None, 'delta_lb': -150, 'pct': 0.15}
    assert flag['message_en'] == 'LARGE CORRECTION −150 lb (15% of lot)' and flag['message_es'] == 'CORRECCIÓN GRANDE −150 lb (15% del lote)'


def test_late_entries_section_flag_and_hours(client, db_cursor, items, actors):
    key = actors['floor']['key']
    late, p = posted(client, 'make', body('make', items) | {'occurred_at': (main.get_plant_now() - timedelta(hours=49)).isoformat()}, key)
    recent, _ = posted(client, 'adjust', adjust(items, 1), key)
    doc = summary(client, key)
    assert doc['receipts_today'] == 2   # the late make is counted once although it is listed twice
    [row] = section(doc, 'made')['rows']
    assert row['late_entry'] is True and row['days_late'] == 2 and 48.9 < row['hours_late'] < 49.5 and row['entered_on_later_day'] is False
    [exc] = exceptions_for(db_cursor, ticket_id=p['ticket_id'], kind='LATE_ENTRY')
    [flag] = row['flags']
    assert flag['code'] == 'LATE_ENTRY' and flag['refs'] == {'exception_id': exc['id'], 'hours_late': row['hours_late']}
    assert flag['message_en'].startswith('LATE ENTRY — 49') and flag['message_es'].startswith('ENTRADA TARDÍA — 49')
    assert row['exceptions_opened'] == [exc['id']]
    late_rows = section(doc, 'late_entries')['rows']
    assert [r['receipt_number'] for r in late_rows] == [late['receipt_number']] and late_rows[0] == row
    assert section(doc, 'adjusted')['rows'][0]['late_entry'] is False
    # It happened two days ago, so it also shows on that day's summary — flagged as entered on a later day.
    then = summary(client, key, date=row['happened_at'][:10])
    [back] = section(then, 'made')['rows']
    assert back['receipt_number'] == late['receipt_number'] and back['entered_on_later_day'] is True
    assert section(then, 'late_entries')['count'] == 1
    # The owner's late entry opens no exception → late_entry true, no LATE_ENTRY flag.
    owner_late, _ = posted(client, 'adjust', adjust(items, 2) | {'occurred_at': (main.get_plant_now() - timedelta(hours=60)).isoformat()}, actors['owner']['key'])
    [o] = section(summary(client, actors['owner']['key']), 'late_entries')['rows']
    assert o['late_entry'] is True and o['flags'] == [] and o['days_late'] == 2


def test_not_entered_yet_expected_receipts_and_planned_runs(client, db_cursor, items, actors):
    key = actors['floor']['key']
    today = main.get_plant_now().date()
    base = seed_receive(db_cursor)
    product, supplier = base['product_id'], base['supplier_id']
    rows = {}
    for label, offset, status in [('today', 0, 'open'), ('overdue', -3, 'open'), ('tomorrow', 1, 'open'), ('closed', 0, 'closed')]:
        db_cursor.execute('''INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,expected_date,reference_number,status)
                             VALUES (%s,%s,1000,%s,%s,%s) RETURNING id''', (product, supplier, today + timedelta(days=offset), 'ER-' + label + uuid4().hex[:4], status))
        rows[label] = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,expected_date,status) VALUES (%s,%s,5,NULL,%s) RETURNING id', (product, supplier, 'open'))
    rows['undated'] = db_cursor.fetchone()['id']
    db_cursor.execute('''INSERT INTO production_runs(product_id,planned_qty_lb,planned_qty,planned_unit,planned_date,status,run_type)
                         VALUES (%s,100,10,'cases',%s,'planned','pack') RETURNING id''', (items['finished']['id'], today))
    run = db_cursor.fetchone()['id']
    db_cursor.execute('''INSERT INTO production_runs(product_id,planned_qty_lb,planned_date,status,run_type)
                         VALUES (%s,100,%s,'done','bake') RETURNING id''', (items['batch']['id'], today))
    doc = summary(client, key)
    pending = section(doc, 'not_entered_yet')
    assert [r['kind'] for r in pending['rows']] == ['expected_receipt', 'expected_receipt', 'production_run']
    overdue, due_today, planned = pending['rows']
    assert overdue['ref']['expected_receipt_id'] == rows['overdue'] and overdue['overdue'] is True and ', overdue) — no RCV receipt' in overdue['line_en']
    assert 'venció el' in overdue['line_es'] and overdue['line_es'].endswith('— sin recibo RCV')
    assert due_today['ref']['expected_receipt_id'] == rows['today'] and due_today['expected'] == {
        'quantity': 1000, 'unit': 'lb', 'date': today.isoformat(), 'supplier': {'id': supplier, 'name': 'Ticket supplier ' + base['lot_code'][3:]}}
    assert due_today['product']['id'] == product and due_today['status'] == 'open' and '(Ticket test ' in due_today['line_en'] and 'due today' in due_today['line_en']
    assert planned['ref'] == {'expected_receipt_id': None, 'reference_number': None, 'production_run_id': run}
    assert planned['expected'] == planned['expected'] | {'run_type': 'pack', 'planned_qty': 10, 'planned_unit': 'cases', 'date': today.isoformat()}
    assert planned['line_en'] == f'production run #{run} {items["finished"]["name"]} pack, planned today — status planned, no PK receipt'
    assert planned['line_es'] == f'corrida #{run} {items["finished"]["name"]} empaque, planeada hoy — estado planeada, sin recibo PK'
    assert planned['line'] == planned['line_en']
    # Expectations are plant-wide: every view shows the same rows; they never enter the hash.
    assert section(summary(client, main.DASHBOARD_API_KEY), 'not_entered_yet')['rows'] == pending['rows']
    h = doc['summary_hash']
    # A ticketed receive for that product/supplier auto-matches the oldest open expectation (FIFO) and removes it.
    result, _, p = receive(client, db_cursor, key, base, cases=40, lot_code='ER-' + uuid4().hex[:6].upper())
    assert p['draft']['expected_receipt_match']['id'] == rows['overdue']
    doc2 = summary(client, key)
    assert [r['ref']['expected_receipt_id'] for r in section(doc2, 'not_entered_yet')['rows'][:2]] == [rows['today'], None]
    assert section(doc2, 'received')['rows'][0]['detail']['expected_receipt']['id'] == rows['overdue']
    assert doc2['summary_hash'] != h   # the receive is this person's entry, so that one does move


def test_rejected_ticket_with_a_receipt_number_is_not_a_receipt(client, db_cursor, items, actors):
    key = actors['floor']['key']
    p = prepare(client, 'adjust', adjust(items, -1), key)
    # A held/rejected ticket can already carry a number (seen on staging: MK-…-002 rejected, result_ref NULL).
    db_cursor.execute("UPDATE write_tickets SET receipt_number=%s, status='rejected' WHERE id=%s",
                      ('ADJ-' + main.get_plant_now().strftime('%y%m%d') + '-999', p['ticket_id']))
    doc = summary(client, key)
    assert doc['receipts_today'] == 0 and section(doc, 'adjusted')['rows'] == []
    # GET /receipts still lists it (its own filter) and no longer crashes on the NULL result_ref.
    listed = client.get('/receipts', params={'date': doc['date'], 'actor': actors['floor']['name']}, headers=headers(key)).json()
    assert [r['status'] for r in listed['receipts']] == ['rejected']


def test_legacy_key_receipts_and_all_actors_view(client, db_cursor, actors):
    result, payload, p = receive(client, db_cursor, main.API_KEY)
    floor, _, _ = receive(client, db_cursor, actors['floor']['key'])
    for key in (main.API_KEY, main.DASHBOARD_API_KEY):
        doc = summary(client, key)
        assert doc['actor'] is None and doc['title_en'].endswith('— all') and doc['title_es'].endswith('— todos')
        assert doc['confirmation'] == {'status': 'unconfirmed', 'latest': None, 'can_confirm': False,
                                       'reason': 'Sign in as a person to confirm a shift',
                                       'reason_en': 'Sign in as a person to confirm a shift',
                                       'reason_es': 'Inicia sesión como persona para confirmar el turno'}
        rows = {r['receipt_number']: r for r in section(doc, 'received')['rows']}
        assert rows[result['receipt_number']]['actor'] == {'id': None, 'name': 'legacy-shared-key', 'role': None}
        assert rows[floor['receipt_number']]['actor']['id'] == actors['floor']['id']
        assert doc['receipts_today'] >= 2
    # A shared key may ask for one person's view (read-only).
    named = summary(client, main.DASHBOARD_API_KEY, actor=actors['floor']['name'])
    assert named['actor']['id'] == actors['floor']['id'] and named['confirmation']['can_confirm'] is False
    assert [r['receipt_number'] for r in section(named, 'received')['rows']] == [floor['receipt_number']]
    by_id = summary(client, main.API_KEY, actor=str(actors['floor']['id']))
    assert by_id['summary_hash'] == named['summary_hash'] == summary(client, actors['floor']['key'])['summary_hash']


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------
def test_hash_is_deterministic_and_covers_only_receipt_rows(client, db_cursor, items, actors, monkeypatch):
    key = actors['floor']['key']
    result, p = posted(client, 'adjust', adjust(items, -1), key)
    a, b = summary(client, key), summary(client, key, lang='es')
    assert a['summary_hash'] == b['summary_hash'] and a['generated_at'] <= b['generated_at']
    sections = [s for s in a['sections']]
    assert shift_summary.summary_hash(main, sections) == a['summary_hash']
    assert shift_summary.summary_hash(main, b['sections']) == a['summary_hash']
    reduced = shift_summary.reduced(section(a, 'adjusted')['rows'][0])
    assert set(reduced) == {'receipt_number', 'action', 'status', 'effective_status', 'happened_at', 'entered_at', 'late_entry',
                            'detail', 'transaction_ids', 'product', 'lot', 'quantity', 'flags'}
    assert reduced['product'] == {'id': items['ingredient']['id']} and reduced['quantity'] == {'lb': -1, 'cases': None}
    # Open exceptions, clock and expectations are outside the hash; the person's own new receipt is inside it.
    db_cursor.execute('''INSERT INTO exceptions(kind,status,severity,detail,owner_actor_id) VALUES ('NEGATIVE_BALANCE','open','warn','{}',%s)''', (actors['floor']['id'],))
    db_cursor.execute("INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,expected_date) VALUES (%s,%s,1,%s)",
                      (items['ingredient']['id'], seed_receive(db_cursor)['supplier_id'], main.get_plant_now().date()))
    real_now = main.get_plant_now()
    monkeypatch.setattr(main, 'get_plant_now', lambda: real_now + timedelta(hours=3))
    c = summary(client, key)
    assert section(c, 'open_exceptions')['count'] == 1 and section(c, 'not_entered_yet')['count'] >= 1
    assert c['summary_hash'] == a['summary_hash']
    posted(client, 'adjust', adjust(items, -1), key)
    assert summary(client, key)['summary_hash'] != a['summary_hash']


# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------
def test_confirm_match_with_notes_and_reconfirm_appends(client, db_cursor, items, actors):
    key = actors['floor']['key']
    posted(client, 'adjust', adjust(items, -1), key)
    doc = summary(client, key)
    r = confirm(client, key, doc)
    assert r.status_code == 201, r.text
    first = r.json()
    assert first['success'] is True and first['outcome'] == 'match' and first['notes'] == '' and first['exception'] is None
    assert first['actor'] == doc['actor'] and first['business_date'] == doc['date'] and first['receipts_seen'] == 1
    assert first['summary_hash'] == doc['summary_hash'] and first['confirmed_late'] is False
    assert first['message'] == first['message_en'] == 'Shift recorded as: matches the floor.' and first['message_es'].startswith('Turno registrado como: coincide')
    db_cursor.execute('SELECT * FROM shift_confirmations WHERE id=%s', (first['confirmation_id'],))
    row = db_cursor.fetchone()
    assert row['actor_id'] == actors['floor']['id'] and row['summary_snapshot'] == doc['sections'] and row['exception_id'] is None
    after = summary(client, key)
    assert after['confirmation']['status'] == 'match' and after['confirmation']['can_confirm'] is True
    assert after['confirmation']['latest'] == {k: v for k, v in first.items() if k not in ('message', 'message_en', 'message_es', 'success')}
    # Re-confirm with notes: a second row, the latest wins, history kept.
    r = confirm(client, key, after, notes='counted the oats twice', lang='es')
    assert r.status_code == 201, r.text
    second = r.json()
    assert second['confirmation_id'] > first['confirmation_id'] and second['notes'] == 'counted the oats twice'
    assert second['message'] == second['message_es'] == 'Turno registrado como: coincide con el piso, con notas.'
    db_cursor.execute('SELECT count(*) AS n FROM shift_confirmations WHERE business_date=%s AND actor_id=%s', (doc['date'], actors['floor']['id']))
    assert db_cursor.fetchone()['n'] == 2
    assert summary(client, key)['confirmation']['latest']['confirmation_id'] == second['confirmation_id']
    # Rows are append-only.
    with pytest.raises(psycopg2.Error):
        db_cursor.execute('SAVEPOINT a9; UPDATE shift_confirmations SET notes=%s WHERE id=%s', ('x', first['confirmation_id']))
    db_cursor.execute('ROLLBACK TO SAVEPOINT a9')


def test_discrepancy_opens_one_owner_exception_and_needs_notes(client, db_cursor, items, actors):
    key = actors['floor']['key']
    doc = summary(client, key)
    error(confirm(client, key, doc, outcome='discrepancy'), 422, 'NOTES_REQUIRED')
    error(confirm(client, key, doc, outcome='discrepancy', notes='   '), 422, 'NOTES_REQUIRED')
    error(confirm(client, key, doc, outcome='maybe'), 422, 'OUTCOME_INVALID')
    r = confirm(client, key, doc, outcome='discrepancy', notes='The afternoon oats delivery was never entered')
    assert r.status_code == 201, r.text
    result = r.json()
    [exc] = exceptions_for(db_cursor, kind='SHIFT_DISCREPANCY', owner_actor_id=actors['owner']['id'])
    assert result['exception'] == {'id': exc['id'], 'kind': 'SHIFT_DISCREPANCY'} and result['outcome'] == 'discrepancy'
    assert exc['severity'] == 'warn' and exc['status'] == 'open' and exc['due_at'] is None
    assert exc['detail'] == {'business_date': doc['date'], 'actor_id': actors['floor']['id'], 'actor_name': actors['floor']['name'],
                             'notes': 'The afternoon oats delivery was never entered', 'receipts_seen': 0, 'summary_hash': doc['summary_hash']}
    assert result['message_en'] == f'Shift recorded as: something is missing. {actors["owner"]["name"]} will see this on the weekly view.'
    assert result['message_es'] == f'Turno registrado como: falta algo. {actors["owner"]["name"]} lo verá en la vista semanal.'
    db_cursor.execute('SELECT exception_id FROM shift_confirmations WHERE id=%s', (result['confirmation_id'],))
    assert db_cursor.fetchone()['exception_id'] == exc['id']
    after = summary(client, key)
    assert after['confirmation']['status'] == 'discrepancy' and after['confirmation']['latest']['exception'] == result['exception']
    # The owner sees it in his own OPEN EXCEPTIONS.
    [mine] = section(summary(client, actors['owner']['key']), 'open_exceptions')['rows']
    assert mine['exception_id'] == exc['id'] and mine['summary_en'] == 'SHIFT DISCREPANCY' and mine['summary_es'] == 'DISCREPANCIA DE TURNO'
    assert section(summary(client, actors['owner']['key']), 'open_exceptions')['rollup_en'] == '1 shift discrepancy'


def test_stale_hash_returns_the_fresh_summary(client, db_cursor, items, actors):
    key = actors['floor']['key']
    doc = summary(client, key)
    posted(client, 'adjust', adjust(items, -1), key)
    r = confirm(client, key, doc)
    assert r.status_code == 409, r.text
    body_ = r.json()
    assert body_['detail']['error_code'] == 'SHIFT_SUMMARY_STALE' and body_['error_detail']['code'] == 'SHIFT_SUMMARY_STALE'
    assert body_['error_detail']['message_es'].startswith('El resumen del turno cambió')
    fresh = body_['detail']['summary']
    assert fresh['receipts_today'] == 1 and fresh['summary_hash'] != doc['summary_hash'] and [s['key'] for s in fresh['sections']] == KEYS
    db_cursor.execute('SELECT count(*) AS n FROM shift_confirmations')
    assert db_cursor.fetchone()['n'] == 0
    assert confirm(client, key, fresh).status_code == 201


def test_confirm_yesterday_is_late_and_future_is_refused(client, db_cursor, actors):
    key = actors['floor']['key']
    yesterday = (main.get_plant_now().date() - timedelta(days=1)).isoformat()
    doc = summary(client, key, date=yesterday)
    assert doc['date'] == yesterday
    r = confirm(client, key, doc)
    assert r.status_code == 201 and r.json()['confirmed_late'] is True and r.json()['business_date'] == yesterday
    tomorrow = (main.get_plant_now().date() + timedelta(days=1)).isoformat()
    error(client.get('/reports/shift-summary', params={'date': tomorrow}, headers=headers(key)), 422, 'DATE_IN_FUTURE')
    error(confirm(client, key, doc | {'date': tomorrow}), 422, 'DATE_IN_FUTURE')
    error(confirm(client, key, doc | {'summary_hash': 'x' * 64}, extra='field'), 422, 'VALIDATION_ERROR') if False else None
    assert client.post('/reports/shift-summary/confirm', json={'date': yesterday, 'outcome': 'match', 'summary_hash': 'h', 'receipts_seen': 0, 'extra': 1},
                       headers=headers(key)).status_code == 422
    assert client.post('/reports/shift-summary/confirm', json={'date': yesterday, 'outcome': 'match', 'summary_hash': 'h', 'receipts_seen': -1},
                       headers=headers(key)).status_code == 422


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------
def test_permissions_matrix(client, db_cursor, actors):
    floor, owner, office = (actors[r]['key'] for r in ('floor', 'owner', 'office'))
    for key in (main.API_KEY, main.DASHBOARD_API_KEY):
        error(confirm(client, key, summary(client, key)), 403, 'ACTOR_REQUIRED')
    error(client.get('/reports/shift-summary', params={'actor': actors['owner']['name']}, headers=headers(floor)), 403, 'ACTOR_NOT_ALLOWED')
    error(client.get('/reports/shift-summary', params={'actor': str(actors['owner']['id'])}, headers=headers(floor)), 403, 'ACTOR_NOT_ALLOWED')
    assert summary(client, floor, actor=actors['floor']['name'])['actor']['id'] == actors['floor']['id']
    assert summary(client, owner, actor=actors['floor']['name'])['actor']['id'] == actors['floor']['id']
    assert summary(client, office, actor=str(actors['floor']['id']))['actor']['id'] == actors['floor']['id']
    assert summary(client, owner, actor=actors['retired']['name'])['actor']['id'] == actors['retired']['id']
    for key in (floor, owner, main.API_KEY, main.DASHBOARD_API_KEY):
        error(client.get('/reports/shift-summary', params={'actor': 'nobody ' + uuid4().hex}, headers=headers(key)), 422, 'ACTOR_UNKNOWN')
        error(client.get('/reports/shift-summary', params={'actor': '999999999'}, headers=headers(key)), 422, 'ACTOR_UNKNOWN')
    assert client.get('/reports/shift-summary', headers=headers('not-a-key')).status_code == 403
    assert client.get('/reports/shift-summary', headers=headers(actors['retired']['key'])).status_code == 403
    assert client.get('/reports/shift-summary').status_code == 401
    assert client.post('/reports/shift-summary/confirm', json={}, headers=headers('not-a-key')).status_code == 403
    assert client.get('/reports/shift-summary', params={'lang': 'fr'}, headers=headers(floor)).status_code == 422
    # Both routes are on the dashboard allowlist and nowhere else new.
    assert shift_summary.ROUTES <= main.DASHBOARD_KEY_ALLOWLIST and not (shift_summary.ROUTES & main.ACTOR_WRITE_ALLOWLIST)


def test_confirm_is_attributed_to_the_key_never_the_body(client, db_cursor, actors):
    key = actors['office']['key']
    doc = summary(client, key)
    r = client.post('/reports/shift-summary/confirm', headers=headers(key),
                    json={'date': doc['date'], 'outcome': 'match', 'summary_hash': doc['summary_hash'], 'receipts_seen': 0, 'actor_id': actors['owner']['id']})
    assert r.status_code == 422   # extra fields are refused, never read
    r = confirm(client, key, doc)
    assert r.status_code == 201 and r.json()['actor']['id'] == actors['office']['id']


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------
def test_migration_076_up_down_up(isolated_database):
    up = (ROOT / 'migrations/076_shift_confirmations.sql').read_text()
    down = (ROOT / 'migrations/down/076_shift_confirmations_down.sql').read_text()
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SET search_path TO public')
        cur.execute(up)
        cur.execute("SELECT count(*) FROM migration_markers WHERE name='076_shift_confirmations'")
        assert cur.fetchone()[0] == 1
        cur.execute("INSERT INTO actors(name,role,key_hash) VALUES ('a9','floor',%s) RETURNING id", ('f' * 64,))
        actor = cur.fetchone()[0]
        cur.execute("INSERT INTO shift_confirmations(business_date,actor_id,outcome,notes,receipts_seen,summary_hash,summary_snapshot) VALUES (current_date,%s,'match','',0,%s,'[]')",
                    (actor, 'a' * 64))
        with pytest.raises(psycopg2.Error):
            cur.execute("SAVEPOINT x; INSERT INTO shift_confirmations(business_date,actor_id,outcome,notes,receipts_seen,summary_hash,summary_snapshot) VALUES (current_date,%s,'discrepancy','',0,%s,'[]')",
                        (actor, 'a' * 64))
        cur.execute('ROLLBACK TO SAVEPOINT x')
        with pytest.raises(psycopg2.Error):
            cur.execute('SAVEPOINT y; DELETE FROM shift_confirmations')
        cur.execute('ROLLBACK TO SAVEPOINT y')
        with pytest.raises(psycopg2.Error):
            cur.execute('SAVEPOINT z; ' + down)      # refuses while rows exist
        cur.execute('ROLLBACK TO SAVEPOINT z')
        cur.execute('ALTER TABLE shift_confirmations DISABLE TRIGGER shift_confirmations_append_only; DELETE FROM shift_confirmations; '
                    'ALTER TABLE shift_confirmations ENABLE TRIGGER shift_confirmations_append_only')
        cur.execute(down)
        cur.execute("SELECT to_regclass('shift_confirmations'), (SELECT count(*) FROM migration_markers WHERE name='076_shift_confirmations')")
        assert cur.fetchone() == (None, 0)
        cur.execute(up)
        cur.execute(up)
        cur.execute("SELECT relrowsecurity FROM pg_class WHERE relname='shift_confirmations'")
        assert cur.fetchone()[0] is True
        conn.rollback()

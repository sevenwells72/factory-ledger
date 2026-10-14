"""A9: GET /reports/shift-summary + POST /reports/shift-summary/confirm against the
frozen contract (docs/contracts/shift-summary.md). Real HTTP through the TestClient
on the rolled-back test transaction; every section is exercised with a ticketed
receipt, the hash rules of §1.3 are asserted directly, and the permission matrix
of §1.1 / §3 is walked key by key."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg2
from psycopg2.extras import RealDictCursor
import pytest

import main
import shift_summary
import write_tickets as wt
from scripts.exceptions_sweep import sweep
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, error, seed as seed_receive, isolated_database  # noqa: F401
from tests.test_write_tickets_part2 import items, body, prepare  # noqa: F401
from tests.test_exceptions_enforcement import set_balance, adjust, found, exceptions_for, seed_photo, race_actors, Race
from tests.test_ship_tickets import seed_shipping, prepared as ship_prepared, upload, storage  # noqa: F401
from tests.test_order_tickets import catalog, create_order  # noqa: F401
from tests.test_a7_review_fixes import incoming  # noqa: F401
from tests.test_lot_confirmation import move as move_lot

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
                           'note': 'spilled', 'photo_attached': False,
                           'kosher_lots': [{'lot_id': items['ingredient']['lot_id'], 'kosher_tier': 'regular',
                                            'kosher_attested_by_actor_id': None, 'kosher_attested_at': None}]}
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
    assert flag['code'] == 'LARGE_CORRECTION'
    assert flag['refs'] == {'exception_id': None, 'delta_lb': -150, 'pct': 0.15, 'resolution': None, 'approved_by_actor_id': None}
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
    assert back['receipt_number'] == late['receipt_number'] and back['entered_on_later_day'] is True and back['late_entry'] is True
    # ...but it is not a late entry OF that day: §2.6 lists receipts ENTERED on the queried day (review item 3).
    assert section(then, 'late_entries')['count'] == 0 and section(then, 'late_entries')['rows'] == []
    assert then['receipts_today'] == 1
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
                            'detail', 'transaction_ids', 'product', 'lot', 'quantity', 'flags', 'kosher_tier',
                            'kosher_attested_by_actor_id', 'kosher_attested_at', 'transaction_kosher_downgrade'}
    assert reduced['product'] == {'id': items['ingredient']['id']} and reduced['quantity'] == {'lb': -1, 'cases': None}
    # Open exceptions, clock and expectations are outside the hash; the person's own new receipt is inside it.
    db_cursor.execute('''INSERT INTO exceptions(kind,status,severity,detail,owner_actor_id) VALUES ('NEGATIVE_BALANCE','open','warn','{}',%s)''', (actors['floor']['id'],))
    db_cursor.execute("INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,expected_date) VALUES (%s,%s,1,%s)",
                      (items['ingredient']['id'], seed_receive(db_cursor)['supplier_id'], main.get_plant_now().date()))
    real_now = main.get_plant_now()
    monkeypatch.setattr(main, 'get_plant_now', lambda: real_now + timedelta(hours=3))
    c = summary(client, key, date=a['date'])  # keep the requested day when +3h crosses midnight
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


# ---------------------------------------------------------------------------
# Fix round 2026-10-13 (Codex review of PR #102, comment 6149165326)
# ---------------------------------------------------------------------------
def test_repeated_discrepancy_reuses_the_open_exception_until_the_owner_resolves_it(client, db_cursor, items, actors):
    """Item 1: one SHIFT_DISCREPANCY per (actor, day) while it is open; every confirmation still appends."""
    key = actors['floor']['key']
    doc = summary(client, key)
    first = confirm(client, key, doc, outcome='discrepancy', notes='oats delivery missing')
    assert first.status_code == 201, first.text
    exc_id = first.json()['exception']['id']
    # The same POST again (a retry) and a different note later: no second exception.
    again = confirm(client, key, doc, outcome='discrepancy', notes='oats delivery missing')
    third = confirm(client, key, doc, outcome='discrepancy', notes='also the pallet move')
    assert again.status_code == 201 and third.status_code == 201, (again.text, third.text)
    assert again.json()['exception'] == third.json()['exception'] == {'id': exc_id, 'kind': 'SHIFT_DISCREPANCY'}
    assert len({first.json()['confirmation_id'], again.json()['confirmation_id'], third.json()['confirmation_id']}) == 3
    [exc] = exceptions_for(db_cursor, kind='SHIFT_DISCREPANCY', owner_actor_id=actors['owner']['id'])
    assert exc['id'] == exc_id and exc['status'] == 'open'
    assert exc['detail']['notes'] == 'also the pallet move' and exc['detail']['confirmations'] == 3 and exc['detail']['reconfirmed_at']
    assert exc['detail']['business_date'] == doc['date'] and exc['detail']['actor_id'] == actors['floor']['id']
    db_cursor.execute('SELECT exception_id FROM shift_confirmations WHERE business_date=%s AND actor_id=%s ORDER BY id', (doc['date'], actors['floor']['id']))
    assert [r['exception_id'] for r in db_cursor.fetchall()] == [exc_id] * 3
    assert section(summary(client, actors['owner']['key']), 'open_exceptions')['rollup_en'] == '1 shift discrepancy'
    # A match in between does not close it; another person's discrepancy for the same day is their own.
    assert confirm(client, key, doc).status_code == 201
    other = confirm(client, actors['office']['key'], summary(client, actors['office']['key']), outcome='discrepancy', notes='mine')
    assert other.status_code == 201 and other.json()['exception']['id'] != exc_id
    # Once the owner resolves it, a new discrepancy opens a new exception.
    db_cursor.execute("UPDATE exceptions SET status='resolved', resolved_at=clock_timestamp(), resolution_kind='acknowledged' WHERE id=%s", (exc_id,))
    fresh = confirm(client, key, summary(client, key), outcome='discrepancy', notes='still missing')
    assert fresh.status_code == 201 and fresh.json()['exception']['id'] > exc_id
    assert len(exceptions_for(db_cursor, kind='SHIFT_DISCREPANCY', owner_actor_id=actors['owner']['id'])) == 3


def test_concurrent_discrepancies_append_two_confirmations_and_open_one_exception(isolated_database, monkeypatch):
    """Item 1, concurrency: two simultaneous "something is missing" taps serialise on the (actor, day) lock."""
    keys, items_, ids = race_actors(isolated_database)
    race = Race(isolated_database)
    monkeypatch.setattr(main, 'get_db_connection', race.connection)
    main._reset_actor_cache()
    with TestClient(main.app) as http:
        doc = summary(http, keys['floor'])
        race.on = True
        post = lambda: confirm(http, keys['floor'], doc, outcome='discrepancy', notes='race')
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [f.result(timeout=30) for f in [executor.submit(post) for _ in range(2)]]
        race.on = False
        assert [r.status_code for r in results] == [201, 201], [r.text for r in results]
        a, b = (r.json() for r in results)
        assert a['confirmation_id'] != b['confirmation_id'] and a['exception'] == b['exception']
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT count(*) AS n FROM exceptions WHERE kind='SHIFT_DISCREPANCY' AND detail->>'actor_id'=%s", (str(ids['floor']),))
            assert cur.fetchone()['n'] == 1
            cur.execute('SELECT count(*) AS n FROM shift_confirmations WHERE actor_id=%s', (ids['floor'],))
            assert cur.fetchone()['n'] == 2


def test_metadata_receipts_are_listed_outside_the_sections_never_as_adjustments(client, db_cursor, items, catalog, incoming, actors):
    """Item 2: ORD / ER / LOT receipts are receipts (GET /receipts parity) but not ledger postings."""
    office, floor = actors['office']['key'], actors['floor']['key']
    order = create_order(client, catalog, office)
    er = commit(client, client.post('/expected-receipts/prepare', json=incoming, headers=headers(office)).json(), office).json()
    adj, _ = posted(client, 'adjust', adjust(items, -1), floor)
    h = summary(client, floor)['summary_hash']
    md, moved = move_lot(client, items['ingredient'], floor)
    for key, mine, theirs in ((office, [order, er], [moved, adj]), (floor, [moved, adj], [order, er])):
        doc = summary(client, key)
        listed = {r['receipt_number'] for s in doc['sections'] if s['key'] in shift_summary.RECEIPT_SECTIONS for r in s['rows']}
        other = {r['receipt_number']: r for r in doc['other_receipts']['rows']}
        assert not any(r['receipt_number'] in (order['receipt_number'], er['receipt_number'], moved['receipt_number']) for r in section(doc, 'adjusted')['rows'])
        assert order['receipt_number'] not in listed and er['receipt_number'] not in listed and moved['receipt_number'] not in listed
        # Parity with GET /receipts: every committed receipt of the day is either a section row or an other_receipts row.
        receipts = client.get('/receipts', params={'date': doc['date'], 'actor': doc['actor']['name']}, headers=headers(key)).json()
        assert {r['receipt_number'] for r in receipts['receipts'] if r['status'] == 'committed'} == listed | set(other)
        assert doc['receipts_today'] == len(listed) and doc['other_receipts']['count'] == len(other)
    office_doc = summary(client, office)
    other = {r['receipt_number']: r for r in office_doc['other_receipts']['rows']}
    o = other[order['receipt_number']]
    assert o['action'] == 'create_order' and o['kind'] == 'order' and o['status'] == 'committed' and o['actor']['id'] == actors['office']['id']
    assert o['refs']['order_id'] == order['order_id'] and o['refs']['order_number'] and o['refs']['order_number'] in o['line_en']
    assert o['line_en'].startswith(f'{order["receipt_number"]}  order created  ') and 'pedido creado' in o['line_es'] and o['line'] == o['line_en']
    e = other[er['receipt_number']]
    assert e['kind'] == 'expected_receipt' and e['refs']['expected_receipt_ids'] == [er['expected_receipt_id']] and 'expected receipt created' in e['line_en']
    assert office_doc['other_receipts']['label_en'].startswith('OTHER ENTRIES') and office_doc['other_receipts']['label'] == office_doc['other_receipts']['label_en']
    floor_doc = summary(client, floor)
    m = {r['receipt_number']: r for r in floor_doc['other_receipts']['rows']}[moved['receipt_number']]
    assert m['kind'] == 'lot_move' and m['refs']['lot_id'] == items['ingredient']['lot_id'] and f'lot {items["ingredient"]["lot_code"]}' in m['line_en']
    assert section(floor_doc, 'adjusted')['count'] == 1 and floor_doc['receipts_today'] == 1
    # They never enter the hash: the floor's hash is the same as before the lot move was recorded.
    assert floor_doc['summary_hash'] == h
    all_view = summary(client, main.DASHBOARD_API_KEY)
    assert all_view['other_receipts']['count'] >= 2 and all_view['other_receipts']['empty_es'] == 'ninguno'


def test_planned_run_is_entered_once_a_matching_make_or_pack_receipt_exists(client, db_cursor, items, actors):
    """Item 2: "without a linked MK/PK" = no committed make/pack receipt for the run's product on its planned day."""
    key = actors['floor']['key']
    today = main.get_plant_now().date()
    runs = {}
    for label, product, run_type in (('bake', items['batch']['id'], 'bake'), ('pack1', items['finished']['id'], 'pack'), ('pack2', items['finished']['id'], 'pack')):
        db_cursor.execute("INSERT INTO production_runs(product_id,planned_qty_lb,planned_date,status,run_type) VALUES (%s,100,%s,'planned',%s) RETURNING id",
                          (product, today, run_type))
        runs[label] = db_cursor.fetchone()['id']
    pending = lambda doc: [r['ref']['production_run_id'] for r in section(doc, 'not_entered_yet')['rows'] if r['kind'] == 'production_run']
    assert pending(summary(client, key)) == [runs['bake'], runs['pack1'], runs['pack2']]
    posted(client, 'make', body('make', items), key)
    assert pending(summary(client, key)) == [runs['pack1'], runs['pack2']]
    # Expectations are plant-wide: another actor's pack covers the first pack run in every view.
    posted(client, 'pack', body('pack', items), actors['owner']['key'])
    assert pending(summary(client, key)) == [runs['pack2']] == pending(summary(client, main.DASHBOARD_API_KEY))
    # A make does not cover a pack run; a second pack does.
    posted(client, 'make', body('make', items), key)
    assert pending(summary(client, key)) == [runs['pack2']]
    posted(client, 'pack', body('pack', items) | {'cases': 3}, key)
    assert pending(summary(client, key)) == []
    # Yesterday's planned run is not covered by today's receipts.
    db_cursor.execute("UPDATE production_runs SET planned_date=%s WHERE id=%s", (today - timedelta(days=1), runs['pack2']))
    assert pending(summary(client, key, date=(today - timedelta(days=1)).isoformat())) == [runs['pack2']]


def test_adjust_and_found_balances_are_the_commit_time_balances(client, db_cursor, items, actors):
    """Item 4: before/after come from the commit, not from the prepare-time draft."""
    key = actors['floor']['key']
    p = prepare(client, 'adjust', adjust(items, 1), key)
    assert p['draft']['current_quantity_lb'] == 100
    posted(client, 'adjust', adjust(items, 10), actors['owner']['key'])       # someone else moves the lot in between
    r = commit(client, p, key, acknowledged_warnings=[w['code'] for w in p['warnings'] if w.get('requires_ack')])
    assert r.status_code == 200, r.text
    [row] = [x for x in section(summary(client, key), 'adjusted')['rows'] if x['receipt_number'] == r.json()['receipt_number']]
    assert (row['detail']['lot_on_hand_before'], row['detail']['lot_on_hand_after'], row['detail']['delta_lb']) == (110, 111, 1)
    # Found stock on an EXISTING lot states the balance it was added to.
    f, _ = posted(client, 'found', found(items, 5, lot_code=items['ingredient']['lot_code']), key)
    assert f['lot_id'] == items['ingredient']['lot_id'] and f['new_balance_lb'] == 116
    [row] = [x for x in section(summary(client, key), 'adjusted')['rows'] if x['receipt_number'] == f['receipt_number']]
    assert (row['detail']['lot_on_hand_before'], row['detail']['lot_on_hand_after'], row['detail']['delta_lb']) == (111, 116, 5)
    # A found lot that did not exist starts at zero.
    g, _ = posted(client, 'found', found(items, 7), key)
    [row] = [x for x in section(summary(client, key), 'adjusted')['rows'] if x['receipt_number'] == g['receipt_number']]
    assert (row['detail']['lot_on_hand_before'], row['detail']['lot_on_hand_after']) == (0, 7)


def test_large_correction_keeps_its_approval_or_photo_reference_after_resolution(client, db_cursor, items, actors):
    """Item 4: the hold's exception is resolved by the approval/photo that posts the receipt — the reference must survive."""
    key, owner = actors['floor']['key'], actors['owner']['key']
    set_balance(db_cursor, items, 'ingredient', 1000)
    # (a) owner approval
    p = prepare(client, 'adjust', adjust(items, 600, 'missing_receipt'), key)
    held = commit(client, p, key)
    assert held.status_code == 202, held.text
    exc_id = held.json()['exception_id']
    approved = client.post(f'/exceptions/{exc_id}/approve', json={'note': 'seen'}, headers=headers(owner))
    assert approved.status_code == 200, approved.text
    doc = summary(client, key)
    [row] = [x for x in section(doc, 'adjusted')['rows'] if x['receipt_number'] == approved.json()['receipt_number']]
    [flag] = [f for f in row['flags'] if f['code'] == 'LARGE_CORRECTION']
    assert flag['refs'] == {'exception_id': exc_id, 'delta_lb': 600, 'pct': 0.6, 'resolution': 'approved', 'approved_by_actor_id': actors['owner']['id']}
    assert flag['message_en'] == f'LARGE CORRECTION +600 lb (60% of lot) — approved by {actors["owner"]["name"]}'
    assert flag['message_es'].endswith(f'— aprobada por {actors["owner"]["name"]}')
    assert row['exceptions_opened'] == [exc_id] and row['detail']['photo_attached'] is False
    assert (row['detail']['lot_on_hand_before'], row['detail']['lot_on_hand_after']) == (1000, 1600)
    [exc] = exceptions_for(db_cursor, id=exc_id)
    assert exc['status'] == 'resolved'                       # not open, still referenced
    # (b) a photo releases a held found
    q = prepare(client, 'found', found(items, 700, 'missing_receipt'), key)   # 700 ≠ 600: no POSSIBLE_DUPLICATE ack needed
    held2 = commit(client, q, key)
    assert held2.status_code == 202, held2.text
    seed_photo(db_cursor, actors['floor']['id'], 'photos/a9.jpg')
    released = commit(client, q, key, attachment_ref='photos/a9.jpg')
    assert released.status_code == 200, released.text
    [row2] = [x for x in section(summary(client, key), 'adjusted')['rows'] if x['receipt_number'] == released.json()['receipt_number']]
    [flag2] = [f for f in row2['flags'] if f['code'] == 'LARGE_CORRECTION']
    assert flag2['refs']['exception_id'] == held2.json()['exception_id'] and flag2['refs']['resolution'] == 'photo_attached'
    assert flag2['refs']['approved_by_actor_id'] is None and flag2['message_en'].endswith('— photo attached') and flag2['message_es'].endswith('— con foto')
    assert row2['detail']['photo_attached'] is True and row2['exceptions_opened'] == [held2.json()['exception_id']]
    # The hash is stable across re-reads (the resolved reference does not drift).
    assert summary(client, key)['summary_hash'] == summary(client, key)['summary_hash']


def test_voided_make_and_pack_keep_their_original_quantities(client, db_cursor, items, actors):
    """Item 5: a voided receipt stays listed with what it recorded; only live balances drop it."""
    key = actors['floor']['key']
    mk, _ = posted(client, 'make', body('make', items), key)
    pk, _ = posted(client, 'pack', body('pack', items), key)
    before = summary(client, key)
    for result in (mk, pk):
        r = client.post(f'/void/{result["transaction_id"]}', json={'reason': 'entered twice'}, headers=headers())
        assert r.status_code == 200, r.text
    doc = summary(client, key)
    [made] = section(doc, 'made')['rows']
    [packed] = section(doc, 'packed')['rows']
    assert made['effective_status'] == 'voided' and made['quantity']['lb'] == 10 and made['detail']['output_lb'] == 10
    assert made['transactions'] == [{'id': mk['transaction_id'], 'type': 'make', 'effective_status': 'voided'}]
    assert [f['code'] for f in made['flags']] == ['VOIDED'] and made['flags'][0]['refs'] == {'voided_by_receipt_number': None}
    assert '  10 lb  lot ' in made['line_en'] and made['detail']['ingredients'][0]['quantity_lb'] == 10
    assert packed['effective_status'] == 'voided' and packed['quantity'] == packed['quantity'] | {'lb': 10, 'cases': 2, 'display_en': '2 cases (10 lb)'}
    assert packed['detail']['cases'] == 2 and packed['detail']['lb'] == 10 and [f['code'] for f in packed['flags']] == ['VOIDED']
    assert doc['receipts_today'] == 2 and doc['summary_hash'] != before['summary_hash']
    # Voided lines never enter LOT BALANCES: nothing of today's is posted any more.
    assert section(doc, 'lot_balances_touched')['rows'] == []
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 100 and main.lot_on_hand(db_cursor, items['batch']['lot_id']) == 100
    # Void one of two: the posted one still moves the balance, the voided one keeps its amount.
    mk2, _ = posted(client, 'make', body('make', items) | {'batches': 2}, key)
    doc = summary(client, key)
    rows = {r['receipt_number']: r for r in section(doc, 'made')['rows']}
    assert rows[mk['receipt_number']]['quantity']['lb'] == 10 and rows[mk2['receipt_number']]['quantity']['lb'] == 20
    balances = {b['lot']['id']: b for b in section(doc, 'lot_balances_touched')['rows']}
    assert balances[items['ingredient']['lot_id']]['delta_lb'] == -20 and balances[items['ingredient']['lot_id']]['receipts'] == [mk2['receipt_number']]
    assert mk['lot_id'] not in balances and balances[mk2['lot_id']]['after_lb'] == 20


def test_dst_edges_elapsed_hours_and_ordering_use_instants(client, db_cursor, items, actors):
    """Item 6: hours_late across spring-forward is real elapsed time; fall-back receipts order by instant."""
    owner = actors['owner']['key']
    # Spring forward (2026-03-08, 02:00 EST → 03:00 EDT): entered 03:00 EDT is ONE real hour after 01:00 EST.
    spring, p = posted(client, 'adjust', adjust(items, 1) | {'occurred_at': '2026-03-08T01:00:00-05:00', 'backfill': True}, owner)
    # Committed tickets are immutable; the fixture lifts the trigger inside the rolled-back test transaction only.
    db_cursor.execute('ALTER TABLE write_tickets DISABLE TRIGGER write_tickets_preserve_committed')
    db_cursor.execute("""UPDATE write_tickets SET committed_at=%s, response=response-'entry_timing' WHERE id=%s""",
                      ('2026-03-08T03:00:00-04:00', p['ticket_id']))
    doc = summary(client, owner, date='2026-03-08')
    [row] = section(doc, 'adjusted')['rows']
    assert row['receipt_number'] == spring['receipt_number'] and row['entered_at'] == '2026-03-08T03:00:00-04:00'
    assert row['hours_late'] == 1.0 and row['days_late'] == 0 and row['late_entry'] is False and row['entered_on_later_day'] is False
    assert section(doc, 'late_entries')['rows'] == []
    # Fall back (2025-11-02, 02:00 EDT → 01:00 EST): 01:15 EST happened AFTER 01:45 EDT.
    a, _ = posted(client, 'adjust', adjust(items, 2) | {'occurred_at': '2025-11-02T01:45:00-04:00', 'backfill': True}, owner)
    b, _ = posted(client, 'adjust', adjust(items, 3) | {'occurred_at': '2025-11-02T01:15:00-05:00', 'backfill': True}, owner)
    doc = summary(client, owner, date='2025-11-02')
    assert [r['receipt_number'] for r in section(doc, 'adjusted')['rows']] == [a['receipt_number'], b['receipt_number']]
    assert [r['happened_at'] for r in section(doc, 'adjusted')['rows']] == ['2025-11-02T01:45:00-04:00', '2025-11-02T01:15:00-05:00']
    # Both were entered today, long after: today's LATE ENTRIES lists them (b first: fewer hours late? no — a happened earlier → more hours).
    late = [r['receipt_number'] for r in section(summary(client, owner), 'late_entries')['rows'] if r['receipt_number'] in (a['receipt_number'], b['receipt_number'])]
    assert late == [a['receipt_number'], b['receipt_number']]
    rows = {r['receipt_number']: r for r in section(summary(client, owner), 'late_entries')['rows']}
    assert rows[a['receipt_number']]['hours_late'] - rows[b['receipt_number']]['hours_late'] == 0.5
    # The stored entry_timing (write_tickets) and the recomputed fallback agree.
    db_cursor.execute("UPDATE write_tickets SET response=response-'entry_timing' WHERE receipt_number=%s", (a['receipt_number'],))
    db_cursor.execute('ALTER TABLE write_tickets ENABLE TRIGGER write_tickets_preserve_committed')
    again = {r['receipt_number']: r for r in section(summary(client, owner), 'late_entries')['rows']}
    assert abs(again[a['receipt_number']]['hours_late'] - rows[a['receipt_number']]['hours_late']) < 0.02


def test_notes_are_stored_verbatim(client, db_cursor, items, actors):
    """Item 7: §3.1 — notes are stored as sent; only the blank check trims."""
    key = actors['floor']['key']
    doc = summary(client, key)
    notes = '  two leading spaces, a tab\t and a trailing newline\n'
    r = confirm(client, key, doc, notes=notes)
    assert r.status_code == 201, r.text
    assert r.json()['notes'] == notes and r.json()['message_en'] == 'Shift recorded as: matches the floor, with notes.'
    db_cursor.execute('SELECT notes FROM shift_confirmations WHERE id=%s', (r.json()['confirmation_id'],))
    assert db_cursor.fetchone()['notes'] == notes
    assert summary(client, key)['confirmation']['latest']['notes'] == notes
    d = confirm(client, key, doc, outcome='discrepancy', notes=' missing \n')
    assert d.status_code == 201, d.text
    assert d.json()['notes'] == ' missing \n'
    [exc] = exceptions_for(db_cursor, id=d.json()['exception']['id'])
    assert exc['detail']['notes'] == ' missing \n'
    error(confirm(client, key, doc, outcome='discrepancy', notes=' \n\t'), 422, 'NOTES_REQUIRED')


def test_actor_me_is_the_caller(client, db_cursor, actors):
    """The merged F1 Assistant (PR #90) asks for `actor=me`."""
    me = summary(client, actors['floor']['key'], actor='me')
    assert me['actor']['id'] == actors['floor']['id'] and me['summary_hash'] == summary(client, actors['floor']['key'])['summary_hash']
    assert summary(client, main.DASHBOARD_API_KEY, actor='me')['actor'] is None

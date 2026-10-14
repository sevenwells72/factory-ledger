"""PR #101 integration checklist 1–7 on a fresh LOCAL DB; no hosted access."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from psycopg2.extras import Json
import pytest

import main
import shift_summary as ss
import write_tickets as wt
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_kosher_tiers import items, made, approve  # noqa: F401
from tests.test_a12_review_fixes import held_found
from tests.test_write_tickets import headers, commit, posted_count, ticket_row, error
from tests.test_write_tickets_part2 import body, prepare, seed
from tests.test_shift_summary import summary, section, confirm, receive, posted
from tests.test_ship_tickets import seed_shipping, prepared as ship_prepared

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parent.parent
FIELDS = {'kosher_tier', 'kosher_label', 'transaction_kosher_downgrade',
          'kosher_attested_by_actor_id', 'kosher_attested_at'}
HASH_KOSHER = FIELDS - {'kosher_label'}


def hold(client, cur, items, actors, action):
    if action == 'make':
        prepared, held = made(client, items, actors)
    else:
        prepared, held, _ = held_found(client, cur, items, actors)
    return prepared, held


def flags(row):
    return {f['code']: f for f in row['flags']}


@pytest.fixture
def regular_items(db_cursor):
    return seed(db_cursor)


@pytest.mark.parametrize('action', ['receive', 'make', 'pack', 'adjust', 'found'])
def test_item1_every_regular_receipt_has_all_five_immutable_fields(client, db_cursor, regular_items, actors, action):
    key = actors['floor']['key']
    if action == 'receive':
        result, _, _ = receive(client, db_cursor, key)
    else:
        result, _ = posted(client, action, body(action, regular_items), key)
    rows = section(summary(client, key), ss.ACTION_SECTION[action])['rows']
    row = next(r for r in rows if r['receipt_number'] == result['receipt_number'])
    assert FIELDS <= row.keys()
    assert row['kosher_tier'] == 'regular'
    assert row['transaction_kosher_downgrade'] is False
    assert row['kosher_attested_by_actor_id'] is None and row['kosher_attested_at'] is None


@pytest.mark.parametrize('action', ['make', 'found'])
def test_items2_3_4_6_hold_to_approved_same_receipt_preparer_and_owner(client, db_cursor, items, actors, action):
    key = actors['floor']['key']
    before = summary(client, key)
    prepared, held = hold(client, db_cursor, items, actors, action)
    waiting = summary(client, key)
    [row] = [r for r in section(waiting, 'not_entered_yet')['rows'] if r.get('kind') == 'kosher_hold']
    assert row['receipt_number'] == held['receipt_number'] and row['action'] == action
    assert row['status'] == 'awaiting_approval' and row['held'] is True and row['posted'] is False
    assert row['actor']['id'] == actors['floor']['id']
    assert row['approval']['status'] == 'awaiting_approval'
    assert row['approval']['owner_actor_id'] == actors['owner']['id']
    assert row['expires_at'] == row['approval']['expires_at'] == ss.iso(ticket_row(db_cursor, prepared)['expires_at'], main)
    assert row['kosher_attested_by_actor_id'] is None
    assert 'approval' in row['line_en'].lower() and 'aprobación' in row['line_es'].lower()
    assert waiting['receipts_today'] == before['receipts_today']
    assert waiting['summary_hash'] == before['summary_hash'] and posted_count(db_cursor, prepared) == 0
    assert not section(waiting, 'lot_balances_touched')['rows']
    spanish = summary(client, key, lang='es')
    [esrow] = [r for r in section(spanish, 'not_entered_yet')['rows'] if r.get('kind') == 'kosher_hold']
    assert esrow['line'] == esrow['line_es']
    assert confirm(client, key, waiting).status_code == 201

    approved = approve(client, held, actors)
    assert approved.status_code == 200, approved.text
    result = approved.json()
    after = summary(client, key)
    assert result['receipt_number'] == held['receipt_number']
    assert not [r for r in section(after, 'not_entered_yet')['rows'] if r.get('receipt_number') == held['receipt_number']]
    [row] = section(after, ss.ACTION_SECTION[action])['rows']
    assert row['receipt_number'] == held['receipt_number']
    assert FIELDS <= row.keys() and row['kosher_tier'] == 'extra_kosher'
    assert row['actor']['id'] == actors['floor']['id'] != actors['owner']['id']
    assert row['kosher_attested_by_actor_id'] == actors['owner']['id']
    assert row['kosher_attested_at'] == result['kosher_attested_at']
    assert flags(row)['EXTRA_KOSHER']['refs'] == {'kosher_tier': 'extra_kosher'}
    assert 'EXTRA-KOSHER (SS)' in row['line_en'] and 'EXTRA-KOSHER (SS)' in row['line_es']
    assert after['receipts_today'] == before['receipts_today'] + 1
    assert posted_count(db_cursor, prepared) == 1 and after['summary_hash'] != waiting['summary_hash']
    stale = confirm(client, key, waiting)
    error(stale, 409, 'SHIFT_SUMMARY_STALE')
    assert stale.json()['detail']['summary']['summary_hash'] == after['summary_hash']
    assert approve(client, held, actors).json()['replayed'] is True


@pytest.mark.parametrize('action', ['make', 'found'])
def test_item3_expired_hold_stays_visible_until_rejected_and_hash_stays_stable(client, db_cursor, items, actors, action):
    prepared, held = hold(client, db_cursor, items, actors, action)
    key = actors['floor']['key']
    current = summary(client, key)
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s", (prepared['ticket_id'],))
    expired = summary(client, key)
    [row] = [r for r in section(expired, 'not_entered_yet')['rows'] if r.get('receipt_number') == held['receipt_number']]
    assert row['approval']['status'] == 'expired' and row['posted'] is False
    assert 'expired' in row['line_en'].lower() and 'vencid' in row['line_es'].lower()
    assert expired['summary_hash'] == current['summary_hash']
    error(approve(client, held, actors), 409, 'TICKET_EXPIRED')
    rejected = client.post(f"/exceptions/{held['exception_id']}/reject", json={'resolution_kind': 'declined', 'note': 'Expired stock lacks approval'}, headers=headers(actors['owner']['key']))
    assert rejected.status_code == 200, rejected.text
    assert not [r for r in section(summary(client, key), 'not_entered_yet')['rows'] if r.get('receipt_number') == held['receipt_number']]
    assert posted_count(db_cursor, prepared) == 0
    # Both A9 guards survive the rebase on a rejected numbered hold.
    detail = client.get('/receipts/' + held['receipt_number'], headers=headers(key))
    assert detail.status_code == 200 and detail.json()['transactions'] == []
    assert detail.json()['entered_at'] is None


def test_item3_holds_from_earlier_days_are_scoped_to_preparer_and_all_actors(client, db_cursor, items, actors):
    prepared, held = made(client, items, actors, occurred_at=(main.get_plant_now()-timedelta(days=1)).isoformat())
    for key in (actors['floor']['key'], main.API_KEY):
        rows = section(summary(client, key), 'not_entered_yet')['rows']
        assert held['receipt_number'] in [r.get('receipt_number') for r in rows]
    assert not [r for r in section(summary(client, actors['office']['key']), 'not_entered_yet')['rows'] if r.get('kind') == 'kosher_hold']


def test_items1_2_extra_pack_and_catalog_edits_use_receipt_evidence(client, db_cursor, items, actors):
    key = actors['floor']['key']
    result, _ = posted(client, 'pack', body('pack', items), key)
    doc = summary(client, key)
    [row] = section(doc, 'packed')['rows']
    assert row['kosher_tier'] == 'extra_kosher'
    assert flags(row)['EXTRA_KOSHER']['refs'] == {'kosher_tier': 'extra_kosher'}
    assert row['transaction_kosher_downgrade'] is False
    db_cursor.execute("UPDATE products SET kosher_tier='regular',name=name||' renamed' WHERE id IN (%s,%s)", (items['batch']['id'], items['finished']['id']))
    later = summary(client, key, lang='es')
    [newrow] = section(later, 'packed')['rows']
    assert newrow['kosher_tier'] == 'extra_kosher' and newrow['product']['name'].endswith('renamed')
    assert later['summary_hash'] == doc['summary_hash']
    assert confirm(client, key, doc).status_code == 201


def test_items2_5_downgrade_flag_uses_posted_pack_transaction_and_output_lot(client, db_cursor, items, actors):
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (items['finished']['id'],))
    payload = body('pack', items) | {'lot_allocations': [{'lot_id': items['extra_lot']['lot_id'], 'quantity_lb': 10}]}
    result, _ = posted(client, 'pack', payload, actors['floor']['key'])
    en = summary(client, actors['floor']['key'])
    [row] = section(en, 'packed')['rows']
    assert row['kosher_tier'] == 'regular' and row['transaction_kosher_downgrade'] is True
    assert 'EXTRA_KOSHER' not in flags(row)
    flag = flags(row)['KOSHER_DOWNGRADE']
    assert flag['refs'] == {'transaction_id': result['transaction_id'], 'lot_id': result['output_lot_id'], 'transaction_kosher_downgrade': True}
    assert flag['message_en'] and flag['message_es'] and flag['message'] == flag['message_en']
    spanish = summary(client, actors['floor']['key'], lang='es')
    assert spanish['summary_hash'] == en['summary_hash']
    assert flags(section(spanish, 'packed')['rows'][0])['KOSHER_DOWNGRADE']['message'] == flag['message_es']


def historical_receive(cur, client, actors):
    """Synthetic pre-gate stored extra receipt; never bypass the live HTTP gate."""
    _, payload, prepared = receive(client, cur, actors['floor']['key'])
    original = ticket_row(cur, prepared)
    number = wt.allocate_receipt(cur, 'receive', main.get_plant_now().date())
    draft = deepcopy(original['draft']) | {'kosher_tier': 'extra_kosher', 'kosher_label': 'Saved historical extra label'}
    now = main.get_plant_now()
    cur.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (payload['product_id'],))
    cur.execute("INSERT INTO lots(product_id,lot_code,kosher_tier,kosher_attested_by_actor_id,kosher_attested_at) VALUES (%s,%s,'extra_kosher',%s,%s) RETURNING id", (payload['product_id'], 'WT-A12-HISTORY-' + uuid4().hex[:12].upper(), actors['owner']['id'], now))
    lot_id = cur.fetchone()['id']
    cur.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (payload['product_id'],))
    cur.execute("""INSERT INTO write_tickets(ticket_hash,action,actor_id,operator_id,key_kind,client_source,payload,payload_hash,state_hash,draft,expires_at)
        SELECT %s,action,actor_id,operator_id,key_kind,client_source,payload,payload_hash,state_hash,%s,expires_at FROM write_tickets WHERE id=%s RETURNING id""", (uuid4().hex + uuid4().hex, Json(draft), prepared['ticket_id']))
    ticket_id = cur.fetchone()['id']
    cur.execute("""INSERT INTO transactions(type,ticket_id,entered_by_actor_id,kosher_tier,kosher_attested_by_actor_id,kosher_attested_at,supplier_id)
        VALUES ('receive',%s,%s,'extra_kosher',%s,%s,%s) RETURNING id""", (ticket_id, actors['floor']['id'], actors['owner']['id'], now, payload['supplier_id']))
    txn_id = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,50)', (txn_id, payload['product_id'], lot_id))
    response = {'receipt_number': number, 'transaction_id': txn_id, 'lot_id': lot_id, 'kosher_tier': 'extra_kosher', 'kosher_label': 'Saved historical extra label', 'kosher_attested_by_actor_id': actors['owner']['id'], 'kosher_attested_at': now.isoformat(), 'transaction_kosher_downgrade': False}
    cur.execute("UPDATE write_tickets SET status='committed',committed_at=%s,receipt_number=%s,result_ref=%s,response=%s WHERE id=%s", (now, number, Json({'transaction_ids': [txn_id], 'lot_ids': [lot_id]}), Json(response), ticket_id))
    return number, response, payload


def test_items1_2_6_historical_extra_receive_renders_stored_evidence_while_new_receive_refuses(client, db_cursor, actors):
    number, saved, payload = historical_receive(db_cursor, client, actors)
    key = actors['floor']['key']
    doc = summary(client, key)
    row = next(r for r in section(doc, 'received')['rows'] if r['receipt_number'] == number)
    assert row['kosher_tier'] == 'extra_kosher' and row['kosher_attested_by_actor_id'] == actors['owner']['id']
    assert row['kosher_attested_at'] == saved['kosher_attested_at']
    assert flags(row)['EXTRA_KOSHER']['refs'] == {'kosher_tier': 'extra_kosher'}
    assert row['status'] == 'committed' and 'EXTRA-KOSHER (SS)' in row['line_en']
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher',name=name||' changed' WHERE id=%s", (payload['product_id'],))
    fresh = summary(client, key)
    assert fresh['summary_hash'] == doc['summary_hash']
    p = client.post('/receive/prepare', json=payload | {'lot_code': 'REFUSED-' + uuid4().hex}, headers=headers(key))
    assert p.status_code == 200 and not p.json()['can_commit']
    refused = commit(client, p.json(), key)
    assert refused.status_code == 409 and posted_count(db_cursor, p.json()) == 0
    assert not [r for r in section(summary(client, key), 'not_entered_yet')['rows'] if r.get('action') == 'receive']


def test_items1_5_adjust_uses_stored_lot_tier_despite_regular_transaction_default(client, db_cursor, items, actors):
    prepared, held = made(client, items, actors)
    approved = approve(client, held, actors).json()
    payload = {k: v for k, v in body('adjust', items).items() if k != 'reason'} | {'lot_id': approved['lot_id'], 'delta_lb': -1, 'reason_code': 'physical_count'}
    result, _ = posted(client, 'adjust', payload, actors['floor']['key'])
    doc = summary(client, actors['floor']['key'])
    [row] = section(doc, 'adjusted')['rows']
    db_cursor.execute('SELECT kosher_tier FROM transactions WHERE id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['kosher_tier'] == 'regular'
    assert row['kosher_tier'] == 'extra_kosher'
    assert row['kosher_attested_by_actor_id'] == actors['owner']['id']
    [evidence] = row['detail']['kosher_lots']
    assert evidence['lot_id'] == approved['lot_id'] and evidence['kosher_tier'] == 'extra_kosher'
    assert evidence['kosher_attested_by_actor_id'] == actors['owner']['id']
    detail = client.get('/receipts/' + result['receipt_number'], headers=headers(actors['floor']['key'])).json()
    assert detail['response']['kosher_lots'] == row['detail']['kosher_lots']
    db_cursor.execute("UPDATE products SET kosher_tier='regular',name=name||' changed' WHERE id=%s", (items['batch']['id'],))
    assert summary(client, actors['floor']['key'])['summary_hash'] == doc['summary_hash']


def test_items1_5_ship_preserves_all_consumed_lot_evidence(client, db_cursor, actors):
    data = seed_shipping(db_cursor)
    db_cursor.execute('SELECT * FROM lots WHERE id=%s', (data['lot_id'],))
    old_lot = db_cursor.fetchone()
    now = main.get_plant_now()
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (old_lot['product_id'],))
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,kosher_tier,kosher_attested_by_actor_id,kosher_attested_at) VALUES (%s,%s,'extra_kosher',%s,%s) RETURNING id", (old_lot['product_id'], 'WT-A12-SHIP-' + uuid4().hex[:12].upper(), actors['owner']['id'], now))
    extra_id = db_cursor.fetchone()['id']
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s", (old_lot['product_id'],))
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','Synthetic ship tier evidence') RETURNING id")
    txn_id = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)', (txn_id, old_lot['product_id'], extra_id))
    db_cursor.execute('SELECT lot_code FROM lots WHERE id=%s', (extra_id,))
    extra_code = db_cursor.fetchone()['lot_code']
    preparation = client.post('/ship/bulk/prepare', json={'customer_id': data['customer'], 'order_reference': 'A12-SHIP', 'bol_reference': 'BOL-A12', 'lines': [{'product_id': old_lot['product_id'], 'lot_id': data['lot_id'], 'quantity_lb': 25}, {'product_id': old_lot['product_id'], 'lot_id': extra_id, 'quantity_lb': 25}], 'lot_confirmations': [{'lot_id': data['lot_id'], 'method': 'full_code', 'value': data['lot_code']}, {'lot_id': extra_id, 'method': 'full_code', 'value': extra_code}]}, headers=headers(actors['office']['key']))
    assert preparation.status_code == 200, preparation.text
    prepared = preparation.json()
    r = commit(client, prepared, actors['office']['key'])
    assert r.status_code == 200, r.text
    doc = summary(client, actors['office']['key'])
    [row] = section(doc, 'shipped')['rows']
    evidence = {e['lot_id']: e for e in row['detail']['kosher_lots']}
    assert set(evidence) == {data['lot_id'], extra_id}
    assert evidence[data['lot_id']]['kosher_tier'] == 'regular'
    assert evidence[extra_id]['kosher_tier'] == 'extra_kosher'
    assert evidence[extra_id]['kosher_attested_by_actor_id'] == actors['owner']['id']
    stored = client.get('/receipts/' + r.json()['receipt_number'], headers=headers(actors['office']['key'])).json()
    assert stored['response']['kosher_lots'] == row['detail']['kosher_lots']
    assert summary(client, actors['office']['key'], lang='es')['summary_hash'] == doc['summary_hash']


@pytest.mark.parametrize('field,value', [('kosher_tier', 'regular'), ('kosher_attested_by_actor_id', 999), ('kosher_attested_at', '2026-10-13T12:00:00-04:00'), ('transaction_kosher_downgrade', True)])
def test_item5_each_stored_kosher_hash_field_moves_hash(client, db_cursor, items, actors, field, value):
    prepared, held = made(client, items, actors)
    assert approve(client, held, actors).status_code == 200
    doc = summary(client, actors['floor']['key'])
    changed = deepcopy(doc['sections'])
    row = next(s for s in changed if s['key'] == 'made')['rows'][0]
    assert field in ss.reduced(row)
    row[field] = value
    assert ss.summary_hash(main, changed) != doc['summary_hash']


def test_item5_labels_names_new_holds_and_nested_catalog_names_do_not_move_hash(client, db_cursor, items, actors):
    key = actors['floor']['key']
    prepared, held = made(client, items, actors)
    assert approve(client, held, actors).status_code == 200
    doc = summary(client, key)
    changed = deepcopy(doc['sections'])
    row = next(s for s in changed if s['key'] == 'made')['rows'][0]
    assert HASH_KOSHER <= ss.reduced(row).keys() and 'kosher_label' not in ss.reduced(row)
    row['kosher_label'] = 'Different display label'
    row['actor']['name'] = 'Renamed preparer'
    row['product']['name'] = 'Renamed output'
    row['detail']['batch_product']['name'] = 'Renamed batch'
    row['detail']['ingredients'][0]['product']['name'] = 'Renamed ingredient'
    assert ss.summary_hash(main, changed) == doc['summary_hash']
    db_cursor.execute("UPDATE products SET name=name||' renamed',kosher_tier='regular' WHERE id IN (%s,%s)", (items['batch']['id'], items['ingredient']['id']))
    assert summary(client, key)['summary_hash'] == doc['summary_hash']
    # A new hold alone never invalidates the existing entry confirmation.
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (items['batch']['id'],))
    held_found(client, db_cursor, items, actors)
    assert summary(client, key)['summary_hash'] == doc['summary_hash']
    assert confirm(client, key, doc).status_code == 201
    altered = deepcopy(doc['sections'])
    f = next(s for s in altered if s['key'] == 'made')['rows'][0]['flags'][0]
    f['refs']['kosher_tier'] = 'regular'
    assert ss.summary_hash(main, altered) != doc['summary_hash']


def test_item7_migration_numbering_route_hooks_and_null_guards_preserved():
    schema = (ROOT / 'tests/schema/schema.sql').read_text()
    fixture = (ROOT / 'tests/test_write_tickets.py').read_text()
    # 068/075/076 are on production (2026-10-13) and in the re-dumped schema: no pending \ir tails.
    pending = [line for line in schema.splitlines() if line.startswith('\\ir ../../migrations/')]
    assert pending in ([], ['\\ir ../../migrations/077_key_sessions.sql'],
                       ['\\ir ../../migrations/080_resolution_receipts.sql'],
                       ['\\ir ../../migrations/082_order_plant_date.sql'])
    for table in ('assistant_attachment_links', 'ticket_attestations', 'lot_events', 'shift_confirmations'):
        assert schema.count('CREATE TABLE public.' + table + ' (') == 1
    for number in ('075', '076'):
        assert 'migrations/' + number + '_' in fixture
    assert len(list((ROOT / 'migrations').glob('075_*.sql'))) == 1
    assert len(list((ROOT / 'migrations').glob('076_*.sql'))) == 1
    session_migrations = list((ROOT / 'migrations').glob('077_*.sql'))
    assert len(session_migrations) <= 1  # PR #103 may now coexist; A12 allocates no duplicate.
    assert not session_migrations or session_migrations[0].name == '077_key_sessions.sql'
    methods = {(method, route.path) for route in main.app.routes for method in getattr(route, 'methods', [])}
    assert ss.ROUTES <= methods and ss.ROUTES <= main.DASHBOARD_KEY_ALLOWLIST
    assert ('POST', '/reports/shift-summary/confirm') in main.permissions.UNGATED_ROUTES
    assert ('GET', '/dash/fl-assistant') in methods


def test_item5_live_reason_labels_never_invalidate_confirmed_entry_hash(client, db_cursor, regular_items, actors):
    key = actors['floor']['key']
    posted(client, 'adjust', body('adjust', regular_items), key)
    before = summary(client, key)
    db_cursor.execute("UPDATE correction_reasons SET label_en=label_en||' renamed', label_es=label_es||' cambiado' WHERE code='physical_count'")
    after = summary(client, key, lang='es')
    assert after['summary_hash'] == before['summary_hash']
    assert confirm(client, key, before).status_code == 201

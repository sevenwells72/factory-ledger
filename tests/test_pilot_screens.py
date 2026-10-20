"""Pilot form contracts, using local PostgreSQL and no model transport."""
from datetime import datetime
from uuid import uuid4

import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets import headers, commit, ticket_row, payload, prepare as receive_prepare
from tests.test_write_tickets_part2 import items, body, prepare as action_prepare
from tests.test_ship_tickets import shipping, new_order, upload, storage
from tests.test_order_tickets import create_body, create_order
from tests.test_customer_tickets import customer_schema

pytestmark = pytest.mark.db


def extra_lot(cur, product_id, qty=100):
    code = 'PILOT-' + uuid4().hex[:12].upper()
    cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id', (product_id, code))
    lid = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type) VALUES('receive') RETURNING id")
    txn = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,%s)',
                (txn, product_id, lid, qty))
    return {'lot_id': lid, 'lot_code': code}


def ship_request(client, shipping, key, action, choices=None, confirmations=None):
    data = {'bol_reference': 'PILOT-BOL', 'lot_confirmations': confirmations or []}
    if action == 'order':
        oid = create_order(client, shipping, key)['order_id']
        r = client.get(f'/sales/orders/{oid}', headers=headers(key)); assert r.status_code == 200, r.text
        line = next(r for r in r.json()['lines'] if r['product_id'] == shipping['products']['food'])
        line = {'line_id': line['line_id'], 'quantity_lb': 50}
        if choices is not None:
            line['lot_allocations'] = choices
        data['lines'] = [line]
        path = f'/sales/orders/{oid}/ship/prepare'
    else:
        path = '/ship/prepare'
        data.update(customer_id=shipping['customer'], product_id=shipping['products']['food'],
                    quantity_lb=50, order_reference='PILOT')
        if choices is not None:
            data['lot_allocations'] = choices
    return client.post(path, json=data, headers=headers(key))


@pytest.mark.parametrize('action', ['order', 'standalone'])
def test_shipping_fifo_default_and_actual_lot_selection(client, db_cursor, shipping, actors, action):
    key = actors['office']['key']
    newer = extra_lot(db_cursor, shipping['products']['food'])
    fifo = ship_request(client, shipping, key, action)
    assert fifo.status_code == 200, fifo.text
    assert fifo.json()['draft']['lines'][0]['lots'][0]['lot_id'] == shipping['lot_id']
    selected = ship_request(client, shipping, key, action, [{'lot_id': newer['lot_id'], 'quantity_lb': 50}])
    assert selected.status_code == 200, selected.text
    prepared = selected.json()
    assert prepared['draft']['lines'][0]['lots'] == [dict(newer, quantity_lb=50)]
    assert not prepared['can_commit']
    denied = commit(client, prepared, key)
    assert denied.status_code == 422 and denied.json()['detail']['error_code'] == 'LOT_NOT_CONFIRMED'
    result = commit(client, prepared, key,
                    lot_confirmations=[dict(lot_id=newer['lot_id'], method='full_code', value=newer['lot_code'])],
                    acknowledged_warnings=[w['code'] for w in prepared['warnings']])
    assert result.status_code == 200, result.text
    assert main.lot_on_hand(db_cursor, shipping['lot_id']) == 2000
    assert main.lot_on_hand(db_cursor, newer['lot_id']) == 50
    detail = client.get('/receipts/' + result.json()['receipt_number'], headers=headers(key)).json()
    assert [l['lot_code'] for l in detail['lots']] == [newer['lot_code']]
    assert commit(client, prepared, key).json() == result.json() | {'replayed': True}


@pytest.mark.parametrize('action', ['order', 'standalone'])
def test_shipping_split_lots_matches_exact_physical_quantities(client, db_cursor, shipping, actors, action):
    key = actors['office']['key']; newer = extra_lot(db_cursor, shipping['products']['food'])
    choices = [{'lot_id': shipping['lot_id'], 'quantity_lb': 15}, {'lot_id': newer['lot_id'], 'quantity_lb': 35}]
    confirmations = [dict(lot_id=r['lot_id'], method='full_code', value=r['lot_code']) for r in (shipping, newer)]
    r = ship_request(client, shipping, key, action, choices, confirmations)
    assert r.status_code == 200, r.text
    p = r.json(); result = commit(client, p, key, acknowledged_warnings=[w['code'] for w in p['warnings']])
    assert result.status_code == 200, result.text
    assert main.lot_on_hand(db_cursor, shipping['lot_id']) == 1985
    assert main.lot_on_hand(db_cursor, newer['lot_id']) == 65
    db_cursor.execute('SELECT lot_id,-SUM(quantity_lb) qty FROM transaction_lines WHERE transaction_id IN '
                      '(SELECT id FROM transactions WHERE ticket_id=%s) GROUP BY lot_id', (p['ticket_id'],))
    assert {r['lot_id']: float(r['qty']) for r in db_cursor.fetchall()} == {shipping['lot_id']: 15, newer['lot_id']: 35}


@pytest.mark.parametrize('action', ['order', 'standalone'])
@pytest.mark.parametrize('problem', ['wrong_product', 'short', 'sum', 'duplicate', 'empty'])
def test_shipping_refuses_invalid_physical_selections(client, db_cursor, shipping, actors, action, problem):
    key = actors['office']['key']
    newer = extra_lot(db_cursor, shipping['products']['food'], qty=20)
    choices = [{'lot_id': newer['lot_id'], 'quantity_lb': 50}]
    if problem == 'wrong_product':
        choices[0]['lot_id'] = extra_lot(db_cursor, shipping['products']['bulk'])['lot_id']
    elif problem == 'sum': choices[0]['quantity_lb'] = 10
    elif problem == 'duplicate': choices = [{'lot_id': newer['lot_id'], 'quantity_lb': 25}] * 2
    elif problem == 'empty': choices = []
    r = ship_request(client, shipping, key, action, choices)
    assert r.status_code in (409, 422), r.text
    assert r.json()['detail']['error_code'] in ('INSUFFICIENT_STOCK', 'INVALID_LOT_ALLOCATIONS')
    assert main.lot_on_hand(db_cursor, shipping['lot_id']) == 2000


@pytest.mark.parametrize('action', ['receive', 'make', 'pack'])
def test_optional_notes_survive_draft_post_replay_and_receipt(client, db_cursor, payload, items, actors, action):
    note = 'Packed for Wednesday delivery — keep upright.'
    key = actors['floor']['key']
    p = receive_prepare(client, payload | {'note': note}, key) if action == 'receive' else action_prepare(client, action, body(action, items) | {'note': note}, key)
    assert p['draft']['note'] == note
    if action == 'pack': assert p['draft']['target_product'] == items['finished']['name']
    r = commit(client, p, key); assert r.status_code == 200, r.text
    assert r.json()['note'] == note
    db_cursor.execute('SELECT notes FROM transactions WHERE id=%s', (r.json()['transaction_id'],))
    assert db_cursor.fetchone()['notes'].endswith(note)
    detail = client.get('/receipts/' + r.json()['receipt_number'], headers=headers(key)).json()
    assert detail['draft']['note'] == note and detail['response']['note'] == note
    assert commit(client, p, key).json() == r.json() | {'replayed': True}


def test_found_location_survives_receipt(client, items, actors):
    key = actors['floor']['key']; location = 'Dry storage rack 3'
    p = action_prepare(client, 'found', body('found', items) | {'found_location': location}, key)
    assert p['draft']['found_location'] == location
    r = commit(client, p, key); assert r.status_code == 200, r.text
    detail = client.get('/receipts/' + r.json()['receipt_number'], headers=headers(key)).json()
    assert detail['draft']['found_location'] == detail['response']['found_location'] == location


def test_void_note_survives_separate_reason_and_receipt(client, payload, actors):
    key = actors['floor']['key']; recorded = commit(client, receive_prepare(client, payload, key), key).json()
    r = client.post(f"/void/{recorded['transaction_id']}/prepare", json={'reason': 'data_entry_error', 'note': 'Wrong pallet entered'}, headers=headers(key))
    assert r.status_code == 200, r.text
    p = r.json(); assert p['draft']['note'] == 'Wrong pallet entered'
    r = commit(client, p, key); assert r.status_code == 200, r.text
    detail = client.get('/receipts/' + r.json()['receipt_number'], headers=headers(key)).json()
    assert detail['response']['note'] == 'Wrong pallet entered'
    assert detail['draft']['reason'] == 'data_entry_error'


@pytest.mark.parametrize('action', ['customer', 'order'])
def test_metadata_receipts_keep_actual_time_and_order_note(client, db_cursor, shipping, actors, customer_schema, action):
    key = actors['office']['key']
    data = {'name': 'Pilot customer ' + uuid4().hex[:10]} if action == 'customer' else create_body(shipping) | {'notes': 'Deliver at rear door'}
    path = '/customers/prepare' if action == 'customer' else '/sales/orders/prepare'
    r = client.post(path, json=data, headers=headers(key)); assert r.status_code == 200, r.text
    p = r.json()
    assert p['draft']['happened_at'] == p['draft']['entry_timing']['happened_at']
    repeated = client.post(path, json=data, headers=headers(key)).json()
    assert repeated['payload_hash'] == p['payload_hash']
    assert ticket_row(db_cursor, p)['status'] == 'superseded'
    r = commit(client, repeated, key); assert r.status_code == 200, r.text
    detail = client.get('/receipts/' + r.json()['receipt_number'], headers=headers(key)).json()
    assert detail['happened_at'] == detail['entered_at']
    timing = r.json()['entry_timing']
    assert datetime.fromisoformat(timing['happened_at']) == datetime.fromisoformat(timing['entered_at'])
    if action == 'order':
        assert detail['draft']['notes'] == detail['response']['notes'] == 'Deliver at rear door'


def test_receipt_exposes_current_proof_after_later_photo(client, db_cursor, shipping, actors, storage):
    key = actors['office']['key']
    r = ship_request(client, shipping, key, 'standalone', confirmations=[{'lot_id': shipping['lot_id'], 'method': 'full_code', 'value': shipping['lot_code']}])
    p = r.json(); r = commit(client, p, key); assert r.status_code == 200, r.text
    assert r.json()['proof_status'] == 'photo_pending'
    photo = upload(client, key)
    attached = client.post(f"/shipments/{r.json()['shipment_id']}/attachments", json={'attachment_id': photo['id']}, headers=headers(key))
    assert attached.status_code == 200, attached.text
    detail = client.get('/receipts/' + r.json()['receipt_number'], headers=headers(key)).json()
    assert detail['shipment']['proof_status'] == 'complete' and detail['attachments']
    assert detail['response']['proof_status'] == 'photo_pending'  # Immutable commit evidence.


def test_ship_lot_availability_respects_other_order_reservations_and_own_pins(client, db_cursor, shipping, actors):
    key = actors['office']['key']; pid = shipping['products']['food']
    newer = extra_lot(db_cursor, pid)
    mine = create_order(client, shipping, key)['order_id']
    other = create_order(client, shipping, key, cases=4)['order_id']
    def food_line(oid):
        rows = client.get(f'/sales/orders/{oid}', headers=headers(key)).json()['lines']
        return next(r['line_id'] for r in rows if r['product_id'] == pid)
    my_line, other_line = food_line(mine), food_line(other)
    db_cursor.execute('INSERT INTO sales_order_allocations(sales_order_id,sales_order_line_id,product_id,lot_id,quantity_lb,source) '
                      "VALUES(%s,%s,%s,%s,80,'manual') RETURNING id", (other, other_line, pid, newer['lot_id']))
    allocation_id = db_cursor.fetchone()['id']
    view = client.get('/ship/lots', params={'product_id': pid, 'line_id': my_line, 'quantity_lb': 50}, headers=headers(key))
    assert view.status_code == 200, view.text
    assert next(l['available_lb'] for l in view.json()['lots'] if l['lot_id'] == newer['lot_id']) == 20
    data = {'bol_reference': 'PILOT', 'lines': [{'line_id': my_line, 'quantity_lb': 50,
            'lot_allocations': [{'lot_id': newer['lot_id'], 'quantity_lb': 50}]}]}
    r = client.post(f'/sales/orders/{mine}/ship/prepare', json=data, headers=headers(key))
    assert r.status_code == 409 and r.json()['detail']['error_code'] == 'INSUFFICIENT_STOCK'
    own_view = client.get('/ship/lots', params={'product_id': pid, 'line_id': other_line, 'quantity_lb': 50}, headers=headers(key)).json()
    assert next(l['available_lb'] for l in own_view['lots'] if l['lot_id'] == newer['lot_id']) == 100
    assert own_view['suggested_lots'] == [dict(newer, quantity_lb=50)]
    data['lines'][0]['line_id'] = other_line
    data['lot_confirmations'] = [{'lot_id': newer['lot_id'], 'method': 'full_code', 'value': newer['lot_code']}]
    r = client.post(f'/sales/orders/{other}/ship/prepare', json=data, headers=headers(key)); assert r.status_code == 200, r.text
    result = commit(client, r.json(), key); assert result.status_code == 200, result.text
    assert main.lot_on_hand(db_cursor, newer['lot_id']) == 50
    db_cursor.execute("SELECT SUM(quantity_lb) qty FROM sales_order_allocations WHERE sales_order_line_id=%s AND status='active'", (other_line,))
    assert float(db_cursor.fetchone()['qty']) == 30
    mismatch = client.get('/ship/lots', params={'product_id': shipping['products']['bulk'], 'line_id': my_line}, headers=headers(key))
    assert mismatch.status_code == 404


@pytest.mark.parametrize('action', ['order', 'standalone'])
def test_selected_lot_stock_change_refuses_without_switching_to_fifo(client, db_cursor, shipping, actors, action):
    key = actors['office']['key']; newer = extra_lot(db_cursor, shipping['products']['food'], 60)
    r = ship_request(client, shipping, key, action, [{'lot_id': newer['lot_id'], 'quantity_lb': 50}],
                     [{'lot_id': newer['lot_id'], 'method': 'full_code', 'value': newer['lot_code']}])
    assert r.status_code == 200, r.text
    p = r.json()
    db_cursor.execute("INSERT INTO transactions(type) VALUES('ship') RETURNING id")
    txn = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,-30)',
                      (txn, shipping['products']['food'], newer['lot_id']))
    refused = commit(client, p, key)
    assert refused.status_code == 409 and refused.json()['detail']['error_code'] == 'TICKET_STALE'
    assert main.lot_on_hand(db_cursor, shipping['lot_id']) == 2000
    assert main.lot_on_hand(db_cursor, newer['lot_id']) == 30
    db_cursor.execute('SELECT count(*) n FROM shipments WHERE ticket_id=%s', (p['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('ship_qty,remaining', [(25, 25), (50, 0)])
def test_different_physical_lot_releases_excess_old_pins_and_void_restores_actual_stock(client, db_cursor, shipping, actors, ship_qty, remaining):
    key = actors['office']['key']; pid = shipping['products']['food']
    newer = extra_lot(db_cursor, pid)
    oid = create_order(client, shipping, key)['order_id']
    line = next(r['line_id'] for r in client.get(f'/sales/orders/{oid}', headers=headers(key)).json()['lines'] if r['product_id'] == pid)
    db_cursor.execute('INSERT INTO sales_order_allocations(sales_order_id,sales_order_line_id,product_id,lot_id,quantity_lb,source) '
                      "VALUES(%s,%s,%s,%s,50,'manual')", (oid, line, pid, shipping['lot_id']))
    data = {'bol_reference': 'PILOT', 'lines': [{'line_id': line, 'quantity_lb': ship_qty,
            'lot_allocations': [{'lot_id': newer['lot_id'], 'quantity_lb': ship_qty}]}],
            'lot_confirmations': [{'lot_id': newer['lot_id'], 'method': 'full_code', 'value': newer['lot_code']}]}
    r = client.post(f'/sales/orders/{oid}/ship/prepare', json=data, headers=headers(key)); assert r.status_code == 200, r.text
    result = commit(client, r.json(), key); assert result.status_code == 200, result.text
    posted = result.json()['lines_shipped'][0]
    assert sum(r['quantity_lb'] for r in posted['allocations_released']) == ship_qty
    db_cursor.execute("SELECT COALESCE(SUM(quantity_lb),0) qty FROM sales_order_allocations WHERE sales_order_line_id=%s AND status='active'", (line,))
    assert float(db_cursor.fetchone()['qty']) == remaining
    db_cursor.execute("SELECT count(*) n FROM sales_order_allocations WHERE sales_order_line_id=%s AND status='shipped'", (line,))
    assert db_cursor.fetchone()['n'] == 0
    owner = actors['owner']['key']
    r = client.post(f"/void/{posted['transaction_id']}/prepare", json={'reason': 'data_entry_error'}, headers=headers(owner))
    assert r.status_code == 200, r.text
    voided = commit(client, r.json(), owner); assert voided.status_code == 200, voided.text
    assert main.lot_on_hand(db_cursor, newer['lot_id']) == 100
    assert main.lot_on_hand(db_cursor, shipping['lot_id']) == 2000
    # Unused OLD lot reservations were released, never falsely consumed/re-shipped.
    db_cursor.execute("SELECT COALESCE(SUM(quantity_lb),0) qty FROM sales_order_allocations WHERE sales_order_line_id=%s AND status='active'", (line,))
    assert float(db_cursor.fetchone()['qty']) == remaining


def test_historical_order_receipt_reads_original_bilingual_note_without_rewriting_snapshots(client, db_cursor, shipping, actors, monkeypatch):
    key = actors['office']['key']
    notes = {'notes': 'Rear door delivery', 'notes_es': 'Entrega en la puerta trasera'}
    import write_tickets as wt
    p = client.post('/sales/orders/prepare', json=create_body(shipping, **notes), headers=headers(key)).json()
    # Recreate the legacy writer before commit; committed rows remain immutable.
    db_cursor.execute("UPDATE write_tickets SET draft=draft-'notes'-'notes_es' WHERE id=%s AND status='prepared'", (p['ticket_id'],))
    original = wt.json_value
    def legacy_response(api, value):
        result = original(api, value)
        if isinstance(result, dict) and result.get('ticket_id') == p['ticket_id'] and result.get('receipt_number'):
            result.pop('notes', None)
            result.pop('notes_es', None)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(wt, 'json_value', legacy_response)
        posted = commit(client, p, key)
        assert posted.status_code == 200, posted.text
        recorded = posted.json()
    # Later editable order changes must not overwrite what this receipt recorded.
    db_cursor.execute('UPDATE sales_orders SET notes=%s,notes_es=%s WHERE id=%s', ('Changed later', 'Cambió después', recorded['order_id']))
    detail = client.get('/receipts/' + recorded['receipt_number'], headers=headers(key)).json()
    for name, value in notes.items():
        assert detail['draft'][name] == detail['response'][name] == value
    db_cursor.execute('SELECT draft,response,payload FROM write_tickets WHERE id=%s', (recorded['ticket_id'],))
    stored = db_cursor.fetchone()
    assert all(name not in stored['draft'] and name not in stored['response'] for name in notes)
    assert all(stored['payload'][name] == value for name, value in notes.items())


@pytest.mark.parametrize('credential', ['actor', 'session'])
def test_ship_lot_lookup_is_authorized_read_only_and_does_not_expire_reservations(client, db_cursor, shipping, actors, monkeypatch, credential):
    from tests.test_named_actor_writes import _snapshot
    pid = shipping['products']['food']
    newer = extra_lot(db_cursor, pid)
    oid = create_order(client, shipping, actors['office']['key'])['order_id']
    line = next(r['line_id'] for r in client.get(f'/sales/orders/{oid}', headers=headers(actors['office']['key'])).json()['lines'] if r['product_id'] == pid)
    db_cursor.execute('INSERT INTO sales_order_allocations(sales_order_id,sales_order_line_id,product_id,lot_id,quantity_lb,source,expires_at) '
                      "VALUES(%s,%s,%s,%s,40,'auto_fifo',clock_timestamp()-interval '1 hour') RETURNING id",
                      (oid, line, pid, shipping['lot_id']))
    expired = db_cursor.fetchone()['id']
    auth = headers(actors['floor']['key'])
    if credential == 'session':
        monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
        response = client.post('/auth/session', json={'actor_key': actors['floor']['key']})
        assert response.status_code == 200, response.text
        auth = headers(response.json()['session_token'])
    before = _snapshot(db_cursor)
    db_cursor.execute('SELECT id,status,payload,payload_hash FROM write_tickets ORDER BY id')
    tickets_before = db_cursor.fetchall()
    response = client.get('/ship/lots', params={'product_id': pid, 'line_id': line, 'quantity_lb': 50}, headers=auth)
    assert response.status_code == 200, response.text
    assert response.json()['lots'] == [
        {'lot_id': shipping['lot_id'], 'lot_code': shipping['lot_code'], 'available_lb': 2000},
        {'lot_id': newer['lot_id'], 'lot_code': newer['lot_code'], 'available_lb': 100}]
    assert response.json()['suggested_lots'] == [{'lot_id': shipping['lot_id'], 'lot_code': shipping['lot_code'], 'quantity_lb': 50}]
    assert client.get('/ship/lots', params={'product_id': pid}, headers=headers(main.DASHBOARD_API_KEY)).status_code == 403
    assert client.get('/ship/lots', params={'product_id': pid}).status_code in (401, 403)
    assert _snapshot(db_cursor) == before
    db_cursor.execute('SELECT id,status,payload,payload_hash FROM write_tickets ORDER BY id')
    assert db_cursor.fetchall() == tickets_before
    db_cursor.execute('SELECT status,released_at,release_reason FROM sales_order_allocations WHERE id=%s', (expired,))
    assert dict(db_cursor.fetchone()) == {'status': 'active', 'released_at': None, 'release_reason': None}

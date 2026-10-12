"""A7: order tickets on real PostgreSQL — every §4.3 order action and every A1 guarantee.

Runs through the real HTTP routes with the savepoint-proxied test connection
(`client`/`actors` from test_actor_attribution); the double-commit race uses a
dedicated database like A1's.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
import pytest

import main
import order_tickets
import permissions
import write_tickets as tickets
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, ticket_row, error, isolated_database  # noqa: F401

pytestmark = pytest.mark.db
ORDER_ACTIONS = list(tickets.ORDER_ACTIONS)


# ---------------------------------------------------------------------------
# Seeds and helpers
# ---------------------------------------------------------------------------
def seed(cur):
    suffix = uuid4().hex[:10]
    cur.execute('INSERT INTO customers(name, active) VALUES (%s, true) RETURNING id', ('Ticket customer ' + suffix,))
    customer = cur.fetchone()['id']
    cur.execute('INSERT INTO customers(name, active) VALUES (%s, false) RETURNING id', ('Retired customer ' + suffix,))
    inactive = cur.fetchone()['id']
    products = {}
    for key, name, service, weight in [('food', 'Granola', False, 25), ('bulk', 'Bulk coconut', False, None),
                                       ('pallet', 'Pallet Charge', True, None)]:
        cur.execute('''INSERT INTO products(name, type, is_service, case_size_lb)
                       VALUES (%s, 'finished', %s, %s) RETURNING id''', (f'{name} {suffix}', service, weight))
        products[key] = cur.fetchone()['id']
    return {'customer': customer, 'inactive_customer': inactive, 'products': products, 'suffix': suffix}


@pytest.fixture
def catalog(db_cursor):
    return seed(db_cursor)


def create_body(catalog, cases=2, **updates):
    """`cases` varies the first line so a test can avoid the 24 h same-lines warning."""
    p = catalog['products']
    body = {'customer_id': catalog['customer'], 'customer_po': 'PO-' + catalog['suffix'],
            'lines': [{'product_id': p['food'], 'quantity': cases, 'unit': 'cases', 'unit_price': 30, 'amount': 30 * cases},
                      {'product_id': p['pallet'], 'quantity': 3, 'unit': 'each', 'unit_price': 12.5, 'amount': 37.5}]}
    body.update(updates)
    return body


def prepare_raw(client, path, body, key=None):
    return client.post(path, json=body, headers=headers(key))


def prepare(client, path, body, key=None):
    response = prepare_raw(client, path, body, key)
    assert response.status_code == 200, response.text
    return response.json()


def create_order(client, catalog, key=None, **updates):
    """Create an order through a ticket and return the commit receipt."""
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog, **updates), key)
    ack = [w['code'] for w in prepared['warnings'] if w.get('requires_ack')]
    response = commit(client, prepared, key, acknowledged_warnings=ack)
    assert response.status_code == 200, response.text
    return response.json()


def order_row(cur, order_id):
    cur.execute('SELECT * FROM sales_orders WHERE id=%s', (order_id,))
    return cur.fetchone()


def line_rows(cur, order_id):
    cur.execute('SELECT * FROM sales_order_lines WHERE sales_order_id=%s ORDER BY id', (order_id,))
    return cur.fetchall()


def count(cur, table):
    cur.execute(f'SELECT count(*) AS n FROM {table}')
    return cur.fetchone()['n']


def office(actors):
    return actors['office']['key']


# ---------------------------------------------------------------------------
# Create: dry-run prepare, single-use commit, replay, receipt
# ---------------------------------------------------------------------------
def test_create_prepare_is_a_dry_run_and_commit_creates_once_with_replay(client, db_cursor, catalog, actors):
    before = {t: count(db_cursor, t) for t in ('sales_orders', 'sales_order_lines', 'actor_write_audit')}
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog), office(actors))
    assert prepared['action'] == 'create_order' and prepared['can_commit'] is True
    assert {t: count(db_cursor, t) for t in before} == before, 'prepare must write nothing but the ticket'
    draft = prepared['draft']
    assert 'order_id' not in draft and draft['order_number_provisional'].startswith('SO-')
    assert draft['customer_po_status'] == 'PO provided' and draft['total_lb'] == 50 and draft['total'] == 97.5
    assert all('line_id' not in line for line in draft['lines'])
    assert draft['summary'].startswith('Create order for Ticket customer') and 'PO PO-' in draft['summary']
    assert draft['actor']['name'] == actors['office']['name'] and draft['entry_timing']['status'] == 'normal'
    assert prepared['warnings'] == []
    row = ticket_row(db_cursor, prepared)
    assert row['action'] == 'create_order' and row['payload']['customer_id'] == catalog['customer']
    assert 'customer_name' not in row['payload'] and row['payload']['lines'][0]['product_id'] == catalog['products']['food']

    first = commit(client, prepared, office(actors))
    assert first.status_code == 200, first.text
    receipt = first.json()
    assert receipt['receipt_number'].startswith('ORD-') and receipt['replayed'] is False
    assert receipt['state_changed'] is False and receipt['status'] == 'confirmed'
    assert receipt['order_number'].startswith('SO-') and receipt['total'] == 97.5
    assert receipt['external_order_reference'] == receipt['receipt_number']
    assert receipt['entry_timing']['entered_by']['name'] == actors['office']['name']
    order = order_row(db_cursor, receipt['order_id'])
    assert order['ticket_id'] == prepared['ticket_id'] and order['customer_po'] == 'PO-' + catalog['suffix']
    assert order['external_order_reference'] == receipt['receipt_number'] and order['status'] == 'confirmed'
    lines = line_rows(db_cursor, receipt['order_id'])
    assert [line['ticket_id'] for line in lines] == [prepared['ticket_id']] * 2
    assert [float(line['ordered_quantity']) for line in lines] == [2, 3]
    assert [float(line['quantity_lb']) for line in lines] == [50, 0]
    assert [float(line['amount']) for line in lines] == [60, 37.5]
    assert count(db_cursor, 'sales_orders') == before['sales_orders'] + 1

    replay = commit(client, prepared, office(actors))
    assert replay.status_code == 200 and replay.json() == receipt | {'replayed': True}
    assert count(db_cursor, 'sales_orders') == before['sales_orders'] + 1
    row = ticket_row(db_cursor, prepared)
    assert row['status'] == 'committed' and row['response'] == receipt
    assert row['result_ref'] == {'order_id': receipt['order_id'], 'order_number': receipt['order_number'],
                                 'line_ids': [l['id'] for l in lines], 'transaction_ids': [], 'lot_ids': []}
    # Attribution: the actor's audit rows name the ticket commit route.
    db_cursor.execute('SELECT target_table, route FROM actor_write_audit WHERE actor_id=%s ORDER BY id',
                      (actors['office']['id'],))
    audit = db_cursor.fetchall()
    assert [a['target_table'] for a in audit] == ['sales_orders', 'sales_order_lines', 'sales_order_lines']
    assert {a['route'] for a in audit} == {'/tickets/{ticket}/commit'}

    detail = client.get('/receipts/' + receipt['receipt_number'], headers=headers()).json()
    assert detail['action'] == 'create_order' and detail['response'] == receipt
    assert detail['order']['order_number'] == receipt['order_number'] and detail['order']['line_ids'] == [l['id'] for l in lines]
    assert detail['transactions'] == [] and detail['lots'] == [] and detail['late_entry'] is False
    listed = client.get('/receipts', params={'action': 'create_order'}, headers=headers()).json()['receipts']
    assert [r['receipt_number'] for r in listed] == [receipt['receipt_number']]


@pytest.mark.parametrize('changes,code', [
    ({'customer_name': 'Someone'}, 'IDS_REQUIRED'),
    ({'customer_address': '1 Main'}, 'IDS_REQUIRED'),
    ({'by': 'Luz'}, 'SELF_REPORTED_IDENTITY_REJECTED'),
    ({'changed_by': 'Luz'}, 'SELF_REPORTED_IDENTITY_REJECTED'),
])
def test_names_and_self_reported_identity_rejected(client, catalog, changes, code):
    error(prepare_raw(client, '/sales/orders/prepare', create_body(catalog, **changes)), 422, code)


def test_line_names_and_bad_shapes_rejected(client, catalog):
    body = create_body(catalog)
    body['lines'][0]['product_name'] = 'Granola'
    error(prepare_raw(client, '/sales/orders/prepare', body), 422, 'IDS_REQUIRED')
    body = create_body(catalog)
    del body['lines'][0]['quantity']
    error(prepare_raw(client, '/sales/orders/prepare', body), 422, 'QUANTITY_REQUIRED')
    for bad in ({'lines': []}, {'mode': 'commit'}, {'customer_id': 'x'}, {'client_source': 'invented'}):
        assert prepare_raw(client, '/sales/orders/prepare', create_body(catalog, **bad)).status_code == 422


def test_unknown_customer_or_product_is_a_blocker_not_an_order(client, db_cursor, catalog):
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog, customer_id=catalog['inactive_customer']))
    assert prepared['can_commit'] is False and prepared['blockers'][0]['code'] == 'CUSTOMER_INACTIVE'
    body = create_body(catalog)
    body['lines'][0]['product_id'] = 999999999
    prepared = prepare(client, '/sales/orders/prepare', body)
    assert prepared['blockers'][0]['code'] == 'PRODUCT_NOT_FOUND'
    error(commit(client, prepared), 409, 'TICKET_STALE')
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'


# ---------------------------------------------------------------------------
# Order rules kept: duplicate PO, No PO, service lines, external reference
# ---------------------------------------------------------------------------
def test_duplicate_po_warning_needs_acknowledgement_or_explicit_override(client, db_cursor, catalog, actors):
    first = create_order(client, catalog, office(actors))
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog, cases=4), office(actors))
    [warning] = prepared['warnings']
    assert warning['code'] == 'DUPLICATE_CUSTOMER_PO'
    assert warning['requires_ack'] is True and prepared['can_commit'] is True
    assert warning['refs']['existing_orders'][0]['order_number'] == first['order_number']
    refused = commit(client, prepared, office(actors))
    error(refused, 409, 'WARNING_NOT_ACKNOWLEDGED')
    assert refused.json()['detail']['missing'] == ['DUPLICATE_CUSTOMER_PO']
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    accepted = commit(client, prepared, office(actors), acknowledged_warnings=['DUPLICATE_CUSTOMER_PO'])
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()['order_id'] != first['order_id']
    assert ticket_row(db_cursor, prepared)['acknowledged'] == ['DUPLICATE_CUSTOMER_PO']
    # Explicit override in the payload: still warned, no acknowledgement needed.
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog, cases=5, allow_duplicate_po=True), office(actors))
    [warning] = prepared['warnings']
    assert warning['requires_ack'] is False
    assert commit(client, prepared, office(actors)).status_code == 200


def test_duplicate_po_appearing_after_prepare_makes_the_ticket_stale(client, db_cursor, catalog, actors):
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog), office(actors))
    assert prepared['warnings'] == []
    create_order(client, catalog, actors['owner']['key'])   # same PO lands first
    before = count(db_cursor, 'sales_orders')
    stale = commit(client, prepared, office(actors))
    error(stale, 409, 'TICKET_STALE')
    assert stale.json()['detail']['blockers'][0]['code'] == 'DUPLICATE_CUSTOMER_PO'
    assert count(db_cursor, 'sales_orders') == before
    row = ticket_row(db_cursor, prepared)
    assert row['status'] == 'rejected' and 'DUPLICATE_CUSTOMER_PO' in row['reject_reason']


def test_order_without_po_is_allowed_and_flagged(client, db_cursor, catalog):
    body = create_body(catalog)
    del body['customer_po']
    prepared = prepare(client, '/sales/orders/prepare', body)
    [warning] = [w for w in prepared['warnings'] if w['code'] == 'NO_PO']
    assert warning['requires_ack'] is False and prepared['draft']['customer_po_status'] == 'No PO'
    assert prepared['draft']['summary'].endswith('no PO')
    receipt = commit(client, prepared).json()
    assert receipt['customer_po_status'] == 'No PO' and receipt['customer_po'] is None
    assert order_row(db_cursor, receipt['order_id'])['customer_po'] is None
    # A blank PO is the same as none.
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog, cases=6, customer_po='  '))
    assert [w['code'] for w in prepared['warnings']] == ['NO_PO']


def test_service_line_has_zero_weight_and_priced_amount(client, db_cursor, catalog):
    receipt = create_order(client, catalog)
    pallet = [l for l in receipt['lines'] if l['product_id'] == catalog['products']['pallet']][0]
    assert pallet['quantity_lb'] == 0 and pallet['unit'] == 'each' and pallet['amount'] == 37.5
    assert pallet['is_service'] is True and receipt['total_lb'] == 50


def test_client_external_reference_is_kept_and_original_create_replays(client, db_cursor, catalog):
    receipt = create_order(client, catalog, external_order_reference='EXT-' + catalog['suffix'])
    assert receipt['external_order_reference'] == 'EXT-' + catalog['suffix']
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog, external_order_reference='EXT-' + catalog['suffix']))
    assert prepared['can_commit'] is True
    assert commit(client, prepared).json() == {**receipt, 'replayed': True}
    # The direct route's PR #67 rule still sees the ticket-created order.
    direct = client.post('/sales/orders', json={**create_body(catalog), 'external_order_reference': 'EXT-' + catalog['suffix']},
                         headers=headers())
    error(direct, 409, 'EXTERNAL_ORDER_REFERENCE_CONFLICT')


def test_possible_duplicate_same_lines_within_24h_requires_acknowledgement(client, db_cursor, catalog, actors):
    body = create_body(catalog)
    del body['customer_po']
    first = create_order(client, catalog, office(actors), customer_po=None)
    prepared = prepare(client, '/sales/orders/prepare', body, actors['owner']['key'])
    [warning] = [w for w in prepared['warnings'] if w['code'] == 'POSSIBLE_DUPLICATE']
    assert warning['requires_ack'] is True
    assert warning['refs']['receipt_number'] == first['receipt_number'] and warning['refs']['order_number'] == first['order_number']
    assert warning['refs']['operator_id'] == actors['office']['name']
    error(commit(client, prepared, actors['owner']['key']), 409, 'WARNING_NOT_ACKNOWLEDGED')
    assert commit(client, prepared, actors['owner']['key'], acknowledged_warnings=['POSSIBLE_DUPLICATE']).status_code == 200
    # Different lines: no warning.
    body['lines'][0]['quantity'] = 3
    body['lines'][0]['amount'] = 90
    assert prepare(client, '/sales/orders/prepare', body, actors['owner']['key'])['warnings'] == [no for no in [order_tickets.no_po_warning()]]


# ---------------------------------------------------------------------------
# Lines, header, status, ready
# ---------------------------------------------------------------------------
def test_add_update_and_cancel_lines_through_tickets(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    base = f"/sales/orders/{order['order_id']}"
    add = {'lines': [{'product_id': catalog['products']['bulk'], 'quantity': 100, 'unit': 'lb', 'unit_price': 2}]}
    prepared = prepare(client, base + '/lines/prepare', add, office(actors))
    assert prepared['draft']['order']['order_number'] == order['order_number']
    assert prepared['draft']['total_lb_added'] == 100 and 'line_id' not in prepared['draft']['lines_added'][0]
    assert len(line_rows(db_cursor, order['order_id'])) == 2, 'dry run must not add lines'
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['receipt_number'].startswith('ORD-') and receipt['lines_added'][0]['amount'] == 200
    lines = line_rows(db_cursor, order['order_id'])
    assert len(lines) == 3 and lines[-1]['ticket_id'] == prepared['ticket_id']
    assert float(lines[-1]['ordered_quantity']) == 100 and lines[-1]['ordered_unit'] == 'lb'
    assert ticket_row(db_cursor, prepared)['result_ref']['line_ids'] == [lines[-1]['id']]
    # The same lines again within 24 h: possible duplicate.
    again = prepare(client, base + '/lines/prepare', add, office(actors))
    assert [w['code'] for w in again['warnings']] == ['POSSIBLE_DUPLICATE']

    line_id = lines[0]['id']
    prepared = prepare(client, base + f'/lines/{line_id}/update/prepare', {'quantity_lb': 75, 'unit_price': 31}, office(actors))
    assert prepared['draft']['summary'] == f"Update line #{line_id} on {order['order_number']} (Ticket customer {catalog['suffix']}): quantity 50 lb → 75 lb, unit price → 31.0"
    assert float(line_rows(db_cursor, order['order_id'])[0]['quantity_lb']) == 50
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['quantity_lb'] == 75 and receipt['unit_price'] == 31
    updated = line_rows(db_cursor, order['order_id'])[0]
    assert float(updated['quantity_lb']) == 75 and float(updated['ordered_quantity']) == 3 and float(updated['amount']) == 93
    assert commit(client, prepared, office(actors)).json()['replayed'] is True
    assert float(line_rows(db_cursor, order['order_id'])[0]['quantity_lb']) == 75

    prepared = prepare(client, base + f'/lines/{line_id}/cancel/prepare', {}, office(actors))
    assert line_rows(db_cursor, order['order_id'])[0]['line_status'] == 'pending'
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['line_status'] == 'cancelled'
    assert line_rows(db_cursor, order['order_id'])[0]['line_status'] == 'cancelled'
    detail = client.get('/receipts/' + receipt['receipt_number'], headers=headers()).json()
    assert detail['order']['id'] == order['order_id'] and detail['order']['line_ids'] == [line_id]
    # Editing a cancelled line is a blocker.
    prepared = prepare(client, base + f'/lines/{line_id}/update/prepare', {'quantity_lb': 10}, office(actors))
    assert prepared['blockers'][0]['code'] == 'LINE_NOT_EDITABLE'
    # A line from another order is not found.
    other = create_order(client, catalog, office(actors), customer_po='OTHER-' + catalog['suffix'])
    prepared = prepare(client, base + f"/lines/{other['lines'][0]['line_id']}/cancel/prepare", {}, office(actors))
    assert prepared['blockers'][0]['code'] == 'LINE_NOT_FOUND'
    # Nothing to update is a blocker from the live core.
    prepared = prepare(client, base + f"/lines/{lines[1]['id']}/update/prepare", {}, office(actors))
    assert prepared['blockers'][0]['code'] == 'NO_FIELDS_TO_UPDATE'


def test_header_update_with_duplicate_po_customer_change_and_no_po(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    other = create_order(client, catalog, office(actors), customer_po='OTHER-' + catalog['suffix'])
    base = f"/sales/orders/{order['order_id']}"
    error(prepare_raw(client, base + '/header/prepare', {}, office(actors)), 422, 'NO_FIELDS_TO_UPDATE')
    prepared = prepare(client, base + '/header/prepare',
                       {'requested_ship_date': '2026-11-02', 'customer_po': 'OTHER-' + catalog['suffix']}, office(actors))
    assert ticket_row(db_cursor, prepared)['payload']['fields'] == ['customer_po', 'requested_ship_date']
    [warning] = prepared['warnings']
    assert warning['code'] == 'DUPLICATE_CUSTOMER_PO' and warning['refs']['existing_orders'][0]['order_id'] == other['order_id']
    assert order_row(db_cursor, order['order_id'])['requested_ship_date'] is None
    error(commit(client, prepared, office(actors)), 409, 'WARNING_NOT_ACKNOWLEDGED')
    receipt = commit(client, prepared, office(actors), acknowledged_warnings=['DUPLICATE_CUSTOMER_PO']).json()
    assert receipt['fields_updated'] == ['customer_po', 'requested_ship_date']
    row = order_row(db_cursor, order['order_id'])
    assert str(row['requested_ship_date']) == '2026-11-02' and row['customer_po'] == 'OTHER-' + catalog['suffix']
    # Clearing the PO flags it; changing to an inactive customer is a blocker.
    prepared = prepare(client, base + '/header/prepare', {'customer_po': None}, office(actors))
    assert [w['code'] for w in prepared['warnings']] == ['NO_PO']
    assert commit(client, prepared, office(actors)).json()['customer_po_status'] == 'No PO'
    prepared = prepare(client, base + '/header/prepare', {'customer_id': catalog['inactive_customer']}, office(actors))
    assert prepared['blockers'][0]['code'] == 'CUSTOMER_INACTIVE'
    # Header edits lock once the order advances.
    db_cursor.execute("UPDATE sales_orders SET status='in_production' WHERE id=%s", (order['order_id'],))
    prepared = prepare(client, base + '/header/prepare', {'notes': 'late'}, office(actors))
    assert prepared['blockers'][0]['code'] == 'ORDER_HEADER_LOCKED'


def test_status_transitions_keep_the_table_and_block_shipped(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    base = f"/sales/orders/{order['order_id']}/status/prepare"
    prepared = prepare(client, base, {'status': 'in_production'}, office(actors))
    assert prepared['draft']['summary'].endswith('confirmed → in_production')
    assert order_row(db_cursor, order['order_id'])['status'] == 'confirmed'
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['previous_status'] == 'confirmed' and receipt['status'] == 'in_production'
    assert order_row(db_cursor, order['order_id'])['status'] == 'in_production'
    for status in ('shipped', 'partial_ship'):
        blocked = prepare(client, base, {'status': status}, office(actors))
        assert blocked['can_commit'] is False and 'set automatically' in blocked['blockers'][0]['message']
    blocked = prepare(client, base, {'status': 'bogus'}, office(actors))
    assert 'Invalid status' in blocked['blockers'][0]['message']
    blocked = prepare(client, base, {'status': 'confirmed'}, office(actors))
    assert 'Invalid status transition' in blocked['blockers'][0]['message']
    # 'cancelled' via the status table is the legacy cancel: office may, floor may not.
    error(prepare_raw(client, base, {'status': 'cancelled'}, actors['floor']['key']), 403, 'ROLE_NOT_ALLOWED')
    prepared = prepare(client, base, {'status': 'cancelled'}, office(actors))
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['state'] == 'cancelled' and order_row(db_cursor, order['order_id'])['state'] == 'cancelled'


def test_status_invoiced_without_recorded_shipment_is_owner_only(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    db_cursor.execute("UPDATE sales_orders SET status='shipped' WHERE id=%s", (order['order_id'],))
    base = f"/sales/orders/{order['order_id']}/status/prepare"
    error(prepare_raw(client, base, {'status': 'invoiced'}, office(actors)), 403, 'ROLE_NOT_ALLOWED')
    prepared = prepare(client, base, {'status': 'invoiced'}, actors['owner']['key'])
    receipt = commit(client, prepared, actors['owner']['key']).json()
    assert receipt['status'] == 'invoiced' and receipt['state_reason'] == 'shipped_not_recorded'


def test_floor_marks_ready_but_cannot_create(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    floor = actors['floor']['key']
    before = count(db_cursor, 'write_tickets')
    error(prepare_raw(client, '/sales/orders/prepare', create_body(catalog), floor), 403, 'ROLE_NOT_ALLOWED')
    assert count(db_cursor, 'write_tickets') == before, 'a denied prepare issues no ticket'
    prepared = prepare(client, f"/sales/orders/{order['order_number']}/ready/prepare", {'note': 'on pallet 3'}, floor)
    assert prepared['draft']['summary'] == f"Mark {order['order_number']} (Ticket customer {catalog['suffix']}) ready to ship"
    db_cursor.execute('SELECT ready FROM sales_order_flags WHERE so_number=%s', (order['order_number'],))
    assert db_cursor.fetchone() is None
    receipt = commit(client, prepared, floor).json()
    assert receipt['ready'] is True and receipt['ready_by'] == actors['floor']['name'] and receipt['note'] == 'on pallet 3'
    db_cursor.execute('SELECT ready, ready_by FROM sales_order_flags WHERE so_number=%s', (order['order_number'],))
    assert dict(db_cursor.fetchone()) == {'ready': True, 'ready_by': actors['floor']['name']}
    prepared = prepare(client, f"/sales/orders/{order['order_id']}/ready/prepare", {'ready': False}, floor)
    assert commit(client, prepared, floor).json()['ready'] is False
    # Floor may not touch anything else on the order.
    for path, body in [('/lines/prepare', {'lines': [{'product_id': catalog['products']['bulk'], 'quantity_lb': 5}]}),
                       ('/header/prepare', {'notes': 'x'}), ('/status/prepare', {'status': 'in_production'}),
                       ('/cancel/prepare', {'reason': 'customer_cancelled'}), ('/close/prepare', {'reason': 'short_closed'}),
                       ('/reopen/prepare', {}), (f"/lines/{order['lines'][0]['line_id']}/cancel/prepare", {}),
                       (f"/lines/{order['lines'][0]['line_id']}/update/prepare", {'quantity_lb': 1})]:
        error(prepare_raw(client, f"/sales/orders/{order['order_id']}" + path, body, floor), 403, 'ROLE_NOT_ALLOWED')


# ---------------------------------------------------------------------------
# Cancel, close, reopen
# ---------------------------------------------------------------------------
def test_cancel_close_and_reopen_follow_the_matrix(client, db_cursor, catalog, actors):
    owner = actors['owner']['key']
    order = create_order(client, catalog, office(actors))
    base = f"/sales/orders/{order['order_id']}"
    prepared = prepare(client, base + '/cancel/prepare', {'reason': 'other'}, office(actors))
    assert prepared['blockers'][0]['code'] == 'STATE_NOTE_REQUIRED'
    prepared = prepare(client, base + '/cancel/prepare', {'reason': 'other', 'note': 'typed twice'}, office(actors))
    assert prepared['draft']['summary'] == f"Cancel {order['order_number']} (Ticket customer {catalog['suffix']}) (other)"
    assert order_row(db_cursor, order['order_id'])['state'] == 'open'
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['state'] == 'cancelled' and receipt['state_changed_by'] == actors['office']['name']
    assert order_row(db_cursor, order['order_id'])['state'] == 'cancelled'
    # Cancelling again: the order is no longer open.
    prepared = prepare(client, base + '/cancel/prepare', {'reason': 'customer_cancelled'}, office(actors))
    assert prepared['blockers'][0]['code'] == 'ORDER_NOT_OPEN'

    error(prepare_raw(client, base + '/reopen/prepare', {}, office(actors)), 403, 'ROLE_NOT_ALLOWED')
    prepared = prepare(client, base + '/reopen/prepare', {'note': 'customer called back'}, owner)
    assert prepared['draft']['summary'].endswith("as 'confirmed'")
    receipt = commit(client, prepared, owner).json()
    assert receipt['state'] == 'open' and receipt['status'] == 'confirmed'
    assert order_row(db_cursor, order['order_id'])['state'] == 'open'

    prepared = prepare(client, base + '/close/prepare', {'reason': 'short_closed'}, office(actors))
    receipt = commit(client, prepared, office(actors)).json()
    assert receipt['state'] == 'closed' and receipt['state_reason'] == 'short_closed' and receipt['status'] == 'shipped'
    assert commit(client, prepared, office(actors)).json()['replayed'] is True
    # The master key reopens too (its existing reach), then the owner-only close reason.
    prepared = prepare(client, base + '/reopen/prepare', {})
    assert commit(client, prepared).json()['state'] == 'open'
    error(prepare_raw(client, base + '/close/prepare', {'reason': 'shipped_not_recorded'}, office(actors)), 403, 'ROLE_NOT_ALLOWED')
    prepared = prepare(client, base + '/close/prepare', {'reason': 'shipped_not_recorded', 'note': 'left on the truck log'}, owner)
    assert prepared['can_commit'] is True
    receipt = commit(client, prepared, owner).json()
    assert receipt['state_reason'] == 'shipped_not_recorded'
    # shipped_recorded passes today's A6 hook (no gate yet) and reaches the core's checks.
    assert order_tickets.shipment_gate(main, db_cursor, {}, 'shipped_recorded') == []


def test_close_shipped_not_recorded_role_is_rechecked_at_commit(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    owner = actors['owner']
    prepared = prepare(client, f"/sales/orders/{order['order_id']}/close/prepare",
                       {'reason': 'shipped_not_recorded'}, owner['key'])
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (owner['id'],))
    main._reset_actor_cache()
    error(commit(client, prepared, owner['key']), 403, 'ROLE_NOT_ALLOWED')
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    assert order_row(db_cursor, order['order_id'])['state'] == 'open'


# ---------------------------------------------------------------------------
# A1 guarantees on order tickets
# ---------------------------------------------------------------------------
def test_wrong_user_payload_mismatch_expiry_and_supersession(client, db_cursor, catalog, actors):
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog), office(actors))
    error(commit(client, prepared, actors['owner']['key']), 403, 'TICKET_WRONG_USER')
    error(commit(client, prepared), 403, 'TICKET_WRONG_USER')
    error(client.post('/tickets/' + prepared['ticket'] + '/commit', json={'payload_hash': 'f' * 64},
                      headers=headers(office(actors))), 409, 'TICKET_PAYLOAD_MISMATCH')
    error(client.post('/tickets/wt_nope/commit', json={'payload_hash': prepared['payload_hash']},
                      headers=headers(office(actors))), 404, 'TICKET_NOT_FOUND')
    error(commit(client, prepared, office(actors), lot_confirmations=[{'lot_id': 1, 'method': 'last4', 'value': 'ABCD'}]),
          422, 'UNUSED_LOT_CONFIRMATION')
    # Same actor, same payload again on the same plant day: the first ticket is superseded.
    again = prepare(client, '/sales/orders/prepare', create_body(catalog), office(actors))
    assert ticket_row(db_cursor, prepared)['status'] == 'superseded'
    assert again['payload_hash'] == prepared['payload_hash']
    assert again['draft']['happened_at'] == main.get_plant_now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    error(commit(client, prepared, office(actors)), 409, 'TICKET_NOT_COMMITTABLE')
    db_cursor.execute("UPDATE write_tickets SET expires_at=now()-interval '1 minute' WHERE id=%s", (again['ticket_id'],))
    error(commit(client, again, office(actors)), 409, 'TICKET_EXPIRED')
    assert ticket_row(db_cursor, again)['status'] == 'expired'
    assert count(db_cursor, 'sales_orders') == 0 or order_row(db_cursor, 0) is None
    db_cursor.execute('SELECT count(*) AS n FROM sales_orders WHERE customer_id=%s', (catalog['customer'],))
    assert db_cursor.fetchone()['n'] == 0


def test_stale_ticket_after_the_order_closed_is_rejected_without_writing(client, db_cursor, catalog, actors):
    order = create_order(client, catalog, office(actors))
    base = f"/sales/orders/{order['order_id']}"
    add = prepare(client, base + '/lines/prepare',
                  {'lines': [{'product_id': catalog['products']['bulk'], 'quantity_lb': 10}]}, office(actors))
    closing = prepare(client, base + '/cancel/prepare', {'reason': 'customer_cancelled'}, office(actors))
    assert commit(client, closing, office(actors)).status_code == 200
    stale = commit(client, add, office(actors))
    error(stale, 409, 'TICKET_STALE')
    assert stale.json()['detail']['blockers'][0]['code'] == 'ORDER_NOT_OPEN'
    assert len(line_rows(db_cursor, order['order_id'])) == 2
    assert ticket_row(db_cursor, add)['status'] == 'rejected'
    # A changed order that is still valid commits with state_changed=true.
    reopened = prepare(client, base + '/reopen/prepare', {}, actors['owner']['key'])
    assert commit(client, reopened, actors['owner']['key']).status_code == 200
    note = prepare(client, base + '/header/prepare', {'notes': 'rush'}, office(actors))
    db_cursor.execute("UPDATE sales_orders SET requested_ship_date='2026-12-01' WHERE id=%s", (order['order_id'],))
    receipt = commit(client, note, office(actors)).json()
    assert receipt['state_changed'] is True and receipt['notes'] == 'rush'


def test_role_change_and_deactivation_between_prepare_and_commit(client, db_cursor, catalog, actors):
    prepared = prepare(client, '/sales/orders/prepare', create_body(catalog), office(actors))
    db_cursor.execute("UPDATE actors SET role='floor' WHERE id=%s", (actors['office']['id'],))
    main._reset_actor_cache()
    error(commit(client, prepared, office(actors)), 403, 'ROLE_NOT_ALLOWED')
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    db_cursor.execute("UPDATE actors SET role='office', active=false WHERE id=%s", (actors['office']['id'],))
    main._reset_actor_cache()
    assert commit(client, prepared, office(actors)).status_code == 403   # unknown key now
    db_cursor.execute('SELECT count(*) AS n FROM sales_orders WHERE customer_id=%s', (catalog['customer'],))
    assert db_cursor.fetchone()['n'] == 0


def test_dashboard_key_gets_nothing_new_master_key_reaches_everything(client, db_cursor, catalog):
    response = prepare_raw(client, '/sales/orders/prepare', create_body(catalog), main.DASHBOARD_API_KEY)
    assert response.status_code == 403 and 'not authorized' in response.text
    receipt = create_order(client, catalog)   # master key
    assert receipt['entry_timing']['entered_by']['key_kind'] == 'legacy_ledger'
    assert order_row(db_cursor, receipt['order_id'])['ticket_id'] is not None
    for path in tickets.ORDER_PREPARE_ROUTES:
        url = path[1].replace('{order_id}', str(receipt['order_id'])).replace('{line_id}', str(receipt['lines'][0]['line_id']))
        assert client.post(url, json={}, headers=headers(main.DASHBOARD_API_KEY)).status_code == 403


def test_direct_order_routes_unchanged(client, db_cursor, catalog, actors):
    before = count(db_cursor, 'write_tickets')
    direct = client.post('/sales/orders', json=create_body(catalog), headers=headers())
    assert direct.status_code == 200, direct.text
    assert order_row(db_cursor, direct.json()['order_id'])['ticket_id'] is None
    assert count(db_cursor, 'write_tickets') == before
    patched = client.patch(f"/sales/orders/{direct.json()['order_id']}", json={'notes': 'direct'}, headers=headers(office(actors)))
    assert patched.status_code == 200 and patched.json()['fields_updated'] == ['notes']
    status = client.patch(f"/sales/orders/{direct.json()['order_id']}/status", json={'status': 'in_production'}, headers=headers())
    assert status.status_code == 200 and status.json()['status'] == 'in_production'
    error(client.post('/sales/orders', json=create_body(catalog, customer_po='F-' + catalog['suffix']),
                      headers=headers(actors['floor']['key'])), 403, 'ROLE_NOT_ALLOWED')
    ready = client.post(f"/sales-orders/{direct.json()['order_number']}/ready", json={'ready': True}, headers=headers())
    assert ready.status_code == 200 and ready.json()['ready_by'] == 'floor'
    closed = client.post(f"/sales/orders/{direct.json()['order_id']}/close",
                         json={'reason': 'short_closed', 'mode': 'commit'}, headers=headers(office(actors)))
    assert closed.status_code == 200 and closed.json()['state'] == 'closed'


def test_matrix_rows_for_order_actions():
    for action in ORDER_ACTIONS:
        assert action in permissions.ROLE_PERMISSIONS
    assert permissions.allowed('mark_order_ready', 'floor')
    assert not any(permissions.allowed(a, 'floor') for a in ORDER_ACTIONS if a != 'mark_order_ready')
    assert all(permissions.allowed(a, 'office') for a in ORDER_ACTIONS if a != 'reopen_order')
    assert not permissions.allowed('close_order_shipped_not_recorded', 'office')
    assert all(permissions.allowed(a, 'owner') for a in ORDER_ACTIONS)
    assert not any(permissions.allowed(a, 'legacy_dashboard') for a in ORDER_ACTIONS)
    assert all(tickets.PREFIXES[a] == 'ORD' for a in ORDER_ACTIONS)


def test_concurrent_double_commit_creates_one_order_and_one_receipt(isolated_database, monkeypatch):
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        catalog = seed(cur)
    gate = Barrier(2)
    race = False

    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='10s'")
            if race:
                gate.wait(timeout=10)
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    with TestClient(main.app) as http:
        prepared = prepare(http, '/sales/orders/prepare', create_body(catalog))
        race = True
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = [f.result(timeout=20) for f in [executor.submit(commit, http, prepared) for _ in range(2)]]
        race = False
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    results = [r.json() for r in responses]
    assert results[0]['receipt_number'] == results[1]['receipt_number']
    assert results[0]['order_id'] == results[1]['order_id']
    assert sorted(r['replayed'] for r in results) == [False, True]
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute('SELECT count(*) AS n FROM sales_orders WHERE customer_id=%s', (catalog['customer'],))
        assert cur.fetchone()['n'] == 1
        cur.execute("SELECT count(*) AS n FROM write_tickets WHERE status='committed'")
        assert cur.fetchone()['n'] == 1
        # The populated 067 down refuses to discard order-ticket evidence.
        with pytest.raises(psycopg2.Error, match='067 rollback refused'):
            cur.execute((main.pathlib.Path(__file__).resolve().parents[1] / 'migrations/down/067_order_tickets_down.sql').read_text())

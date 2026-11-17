"""Cancel as duplicate: the read-only duplicate-check facts and the cancel_order guard.

The panel reads GET /sales/orders/{id}/duplicate-check and then uses the
existing cancel_order ticket (prepare -> commit). These tests pin the
suggestion rules, the self / cancelled-original refusals in BOTH the read and
the live cancel rules, and that shipments and reservations follow the existing
cancel behaviour (shipped refused, reservations listed and then released).
"""
import time
from uuid import uuid4

import pytest

import order_tickets

from tests.test_order_tickets import client, actors, catalog, create_order, prepare, prepare_raw, office, order_row  # noqa: F401
from tests.test_write_tickets import headers, commit
from tests.test_sales_order_state_model import _allocate, _post_ship

pytestmark = pytest.mark.db


def check(client, order_id, key, original=None):
    path = f'/sales/orders/{order_id}/duplicate-check' + (f'?original={original}' if original is not None else '')
    response = client.get(path, headers=headers(key))
    assert response.status_code == 200, response.text
    return response.json()


def cancel_duplicate(client, order_id, original_id, key, note=None):
    return prepare(client, f'/sales/orders/{order_id}/cancel/prepare',
                   {'reason': 'duplicate', 'related_so_id': original_id,
                    'note': note or f'Duplicate of #{original_id}'}, key)


def table_counts(cur):
    counts = {}
    for table in ('sales_orders', 'sales_order_allocations', 'write_tickets', 'transactions', 'actor_write_audit'):
        cur.execute(f'SELECT count(*) AS n FROM {table}')
        counts[table] = cur.fetchone()['n']
    cur.execute("SELECT count(*) AS n FROM sales_orders WHERE state <> 'open'")
    counts['exited'] = cur.fetchone()['n']
    return counts


def test_suggests_the_same_po_order_and_shows_both_sides(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)   # same customer, same PO, same lines
    before = table_counts(db_cursor)
    data = check(client, duplicate['order_id'], key)
    assert table_counts(db_cursor) == before, 'duplicate-check must write nothing'
    assert data['cancellable'] is True and data['blockers'] == []
    assert data['suggestion']['order_number'] == original['order_number']
    assert data['suggestion']['matches'] == ['customer_po', 'lines']
    order = data['order']
    assert order['order_number'] == duplicate['order_number']
    assert order['customer_po'] == 'PO-' + catalog['suffix']
    assert order['created_by'] == actors['office']['name']
    assert order['created_date'] and [l['product_id'] for l in order['lines']] == [
        catalog['products']['food'], catalog['products']['pallet']]
    # Read by order number too, the way the Office page links to it.
    assert check(client, duplicate['order_number'], key)['order']['order_id'] == duplicate['order_id']


def test_po_match_ranks_before_a_line_match_and_lines_alone_still_suggest(client, db_cursor, catalog, actors):
    key = office(actors)
    by_lines = create_order(client, catalog, key, customer_po='OTHER-' + catalog['suffix'])
    by_po = create_order(client, catalog, key, cases=7)
    duplicate = create_order(client, catalog, key)
    data = check(client, duplicate['order_id'], key)
    assert [c['order_number'] for c in data['candidates']] == [by_po['order_number'], by_lines['order_number']]
    assert data['candidates'][0]['matches'] == ['customer_po']
    assert data['candidates'][1]['matches'] == ['lines']
    # No PO on the duplicate: the line match alone suggests.
    no_po = create_order(client, catalog, key, customer_po=None, cases=7)
    assert check(client, no_po['order_id'], key)['suggestion']['order_number'] == by_po['order_number']


def test_suggestion_excludes_cancelled_old_and_other_customer_orders(client, db_cursor, catalog, actors, monkeypatch):
    key = office(actors)
    stale = create_order(client, catalog, key)
    time.sleep(1.2)
    # created_at is database-owned and immutable, so shrink the window instead of back-dating.
    monkeypatch.setattr(order_tickets, 'DUPLICATE_LOOKBACK_DAYS', 1 / 86400)
    cancelled = create_order(client, catalog, key)
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='other', status='cancelled' WHERE id=%s",
                      (cancelled['order_id'],))
    db_cursor.execute('INSERT INTO customers(name, active) VALUES (%s, true) RETURNING id', ('Other ' + uuid4().hex,))
    other_customer = create_order(client, {**catalog, 'customer': db_cursor.fetchone()['id']}, key)
    duplicate = create_order(client, catalog, key)
    data = check(client, duplicate['order_id'], key)
    numbers = [c['order_number'] for c in data['candidates']]
    assert cancelled['order_number'] not in numbers
    assert stale['order_number'] not in numbers
    assert other_customer['order_number'] not in numbers
    assert data['suggestion'] is None


def test_chosen_original_reports_self_cancelled_and_other_customer(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    data = check(client, duplicate['order_id'], key, original=original['order_number'])
    assert data['original']['order_id'] == original['order_id'] and data['original']['problems'] == []
    own = check(client, duplicate['order_id'], key, original=duplicate['order_number'])['original']['problems']
    assert [(p['code'], p['blocks']) for p in own] == [('RELATED_SO_INVALID', True)]
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='other', status='cancelled' WHERE id=%s",
                      (original['order_id'],))
    gone = check(client, duplicate['order_id'], key, original=original['order_id'])['original']['problems']
    assert [(p['code'], p['blocks']) for p in gone] == [('RELATED_SO_CANCELLED', True)]
    db_cursor.execute('INSERT INTO customers(name, active) VALUES (%s, true) RETURNING id', ('Other ' + uuid4().hex,))
    elsewhere = create_order(client, {**catalog, 'customer': db_cursor.fetchone()['id']}, key)
    problems = check(client, duplicate['order_id'], key, original=elsewhere['order_number'])['original']['problems']
    assert [(p['code'], p['blocks']) for p in problems] == [('DIFFERENT_CUSTOMER', False)]
    missing = client.get(f"/sales/orders/{duplicate['order_id']}/duplicate-check?original=SO-NOPE-000",
                         headers=headers(key))
    assert missing.status_code == 404 and missing.json()['detail']['error_code'] == 'RELATED_SO_NOT_FOUND'


def test_cancel_as_duplicate_uses_the_ticket_and_records_reason_note_and_original(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    note = 'Duplicate of ' + original['order_number']
    prepared = cancel_duplicate(client, duplicate['order_id'], original['order_id'], key, note)
    assert prepared['can_commit'], prepared
    assert order_row(db_cursor, duplicate['order_id'])['state'] == 'open', 'prepare is a dry run'
    receipt = commit(client, prepared, key).json()
    assert receipt['receipt_number'].startswith('ORD-')
    row = order_row(db_cursor, duplicate['order_id'])
    assert (row['state'], row['state_reason'], row['related_so_id'], row['state_note']) == (
        'cancelled', 'duplicate', original['order_id'], note)
    assert order_row(db_cursor, original['order_id'])['state'] == 'open'
    # The original cannot now be cancelled as a duplicate of the cancelled copy.
    back = cancel_duplicate(client, original['order_id'], duplicate['order_id'], key)
    assert [b['code'] for b in back['blockers']] == ['RELATED_SO_CANCELLED']
    assert back['can_commit'] is False


def test_cancel_as_duplicate_of_itself_or_a_cancelled_order_is_blocked(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    own = cancel_duplicate(client, duplicate['order_id'], duplicate['order_id'], key)
    assert [b['code'] for b in own['blockers']] == ['RELATED_SO_INVALID']
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='other', status='cancelled' WHERE id=%s",
                      (original['order_id'],))
    gone = cancel_duplicate(client, duplicate['order_id'], original['order_id'], key)
    assert [b['code'] for b in gone['blockers']] == ['RELATED_SO_CANCELLED']
    # Superseded keeps its old rule: the replacing order may be in any state.
    superseded = prepare(client, f"/sales/orders/{duplicate['order_id']}/cancel/prepare",
                         {'reason': 'superseded', 'related_so_id': original['order_id']}, key)
    assert superseded['blockers'] == [] and superseded['can_commit']


def test_original_cancelled_after_prepare_refuses_the_commit(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    prepared = cancel_duplicate(client, duplicate['order_id'], original['order_id'], key)
    assert prepared['can_commit']
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='other', status='cancelled' WHERE id=%s",
                      (original['order_id'],))
    response = commit(client, prepared, key)
    # The commit re-checks the live state and refuses; nothing is cancelled.
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] in ('TICKET_STALE', 'RELATED_SO_CANCELLED')
    assert order_row(db_cursor, duplicate['order_id'])['state'] == 'open'


def test_direct_cancel_route_has_the_same_cancelled_original_guard(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='other', status='cancelled' WHERE id=%s",
                      (original['order_id'],))
    response = client.post(f"/sales/orders/{duplicate['order_id']}/cancel",
                           json={'reason': 'duplicate', 'related_so_id': original['order_id'], 'mode': 'preview'},
                           headers=headers(None))
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == 'RELATED_SO_CANCELLED'


def test_reservations_are_shown_then_released_by_the_existing_cancel(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    line = duplicate['lines'][0]
    allocation = _allocate(db_cursor, duplicate['order_id'], line['line_id'], catalog['products']['food'], 25)
    data = check(client, duplicate['order_id'], key)
    assert data['reservations'] == {'count': 1, 'quantity_lb': 25.0}
    db_cursor.execute('SELECT status FROM sales_order_allocations WHERE id=%s', (allocation,))
    assert db_cursor.fetchone()['status'] == 'active', 'reading must not release anything'
    prepared = cancel_duplicate(client, duplicate['order_id'], original['order_id'], key)
    assert [r['id'] for r in prepared['draft']['reservations_released']] == [allocation]
    commit(client, prepared, key)
    db_cursor.execute('SELECT status FROM sales_order_allocations WHERE id=%s', (allocation,))
    assert db_cursor.fetchone()['status'] == 'released'


def test_shipped_duplicate_is_reported_and_cannot_be_cancelled(client, db_cursor, catalog, actors):
    key = office(actors)
    original = create_order(client, catalog, key)
    duplicate = create_order(client, catalog, key)
    food = catalog['products']['food']
    db_cursor.execute('INSERT INTO lots(product_id, lot_code) VALUES (%s, %s) RETURNING id', (food, 'SST-LOT-DUP-' + uuid4().hex[:8].upper()))
    _post_ship(db_cursor, duplicate['lines'][0]['line_id'], food, db_cursor.fetchone()['id'], 25)
    data = check(client, duplicate['order_id'], key)
    assert data['cancellable'] is False
    assert [b['code'] for b in data['blockers']] == ['ORDER_ALREADY_SHIPPED']
    assert data['order']['fulfillment'] != 'unshipped' and data['order']['shipped_lb'] == 25.0
    prepared = cancel_duplicate(client, duplicate['order_id'], original['order_id'], key)
    assert [b['code'] for b in prepared['blockers']] == ['ORDER_ALREADY_SHIPPED']
    assert order_row(db_cursor, duplicate['order_id'])['state'] == 'open'


def test_duplicate_check_needs_a_key_and_reports_a_missing_order(client, catalog, actors):
    order = create_order(client, catalog, office(actors))
    assert client.get(f"/sales/orders/{order['order_id']}/duplicate-check", headers={'X-API-Key': ''}).status_code == 401
    missing = client.get('/sales/orders/999999999/duplicate-check', headers=headers(office(actors)))
    assert missing.status_code == 404

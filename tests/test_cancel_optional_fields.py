"""Cancel order: the reason is the only required input (Michael, 2026-11-17).

The note and the related order are optional for EVERY cancel reason, Duplicate
and Replaced (superseded) included, on the ticket path the Office page uses
(prepare -> commit) and on the direct route (preview -> commit). A related order
or note that IS supplied is still stored and still validated (self, missing,
cancelled original, wrong reason). The eligibility rules do not move: shipped
and already-cancelled orders are still refused, and nothing but the cancelled
order changes.
"""
from uuid import uuid4

import pytest

from tests.test_order_tickets import client, actors, catalog, create_order, prepare, office, order_row  # noqa: F401
from tests.test_write_tickets import headers, commit, ticket_row
from tests.test_sales_order_state_model import _allocate, _post_ship

pytestmark = pytest.mark.db

REASONS = ['customer_cancelled', 'cns_declined', 'duplicate', 'superseded', 'other']


def cancel_prepare(client, order_id, key, **body):
    return prepare(client, f'/sales/orders/{order_id}/cancel/prepare', body, key)


def audit_rows(cur, actor_id, order_id):
    cur.execute("SELECT route, target_table, target_id FROM actor_write_audit "
                "WHERE actor_id=%s AND target_table='sales_orders' AND target_id=%s",
                (actor_id, order_id))
    return cur.fetchall()


def other_orders_snapshot(cur, order_id):
    cur.execute('SELECT id, state, state_reason, status FROM sales_orders WHERE id<>%s ORDER BY id', (order_id,))
    return cur.fetchall()


@pytest.mark.parametrize('blank', [{}, {'note': None, 'related_so_id': None}, {'note': '', 'related_so_id': None},
                                   {'note': '   '}])
@pytest.mark.parametrize('reason', REASONS)
def test_ticket_cancel_needs_only_the_reason(client, db_cursor, catalog, actors, reason, blank):
    key = office(actors)
    order = create_order(client, catalog, key)
    prepared = cancel_prepare(client, order['order_id'], key, reason=reason, **blank)
    assert prepared['blockers'] == [] and prepared['can_commit'], prepared
    assert order_row(db_cursor, order['order_id'])['state'] == 'open', 'prepare is a dry run'
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt['receipt_number'].startswith('ORD-')
    assert receipt['state'] == 'cancelled' and receipt['state_reason'] == reason
    row = order_row(db_cursor, order['order_id'])
    assert (row['state'], row['status'], row['state_reason']) == ('cancelled', 'cancelled', reason)
    assert row['state_note'] is None, 'a blank note is stored as NULL'
    assert row['related_so_id'] is None
    assert row['state_changed_by'] == actors['office']['name']
    # The audit trail keeps the reason: the committed ticket payload and the named-actor write row.
    ticket = ticket_row(db_cursor, prepared)
    assert ticket['payload']['reason'] == reason
    assert audit_rows(db_cursor, actors['office']['id'], order['order_id'])


@pytest.mark.parametrize('reason', REASONS)
def test_direct_cancel_preview_and_commit_need_only_the_reason(client, db_cursor, catalog, actors, reason):
    order = create_order(client, catalog, office(actors))
    path = f"/sales/orders/{order['order_id']}/cancel"
    preview = client.post(path, json={'reason': reason, 'note': '', 'mode': 'preview'}, headers=headers(None))
    assert preview.status_code == 200, preview.text
    assert preview.json()['resulting_state_reason'] == reason
    done = client.post(path, json={'reason': reason, 'mode': 'commit'}, headers=headers(None))
    assert done.status_code == 200, done.text
    body = done.json()
    assert (body['state'], body['state_reason'], body['state_note'], body['related_so_id']) == (
        'cancelled', reason, None, None)


@pytest.mark.parametrize('reason', ['duplicate', 'superseded'])
def test_voluntary_related_order_and_note_are_still_saved(client, db_cursor, catalog, actors, reason):
    key = office(actors)
    kept = create_order(client, catalog, key)
    order = create_order(client, catalog, key)
    note = f'{reason} of {kept["order_number"]}'
    prepared = cancel_prepare(client, order['order_id'], key, reason=reason,
                              related_so_id=kept['order_id'], note=note)
    assert prepared['can_commit'], prepared
    commit(client, prepared, key)
    row = order_row(db_cursor, order['order_id'])
    assert (row['state_reason'], row['related_so_id'], row['state_note']) == (reason, kept['order_id'], note)
    assert order_row(db_cursor, kept['order_id'])['state'] == 'open'


@pytest.mark.parametrize('reason', ['customer_cancelled', 'cns_declined', 'other'])
def test_voluntary_note_is_saved_for_other_reasons(client, db_cursor, catalog, actors, reason):
    key = office(actors)
    order = create_order(client, catalog, key)
    commit(client, cancel_prepare(client, order['order_id'], key, reason=reason, note='  called Tuesday  '), key)
    assert order_row(db_cursor, order['order_id'])['state_note'] == '  called Tuesday  '


def test_a_supplied_related_order_is_still_validated(client, db_cursor, catalog, actors):
    key = office(actors)
    kept = create_order(client, catalog, key)
    order = create_order(client, catalog, key)
    codes = lambda prepared: [b['code'] for b in prepared['blockers']]
    assert codes(cancel_prepare(client, order['order_id'], key, reason='duplicate',
                                related_so_id=order['order_id'])) == ['RELATED_SO_INVALID']
    assert codes(cancel_prepare(client, order['order_id'], key, reason='superseded',
                                related_so_id=order['order_id'])) == ['RELATED_SO_INVALID']
    assert codes(cancel_prepare(client, order['order_id'], key, reason='duplicate',
                                related_so_id=999999999)) == ['RELATED_SO_NOT_FOUND']
    assert codes(cancel_prepare(client, order['order_id'], key, reason='customer_cancelled',
                                related_so_id=kept['order_id'])) == ['RELATED_SO_NOT_APPLICABLE']
    db_cursor.execute("UPDATE sales_orders SET state='cancelled', state_reason='other', status='cancelled' WHERE id=%s",
                      (kept['order_id'],))
    assert codes(cancel_prepare(client, order['order_id'], key, reason='duplicate',
                                related_so_id=kept['order_id'])) == ['RELATED_SO_CANCELLED']
    assert order_row(db_cursor, order['order_id'])['state'] == 'open'


@pytest.mark.parametrize('reason', REASONS)
def test_shipped_and_already_cancelled_orders_are_still_refused(client, db_cursor, catalog, actors, reason):
    key = office(actors)
    shipped = create_order(client, catalog, key)
    food = catalog['products']['food']
    db_cursor.execute('INSERT INTO lots(product_id, lot_code) VALUES (%s, %s) RETURNING id',
                      (food, 'SST-LOT-OPT-' + uuid4().hex[:8].upper()))
    _post_ship(db_cursor, shipped['lines'][0]['line_id'], food, db_cursor.fetchone()['id'], 25)
    refused = cancel_prepare(client, shipped['order_id'], key, reason=reason)
    assert [b['code'] for b in refused['blockers']] == ['ORDER_ALREADY_SHIPPED'] and not refused['can_commit']
    assert order_row(db_cursor, shipped['order_id'])['state'] == 'open'

    order = create_order(client, catalog, key)
    commit(client, cancel_prepare(client, order['order_id'], key, reason=reason), key)
    again = cancel_prepare(client, order['order_id'], key, reason=reason)
    assert [b['code'] for b in again['blockers']] == ['ORDER_NOT_OPEN'] and not again['can_commit']


def test_blank_cancel_releases_only_its_own_reservations_and_touches_nothing_else(client, db_cursor, catalog, actors):
    key = office(actors)
    neighbour = create_order(client, catalog, key)
    order = create_order(client, catalog, key)
    food = catalog['products']['food']
    mine = _allocate(db_cursor, order['order_id'], order['lines'][0]['line_id'], food, 25)
    theirs = _allocate(db_cursor, neighbour['order_id'], neighbour['lines'][0]['line_id'], food, 10)
    before = other_orders_snapshot(db_cursor, order['order_id'])
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    transactions = db_cursor.fetchone()['n']
    prepared = cancel_prepare(client, order['order_id'], key, reason='duplicate')
    assert [r['id'] for r in prepared['draft']['reservations_released']] == [mine]
    commit(client, prepared, key)
    db_cursor.execute('SELECT id, status FROM sales_order_allocations WHERE id IN (%s, %s) ORDER BY id', (mine, theirs))
    assert {r['id']: r['status'] for r in db_cursor.fetchall()} == {mine: 'released', theirs: 'active'}
    assert other_orders_snapshot(db_cursor, order['order_id']) == before
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == transactions, 'cancelling posts no inventory transaction'

"""Cutover reset (Michael, 2026-10-13): a physical_count adjust ticket that names the
lot's open SHORTAGE exceptions closes them as 'counted' when it posts — receipt number,
count time and the preparer's key — instead of being refused. Everything else about
the A3b "never add stock to cover a shortage" rule is unchanged."""
from datetime import timedelta

import pytest

import main
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, error, posted_count  # noqa: F401
from tests.test_write_tickets_part2 import items, prepare  # noqa: F401
from tests.test_exceptions_enforcement import (adjust, open_shortage, exceptions_for, approve,  # noqa: F401
                                               private_evidence_volume)

pytestmark = pytest.mark.db


def closing(items_, delta, ids, **extra):
    return adjust(items_, delta, resolves_shortage_ids=ids, **extra)


def test_named_shortage_closes_as_counted_with_the_receipt(client, db_cursor, items, actors):
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    lot_id = items['ingredient']['lot_id']
    assert main.lot_on_hand(db_cursor, lot_id) == -6.0
    owner = actors['owner']['key']
    counted_at = (main.get_plant_now() - timedelta(hours=2)).isoformat()
    prepared = prepare(client, 'adjust', closing(items, 9, [exception_id], note='OPENING BALANCE test',
                                                 occurred_at=counted_at, backfill=True), owner)
    assert prepared['can_commit'] and prepared['draft']['resolves_shortage_ids'] == [exception_id]
    posted = commit(client, prepared, owner)
    assert posted.status_code == 200, posted.text
    result = posted.json()
    assert result['receipt_number'].startswith('ADJ-') and result['reason_code'] == 'physical_count'
    assert main.lot_on_hand(db_cursor, lot_id) == 3.0
    [closed] = result['shortages_closed']
    assert closed['exception_id'] == exception_id
    res = closed['resolution']
    assert (res['kind'], res['via'], res['book_before_lb'], res['delta_lb'], res['counted_lb'], res['short_lb']) == \
        ('counted', 'adjust_ticket', -6.0, 9.0, 3.0, 6.0)
    assert res['receipt_number'] == result['receipt_number'] and res['transaction_id'] == result['transaction_id']
    [row] = exceptions_for(db_cursor, id=exception_id)
    assert (row['status'], row['resolution_kind'], row['resolution_ticket_id'], row['resolved_by_actor_id']) == \
        ('resolved', 'counted', prepared['ticket_id'], actors['owner']['id'])
    db_cursor.execute('SELECT status, resolution_kind, resolution_ticket_id FROM shortage_flags WHERE exception_id=%s', (exception_id,))
    assert dict(db_cursor.fetchone()) == {'status': 'resolved', 'resolution_kind': 'counted', 'resolution_ticket_id': prepared['ticket_id']}
    db_cursor.execute('SELECT receipt_number, reason_code, operator_id, notes, occurred_at FROM transactions WHERE id=%s',
                      (result['transaction_id'],))
    txn = db_cursor.fetchone()
    assert txn['receipt_number'] == result['receipt_number'] and txn['operator_id'] == actors['owner']['name']
    assert 'OPENING BALANCE test' in txn['notes']
    assert txn['occurred_at'].isoformat()[:16] == counted_at[:16]


def test_without_naming_it_the_rule_still_refuses(client, db_cursor, items, actors):
    open_shortage(client, db_cursor, items, actors)
    response = client.post('/adjust/prepare', json=adjust(items, 9), headers=headers(actors['owner']['key']))
    error(response, 409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')


def test_only_physical_count_and_the_exact_open_set(client, db_cursor, items, actors):
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    owner = headers(actors['owner']['key'])
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    before = db_cursor.fetchone()['n']
    error(client.post('/adjust/prepare', json=closing(items, 9, [exception_id], reason='missing_receipt'), headers=owner),
          422, 'SHORTAGE_CLOSE_REASON')
    error(client.post('/adjust/prepare', json=closing(items, 9, [exception_id + 999]), headers=owner), 409, 'SHORTAGE_SET_CHANGED')
    error(client.post('/adjust/prepare', json=closing(items, 9, [exception_id, exception_id]), headers=owner), 409, 'SHORTAGE_SET_CHANGED')
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == before    # refused, never drafted


def test_shared_and_office_keys_cannot_close_shortages(client, db_cursor, items, actors):
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    error(client.post('/adjust/prepare', json=closing(items, 9, [exception_id]), headers=headers()), 403, 'ROLE_NOT_ALLOWED')
    error(client.post('/adjust/prepare', json=closing(items, 9, [exception_id]), headers=headers(actors['office']['key'])),
          403, 'ROLE_NOT_ALLOWED')
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'


def test_shortage_closed_elsewhere_after_prepare_makes_the_ticket_stale(client, db_cursor, items, actors):
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    owner = actors['owner']['key']
    prepared = prepare(client, 'adjust', closing(items, 9, [exception_id]), owner)
    resolved = client.post(f'/exceptions/{exception_id}/resolve', headers=headers(actors['floor']['key']),
                           json={'resolution_kind': 'counted', 'counted_lb': 1, 'note': 'counted by the floor first'})
    assert resolved.status_code == 200, resolved.text
    stale = commit(client, prepared, owner)
    assert stale.status_code == 409 and stale.json()['detail']['error_code'] == 'TICKET_STALE'
    assert stale.json()['detail']['blockers'][0]['code'] == 'SHORTAGE_SET_CHANGED'
    assert posted_count(db_cursor, prepared) == 0
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 1.0


def test_large_count_is_held_then_owner_approval_posts_and_closes(client, db_cursor, items, actors):
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    owner = actors['owner']['key']
    prepared = prepare(client, 'adjust', closing(items, 606, [exception_id]), owner)
    held = commit(client, prepared, owner)
    assert held.status_code == 202, held.text
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'    # nothing closes while held
    approved = approve(client, held.json()['exception_id'], owner, note='Signed cutover reset preview')
    assert approved.status_code == 200, approved.text
    assert approved.json()['shortages_closed'][0]['exception_id'] == exception_id
    assert approved.json()['receipt_number'].startswith('ADJ-')
    assert exceptions_for(db_cursor, id=exception_id)[0]['resolution_kind'] == 'counted'
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 600.0


def test_ordinary_adjust_payload_is_unchanged(client, db_cursor, items, actors):
    prepared = prepare(client, 'adjust', adjust(items, -2), actors['owner']['key'])
    db_cursor.execute('SELECT payload FROM write_tickets WHERE id=%s', (prepared['ticket_id'],))
    assert 'resolves_shortage_ids' not in db_cursor.fetchone()['payload']

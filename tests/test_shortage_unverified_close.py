"""Resolve a stock shortage without a physical count (owner decision, Michael, 2026-11-05).

'unverified' closes the shortage on a note alone: no stock posts, the lot keeps its
book balance, and the closed exception is marked unverified. Gated on migration 092.
Real HTTP routes against the local test database (never staging/prod).
"""
from pathlib import Path

import psycopg2
import pytest

import main
import exceptions_enforcement as a3b
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, error, isolated_database  # noqa: F401
from tests.test_write_tickets_part2 import items  # noqa: F401
from tests.test_exceptions_enforcement import post_short_make

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / 'migrations/092_shortage_unverified_close.sql'
DOWN = ROOT / 'migrations/down/092_shortage_unverified_close_down.sql'
NOTE = 'Bag was empty when we looked; cannot count, the make already used it'


@pytest.fixture
def migrated(db_cursor):
    db_cursor.execute(MIGRATION.read_text())


def shortage(client_, db_cursor, items_, actors_):
    _, result = post_short_make(client_, db_cursor, items_, actors_['floor']['key'])
    return result['shortages'][0]['exception_id']


def resolve(client_, eid, key, **body):
    return client_.post(f'/exceptions/{eid}/resolve', json=body, headers=headers(key))


def ledger_state(cur, lot_id):
    cur.execute('SELECT count(*) AS n FROM transactions')
    count = cur.fetchone()['n']
    cur.execute('SELECT count(*) AS n FROM transaction_lines WHERE lot_id=%s', (lot_id,))
    return count, cur.fetchone()['n'], float(main.lot_on_hand(cur, lot_id))


def status_of(cur, eid):
    cur.execute('SELECT status, resolution_kind FROM exceptions WHERE id=%s', (eid,))
    return tuple(cur.fetchone().values())


def test_resolve_without_count_changes_no_stock_and_is_marked_unverified(client, db_cursor, items, actors, migrated):
    eid = shortage(client, db_cursor, items, actors)
    lot_id = items['ingredient']['lot_id']
    before = ledger_state(db_cursor, lot_id)
    assert before[2] == -6.0                       # the make left the lot 6 lb short on the books
    r = resolve(client, eid, actors['floor']['key'], resolution_kind='unverified', note=NOTE)
    assert r.status_code == 200, r.text
    view = r.json()
    assert view['status'] == 'resolved' and view['resolution_kind'] == 'unverified'
    assert view['unverified'] is True
    assert view['resolution_note'] == NOTE and view['resolved_by']['id'] == actors['floor']['id']
    res = view['detail']['resolution']
    assert res == {'kind': 'unverified', 'verified': False, 'inventory_changed': False,
                   'short_lb': 6.0, 'book_balance_lb': -6.0}
    # No inventory change: no transaction, no line, same balance, no fixing ticket.
    assert ledger_state(db_cursor, lot_id) == before
    assert view['resolution_ticket_id'] is None and view['resolution_posting_receipt_number'] is None
    assert view['shortage_flag']['status'] == 'resolved' and view['shortage_flag']['resolution_kind'] == 'unverified'
    # The decision still gets its own XR receipt and the named actor.
    xr = view['resolution_receipt_number']
    assert xr.startswith('XR-')
    receipt = client.get('/receipts/' + xr, headers=headers()).json()
    assert receipt['action'] == 'resolve_exception' and receipt['transactions'] == []
    db_cursor.execute("SELECT actor_id FROM actor_write_audit WHERE target_table='exceptions' AND target_id=%s", (eid,))
    assert actors['floor']['id'] in [row['actor_id'] for row in db_cursor.fetchall()]
    error(resolve(client, eid, actors['floor']['key'], resolution_kind='unverified', note='again'), 409, 'EXCEPTION_CLOSED')


@pytest.mark.parametrize('note', [None, '', '   ', '\t\n'])
def test_resolve_without_count_requires_a_note(client, db_cursor, items, actors, migrated, note):
    eid = shortage(client, db_cursor, items, actors)
    before = ledger_state(db_cursor, items['ingredient']['lot_id'])
    body = {'resolution_kind': 'unverified'} | ({} if note is None else {'note': note})
    r = resolve(client, eid, actors['floor']['key'], **body)
    assert r.status_code == 422, r.text
    assert status_of(db_cursor, eid) == ('open', None)
    assert ledger_state(db_cursor, items['ingredient']['lot_id']) == before


@pytest.mark.parametrize('extra', [{'counted_lb': 0}, {'counted_lb': 4}, {'receipt_number': 'RCV-261105-001'},
                                   {'attachment_ref': 'photo-1'}])
def test_resolve_without_count_takes_only_a_note(client, db_cursor, items, actors, migrated, extra):
    eid = shortage(client, db_cursor, items, actors)
    before = ledger_state(db_cursor, items['ingredient']['lot_id'])
    r = resolve(client, eid, actors['floor']['key'], resolution_kind='unverified', note=NOTE, **extra)
    error(r, 422, 'UNVERIFIED_NOTE_ONLY')
    assert status_of(db_cursor, eid) == ('open', None)
    assert ledger_state(db_cursor, items['ingredient']['lot_id']) == before


def test_without_migration_092_it_is_neither_offered_nor_accepted(client, db_cursor, items, actors):
    eid = shortage(client, db_cursor, items, actors)
    for role in ('floor', 'owner'):
        assert client.get(f'/exceptions/{eid}', headers=headers(actors[role]['key'])).json()['allowed_actions'] == \
            ['counted', 'missing_movement', 'voided']
    error(resolve(client, eid, actors['floor']['key'], resolution_kind='unverified', note=NOTE), 503, 'UNVERIFIED_CLOSE_UNAVAILABLE')
    error(client.post(f'/exceptions/{eid}/resolve/prepare', json={'resolution_kind': 'unverified', 'note': NOTE},
                      headers=headers(actors['floor']['key'])), 503, 'UNVERIFIED_CLOSE_UNAVAILABLE')
    assert status_of(db_cursor, eid) == ('open', None)


@pytest.mark.parametrize('role,expected', [('owner', ['counted', 'missing_movement', 'voided', 'unverified']),
                                           ('floor', ['counted', 'missing_movement', 'voided', 'unverified']),
                                           ('office', [])])
def test_offered_to_everyone_who_may_resolve_a_shortage(client, db_cursor, items, actors, migrated, role, expected):
    eid = shortage(client, db_cursor, items, actors)
    auth = headers(actors[role]['key'])
    assert client.get(f'/exceptions/{eid}', headers=auth).json()['allowed_actions'] == expected
    rows = client.get('/exceptions', headers=auth).json()['exceptions']
    assert next(r for r in rows if r['id'] == eid)['allowed_actions'] == expected


def test_only_shortages_offer_it(client, db_cursor, actors, migrated):
    db_cursor.execute("INSERT INTO exceptions(kind,severity,detail) VALUES ('UNIDENTIFIED_LOT','info','{}') RETURNING id")
    eid = db_cursor.fetchone()['id']
    assert 'unverified' not in client.get(f'/exceptions/{eid}', headers=headers(actors['owner']['key'])).json()['allowed_actions']
    error(resolve(client, eid, actors['owner']['key'], resolution_kind='unverified', note=NOTE), 422, 'RESOLUTION_KIND_INVALID')


def test_resolved_unverified_list_filter(client, db_cursor, items, actors, migrated):
    eid = shortage(client, db_cursor, items, actors)
    assert resolve(client, eid, actors['floor']['key'], resolution_kind='unverified', note=NOTE).status_code == 200
    listed = client.get('/exceptions?status=resolved&resolution_kind=unverified',
                        headers=headers(actors['owner']['key'])).json()['exceptions']
    assert [(r['id'], r['unverified']) for r in listed] == [(eid, True)]
    counted = client.get('/exceptions?status=resolved&resolution_kind=counted',
                         headers=headers(actors['owner']['key'])).json()['exceptions']
    assert eid not in [r['id'] for r in counted]


def test_prepared_ticket_path_closes_without_stock_change(client, db_cursor, items, actors, migrated):
    eid = shortage(client, db_cursor, items, actors)
    key = actors['floor']['key']
    before = ledger_state(db_cursor, items['ingredient']['lot_id'])
    prepared = client.post(f'/exceptions/{eid}/resolve/prepare', json={'resolution_kind': 'unverified', 'note': NOTE},
                           headers=headers(key))
    assert prepared.status_code == 200, prepared.text
    prepared = prepared.json()
    assert prepared['draft']['verified'] is False and prepared['draft']['inventory_changed'] is False
    r = commit(client, prepared, key)
    assert r.status_code == 200, r.text
    assert status_of(db_cursor, eid) == ('resolved', 'unverified')
    assert ledger_state(db_cursor, items['ingredient']['lot_id']) == before


def test_migration_092_rerunnable_and_down_refuses_while_used(client, db_cursor, items, actors, migrated):
    db_cursor.execute(MIGRATION.read_text())        # idempotent
    eid = shortage(client, db_cursor, items, actors)
    assert resolve(client, eid, actors['floor']['key'], resolution_kind='unverified', note=NOTE).status_code == 200
    db_cursor.execute('SAVEPOINT down092')
    with pytest.raises(psycopg2.Error, match='keep migration 092'):
        db_cursor.execute(DOWN.read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT down092')
    assert a3b.unverified_close_enabled(db_cursor)


def test_migration_092_down_restores_the_old_check(db_cursor, migrated):
    db_cursor.execute(DOWN.read_text())
    assert not a3b.unverified_close_enabled(db_cursor)
    db_cursor.execute("""SELECT pg_get_constraintdef(oid) AS d FROM pg_constraint
                         WHERE conname='shortage_flags_resolution_kind_check'""")
    assert 'unverified' not in db_cursor.fetchone()['d']
    db_cursor.execute(MIGRATION.read_text())
    assert a3b.unverified_close_enabled(db_cursor)

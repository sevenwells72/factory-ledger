"""Regression contract for D1/D2 receipt gaps and the count-tool threshold."""
from pathlib import Path

import pytest
import main
import exceptions_enforcement as a3b
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets import headers, commit, error, isolated_database
from tests.test_write_tickets_part2 import items, prepare
from tests.test_exceptions_enforcement import post_short_make, adjust, set_balance

pytestmark = pytest.mark.db


def shortage(client, db_cursor, items, actors):
    _, result = post_short_make(client, db_cursor, items, actors['floor']['key'])
    return result['shortages'][0]['exception_id'], result


def resolve(client, eid, actors, **extra):
    return client.post(f'/exceptions/{eid}/resolve', headers=headers(actors['floor']['key']),
        json={'resolution_kind': 'counted', 'counted_lb': 4, 'note': 'Counted on pallet', **extra})


def test_counted_adjustment_and_resolution_have_distinct_linked_receipts(client, db_cursor, items, actors):
    eid, source = shortage(client, db_cursor, items, actors)
    result = resolve(client, eid, actors)
    assert result.status_code == 200, result.text
    view = result.json()
    detail = view['detail']['resolution']
    assert detail['receipt_number'].startswith('ADJ-')
    assert view['resolution_receipt_number'].startswith('XR-')
    assert view['receipt_number'] == source['receipt_number']  # still the SOURCE
    assert view['resolution_posting_receipt_number'] == detail['receipt_number']
    for number in (detail['receipt_number'], view['resolution_receipt_number']):
        receipt = client.get('/receipts/' + number, headers=headers()).json()
        assert receipt['status'] == 'committed'
        assert receipt['actor']['id'] == actors['floor']['id']
    db_cursor.execute('SELECT receipt_number,ticket_id FROM transactions WHERE id=%s', (detail['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'receipt_number': detail['receipt_number'], 'ticket_id': view['resolution_ticket_id']}


def test_nonstock_resolution_gets_xr(client, db_cursor, actors):
    db_cursor.execute("INSERT INTO exceptions(kind,severity,detail) VALUES ('LATE_ENTRY','info','{}') RETURNING id")
    eid = db_cursor.fetchone()['id']
    r = client.post(f'/exceptions/{eid}/resolve', headers=headers(actors['owner']['key']),
                    json={'resolution_kind': 'acknowledged', 'note': 'Reviewed'})
    assert r.status_code == 200, r.text
    number = r.json()['resolution_receipt_number']
    assert number.startswith('XR-')
    receipt = client.get('/receipts/' + number, headers=headers()).json()
    assert receipt['action'] == 'resolve_exception'
    assert receipt['response']['id'] == eid
    assert receipt['transactions'] == []


def test_void_receipt_links_back_from_original(client, db_cursor, items, actors):
    key = actors['owner']['key']
    original = commit(client, prepare(client, 'adjust', adjust(items, -2), key), key).json()
    r = client.post('/void/' + str(original['transaction_id']), json={'reason': 'Wrong lot'}, headers=headers(key))
    assert r.status_code == 200, r.text
    number = r.json()['receipt_number']
    assert number.startswith('VD-')
    void = client.get('/receipts/' + number, headers=headers()).json()
    assert void['action'] == 'void' and void['response']['correction_id'] == r.json()['correction_id']
    old = client.get('/receipts/' + original['receipt_number'], headers=headers()).json()
    assert old['effective_status'] == 'voided' and old['void_receipt_number'] == number
    assert old['transactions'][0]['void_receipt_number'] == number


@pytest.mark.parametrize('kosher', [False, True])
@pytest.mark.parametrize('note', ['', '   ', None])
def test_approve_blank_note_is_optional_and_has_xr(client, db_cursor, items, actors, kosher, note):
    key = actors['floor']['key']
    if kosher:
        from tests.test_a12_review_fixes import held_found
        db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id=%s", (items['batch']['id'],))
        _, held, _ = held_found(client, db_cursor, items, actors)
    else:
        held = commit(client, prepare(client, 'adjust', adjust(items, 600), key), key).json()
    r = client.post(f"/exceptions/{held['exception_id']}/approve", json={'note': note}, headers=headers(actors['owner']['key']))
    assert r.status_code == 200, r.text
    assert r.json()['resolution_receipt_number'].startswith('XR-')
    replay = client.post(f"/exceptions/{held['exception_id']}/approve", json={}, headers=headers(actors['owner']['key'])).json()
    assert replay == r.json() | {'replayed': True}


@pytest.mark.parametrize('role,expected', [('owner', ['counted', 'missing_movement', 'voided']),
    ('floor', ['counted', 'missing_movement', 'voided']), ('office', [])])
def test_exception_allowed_actions_per_caller(client, db_cursor, items, actors, role, expected):
    eid, _ = shortage(client, db_cursor, items, actors)
    auth = headers(actors[role]['key'])
    detail = client.get(f'/exceptions/{eid}', headers=auth).json()
    assert detail['allowed_actions'] == expected
    rows = client.get('/exceptions', headers=auth).json()['exceptions']
    assert next(r for r in rows if r['id'] == eid)['allowed_actions'] == expected
    assert resolve(client, eid, actors).json()['allowed_actions'] == []


@pytest.mark.parametrize('delta,photo,pct', [(-40, False, .04), (-101, False, .101), (-501, True, .501)])
def test_case_unit_adjust_is_pounds_not_cases(client, db_cursor, items, actors, delta, photo, pct):
    db_cursor.execute("UPDATE products SET uom='case',case_size_lb=15 WHERE id=%s", (items['ingredient']['id'],))
    set_balance(db_cursor, items, 'ingredient', 1000)
    key = actors['floor']['key']
    draft = prepare(client, 'adjust', adjust(items, delta), key)
    review = draft['draft']['correction_review']
    assert review['lb_equivalent'] == delta
    assert review['pct_of_book'] == pct and review['photo_required'] is photo
    r = commit(client, draft, key)
    assert r.status_code == (202 if photo else 200), r.text


@pytest.mark.parametrize('endpoint', ['reject', 'write_off'])
@pytest.mark.parametrize('note', ['', '   ', None])
def test_reject_and_writeoff_still_require_note(client, db_cursor, items, actors, endpoint, note):
    if endpoint == 'reject':
        key = actors['floor']['key']
        held = commit(client, prepare(client, 'adjust', adjust(items, 600), key), key).json()
        eid, action, kind = held['exception_id'], 'reject', 'declined'
    else:
        db_cursor.execute("INSERT INTO exceptions(kind,severity,detail) VALUES ('UNIDENTIFIED_LOT','warn','{}') RETURNING id")
        eid, action, kind = db_cursor.fetchone()['id'], 'resolve', 'written_off'
    assert client.post(f'/exceptions/{eid}/{action}', json={'resolution_kind': kind, 'note': note},
                       headers=headers(actors['owner']['key'])).status_code == 422


def prepared_resolution(client, eid, actors, **extra):
    r = client.post(f'/exceptions/{eid}/resolve/prepare', headers=headers(actors['floor']['key']),
        json={'resolution_kind': 'counted', 'counted_lb': 4, 'note': 'Counted on pallet', **extra})
    assert r.status_code == 200, r.text
    return r.json()


def test_prepared_counted_resolution_replay_and_shift_summary(client, db_cursor, items, actors):
    from tests.test_shift_summary import summary, section
    # prepare records no stock and no numbered receipt
    eid, _ = shortage(client, db_cursor, items, actors)
    key = actors['floor']['key']
    draft = prepared_resolution(client, eid, actors)
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == -6
    db_cursor.execute('SELECT receipt_number FROM write_tickets WHERE id=%s', (draft['ticket_id'],))
    assert db_cursor.fetchone()['receipt_number'] is None
    recorded = commit(client, draft, key)
    assert recorded.status_code == 200, recorded.text
    r = recorded.json()
    assert r['receipt_number'].startswith('XR-') and r['adjustment_receipt_number'].startswith('ADJ-')
    assert r['exception']['resolution_receipt_number'] == r['receipt_number']
    assert commit(client, draft, key).json() == r | {'replayed': True}
    assert client.get('/receipts/' + r['receipt_number'], headers=headers(key)).json()['response'] == r
    doc = summary(client, key)
    adjusted = section(doc, 'adjusted')['rows']
    assert [row['receipt_number'] for row in adjusted] == [r['adjustment_receipt_number']]
    assert adjusted[0]['quantity']['lb'] == 10
    assert any(row['receipt_number'] == r['receipt_number'] for row in doc['other_receipts']['rows'])


def test_stale_count_does_not_overwrite_changed_stock(client, db_cursor, items, actors):
    eid, _ = shortage(client, db_cursor, items, actors)
    draft = prepared_resolution(client, eid, actors)
    set_balance(db_cursor, items, 'ingredient', 2)
    error(commit(client, draft, actors['floor']['key']), 409, 'TICKET_STALE')
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 2
    assert client.get(f'/exceptions/{eid}', headers=headers(actors['floor']['key'])).json()['status'] == 'open'


def test_resolution_failure_rolls_back_adjustment_receipts_and_counters(client, db_cursor, items, actors, monkeypatch):
    import resolution_receipts as receipts
    eid, _ = shortage(client, db_cursor, items, actors)
    db_cursor.execute('SELECT count(*) n FROM write_tickets')
    before = db_cursor.fetchone()['n']
    db_cursor.execute('SELECT prefix,business_date,next FROM receipt_counters ORDER BY prefix,business_date')
    counters = db_cursor.fetchall()
    original = receipts.finish
    def fail_xr(api, cur, receipt, response, refs):
        if receipt['receipt_number'].startswith('XR-'):
            raise RuntimeError('Injected failure after the ADJ and XR link')
        return original(api, cur, receipt, response, refs)
    monkeypatch.setattr(receipts, 'finish', fail_xr)
    with pytest.raises(RuntimeError, match='Injected failure'):
        resolve(client, eid, actors)
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == -6
    db_cursor.execute('SELECT count(*) n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == before
    db_cursor.execute('SELECT prefix,business_date,next FROM receipt_counters ORDER BY prefix,business_date')
    assert db_cursor.fetchall() == counters
    view = client.get(f'/exceptions/{eid}', headers=headers(actors['floor']['key'])).json()
    assert view['status'] == 'open' and view['resolution_receipt_number'] is None


def test_zero_count_correction_has_xr_without_fabricating_adj(client, db_cursor, items, actors):
    eid, _ = shortage(client, db_cursor, items, actors)
    set_balance(db_cursor, items, 'ingredient', 4)
    view = resolve(client, eid, actors).json()
    assert view['resolution_receipt_number'].startswith('XR-')
    assert view['resolution_ticket_id'] is None
    assert view['detail']['resolution']['transaction_id'] is None
    assert view['detail']['resolution']['delta_lb'] == 0


def test_partial_evidence_gets_immutable_xr_and_stays_open(client, db_cursor, items, actors):
    from tests.test_exceptions_enforcement import receive_onto
    eid, _ = shortage(client, db_cursor, items, actors)
    key = actors['floor']['key']
    small = receive_onto(client, db_cursor, items, key, 1, 2)
    draft = prepared_resolution(client, eid, actors, resolution_kind='missing_movement',
                                receipt_number=small['receipt_number'])
    first = commit(client, draft, key).json()
    assert first['exception']['status'] == 'open'
    assert first['exception']['evidence']['remaining_lb'] == 4
    assert commit(client, draft, key).json() == first | {'replayed': True}
    later = resolve(client, eid, actors).json()
    assert later['resolution_receipt_number'] != first['receipt_number']
    old = client.get('/receipts/' + first['receipt_number'], headers=headers(key)).json()
    assert old['response'] == first


def test_prepared_void_replays_and_restoration_clears_backlink(client, db_cursor, items, actors):
    key = actors['owner']['key']
    original = commit(client, prepare(client, 'adjust', adjust(items, -2), key), key).json()
    path = f"/void/{original['transaction_id']}/prepare"
    p = client.post(path, json={'reason': 'Mistake'}, headers=headers(key))
    assert p.status_code == 200, p.text
    result = commit(client, p.json(), key)
    assert result.status_code == 200, result.text
    r = result.json()
    assert r['receipt_number'].startswith('VD-')
    assert commit(client, p.json(), key).json() == r | {'replayed': True}
    main._append_transaction_correction(db_cursor, original['transaction_id'], 'restore', 'Restored', None, 'test')
    detail = client.get('/receipts/' + original['receipt_number'], headers=headers()).json()
    assert detail['void_receipt_number'] is None and detail['effective_status'] == 'posted'
    assert client.get('/receipts/' + r['receipt_number'], headers=headers()).json()['response'] == r


@pytest.mark.parametrize('kind,owner,floor', [('UNIDENTIFIED_LOT', ['identified', 'written_off'], ['identified']),
    ('LATE_ENTRY', ['acknowledged'], []), ('BOL_PHOTO_MISSING', ['photo_attached', 'waived'], [])])
def test_action_matrix_filters_owner_only_decisions(client, db_cursor, actors, kind, owner, floor):
    db_cursor.execute('INSERT INTO exceptions(kind,severity,detail) VALUES (%s,\'info\',\'{}\') RETURNING id', (kind,))
    eid = db_cursor.fetchone()['id']
    for role, expected in [('owner', owner), ('floor', floor), ('office', [])]:
        view = client.get(f'/exceptions/{eid}', headers=headers(actors[role]['key'])).json()
        assert view['allowed_actions'] == expected


def test_hold_actions_and_reject_xr_replay(client, db_cursor, items, actors):
    key = actors['floor']['key']
    held = commit(client, prepare(client, 'adjust', adjust(items, 600), key), key).json()
    eid = held['exception_id']
    for role, actions in [('owner', ['approve', 'reject']), ('floor', []), ('office', [])]:
        assert client.get(f'/exceptions/{eid}', headers=headers(actors[role]['key'])).json()['allowed_actions'] == actions
    body = {'resolution_kind': 'declined', 'note': 'Wrong lot'}
    owner = headers(actors['owner']['key'])
    r = client.post(f'/exceptions/{eid}/reject', json=body, headers=owner).json()
    assert r['resolution_receipt_number'].startswith('XR-')
    assert client.post(f'/exceptions/{eid}/reject', json=body, headers=owner).json() == r | {'replayed': True}
    assert client.get(f'/exceptions/{eid}', headers=owner).json()['allowed_actions'] == []


def test_void_permission_at_prepare_and_after_role_change(client, db_cursor, items, actors):
    owner, floor = actors['owner']['key'], actors['floor']['key']
    original = commit(client, prepare(client, 'adjust', adjust(items, -2), owner), owner).json()
    path = f"/void/{original['transaction_id']}/prepare"
    error(client.post(path, json={'reason': 'Mistake'}, headers=headers(floor)), 403, 'VOID_OWNER_REQUIRED')
    r = client.post(path, json={'reason': 'Mistake'}, headers=headers(owner)).json()
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (actors['owner']['id'],))
    error(commit(client, r, owner), 403, 'ROLE_NOT_ALLOWED')


def test_migration_is_idempotent_preserves_actions_and_permissions(db_cursor):
    migration = (Path(__file__).parents[1] / 'migrations/080_resolution_receipts.sql').read_text()
    db_cursor.execute(migration)
    db_cursor.execute(migration)
    db_cursor.execute("SELECT pg_get_constraintdef(oid) definition FROM pg_constraint WHERE conrelid='write_tickets'::regclass AND conname='write_tickets_action_check'")
    definition = db_cursor.fetchone()['definition']
    for action in ('resolve_exception', 'void', 'update_supplier_lot', 'receive', 'adjust', 'ship_order'):
        assert action in definition
    db_cursor.execute("SELECT relrowsecurity FROM pg_class WHERE oid='write_tickets'::regclass")
    assert db_cursor.fetchone()['relrowsecurity']


@pytest.mark.parametrize('action', ['counted', 'void'])
def test_concurrent_prepared_commits_issue_receipts_once(isolated_database, monkeypatch, action):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from tests.test_exceptions_enforcement import Race, race_actors
    from tests.test_write_tickets_part2 import body
    keys, stock, actor_ids = race_actors(isolated_database)
    race = Race(isolated_database)
    monkeypatch.setattr(main, 'get_db_connection', race.connection)
    main._reset_actor_cache()
    http = TestClient(main.app)
    key = keys['floor']
    if action == 'counted':
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            set_balance(cur, stock, 'ingredient', 4)
        made = commit(http, prepare(http, 'make', body('make', stock), key), key).json()
        eid = made['shortages'][0]['exception_id']
        draft = http.post(f'/exceptions/{eid}/resolve/prepare', json={'resolution_kind':'counted',
            'counted_lb':4, 'note':'Concurrent physical count'}, headers=headers(key)).json()
    else:
        original = commit(http, prepare(http, 'adjust', adjust(stock, -2), key), key).json()
        draft = http.post(f"/void/{original['transaction_id']}/prepare", json={'reason':'Concurrent void'},
                          headers=headers(key)).json()
    assert draft['can_commit']
    race.on = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: commit(http, draft, key), range(2)))
    race.on = False
    assert all(r.status_code == 200 for r in responses)
    assert {r.json()['replayed'] for r in responses} == {False, True}
    assert len({r.json()['receipt_number'] for r in responses}) == 1
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if action == 'counted':
            assert main.lot_on_hand(cur, stock['ingredient']['lot_id']) == 4
            cur.execute("SELECT count(*) n FROM write_tickets WHERE action='adjust' AND status='committed'")
            assert cur.fetchone()['n'] == 1
        else:
            cur.execute("SELECT count(*) n FROM ledger_corrections WHERE target_id=%s AND event_type='void'", (original['transaction_id'],))
            assert cur.fetchone()['n'] == 1
    http.close()
    main._reset_actor_cache()

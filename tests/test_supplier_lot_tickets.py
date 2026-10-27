"""A5 part 2: real HTTP/PG corrections, identity/exception atomicity and races."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
import pytest

import main
import permissions
import supplier_lot_tickets
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import (payload, headers, commit, ticket_row, error,
    prepare as receive_prepare, seed, isolated_database)  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.db


def prepare(client, lot_id, key, **changes):
    response = client.post(f'/lots/{lot_id}/supplier-lot/prepare', headers=headers(key), json={
        'supplier_lot_code': ' SUP-2026/8812-A ', 'reason_code': 'data_entry_error',
        'note': 'Supplier label recovered.', **changes})
    assert response.status_code == 200, response.text
    return response.json()


def get_lot(cur, lot_id):
    cur.execute('SELECT * FROM lots WHERE id=%s', (lot_id,))
    return dict(cur.fetchone())


def get_exception(cur, exception_id):
    cur.execute('SELECT * FROM exceptions WHERE id=%s', (exception_id,))
    return dict(cur.fetchone())


@pytest.fixture
def unidentified(client, payload, actors):
    draft = receive_prepare(client, payload | {'supplier_lot_code': 'UNKNOWN', 'no_supplier_lot': True}, actors['floor']['key'])
    response = commit(client, draft, actors['floor']['key'])
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['identity_status'] == 'unidentified' and result['exception_id']
    return result


@pytest.mark.parametrize('role', ['owner', 'floor', 'office'])
@pytest.mark.parametrize('session', [False, True])
def test_correct_identify_audit_and_replay(client, db_cursor, actors, unidentified, role, session, monkeypatch):
    actor = actors[role]
    key = actor['key']
    if session:
        monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
        monkeypatch.setenv('PIN_LOGIN_ENABLED', '0')
        signed = client.post('/auth/session', json={'actor_key': key})
        assert signed.status_code == 200
        key = signed.json()['session_token']
    before = get_lot(db_cursor, unidentified['lot_id'])
    balance = main.lot_on_hand(db_cursor, before['id'])
    draft = prepare(client, before['id'], key, client_source='dashboard')
    assert draft['can_commit'] and draft['action'] == 'update_supplier_lot'
    assert draft['draft']['exception_id'] == unidentified['exception_id']
    assert draft['draft']['previous_supplier_lot_code'] == 'UNKNOWN'
    assert draft['draft']['supplier_lot_code'] == 'SUP-2026/8812-A'
    assert get_lot(db_cursor, before['id']) == before
    row = ticket_row(db_cursor, draft)
    assert row['expires_at']-row['prepared_at'] == timedelta(minutes=30)
    result = commit(client, draft, key)
    assert result.status_code == 200, result.text
    result = result.json()
    assert result['receipt_number'].startswith('LOT-')
    assert result['state_changed'] is False and result['replayed'] is False
    assert result['resolved_exception_ids'] == [unidentified['exception_id']]
    assert result['entered_by_actor_id'] == actor['id']
    assert result['entry_timing']['entered_by']['id'] == actor['id']
    after = get_lot(db_cursor, before['id'])
    assert after['identity_status'] == 'identified' and after['identify_by'] is None
    assert after['supplier_lot_code'] == 'SUP-2026/8812-A'
    assert after['supplier_id'] == before['supplier_id'] and after['lot_code'] == before['lot_code']
    assert main.lot_on_hand(db_cursor, before['id']) == balance
    exception = get_exception(db_cursor, unidentified['exception_id'])
    assert exception['status'] == 'resolved' and exception['resolution_kind'] == 'identified'
    assert exception['resolved_by_actor_id'] == actor['id'] and exception['resolved_at']
    assert exception['resolution_ticket_id'] == draft['ticket_id']
    assert exception['resolution_note'] == 'Supplier label recovered.'
    db_cursor.execute('SELECT * FROM lot_events WHERE ticket_id=%s', (draft['ticket_id'],))
    events = db_cursor.fetchall()
    assert len(events) == 1
    event = events[0]
    assert event['id'] == result['lot_event_id'] and event['event'] == 'supplier_lot'
    assert event['old']['supplier_lot_code'] == 'UNKNOWN' and event['old']['identity_status'] == 'unidentified'
    assert event['new'] == {'supplier_lot_code': 'SUP-2026/8812-A', 'identity_status': 'identified', 'identify_by': None}
    assert event['reason_code'] == 'data_entry_error' and event['entered_by_actor_id'] == actor['id']
    assert event['transaction_id'] is None
    # Session and personal key identify the same person; replay works across them.
    assert commit(client, draft, actor['key']).json() == result | {'replayed': True}
    assert get_exception(db_cursor, exception['id']) == exception
    detail = client.get('/receipts/' + result['receipt_number'], headers=headers(key)).json()
    assert detail['response'] == result and detail['transactions'] == []
    assert detail['lots'][0]['id'] == before['id']
    assert detail['lots'][0]['identity_status'] == 'identified'
    db_cursor.execute('SELECT target_table,actor_id FROM actor_write_audit WHERE target_id=%s AND target_table=%s', (before['id'], 'lots'))
    assert any(r['actor_id'] == actor['id'] for r in db_cursor.fetchall())


@pytest.mark.parametrize('value', ['', '   ', 'N/A', 'n.a.', 'NA', 'unknown', 'Unidentified', 'none', 'null',
    'TBD', 'pending', '???', '---', '<script>bad</script>', 'SUP\x00A', 'SUP\nA', 'A'*201, None, 123, True, {}])
def test_blank_garbage_and_nonstring_code_rejected(client, db_cursor, unidentified, actors, value):
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    count = db_cursor.fetchone()['n']
    r = client.post(f'/lots/{unidentified["lot_id"]}/supplier-lot/prepare', headers=headers(actors['office']['key']),
                    json={'supplier_lot_code': value, 'reason_code': 'data_entry_error'})
    error(r, 422, 'SUPPLIER_LOT_CODE_INVALID')
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == count
    assert get_exception(db_cursor, unidentified['exception_id'])['status'] == 'open'


@pytest.mark.parametrize('reason', ['made_up', 'physical_count', 'missing_receipt', 'missing_production',
                                     'damage_disposal', 'unrecorded_usage', 'unknown'])
def test_only_applicable_fixed_reasons(client, unidentified, actors, reason):
    r = client.post(f'/lots/{unidentified["lot_id"]}/supplier-lot/prepare', headers=headers(actors['floor']['key']),
                    json={'supplier_lot_code': 'REAL-123', 'reason_code': reason, 'note': 'A note cannot widen the list.'})
    error(r, 422, 'REASON_CODE_INVALID')


def test_wrong_role_shared_keys_and_wrong_actor(client, db_cursor, unidentified, actors, monkeypatch):
    url = f'/lots/{unidentified["lot_id"]}/supplier-lot/prepare'
    body = {'supplier_lot_code': 'REAL-123', 'reason_code': 'wrong_lot'}
    error(client.post(url, headers=headers(), json=body), 403, 'ACTOR_REQUIRED')
    assert client.post(url, headers=headers(main.DASHBOARD_API_KEY), json=body).status_code == 403
    assert client.post(url, headers=headers(actors['retired']['key']), json=body).status_code == 403
    # All three named roles are allowed by §4.3; an unrecognized role fails closed.
    original = main._resolve_actor
    monkeypatch.setattr(main, '_resolve_actor', lambda key: dict(original(key), role='viewer') if key == actors['floor']['key'] else original(key))
    error(client.post(url, headers=headers(actors['floor']['key']), json=body), 403, 'ROLE_NOT_ALLOWED')
    monkeypatch.setattr(main, '_resolve_actor', original)
    draft = prepare(client, unidentified['lot_id'], actors['office']['key'], reason_code='wrong_lot')
    error(commit(client, draft, actors['floor']['key']), 403, 'TICKET_WRONG_USER')
    error(commit(client, draft), 403, 'TICKET_WRONG_USER')
    assert get_exception(db_cursor, unidentified['exception_id'])['status'] == 'open'


def test_expiry_hash_and_attribution_cannot_be_overridden(client, db_cursor, unidentified, actors):
    key = actors['floor']['key']
    url = f'/lots/{unidentified["lot_id"]}/supplier-lot/prepare'
    for injected in ({'entered_by_actor_id': actors['owner']['id']}, {'actor_id': actors['owner']['id']}, {'supplier_id': 1}):
        assert client.post(url, headers=headers(key), json={'supplier_lot_code': 'REAL-123', 'reason_code': 'wrong_lot', **injected}).status_code == 422
    draft = prepare(client, unidentified['lot_id'], key)
    row = ticket_row(db_cursor, draft)
    assert row['expires_at']-row['prepared_at'] == timedelta(minutes=10)
    error(commit(client, draft, key, payload_hash='0'*64), 409, 'TICKET_PAYLOAD_MISMATCH')
    assert commit(client, draft, key, supplier_lot_code='OTHER').status_code == 422
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s", (draft['ticket_id'],))
    error(commit(client, draft, key), 409, 'TICKET_EXPIRED')
    assert ticket_row(db_cursor, draft)['status'] == 'expired'
    assert get_exception(db_cursor, unidentified['exception_id'])['status'] == 'open'


def test_live_permission_and_reason_recheck(client, db_cursor, unidentified, actors, monkeypatch):
    key = actors['floor']['key']
    draft = prepare(client, unidentified['lot_id'], key)
    with monkeypatch.context() as policy:
        policy.setitem(permissions.ROLE_PERMISSIONS, 'update_supplier_lot', frozenset({'owner'}))
        error(commit(client, draft, key), 403, 'ROLE_NOT_ALLOWED')
    db_cursor.execute("UPDATE correction_reasons SET active=false WHERE code='data_entry_error'")
    r = commit(client, draft, key)
    error(r, 409, 'TICKET_STALE')
    assert r.json()['detail']['blockers'][0]['code'] == 'REASON_CODE_INVALID'
    assert get_exception(db_cursor, unidentified['exception_id'])['status'] == 'open'


def test_inactive_actor_rechecked(client, db_cursor, unidentified, actors):
    key = actors['office']['key']
    draft = prepare(client, unidentified['lot_id'], key)
    db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (actors['office']['id'],))
    r = commit(client, draft, key)
    error(r, 409, 'TICKET_STALE')
    assert r.json()['detail']['blockers'][0]['code'] == 'ACTOR_INACTIVE'


def test_correct_twice_resolves_once_and_rejects_stale_draft(client, db_cursor, unidentified, actors):
    key = actors['office']['key']
    first = prepare(client, unidentified['lot_id'], key)
    stale = prepare(client, unidentified['lot_id'], actors['floor']['key'], supplier_lot_code='OTHER-123')
    assert commit(client, first, key).status_code == 200
    resolved = get_exception(db_cursor, unidentified['exception_id'])
    r = commit(client, stale, actors['floor']['key'])
    error(r, 409, 'TICKET_STALE')
    assert r.json()['detail']['blockers'][0]['code'] == 'LOT_IDENTITY_CHANGED'
    fresh = prepare(client, unidentified['lot_id'], key, supplier_lot_code='OTHER-123')
    second = commit(client, fresh, key)
    assert second.status_code == 200, second.text
    assert second.json()['resolved_exception_ids'] == []
    assert get_exception(db_cursor, unidentified['exception_id']) == resolved


def test_escalated_and_count_packet_lot_without_receive(client, db_cursor, payload, actors):
    db_cursor.execute("""INSERT INTO lots(product_id,lot_code,identity_status,identify_by)
        VALUES (%s,'COUNT-UNIDENTIFIED','unidentified',CURRENT_DATE) RETURNING id""", (payload['product_id'],))
    lot_id = db_cursor.fetchone()['id']
    db_cursor.execute("""INSERT INTO exceptions(kind,status,severity,lot_id,product_id,detail,escalated_at)
        VALUES ('UNIDENTIFIED_LOT','escalated','warn',%s,%s,'{}',clock_timestamp()) RETURNING id""", (lot_id, payload['product_id']))
    eid = db_cursor.fetchone()['id']
    key = actors['office']['key']
    draft = prepare(client, lot_id, key)
    result = commit(client, draft, key)
    assert result.status_code == 200, result.text
    assert result.json()['resolved_exception_ids'] == [eid]
    assert get_exception(db_cursor, eid)['resolution_kind'] == 'identified'


@pytest.mark.parametrize('kind', ['merged', 'commingled', 'components', 'inactive'])
def test_unsafe_lots_block(client, db_cursor, unidentified, actors, kind):
    lid = unidentified['lot_id']
    if kind == 'merged':
        db_cursor.execute("UPDATE lots SET status='merged' WHERE id=%s", (lid,))
    elif kind == 'commingled':
        db_cursor.execute("UPDATE lots SET lot_type='commingled' WHERE id=%s", (lid,))
    elif kind == 'components':
        db_cursor.execute("INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code) VALUES (%s,'UNKNOWN')", (lid,))
    else:
        db_cursor.execute('UPDATE products SET active=false WHERE id=(SELECT product_id FROM lots WHERE id=%s)', (lid,))
    draft = prepare(client, lid, actors['floor']['key'])
    assert not draft['can_commit']
    error(commit(client, draft, actors['floor']['key']), 409, 'TICKET_STALE')
    assert get_exception(db_cursor, unidentified['exception_id'])['status'] == 'open'


def test_receive_never_resolves_or_extends_exception(client, db_cursor, payload, unidentified, actors):
    key = actors['floor']['key']
    before = get_exception(db_cursor, unidentified['exception_id'])
    missing = receive_prepare(client, payload | {'supplier_lot_code': 'UNKNOWN', 'no_supplier_lot': True}, key)
    r = commit(client, missing, key, acknowledged_warnings=['POSSIBLE_DUPLICATE'])
    assert r.status_code == 200, r.text
    assert get_exception(db_cursor, before['id']) == before
    identified = receive_prepare(client, payload | {'supplier_lot_code': 'REAL-123'}, key)
    assert identified['blockers'][0]['code'] == 'LOT_IDENTITY_REQUIRES_CORRECTION'
    error(commit(client, identified, key, acknowledged_warnings=['POSSIBLE_DUPLICATE']), 409, 'TICKET_STALE')
    assert get_exception(db_cursor, before['id']) == before
    # Legacy receive is still reachable until A10; it also must never close R8.
    r = client.post('/receive', headers=headers(key), json={'product_name': missing['draft']['product_name'],
        'shipper_name': missing['draft']['supplier_name'], 'cases': 1, 'case_size_lb': 25,
        'lot_code': payload['lot_code'], 'supplier_lot_code': 'REAL-123', 'bol_reference': 'A5 legacy top-up'})
    error(r, 422, 'LOT_IDENTITY_REQUIRES_CORRECTION')
    assert get_exception(db_cursor, before['id']) == before
    assert get_lot(db_cursor, unidentified['lot_id'])['identity_status'] == 'unidentified'


def test_receive_prepared_before_correction_cannot_replace_it(client, db_cursor, payload, unidentified, actors):
    key = actors['floor']['key']
    receive = receive_prepare(client, payload | {'supplier_lot_code': 'UNKNOWN', 'no_supplier_lot': True}, key)
    correction = prepare(client, unidentified['lot_id'], key)
    assert commit(client, correction, key).status_code == 200
    r = commit(client, receive, key, acknowledged_warnings=['POSSIBLE_DUPLICATE'])
    error(r, 409, 'TICKET_STALE')
    assert get_lot(db_cursor, unidentified['lot_id'])['supplier_lot_code'] == 'SUP-2026/8812-A'


def test_failure_rolls_back_lot_event_exception_and_ticket(client, db_cursor, unidentified, actors, monkeypatch):
    key = actors['floor']['key']
    draft = prepare(client, unidentified['lot_id'], key)
    old_lot = get_lot(db_cursor, unidentified['lot_id'])
    old_exception = get_exception(db_cursor, unidentified['exception_id'])
    original = supplier_lot_tickets.post
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected after lot event and exception resolution')
    monkeypatch.setattr(supplier_lot_tickets, 'post', crash)
    with pytest.raises(RuntimeError, match='injected'):
        commit(client, draft, key)
    assert get_lot(db_cursor, unidentified['lot_id']) == old_lot
    assert get_exception(db_cursor, unidentified['exception_id']) == old_exception
    assert ticket_row(db_cursor, draft)['status'] == 'prepared'
    db_cursor.execute('SELECT count(*) AS n FROM lot_events WHERE ticket_id=%s', (draft['ticket_id'],))
    assert db_cursor.fetchone()['n'] == 0
    monkeypatch.setattr(supplier_lot_tickets, 'post', original)
    assert commit(client, draft, key).status_code == 200


def test_migration_rerun_down_up_and_evidence_retention(client, db_cursor, unidentified, actors):
    up = (ROOT/'migrations/079_supplier_lot_tickets.sql').read_text()
    down = (ROOT/'migrations/down/079_supplier_lot_tickets_down.sql').read_text()
    db_cursor.execute(up)
    db_cursor.execute(up)
    db_cursor.execute(down)
    db_cursor.execute(down)
    db_cursor.execute(up)
    draft = prepare(client, unidentified['lot_id'], actors['floor']['key'])
    assert commit(client, draft, actors['floor']['key']).status_code == 200
    db_cursor.execute('SAVEPOINT retain_correction')
    with pytest.raises(psycopg2.Error, match='rollback refused'):
        db_cursor.execute(down)
    db_cursor.execute('ROLLBACK TO SAVEPOINT retain_correction')
    db_cursor.execute('SAVEPOINT event_immutable')
    with pytest.raises(psycopg2.Error):
        db_cursor.execute('DELETE FROM lot_events WHERE ticket_id=%s', (draft['ticket_id'],))
    db_cursor.execute('ROLLBACK TO SAVEPOINT event_immutable')


@pytest.mark.parametrize('same_ticket', [True, False])
def test_concurrent_commits_post_once(isolated_database, monkeypatch, same_ticket):
    key = 'a5-test-' + uuid4().hex
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        receive_body = seed(cur) | {'supplier_lot_code': 'UNKNOWN', 'no_supplier_lot': True}
        cur.execute("INSERT INTO actors(name,role,key_hash) VALUES (%s,'office',%s) RETURNING id",
                    ('A5 concurrent ' + uuid4().hex, sha256(key.encode()).hexdigest()))
        actor_id = cur.fetchone()['id']
    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='10s'")
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    main._reset_actor_cache()
    try:
        with TestClient(main.app) as http:
            received = commit(http, receive_prepare(http, receive_body, key), key).json()
            a = prepare(http, received['lot_id'], key)
            b = a if same_ticket else prepare(http, received['lot_id'], key, supplier_lot_code='DIFFERENT-123')
            gate = Barrier(2)
            def race(draft):
                gate.wait(timeout=10)
                return commit(http, draft, key)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(race, draft) for draft in (a, b)]
                results = [f.result(timeout=25) for f in futures]
            assert sorted(r.status_code for r in results) == ([200, 200] if same_ticket else [200, 409]), [r.text for r in results]
            if same_ticket:
                assert sorted(r.json()['replayed'] for r in results) == [False, True]
                assert len({r.json()['receipt_number'] for r in results}) == 1
            else:
                assert next(r for r in results if r.status_code == 409).json()['detail']['blockers'][0]['code'] == 'LOT_IDENTITY_CHANGED'
            with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT * FROM lot_events WHERE lot_id=%s', (received['lot_id'],))
                events = cur.fetchall()
                assert len(events) == 1 and events[0]['entered_by_actor_id'] == actor_id
                ex = get_exception(cur, received['exception_id'])
                assert ex['status'] == 'resolved' and ex['resolution_kind'] == 'identified'
                assert ex['resolution_ticket_id'] == events[0]['ticket_id']
                assert main.lot_on_hand(cur, received['lot_id']) == 50
    finally:
        main._reset_actor_cache()

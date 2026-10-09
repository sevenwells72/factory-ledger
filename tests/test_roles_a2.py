"""A2: roles enforced in FL, entered_by on every write, back-dating limits (design rev 3.7 §4.3, §6)."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Event
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
import pytest

import main
import permissions
import ticket_actions
import write_tickets
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import (headers, commit, ticket_row, posted_count, error,
                                      isolated_database, seed as seed_receive)  # noqa: F401
from tests.test_write_tickets_part2 import items, body, prepare, seed as seed_items  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.db
TICKET_ACTIONS = ['receive', 'make', 'pack', 'adjust', 'found']
ROLES = ['owner', 'floor', 'office', 'legacy_ledger', 'legacy_dashboard']

# Design §4.3 restated by hand — the module must match THIS, not the other way round.
EXPECTED = {
    'receive':                          'owner floor office legacy_ledger legacy_dashboard',
    'make':                             'owner floor legacy_ledger',
    'pack':                             'owner floor legacy_ledger',
    'adjust':                           'owner floor legacy_ledger',
    'found':                            'owner floor legacy_ledger',
    'void':                             'owner floor legacy_ledger',
    'rename_lot':                       'owner floor office legacy_ledger',
    'update_supplier_lot':              'owner floor office legacy_ledger',
    'move_lot':                         'owner floor office',
    'ship_order':                       'owner floor office legacy_ledger',
    'ship_standalone':                  'owner office legacy_ledger',
    'ship_bulk':                        'owner floor office legacy_ledger',
    'create_order':                     'owner office legacy_ledger',
    'add_order_lines':                  'owner office legacy_ledger',
    'update_order_line':                'owner office legacy_ledger',
    'cancel_order_line':                'owner office legacy_ledger',
    'update_order_header':              'owner office legacy_ledger',
    'update_order_status':              'owner office legacy_ledger',
    'mark_order_ready':                 'owner floor office legacy_ledger',
    'cancel_order':                     'owner office legacy_ledger',
    'close_order_shipped_not_recorded': 'owner legacy_ledger',
    'reopen_order':                     'owner legacy_ledger',
    'create_expected_receipt':          'owner office legacy_ledger',
    'manage_customers':                 'owner office legacy_ledger',
    'manage_aliases':                   'owner office',
    'list_exceptions':                  'owner floor office',
    'resolve_exception':                'owner floor',
    'approve_exception':                'owner',
    'kosher_attestation':               'owner',
    'backdate_over_14d':                'owner legacy_ledger',
}


def ident(role=None, key_kind='actor', name='Someone', id=1):
    return {'id': id if key_kind == 'actor' else None, 'name': name, 'role': role, 'key_kind': key_kind}


def identity_for(role):
    if role in ('legacy_ledger', 'legacy_dashboard'):
        return ident(None, role, name=None)
    return ident(role)


def key_for(which, actors):
    return {'legacy_ledger': main.API_KEY, 'legacy_dashboard': main.DASHBOARD_API_KEY}.get(which) or actors[which]['key']


def at(hours=0, days=0):
    return (main.get_plant_now() - timedelta(hours=hours, days=days)).isoformat()


def blocked_commit(client, prepared, key):
    """A blocked draft never posts: 409 TICKET_STALE (revalidation fails again) or DRAFT_BLOCKED."""
    response = commit(client, prepared, key)
    assert response.status_code == 409, response.text
    detail = response.json()['detail']
    assert detail['error_code'] in ('TICKET_STALE', 'DRAFT_BLOCKED'), detail
    return [b['code'] for b in detail['blockers']]


def late_exceptions(cur, prepared):
    cur.execute("SELECT * FROM exceptions WHERE kind='LATE_ENTRY' AND ticket_id=%s", (prepared['ticket_id'],))
    return cur.fetchall()


# ── 1. The matrix itself ──────────────────────────────────────────────────

def test_matrix_covers_exactly_the_design_actions():
    assert set(permissions.ACTIONS) == set(EXPECTED)
    assert tuple(permissions.ROLES) == tuple(ROLES)


@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('action', sorted(EXPECTED))
def test_every_allowed_and_denied_combination(action, role):
    expected = role in EXPECTED[action].split()
    identity = identity_for(role)
    assert permissions.allowed(action, role) is expected
    assert permissions.permissions_for(identity)[action] is expected
    if expected:
        assert permissions.require(action, identity) == role
    else:
        with pytest.raises(Exception) as exc:
            permissions.require(action, identity)
        detail = exc.value.detail
        assert exc.value.status_code == 403
        assert detail['error_code'] == 'ROLE_NOT_ALLOWED'
        assert detail['action'] == action and detail['role'] == role
        assert action.replace('_', ' ') in detail['message'] or permissions.label(action)[0] in detail['message']
        assert detail['message_es']


def test_unknown_role_or_key_kind_is_denied_everywhere():
    for identity in (ident('admin'), ident(None, 'session'), ident('owner', 'readonly'), {}):
        assert not any(permissions.permissions_for(identity).values())
        with pytest.raises(Exception):
            permissions.require('receive', identity)


def test_markdown_table_is_the_live_matrix():
    table = permissions.matrix_markdown()
    for action in EXPECTED:
        row = next(line for line in table.splitlines() if line.startswith(f'| `{action}` |'))
        marks = [cell.strip() for cell in row.strip('|').split('|')[1:]]
        assert marks == ['✓' if r in EXPECTED[action].split() else '✗' for r in ROLES], action


# ── 2. Enforced at PREPARE, keyed on the ticket action ────────────────────

@pytest.mark.parametrize('which', ROLES)
@pytest.mark.parametrize('action', TICKET_ACTIONS)
def test_prepare_enforces_the_matrix_per_key(client, db_cursor, items, actors, action, which):
    payload = seed_receive(db_cursor) if action == 'receive' else body(action, items)
    path = {'receive': '/receive/prepare', 'found': '/inventory/found/prepare'}.get(action, f'/{action}/prepare')
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    before = db_cursor.fetchone()['n']
    response = client.post(path, json=payload, headers=headers(key_for(which, actors)))
    if which in EXPECTED[action].split():
        assert response.status_code == 200, response.text
        # A5: make/pack prepares never pre-confirm lots; the matrix answered
        # 200, so re-prepare with matching evidence to reach can_commit.
        if action in ('make', 'pack') and response.json()['draft'].get('input_plan') and all(
                b['code'] == 'LOT_NOT_CONFIRMED' for b in response.json()['blockers']):
            confirmations = [{'lot_id': i['lot_id'], 'method': 'full_code', 'value': i['lot_code']}
                             for i in response.json()['draft']['input_plan']]
            response = client.post(path, json=payload | {'lot_confirmations': confirmations},
                                   headers=headers(key_for(which, actors)))
            assert response.status_code == 200, response.text
        assert response.json()['can_commit'], response.text
        assert response.json()['actor']['role'] == (which if which in ('owner', 'floor', 'office') else None)
    elif which == 'legacy_dashboard':
        # Route-level scope (A1 part 1) answers before the matrix: unchanged status quo.
        assert response.status_code == 403 and response.json()['detail'] == 'API key not authorized for this endpoint'
    else:
        error(response, 403, 'ROLE_NOT_ALLOWED')
        assert response.json()['detail']['action'] == action
        assert response.json()['detail']['role'] == which
        assert response.json()['error_detail']['code'] == 'ROLE_NOT_ALLOWED'
    if response.status_code != 200:
        db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
        assert db_cursor.fetchone()['n'] == before


# ── 3. Enforced AGAIN at COMMIT, with the role as it is now ───────────────

@pytest.mark.parametrize('action', ['make', 'adjust'])
def test_commit_denies_when_the_role_changed_after_prepare(client, db_cursor, items, actors, action):
    key = actors['floor']['key']
    prepared = prepare(client, action, body(action, items), key)
    assert prepared['can_commit']
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (actors['floor']['id'],))
    # The auth cache may still say 'floor'; commit reads the role from the row.
    response = commit(client, prepared, key)
    error(response, 403, 'ROLE_NOT_ALLOWED')
    assert response.json()['detail'] == response.json()['detail'] | {'action': action, 'role': 'office'}
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute("UPDATE actors SET role='floor' WHERE id=%s", (actors['floor']['id'],))
    assert commit(client, prepared, key).status_code == 200


def test_commit_checks_the_stored_action_not_the_route(client, db_cursor, items, actors):
    """An office receive ticket relabelled as a make in the DB is denied at commit."""
    key = actors['office']['key']
    payload = seed_receive(db_cursor)
    prepared = client.post('/receive/prepare', json=payload, headers=headers(key)).json()
    db_cursor.execute("UPDATE write_tickets SET action='make' WHERE id=%s", (prepared['ticket_id'],))
    response = commit(client, prepared, key)
    error(response, 403, 'ROLE_NOT_ALLOWED')
    assert response.json()['detail']['action'] == 'make'
    assert posted_count(db_cursor, prepared) == 0


# ── 4. Back-dating: 48 h / 14 d boundaries, prepare and commit ────────────

@pytest.mark.parametrize('hours,status', [(47, 'normal'), (48 - 1/60, 'normal'), (49, 'late'),
                                          (14*24 - 1, 'late'), (14*24 + 1, 'backfill')])
def test_timing_classification(hours, status):
    now = main.get_plant_now()
    info = permissions.timing(now - timedelta(hours=hours), now)
    assert info['status'] == status
    assert info['days_late'] == int(hours // 24)


def test_timing_future_and_grace():
    now = main.get_plant_now()
    assert permissions.timing(now + timedelta(minutes=4), now)['status'] == 'normal'
    assert permissions.timing(now + timedelta(minutes=6), now)['status'] == 'future'
    assert permissions.timing(now + timedelta(minutes=4), now)['hours_late'] == 0.0


@pytest.mark.parametrize('which', ['floor', 'office', 'legacy_ledger', 'legacy_dashboard'])
def test_47h_is_normal_for_everyone(client, db_cursor, items, actors, which):
    action = 'receive' if which in ('office', 'legacy_dashboard') else 'make'
    payload = (seed_receive(db_cursor) if action == 'receive' else body(action, items)) | {'occurred_at': at(hours=47)}
    prepared = prepare(client, action, payload, key_for(which, actors))
    assert prepared['draft']['entry_timing']['status'] == 'normal'
    assert [w['code'] for w in prepared['warnings']] == []
    posted = commit(client, prepared, key_for(which, actors))
    assert posted.status_code == 200, posted.text
    assert posted.json()['entry_timing']['status'] == 'normal'
    assert posted.json()['entry_timing']['late_entry_exception_id'] is None
    assert late_exceptions(db_cursor, prepared) == []


@pytest.mark.parametrize('which,hours', [('floor', 49), ('office', 49), ('legacy_ledger', 49),
                                         ('legacy_dashboard', 49), ('floor', 14*24 - 1)])
def test_49h_to_14d_posts_and_opens_late_entry_for_non_owners(client, db_cursor, items, actors, which, hours):
    action = 'receive' if which in ('office', 'legacy_dashboard') else 'make'
    payload = (seed_receive(db_cursor) if action == 'receive' else body(action, items)) | {'occurred_at': at(hours=hours)}
    key = key_for(which, actors)
    prepared = prepare(client, action, payload, key)
    assert prepared['can_commit'], prepared
    assert prepared['draft']['entry_timing']['status'] == 'late'
    warning = next(w for w in prepared['warnings'] if w['code'] == 'LATE_ENTRY')
    assert warning['requires_ack'] is False
    assert warning['refs']['days_late'] == hours // 24
    assert f'({hours // 24} days)' in warning['message'] and warning['message_es']
    posted = commit(client, prepared, key)   # no acknowledgement needed
    assert posted.status_code == 200, posted.text
    result = posted.json()
    rows = late_exceptions(db_cursor, prepared)
    assert len(rows) == 1 and posted_count(db_cursor, prepared) == 1
    row = rows[0]
    assert result['entry_timing'] == result['entry_timing'] | {
        'status': 'late', 'days_late': hours // 24, 'late_entry_exception_id': row['id']}
    assert result['entry_timing']['entered_by']['role'] == (which if which in ('floor', 'office') else None)
    assert (row['status'], row['severity'], row['kind']) == ('open', 'warn', 'LATE_ENTRY')
    assert row['transaction_id'] == result['transaction_id']
    assert row['receipt_number'] == result['receipt_number']
    assert row['owner_actor_id'] == actors['owner']['id']
    assert row['detail']['days_late'] == hours // 24 and row['detail']['action'] == action
    assert row['detail']['role'] == (which if which in ('floor', 'office') else which)
    assert row['detail']['entered_at'] and row['detail']['happened_at'] == prepared['draft']['happened_at']
    db_cursor.execute('SELECT entry_backfilled, created_at_source FROM transactions WHERE id=%s', (result['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'entry_backfilled': False, 'created_at_source': 'database'}
    # The committed ticket's receipt carries the same timing block.
    assert client.get('/receipts/' + result['receipt_number'], headers=headers(key)).json()['response']['entry_timing']['status'] == 'late'


def test_owner_late_entry_posts_without_an_exception(client, db_cursor, items, actors):
    key = actors['owner']['key']
    prepared = prepare(client, 'make', body('make', items) | {'occurred_at': at(hours=49)}, key)
    assert prepared['draft']['entry_timing']['status'] == 'late'
    assert [w['code'] for w in prepared['warnings']] == []
    posted = commit(client, prepared, key)
    assert posted.status_code == 200, posted.text
    assert posted.json()['entry_timing']['late_entry_exception_id'] is None
    assert late_exceptions(db_cursor, prepared) == []


@pytest.mark.parametrize('which', ['floor', 'office', 'legacy_dashboard'])
@pytest.mark.parametrize('backfill', [False, True])
def test_over_14d_is_owner_only_at_prepare(client, db_cursor, items, actors, which, backfill):
    action = 'receive' if which in ('office', 'legacy_dashboard') else 'adjust'
    payload = (seed_receive(db_cursor) if action == 'receive' else body(action, items))
    payload |= {'occurred_at': at(hours=14*24 + 1), 'backfill': backfill}
    path = {'receive': '/receive/prepare'}.get(action, f'/{action}/prepare')
    response = client.post(path, json=payload, headers=headers(key_for(which, actors)))
    error(response, 403, 'BACKFILL_OWNER_ONLY')
    detail = response.json()['detail']
    assert (detail['action'], detail['role'], detail['days_late']) == ('backdate_over_14d', which, 14)
    assert '14 days' in detail['message'] and detail['message_es']
    db_cursor.execute("SELECT count(*) AS n FROM write_tickets WHERE payload->>'occurred_at'=%s", (payload['occurred_at'],))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.parametrize('which', ['owner', 'legacy_ledger'])
def test_over_14d_owner_needs_backfill_flag_then_posts_without_exception(client, db_cursor, items, actors, which):
    key = key_for(which, actors)
    payload = body('adjust', items) | {'occurred_at': at(hours=14*24 + 1)}
    blocked = prepare(client, 'adjust', payload, key)
    assert blocked['can_commit'] is False
    assert [b['code'] for b in blocked['blockers']] == ['OCCURRED_AT_BACKFILL_REQUIRED']
    assert blocked['draft']['entry_timing']['status'] == 'backfill'
    assert blocked_commit(client, blocked, key) == ['OCCURRED_AT_BACKFILL_REQUIRED']
    prepared = prepare(client, 'adjust', payload | {'backfill': True}, key)
    assert prepared['can_commit'], prepared
    posted = commit(client, prepared, key)
    assert posted.status_code == 200, posted.text
    assert posted.json()['entry_timing']['status'] == 'backfill'
    assert late_exceptions(db_cursor, prepared) == []
    db_cursor.execute('SELECT entry_backfilled, created_at_source FROM transactions WHERE id=%s',
                      (posted.json()['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'entry_backfilled': True, 'created_at_source': 'api_backfill'}


def test_future_dated_entries_are_rejected(client, db_cursor, items, actors):
    key = actors['floor']['key']
    payload = body('make', items) | {'occurred_at': (main.get_plant_now() + timedelta(minutes=10)).isoformat()}
    prepared = prepare(client, 'make', payload, key)
    assert prepared['can_commit'] is False
    assert [b['code'] for b in prepared['blockers']] == ['OCCURRED_AT_IN_FUTURE']
    assert prepared['draft']['entry_timing']['status'] == 'future'
    assert blocked_commit(client, prepared, key) == ['OCCURRED_AT_IN_FUTURE']
    assert posted_count(db_cursor, prepared) == 0


def test_commit_reclassifies_against_the_commit_clock(client, db_cursor, items, actors, monkeypatch):
    """Prepared at 47 h, committed after the 48 h line → late entry opened at commit."""
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items) | {'occurred_at': at(hours=47)}, key)
    assert prepared['draft']['entry_timing']['status'] == 'normal' and prepared['warnings'] == []
    real_now = main.get_plant_now()
    monkeypatch.setattr(main, 'get_plant_now', lambda: real_now + timedelta(hours=2))
    posted = commit(client, prepared, key)
    assert posted.status_code == 200, posted.text
    assert posted.json()['entry_timing']['status'] == 'late'
    assert len(late_exceptions(db_cursor, prepared)) == 1


def test_commit_denies_over_14d_against_the_commit_clock(client, db_cursor, items, actors, monkeypatch):
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items) | {'occurred_at': at(days=13)}, key)
    assert prepared['can_commit'] and prepared['draft']['entry_timing']['status'] == 'late'
    real_now = main.get_plant_now()
    monkeypatch.setattr(main, 'get_plant_now', lambda: real_now + timedelta(days=2))
    error(commit(client, prepared, key), 403, 'BACKFILL_OWNER_ONLY')
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    assert posted_count(db_cursor, prepared) == 0


def test_late_entry_exception_has_no_owner_when_no_owner_actor_exists(client, db_cursor, items, actors):
    db_cursor.execute("UPDATE actors SET active=false WHERE id=%s", (actors['owner']['id'],))
    main._reset_actor_cache()
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items) | {'occurred_at': at(hours=49)}, key)
    assert commit(client, prepared, key).status_code == 200
    assert late_exceptions(db_cursor, prepared)[0]['owner_actor_id'] is None


# ── 5. entered_by on every write ─────────────────────────────────────────

@pytest.mark.parametrize('action', TICKET_ACTIONS)
@pytest.mark.parametrize('which', ['owner', 'floor', 'legacy_ledger'])
def test_ticket_writes_record_entered_by(client, db_cursor, items, actors, action, which):
    payload = seed_receive(db_cursor) if action == 'receive' else body(action, items)
    key = key_for(which, actors)
    prepared = prepare(client, action, payload, key)
    posted = commit(client, prepared, key)
    assert posted.status_code == 200, posted.text
    db_cursor.execute('SELECT operator_id, entered_by_actor_id, occurred_at, created_at FROM transactions WHERE ticket_id=%s',
                      (prepared['ticket_id'],))
    row = db_cursor.fetchone()
    if which == 'legacy_ledger':
        assert (row['operator_id'], row['entered_by_actor_id']) == ('legacy-shared-key', None)
    else:
        assert (row['operator_id'], row['entered_by_actor_id']) == (actors[which]['name'], actors[which]['id'])
    # happened_at (occurred_at, user-supplied) and entered_at (created_at, DB clock) are distinct columns.
    assert row['occurred_at'] == datetime.fromisoformat(prepared['draft']['happened_at'])
    assert row['created_at'] >= row['occurred_at'] - timedelta(minutes=5)
    timing = posted.json()['entry_timing']
    assert timing['entered_by']['id'] == (None if which == 'legacy_ledger' else actors[which]['id'])
    assert timing['entered_at'] and timing['happened_at']


@pytest.mark.parametrize('which', ['floor', 'legacy_ledger'])
def test_direct_routes_record_entered_by_and_keep_legacy_placeholder(client, db_cursor, items, actors, which):
    key = key_for(which, actors)
    legacy = {'product_name': items['ingredient']['name'], 'lot_code': items['ingredient']['lot_code'],
              'adjustment_lb': -1, 'reason': 'Count correction', 'reason_es': 'Corrección de conteo',
              'mode': 'commit'}
    response = client.post('/adjust', json=legacy, headers=headers(key))
    assert response.status_code == 200, response.text
    txn = response.json()['transaction_id']
    db_cursor.execute('SELECT operator_id, entered_by_actor_id FROM transactions WHERE id=%s', (txn,))
    expected = ('legacy-shared-key', None) if which == 'legacy_ledger' else (actors['floor']['name'], actors['floor']['id'])
    assert tuple(db_cursor.fetchone().values()) == expected
    void = client.post(f'/void/{txn}', json={'reason': 'A2 test'}, headers=headers(key))
    assert void.status_code == 200, void.text
    db_cursor.execute("SELECT operator_id, entered_by_actor_id FROM ledger_corrections WHERE target_id=%s AND event_type='void'", (txn,))
    assert tuple(db_cursor.fetchone().values()) == expected


LEGACY_BODIES = {
    'make': lambda items: {'product_name': items['batch']['name'], 'batches': 1, 'mode': 'commit'},
    'pack': lambda items: {'source_product': items['batch']['name'], 'target_product': items['finished']['name'],
                           'cases': 1, 'mode': 'commit'},
    'adjust': lambda items: {'product_name': items['ingredient']['name'], 'lot_code': items['ingredient']['lot_code'],
                             'adjustment_lb': -1, 'reason': 'Count correction', 'reason_es': 'Corrección de conteo',
                             'mode': 'commit'},
}


@pytest.mark.parametrize('action', ['make', 'pack', 'adjust'])
def test_direct_routes_deny_office_like_the_ticket_does(client, db_cursor, items, actors, action):
    """Reviewer case: an office key could post `POST /make` while `/make/prepare` denied it."""
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    response = client.post(f'/{action}', json=LEGACY_BODIES[action](items), headers=headers(actors['office']['key']))
    error(response, 403, 'ROLE_NOT_ALLOWED')
    detail = response.json()['detail']
    assert (detail['action'], detail['role'], detail['actor'], detail['key_kind']) == (action, 'office', actors['office']['name'], 'actor')
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before
    # The same key on the same route's ticket twin gets the same answer.
    error(client.post(f'/{action}/prepare', json=body(action, items), headers=headers(actors['office']['key'])),
          403, 'ROLE_NOT_ALLOWED')


def test_direct_routes_gate_before_the_body_is_read(client, actors):
    """The 403 comes from the auth dependency: no body validation, no handler, no 422."""
    error(client.post('/make', json={}, headers=headers(actors['office']['key'])), 403, 'ROLE_NOT_ALLOWED')
    error(client.post('/sales/orders', json={}, headers=headers(actors['floor']['key'])), 403, 'ROLE_NOT_ALLOWED')
    assert client.post('/sales/orders', json={}, headers=headers(actors['office']['key'])).status_code == 422


@pytest.mark.parametrize('which,method,path,action', [
    ('floor', 'POST', '/sales/orders', 'create_order'),
    ('floor', 'POST', '/customers', 'manage_customers'),
    ('floor', 'PATCH', '/customers/1', 'manage_customers'),
    ('floor', 'POST', '/ship', 'ship_standalone'),
    ('floor', 'POST', '/sales/orders/1/lines', 'add_order_lines'),
    ('floor', 'PATCH', '/sales/orders/1', 'update_order_header'),
    ('floor', 'PATCH', '/sales/orders/1/status', 'update_order_status'),
    ('floor', 'POST', '/sales/orders/1/cancel', 'cancel_order'),
    ('floor', 'POST', '/expected-receipts', 'create_expected_receipt'),
    ('office', 'POST', '/void/1', 'void'),
    ('office', 'POST', '/sales/orders/1/reopen', 'reopen_order'),
    ('floor', 'POST', '/sales/orders/1/close', 'cancel_order'),
])
def test_direct_routes_apply_the_matrix_per_role(client, actors, which, method, path, action):
    response = client.request(method, path, json={}, headers=headers(actors[which]['key']))
    error(response, 403, 'ROLE_NOT_ALLOWED')
    assert (response.json()['detail']['action'], response.json()['detail']['role']) == (action, which)


def test_close_shipped_not_recorded_is_owner_only_on_the_direct_route(client, actors):
    """§4.3 R6: office may close an order, but not as 'shipped, not recorded'."""
    body_ = {'reason': 'shipped_not_recorded', 'mode': 'commit'}
    response = client.post('/sales/orders/1/close', json=body_, headers=headers(actors['office']['key']))
    error(response, 403, 'ROLE_NOT_ALLOWED')
    assert response.json()['detail']['action'] == 'close_order_shipped_not_recorded'
    assert client.post('/sales/orders/1/close', json=body_ | {'reason': 'short_closed'},
                       headers=headers(actors['office']['key'])).status_code != 403
    assert client.post('/sales/orders/1/close', json=body_, headers=headers(actors['owner']['key'])).status_code != 403
    assert client.post('/sales/orders/1/close', json=body_, headers=headers(main.API_KEY)).status_code != 403


def test_direct_routes_still_open_where_the_matrix_allows(client, db_cursor, items, actors):
    for which in ('owner', 'floor'):
        response = client.post('/make', json=LEGACY_BODIES['make'](items), headers=headers(actors[which]['key']))
        assert response.status_code == 200, response.text
    office = client.post('/receive', json={'product_name': items['ingredient']['name'], 'cases': 1, 'case_size_lb': 5,
                                           'shipper_name': 'ACME', 'bol_reference': 'A2', 'mode': 'preview'},
                         headers=headers(actors['office']['key']))
    assert office.status_code == 200, office.text
    # Floor's one status change stays reachable (route exists; 404 is the handler's answer, not a 403).
    assert client.post('/sales-orders/SO-000000-000/ready', headers=headers(actors['floor']['key'])).status_code != 403


def test_shared_keys_keep_their_legacy_reach_on_direct_routes(client, db_cursor, items, actors):
    """Only the master key keeps today's reach until A10; the gate is for named actors."""
    assert permissions.require_route(('POST', '/make'), {'key_kind': 'legacy_ledger'}) is None
    assert permissions.require_route(('POST', '/sales/orders/{order_id}/cancel'), {'key_kind': 'legacy_dashboard'}) is None
    response = client.post('/make', json=LEGACY_BODIES['make'](items), headers=headers(main.API_KEY))
    assert response.status_code == 200, response.text
    # Dashboard key: the route allowlist answers, the matrix never runs.
    assert client.post('/sales/orders/1/cancel', json={}, headers=headers(main.DASHBOARD_API_KEY)).status_code not in (403,)


def test_every_actor_reachable_write_route_is_gated_or_named_exempt():
    reachable = {r for r in (main.ACTOR_WRITE_ALLOWLIST | main.DASHBOARD_KEY_ALLOWLIST) if r[0] != 'GET'}
    tickets = write_tickets.ACTOR_ROUTES | write_tickets.DASHBOARD_ROUTES
    unaccounted = reachable - set(permissions.ROUTE_ACTIONS) - permissions.UNGATED_ROUTES - tickets
    assert unaccounted == set(), sorted(unaccounted)
    registered = {(m, r.path) for r in main.app.routes for m in getattr(r, 'methods', ()) or ()}
    assert set(permissions.ROUTE_ACTIONS) <= registered, sorted(set(permissions.ROUTE_ACTIONS) - registered)
    assert set(permissions.ROUTE_ACTIONS.values()) <= set(permissions.ACTIONS)
    assert not set(permissions.ROUTE_ACTIONS) & permissions.UNGATED_ROUTES


@pytest.mark.parametrize('backfill', [False, True])
def test_direct_routes_deny_floor_over_14d_whatever_backfill_says(client, db_cursor, items, actors, backfill):
    """Reviewer case: `POST /adjust {backfill:true}` let floor write 15-day-old entries."""
    legacy = LEGACY_BODIES['adjust'](items) | {'occurred_at': at(hours=14*24 + 1), 'backfill': backfill}
    response = client.post('/adjust', json=legacy, headers=headers(actors['floor']['key']))
    error(response, 403, 'BACKFILL_OWNER_ONLY')
    assert (response.json()['detail']['role'], response.json()['detail']['days_late']) == ('floor', 14)
    make = LEGACY_BODIES['make'](items) | {'occurred_at': at(hours=14*24 + 1), 'backfill': backfill}
    error(client.post('/make', json=make, headers=headers(actors['floor']['key'])), 403, 'BACKFILL_OWNER_ONLY')


@pytest.mark.parametrize('which', ['owner', 'legacy_ledger'])
def test_direct_routes_owner_and_master_over_14d_keep_the_backfill_path(client, db_cursor, items, actors, which):
    legacy = LEGACY_BODIES['adjust'](items) | {'occurred_at': at(hours=14*24 + 1)}
    error(client.post('/adjust', json=legacy, headers=headers(key_for(which, actors))), 400, 'OCCURRED_AT_BACKFILL_REQUIRED')
    posted = client.post('/adjust', json=legacy | {'backfill': True}, headers=headers(key_for(which, actors)))
    assert posted.status_code == 200, posted.text
    db_cursor.execute('SELECT entry_backfilled FROM transactions WHERE id=%s', (posted.json()['transaction_id'],))
    assert db_cursor.fetchone()['entry_backfilled'] is True


def test_direct_routes_floor_late_entry_still_posts(client, db_cursor, items, actors):
    legacy = LEGACY_BODIES['adjust'](items) | {'occurred_at': at(hours=49)}
    posted = client.post('/adjust', json=legacy, headers=headers(actors['floor']['key']))
    assert posted.status_code == 200, posted.text


def test_whoami_permissions_are_the_matrix(client, actors):
    for which in ROLES:
        body_ = client.get('/auth/whoami', headers=headers(key_for(which, actors))).json()
        assert body_['permissions'] == {a: which in EXPECTED[a].split() for a in EXPECTED}, which


# ── 6. Commit holds the actor row lock through posting ───────────────────

def test_commit_locks_the_actor_row_so_a_role_change_waits(isolated_database, monkeypatch):
    """Reviewer case: a role change committed between the commit's role check
    and the post must not slip through. The commit reads role+active FOR SHARE
    in the posting transaction; a concurrent UPDATE actors blocks until it ends."""
    token = uuid4().hex[:8].upper()
    key = f'actor-key-floor-{token}'
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        items_ = seed_items(cur)
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'floor',%s,true) RETURNING id",
                    (f'ACT floor {token}', sha256(key.encode()).hexdigest()))
        actor_id = cur.fetchone()['id']
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'owner',%s,true)",
                    (f'ACT owner {token}', sha256(f'actor-key-owner-{token}'.encode()).hexdigest()))

    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    main._reset_actor_cache()

    locked = Barrier(2)       # commit thread has passed the role check and is about to post
    release = Event()         # main thread has tried the concurrent UPDATE
    pause = False
    original = permissions.require

    def require(action, identity):
        # The window the reviewer named: role checked, post not yet made.
        result = original(action, identity)
        if pause:
            locked.wait(timeout=10)
            assert release.wait(timeout=10)
        return result
    monkeypatch.setattr(permissions, 'require', require)

    with TestClient(main.app) as http:
        prepared = prepare(http, 'adjust', body('adjust', items_), key)
        assert prepared['can_commit'], prepared
        pause = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(commit, http, prepared, key)
            locked.wait(timeout=10)
            # The role change cannot be committed while the post is in flight.
            with psycopg2.connect(isolated_database) as other, other.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='500ms'")
                with pytest.raises(psycopg2.errors.LockNotAvailable):
                    cur.execute("UPDATE actors SET role='office' WHERE id=%s", (actor_id,))
                other.rollback()
            release.set()
            posted = future.result(timeout=20)
        pause = False
        assert posted.status_code == 200, posted.text
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, prepared) == 1
            cur.execute('SELECT entered_by_actor_id FROM transactions WHERE ticket_id=%s', (prepared['ticket_id'],))
            assert cur.fetchone()['entered_by_actor_id'] == actor_id
            # Once the post is done the same change goes straight through …
            cur.execute("UPDATE actors SET role='office' WHERE id=%s", (actor_id,))
        # … and the next commit by that key sees it (ticket bound to the person, role read fresh).
        main._reset_actor_cache()
        error(http.post('/adjust/prepare', json=body('adjust', items_), headers=headers(key)), 403, 'ROLE_NOT_ALLOWED')
    main._reset_actor_cache()


def test_commit_denies_an_actor_deactivated_after_prepare(client, db_cursor, items, actors):
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items), key)
    db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (actors['floor']['id'],))
    response = commit(client, prepared, key)   # the auth cache may still admit the key
    error(response, 409, 'TICKET_STALE')
    assert [b['code'] for b in response.json()['detail']['blockers']] == ['ACTOR_INACTIVE']
    assert posted_count(db_cursor, prepared) == 0


# ── 7. DST: every comparison on UTC instants ──────────────────────────────
# America/New_York 2026: spring forward Mar 8 02:00 → 03:00; fall back Nov 1 02:00 → 01:00.

SPRING_NOW = datetime(2026, 3, 9, 12, 0, tzinfo=main.PLANT_TIMEZONE)                 # EDT, after the change
FALL_NOW = datetime(2026, 11, 1, 1, 45, fold=0, tzinfo=main.PLANT_TIMEZONE)          # 01:45 EDT, 15 min before


def spring_335h30_naive():
    """335.5 real hours before SPRING_NOW, written as a plant wall-clock time with no offset."""
    return (SPRING_NOW.astimezone(timezone.utc) - timedelta(hours=335.5)).astimezone(main.PLANT_TIMEZONE)


def fall_30min_future():
    """30 real minutes after FALL_NOW: 06:15Z = 01:15 EST — a smaller wall-clock reading than 01:45 EDT."""
    return (FALL_NOW.astimezone(timezone.utc) + timedelta(minutes=30)).astimezone(main.PLANT_TIMEZONE)


def test_timing_across_spring_forward_uses_real_elapsed_hours():
    happened = spring_335h30_naive()
    assert happened.tzinfo is SPRING_NOW.tzinfo                     # the shared-ZoneInfo case
    assert (SPRING_NOW - happened) == timedelta(hours=336.5)        # Python's wall-clock subtraction
    info = permissions.timing(happened, SPRING_NOW)
    assert (info['status'], info['hours_late'], info['days_late']) == ('late', 335.5, 13)
    naive = happened.replace(tzinfo=None)
    assert permissions.timing(naive, SPRING_NOW)['status'] == 'late'


def test_timing_across_fall_back_sees_the_future():
    happened = fall_30min_future()
    assert happened.tzinfo is FALL_NOW.tzinfo and happened < FALL_NOW   # wall clock says "30 min ago"
    assert permissions.timing(happened, FALL_NOW)['status'] == 'future'
    assert permissions.timing(happened.astimezone(timezone.utc), FALL_NOW)['status'] == 'future'


def test_validate_occurred_at_across_dst(monkeypatch):
    monkeypatch.setattr(main, 'get_plant_now', lambda: FALL_NOW)
    with pytest.raises(Exception) as exc:
        main.validate_inventory_occurred_at(fall_30min_future())
    assert exc.value.detail['error_code'] == 'OCCURRED_AT_IN_FUTURE'
    monkeypatch.setattr(main, 'get_plant_now', lambda: SPRING_NOW)
    event, source = main.validate_inventory_occurred_at(spring_335h30_naive().replace(tzinfo=None))
    assert source is None and event == spring_335h30_naive()
    with pytest.raises(Exception) as exc:
        main.validate_inventory_occurred_at(spring_335h30_naive() - timedelta(hours=1))
    assert exc.value.detail['error_code'] == 'OCCURRED_AT_BACKFILL_REQUIRED'


def test_spring_forward_335h_is_a_late_entry_at_prepare_and_commit(client, db_cursor, items, actors, monkeypatch):
    """Wall-clock math said 336.5 h (> 14 d) and denied floor; real elapsed time is 335.5 h."""
    monkeypatch.setattr(main, 'get_plant_now', lambda: SPRING_NOW)
    key = actors['floor']['key']
    payload = body('make', items) | {'occurred_at': spring_335h30_naive().replace(tzinfo=None).isoformat()}
    prepared = prepare(client, 'make', payload, key)
    assert prepared['can_commit'], prepared
    assert prepared['draft']['entry_timing'] == prepared['draft']['entry_timing'] | {
        'status': 'late', 'hours_late': 335.5, 'days_late': 13}
    assert prepared['draft']['happened_vs_now_minutes'] == 335.5 * 60
    assert [w['code'] for w in prepared['warnings']] == ['LATE_ENTRY']
    posted = commit(client, prepared, key)
    assert posted.status_code == 200, posted.text
    assert posted.json()['entry_timing']['status'] == 'late'
    assert posted.json()['entry_timing']['hours_late'] == 335.5
    assert len(late_exceptions(db_cursor, prepared)) == 1


def test_spring_forward_335h_on_the_direct_route(client, db_cursor, items, actors, monkeypatch):
    monkeypatch.setattr(main, 'get_plant_now', lambda: SPRING_NOW)
    legacy = LEGACY_BODIES['adjust'](items) | {'occurred_at': spring_335h30_naive().replace(tzinfo=None).isoformat()}
    posted = client.post('/adjust', json=legacy, headers=headers(actors['floor']['key']))
    assert posted.status_code == 200, posted.text


def test_fall_back_30min_future_is_rejected_at_prepare_and_commit(client, db_cursor, items, actors, monkeypatch):
    """Wall-clock math read 01:15 EST as 30 min before 01:45 EDT and accepted a future entry."""
    monkeypatch.setattr(main, 'get_plant_now', lambda: FALL_NOW)
    key = actors['floor']['key']
    payload = body('make', items) | {'occurred_at': fall_30min_future().isoformat()}
    prepared = prepare(client, 'make', payload, key)
    assert prepared['can_commit'] is False
    assert [b['code'] for b in prepared['blockers']] == ['OCCURRED_AT_IN_FUTURE']
    assert prepared['draft']['entry_timing']['status'] == 'future'
    assert blocked_commit(client, prepared, key) == ['OCCURRED_AT_IN_FUTURE']
    assert posted_count(db_cursor, prepared) == 0


def test_fall_back_30min_future_on_the_direct_route(client, db_cursor, items, actors, monkeypatch):
    monkeypatch.setattr(main, 'get_plant_now', lambda: FALL_NOW)
    legacy = LEGACY_BODIES['adjust'](items) | {'occurred_at': fall_30min_future().isoformat()}
    error(client.post('/adjust', json=legacy, headers=headers(actors['floor']['key'])), 400, 'OCCURRED_AT_IN_FUTURE')


def test_prepare_and_commit_agree_across_a_dst_change(client, db_cursor, items, actors, monkeypatch):
    """Prepared after fall-back for an entry 48.5 real hours earlier (47.5 h by wall clock):
    prepare must already say late, and commit an hour later must say the same thing."""
    now = datetime(2026, 11, 1, 8, 0, tzinfo=main.PLANT_TIMEZONE)                   # 08:00 EST
    happened = (now.astimezone(timezone.utc) - timedelta(hours=48.5)).astimezone(main.PLANT_TIMEZONE)
    assert now - happened == timedelta(hours=47.5)                                   # the wall-clock trap
    monkeypatch.setattr(main, 'get_plant_now', lambda: now)
    key = actors['floor']['key']
    prepared = prepare(client, 'make', body('make', items) | {'occurred_at': happened.replace(tzinfo=None).isoformat()}, key)
    assert prepared['draft']['entry_timing']['hours_late'] == 48.5
    assert [w['code'] for w in prepared['warnings']] == ['LATE_ENTRY']
    monkeypatch.setattr(main, 'get_plant_now', lambda: now + timedelta(hours=1))
    posted = commit(client, prepared, key)
    assert posted.status_code == 200, posted.text
    assert posted.json()['entry_timing']['hours_late'] == 49.5
    assert len(late_exceptions(db_cursor, prepared)) == 1


# ── 8. Migration 065 ─────────────────────────────────────────────────────

def test_migration_065_up_down_up(isolated_database):
    up = (ROOT / 'migrations/065_entered_by.sql').read_text()
    down = (ROOT / 'migrations/down/065_entered_by_down.sql').read_text()

    def columns(cur):
        cur.execute("""SELECT table_name FROM information_schema.columns
                       WHERE column_name='entered_by_actor_id' ORDER BY 1""")
        return [r[0] for r in cur.fetchall()]

    # A3b's 069 view passes entered_by_actor_id through; it rolls back first and
    # goes back on last (rollback order documented in the 069 down file).
    up_069 = (ROOT / 'migrations/069_exceptions_enforcement.sql').read_text()
    down_069 = (ROOT / 'migrations/down/069_exceptions_enforcement_down.sql').read_text()
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL search_path TO public')
        cur.execute(down_069)
        assert columns(cur) == ['ledger_corrections', 'transactions']
        cur.execute(up)   # rerun is a no-op
        cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgname='trg_transactions_original_append_only'")
        assert cur.fetchone()[0] != 'D'
        cur.execute(down)
        assert columns(cur) == []
        cur.execute("SELECT count(*) FROM migration_markers WHERE name='065_entered_by'")
        assert cur.fetchone()[0] == 0
        cur.execute(up)
        assert columns(cur) == ['ledger_corrections', 'transactions']
        cur.execute("SELECT count(*) FROM migration_markers WHERE name='065_entered_by'")
        assert cur.fetchone()[0] == 1
        cur.execute("""SELECT confrelid::regclass::text FROM pg_constraint
                       WHERE conrelid='transactions'::regclass AND contype='f'
                         AND conkey=ARRAY[(SELECT attnum FROM pg_attribute WHERE attrelid='transactions'::regclass
                                                                             AND attname='entered_by_actor_id')]""")
        assert cur.fetchone()[0] == 'actors'
        cur.execute(up_069)
        assert 'ledger_current_transactions' in columns(cur)

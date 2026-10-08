"""A2: roles enforced in FL, entered_by on every write, back-dating limits (design rev 3.7 §4.3, §6)."""
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
import pytest

import main
import permissions
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import (headers, commit, ticket_row, posted_count, error,
                                      isolated_database, seed as seed_receive)  # noqa: F401
from tests.test_write_tickets_part2 import items, body, prepare  # noqa: F401

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
    'move_lot':                         'owner floor office legacy_ledger',
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


@pytest.mark.parametrize('action', ['make', 'adjust'])
def test_direct_legacy_routes_are_unchanged_for_office(client, db_cursor, items, actors, action):
    """A2 enforces on tickets only; the direct routes close in A10."""
    legacy = {'make': {'product_name': items['batch']['name'], 'batches': 1, 'mode': 'commit'},
              'adjust': {'product_name': items['ingredient']['name'], 'lot_code': items['ingredient']['lot_code'],
                         'adjustment_lb': -1, 'reason': 'Count correction', 'reason_es': 'Corrección de conteo',
                         'mode': 'commit'}}[action]
    response = client.post(f'/{action}', json=legacy, headers=headers(actors['office']['key']))
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT entered_by_actor_id FROM transactions WHERE id=%s', (response.json()['transaction_id'],))
    assert db_cursor.fetchone()['entered_by_actor_id'] == actors['office']['id']


def test_whoami_permissions_are_the_matrix(client, actors):
    for which in ROLES:
        body_ = client.get('/auth/whoami', headers=headers(key_for(which, actors))).json()
        assert body_['permissions'] == {a: which in EXPECTED[a].split() for a in EXPECTED}, which


# ── 6. Migration 065 ─────────────────────────────────────────────────────

def test_migration_065_up_down_up(isolated_database):
    up = (ROOT / 'migrations/065_entered_by.sql').read_text()
    down = (ROOT / 'migrations/down/065_entered_by_down.sql').read_text()

    def columns(cur):
        cur.execute("""SELECT table_name FROM information_schema.columns
                       WHERE column_name='entered_by_actor_id' ORDER BY 1""")
        return [r[0] for r in cur.fetchall()]

    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL search_path TO public')
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

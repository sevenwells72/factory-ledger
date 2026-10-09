"""A3b: exceptions enforcement (design rev 3.7 §5 R2/R3, §5.1, §7.1, §11 item 7; Michael's
A3b decisions 2026-10-08) — reason codes, large-correction holds + owner approval,
shortage post-and-flag, business-day deadlines, /exceptions routes, nightly sweep."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
import threading
from threading import Barrier
import time
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg2
from psycopg2.extras import RealDictCursor
import pytest

import main
import exceptions_enforcement as a3b
import lot_confirmation
from scripts.exceptions_sweep import sweep
from scripts.expire_tickets import expire
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import (headers, commit, ticket_row, posted_count, error,
                                      isolated_database)  # noqa: F401
from tests.test_write_tickets_part2 import items, body, prepare, seed as seed_items  # noqa: F401

from tests.pin_test_support import headers, commit  # A11: real per-request owner proof

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.db
NY = a3b.PLANT_TIMEZONE


# ── helpers ───────────────────────────────────────────────────────────────

def set_balance(cur, items_, which, balance):
    """Append a ledger line so the fixture lot (100 lb) reads `balance`."""
    lot = items_[which]
    cur.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','A3b fixture') RETURNING id")
    txn = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                (txn, lot['id'], lot['lot_id'], balance - main.lot_on_hand(cur, lot['lot_id'])))


def adjust(items_, delta, reason='physical_count', **extra):
    return {'occurred_at': main.get_plant_now().isoformat(), 'lot_id': items_['ingredient']['lot_id'],
            'delta_lb': delta, 'reason_code': reason, **extra}


def found(items_, qty, reason='physical_count', **extra):
    return {'occurred_at': main.get_plant_now().isoformat(), 'product_id': items_['ingredient']['id'],
            'quantity': qty, 'reason_code': reason, **extra}


def exceptions_for(cur, **where):
    clause = ' AND '.join(f'{k}=%s' for k in where)
    cur.execute(f'SELECT * FROM exceptions WHERE {clause} ORDER BY id', tuple(where.values()))
    return cur.fetchall()


def flags_for(cur, transaction_id):
    cur.execute('SELECT * FROM shortage_flags WHERE transaction_id=%s ORDER BY id', (transaction_id,))
    return cur.fetchall()


def post_short_make(client_, db_cursor, items_, key, on_hand=4.0):
    """Ingredient lot at `on_hand`, make needs 10 → posts short (10 − on_hand)."""
    set_balance(db_cursor, items_, 'ingredient', on_hand)
    prepared = prepare(client_, 'make', body('make', items_), key)
    assert prepared['can_commit'], prepared
    response = commit(client_, prepared, key)
    assert response.status_code == 200, response.text
    return prepared, response.json()


# ── A. reason codes (R2, §5.1) ────────────────────────────────────────────

@pytest.mark.parametrize('action', ['adjust', 'found'])
def test_reason_code_must_be_on_the_fixed_list(client, db_cursor, items, action):
    make = adjust if action == 'adjust' else found
    path = '/adjust/prepare' if action == 'adjust' else '/inventory/found/prepare'
    before = ticket_count(db_cursor)
    response = client.post(path, json=make(items, -2 if action == 'adjust' else 2, reason='swept the floor'), headers=headers())
    error(response, 422, 'REASON_CODE_INVALID')
    valid = [r['code'] for r in response.json()['detail']['valid_reasons']]
    assert 'physical_count' in valid and 'unknown' in valid
    assert ('damage_disposal' in valid) == (action == 'adjust')     # applies_to is enforced
    # A code that exists but does not apply to this action is just as invalid.
    other = 'wrong_lot' if action == 'found' else 'physical_count'
    if action == 'found':
        error(client.post(path, json=make(items, 2, reason=other), headers=headers()), 422, 'REASON_CODE_INVALID')
    assert ticket_count(db_cursor) == before      # refused, never drafted
    assert client.post(path, json=make(items, -2 if action == 'adjust' else 2, reason=''), headers=headers()).status_code == 422


def ticket_count(cur):
    cur.execute('SELECT count(*) AS n FROM write_tickets')
    return cur.fetchone()['n']


def test_unknown_needs_a_note_and_the_note_is_kept(client, db_cursor, items):
    error(client.post('/adjust/prepare', json=adjust(items, -3, 'unknown'), headers=headers()), 422, 'NOTE_REQUIRED')
    error(client.post('/adjust/prepare', json=adjust(items, -3, 'unknown', note='   '), headers=headers()), 422, 'NOTE_REQUIRED')
    error(client.post('/inventory/found/prepare', json=found(items, 3, 'unknown'), headers=headers()), 422, 'NOTE_REQUIRED')
    prepared = prepare(client, 'adjust', adjust(items, -3, 'Unknown', note='bag split on the floor'))
    assert prepared['draft']['reason_code'] == 'unknown' and prepared['draft']['note'] == 'bag split on the floor'
    assert prepared['draft']['correction_review']['highlighted'] is True          # §5.1: unknown is always highlighted
    assert prepared['draft']['correction_review']['rules'] == ['REASON_UNKNOWN']
    response = commit(client, prepared)
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT reason_code, adjust_reason, adjust_reason_es, notes FROM transactions WHERE id=%s',
                      (response.json()['transaction_id'],))
    row = db_cursor.fetchone()
    assert (row['reason_code'], row['adjust_reason'], row['adjust_reason_es']) == ('unknown', 'Unknown', 'Desconocido')
    assert row['notes'].endswith('— bag split on the floor')
    # found: `notes` is the note
    prepared = prepare(client, 'found', found(items, 3, 'unknown', notes='pallet with no paperwork'))
    assert commit(client, prepared).status_code == 200


def test_legacy_codes_translate_with_note_prefill(client, db_cursor, items):
    prepared = prepare(client, 'adjust', adjust(items, -3, 'hydration_yield'))
    assert prepared['draft']['reason_code'] == 'unrecorded_usage'
    assert prepared['draft']['reason']['legacy_code'] == 'hydration_yield'
    assert prepared['draft']['note'] == 'hydration/yield'
    prepared = prepare(client, 'found', found(items, 3, 'found_back_stock'))
    assert prepared['draft']['reason_code'] == 'missing_receipt'
    response = commit(client, prepared)
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT reason_code, notes FROM transactions WHERE id=%s', (response.json()['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'reason_code': 'missing_receipt', 'notes': 'Found inventory: missing_receipt'}
    db_cursor.execute('SELECT reason_code FROM inventory_adjustments WHERE lot_id=%s', (response.json()['lot_id'],))
    assert db_cursor.fetchone()['reason_code'] == 'missing_receipt'


@pytest.mark.parametrize('delta,reason', [(5, 'damage_disposal'), (5, 'unrecorded_usage'), (-5, 'missing_receipt')])
def test_adjust_sign_is_enforced(client, items, delta, reason):
    error(client.post('/adjust/prepare', json=adjust(items, delta, reason), headers=headers()), 422, 'REASON_SIGN_MISMATCH')
    assert prepare(client, 'adjust', adjust(items, -delta, reason))['draft']['reason_code'] == reason


def test_reason_code_written_at_insert_and_exposed_by_the_view(client, db_cursor, items, actors):
    key = actors['floor']['key']
    ticket_txn = commit(client, prepare(client, 'adjust', adjust(items, -3, 'damage_disposal'), key), key).json()['transaction_id']
    found_txn = commit(client, prepare(client, 'found', found(items, 3, 'physical_count'), key), key).json()['transaction_id']
    db_cursor.execute('SELECT id, reason_code, entered_by_actor_id, effective_status FROM ledger_current_transactions WHERE id IN (%s,%s) ORDER BY id',
                      (ticket_txn, found_txn))
    rows = db_cursor.fetchall()
    assert [(r['reason_code'], r['entered_by_actor_id'], r['effective_status']) for r in rows] == [
        ('damage_disposal', actors['floor']['id'], 'posted'), ('physical_count', actors['floor']['id'], 'posted')]
    # Direct (legacy) routes are unchanged for callers but no longer leave the column NULL (P1.8).
    legacy = {'product_name': items['ingredient']['name'], 'lot_code': items['ingredient']['lot_code'],
              'adjustment_lb': -1, 'reason_es': 'x', 'mode': 'commit'}
    for text, expected in [('spoilage', 'damage_disposal'), ('Physical count 2026-10-08 Arturo', 'physical_count'),
                           ('Damage Disposal', 'damage_disposal'), ('swept the floor', 'unknown')]:
        response = client.post('/adjust', json=legacy | {'reason': text}, headers=headers())
        assert response.status_code == 200, response.text
        assert response.json()['reason_code'] == expected
        db_cursor.execute('SELECT reason_code, adjust_reason FROM transactions WHERE id=%s', (response.json()['transaction_id'],))
        assert dict(db_cursor.fetchone()) == {'reason_code': expected, 'adjust_reason': text}
    response = client.post('/inventory/found', json={'product_id': items['ingredient']['id'], 'quantity': 2,
                                                     'reason_code': 'predates_system'}, headers=headers())
    assert response.status_code == 200, response.text
    db_cursor.execute("SELECT reason_code FROM transactions WHERE type='adjust' AND notes='Found inventory: predates_system' ORDER BY id DESC LIMIT 1")
    assert db_cursor.fetchone()['reason_code'] == 'missing_receipt'
    db_cursor.execute("SELECT count(*) AS n FROM transactions WHERE type='adjust' AND reason_code IS NULL AND notes IN ('Adjustment: -1 lb','Found inventory: predates_system')")
    assert db_cursor.fetchone()['n'] == 0


# ── A. thresholds: highlight, book balance, photo ──────────────────────────

@pytest.mark.parametrize('balance,delta,rules', [
    (100, -10, []),                                    # exactly 10 % is not "more than"
    (100, -10.5, ['OVER_10_PERCENT_OF_BOOK']),
    (100, 10.5, ['OVER_10_PERCENT_OF_BOOK']),
    (10000, -501, ['OVER_500_LB']),                    # 5 % of book but over 500 lb
    (10000, -500, []),
    (100, 600, ['OVER_10_PERCENT_OF_BOOK', 'OVER_500_LB']),
    (0, -1, ['BOOK_BALANCE_NOT_POSITIVE']),            # zero book balance → always highlighted
    (-5, 1, ['BOOK_BALANCE_NOT_POSITIVE']),            # negative too
])
def test_adjust_highlight_rules_on_book_balance(client, db_cursor, items, balance, delta, rules):
    set_balance(db_cursor, items, 'ingredient', balance)
    reason = 'physical_count'
    prepared = prepare(client, 'adjust', adjust(items, delta, reason))
    review = prepared['draft']['correction_review']
    assert review['rules'] == rules
    assert review['highlighted'] is bool(rules)
    assert review['book_balance_before'] == float(balance)
    assert review['photo_required'] is (abs(delta) > 500)
    assert review['weekly_view'] is True
    assert review['lb_equivalent'] == float(delta)


@pytest.mark.parametrize('qty,highlighted', [(400, False), (500, False), (501, True)])
def test_found_uses_the_500_lb_rule_only_and_is_always_on_the_weekly_view(client, db_cursor, items, qty, highlighted):
    set_balance(db_cursor, items, 'ingredient', 0)   # no book balance rule for found stock
    review = prepare(client, 'found', found(items, qty))['draft']['correction_review']
    assert review['highlighted'] is highlighted and review['photo_required'] is highlighted
    assert review['rules'] == (['OVER_500_LB'] if highlighted else [])
    assert review['weekly_view'] is True and review['pct_of_book'] is None


def test_highlight_is_recomputed_under_lock_at_commit(client, db_cursor, items):
    prepared = prepare(client, 'adjust', adjust(items, -20))          # 20 % of 100 → highlighted at prepare
    assert prepared['draft']['correction_review']['highlighted'] is True
    set_balance(db_cursor, items, 'ingredient', 1000)                   # stock arrived: 2 % of 1000
    response = commit(client, prepared)
    assert response.status_code == 200, response.text
    review = response.json()['correction_review']
    assert review['highlighted'] is False and review['book_balance_before'] == 1000.0
    assert client.get('/receipts/' + response.json()['receipt_number'], headers=headers()).json()['response']['correction_review'] == review


def test_count_based_product_applies_the_percent_rule_only(client, db_cursor, items):
    db_cursor.execute("UPDATE products SET uom='each' WHERE id=%s", (items['ingredient']['id'],))
    review = prepare(client, 'adjust', adjust(items, -600))['draft']['correction_review']
    assert review['lb_rule'] == 'not_applicable' and review['lb_equivalent'] is None
    assert review['rules'] == ['OVER_10_PERCENT_OF_BOOK'] and review['photo_required'] is False


def test_large_correction_with_photo_posts_highlighted(client, db_cursor, items, actors):
    key = actors['floor']['key']
    prepared = prepare(client, 'adjust', adjust(items, 600, 'missing_receipt', attachment_ref='photos/2026-10-08/bag-count.jpg'), key)
    assert prepared['can_commit'] and prepared['draft']['correction_review']['photo_required']
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    review = response.json()['correction_review']
    assert review['highlighted'] and review['attachment_ref'] == 'photos/2026-10-08/bag-count.jpg'
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 700.0
    assert exceptions_for(db_cursor, ticket_id=prepared['ticket_id']) == []
    assert ticket_row(db_cursor, prepared)['status'] == 'committed'


def test_large_correction_without_photo_is_held_then_released_by_a_late_photo(client, db_cursor, items, actors):
    key = actors['floor']['key']
    prepared = prepare(client, 'adjust', adjust(items, 600, 'missing_receipt'), key)
    assert prepared['can_commit']        # the hold happens at commit, where the balance is locked
    held = commit(client, prepared, key, acknowledged_warnings=[])
    assert held.status_code == 202, held.text
    data = held.json()
    assert data['held'] is True and data['status'] == 'awaiting_approval' and data['error_code'] == 'PHOTO_REQUIRED'
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'awaiting_approval'
    [exc] = exceptions_for(db_cursor, ticket_id=prepared['ticket_id'])
    assert (exc['kind'], exc['status'], exc['severity']) == ('LARGE_CORRECTION', 'open', 'block')
    assert exc['detail']['payload_hash'] == prepared['payload_hash']
    assert exc['detail']['correction_review']['photo_required'] is True
    assert exc['owner_actor_id'] == actors['owner']['id'] and exc['lot_id'] == items['ingredient']['lot_id']
    assert data['exception_id'] == exc['id']
    # Held tickets do not expire and a repeat commit without a photo is the same hold, not a second exception.
    db_cursor.execute("UPDATE write_tickets SET expires_at=now()-interval '1 day' WHERE id=%s", (prepared['ticket_id'],))
    assert expire(db_cursor) == 0
    again = commit(client, prepared, key)
    assert again.status_code == 202 and again.json()['exception_id'] == exc['id']
    assert len(exceptions_for(db_cursor, ticket_id=prepared['ticket_id'])) == 1
    assert posted_count(db_cursor, prepared) == 0
    # A later photo releases the hold through the normal commit — once.
    released = commit(client, prepared, key, attachment_ref='photos/late.jpg')
    assert released.status_code == 200, released.text
    assert released.json()['correction_review']['attachment_ref'] == 'photos/late.jpg'
    assert released.json()['correction_review']['highlighted'] is True
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 700.0
    [exc] = exceptions_for(db_cursor, ticket_id=prepared['ticket_id'])
    assert (exc['status'], exc['resolution_kind'], exc['resolved_by_actor_id']) == ('resolved', 'photo_attached', actors['floor']['id'])
    assert exc['transaction_id'] == released.json()['transaction_id'] and exc['receipt_number'] == released.json()['receipt_number']
    assert commit(client, prepared, key, attachment_ref='photos/late.jpg').json() == released.json() | {'replayed': True}
    assert commit(client, prepared, key).json() == released.json() | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1


def test_found_over_500_is_held_too(client, db_cursor, items, actors):
    key = actors['floor']['key']
    prepared = prepare(client, 'found', found(items, 600, 'missing_receipt'), key)
    held = commit(client, prepared, key)
    assert held.status_code == 202, held.text
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute('SELECT count(*) AS n FROM lots WHERE product_id=%s', (items['ingredient']['id'],))
    assert db_cursor.fetchone()['n'] == 1          # no found lot was created
    assert commit(client, prepared, key, attachment_ref='p.jpg').status_code == 200


def hold(client_, db_cursor, items_, actors_, delta=600):
    key = actors_['floor']['key']
    prepared = prepare(client_, 'adjust', adjust(items_, delta, 'missing_receipt'), key)
    held = commit(client_, prepared, key)
    assert held.status_code == 202, held.text
    return prepared, held.json()['exception_id']


def test_owner_approval_is_the_commit_posts_once_as_the_preparer(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    error(client.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(actors['floor']['key'])), 403, 'ROLE_NOT_ALLOWED')
    error(client.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(actors['office']['key'])), 403, 'ROLE_NOT_ALLOWED')
    error(client.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers()), 403, 'ROLE_NOT_ALLOWED')   # master key: nothing new
    assert posted_count(db_cursor, prepared) == 0
    approved = client.post(f'/exceptions/{exception_id}/approve', json={'note': 'Photo seen on the floor'},
                           headers=headers(actors['owner']['key']))
    assert approved.status_code == 200, approved.text
    receipt = approved.json()
    assert receipt['receipt_number'].startswith('ADJ-') and receipt['replayed'] is False
    assert receipt['approval']['approved_by']['id'] == actors['owner']['id'] and receipt['approval']['exception_id'] == exception_id
    assert receipt['entry_timing']['entered_by']['id'] == actors['floor']['id']
    db_cursor.execute('SELECT operator_id, entered_by_actor_id, reason_code FROM transactions WHERE id=%s', (receipt['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'operator_id': actors['floor']['name'], 'entered_by_actor_id': actors['floor']['id'],
                                          'reason_code': 'missing_receipt'}
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 700.0
    row = ticket_row(db_cursor, prepared)
    assert row['status'] == 'committed' and row['response'] == receipt
    [exc] = exceptions_for(db_cursor, id=exception_id)
    assert (exc['status'], exc['resolution_kind'], exc['resolved_by_actor_id'], exc['resolution_note']) == \
        ('resolved', 'approved', actors['owner']['id'], 'Photo seen on the floor')
    assert exc['transaction_id'] == receipt['transaction_id'] and exc['resolution_ticket_id'] == prepared['ticket_id']
    # Replay: the approval, the preparer's commit — same receipt, nothing posts twice.
    assert client.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(actors['owner']['key'])).json() == receipt | {'replayed': True}
    assert commit(client, prepared, actors['floor']['key']).json() == receipt | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1
    error(client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'no'},
                      headers=headers(actors['owner']['key'])), 409, 'ALREADY_APPROVED')
    db_cursor.execute("SELECT actor_id, route FROM actor_write_audit WHERE target_table='exceptions' AND target_id=%s", (exception_id,))
    assert db_cursor.fetchone()['actor_id'] == actors['owner']['id']


def test_owner_reject_posts_nothing(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    owner = headers(actors['owner']['key'])
    assert client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined'}, headers=owner).status_code == 422
    error(client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'approved', 'note': 'x'}, headers=owner), 422, 'RESOLUTION_KIND_INVALID')
    error(client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'x'},
                      headers=headers(actors['floor']['key'])), 403, 'ROLE_NOT_ALLOWED')
    rejected = client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'Count it again first'}, headers=owner)
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()['posted'] is False and rejected.json()['replayed'] is False
    assert posted_count(db_cursor, prepared) == 0
    row = ticket_row(db_cursor, prepared)
    assert row['status'] == 'rejected' and 'Count it again first' in row['reject_reason']
    [exc] = exceptions_for(db_cursor, id=exception_id)
    assert (exc['status'], exc['resolution_kind'], exc['resolved_by_actor_id']) == ('resolved', 'declined', actors['owner']['id'])
    assert client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'again'}, headers=owner).json()['replayed'] is True
    error(client.post(f'/exceptions/{exception_id}/approve', json={}, headers=owner), 409, 'EXCEPTION_CLOSED')
    error(commit(client, prepared, actors['floor']['key']), 409, 'TICKET_NOT_COMMITTABLE')
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 100.0


def test_approval_revalidates_and_a_stale_draft_is_not_consumed(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    db_cursor.execute("UPDATE lots SET status='merged' WHERE id=%s", (items['ingredient']['lot_id'],))
    response = client.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(actors['owner']['key']))
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'LOT_MERGED'
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'awaiting_approval'      # still the owner's decision
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'
    # The owner can still decline it explicitly.
    assert client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'lot merged'},
                       headers=headers(actors['owner']['key'])).status_code == 200


def test_approve_wrong_kind_and_resolve_wrong_route(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    owner = headers(actors['owner']['key'])
    error(client.post(f'/exceptions/{exception_id}/resolve', json={'resolution_kind': 'approved', 'note': 'x'}, headers=owner),
          409, 'USE_APPROVE_OR_REJECT')
    _, response = post_short_make(client, db_cursor, items, actors['floor']['key'])
    shortage_id = response['shortages'][0]['exception_id']
    error(client.post(f'/exceptions/{shortage_id}/approve', json={}, headers=owner), 409, 'APPROVAL_NOT_APPLICABLE')
    error(client.post('/exceptions/999999999/approve', json={}, headers=owner), 404, 'EXCEPTION_NOT_FOUND')


def race_actors(isolated_database):
    """Committed fixtures on the dedicated database: products/lots + a floor and an owner actor."""
    token = uuid4().hex[:8].upper()
    keys = {role: f'actor-key-{role}-{token}' for role in ('floor', 'owner')}
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        items_ = seed_items(cur)
        ids = {}
        for role, key in keys.items():
            cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id",
                        (f'ACT {role} {token}', role, sha256(key.encode()).hexdigest()))
            ids[role] = cur.fetchone()['id']
        import os, secrets
        from tests.pin_test_support import seed_owner_pin
        os.environ.setdefault('PIN_PEPPER', secrets.token_hex(32))
        cur.execute((ROOT / 'migrations/073_pin_sessions.sql').read_text())
        seed_owner_pin(cur, ids['owner'], keys['owner'])
    return keys, items_, ids


class Race:
    """App connections that meet at a barrier once per thread while `on` is set
    (actor lookups open extra connections, so each thread waits only once)."""
    def __init__(self, database, parties=2):
        self.database, self.gate, self.on, self.local = database, Barrier(parties), False, threading.local()

    @contextmanager
    def connection(self):
        with psycopg2.connect(self.database) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='10s'")
            if self.on and not getattr(self.local, 'waited', False):
                self.local.waited = True
                self.gate.wait(timeout=10)
            yield conn


def test_concurrent_approvals_post_once(isolated_database, monkeypatch):
    keys, items_, ids = race_actors(isolated_database)
    race = Race(isolated_database)
    monkeypatch.setattr(main, 'get_db_connection', race.connection)
    main._reset_actor_cache()
    with TestClient(main.app) as http:
        prepared = prepare(http, 'adjust', adjust(items_, 600, 'missing_receipt'), keys['floor'])
        held = commit(http, prepared, keys['floor'])
        assert held.status_code == 202, held.text
        exception_id = held.json()['exception_id']
        race.on = True
        approve = lambda: http.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(keys['owner']))
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [f.result(timeout=30) for f in [executor.submit(approve) for _ in range(2)]]
        race.on = False
        assert [r.status_code for r in results] == [200, 200], [r.text for r in results]
        data = [r.json() for r in results]
        assert data[0]['receipt_number'] == data[1]['receipt_number']
        assert sorted(r['replayed'] for r in data) == [False, True]
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, prepared) == 1
            cur.execute("SELECT count(*) AS n FROM exceptions WHERE ticket_id=%s", (prepared['ticket_id'],))
            assert cur.fetchone()['n'] == 1
            cur.execute('SELECT entered_by_actor_id FROM transactions WHERE id=%s', (data[0]['transaction_id'],))
            assert cur.fetchone()['entered_by_actor_id'] == ids['floor']
            assert main.lot_on_hand(cur, items_['ingredient']['lot_id']) == 700.0


# ── B. shortages (R3) ─────────────────────────────────────────────────────

def test_short_make_posts_flags_once_and_replays_without_a_second_flag(client, db_cursor, items, actors):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 4)
    prepared = prepare(client, 'make', body('make', items), key)
    assert prepared['can_commit'] and prepared['blockers'] == []
    [warning] = [w for w in prepared['warnings'] if w['code'] == 'WILL_CREATE_SHORTAGE']
    assert warning['requires_ack'] is False and warning['refs']['short_lb'] == 6.0
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    result = response.json()
    txn = result['transaction_id']
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == -6.0           # never add stock to cover
    db_cursor.execute('SELECT quantity_lb FROM transaction_lines WHERE transaction_id=%s ORDER BY id', (txn,))
    assert [float(r['quantity_lb']) for r in db_cursor.fetchall()] == [10, -10]
    [flag] = flags_for(db_cursor, txn)
    [exc] = exceptions_for(db_cursor, transaction_id=txn)
    assert float(flag['short_lb']) == 6.0 and flag['lot_id'] == items['ingredient']['lot_id'] and flag['status'] == 'open'
    assert flag['exception_id'] == exc['id'] and exc['kind'] == 'SHORTAGE' and exc['severity'] == 'warn'
    assert exc['owner_actor_id'] == actors['floor']['id'] == flag['owner_actor_id']
    assert exc['receipt_number'] == result['receipt_number'] and exc['ticket_id'] == prepared['ticket_id']
    db_cursor.execute('SELECT created_at FROM transactions WHERE id=%s', (txn,))
    entered = db_cursor.fetchone()['created_at']
    assert exc['due_at'] == flag['due_at'] == a3b.business_deadline(entered, 2)
    assert exc['opened_at'] == entered
    assert result['shortages'][0]['exception_id'] == exc['id'] and result['shortages'][0]['short_lb'] == 6.0
    # Replay: original receipt, still one flag and one exception.
    assert commit(client, prepared, key).json() == result | {'replayed': True}
    assert len(flags_for(db_cursor, txn)) == 1 and len(exceptions_for(db_cursor, transaction_id=txn)) == 1
    # And the database would refuse a duplicate even if code tried.
    db_cursor.execute('SAVEPOINT dup')
    with pytest.raises(psycopg2.IntegrityError):
        db_cursor.execute('INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,due_at) VALUES (%s,%s,%s,1,now())',
                          (txn, items['ingredient']['id'], items['ingredient']['lot_id']))
    db_cursor.execute('ROLLBACK TO SAVEPOINT dup')
    with pytest.raises(psycopg2.IntegrityError):
        db_cursor.execute("INSERT INTO exceptions(kind,severity,transaction_id,lot_id) VALUES ('SHORTAGE','warn',%s,%s)",
                          (txn, items['ingredient']['lot_id']))
    db_cursor.execute('ROLLBACK TO SAVEPOINT dup')


def test_short_pack_posts_against_the_pinned_batch_lot(client, db_cursor, items, actors):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'batch', 7)                 # pack 2 cases × 5 lb = 10 lb
    prepared = prepare(client, 'pack', body('pack', items), key)
    assert prepared['can_commit'], prepared
    assert prepared['draft']['input_plan'][0]['short_lb'] == 3.0
    assert [w['code'] for w in prepared['warnings']] == ['WILL_CREATE_SHORTAGE']
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    assert main.lot_on_hand(db_cursor, items['batch']['lot_id']) == -3.0
    [flag] = flags_for(db_cursor, response.json()['transaction_id'])
    assert float(flag['short_lb']) == 3.0 and flag['lot_id'] == items['batch']['lot_id']


def test_pack_from_an_empty_batch_lot_still_posts_and_flags(client, db_cursor, items, actors):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'batch', 0)
    prepared = prepare(client, 'pack', body('pack', items) | {'lot_allocations': [{'lot_id': items['batch']['lot_id'], 'quantity_lb': 10}]}, key)
    assert prepared['can_commit'], prepared
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    assert main.lot_on_hand(db_cursor, items['batch']['lot_id']) == -10.0
    assert float(flags_for(db_cursor, response.json()['transaction_id'])[0]['short_lb']) == 10.0


def test_shortage_is_recomputed_under_lock_at_commit(client, db_cursor, items, actors):
    key = actors['floor']['key']
    set_balance(db_cursor, items, 'ingredient', 4)
    prepared = prepare(client, 'make', body('make', items), key)
    assert any(w['code'] == 'WILL_CREATE_SHORTAGE' for w in prepared['warnings'])
    set_balance(db_cursor, items, 'ingredient', 100)         # the delivery was entered before the commit
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    assert response.json()['shortages'] == [] and flags_for(db_cursor, response.json()['transaction_id']) == []
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 90.0


def test_product_with_no_lot_at_all_still_blocks(client, db_cursor, items):
    db_cursor.execute("INSERT INTO products(name,odoo_code,type,uom) VALUES (%s,%s,'ingredient','lb') RETURNING id",
                      ('A3b lotless ' + uuid4().hex[:8], 'A3B-' + uuid4().hex[:8]))
    pid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,1)', (items['batch']['id'], pid))
    prepared = prepare(client, 'make', body('make', items))
    assert not prepared['can_commit']
    assert prepared['blockers'][0]['code'] == 'INSUFFICIENT_STOCK' and 'no lot' in prepared['blockers'][0]['message']


def test_everything_else_still_blocks_a_short_make(client, db_cursor, items, actors):
    set_balance(db_cursor, items, 'ingredient', 4)
    payload = body('make', items)
    # A2 permissions: office cannot make, short or not.
    error(client.post('/make/prepare', json=payload, headers=headers(actors['office']['key'])), 403, 'ROLE_NOT_ALLOWED')
    # A2 back-dating: floor > 14 days is owner-only.
    old = payload | {'occurred_at': (main.get_plant_now() - timedelta(days=15)).isoformat(), 'backfill': True}
    error(client.post('/make/prepare', json=old, headers=headers(actors['floor']['key'])), 403, 'BACKFILL_OWNER_ONLY')
    # A5 lot confirmation: the short lot still has to be confirmed.
    raw = client.post('/make/prepare', json=payload, headers=headers(actors['floor']['key'])).json()
    assert [b['code'] for b in raw['blockers']] == ['LOT_NOT_CONFIRMED']
    assert any(w['code'] == 'WILL_CREATE_SHORTAGE' for w in raw['warnings'])
    unconfirmed = commit(client, raw, actors['floor']['key'])
    error(unconfirmed, 422, 'LOT_NOT_CONFIRMED')
    assert posted_count(db_cursor, raw) == 0
    # A1 re-validation: a merged lot is stale, shortage or not.
    prepared = prepare(client, 'make', payload, actors['floor']['key'])
    db_cursor.execute("UPDATE lots SET status='merged' WHERE id=%s", (items['ingredient']['lot_id'],))
    stale = commit(client, prepared, actors['floor']['key'])
    error(stale, 409, 'TICKET_STALE')
    assert stale.json()['detail']['blockers'][0]['code'] == 'LOT_MERGED'
    assert posted_count(db_cursor, prepared) == 0
    db_cursor.execute("SELECT count(*) AS n FROM shortage_flags WHERE lot_id=%s", (items['ingredient']['lot_id'],))
    assert db_cursor.fetchone()['n'] == 0


def test_never_add_stock_to_cover_an_open_shortage(client, db_cursor, items, actors):
    key = actors['floor']['key']
    _, result = post_short_make(client, db_cursor, items, key)
    exception_id = result['shortages'][0]['exception_id']
    refused = client.post('/adjust/prepare', json=adjust(items, 6, 'missing_receipt'), headers=headers(key))
    error(refused, 409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')
    assert refused.json()['detail']['exception_id'] == exception_id and refused.json()['detail']['short_lb'] == 6.0
    error(client.post('/inventory/found/prepare', json=found(items, 6, 'missing_receipt'), headers=headers(key)), 409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')
    assert prepare(client, 'adjust', adjust(items, -1, 'damage_disposal'), key)['can_commit']   # taking more is not a cover-up
    # Resolve first (who/when/why recorded), then stock may be corrected.
    assert client.post(f'/exceptions/{exception_id}/resolve', json={'resolution_kind': 'counted'}, headers=headers(key)).status_code == 422
    error(client.post(f'/exceptions/{exception_id}/resolve', json={'resolution_kind': 'approved', 'note': 'x'}, headers=headers(key)), 422, 'RESOLUTION_KIND_INVALID')
    resolved = client.post(f'/exceptions/{exception_id}/resolve', json={'resolution_kind': 'counted', 'counted_lb': 4, 'note': 'Counted 4 lb left on the pallet'}, headers=headers(key))
    assert resolved.status_code == 200, resolved.text
    view = resolved.json()
    assert view['status'] == 'resolved' and view['resolution_kind'] == 'counted' and view['resolved_by']['id'] == actors['floor']['id']
    assert view['resolved_at'] and view['resolution_note'] == 'Counted 4 lb left on the pallet'
    assert view['shortage_flag']['status'] == 'resolved' and view['shortage_flag']['resolution_kind'] == 'counted'
    error(client.post(f'/exceptions/{exception_id}/resolve', json={'resolution_kind': 'counted', 'note': 'again'}, headers=headers(key)), 409, 'EXCEPTION_CLOSED')
    prepared = prepare(client, 'adjust', adjust(items, 6, 'missing_receipt'), key)
    assert commit(client, prepared, key).status_code == 200
    db_cursor.execute("SELECT actor_id FROM actor_write_audit WHERE target_table='exceptions' AND target_id=%s", (exception_id,))
    assert db_cursor.fetchone()['actor_id'] == actors['floor']['id']


# ── C. deadlines ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('entered,days,expected', [
    ('2026-10-08T15:00:00-04:00', 2, '2026-10-12T23:59:00-04:00'),   # Thu → Mon (weekend skipped)
    ('2026-10-09T09:00:00-04:00', 2, '2026-10-13T23:59:00-04:00'),   # Fri → Tue
    ('2026-10-10T09:00:00-04:00', 2, '2026-10-13T23:59:00-04:00'),   # Sat → Tue
    ('2026-10-06T23:30:00-04:00', 2, '2026-10-08T23:59:00-04:00'),   # Tue late → Thu
    ('2026-10-07T03:30:00+00:00', 2, '2026-10-08T23:59:00-04:00'),   # UTC instant that is Tue 23:30 in NY
    ('2026-10-30T10:00:00-04:00', 2, '2026-11-03T23:59:00-05:00'),   # across the Nov 1 fall-back
    ('2026-10-09T09:00:00-04:00', 7, '2026-10-20T23:59:00-04:00'),   # unidentified lot: seven weekdays
])
def test_business_day_deadline(entered, days, expected):
    assert a3b.business_deadline(datetime.fromisoformat(entered), days).isoformat() == expected


def test_a5_identification_deadline_is_the_same_clock():
    when = datetime.fromisoformat('2026-10-09T09:00:00-04:00')
    assert lot_confirmation.identification_deadline(when) == a3b.business_deadline(when, 7)
    with pytest.raises(ValueError):
        a3b.business_deadline(datetime(2026, 10, 9, 9), 2)


# ── D/E. permissions and routes ───────────────────────────────────────────

def test_exceptions_routes_are_named_actor_only(client, db_cursor, items, actors):
    _, result = post_short_make(client, db_cursor, items, actors['floor']['key'])
    exception_id = result['shortages'][0]['exception_id']
    for role in ('owner', 'floor', 'office'):
        listed = client.get('/exceptions', headers=headers(actors[role]['key']))
        assert listed.status_code == 200, listed.text
        assert exception_id in [e['id'] for e in listed.json()['exceptions']]
        assert client.get(f'/exceptions/{exception_id}', headers=headers(actors[role]['key'])).json()['id'] == exception_id
    error(client.get('/exceptions', headers=headers()), 403, 'ROLE_NOT_ALLOWED')                 # master key
    response = client.get('/exceptions', headers=headers(main.DASHBOARD_API_KEY))                    # dashboard key
    assert response.status_code == 403 and response.json()['detail'] == 'API key not authorized for this endpoint'
    assert client.get('/exceptions', headers={'X-API-Key': 'not-a-key'}).status_code == 403
    assert client.get('/exceptions', headers=headers(actors['retired']['key'])).status_code == 403
    body_ = {'resolution_kind': 'counted', 'counted_lb': 4, 'note': 'x'}
    error(client.post(f'/exceptions/{exception_id}/resolve', json=body_, headers=headers(actors['office']['key'])), 403, 'ROLE_NOT_ALLOWED')
    error(client.post(f'/exceptions/{exception_id}/resolve', json=body_, headers=headers()), 403, 'ROLE_NOT_ALLOWED')
    assert client.post(f'/exceptions/{exception_id}/resolve', json=body_, headers=headers(main.DASHBOARD_API_KEY)).status_code == 403
    # Owner-only kinds: floor may resolve a shortage but not acknowledge a late entry.
    db_cursor.execute("""INSERT INTO exceptions(kind,severity,detail,owner_actor_id) VALUES ('LATE_ENTRY','warn','{}',%s) RETURNING id""",
                      (actors['owner']['id'],))
    late = db_cursor.fetchone()['id']
    error(client.post(f'/exceptions/{late}/resolve', json={'resolution_kind': 'acknowledged', 'note': 'ok'},
                      headers=headers(actors['floor']['key'])), 403, 'ROLE_NOT_ALLOWED')
    acknowledged = client.post(f'/exceptions/{late}/resolve', json={'resolution_kind': 'acknowledged', 'note': 'Seen, paper was late'},
                               headers=headers(actors['owner']['key']))
    assert acknowledged.status_code == 200 and acknowledged.json()['resolved_by']['id'] == actors['owner']['id']
    assert client.post(f'/exceptions/{exception_id}/resolve', json=body_, headers=headers(actors['floor']['key'])).status_code == 200
    assert exception_id not in [e['id'] for e in client.get('/exceptions', headers=headers(actors['floor']['key'])).json()['exceptions']]


def test_list_filters_view_and_overdue(client, db_cursor, items, actors):
    key = actors['owner']['key']
    _, result = post_short_make(client, db_cursor, items, actors['floor']['key'])
    shortage_id = result['shortages'][0]['exception_id']
    db_cursor.execute("INSERT INTO exceptions(kind,severity,detail,due_at) VALUES ('UNIDENTIFIED_LOT','warn','{}',now()-interval '1 day') RETURNING id")
    overdue_id = db_cursor.fetchone()['id']
    listed = client.get('/exceptions', headers=headers(key)).json()
    ids = [e['id'] for e in listed['exceptions']]
    assert ids[0] == overdue_id            # overdue first
    assert shortage_id in ids
    assert client.get('/exceptions?overdue=true', headers=headers(key)).json()['exceptions'][0]['id'] == overdue_id
    assert shortage_id not in [e['id'] for e in client.get('/exceptions?overdue=true', headers=headers(key)).json()['exceptions']]
    assert shortage_id in [e['id'] for e in client.get('/exceptions?overdue=false', headers=headers(key)).json()['exceptions']]
    assert [e['kind'] for e in client.get('/exceptions?kind=shortage', headers=headers(key)).json()['exceptions']] == ['SHORTAGE'] * \
        len(client.get('/exceptions?kind=shortage', headers=headers(key)).json()['exceptions'])
    assert client.get(f'/exceptions?lot_id={items["ingredient"]["lot_id"]}', headers=headers(key)).json()['exceptions'][0]['id'] == shortage_id
    view = client.get(f'/exceptions/{shortage_id}', headers=headers(key)).json()
    assert view['lot'] == {'id': items['ingredient']['lot_id'], 'lot_code': items['ingredient']['lot_code']}
    assert view['product']['id'] == items['ingredient']['id'] and view['owner']['id'] == actors['floor']['id']
    assert view['overdue'] is False and view['shortage_flag']['short_lb'] == 6.0 and view['ticket_receipt_number'] == result['receipt_number']
    assert client.get(f'/exceptions/{overdue_id}', headers=headers(key)).json()['overdue'] is True
    error(client.get('/exceptions/999999999', headers=headers(key)), 404, 'EXCEPTION_NOT_FOUND')
    assert client.get('/exceptions?status=bogus', headers=headers(key)).status_code == 422
    assert shortage_id not in [e['id'] for e in client.get('/exceptions?status=resolved', headers=headers(key)).json()['exceptions']]
    assert client.post(f'/exceptions/{shortage_id}/resolve', json={'resolution_kind': 'counted', 'counted_lb': 0, 'note': 'nothing left'},
                       headers=headers(key)).status_code == 200
    assert shortage_id in [e['id'] for e in client.get('/exceptions?status=resolved', headers=headers(key)).json()['exceptions']]
    assert shortage_id not in [e['id'] for e in client.get('/exceptions', headers=headers(key)).json()['exceptions']]
    assert shortage_id in [e['id'] for e in client.get('/exceptions?status=all', headers=headers(key)).json()['exceptions']]


def test_nightly_sweep_escalates_overdue_once(client, db_cursor, items, actors):
    key = actors['owner']['key']
    _, result = post_short_make(client, db_cursor, items, actors['floor']['key'])
    shortage_id = result['shortages'][0]['exception_id']
    db_cursor.execute("UPDATE exceptions SET due_at=now()-interval '1 hour' WHERE id=%s", (shortage_id,))
    db_cursor.execute("UPDATE shortage_flags SET due_at=now()-interval '1 hour' WHERE exception_id=%s", (shortage_id,))
    db_cursor.execute("INSERT INTO exceptions(kind,severity,detail,status,resolved_at,due_at) VALUES ('SHORTAGE','warn','{}','resolved',now(),now()-interval '2 days') RETURNING id")
    resolved_id = db_cursor.fetchone()['id']
    assert sweep(db_cursor) == (1, 1)
    assert sweep(db_cursor) == (0, 0)                                      # idempotent
    [exc] = exceptions_for(db_cursor, id=shortage_id)
    assert exc['status'] == 'escalated' and exc['escalated_at'] is not None
    assert exceptions_for(db_cursor, id=resolved_id)[0]['status'] == 'resolved'
    db_cursor.execute('SELECT status FROM shortage_flags WHERE exception_id=%s', (shortage_id,))
    assert db_cursor.fetchone()['status'] == 'escalated'
    listed = client.get('/exceptions', headers=headers(key)).json()['exceptions']    # escalated is still in the open queue
    assert [e['status'] for e in listed if e['id'] == shortage_id] == ['escalated']
    assert client.get('/exceptions?status=escalated', headers=headers(key)).json()['exceptions'][0]['overdue'] is True
    # It can still be resolved, and the flag follows.
    assert client.post(f'/exceptions/{shortage_id}/resolve', json={'resolution_kind': 'counted', 'counted_lb': 0, 'note': 'nothing left'},
                       headers=headers(key)).status_code == 200
    db_cursor.execute('SELECT status, resolution_kind FROM shortage_flags WHERE exception_id=%s', (shortage_id,))
    assert dict(db_cursor.fetchone()) == {'status': 'resolved', 'resolution_kind': 'counted'}


def test_sweep_script_refuses_unguarded_targets(monkeypatch):
    import scripts.exceptions_sweep as script
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pw@db.example.com:5432/x')
    monkeypatch.delenv('ENVIRONMENT', raising=False)
    with pytest.raises(RuntimeError):
        script.main()
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('PRODUCTION_DATABASE_HOST', 'somewhere.else')
    with pytest.raises(RuntimeError):
        script.main()


def test_attachment_only_on_corrections(client, items):
    prepared = prepare(client, 'make', body('make', items))
    error(commit(client, prepared, attachment_ref='x.jpg'), 422, 'UNUSED_ATTACHMENT')


def test_permission_rows_are_the_a2_ones(client):
    import permissions
    assert permissions.ROUTE_ACTIONS[('POST', '/exceptions/{exception_id}/resolve')] == 'resolve_exception'
    assert permissions.ROUTE_ACTIONS[('POST', '/exceptions/{exception_id}/approve')] == 'approve_exception'
    assert permissions.ROUTE_ACTIONS[('POST', '/exceptions/{exception_id}/reject')] == 'approve_exception'
    assert permissions.allowed('list_exceptions', 'office') and not permissions.allowed('list_exceptions', 'legacy_ledger')
    assert permissions.allowed('resolve_exception', 'floor') and not permissions.allowed('resolve_exception', 'office')
    assert permissions.allowed('approve_exception', 'owner') and not permissions.allowed('approve_exception', 'floor')


# ── migration 069 ─────────────────────────────────────────────────────────

def test_migration_069_rerunnable_down_and_up(isolated_database):
    up = (ROOT / 'migrations/069_exceptions_enforcement.sql').read_text()
    down = (ROOT / 'migrations/down/069_exceptions_enforcement_down.sql').read_text()

    def state(cur):
        cur.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname='write_tickets_status_check'")
        check = cur.fetchone()[0]
        cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_name='ledger_current_transactions'
                       AND column_name IN ('reason_code','entered_by_actor_id') ORDER BY 1""")
        columns = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname IN ('shortage_flags_one_per_line_idx','exceptions_one_shortage_per_line_idx','exceptions_one_hold_per_ticket_idx')")
        indexes = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM migration_markers WHERE name='069_exceptions_enforcement'")
        return 'awaiting_approval' in check, columns, indexes, cur.fetchone()[0]

    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL search_path TO public')
        assert state(cur) == (True, ['entered_by_actor_id', 'reason_code'], 3, 1)   # fixture applied it
        cur.execute(up)                                                            # rerun: no-op
        assert state(cur) == (True, ['entered_by_actor_id', 'reason_code'], 3, 1)
        cur.execute("SELECT count(*) FROM pg_views WHERE viewname IN ('inventory_summary','lot_balances','v_lot_quantities','v_test_batches_for_review')")
        assert cur.fetchone()[0] == 4
        # The module-scoped database already carries shortage/hold rows from the
        # tests above: the down refuses them until the export is confirmed.
        cur.execute('SAVEPOINT refused')
        with pytest.raises(psycopg2.errors.RaiseException):
            cur.execute(down)
        cur.execute('ROLLBACK TO SAVEPOINT refused')
        cur.execute("SET LOCAL factory_ledger.confirm_exceptions_export = 'yes'")
        cur.execute(down)
        assert state(cur) == (False, [], 0, 0)
        cur.execute("SELECT count(*) FROM pg_views WHERE viewname IN ('inventory_summary','lot_balances','v_lot_quantities','v_test_batches_for_review')")
        assert cur.fetchone()[0] == 4                                              # dependents recreated
        cur.execute(up)
        assert state(cur) == (True, ['entered_by_actor_id', 'reason_code'], 3, 1)
        # The down refuses while a ticket is held.
        cur.execute("""INSERT INTO write_tickets(ticket_hash,action,operator_id,key_kind,client_source,payload,payload_hash,state_hash,draft,warnings,status,expires_at)
                       VALUES (repeat('a',64),'adjust','x','legacy_ledger','api','{}',repeat('b',64),repeat('c',64),'{}','[]','awaiting_approval',now())""")
        cur.execute('SAVEPOINT held')
        with pytest.raises(psycopg2.errors.RaiseException):
            cur.execute(down)
        cur.execute('ROLLBACK TO SAVEPOINT held')


def test_migration_069_sweeps_the_p1_8_null_window(isolated_database):
    up = (ROOT / 'migrations/069_exceptions_enforcement.sql').read_text()
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute('SET LOCAL search_path TO public')
        cur.execute("""INSERT INTO transactions(type,adjust_reason,notes) VALUES ('adjust','spoilage','x'),
                       ('adjust',NULL,'Found inventory: found_during_count'), ('adjust','whatever','x') RETURNING id""")
        ids = [r['id'] for r in cur.fetchall()]
        cur.execute('SELECT reason_code FROM transactions WHERE id=ANY(%s) ORDER BY id', (ids,))
        assert [r['reason_code'] for r in cur.fetchall()] == [None, None, None]
        cur.execute(up)
        cur.execute('SELECT reason_code FROM transactions WHERE id=ANY(%s) ORDER BY id', (ids,))
        assert [r['reason_code'] for r in cur.fetchall()] == ['damage_disposal', 'physical_count', 'unknown']
        cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgname='trg_transactions_original_append_only'")
        assert cur.fetchone()['tgenabled'] != 'D'


# ── Codex review of PR #94 (2026-10-09): evidence, approver re-checks, lock order,
#    shared-key preparers, §5 / §5.1 flags ──────────────────────────────────

def open_shortage(client_, db_cursor, items_, actors_):
    _, result = post_short_make(client_, db_cursor, items_, actors_['floor']['key'])
    return result['shortages'][0]['exception_id'], result


def resolve(client_, exception_id, key, **body_):
    return client_.post(f'/exceptions/{exception_id}/resolve', json=body_, headers=headers(key))


def approve(client_, exception_id, key, **body_):
    return client_.post(f'/exceptions/{exception_id}/approve', json=body_, headers=headers(key))


@pytest.mark.parametrize('route,body_', [('resolve', {'resolution_kind': 'counted', 'counted_lb': 2, 'note': '   '}),
                                         ('reject', {'resolution_kind': 'declined', 'note': '\t\n'})])
def test_blank_notes_are_refused(client, db_cursor, items, actors, route, body_):
    if route == 'resolve':
        exception_id, _ = open_shortage(client, db_cursor, items, actors)
        key = actors['floor']['key']
    else:
        _, exception_id = hold(client, db_cursor, items, actors)
        key = actors['owner']['key']
    response = client.post(f'/exceptions/{exception_id}/{route}', json=body_, headers=headers(key))
    assert response.status_code == 422, response.text
    assert 'blank' in response.text
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'


def test_counted_resolution_posts_the_counted_correction_atomically(client, db_cursor, items, actors):
    key = actors['floor']['key']
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    lot_id = items['ingredient']['lot_id']
    assert main.lot_on_hand(db_cursor, lot_id) == -6.0
    error(resolve(client, exception_id, key, resolution_kind='counted', note='counted it'), 422, 'COUNT_REQUIRED')
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'
    resolved = resolve(client, exception_id, key, resolution_kind='counted', counted_lb=2, note='Two pounds left on the pallet')
    assert resolved.status_code == 200, resolved.text
    view = resolved.json()
    assert view['status'] == 'resolved' and view['resolution_kind'] == 'counted'
    res = view['detail']['resolution']
    assert (res['book_before_lb'], res['delta_lb'], res['counted_lb'], res['new_balance_lb'], res['short_lb']) == (-6.0, 8.0, 2.0, 2.0, 6.0)
    assert main.lot_on_hand(db_cursor, lot_id) == 2.0
    db_cursor.execute('SELECT type, reason_code, entered_by_actor_id, notes FROM transactions WHERE id=%s', (res['transaction_id'],))
    txn = db_cursor.fetchone()
    assert (txn['type'], txn['reason_code'], txn['entered_by_actor_id']) == ('adjust', 'physical_count', actors['floor']['id'])
    assert f'Shortage #{exception_id}' in txn['notes'] and 'Two pounds left' in txn['notes']
    assert view['shortage_flag']['status'] == 'resolved' and view['shortage_flag']['resolution_kind'] == 'counted'
    # Now, and only now, a positive adjust reaches the lot through a ticket.
    assert prepare(client, 'adjust', adjust(items, 1, 'missing_receipt'), key)['can_commit']


def test_counted_resolution_over_500_lb_needs_a_photo(client, db_cursor, items, actors):
    key = actors['floor']['key']
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    lot_id = items['ingredient']['lot_id']
    refused = resolve(client, exception_id, key, resolution_kind='counted', counted_lb=600, note='a whole pallet was mis-shelved')
    error(refused, 422, 'PHOTO_REQUIRED')
    assert main.lot_on_hand(db_cursor, lot_id) == -6.0 and exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'
    ok = resolve(client, exception_id, key, resolution_kind='counted', counted_lb=600, note='a whole pallet was mis-shelved',
                 attachment_ref='photos/pallet.jpg')
    assert ok.status_code == 200, ok.text
    review = ok.json()['detail']['resolution']['correction_review']
    assert review['rules'] == ['BOOK_BALANCE_NOT_POSITIVE', 'OVER_500_LB'] and review['highlighted'] is True
    assert ok.json()['detail']['resolution']['attachment_ref'] == 'photos/pallet.jpg'
    assert main.lot_on_hand(db_cursor, lot_id) == 600.0


def ensure_supplier(cur, items_):
    """One supplier per fixture lot, as A5 requires for repeat receives onto the same lot."""
    cur.execute('SELECT id FROM suppliers WHERE name=%s', ('A3b supplier ' + items_['ingredient']['lot_code'],))
    supplier = cur.fetchone()
    if not supplier:
        cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id', ('A3b supplier ' + items_['ingredient']['lot_code'],))
        supplier = cur.fetchone()
    return supplier['id']


def receive_onto(client_, db_cursor, items_, key, cases, case_size_lb, supplier_id=None):
    """A receive ticket onto the fixture ingredient lot (the movement that was never entered);
    one supplier and one supplier lot code per lot, as A5 requires."""
    supplier_id = supplier_id or ensure_supplier(db_cursor, items_)
    payload = {'product_id': items_['ingredient']['id'], 'supplier_id': supplier_id, 'cases': cases,
               'case_size_lb': case_size_lb, 'bol_reference': 'A3B-' + uuid4().hex[:8], 'lot_code': items_['ingredient']['lot_code'],
               'supplier_lot_code': 'SUP-' + items_['ingredient']['lot_code'], 'occurred_at': main.get_plant_now().isoformat()}
    prepared = client_.post('/receive/prepare', json=payload, headers=headers(key))
    assert prepared.status_code == 200, prepared.text
    posted = commit(client_, prepared.json(), key,
                    acknowledged_warnings=[w['code'] for w in prepared.json()['warnings'] if w.get('requires_ack')])
    assert posted.status_code == 200, posted.text
    return posted.json()


def test_missing_movement_needs_the_matching_receipt(client, db_cursor, items, actors):
    key = actors['floor']['key']
    exception_id, result = open_shortage(client, db_cursor, items, actors)
    error(resolve(client, exception_id, key, resolution_kind='missing_movement', note='the receive'), 422, 'RECEIPT_REQUIRED')
    error(resolve(client, exception_id, key, resolution_kind='missing_movement', receipt_number='RCV-000000-000', note='x'), 404, 'RECEIPT_NOT_FOUND')
    # The short make's own receipt puts nothing on the short lot — unrelated.
    refused = resolve(client, exception_id, key, resolution_kind='missing_movement', receipt_number=result['receipt_number'], note='x')
    error(refused, 409, 'RECEIPT_NOT_MATCHING')
    assert 'same lot' in refused.json()['detail']['problems'][0]
    # A receive onto the lot that is too small to cover the shortage: claimed in part, still open.
    small = receive_onto(client, db_cursor, items, key, cases=1, case_size_lb=2)
    partial = resolve(client, exception_id, key, resolution_kind='missing_movement', receipt_number=small['receipt_number'], note='x')
    assert partial.status_code == 200, partial.text
    assert partial.json()['status'] == 'open' and partial.json()['shortage_flag']['status'] == 'open'
    assert partial.json()['evidence']['claimed_lb'] == 2.0 and partial.json()['evidence']['remaining_lb'] == 4.0
    assert partial.json()['detail']['last_claim']['closes'] is False and 'resolution' not in partial.json()['detail']
    error(client.post('/adjust/prepare', json=adjust(items, 1, 'missing_receipt'), headers=headers(key)), 409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')
    # The same receipt cannot be spent twice.
    reused = resolve(client, exception_id, key, resolution_kind='missing_movement', receipt_number=small['receipt_number'], note='x')
    error(reused, 409, 'RECEIPT_NOT_MATCHING')
    assert 'already used' in reused.json()['detail']['problems'][0] and reused.json()['detail']['remaining_lb'] == 4.0
    # The receive that was never entered: same lot, after the shortage opened, covers the remainder.
    covering = receive_onto(client, db_cursor, items, key, cases=2, case_size_lb=5)
    ok = resolve(client, exception_id, key, resolution_kind='missing_movement', receipt_number=covering['receipt_number'],
                 note='Receive from Tuesday was never entered')
    assert ok.status_code == 200, ok.text
    view = ok.json()
    assert view['status'] == 'resolved' and view['resolution_ticket_id'] == covering['ticket_id'] and view['shortage_flag']['status'] == 'resolved'
    res = view['detail']['resolution']
    assert (res['kind'], res['receipt_number'], res['transaction_ids'], res['added_lb'], res['receipt_used_before_lb'], res['claimed_lb'],
            res['claimed_total_lb'], res['short_lb'], res['remaining_lb'], res['closes'], res['balance_now_lb']) == \
        ('missing_movement', covering['receipt_number'], [covering['transaction_id']], 10.0, 0.0, 4.0, 6.0, 6.0, 0.0, True, 6.0)
    assert view['evidence']['claimed_lb'] == 6.0 and [c['claimed_lb'] for c in view['evidence']['claims']] == [2.0, 4.0]
    assert prepare(client, 'adjust', adjust(items, 1, 'missing_receipt'), key)['can_commit']


def test_voided_resolution_needs_the_short_posting_voided(client, db_cursor, items, actors):
    key = actors['floor']['key']
    exception_id, result = open_shortage(client, db_cursor, items, actors)
    refused = resolve(client, exception_id, key, resolution_kind='voided', note='entered twice')
    error(refused, 409, 'POSTING_STILL_EFFECTIVE')
    assert refused.json()['detail']['effective_status'] == 'posted'
    db_cursor.execute("""INSERT INTO ledger_corrections(target_table,target_id,event_type,previous_values,replacement_values,reason)
                         VALUES ('transactions',%s,'void','{"status":"posted"}','{"status":"voided"}','A3b test void')""",
                      (result['transaction_id'],))
    ok = resolve(client, exception_id, key, resolution_kind='voided', note='entered twice')
    assert ok.status_code == 200, ok.text
    assert ok.json()['detail']['resolution'] == {'kind': 'voided', 'transaction_id': result['transaction_id'], 'effective_status': 'voided'}


def test_write_off_is_owner_only_and_identified_needs_the_lot_identified(client, db_cursor, items, actors):
    def unidentified():
        db_cursor.execute("""INSERT INTO exceptions(kind,severity,detail,owner_actor_id,lot_id,product_id)
                             VALUES ('UNIDENTIFIED_LOT','warn','{}',%s,%s,%s) RETURNING id""",
                          (actors['floor']['id'], items['ingredient']['lot_id'], items['ingredient']['id']))
        return db_cursor.fetchone()['id']
    floor, owner = actors['floor']['key'], actors['owner']['key']
    first = unidentified()
    error(resolve(client, first, floor, resolution_kind='written_off', note='label gone'), 403, 'ROLE_NOT_ALLOWED')
    refused = resolve(client, first, floor, resolution_kind='identified', note='found the label')
    error(refused, 409, 'LOT_NOT_IDENTIFIED')
    assert exceptions_for(db_cursor, id=first)[0]['status'] == 'open'
    db_cursor.execute("UPDATE lots SET identity_status='identified' WHERE id=%s", (items['ingredient']['lot_id'],))
    assert resolve(client, first, floor, resolution_kind='identified', note='found the label').json()['status'] == 'resolved'
    second = unidentified()
    written_off = resolve(client, second, owner, resolution_kind='written_off', note='disposed, no label')
    assert written_off.status_code == 200 and written_off.json()['status'] == 'waived'


def test_approval_rechecks_the_approver_under_lock(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    owner = actors['owner']['key']
    db_cursor.execute("UPDATE actors SET active=false WHERE id=%s", (actors['owner']['id'],))
    denied = approve(client, exception_id, owner, note='ok')          # the auth cache may still admit the key
    assert denied.status_code == 403, denied.text
    db_cursor.execute("UPDATE actors SET active=true, role='office' WHERE id=%s", (actors['owner']['id'],))
    demoted = approve(client, exception_id, owner, note='ok')
    assert demoted.status_code == 403, demoted.text
    rejected = client.post(f'/exceptions/{exception_id}/reject', json={'resolution_kind': 'declined', 'note': 'no'}, headers=headers(owner))
    assert rejected.status_code == 403, rejected.text
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'awaiting_approval'
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'
    db_cursor.execute("UPDATE actors SET role='owner' WHERE id=%s", (actors['owner']['id'],))
    assert approve(client, exception_id, owner, note='ok').status_code == 200


def test_approval_fails_when_the_preparer_may_no_longer_adjust(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    owner = actors['owner']['key']
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s", (actors['floor']['id'],))
    response = approve(client, exception_id, owner)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'ROLE_NOT_ALLOWED'
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'awaiting_approval'
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'
    db_cursor.execute("UPDATE actors SET role='floor' WHERE id=%s", (actors['floor']['id'],))
    posted = approve(client, exception_id, owner)
    assert posted.status_code == 200 and posted.json()['entry_timing']['entered_by']['id'] == actors['floor']['id']


def test_approving_a_hold_without_a_named_preparer_is_a_409_not_a_crash(client, db_cursor, items, actors):
    prepared, exception_id = hold(client, db_cursor, items, actors)
    db_cursor.execute("UPDATE write_tickets SET actor_id=NULL, key_kind='legacy_ledger' WHERE id=%s", (prepared['ticket_id'],))
    response = approve(client, exception_id, actors['owner']['key'])
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'PREPARER_UNKNOWN'
    assert posted_count(db_cursor, prepared) == 0 and ticket_row(db_cursor, prepared)['status'] == 'awaiting_approval'


def test_shared_key_gets_photo_required_not_a_hold(client, db_cursor, items):
    prepared = prepare(client, 'adjust', adjust(items, 600, 'missing_receipt'))      # master key
    response = commit(client, prepared)
    error(response, 422, 'PHOTO_REQUIRED')
    assert response.json()['detail']['held'] is False
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    assert exceptions_for(db_cursor, ticket_id=prepared['ticket_id']) == []
    assert posted_count(db_cursor, prepared) == 0
    posted = commit(client, prepared, attachment_ref='photos/pallet.jpg')
    assert posted.status_code == 200, posted.text
    assert posted.json()['correction_review']['highlighted'] is True and posted.json()['correction_review']['attachment_ref'] == 'photos/pallet.jpg'
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 700.0


def test_approve_locks_the_ticket_before_the_exception(isolated_database, monkeypatch):
    """The photo-release commit locks the ticket first and then touches the exception.
    A session holding the ticket row must be able to update the exception while an
    approve waits — i.e. the waiting approve has not taken the exception lock."""
    keys, items_, ids = race_actors(isolated_database)
    race = Race(isolated_database)
    monkeypatch.setattr(main, 'get_db_connection', race.connection)
    main._reset_actor_cache()
    with TestClient(main.app) as http:
        prepared = prepare(http, 'adjust', adjust(items_, 600, 'missing_receipt'), keys['floor'])
        held = commit(http, prepared, keys['floor'])
        assert held.status_code == 202, held.text
        exception_id = held.json()['exception_id']
        holder = psycopg2.connect(isolated_database)
        try:
            with holder.cursor() as hc:
                hc.execute('SELECT id FROM write_tickets WHERE id=%s FOR UPDATE', (prepared['ticket_id'],))
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: http.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(keys['owner'])))
                    deadline = time.time() + 15
                    while time.time() < deadline:
                        # Backend activity is snapshotted per transaction; refresh it each poll.
                        hc.execute("""SELECT pg_stat_clear_snapshot(), count(*) FROM pg_stat_activity
                                      WHERE datname=current_database() AND wait_event_type='Lock' AND pid<>pg_backend_pid()""")
                        if hc.fetchone()[1] >= 1:
                            break
                        time.sleep(0.05)
                    else:
                        pytest.fail('the approve never waited on the ticket lock')
                    hc.execute("SET LOCAL lock_timeout='2s'")
                    hc.execute('UPDATE exceptions SET detail = detail WHERE id=%s', (exception_id,))   # must not wait on the approve
                    holder.rollback()
                    response = future.result(timeout=30)
        finally:
            holder.rollback()
            holder.close()
        assert response.status_code == 200, response.text
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, prepared) == 1


def test_concurrent_approve_and_photo_commit_post_once_without_deadlock(isolated_database, monkeypatch):
    keys, items_, ids = race_actors(isolated_database)
    race = Race(isolated_database)
    monkeypatch.setattr(main, 'get_db_connection', race.connection)
    main._reset_actor_cache()
    with TestClient(main.app) as http:
        prepared = prepare(http, 'adjust', adjust(items_, 600, 'missing_receipt'), keys['floor'])
        held = commit(http, prepared, keys['floor'])
        assert held.status_code == 202, held.text
        exception_id = held.json()['exception_id']
        race.on = True
        calls = [lambda: http.post(f'/exceptions/{exception_id}/approve', json={}, headers=headers(keys['owner'])),
                 lambda: commit(http, prepared, keys['floor'], attachment_ref='photos/late.jpg')]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [f.result(timeout=30) for f in [executor.submit(call) for call in calls]]
        race.on = False
        assert [r.status_code for r in results] == [200, 200], [r.text for r in results]
        data = [r.json() for r in results]
        assert data[0]['receipt_number'] == data[1]['receipt_number']
        assert sorted(r['replayed'] for r in data) == [False, True]
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, prepared) == 1
            [exc] = exceptions_for(cur, id=exception_id)
            assert exc['status'] == 'resolved' and exc['resolution_kind'] in ('approved', 'photo_attached')
            cur.execute('SELECT status FROM write_tickets WHERE id=%s', (prepared['ticket_id'],))
            assert cur.fetchone()['status'] == 'committed'


@pytest.mark.parametrize('action', ['adjust', 'found'])
def test_small_unknown_corrections_are_highlighted(client, items, actors, action):
    key = actors['floor']['key']
    small = adjust(items, 5, 'unknown', note='no idea where it came from') if action == 'adjust' else \
        found(items, 5, 'unknown', notes='no idea where it came from')
    review = prepare(client, action, small, key)['draft']['correction_review']
    assert (review['highlighted'], review['rules'], review['photo_required'], review['reason_code']) == (True, ['REASON_UNKNOWN'], False, 'unknown')
    plain = adjust(items, 5, 'missing_receipt') if action == 'adjust' else found(items, 5, 'missing_receipt')
    assert prepare(client, action, plain, key)['draft']['correction_review']['highlighted'] is False


def test_positive_adjust_shortly_before_a_make_is_tagged_pre_make_adjust(client, db_cursor, items, actors):
    key, owner = actors['floor']['key'], actors['owner']['key']
    lot = items['ingredient']
    # Too old (31 min), by the floor: outside the window. (The created_at trigger is
    # paused for this one fixture row only; the ledger never back-dates entry time.)
    db_cursor.execute('ALTER TABLE transactions DISABLE TRIGGER trg_transactions_created_at')
    db_cursor.execute("""INSERT INTO transactions(type,notes,created_at,entered_by_actor_id,operator_id)
                         VALUES ('adjust','old', clock_timestamp() - interval '31 minutes', %s, %s) RETURNING id""",
                      (actors['floor']['id'], actors['floor']['name']))
    old = db_cursor.fetchone()['id']
    db_cursor.execute('ALTER TABLE transactions ENABLE TRIGGER trg_transactions_created_at')
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,4)', (old, lot['id'], lot['lot_id']))
    plus = commit(client, prepare(client, 'adjust', adjust(items, 3, 'missing_receipt'), key), key).json()        # tagged
    commit(client, prepare(client, 'adjust', adjust(items, -1, 'damage_disposal'), key), key)                      # negative: never
    commit(client, prepare(client, 'adjust', adjust(items, 2, 'missing_receipt'), owner), owner)                   # other actor: not
    made = commit(client, prepare(client, 'make', body('make', items), key), key).json()
    tags = made['pre_make_adjusts']
    assert [t['adjust_transaction_id'] for t in tags] == [plus['transaction_id']]
    assert tags[0]['same_lot'] is True and tags[0]['adjust_lb'] == 3.0 and tags[0]['minutes_before'] < 5
    assert tags[0]['adjust_receipt_number'] == plus['receipt_number']
    [exc] = exceptions_for(db_cursor, id=tags[0]['exception_id'])
    assert (exc['kind'], exc['severity'], exc['status'], exc['transaction_id'], exc['receipt_number'], exc['owner_actor_id']) == \
        ('PRE_MAKE_ADJUST', 'info', 'open', plus['transaction_id'], plus['receipt_number'], actors['owner']['id'])
    assert exc['detail']['pre_make_adjust'] is True and exc['detail']['followed_by']['receipt_number'] == made['receipt_number']
    # A second (different) make right after does not re-tag the same adjust (070 unique index).
    second = prepare(client, 'make', body('make', items) | {'batches': 2}, key)
    again = commit(client, second, key, acknowledged_warnings=[w['code'] for w in second['warnings'] if w.get('requires_ack')]).json()
    assert again['replayed'] is False
    assert [t['exception_id'] for t in again['pre_make_adjusts']] == [tags[0]['exception_id']]
    assert len(exceptions_for(db_cursor, kind='PRE_MAKE_ADJUST', transaction_id=plus['transaction_id'])) == 1
    # Listed for the owner, acknowledged by the owner only.
    listed = client.get('/exceptions?kind=PRE_MAKE_ADJUST', headers=headers(owner)).json()
    assert tags[0]['exception_id'] in [e['id'] for e in listed['exceptions']]
    error(resolve(client, tags[0]['exception_id'], key, resolution_kind='acknowledged', note='seen'), 403, 'ROLE_NOT_ALLOWED')
    assert resolve(client, tags[0]['exception_id'], owner, resolution_kind='acknowledged', note='Asked; the bag had been found').status_code == 200


def test_migration_070_rerunnable_down_and_up(isolated_database):
    up = (ROOT / 'migrations/070_pre_make_adjust.sql').read_text()
    down = (ROOT / 'migrations/down/070_pre_make_adjust_down.sql').read_text()

    def state(cur):
        cur.execute("SELECT pg_get_constraintdef(oid) LIKE '%%PRE_MAKE_ADJUST%%' FROM pg_constraint WHERE conname='exceptions_kind_check'")
        kind_ok = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname='exceptions_one_pre_make_tag_idx'")
        indexes = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM migration_markers WHERE name='070_pre_make_adjust'")
        return kind_ok, indexes, cur.fetchone()[0]

    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL search_path TO public')
        assert state(cur) == (True, 1, 1)      # fixture applied it
        cur.execute(up)                        # rerun: no-op
        assert state(cur) == (True, 1, 1)
        cur.execute("INSERT INTO exceptions(kind,severity,detail) VALUES ('PRE_MAKE_ADJUST','info','{}')")
        cur.execute('SAVEPOINT refused')
        with pytest.raises(psycopg2.errors.RaiseException, match='070 down refused'):
            cur.execute(down)
        cur.execute('ROLLBACK TO SAVEPOINT refused')
        cur.execute("SET LOCAL factory_ledger.confirm_exceptions_export = 'yes'")
        cur.execute(down)
        assert state(cur) == (False, 0, 0)
        cur.execute('SAVEPOINT narrowed')
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("INSERT INTO exceptions(kind,severity,detail) VALUES ('PRE_MAKE_ADJUST','info','{}')")
        cur.execute('ROLLBACK TO SAVEPOINT narrowed')
        cur.execute(up)
        assert state(cur) == (True, 1, 1)
        conn.rollback()


# ── Codex re-check of PR #94 (2026-10-09): P1 receipt evidence is consumed, P2 finite counts ──

def second_short_make(client_, items_, key):
    """A second, different make (2 batches) on the already-short ingredient lot: short 20 lb."""
    prepared = prepare(client_, 'make', body('make', items_) | {'batches': 2}, key)
    response = commit(client_, prepared, key, acknowledged_warnings=[w['code'] for w in prepared['warnings'] if w.get('requires_ack')])
    assert response.status_code == 200, response.text
    return response.json()


def claims_by_receipt(cur, ticket_id):
    cur.execute('SELECT COALESCE(SUM(claimed_lb), 0) AS used, count(*) AS n FROM shortage_evidence_claims WHERE evidence_ticket_id=%s', (ticket_id,))
    row = cur.fetchone()
    return float(row['used']), row['n']


def test_one_receipt_cannot_close_more_shortage_than_it_delivered(client, db_cursor, items, actors):
    key = actors['floor']['key']
    first_id, _ = open_shortage(client, db_cursor, items, actors)                        # short 6 (lot −6)
    second_id = second_short_make(client, items, key)['shortages'][0]['exception_id']    # short 20 (lot −26)
    twenty = receive_onto(client, db_cursor, items, key, cases=4, case_size_lb=5)        # 20 lb → lot −6
    # 6 of 20 close the first shortage.
    first = resolve(client, first_id, key, resolution_kind='missing_movement', receipt_number=twenty['receipt_number'], note='late receive')
    assert first.status_code == 200 and first.json()['status'] == 'resolved', first.text
    assert claims_by_receipt(db_cursor, twenty['ticket_id']) == (6.0, 1)
    # The remaining 14 of 20 cover the second shortage only in part: it stays open for 6.
    second = resolve(client, second_id, key, resolution_kind='missing_movement', receipt_number=twenty['receipt_number'], note='same receive')
    assert second.status_code == 200, second.text
    assert second.json()['status'] == 'open' and second.json()['shortage_flag']['status'] == 'open'
    assert second.json()['evidence'] == {'claimed_lb': 14.0, 'remaining_lb': 6.0, 'claims': second.json()['evidence']['claims']}
    assert claims_by_receipt(db_cursor, twenty['ticket_id']) == (20.0, 2)
    # The 20 lb receipt is spent: it cannot be used again, and stock still cannot be added.
    spent = resolve(client, second_id, key, resolution_kind='missing_movement', receipt_number=twenty['receipt_number'], note='again')
    error(spent, 409, 'RECEIPT_NOT_MATCHING')
    assert spent.json()['detail']['receipt_used_lb'] == 20.0 and spent.json()['detail']['receipt_added_lb'] == 20.0
    error(client.post('/adjust/prepare', json=adjust(items, 1, 'missing_receipt'), headers=headers(key)), 409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD')
    assert exceptions_for(db_cursor, id=second_id)[0]['status'] == 'open'
    # A further 6 lb receive closes the remainder; the ledger of claims adds up to 26 = 20 + 6.
    six = receive_onto(client, db_cursor, items, key, cases=3, case_size_lb=2)
    closed = resolve(client, second_id, key, resolution_kind='missing_movement', receipt_number=six['receipt_number'], note='second late receive')
    assert closed.status_code == 200 and closed.json()['status'] == 'resolved', closed.text
    assert closed.json()['evidence']['claimed_lb'] == 20.0 and closed.json()['evidence']['remaining_lb'] == 0.0
    assert claims_by_receipt(db_cursor, six['ticket_id']) == (6.0, 1)
    db_cursor.execute('SELECT COALESCE(SUM(claimed_lb), 0) AS total FROM shortage_evidence_claims WHERE exception_id IN (%s, %s)', (first_id, second_id))
    assert float(db_cursor.fetchone()['total']) == 26.0
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == 0.0
    assert prepare(client, 'adjust', adjust(items, 1, 'missing_receipt'), key)['can_commit']


def test_concurrent_resolutions_cannot_spend_the_same_receipt_twice(isolated_database, monkeypatch):
    keys, items_, ids = race_actors(isolated_database)
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        set_balance(cur, items_, 'ingredient', 4.0)
        supplier_id = ensure_supplier(cur, items_)
    race = Race(isolated_database)
    monkeypatch.setattr(main, 'get_db_connection', race.connection)
    main._reset_actor_cache()
    with TestClient(main.app) as http:
        floor = keys['floor']
        first = commit(http, prepare(http, 'make', body('make', items_), floor), floor).json()
        shortages = [first['shortages'][0]['exception_id'], second_short_make(http, items_, floor)['shortages'][0]['exception_id']]
        twenty = receive_onto(http, None, items_, floor, cases=4, case_size_lb=5, supplier_id=supplier_id)
        race.on = True
        calls = [lambda i=i: resolve(http, i, floor, resolution_kind='missing_movement', receipt_number=twenty['receipt_number'], note='late')
                 for i in shortages]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [f.result(timeout=30) for f in [executor.submit(call) for call in calls]]
        race.on = False
        assert all(r.status_code in (200, 409) for r in results), [r.text for r in results]
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert claims_by_receipt(cur, twenty['ticket_id'])[0] == 20.0            # never 26
            cur.execute('SELECT status FROM exceptions WHERE id=ANY(%s) ORDER BY id', (shortages,))
            statuses = [r['status'] for r in cur.fetchall()]
            assert statuses.count('resolved') == 1 and statuses.count('open') == 1, statuses


@pytest.mark.parametrize('raw', ['NaN', 'Infinity', '-Infinity', '1e400', '1000000000', '-1', '"two"'])
def test_counted_lb_must_be_a_finite_sane_number(client, db_cursor, items, actors, raw):
    key = actors['floor']['key']
    exception_id, _ = open_shortage(client, db_cursor, items, actors)
    response = client.post(f'/exceptions/{exception_id}/resolve', headers={**headers(key), 'Content-Type': 'application/json'},
                           content=f'{{"resolution_kind": "counted", "counted_lb": {raw}, "note": "counted"}}')
    assert response.status_code == 422, response.text
    assert exceptions_for(db_cursor, id=exception_id)[0]['status'] == 'open'
    assert main.lot_on_hand(db_cursor, items['ingredient']['lot_id']) == -6.0
    db_cursor.execute("SELECT count(*) AS n FROM transactions WHERE type='adjust' AND notes LIKE %s", (f'%Shortage #{exception_id}%',))
    assert db_cursor.fetchone()['n'] == 0


def test_migration_071_rerunnable_down_and_up(isolated_database):
    up = (ROOT / 'migrations/071_shortage_evidence_claims.sql').read_text()
    down = (ROOT / 'migrations/down/071_shortage_evidence_claims_down.sql').read_text()

    def state(cur):
        cur.execute("SELECT to_regclass('shortage_evidence_claims') IS NOT NULL")
        table = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname IN ('shortage_evidence_claims_receipt_idx','shortage_evidence_claims_exception_idx')")
        indexes = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM migration_markers WHERE name='071_shortage_evidence_claims'")
        return table, indexes, cur.fetchone()[0]

    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL search_path TO public')
        assert state(cur) == (True, 2, 1)      # fixture applied it
        cur.execute(up)                        # rerun: no-op
        assert state(cur) == (True, 2, 1)
        cur.execute('SELECT count(*) FROM shortage_evidence_claims')
        if cur.fetchone()[0]:                  # claims from the tests above refuse the down until the export is confirmed
            cur.execute('SAVEPOINT refused')
            with pytest.raises(psycopg2.errors.RaiseException, match='071 down refused'):
                cur.execute(down)
            cur.execute('ROLLBACK TO SAVEPOINT refused')
        cur.execute("SET LOCAL factory_ledger.confirm_exceptions_export = 'yes'")
        cur.execute(down)
        assert state(cur) == (False, 0, 0)
        cur.execute(up)
        assert state(cur) == (True, 2, 1)
        conn.rollback()

"""A1 part 1: receive lifecycle on real PostgreSQL, including real HTTP races."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
import pytest

import main
import write_tickets as tickets
from scripts.expire_tickets import expire
from tests.test_actor_attribution import client, actors  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.db


def seed(cur):
    suffix = uuid4().hex[:12].upper()
    cur.execute("INSERT INTO products(name,odoo_code,type,uom) VALUES (%s,%s,'ingredient','lb') RETURNING id",
                ('Ticket test ' + suffix, 'WT-' + suffix))
    product = cur.fetchone()['id']
    cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id', ('Ticket supplier ' + suffix,))
    return {'product_id': product, 'supplier_id': cur.fetchone()['id'], 'cases': 2,
            'case_size_lb': 25.0, 'bol_reference': 'WT-' + suffix, 'lot_code': 'WT-' + suffix,
            'supplier_lot_code': 'SUP-' + suffix, 'occurred_at': main.get_plant_now().isoformat()}


@pytest.fixture
def payload(db_cursor):
    return seed(db_cursor)


def headers(key=None):
    return {'X-API-Key': key or main.API_KEY}


def prepare(client, payload, key=None):
    response = client.post('/receive/prepare', json=payload, headers=headers(key))
    assert response.status_code == 200, response.text
    return response.json()


def commit(client, prepared, key=None, **updates):
    body = {'payload_hash': prepared['payload_hash'], **updates}
    return client.post('/tickets/' + prepared['ticket'] + '/commit', json=body, headers=headers(key))


def ticket_row(cur, prepared):
    cur.execute('SELECT * FROM write_tickets WHERE id=%s', (prepared['ticket_id'],))
    return cur.fetchone()


def posted_count(cur, prepared):
    cur.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (prepared['ticket_id'],))
    return cur.fetchone()['n']


def error(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()['detail']['error_code'] == code


@pytest.mark.parametrize('source,minutes', [('api', 10), ('mcp', 10), ('fl_assistant', 10), ('dashboard', 30)])
def test_prepare_expiry_normalized_draft_and_no_post(client, db_cursor, payload, source, minutes):
    prepared = prepare(client, payload | {'client_source': source})
    assert prepared['can_commit'] is True and prepared['blockers'] == []
    assert prepared['ticket'].startswith('wt_') and len(prepared['ticket']) == 46
    row = ticket_row(db_cursor, prepared)
    assert row['expires_at'] - row['prepared_at'] == timedelta(minutes=minutes)
    assert row['ticket_hash'] == sha256(prepared['ticket'].encode()).hexdigest()
    assert prepared['ticket'] not in str(row)
    assert tickets.canonical_hash(row['payload']) == prepared['payload_hash']
    assert row['payload']['product_id'] == payload['product_id']
    assert 'product_name' not in row['payload'] and 'shipper_name' not in row['payload']
    assert row['draft']['product_id'] == payload['product_id']
    assert row['draft']['happened_at'] == row['payload']['occurred_at']
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('names', [ {'product_name': 'Test'}, {'supplier_name': 'Test'}, {'shipper_name': 'Test'}])
def test_names_rejected(client, payload, names):
    error(client.post('/receive/prepare', json=payload | names, headers=headers()), 422, 'IDS_REQUIRED')


@pytest.mark.parametrize('changes', [{'product_id': '42'}, {'product_id': True}, {'cases': -1},
                                     {'case_size_lb': 0}, {'client_source': 'invented'}, {'mode': 'commit'}])
def test_input_schema_rejects_invalid_or_unposted_fields(client, payload, changes):
    response = client.post('/receive/prepare', json=payload | changes, headers=headers())
    assert response.status_code == 422


def test_commit_single_use_replay_and_receipt_lookup(client, db_cursor, payload):
    prepared = prepare(client, payload)
    first = commit(client, prepared)
    assert first.status_code == 200, first.text
    receipt = first.json()
    assert receipt['replayed'] is False and receipt['state_changed'] is False
    assert receipt['receipt_number'].startswith('RCV-')
    assert posted_count(db_cursor, prepared) == 1
    replay = commit(client, prepared).json()
    assert replay == receipt | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1
    row = ticket_row(db_cursor, prepared)
    assert row['response'] == receipt and row['status'] == 'committed'
    db_cursor.execute('SELECT receipt_number, ticket_id FROM transactions WHERE id=%s', (receipt['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'receipt_number': receipt['receipt_number'], 'ticket_id': row['id']}
    detail = client.get('/receipts/' + receipt['receipt_number'], headers=headers()).json()
    assert detail['response'] == receipt
    assert detail['transactions'][0]['lines'][0]['quantity_lb'] == 50
    assert detail['lots'][0]['id'] == receipt['lot_id']
    reverse = client.get('/receipts/by-transaction/' + str(receipt['transaction_id']), headers=headers())
    assert reverse.json() == detail
    # Replay does not re-check expiry, warnings or current product state.
    db_cursor.execute('UPDATE products SET active=false WHERE id=%s', (payload['product_id'],))
    assert commit(client, prepared).json() == replay


@pytest.mark.parametrize('url', ['/receipts/NO-SUCH-RECEIPT', '/receipts/by-transaction/2147483647'])
def test_receipt_not_recorded_is_json_404(client, url):
    error(client.get(url, headers=headers()), 404, 'RECEIPT_NOT_FOUND')


def test_legacy_transaction_has_no_ticket_receipt(client, db_cursor, payload):
    txn = direct_receive(client, db_cursor, payload)
    error(client.get('/receipts/by-transaction/' + str(txn['transaction_id']), headers=headers()),
          404, 'RECEIPT_NOT_FOUND')


def test_hash_tampering_and_commit_body_cannot_replace_payload(client, db_cursor, payload):
    prepared = prepare(client, payload)
    error(commit(client, prepared, payload_hash='0'*64), 409, 'TICKET_PAYLOAD_MISMATCH')
    assert commit(client, prepared, cases=900).status_code == 422
    assert commit(client, prepared, client_source='dashboard').status_code == 422
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    db_cursor.execute("UPDATE write_tickets SET payload=jsonb_set(payload,'{cases}','900') WHERE id=%s",
                      (prepared['ticket_id'],))
    error(commit(client, prepared), 409, 'TICKET_PAYLOAD_MISMATCH')
    assert posted_count(db_cursor, prepared) == 0


def test_wrong_identity_including_legacy_key_kinds(client, db_cursor, payload, actors):
    prepared = prepare(client, payload, actors['floor']['key'])
    error(commit(client, prepared, actors['office']['key']), 403, 'TICKET_WRONG_USER')
    error(commit(client, prepared), 403, 'TICKET_WRONG_USER')
    assert commit(client, prepared, actors['floor']['key']).status_code == 200
    shared = prepare(client, payload)
    error(commit(client, shared, main.DASHBOARD_API_KEY), 403, 'TICKET_WRONG_USER')
    assert ticket_row(db_cursor, prepared)['actor_id'] == actors['floor']['id']


@pytest.mark.parametrize('source', ['mcp', 'fl_assistant', 'dashboard', 'api'])
def test_expiry_persists_and_blocks_post(client, db_cursor, payload, source):
    prepared = prepare(client, payload | {'client_source': source})
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s",
                      (prepared['ticket_id'],))
    error(commit(client, prepared), 409, 'TICKET_EXPIRED')
    assert ticket_row(db_cursor, prepared)['status'] == 'expired'
    error(commit(client, prepared), 409, 'TICKET_NOT_COMMITTABLE')
    assert posted_count(db_cursor, prepared) == 0


def test_superseded_and_unknown_tickets(client, db_cursor, payload):
    first = prepare(client, payload)
    second = prepare(client, payload)
    assert first['payload_hash'] == second['payload_hash']
    assert ticket_row(db_cursor, first)['status'] == 'superseded'
    error(commit(client, first), 409, 'TICKET_NOT_COMMITTABLE')
    error(commit(client, first | {'ticket': 'wt_missing'}), 404, 'TICKET_NOT_FOUND')


def test_future_date_is_blocked_at_prepare_and_rejected_at_commit(client, db_cursor, payload):
    prepared = prepare(client, payload | {'occurred_at': (main.get_plant_now()+timedelta(hours=2)).isoformat()})
    assert prepared['can_commit'] is False
    assert prepared['blockers'][0]['code'] == 'OCCURRED_AT_IN_FUTURE'
    response = commit(client, prepared)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'] == prepared['blockers']
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'


def test_revalidation_failure_rolls_back_and_records_rejection(client, db_cursor, payload):
    prepared = prepare(client, payload)
    db_cursor.execute('UPDATE products SET active=false WHERE id=%s', (payload['product_id'],))
    response = commit(client, prepared)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'PRODUCT_NOT_FOUND'
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'


def direct_receive(client, cur, payload, **changes):
    cur.execute('SELECT name FROM products WHERE id=%s', (payload['product_id'],))
    product = cur.fetchone()['name']
    cur.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    shipper = cur.fetchone()['name']
    legacy = {k: v for k, v in payload.items() if k not in ('product_id', 'supplier_id', 'client_source')}
    response = client.post('/receive', json=legacy | {'product_name': product, 'shipper_name': shipper,
                           'mode': 'commit'} | changes, headers=headers())
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize('shipper_code_override', [None, 'RACE'])
@pytest.mark.parametrize('winner', ['legacy', 'ticket'])
def test_generated_lot_taken_after_prepare_requires_fresh_ticket(
        client, db_cursor, payload, shipper_code_override, winner):
    payload = payload | {'lot_code': None, 'shipper_code_override': shipper_code_override}
    first = prepare(client, payload)
    reserved_code = first['draft']['lot_code']
    assert first['can_commit'] is True
    assert not first['draft'].get('lot_exists')

    # Another delivery takes the displayed code before the first ticket commits.
    competing_payload = payload | {'bol_reference': payload['bol_reference'] + '-OTHER'}
    if winner == 'ticket':
        second = prepare(client, competing_payload)
        assert second['draft']['lot_code'] == reserved_code
        response = commit(client, second)
        assert response.status_code == 200, response.text
        competing = response.json()
    else:
        competing = direct_receive(client, db_cursor, competing_payload)
    assert competing['lot_code'] == reserved_code
    db_cursor.execute('SELECT prefix,business_date,next FROM receipt_counters ORDER BY prefix,business_date')
    counters_before = db_cursor.fetchall()

    response = commit(client, first)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'LOT_CODE_TAKEN'
    row = ticket_row(db_cursor, first)
    assert row['status'] == 'rejected' and 'LOT_CODE_TAKEN' in row['reject_reason']
    assert row['receipt_number'] is None
    assert posted_count(db_cursor, first) == 0
    assert main.lot_on_hand(db_cursor, competing['lot_id']) == 50
    db_cursor.execute('SELECT prefix,business_date,next FROM receipt_counters ORDER BY prefix,business_date')
    assert db_cursor.fetchall() == counters_before
    error(commit(client, first), 409, 'TICKET_NOT_COMMITTABLE')

    fresh = prepare(client, payload)
    assert fresh['can_commit'] is True
    assert fresh['draft']['lot_code'] != reserved_code
    assert fresh['warnings'][0]['code'] == 'POSSIBLE_DUPLICATE'
    response = commit(client, fresh, acknowledged_warnings=['POSSIBLE_DUPLICATE'])
    assert response.status_code == 200, response.text
    assert response.json()['lot_code'] == fresh['draft']['lot_code']
    assert response.json()['lot_id'] != competing['lot_id']
    assert posted_count(db_cursor, fresh) == 1


def test_state_changed_but_valid_and_duplicate_acknowledgement(client, db_cursor, payload):
    direct = direct_receive(client, db_cursor, payload)
    first = prepare(client, payload)
    assert first['draft']['lot_exists'] is True
    direct_receive(client, db_cursor, payload)
    # The draft explicitly allows adding to this existing lot; only its stock changed.
    response = commit(client, first, acknowledged_warnings=['POSSIBLE_DUPLICATE'])
    assert response.status_code == 200, response.text
    assert response.json()['state_changed'] is True
    duplicate = prepare(client, payload)
    warning = duplicate['warnings'][0]
    assert warning['code'] == 'POSSIBLE_DUPLICATE' and warning['requires_ack'] is True
    assert warning['refs']['receipt_number'] == response.json()['receipt_number']
    assert warning['message_es']
    error(commit(client, duplicate), 409, 'WARNING_NOT_ACKNOWLEDGED')
    assert commit(client, duplicate, acknowledged_warnings=['POSSIBLE_DUPLICATE']).status_code == 200
    assert ticket_row(db_cursor, duplicate)['acknowledged'] == ['POSSIBLE_DUPLICATE']
    assert direct['transaction_id'] != response.json()['transaction_id']


def test_possible_duplicate_from_any_actor_uses_posted_effective_lines(client, db_cursor, payload, actors):
    first = prepare(client, payload, actors['floor']['key'])
    original = commit(client, first, actors['floor']['key']).json()
    duplicate = prepare(client, payload, actors['office']['key'])
    assert duplicate['warnings'][0]['refs']['operator_id'] == actors['floor']['name']
    response = client.post('/void/' + str(original['transaction_id']), json={'reason': 'test'}, headers=headers())
    assert response.status_code == 200
    assert prepare(client, payload)['warnings'] == []
    detail = client.get('/receipts/' + original['receipt_number'], headers=headers()).json()
    assert detail['transactions'][0]['effective_status'] == 'voided'


def test_duplicate_outside_window(client, db_cursor, payload):
    # Entry time is DB-owned. Use a SELECT-only view shadow to represent history
    # beyond the window, without disabling append-only or timestamp triggers.
    direct_receive(client, db_cursor, payload)
    db_cursor.execute('''CREATE TEMP VIEW ledger_current_transactions AS
        SELECT id,type,effective_status,operator_id,created_at-interval '25 hours' AS created_at
        FROM public.ledger_current_transactions''')
    assert prepare(client, payload)['warnings'] == []


def test_expected_receipt_closed_between_prepare_and_commit(client, db_cursor, payload):
    db_cursor.execute('''INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,status)
                         VALUES (%s,%s,50,'open') RETURNING id''',
                      (payload['product_id'], payload['supplier_id']))
    er_id = db_cursor.fetchone()['id']
    prepared = prepare(client, payload)
    assert prepared['draft']['expected_receipt_match']['id'] == er_id
    direct_receive(client, db_cursor, payload)
    error(commit(client, prepared), 409, 'TICKET_STALE')
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'
    assert posted_count(db_cursor, prepared) == 0


def test_mid_post_failure_rolls_back_counter_lot_ledger_and_ticket(client, db_cursor, payload, monkeypatch):
    prepared = prepare(client, payload)
    core = main._receive_commit_core
    def crash(*args, **kwargs):
        core(*args, **kwargs)
        raise RuntimeError('simulated lost process before receipt save')
    monkeypatch.setattr(main, '_receive_commit_core', crash)
    with pytest.raises(RuntimeError, match='simulated'):
        commit(client, prepared)
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    db_cursor.execute('SELECT count(*) AS n FROM lots WHERE product_id=%s', (payload['product_id'],))
    assert db_cursor.fetchone()['n'] == 0
    monkeypatch.setattr(main, '_receive_commit_core', core)
    assert commit(client, prepared).status_code == 200


@pytest.mark.parametrize('action,prefix', list(tickets.PREFIXES.items()))
def test_receipt_counter_prefix_increments_resets_and_survives_beyond_999(db_cursor, action, prefix):
    day = main.get_plant_now().date() - timedelta(days=366)
    first = tickets.allocate_receipt(db_cursor, action, day)
    assert first == f'{prefix}-{day:%y%m%d}-001'
    assert tickets.allocate_receipt(db_cursor, action, day).endswith('-002')
    assert tickets.allocate_receipt(db_cursor, action, day+timedelta(days=1)).endswith('-001')
    db_cursor.execute('UPDATE receipt_counters SET next=1000 WHERE prefix=%s AND business_date=%s', (prefix, day))
    assert tickets.allocate_receipt(db_cursor, action, day).endswith('-1000')


def test_backdated_plant_day_receipt_lists_on_both_days_and_filters(client, db_cursor, payload, actors):
    yesterday = main.get_plant_now()-timedelta(days=1)
    payload.update(occurred_at=yesterday.isoformat(), client_source='dashboard')
    prepared = prepare(client, payload, actors['floor']['key'])
    receipt = commit(client, prepared, actors['floor']['key']).json()
    assert receipt['receipt_number'].startswith(f'RCV-{yesterday:%y%m%d}-')
    for day in (yesterday.date(), main.get_plant_now().date()):
        response = client.get('/receipts', params={'date': day.isoformat(), 'actor': actors['floor']['name'],
            'action': 'receive', 'status': 'committed', 'client_source': 'dashboard'}, headers=headers())
        assert response.status_code == 200, response.text
        assert len(response.json()['receipts']) == 1
        assert response.json()['receipts'][0]['late_entry'] is True
    assert client.get('/receipts?actor=nobody', headers=headers()).json()['receipts'] == []


@pytest.mark.parametrize('which', ['master', 'dashboard', 'floor', 'office'])
def test_route_auth_matrix_and_attribution(client, db_cursor, payload, actors, which):
    key = {'master': main.API_KEY, 'dashboard': main.DASHBOARD_API_KEY}.get(which)
    key = key or actors[which]['key']
    prepared = prepare(client, payload, key)
    response = commit(client, prepared, key)
    assert response.status_code == 200, response.text
    receipt = response.json()
    for path in ('/receipts', '/receipts/' + receipt['receipt_number'],
                 '/receipts/by-transaction/' + str(receipt['transaction_id'])):
        assert client.get(path, headers=headers(key)).status_code == 200
    db_cursor.execute('SELECT operator_id FROM transactions WHERE id=%s', (receipt['transaction_id'],))
    assert db_cursor.fetchone()['operator_id'] == (actors[which]['name'] if which in actors else 'legacy-shared-key')


@pytest.mark.parametrize('method,path', sorted(tickets.ACTOR_ROUTES))
def test_every_new_route_rejects_unknown_key(client, method, path):
    path = path.replace('{ticket}', 'wt_unknown').replace('{transaction_id}', '1').replace('{receipt_number}', 'NOPE')
    response = client.request(method, path, json={}, headers=headers('unknown-key'))
    assert response.status_code == 403


def test_committed_evidence_immutable_and_sweep_idempotent(client, db_cursor, payload):
    prepared = prepare(client, payload)
    assert commit(client, prepared).status_code == 200
    for statement in ("UPDATE write_tickets SET status='prepared' WHERE id=%s", 'DELETE FROM write_tickets WHERE id=%s'):
        db_cursor.execute('SAVEPOINT immutable')
        with pytest.raises(psycopg2.IntegrityError):
            db_cursor.execute(statement, (prepared['ticket_id'],))
        db_cursor.execute('ROLLBACK TO SAVEPOINT immutable')
    another = prepare(client, payload)
    db_cursor.execute("UPDATE write_tickets SET expires_at=now()-interval '1 minute' WHERE id=%s", (another['ticket_id'],))
    assert expire(db_cursor) == 1
    assert expire(db_cursor) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'committed'


@pytest.fixture(scope='module')
def isolated_database(_db_connection):
    """Dedicated local database for committed races and reversible DDL."""
    parts = urlsplit(main.DATABASE_URL)
    assert parts.hostname in ('localhost', '127.0.0.1', '::1')
    name = 'a1_race_' + uuid4().hex
    url = urlunsplit(parts._replace(path='/' + name))
    admin = psycopg2.connect(main.DATABASE_URL)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute('CREATE DATABASE ' + name)
    try:
        # Schema dump has only psql restrict/include meta-commands; execute
        # its SQL on the guarded local connection, then the pending migration.
        schema = '\n'.join(line for line in (ROOT/'tests/schema/schema.sql').read_text().splitlines()
                           if not line.startswith('\\'))
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute('CREATE EXTENSION pg_trgm')
            cur.execute(schema)
            cur.execute('SET LOCAL search_path TO public')
            cur.execute((ROOT/'migrations/058_write_tickets.sql').read_text())
            cur.execute((ROOT/'migrations/064_unidentified_lots.sql').read_text())
        yield url
    finally:
        with admin.cursor() as cur:
            cur.execute('DROP DATABASE ' + name + ' WITH (FORCE)')
        admin.close()


def test_migration_up_down_up_and_marker_stability(isolated_database):
    up = (ROOT/'migrations/058_write_tickets.sql').read_text()
    down = (ROOT/'migrations/down/058_write_tickets_down.sql').read_text()
    # 061 (exceptions/shortage_flags) holds FKs to write_tickets, so it must
    # come off before 058 can and go back on afterwards.
    up_061 = (ROOT/'migrations/061_exceptions_tables.sql').read_text()
    down_061 = (ROOT/'migrations/down/061_exceptions_tables_down.sql').read_text()
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SELECT oid FROM pg_class WHERE relname IN (%s,%s) ORDER BY oid',
                    ('ledger_current_transactions', 'ledger_current_transaction_lines'))
        views = cur.fetchall()
        cur.execute('SET LOCAL search_path TO public')
        cur.execute(down_061)
        cur.execute(down)
        cur.execute("SELECT to_regclass('public.write_tickets')")
        assert cur.fetchone()[0] is None
        cur.execute(up)
        cur.execute("SELECT applied_at FROM migration_markers WHERE name='058_write_tickets'")
        applied = cur.fetchone()[0]
        cur.execute(up)
        cur.execute("SELECT applied_at FROM migration_markers WHERE name='058_write_tickets'")
        assert cur.fetchone()[0] == applied
        cur.execute("SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE relname IN ('write_tickets','receipt_counters')")
        assert cur.fetchall() == [(True, False), (True, False)]
        cur.execute(up_061)
        cur.execute('SELECT oid FROM pg_class WHERE relname IN (%s,%s) ORDER BY oid',
                    ('ledger_current_transactions', 'ledger_current_transaction_lines'))
        assert cur.fetchall() == views


def test_concurrent_double_commit_one_transaction_and_same_receipt(isolated_database, monkeypatch):
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        payload = seed(cur)
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
    # Route I/O uses the dedicated database; startup uses the existing test URL.
    with TestClient(main.app) as http:
        prepared = prepare(http, payload)
        race = True
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = [f.result(timeout=20) for f in
                         [executor.submit(commit, http, prepared) for _ in range(2)]]
        race = False
        assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
        results = [r.json() for r in responses]
        assert results[0]['receipt_number'] == results[1]['receipt_number']
        assert sorted(r['replayed'] for r in results) == [False, True]
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, prepared) == 1
            cur.execute('SELECT count(*) AS n FROM transaction_lines WHERE transaction_id=%s', (results[0]['transaction_id'],))
            assert cur.fetchone()['n'] == 1
        # Populated down migration refuses to discard replay/audit evidence.
        with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg2.Error, match='rollback refused'):
                cur.execute((ROOT/'migrations/down/058_write_tickets_down.sql').read_text())
            conn.rollback()



def test_merged_and_code_twin_lots_are_blockers_without_posting(client, db_cursor, payload):
    direct_receive(client, db_cursor, payload)
    db_cursor.execute("UPDATE lots SET status='merged' WHERE product_id=%s", (payload['product_id'],))
    merged = prepare(client, payload)
    assert merged['can_commit'] is False
    assert merged['blockers'][0]['code'] == 'LOT_MERGED'
    error(commit(client, merged, acknowledged_warnings=['POSSIBLE_DUPLICATE']), 409, 'TICKET_STALE')
    db_cursor.execute("UPDATE lots SET status='active' WHERE product_id=%s", (payload['product_id'],))
    twin = prepare(client, payload | {'lot_code': payload['lot_code'] + ' LOT'})
    assert twin['can_commit'] is False
    assert 'normalizes' in twin['blockers'][0]['message']
    assert posted_count(db_cursor, twin) == 0


def test_missing_supplier_and_unknown_product_still_issue_blocked_tickets(client, db_cursor, payload):
    for changes, code in (({'supplier_id': 2147483647}, 'SUPPLIER_NOT_FOUND'),
                          ({'product_id': 2147483647}, 'PRODUCT_NOT_FOUND')):
        prepared = prepare(client, payload | changes)
        assert prepared['can_commit'] is False
        assert prepared['blockers'][0]['code'] == code
        error(commit(client, prepared), 409, 'TICKET_STALE')
        assert posted_count(db_cursor, prepared) == 0


def test_rls_hides_tickets_and_counters_from_nonowner(client, db_cursor, payload):
    prepared = prepare(client, payload)
    assert commit(client, prepared).status_code == 200
    role = 'wt_rls_' + uuid4().hex
    db_cursor.execute('CREATE ROLE ' + role + ' NOLOGIN')
    db_cursor.execute('GRANT USAGE ON SCHEMA public TO ' + role)
    db_cursor.execute('GRANT SELECT,INSERT ON write_tickets,receipt_counters TO ' + role)
    db_cursor.execute('SAVEPOINT rls_test')
    db_cursor.execute('SET LOCAL ROLE ' + role)
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute('SELECT count(*) AS n FROM receipt_counters')
    assert db_cursor.fetchone()['n'] == 0
    with pytest.raises(psycopg2.errors.InsufficientPrivilege):
        db_cursor.execute("INSERT INTO receipt_counters(prefix,business_date) VALUES ('RCV',CURRENT_DATE)")
    db_cursor.execute('ROLLBACK TO SAVEPOINT rls_test')


def test_missing_supplier_defaults_and_happened_alias_freeze_the_draft(client, db_cursor, payload):
    body = {k: v for k, v in payload.items() if k not in ('supplier_id', 'occurred_at', 'lot_code')}
    body['happened_at'] = payload['occurred_at']
    prepared = prepare(client, body)
    assert prepared['can_commit'] is True
    receipt = commit(client, prepared).json()
    assert receipt['lot_code'] == prepared['draft']['lot_code']
    assert ticket_row(db_cursor, prepared)['payload']['occurred_at'] == payload['occurred_at']


def test_successful_expected_receipt_stays_pinned_and_settles(client, db_cursor, payload):
    db_cursor.execute("INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,status) VALUES (%s,%s,50,'open') RETURNING id",
                      (payload['product_id'], payload['supplier_id']))
    er_id = db_cursor.fetchone()['id']
    prepared = prepare(client, payload)
    receipt = commit(client, prepared)
    assert receipt.status_code == 200, receipt.text
    db_cursor.execute('SELECT expected_receipt_id FROM transactions WHERE id=%s', (receipt.json()['transaction_id'],))
    assert db_cursor.fetchone()['expected_receipt_id'] == er_id
    db_cursor.execute('SELECT status FROM expected_receipts WHERE id=%s', (er_id,))
    assert db_cursor.fetchone()['status'] == 'closed'

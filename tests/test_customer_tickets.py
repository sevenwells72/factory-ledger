"""Customer metadata tickets reuse A7's real HTTP/DB lifecycle and receipts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
import re
from threading import Barrier, Event
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
import pytest

import customer_tickets
import main
import permissions
import write_tickets as tickets
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, commit, ticket_row, error, isolated_database  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / 'migrations/083_customer_tickets.sql'
DOWN = ROOT / 'migrations/down/083_customer_tickets_down.sql'


@pytest.fixture
def customer_schema(db_cursor):
    db_cursor.execute(UP.read_text())


def body(**updates):
    return {'name': 'New customer ' + uuid4().hex, **updates}


def prepare(client, actors, payload=None, role='office'):
    response = client.post('/customers/prepare', json=payload or body(), headers=headers(actors[role]['key']))
    assert response.status_code == 200, response.text
    return response.json()


def customer_count(cur):
    cur.execute('SELECT count(*) AS n FROM customers')
    return cur.fetchone()['n']


# Request validation and role policy have no database dependency.
def test_request_matches_direct_customer_fields_and_required_name():
    assert set(customer_tickets.CreateCustomerRequest.model_fields) - {'client_source'} == set(main.CustomerCreate.model_fields)
    assert customer_tickets.CreateCustomerRequest.model_fields['name'].is_required()
    assert customer_tickets.CreateCustomerRequest(name='  New \t Customer  ').name == 'New Customer'


@pytest.mark.parametrize('role,allowed', [(role, role in ('office', 'owner')) for role in permissions.ROLES])
def test_create_customer_role_policy(role, allowed):
    assert permissions.allowed('create_customer', role) is allowed


@pytest.mark.db
def test_prepare_dry_run_commit_replay_receipt_and_atomic_audit(client, db_cursor, actors, customer_schema):
    payload = body(name='  New   Customer ' + uuid4().hex, contact_name='Pat', email='pat@example.test',
                   phone='555-0100', address='1 Factory Way', notes='Call first', notes_es='Llame primero')
    count = customer_count(db_cursor)
    db_cursor.execute('SELECT count(*) AS n FROM actor_write_audit')
    audits = db_cursor.fetchone()['n']
    prepared = prepare(client, actors, payload)
    assert prepared['can_commit'] and prepared['action'] == 'create_customer'
    assert prepared['blockers'] == [] and prepared['warnings'] == []
    assert prepared['draft']['summary'].startswith('Create customer New Customer ')
    assert prepared['draft']['customer']['name'] == ' '.join(payload['name'].split())
    assert 'id' not in prepared['draft']['customer']
    assert customer_count(db_cursor) == count
    db_cursor.execute('SELECT count(*) AS n FROM actor_write_audit')
    assert db_cursor.fetchone()['n'] == audits
    row = ticket_row(db_cursor, prepared)
    assert tickets.canonical_hash(row['payload']) == prepared['payload_hash']
    assert row['ticket_hash'] == sha256(prepared['ticket'].encode()).hexdigest()
    assert prepared['ticket'] not in str(row)

    first = commit(client, prepared, actors['office']['key'])
    assert first.status_code == 200, first.text
    result = first.json()
    assert result['receipt_number'].startswith('CUS-') and result['success'] is True
    assert result['replayed'] is False and result['state_changed'] is False
    assert result['customer']['id'] == result['customer_id']
    assert result['customer']['ticket_id'] == prepared['ticket_id']
    assert result['entry_timing']['entered_by']['id'] == actors['office']['id']
    assert result['customer']['email'] == payload['email']
    assert customer_count(db_cursor) == count + 1
    replay = commit(client, prepared, actors['office']['key'])
    assert replay.status_code == 200 and replay.json() == result | {'replayed': True}
    assert customer_count(db_cursor) == count + 1
    row = ticket_row(db_cursor, prepared)
    assert row['response'] == result and row['status'] == 'committed'
    assert row['result_ref'] == {'customer_id': result['customer_id'], 'transaction_ids': [], 'lot_ids': []}
    db_cursor.execute('SELECT target_table,target_id,route FROM actor_write_audit WHERE actor_id=%s', (actors['office']['id'],))
    assert [dict(row) for row in db_cursor.fetchall()] == [
        {'target_table': 'customers', 'target_id': result['customer_id'], 'route': '/tickets/{ticket}/commit'}]
    receipt = client.get('/receipts/' + result['receipt_number'], headers=headers(actors['office']['key']))
    assert receipt.status_code == 200, receipt.text
    detail = receipt.json()
    assert detail['customer']['id'] == result['customer_id']
    assert detail['action'] == 'create_customer' and detail['response'] == result
    assert detail['transactions'] == [] and detail['lots'] == []
    listed = client.get('/receipts?action=create_customer', headers=headers(actors['office']['key'])).json()['receipts']
    assert [row['receipt_number'] for row in listed] == [result['receipt_number']]


@pytest.mark.db
@pytest.mark.parametrize('source,minutes', [('api', 10), ('mcp', 10), ('fl_assistant', 10), ('dashboard', 30)])
def test_ticket_sources_expire_with_standard_ttl(client, db_cursor, actors, customer_schema, source, minutes):
    prepared = prepare(client, actors, body(client_source=source))
    row = ticket_row(db_cursor, prepared)
    assert row['expires_at'] - row['prepared_at'] == timedelta(minutes=minutes)
    assert row['client_source'] == source


@pytest.mark.db
@pytest.mark.parametrize('role,allowed', [('owner', True), ('office', True), ('floor', False),
    ('retired', False), ('master', False), ('dashboard', False), ('unknown', False)])
def test_prepare_role_and_key_permissions(client, db_cursor, actors, customer_schema, role, allowed):
    key = {'master': main.API_KEY, 'dashboard': main.DASHBOARD_API_KEY, 'unknown': 'unknown-key'}.get(role)
    key = key or actors[role]['key']
    before = customer_count(db_cursor)
    response = client.post('/customers/prepare', json=body(), headers=headers(key))
    assert response.status_code == (200 if allowed else 403), response.text
    assert customer_count(db_cursor) == before
    if allowed:
        assert commit(client, response.json(), key).status_code == 200


@pytest.mark.db
@pytest.mark.parametrize('existing,new', [('Acme Foods', 'acme foods'), ('Acme Foods', ' ACME   Foods '),
    (' Acme   Foods ', 'acme foods'), ('Acme\tFoods', 'ACME FOODS'), ('\tAcme Foods\n', 'acme foods'), ('\u00a0Acme\u00a0Foods\u00a0', 'acme foods'),
    ('\u2003Acme\u2003Foods\u2003', 'acme foods')])
@pytest.mark.parametrize('active', [True, False])
def test_normalized_duplicate_offers_existing_customer(client, db_cursor, actors, customer_schema, existing, new, active):
    token = uuid4().hex
    existing = re.sub('Foods', 'Foods ' + token, existing)
    new = re.sub('foods', lambda match: match.group() + ' ' + token, new, flags=re.IGNORECASE)
    db_cursor.execute('INSERT INTO customers(name,active) VALUES (%s,%s) RETURNING id', (existing, active))
    customer_id = db_cursor.fetchone()['id']
    before = customer_count(db_cursor)
    prepared = prepare(client, actors, body(name=new))
    assert prepared['can_commit'] is False
    [blocker] = prepared['blockers']
    assert blocker['code'] == 'CUSTOMER_ALREADY_EXISTS'
    assert blocker['existing_customer'] == {'id': customer_id, 'name': existing, 'active': active}
    assert ('Use this customer' if active else 'reactivate this customer') in blocker['message']
    refused = commit(client, prepared, actors['office']['key'])
    error(refused, 409, 'TICKET_STALE')
    assert refused.json()['detail']['blockers'][0] == blocker
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'
    assert customer_count(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('existing,new', [
    ('Sunshine Bakery {token}', 'Sunshine Bakery {token}.'),
    ('Sunshine Bakery {token}.', 'Sunshine Bakery {token}'),
    ('Sunshine Bakery {token}', 'Sunshine Bakery {token},Inc.'),
    ('Sunshine Bakery {token},LLC', 'Sunshine Bakery {token}'),
    ('Sunshine, Bakery {token}', 'Sunshine Bakery {token}'),
    ('Sunshine-Bakery {token}', 'Sunshine Bakery {token}'),
    *[(f'Sunshine Bakery {{token}}{suffix}', 'Sunshine Bakery {token}')
      for suffix in (' Inc', ', Inc.', ',Inc.', ' (Inc.)', ' LLC', ',LLC', ' L.L.C.', ' Co.', ',Co.', ' Corp', ',Corp.', ' Ltd.', ',Ltd.')],
    *[('Sunshine Bakery {token}', f'Sunshine Bakery {{token}}{suffix}')
      for suffix in (' Inc', ', Inc.', ',Inc.', ' (Inc.)', ' LLC', ',LLC', ' L.L.C.', ' Co.', ',Co.', ' Corp', ',Corp.', ' Ltd.', ',Ltd.')],
    ('Sunshine Bakery {token}, Inc. LLC.', 'Sunshine Bakery {token}'),
])
def test_punctuation_and_company_suffix_duplicate_offers_existing_customer(
        client, db_cursor, actors, customer_schema, existing, new):
    token = uuid4().hex
    existing, new = existing.format(token=token), new.format(token=token)
    db_cursor.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (existing,))
    customer_id = db_cursor.fetchone()['id']
    before = customer_count(db_cursor)
    prepared = prepare(client, actors, body(name=new))
    assert prepared['can_commit'] is False
    [blocker] = prepared['blockers']
    assert blocker['code'] == 'CUSTOMER_ALREADY_EXISTS'
    assert blocker['existing_customer'] == {'id': customer_id, 'name': existing, 'active': True}
    assert existing in blocker['message'] and 'Use this customer' in blocker['message']
    response = commit(client, prepared, actors['office']['key'])
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0] == blocker
    assert customer_count(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('alias_source', ['customer_aliases', 'search_aliases'])
@pytest.mark.parametrize('active', [True, False])
@pytest.mark.parametrize('variant', ['', '.', ', Inc.', ',Inc.'])
def test_alias_duplicate_offers_canonical_customer(
        client, db_cursor, actors, customer_schema, alias_source, active, variant):
    token = uuid4().hex
    existing, alias = 'Sunshine Bakery ' + token, 'SS ' + token
    db_cursor.execute('INSERT INTO customers(name,active) VALUES (%s,%s) RETURNING id', (existing, active))
    customer_id = db_cursor.fetchone()['id']
    if alias_source == 'customer_aliases':
        db_cursor.execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)', (customer_id, alias))
    else:
        db_cursor.execute("INSERT INTO search_aliases(kind,customer_id,alias) VALUES ('customer',%s,%s)", (customer_id, alias))
    before = customer_count(db_cursor)
    prepared = prepare(client, actors, body(name=alias.lower() + variant))
    assert prepared['can_commit'] is False
    [blocker] = prepared['blockers']
    assert blocker['code'] == 'CUSTOMER_ALREADY_EXISTS'
    assert blocker['existing_customer'] == {'id': customer_id, 'name': existing, 'active': active}
    assert existing in blocker['message']
    assert ('Use this customer' if active else 'reactivate this customer') in blocker['message']
    response = commit(client, prepared, actors['office']['key'])
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0] == blocker
    assert customer_count(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('alias_source', ['customer_aliases', 'search_aliases'])
def test_alias_added_after_prepare_is_refused_with_canonical_customer(
        client, db_cursor, actors, customer_schema, alias_source):
    token = uuid4().hex
    existing, alias = 'Sunshine Bakery ' + token, 'SS ' + token
    db_cursor.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (existing,))
    customer_id = db_cursor.fetchone()['id']
    prepared = prepare(client, actors, body(name=alias + '.'))
    assert prepared['can_commit'] is True
    if alias_source == 'customer_aliases':
        db_cursor.execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)', (customer_id, alias))
    else:
        db_cursor.execute("INSERT INTO search_aliases(kind,customer_id,alias) VALUES ('customer',%s,%s)", (customer_id, alias))
    before = customer_count(db_cursor)
    response = commit(client, prepared, actors['office']['key'])
    error(response, 409, 'TICKET_STALE')
    [blocker] = response.json()['detail']['blockers']
    assert blocker['code'] == 'CUSTOMER_ALREADY_EXISTS'
    assert blocker['existing_customer'] == {'id': customer_id, 'name': existing, 'active': True}
    assert existing in blocker['message'] and 'Use this customer' in blocker['message']
    assert customer_count(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('new', ['Sunrise Bakery {token}', 'Sunshine Bakery {token} East', 'Sunshine Bakery {token} Co-op'])
def test_similar_but_different_customer_names_can_be_created(client, db_cursor, actors, customer_schema, new):
    token = uuid4().hex
    existing, new = 'Sunshine Bakery ' + token, new.format(token=token)
    db_cursor.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (existing,))
    existing_id = db_cursor.fetchone()['id']
    before = customer_count(db_cursor)
    prepared = prepare(client, actors, body(name=new))
    assert prepared['can_commit'] is True and prepared['blockers'] == []
    response = commit(client, prepared, actors['office']['key'])
    assert response.status_code == 200, response.text
    assert response.json()['name'] == new and response.json()['customer_id'] != existing_id
    assert customer_count(db_cursor) == before + 1


@pytest.mark.db
def test_direct_customer_appearing_after_prepare_is_refused_with_existing_choice(client, db_cursor, actors, customer_schema):
    payload = body()
    prepared = prepare(client, actors, payload)
    direct = client.post('/customers', json=payload | {'name': payload['name'].upper()}, headers=headers())
    assert direct.status_code == 200, direct.text
    response = commit(client, prepared, actors['office']['key'])
    error(response, 409, 'TICKET_STALE')
    [blocker] = response.json()['detail']['blockers']
    assert blocker['code'] == 'CUSTOMER_ALREADY_EXISTS'
    assert blocker['existing_customer']['id'] == direct.json()['customer_id']
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'


@pytest.mark.db
@pytest.mark.parametrize('change,code', [('role', 'ROLE_NOT_ALLOWED'), ('inactive', None), ('wrong_user', 'TICKET_WRONG_USER')])
def test_commit_rechecks_permissions_and_bound_person(client, db_cursor, actors, customer_schema, change, code):
    prepared = prepare(client, actors)
    key = actors['office']['key']
    if change == 'role':
        db_cursor.execute("UPDATE actors SET role='floor' WHERE id=%s", (actors['office']['id'],))
    elif change == 'inactive':
        db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (actors['office']['id'],))
        main._reset_actor_cache()
    else:
        key = actors['owner']['key']
    response = commit(client, prepared, key)
    assert response.status_code == 403, response.text
    if code:
        assert response.json()['detail']['error_code'] == code
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'


@pytest.mark.db
def test_payload_hash_supersession_and_expiry_are_shared_a7_guards(client, db_cursor, actors, customer_schema):
    payload = body()
    first = prepare(client, actors, payload)
    second = prepare(client, actors, payload)
    assert first['payload_hash'] == second['payload_hash']
    assert ticket_row(db_cursor, first)['status'] == 'superseded'
    error(commit(client, first, actors['office']['key']), 409, 'TICKET_NOT_COMMITTABLE')
    error(commit(client, second, actors['office']['key'], payload_hash='wrong'), 409, 'TICKET_PAYLOAD_MISMATCH')
    assert ticket_row(db_cursor, second)['status'] == 'prepared'
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 minute' WHERE id=%s", (second['ticket_id'],))
    error(commit(client, second, actors['office']['key']), 409, 'TICKET_EXPIRED')
    assert ticket_row(db_cursor, second)['status'] == 'expired'


@pytest.mark.db
@pytest.mark.parametrize('changes', [{'name': ''}, {'name': '  '}, {'name': None}, {'name': 123},
    {'by': 'Someone'}, {'created_by': 'Someone'}, {'active': True}, {'client_source': 'invented'}, {'mode': 'commit'}])
def test_invalid_request_never_issues_ticket(client, db_cursor, actors, customer_schema, changes):
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    before = db_cursor.fetchone()['n']
    response = client.post('/customers/prepare', json=body(**changes), headers=headers(actors['office']['key']))
    assert response.status_code == 422, response.text
    db_cursor.execute('SELECT count(*) AS n FROM write_tickets')
    assert db_cursor.fetchone()['n'] == before


@pytest.mark.db
def test_direct_bilingual_rule_and_direct_routes_are_preserved(client, db_cursor, actors, customer_schema):
    blocked = prepare(client, actors, body(notes_es='Solo español'))
    assert blocked['can_commit'] is False and 'English version required' in blocked['blockers'][0]['message']
    english = prepare(client, actors, body(notes='English is sufficient'))
    assert commit(client, english, actors['office']['key']).status_code == 200
    # Existing exact-string direct-route uniqueness stays unchanged by 083.
    name = body()['name']
    one = client.post('/customers', json={'name': name}, headers=headers())
    two = client.post('/customers', json={'name': name.upper()}, headers=headers())
    assert one.status_code == two.status_code == 200
    assert one.json()['customer_id'] != two.json()['customer_id']


@pytest.mark.db
def test_failed_audit_or_response_save_rolls_customer_back_and_allows_retry(client, db_cursor, actors, customer_schema, monkeypatch):
    prepared = prepare(client, actors)
    before = customer_count(db_cursor)
    original = main._record_actor_write
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('audit failure')
    monkeypatch.setattr(main, '_record_actor_write', crash)
    with pytest.raises(RuntimeError, match='audit failure'):
        commit(client, prepared, actors['office']['key'])
    assert customer_count(db_cursor) == before and ticket_row(db_cursor, prepared)['status'] == 'prepared'
    monkeypatch.setattr(main, '_record_actor_write', original)
    assert commit(client, prepared, actors['office']['key']).status_code == 200


@pytest.mark.db
@pytest.mark.parametrize('extra', [{'lot_confirmations': [{'ingredient_product_id': 1, 'lot_id': 1, 'method': 'last4', 'value': '1234'}]},
    {'attachment_ref': 'not-used'}, {'attachment_ids': [1]}])
def test_unused_inventory_evidence_refused(client, db_cursor, actors, customer_schema, extra):
    prepared = prepare(client, actors)
    assert commit(client, prepared, actors['office']['key'], **extra).status_code == 422
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'


@pytest.mark.db
@pytest.mark.parametrize('sessions_enabled', [False, True])
def test_session_person_can_record_and_continue_with_new_customer_order(client, db_cursor, actors, customer_schema, monkeypatch, sessions_enabled):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1' if sessions_enabled else '0')
    key = actors['office']['key']
    if sessions_enabled:
        response = client.post('/auth/session', json={'actor_key': key})
        assert response.status_code == 200, response.text
        key = response.json()['session_token']
    prepared = client.post('/customers/prepare', json=body(), headers=headers(key)).json()
    recorded = commit(client, prepared, key)
    assert recorded.status_code == 200, recorded.text
    db_cursor.execute("INSERT INTO products(name,type,case_size_lb,is_service) VALUES (%s,'finished',25,false) RETURNING id", ('Customer order product ' + uuid4().hex,))
    product_id = db_cursor.fetchone()['id']
    order = client.post('/sales/orders/prepare', json={'customer_id': recorded.json()['customer_id'],
        'customer_po': 'PO-' + uuid4().hex, 'lines': [{'product_id': product_id, 'quantity': 1, 'unit': 'cases'}]}, headers=headers(key))
    assert order.status_code == 200 and order.json()['can_commit'], order.text
    result = commit(client, order.json(), key)
    assert result.status_code == 200 and result.json()['receipt_number'].startswith('ORD-'), result.text


@pytest.mark.db
def test_migration_up_down_up_preserves_other_actions_and_refuses_existing_evidence(isolated_database):
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(UP.read_text())
        cur.execute(UP.read_text())
        cur.execute("SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint WHERE conname='write_tickets_action_check'")
        before = cur.fetchone()['definition']
        assert 'create_customer' in before and 'create_order' in before and 'ship_order' in before
        cur.execute(DOWN.read_text())
        cur.execute("SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint WHERE conname='write_tickets_action_check'")
        after = cur.fetchone()['definition']
        assert 'create_customer' not in after and 'create_order' in after and 'ship_order' in after
        cur.execute(UP.read_text())
        cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name='083_customer_tickets'")
        assert cur.fetchone()['n'] == 1
        cur.execute("""INSERT INTO write_tickets(ticket_hash,action,operator_id,key_kind,client_source,payload,payload_hash,state_hash,draft,expires_at)
                       VALUES (%s,'create_customer','test','legacy_ledger','api','{}',%s,%s,'{}',clock_timestamp()+interval '10 minutes')""",
                    (sha256(uuid4().bytes).hexdigest(), '0' * 64, '0' * 64))
        cur.execute('SAVEPOINT rollback_guard')
        with pytest.raises(psycopg2.Error, match='ticket evidence exists'):
            cur.execute(DOWN.read_text())
        cur.execute('ROLLBACK TO SAVEPOINT rollback_guard')
        # Roll back all DDL/evidence to leave the shared isolated fixture clean.
        conn.rollback()


@pytest.mark.db
@pytest.mark.parametrize('same_ticket', [True, False])
def test_concurrent_commits_create_one_customer_and_one_receipt(isolated_database, monkeypatch, same_ticket):
    @contextmanager
    def connection():
        conn = psycopg2.connect(isolated_database)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    monkeypatch.setattr(main, 'get_db_connection', connection)
    token = uuid4().hex
    key = 'customer-race-' + token
    with connection() as conn, conn.cursor() as cur:
        cur.execute(UP.read_text())
        cur.execute("INSERT INTO actors(name,role,key_hash) VALUES (%s,'office',%s) RETURNING id", ('Customer race ' + token, sha256(key.encode()).hexdigest()))
        actor_id = cur.fetchone()[0]
    main._reset_actor_cache()
    with TestClient(main.app) as api:
        first = api.post('/customers/prepare', json={'name': 'Race customer ' + token}, headers=headers(key)).json()
        second = first if same_ticket else api.post('/customers/prepare', json={'name': ' RACE   CUSTOMER ' + token}, headers=headers(key)).json()
        gate = Barrier(2)
        def run(prepared):
            gate.wait(timeout=10)
            return commit(api, prepared, key)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, [first, second]))
        assert sorted(result.status_code for result in results) == ([200, 200] if same_ticket else [200, 409])
        recorded = next(result.json() for result in results if result.status_code == 200)
        if same_ticket:
            assert {result.json()['customer_id'] for result in results} == {recorded['customer_id']}
            assert sorted(result.json()['replayed'] for result in results) == [False, True]
        else:
            rejected = next(result.json() for result in results if result.status_code == 409)
            assert rejected['detail']['blockers'][0]['existing_customer']['id'] == recorded['customer_id']
        with connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM customers WHERE name ILIKE %s', ('%'+token,))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM write_tickets WHERE actor_id=%s AND status='committed'", (actor_id,))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM actor_write_audit WHERE actor_id=%s AND target_table='customers'", (actor_id,))
            assert cur.fetchone()[0] == 1
    main._reset_actor_cache()


@pytest.mark.db
def test_shift_summary_calls_customer_receipt_plainly_and_preserves_recorded_name(client, db_cursor, actors, customer_schema):
    original = body()['name']
    prepared = prepare(client, actors, {'name': original})
    result = commit(client, prepared, actors['office']['key']).json()
    db_cursor.execute('UPDATE customers SET name=%s WHERE id=%s', ('Renamed ' + uuid4().hex, result['customer_id']))
    response = client.get('/reports/shift-summary', headers=headers(actors['office']['key']))
    assert response.status_code == 200, response.text
    [row] = [r for r in response.json()['other_receipts']['rows'] if r['receipt_number'] == result['receipt_number']]
    assert row['kind'] == 'customer' and row['refs']['customer_id'] == result['customer_id']
    assert 'customer created' in row['line_en'] and 'cliente creado' in row['line_es']
    assert original in row['line_en'] and 'Renamed' not in row['line_en']


@pytest.mark.db
def test_ticket_commit_waits_for_direct_customer_insert_and_offers_it(isolated_database, monkeypatch):
    @contextmanager
    def connection():
        conn = psycopg2.connect(isolated_database)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    monkeypatch.setattr(main, 'get_db_connection', connection)
    token = uuid4().hex
    key = 'direct-customer-race-' + token
    with connection() as conn, conn.cursor() as cur:
        cur.execute(UP.read_text())
        cur.execute("INSERT INTO actors(name,role,key_hash) VALUES (%s,'office',%s)",
                    ('Direct customer race ' + token, sha256(key.encode()).hexdigest()))
    main._reset_actor_cache()
    with TestClient(main.app) as api:
        prepared = api.post('/customers/prepare', json={'name': 'Direct race ' + token}, headers=headers(key)).json()
        # Hold the direct route's exact INSERT on another real connection,
        # proving the ticket waits before checking normalized-name absence.
        original_duplicate = customer_tickets.duplicate
        original_post = customer_tickets.post
        posting = Event()
        duplicate_checked = Event()
        def observed_post(*args, **kwargs):
            posting.set()
            return original_post(*args, **kwargs)
        def observed_duplicate(*args, **kwargs):
            duplicate_checked.set()
            return original_duplicate(*args, **kwargs)
        monkeypatch.setattr(customer_tickets, 'duplicate', observed_duplicate)
        monkeypatch.setattr(customer_tickets, 'post', observed_post)
        direct = psycopg2.connect(isolated_database)
        try:
            with direct.cursor() as cur:
                cur.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (' DIRECT   RACE ' + token,))
                customer_id = cur.fetchone()[0]
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(commit, api, prepared, key)
                # The duplicate check is after the catalog lock. An uncommitted
                # direct INSERT must keep that check waiting until it commits.
                assert posting.wait(timeout=5)
                assert not duplicate_checked.wait(timeout=0.1)
                direct.commit()
                response = future.result(timeout=10)
            error(response, 409, 'TICKET_STALE')
            assert duplicate_checked.is_set()
            blocker = response.json()['detail']['blockers'][0]
            assert blocker['existing_customer']['id'] == customer_id
        finally:
            direct.close()
    main._reset_actor_cache()

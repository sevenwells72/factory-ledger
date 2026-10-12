"""PR #93 review regressions: immutable references, live numbering race and ER tickets."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
import pytest

import main
import order_tickets
from tests.test_order_tickets import (client, actors, catalog, seed, create_body, create_order,
                                      prepare, prepare_raw, count, office)
from tests.test_write_tickets import headers, commit, ticket_row, error, isolated_database
from tests.test_expected_receipt_extract import _insert_document

pytestmark = pytest.mark.db


def test_reference_retry_after_customer_reassignment_returns_original_receipt(client, db_cursor, catalog, actors):
    body = create_body(catalog, external_order_reference='RETRY-' + catalog['suffix'])
    original = create_order(client, catalog, office(actors), external_order_reference=body['external_order_reference'])
    db_cursor.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', ('Moved ' + uuid4().hex,))
    moved = db_cursor.fetchone()['id']
    edit = prepare(client, f"/sales/orders/{original['order_id']}/header/prepare", {'customer_id': moved}, office(actors))
    assert commit(client, edit, office(actors)).status_code == 200
    before = count(db_cursor, 'sales_orders')
    retry = prepare(client, '/sales/orders/prepare', body, office(actors))
    assert retry['can_commit'], retry
    result = commit(client, retry, office(actors))
    assert result.status_code == 200, result.text
    assert result.json() == {**original, 'replayed': True}
    assert commit(client, retry, office(actors)).json() == result.json()
    assert count(db_cursor, 'sales_orders') == before
    # Changing mutable fields never makes the same reference available again.
    conflict = prepare(client, '/sales/orders/prepare', {**body, 'customer_id': moved}, office(actors))
    assert not conflict['can_commit']
    assert conflict['blockers'][0]['code'] == 'EXTERNAL_ORDER_REFERENCE_CONFLICT'


@pytest.mark.parametrize('action', ['create', 'add'])
def test_mixed_quantity_forms_on_repeated_product_warn_without_crashing(client, catalog, action):
    lines = [{'product_id': catalog['products']['food'], 'quantity': 2, 'unit': 'cases'},
             {'product_id': catalog['products']['food'], 'quantity_lb': 30}]
    if action == 'create':
        path = '/sales/orders/prepare'
        body = create_body(catalog, customer_po=None, lines=lines)
    else:
        order = create_order(client, catalog)
        path = f"/sales/orders/{order['order_id']}/lines/prepare"
        body = {'lines': lines}
    first = prepare(client, path, body)
    assert first['can_commit']
    assert commit(client, first).status_code == 200
    again = prepare(client, path, {**body, 'lines': list(reversed(lines))})
    assert again['can_commit']
    assert 'POSSIBLE_DUPLICATE' in [w['code'] for w in again['warnings']]
    error(commit(client, again), 409, 'WARNING_NOT_ACKNOWLEDGED')


def test_live_prepare_while_create_commit_is_uncommitted(isolated_database, monkeypatch):
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        data = seed(cur)
    inserted, drafting, release = Event(), Event(), Event()
    original = main._create_sales_order_core

    def paused(cur, *args, **kwargs):
        # notes is the fourth positional argument after the cursor.
        note = args[3]
        if note == 'racing prepare':
            drafting.set()
        result = original(cur, *args, **kwargs)
        if note == 'paused commit' and release.is_set() is False and armed:
            inserted.set()
            assert release.wait(10)
        return result

    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database, application_name='a7_number_race') as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout='12s'")
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    monkeypatch.setattr(main, '_create_sales_order_core', paused)
    armed = False
    with TestClient(main.app, raise_server_exceptions=False) as http:
        first = prepare(http, '/sales/orders/prepare', create_body(data, notes='paused commit', notes_es='confirmar', customer_po=None))
        armed = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            committed = pool.submit(commit, http, first)
            assert inserted.wait(10)
            prepared = pool.submit(prepare_raw, http, '/sales/orders/prepare',
                                  create_body(data, notes='racing prepare', notes_es='preparar', customer_po=None, cases=4))
            try:
                assert drafting.wait(10)
                # Prove overlapping PostgreSQL operations, rather than relying on timing.
                deadline = monotonic() + 5
                blocked = False
                with psycopg2.connect(isolated_database) as observer:
                    observer.autocommit = True
                    with observer.cursor() as cur:
                        while monotonic() < deadline:
                            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE application_name='a7_number_race' AND wait_event_type='Lock')")
                            if cur.fetchone()[0]:
                                blocked = True
                                break
                            sleep(.01)
                assert blocked, 'prepare must overlap the uncommitted allocation'
            finally:
                release.set()
            result, draft = committed.result(timeout=15), prepared.result(timeout=15)
        assert result.status_code == 200, result.text
        assert draft.status_code == 200, draft.text
        assert draft.json()['can_commit'], draft.text
        second = commit(http, draft.json())
        assert second.status_code == 200, second.text
        assert second.json()['order_number'] != result.json()['order_number']
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM sales_orders WHERE customer_id=%s', (data['customer'],))
        assert cur.fetchone()[0] == 2


@pytest.fixture
def incoming(db_cursor, catalog):
    db_cursor.execute('INSERT INTO suppliers(name, active) VALUES (%s,true) RETURNING id', ('Incoming '+uuid4().hex,))
    return {'product_id': catalog['products']['food'], 'supplier_id': db_cursor.fetchone()['id'],
            'expected_qty': 100, 'reference_number': 'PO-'+uuid4().hex, 'client_source': 'dashboard'}


def test_expected_receipt_create_dry_run_commit_replay_and_receipt(client, db_cursor, incoming, actors):
    before = count(db_cursor, 'expected_receipts')
    draft = prepare(client, '/expected-receipts/prepare', incoming, office(actors))
    assert draft['action'] == 'create_expected_receipt' and draft['can_commit']
    assert count(db_cursor, 'expected_receipts') == before
    response = commit(client, draft, office(actors))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['expected_receipt']['expected_qty'] == 100
    assert result['expected_receipt']['created_by'] == actors['office']['name']
    assert result['receipt_number'].startswith('ER-')
    assert commit(client, draft, office(actors)).json() == {**result, 'replayed': True}
    assert count(db_cursor, 'expected_receipts') == before + 1
    detail = client.get('/receipts/'+result['receipt_number'], headers=headers(office(actors))).json()
    assert detail['expected_receipts'][0]['id'] == result['expected_receipt_id']


@pytest.mark.parametrize('role,status', [('owner',200), ('office',200), ('floor',403)])
def test_expected_receipt_role_gate(client, incoming, actors, role, status):
    response = prepare_raw(client, '/expected-receipts/prepare', incoming, actors[role]['key'])
    assert response.status_code == status, response.text


@pytest.mark.parametrize('operation,payload', [('update', {'expected_qty': 80, 'notes': 'revised'}),
                                               ('cancel', {}), ('update', {'status': 'closed'})])
def test_expected_receipt_edits_are_tickets(client, db_cursor, incoming, actors, operation, payload):
    draft = prepare(client, '/expected-receipts/prepare', incoming, office(actors))
    original = commit(client, draft, office(actors)).json()
    er_id = original['expected_receipt_id']
    prepared = prepare(client, f'/expected-receipts/{er_id}/{operation}/prepare', payload, office(actors))
    assert main.fetch_expected_receipt(db_cursor, er_id)['expected_qty'] == 100
    response = commit(client, prepared, office(actors))
    assert response.status_code == 200, response.text
    record = response.json()['expected_receipt']
    assert record['status'] == ('cancelled' if operation == 'cancel' else payload.get('status', 'open'))
    assert record['expected_qty'] == payload.get('expected_qty', 100)


def test_expected_receipt_rechecks_supplier_and_role_on_commit(client, db_cursor, incoming, actors):
    draft = prepare(client, '/expected-receipts/prepare', incoming, office(actors))
    db_cursor.execute('UPDATE actors SET role=%s WHERE id=%s', ('floor', actors['office']['id']))
    error(commit(client, draft, office(actors)), 403, 'ROLE_NOT_ALLOWED')
    db_cursor.execute('UPDATE actors SET role=%s WHERE id=%s', ('office', actors['office']['id']))
    db_cursor.execute('UPDATE suppliers SET active=false WHERE id=%s', (incoming['supplier_id'],))
    error(commit(client, draft, office(actors)), 409, 'TICKET_STALE')


def test_expected_intake_ticket_preserves_atomic_document_and_alias_flow(client, db_cursor, incoming, actors):
    doc = _insert_document(db_cursor, path=uuid4().hex+'.png', status='extracted')
    body = {'document_id': doc['id'], 'supplier_id': incoming['supplier_id'], 'reference_number': incoming['reference_number'],
            'lines': [{'product_id': incoming['product_id'], 'expected_qty_lb': 50,
                       'vendor_description': 'Vendor product', 'quantity': 2, 'unit': 'cases', 'lb_per_unit': 25}]}
    before = count(db_cursor, 'expected_receipts')
    prepared = prepare(client, '/expected-receipts/extract/approve/prepare', body, office(actors))
    assert prepared['action'] == 'create_expected_receipt' and prepared['can_commit']
    assert count(db_cursor, 'expected_receipts') == before
    db_cursor.execute('SELECT status FROM purchase_documents WHERE id=%s', (doc['id'],))
    assert db_cursor.fetchone()['status'] == 'extracted'
    response = commit(client, prepared, office(actors))
    assert response.status_code == 200, response.text
    assert response.json()['created_count'] == 1 and response.json()['aliases_saved'] == 1
    assert response.json()['created'][0]['source_document_id'] == doc['id']
    assert commit(client, prepared, office(actors)).json()['replayed']
    assert count(db_cursor, 'expected_receipts') == before + 1


def test_reference_retry_ignores_mutable_customer_and_product_activity(client, db_cursor, catalog):
    reference = 'DURABLE-' + catalog['suffix']
    original = create_order(client, catalog, external_order_reference=reference)
    db_cursor.execute('UPDATE customers SET active=false WHERE id=%s', (catalog['customer'],))
    db_cursor.execute('UPDATE products SET active=false WHERE id=%s', (catalog['products']['food'],))
    draft = prepare(client, '/sales/orders/prepare', create_body(catalog, external_order_reference=reference))
    assert draft['can_commit']
    assert commit(client, draft).json() == {**original, 'replayed': True}


def test_same_reference_prepared_by_two_people_creates_once(client, db_cursor, catalog, actors):
    body = create_body(catalog, external_order_reference='ONE-'+catalog['suffix'])
    first = prepare(client, '/sales/orders/prepare', body, office(actors))
    second = prepare(client, '/sales/orders/prepare', body, actors['owner']['key'])
    before = count(db_cursor, 'sales_orders')
    original = commit(client, first, office(actors))
    replay = commit(client, second, actors['owner']['key'])
    assert replay.status_code == 200, replay.text
    assert replay.json() == {**original.json(), 'replayed': True}
    assert count(db_cursor, 'sales_orders') == before + 1


@pytest.mark.parametrize('extra', [{'product_name':'name'}, {'supplier_name':'name'}, {'created_by':'someone'},
                                  {'expected_qty':float('inf')}, {'product_id':True}])
def test_expected_receipt_strict_inputs(client, incoming, extra):
    # Raw JSON allows the non-finite case to exercise request validation, not httpx encoding.
    import json
    response = client.post('/expected-receipts/prepare', content=json.dumps({**incoming, **extra}),
                           headers={**headers(), 'Content-Type':'application/json'})
    assert response.status_code == 422, response.text


def test_expected_receipt_lifecycle_expiry_hash_binding_supersession(client, db_cursor, incoming, actors):
    first = prepare(client, '/expected-receipts/prepare', incoming, office(actors))
    second = prepare(client, '/expected-receipts/prepare', incoming, office(actors))
    error(commit(client, first, office(actors)), 409, 'TICKET_NOT_COMMITTABLE')
    error(commit(client, second, actors['owner']['key']), 403, 'TICKET_WRONG_USER')
    error(commit(client, {**second, 'payload_hash':'bad'}, office(actors)), 409, 'TICKET_PAYLOAD_MISMATCH')
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 minute' WHERE id=%s", (second['ticket_id'],))
    error(commit(client, second, office(actors)), 409, 'TICKET_EXPIRED')


def test_expected_receipt_closed_after_prepare_rejects_without_partial_edit(client, db_cursor, incoming, actors):
    original = commit(client, prepare(client, '/expected-receipts/prepare', incoming, office(actors)), office(actors)).json()
    er_id = original['expected_receipt_id']
    edit = prepare(client, f'/expected-receipts/{er_id}/update/prepare', {'expected_qty':200}, office(actors))
    db_cursor.execute("UPDATE expected_receipts SET status='closed' WHERE id=%s", (er_id,))
    error(commit(client, edit, office(actors)), 409, 'TICKET_STALE')
    assert main.fetch_expected_receipt(db_cursor, er_id)['expected_qty'] == 100


def test_expected_receipt_duplicate_warning_and_direct_sync_contract(client, db_cursor, incoming):
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (incoming['supplier_id'],))
    supplier_name = db_cursor.fetchone()['name']
    # Existing integrations use the legacy master endpoint; ticket provenance is optional.
    body = {k:v for k,v in incoming.items() if k not in ('supplier_id','client_source')}
    direct = client.post('/expected-receipts', json={**body, 'supplier_name':supplier_name, 'created_by':'qbo-po-sync'}, headers=headers())
    assert direct.status_code == 201, direct.text
    er_id = direct.json()['expected_receipt_id']
    assert client.patch(f'/expected-receipts/{er_id}', json={'expected_qty':120}, headers=headers()).status_code == 200
    db_cursor.execute('SELECT ticket_id FROM expected_receipts WHERE id=%s', (er_id,))
    assert db_cursor.fetchone()['ticket_id'] is None
    first = prepare(client, '/expected-receipts/prepare', incoming)
    assert commit(client, first).status_code == 200
    duplicate = prepare(client, '/expected-receipts/prepare', incoming)
    assert 'POSSIBLE_DUPLICATE' in [w['code'] for w in duplicate['warnings']]
    error(commit(client, duplicate), 409, 'WARNING_NOT_ACKNOWLEDGED')


def test_expected_receipt_atomic_intake_rollback_and_duplicate_ack(client, db_cursor, incoming, actors):
    existing = commit(client, prepare(client, '/expected-receipts/prepare', incoming, office(actors)), office(actors)).json()
    doc = _insert_document(db_cursor, path=uuid4().hex+'.png', status='extracted')
    body = {'document_id':doc['id'], 'supplier_id':incoming['supplier_id'], 'reference_number':incoming['reference_number'],
            'lines':[{'product_id':incoming['product_id'], 'expected_qty_lb':50, 'vendor_description':'incoming',
                      'quantity':2, 'lb_per_unit':25}]}
    draft = prepare(client, '/expected-receipts/extract/approve/prepare', body, office(actors))
    assert 'DUPLICATE_REFERENCE' in [w['code'] for w in draft['warnings']]
    error(commit(client, draft, office(actors)), 409, 'WARNING_NOT_ACKNOWLEDGED')
    result = commit(client, draft, office(actors), acknowledged_warnings=['DUPLICATE_REFERENCE'])
    assert result.status_code == 200 and result.json()['duplicate_overridden']
    # A second independently prepared approval cannot reuse the source document.
    stale = prepare(client, '/expected-receipts/extract/approve/prepare', {**body,'force':True}, office(actors))
    assert not stale['can_commit']
    assert stale['blockers'][0]['code'] == 'DOCUMENT_ALREADY_APPROVED'
    error(commit(client, stale, office(actors)), 409, 'TICKET_STALE')


def test_expected_routes_remain_usable_when_a10_removes_direct_allowlist(client, incoming, actors, monkeypatch):
    old = {('POST','/expected-receipts'), ('PATCH','/expected-receipts/{expected_receipt_id}'),
           ('POST','/expected-receipts/extract/approve')}
    monkeypatch.setattr(main, 'ACTOR_WRITE_ALLOWLIST', main.ACTOR_WRITE_ALLOWLIST - old)
    monkeypatch.setattr(main, 'DASHBOARD_KEY_ALLOWLIST', main.DASHBOARD_KEY_ALLOWLIST - old)
    assert prepare(client, '/expected-receipts/prepare', incoming, office(actors))['can_commit']
    for method,path in __import__('write_tickets').EXPECTED_RECEIPT_PREPARE_ROUTES:
        assert client.request(method, path.replace('{expected_receipt_id}','1'), json={}, headers=headers(main.DASHBOARD_API_KEY)).status_code == 403


def test_migration_072_rerunnable_down_up_and_large_so_counter(db_cursor, catalog):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    up = (root/'migrations/072_order_ticket_review_fixes.sql').read_text()
    down = (root/'migrations/down/072_order_ticket_review_fixes_down.sql').read_text()
    db_cursor.execute(up)
    db_cursor.execute(down)
    db_cursor.execute(up)
    db_cursor.execute(up)
    db_cursor.execute("INSERT INTO sales_order_number_counters(order_day,last_number) VALUES(CURRENT_DATE,999) ON CONFLICT(order_day) DO UPDATE SET last_number=999")
    db_cursor.execute("INSERT INTO sales_orders(customer_id,order_number) VALUES(%s,'') RETURNING order_number", (catalog['customer'],))
    assert db_cursor.fetchone()['order_number'].endswith('-1000')
    db_cursor.execute('SAVEPOINT number_rollback')
    db_cursor.execute("INSERT INTO sales_orders(customer_id,order_number) VALUES(%s,'') RETURNING order_number", (catalog['customer'],))
    discarded = db_cursor.fetchone()['order_number']
    db_cursor.execute('ROLLBACK TO SAVEPOINT number_rollback')
    db_cursor.execute("INSERT INTO sales_orders(customer_id,order_number) VALUES(%s,'') RETURNING order_number", (catalog['customer'],))
    assert db_cursor.fetchone()['order_number'] == discarded


@pytest.mark.parametrize('kind', ['orders', 'expected'])
def test_concurrent_prepares_and_commits_use_live_connections(isolated_database, monkeypatch, kind):
    from threading import Barrier
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        data = seed(cur)
        cur.execute('INSERT INTO suppliers(name) VALUES(%s) RETURNING id', ('race '+uuid4().hex,))
        supplier_id = cur.fetchone()['id']
    gate = Barrier(2)
    racing = False
    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout='10s'")
            if racing:
                gate.wait(timeout=10)
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    with TestClient(main.app) as http:
        if kind == 'orders':
            path = '/sales/orders/prepare'
            bodies = [create_body(data, customer_po=None, cases=n) for n in (2,3)]
        else:
            path = '/expected-receipts/prepare'
            bodies = [{'product_id':data['products']['food'],'supplier_id':supplier_id,'expected_qty':n} for n in (10,20)]
        racing = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prepare_raw,http,path,b) for b in bodies]
            drafts = [f.result(timeout=15) for f in futures]
        assert [r.status_code for r in drafts] == [200,200], [r.text for r in drafts]
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Orders: two distinct commits compete for the allocator. ER: double
            # commit one ticket proves exactly one expected delivery + receipt.
            selected = drafts if kind == 'orders' else [drafts[0], drafts[0]]
            futures = [pool.submit(commit,http,d.json()) for d in selected]
            responses = [f.result(timeout=15) for f in futures]
        racing = False
        assert [r.status_code for r in responses] == [200,200], [r.text for r in responses]
        results = [r.json() for r in responses]
        if kind == 'orders':
            assert results[0]['order_number'] != results[1]['order_number']
        else:
            assert results[0]['expected_receipt_id'] == results[1]['expected_receipt_id']
            assert results[0]['receipt_number'] == results[1]['receipt_number']
            assert sorted(r['replayed'] for r in results) == [False,True]


def test_migration_072_refuses_to_discard_expected_ticket_evidence(client, db_cursor, incoming):
    from pathlib import Path
    result = commit(client, prepare(client, '/expected-receipts/prepare', incoming))
    assert result.status_code == 200
    db_cursor.execute('SAVEPOINT guarded_down')
    with pytest.raises(psycopg2.Error, match='072 rollback refused'):
        db_cursor.execute((Path(__file__).resolve().parents[1]/'migrations/down/072_order_ticket_review_fixes_down.sql').read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT guarded_down')

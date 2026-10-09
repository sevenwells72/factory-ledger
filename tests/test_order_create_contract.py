"""Row 158: real PostgreSQL contract, rollback, and concurrent HTTP requests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import psycopg2
import pytest
from fastapi.testclient import TestClient

import main
from tests.test_sales_order_line_fields import client  # noqa: F401
from tests.test_named_actor_writes import named_actors  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'migrations/057_order_create_contract.sql'
DOWN = ROOT / 'migrations/down/057_order_create_contract_down.sql'
pytestmark = pytest.mark.db


def seed(cur):
    suffix = uuid4().hex[:12]
    cur.execute("INSERT INTO customers(name) VALUES (%s) RETURNING id", ('Contract ' + suffix,))
    customer = cur.fetchone()['id']
    products = []
    for name, service, weight in [('Food', False, 25), ('Pallet Charge', True, None), ('Freight', True, None)]:
        cur.execute('''INSERT INTO products(name, type, is_service, case_size_lb)
                       VALUES (%s,'finished',%s,%s) RETURNING id''',
                    (name + ' ' + suffix, service, weight))
        products.append(cur.fetchone()['id'])
    return customer, products


@pytest.fixture
def catalog(db_cursor):
    return seed(db_cursor)


def payload(catalog, **updates):
    customer, products = catalog
    body = {'customer_id': customer, 'customer_po': '062732',
            'external_order_reference': 'ORD28100', 'lines': [
                {'product_id': products[0], 'quantity': 2, 'unit': 'cases', 'unit_price': 30, 'amount': 60},
                {'product_id': products[1], 'quantity': 3, 'unit': 'each', 'unit_price': 12.5, 'amount': 37.5},
                {'product_id': products[2], 'quantity': 1, 'unit': 'each', 'unit_price': 0, 'amount': 0},
            ]}
    body.update(updates)
    return body


def post(client, body):
    result = client.post('/sales/orders', json=body)
    assert result.status_code == 200, result.text
    return result.json()


def counts(cur):
    tables = ('sales_orders', 'sales_order_lines', 'sales_order_create_receipts',
              'actor_write_audit', 'transactions', 'transaction_lines', 'sales_order_allocations')
    result = {}
    for table in tables:
        cur.execute(f'SELECT COUNT(*) AS n FROM {table}')
        result[table] = cur.fetchone()['n']
    return result


def test_exact_po_approved_ids_service_amounts_and_product_weights(client, db_cursor, catalog, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Approved IDs must not re-resolve names')
    monkeypatch.setattr(main, 'resolve_customer_id', unexpected)
    monkeypatch.setattr(main, 'resolve_product_id', unexpected)
    before = counts(db_cursor)
    body = payload(catalog)
    body['customer_name'] = 'Ignored stale customer display name'
    body['lines'][0]['product_name'] = 'Ignored stale product display name'
    result = post(client, body)
    assert result['order_number'].startswith('SO-')
    assert result['customer_po'] == '062732'
    assert result['external_order_reference'] == 'ORD28100'
    assert result['total_lb'] == 50
    assert result['total'] == 97.5
    for line, pid, qty, price, amount in zip(result['lines'], catalog[1], (2, 3, 1), (30, 12.5, 0), (60, 37.5, 0)):
        assert (line['product_id'], line['quantity'], line['unit_price'], line['amount']) == (pid, qty, price, amount)
    db_cursor.execute('SELECT customer_po, external_order_reference FROM sales_orders WHERE id=%s', (result['order_id'],))
    assert db_cursor.fetchone() == {'customer_po': '062732', 'external_order_reference': 'ORD28100'}
    db_cursor.execute('SELECT * FROM sales_order_lines WHERE sales_order_id=%s ORDER BY id', (result['order_id'],))
    lines = db_cursor.fetchall()
    assert [float(line['ordered_quantity']) for line in lines] == [2, 3, 1]
    assert [float(line['quantity_lb']) for line in lines] == [50, 0, 0]
    assert [float(line['amount']) for line in lines] == [60, 37.5, 0]
    detail = client.get(f"/sales/orders/{result['order_id']}").json()
    assert detail['customer_po'] == '062732'
    assert detail['external_order_reference'] == 'ORD28100'
    assert detail['totals']['total_ordered_lb'] == 50
    assert detail['totals']['total_ordered_units'] == 2
    assert detail['totals']['total_value'] == 97.5
    assert detail['lines'][1]['unit_quantity'] == 3
    assert detail['lines'][2]['unit_price'] == 0
    assert detail['lines'][2]['line_value'] == 0
    after = counts(db_cursor)
    for table in ('transactions', 'transaction_lines', 'sales_order_allocations'):
        assert after[table] == before[table]


@pytest.mark.parametrize('po', [None, '', '   '])
def test_no_po_never_blocks_and_is_visible(client, catalog, po):
    first = post(client, payload(catalog, customer_po=po, external_order_reference=None))
    second = post(client, payload(catalog, customer_po=po, external_order_reference=None))
    assert first['order_id'] != second['order_id']
    assert first['customer_po'] is None
    assert first['customer_po_status'] == 'No PO'
    detail = client.get(f"/sales/orders/{first['order_id']}").json()
    assert detail['customer_po_status'] == 'No PO'


def test_po_duplicate_warning_explicit_override_and_trimmed_text(client, db_cursor, catalog):
    first = post(client, payload(catalog, customer_po='  062732  '))
    duplicate = payload(catalog, external_order_reference='ORD28101')
    response = client.post('/sales/orders', json=duplicate)
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == 'DUPLICATE_CUSTOMER_PO'
    assert 'allow_duplicate_po=true' in response.json()['detail']['message']
    duplicate['allow_duplicate_po'] = True
    second = post(client, duplicate)
    assert second['order_id'] != first['order_id']
    db_cursor.execute('SELECT customer_po FROM sales_orders WHERE id=%s', (first['order_id'],))
    assert db_cursor.fetchone()['customer_po'] == '062732'


def test_retry_is_original_even_after_header_and_catalog_edits(client, db_cursor, catalog):
    body = payload(catalog)
    original = post(client, body)
    before = counts(db_cursor)
    assert post(client, body) == original  # retry wins over duplicate PO warning
    assert counts(db_cursor) == before
    assert client.patch(f"/sales/orders/{original['order_id']}", json={'customer_po': 'NEW-PO'}).status_code == 200
    db_cursor.execute('UPDATE products SET case_size_lb=50, name=%s WHERE id=%s', ('Renamed ' + uuid4().hex, catalog[1][0]))
    assert post(client, body) == original
    assert counts(db_cursor) == before


@pytest.mark.parametrize('change', ['quantity', 'price', 'po', 'notes', 'date'])
def test_conflicting_reference_is_clear_and_atomic(client, db_cursor, catalog, change):
    body = payload(catalog)
    post(client, body)
    before = counts(db_cursor)
    if change == 'quantity':
        body['lines'][0]['quantity'] = 3
    elif change == 'price':
        body['lines'][0]['unit_price'] = 31
    else:
        body[{'po': 'customer_po', 'notes': 'notes', 'date': 'requested_ship_date'}[change]] = '2026-10-01' if change == 'date' else 'different'
    response = client.post('/sales/orders', json=body)
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == 'EXTERNAL_ORDER_REFERENCE_CONFLICT'
    assert counts(db_cursor) == before


def test_reference_is_globally_unique_in_database(client, db_cursor, catalog):
    original = post(client, payload(catalog))
    db_cursor.execute('SAVEPOINT uniqueness')
    with pytest.raises(psycopg2.errors.UniqueViolation):
        db_cursor.execute("INSERT INTO sales_orders(customer_id, order_number, external_order_reference) VALUES (%s,'','ORD28100')", (catalog[0],))
    db_cursor.execute('ROLLBACK TO SAVEPOINT uniqueness')
    db_cursor.execute("INSERT INTO customers(name) VALUES (%s) RETURNING id", ('Other ' + uuid4().hex,))
    other = db_cursor.fetchone()['id']
    response = client.post('/sales/orders', json=payload((other, catalog[1])))
    assert response.status_code == 409
    assert response.json()['detail']['error_code'] == 'EXTERNAL_ORDER_REFERENCE_CONFLICT'


@pytest.mark.parametrize('bad_line', [
    {'quantity': -1, 'unit': 'each'},
    {'quantity': 2, 'unit': 'each'},  # physical product cannot be each
    {'quantity': 2, 'unit': 'cases', 'amount': 999, 'unit_price': 30},
    {'quantity': 2, 'unit': 'cases', 'quantity_lb': 17},
])
def test_bad_later_line_rolls_back_header_lines_audit_and_receipt(client, db_cursor, catalog, named_actors, bad_line):
    before = counts(db_cursor)
    body = payload(catalog)
    body['lines'].append({'product_id': catalog[1][0], **bad_line})
    response = client.post('/sales/orders', json=body, headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == 422, response.text
    assert counts(db_cursor) == before


@pytest.mark.parametrize('failure', ['audit', 'receipt'])
def test_atomic_failure_after_lines(client, db_cursor, catalog, named_actors, monkeypatch, failure):
    before = counts(db_cursor)
    if failure == 'audit':
        def fail(*args):
            raise RuntimeError('Injected audit failure')
        monkeypatch.setattr(main, '_record_actor_write', fail)
    else:
        db_cursor.execute('''CREATE FUNCTION pg_temp.fail_receipt() RETURNS trigger LANGUAGE plpgsql AS $$
                            BEGIN RAISE EXCEPTION 'Injected receipt failure'; END $$;
                            CREATE TRIGGER fail_receipt BEFORE INSERT ON sales_order_create_receipts
                            FOR EACH ROW EXECUTE FUNCTION pg_temp.fail_receipt()''')
    response = client.post('/sales/orders', json=payload(catalog), headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == 500, response.text
    assert counts(db_cursor) == before


@pytest.mark.parametrize('actor', ['Blubber', 'Arturo', 'Luz', 'Miriam'])
def test_named_create_and_header_audit_same_transaction(client, db_cursor, catalog, named_actors, actor):
    client.headers['X-API-Key'] = named_actors[actor]['key']
    if actor in ('Arturo', 'Luz'):   # floor: §4.3 (A2) denies create_order on the direct route too
        response = client.post('/sales/orders', json=payload(catalog))
        assert response.status_code == 403, response.text
        assert response.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
        db_cursor.execute('SELECT count(*) AS n FROM actor_write_audit')
        assert db_cursor.fetchone()['n'] == 0
        return
    result = post(client, payload(catalog))
    assert post(client, payload(catalog)) == result
    response = client.patch(f"/sales/orders/{result['order_id']}", json={'customer_po': '000008'})
    assert response.status_code == 200, response.text
    assert response.json()['customer_po'] == '000008'
    db_cursor.execute('SELECT operator_id, method, target_table FROM actor_write_audit ORDER BY id')
    audits = db_cursor.fetchall()
    assert len(audits) == 5  # one header + three lines; retry adds none; PATCH adds one
    assert all(row['operator_id'] == actor for row in audits)
    assert audits[-1]['method'] == 'PATCH'
    assert audits[-1]['target_table'] == 'sales_orders'


def test_header_po_clear_duplicate_override_customer_move_and_locked_state(client, db_cursor, catalog):
    first = post(client, payload(catalog))
    second = post(client, payload(catalog, customer_po=None, external_order_reference='SECOND'))
    url = f"/sales/orders/{second['order_id']}"
    assert client.patch(url, json={'customer_po': '062732'}).status_code == 409
    updated = client.patch(url, json={'customer_po': '062732', 'allow_duplicate_po': True})
    assert updated.status_code == 200, updated.text
    assert client.patch(url, json={'customer_po': None}).json()['customer_po_status'] == 'No PO'
    db_cursor.execute("INSERT INTO customers(name) VALUES (%s) RETURNING id", ('Move ' + uuid4().hex,))
    other = db_cursor.fetchone()['id']
    # A reference is reserved globally; moving the original order does not free it.
    conflict = client.post('/sales/orders', json=payload((other, catalog[1]), external_order_reference='SECOND'))
    assert conflict.status_code == 409
    assert client.patch(url, json={'customer_id': other}).status_code == 200
    assert post(client, payload(catalog, customer_po=None, external_order_reference='SECOND')) == second
    first_url = f"/sales/orders/{first['order_id']}"
    assert client.patch(first_url, json={'customer_po': '062732'}).status_code == 200  # excludes self
    db_cursor.execute("UPDATE sales_orders SET status='ready' WHERE id=%s", (first['order_id'],))
    assert client.patch(first_url, json={'customer_po': '001'}).status_code == 400


def test_header_audit_failure_rolls_back_po(client, db_cursor, catalog, named_actors, monkeypatch):
    original = post(client, payload(catalog))
    def fail(*args):
        raise RuntimeError('Injected header audit failure')
    monkeypatch.setattr(main, '_record_actor_write', fail)
    response = client.patch(f"/sales/orders/{original['order_id']}", json={'customer_po': 'CHANGED'},
                            headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == 500, response.text
    db_cursor.execute('SELECT customer_po FROM sales_orders WHERE id=%s', (original['order_id'],))
    assert db_cursor.fetchone()['customer_po'] == '062732'


def test_legacy_request_response_and_no_receipt_unchanged(client, db_cursor, catalog):
    db_cursor.execute('SELECT name FROM customers WHERE id=%s', (catalog[0],))
    customer = db_cursor.fetchone()['name']
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (catalog[1][0],))
    product = db_cursor.fetchone()['name']
    before = counts(db_cursor)
    response = post(client, {'customer_name': customer, 'lines': [{'product_name': product, 'quantity_lb': 50, 'unit_price': 30}]})
    assert response == {
        'order_id': response['order_id'], 'order_number': response['order_number'],
        'customer': customer, 'requested_ship_date': None, 'status': 'confirmed',
        'total_lb': 50, 'lines': [{'line_id': response['lines'][0]['line_id'],
          'product': product, 'quantity_lb': 50, 'original_quantity': None,
          'original_unit': 'lb', 'case_weight_lb': None, 'unit_price': 30}],
        'warnings': None, 'message': f"Order {response['order_number']} created with 1 line(s)", 'success': True,
    }
    after = counts(db_cursor)
    assert after['sales_order_create_receipts'] == before['sales_order_create_receipts']
    assert after['actor_write_audit'] == before['actor_write_audit']


def test_line_edit_keeps_saved_amount_and_original_receipt(client, catalog):
    body = payload(catalog)
    original = post(client, body)
    order_id = original['order_id']
    line_id = original['lines'][0]['line_id']
    response = client.patch(f'/sales/orders/{order_id}/lines/{line_id}/update', params={'quantity_lb': 100, 'unit_price': 0})
    assert response.status_code == 200, response.text
    assert response.json()['unit_price'] == 0
    line = client.get(f'/sales/orders/{order_id}').json()['lines'][0]
    assert line['quantity'] == 4
    assert line['amount'] == 0
    assert line['unit_price'] == 0
    assert post(client, body) == original


@pytest.mark.parametrize('service_id,service_name', [(102, 'Pallets'), (176, 'Pallet Charge')])
def test_both_pallet_products_and_other_services_ship_without_inventory(client, db_cursor, catalog, service_id, service_name):
    # Product identity is authoritative; no product-name heuristics or special
    # pallet count in a pounds column. Both deployed pallet SKUs use is_service.
    db_cursor.execute("INSERT INTO products(id,name,type,is_service,active) VALUES (%s,%s,'packaging',true,true)",
                      (service_id, service_name))
    body = payload(catalog)
    body['lines'][1]['product_id'] = service_id
    original = post(client, body)
    service = original['lines'][1]
    assert service['product_id'] == service_id
    assert service['is_service'] is True
    assert (service['quantity'], service['unit_price'], service['amount'], service['quantity_lb']) == (3, 12.5, 37.5, 0)
    assert original['total_lb'] == 50
    assert original['total'] == 97.5
    order_id = original['order_id']
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,supplier_lot_code,entry_source) VALUES (%s,'CONTRACT-LOT-001','N/A','received') RETURNING id", (catalog[1][0],))
    lot_id = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,timestamp) VALUES ('receive',now()) RETURNING id")
    transaction_id = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,50)', (transaction_id,catalog[1][0],lot_id))
    preview = client.post(f'/sales/orders/{order_id}/ship', json={'mode': 'preview', 'ship_all': True})
    assert preview.status_code == 200, preview.text
    assert len(preview.json()['lines']) == 3
    service_preview = [line for line in preview.json()['lines'] if line.get('is_service')]
    assert len(service_preview) == 2
    assert all(line['on_hand_lb'] is None for line in service_preview)
    response = client.post(f'/sales/orders/{order_id}/ship', json={'mode': 'commit', 'ship_all': True})
    assert response.status_code == 200, response.text
    assert response.json()['order_status'] == 'shipped'
    detail = client.get(f'/sales/orders/{order_id}').json()
    assert detail['lines'][1]['line_status'] == 'fulfilled'
    assert detail['lines'][1]['remaining_units'] == 0
    assert detail['lines'][1]['quantity'] == 3
    assert detail['totals']['total_shipped_lb'] == 50
    assert detail['totals']['total_value'] == 97.5
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lines WHERE product_id IN (%s,%s)', (service_id, catalog[1][2]))
    assert db_cursor.fetchone()['n'] == 0


def test_all_zero_prices_produce_zero_total(client, catalog):
    body = payload(catalog)
    for line in body['lines']:
        line.update(unit_price=0, amount=0)
    result = post(client, body)
    assert result['total'] == 0
    assert client.get(f"/sales/orders/{result['order_id']}").json()['totals']['total_value'] == 0


def test_detail_and_price_edits_keep_saved_weight_and_use_service_flag(client, db_cursor, catalog):
    original = post(client, payload(catalog))
    db_cursor.execute("UPDATE products SET case_size_lb=50, name=%s WHERE id=%s", ('Coffee ' + uuid4().hex, catalog[1][0]))
    order_id = original['order_id']
    line_id = original['lines'][0]['line_id']
    detail = client.get(f'/sales/orders/{order_id}').json()
    assert detail['lines'][0]['case_size_lb'] == 25
    assert detail['lines'][0]['cases'] == 2
    assert not detail['lines'][0].get('is_non_weight')
    response = client.patch(f'/sales/orders/{order_id}/lines/{line_id}/update', params={'unit_price': 40})
    assert response.status_code == 200, response.text
    assert client.get(f'/sales/orders/{order_id}').json()['lines'][0]['amount'] == 80


@pytest.mark.parametrize('update', [{'customer_po': 62732}, {'allow_duplicate_po': 'true'}, {'lines': []}])
def test_invalid_new_fields_rejected_without_writes(client, db_cursor, catalog, update):
    before = counts(db_cursor)
    response = client.post('/sales/orders', json=payload(catalog, **update))
    assert response.status_code == 422, response.text
    assert counts(db_cursor) == before


def test_po_header_customer_move_checks_duplicate_po(client, db_cursor, catalog):
    first = post(client, payload(catalog))
    db_cursor.execute("INSERT INTO customers(name) VALUES (%s) RETURNING id", ('MovePO ' + uuid4().hex,))
    other = db_cursor.fetchone()['id']
    post(client, payload((other, catalog[1]), external_order_reference='OTHER'))
    url = f"/sales/orders/{first['order_id']}"
    response = client.patch(url, json={'customer_id': other})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == 'DUPLICATE_CUSTOMER_PO'
    assert client.patch(url, json={'customer_id': other, 'allow_duplicate_po': True}).status_code == 200
    # Original customer/reference still returns its original receipt after move.
    assert post(client, payload(catalog)) == first


def test_migration_057_reversible_rerunnable_and_preserves_ledger_views(db_cursor):
    db_cursor.execute("SELECT oid FROM pg_class WHERE relname IN ('ledger_current_transactions', 'ledger_current_transaction_lines') ORDER BY oid")
    views = db_cursor.fetchall()
    db_cursor.execute(DOWN.read_text())
    db_cursor.execute(UP.read_text())
    db_cursor.execute(UP.read_text())
    db_cursor.execute("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid='sales_order_create_receipts'::regclass")
    assert db_cursor.fetchone() == {'relrowsecurity': True, 'relforcerowsecurity': False}
    db_cursor.execute("SELECT oid FROM pg_class WHERE relname IN ('ledger_current_transactions', 'ledger_current_transaction_lines') ORDER BY oid")
    assert db_cursor.fetchall() == views


@pytest.mark.parametrize('mode', ['same', 'conflicting', 'po'])
def test_concurrent_requests_commit_exactly_one_order(_db_connection, monkeypatch, mode):
    # UUID-named fixture rows isolate committed data from every other test.
    # Every connection uses ONLY the guarded disposable test URL.
    url = main.DATABASE_URL
    from psycopg2.extras import RealDictCursor
    with psycopg2.connect(url) as setup:
        with setup.cursor(cursor_factory=RealDictCursor) as cur:
            catalog = seed(cur)
        setup.commit()
    gate = Barrier(2)
    original_lock = main._lock_order_reference
    def rendezvous(cur, customer_id, reference):
        gate.wait(timeout=10)
        original_lock(cur, customer_id, reference)
    monkeypatch.setattr(main, '_lock_order_reference', rendezvous)
    @contextmanager
    def isolated_connection():
        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout = '10s'")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    monkeypatch.setattr(main, 'get_db_connection', isolated_connection)
    first = payload(catalog)
    second = payload(catalog)
    if mode == 'conflicting':
        second['notes'] = 'Different intent'
    elif mode == 'po':
        second['external_order_reference'] = 'SECOND'
    try:
        with TestClient(main.app) as http:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(http.post, '/sales/orders', json=body, headers={'X-API-Key': main.API_KEY})
                           for body in (first, second)]
                responses = [f.result(timeout=20) for f in futures]
        assert sorted(r.status_code for r in responses) == ([200, 200] if mode == 'same' else [200, 409]), [r.text for r in responses]
        if mode == 'same':
            assert responses[0].json() == responses[1].json()
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM sales_orders WHERE customer_id=%s', (catalog[0],))
            assert cur.fetchone()[0] == 1
            cur.execute('SELECT count(*) FROM sales_order_create_receipts WHERE customer_id=%s', (catalog[0],))
            assert cur.fetchone()[0] == 1
    finally:
        # Only rows created by this fixture, never a shared database reset.
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute('DELETE FROM sales_order_create_receipts WHERE customer_id=%s', (catalog[0],))
            cur.execute('DELETE FROM sales_order_lines WHERE sales_order_id IN (SELECT id FROM sales_orders WHERE customer_id=%s)', (catalog[0],))
            cur.execute('DELETE FROM sales_orders WHERE customer_id=%s', (catalog[0],))
            cur.execute('DELETE FROM products WHERE id=ANY(%s)', (catalog[1],))
            cur.execute('DELETE FROM customers WHERE id=%s', (catalog[0],))

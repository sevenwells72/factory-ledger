"""PR #67 cross-review probes: real HTTP writes, PDFs, and two-connection races."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from threading import Barrier
from uuid import uuid4
import re

import psycopg2
from psycopg2.extras import RealDictCursor
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
import yaml

import main
from tests.test_sales_order_line_fields import client  # noqa: F401
from tests.test_named_actor_writes import named_actors  # noqa: F401
from tests.test_order_create_contract import catalog, seed, payload, post, counts, ROOT, UP, DOWN  # noqa: F401
from tests.test_sales_order_extract import _insert_document, _approve_payload, _approve_line

pytestmark = pytest.mark.db


@pytest.mark.parametrize('path', ['manual', 'receipt', 'intake'])
def test_rebased_creation_audits_each_target_exactly_once(client, db_cursor, catalog, named_actors, path):
    client.headers['X-API-Key'] = named_actors['Miriam']['key']
    new_customer = 'Zq' + uuid4().hex
    if path == 'intake':
        doc = _insert_document(db_cursor, path=uuid4().hex, kind='sales', status='extracted', sha=uuid4().hex)
        body = _approve_payload(doc['id'], catalog[0], [
            _approve_line(catalog[1][0], 20), _approve_line(catalog[1][0], 30)], po=uuid4().hex)
        route, status = '/sales/orders/extract/approve', 201
    elif path == 'manual':
        body = legacy_body(db_cursor, catalog, {'quantity_lb': 50})
        body['customer_name'] = new_customer
        body['lines'].append(dict(body['lines'][0]))
        route, status = '/sales/orders', 200
    else:
        body = payload(catalog)
        del body['customer_id']
        body['customer_name'] = new_customer
        route, status = '/sales/orders', 200
    response = client.post(route, json=body)
    assert response.status_code == status, response.text
    result = response.json()
    expected = {('sales_orders', result['order_id'])} | {
        ('sales_order_lines', line['line_id']) for line in result['lines']}
    if path != 'intake':
        db_cursor.execute('SELECT id FROM customers WHERE name=%s', (new_customer,))
        expected.add(('customers', db_cursor.fetchone()['id']))
    db_cursor.execute('SELECT * FROM actor_write_audit ORDER BY id')
    audit = db_cursor.fetchall()
    assert len(audit) == len(expected) == {'manual': 4, 'receipt': 5, 'intake': 3}[path]
    assert {(r['target_table'], r['target_id']) for r in audit} == expected
    assert {(r['actor_id'], r['operator_id'], r['method'], r['route']) for r in audit} == {
        (named_actors['Miriam']['id'], 'Miriam', 'POST', route)}
    if path == 'receipt':
        before = counts(db_cursor)
        assert client.post(route, json=body).json() == result
        assert counts(db_cursor) == before
        assert before['sales_order_create_receipts'] == 1


@pytest.mark.parametrize('failure', ['second_line_audit', 'receipt'])
def test_rebased_receipt_failure_undoes_auto_customer_and_all_audits(
        client, db_cursor, catalog, named_actors, monkeypatch, failure):
    body = payload(catalog)
    del body['customer_id']
    body['customer_name'] = 'Zq' + uuid4().hex
    before = counts(db_cursor)
    if failure == 'receipt':
        db_cursor.execute('''CREATE FUNCTION pg_temp.reject_receipt() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'receipt rollback probe'; END $$;
            CREATE TRIGGER reject_receipt BEFORE INSERT ON sales_order_create_receipts
            FOR EACH ROW EXECUTE FUNCTION pg_temp.reject_receipt()''')
    else:
        original = main._record_actor_write
        line_audits = []
        def fail_second_line(cur, request, table, target):
            if table == 'sales_order_lines':
                line_audits.append(target)
                if len(line_audits) == 2:
                    cur.execute('SELECT 1 / 0')
            original(cur, request, table, target)
        monkeypatch.setattr(main, '_record_actor_write', fail_second_line)
    response = client.post('/sales/orders', json=body,
                           headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == 500, response.text
    assert counts(db_cursor) == before
    db_cursor.execute('SELECT id FROM customers WHERE name=%s', (body['customer_name'],))
    assert db_cursor.fetchone() is None


@pytest.mark.parametrize('legacy', [False, True])
def test_reduced_partial_line_does_not_block_other_line_completion(client, db_cursor, catalog, legacy):
    body = (legacy_body(db_cursor, catalog, {'quantity_lb': 50}) if legacy else
            payload(catalog, lines=[{'product_id': catalog[1][0], 'quantity_lb': 50}]))
    body['lines'].append(dict(body['lines'][0]))
    created = post(client, body)
    first, second = [line['line_id'] for line in created['lines']]
    url = f"/sales/orders/{created['order_id']}/ship"
    stock(db_cursor, catalog[1][0], 90)
    response = client.post(url, json={'mode': 'commit', 'lines': [{'line_id': first, 'quantity_lb': 40}]})
    assert response.status_code == 200, response.text
    assert response.json()['order_status'] == 'partial_ship'
    response = client.patch(f"/sales/orders/{created['order_id']}/lines/{first}/update", params={'quantity_lb': 40})
    assert response.status_code == 200, response.text
    response = client.post(url, json={'mode': 'commit', 'lines': [{'line_id': second, 'quantity_lb': 50}]})
    assert response.status_code == 200, response.text
    assert response.json()['order_status'] == 'shipped'
    db_cursor.execute('SELECT status FROM sales_orders WHERE id=%s', (created['order_id'],))
    assert db_cursor.fetchone()['status'] == 'shipped'


@pytest.mark.parametrize('fail_audit', [False, True])
def test_legacy_stuck_partial_order_finishes_without_new_shipment(
        client, db_cursor, catalog, named_actors, monkeypatch, fail_audit):
    created = post(client, legacy_body(db_cursor, catalog, {'quantity_lb': 50}))
    line_id = created['lines'][0]['line_id']
    url = f"/sales/orders/{created['order_id']}/ship"
    stock(db_cursor, catalog[1][0], 40)
    assert client.post(url, json={'mode': 'commit', 'ship_all': True}).status_code == 200
    # Persist the exact pre-fix stuck state: partial status but nothing remaining.
    db_cursor.execute('UPDATE sales_order_lines SET quantity_lb=40 WHERE id=%s', (line_id,))
    before = counts(db_cursor)
    db_cursor.execute('SELECT count(*) AS n FROM shipments')
    shipments_before = db_cursor.fetchone()['n']
    if fail_audit:
        def fail(cur, *args):
            cur.execute('SELECT 1 / 0')
        monkeypatch.setattr(main, '_record_actor_write', fail)
    response = client.post(url, json={'mode': 'commit', 'ship_all': True},
                           headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == (500 if fail_audit else 200), response.text
    db_cursor.execute('SELECT status FROM sales_orders WHERE id=%s', (created['order_id'],))
    assert db_cursor.fetchone()['status'] == ('partial_ship' if fail_audit else 'shipped')
    if not fail_audit:
        assert response.json()['shipment_id'] is None
        assert response.json()['lines_shipped'] == []
        before['actor_write_audit'] += 1
        repeated = client.post(url, json={'mode': 'commit', 'ship_all': True})
        assert repeated.status_code == 409
    assert counts(db_cursor) == before
    db_cursor.execute('SELECT count(*) AS n FROM shipments')
    assert db_cursor.fetchone()['n'] == shipments_before


def test_legacy_422_reports_all_bad_lines_with_original_indices(client, db_cursor, catalog):
    body = legacy_body(db_cursor, catalog, {'quantity_lb': 50})
    product = body['lines'][0]['product_name']
    body['lines'] += [{'product_name': product, 'quantity': 2, 'unit': 'cases'},
                      {'product_name': product, 'quantity': 3, 'unit': 'bags'}]
    before = counts(db_cursor)
    response = client.post('/sales/orders', json=body)
    assert response.status_code == 422, response.text
    errors = response.json()['detail']
    assert [e['loc'] for e in errors] == [
        ['body', 'lines', 1, 'quantity_lb'], ['body', 'lines', 2, 'quantity_lb']]
    assert "unit is 'cases'" in errors[0]['msg']
    assert "unit is 'bags'" in errors[1]['msg']
    assert counts(db_cursor) == before


@pytest.mark.parametrize('price,status', [(30, 422), (0, 422), (None, 200)])
def test_priced_case_line_edit_requires_whole_cases_and_names_product(client, db_cursor, catalog, price, status):
    created = post(client, payload(catalog, lines=[{
        'product_id': catalog[1][0], 'quantity': 4, 'unit': 'cases', 'unit_price': price}]))
    line_id = created['lines'][0]['line_id']
    db_cursor.execute('SELECT row_to_json(l) AS data FROM sales_order_lines l WHERE id=%s', (line_id,))
    original = db_cursor.fetchone()['data']
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (catalog[1][0],))
    name = db_cursor.fetchone()['name']
    response = client.patch(f"/sales/orders/{created['order_id']}/lines/{line_id}/update", params={'quantity_lb': 110})
    assert response.status_code == status, response.text
    if status == 422:
        detail = response.json()['detail']
        assert detail['error_code'] == 'WHOLE_CASE_QUANTITY_REQUIRED'
        assert name in detail['message'] and '25' in detail['message']
        db_cursor.execute('SELECT row_to_json(l) AS data FROM sales_order_lines l WHERE id=%s', (line_id,))
        assert db_cursor.fetchone()['data'] == original


@pytest.mark.parametrize('kind', ['cancelled', 'service_first'])
def test_shipping_rejections_explain_cancelled_and_service_lines(client, db_cursor, catalog, kind):
    created = post(client, payload(catalog))
    if kind == 'cancelled':
        line = created['lines'][0]
        cancelled = client.patch(f"/sales/orders/{created['order_id']}/lines/{line['line_id']}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        quantity, code, wording = 50, 'LINE_CANCELLED', 'cancelled'
    else:
        stock(db_cursor, catalog[1][0])  # Stock exists; this is a sequencing error.
        line = created['lines'][1]
        quantity, code, wording = 0, 'PHYSICAL_LINES_PENDING', 'physical product lines'
    before = counts(db_cursor)
    db_cursor.execute('SELECT line_status FROM sales_order_lines WHERE id=%s', (line['line_id'],))
    status = db_cursor.fetchone()['line_status']
    response = client.post(f"/sales/orders/{created['order_id']}/ship", json={
        'mode': 'commit', 'lines': [{'line_id': line['line_id'], 'quantity_lb': quantity}]})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == code
    assert wording in response.json()['detail']['message']
    assert 'zero available stock' not in response.text
    assert counts(db_cursor) == before
    db_cursor.execute('SELECT line_status FROM sales_order_lines WHERE id=%s', (line['line_id'],))
    assert db_cursor.fetchone()['line_status'] == status


def stock(cur, product_id, pounds=50):
    cur.execute("INSERT INTO lots(product_id,lot_code,supplier_lot_code,entry_source) VALUES (%s,%s,'N/A','received') RETURNING id",
                (product_id, 'REVIEW-' + uuid4().hex[:12].upper()))
    lot_id = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type,timestamp) VALUES ('receive',now()) RETURNING id")
    transaction_id = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                (transaction_id, product_id, lot_id, pounds))


@pytest.mark.parametrize('shipped', [False, True])
def test_packing_slip_renders_service_counts_before_and_after_ship(client, db_cursor, catalog, shipped):
    # Simple names make assertions inspect the generated PDF's table rows.
    for pid, name in zip(catalog[1], ('Review Food', 'Review Pallets', 'Review Freight')):
        db_cursor.execute('UPDATE products SET name=%s WHERE id=%s', (name, pid))
    created = post(client, payload(catalog))
    if shipped:
        stock(db_cursor, catalog[1][0])
        response = client.post(f"/sales/orders/{created['order_id']}/ship", json={'mode': 'commit', 'ship_all': True})
        assert response.status_code == 200, response.text
    response = client.get(f"/sales/orders/{created['order_id']}/packing-slip")
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'application/pdf'
    text = '\n'.join(page.extract_text() for page in PdfReader(BytesIO(response.content)).pages)
    assert re.search(r'Review Pallets\s+N/A\s+3\b', text), text
    assert re.search(r'Review Freight\s+N/A\s+1\b', text), text


@pytest.mark.parametrize('finish', ['ship_all', 'individual', 'all_at_once'])
def test_services_finish_without_more_physical_inventory(client, db_cursor, catalog, finish):
    created = post(client, payload(catalog))
    order_id = created['order_id']
    url = f'/sales/orders/{order_id}/ship'
    stock(db_cursor, catalog[1][0])
    if finish != 'all_at_once':
        response = client.post(url, json={'mode': 'commit', 'lines': [
            {'line_id': created['lines'][0]['line_id'], 'quantity_lb': 50}]})
        assert response.status_code == 200, response.text
        assert response.json()['order_status'] == 'partial_ship'
        before = counts(db_cursor)
    if finish == 'individual':
        for line, expected in zip(created['lines'][1:], ('partial_ship', 'shipped')):
            response = client.post(url, json={'mode': 'commit', 'lines': [
                {'line_id': line['line_id'], 'quantity_lb': 0}]})
            assert response.status_code == 200, response.text
            assert response.json()['order_status'] == expected
        repeated = client.post(url, json={'mode': 'commit', 'lines': [
            {'line_id': created['lines'][1]['line_id'], 'quantity_lb': 0}]})
        assert repeated.status_code == 409
    else:
        response = client.post(url, json={'mode': 'commit', 'ship_all': True})
        assert response.status_code == 200, response.text
        assert response.json()['order_status'] == 'shipped'
    if finish != 'all_at_once':
        after = counts(db_cursor)
        assert after['transactions'] == before['transactions']
        assert after['transaction_lines'] == before['transaction_lines']
    detail = client.get(f'/sales/orders/{order_id}').json()
    assert all(line['line_status'] == 'fulfilled' for line in detail['lines'])
    assert detail['totals']['total_shipped_lb'] == 50
    assert detail['lines'][1]['quantity_lb'] == 0


def test_zero_shipment_still_rolls_back_with_unfulfilled_physical_lines(client, db_cursor, catalog):
    created = post(client, payload(catalog))
    before = counts(db_cursor)
    response = client.post(f"/sales/orders/{created['order_id']}/ship", json={'mode': 'commit', 'ship_all': True})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['error_code'] == 'ZERO_SHIPMENT'
    assert counts(db_cursor) == before
    db_cursor.execute('SELECT line_status FROM sales_order_lines WHERE sales_order_id=%s', (created['order_id'],))
    assert all(row['line_status'] == 'pending' for row in db_cursor.fetchall())


def test_shipping_one_physical_line_does_not_finish_other_pending_lines(client, db_cursor, catalog):
    body = payload(catalog)
    body['lines'] = [body['lines'][0], dict(body['lines'][0])]
    created = post(client, body)
    stock(db_cursor, catalog[1][0], 100)
    url = f"/sales/orders/{created['order_id']}/ship"
    response = client.post(url, json={'mode': 'commit', 'lines': [
        {'line_id': created['lines'][0]['line_id'], 'quantity_lb': 50}]})
    assert response.status_code == 200, response.text
    assert response.json()['order_status'] == 'partial_ship'
    db_cursor.execute("UPDATE sales_order_lines SET line_status='cancelled' WHERE id=%s", (created['lines'][1]['line_id'],))
    # The cancellation path is tested elsewhere; include cancelled rows in the
    # same completion query via a separate service-only finishing request.
    db_cursor.execute("INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,ordered_quantity,ordered_unit) VALUES (%s,%s,0,1,'each')", (created['order_id'], catalog[1][1]))
    response = client.post(url, json={'mode': 'commit', 'ship_all': True})
    assert response.status_code == 200, response.text
    assert response.json()['order_status'] == 'shipped'


@pytest.mark.parametrize('fail_audit', [False, True])
def test_line_edit_audits_named_actor_atomically(client, db_cursor, catalog, named_actors, monkeypatch, fail_audit):
    created = post(client, payload(catalog))
    line_id = created['lines'][0]['line_id']
    before = counts(db_cursor)
    if fail_audit:
        def fail(*args):
            raise RuntimeError('Injected line audit failure')
        monkeypatch.setattr(main, '_record_actor_write', fail)
    response = client.patch(f"/sales/orders/{created['order_id']}/lines/{line_id}/update",
                            params={'quantity_lb': 100, 'unit_price': 40},
                            headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == (500 if fail_audit else 200), response.text
    db_cursor.execute('SELECT quantity_lb,ordered_quantity,amount FROM sales_order_lines WHERE id=%s', (line_id,))
    saved = db_cursor.fetchone()
    assert tuple(saved.values()) == ((50, 2, 60) if fail_audit else (100, 4, 160))
    if fail_audit:
        assert counts(db_cursor) == before
    else:
        db_cursor.execute("SELECT actor_id,operator_id,method,route FROM actor_write_audit WHERE target_table='sales_order_lines' AND target_id=%s", (line_id,))
        assert db_cursor.fetchone() == {'actor_id': named_actors['Miriam']['id'], 'operator_id': 'Miriam',
                                       'method': 'PATCH', 'route': '/sales/orders/{order_id}/lines/{line_id}/update'}


def test_trimmed_reference_retries_and_po_text_preserves_zeros_and_internal_spaces(client, db_cursor, catalog):
    body = payload(catalog, customer_po=' \t062  732 \n')
    original = post(client, body)
    assert original['customer_po'] == '062  732'
    body.update(external_order_reference=' ORD28100 ', customer_po='062  732')
    assert post(client, body) == original
    response = client.patch(f"/sales/orders/{original['order_id']}", json={'customer_po': '  000123  '})
    assert response.status_code == 200, response.text
    assert response.json()['customer_po'] == '000123'
    for table in ('sales_orders', 'sales_order_create_receipts'):
        db_cursor.execute('SAVEPOINT trim_check')
        with pytest.raises(psycopg2.errors.CheckViolation):
            db_cursor.execute(f"UPDATE {table} SET external_order_reference='ORD28100 ' WHERE customer_id=%s", (catalog[0],))
        db_cursor.execute('ROLLBACK TO SAVEPOINT trim_check')


def legacy_body(cur, catalog, line):
    cur.execute('SELECT name FROM customers WHERE id=%s', (catalog[0],))
    customer = cur.fetchone()['name']
    cur.execute('SELECT name FROM products WHERE id=%s', (catalog[1][0],))
    product = cur.fetchone()['name']
    return {'customer_name': customer, 'lines': [{'product_name': product, **line}]}


@pytest.mark.parametrize('new_style', [False, True])
@pytest.mark.parametrize('stored_price', [None, 0], ids=['null-price', 'zero-price'])
def test_null_and_zero_prices_preserve_legacy_reads(client, db_cursor, catalog, new_style, stored_price):
    body = (payload(catalog, lines=[{'product_id': catalog[1][0], 'quantity_lb': 50, 'unit_price': stored_price}])
            if new_style else legacy_body(db_cursor, catalog, {'quantity_lb': 50, 'unit_price': stored_price}))
    created = post(client, body)
    order_id = created['order_id']
    expected = 0 if new_style and stored_price == 0 else None
    detail = client.get(f'/sales/orders/{order_id}').json()
    assert detail['lines'][0]['case_price'] == expected
    assert detail['lines'][0]['line_value'] == expected
    assert detail['totals']['total_value'] == expected
    line_id = created['lines'][0]['line_id']
    params = {'unit_price': 0} if stored_price == 0 else {'quantity_lb': 100}
    edited = client.patch(f"/sales/orders/{order_id}/lines/{line_id}/update", params=params)
    assert edited.status_code == 200, edited.text
    assert edited.json()['unit_price'] == expected
    detail = client.get(f'/sales/orders/{order_id}').json()
    assert detail['lines'][0]['case_price'] == expected
    assert detail['lines'][0]['line_value'] == expected
    assert detail['totals']['total_value'] == expected
    db_cursor.execute('SELECT unit_price, ordered_quantity FROM sales_order_lines WHERE id=%s', (line_id,))
    stored = db_cursor.fetchone()
    assert stored['unit_price'] == stored_price
    assert (stored['ordered_quantity'] is not None) == new_style


@pytest.mark.parametrize('line,status,warning', [
    ({'quantity': 2, 'unit': 'cases'}, 422, False),
    ({'quantity': 2, 'unit': 'bags'}, 422, False),
    ({'quantity': 2, 'unit': 'boxes'}, 422, False),
    ({'quantity': 2, 'unit': 'cases', 'case_weight_lb': 25}, 200, False),
    ({'quantity': 50}, 200, True),
    ({'quantity': 50, 'unit': 'lb'}, 200, False),
    ({'quantity_lb': 50}, 200, False),
])
def test_gpt_schema_shapes_keep_validation_and_warning(client, db_cursor, catalog, line, status, warning):
    schema = yaml.safe_load((ROOT / 'openapi-gpt-v3.yaml').read_text())['components']['schemas']
    body = legacy_body(db_cursor, catalog, line)
    assert set(body) <= schema['OrderCreate']['properties'].keys()
    assert set(body['lines'][0]) <= schema['OrderLineInput']['properties'].keys()
    response = client.post('/sales/orders', json=body)
    assert response.status_code == status, response.text
    if status == 422:
        assert f"case_weight_lb is required when unit is '{line['unit']}'" in response.text
    else:
        warnings = response.json()['warnings'] or []
        assert any('No unit specified' in w and 'Did you mean cases?' in w for w in warnings) == warning
        db_cursor.execute('SELECT ordered_quantity FROM sales_order_lines WHERE sales_order_id=%s', (response.json()['order_id'],))
        assert db_cursor.fetchone()['ordered_quantity'] is None


def test_new_path_keeps_no_unit_warning_and_product_case_weight_lookup(client, catalog):
    result = post(client, payload(catalog, lines=[{'product_id': catalog[1][0], 'quantity': 100, 'unit_price': 30}]))
    assert any('No unit specified' in w and 'Did you mean cases?' in w for w in result['warnings'])
    assert result['total'] == 120
    cases = post(client, payload(catalog, customer_po=None, external_order_reference='CASES'))
    assert cases['lines'][0]['quantity_lb'] == 50


@pytest.mark.parametrize('pounds,price,status', [(110, 30, 422), (110, 0, 422), (100, 30, 200), (110, None, 200)])
def test_new_cased_product_in_lb_needs_whole_cases_when_priced(client, db_cursor, catalog, pounds, price, status):
    before = counts(db_cursor)
    body = payload(catalog, lines=[{'product_id': catalog[1][0], 'quantity': pounds, 'unit': 'lb', 'unit_price': price}])
    response = client.post('/sales/orders', json=body)
    assert response.status_code == status, response.text
    if status == 422:
        assert response.json()['detail']['error_code'] == 'WHOLE_CASE_QUANTITY_REQUIRED'
        assert 'cases or whole-case pounds' in response.json()['detail']['message']
        assert counts(db_cursor) == before


def test_migration_down_removes_trim_checks_and_marker_and_up_restores_them(db_cursor):
    db_cursor.execute("SELECT name FROM migration_markers WHERE name <> '057_order_create_contract'")
    other_markers = {row['name'] for row in db_cursor.fetchall()}
    db_cursor.execute(DOWN.read_text())
    db_cursor.execute('SELECT name FROM migration_markers')
    assert {row['name'] for row in db_cursor.fetchall()} == other_markers
    db_cursor.execute("SELECT count(*) AS n FROM pg_constraint WHERE conname IN ('sales_orders_external_reference_trimmed_check','sales_order_create_receipts_reference_trimmed_check')")
    assert db_cursor.fetchone()['n'] == 0
    db_cursor.execute(UP.read_text())
    db_cursor.execute("SELECT * FROM migration_markers WHERE name='057_order_create_contract'")
    marker = db_cursor.fetchone()
    assert marker is not None
    db_cursor.execute(UP.read_text())
    db_cursor.execute("SELECT * FROM migration_markers WHERE name='057_order_create_contract'")
    assert db_cursor.fetchall() == [marker]  # rerun neither duplicates nor rewrites it
    db_cursor.execute('SELECT name FROM migration_markers')
    assert {row['name'] for row in db_cursor.fetchall()} == other_markers | {'057_order_create_contract'}
    db_cursor.execute("SELECT count(*) AS n FROM pg_constraint WHERE conname IN ('sales_orders_external_reference_trimmed_check','sales_order_create_receipts_reference_trimmed_check')")
    assert db_cursor.fetchone()['n'] == 2


@pytest.fixture
def committed_catalog(_db_connection, monkeypatch):
    # Committed fixture rows for real concurrent transactions; never a real DB.
    url = main.DATABASE_URL
    with psycopg2.connect(url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        made = seed(cur)
    extra_customers = []
    @contextmanager
    def connection():
        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='10s'")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    monkeypatch.setattr(main, 'get_db_connection', connection)
    try:
        yield made, extra_customers
    finally:
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute('SELECT id FROM customers WHERE id=%s OR name=ANY(%s)', (made[0], extra_customers))
            ids = [r[0] for r in cur.fetchall()]
            cur.execute('DELETE FROM sales_order_create_receipts WHERE customer_id=ANY(%s)', (ids,))
            cur.execute('DELETE FROM sales_orders WHERE customer_id=ANY(%s)', (ids,))
            cur.execute('DELETE FROM customers WHERE id=ANY(%s)', (ids,))
            cur.execute('DELETE FROM products WHERE id=ANY(%s)', (made[1],))


def test_concurrent_new_customer_name_reference_returns_same_receipt(committed_catalog, monkeypatch):
    catalog, extra_customers = committed_catalog
    name = 'NEW-REVIEW-' + uuid4().hex
    extra_customers.append(name)
    body = payload(catalog)
    del body['customer_id']
    body['customer_name'] = name
    gate = Barrier(2)
    lock = main._lock_order_customer_reference
    def rendezvous(cur, name, reference):
        gate.wait(timeout=10)
        lock(cur, name, reference)
    monkeypatch.setattr(main, '_lock_order_customer_reference', rendezvous)
    with TestClient(main.app) as http, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(http.post, '/sales/orders', json=body, headers={'X-API-Key': main.API_KEY}) for _ in range(2)]
        responses = [f.result(timeout=20) for f in futures]
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert responses[0].json() == responses[1].json()
    with psycopg2.connect(main.DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM customers WHERE name=%s', (name,))
        assert cur.fetchone()[0] == 1
        cur.execute('SELECT count(*) FROM sales_orders WHERE customer_id=%s', (responses[0].json()['customer_id'],))
        assert cur.fetchone()[0] == 1


def test_header_po_edit_races_create_with_same_po(committed_catalog, monkeypatch):
    catalog, _ = committed_catalog
    with TestClient(main.app) as http:
        http.headers['X-API-Key'] = main.API_KEY
        original = post(http, payload(catalog, customer_po=None, external_order_reference='ORIGINAL'))
        gate = Barrier(2)
        lock = main._lock_customer_po
        def rendezvous(cur, customer_id, po):
            gate.wait(timeout=10)
            lock(cur, customer_id, po)
        monkeypatch.setattr(main, '_lock_customer_po', rendezvous)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(http.patch, f"/sales/orders/{original['order_id']}", json={'customer_po': 'RACE-PO'}),
                       executor.submit(http.post, '/sales/orders', json=payload(catalog, customer_po='RACE-PO'))]
            responses = [f.result(timeout=20) for f in futures]
    assert sorted(r.status_code for r in responses) == [200, 409], [r.text for r in responses]
    assert next(r for r in responses if r.status_code == 409).json()['detail']['error_code'] == 'DUPLICATE_CUSTOMER_PO'
    with psycopg2.connect(main.DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM sales_orders WHERE customer_id=%s AND customer_po='RACE-PO'", (catalog[0],))
        assert cur.fetchone()[0] == 1

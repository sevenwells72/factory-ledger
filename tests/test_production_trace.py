"""Dashboard trace reads the actual corrected ledger, never consumption snapshots."""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor

import main


@pytest.fixture
def trace_client(_db_connection, db_cursor, monkeypatch):
    @contextmanager
    def transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
    monkeypatch.setattr(main, 'get_transaction', transaction)
    # No application lifespan: fixtures own the local transaction and rollback.
    return TestClient(main.app, headers={'X-API-Key': main.DASHBOARD_API_KEY})


def product(cur, name, kind, unit='lb'):
    cur.execute("INSERT INTO products (name, type, uom) VALUES (%s,%s,%s) RETURNING id", (name, kind, unit))
    pid = cur.fetchone()['id']
    cur.execute("INSERT INTO lots (product_id, lot_code, supplier_lot_code) VALUES (%s,'SAME-CODE','SUP-123') RETURNING id", (pid,))
    return pid, cur.fetchone()['id']


def event(cur, kind, stamp, lines):
    cur.execute("INSERT INTO transactions (type,timestamp,status) VALUES (%s,%s,'posted') RETURNING id", (kind, stamp))
    tid = cur.fetchone()['id']
    ids = []
    for (pid, lid), qty in lines:
        cur.execute("INSERT INTO transaction_lines (transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s) RETURNING id", (tid,pid,lid,qty))
        ids.append(cur.fetchone()['id'])
    return tid, ids


def row_trace(client, pid, kind='make', day='2026-09-16'):
    response = client.get('/dashboard/api/production/trace', params={
        'product_id': pid, 'kind': kind, 'start_date': day, 'end_date': day})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.db
def test_trace_void_and_amended_consumption(trace_client, db_cursor):
    batch = product(db_cursor, 'Trace granola', 'batch')
    oats = product(db_cursor, 'Trace oats', 'ingredient', '50 lb bag')
    tid, lines = event(db_cursor, 'make', '2026-09-17 01:30', [(batch, 20), (oats, -20)])
    # UTC 01:30 belongs to the previous ET calendar day.
    main._append_transaction_line_correction(db_cursor, lines[1], {'quantity_lb': -18.1234}, 'correct usage', 'test')
    data = row_trace(trace_client, batch[0])
    assert len(data['transactions']) == 1
    consumed = data['transactions'][0]['consumed'][0]
    assert consumed['quantity'] == 18.1234
    assert consumed['unit'] == 'lb'
    assert consumed['supplier_lot_code'] == 'SUP-123'
    assert data['subtotals'][0]['quantity'] == 18.1234
    assert not row_trace(trace_client, batch[0], day='2026-09-17')['transactions']
    main._append_transaction_correction(db_cursor, tid, 'void', 'duplicate', {}, 'test')
    assert not row_trace(trace_client, batch[0])['transactions']
    result = trace_client.get('/dashboard/api/production/trace', params={'lot_id':batch[1]})
    assert result.status_code == 200
    assert result.json()['transactions'] == []


@pytest.mark.db
def test_packed_lots_prior_day_batch_and_packaging(trace_client, db_cursor):
    batch = product(db_cursor, 'Trace bulk', 'batch')
    other_batch = product(db_cursor, 'Trace other bulk', 'batch')
    oats = product(db_cursor, 'Trace ingredient', 'ingredient')
    bags = product(db_cursor, 'Trace bags', 'ingredient', 'each')
    finished = product(db_cursor, 'Trace finished', 'finished')
    made, _ = event(db_cursor, 'make', '2026-09-14 15:00', [(batch, 30), (oats, -30)])
    event(db_cursor, 'make', '2026-09-14 16:00', [(other_batch, 40), (oats, -40)])
    packed, _ = event(db_cursor, 'pack', '2026-09-16 15:00', [(finished, 15), (batch, -15), (bags, -2)])
    event(db_cursor, 'pack', '2026-09-16 16:00', [(finished, 10), (batch, -10), (bags, -1)])
    data = row_trace(trace_client, finished[0], 'pack')
    assert len(data['transactions']) == 2
    assert data['transactions'][0]['transaction_id'] == packed
    totals = {item['product_id']: item for item in data['subtotals']}
    assert totals[batch[0]]['quantity'] == 25
    assert totals[bags[0]]['quantity'] == 3
    assert totals[bags[0]]['unit'] == 'each'
    result = trace_client.get('/dashboard/api/production/trace', params={'lot_id':batch[1]})
    assert result.status_code == 200
    assert [t['transaction_id'] for t in result.json()['transactions']] == [made]
    assert result.json()['subtotals'][0]['quantity'] == 30
    main._append_transaction_correction(db_cursor, packed, 'void', 'duplicate pack', {}, 'test')
    assert len(row_trace(trace_client, finished[0], 'pack')['transactions']) == 1


@pytest.mark.db
def test_trace_auth_and_invalid_scopes(trace_client):
    path = '/dashboard/api/production/trace'
    assert trace_client.get(path, params={'lot_id':99999999}).status_code == 200
    assert TestClient(main.app).get(path, params={'lot_id':1}).status_code == 401
    for params in ({}, {'lot_id':1, 'product_id':1}, {'lot_id':1, 'kind':'pack'},
                   {'product_id':1, 'start_date':'2026-09-17', 'end_date':'2026-09-16'},
                   {'product_id':1, 'start_date':'2026-08-01', 'end_date':'2026-09-17'}):
        assert trace_client.get(path, params=params).status_code == 400
    assert trace_client.get(path, params={'lot_id':-1}).status_code == 422


@pytest.mark.db
def test_amended_output_lot_uses_current_lines(trace_client, db_cursor):
    batch = product(db_cursor, 'Trace amended batch', 'batch')
    oats = product(db_cursor, 'Trace amended oats', 'ingredient')
    db_cursor.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, 'CORRECTED-LOT') RETURNING id",
        (batch[0],))
    current_lot = db_cursor.fetchone()['id']
    made, lines = event(db_cursor, 'make', '2026-09-14 15:00', [(batch, 30), (oats, -30)])
    main._append_transaction_line_correction(
        db_cursor, lines[0], {'lot_id': current_lot}, 'correct output lot', 'test')
    main._append_transaction_line_correction(
        db_cursor, lines[1], {'quantity_lb': -28}, 'correct consumed weight', 'test')

    old = trace_client.get('/dashboard/api/production/trace', params={'lot_id': batch[1]})
    assert old.status_code == 200
    assert old.json()['transactions'] == []
    assert old.json()['subtotals'] == []
    current = trace_client.get('/dashboard/api/production/trace', params={'lot_id': current_lot})
    assert current.status_code == 200
    data = current.json()
    assert [t['transaction_id'] for t in data['transactions']] == [made]
    assert data['transactions'][0]['consumed'][0]['lot_id'] == oats[1]
    assert data['transactions'][0]['consumed'][0]['quantity'] == 28
    assert data['subtotals'][0]['quantity'] == 28

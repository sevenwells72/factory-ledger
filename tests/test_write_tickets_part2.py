"""Part-2 ticket lifecycle, pinned inputs, atomicity and actual HTTP races."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from threading import Barrier
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg2
from psycopg2.extras import RealDictCursor
import pytest

import main
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import (headers, commit, ticket_row, posted_count, error,
                                      isolated_database)  # noqa: F401

pytestmark = pytest.mark.db
ACTIONS = ['make', 'pack', 'adjust', 'found']


def seed(cur):
    token = uuid4().hex[:12].upper()
    products = {}
    for name, kind in [('ingredient', 'ingredient'), ('batch', 'batch'), ('finished', 'finished')]:
        cur.execute('''INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active)
                       VALUES (%s,%s,%s,'lb',10,5,true) RETURNING id,name''',
                    (f'WT2 {name} {token}', f'WT2-{name}-{token}', kind))
        products[name] = dict(cur.fetchone())
    cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',
                (products['batch']['id'], products['finished']['id']))
    cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,10)',
                (products['batch']['id'], products['ingredient']['id']))
    for name in ['ingredient', 'batch']:
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id,lot_code',
                    (products[name]['id'], f'WT2-{name.upper()}-{token}'))
        products[name].update(dict(cur.fetchone()))  # replace id with lot id below
        products[name]['lot_id'] = products[name].pop('id')
        cur.execute('SELECT product_id FROM lots WHERE id=%s', (products[name]['lot_id'],))
        products[name]['id'] = cur.fetchone()['product_id']
        cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive','WT2 fixture') RETURNING id")
        txn = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',
                    (txn, products[name]['id'], products[name]['lot_id']))
    return products


@pytest.fixture
def items(db_cursor):
    return seed(db_cursor)


def body(action, items):
    common = {'occurred_at': main.get_plant_now().isoformat()}
    data = {
        'make': {'product_id': items['batch']['id'], 'batches': 1},
        'pack': {'source_product_id': items['batch']['id'], 'target_product_id': items['finished']['id'], 'cases': 2},
        'adjust': {'lot_id': items['ingredient']['lot_id'], 'delta_lb': -2,
                   'reason': 'Count correction', 'reason_es': 'Corrección de conteo'},
        'found': {'product_id': items['ingredient']['id'], 'quantity': 2, 'reason_code': 'physical_count'},
    }
    return common | data[action]


def prepare(client, action, payload, key=None):
    path = '/inventory/found/prepare' if action == 'found' else f'/{action}/prepare'
    response = client.post(path, json=payload, headers=headers(key))
    assert response.status_code == 200, response.text
    return response.json()


def output_product(action, items):
    return items[{'make': 'batch', 'pack': 'finished', 'adjust': 'ingredient', 'found': 'ingredient'}[action]]['id']


def counters(cur):
    cur.execute('SELECT * FROM receipt_counters ORDER BY prefix,business_date')
    return cur.fetchall()


@pytest.mark.parametrize('action', ACTIONS)
def test_single_use_replay_receipts_and_exact_lines(client, db_cursor, items, action, actors):
    key = actors['floor']['key']
    prepared = prepare(client, action, body(action, items), key)
    assert prepared['can_commit'], prepared
    assert posted_count(db_cursor, prepared) == 0
    first = commit(client, prepared, key)
    assert first.status_code == 200, first.text
    result = first.json()
    assert result['receipt_number'].startswith({'make': 'MK-', 'pack': 'PK-', 'adjust': 'ADJ-', 'found': 'FND-'}[action])
    assert result['state_changed'] is False
    assert commit(client, prepared, key).json() == result | {'replayed': True}
    assert posted_count(db_cursor, prepared) == 1
    db_cursor.execute('SELECT operator_id,receipt_number FROM transactions WHERE ticket_id=%s', (prepared['ticket_id'],))
    assert dict(db_cursor.fetchone()) == {'operator_id': actors['floor']['name'], 'receipt_number': result['receipt_number']}
    detail = client.get('/receipts/' + result['receipt_number'], headers=headers(key)).json()
    assert detail['response'] == result
    assert client.get('/receipts/by-transaction/' + str(result['transaction_id']), headers=headers(key)).json() == detail
    db_cursor.execute('SELECT product_id,lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s ORDER BY id',
                      (result['transaction_id'],))
    lines = db_cursor.fetchall()
    assert len(lines) == (2 if action in ('make', 'pack') else 1)
    if action in ('make', 'pack'):
        assert [float(l['quantity_lb']) for l in lines] == [10, -10]
        assert lines[1]['lot_id'] == prepared['draft']['input_plan'][0]['lot_id']
    assert sorted(ticket_row(db_cursor, prepared)['result_ref']['lot_ids']) == sorted({l['lot_id'] for l in lines})
    db_cursor.execute('SELECT count(*) AS n FROM trace_events WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['n'] == 1
    error(commit(client, prepared, actors['office']['key']), 403, 'TICKET_WRONG_USER')


@pytest.mark.parametrize('action', ACTIONS)
@pytest.mark.parametrize('source,minutes', [('mcp', 10), ('dashboard', 30)])
def test_expiry_and_supersession(client, db_cursor, items, action, source, minutes):
    payload = body(action, items) | {'client_source': source}
    first = prepare(client, action, payload)
    second = prepare(client, action, payload)
    error(commit(client, first), 409, 'TICKET_NOT_COMMITTABLE')
    row = ticket_row(db_cursor, second)
    assert row['expires_at'] - row['prepared_at'] == timedelta(minutes=minutes)
    db_cursor.execute("UPDATE write_tickets SET expires_at=now()-interval '1 minute' WHERE id=%s", (second['ticket_id'],))
    error(commit(client, second), 409, 'TICKET_EXPIRED')
    assert ticket_row(db_cursor, second)['status'] == 'expired'
    assert posted_count(db_cursor, second) == 0


@pytest.mark.parametrize('action', ACTIONS)
def test_tamper_and_client_cannot_replace_payload(client, db_cursor, items, action):
    prepared = prepare(client, action, body(action, items))
    error(commit(client, prepared, payload_hash='bad'), 409, 'TICKET_PAYLOAD_MISMATCH')
    assert commit(client, prepared, quantity=500).status_code == 422
    assert commit(client, prepared, client_source='api').status_code == 422
    db_cursor.execute("UPDATE write_tickets SET payload=payload || '{\"backfill\":true}'::jsonb WHERE id=%s", (prepared['ticket_id'],))
    error(commit(client, prepared), 409, 'TICKET_PAYLOAD_MISMATCH')
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('action', ACTIONS)
def test_archive_revalidation_is_terminal_without_stock_or_counter_changes(client, db_cursor, items, action):
    prepared = prepare(client, action, body(action, items))
    before = counters(db_cursor)
    db_cursor.execute('UPDATE products SET active=false WHERE id=%s', (output_product(action, items),))
    result = commit(client, prepared)
    error(result, 409, 'TICKET_STALE')
    assert result.json()['detail']['blockers'][0]['code'] == 'PRODUCT_NOT_FOUND'
    assert ticket_row(db_cursor, prepared)['status'] == 'rejected'
    assert posted_count(db_cursor, prepared) == 0
    assert counters(db_cursor) == before


@pytest.mark.parametrize('action', ACTIONS)
def test_duplicate_warning_cross_actor_ack_and_effective_void(client, db_cursor, items, action, actors):
    payload = body(action, items)
    first = prepare(client, action, payload, actors['office']['key'])
    posted = commit(client, first, actors['office']['key'])
    assert posted.status_code == 200, posted.text
    second = prepare(client, action, payload, actors['floor']['key'])
    warning = next(w for w in second['warnings'] if w['code'] == 'POSSIBLE_DUPLICATE')
    assert warning['refs']['receipt_number'] == posted.json()['receipt_number']
    assert warning['refs']['operator_id'] == actors['office']['name']
    error(commit(client, second, actors['floor']['key']), 409, 'WARNING_NOT_ACKNOWLEDGED')
    assert commit(client, second, actors['floor']['key'], acknowledged_warnings=['POSSIBLE_DUPLICATE']).status_code == 200
    for prepared in (first, second):
        txn = ticket_row(db_cursor, prepared)['response']['transaction_id']
        assert client.post('/void/' + str(txn), json={'reason': 'test'}, headers=headers()).status_code == 200
    assert prepare(client, action, payload)['warnings'] == []


@pytest.mark.parametrize('action', ACTIONS)
def test_duplicate_window_excludes_old_entries(client, db_cursor, items, action):
    payload = body(action, items)
    first = prepare(client, action, payload)
    assert commit(client, first).status_code == 200
    db_cursor.execute("""CREATE TEMP VIEW ledger_current_transactions AS
        SELECT id,type,effective_status,operator_id,created_at-interval '3 hours' AS created_at
        FROM public.ledger_current_transactions""")
    assert prepare(client, action, payload)['warnings'] == []


@pytest.mark.parametrize('action', ['make', 'pack', 'found'])
@pytest.mark.parametrize('winner', ['legacy', 'ticket'])
def test_lot_code_race_rejects_without_ledger_or_counter(client, db_cursor, items, action, winner):
    payload = body(action, items)
    first = prepare(client, action, payload)
    assert first['can_commit'], first
    code = first['draft'].get('lot_code') or first['draft']['output_lot_code']
    if winner == 'ticket':
        # Distinct payload avoids superseding the first ticket.
        second = prepare(client, action, payload | {'occurred_at': (main.get_plant_now()-timedelta(seconds=1)).isoformat()})
        assert commit(client, second).status_code == 200
    else:
        legacy = {'make': {'product_name': items['batch']['name'], 'batches': 1, 'mode': 'commit'},
                  'pack': {'source_product': items['batch']['name'], 'target_product': items['finished']['name'], 'cases': 2, 'mode': 'commit'},
                  'found': {'product_id': items['ingredient']['id'], 'quantity': 2, 'reason_code': 'physical_count'}}[action]
        path = '/inventory/found' if action == 'found' else '/' + action
        response = client.post(path, json=legacy, headers=headers())
        assert response.status_code == 200, response.text
    before = counters(db_cursor)
    result = commit(client, first)
    error(result, 409, 'TICKET_STALE')
    assert result.json()['detail']['blockers'][0]['code'] == 'LOT_CODE_TAKEN'
    assert counters(db_cursor) == before
    assert posted_count(db_cursor, first) == 0
    fresh = prepare(client, action, payload)
    if action in ('make', 'found'):
        assert fresh['draft']['lot_code'] != code
    else:
        assert fresh['draft']['lot_exists'] is True  # inherited code is explicitly shown as existing
    assert commit(client, fresh, acknowledged_warnings=['POSSIBLE_DUPLICATE']).status_code == 200


@pytest.mark.parametrize('action', ['make', 'pack', 'adjust'])
def test_merged_input_or_adjust_lot_rejected(client, db_cursor, items, action):
    prepared = prepare(client, action, body(action, items))
    lid = items['batch' if action == 'pack' else 'ingredient']['lot_id']
    db_cursor.execute("UPDATE lots SET status='merged' WHERE id=%s", (lid,))
    result = commit(client, prepared)
    error(result, 409, 'TICKET_STALE')
    assert result.json()['detail']['blockers'][0]['code'] == 'LOT_MERGED'
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('action', ['make', 'pack'])
@pytest.mark.parametrize('remaining', [50, 1])
def test_stock_revalidation_keeps_pinned_inputs(client, db_cursor, items, action, remaining):
    prepared = prepare(client, action, body(action, items))
    lid = prepared['draft']['input_plan'][0]['lot_id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('adjust','Competing count') RETURNING id")
    txn = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) SELECT %s,product_id,id,%s FROM lots WHERE id=%s', (txn, remaining-100, lid))
    result = commit(client, prepared)
    if remaining == 1:
        error(result, 409, 'TICKET_STALE')
        assert posted_count(db_cursor, prepared) == 0
    else:
        assert result.status_code == 200, result.text
        assert result.json()['state_changed'] is True
        db_cursor.execute('SELECT lot_id FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0', (result.json()['transaction_id'],))
        assert db_cursor.fetchone()['lot_id'] == lid


@pytest.mark.parametrize('action', ACTIONS)
def test_failure_after_trace_rolls_back_all_writes(client, db_cursor, items, action, monkeypatch):
    prepared = prepare(client, action, body(action, items))
    before = counters(db_cursor)
    original = main.emit_trace_event
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected post-trace failure')
    monkeypatch.setattr(main, 'emit_trace_event', crash)
    with pytest.raises(RuntimeError, match='injected'):
        commit(client, prepared)
    assert posted_count(db_cursor, prepared) == 0
    assert ticket_row(db_cursor, prepared)['status'] == 'prepared'
    assert counters(db_cursor) == before
    monkeypatch.setattr(main, 'emit_trace_event', original)
    assert commit(client, prepared).status_code == 200


@pytest.mark.parametrize('action', ACTIONS)
def test_concurrent_double_commit(isolated_database, monkeypatch, action):
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        items = seed(cur)
    gate = Barrier(2)
    racing = False
    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='10s'")
            if racing:
                gate.wait(timeout=10)
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    with TestClient(main.app) as http:
        prepared = prepare(http, action, body(action, items))
        racing = True
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(commit, http, prepared) for _ in range(2)]
            results = [f.result(timeout=20) for f in futures]
        racing = False
        assert [r.status_code for r in results] == [200, 200], [r.text for r in results]
        data = [r.json() for r in results]
        assert data[0]['receipt_number'] == data[1]['receipt_number']
        assert sorted(r['replayed'] for r in data) == [False, True]
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, prepared) == 1


@pytest.mark.parametrize('action', ACTIONS)
def test_ids_only_and_strict_request_schema(client, items, action):
    path = '/inventory/found/prepare' if action == 'found' else f'/{action}/prepare'
    payload = body(action, items)
    error(client.post(path, json=payload | {'product_name': 'Guess'}, headers=headers()), 422, 'IDS_REQUIRED')
    for extra in ({'mode': 'commit'}, {'input_plan': []}, {'performed_by': 'Somebody else'}):
        assert client.post(path, json=payload | extra, headers=headers()).status_code == 422
    field = {'make': 'batches', 'pack': 'cases', 'adjust': 'delta_lb', 'found': 'quantity'}[action]
    assert client.post(path, json=payload | {field: 0}, headers=headers()).status_code == 422


@pytest.mark.parametrize('action', ACTIONS)
def test_blocked_time_never_posts(client, db_cursor, items, action):
    payload = body(action, items) | {'occurred_at': (main.get_plant_now()+timedelta(hours=2)).isoformat()}
    prepared = prepare(client, action, payload)
    assert not prepared['can_commit']
    error(commit(client, prepared), 409, 'TICKET_STALE')
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('action', ['make', 'pack', 'found'])
def test_existing_output_lot_rename_does_not_create_replacement(client, db_cursor, items, action):
    pid = output_product(action, items)
    code = 'WT2-EXISTING-' + uuid4().hex[:10].upper()
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (pid, code))
    lid = db_cursor.fetchone()['id']
    field = 'target_lot_code' if action == 'pack' else 'lot_code'
    prepared = prepare(client, action, body(action, items) | {field: code})
    assert prepared['draft']['lot_exists'] is True
    db_cursor.execute('UPDATE lots SET lot_code=%s WHERE id=%s', (code+'-RENAMED', lid))
    response = commit(client, prepared)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'LOT_IDENTITY_CHANGED'
    assert posted_count(db_cursor, prepared) == 0


def test_adjust_pins_id_when_another_lot_takes_old_code(client, db_cursor, items):
    prepared = prepare(client, 'adjust', body('adjust', items))
    current = items['ingredient']
    db_cursor.execute('UPDATE lots SET lot_code=%s WHERE id=%s', (current['lot_code']+'-RENAMED', current['lot_id']))
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s)', (current['id'], current['lot_code']))
    response = commit(client, prepared)
    assert response.status_code == 200, response.text
    assert response.json()['lot_id'] == current['lot_id']
    assert response.json()['state_changed'] is True
    db_cursor.execute('SELECT lot_id FROM transaction_lines WHERE transaction_id=%s', (response.json()['transaction_id'],))
    assert db_cursor.fetchone()['lot_id'] == current['lot_id']


def test_changed_make_formula_requires_fresh_draft(client, db_cursor, items):
    prepared = prepare(client, 'make', body('make', items))
    db_cursor.execute('UPDATE batch_formulas SET quantity_lb=12 WHERE product_id=%s', (items['batch']['id'],))
    response = commit(client, prepared)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'INPUT_PLAN_CHANGED'
    assert posted_count(db_cursor, prepared) == 0


def test_make_pins_multiple_fifo_lots_even_after_new_stock_arrives(client, db_cursor, items):
    ing = items['ingredient']
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('adjust') RETURNING id")
    txn = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-95)',
                      (txn, ing['id'], ing['lot_id']))
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (ing['id'], ing['lot_code']+'-SECOND'))
    second = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,10)',
                      (txn, ing['id'], second))
    prepared = prepare(client, 'make', body('make', items))
    assert len(prepared['draft']['input_plan']) == 2
    # Replenishing the oldest lot must not silently remove the second lot from the post.
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,10)',
                      (txn, ing['id'], ing['lot_id']))
    response = commit(client, prepared)
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0 ORDER BY lot_id',
                      (response.json()['transaction_id'],))
    assert [(r['lot_id'], float(r['quantity_lb'])) for r in db_cursor.fetchall()] == [(ing['lot_id'], -5), (second, -5)]


@pytest.mark.parametrize('shortage', [False, True])
def test_pack_add_ins_are_pinned_revalidated_and_traced(client, db_cursor, items, shortage):
    cur = db_cursor
    cur.execute("INSERT INTO products(name,type,odoo_code,uom) VALUES (%s,'batch',%s,'lb') RETURNING id",
                ('WT2 intermediate '+uuid4().hex, 'WT2-'+uuid4().hex))
    intermediate = cur.fetchone()['id']
    cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (intermediate, items['finished']['id']))
    for pid, qty in [(items['batch']['id'], 10), (items['ingredient']['id'], 2)]:
        cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,%s)', (intermediate, pid, qty))
    prepared = prepare(client, 'pack', body('pack', items))
    assert prepared['can_commit'], prepared
    assert len(prepared['draft']['input_plan']) == 2
    if shortage:
        cur.execute("INSERT INTO transactions(type) VALUES ('adjust') RETURNING id")
        txn = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-99)',
                    (txn, items['ingredient']['id'], items['ingredient']['lot_id']))
    response = commit(client, prepared)
    if shortage:
        error(response, 409, 'TICKET_STALE')
        assert posted_count(cur, prepared) == 0
    else:
        assert response.status_code == 200, response.text
        cur.execute('SELECT quantity_lb FROM transaction_lines WHERE transaction_id=%s ORDER BY id', (response.json()['transaction_id'],))
        assert [float(r['quantity_lb']) for r in cur.fetchall()] == [10, -10, -2]
        assert response.json()['add_in_ingredients_consumed'][0]['lot_id'] == items['ingredient']['lot_id']


@pytest.mark.parametrize('action', ['make', 'pack', 'found'])
def test_creation_guard_catches_lot_taken_after_validation(client, db_cursor, items, action, monkeypatch):
    prepared = prepare(client, action, body(action, items))
    original = main.find_or_create_lot
    def competing_insert(cur, pid, code, *args, **kwargs):
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) ON CONFLICT DO NOTHING', (pid, code))
        return original(cur, pid, code, *args, **kwargs)
    monkeypatch.setattr(main, 'find_or_create_lot', competing_insert)
    before = counters(db_cursor)
    response = commit(client, prepared)
    error(response, 409, 'TICKET_STALE')
    assert response.json()['detail']['blockers'][0]['code'] == 'LOT_CODE_TAKEN'
    assert counters(db_cursor) == before
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('action', ['make', 'pack', 'found'])
def test_two_tickets_race_for_generated_code(isolated_database, monkeypatch, action):
    with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        items = seed(cur)
    gate = Barrier(2)
    racing = False
    @contextmanager
    def connection():
        with psycopg2.connect(isolated_database) as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout='10s'")
            if racing:
                gate.wait(timeout=10)
            yield conn
    monkeypatch.setattr(main, 'get_db_connection', connection)
    with TestClient(main.app) as http:
        payload = body(action, items)
        first = prepare(http, action, payload)
        second = prepare(http, action, payload | {'occurred_at': (main.get_plant_now()-timedelta(seconds=1)).isoformat()})
        racing = True
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(commit, http, prepared) for prepared in (first, second)]
            results = [f.result(timeout=20) for f in futures]
        racing = False
        assert sorted(r.status_code for r in results) == [200, 409], [r.text for r in results]
        rejected = next(r for r in results if r.status_code == 409)
        assert rejected.json()['detail']['blockers'][0]['code'] == 'LOT_CODE_TAKEN'
        with psycopg2.connect(isolated_database) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            assert posted_count(cur, first) + posted_count(cur, second) == 1


def test_make_duplicate_matches_stored_decimal_yield(client, db_cursor, items):
    db_cursor.execute('UPDATE products SET default_batch_lb=0.1,yield_multiplier=1.2345 WHERE id=%s', (items['batch']['id'],))
    payload = body('make', items) | {'batches': 3}
    prepared = prepare(client, 'make', payload)
    response = commit(client, prepared)
    assert response.status_code == 200, response.text
    duplicate = prepare(client, 'make', payload)
    assert duplicate['warnings'][0]['refs']['receipt_number'] == response.json()['receipt_number']

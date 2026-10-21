"""Real competing transactions must not reopen a confirmed Charge/Product check.

The check and the production confirmation handler use separate committed
connections. Each test proves the second writer is waiting on the first
writer's product lock before allowing the first transaction to commit.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import pytest
from starlette.requests import Request

import charge_setting
import main


@pytest.fixture
def race_item(_db_connection):
    connections = [psycopg2.connect(os.environ['TEST_DATABASE_URL'],
                                  application_name='charge-race-' + uuid4().hex)
                   for _ in range(3)]
    checker, confirmer, observer = connections
    observer.autocommit = True
    token = uuid4().hex
    schema = 'charge_race_' + token
    with observer.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        # Private copies retain the real columns, constraints and unique open
        # question index. Dropping the schema also removes append-only audit
        # evidence without weakening its trigger on the suite's public table.
        for table in ('actors', 'products', 'exceptions', 'actor_write_audit'):
            cur.execute(sql.SQL('CREATE TABLE {}.{} (LIKE public.{} INCLUDING ALL)').format(
                sql.Identifier(schema), sql.Identifier(table), sql.Identifier(table)))
        cur.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(schema)))
        cur.execute("INSERT INTO actors(name, role, key_hash, active) VALUES (%s, 'owner', %s, true) RETURNING id",
                    ('Charge race ' + token, token))
        actor = {'id': cur.fetchone()['id'], 'name': 'Charge race ' + token, 'role': 'owner'}
        cur.execute("""INSERT INTO products(name, type, uom, active, is_service)
                       VALUES (%s, 'finished', 'lb', true, false) RETURNING id""",
                    ('Coffee race ' + str(uuid4().int),))
        product_id = cur.fetchone()['id']
    for connection in (checker, confirmer):
        with connection.cursor() as cur:
            cur.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(schema)))
        connection.commit()
    try:
        yield checker, confirmer, observer, product_id, actor
    finally:
        try:
            checker.close()
            confirmer.close()
            with observer.cursor() as cur:
                cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        finally:
            observer.close()


def _confirmation(monkeypatch, connection, product_id, actor):
    route = next(r for r in main.app.routes if getattr(r, 'path', '') == '/catalog/items/{product_id}/confirm-kind')
    request = Request({'type': 'http', 'method': 'POST', 'path': f'/catalog/items/{product_id}/confirm-kind',
                       'headers': [], 'route': route, 'state': {'actor': actor, 'key_kind': 'actor'}})

    @contextmanager
    def transaction():
        try:
            with connection.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    monkeypatch.setattr(main, 'get_transaction', transaction)
    return lambda: route.endpoint(product_id, charge_setting.ItemKindChoice(item_kind='product'), request)


def _assert_waiting(observer, waiter, holder, future):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        with observer.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT query, pg_blocking_pids(pid) AS blockers FROM pg_stat_activity WHERE pid=%s',
                        (waiter.get_backend_pid(),))
            row = cur.fetchone()
        if holder.get_backend_pid() in row['blockers']:
            assert 'FROM products WHERE id=' in row['query'] and 'FOR UPDATE' in row['query']
            return
        if future.done():
            future.result()  # Surface an endpoint failure rather than hiding it as a missing lock.
            pytest.fail('The competing operation completed without waiting for the product lock')
        sleep(0.01)
    pytest.fail('The competing operation did not wait on the expected product lock')


def _assert_answer_stays_saved(observer, product_id, actor):
    with observer.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute('SELECT is_service, charge_setting_confirmed_at, charge_setting_confirmed_by_actor_id '
                    'FROM products WHERE id=%s', (product_id,))
        item = cur.fetchone()
        assert item['is_service'] is False  # Coffee deliberately remains a name/setting mismatch.
        assert item['charge_setting_confirmed_at'] is not None
        assert item['charge_setting_confirmed_by_actor_id'] == actor['id']
        cur.execute("SELECT status, resolution_kind FROM exceptions WHERE product_id=%s", (product_id,))
        assert all(row['status'] == 'resolved' and row['resolution_kind'] == 'confirmed_product'
                   for row in cur.fetchall())
        assert charge_setting.check(cur, product_id) is None


@pytest.mark.db
@pytest.mark.parametrize('existing_question', [False, True])
def test_confirmation_waits_for_check_then_closes_its_question(race_item, monkeypatch, existing_question):
    checker, confirmer, observer, product_id, actor = race_item
    confirm = _confirmation(monkeypatch, confirmer, product_id, actor)
    if existing_question:
        with checker.cursor(cursor_factory=RealDictCursor) as cur:
            charge_setting.check(cur, product_id)
        checker.commit()
    with checker.cursor(cursor_factory=RealDictCursor) as cur:
        check_id = charge_setting.check(cur, product_id)
    assert check_id is not None
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(confirm)
        try:
            _assert_waiting(observer, confirmer, checker, pending)
        finally:
            checker.commit()
        result = pending.result(timeout=5)
    assert result['closed_exception_ids'] == [check_id]
    assert result['item']['confirmed'] is True
    _assert_answer_stays_saved(observer, product_id, actor)


@pytest.mark.db
@pytest.mark.parametrize('existing_question', [False, True])
def test_nightly_check_waits_for_confirmation_and_reads_the_saved_answer(race_item, monkeypatch, existing_question):
    checker, confirmer, observer, product_id, actor = race_item
    if existing_question:
        with checker.cursor(cursor_factory=RealDictCursor) as cur:
            charge_setting.check(cur, product_id)
        checker.commit()
    confirm = _confirmation(monkeypatch, confirmer, product_id, actor)
    saved, release = Event(), Event()
    record_write = main._record_actor_write

    def pause_after_saved_answer(cur, request, table, row_id):
        record_write(cur, request, table, row_id)
        if table == 'products':
            saved.set()
            assert release.wait(timeout=10), 'The test did not release the confirmation transaction'

    monkeypatch.setattr(main, '_record_actor_write', pause_after_saved_answer)

    def nightly_check():
        try:
            with checker.cursor(cursor_factory=RealDictCursor) as cur:
                # Reproduce the nightly candidate read while the answer is
                # uncommitted, even when a question was already open.
                cur.execute('SELECT id FROM products WHERE id=%s AND charge_setting_confirmed_at IS NULL', (product_id,))
                assert cur.fetchone()['id'] == product_id
                return charge_setting.check(cur, product_id)
        finally:
            checker.commit()

    with ThreadPoolExecutor(max_workers=2) as workers:
        confirmation = workers.submit(confirm)
        checking = None
        try:
            assert saved.wait(timeout=5), 'The confirmation did not save its answer'
            checking = workers.submit(nightly_check)
            _assert_waiting(observer, checker, confirmer, checking)
        finally:
            release.set()
        assert confirmation.result(timeout=5)['item']['confirmed'] is True
        assert checking.result(timeout=5) is None
    _assert_answer_stays_saved(observer, product_id, actor)

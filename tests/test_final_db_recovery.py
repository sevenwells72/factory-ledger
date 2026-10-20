"""Connection faults: discard broken sockets, retry only marked reads, never writes."""
from unittest.mock import Mock
import psycopg2
import pytest
from fastapi import HTTPException
import main


@pytest.mark.parametrize('error', [psycopg2.OperationalError('server closed'), psycopg2.InterfaceError('closed')])
def test_failed_connection_is_discarded_without_rollback_masking(monkeypatch, error):
    connection = Mock(); connection.rollback.side_effect = RuntimeError('closed rollback')
    pool = Mock(); pool.getconn.return_value = connection
    monkeypatch.setattr(main, 'db_pool', pool)
    work = []
    with pytest.raises(type(error), match=str(error)):
        with main.get_db_connection():
            work.append('write attempted')
            raise error
    assert work == ['write attempted']
    pool.getconn.assert_called_once()
    pool.putconn.assert_called_once_with(connection, close=True)
    connection.rollback.assert_not_called()
    connection.commit.assert_not_called()


def test_read_retry_is_bounded_and_bilingual(caplog):
    calls = []
    @main.retry_database_read
    def read():
        calls.append(1)
        raise psycopg2.OperationalError('server closed')
    with pytest.raises(HTTPException) as caught:
        read()
    assert len(calls) == 2
    assert caught.value.status_code == 503
    assert caught.value.detail['error_code'] == 'DATABASE_TEMPORARILY_UNAVAILABLE'
    assert 'conexión' in caught.value.detail['message_es']
    assert caplog.text.count('DB_READ_RETRY_ONCE') == 1


def test_retry_does_not_replay_sql_or_business_errors():
    calls = []
    @main.retry_database_read
    def read():
        calls.append(1)
        raise psycopg2.ProgrammingError('bad SQL')
    with pytest.raises(psycopg2.ProgrammingError):
        read()
    assert len(calls) == 1


def test_successful_retry_uses_second_read_result():
    calls = []
    @main.retry_database_read
    def read():
        calls.append(1)
        if len(calls) == 1:
            raise psycopg2.InterfaceError('socket closed')
        return {'orders': ['SO-261016-001']}
    assert read() == {'orders': ['SO-261016-001']}
    assert len(calls) == 2


def test_session_capacity_failure_is_logged_and_read_retries_once(monkeypatch, caplog):
    pool = Mock()
    pool.getconn.side_effect = psycopg2.OperationalError('FATAL: (EMAXCONNSESSION) max clients reached in session mode')
    monkeypatch.setattr(main, 'db_pool', pool)
    with pytest.raises(HTTPException) as caught:
        main.search_products(q='Oats', limit=5, _=True)
    assert caught.value.status_code == 503
    assert pool.getconn.call_count == 2
    assert caplog.text.count('DB_POOL_CAPACITY') == 2
    assert caplog.text.count('DB_READ_RETRY_ONCE') == 1
    pool.putconn.assert_not_called()


def test_local_pool_capacity_is_bounded_too(monkeypatch):
    database_pool = Mock()
    database_pool.getconn.side_effect = main.pool.PoolError('connection pool exhausted')
    monkeypatch.setattr(main, 'db_pool', database_pool)
    # The timeout itself reaches the app, which answers 503 DATABASE_BUSY
    # (PR #141); getconn already waited, so a retry would double the wait.
    with pytest.raises(main.pool.PoolError, match='connection pool exhausted'):
        main.search_products(q='Oats', limit=5, _=True)
    assert database_pool.getconn.call_count == 1


@pytest.mark.parametrize('value,display', [(88.80000000000001, '88.8'), (0.5, '0.5'),
    (1.005, '1.01'), (1234.567, '1,234.57'), (-0.00001, '0'), (100, '100')])
def test_worker_quantity_sentences_keep_two_decimals(value, display):
    assert main._fmt_quantity(value) == display

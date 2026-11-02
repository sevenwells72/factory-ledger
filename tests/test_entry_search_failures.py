"""Entry-screen searches when the database is slow or fails: a JSON error, never a hang
or a partial list, and the next search still works (PR #173 staging follow-up)."""
import psycopg2
import pytest
import main
import resolution
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import headers

pytestmark = pytest.mark.db

SEARCHES = [
    ('entry_make_products', 'GET', '/entry/make-products', {'params': {'query': 'classic'}}),
    ('entry_lots', 'GET', '/entry/lots', {'params': {'query': 'OPEN'}}),
    ('resolve', 'POST', '/resolve', {'json': {'kind': 'product', 'query': 'oats', 'context': {}, 'limit': 8}}),
]


def call(client, method, path, kwargs, key):
    return client.request(method, path, headers=headers(key), **kwargs)


def timed_out(cur, *args, **kwargs):
    # A real statement timeout from PostgreSQL, as a slow plan or lock wait would raise.
    cur.execute("SET LOCAL statement_timeout = '50ms'")
    cur.execute('SELECT pg_sleep(1)')


@pytest.mark.parametrize('name,method,path,kwargs', SEARCHES, ids=[s[2] for s in SEARCHES])
def test_search_statement_timeout_is_a_json_error_and_the_next_search_works(client, actors, monkeypatch, name, method, path, kwargs):
    real = getattr(resolution, name)
    monkeypatch.setattr(resolution, name, timed_out)
    failed = call(client, method, path, kwargs, actors['floor']['key'])
    assert failed.status_code >= 500, failed.text
    body = failed.json()
    # Current server contract (app-wide handler): a 5xx JSON failure, never a partial list.
    # The raw database text can appear in `error`; the Make screen shows its own wording
    # for any 5xx (tests/test_pilot_entry_ui.js), so workers never see it.
    assert body['success'] is False and body.get('error_code'), body
    assert 'products' not in body and 'lots' not in body and 'candidates' not in body
    assert 'pg_sleep' not in failed.text, 'no SQL text in the error'
    monkeypatch.setattr(resolution, name, real)
    again = call(client, method, path, kwargs, actors['floor']['key'])
    assert again.status_code == 200, again.text


@pytest.mark.parametrize('name,method,path,kwargs', SEARCHES, ids=[s[2] for s in SEARCHES])
def test_search_database_error_is_a_json_error_with_the_dashboard_key_too(client, monkeypatch, name, method, path, kwargs):
    def broken(*args, **kwargs):
        raise psycopg2.OperationalError('server closed the connection unexpectedly')
    monkeypatch.setattr(resolution, name, broken)
    failed = call(client, method, path, kwargs, main.DASHBOARD_API_KEY)
    assert failed.status_code >= 500, failed.text
    body = failed.json()
    assert body['success'] is False and body.get('error_code'), body
    assert 'products' not in body and 'lots' not in body and 'candidates' not in body


def test_search_query_length_limits_answer_422_not_a_server_error(client, actors):
    key = actors['floor']['key']
    assert client.get('/entry/make-products', params={'query': 'x' * 201}, headers=headers(key)).status_code == 422
    assert client.get('/entry/lots', params={'query': 'x' * 501}, headers=headers(key)).status_code == 422

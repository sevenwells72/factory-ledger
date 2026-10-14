"""Automatic reads authenticate without renewing the personal idle deadline."""
from contextlib import contextmanager
from datetime import timedelta

import pytest
import key_sessions
import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_key_sessions import pilot, sign_in  # noqa: F401

pytestmark = pytest.mark.db


def session_row(cur, token):
    cur.execute('SELECT last_seen_at,expires_at,ended_reason FROM key_actor_sessions WHERE token_hash=%s',
                (key_sessions.digest(token),))
    return dict(cur.fetchone())


def session_clock(monkeypatch, initial):
    """Move just the session check's clock; all authentication/storage uses real SQL."""
    timeline = {'now': initial}
    transaction = main.get_transaction

    class ClockCursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def __getattr__(self, name):
            return getattr(self.cursor, name)

        def execute(self, sql, params=None):
            if sql == 'SELECT clock_timestamp() AS now':
                return self.cursor.execute('SELECT %s::timestamptz AS now', (timeline['now'],))
            return self.cursor.execute(sql, params)

    @contextmanager
    def timed_transaction():
        with transaction() as cursor:
            yield ClockCursor(cursor)

    monkeypatch.setattr(main, 'get_transaction', timed_transaction)
    return timeline


@pytest.mark.parametrize('path', ['/auth/whoami', '/ledger/recent?limit=5'])
def test_background_poll_preserves_activity_and_expiry(client, actors, db_cursor, path):
    result, credential = sign_in(client, actors)
    token = result['session_token']
    before = session_row(db_cursor, token)
    response = client.get(path, headers={**credential, 'X-FL-Background': '1'})
    assert response.status_code == 200
    assert session_row(db_cursor, token) == before
    if path == '/auth/whoami':
        assert response.json()['actor']['id'] == actors['floor']['id']
        assert response.json()['key_kind'] == 'session'


@pytest.mark.parametrize('marker', [None, '0', 'true'])
def test_user_action_extends_activity_and_expiry(client, actors, db_cursor, monkeypatch, marker):
    result, credential = sign_in(client, actors)
    token = result['session_token']
    before = session_row(db_cursor, token)
    clock = session_clock(monkeypatch, before['last_seen_at'] + timedelta(minutes=170))
    if marker is not None:
        credential = {**credential, 'X-FL-Background': marker}
    assert client.get('/auth/whoami', headers=credential).status_code == 200
    after = session_row(db_cursor, token)
    assert after['last_seen_at'] == clock['now'] > before['last_seen_at']
    assert after['expires_at'] == clock['now'] + timedelta(minutes=180) > before['expires_at']
    clock['now'] = before['expires_at']
    assert client.get('/auth/whoami', headers={**credential, 'X-FL-Background': '1'}).status_code == 200
    assert session_row(db_cursor, token) == after


def test_every_minute_polling_still_expires_after_180_idle_minutes(client, actors, db_cursor, monkeypatch):
    result, credential = sign_in(client, actors)
    token = result['session_token']
    before = session_row(db_cursor, token)
    start = before['expires_at'] - timedelta(minutes=180)
    clock = session_clock(monkeypatch, start)
    background = {**credential, 'X-FL-Background': '1'}
    for minute in range(1, 180):
        clock['now'] = start + timedelta(minutes=minute)
        assert client.get('/ledger/recent?limit=5', headers=background).status_code == 200
        assert session_row(db_cursor, token) == before
    clock['now'] = start + timedelta(minutes=180)
    expired = client.get('/ledger/recent?limit=5', headers=background)
    assert expired.status_code == 401
    assert expired.json()['detail']['error_code'] == 'SESSION_EXPIRED'
    after = session_row(db_cursor, token)
    assert after['last_seen_at'] == before['last_seen_at']
    assert after['expires_at'] == before['expires_at']
    assert after['ended_reason'] == 'idle'
    assert client.get('/auth/whoami', headers=credential).status_code == 401


@pytest.mark.parametrize('change', ['inactive', 'rotated'])
def test_background_marker_does_not_bypass_live_revocation(client, actors, db_cursor, change):
    _, credential = sign_in(client, actors)
    if change == 'inactive':
        db_cursor.execute('UPDATE actors SET active=false WHERE id=%s', (actors['floor']['id'],))
    else:
        db_cursor.execute('UPDATE actors SET key_hash=%s WHERE id=%s',
                          (key_sessions.digest('replacement-test-only-key'), actors['floor']['id']))
    response = client.get('/auth/whoami', headers={**credential, 'X-FL-Background': '1'})
    assert response.status_code == 401
    assert response.json()['detail']['error_code'] == 'SESSION_EXPIRED'

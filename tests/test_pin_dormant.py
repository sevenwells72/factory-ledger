"""Default-off compatibility and explicit activation, on a fresh local DB.

The pre-A11 actor/A2/A3b test modules are unchanged from main and run flag off.
These cases additionally prove no PIN configuration or tables are needed.
"""
from contextlib import contextmanager
from pathlib import Path
import secrets
from types import SimpleNamespace

from fastapi.testclient import TestClient
import psycopg2
import pytest
from starlette.requests import Request
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import main
import pin_sessions as pins
from proxy_server import RailwayEdgeHeaders
from scripts.bootstrap_owner_pin import bootstrap
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets import isolated_database, payload, prepare, commit, headers, posted_count  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PAGES = ['index.html', 'history.html', 'runs.html', 'traceability.html',
         'process-flow.html', 'sankey.html', 'legal/eula.html', 'legal/privacy.html']


@pytest.fixture(autouse=True)
def dormant(monkeypatch):
    monkeypatch.delenv('PIN_LOGIN_ENABLED', raising=False)
    monkeypatch.delenv('PIN_PEPPER', raising=False)


@pytest.mark.parametrize('setting', [None, '', '0', 'false', 'true', 'yes', ' 1', '1'])
def test_only_explicit_one_enables(setting, monkeypatch):
    if setting is not None:
        monkeypatch.setenv('PIN_LOGIN_ENABLED', setting)
    assert pins.enabled() is (setting == '1')


@pytest.mark.parametrize('page', PAGES)
def test_all_existing_dashboard_pages_are_public_without_pepper(client, page):
    response = client.get('/dashboard/' + page)
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


@pytest.mark.parametrize('method,path', [
    ('POST', '/auth/session'), ('POST', '/auth/session/key'),
    ('GET', '/auth/session'), ('DELETE', '/auth/session'),
    ('GET', '/actors/pins'), ('POST', '/actors/1/pin'),
    ('GET', '/dashboard/pin-management.html'), ('GET', '/dashboard/pin-management.js')])
def test_pin_surfaces_are_404_even_without_credentials_or_valid_body(client, method, path):
    assert client.request(method, path).status_code == 404


def test_browser_key_and_cors_unchanged(client, monkeypatch):
    result = client.get('/auth/config')
    assert result.status_code == 200 and result.headers['cache-control'] == 'no-store'
    assert result.json()['pin_login_enabled'] is False
    # Boolean assertion prevents pytest from printing the credential on failure.
    assert bool(result.json()['dashboard_key'] == main.DASHBOARD_API_KEY)
    monkeypatch.setenv('PIN_LOGIN_ENABLED', '1')
    active = client.get('/auth/config').json()
    assert active == {'pin_login_enabled': True}


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('origin', ['https://cns-factory-ledger.netlify.app', 'https://old-dashboard.example'])
@pytest.mark.parametrize('preflight', [False, True])
def test_actual_app_cors_policy(client, monkeypatch, enabled, origin, preflight):
    monkeypatch.setenv('PIN_LOGIN_ENABLED', '1' if enabled else '0')
    request_headers = {'Origin': origin}
    if preflight:
        request_headers.update({'Access-Control-Request-Method': 'POST',
                                'Access-Control-Request-Headers': 'X-API-Key,Content-Type'})
    response = client.request('OPTIONS' if preflight else 'GET', '/auth/whoami', headers=request_headers)
    allowed = not enabled or origin == 'https://cns-factory-ledger.netlify.app'
    if preflight:
        assert response.status_code == (200 if allowed else 400)
    expected_origin = origin if allowed and (enabled or preflight) else '*' if allowed else None
    assert response.headers.get('access-control-allow-origin') == expected_origin
    assert response.headers.get('access-control-allow-credentials') == 'true'
    assert response.headers.get('access-control-expose-headers') == ('Retry-After, Content-Disposition' if enabled and not preflight else None)


def test_legacy_and_actor_keys_keep_main_permissions(client, actors, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dormant auth must not access PIN/session storage')
    monkeypatch.setattr(pins, 'resolve', forbidden)
    monkeypatch.setattr(pins, 'pepper', forbidden)
    for key, kind, person in [(main.API_KEY, 'legacy_ledger', None),
                             (main.DASHBOARD_API_KEY, 'legacy_dashboard', None),
                             *[(p['key'], 'actor', p) for label, p in actors.items() if label != 'retired']]:
        response = client.get('/auth/whoami', headers=headers(key))
        assert response.status_code == 200
        assert response.json()['key_kind'] == kind
        assert response.json()['actor'] == ({k:person[k] for k in ('id','name','role')} if person else None)
    assert client.get('/auth/whoami', headers=headers(actors['retired']['key'])).status_code == 403
    assert client.get('/auth/whoami', headers=headers(pins.SESSION_PREFIX + secrets.token_urlsafe(32))).status_code == 403
    for action in ('approve', 'reject'):
        path = '/exceptions/2147483000/' + action
        body = {'note': 'Review parity'} | ({'resolution_kind': 'declined'} if action == 'reject' else {})
        assert client.post(path, json=body, headers=headers(actors['owner']['key'])).status_code == 404
        assert client.post(path, json=body, headers=headers(actors['floor']['key'])).status_code == 403
    assert client.get('/exceptions', headers=headers(main.DASHBOARD_API_KEY)).status_code == 403


def test_legacy_backdated_owner_commit_needs_no_pin(client, actors, payload, db_cursor):
    from datetime import timedelta
    prepared = prepare(client, payload | {'occurred_at': (main.get_plant_now()-timedelta(days=16)).isoformat(),
                                         'backfill': True}, actors['owner']['key'])
    assert commit(client, prepared, actors['owner']['key']).status_code == 200
    assert posted_count(db_cursor, prepared) == 1


def test_offline_bootstrap_before_activation(client, actors, db_cursor, monkeypatch):
    monkeypatch.setenv('PIN_PEPPER', secrets.token_hex(32))
    value = next(str(i) for i in range(5100, 5900) if pins.valid_pin(str(i)))
    owner = actors['owner']
    for role in ('floor', 'office', 'retired'):
        with pytest.raises(ValueError):
            bootstrap(db_cursor, actors[role]['id'], value)
    token_hash = pins.digest(secrets.token_urlsafe(32))
    db_cursor.execute("INSERT INTO actor_sessions(token_hash,actor_id,auth_method,expires_at) VALUES (%s,%s,'actor_key',clock_timestamp()+interval '10 minutes')", (token_hash, owner['id']))
    bootstrap(db_cursor, owner['id'], value)
    db_cursor.execute('SELECT ended_reason FROM actor_sessions WHERE token_hash=%s', (token_hash,))
    assert db_cursor.fetchone()['ended_reason'] == 'pin_reset'
    with pytest.raises(ValueError):
        bootstrap(db_cursor, owner['id'], value)
    assert client.post('/auth/session', json={'pin': value}).status_code == 404
    monkeypatch.setenv('PIN_LOGIN_ENABLED', '1')
    response = client.post('/auth/session', json={'pin': value})
    assert response.status_code == 200
    assert response.json()['actor']['id'] == owner['id']


def test_migration_refuses_nonowner_before_rls(isolated_database):
    # Disposable local DB only. Session role is deliberately not the table owner.
    with psycopg2.connect(isolated_database) as conn, conn.cursor() as cur:
        cur.execute('CREATE ROLE a11_nonowner NOLOGIN')
        try:
            cur.execute('SAVEPOINT migration_guard')
            cur.execute('SET LOCAL ROLE a11_nonowner')
            with pytest.raises(psycopg2.errors.RaiseException):
                cur.execute((ROOT/'migrations/073_pin_sessions.sql').read_text())
            cur.execute('ROLLBACK TO SAVEPOINT migration_guard')
            cur.execute("SELECT relrowsecurity FROM pg_class WHERE oid='actors'::regclass")
            assert cur.fetchone()[0] is False
        finally:
            cur.execute('DROP ROLE a11_nonowner')


def test_uvicorn_forwarded_clients_have_separate_ip_buckets(client, db_cursor, actors, monkeypatch):
    monkeypatch.setenv('PIN_LOGIN_ENABLED', '1')
    monkeypatch.setenv('PIN_PEPPER', secrets.token_hex(32))
    # TestClient's peer is 'testclient'. Real uvicorn middleware translates the
    # forwarded addresses before the actual login handler hashes request.client.
    proxy = ProxyHeadersMiddleware(main.app, trusted_hosts=['testclient'])
    http = TestClient(proxy)
    try:
        for _ in range(5):
            http.cookies.clear()
            response = http.post('/auth/session', json={'pin': 'invalid'}, headers={'X-Forwarded-For': '198.51.100.11'})
        assert 'retry-after' in response.headers
        http.cookies.clear()
        other = http.post('/auth/session', json={'pin': 'invalid'}, headers={'X-Forwarded-For': '198.51.100.12'})
        assert other.status_code == 401 and 'retry-after' not in other.headers
    finally:
        http.close()
    db_cursor.execute("SELECT count(*) AS n FROM pin_rate_limits WHERE source LIKE 'ip:%'")
    assert db_cursor.fetchone()['n'] == 2


@pytest.mark.parametrize('trusted', ['*', 'testclient', 'untrusted'])
def test_railway_edge_identity_overrides_spoofed_xff(client, db_cursor, monkeypatch, trusted):
    monkeypatch.setenv('PIN_LOGIN_ENABLED', '1')
    monkeypatch.setenv('PIN_PEPPER', secrets.token_hex(32))
    proxy = RailwayEdgeHeaders(ProxyHeadersMiddleware(main.app, trusted_hosts=trusted), trusted)
    http = TestClient(proxy)
    try:
        for i in range(6):
            http.cookies.clear()
            response = http.post('/auth/session', json={'pin': 'invalid'},
                                 headers={'X-Real-IP': '198.51.100.20', 'X-Forwarded-For': '203.0.113.'+str(i)})
        assert 'retry-after' in response.headers
        http.cookies.clear()
        other = http.post('/auth/session', json={'pin': 'invalid'},
                          headers={'X-Real-IP': '198.51.100.21', 'X-Forwarded-For': '203.0.113.250'})
        assert ('retry-after' not in other.headers) is (trusted != 'untrusted')
    finally:
        http.close()
    db_cursor.execute("SELECT count(*) AS n FROM pin_rate_limits WHERE source LIKE 'ip:%'")
    assert db_cursor.fetchone()['n'] == (1 if trusted == 'untrusted' else 2)


def test_proxy_startup_reads_env_and_enables_uvicorn_headers(monkeypatch):
    import proxy_server
    def no_app_load(config):
        config.loaded_app = object()
    monkeypatch.setattr(proxy_server.uvicorn.Config, 'load', no_app_load)
    monkeypatch.setenv('FORWARDED_ALLOW_IPS', '192.0.2.10,192.0.2.11')
    monkeypatch.setenv('RAILWAY_ENVIRONMENT_ID', 'test-environment')
    config = proxy_server.config()
    assert config.proxy_headers is True
    assert config.forwarded_allow_ips == '192.0.2.10,192.0.2.11'
    assert isinstance(config.loaded_app, RailwayEdgeHeaders)


def test_dormant_auth_works_without_any_pin_tables_or_columns(client, db_cursor, actors):
    db_cursor.execute('DROP TABLE actor_sessions, pin_rate_limits, pin_attempts, pin_management_audit')
    db_cursor.execute('ALTER TABLE actors DROP COLUMN pin_hash, DROP COLUMN pin_set_at, DROP COLUMN pin_locked_until, DROP COLUMN pin_failed_attempts')
    main._reset_actor_cache()
    for key in (main.API_KEY, main.DASHBOARD_API_KEY, actors['owner']['key']):
        assert client.get('/auth/whoami', headers=headers(key)).status_code == 200
    assert client.post('/exceptions/2147483000/approve', json={}, headers=headers(actors['owner']['key'])).status_code == 404
    assert client.post('/auth/session', json={'pin': 'invalid'}).status_code == 404


def test_session_prefixed_actor_key_keeps_legacy_precedence_when_off(client, db_cursor, actors):
    key = pins.SESSION_PREFIX + secrets.token_urlsafe(32)
    db_cursor.execute('UPDATE actors SET key_hash=%s WHERE id=%s', (pins.digest(key), actors['owner']['id']))
    main._reset_actor_cache()
    response = client.get('/auth/whoami', headers=headers(key))
    assert response.status_code == 200
    assert response.json()['key_kind'] == 'actor'
    assert response.json()['actor']['id'] == actors['owner']['id']

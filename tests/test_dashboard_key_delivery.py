"""O-7: the dashboard reads its scoped key from GET /auth/dashboard-key.

The route hands out exactly the dashboard scope (never the master or read-only
key), and the key it hands out still cannot open a session or prepare a
make/pack/adjust/found/ship. The literal guard keeps key values out of git.
"""
from pathlib import Path

import pytest

import main
from scripts import check_no_key_literals as guard
from tests.test_actor_attribution import actors, client  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def test_route_is_public_no_store_and_returns_only_the_dashboard_key(client):
    response = client.get('/auth/dashboard-key')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json() == {'dashboard_key': main.DASHBOARD_API_KEY}
    assert main.API_KEY not in response.text


@pytest.mark.parametrize('clash', ['empty', 'master', 'readonly'])
def test_route_refuses_rather_than_leak_another_key(client, monkeypatch, clash):
    if clash == 'empty':
        monkeypatch.setattr(main, 'DASHBOARD_API_KEY', '')
    elif clash == 'master':
        monkeypatch.setattr(main, 'DASHBOARD_API_KEY', main.API_KEY)
    else:
        monkeypatch.setattr(main, 'READONLY_API_KEY', main.DASHBOARD_API_KEY)
    response = client.get('/auth/dashboard-key')
    assert response.status_code == 503
    assert response.headers['cache-control'] == 'no-store'
    assert main.API_KEY not in response.text and 'dashboard_key' not in response.text


def test_rotation_is_an_env_change_only(client, monkeypatch):
    monkeypatch.setattr(main, 'DASHBOARD_API_KEY', 'rotated-dashboard-key')
    served = client.get('/auth/dashboard-key').json()['dashboard_key']
    assert served == 'rotated-dashboard-key'
    assert client.get('/auth/whoami', headers={'X-API-Key': served}).status_code == 200
    assert client.get('/auth/whoami', headers={'X-API-Key': 'test-dashboard-key'}).status_code == 403


@pytest.mark.db
def test_served_key_cannot_open_a_session(client, actors, monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED', '1')
    served = client.get('/auth/dashboard-key').json()['dashboard_key']
    response = client.post('/auth/session', json={'actor_key': served})
    assert response.status_code == 401
    assert response.json()['detail']['error_code'] == 'ACTOR_KEY_INVALID'


@pytest.mark.db
@pytest.mark.parametrize('path', ['/make/prepare', '/pack/prepare', '/adjust/prepare',
                                  '/inventory/found/prepare', '/ship/prepare', '/ship/bulk/prepare',
                                  '/sales/orders/1/ship/prepare'])
def test_served_key_cannot_prepare_floor_writes(client, path):
    served = client.get('/auth/dashboard-key').json()['dashboard_key']
    response = client.post(path, json={}, headers={'X-API-Key': served})
    assert response.status_code == 403, response.text


def test_route_is_not_on_any_key_allowlist():
    route = ('GET', '/auth/dashboard-key')
    assert route not in main.DASHBOARD_KEY_ALLOWLIST and route not in main.ACTOR_WRITE_ALLOWLIST


# ── literal guard ──

def test_repository_has_no_key_literals():
    assert guard.scan(str(ROOT), guard.tracked_files(str(ROOT))) == []


def test_guard_flags_each_kind_without_printing_values(tmp_path):
    secret = 'live-secret-value-123'
    (tmp_path / 'dashboard').mkdir()
    (tmp_path / 'dashboard' / 'page.js').write_text(
        "const SALES_API_KEY = 'abcdefgh1234';\nfetch(u,{headers:{'X-API-Key':'zzzzzzzz'}});\n")
    (tmp_path / 'notes.txt').write_text(f'curl -H "X-API-Key: {secret}"\n')
    findings = guard.scan(str(tmp_path), ['dashboard/page.js', 'notes.txt'], [secret])
    assert [f.split(':')[0] + ':' + f.split(':')[1] for f in findings] == [
        'dashboard/page.js:1', 'dashboard/page.js:2', 'notes.txt:1']
    assert not any(secret in f or 'abcdefgh1234' in f or 'zzzzzzzz' in f for f in findings)


def test_guard_ignores_storage_names_and_runtime_lookups(tmp_path):
    (tmp_path / 'dashboard').mkdir()
    (tmp_path / 'dashboard' / 'ok.js').write_text(
        'const LS_KEY="board-state-v1";\n'
        "headers:{'X-API-Key':await FLDashboard.key()}\n")
    assert guard.scan(str(tmp_path), ['dashboard/ok.js']) == []


def test_guard_retired_fingerprint_matches_whole_tokens_only(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(guard, 'RETIRED_FINGERPRINTS',
                        {hashlib.sha256(b'retired-key-0001').hexdigest(): 'test key'})
    (tmp_path / 'a.js').write_text("x='retired-key-0001'\n")
    (tmp_path / 'CHANGE_LOG.md').write_text('old retired-key-0001 mention\n')
    findings = guard.scan(str(tmp_path), ['a.js', 'CHANGE_LOG.md'])
    assert findings == ['a.js:1: retired test key literal [redacted]']

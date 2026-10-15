"""O-7: the dashboard reads its scoped key from GET /auth/dashboard-key.

The route hands out exactly the dashboard scope (never the master or read-only
key), and the key it hands out still cannot open a session or prepare a
make/pack/adjust/found/ship. The literal guard keeps key values out of git.
"""
import asyncio
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import pytest

import main
from scripts import check_no_key_literals as guard
from tests.test_actor_attribution import actors, client  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_clash_cache():
    main._dashboard_key_clash_cache.clear()
    yield
    main._dashboard_key_clash_cache.clear()


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


@pytest.mark.db
@pytest.mark.parametrize('role', ['owner', 'retired'])
def test_route_refuses_a_value_that_is_an_actor_key(client, actors, monkeypatch, role):
    """Codex FIX ROUND 1: publishing an actor's key would hand out a personal
    credential that becomes a full actor key once the dashboard key rotates."""
    monkeypatch.setattr(main, 'DASHBOARD_API_KEY', actors[role]['key'])
    main._dashboard_key_clash_cache.clear()
    response = client.get('/auth/dashboard-key')
    assert response.status_code == 503 and actors[role]['key'] not in response.text


def test_route_fails_closed_when_the_actor_lookup_fails(client, monkeypatch):
    @contextmanager
    def _broken():
        raise RuntimeError('db down')
        yield  # pragma: no cover

    monkeypatch.setattr(main, 'get_db_connection', _broken)
    main._dashboard_key_clash_cache.clear()  # the client's startup cached a clean lookup
    response = client.get('/auth/dashboard-key')
    assert response.status_code == 503 and 'dashboard_key' not in response.text
    assert main._dashboard_key_clash_cache == {}, 'a failed lookup is never cached'


def _startup(monkeypatch):
    monkeypatch.setattr(main, 'assert_staging_database', lambda url: None)
    monkeypatch.setattr(main.pool, 'ThreadedConnectionPool', lambda **kw: None)
    monkeypatch.setattr(main, '_run_startup_migrations', lambda: None)
    pool_before = main.db_pool
    try:
        asyncio.run(main.startup())
    finally:
        main.db_pool = pool_before


@pytest.mark.db
def test_startup_refuses_a_dashboard_key_equal_to_an_actor_key(client, actors, monkeypatch):
    monkeypatch.setattr(main, 'DASHBOARD_API_KEY', actors['owner']['key'])
    with pytest.raises(RuntimeError, match='every actor key'):
        _startup(monkeypatch)


@pytest.mark.parametrize('other', ['API_KEY', 'READONLY_API_KEY'])
def test_startup_refuses_a_dashboard_key_equal_to_master_or_readonly(monkeypatch, other):
    monkeypatch.setattr(main, 'DASHBOARD_API_KEY', 'shared-value-for-startup-test')
    monkeypatch.setattr(main, other, 'shared-value-for-startup-test')
    with pytest.raises(RuntimeError, match='must differ'):
        _startup(monkeypatch)


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
        'dashboard/page.js:1', 'dashboard/page.js:2', 'notes.txt:1', 'notes.txt:1']  # pattern + live value
    assert not any(secret in f or 'abcdefgh1234' in f or 'zzzzzzzz' in f for f in findings)


def test_guard_ignores_storage_names_and_runtime_lookups(tmp_path):
    (tmp_path / 'dashboard').mkdir()
    (tmp_path / 'dashboard' / 'ok.js').write_text(
        'const LS_KEY="board-state-v1";\n'
        "headers:{'X-API-Key':await FLDashboard.key()}\n")
    assert guard.scan(str(tmp_path), ['dashboard/ok.js']) == []


def test_guard_retired_fingerprint_matches_whole_tokens_everywhere(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(guard, 'RETIRED_FINGERPRINTS',
                        {hashlib.sha256(b'retired-key-0001').hexdigest(): 'test key'})
    (tmp_path / 'a.js').write_text("x='retired-key-0001'\n")
    (tmp_path / 'CHANGE_LOG.md').write_text('old retired-key-0001 mention; not retired-key-00012\n')
    findings = guard.scan(str(tmp_path), ['a.js', 'CHANGE_LOG.md'])
    assert findings == ['a.js:1: retired test key literal [redacted]',
                        'CHANGE_LOG.md:1: retired test key literal [redacted]'], 'history is not exempt'


def test_both_retired_literals_are_fingerprinted():
    labels = set(guard.RETIRED_FINGERPRINTS.values())
    assert any('dashboard' in label for label in labels) and any('master' in label for label in labels)


PLANTED = 'fld_Pl4ntedFakeKey_9x7Q2'


@pytest.mark.parametrize('name,line', [
    ('docs/runbook.md', f'curl -H "X-API-Key: {PLANTED}" $API/customers'),
    ('docs/runbook.md', f'curl -H "Authorization: Bearer {PLANTED}" $API/health'),
    ('gpt-instructions.md', f'https://api.example/sales/orders/1/packing-slip?key={PLANTED}'),
    ('DEPLOYMENT.md', f'DASHBOARD_API_KEY={PLANTED}'),
    ('scripts/tool.py', f"requests.get(url, headers={{'X-API-Key': '{PLANTED}'}})"),
    ('dashboard/page.js', f"const API_KEY = '{PLANTED}';"),
])
def test_guard_fails_on_a_planted_fake_key_without_printing_it(tmp_path, name, line):
    """Codex FIX ROUND 1: an unknown key-shaped literal must fail the guard too."""
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / name).write_text(line + '\n')
    findings = guard.scan(str(tmp_path), [name])
    assert len(findings) == 1 and findings[0].startswith(name + ':1:')
    assert PLANTED not in findings[0]


@pytest.mark.parametrize('line', [
    'curl -H "X-API-Key: <redacted>" $API/customers',
    'curl -H "X-API-Key: $DASHBOARD_API_KEY" $API/customers',
    "API_KEY=os.environ['API_KEY']",
    'tests/conftest.py sets DASHBOARD_API_KEY=test-dashboard-key',
    '# Signed packing-slip URLs are bearer capabilities',
])
def test_guard_ignores_placeholders_variables_and_documented_fakes(tmp_path, line):
    (tmp_path / 'doc.md').write_text(line + '\n')
    assert guard.scan(str(tmp_path), ['doc.md']) == []


def test_guard_cli_fails_on_a_planted_tracked_file_and_redacts(tmp_path):
    import subprocess, sys
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'notes.md').write_text(f'curl -H "X-API-Key: {PLANTED}"\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'notes.md'], check=True)
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_no_key_literals.py'), '--root', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and 'notes.md:1' in result.stdout and PLANTED not in result.stdout + result.stderr

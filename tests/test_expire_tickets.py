"""Expiry routing must fail closed before connecting; all remote URLs are fake."""
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from scripts import expire_tickets

LOCAL = 'postgresql://tester@127.0.0.1/test'
PROD = 'postgresql://tester:synthetic@production.example/postgres'
STAGE = 'postgresql://tester:synthetic@stage.example/postgres'


@pytest.fixture(autouse=True)
def guarded_env(monkeypatch):
    for key in ('ENVIRONMENT', 'DATABASE_URL', 'PRODUCTION_DATABASE_HOST',
                'STAGING_DATABASE_HOST', 'STAGING_DATABASE_PROJECT_REF',
                'PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PRODUCTION_DATABASE_HOST', 'production.example')
    connection = MagicMock()
    connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value.rowcount = 3
    monkeypatch.setattr(expire_tickets.psycopg2, 'connect', connection)
    return connection


@pytest.mark.parametrize('environment,url', [
    ('', LOCAL), ('development', LOCAL), ('staging', STAGE),
    ('production', PROD), (' Production ', PROD.replace('production.example', 'PRODUCTION.EXAMPLE.')),
    ('production', PROD + '?sslmode=require&connect_timeout=5&application_name=expiry'),
])
def test_allowed_database_connects_and_expires(monkeypatch, guarded_env, capsys, environment, url):
    monkeypatch.setenv('ENVIRONMENT', environment)
    monkeypatch.setenv('DATABASE_URL', url)
    expire_tickets.main()
    guarded_env.assert_called_once_with(url)
    cur = guarded_env.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    cur.execute.assert_called_once_with("UPDATE write_tickets SET status='expired' WHERE status='prepared' AND expires_at < now()")
    assert capsys.readouterr().out == 'Expired 3 prepared tickets\n'


@pytest.mark.parametrize('environment,url,host', [
    ('', PROD, 'production.example'), ('prod', PROD, 'production.example'),
    ('development', PROD, 'production.example'), ('staging', PROD, 'production.example'),
    ('production', PROD, ''), ('production', STAGE, 'production.example'),
    ('production', LOCAL, 'production.example'),
    ('production', PROD.replace('production.example', 'production.example.evil'), 'production.example'),
    ('production', PROD + '?host=other.example', 'production.example'),
    ('production', PROD + '?hostaddr=127.0.0.1', 'production.example'),
    ('production', PROD + '?service=other', 'production.example'),
    ('production', PROD + '?user=other', 'production.example'),
    ('production', PROD + '?dbname=other', 'production.example'),
    ('production', PROD + '?%68ost=other.example', 'production.example'),
    ('production', PROD + '#fragment', 'production.example'),
    ('production', PROD.replace('postgresql:', 'https:'), 'production.example'),
    ('production', PROD.replace('production.example', 'production.example,other.example'), 'production.example'),
    ('production', PROD.replace('/postgres', ''), 'production.example'),
    ('production', 'dbname=postgres host=production.example', 'production.example'),
    ('production', '', 'production.example'),
    ('staging', STAGE, ''), ('', LOCAL + '?host=production.example', 'production.example'),
])
def test_refuses_before_connection(monkeypatch, guarded_env, environment, url, host):
    monkeypatch.setenv('ENVIRONMENT', environment)
    monkeypatch.setenv('DATABASE_URL', url)
    monkeypatch.setenv('PRODUCTION_DATABASE_HOST', host)
    with pytest.raises(RuntimeError) as exc:
        expire_tickets.main()
    guarded_env.assert_not_called()
    assert 'synthetic' not in str(exc.value)
    assert 'postgresql://' not in str(exc.value)


@pytest.mark.parametrize('environment,url', [('', LOCAL), ('staging', STAGE), ('production', PROD)])
@pytest.mark.parametrize('key', ['PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE'])
def test_environment_routing_overrides_never_connect(monkeypatch, guarded_env, environment, url, key):
    monkeypatch.setenv('ENVIRONMENT', environment)
    monkeypatch.setenv('DATABASE_URL', url)
    monkeypatch.setenv(key, 'override')
    with pytest.raises(RuntimeError, match='routing overrides'):
        expire_tickets.main()
    guarded_env.assert_not_called()


def test_staging_host_pin_still_applies(monkeypatch, guarded_env):
    monkeypatch.setenv('ENVIRONMENT', 'staging')
    monkeypatch.setenv('DATABASE_URL', STAGE)
    monkeypatch.setenv('STAGING_DATABASE_HOST', 'different.example')
    with pytest.raises(RuntimeError, match='STAGING_DATABASE_HOST'):
        expire_tickets.main()
    guarded_env.assert_not_called()


def test_cli_hides_driver_error_and_returns_nonzero():
    env = {k: v for k, v in os.environ.items() if not k.startswith('PG')}
    env.update(ENVIRONMENT='production', DATABASE_URL=PROD,
               PRODUCTION_DATABASE_HOST='production.example', PYTHONDONTWRITEBYTECODE='1')
    script = Path(expire_tickets.__file__).resolve()
    # The stub raises before any network access, with a deliberately sensitive-looking message.
    code = ("import psycopg2,runpy,os\n"
            "def rejected(*a, **k): raise psycopg2.OperationalError(os.environ['DATABASE_URL'])\n"
            "psycopg2.connect=rejected\n"
            f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == ''
    assert result.stderr == 'Ticket expiry failed (OperationalError)\n'

"""Prove staging fails before a connection, migration or sweep can be started."""
import asyncio
import os
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest

import main
from staging_safety import assert_staging_database, PRODUCTION_PROJECT_REF
from scripts.staging_start_command import start_command

GOOD = "postgresql://postgres.stage:synthetic@stage.example:5432/postgres"
PROD = "postgresql://postgres.prod:synthetic@production.example:5432/postgres"


@pytest.fixture(autouse=True)
def staging_env(monkeypatch):
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE", "STAGING_DATABASE_HOST",
                 "STAGING_DATABASE_PROJECT_REF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("PRODUCTION_DATABASE_HOST", "production.example")


@pytest.mark.parametrize("url", [
    PROD, PROD.replace("production.example", "PRODUCTION.EXAMPLE."),
    "postgresql://postgres:synthetic@aws-1-us-east-1.pooler.supabase.com/postgres",
    f"postgresql://postgres:synthetic@db.{PRODUCTION_PROJECT_REF}.supabase.co/postgres",
    f"postgresql://postgres.{PRODUCTION_PROJECT_REF}:synthetic@other-pooler.example/postgres",
    f"postgresql://postgres.%76{PRODUCTION_PROJECT_REF[1:]}:synthetic@other.example/postgres",
])
def test_rejects_production_identity_without_leaking_url(url):
    with pytest.raises(RuntimeError, match="matches production") as exc:
        assert_staging_database(url)
    assert url not in str(exc.value)
    assert "synthetic" not in str(exc.value)


@pytest.mark.parametrize("url", ["", "dbname=postgres host=production.example",
    GOOD+"?host=production.example", GOOD+"?hostaddr=1.2.3.4", GOOD+"?service=prod",
    GOOD+"?user=postgres.prod", GOOD+"#fragment"])
def test_fails_closed_on_ambiguous_routing(url):
    with pytest.raises(RuntimeError):
        assert_staging_database(url)


def test_requires_production_host(monkeypatch):
    monkeypatch.delenv("PRODUCTION_DATABASE_HOST")
    with pytest.raises(RuntimeError, match="requires PRODUCTION_DATABASE_HOST"):
        assert_staging_database(GOOD)


def test_allows_distinct_staging_host():
    assert_staging_database(GOOD)


def test_production_behavior_unchanged(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert_staging_database(PROD)


def test_staging_identity_pin(monkeypatch):
    monkeypatch.setenv("STAGING_DATABASE_PROJECT_REF", "expected")
    with pytest.raises(RuntimeError, match="PROJECT_REF"):
        assert_staging_database(GOOD)
    assert_staging_database(GOOD.replace("postgres.stage", "postgres.expected"))


def test_libpq_environment_override_rejected(monkeypatch):
    monkeypatch.setenv("PGHOSTADDR", "1.2.3.4")
    with pytest.raises(RuntimeError, match="routing overrides"):
        assert_staging_database(GOOD)


def test_startup_refuses_before_pool_or_sweeps(monkeypatch):
    pool = Mock(side_effect=AssertionError("must not connect"))
    sweeps = Mock(side_effect=AssertionError("must not sweep"))
    monkeypatch.setattr(main, "DATABASE_URL", PROD)
    monkeypatch.setattr(main.pool, "ThreadedConnectionPool", pool)
    monkeypatch.setattr(main, "_run_startup_migrations", sweeps)
    with pytest.raises(RuntimeError, match="matches production"):
        asyncio.run(main.startup())
    pool.assert_not_called()
    sweeps.assert_not_called()


def test_direct_sweep_entrypoint_is_guarded(monkeypatch):
    marker = Mock(side_effect=AssertionError("must not execute SQL"))
    monkeypatch.setattr(main, "DATABASE_URL", PROD)
    monkeypatch.setattr(main, "_ensure_migration_markers", marker)
    with pytest.raises(RuntimeError, match="matches production"):
        main._run_startup_migrations()
    marker.assert_not_called()


@pytest.mark.parametrize("environment,url,allowed", [
    ("staging", PROD, False), ("production", GOOD, False), ("staging", GOOD, True)
])
def test_exact_railway_launcher_before_uvicorn(environment, url, allowed):
    # Run the exact generated command with a fake uvicorn module. No network.
    launcher = shlex.split(start_command())[2]
    stub = ('import sys,types\n'
            'u=types.ModuleType("uvicorn")\n'
            'u.run=lambda *a,**k: print("UVICORN_REACHED")\n'
            'sys.modules["uvicorn"]=u\n')
    env = {k:v for k,v in os.environ.items()
           if k not in ("DATABASE_URL", "TEST_DATABASE_URL")}
    env.update(ENVIRONMENT=environment,DATABASE_URL=url)
    result = subprocess.run([sys.executable,"-c",stub+launcher],
                            env=env,capture_output=True,text=True)
    assert ("UVICORN_REACHED" in result.stdout) is allowed
    assert (result.returncode == 0) is allowed
    assert url not in result.stderr

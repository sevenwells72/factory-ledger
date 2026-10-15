"""The pre-merge key check is read-only, complete, and never prints secrets."""

from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_dashboard_key_preflight.py"
DASHBOARD = "preflight-private-dashboard-value"
MASTER = "preflight-private-master-value"
READONLY = "preflight-private-readonly-value"
DSN = "postgresql://private-user:private-password@localhost/fl_pr122fx_preflight"


@pytest.fixture
def preflight(monkeypatch):
    spec = importlib.util.spec_from_file_location("dashboard_key_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("DASHBOARD_API_KEY", DASHBOARD)
    monkeypatch.setenv("API_KEY", MASTER)
    monkeypatch.setenv("READONLY_API_KEY", READONLY)
    monkeypatch.setenv("DATABASE_URL", DSN)
    return module


class DatabaseDouble:
    """Exercise the actual query/transaction code, with no network connection."""

    def __init__(self, actors=(), error_at=None):
        self.actors = actors
        self.error_at = error_at
        self.statements = []
        self.closed = False
        self.transaction_ended = False

    def connect(self, dsn, **kwargs):
        assert dsn == DSN
        assert kwargs == {"connect_timeout": 5}
        if self.error_at == "connect":
            raise RuntimeError(DSN + DASHBOARD)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.transaction_ended = True

    def cursor(self):
        return self

    def execute(self, statement, params=None):
        self.statements.append((statement, params))
        if self.error_at == "query":
            raise RuntimeError(DSN + DASHBOARD)
        if statement.startswith("SELECT"):
            assert self.statements[:2] == [
                ("SET TRANSACTION READ ONLY", None),
                ("SET LOCAL statement_timeout = '5s'", None),
            ]
            assert statement == (
                "SELECT EXISTS (SELECT 1 FROM public.actors WHERE key_hash = %s)"
            )
            assert params == (sha256(DASHBOARD.encode()).hexdigest(),)

    def fetchone(self):
        target_hash = self.statements[-1][1][0]
        return (any(actor["key_hash"] == target_hash for actor in self.actors),)

    def close(self):
        self.closed = True


def install_database(monkeypatch, database):
    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", database.connect)


def assert_status(capsys, expected):
    captured = capsys.readouterr()
    assert captured.out == (expected + "\n" if expected else "")
    assert captured.err == ""
    for secret in (DASHBOARD, MASTER, READONLY, DSN, sha256(DASHBOARD.encode()).hexdigest()):
        assert secret not in captured.out + captured.err


@pytest.mark.parametrize("kind", ["master", "readonly", "active_personal", "inactive_personal"])
def test_preflight_rejects_every_key_kind(preflight, monkeypatch, capsys, kind):
    actors = []
    if kind in ("master", "readonly"):
        monkeypatch.setenv("API_KEY" if kind == "master" else "READONLY_API_KEY", DASHBOARD)
    else:
        actors = [{"key_hash": sha256(DASHBOARD.encode()).hexdigest(),
                   "active": kind == "active_personal"}]
    database = DatabaseDouble(actors)
    install_database(monkeypatch, database)
    assert preflight.main() == 1
    assert_status(capsys, "COLLISION " + ("personal" if actors else kind))
    assert database.closed and database.transaction_ended


def test_preflight_lists_all_clashing_kinds(preflight, monkeypatch, capsys):
    monkeypatch.setenv("API_KEY", DASHBOARD)
    monkeypatch.setenv("READONLY_API_KEY", DASHBOARD)
    install_database(monkeypatch, DatabaseDouble([
        {"key_hash": sha256(DASHBOARD.encode()).hexdigest(), "active": False},
    ]))
    assert preflight.main() == 1
    assert_status(capsys, "COLLISION master readonly personal")


@pytest.mark.parametrize("readonly", [READONLY, "", "absent"])
def test_preflight_clean_is_readonly_and_status_only(preflight, monkeypatch, capsys, readonly):
    if readonly == "absent":
        monkeypatch.delenv("READONLY_API_KEY")
    else:
        monkeypatch.setenv("READONLY_API_KEY", readonly)
    database = DatabaseDouble([{"key_hash": sha256(MASTER.encode()).hexdigest(), "active": True}])
    install_database(monkeypatch, database)
    assert preflight.main() == 0
    assert_status(capsys, "OK")
    assert database.closed and database.transaction_ended


@pytest.mark.parametrize("missing", ["DASHBOARD_API_KEY", "API_KEY", "DATABASE_URL"])
def test_preflight_missing_required_configuration_fails_closed(preflight, monkeypatch, capsys, missing):
    monkeypatch.setenv(missing, "   ")
    monkeypatch.setattr(preflight, "actor_key_collision", lambda *args: pytest.fail("must not connect"))
    assert preflight.main() == 2
    assert_status(capsys, "")


@pytest.mark.parametrize("error_at", ["connect", "query"])
def test_preflight_lookup_failure_is_redacted_and_fails_closed(preflight, monkeypatch, capsys, error_at):
    database = DatabaseDouble(error_at=error_at)
    install_database(monkeypatch, database)
    assert preflight.main() == 2
    assert_status(capsys, "")
    if error_at == "query":
        assert database.closed and database.transaction_ended


def test_preflight_strips_config_like_application(preflight, monkeypatch, capsys):
    monkeypatch.setenv("DASHBOARD_API_KEY", " " + DASHBOARD + " \n")
    monkeypatch.setenv("API_KEY", "\n" + DASHBOARD + " ")
    install_database(monkeypatch, DatabaseDouble())
    assert preflight.main() == 1
    assert_status(capsys, "COLLISION master")


@pytest.mark.parametrize("mode, expected_code, expected_status", [
    ("clean", 0, "OK"),
    ("master", 1, "COLLISION master"),
    ("personal", 1, "COLLISION personal"),
    ("failure", 2, ""),
])
def test_preflight_cli_exit_and_redaction(tmp_path, mode, expected_code, expected_status):
    # A subprocess verifies the public command's exit code and both streams.
    # It cannot reach a real database even if the caller has production envs.
    (tmp_path / "psycopg2.py").write_text(
        "import os\n"
        "class Connection:\n"
        " def __enter__(self): return self\n"
        " def __exit__(self, *args): pass\n"
        " def cursor(self): return self\n"
        " def execute(self, *args): pass\n"
        " def fetchone(self): return (os.environ['PREFLIGHT_MODE'] == 'personal',)\n"
        " def close(self): pass\n"
        "def connect(*args, **kwargs):\n"
        " if os.environ['PREFLIGHT_MODE'] == 'failure':\n"
        "  raise RuntimeError(os.environ['DATABASE_URL'] + os.environ['DASHBOARD_API_KEY'])\n"
        " return Connection()\n"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "PREFLIGHT_MODE": mode,
           "DASHBOARD_API_KEY": DASHBOARD, "API_KEY": DASHBOARD if mode == "master" else MASTER,
           "READONLY_API_KEY": READONLY, "DATABASE_URL": DSN}
    result = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == expected_code
    assert result.stdout == (expected_status + "\n" if expected_status else "")
    assert result.stderr == ""
    assert all(secret not in result.stdout + result.stderr for secret in (DASHBOARD, MASTER, READONLY, DSN))

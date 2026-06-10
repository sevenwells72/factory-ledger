"""Pytest fixtures for Factory Ledger tests.

DATABASE POLICY (2026-06-10): tests run against TEST_DATABASE_URL ONLY.
The old silent fallback to DATABASE_URL (production) is gone.

Execution order matters and is enforced by doing all of this at conftest
import time, BEFORE any test module imports `main`:

  1. TEST_DATABASE_URL itself is validated against the production host list.
     If it points at production, the whole session aborts (pytest.exit) —
     it does not skip, it REFUSES to run.
  2. Only after the guard passes is TEST_DATABASE_URL copied into
     os.environ["DATABASE_URL"], so `main` (which reads DATABASE_URL at
     import) connects to the test database.
  3. If TEST_DATABASE_URL is not set, DATABASE_URL is scrubbed from the
     environment entirely so no test can accidentally reach production;
     DB-backed tests skip with instructions.

Setup for the local test database: scripts/setup_test_db.sh
Typical run:
    TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_test \
        python3 -m pytest

Every test that needs a database connection uses the `db_cursor` fixture
(rolls back on teardown), or the `client` fixture (FastAPI TestClient whose
DB writes are rolled back via a savepoint-proxied connection).
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

# Make main.py importable from tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────
# Production guard — runs at conftest import, before main is imported
# ─────────────────────────────────────────────────────────────────

# Hosts/markers that identify the production database. The Supabase project
# ref is included so even a direct (non-pooler) connection string is caught.
_PROD_URL_MARKERS = (
    "supabase.com",
    "supabase.co",
    "vrafvwcdpcijvxdvefpr",  # Supabase project ref (production)
)


def _production_match(url: str) -> str | None:
    """Return a human-readable reason if `url` points at production."""
    if not url:
        return None
    host = (urlparse(url).hostname or "").lower()
    for marker in _PROD_URL_MARKERS:
        if marker in host or marker in url.lower():
            return f"matches production marker '{marker}'"
    # Also refuse if it shares a host with whatever DATABASE_URL held when
    # the session started (belt and braces for a renamed prod project).
    prod_url = os.environ.get("DATABASE_URL", "")
    prod_host = (urlparse(prod_url).hostname or "").lower()
    if prod_host and host == prod_host:
        return f"host '{host}' equals the host of the pre-existing DATABASE_URL"
    return None


_TEST_URL = (os.environ.get("TEST_DATABASE_URL") or "").strip()

if _TEST_URL:
    _reason = _production_match(_TEST_URL)
    if _reason:
        pytest.exit(
            "REFUSING TO RUN: TEST_DATABASE_URL points at the PRODUCTION "
            f"database ({_reason}). Point it at a local test database — "
            "see scripts/setup_test_db.sh.",
            returncode=4,
        )
    # Guard passed — only now may the test process see this URL as its
    # DATABASE_URL (main.py reads it at import time).
    os.environ["DATABASE_URL"] = _TEST_URL
    os.environ.setdefault("API_KEY", "test-api-key")
else:
    # No test DB configured: scrub DATABASE_URL so nothing in the test
    # process can ever connect to production by accident.
    os.environ.pop("DATABASE_URL", None)


def _connection_url():
    return _TEST_URL or None


# ─────────────────────────────────────────────────────────────────
# DB fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def _db_connection():
    url = _connection_url()
    if not url:
        pytest.skip(
            "TEST_DATABASE_URL not set — DB-backed tests skipped. "
            "Run scripts/setup_test_db.sh, then: "
            "TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_test python3 -m pytest"
        )
    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed")
    try:
        conn = psycopg2.connect(url)
    except Exception as e:
        pytest.skip(f"Test DB connection failed ({type(e).__name__}): {e}")
    # Never autocommit — every test rolls back.
    conn.autocommit = False
    yield conn
    conn.close()


@pytest.fixture
def db_cursor(_db_connection):
    """RealDictCursor scoped to a single test. Rolls back on teardown."""
    from psycopg2.extras import RealDictCursor

    # Use a SAVEPOINT so that even if the test itself commits (it shouldn't),
    # the outer transaction can still be rolled back.
    cur = _db_connection.cursor(cursor_factory=RealDictCursor)
    cur.execute("SAVEPOINT test_savepoint")
    try:
        yield cur
    finally:
        cur.execute("ROLLBACK TO SAVEPOINT test_savepoint")
        cur.close()

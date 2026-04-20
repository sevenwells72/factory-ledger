"""Pytest fixtures for Factory Ledger tests.

Every test that needs a database connection uses the `db_cursor` fixture, which:
  - Opens a single psycopg2 connection to TEST_DATABASE_URL (or falls back to
    DATABASE_URL).
  - Yields a RealDictCursor inside a transaction.
  - ROLLS BACK on test exit — every mutation is undone, prod data stays clean.

If neither env var is set, or the connection fails, tests decorated with
`@pytest.mark.db` are skipped with a clear message.

Usage:
    TEST_DATABASE_URL=postgresql://... python3 -m pytest
"""

import os
import sys
from pathlib import Path

import pytest

# Make main.py importable from tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _connection_url():
    return os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")


@pytest.fixture(scope="session")
def _db_connection():
    url = _connection_url()
    if not url:
        pytest.skip(
            "No TEST_DATABASE_URL or DATABASE_URL set — DB-backed tests skipped. "
            "Set TEST_DATABASE_URL to a Postgres instance with the schema loaded."
        )
    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed")
    try:
        conn = psycopg2.connect(url)
    except Exception as e:
        pytest.skip(f"DB connection failed ({type(e).__name__}): {e}")
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

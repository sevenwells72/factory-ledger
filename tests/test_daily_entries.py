"""Daily Entries dashboard view — entry timestamps and late-entry flagging.

The endpoint under test: GET /dashboard/api/activity/daily-entries
- reads effective state only (posted transactions; voided excluded)
- 'late' = created_at falls on a later America/New_York calendar day
  than the event date (business_date)
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor

import main


class _ConnProxy:
    def __init__(self, conn, savepoint):
        self._conn = conn
        self._savepoint = savepoint
        with conn.cursor() as cur:
            cur.execute(f"SAVEPOINT {savepoint}")

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        with self._conn.cursor() as cur:
            cur.execute(f"RELEASE SAVEPOINT {self._savepoint}")
            cur.execute(f"SAVEPOINT {self._savepoint}")

    def rollback(self):
        with self._conn.cursor() as cur:
            cur.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint}")
            cur.execute(f"SAVEPOINT {self._savepoint}")


@pytest.fixture
def daily_client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "daily_entries_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as client:
        client.headers["X-API-Key"] = main.API_KEY
        yield client
    _db_connection.rollback()


def _seed_transaction(cur, name, txn_type="receive", quantity_lb=10, event_naive_utc=None):
    """Insert a posted transaction. `timestamp` is naive UTC per the legacy
    convention; the 039 trigger derives business_date (America/New_York) from it
    and forces created_at to the real insert time."""
    if event_naive_utc is None:
        event_naive_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, active) "
        "VALUES (%s, 'ingredient', %s, true) RETURNING id",
        (name, f"SKU-{name}"),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, f"{name}-LOT"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute(
        """INSERT INTO transactions (type, timestamp, status)
           VALUES (%s, %s, 'posted')
           RETURNING id, business_date, created_at, created_at_source""",
        (txn_type, event_naive_utc),
    )
    transaction = cur.fetchone()
    cur.execute(
        """INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
           VALUES (%s, %s, %s, %s)""",
        (transaction["id"], product_id, lot_id, quantity_lb),
    )
    return transaction


def _fetch_entries(client, day, date_mode="event"):
    response = client.get(
        "/dashboard/api/activity/daily-entries",
        params={"date": day.isoformat(), "date_mode": date_mode},
    )
    assert response.status_code == 200, response.text
    return response.json()["entries"]


@pytest.mark.db
def test_same_day_entry_appears_with_created_at(_db_connection, daily_client):
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        txn = _seed_transaction(cur, "DAILY-SAMEDAY", txn_type="ship", quantity_lb=-5)

    entries = _fetch_entries(daily_client, txn["business_date"])
    entry = next(e for e in entries if e["transaction_id"] == txn["id"])

    assert entry["type"] == "ship"
    assert entry["created_at"]
    assert entry["created_date"]
    assert entry["created_time"].endswith(" ET")
    assert entry["created_at_source"] == "database"
    assert entry["entry_time_reliable"] is True
    assert entry["late_entry"] is False
    assert entry["days_late"] == 0

    line = entry["lines"][0]
    assert line["product_name"] == "DAILY-SAMEDAY"
    assert line["sku"] == "SKU-DAILY-SAMEDAY"
    assert float(line["quantity_lb"]) == -5  # signed, not absolute


@pytest.mark.db
def test_late_entry_is_flagged(_db_connection, daily_client):
    event_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        txn = _seed_transaction(cur, "DAILY-LATE", event_naive_utc=event_time)

    entries = _fetch_entries(daily_client, txn["business_date"])
    entry = next(e for e in entries if e["transaction_id"] == txn["id"])

    assert entry["entry_time_reliable"] is True
    assert entry["late_entry"] is True
    assert entry["days_late"] >= 2
    assert entry["event_date"] == txn["business_date"].isoformat()

    # date_mode=entered finds the same row by its entry day instead
    entered_day = txn["created_at"].astimezone(main.PLANT_TIMEZONE).date()
    entered_entries = _fetch_entries(daily_client, entered_day, date_mode="entered")
    assert any(e["transaction_id"] == txn["id"] for e in entered_entries)
    # ...and by-event mode for the entry day must NOT include it
    event_day_entries = _fetch_entries(daily_client, entered_day, date_mode="event")
    assert all(e["transaction_id"] != txn["id"] for e in event_day_entries)


@pytest.mark.db
def test_voided_transaction_is_excluded(_db_connection, daily_client):
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        txn = _seed_transaction(cur, "DAILY-VOIDED")

    entries = _fetch_entries(daily_client, txn["business_date"])
    assert any(e["transaction_id"] == txn["id"] for e in entries)

    voided = daily_client.post(
        f"/records/transactions/{txn['id']}/corrections",
        json={"event_type": "void", "reason": "daily entries effective-state test"},
    )
    assert voided.status_code == 200, voided.text

    entries_after = _fetch_entries(daily_client, txn["business_date"])
    assert all(e["transaction_id"] != txn["id"] for e in entries_after)

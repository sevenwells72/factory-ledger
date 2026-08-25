"""FR-12 contract tests for the authenticated global recent-ledger feed."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

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
def client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "recent_ledger_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as c:
        c.headers["X-API-Key"] = main.API_KEY
        yield c
    _db_connection.rollback()


@pytest.fixture
def cur(_db_connection):
    cursor = _db_connection.cursor(cursor_factory=RealDictCursor)
    yield cursor
    cursor.close()


def _seed_transaction(cur, name, txn_type="receive", quantity=10):
    cur.execute(
        """INSERT INTO products (name, type, uom, active)
           VALUES (%s, 'ingredient', 'lb', true) RETURNING id""",
        (name,),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, f"{name}-LOT"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute(
        """INSERT INTO transactions (type, timestamp, status)
           VALUES (%s, %s, 'posted') RETURNING id""",
        (txn_type, datetime.now(timezone.utc).replace(tzinfo=None)),
    )
    transaction_id = cur.fetchone()["id"]
    cur.execute(
        """INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
           VALUES (%s, %s, %s, %s)""",
        (transaction_id, product_id, lot_id, quantity),
    )
    return transaction_id


def _recent(client, limit=50, headers=None):
    response = client.get("/ledger/recent", params={"limit": limit}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.db
def test_correction_is_newer_feed_event_and_voided_original_remains(client, cur):
    txn_id = _seed_transaction(cur, "RECENT-VOID", quantity=11)

    voided = client.post(
        f"/records/transactions/{txn_id}/corrections",
        json={"event_type": "void", "reason": "FR-12 audit trail test"},
    )
    assert voided.status_code == 200, voided.text

    events = _recent(client)["events"]
    correction_index = next(i for i, e in enumerate(events)
                            if e["event_kind"] == "correction" and e["transaction_id"] == txn_id)
    original_index = next(i for i, e in enumerate(events)
                          if e["event_kind"] == "transaction" and e["transaction_id"] == txn_id)
    correction = events[correction_index]
    original = events[original_index]
    assert correction_index < original_index
    assert correction["event_type"] == "void"
    assert correction["occurred_at"]
    assert correction["created_at_source"] == "database"
    assert correction["lines"] == []
    assert correction["correction"]["target_id"] == txn_id
    assert original["effective_status"] == "voided"
    assert original["occurred_at"] == correction["occurred_at"]
    assert original["created_at_source"] == "database"
    assert original["lines"][0] == {
        "product_name": "RECENT-VOID", "quantity": 11, "unit": "lb", "lot_code": "RECENT-VOID-LOT"
    }


@pytest.mark.db
def test_restore_is_its_own_recent_event(client, cur):
    txn_id = _seed_transaction(cur, "RECENT-RESTORE")
    assert client.post(f"/records/transactions/{txn_id}/corrections", json={
        "event_type": "void", "reason": "FR-12 void before restore"
    }).status_code == 200
    restored = client.post(f"/records/transactions/{txn_id}/corrections", json={
        "event_type": "restore", "reason": "FR-12 restore"
    })
    assert restored.status_code == 200, restored.text

    events = _recent(client)["events"]
    restore = next(e for e in events if e["event_kind"] == "correction"
                   and e["transaction_id"] == txn_id and e["event_type"] == "restore")
    original = next(e for e in events if e["event_kind"] == "transaction" and e["transaction_id"] == txn_id)
    assert restore["effective_status"] == "posted"
    assert original["effective_status"] == "posted"


@pytest.mark.db
def test_limit_maximum_and_direction_mapping(client, cur):
    expected = {
        "receive": "received", "make": "produced", "pack": "packed", "ship": "shipped",
        "adjust-in": "adjusted_in", "adjust-out": "adjusted_out",
    }
    transaction_ids = {
        "receive": _seed_transaction(cur, "RECENT-RECEIVE", "receive", 1),
        "make": _seed_transaction(cur, "RECENT-MAKE", "make", 1),
        "pack": _seed_transaction(cur, "RECENT-PACK", "pack", -1),
        "ship": _seed_transaction(cur, "RECENT-SHIP", "ship", -1),
        "adjust-in": _seed_transaction(cur, "RECENT-ADJUST-IN", "adjust", 1),
        "adjust-out": _seed_transaction(cur, "RECENT-ADJUST-OUT", "adjust", -1),
    }
    feed = _recent(client, limit=50)
    by_id = {event["transaction_id"]: event for event in feed["events"] if event["event_kind"] == "transaction"}
    for key, transaction_id in transaction_ids.items():
        assert by_id[transaction_id]["direction"] == expected[key]

    limited = _recent(client, limit=1)
    assert limited["count"] == 1 and len(limited["events"]) == 1
    too_large = client.get("/ledger/recent", params={"limit": 51})
    assert too_large.status_code == 422


@pytest.mark.db
def test_recent_ledger_requires_and_accepts_dashboard_scoped_key(client):
    # Reuse the fixture client: creating a second TestClient would run the
    # application shutdown hook and close the shared test pool mid-test.
    missing = client.get("/ledger/recent", headers={"X-API-Key": ""})
    assert missing.status_code == 401

    scoped = _recent(client, headers={"X-API-Key": main.DASHBOARD_API_KEY})
    assert "events" in scoped


def test_dashboard_activity_renders_occurred_entered_lag_and_backfill_badge():
    root = Path(__file__).resolve().parent.parent
    dashboard = (root / "dashboard/dashboard.js").read_text(encoding="utf-8")
    styles = (root / "dashboard/dashboard.css").read_text(encoding="utf-8")
    index = (root / "dashboard/index.html").read_text(encoding="utf-8")

    assert "<strong>Occurred:</strong>" in dashboard
    assert "<strong>Entered:</strong>" in dashboard
    assert "if (Math.abs(lagMinutes) <= 60) return '';" in dashboard
    assert "recent-entry-backfilled" in dashboard
    assert "source !== 'api_backfill'" in dashboard
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in styles
    assert 'dashboard.css?v=27' in index
    assert 'dashboard.js?v=38' in index

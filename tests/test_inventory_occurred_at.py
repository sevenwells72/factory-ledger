"""Inventory occurrence time: API validation, persistence, and trigger behavior."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

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
def occurred_client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "occurred_at_api")
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


def _seed_adjustment_target(conn):
    token = uuid4().hex[:10]
    product_name = f"Occurred At Product {token}"
    lot_code = f"OCCURRED-{token}"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO products (name, type, odoo_code, active) "
            "VALUES (%s, 'finished', %s, true) RETURNING id",
            (product_name, f"OCC-{token}"),
        )
        product_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (product_id, lot_code),
        )
        lot_id = cur.fetchone()["id"]
    return product_name, lot_code, lot_id


def _adjust(client, product_name, lot_code, **timing):
    return client.post(
        "/adjust",
        json={
            "mode": "commit",
            "product_name": product_name,
            "lot_code": lot_code,
            "adjustment_lb": 1,
            "reason": "occurred_at regression",
            **timing,
        },
    )


def _transaction(conn, transaction_id):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, timestamp, occurred_at, business_date, created_at, "
            "created_at_source FROM transactions WHERE id = %s",
            (transaction_id,),
        )
        return cur.fetchone()


def _migration_marker_sql(relative_path):
    root = Path(__file__).resolve().parent.parent
    text = (root / relative_path).read_text(encoding="utf-8")
    return text.split("-- BEGIN 8/17 RECON BACKFILL MARKER", 1)[1].split(
        "-- END 8/17 RECON BACKFILL MARKER", 1
    )[0]


def _force_transaction_provenance(cur, transaction_ids, source, created_at):
    cur.execute(
        "ALTER TABLE transactions DISABLE TRIGGER trg_transactions_original_append_only"
    )
    cur.execute(
        "ALTER TABLE transactions DISABLE TRIGGER trg_transactions_created_at"
    )
    cur.execute(
        """UPDATE transactions
           SET created_at = %s, created_at_source = %s
           WHERE id = ANY(%s)""",
        (created_at, source, transaction_ids),
    )
    cur.execute(
        "ALTER TABLE transactions ENABLE TRIGGER trg_transactions_created_at"
    )
    cur.execute(
        "ALTER TABLE transactions ENABLE TRIGGER trg_transactions_original_append_only"
    )


@pytest.mark.db
def test_past_occurred_at_inside_window_is_preserved(
    _db_connection, occurred_client
):
    product_name, lot_code, _ = _seed_adjustment_target(_db_connection)
    expected = main.get_plant_now() - timedelta(hours=2)
    plant_local_iso = expected.replace(tzinfo=None).isoformat(timespec="seconds")

    response = _adjust(
        occurred_client, product_name, lot_code, occurred_at=plant_local_iso
    )
    assert response.status_code == 200, response.text
    transaction = _transaction(_db_connection, response.json()["transaction_id"])

    assert transaction["occurred_at"] == expected.replace(microsecond=0)
    assert transaction["created_at"] > transaction["occurred_at"]
    assert transaction["created_at_source"] == "database"
    assert transaction["business_date"] == expected.date()


@pytest.mark.db
def test_future_occurred_at_is_rejected(_db_connection, occurred_client):
    product_name, lot_code, _ = _seed_adjustment_target(_db_connection)
    future = (main.get_plant_now() + timedelta(minutes=6)).isoformat()

    response = _adjust(
        occurred_client, product_name, lot_code, occurred_at=future
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "OCCURRED_AT_IN_FUTURE"


@pytest.mark.db
def test_older_than_fourteen_days_requires_backfill(
    _db_connection, occurred_client
):
    product_name, lot_code, _ = _seed_adjustment_target(_db_connection)
    old = (main.get_plant_now() - timedelta(days=15)).isoformat()

    response = _adjust(occurred_client, product_name, lot_code, occurred_at=old)
    assert response.status_code == 400
    assert (
        response.json()["detail"]["error_code"]
        == "OCCURRED_AT_BACKFILL_REQUIRED"
    )


@pytest.mark.db
def test_backfill_accepts_old_event_and_sets_existing_marker(
    _db_connection, occurred_client
):
    product_name, lot_code, _ = _seed_adjustment_target(_db_connection)
    old = main.get_plant_now() - timedelta(days=15)

    response = _adjust(
        occurred_client,
        product_name,
        lot_code,
        occurred_at=old.isoformat(),
        backfill=True,
    )
    assert response.status_code == 200, response.text
    transaction = _transaction(_db_connection, response.json()["transaction_id"])

    assert transaction["occurred_at"] == old
    assert transaction["created_at_source"] == main.INVENTORY_BACKFILL_SOURCE
    assert transaction["created_at"] > transaction["occurred_at"]

    recent = occurred_client.get("/ledger/recent", params={"limit": 50})
    assert recent.status_code == 200, recent.text
    event = next(
        event for event in recent.json()["events"]
        if event["event_kind"] == "transaction"
        and event["transaction_id"] == transaction["id"]
    )
    assert event["created_at_source"] == "api_backfill"


@pytest.mark.db
def test_migration_046_marks_only_8_17_recon_rows_and_down_restores_database(
    _db_connection,
):
    entered_at = datetime(2026, 8, 17, 19, 48, tzinfo=timezone.utc)
    event_at = datetime(2026, 8, 14, 16, 0, tzinfo=timezone.utc)
    legacy_timestamp = event_at.replace(tzinfo=None)
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        transaction_ids = []
        for operator_id, notes in (
            ("inv-recon-2026-08-17", "INV-RECON-2026-08-17 test row"),
            ("not-the-recon", "ordinary entry"),
        ):
            cur.execute(
                """INSERT INTO transactions
                       (type, timestamp, occurred_at, business_date, operator_id,
                        notes, status)
                   VALUES ('adjust', %s, %s, DATE '2026-08-14', %s, %s, 'posted')
                   RETURNING id""",
                (legacy_timestamp, event_at, operator_id, notes),
            )
            transaction_ids.append(cur.fetchone()["id"])

        recon_id, decoy_id = transaction_ids
        _force_transaction_provenance(
            cur, transaction_ids, "database", entered_at
        )
        cur.execute(_migration_marker_sql("migrations/046_inventory_occurred_at.sql"))
        cur.execute(
            "SELECT id, created_at_source FROM transactions WHERE id = ANY(%s)",
            (transaction_ids,),
        )
        sources = {row["id"]: row["created_at_source"] for row in cur.fetchall()}
        assert sources == {recon_id: "api_backfill", decoy_id: "database"}

        cur.execute(
            _migration_marker_sql("migrations/down/046_inventory_occurred_at_down.sql")
        )
        cur.execute(
            "SELECT id, created_at_source FROM transactions WHERE id = ANY(%s)",
            (transaction_ids,),
        )
        sources = {row["id"]: row["created_at_source"] for row in cur.fetchall()}
        assert sources == {recon_id: "database", decoy_id: "database"}


@pytest.mark.db
def test_absent_occurred_at_keeps_legacy_timestamp_fallback(
    _db_connection, occurred_client
):
    product_name, lot_code, _ = _seed_adjustment_target(_db_connection)

    response = _adjust(occurred_client, product_name, lot_code)
    assert response.status_code == 200, response.text
    transaction = _transaction(_db_connection, response.json()["transaction_id"])

    legacy_timestamp_as_utc = transaction["timestamp"].replace(tzinfo=timezone.utc)
    assert transaction["occurred_at"] == legacy_timestamp_as_utc
    assert transaction["created_at_source"] == "database"


@pytest.mark.db
def test_business_time_trigger_leaves_explicit_occurred_at_alone(db_cursor):
    explicit = main.get_plant_now() - timedelta(days=3, hours=4)
    entry_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    db_cursor.execute(
        """
        INSERT INTO transactions (type, timestamp, occurred_at)
        VALUES ('adjust', %s, %s)
        RETURNING occurred_at, business_date, created_at
        """,
        (entry_timestamp, explicit),
    )
    transaction = db_cursor.fetchone()

    assert transaction["occurred_at"] == explicit
    assert transaction["business_date"] == explicit.date()
    assert transaction["created_at"] > transaction["occurred_at"]


def test_every_inventory_write_model_accepts_occurrence_metadata():
    models = (
        main.ReceiveRequest,
        main.ShipRequest,
        main.MakeRequest,
        main.PackRequest,
        main.AdjustRequest,
        main.AddFoundInventoryRequest,
        main.AddFoundInventoryWithNewProductRequest,
        main.ShipOrderRequest,
        main.CommitShipOrderRequest,
    )
    for model in models:
        assert "occurred_at" in model.model_fields
        assert "backfill" in model.model_fields

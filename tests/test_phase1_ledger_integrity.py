"""Trial v8.1 Phase 1: timestamp, correction, certification, and cutoff proof."""

from contextlib import contextmanager
from datetime import date

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
def phase1_client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "phase1_api")
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


def _seed_transaction(cur, name):
    cur.execute(
        "INSERT INTO products (name, type, active) VALUES (%s, 'ingredient', true) RETURNING id",
        (name,),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, f"{name}-LOT"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute(
        """INSERT INTO transactions (type, timestamp, status, created_at)
           VALUES ('receive', NOW(), 'posted', '2000-01-01T00:00:00Z')
           RETURNING id, business_date, created_at, created_at_source""",
    )
    transaction = cur.fetchone()
    cur.execute(
        """INSERT INTO transaction_lines
               (transaction_id, product_id, lot_id, quantity_lb, created_at)
           VALUES (%s, %s, %s, 10, '2000-01-01T00:00:00Z')
           RETURNING id, created_at, created_at_source""",
        (transaction["id"], product_id, lot_id),
    )
    line = cur.fetchone()
    return transaction, line


@pytest.mark.db
def test_database_forces_and_protects_created_at(db_cursor):
    transaction, line = _seed_transaction(db_cursor, "PHASE1-IMMUTABLE")

    assert transaction["created_at"].year != 2000
    assert transaction["created_at_source"] == "database"
    assert line["created_at"].year != 2000
    assert line["created_at_source"] == "database"

    db_cursor.execute("SAVEPOINT immutable_attempt")
    with pytest.raises(Exception, match="created_at.*immutable"):
        db_cursor.execute(
            "UPDATE transactions SET created_at = created_at - interval '1 day' WHERE id = %s",
            (transaction["id"],),
        )
    db_cursor.execute("ROLLBACK TO SAVEPOINT immutable_attempt")

    with pytest.raises(Exception, match="append-only"):
        db_cursor.execute(
            "UPDATE transactions SET notes = 'rewritten' WHERE id = %s",
            (transaction["id"],),
        )
    db_cursor.execute("ROLLBACK TO SAVEPOINT immutable_attempt")

    with pytest.raises(Exception, match="append-only"):
        db_cursor.execute(
            "UPDATE transaction_lines SET quantity_lb = 99 WHERE id = %s",
            (line["id"],),
        )
    db_cursor.execute("ROLLBACK TO SAVEPOINT immutable_attempt")


@pytest.mark.db
def test_void_preserves_original_and_api_rejects_created_at_amendment(
    _db_connection, phase1_client
):
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        transaction, _ = _seed_transaction(cur, "PHASE1-CORRECTION")
    original_created_at = transaction["created_at"]

    voided = phase1_client.post(
        f"/void/{transaction['id']}", json={"reason": "duplicate floor entry"}
    )
    assert voided.status_code == 200, voided.text
    assert voided.json()["correction_id"]

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT status, created_at FROM transactions WHERE id = %s",
            (transaction["id"],),
        )
        original = cur.fetchone()
    assert original["status"] == "posted"
    assert original["created_at"] == original_created_at

    bad_amendment = phase1_client.post(
        f"/records/transactions/{transaction['id']}/corrections",
        json={
            "event_type": "amend",
            "reason": "attempt timestamp rewrite",
            "replacement_values": {"created_at": "1999-01-01T00:00:00Z"},
        },
    )
    assert bad_amendment.status_code == 422
    assert "Immutable" in bad_amendment.text


@pytest.mark.db
def test_floor_adjust_route_then_append_only_void(_db_connection, phase1_client):
    """Keep the unchanged Floor 2.0 adjust action compatible with Phase 1."""
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """INSERT INTO products (name, type, active)
               VALUES ('PHASE1-FLOOR-SMOKE', 'ingredient', true)
               RETURNING id"""
        )
        product_id = cur.fetchone()["id"]
        cur.execute(
            """INSERT INTO lots (product_id, lot_code)
               VALUES (%s, 'PHASE1-FLOOR-SMOKE-LOT')""",
            (product_id,),
        )

    written = phase1_client.post(
        "/adjust",
        json={
            "mode": "commit",
            "product_name": "PHASE1-FLOOR-SMOKE",
            "lot_code": "PHASE1-FLOOR-SMOKE-LOT",
            "adjustment_lb": 0.01,
            "reason": "PHASE1 FLOOR 2.0 COMPATIBILITY SMOKE TEST",
        },
    )
    assert written.status_code == 200, written.text
    transaction_id = written.json()["transaction_id"]

    voided = phase1_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={
            "event_type": "void",
            "reason": "PHASE1 SMOKE CLEANUP — PRESERVE AS VOIDED EVIDENCE",
        },
    )
    assert voided.status_code == 200, voided.text

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """SELECT t.status AS original_status, ct.effective_status,
                      t.created_at_source, ct.latest_correction_id
               FROM transactions t
               JOIN ledger_current_transactions ct ON ct.id = t.id
               WHERE t.id = %s""",
            (transaction_id,),
        )
        row = cur.fetchone()
    assert row["original_status"] == "posted"
    assert row["effective_status"] == "voided"
    assert row["created_at_source"] == "database"
    assert row["latest_correction_id"]


@pytest.mark.db
def test_dashboard_daily_views_show_created_at_and_respect_effective_voids(
    _db_connection, phase1_client
):
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        received, line = _seed_transaction(cur, "PHASE1-DASHBOARD-TIMESTAMP")
        cur.execute(
            """INSERT INTO transactions
                   (type, timestamp, status, customer_name, order_reference)
               VALUES ('ship', NOW(), 'posted', 'PHASE1 TEST CUSTOMER', 'PHASE1-DASHBOARD')
               RETURNING id"""
        )
        shipment_id = cur.fetchone()["id"]
        cur.execute(
            """INSERT INTO transaction_lines
                   (transaction_id, product_id, lot_id, quantity_lb)
               SELECT %s, product_id, lot_id, -1
               FROM transaction_lines WHERE id = %s""",
            (shipment_id, line["id"]),
        )
        cur.execute(
            "SELECT lot_code FROM lots WHERE id = "
            "(SELECT lot_id FROM transaction_lines WHERE id = %s)",
            (line["id"],),
        )
        lot_code = cur.fetchone()["lot_code"]

    receipts = phase1_client.get("/dashboard/api/activity/receipts").json()["receipts"]
    receipt = next(row for row in receipts if row["transaction_id"] == received["id"])
    assert receipt["created_at"]
    assert receipt["created_date"]
    assert receipt["created_time"].endswith(" ET")
    assert receipt["created_at_source"] == "database"

    shipments = phase1_client.get("/dashboard/api/activity/shipments").json()["shipments"]
    shipment = next(row for row in shipments if row["transaction_id"] == shipment_id)
    assert shipment["created_at"]
    assert shipment["created_date"]
    assert shipment["created_time"].endswith(" ET")
    assert shipment["created_at_source"] == "database"

    lot_detail = phase1_client.get(f"/dashboard/api/lot/{lot_code}").json()
    timeline_row = next(
        row for row in lot_detail["timeline"] if row["transaction_id"] == shipment_id
    )
    assert timeline_row["created_at"]
    assert timeline_row["created_date"]
    assert timeline_row["created_time"].endswith(" ET")

    voided = phase1_client.post(
        f"/records/transactions/{shipment_id}/corrections",
        json={"event_type": "void", "reason": "dashboard effective-state test"},
    )
    assert voided.status_code == 200, voided.text
    shipments_after = phase1_client.get("/dashboard/api/activity/shipments").json()["shipments"]
    assert all(row["transaction_id"] != shipment_id for row in shipments_after)
    lot_after = phase1_client.get(f"/dashboard/api/lot/{lot_code}").json()["timeline"]
    assert all(row["transaction_id"] != shipment_id for row in lot_after)


@pytest.mark.db
def test_production_summaries_use_corrected_lines_and_effective_status(
    _db_connection, phase1_client
):
    product_name = "PHASE1-GRAHAM-PRODUCTION"
    target_date = main.get_plant_now().date()
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """INSERT INTO products (name, type, active, default_batch_lb)
               VALUES (%s, 'batch', true, 10)
               RETURNING id""",
            (product_name,),
        )
        product_id = cur.fetchone()["id"]
        cur.execute(
            """INSERT INTO lots (product_id, lot_code)
               VALUES (%s, 'PHASE1-GRAHAM-PRODUCTION-LOT')
               RETURNING id""",
            (product_id,),
        )
        lot_id = cur.fetchone()["id"]
        cur.execute(
            """INSERT INTO transactions (type, timestamp, status)
               VALUES ('make', NOW(), 'posted')
               RETURNING id""",
        )
        transaction_id = cur.fetchone()["id"]
        cur.execute(
            """INSERT INTO transaction_lines
                   (transaction_id, product_id, lot_id, quantity_lb)
               VALUES (%s, %s, %s, 10)
               RETURNING id""",
            (transaction_id, product_id, lot_id),
        )
        line_id = cur.fetchone()["id"]
        main._append_transaction_line_correction(
            cur,
            line_id,
            {"quantity_lb": 25},
            "correct production quantity",
            "phase1-test",
        )

    calendar = phase1_client.get("/dashboard/api/production", params={"days": 1})
    assert calendar.status_code == 200, calendar.text
    calendar_rows = [
        row
        for day in calendar.json()["days"]
        for row in day["batches"]
        if row["product_name"] == product_name
    ]
    assert len(calendar_rows) == 1
    assert calendar_rows[0]["total_lbs"] == 25

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        daily = main.get_daily_production_summary(cur, target_date)
    daily_row = next(
        row for row in daily["production"] if row["product_name"] == product_name
    )
    assert daily_row["total_lb"] == 25

    day_summary = phase1_client.get(
        "/production/day-summary", params={"date": target_date.isoformat()}
    )
    assert day_summary.status_code == 200, day_summary.text
    batch_row = next(
        row
        for row in day_summary.json()["batch_products"]
        if row["product_name"] == product_name
    )
    assert batch_row["total_produced_lb"] == 25

    voided = phase1_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "void", "reason": "production summary effective-state test"},
    )
    assert voided.status_code == 200, voided.text

    calendar_after = phase1_client.get(
        "/dashboard/api/production", params={"days": 1}
    ).json()
    assert all(
        row["product_name"] != product_name
        for day in calendar_after["days"]
        for row in day["batches"]
    )
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        daily_after = main.get_daily_production_summary(cur, target_date)
    assert all(row["product_name"] != product_name for row in daily_after["production"])
    day_summary_after = phase1_client.get(
        "/production/day-summary", params={"date": target_date.isoformat()}
    ).json()
    assert all(
        row["product_name"] != product_name
        for row in day_summary_after["batch_products"]
    )


@pytest.mark.db
def test_late_cutoff_boundary_and_csv_include_corrections(_db_connection, phase1_client):
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        first, _ = _seed_transaction(cur, "PHASE1-BOUNDARY-A")
    target_date = first["business_date"]

    certification = phase1_client.post(
        "/records/certifications",
        json={
            "business_date": target_date.isoformat(),
            "certified_at": first["created_at"].isoformat(),
            "source_type": "whatsapp_export",
            "source_message_id": "wa-phase1-boundary",
        },
    )
    assert certification.status_code == 200, certification.text

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        second, _ = _seed_transaction(cur, "PHASE1-BOUNDARY-B")
    correction = phase1_client.post(
        f"/records/transactions/{first['id']}/corrections",
        json={
            "event_type": "amend",
            "reason": "self-caught same-day note fix",
            "replacement_values": {"notes": "corrected"},
        },
    )
    assert correction.status_code == 200, correction.text

    response = phase1_client.get(
        "/records/late", params={"business_date": target_date.isoformat()}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    rows = {row["transaction_id"]: row for row in data["entries"]}
    assert rows[first["id"]]["late"] is False, "created_at == cutoff is on time"
    assert rows[second["id"]]["late"] is True, "created_at > cutoff is late"
    assert any(row["transaction_id"] == first["id"] for row in data["corrections"])
    assert all(row["late"] is True for row in data["corrections"])

    csv_response = phase1_client.get(
        "/records/late.csv", params={"business_date": target_date.isoformat()}
    )
    assert csv_response.status_code == 200
    assert "record_kind" in csv_response.text
    assert "correction" in csv_response.text


@pytest.mark.db
def test_certification_is_append_only_and_correctable(_db_connection, phase1_client):
    target_date = date(2099, 1, 2)
    created = phase1_client.post(
        "/records/certifications",
        json={
            "business_date": target_date.isoformat(),
            "certified_at": "2099-01-02T20:00:00-05:00",
        },
    )
    assert created.status_code == 200, created.text
    certification_id = created.json()["certification_id"]

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SAVEPOINT certification_update")
        with pytest.raises(Exception, match="append-only"):
            cur.execute(
                "UPDATE certifications SET certified_at = certified_at + interval '1 hour' WHERE id = %s",
                (certification_id,),
            )
        cur.execute("ROLLBACK TO SAVEPOINT certification_update")

    corrected = phase1_client.post(
        f"/records/certifications/{certification_id}/corrections",
        json={
            "certified_at": "2099-01-02T20:15:00-05:00",
            "reason": "wrong WhatsApp message time",
        },
    )
    assert corrected.status_code == 200, corrected.text
    current = phase1_client.get(f"/records/certifications/{target_date.isoformat()}")
    assert current.status_code == 200
    assert len(current.json()["chain"]) == 2
    assert current.json()["certified_at"] == "2099-01-02T20:15:00-05:00"

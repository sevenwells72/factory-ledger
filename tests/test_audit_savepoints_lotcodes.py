"""Regression coverage for Problems 1-3 of the 2026-08-25 data-entry audit
(docs/data-entry-inventory.md "Risk notes", branch fix/audit-insert-savepoints).

Problem 1 — best-effort audit inserts (product_verification_history,
lot_reassignments, inventory_adjustments) ran on the primary write's
transaction with a bare try/except: a DB-level failure aborted the whole
transaction and the endpoint returned success:true while the primary write
was silently rolled back. They now run inside a SAVEPOINT
(best_effort_audit_insert) and surface failures via an audit_warning
response field.

Problem 2 — /make commit took no advisory lock around lot-code generation
(unlike /receive, lock 1, and /inventory/found*, lock 2), so two concurrent
makes could mint identical B-codes or silently fold into one lot. It now
takes pg_advisory_xact_lock(3).

Problem 3 — the sequence probe parsed int() from the lexically-highest
matching lot code, so a manual code with a nonnumeric suffix (which sorts
above the numeric ones) reset the counter to 001. next_lot_sequence() now
takes MAX over the numeric suffixes only.

No test in this module may use production; everything runs through the
guarded TEST_DATABASE_URL fixtures and rolls back.
"""

import os
import threading
import time
from contextlib import contextmanager
from uuid import uuid4

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed", allow_module_level=True)

from psycopg2.extras import RealDictCursor

import main


LONG_NAME = "x" * 150      # > varchar(100) performed_by/adjusted_by columns
LONG_REASON = "r" * 60     # > varchar(50) reason_code columns


class _ConnProxy:
    """Wrap the test connection so commit()/rollback() act on an inner
    SAVEPOINT; the outer _db_connection fixture rolls everything back."""

    def __init__(self, conn, sp_name):
        self._conn = conn
        self._sp = sp_name
        with self._conn.cursor() as c:
            c.execute(f"SAVEPOINT {self._sp}")

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        with self._conn.cursor() as c:
            c.execute(f"RELEASE SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")

    def rollback(self):
        with self._conn.cursor() as c:
            c.execute(f"ROLLBACK TO SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")


@pytest.fixture
def client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "audit_sp_inner")
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


def _cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


def _seed_product(conn, name=None, product_type="ingredient"):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, %s, 'lb', true) RETURNING id",
            (name or f"AuditSP {uuid4().hex[:10]}", product_type),
        )
        return cur.fetchone()["id"]


def _count(conn, sql, params):
    with _cur(conn) as cur:
        cur.execute(sql, params)
        return cur.fetchone()["n"]


# ─────────────────────────────────────────────────────────────────
# Problem 1 — audit-insert failures must not roll back the primary write
# ─────────────────────────────────────────────────────────────────

def test_found_inventory_primary_write_survives_audit_failure(client, _db_connection):
    product_id = _seed_product(_db_connection)
    resp = client.post("/inventory/found", json={
        "product_id": product_id,
        "quantity": 25,
        "reason_code": LONG_REASON,  # overflows inventory_adjustments.reason_code varchar(50)
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "audit_warning" in body

    lot_id = body["lot_id"]
    assert _count(_db_connection, "SELECT COUNT(*) AS n FROM lots WHERE id = %s", (lot_id,)) == 1
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM transaction_lines WHERE lot_id = %s AND quantity_lb = 25",
        (lot_id,),
    ) == 1
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM inventory_adjustments WHERE lot_id = %s",
        (lot_id,),
    ) == 0


def test_found_inventory_audit_row_written_on_success(client, _db_connection):
    product_id = _seed_product(_db_connection)
    resp = client.post("/inventory/found", json={
        "product_id": product_id,
        "quantity": 10,
        "reason_code": "found_count",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "audit_warning" not in body
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM inventory_adjustments WHERE lot_id = %s AND adjustment_type = 'found'",
        (body["lot_id"],),
    ) == 1


def test_quick_create_survives_audit_failure(client, _db_connection):
    name = f"AuditSP QC {uuid4().hex[:10]}"
    resp = client.post("/products/quick-create", json={
        "product_name": name,
        "product_type": "ingredient",
        "performed_by": LONG_NAME,  # overflows product_verification_history.performed_by varchar(100)
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "audit_warning" in body
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM products WHERE id = %s",
        (body["product_id"],),
    ) == 1
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM product_verification_history WHERE product_id = %s",
        (body["product_id"],),
    ) == 0


def test_quick_create_audit_row_written_on_success(client, _db_connection):
    name = f"AuditSP QC {uuid4().hex[:10]}"
    resp = client.post("/products/quick-create", json={
        "product_name": name,
        "product_type": "ingredient",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "audit_warning" not in body
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM product_verification_history "
        "WHERE product_id = %s AND action = 'created'",
        (body["product_id"],),
    ) == 1


def test_verify_survives_audit_failure(client, _db_connection):
    product_id = _seed_product(_db_connection)
    resp = client.post(f"/products/{product_id}/verify", json={
        "action": "verify",
        "performed_by": LONG_NAME,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "audit_warning" in body
    with _cur(_db_connection) as cur:
        cur.execute("SELECT verification_status FROM products WHERE id = %s", (product_id,))
        assert cur.fetchone()["verification_status"] == "verified"
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM product_verification_history WHERE product_id = %s",
        (product_id,),
    ) == 0


def test_reassign_survives_audit_failure(client, _db_connection):
    from_id = _seed_product(_db_connection)
    to_id = _seed_product(_db_connection)
    with _cur(_db_connection) as cur:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (from_id, f"AUDITSP-{uuid4().hex[:8]}"),
        )
        lot_id = cur.fetchone()["id"]

    resp = client.post(f"/lots/{lot_id}/reassign", json={
        "to_product_id": to_id,
        "reason_code": "data_entry_error",
        "performed_by": LONG_NAME,  # overflows lot_reassignments.reassigned_by varchar(100)
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["reassignment_id"] is None
    assert "audit_warning" in body
    with _cur(_db_connection) as cur:
        cur.execute("SELECT product_id FROM lots WHERE id = %s", (lot_id,))
        assert cur.fetchone()["product_id"] == to_id
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM lot_reassignments WHERE lot_id = %s",
        (lot_id,),
    ) == 0


def test_reassign_audit_row_written_on_success(client, _db_connection):
    from_id = _seed_product(_db_connection)
    to_id = _seed_product(_db_connection)
    with _cur(_db_connection) as cur:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (from_id, f"AUDITSP-{uuid4().hex[:8]}"),
        )
        lot_id = cur.fetchone()["id"]

    resp = client.post(f"/lots/{lot_id}/reassign", json={
        "to_product_id": to_id,
        "reason_code": "data_entry_error",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reassignment_id"] is not None
    assert "audit_warning" not in body
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM lot_reassignments WHERE id = %s",
        (body["reassignment_id"],),
    ) == 1


def test_found_with_new_product_writes_adjustment_row(client, _db_connection):
    """Coverage gap closed: this path previously wrote no inventory_adjustments
    row at all (2026-08-25 audit, Risk notes §3)."""
    resp = client.post("/inventory/found-with-new-product", json={
        "product_name": f"AuditSP FNP {uuid4().hex[:10]}",
        "product_type": "ingredient",
        "quantity": 12,
        "reason_code": "found_count",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "audit_warning" not in body
    assert _count(
        _db_connection,
        "SELECT COUNT(*) AS n FROM inventory_adjustments "
        "WHERE lot_id = %s AND adjustment_type = 'found'",
        (body["lot_id"],),
    ) == 1

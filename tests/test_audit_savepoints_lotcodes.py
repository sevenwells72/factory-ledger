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


# ─────────────────────────────────────────────────────────────────
# Problem 2 — /make commit must serialize lot-code generation (advisory lock 3)
# ─────────────────────────────────────────────────────────────────

MAKE_LOCK_ID = 3


def _connect_raw():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set — DB-backed tests skipped")
    import psycopg2

    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


@pytest.fixture
def raw_client(monkeypatch):
    """TestClient whose get_db_connection yields ONE raw connection and never
    commits: requests within a test share a single open transaction (so a
    second request sees the first's writes), and teardown rolls everything
    back — which also releases any pg_advisory_xact_lock the request took.
    Needed because advisory-lock blocking is only observable across real
    connections, not through the shared savepoint-proxied test connection."""
    conn = _connect_raw()

    @contextmanager
    def _fake_get_conn():
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as c:
        c.headers["X-API-Key"] = main.API_KEY
        yield c, conn
    conn.rollback()
    conn.close()


def _seed_makeable_product(conn, name):
    """Batch product + 1-ingredient formula + 100 lb posted ingredient stock,
    written on `conn`'s open transaction (not committed)."""
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, default_batch_lb, active) "
            "VALUES (%s, 'batch', 'lb', 50, true) RETURNING id",
            (name,),
        )
        batch_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, 'ingredient', 'lb', true) RETURNING id",
            (f"{name} ING",),
        )
        ing_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) "
            "VALUES (%s, %s, 50)",
            (batch_id, ing_id),
        )
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (ing_id, f"MAKELOCK-ING-{uuid4().hex[:8]}"),
        )
        lot_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status, notes) "
            "VALUES ('receive', NOW(), 'posted', 'make-lock test seed') RETURNING id",
        )
        txn_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, 100)",
            (txn_id, ing_id, lot_id),
        )
    return batch_id


def test_make_commit_blocks_on_advisory_lock(raw_client):
    """A make commit must WAIT while another session holds advisory lock 3 —
    proving concurrent makes are serialized around lot-code generation."""
    client, conn_a = raw_client
    name = f"MakeLock Batch {uuid4().hex[:8]}"
    _seed_makeable_product(conn_a, name)

    blocker = _connect_raw()
    try:
        with blocker.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND classid = 0 AND objid = %s AND granted",
                (MAKE_LOCK_ID,),
            )
            if cur.fetchone()[0]:
                pytest.skip("advisory lock 3 already held by another session")
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (MAKE_LOCK_ID,))

        result = {}

        def _commit_make():
            resp = client.post(
                "/make", json={"mode": "commit", "product_name": name, "batches": 1}
            )
            result["status"] = resp.status_code
            result["body"] = resp.json()

        worker = threading.Thread(target=_commit_make)
        worker.start()

        # The request must show up WAITING on advisory lock 3 (not granted).
        blocked = False
        deadline = time.time() + 5
        while time.time() < deadline:
            with blocker.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND classid = 0 AND objid = %s "
                    "AND NOT granted",
                    (MAKE_LOCK_ID,),
                )
                if cur.fetchone()[0]:
                    blocked = True
                    break
            if not worker.is_alive():
                break  # completed without ever waiting — the lock is missing
            time.sleep(0.05)

        assert blocked, "/make commit never waited on advisory lock 3"

        blocker.rollback()  # releases the xact lock; the make proceeds
        worker.join(timeout=10)
        assert not worker.is_alive(), "/make commit did not finish after lock release"
        assert result["status"] == 200, result.get("body")
        assert result["body"]["success"] is True
    finally:
        blocker.rollback()
        blocker.close()


def test_sequential_make_commits_mint_distinct_codes(raw_client):
    """Two makes of the same product on the same day must get consecutive
    B-codes and distinct lots — not fold into one lot (Risk notes §1.2)."""
    client, conn_a = raw_client
    name = f"MakeSeq Batch {uuid4().hex[:8]}"
    _seed_makeable_product(conn_a, name)

    first = client.post(
        "/make", json={"mode": "commit", "product_name": name, "batches": 1}
    ).json()
    second = client.post(
        "/make", json={"mode": "commit", "product_name": name, "batches": 1}
    ).json()

    assert first["success"] is True and second["success"] is True
    assert first["lot_code"] != second["lot_code"]
    assert first["lot_id"] != second["lot_id"]
    first_seq = int(first["lot_code"].rsplit("-", 1)[1])
    second_seq = int(second["lot_code"].rsplit("-", 1)[1])
    assert second_seq == first_seq + 1


# ─────────────────────────────────────────────────────────────────
# Problem 3 — nonnumeric lot-code suffixes must not reset the sequence
# ─────────────────────────────────────────────────────────────────

def _seed_lots(conn, product_id, codes):
    with _cur(conn) as cur:
        for code in codes:
            cur.execute(
                "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s)",
                (product_id, code),
            )


@pytest.mark.db
def test_next_lot_sequence_ignores_nonnumeric_suffix(db_cursor, _db_connection):
    pid = _seed_product(_db_connection)
    # '-A' sorts ABOVE '-001' lexically; the old probe parsed it, failed,
    # and reset the counter to 001.
    _seed_lots(_db_connection, pid, ["ZZSEQ-TEST-001", "ZZSEQ-TEST-A"])
    assert main.next_lot_sequence(db_cursor, "ZZSEQ-TEST-%") == 2


@pytest.mark.db
def test_next_lot_sequence_counts_past_999(db_cursor, _db_connection):
    pid = _seed_product(_db_connection)
    # '1000' < '999' lexically, so the old probe stuck at 1000 forever.
    _seed_lots(_db_connection, pid, ["ZZSEQ-BIG-999", "ZZSEQ-BIG-1000"])
    assert main.next_lot_sequence(db_cursor, "ZZSEQ-BIG-%") == 1001


@pytest.mark.db
def test_next_lot_sequence_fresh_prefix_starts_at_1(db_cursor, _db_connection):
    pid = _seed_product(_db_connection)
    assert main.next_lot_sequence(db_cursor, "ZZSEQ-FRESH-%") == 1
    # Only nonnumeric suffixes on the prefix → still start at 1.
    _seed_lots(_db_connection, pid, ["ZZSEQ-FRESH-A", "ZZSEQ-FRESH-XY"])
    assert main.next_lot_sequence(db_cursor, "ZZSEQ-FRESH-%") == 1


def test_receive_preview_seq_survives_manual_nonnumeric_code(client, _db_connection):
    pid = _seed_product(_db_connection)
    date_part = main.get_plant_now().strftime("%y-%m-%d")
    _seed_lots(_db_connection, pid, [f"{date_part}-QQZZ-001", f"{date_part}-QQZZ-A"])
    with _cur(_db_connection) as cur:
        cur.execute("SELECT name FROM products WHERE id = %s", (pid,))
        name = cur.fetchone()["name"]
    resp = client.post("/receive", json={
        "mode": "preview",
        "product_name": name,
        "cases": 1, "case_size_lb": 50,
        "shipper_name": "QQZZ Foods",
        "bol_reference": "BOL-SEQ-TEST",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["lot_code"] == f"{date_part}-QQZZ-002"


def test_found_seq_survives_manual_nonnumeric_code(client, _db_connection):
    pid = _seed_product(_db_connection)
    date_part = main.get_plant_now().strftime("%y-%m-%d")
    _seed_lots(_db_connection, pid, [f"{date_part}-FOUND-007", f"{date_part}-FOUND-B"])
    resp = client.post("/inventory/found", json={
        "product_id": pid,
        "quantity": 5,
        "reason_code": "found_count",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["lot_code"] == f"{date_part}-FOUND-008"


def test_make_preview_seq_survives_manual_nonnumeric_code(client, _db_connection):
    pid = _seed_product(_db_connection, product_type="batch")
    with _cur(_db_connection) as cur:
        cur.execute("UPDATE products SET default_batch_lb = 50 WHERE id = %s", (pid,))
        cur.execute("SELECT name FROM products WHERE id = %s", (pid,))
        name = cur.fetchone()["name"]
    date_part = main.get_plant_now().strftime("%y-%m%d")
    _seed_lots(_db_connection, pid, [f"B{date_part}-004", f"B{date_part}-XYZ"])
    resp = client.post("/make", json={
        "mode": "preview", "product_name": name, "batches": 1,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["lot_code"] == f"B{date_part}-005"


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

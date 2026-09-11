"""Sales-order state model: migration 051 backfill, administrative exits,
derived fulfillment, tiered health, and the counts endpoint.

Three things are deliberately orthogonal here and the tests keep them that way:

  state        stored, administrative — why a human took the order off the board
  fulfillment  derived, physical      — what the LEDGER says shipped
  health       derived, advisory      — what needs attention now

The regression that matters most is at the bottom: `status=open` consumers must
behave exactly as they did before, because the deprecation-window mirror keeps
sales_orders.status in step with state.
"""

import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor

import main

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "051_sales_order_state_model.sql"


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
        proxy = _ConnProxy(_db_connection, "so_state_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as test_client:
        test_client.headers["X-API-Key"] = main.API_KEY
        yield test_client


# ─────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────

def _seed_customer(cur):
    token = uuid4().hex[:10].upper()
    cur.execute(
        "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
        (f"SOSTATE Customer {token}",),
    )
    return cur.fetchone()["id"], token


def _seed_product(cur, token, *, service=False, with_lot=False, stock=0):
    suffix = uuid4().hex[:6].upper()
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, uom, is_service, active) "
        "VALUES (%s, %s, %s, %s, %s, true) RETURNING id",
        (
            f"SOSTATE {'Service' if service else 'FG'} {token} {suffix}",
            "packaging" if service else "finished",
            f"SST-{token}-{suffix}",
            "each" if service else "lb",
            service,
        ),
    )
    product_id = cur.fetchone()["id"]
    lot_id = None
    if with_lot:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code, entry_source, received_at) "
            "VALUES (%s, %s, 'received', NOW()) RETURNING id",
            (product_id, f"SST-LOT-{token}-{suffix}"),
        )
        lot_id = cur.fetchone()["id"]
        if stock:
            cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
            txn = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
                "VALUES (%s, %s, %s, %s)",
                (txn, product_id, lot_id, stock),
            )
    return product_id, lot_id


def _seed_order(cur, customer_id, token, *, status="confirmed",
                ship_date=None, floor_ready=True, state=None, reason=None):
    order_number = f"SST-SO-{token}-{uuid4().hex[:5]}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status, requested_ship_date) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (customer_id, order_number, status, ship_date),
    )
    order_id = cur.fetchone()["id"]
    if state is not None:
        cur.execute(
            "UPDATE sales_orders SET state = %s, state_reason = %s WHERE id = %s",
            (state, reason, order_id),
        )
    if floor_ready:
        cur.execute(
            "INSERT INTO sales_order_flags (so_number, ready, ready_at, ready_by) "
            "VALUES (%s, true, NOW(), 'test')",
            (order_number,),
        )
    return order_id, order_number


def _add_line(cur, order_id, product_id, qty, *, shipped=0, status="pending"):
    cur.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (order_id, product_id, qty, shipped, status),
    )
    return cur.fetchone()["id"]


def _allocate(cur, order_id, line_id, product_id, qty, *, lot_id=None):
    cur.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, lot_id, quantity_lb, source) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (order_id, line_id, product_id, lot_id, qty,
         "staged_lot" if lot_id else "manual"),
    )
    return cur.fetchone()["id"]


def _post_ship(cur, line_id, product_id, lot_id, qty, *, recorded=None):
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('ship', NOW()) RETURNING id")
    txn = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)",
        (txn, product_id, lot_id, -qty),
    )
    cur.execute(
        "INSERT INTO sales_order_shipments (sales_order_line_id, transaction_id, quantity_lb) "
        "VALUES (%s, %s, %s)",
        (line_id, txn, round(qty, 2)),
    )
    if recorded is not None:
        cur.execute(
            "UPDATE sales_order_lines SET quantity_shipped_lb = %s, line_status = 'fulfilled' "
            "WHERE id = %s",
            (recorded, line_id),
        )
    return txn


def _void_shipment(client, transaction_id, reason):
    """Void through the real corrections endpoint — the same path production
    uses — so effective_status flips exactly the way it does in the field."""
    resp = client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "void", "reason": reason},
    )
    assert resp.status_code == 200, resp.text
    return resp


def _ledger_count(cur):
    cur.execute("SELECT COUNT(*) AS n FROM transactions")
    txns = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM transaction_lines")
    lines = cur.fetchone()["n"]
    return txns, lines


def _trace_count(cur):
    cur.execute("SELECT COUNT(*) AS n FROM trace_events")
    return cur.fetchone()["n"]


def _state_row(cur, order_id):
    cur.execute(
        "SELECT status, state, state_reason, state_note, state_changed_at, "
        "       state_changed_by, related_so_id, status_before_exit "
        "  FROM sales_orders WHERE id = %s",
        (order_id,),
    )
    return dict(cur.fetchone())


# ═════════════════════════════════════════════════════════════════
# Migration 051 — shape and backfill
# ═════════════════════════════════════════════════════════════════

def test_migration_does_not_touch_status_or_line_status():
    """The legacy columns are load-bearing for fifteen readers — hands off."""
    sql = MIGRATION.read_text().lower()
    body = "\n".join(ln for ln in sql.splitlines() if not ln.strip().startswith("--"))
    assert "drop column" not in body
    assert "rename" not in body
    assert "line_status" not in body.replace("sol.line_status <> 'cancelled'", "")
    assert "drop constraint" not in body
    assert "sales_orders_status_check" not in body
    # Owner ruling 1: the placeholder actor is never written.
    assert "legacy-shared-key" not in body
    assert "set session" not in body


@pytest.mark.db
def test_state_constraints_reject_mismatched_reasons(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)

    # open must carry no reason
    with pytest.raises(Exception):
        db_cursor.execute(
            "UPDATE sales_orders SET state='open', state_reason='short_closed' WHERE id=%s",
            (order_id,),
        )


@pytest.mark.db
def test_closed_rejects_a_cancel_reason(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    with pytest.raises(Exception):
        db_cursor.execute(
            "UPDATE sales_orders SET state='closed', state_reason='customer_cancelled' WHERE id=%s",
            (order_id,),
        )


@pytest.mark.db
def test_cancelled_rejects_a_close_reason(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    with pytest.raises(Exception):
        db_cursor.execute(
            "UPDATE sales_orders SET state='cancelled', state_reason='short_closed' WHERE id=%s",
            (order_id,),
        )


BACKFILL_MARKER = "051_sales_order_state_backfill"


def _backfill_block():
    sql = MIGRATION.read_text()
    start = sql.index("DO $$\nDECLARE")
    end = sql.index("END $$;", start) + len("END $$;")
    return sql[start:end]


def _run_backfill(cur, *, fresh=True):
    """Run the migration's backfill DO block against rows seeded in this test.

    fresh=True clears the applied marker first, so each test sees the block as
    a first-ever apply. The suite's own test DB has already been migrated, so
    without this every backfill test would just watch the block no-op.
    """
    if fresh:
        cur.execute("DELETE FROM migration_markers WHERE name = %s", (BACKFILL_MARKER,))
    cur.execute(_backfill_block())


@pytest.mark.db
@pytest.mark.parametrize("status", ["new", "confirmed", "in_production", "ready", "partial_ship"])
def test_backfill_maps_operational_statuses_to_open(db_cursor, status):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status=status)
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "open"
    assert row["state_reason"] is None
    assert row["state_note"] == f"backfilled from legacy status={status}"
    assert row["state_changed_by"] == "migration-051"
    assert row["state_changed_at"] is not None
    # the legacy column is untouched by the backfill
    assert row["status"] == status


@pytest.mark.db
@pytest.mark.parametrize("status", ["shipped", "invoiced"])
def test_backfill_shipped_recorded_when_nothing_remains(db_cursor, status):
    """Everything the ledger says was ordered went out → shipped_recorded."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status=status)
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _post_ship(db_cursor, line_id, product_id, lot_id, 100, recorded=100)
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "closed"
    assert row["state_reason"] == "shipped_recorded"
    assert row["state_note"] == f"backfilled from legacy status={status}"
    assert row["state_changed_by"] == "migration-051"


@pytest.mark.db
def test_backfill_shipped_not_recorded_when_pounds_remain(db_cursor):
    """status says shipped, but the ledger still shows pounds owed."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _post_ship(db_cursor, line_id, product_id, lot_id, 40, recorded=100)
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "closed"
    assert row["state_reason"] == "shipped_not_recorded"


@pytest.mark.db
def test_backfill_uses_the_ledger_as_tiebreaker_not_recorded_pounds(db_cursor, client):
    """The recorded column says 100 shipped; the void means the ledger says 0.

    The ledger wins: this is the split the tiebreaker exists for.
    """
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    txn = _post_ship(db_cursor, line_id, product_id, lot_id, 100, recorded=100)
    _void_shipment(client, txn, "backfill tiebreaker test")
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "closed"
    assert row["state_reason"] == "shipped_not_recorded", (
        "recorded pounds said fully shipped, but the voided ledger says otherwise"
    )


@pytest.mark.db
def test_backfill_maps_cancelled_to_cancelled_other(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="cancelled")
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "cancelled"
    assert row["state_reason"] == "other"
    assert row["state_note"] == "backfilled from legacy status=cancelled"


@pytest.mark.db
def test_backfill_excludes_cancelled_lines_from_the_split(db_cursor):
    """A cancelled line's unshipped pounds are not pounds still owed."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    shipped_line = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _post_ship(db_cursor, shipped_line, product_id, lot_id, 100, recorded=100)
    _add_line(db_cursor, order_id, product_id, 250, status="cancelled")
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)

    assert _state_row(db_cursor, order_id)["state_reason"] == "shipped_recorded"


@pytest.mark.db
def test_backfill_is_idempotent(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="confirmed")
    db_cursor.execute(
        "UPDATE sales_orders SET state_changed_at = NULL, state_changed_by = NULL WHERE id = %s",
        (order_id,),
    )
    _run_backfill(db_cursor)
    first = _state_row(db_cursor, order_id)
    _run_backfill(db_cursor)
    assert _state_row(db_cursor, order_id) == first


# ═════════════════════════════════════════════════════════════════
# Administrative exits
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_close_sets_state_mirrors_status_and_preserves_prior_status(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="ready")
    _add_line(db_cursor, order_id, product_id, 100)

    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["state"] == "closed"
    assert body["state_reason"] == "short_closed"
    assert body["status"] == "shipped", "mirror must keep legacy status consumers right"

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "closed"
    assert row["status"] == "shipped"
    assert row["status_before_exit"] == "ready"
    assert row["state_changed_at"] is not None


@pytest.mark.db
def test_close_is_rejected_when_already_closed(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              state="closed", reason="short_closed")
    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "ORDER_NOT_OPEN"


@pytest.mark.db
def test_cancel_rejected_when_partially_shipped_and_points_at_short_closed(db_cursor, client):
    """The headline rule: you cannot un-ship pounds that physically left."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="partial_ship")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=40, status="partial")
    _post_ship(db_cursor, line_id, product_id, lot_id, 40, recorded=40)

    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "mode": "commit"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ORDER_ALREADY_SHIPPED"
    assert detail["fulfillment"] == "partial"
    assert detail["suggested_reason"] == "short_closed"
    assert "short_closed" in detail["message"]
    # and nothing was written
    assert _state_row(db_cursor, order_id)["state"] == "open"


@pytest.mark.db
def test_cancel_allowed_when_unshipped(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="confirmed")
    _add_line(db_cursor, order_id, product_id, 100)

    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "mode": "commit"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == "cancelled"
    assert body["status"] == "cancelled"
    assert _state_row(db_cursor, order_id)["status_before_exit"] == "confirmed"


@pytest.mark.db
def test_cancel_allowed_after_its_only_shipment_is_voided(db_cursor, client):
    """Voiding walks fulfillment back to unshipped, which re-opens the door."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="partial_ship")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=40, status="partial")
    txn = _post_ship(db_cursor, line_id, product_id, lot_id, 40, recorded=40)

    blocked = client.post(f"/sales/orders/{order_id}/cancel",
                          json={"reason": "customer_cancelled", "mode": "commit"})
    assert blocked.status_code == 409

    _void_shipment(client, txn, "cancel-after-void test")
    allowed = client.post(f"/sales/orders/{order_id}/cancel",
                          json={"reason": "customer_cancelled", "mode": "commit"})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["state"] == "cancelled"


@pytest.mark.db
def test_reason_other_requires_a_note(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "other", "mode": "commit"})
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error_code"] == "STATE_NOTE_REQUIRED"

    ok = client.post(f"/sales/orders/{order_id}/cancel",
                     json={"reason": "other", "note": "customer went bust",
                           "mode": "commit"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["state_note"] == "customer went bust"


@pytest.mark.db
@pytest.mark.parametrize("reason", ["duplicate", "superseded"])
def test_duplicate_and_superseded_require_related_so_id(db_cursor, client, reason):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    other_id, _ = _seed_order(db_cursor, customer_id, token)

    missing = client.post(f"/sales/orders/{order_id}/cancel",
                          json={"reason": reason, "mode": "commit"})
    assert missing.status_code == 422, missing.text
    assert missing.json()["detail"]["error_code"] == "RELATED_SO_REQUIRED"

    ok = client.post(f"/sales/orders/{order_id}/cancel",
                     json={"reason": reason, "related_so_id": other_id,
                           "mode": "commit"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["related_so_id"] == other_id


@pytest.mark.db
def test_reopen_restores_the_status_the_order_had_before_it_closed(db_cursor, client):
    """Owner ruling 2: a 'ready' order comes back 'ready', not 'confirmed'."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="ready")

    closed = client.post(f"/sales/orders/{order_id}/close",
                         json={"reason": "short_closed", "mode": "commit"})
    assert closed.status_code == 200, closed.text

    reopened = client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert reopened.status_code == 200, reopened.text
    body = reopened.json()
    assert body["state"] == "open"
    assert body["state_reason"] is None
    assert body["status"] == "ready"

    row = _state_row(db_cursor, order_id)
    assert row["status"] == "ready"
    assert row["status_before_exit"] is None, "the carry-over slot is cleared on reopen"


@pytest.mark.db
def test_reopen_falls_back_to_confirmed_when_no_prior_status_was_captured(db_cursor, client):
    """Backfilled rows never went through close, so they have no carry-over."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped",
                              state="closed", reason="shipped_recorded")
    db_cursor.execute(
        "UPDATE sales_orders SET status_before_exit = NULL WHERE id = %s", (order_id,))

    resp = client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "confirmed"
    assert _state_row(db_cursor, order_id)["status"] == "confirmed"


@pytest.mark.db
def test_reopen_works_from_cancelled_too(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              state="cancelled", reason="customer_cancelled")
    resp = client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "open"


@pytest.mark.db
def test_reopen_rejected_on_an_open_order(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "ORDER_NOT_CLOSED"


# ─── reservations ────────────────────────────────────────────────

@pytest.mark.db
def test_close_releases_reservations_and_audits_the_release(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    alloc_id = _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit",
                             "changed_by": "office-jo"})
    assert resp.status_code == 200, resp.text
    assert [r["id"] for r in resp.json()["reservations_released"]] == [alloc_id]
    assert resp.json()["state_changed_by"] == "office-jo"

    db_cursor.execute(
        "SELECT status, released_at, released_by, release_reason "
        "  FROM sales_order_allocations WHERE id = %s",
        (alloc_id,),
    )
    row = db_cursor.fetchone()
    assert row["status"] == "released"
    assert row["released_at"] is not None, "the audit is the released_at/by/reason stamp"
    assert row["release_reason"] == "order_closed"
    # released_by is the CALLER SURFACE, never the body's changed_by: the body
    # is a claim about who asked for the exit, not about who released stock.
    assert row["released_by"] != "office-jo"
    assert row["released_by"] is None, "master key with no tag -> NULL"


@pytest.mark.db
def test_cancel_releases_reservations_and_audits_the_release(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    alloc_id = _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "mode": "commit"})
    assert resp.status_code == 200, resp.text

    db_cursor.execute(
        "SELECT status, released_at, release_reason FROM sales_order_allocations WHERE id = %s",
        (alloc_id,),
    )
    row = db_cursor.fetchone()
    assert row["status"] == "released"
    assert row["release_reason"] == "order_cancelled"
    assert row["released_at"] is not None


@pytest.mark.db
def test_preview_reports_the_reservations_and_state_without_writing(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="ready")
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    alloc_id = _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "preview"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "preview"
    assert body["resulting_state"] == "closed"
    assert body["resulting_state_reason"] == "short_closed"
    assert [r["id"] for r in body["reservations_to_release"]] == [alloc_id]

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "open", "preview must not write"
    assert row["status"] == "ready"
    db_cursor.execute("SELECT status FROM sales_order_allocations WHERE id = %s", (alloc_id,))
    assert db_cursor.fetchone()["status"] == "active"


# ─── no ledger, no trace ─────────────────────────────────────────

@pytest.mark.db
def test_state_changes_create_no_ledger_rows_and_no_trace_events(db_cursor, client, monkeypatch):
    """Administrative exits are bookkeeping about intent, not stock movement.

    trace emission is forced ON so this proves the endpoints never call it,
    rather than proving the feature flag happened to be off.
    """
    monkeypatch.setattr(main, "trace_emit_enabled", lambda: True)

    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="ready")
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    ledger_before = _ledger_count(db_cursor)
    trace_before = _trace_count(db_cursor)

    assert client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit"}).status_code == 200
    assert client.post(f"/sales/orders/{order_id}/reopen",
                       json={"mode": "commit"}).status_code == 200
    assert client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "mode": "commit"}).status_code == 200

    assert _ledger_count(db_cursor) == ledger_before, "no ledger postings"
    assert _trace_count(db_cursor) == trace_before, "no trace events"


# ─── attribution ─────────────────────────────────────────────────

@pytest.mark.db
def test_changed_by_prefers_the_body_field_then_the_source_tag(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_a, _ = _seed_order(db_cursor, customer_id, token)
    order_b, _ = _seed_order(db_cursor, customer_id, token)

    verbatim = client.post(f"/sales/orders/{order_a}/close",
                           json={"reason": "short_closed", "mode": "commit",
                                 "changed_by": "floor-tablet-2"})
    assert verbatim.json()["state_changed_by"] == "floor-tablet-2"

    # master key with no tag → NULL, exactly like created_by elsewhere.
    # Never the 'legacy-shared-key' placeholder.
    untagged = client.post(f"/sales/orders/{order_b}/close",
                           json={"reason": "short_closed", "mode": "commit"})
    assert untagged.json()["state_changed_by"] is None
    assert _state_row(db_cursor, order_b)["state_changed_by"] != "legacy-shared-key"


@pytest.mark.db
def test_dashboard_key_is_recorded_as_the_dashboard_surface(db_cursor, client, monkeypatch):
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "sostate-dashboard-key")
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)

    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit"},
                       headers={"X-API-Key": "sostate-dashboard-key"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state_changed_by"] == "dashboard"


# ═════════════════════════════════════════════════════════════════
# Derived fulfillment
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_fulfillment_tracks_the_ledger_not_the_recorded_column(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    txn = _post_ship(db_cursor, line_id, product_id, lot_id, 100, recorded=100)

    assert client.get(f"/sales/orders/{order_id}").json()["fulfillment"] == "shipped"

    _void_shipment(client, txn, "fulfillment void test")
    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["fulfillment"] == "unshipped", "a voided shipment un-ships the order"
    assert body["shipped_recorded_lb"] == pytest.approx(100), "recorded pounds are untouched"


@pytest.mark.db
def test_fulfillment_ignores_cancelled_lines(db_cursor, client):
    """A cancelled line's pounds are not outstanding, so the order reads shipped."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="partial_ship")
    shipped_line = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _post_ship(db_cursor, shipped_line, product_id, lot_id, 100, recorded=100)
    cancelled_line = _add_line(db_cursor, order_id, product_id, 900, status="cancelled")

    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["fulfillment"] == "shipped"

    # un-cancel it and the same order is only partially fulfilled
    db_cursor.execute(
        "UPDATE sales_order_lines SET line_status = 'pending' WHERE id = %s",
        (cancelled_line,),
    )
    assert client.get(f"/sales/orders/{order_id}").json()["fulfillment"] == "partial"


@pytest.mark.db
def test_fulfillment_partial_and_unshipped(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    unshipped_id, _ = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, unshipped_id, product_id, 100)
    assert client.get(f"/sales/orders/{unshipped_id}").json()["fulfillment"] == "unshipped"

    partial_id, _ = _seed_order(db_cursor, customer_id, token, status="partial_ship")
    line_id = _add_line(db_cursor, partial_id, product_id, 100, shipped=40, status="partial")
    _post_ship(db_cursor, line_id, product_id, lot_id, 40, recorded=40)
    assert client.get(f"/sales/orders/{partial_id}").json()["fulfillment"] == "partial"


# ═════════════════════════════════════════════════════════════════
# Tiered health — v2.1, time-aware and factory-local
#
# The tier matrix below drives compute_so_health() directly, with a pinned
# `today` and synthetic line readiness. That is deliberate: the tier rules are
# a pure function of (shortage, ship date, readiness flag, fulfillment), and
# proving every row plus the boundaries through the database would cost a
# seeded order apiece to test arithmetic no database is involved in. The
# DB-backed tests that follow prove the WIRING — that the endpoint feeds this
# function real readiness and returns what it produces.
#
# The clock tests below the matrix are the exception: they pin an instant
# instead of a date, because what they are proving is which timezone the
# unpinned `today` comes from.
# ═════════════════════════════════════════════════════════════════

_TODAY = date(2026, 6, 15)


def _fake_line(shortage=0.0, unallocated=0.0, line_id=1, sku="SKU-1",
               product=None):
    """One entry shaped like `_line_readiness()` yields, with only the fields
    health reads populated — including `product`, which the readiness query
    already carries and the info strings now name."""
    return {
        "line_id": line_id,
        "sku": sku,
        "product": product,
        "readiness": {
            "shortage_lb": shortage,
            "unallocated_need_lb": unallocated,
        },
    }


def _health(*, days_out=None, shortages=(), floor_ready=True, state="open",
            fulfillment="unshipped", unallocated=0.0, unallocateds=(),
            products=(), today=_TODAY):
    """compute_so_health() addressed in the vocabulary of the matrix.

    `unallocated` puts pounds on the first line (or invents one); `unallocateds`
    spreads a tuple of them across as many lines, which is what the aggregation
    rule needs. `products` names those lines when the label is under test.
    """
    lines = [
        _fake_line(shortage=shortage, line_id=i + 1, sku=f"SKU-{i + 1}")
        for i, shortage in enumerate(shortages)
    ]
    if unallocateds:
        lines = [
            _fake_line(unallocated=pounds, line_id=i + 1, sku=f"SKU-{i + 1}",
                       product=(products[i] if i < len(products) else None))
            for i, pounds in enumerate(unallocateds)
        ]
    elif unallocated:
        if lines:
            lines[0]["readiness"]["unallocated_need_lb"] = unallocated
            if products:
                lines[0]["product"] = products[0]
        else:
            lines = [_fake_line(unallocated=unallocated,
                                product=(products[0] if products else None))]
    return main.compute_so_health(
        state=state,
        fulfillment=fulfillment,
        requested_ship_date=(None if days_out is None
                             else today + timedelta(days=days_out)),
        floor_ready=floor_ready,
        line_readiness=lines,
        today=today,
    )


# (case id, kwargs, expected level, exact reasons)
_TIER_MATRIX = [
    # ── critical: a shortage inside the 5-day window, or already past ──────
    (
        "shortage-3-days-out",
        dict(days_out=3, shortages=(500, 235)),
        "critical",
        ["Short 735 lb on 2 lines — ships in 3 days"],
    ),
    (
        "shortage-exactly-the-boundary",           # ship_by == today + 5
        dict(days_out=5, shortages=(500,)),
        "critical",
        ["Short 500 lb — ships in 5 days"],
    ),
    (
        "shortage-exactly-today",                  # ship_by == today
        dict(days_out=0, shortages=(500,)),
        "critical",
        ["Short 500 lb — ships today"],
    ),
    (
        "shortage-yesterday",                      # ship_by == today - 1
        dict(days_out=-1, shortages=(500,)),
        "critical",
        ["Short 500 lb — 1 day overdue"],
    ),
    (
        "shortage-long-overdue",
        dict(days_out=-12, shortages=(500, 235)),
        "critical",
        ["Short 735 lb on 2 lines — 12 days overdue"],
    ),
    (
        "shortage-tomorrow",
        dict(days_out=1, shortages=(500,)),
        "critical",
        ["Short 500 lb — ships tomorrow"],
    ),
    # ── warning: a shortage further out than the window ───────────────────
    (
        "shortage-one-day-past-the-boundary",      # ship_by == today + 6
        dict(days_out=6, shortages=(500,)),
        "warning",
        ["Short 500 lb — ships in 6 days"],
    ),
    (
        "shortage-exactly-the-warning-boundary",   # ship_by == today + 10
        dict(days_out=10, shortages=(500,)),
        "warning",
        ["Short 500 lb — ships in 10 days"],
    ),
    (
        "shortage-with-no-ship-date",              # no deadline to be inside of
        dict(days_out=None, shortages=(500,)),
        "warning",
        ["Short 500 lb"],
    ),
    # ── info: a shortage further out than the warning window ───────────────
    #
    # The level does not move and `reasons` stays empty: an unproduced order
    # with time to spare is the normal state of a make-to-order book.
    (
        "shortage-one-day-past-the-warning-boundary",   # ship_by == today + 11
        dict(days_out=11, shortages=(500,)),
        "quiet",
        [],
        ["Short 500 lb — ships in 11 days"],
    ),
    (
        "shortage-13-days-out",
        dict(days_out=13, shortages=(1400,)),
        "quiet",
        [],
        ["Short 1,400 lb — ships in 13 days"],
    ),
    (
        "shortage-14-days-out",
        dict(days_out=14, shortages=(500,)),
        "quiet",
        [],
        ["Short 500 lb — ships in 14 days"],
    ),
    (
        "shortage-months-away",
        dict(days_out=90, shortages=(500, 235)),
        "quiet",
        [],
        ["Short 735 lb on 2 lines — ships in 90 days"],
    ),
    # ── warning: overdue with the stock on hand ───────────────────────────
    (
        "overdue-12-days-no-shortage",
        dict(days_out=-12),
        "warning",
        ["12 days overdue — stock on hand"],
    ),
    (
        "overdue-yesterday-no-shortage",
        dict(days_out=-1),
        "warning",
        ["1 day overdue — stock on hand"],
    ),
    (
        "overdue-but-fully-shipped",               # nothing left to chase
        dict(days_out=-12, fulfillment="shipped"),
        "quiet",
        [],
    ),
    # ── warning: Ready to Ship unset with the date closing in ─────────────
    (
        "not-ready-tomorrow",
        dict(days_out=1, floor_ready=False),
        "warning",
        ["Not Ready to Ship — ships tomorrow"],
    ),
    (
        "not-ready-exactly-the-boundary",          # ship_by == today + 2
        dict(days_out=2, floor_ready=False),
        "warning",
        ["Not Ready to Ship — ships in 2 days"],
    ),
    (
        "not-ready-today",
        dict(days_out=0, floor_ready=False),
        "warning",
        ["Not Ready to Ship — ships today"],
    ),
    (
        "not-ready-one-day-past-the-boundary",     # ship_by == today + 3
        dict(days_out=3, floor_ready=False),
        "quiet",
        [],
    ),
    (
        "not-ready-with-no-ship-date",
        dict(days_out=None, floor_ready=False),
        "quiet",
        [],
    ),
    # ── several reasons on one order: highest tier wins, all are listed ────
    (
        # v2.1: the Not-Ready reason is suppressed by the shortage. The order
        # cannot ship for want of material; the missing flag is not news.
        "shortage-and-not-ready",
        dict(days_out=2, shortages=(500,), floor_ready=False),
        "critical",
        ["Short 500 lb — ships in 2 days"],
    ),
    (
        "overdue-and-not-ready",                   # stock on hand: flag kept
        dict(days_out=-3, floor_ready=False),
        "warning",
        ["3 days overdue — stock on hand",
         "Not Ready to Ship — 3 days overdue"],
    ),
    (
        "shortage-and-not-ready-both-due-today",
        dict(days_out=0, shortages=(500,), floor_ready=False),
        "critical",
        ["Short 500 lb — ships today"],
    ),
    (
        "overdue-shortage-and-not-ready",          # only the shortage speaks
        dict(days_out=-4, shortages=(500,), floor_ready=False),
        "critical",
        ["Short 500 lb — 4 days overdue"],
    ),
    # ── quiet ─────────────────────────────────────────────────────────────
    (
        "nothing-wrong",
        dict(days_out=30),
        "quiet",
        [],
    ),
    (
        "no-ship-date-nothing-wrong",
        dict(days_out=None),
        "quiet",
        [],
    ),
    # ── closed and cancelled are quiet no matter what is wrong ────────────
    (
        "closed-with-an-overdue-shortage",
        dict(days_out=-12, shortages=(500,), floor_ready=False, state="closed"),
        "quiet",
        [],
    ),
    (
        "cancelled-with-an-overdue-shortage",
        dict(days_out=-12, shortages=(500,), floor_ready=False,
             state="cancelled"),
        "quiet",
        [],
    ),
]


@pytest.mark.parametrize(
    "kwargs,expected_level,expected_reasons,expected_info",
    # Rows carry an expected-`info` list only when they have one to assert;
    # the rest keep the three-column shape they have always had.
    [(case[1], case[2], case[3], case[4] if len(case) > 4 else [])
     for case in _TIER_MATRIX],
    ids=[case[0] for case in _TIER_MATRIX],
)
def test_health_tier_matrix(kwargs, expected_level, expected_reasons,
                            expected_info):
    """Every row of the v2.1 tier table, plus both boundary days on all three
    windows: critical at today+5 vs warning at today+6, warning at today+10 vs
    info at today+11, Not-Ready at today+2 vs quiet at today+3."""
    health = _health(**kwargs)
    assert health["level"] == expected_level, health
    assert health["reasons"] == expected_reasons, health
    if expected_info:
        assert health["info"] == expected_info, health


def test_health_reasons_carry_the_pounds_the_lines_and_the_date():
    """The reason is the whole instruction: how short, over how many lines,
    and by when. A bare 'Short' tells the operator to go open the order."""
    health = _health(days_out=3, shortages=(500, 235))
    assert health["reasons"] == ["Short 735 lb on 2 lines — ships in 3 days"]


def test_health_single_short_line_does_not_say_on_1_lines():
    health = _health(days_out=8, shortages=(500,))
    assert health["reasons"] == ["Short 500 lb — ships in 8 days"]


def test_the_info_only_shortage_is_worded_exactly_like_the_reason():
    """Crossing the warning window changes which list the line lands in, not
    what it says — including the 'on N lines' rule."""
    one = _health(days_out=14, shortages=(500,))
    assert one["info"] == ["Short 500 lb — ships in 14 days"]
    two = _health(days_out=14, shortages=(500, 235))
    assert two["info"] == ["Short 735 lb on 2 lines — ships in 14 days"]


def test_health_closed_and_cancelled_carry_no_info_either():
    """Quiet means silent: an order off the board reports nothing at all,
    not even the unallocated pounds it still nominally has."""
    for state in ("closed", "cancelled"):
        health = _health(days_out=-12, shortages=(500,), unallocated=60,
                         state=state)
        assert health == {"level": "quiet", "reasons": [], "info": [],
                          "info_detail": []}, state


# ═════════════════════════════════════════════════════════════════
# Health strings — aggregation, one number format, product names
#
# These are the only tests of the *wording*. The tier matrix above proves which
# list a line lands in; these prove what it says once it is there. They drive
# compute_so_health() directly for the same reason the matrix does: string
# assembly is a pure function of the readiness rows, and the DB-backed pair at
# the end proves the endpoint feeds it real ones.
# ═════════════════════════════════════════════════════════════════

# ── unallocated pounds aggregate to one entry per order ───────────────────

def test_unallocated_on_one_line_names_that_line_and_says_no_count(monkeypatch):
    """With a single line there is no 'across N lines' to say — and nothing to
    disambiguate, so the line names itself instead."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30, unallocateds=(60,),
                     products=("Granola SS Chocolate Chip",))
    assert health["info"] == [
        "60 lb not allocated on Granola SS Chocolate Chip (SKU-1)"]


def test_unallocated_across_many_lines_is_one_entry_carrying_the_total(monkeypatch):
    """Five lines, one sentence. v2.1 emitted five near-identical sentences and
    left the total — the number the operator is asking for — out of all five."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30,
                     unallocateds=(10000, 6000, 4000, 3000, 1000))
    assert health["info"] == ["24,000 lb not allocated across 5 lines"]


def test_unallocated_two_lines_still_says_across_2_lines(monkeypatch):
    """The 'omit when N is 1' rule is about N being 1, not about N being small."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30, unallocateds=(600, 400))
    assert health["info"] == ["1,000 lb not allocated across 2 lines"]


def test_unallocated_lines_below_the_epsilon_are_not_counted(monkeypatch):
    """A rounding crumb is not a line. It must not inflate the count, and it
    must not turn a one-line order into an 'across 2 lines' one."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30, unallocateds=(60, 0.0),
                     products=("Granola SS Chocolate Chip", "Granola Maple"))
    assert health["info"] == [
        "60 lb not allocated on Granola SS Chocolate Chip (SKU-1)"]
    assert [d["line_id"] for d in health["info_detail"]] == [1]


# ── the not-enforced note is appended once, to the one entry ──────────────

def test_not_enforced_note_is_appended_once_across_many_lines(monkeypatch):
    """STATUS-011: the caveat is stated once where it first applies. Repeated
    down a list it stops being read, including the time it mattered."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    health = _health(days_out=30, unallocateds=(10000, 6000, 4000, 3000, 1000))
    assert health["info"] == [
        "24,000 lb not allocated across 5 lines (allocations not enforced)"]
    assert health["info"][0].count("allocations not enforced") == 1


def test_enforced_on_carries_no_note(monkeypatch):
    """Same pounds, same lines, flag on: the sentence ends at the count."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30, unallocateds=(10000, 6000, 4000, 3000, 1000))
    assert health["info"] == ["24,000 lb not allocated across 5 lines"]
    assert "not enforced" not in health["info"][0]


def test_the_note_is_the_only_difference_between_enforced_and_not(monkeypatch):
    """Stated as a pair so the suffix cannot drift into carrying meaning: the
    pounds are unallocated either way, the flag only says whether that blocks a
    shipment."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    on = _health(days_out=30, unallocateds=(600, 400))
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    off = _health(days_out=30, unallocateds=(600, 400))
    assert off["info"][0] == on["info"][0] + " (allocations not enforced)"
    assert off["level"] == on["level"] == "quiet"
    assert off["info_detail"] == on["info_detail"]


def test_the_not_enforced_note_never_raises_the_level(monkeypatch):
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    health = _health(days_out=30, unallocateds=(24000,))
    assert health["level"] == "quiet", health
    assert health["reasons"] == []


# ── info_detail — the rows behind the sentence ────────────────────────────

def test_info_detail_carries_one_row_per_unallocated_line():
    """The per-line facts are not lost to aggregation, they move here: the
    popover expands the sentence back into the lines that made it."""
    health = _health(days_out=30, unallocateds=(10000, 14000),
                     products=("Granola SS Chocolate Chip", "Granola Maple"))
    assert health["info_detail"] == [
        {"line_id": 1, "sku": "SKU-1",
         "product_name": "Granola SS Chocolate Chip",
         "unallocated_lb": 10000.0},
        {"line_id": 2, "sku": "SKU-2", "product_name": "Granola Maple",
         "unallocated_lb": 14000.0},
    ]


def test_info_detail_pounds_sum_to_the_pounds_in_the_sentence():
    health = _health(days_out=30, unallocateds=(10000, 6000, 4000, 3000, 1000))
    total = sum(d["unallocated_lb"] for d in health["info_detail"])
    assert total == 24000.0
    assert main._fmt_number(total) in health["info"][0]


def test_info_detail_is_always_present_and_empty_when_there_is_nothing():
    """Shape is the contract: a caller may read `info_detail` without checking
    whether the key is there."""
    for health in (_health(days_out=30),
                   _health(days_out=3, shortages=(500,)),
                   _health(days_out=-12, state="closed")):
        assert health["info_detail"] == [], health


def test_info_detail_is_unaffected_by_the_enforcement_flag(monkeypatch):
    """The note is wording. The rows are facts."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    assert _health(days_out=30, unallocateds=(60,))["info_detail"] == [
        {"line_id": 1, "sku": "SKU-1", "product_name": None,
         "unallocated_lb": 60.0}]


# ── one number formatter — STATUS-006 ─────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (999, "999"),                 # below the separator
    (1000, "1,000"),              # the separator boundary
    (999.4, "999"),               # rounds down, stays below
    (999.5, "1,000"),             # rounds UP and gains a separator
    (607.5, "608"),               # half rounds up, not to even (606.5 -> 607)
    (606.5, "607"),               # the control: Python's own format gives 606
    (12345.4, "12,345"),
    (12345.5, "12,346"),
    (13500.0000, "13,500"),       # the stored representation, formatted
    (0, "0"),
    (0.5, "1"),
    (1000000, "1,000,000"),
    (None, "0"),
])
def test_fmt_number_boundaries(value, expected):
    """STATUS-006 in one table: separators always, pounds whole, no trailing
    decimal, half rounds up. A shortage that prints smaller than it is reads as
    less urgent than it is."""
    assert main._fmt_number(value) == expected


def test_no_health_string_ever_carries_a_decimal_point_or_a_bare_thousand(monkeypatch):
    """The audit STATUS-006 names, run against the strings this module makes:
    three or more decimal places anywhere, or a value of 1,000 or more without
    a separator."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=3, shortages=(1234.5, 6789.4),
                     unallocateds=())
    with_pounds = _health(days_out=30, unallocateds=(13500.0000, 2499.6))
    for text in health["reasons"] + health["info"] + with_pounds["info"]:
        assert not re.search(r"\d\.\d", text), text
        assert not re.search(r"(?<![\d,])\d{4,}", text), text
    assert health["reasons"] == ["Short 8,024 lb on 2 lines — ships in 3 days"]
    assert with_pounds["info"] == ["16,000 lb not allocated across 2 lines"]


def test_the_shortage_reason_is_formatted_the_same_in_reasons_and_in_info():
    """Crossing the warning window changes which list the line lands in, not
    how its number is written."""
    near = _health(days_out=3, shortages=(1400,))
    far = _health(days_out=90, shortages=(1400,))
    assert near["reasons"] == ["Short 1,400 lb — ships in 3 days"]
    assert far["info"] == ["Short 1,400 lb — ships in 90 days"]


def test_a_four_figure_day_count_is_separated_too():
    """Counts go through the same formatter as pounds — an order four years
    overdue is a data problem, and it should read like one."""
    health = _health(days_out=-1200)
    assert health["reasons"] == ["1,200 days overdue — stock on hand"]


# ── product name (SKU) ────────────────────────────────────────────────────

def test_a_named_line_renders_name_then_code(monkeypatch):
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30, unallocateds=(24000,),
                     products=("Granola SS Chocolate Chip",))
    assert health["info"] == [
        "24,000 lb not allocated on Granola SS Chocolate Chip (SKU-1)"]


def test_a_line_with_no_product_name_falls_back_to_the_bare_code(monkeypatch):
    """Degrade to whichever half exists — never print an empty parenthesis or
    a dangling name."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    health = _health(days_out=30, unallocateds=(60,))
    assert health["info"] == ["60 lb not allocated on SKU-1"]


@pytest.mark.parametrize("line,expected", [
    ({"product": "Granola SS Chocolate Chip", "sku": "70003"},
     "Granola SS Chocolate Chip (70003)"),
    ({"product": "Granola Maple", "sku": None}, "Granola Maple"),
    ({"product": None, "sku": "70003"}, "70003"),
    ({"product": "  Granola Maple  ", "sku": "  70003  "},
     "Granola Maple (70003)"),
    ({"product": None, "sku": None}, "line"),
    ({"product": "", "sku": ""}, "line"),
    ({}, "line"),
])
def test_so_line_label(line, expected):
    assert main._so_line_label(line) == expected


@pytest.mark.db
def test_health_info_names_the_product_end_to_end(db_cursor, client, monkeypatch):
    """The wiring: the name and the code come from the readiness query's own
    columns, so naming the line costs no extra round trip."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    db_cursor.execute("SELECT name, odoo_code FROM products WHERE id = %s",
                      (product_id,))
    product = db_cursor.fetchone()
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 40, lot_id=lot_id)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["info"] == [
        f"60 lb not allocated on {product['name']} ({product['odoo_code']})"
        " (allocations not enforced)"], health["info"]
    assert health["info_detail"] == [{
        "line_id": line_id,
        "sku": product["odoo_code"],
        "product_name": product["name"],
        "unallocated_lb": 60.0,
    }], health["info_detail"]


@pytest.mark.db
def test_health_info_aggregates_two_lines_end_to_end(db_cursor, client, monkeypatch):
    """Two partially-allocated lines on one order: one sentence, both rows."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: True)
    customer_id, token = _seed_customer(db_cursor)
    first_id, first_lot = _seed_product(db_cursor, token, with_lot=True, stock=5000)
    second_id, second_lot = _seed_product(db_cursor, token, with_lot=True, stock=5000)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    line_one = _add_line(db_cursor, order_id, first_id, 1000)
    line_two = _add_line(db_cursor, order_id, second_id, 800)
    _allocate(db_cursor, order_id, line_one, first_id, 400, lot_id=first_lot)
    _allocate(db_cursor, order_id, line_two, second_id, 200, lot_id=second_lot)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["info"] == ["1,200 lb not allocated across 2 lines"], \
        health["info"]
    assert sorted(d["line_id"] for d in health["info_detail"]) == sorted(
        [line_one, line_two]), health["info_detail"]
    assert sum(d["unallocated_lb"] for d in health["info_detail"]) == 1200.0


# ── the Not-Ready reason yields to a shortage ─────────────────────────────

def test_not_ready_is_suppressed_when_the_same_order_is_short():
    """An order that cannot ship for want of material is not also news for
    want of a flag. The shortage reason stays; the flag reason goes."""
    short = _health(days_out=1, shortages=(500,), floor_ready=False)
    assert short["reasons"] == ["Short 500 lb — ships tomorrow"]
    assert not any("Not Ready" in r for r in short["reasons"]), short


def test_not_ready_is_kept_when_the_stock_is_on_hand():
    """Same order, same date, nothing short: now the flag IS the blocker."""
    stocked = _health(days_out=1, shortages=(), floor_ready=False)
    assert stocked["reasons"] == ["Not Ready to Ship — ships tomorrow"]


def test_suppression_is_the_only_difference_between_the_two():
    """Stated as a pair so the rule cannot be half-reverted: identical inputs
    apart from the shortage, and the flag reason appears in exactly one."""
    kwargs = dict(days_out=2, floor_ready=False)
    short = _health(shortages=(500,), **kwargs)["reasons"]
    stocked = _health(shortages=(), **kwargs)["reasons"]
    assert [r for r in short if "Not Ready" in r] == []
    assert [r for r in stocked if "Not Ready" in r] == [
        "Not Ready to Ship — ships in 2 days"]


def test_a_suppressed_not_ready_does_not_reappear_as_info():
    """Suppressed means gone, not demoted: `info` carries pounds, not flags."""
    health = _health(days_out=1, shortages=(500,), floor_ready=False)
    assert not any("Not Ready" in i for i in health["info"]), health["info"]


# ── the critical window is an env var ─────────────────────────────────────

def test_health_critical_window_widens_with_the_env_var(monkeypatch):
    """A shortage 7 days out is a warning at the default and critical at 10."""
    assert _health(days_out=7, shortages=(500,))["level"] == "warning"
    monkeypatch.setenv("SO_HEALTH_CRITICAL_DAYS", "10")
    assert _health(days_out=7, shortages=(500,))["level"] == "critical"


def test_health_critical_window_narrows_with_the_env_var(monkeypatch):
    monkeypatch.setenv("SO_HEALTH_CRITICAL_DAYS", "1")
    assert _health(days_out=3, shortages=(500,))["level"] == "warning"
    assert _health(days_out=1, shortages=(500,))["level"] == "critical"


def test_health_critical_window_defaults_to_five(monkeypatch):
    monkeypatch.delenv("SO_HEALTH_CRITICAL_DAYS", raising=False)
    assert main._so_health_critical_days() == 5


@pytest.mark.parametrize("bad", ["", "   ", "soon", "5.5", "-3"])
def test_health_critical_window_falls_back_on_a_bad_value(monkeypatch, bad):
    """A typo in an env var must not 500 a read path — it falls back."""
    monkeypatch.setenv("SO_HEALTH_CRITICAL_DAYS", bad)
    assert main._so_health_critical_days() == 5
    assert _health(days_out=3, shortages=(500,))["level"] == "critical"


# ── the warning window is an env var too ──────────────────────────────────

def test_health_warning_window_widens_with_the_env_var(monkeypatch):
    """A shortage 14 days out is info at the default and a warning at 20."""
    assert _health(days_out=14, shortages=(500,))["level"] == "quiet"
    monkeypatch.setenv("SO_HEALTH_WARNING_DAYS", "20")
    assert _health(days_out=14, shortages=(500,))["level"] == "warning"


def test_health_warning_window_narrows_with_the_env_var(monkeypatch):
    monkeypatch.setenv("SO_HEALTH_WARNING_DAYS", "7")
    assert _health(days_out=8, shortages=(500,))["level"] == "quiet"
    assert _health(days_out=7, shortages=(500,))["level"] == "warning"


def test_health_warning_window_defaults_to_ten(monkeypatch):
    monkeypatch.delenv("SO_HEALTH_WARNING_DAYS", raising=False)
    assert main._so_health_warning_days() == 10


@pytest.mark.parametrize("bad", ["", "   ", "later", "10.5", "-3"])
def test_health_warning_window_falls_back_on_a_bad_value(monkeypatch, bad):
    """Same rule as the critical window: a typo must not 500 a read path."""
    monkeypatch.setenv("SO_HEALTH_WARNING_DAYS", bad)
    assert main._so_health_warning_days() == 10
    assert _health(days_out=8, shortages=(500,))["level"] == "warning"


def test_the_two_windows_are_independent(monkeypatch):
    """Moving one must not move the other — they answer different questions."""
    monkeypatch.setenv("SO_HEALTH_CRITICAL_DAYS", "2")
    assert main._so_health_warning_days() == 10
    monkeypatch.setenv("SO_HEALTH_WARNING_DAYS", "30")
    assert main._so_health_critical_days() == 2
    assert _health(days_out=3, shortages=(500,))["level"] == "warning"
    assert _health(days_out=2, shortages=(500,))["level"] == "critical"
    assert _health(days_out=29, shortages=(500,))["level"] == "warning"


def test_inverted_windows_never_produce_a_level_without_a_reason(monkeypatch):
    """A misconfiguration must not paint an order red with an empty `reasons`
    list — a badge the operator cannot act on is worse than no badge."""
    monkeypatch.setenv("SO_HEALTH_CRITICAL_DAYS", "20")
    monkeypatch.setenv("SO_HEALTH_WARNING_DAYS", "5")
    health = _health(days_out=10, shortages=(500,))
    assert health["level"] == "quiet", health
    assert health["reasons"] == []
    assert health["info"] == ["Short 500 lb — ships in 10 days"]


def test_health_function_is_labelled_v2_1_and_the_shape_is_the_contract():
    """Owner ruling 4 survives v2.1: the shape is the contract, tiers are not."""
    doc = main.compute_so_health.__doc__ or ""
    assert "v2.1 — time-aware. Shape is the contract." in doc


# ═════════════════════════════════════════════════════════════════
# The clock — "today" is the FACTORY's date, never the server's
#
# These pin an INSTANT rather than a date, because what is under test is which
# timezone the unpinned `today` is read in. `_freeze()` stands the process up as
# a UTC server: datetime.now(tz) honours the tz it is handed, and date.today()
# returns the UTC calendar date — which is exactly what v2 used and what these
# tests exist to keep out. Reverting compute_so_health() to date.today() makes
# the 20:00 test below fail; that is the mutation check, and it is asserted
# directly in test_the_utc_clock_and_the_factory_clock_actually_disagree.
# ═════════════════════════════════════════════════════════════════

# 20:00 Eastern on 2026-09-10 — the UTC calendar has already rolled to the 11th.
_EVENING_ET = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
# 00:30 Eastern on 2026-09-11 — four and a half hours later, next factory day.
_AFTER_MIDNIGHT_ET = datetime(2026, 9, 11, 4, 30, tzinfo=timezone.utc)

_SHIP_DAY = date(2026, 9, 10)


def _freeze(monkeypatch, instant):
    """Pin the process clock at `instant`, UTC-server style.

    Both names are patched on `main` because both are ways to ask what day it
    is: datetime.now(tz) is the factory-local route, date.today() the server
    route. Subclasses, so every isinstance(x, date) check in main.py still holds.
    """
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    class _FrozenDate(date):
        @classmethod
        def today(cls):
            return instant.date()          # the UTC day, as a UTC server sees it

    monkeypatch.setattr(main, "datetime", _FrozenDatetime)
    monkeypatch.setattr(main, "date", _FrozenDate)


def _unpinned_health(**kwargs):
    """compute_so_health() with NO `today` argument — the point of these tests
    is which date it reaches for when the caller does not supply one."""
    kwargs.setdefault("state", "open")
    kwargs.setdefault("fulfillment", "unshipped")
    kwargs.setdefault("floor_ready", True)
    kwargs.setdefault("line_readiness", [])
    kwargs.setdefault("requested_ship_date", _SHIP_DAY)
    return main.compute_so_health(**kwargs)


def test_the_utc_clock_and_the_factory_clock_actually_disagree(monkeypatch):
    """The premise of every test below: at this instant the two answers differ.
    If this ever stops being true the rest of the clock tests prove nothing."""
    _freeze(monkeypatch, _EVENING_ET)
    assert main.date.today() == date(2026, 9, 11)      # the server's answer
    assert main._factory_today() == date(2026, 9, 10)  # the factory's answer


def test_an_order_due_today_is_not_overdue_at_8pm_eastern(monkeypatch):
    """The v2 bug, stated as a test: at 20:00 ET on the ship date the API called
    a Sep 10 order '1 day overdue' because UTC had already turned over."""
    _freeze(monkeypatch, _EVENING_ET)
    health = _unpinned_health()
    assert health["level"] == "quiet", health
    assert health["reasons"] == [], health


def test_the_same_order_is_one_day_overdue_at_half_past_midnight(monkeypatch):
    """And the factory clock does turn over — four and a half hours later."""
    _freeze(monkeypatch, _AFTER_MIDNIGHT_ET)
    health = _unpinned_health()
    assert health["level"] == "warning", health
    assert health["reasons"] == ["1 day overdue — stock on hand"], health


def test_the_ship_phrase_follows_the_factory_clock_too(monkeypatch):
    """Not just the tier — the words. 'ships today' must not read 'overdue'."""
    _freeze(monkeypatch, _EVENING_ET)
    evening = _unpinned_health(line_readiness=[_fake_line(shortage=500)])
    assert evening["reasons"] == ["Short 500 lb — ships today"], evening
    assert evening["level"] == "critical"

    _freeze(monkeypatch, _AFTER_MIDNIGHT_ET)
    after = _unpinned_health(line_readiness=[_fake_line(shortage=500)])
    assert after["reasons"] == ["Short 500 lb — 1 day overdue"], after


def test_the_shortage_windows_are_measured_from_the_factory_date(monkeypatch):
    """The windows count days from the factory's today, so at 20:00 ET a ship
    date five days out is still inside the critical window — under the UTC
    clock it would already have been counted as four."""
    _freeze(monkeypatch, _EVENING_ET)
    lines = [_fake_line(shortage=500)]
    inside = _unpinned_health(requested_ship_date=_SHIP_DAY + timedelta(days=5),
                              line_readiness=lines)
    assert inside["reasons"] == ["Short 500 lb — ships in 5 days"], inside
    assert inside["level"] == "critical"
    outside = _unpinned_health(requested_ship_date=_SHIP_DAY + timedelta(days=6),
                               line_readiness=lines)
    assert outside["reasons"] == ["Short 500 lb — ships in 6 days"], outside
    assert outside["level"] == "warning"


def test_not_ready_window_is_measured_from_the_factory_date(monkeypatch):
    """Two days out on the factory's calendar, not the server's."""
    _freeze(monkeypatch, _EVENING_ET)
    health = _unpinned_health(requested_ship_date=_SHIP_DAY + timedelta(days=2),
                             floor_ready=False)
    assert health["reasons"] == ["Not Ready to Ship — ships in 2 days"], health


def test_an_explicit_today_still_wins(monkeypatch):
    """The pinned-`today` argument the tier matrix relies on is not broken by
    the new default — a caller that supplies a date gets that date."""
    _freeze(monkeypatch, _AFTER_MIDNIGHT_ET)
    health = main.compute_so_health(
        state="open", fulfillment="unshipped", requested_ship_date=_SHIP_DAY,
        floor_ready=True, line_readiness=[], today=_SHIP_DAY)
    assert health["level"] == "quiet", health


# ── FACTORY_TZ is the knob ────────────────────────────────────────────────

def test_factory_tz_defaults_to_eastern(monkeypatch):
    monkeypatch.delenv("FACTORY_TZ", raising=False)
    assert main._factory_tz() == ZoneInfo("America/New_York")


def test_factory_tz_env_var_moves_the_calendar(monkeypatch):
    """01:00 Eastern on the 11th is 22:00 Pacific on the 10th: one instant,
    two factory dates, and the env var decides which one is 'today'."""
    _freeze(monkeypatch, datetime(2026, 9, 11, 5, 0, tzinfo=timezone.utc))
    monkeypatch.setenv("FACTORY_TZ", "America/New_York")
    assert main._factory_today() == date(2026, 9, 11)
    monkeypatch.setenv("FACTORY_TZ", "America/Los_Angeles")
    assert main._factory_today() == date(2026, 9, 10)


@pytest.mark.parametrize("bad", ["", "   ", "Mars/Olympus_Mons", "EST5EDT/nope",
                                 "../../etc/passwd"])
def test_factory_tz_falls_back_on_a_bad_value(monkeypatch, bad):
    """Same rule as the windows: a bad env var must not 500 the board."""
    monkeypatch.setenv("FACTORY_TZ", bad)
    assert main._factory_tz() == ZoneInfo("America/New_York")


def test_factory_today_is_not_the_module_level_plant_timezone(monkeypatch):
    """PLANT_TIMEZONE belongs to the ledger paths. Health reads FACTORY_TZ, and
    retuning one must not retune the other."""
    _freeze(monkeypatch, datetime(2026, 9, 11, 5, 0, tzinfo=timezone.utc))
    monkeypatch.setenv("FACTORY_TZ", "America/Los_Angeles")
    assert main._factory_today() == date(2026, 9, 10)
    assert main.datetime.now(main.PLANT_TIMEZONE).date() == date(2026, 9, 11)


# ═════════════════════════════════════════════════════════════════
# Tiered health — the endpoint is wired to it
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_health_critical_on_a_stock_shortage_with_the_date_closing_in(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=10)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=3))
    _add_line(db_cursor, order_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "critical"
    assert any("Short 90 lb" in r and "ships in 3 days" in r
               for r in health["reasons"]), health["reasons"]


@pytest.mark.db
def test_health_only_warns_on_a_shortage_inside_the_warning_window(db_cursor, client):
    """The v1 rule that made this critical is what v2 exists to fix."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=10)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=8))
    _add_line(db_cursor, order_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "warning"
    assert any("Short 90 lb" in r and "ships in 8 days" in r
               for r in health["reasons"]), health["reasons"]


@pytest.mark.db
def test_health_reports_a_shortage_months_away_as_info_only(db_cursor, client):
    """v2.1: a real shortage, a real reason to look at it eventually, and no
    tier at all — on a make-to-order book this is what 'not made yet' looks
    like, and v2 spent a warning badge on every one of them."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=10)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    _add_line(db_cursor, order_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "quiet", health
    assert health["reasons"] == [], health
    assert any("Short 90 lb" in i and "ships in 30 days" in i
               for i in health["info"]), health["info"]


@pytest.mark.db
def test_health_suppresses_not_ready_on_a_short_order_end_to_end(db_cursor, client):
    """The suppression rule through the endpoint: same order, same date, and
    the only difference is whether the stock is there."""
    customer_id, token = _seed_customer(db_cursor)
    soon = main._factory_today() + timedelta(days=1)

    short_product, _ = _seed_product(db_cursor, token, with_lot=True, stock=0)
    short_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                              ship_date=soon)
    _add_line(db_cursor, short_id, short_product, 100)
    short_health = client.get(f"/sales/orders/{short_id}").json()["health"]
    assert short_health["level"] == "critical", short_health
    assert not any("Not Ready" in r for r in short_health["reasons"]), short_health

    stocked_product, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    stocked_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                                ship_date=soon)
    line_id = _add_line(db_cursor, stocked_id, stocked_product, 100)
    _allocate(db_cursor, stocked_id, line_id, stocked_product, 100)
    stocked_health = client.get(f"/sales/orders/{stocked_id}").json()["health"]
    assert stocked_health["level"] == "warning", stocked_health
    assert "Not Ready to Ship — ships tomorrow" in stocked_health["reasons"], \
        stocked_health


@pytest.mark.db
def test_health_warning_when_overdue_and_not_shipped(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() - timedelta(days=3))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "warning"
    assert "3 days overdue — stock on hand" in health["reasons"], health["reasons"]


@pytest.mark.db
def test_health_warning_when_ready_to_ship_unset_and_ship_date_is_close(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                              ship_date=main._factory_today() + timedelta(days=2))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "warning"
    assert "Not Ready to Ship — ships in 2 days" in health["reasons"], health["reasons"]


@pytest.mark.db
def test_health_quiet_when_nothing_is_wrong(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    assert client.get(f"/sales/orders/{order_id}").json()["health"]["level"] == "quiet"


@pytest.mark.db
def test_health_info_for_unallocated_never_raises_the_level(db_cursor, client, monkeypatch):
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    _add_line(db_cursor, order_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "quiet", "info must not escalate"
    assert health["reasons"] == []
    assert any("not allocated" in i for i in health["info"]), health["info"]
    assert len(health["info_detail"]) == 1, health["info_detail"]


@pytest.mark.db
def test_health_is_quiet_once_the_order_is_closed(db_cursor, client):
    """An order off the board stops asking for attention, shortage or not."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=0)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() - timedelta(days=10))
    _add_line(db_cursor, order_id, product_id, 100)

    assert client.get(f"/sales/orders/{order_id}").json()["health"]["level"] == "critical"

    closed = client.post(f"/sales/orders/{order_id}/close",
                         json={"reason": "short_closed", "mode": "commit"})
    assert closed.status_code == 200, closed.text

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health == {"level": "quiet", "reasons": [], "info": [],
                      "info_detail": []}


@pytest.mark.db
def test_health_is_quiet_once_the_order_is_cancelled(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=0)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() - timedelta(days=10))
    _add_line(db_cursor, order_id, product_id, 100)
    client.post(f"/sales/orders/{order_id}/cancel",
                json={"reason": "customer_cancelled", "mode": "commit"})
    assert client.get(f"/sales/orders/{order_id}").json()["health"] == {
        "level": "quiet", "reasons": [], "info": [], "info_detail": []}


# ═════════════════════════════════════════════════════════════════
# Read side: fields, filter, counts
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_list_and_detail_expose_the_state_fields(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, order_number = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_id, product_id, 100)
    client.post(f"/sales/orders/{order_id}/close",
                json={"reason": "short_closed", "note": "n/a", "mode": "commit",
                      "changed_by": "office-jo"})

    expected = {"state", "state_reason", "state_note", "state_changed_at",
                "state_changed_by", "related_so_id", "fulfillment", "health"}

    detail = client.get(f"/sales/orders/{order_id}").json()
    assert expected <= set(detail)
    assert detail["state"] == "closed"
    assert detail["state_changed_by"] == "office-jo"

    listed = client.get("/sales/orders", params={"state": "closed", "limit": 200}).json()
    row = next(o for o in listed["orders"] if o["order_number"] == order_number)
    assert expected <= set(row)
    assert row["state"] == "closed"


@pytest.mark.db
def test_list_keeps_every_pre_existing_field(db_cursor, client):
    """Additive only: nothing that was in a list item may disappear."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, order_number = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_id, product_id, 100)

    row = next(o for o in client.get("/sales/orders", params={"limit": 200}).json()["orders"]
               if o["order_number"] == order_number)
    for field in ("order_id", "order_number", "customer", "order_date",
                  "requested_ship_date", "status", "customer_po",
                  "source_document_id", "line_count", "total_lb", "shipped_lb",
                  "remaining_lb", "total_units", "shipped_units",
                  "remaining_units", "pallet_lines", "ready", "ready_at",
                  "ready_by", "note", "overdue", "inventory_ready",
                  "dispatch_ready", "fulfillment_diverged", "shortage_lb",
                  "allocated_lb", "remaining_effective_lb", "blockers"):
        assert field in row, f"regression: list item lost {field}"


@pytest.mark.db
def test_list_exposes_effective_quantities_without_cancelled_or_voided_pounds(
        db_cursor, client):
    """The list's fulfillment explanation must describe the same ledger facts
    as its fulfillment value, even when recorded quantities have diverged."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    service_id, _ = _seed_product(db_cursor, token, service=True)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    active_line = _add_line(db_cursor, order_id, product_id, 100)
    _post_ship(db_cursor, active_line, product_id, lot_id, 40, recorded=40)
    voided_txn = _post_ship(db_cursor, active_line, product_id, lot_id, 60,
                            recorded=100)
    _void_shipment(client, voided_txn, "list must exclude voided shipment")
    cancelled_line = _add_line(db_cursor, order_id, product_id, 900,
                               shipped=200, status="cancelled")
    _post_ship(db_cursor, cancelled_line, product_id, lot_id, 200)
    _add_line(db_cursor, order_id, service_id, 10)

    response = client.get("/sales/orders", params={"customer": token})
    assert response.status_code == 200, response.text
    row = response.json()["orders"][0]
    assert row["order_id"] == order_id
    assert row["ordered_lb"] == pytest.approx(100)
    assert row["shipped_effective_lb"] == pytest.approx(40)
    assert row["remaining_effective_lb"] == pytest.approx(60)
    assert row["fulfillment"] == "partial"
    # Preserve the legacy line list, but give the list's pallet calculation
    # the state it needs to exclude cancelled case quantities from its total.
    pallet_lines = {line["line_id"]: line for line in row["pallet_lines"]}
    assert pallet_lines[active_line]["line_status"] == "fulfilled"
    assert pallet_lines[cancelled_line]["line_status"] == "cancelled"
    # Existing fields retain their recorded meaning for legacy callers.
    assert row["total_lb"] == pytest.approx(1000)
    assert row["shipped_lb"] == pytest.approx(300)


@pytest.mark.db
@pytest.mark.parametrize("state,reason,status,shipped,expected_overdue", [
    ("open", None, "shipped", 0, True),
    ("open", None, "confirmed", 100, False),
    ("open", None, "partial_ship", 40, True),
    ("closed", "short_closed", "confirmed", 0, False),
    ("cancelled", "customer_cancelled", "confirmed", 0, False),
])
def test_list_overdue_uses_state_and_effective_fulfillment_like_counts(
        db_cursor, client, state, reason, status, shipped, expected_overdue):
    """Legacy status can disagree with either dimension; it is not the board's
    definition of overdue. The list field, filter, and count must agree."""
    before = client.get("/sales/orders/counts").json()["overdue"]
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    order_id, _ = _seed_order(
        db_cursor, customer_id, token, state=state, reason=reason, status=status,
        ship_date=main._factory_today() - timedelta(days=2))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    if shipped:
        _post_ship(db_cursor, line_id, product_id, lot_id, shipped, recorded=shipped)

    listed = client.get("/sales/orders", params={"customer": token})
    assert listed.status_code == 200, listed.text
    assert listed.json()["orders"][0]["overdue"] is expected_overdue
    filtered = client.get("/sales/orders", params={
        "customer": token, "overdue_only": True})
    assert filtered.status_code == 200, filtered.text
    assert [row["order_id"] for row in filtered.json()["orders"]] == (
        [order_id] if expected_overdue else [])
    after = client.get("/sales/orders/counts").json()["overdue"]
    assert after - before == int(expected_overdue)


@pytest.mark.db
def test_overdue_filter_preserves_explicit_legacy_status_filter(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, _ = _seed_order(
        db_cursor, customer_id, token, status="shipped", state="open",
        ship_date=main._factory_today() - timedelta(days=2))
    _add_line(db_cursor, order_id, product_id, 100)

    for status, expected_ids in (("open", []), ("shipped", [order_id])):
        response = client.get("/sales/orders", params={
            "customer": token, "overdue_only": True, "status": status})
        assert response.status_code == 200, response.text
        assert [row["order_id"] for row in response.json()["orders"]] == expected_ids


@pytest.mark.db
def test_overdue_filter_applies_limit_after_excluding_effectively_shipped(
        db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    shipped_id, _ = _seed_order(
        db_cursor, customer_id, token,
        ship_date=main._factory_today() - timedelta(days=3))
    line_id = _add_line(db_cursor, shipped_id, product_id, 100)
    _post_ship(db_cursor, line_id, product_id, lot_id, 100, recorded=100)
    overdue_id, _ = _seed_order(
        db_cursor, customer_id, token,
        ship_date=main._factory_today() - timedelta(days=2))
    _add_line(db_cursor, overdue_id, product_id, 100)

    response = client.get("/sales/orders", params={
        "customer": token, "overdue_only": True, "limit": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert [row["order_id"] for row in body["orders"]] == [overdue_id]


@pytest.mark.db
def test_ready_flag_remains_editable_on_open_physically_shipped_order(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    order_id, order_number = _seed_order(
        db_cursor, customer_id, token, status="shipped", state="open", floor_ready=False)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _post_ship(db_cursor, line_id, product_id, lot_id, 100, recorded=100)

    for ready in (True, False):
        response = client.post(f"/sales-orders/{order_number}/ready", json={"ready": ready})
        assert response.status_code == 200, response.text
        assert response.json()["ready"] is ready
        row = client.get("/sales/orders", params={"customer": token}).json()["orders"][0]
        assert row["ready"] is ready
        assert row["state"] == "open"
        assert row["fulfillment"] == "shipped"


@pytest.mark.db
@pytest.mark.parametrize("state,reason", [
    ("closed", "short_closed"), ("cancelled", "customer_cancelled"),
])
def test_ready_flag_rejects_exited_orders_even_with_open_legacy_status(
        db_cursor, client, state, reason):
    customer_id, token = _seed_customer(db_cursor)
    _, order_number = _seed_order(
        db_cursor, customer_id, token, status="confirmed", state=state,
        reason=reason, floor_ready=False)

    response = client.post(f"/sales-orders/{order_number}/ready", json={"ready": True})
    assert response.status_code == 400, response.text
    db_cursor.execute("SELECT ready FROM sales_order_flags WHERE so_number = %s", (order_number,))
    assert db_cursor.fetchone() is None, "an exited order must not gain a ready flag"


@pytest.mark.db
def test_state_filter_selects_by_state(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    open_id, open_number = _seed_order(db_cursor, customer_id, token)
    closed_id, closed_number = _seed_order(db_cursor, customer_id, token)
    client.post(f"/sales/orders/{closed_id}/close",
                json={"reason": "short_closed", "mode": "commit"})

    numbers = {o["order_number"] for o in
               client.get("/sales/orders", params={"state": "closed", "limit": 200}).json()["orders"]}
    assert closed_number in numbers
    assert open_number not in numbers


@pytest.mark.db
def test_state_filter_rejects_an_unknown_value(client):
    resp = client.get("/sales/orders", params={"state": "banana"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "INVALID_STATE_FILTER"


@pytest.mark.db
def test_counts_endpoint_buckets(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)

    before = client.get("/sales/orders/counts").json()

    # open + Factory Ready, ship date in the future
    ready_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=True,
                              ship_date=main._factory_today() + timedelta(days=5))
    _add_line(db_cursor, ready_id, product_id, 10)

    # open + overdue + not shipped
    overdue_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                                ship_date=main._factory_today() - timedelta(days=2))
    _add_line(db_cursor, overdue_id, product_id, 10)

    # open but physically shipped (the mirror never touched status: still open)
    shipped_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                                status="partial_ship")
    line_id = _add_line(db_cursor, shipped_id, product_id, 10, shipped=10, status="fulfilled")
    _post_ship(db_cursor, line_id, product_id, lot_id, 10, recorded=10)

    closed_id, _ = _seed_order(db_cursor, customer_id, token)
    client.post(f"/sales/orders/{closed_id}/close",
                json={"reason": "short_closed", "mode": "commit"})
    cancelled_id, _ = _seed_order(db_cursor, customer_id, token)
    client.post(f"/sales/orders/{cancelled_id}/cancel",
                json={"reason": "customer_cancelled", "mode": "commit"})

    after = client.get("/sales/orders/counts").json()
    assert set(after) >= {"open", "ready_to_ship", "overdue", "shipped",
                          "closed", "cancelled"}
    assert "ready" not in after, "owner ruling 5: the key is ready_to_ship"
    assert after["open"] - before["open"] == 3
    assert after["ready_to_ship"] - before["ready_to_ship"] == 1
    assert after["overdue"] - before["overdue"] == 1
    assert after["shipped"] - before["shipped"] == 1
    assert after["closed"] - before["closed"] == 1
    assert after["cancelled"] - before["cancelled"] == 1


@pytest.mark.db
def test_counts_overdue_bucket_is_on_the_factory_clock(db_cursor, client,
                                                      monkeypatch):
    """The board's overdue COUNT and an order's overdue BADGE must never
    disagree about what day it is. Seeded due on the factory's today, then
    counted at 20:00 Eastern (UTC already on the next day) and again at 00:30
    Eastern the morning after."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    due_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                            ship_date=_SHIP_DAY)
    _add_line(db_cursor, due_id, product_id, 10)

    _freeze(monkeypatch, _EVENING_ET)
    evening = client.get("/sales/orders/counts").json()
    _freeze(monkeypatch, _AFTER_MIDNIGHT_ET)
    after_midnight = client.get("/sales/orders/counts").json()

    assert after_midnight["overdue"] - evening["overdue"] == 1, (
        evening, after_midnight)
    # Nothing else moved: only the calendar did.
    assert evening["open"] == after_midnight["open"]


@pytest.mark.db
def test_the_counts_bucket_and_the_order_badge_agree_at_8pm(db_cursor, client,
                                                            monkeypatch):
    """Stated as a pair, because a mismatch between them is the bug the factory
    clock exists to prevent: one order, one instant, two endpoints."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    due_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=True,
                            ship_date=_SHIP_DAY)
    line_id = _add_line(db_cursor, due_id, product_id, 10)
    _allocate(db_cursor, due_id, line_id, product_id, 10)

    _freeze(monkeypatch, _EVENING_ET)
    before = client.get("/sales/orders/counts").json()["overdue"]
    order = client.get(f"/sales/orders/{due_id}").json()
    assert order["health"]["reasons"] == [], order["health"]

    _freeze(monkeypatch, _AFTER_MIDNIGHT_ET)
    after = client.get("/sales/orders/counts").json()["overdue"]
    order = client.get(f"/sales/orders/{due_id}").json()
    assert after - before == 1
    assert order["health"]["reasons"] == ["1 day overdue — stock on hand"], \
        order["health"]


@pytest.mark.db
def test_the_list_overdue_field_and_filter_share_the_factory_clock(
        db_cursor, client, monkeypatch):
    """?overdue_only=true is a SQL filter and `overdue` is computed in Python;
    before v2.1 the filter ran on the server's CURRENT_DATE, so the two could
    disagree inside a single response."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    due_id, due_number = _seed_order(db_cursor, customer_id, token,
                                     ship_date=_SHIP_DAY)
    _add_line(db_cursor, due_id, product_id, 10)

    _freeze(monkeypatch, _EVENING_ET)
    listed = client.get("/sales/orders?limit=200").json()["orders"]
    mine = [o for o in listed if o["order_number"] == due_number]
    assert mine and mine[0]["overdue"] is False, mine
    filtered = client.get("/sales/orders?overdue_only=true&limit=200").json()
    assert due_number not in [o["order_number"] for o in filtered["orders"]]

    _freeze(monkeypatch, _AFTER_MIDNIGHT_ET)
    listed = client.get("/sales/orders?limit=200").json()["orders"]
    mine = [o for o in listed if o["order_number"] == due_number]
    assert mine and mine[0]["overdue"] is True, mine
    filtered = client.get("/sales/orders?overdue_only=true&limit=200").json()
    assert due_number in [o["order_number"] for o in filtered["orders"]]


@pytest.mark.db
def test_counts_route_is_not_swallowed_by_the_order_id_route(client):
    """'counts' must not be resolved as an order number."""
    resp = client.get("/sales/orders/counts")
    assert resp.status_code == 200, resp.text
    assert "open" in resp.json()


# ═════════════════════════════════════════════════════════════════
# Regression: existing status consumers are untouched
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_status_open_consumers_are_unchanged_by_the_mirror(db_cursor, client):
    """The whole point of the mirror: closing drops an order out of
    ?status=open exactly as the legacy cancel/ship paths always did."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, order_number = _seed_order(db_cursor, customer_id, token, status="confirmed")
    _add_line(db_cursor, order_id, product_id, 100)

    def open_numbers():
        return {o["order_number"] for o in
                client.get("/sales/orders",
                           params={"status": "open", "limit": 200}).json()["orders"]}

    assert order_number in open_numbers()

    client.post(f"/sales/orders/{order_id}/close",
                json={"reason": "short_closed", "mode": "commit"})
    assert order_number not in open_numbers(), "mirror should hide it from status=open"

    client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert order_number in open_numbers(), "reopen should bring it back"


@pytest.mark.db
def test_cancel_mirror_matches_the_legacy_cancelled_status(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, order_number = _seed_order(db_cursor, customer_id, token)
    client.post(f"/sales/orders/{order_id}/cancel",
                json={"reason": "customer_cancelled", "mode": "commit"})

    numbers = {o["order_number"] for o in
               client.get("/sales/orders",
                          params={"status": "cancelled", "limit": 200}).json()["orders"]}
    assert order_number in numbers


@pytest.mark.db
def test_open_state_never_touches_status(db_cursor, client):
    """While an order stays open, status is nobody's business but the
    operational paths'."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="in_production")
    before = _state_row(db_cursor, order_id)["status"]

    # a rejected cancel must not have written anything
    client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert _state_row(db_cursor, order_id)["status"] == before


@pytest.mark.db
def test_new_orders_default_to_open_with_confirmed_status(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    db_cursor.execute("SELECT name FROM customers WHERE id = %s", (customer_id,))
    customer_name = db_cursor.fetchone()["name"]
    product_id, _ = _seed_product(db_cursor, token)
    db_cursor.execute("SELECT name FROM products WHERE id = %s", (product_id,))
    product_name = db_cursor.fetchone()["name"]

    resp = client.post("/sales/orders", json={
        "customer_name": customer_name,
        "lines": [{"product_name": product_name, "quantity_lb": 50}],
    })
    assert resp.status_code == 200, resp.text
    order_id = resp.json()["order_id"]

    row = _state_row(db_cursor, order_id)
    assert row["status"] == "confirmed", "creation path is unchanged"
    assert row["state"] == "open", "and picks up the new default"
    assert row["state_reason"] is None


@pytest.mark.db
def test_patch_status_endpoint_still_works_untouched(db_cursor, client):
    """Migration 051 must not disturb the legacy transition endpoint."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="confirmed")
    resp = client.patch(f"/sales/orders/{order_id}/status",
                        json={"status": "in_production"})
    assert resp.status_code == 200, resp.text
    assert _state_row(db_cursor, order_id)["status"] == "in_production"


def test_gpt_schema_has_no_state_operations():
    """Dashboard-only by design.

    The assertion is `<= 30`, not `== 30`: 30 is a CEILING from CLAUDE.md, not
    a required count. Pinning it to exactly 30 would turn retiring an operation
    — a perfectly good thing to do — into a test failure.
    """
    yaml = __import__("yaml")
    spec = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "openapi-gpt-v3.yaml").read_text())
    paths = spec["paths"]
    for path in ("/sales/orders/counts",
                 "/sales/orders/{order_id}/close",
                 "/sales/orders/{order_id}/cancel",
                 "/sales/orders/{order_id}/reopen"):
        assert path not in paths, f"{path} must not be exposed to the GPT"
    ops = sum(1 for methods in paths.values() for m in methods
              if m in ("get", "post", "patch", "put", "delete"))
    assert ops <= 30, f"openapi-gpt-v3.yaml operation ceiling breached: {ops}"


def test_new_endpoints_are_on_the_dashboard_allowlist():
    for entry in (("GET", "/sales/orders/counts"),
                  ("POST", "/sales/orders/{order_id}/close"),
                  ("POST", "/sales/orders/{order_id}/cancel"),
                  ("POST", "/sales/orders/{order_id}/reopen")):
        assert entry in main.DASHBOARD_KEY_ALLOWLIST, entry


# ═════════════════════════════════════════════════════════════════
# Migration hardening: terminal states, marker idempotency
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
@pytest.mark.parametrize("state", ["closed", "cancelled"])
def test_terminal_state_with_null_reason_is_rejected(db_cursor, state):
    """A terminal state with no reason is the one thing the column exists to
    prevent. NULL IN (...) is NULL, and a CHECK admits NULL — only an explicit
    IS NOT NULL makes the constraint bite."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    db_cursor.execute("SAVEPOINT null_reason")
    with pytest.raises(Exception):
        db_cursor.execute(
            "UPDATE sales_orders SET state = %s, state_reason = NULL WHERE id = %s",
            (state, order_id),
        )
    db_cursor.execute("ROLLBACK TO SAVEPOINT null_reason")


@pytest.mark.db
def test_backfill_marker_is_written_and_is_the_gate(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    _seed_order(db_cursor, customer_id, token, status="confirmed")
    db_cursor.execute("UPDATE sales_orders SET state_changed_at=NULL, state_changed_by=NULL")
    _run_backfill(db_cursor)
    db_cursor.execute("SELECT applied_at FROM migration_markers WHERE name = %s",
                      (BACKFILL_MARKER,))
    assert db_cursor.fetchone() is not None


# ─── migration rerun, in a genuinely isolated schema ─────────────
#
# These three make claims about a whole database — "empty", "EVERY backfilled
# row has since changed" — that the shared test DB cannot honour: it holds rows
# from the schema load and from whatever else has run. Asserting them there
# would be calling something a precondition without ever establishing it.
#
# Each therefore runs in a throwaway schema, on its OWN connection, where the
# precondition is created and then asserted before the rerun happens.

ISOLATED_SCHEMA_DDL = """
CREATE TABLE sales_orders (
    id                 serial PRIMARY KEY,
    order_number       text NOT NULL,
    status             text NOT NULL,
    state              text NOT NULL DEFAULT 'open',
    state_reason       text,
    state_note         text,
    state_changed_at   timestamptz,
    state_changed_by   text,
    related_so_id      integer REFERENCES sales_orders(id),
    status_before_exit text
);
CREATE TABLE products (
    id         serial PRIMARY KEY,
    is_service boolean NOT NULL DEFAULT false
);
CREATE TABLE sales_order_lines (
    id             serial PRIMARY KEY,
    sales_order_id integer NOT NULL REFERENCES sales_orders(id),
    product_id     integer NOT NULL REFERENCES products(id),
    quantity_lb    numeric(14,4) NOT NULL,
    line_status    text NOT NULL DEFAULT 'pending'
);
CREATE TABLE sales_order_shipments (
    sales_order_line_id integer NOT NULL,
    transaction_id      integer NOT NULL,
    quantity_lb         numeric(14,4) NOT NULL
);
CREATE TABLE ledger_current_transactions (
    id integer PRIMARY KEY, type text, effective_status text
);
CREATE TABLE ledger_current_transaction_lines (
    transaction_id integer, product_id integer, quantity_lb numeric(14,4)
);
CREATE TABLE migration_markers (
    name text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
"""


@contextmanager
def _isolated_schema():
    """An empty schema on its OWN connection.

    The separate connection matters as much as the separate schema: the suite's
    db_cursor runs inside one long-lived, never-committed transaction shared by
    every test, so DDL issued through it holds locks for the rest of the
    session and can wedge a later TestClient's startup migrations. An
    autocommit connection of its own cannot do that.
    """
    import os
    import psycopg2 as pg

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    name = f"so_state_iso_{uuid4().hex[:8]}"
    conn = pg.connect(url, application_name="sostate-isolated-migration")
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"CREATE SCHEMA {name}")
            try:
                cur.execute(f"SET search_path TO {name}")
                cur.execute(ISOLATED_SCHEMA_DDL)
                yield cur
            finally:
                cur.execute("SET search_path TO public")
                cur.execute(f"DROP SCHEMA {name} CASCADE")
    finally:
        conn.close()


def _isolated_backfill_block():
    """The migration's backfill with its public. prefixes stripped so it
    resolves through search_path. The LOGIC is byte-identical."""
    return _backfill_block().replace("public.", "")


@pytest.mark.db
def test_rerun_is_a_noop_on_a_genuinely_empty_database():
    """Zero rows — asserted, not assumed. With nothing to derive a guard from,
    the marker is the only thing that can carry it."""
    with _isolated_schema() as cur:
        cur.execute("SELECT count(*) AS n FROM sales_orders")
        assert cur.fetchone()["n"] == 0, "precondition: the schema really is empty"
        cur.execute("SELECT count(*) AS n FROM migration_markers")
        assert cur.fetchone()["n"] == 0, "precondition: no marker yet"

        cur.execute(_isolated_backfill_block())
        cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name = %s",
                    (BACKFILL_MARKER,))
        assert cur.fetchone()["n"] == 1, "the first run writes the marker"

        cur.execute(_isolated_backfill_block())
        cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name = %s",
                    (BACKFILL_MARKER,))
        assert cur.fetchone()["n"] == 1, "the rerun is a no-op; no duplicate marker"


@pytest.mark.db
def test_rerun_ignores_orders_created_after_the_migration():
    """A row-derived guard would re-stamp these with a false
    'backfilled from legacy status=…' note."""
    with _isolated_schema() as cur:
        cur.execute("INSERT INTO sales_orders (order_number, status) "
                    "VALUES ('OLD', 'confirmed') RETURNING id")
        old_id = cur.fetchone()["id"]
        cur.execute(_isolated_backfill_block())
        cur.execute("SELECT state_changed_by FROM sales_orders WHERE id = %s", (old_id,))
        assert cur.fetchone()["state_changed_by"] == "migration-051"

        cur.execute("INSERT INTO sales_orders (order_number, status) "
                    "VALUES ('NEW', 'confirmed') RETURNING id")
        new_id = cur.fetchone()["id"]
        cur.execute(_isolated_backfill_block())

        cur.execute("SELECT state, state_note, state_changed_by "
                    "  FROM sales_orders WHERE id = %s", (new_id,))
        row = cur.fetchone()
        assert row["state"] == "open"
        assert row["state_note"] is None, "a post-migration order was never backfilled"
        assert row["state_changed_by"] is None


@pytest.mark.db
def test_rerun_is_a_noop_after_every_backfilled_row_changed_state():
    """The case a row-derived guard cannot see: the migration did its work and
    then every row it touched legitimately moved on.

    Every backfilled row is actually moved here, and the absence of the
    migration's own stamp is asserted before the rerun — so the precondition in
    the name is established, not hoped for.
    """
    with _isolated_schema() as cur:
        for n, status in enumerate(
                ["new", "confirmed", "in_production", "ready", "partial_ship",
                 "shipped", "invoiced", "cancelled"]):
            cur.execute("INSERT INTO sales_orders (order_number, status) VALUES (%s, %s)",
                        (f"SO-{n}", status))

        cur.execute(_isolated_backfill_block())
        cur.execute("SELECT id, state FROM sales_orders "
                    " WHERE state_changed_by = 'migration-051' ORDER BY id")
        backfilled = [(r["id"], r["state"]) for r in cur.fetchall()]
        assert len(backfilled) == 8, "every seeded row was backfilled"

        for order_id, state in backfilled:
            target = (("cancelled", "other") if state != "cancelled"
                      else ("closed", "short_closed"))
            cur.execute(
                "UPDATE sales_orders SET state = %s, state_reason = %s, "
                "       state_note = 'moved on', state_changed_by = 'a-human' "
                " WHERE id = %s",
                (target[0], target[1], order_id))
        cur.execute("SELECT count(*) AS n FROM sales_orders "
                    " WHERE state_changed_by = 'migration-051'")
        assert cur.fetchone()["n"] == 0, (
            "precondition: not one row still carries the migration's own stamp"
        )

        cur.execute("SELECT id, state, state_reason, state_note, state_changed_by "
                    "  FROM sales_orders ORDER BY id")
        before = [dict(r) for r in cur.fetchall()]

        cur.execute(_isolated_backfill_block())

        cur.execute("SELECT id, state, state_reason, state_note, state_changed_by "
                    "  FROM sales_orders ORDER BY id")
        assert [dict(r) for r in cur.fetchall()] == before, (
            "the rerun must not re-backfill rows that have since changed state"
        )


# ═════════════════════════════════════════════════════════════════
# State is authoritative for advancing writes
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
@pytest.mark.parametrize("exit_kind", ["close", "cancel"])
def test_shipping_a_closed_or_cancelled_order_is_rejected(db_cursor, client, exit_kind):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="ready")
    _add_line(db_cursor, order_id, product_id, 100)

    body = ({"reason": "short_closed", "mode": "commit"} if exit_kind == "close"
            else {"reason": "customer_cancelled", "mode": "commit"})
    assert client.post(f"/sales/orders/{order_id}/{exit_kind}", json=body).status_code == 200

    resp = client.post(f"/sales/orders/{order_id}/ship/commit", json={"ship_all": True})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ORDER_NOT_OPEN"
    assert detail["state"] == ("closed" if exit_kind == "close" else "cancelled")
    assert "reopen" in detail["message"]
    assert detail["suggested_action"] == "reopen"


@pytest.mark.db
def test_ship_preview_refuses_a_closed_order_too(db_cursor, client):
    """Preview exists to answer 'can I ship this'; answering yes to something
    the commit will refuse is worse than refusing early."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="ready")
    _add_line(db_cursor, order_id, product_id, 100)
    client.post(f"/sales/orders/{order_id}/close",
                json={"reason": "short_closed", "mode": "commit"})

    resp = client.post(f"/sales/orders/{order_id}/ship/preview", json={"ship_all": True})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "ORDER_NOT_OPEN"


@pytest.mark.db
@pytest.mark.parametrize("exit_kind,expected_state", [
    ("close", "closed"), ("cancel", "cancelled"),
])
def test_adding_a_line_to_a_closed_or_cancelled_order_is_rejected(
        db_cursor, client, exit_kind, expected_state):
    """Adding a line advances the order, so state gates it like every other
    SO write."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    db_cursor.execute("SELECT name FROM products WHERE id = %s", (product_id,))
    product_name = db_cursor.fetchone()["name"]
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_id, product_id, 100)

    body = ({"reason": "short_closed", "mode": "commit"} if exit_kind == "close"
            else {"reason": "customer_cancelled", "mode": "commit"})
    assert client.post(f"/sales/orders/{order_id}/{exit_kind}",
                       json=body).status_code == 200

    resp = client.post(f"/sales/orders/{order_id}/lines",
                       json={"lines": [{"product_name": product_name,
                                        "quantity_lb": 25}]})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ORDER_NOT_OPEN"
    assert detail["state"] == expected_state
    assert detail["suggested_action"] == "reopen"
    assert "reopen" in detail["message"]

    db_cursor.execute("SELECT count(*) AS n FROM sales_order_lines "
                      " WHERE sales_order_id = %s", (order_id,))
    assert db_cursor.fetchone()["n"] == 1, "nothing was inserted"


@pytest.mark.db
def test_adding_a_line_works_again_after_reopen(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    db_cursor.execute("SELECT name FROM products WHERE id = %s", (product_id,))
    product_name = db_cursor.fetchone()["name"]
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_id, product_id, 100)
    client.post(f"/sales/orders/{order_id}/close",
                json={"reason": "short_closed", "mode": "commit"})
    client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})

    resp = client.post(f"/sales/orders/{order_id}/lines",
                       json={"lines": [{"product_name": product_name,
                                        "quantity_lb": 25}]})
    assert resp.status_code == 200, resp.text


@pytest.mark.db
def test_allocation_after_close_is_rejected(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)

    assert client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit"}).status_code == 200

    resp = client.post(f"/sales/orders/{order_id}/allocations",
                       json={"mode": "manual", "line_id": line_id, "quantity_lb": 50})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ORDER_NOT_OPEN"
    assert detail["state"] == "closed"


@pytest.mark.db
def test_allocation_after_reopen_works_again(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    client.post(f"/sales/orders/{order_id}/close",
                json={"reason": "short_closed", "mode": "commit"})
    client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})

    resp = client.post(f"/sales/orders/{order_id}/allocations",
                       json={"mode": "manual", "line_id": line_id, "quantity_lb": 50})
    assert resp.status_code == 200, resp.text


@pytest.mark.db
def test_double_close_is_rejected(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    first = client.post(f"/sales/orders/{order_id}/close",
                        json={"reason": "short_closed", "mode": "commit"})
    assert first.status_code == 200, first.text
    second = client.post(f"/sales/orders/{order_id}/close",
                         json={"reason": "short_closed", "mode": "commit"})
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["error_code"] == "ORDER_NOT_OPEN"
    # and the first close's record is intact
    row = _state_row(db_cursor, order_id)
    assert row["state_reason"] == "short_closed"


# ─── reservation scoping ─────────────────────────────────────────

@pytest.mark.db
def test_exit_leaves_other_orders_reservations_alone(db_cursor, client):
    """Closing one order must not release another customer's stock — not even
    as a side effect of expiring stale auto-FIFO rows on the shared product."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)

    mine, _ = _seed_order(db_cursor, customer_id, token)
    my_line = _add_line(db_cursor, mine, product_id, 100)
    my_alloc = _allocate(db_cursor, mine, my_line, product_id, 100, lot_id=lot_id)

    theirs, _ = _seed_order(db_cursor, customer_id, token)
    their_line = _add_line(db_cursor, theirs, product_id, 100)
    their_alloc = _allocate(db_cursor, theirs, their_line, product_id, 100, lot_id=lot_id)

    # a stale auto-FIFO row on the SAME product, belonging to the other order:
    # the legacy cancel path would expire this as a side effect. It needs its
    # own line — soa_active_lot_uniq forbids two active rows per (line, lot).
    their_stale_line = _add_line(db_cursor, theirs, product_id, 25)
    db_cursor.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, lot_id, quantity_lb, source, expires_at) "
        "VALUES (%s,%s,%s,%s,%s,'auto_fifo', NOW() - INTERVAL '1 hour') RETURNING id",
        (theirs, their_stale_line, product_id, lot_id, 25),
    )
    their_stale = db_cursor.fetchone()["id"]

    resp = client.post(f"/sales/orders/{mine}/close",
                       json={"reason": "short_closed", "mode": "commit"})
    assert resp.status_code == 200, resp.text
    assert [r["id"] for r in resp.json()["reservations_released"]] == [my_alloc], (
        "reservations_released must list exactly the rows this exit changed"
    )

    db_cursor.execute(
        "SELECT id, status FROM sales_order_allocations WHERE id = ANY(%s) ORDER BY id",
        ([their_alloc, their_stale],))
    assert [r["status"] for r in db_cursor.fetchall()] == ["active", "active"], (
        "the other order's rows — including its stale auto-FIFO row — are untouched"
    )


@pytest.mark.db
def test_exit_releases_this_orders_expired_row_too(db_cursor, client):
    """An expired auto-FIFO row is still status='active' until something
    releases it, and it belongs to this order — so the exit takes it, and
    preview must have said so."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    db_cursor.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, lot_id, quantity_lb, source, expires_at) "
        "VALUES (%s,%s,%s,%s,50,'auto_fifo', NOW() - INTERVAL '1 hour') RETURNING id",
        (order_id, line_id, product_id, lot_id),
    )
    expired = db_cursor.fetchone()["id"]

    preview = client.post(f"/sales/orders/{order_id}/close",
                          json={"reason": "short_closed", "mode": "preview"})
    assert expired in [r["id"] for r in preview.json()["reservations_to_release"]]

    commit = client.post(f"/sales/orders/{order_id}/close",
                         json={"reason": "short_closed", "mode": "commit"})
    assert [r["id"] for r in commit.json()["reservations_released"]] == [expired]


# ─── attribution ─────────────────────────────────────────────────

@pytest.mark.db
def test_changed_by_is_stripped_but_never_truncated(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    at_limit = "j" * 120
    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit",
                             "changed_by": f"  {at_limit}  "})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state_changed_by"] == at_limit
    assert _state_row(db_cursor, order_id)["state_changed_by"] == at_limit


@pytest.mark.db
def test_changed_by_over_the_limit_is_rejected_not_truncated(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "commit",
                             "changed_by": "j" * 121})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "CHANGED_BY_TOO_LONG"
    assert _state_row(db_cursor, order_id)["state"] == "open", "nothing written"


@pytest.mark.db
def test_changed_by_is_validated_in_preview_too(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "preview",
                             "changed_by": "j" * 121})
    assert resp.status_code == 400, resp.text


@pytest.mark.db
def test_body_changed_by_never_becomes_released_by(db_cursor, client, monkeypatch):
    """changed_by is a claim about who asked for the exit. released_by is a
    fact about which surface released stock. They are not the same field."""
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "sostate-dashboard-key")
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    alloc_id = _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "mode": "commit",
                             "changed_by": "impersonated-surface"},
                       headers={"X-API-Key": "sostate-dashboard-key"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state_changed_by"] == "impersonated-surface"

    db_cursor.execute("SELECT released_by FROM sales_order_allocations WHERE id = %s",
                      (alloc_id,))
    assert db_cursor.fetchone()["released_by"] == "dashboard", (
        "released_by comes from the authenticated surface, never the body"
    )


# ─── related_so_id ───────────────────────────────────────────────

@pytest.mark.db
def test_related_so_id_must_exist(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "duplicate", "related_so_id": 99999999,
                             "mode": "commit"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "RELATED_SO_NOT_FOUND"


@pytest.mark.db
def test_related_so_id_may_not_be_the_order_itself(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "duplicate", "related_so_id": order_id,
                             "mode": "commit"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "RELATED_SO_INVALID"


@pytest.mark.db
def test_related_so_id_is_refused_for_a_reason_that_cannot_use_it(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    other_id, _ = _seed_order(db_cursor, customer_id, token)
    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "related_so_id": other_id,
                             "mode": "commit"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "RELATED_SO_NOT_APPLICABLE"


@pytest.mark.db
@pytest.mark.parametrize("bad,code", [
    (99999999, "RELATED_SO_NOT_FOUND"),
    ("self", "RELATED_SO_INVALID"),
])
def test_related_so_id_validation_is_identical_in_preview(db_cursor, client, bad, code):
    """A preview that reports success for a body the commit rejects is a lie."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    related = order_id if bad == "self" else bad
    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "duplicate", "related_so_id": related,
                             "mode": "preview"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == code


# ─── preview coverage ────────────────────────────────────────────

@pytest.mark.db
def test_cancel_preview_reports_state_and_reservations_without_writing(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="confirmed")
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    alloc_id = _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    resp = client.post(f"/sales/orders/{order_id}/cancel",
                       json={"reason": "customer_cancelled", "mode": "preview"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "preview"
    assert body["resulting_state"] == "cancelled"
    assert body["resulting_status"] == "cancelled"
    assert [r["id"] for r in body["reservations_to_release"]] == [alloc_id]
    assert body["attribution_note"]

    assert _state_row(db_cursor, order_id)["state"] == "open"
    db_cursor.execute("SELECT status FROM sales_order_allocations WHERE id=%s", (alloc_id,))
    assert db_cursor.fetchone()["status"] == "active"


@pytest.mark.db
def test_reopen_preview_reports_the_restored_status_without_writing(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="in_production")
    client.post(f"/sales/orders/{order_id}/close",
                json={"reason": "short_closed", "mode": "commit"})

    resp = client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "preview"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["resulting_state"] == "open"
    assert body["resulting_status"] == "in_production"
    assert body["reservations_to_release"] == []
    assert body["attribution_note"]
    assert _state_row(db_cursor, order_id)["state"] == "closed", "preview must not write"


@pytest.mark.db
def test_preview_carries_the_attribution_note(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    body = client.post(f"/sales/orders/{order_id}/close",
                       json={"reason": "short_closed", "mode": "preview",
                             "changed_by": "office-jo"}).json()
    assert body["attribution_note"] == main.SO_STATE_ATTRIBUTION_NOTE
    assert body["resulting_state_changed_by"] == "office-jo"


# ═════════════════════════════════════════════════════════════════
# Health: partial allocation
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_health_info_reports_partially_allocated_pounds(db_cursor, client, monkeypatch):
    """100 remaining with 40 allocated is 60 lb unallocated. Reporting nothing
    for it hid exactly the partially-covered lines worth seeing."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 40, lot_id=lot_id)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "quiet", "info never escalates"
    assert any("60 lb not allocated" in i for i in health["info"]), health["info"]
    detail = health["info_detail"]
    assert len(detail) == 1, detail
    assert detail[0]["line_id"] == line_id
    assert detail[0]["unallocated_lb"] == 60.0
    assert detail[0]["sku"] and detail[0]["product_name"], detail
    # The string names the line the same way, name-then-code.
    assert (f"{detail[0]['product_name']} ({detail[0]['sku']})"
            in health["info"][0]), health["info"]


@pytest.mark.db
def test_health_info_is_silent_when_fully_allocated(db_cursor, client, monkeypatch):
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=1000)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    assert client.get(f"/sales/orders/{order_id}").json()["health"]["info"] == []


# ═════════════════════════════════════════════════════════════════
# Legacy PATCH /status — the legacy-cancellation policy
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_legacy_patch_cancelled_routes_through_the_cancel_logic(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="confirmed")
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    alloc_id = _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    resp = client.patch(f"/sales/orders/{order_id}/status", json={"status": "cancelled"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "cancelled", "legacy response shape is unchanged"
    assert body["previous_status"] == "confirmed"
    assert body["state"] == "cancelled"
    assert body["state_reason"] == "other"
    assert body["state_note"] == "via legacy status endpoint"

    row = _state_row(db_cursor, order_id)
    assert (row["status"], row["state"], row["state_reason"]) == ("cancelled", "cancelled", "other")
    assert row["status_before_exit"] == "confirmed"

    db_cursor.execute("SELECT status, release_reason FROM sales_order_allocations WHERE id=%s",
                      (alloc_id,))
    alloc = db_cursor.fetchone()
    assert alloc["status"] == "released"
    assert alloc["release_reason"] == "order_cancelled"


@pytest.mark.db
def test_legacy_patch_cancelled_is_refused_when_partially_shipped(db_cursor, client):
    """The guard has to live here too, or the legacy endpoint becomes the way
    round it."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="partial_ship")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=40, status="partial")
    _post_ship(db_cursor, line_id, product_id, lot_id, 40, recorded=40)

    resp = client.patch(f"/sales/orders/{order_id}/status", json={"status": "cancelled"})
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ORDER_ALREADY_SHIPPED"
    assert detail["suggested_reason"] == "short_closed"

    row = _state_row(db_cursor, order_id)
    assert row["state"] == "open" and row["status"] == "partial_ship", "nothing written"


@pytest.mark.db
def test_legacy_patch_invoiced_routes_through_close_and_mirrors_invoiced(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _post_ship(db_cursor, line_id, product_id, lot_id, 100, recorded=100)

    resp = client.patch(f"/sales/orders/{order_id}/status", json={"status": "invoiced"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "invoiced", "mirrored to 'invoiced', NOT close's usual 'shipped'"
    assert body["state"] == "closed"
    assert body["state_reason"] == "shipped_recorded"
    assert body["state_note"] == "via legacy status endpoint (invoiced)"

    row = _state_row(db_cursor, order_id)
    assert (row["status"], row["state"]) == ("invoiced", "closed")
    assert row["status_before_exit"] == "shipped"


@pytest.mark.db
def test_legacy_patch_invoiced_picks_not_recorded_when_pounds_remain(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _post_ship(db_cursor, line_id, product_id, lot_id, 40, recorded=100)

    resp = client.patch(f"/sales/orders/{order_id}/status", json={"status": "invoiced"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state_reason"] == "shipped_not_recorded"


@pytest.mark.db
@pytest.mark.parametrize("start,target", [
    ("new", "confirmed"),
    ("confirmed", "in_production"),
    ("in_production", "ready"),
    ("ready", "in_production"),
])
def test_legacy_patch_operational_values_never_touch_state(db_cursor, client, start, target):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status=start)
    before = _state_row(db_cursor, order_id)

    resp = client.patch(f"/sales/orders/{order_id}/status", json={"status": target})
    assert resp.status_code == 200, resp.text

    after = _state_row(db_cursor, order_id)
    assert after["status"] == target
    assert after["state"] == before["state"] == "open"
    assert after["state_reason"] is None
    assert after["state_changed_at"] == before["state_changed_at"]
    assert after["status_before_exit"] is None
    assert "state" not in resp.json(), "operational transitions report no state change"


@pytest.mark.db
def test_legacy_patch_is_refused_on_an_order_that_left_the_board(db_cursor, client):
    """Belt and braces: the mirror already makes the transition table refuse
    these, but the state gate says so in the language of the new model."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped",
                              state="closed", reason="shipped_recorded")
    resp = client.patch(f"/sales/orders/{order_id}/status", json={"status": "invoiced"})
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "ORDER_NOT_OPEN"


# ═════════════════════════════════════════════════════════════════
# Startup reconciliation
# ═════════════════════════════════════════════════════════════════

RECONCILE_SQL = """
    UPDATE sales_orders
       SET state = 'cancelled',
           state_reason = 'other',
           state_note = 'reconciled from legacy status at startup',
           state_changed_by = 'startup-reconcile',
           state_changed_at = clock_timestamp()
     WHERE status = 'cancelled'
       AND state = 'open'
"""


@pytest.mark.db
def test_startup_reconcile_moves_legacy_cancelled_rows(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="cancelled")
    assert _state_row(db_cursor, order_id)["state"] == "open"

    db_cursor.execute(RECONCILE_SQL)
    row = _state_row(db_cursor, order_id)
    assert row["state"] == "cancelled"
    assert row["state_reason"] == "other"
    assert row["state_note"] == "reconciled from legacy status at startup"
    assert row["state_changed_by"] == "startup-reconcile"


@pytest.mark.db
def test_startup_reconcile_is_idempotent(db_cursor):
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="cancelled")
    db_cursor.execute(RECONCILE_SQL)
    first = _state_row(db_cursor, order_id)
    db_cursor.execute(RECONCILE_SQL)
    assert db_cursor.rowcount == 0, "second sweep matches nothing"
    assert _state_row(db_cursor, order_id) == first


@pytest.mark.db
@pytest.mark.parametrize("status", ["shipped", "invoiced"])
def test_startup_reconcile_leaves_open_shipped_orders_alone(db_cursor, status):
    """'Open · Shipped' is legitimate: everything physically went out, nobody
    has administratively closed it yet. That distinction is the whole point of
    the state model — auto-closing these would erase it."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status=status)
    before = _state_row(db_cursor, order_id)

    db_cursor.execute(RECONCILE_SQL)
    assert _state_row(db_cursor, order_id) == before
    assert _state_row(db_cursor, order_id)["state"] == "open"


@pytest.mark.db
def test_startup_reconcile_does_not_disturb_a_deliberate_reopen(db_cursor, client):
    """An order reopened on purpose has its status restored from
    status_before_exit, so it no longer reads status='cancelled' and the sweep
    cannot claw it back."""
    customer_id, token = _seed_customer(db_cursor)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="confirmed")
    client.post(f"/sales/orders/{order_id}/cancel",
                json={"reason": "customer_cancelled", "mode": "commit"})
    client.post(f"/sales/orders/{order_id}/reopen", json={"mode": "commit"})
    assert _state_row(db_cursor, order_id)["status"] == "confirmed"

    db_cursor.execute(RECONCILE_SQL)
    row = _state_row(db_cursor, order_id)
    assert row["state"] == "open", "a deliberate reopen survives the sweep"


def test_startup_reconcile_block_exists_and_spares_shipped_rows():
    """Guard the sweep's WHERE clause itself — the dangerous edit is widening
    it to cover shipped/invoiced."""
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text()
    start = source.index("# Migration 051-reconcile:")
    end = source.index("# Migration 008:", start)
    block = source[start:end]
    assert "state_changed_by = 'startup-reconcile'" in block
    assert "WHERE status = 'cancelled'" in block
    assert "AND state = 'open'" in block
    assert "shipped" not in block.split('"""')[1] if '"""' in block else True


# ═════════════════════════════════════════════════════════════════
# Concurrency harness
#
# Every concurrency test here works the same way:
#
#   1. Pre-create each endpoint's DB connection, with a UUID application_name
#      of its own, and read its exact backend PID with pg_backend_pid() BEFORE
#      the endpoint runs. Every later pg_locks / pg_blocking_pids assertion
#      names that PID. Nothing matches on application_name — two runs, or a
#      stray connection, must never be able to satisfy an assertion by
#      accident; the name exists only to make a stuck backend identifiable by
#      a human reading pg_stat_activity.
#
#   2. Park a HOLDER connection on the exact resource the writers contend on,
#      positioned at the step AFTER their first lock. That is what forces the
#      interleaving: each transaction gets far enough to take its first-step
#      lock and then queues, so they are genuinely overlapping and holding
#      partial lock sets — the state in which a disagreeing lock order
#      deadlocks. Without the holder, one transaction usually just finishes
#      before the other starts and the test proves nothing.
#
#   3. Assert through pg_blocking_pids() that EVERY participating backend is
#      blocked on the holder before the holder lets go.
#
#   4. Release, join, and only then judge.
#
# NOTE ON WHAT THESE PROVE. The two-writer tests call the endpoint FUNCTIONS
# directly rather than issuing HTTP requests, so they are transaction-body
# concurrency tests: they exercise the real lock sequence of each endpoint's
# transaction, not the full request path (no routing, no auth dependency, no
# response envelope). That is deliberate — one TestClient serialises both
# threads through a single ASGI portal, and two TestClients race to close the
# same module-level db_pool — but it does mean they are not request-identical.
#
# They use real committed rows, not the rolled-back fixture transaction, and
# clean up after themselves. Cleanup failures are NOT swallowed: a wedge must
# surface as a failure here rather than as a hang in whatever runs next.
# ═════════════════════════════════════════════════════════════════


class _EndpointConn:
    """A pre-created endpoint connection whose backend PID is known up front."""

    def __init__(self, url, label):
        import psycopg2 as pg

        self.app_name = f"sostate-{label}-{uuid4().hex[:12]}"
        self.conn = pg.connect(url, application_name=self.app_name)
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            self.pid = cur.fetchone()[0]

    def close(self):
        """Propagates. A connection that will not close is a symptom — a
        transaction still parked, a backend still holding locks — and
        swallowing it here would hand the next test a wedged database with no
        indication of where it came from."""
        self.conn.close()


def _provider_for(mapping):
    """monkeypatch target for main.get_db_connection.

    Hands each thread the connection pre-created for it, so the PID captured
    before the run is genuinely the one the endpoint uses.
    """
    import threading
    from contextlib import contextmanager as _ctx

    @_ctx
    def _provider():
        econn = mapping[threading.current_thread().name]
        try:
            yield econn.conn
            econn.conn.commit()
        except Exception:
            econn.conn.rollback()
            raise

    return _provider


def _lock_graph(watch_cur, pids):
    """{pid: (waiting_query, [direct blockers])} for the pids given."""
    watch_cur.execute(
        """SELECT pid, query, pg_blocking_pids(pid) AS blockers
             FROM pg_stat_activity
            WHERE pid = ANY(%s)""",
        (list(pids),),
    )
    return {r["pid"]: (r["query"], list(r["blockers"] or []))
            for r in watch_cur.fetchall()}


def _reaches_holder(pid, holder_pid, graph, _seen=None):
    """Is `pid` parked behind the holder, directly or through another writer?

    Postgres reports only the DIRECT blocker. When two writers queue on the
    same rows the first names the holder and the second names the first — so a
    check that demanded every writer name the holder would fail on a genuine
    two-deep queue. Both are nonetheless stopped behind the holder while
    holding their own step-1 locks, which is exactly the interleaving these
    tests exist to force, so the check follows the chain.
    """
    _seen = _seen or set()
    if pid in _seen:
        return False
    _seen.add(pid)
    for blocker in graph.get(pid, (None, []))[1]:
        if blocker == holder_pid or _reaches_holder(blocker, holder_pid, graph, _seen):
            return True
    return False


def _blocked_on(watch_cur, pid, holder_pid):
    """The statement `pid` is stuck on, if the holder is what is blocking it."""
    graph = _lock_graph(watch_cur, [pid])
    if pid not in graph:
        return None
    query, _blockers = graph[pid]
    return query if _reaches_holder(pid, holder_pid, graph) else None


def _wait_until_all_blocked(watch_cur, pids, holder_pid, deadline_s=25):
    """Wait for EVERY pid to be parked behind the holder; return {pid: query}.

    Returns whatever it managed to observe if the deadline passes, so the
    caller can report which backend never queued.
    """
    import time

    pids = list(pids)
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        # One snapshot per poll: the whole chain has to be read consistently,
        # or a writer can look unblocked simply because its blocker was read
        # a moment earlier.
        graph = _lock_graph(watch_cur, pids + [holder_pid])
        seen = {pid: graph[pid][0] for pid in pids
                if pid in graph and _reaches_holder(pid, holder_pid, graph)}
        if len(seen) == len(pids):
            return seen
        time.sleep(0.05)
    graph = _lock_graph(watch_cur, pids + [holder_pid])
    return {pid: graph[pid][0] for pid in pids
            if pid in graph and _reaches_holder(pid, holder_pid, graph)}


def _run_concurrently(named_calls, timeout=60):
    """Start every call on its own named thread at a barrier.

    Thread names are the keys, which is how _provider_for hands each one its
    own pre-created connection.
    """
    import threading

    start = threading.Barrier(len(named_calls))
    out = {}

    def _wrap(name, fn):
        def _run():
            start.wait(timeout=25)
            try:
                out[name] = fn()
            except Exception as exc:            # noqa: BLE001 - recorded, asserted on
                out[name] = exc
        return _run

    threads = [threading.Thread(target=_wrap(name, fn), name=name)
               for name, fn in named_calls.items()]
    for t in threads:
        t.start()
    return threads, out


def _join_all(threads, timeout=60):
    for t in threads:
        t.join(timeout=timeout)
    alive = [t.name for t in threads if t.is_alive()]
    assert not alive, f"these writers never finished (deadlock or hang): {alive}"


def _assert_no_deadlock(result, label):
    """40P01 is Postgres's deadlock_detected.

    Three shapes have to be recognised, because a deadlock does not always
    reach the caller as an exception: these endpoints wrap their bodies in
    `except Exception -> JSONResponse(500)`, so a deadlocked transaction comes
    back as a 500 whose body carries the message. A check that only looked at
    exceptions would let that through as a plain "unexpected result" and hide
    what actually happened.

    Matching the SQLSTATE and Postgres's own phrase — never a bare "deadlock" —
    keeps fixture names out of the assertion.
    """
    import psycopg2
    from starlette.responses import JSONResponse

    def _is_deadlock(text):
        return "40P01" in text or "deadlock detected" in text.lower()

    if isinstance(result, JSONResponse):
        body = result.body.decode("utf-8", "replace")
        assert not _is_deadlock(body), f"{label} deadlocked: {body}"
        raise AssertionError(f"{label} returned {result.status_code}: {body[:300]}")
    if isinstance(result, psycopg2.Error):
        assert result.pgcode != "40P01", f"{label} deadlocked: {result}"
        raise AssertionError(f"{label} failed: {result!r}")
    if isinstance(result, Exception):
        assert not _is_deadlock(str(result)), f"{label} deadlocked: {result}"
        raise AssertionError(f"{label} failed: {result!r}")


class _StubRequest:
    """Just enough Request for caller_source_tag(): it reads one header."""

    def __init__(self, api_key=None):
        self.headers = {"X-API-Key": api_key} if api_key else {}


def _test_url():
    import os

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


# ─────────────────────────────────────────────────────────────────
# Committed fixtures
# ─────────────────────────────────────────────────────────────────

def _seed_committed(seed_cur, *, status, with_alloc=False, stock=1000):
    token = uuid4().hex[:10].upper()
    seed_cur.execute("INSERT INTO customers (name, active) VALUES (%s,true) RETURNING id",
                     (f"RACE Cust {token}",))
    customer_id = seed_cur.fetchone()["id"]
    seed_cur.execute(
        "INSERT INTO products (name,type,odoo_code,uom,is_service,active) "
        "VALUES (%s,'finished',%s,'lb',false,true) RETURNING id",
        (f"RACE FG {token}", f"RACE-{token}"))
    product_id = seed_cur.fetchone()["id"]
    seed_cur.execute(
        "INSERT INTO lots (product_id,lot_code,entry_source,received_at) "
        "VALUES (%s,%s,'received',NOW()) RETURNING id",
        (product_id, f"RACE-LOT-{token}"))
    lot_id = seed_cur.fetchone()["id"]
    seed_cur.execute("INSERT INTO transactions (type,timestamp) VALUES ('receive',NOW()) RETURNING id")
    txn = seed_cur.fetchone()["id"]
    seed_cur.execute(
        "INSERT INTO transaction_lines (transaction_id,product_id,lot_id,quantity_lb) "
        "VALUES (%s,%s,%s,%s)", (txn, product_id, lot_id, stock))
    order_number = f"RACE-SO-{token}"
    seed_cur.execute(
        "INSERT INTO sales_orders (customer_id,order_number,status) VALUES (%s,%s,%s) RETURNING id",
        (customer_id, order_number, status))
    order_id = seed_cur.fetchone()["id"]
    seed_cur.execute(
        "INSERT INTO sales_order_flags (so_number,ready,ready_at,ready_by) "
        "VALUES (%s,true,NOW(),'test')", (order_number,))
    seed_cur.execute(
        "INSERT INTO sales_order_lines (sales_order_id,product_id,quantity_lb,line_status) "
        "VALUES (%s,%s,100,'pending') RETURNING id", (order_id, product_id))
    line_id = seed_cur.fetchone()["id"]
    alloc_id = None
    if with_alloc:
        seed_cur.execute(
            "INSERT INTO sales_order_allocations "
            "(sales_order_id,sales_order_line_id,product_id,lot_id,quantity_lb,source) "
            "VALUES (%s,%s,%s,%s,100,'staged_lot') RETURNING id",
            (order_id, line_id, product_id, lot_id))
        alloc_id = seed_cur.fetchone()["id"]
    return {"customer_id": customer_id, "product_id": product_id, "lot_id": lot_id,
            "order_id": order_id, "order_number": order_number,
            "line_id": line_id, "alloc_id": alloc_id}


def _cleanup_order(seed_cur, order_id, order_number):
    """Remove one committed order. Raises if anything unexpected blocks it.

    Nothing is swallowed: a cleanup that quietly fails leaves rows behind that
    wedge a LATER test, which then looks like an unrelated hang. Failing here
    keeps the cause attached to the cause.

    The one branch that is expected rather than exceptional: trace_events is
    append-only and references sales_orders, so a shipped order genuinely
    cannot be deleted. Those are taken OFF THE BOARD instead — a leftover
    *open* order is not inert, because GET /sales/orders/counts walks every
    open order and touches products, and the client fixture's RELEASE SAVEPOINT
    promotes that lock into the suite's shared session transaction, where it
    blocks the next TestClient's startup ALTER on products.
    """
    seed_cur.execute(
        """DELETE FROM shipment_lines
            WHERE sales_order_line_id IN (SELECT id FROM sales_order_lines
                                           WHERE sales_order_id = %s)""",
        (order_id,))
    seed_cur.execute(
        """DELETE FROM sales_order_shipments
            WHERE sales_order_line_id IN (SELECT id FROM sales_order_lines
                                           WHERE sales_order_id = %s)""",
        (order_id,))
    seed_cur.execute("DELETE FROM shipments WHERE sales_order_id = %s", (order_id,))
    seed_cur.execute("DELETE FROM sales_order_allocations WHERE sales_order_id = %s",
                     (order_id,))
    seed_cur.execute("UPDATE sales_orders SET related_so_id = NULL WHERE related_so_id = %s",
                     (order_id,))
    seed_cur.execute("UPDATE sales_orders SET related_so_id = NULL WHERE id = %s",
                     (order_id,))

    seed_cur.execute("SELECT EXISTS (SELECT 1 FROM trace_events WHERE sales_order_id = %s) AS pinned",
                     (order_id,))
    if seed_cur.fetchone()["pinned"]:
        seed_cur.execute(
            "UPDATE sales_orders SET state = 'closed', state_reason = 'short_closed', "
            "       status = 'shipped', state_note = 'race fixture teardown' "
            " WHERE id = %s", (order_id,))
    else:
        seed_cur.execute("DELETE FROM sales_order_lines WHERE sales_order_id = %s", (order_id,))
        seed_cur.execute("DELETE FROM sales_order_flags WHERE so_number = %s", (order_number,))
        seed_cur.execute("DELETE FROM sales_orders WHERE id = %s", (order_id,))


def _cleanup_committed(seed_cur, ids):
    """Order side removed; the product is deactivated because append-only
    ledger rows still reference it; the customer goes once unreferenced."""
    _cleanup_order(seed_cur, ids["order_id"], ids["order_number"])
    seed_cur.execute("UPDATE products SET active = false WHERE id = %s", (ids["product_id"],))
    seed_cur.execute(
        "SELECT EXISTS (SELECT 1 FROM sales_orders WHERE customer_id = %s) AS used",
        (ids["customer_id"],))
    if not seed_cur.fetchone()["used"]:
        seed_cur.execute("DELETE FROM customers WHERE id = %s", (ids["customer_id"],))


class _Holder:
    """A connection parked on a resource, with its backend PID known."""

    def __init__(self, url, label):
        import psycopg2 as pg

        self.conn = pg.connect(url, application_name=f"sostate-holder-{label}")
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            self.pid = cur.fetchone()[0]

    def park_on_lots(self, product_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM lots WHERE product_id = %s ORDER BY id FOR UPDATE",
                        (product_id,))

    def park_on_orders_share(self, order_ids):
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM sales_orders WHERE id = ANY(%s) "
                        " ORDER BY id FOR SHARE", (list(order_ids),))

    def release(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


def _spawn(name, fn, out):
    """Start one writer on its own named thread. No barrier: these tests stage
    the writers in a fixed order, because 'both started at once' is not the
    same as 'both reached the step we need them at'."""
    import threading

    def _run():
        try:
            out[name] = fn()
        except Exception as exc:                # noqa: BLE001 - recorded, asserted on
            out[name] = exc

    t = threading.Thread(target=_run, name=name)
    t.start()
    return t


def _wait_until_reaches(watch_cur, pid, holder_pid, participants=(), deadline_s=25):
    """Wait until `pid` is parked behind `holder_pid`; return its waiting query.

    `participants` must list every other backend that could sit between the two
    — the chain is walked over the snapshot, so an edge through a writer that
    was not snapshotted is invisible and the wait times out on a queue that is
    in fact correct.
    """
    import time

    pids = list({pid, holder_pid, *participants})
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        graph = _lock_graph(watch_cur, pids)
        if pid in graph and _reaches_holder(pid, holder_pid, graph):
            return graph[pid][0]
        time.sleep(0.05)
    return None


def _wait_until_blocked_by(watch_cur, pid, blocker_pid, deadline_s=25):
    """Wait until `blocker_pid` is a DIRECT blocker of `pid`."""
    import time

    deadline = time.time() + deadline_s
    while time.time() < deadline:
        graph = _lock_graph(watch_cur, [pid])
        if pid in graph and blocker_pid in graph[pid][1]:
            return graph[pid][0]
        time.sleep(0.05)
    return None


def _chain_snapshot(watch_cur, pids):
    """{pid: (waiting query, direct blockers)} — for reporting a failed stage."""
    return _lock_graph(watch_cur, list(pids))

# ═════════════════════════════════════════════════════════════════
# One writer vs a parked holder: does the endpoint take the ORDER ROW lock?
# ═════════════════════════════════════════════════════════════════

class TestStateRaces:
    """Each test parks a holder on the sales_orders row, watches the endpoint
    queue on THAT row, then commits a conflicting state change and lets the
    endpoint through to re-read under its own lock."""

    @staticmethod
    def _assert_waiting_on_order_row_select(query, table="sales_orders"):
        """The blocked statement must be the initial locking read of the order
        row — not some later write that would block regardless.

        This is the assertion that bites when the row lock is removed: without
        it, an endpoint with no row lock still ends up blocked on the holder at
        its final UPDATE, and a naive "is it blocked?" check would pass.
        """
        assert query, "the endpoint never blocked on the holder connection"
        normalized = " ".join(query.lower().split())
        assert "for no key update" in normalized, (
            f"the endpoint is not blocked on a locking read: {query!r}"
        )
        assert normalized.startswith("select"), (
            f"the endpoint blocked on a write, not the initial order-row "
            f"SELECT — the row lock has been lost: {query!r}"
        )
        assert table in normalized, (
            f"the endpoint is blocked on the wrong relation: {query!r}"
        )

    def _run_race(self, monkeypatch, *, seed_kwargs, call, conflicting_sql):
        import psycopg2 as pg

        url = _test_url()
        seed = pg.connect(url)
        seed.autocommit = True
        ids = None
        endpoint = None
        holder = None
        try:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                ids = _seed_committed(sc, **seed_kwargs)

            endpoint = _EndpointConn(url, "race")
            monkeypatch.setattr(main, "get_db_connection",
                                _provider_for({"endpoint": endpoint}))

            holder = pg.connect(url, application_name="sostate-race-holder")
            with holder.cursor() as hc:
                hc.execute("SELECT pg_backend_pid()")
                holder_pid = hc.fetchone()[0]
                # Parked on the order row itself: the very first lock the
                # endpoint should take.
                hc.execute("SELECT id FROM sales_orders WHERE id = %s FOR NO KEY UPDATE",
                           (ids["order_id"],))

            threads, out = _run_concurrently({"endpoint": lambda: call(ids)})
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                blocked = _wait_until_all_blocked(sc, [endpoint.pid], holder_pid)
            blocked_query = blocked.get(endpoint.pid)

            # Release BEFORE asserting. The endpoint is mid-transaction on a
            # connection this test owns; raising here would leave it parked
            # while teardown tried to clean up the very rows it holds.
            with holder.cursor() as hc:
                hc.execute(conflicting_sql, (ids["order_id"],))
            holder.commit()
            _join_all(threads)

            self._assert_waiting_on_order_row_select(blocked_query)
            return out["endpoint"], ids, seed
        except Exception:
            if ids is not None:
                with seed.cursor(cursor_factory=RealDictCursor) as sc:
                    _cleanup_committed(sc, ids)
            seed.close()
            raise
        finally:
            if holder is not None:
                holder.close()
            if endpoint is not None:
                endpoint.close()

    CANCEL_SQL = ("UPDATE sales_orders SET state='cancelled', state_reason='other', "
                  "state_note='race', status='cancelled', status_before_exit='ready' "
                  "WHERE id = %s")
    CLOSE_SQL = ("UPDATE sales_orders SET state='closed', state_reason='short_closed', "
                 "state_note='race', status='shipped', status_before_exit='confirmed' "
                 "WHERE id = %s")

    @pytest.mark.db
    def test_shipment_loses_to_a_concurrent_cancel(self, monkeypatch):
        """A ship that began before the cancel committed must still refuse: it
        re-reads state under its own row lock after the cancel lands."""
        result, ids, seed = self._run_race(
            monkeypatch,
            seed_kwargs={"status": "ready", "with_alloc": True},
            call=lambda i: main.ship_order(
                _StubRequest(), i["order_id"],
                main.ShipOrderRequest(mode="commit", ship_all=True), True),
            conflicting_sql=self.CANCEL_SQL,
        )
        try:
            assert isinstance(result, HTTPException), result
            assert result.status_code == 409
            assert result.detail["error_code"] == "ORDER_NOT_OPEN"
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT count(*) AS n FROM shipments WHERE sales_order_id = %s",
                           (ids["order_id"],))
                assert sc.fetchone()["n"] == 0, "no shipment may exist for a cancelled order"
                sc.execute("SELECT quantity_shipped_lb FROM sales_order_lines WHERE id = %s",
                           (ids["line_id"],))
                assert float(sc.fetchone()["quantity_shipped_lb"]) == 0
        finally:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_allocation_loses_to_a_concurrent_close(self, monkeypatch):
        result, ids, seed = self._run_race(
            monkeypatch,
            seed_kwargs={"status": "confirmed"},
            call=lambda i: main.create_sales_order_allocation(
                main.SalesOrderAllocationCreate(
                    mode="manual", line_id=i["line_id"], quantity_lb=50),
                _StubRequest(), i["order_id"], True),
            conflicting_sql=self.CLOSE_SQL,
        )
        try:
            assert isinstance(result, HTTPException), result
            assert result.status_code == 409
            assert result.detail["error_code"] == "ORDER_NOT_OPEN"
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT count(*) AS n FROM sales_order_allocations "
                           " WHERE sales_order_id = %s AND status = 'active'",
                           (ids["order_id"],))
                assert sc.fetchone()["n"] == 0, "no reservation on a closed order"
        finally:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_close_loses_to_a_concurrent_close(self, monkeypatch):
        """Two exits racing: the second re-reads under its own lock and refuses
        rather than overwriting the first one's reason."""
        result, ids, seed = self._run_race(
            monkeypatch,
            seed_kwargs={"status": "confirmed"},
            call=lambda i: main.close_sales_order(
                main.SalesOrderCloseRequest(reason="shipped_recorded", mode="commit"),
                _StubRequest(), i["order_id"], True),
            conflicting_sql=self.CLOSE_SQL,
        )
        try:
            assert isinstance(result, HTTPException), result
            assert result.status_code == 409
            assert result.detail["error_code"] == "ORDER_NOT_OPEN"
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT state_reason FROM sales_orders WHERE id = %s",
                           (ids["order_id"],))
                assert sc.fetchone()["state_reason"] == "short_closed", (
                    "the winner's reason survives"
                )
        finally:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_quantity_reduction_queues_on_the_order_row(self, monkeypatch):
        """A quantity reduction shrinks that line's reservations, so it
        contends with the exits over the same allocation rows. It must queue on
        the ORDER row rather than walking straight into them."""
        result, ids, seed = self._run_race(
            monkeypatch,
            seed_kwargs={"status": "confirmed", "with_alloc": True},
            call=lambda i: main.update_order_line(
                _StubRequest(), i["order_id"], i["line_id"], 10, None, True),
            conflicting_sql=self.CANCEL_SQL,
        )
        try:
            _ = result
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT state FROM sales_orders WHERE id = %s", (ids["order_id"],))
                assert sc.fetchone()["state"] == "cancelled", "the cancel won"
        finally:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_line_cancel_queues_on_the_order_row(self, monkeypatch):
        result, ids, seed = self._run_race(
            monkeypatch,
            seed_kwargs={"status": "confirmed", "with_alloc": True},
            call=lambda i: main.cancel_order_line(
                _StubRequest(), i["order_id"], i["line_id"], True),
            conflicting_sql=self.CLOSE_SQL,
        )
        try:
            _ = result
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT state FROM sales_orders WHERE id = %s", (ids["order_id"],))
                assert sc.fetchone()["state"] == "closed", "the close won"
        finally:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_add_line_queues_on_the_order_row(self, monkeypatch):
        """Adding a line advances the order, so it takes the order row lock
        first like every other SO write."""
        result, ids, seed = self._run_race(
            monkeypatch,
            seed_kwargs={"status": "confirmed"},
            call=lambda i: main.add_order_lines(
                i["order_id"],
                main.AddOrderLines(lines=[main.OrderLineInput(
                    product_name="ignored", quantity_lb=5)]),
                True),
            conflicting_sql=self.CANCEL_SQL,
        )
        try:
            _ = result
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT state FROM sales_orders WHERE id = %s", (ids["order_id"],))
                assert sc.fetchone()["state"] == "cancelled", "the cancel won"
                sc.execute("SELECT count(*) AS n FROM sales_order_lines "
                           " WHERE sales_order_id = %s", (ids["order_id"],))
                assert sc.fetchone()["n"] == 1, (
                    "no line may land on an order that has been cancelled"
                )
        finally:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                _cleanup_committed(sc, ids)
            seed.close()


# ═════════════════════════════════════════════════════════════════
# Two writers, forced to overlap: deadlock freedom
#
# Scope: SALES-ORDER write paths only. The production/batch commit paths
# (make, pack, reassign_lot) take product locks under their own rules and are
# NOT covered by these tests or by this PR — see the findings doc.
# ═════════════════════════════════════════════════════════════════

class TestNoDeadlock:
    """Two writers, staged into the exact interleaving that the pre-fix
    locking would have deadlocked on.

    Each test drives the writers to a known step with parked holders, asserts
    the resulting blocking chain, and only then releases. Asserting the CHAIN —
    not merely that both are blocked somewhere — is what makes these fail under
    the pre-fix ordering rather than passing vacuously.

    Postgres reports only the DIRECT blocker, so a two-deep queue has the
    second writer naming the first, which names the holder. `_reaches_holder`
    follows that chain; `_wait_until_blocked_by` pins a direct edge where the
    test needs one specific link rather than reachability.
    """

    @pytest.mark.db
    def test_multi_product_ship_and_exit_do_not_deadlock(self, monkeypatch):
        """Ship order A (lines P_high, P_low) vs close order B (reservations
        P_low, P_high), staged with a holder on EACH product.

        Under the fixed ordering both writers sort their products ascending, so
        both reach for P_low FIRST and queue behind H1. Neither ever touches
        P_high, so neither is behind H2.

        Under the OLD ordering — products locked in line order — ship would take
        P_high (queuing behind H2) while close took P_low (queuing behind H1).
        With the holders gone that is the classic cycle: ship holds P_high and
        waits on P_low while close holds P_low and waits on P_high. The
        assertion below discriminates the two directly: it demands ship be in
        H1's chain and NOT in H2's.
        """
        import psycopg2 as pg

        url = _test_url()
        seed = pg.connect(url)
        seed.autocommit = True
        made = {}
        conns = {}
        h_low = h_high = None
        threads = []
        try:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                token = uuid4().hex[:10].upper()
                sc.execute("INSERT INTO customers (name, active) VALUES (%s,true) RETURNING id",
                           (f"LOCKORDER Cust {token}",))
                customer_id = sc.fetchone()["id"]

                products = []
                for n in (1, 2):
                    sc.execute(
                        "INSERT INTO products (name,type,odoo_code,uom,is_service,active) "
                        "VALUES (%s,'finished',%s,'lb',false,true) RETURNING id",
                        (f"LOCKORDER FG{n} {token}", f"LO{n}-{token}"))
                    pid_ = sc.fetchone()["id"]
                    sc.execute(
                        "INSERT INTO lots (product_id,lot_code,entry_source,received_at) "
                        "VALUES (%s,%s,'received',NOW()) RETURNING id",
                        (pid_, f"LO{n}-LOT-{token}"))
                    lot = sc.fetchone()["id"]
                    sc.execute("INSERT INTO transactions (type,timestamp) "
                               "VALUES ('receive',NOW()) RETURNING id")
                    txn = sc.fetchone()["id"]
                    sc.execute("INSERT INTO transaction_lines "
                               "(transaction_id,product_id,lot_id,quantity_lb) "
                               "VALUES (%s,%s,%s,1000)", (txn, pid_, lot))
                    products.append((pid_, lot))
                products.sort()
                (p_low, lot_low), (p_high, lot_high) = products

                def _order(number, status, line_products):
                    sc.execute(
                        "INSERT INTO sales_orders (customer_id,order_number,status) "
                        "VALUES (%s,%s,%s) RETURNING id", (customer_id, number, status))
                    oid = sc.fetchone()["id"]
                    sc.execute("INSERT INTO sales_order_flags "
                               "(so_number,ready,ready_at,ready_by) "
                               "VALUES (%s,true,NOW(),'test')", (number,))
                    lines = []
                    for pid_, lot in line_products:
                        sc.execute(
                            "INSERT INTO sales_order_lines "
                            "(sales_order_id,product_id,quantity_lb,line_status) "
                            "VALUES (%s,%s,50,'pending') RETURNING id", (oid, pid_))
                        lines.append((sc.fetchone()["id"], pid_, lot))
                    return oid, number, lines

                # A's lines are created P_HIGH first, so line order and product
                # order deliberately disagree.
                a_id, a_num, _a_lines = _order(
                    f"LO-A-{token}", "ready", [(p_high, lot_high), (p_low, lot_low)])
                b_id, b_num, b_lines = _order(
                    f"LO-B-{token}", "confirmed", [(p_low, lot_low), (p_high, lot_high)])
                for line_id, pid_, lot in b_lines:
                    sc.execute(
                        "INSERT INTO sales_order_allocations "
                        "(sales_order_id,sales_order_line_id,product_id,lot_id,"
                        " quantity_lb,source) VALUES (%s,%s,%s,%s,50,'staged_lot')",
                        (b_id, line_id, pid_, lot))
                made = {"customer_id": customer_id, "products": [p_low, p_high],
                        "orders": [(a_id, a_num), (b_id, b_num)]}

            conns = {"ship": _EndpointConn(url, "lockorder-ship"),
                     "close": _EndpointConn(url, "lockorder-close")}
            monkeypatch.setattr(main, "get_db_connection", _provider_for(conns))

            h_low = _Holder(url, "p-low")
            h_low.park_on_lots(p_low)
            h_high = _Holder(url, "p-high")
            h_high.park_on_lots(p_high)

            out = {}
            # Staged, not raced: ship first, and only once it has actually
            # parked do we start the close. "Both started" is not the same as
            # "both reached the step we need".
            threads.append(_spawn("ship", lambda: main.ship_order(
                _StubRequest(), a_id,
                main.ShipOrderRequest(mode="commit", ship_all=True), True), out))
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                ship_q = _wait_until_reaches(sc, conns["ship"].pid, h_low.pid,
                                             participants=[conns["close"].pid])

            threads.append(_spawn("close", lambda: main.close_sales_order(
                main.SalesOrderCloseRequest(reason="short_closed", mode="commit"),
                _StubRequest(), b_id, True), out))
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                close_q = _wait_until_reaches(sc, conns["close"].pid, h_low.pid,
                                              participants=[conns["ship"].pid])
                graph = _chain_snapshot(
                    sc, [conns["ship"].pid, conns["close"].pid, h_low.pid, h_high.pid])
                ship_on_high = _reaches_holder(conns["ship"].pid, h_high.pid, graph)
                close_on_high = _reaches_holder(conns["close"].pid, h_high.pid, graph)

            # Unwind before judging: both writers are parked on connections this
            # test owns, and raising here would strand them.
            h_low.release()
            h_high.release()
            _join_all(threads)

            assert ship_q, (
                "the ship never queued on the LOW product — under line-order "
                "locking it would be queued on the high product instead"
            )
            assert close_q, "the close never queued on the low product"
            assert not ship_on_high, (
                "the ship reached the HIGH product first: products are not "
                "being locked ascending"
            )
            assert not close_on_high, "the close reached the high product first"

            _assert_no_deadlock(out["ship"], "ship")
            _assert_no_deadlock(out["close"], "close")
            assert out["ship"]["order_status"] in ("shipped", "partial_ship"), out["ship"]
            assert out["close"]["state"] == "closed", out["close"]
        finally:
            for h in (h_low, h_high):
                if h is not None:
                    h.close()
            for c in conns.values():
                c.close()
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                for oid, num in made.get("orders", []):
                    _cleanup_order(sc, oid, num)
                for pid_ in made.get("products", []):
                    sc.execute("UPDATE products SET active = false WHERE id = %s", (pid_,))
                if made.get("customer_id"):
                    sc.execute("SELECT EXISTS (SELECT 1 FROM sales_orders "
                               " WHERE customer_id = %s) AS used", (made["customer_id"],))
                    if not sc.fetchone()["used"]:
                        sc.execute("DELETE FROM customers WHERE id = %s",
                                   (made["customer_id"],))
            seed.close()

    @pytest.mark.db
    def test_reciprocal_duplicate_cancellations_do_not_deadlock(self, monkeypatch):
        """A→B and B→A, released together from a single holder.

        Writing related_so_id takes FOR KEY SHARE on the OTHER order's row, so
        each cancel holds its own order row and then reaches for its partner's.
        The holder sits on both rows FOR SHARE — which conflicts with FOR NO KEY
        UPDATE — so neither can complete step 1 until it releases, and they are
        released into that reach-across simultaneously.

        This test on its own establishes that the schedule COMPLETES under the
        current locking. It does not, by itself, prove the schedule would
        deadlock under FOR UPDATE — that comes from mutation (b) in the PR
        body, which switches _load_so_for_state_change back to FOR UPDATE and
        observes this test fail with SQLSTATE 40P01.
        """
        import psycopg2 as pg

        url = _test_url()
        seed = pg.connect(url)
        seed.autocommit = True
        ids_a = ids_b = None
        conns = {}
        holder = None
        threads = []
        try:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                ids_a = _seed_committed(sc, status="confirmed")
                ids_b = _seed_committed(sc, status="confirmed")

            conns = {"a": _EndpointConn(url, "recip-a"),
                     "b": _EndpointConn(url, "recip-b")}
            monkeypatch.setattr(main, "get_db_connection", _provider_for(conns))

            holder = _Holder(url, "recip")
            holder.park_on_orders_share([ids_a["order_id"], ids_b["order_id"]])

            def _cancel(mine, other):
                return lambda: main.cancel_sales_order(
                    main.SalesOrderCancelRequest(
                        reason="duplicate", related_so_id=other["order_id"],
                        mode="commit"),
                    _StubRequest(), mine["order_id"], True)

            out = {}
            threads.append(_spawn("a", _cancel(ids_a, ids_b), out))
            threads.append(_spawn("b", _cancel(ids_b, ids_a), out))

            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                a_q = _wait_until_reaches(sc, conns["a"].pid, holder.pid,
                                          participants=[conns["b"].pid])
                b_q = _wait_until_reaches(sc, conns["b"].pid, holder.pid,
                                          participants=[conns["a"].pid])

            holder.release()
            _join_all(threads)

            assert a_q, "cancel A→B never queued on the holder"
            assert b_q, "cancel B→A never queued on the holder"

            for label in ("a", "b"):
                _assert_no_deadlock(out[label], label)
                assert out[label]["state"] == "cancelled", out[label]

            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT id, related_so_id FROM sales_orders WHERE id = ANY(%s)",
                           ([ids_a["order_id"], ids_b["order_id"]],))
                rows = {r["id"]: r["related_so_id"] for r in sc.fetchall()}
            assert rows[ids_a["order_id"]] == ids_b["order_id"]
            assert rows[ids_b["order_id"]] == ids_a["order_id"], (
                "both reciprocal references survived"
            )
        finally:
            if holder is not None:
                holder.close()
            for c in conns.values():
                c.close()
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                for ids in (ids_a, ids_b):
                    if ids:
                        _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_quantity_reduction_and_exit_on_the_same_order_do_not_deadlock(self, monkeypatch):
        """A quantity reduction and an exit on the SAME order and line.

        Staged: the reduction starts first and runs order row → line →
        product, where a holder on that product's lots stops it. The exit then
        starts and cannot even take step 1, because the reduction is holding
        the order row. The chain is therefore exactly:

            exit  ->  reduction  ->  holder

        and the test asserts that precise shape, including the DIRECT edge from
        the exit to the reduction.

        Under the old code the reduction took no order row lock: it held the
        line and the product, and its allocation-shrink write needed KEY SHARE
        on the order — while the exit held the order row FOR UPDATE (which
        conflicts with KEY SHARE) and was itself waiting on the product. Each
        held what the other needed. Mutation (c) in the PR body removes the
        order lock from the reduction and this assertion fails, because the
        exit is then blocked directly by the holder instead of by the
        reduction.
        """
        import psycopg2 as pg

        url = _test_url()
        seed = pg.connect(url)
        seed.autocommit = True
        ids = None
        conns = {}
        holder = None
        threads = []
        try:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                ids = _seed_committed(sc, status="confirmed", with_alloc=True)

            conns = {"reduce": _EndpointConn(url, "qty-reduce"),
                     "exit": _EndpointConn(url, "qty-exit")}
            monkeypatch.setattr(main, "get_db_connection", _provider_for(conns))

            holder = _Holder(url, "qty")
            holder.park_on_lots(ids["product_id"])

            out = {}
            threads.append(_spawn("reduce", lambda: main.update_order_line(
                _StubRequest(), ids["order_id"], ids["line_id"], 10, None, True), out))
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                reduce_q = _wait_until_reaches(sc, conns["reduce"].pid, holder.pid,
                                               participants=[conns["exit"].pid])

            threads.append(_spawn("exit", lambda: main.close_sales_order(
                main.SalesOrderCloseRequest(reason="short_closed", mode="commit"),
                _StubRequest(), ids["order_id"], True), out))
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                exit_q = _wait_until_blocked_by(
                    sc, conns["exit"].pid, conns["reduce"].pid)

            holder.release()
            _join_all(threads)

            assert reduce_q, (
                "the quantity reduction never queued on the product holder"
            )
            assert "for update" in " ".join(reduce_q.lower().split()), (
                f"the reduction is not parked on a product lock: {reduce_q!r}"
            )
            assert exit_q, (
                "the exit was not blocked BY THE REDUCTION — the reduction is "
                "not holding the order row, so the exit reached the product "
                "holder directly"
            )
            assert "for no key update" in " ".join(exit_q.lower().split()), (
                f"the exit is not parked on the order row: {exit_q!r}"
            )

            _assert_no_deadlock(out["reduce"], "quantity reduction")
            _assert_no_deadlock(out["exit"], "close")
            assert out["exit"]["state"] == "closed", out["exit"]
            assert out["reduce"]["allocations_released"], (
                "the reduction's allocation shrink must have landed"
            )
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT quantity_lb FROM sales_order_lines WHERE id = %s",
                           (ids["line_id"],))
                assert float(sc.fetchone()["quantity_lb"]) == 10
        finally:
            if holder is not None:
                holder.close()
            for c in conns.values():
                c.close()
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                if ids:
                    _cleanup_committed(sc, ids)
            seed.close()

    @pytest.mark.db
    def test_add_line_and_cancel_do_not_deadlock_and_no_line_lands(self, monkeypatch):
        """Adding a line to an order while that same order is being cancelled.

        HOLDER POSITION: the order row itself, FOR SHARE — which conflicts with
        FOR NO KEY UPDATE, so BOTH writers queue at step 1 and are released
        together. Whichever wins, the loser must see the winner's committed
        state under its own lock: no line may land on a cancelled order.
        """
        import psycopg2 as pg

        url = _test_url()
        seed = pg.connect(url)
        seed.autocommit = True
        ids = None
        conns = {}
        holder = None
        threads = []
        product_name = None
        try:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                ids = _seed_committed(sc, status="confirmed")
                sc.execute("SELECT name FROM products WHERE id = %s", (ids["product_id"],))
                product_name = sc.fetchone()["name"]

            conns = {"add": _EndpointConn(url, "addline-add"),
                     "cancel": _EndpointConn(url, "addline-cancel")}
            monkeypatch.setattr(main, "get_db_connection", _provider_for(conns))

            holder = _Holder(url, "addline")
            holder.park_on_orders_share([ids["order_id"]])

            out = {}
            threads.append(_spawn("add", lambda: main.add_order_lines(
                ids["order_id"],
                main.AddOrderLines(lines=[main.OrderLineInput(
                    product_name=product_name, quantity_lb=25)]),
                True), out))
            threads.append(_spawn("cancel", lambda: main.cancel_sales_order(
                main.SalesOrderCancelRequest(reason="customer_cancelled", mode="commit"),
                _StubRequest(), ids["order_id"], True), out))

            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                add_q = _wait_until_reaches(sc, conns["add"].pid, holder.pid,
                                            participants=[conns["cancel"].pid])
                cancel_q = _wait_until_reaches(sc, conns["cancel"].pid, holder.pid,
                                               participants=[conns["add"].pid])

            holder.release()
            _join_all(threads)

            assert add_q, "the add never queued on the order row"
            assert cancel_q, "the cancel never queued on the order row"

            for label in ("add", "cancel"):
                if isinstance(out[label], HTTPException):
                    assert out[label].status_code == 409, out[label].detail
                else:
                    _assert_no_deadlock(out[label], label)

            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT state FROM sales_orders WHERE id = %s", (ids["order_id"],))
                state = sc.fetchone()["state"]
                sc.execute("SELECT count(*) AS n FROM sales_order_lines "
                           " WHERE sales_order_id = %s", (ids["order_id"],))
                lines = sc.fetchone()["n"]

            # The cancel always wins the state — it is the only writer that
            # sets one. Which of the two got the row lock first is genuinely
            # nondeterministic, so the invariant is keyed on the ADD's own
            # outcome, not on the final state:
            #
            #   add succeeded  -> it ran while the order was still open, and
            #                     the cancel landed afterwards. Its line is a
            #                     legitimate line on an order later cancelled.
            #   add refused    -> it ran after the cancel committed, saw
            #                     state='cancelled' under its own row lock,
            #                     and inserted nothing.
            #
            # What must never happen is an add that observed 'cancelled' and
            # inserted anyway, or one that succeeded without leaving its line.
            assert state == "cancelled", "the cancel is the only writer that sets state"
            if isinstance(out["add"], HTTPException):
                assert out["add"].detail["error_code"] == "ORDER_NOT_OPEN", out["add"].detail
                assert lines == 1, (
                    "the add was refused, so it must not have inserted anything"
                )
            else:
                assert lines == 2, (
                    "the add reported success, so its line must actually be there"
                )
        finally:
            if holder is not None:
                holder.close()
            for c in conns.values():
                c.close()
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                if ids:
                    _cleanup_committed(sc, ids)
            seed.close()


def test_allocation_path_locks_one_product_and_lock_helper_sorts():
    """The allocation endpoint locks exactly one product per request, so
    "ascending" is trivially satisfied there; the multi-product helper is what
    has to sort, and _lock_allocation_product itself orders lots and
    allocations by id."""
    import inspect
    alloc_src = inspect.getsource(main.create_sales_order_allocation)
    assert alloc_src.count("_lock_allocation_product(") == 1, (
        "the allocation path is expected to lock a single product"
    )
    assert "_lock_allocation_products(" not in alloc_src

    helper = inspect.getsource(main._lock_allocation_products)
    assert "sorted(" in helper, "multi-product locking must be ascending by id"

    one = inspect.getsource(main._lock_allocation_product)
    assert one.count("ORDER BY id") >= 2, "lots and allocations both locked by id"


def test_ship_commit_takes_locks_in_the_normative_order():
    """order row -> line rows -> products, with products pre-acquired before
    either planning loop."""
    import inspect
    src = inspect.getsource(main.ship_order)
    commit = src[src.index('# mode == "commit"'):]

    i_order = commit.index("FOR NO KEY UPDATE OF so")
    i_lines = commit.index("_lock_sales_order_lines(")
    i_products = commit.index("_lock_allocation_products(")
    i_preflight = commit.index("if _allocations_enforced():")
    i_plan = commit.index("plan = _sales_order_ship_plan(")

    assert i_order < i_lines < i_products, (
        "lock order must be order row -> lines -> products"
    )
    assert i_products < i_preflight, "products pre-acquired before the preflight loop"
    assert i_products < i_plan, "products pre-acquired before the shipping loop"

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

from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

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


def _run_backfill(cur):
    """Re-run only the backfill DO block against rows seeded in this test."""
    sql = MIGRATION.read_text()
    start = sql.index("DO $$\nDECLARE")
    end = sql.index("END $$;", start) + len("END $$;")
    cur.execute(sql[start:end])


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

    db_cursor.execute(
        "SELECT status, released_at, released_by, release_reason "
        "  FROM sales_order_allocations WHERE id = %s",
        (alloc_id,),
    )
    row = db_cursor.fetchone()
    assert row["status"] == "released"
    assert row["released_at"] is not None, "the audit is the released_at/by/reason stamp"
    assert row["release_reason"] == "order_closed"
    assert row["released_by"] == "office-jo"


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
# Tiered health
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_health_critical_on_a_stock_shortage_and_names_the_line_and_pounds(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=10)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=date.today() + timedelta(days=30))
    line_id = _add_line(db_cursor, order_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "critical"
    assert any("90" in r and str(line_id) in r for r in health["reasons"]), health["reasons"]


@pytest.mark.db
def test_health_warning_when_overdue_and_not_shipped(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=date.today() - timedelta(days=3))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "warning"
    assert any("3 days overdue" in r for r in health["reasons"]), health["reasons"]


@pytest.mark.db
def test_health_warning_when_factory_ready_unset_and_ship_date_is_close(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                              ship_date=date.today() + timedelta(days=2))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "warning"
    assert any("Factory Ready not set" in r for r in health["reasons"]), health["reasons"]


@pytest.mark.db
def test_health_quiet_when_nothing_is_wrong(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=date.today() + timedelta(days=30))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    assert client.get(f"/sales/orders/{order_id}").json()["health"]["level"] == "quiet"


@pytest.mark.db
def test_health_info_for_unallocated_never_raises_the_level(db_cursor, client, monkeypatch):
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=500)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=date.today() + timedelta(days=30))
    _add_line(db_cursor, order_id, product_id, 100)

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health["level"] == "quiet", "info must not escalate"
    assert health["reasons"] == []
    assert any("not allocated" in i for i in health["info"]), health["info"]


@pytest.mark.db
def test_health_is_quiet_once_the_order_is_closed(db_cursor, client):
    """An order off the board stops asking for attention, shortage or not."""
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=0)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=date.today() - timedelta(days=10))
    _add_line(db_cursor, order_id, product_id, 100)

    assert client.get(f"/sales/orders/{order_id}").json()["health"]["level"] == "critical"

    closed = client.post(f"/sales/orders/{order_id}/close",
                         json={"reason": "short_closed", "mode": "commit"})
    assert closed.status_code == 200, closed.text

    health = client.get(f"/sales/orders/{order_id}").json()["health"]
    assert health == {"level": "quiet", "reasons": [], "info": []}


@pytest.mark.db
def test_health_is_quiet_once_the_order_is_cancelled(db_cursor, client):
    customer_id, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=0)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=date.today() - timedelta(days=10))
    _add_line(db_cursor, order_id, product_id, 100)
    client.post(f"/sales/orders/{order_id}/cancel",
                json={"reason": "customer_cancelled", "mode": "commit"})
    assert client.get(f"/sales/orders/{order_id}").json()["health"] == {
        "level": "quiet", "reasons": [], "info": []}


def test_health_function_is_labelled_provisional():
    """Owner ruling 4: the shape is the contract, the tier rules are not."""
    doc = main.compute_so_health.__doc__ or ""
    assert "v1 — provisional" in doc
    assert "response shape is the contract" in doc


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
                              ship_date=date.today() + timedelta(days=5))
    _add_line(db_cursor, ready_id, product_id, 10)

    # open + overdue + not shipped
    overdue_id, _ = _seed_order(db_cursor, customer_id, token, floor_ready=False,
                                ship_date=date.today() - timedelta(days=2))
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
    """Dashboard-only by design; the GPT yaml is already at its 30-op ceiling."""
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

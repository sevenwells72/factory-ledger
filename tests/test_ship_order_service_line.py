"""Tests for F05-04: ship_order auto-fulfills is_service lines.

Context: `ship_order` (commit path) previously ran every non-terminal
sales_order_line through FIFO inventory lookup. Service products
(Pallet Charge, freight, etc.) have no lots, so commit returned
`status='no_stock'` for those lines and the order was stuck at
`partial_ship` forever. See `audits/2026-04/15-traceability-gaps.md`
F05-04 and migration 024 (operator workaround).

The fixture `ship_order_db` redirects `main.get_db_connection()` to the
test's savepoint-backed connection, so ship_order's own writes are
rolled back along with the rest of the test transaction.
"""

from contextlib import contextmanager

import pytest

try:
    from fastapi import HTTPException
except ImportError:  # pragma: no cover — FastAPI should be installed in the test env
    pytest.skip("fastapi not installed", allow_module_level=True)

try:
    import main
    from main import ship_order, ShipOrderRequest
except Exception as e:  # pragma: no cover — PEP 604 syntax requires Python 3.10+
    pytest.skip(f"cannot import main ({e})", allow_module_level=True)


class _ConnProxy:
    """Wrap a real psycopg2 connection so commit()/rollback() operate on
    an inner SAVEPOINT. The outer `db_cursor` fixture rolls back the enclosing
    savepoint on teardown, so every write made during the test — including
    those ship_order commits internally — is undone."""

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
def ship_order_db(_db_connection, monkeypatch):
    """Monkeypatch `main.get_db_connection` to yield the test connection via
    a proxy that can't actually commit past the outer savepoint."""

    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "ship_order_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    return _db_connection


def _seed(cur, *, stock_lb, service_only=False, physical_qty_lb=100):
    """Seed customer + confirmed order with a physical line (optional) and a
    service line. `stock_lb` is the on-hand inventory for the physical
    product (set to 0 for under-stocked scenarios)."""

    cur.execute(
        "INSERT INTO customers (name, active) VALUES ('TEST-F05-04 Customer', true) RETURNING id"
    )
    customer_id = cur.fetchone()["id"]

    # Existing service products in prod use type='packaging' (pallet charges, etc.)
    cur.execute(
        "INSERT INTO products (name, type, uom, is_service, active) "
        "VALUES ('TEST-F05-04 Pallet Charge', 'packaging', 'each', true, true) RETURNING id"
    )
    service_pid = cur.fetchone()["id"]

    physical_pid = None
    if not service_only:
        cur.execute(
            "INSERT INTO products (name, type, uom, is_service, active) "
            "VALUES ('TEST-F05-04 Flour 25LB', 'ingredient', 'lb', false, true) RETURNING id"
        )
        physical_pid = cur.fetchone()["id"]

        cur.execute(
            "INSERT INTO lots (product_id, lot_code, supplier_lot_code, entry_source) "
            "VALUES (%s, 'TEST-F05-04-LOT-001', 'N/A', 'received') RETURNING id",
            (physical_pid,),
        )
        lot_id = cur.fetchone()["id"]

        if stock_lb > 0:
            cur.execute(
                "INSERT INTO transactions (type, timestamp) VALUES ('receive', now()) RETURNING id"
            )
            receive_txn_id = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
                "VALUES (%s, %s, %s, %s)",
                (receive_txn_id, physical_pid, lot_id, stock_lb),
            )

    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status) "
        "VALUES (%s, 'TEST-F05-04-SO', 'confirmed') RETURNING id",
        (customer_id,),
    )
    order_id = cur.fetchone()["id"]

    physical_line_id = None
    if physical_pid is not None:
        cur.execute(
            "INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb) "
            "VALUES (%s, %s, %s) RETURNING id",
            (order_id, physical_pid, physical_qty_lb),
        )
        physical_line_id = cur.fetchone()["id"]

    cur.execute(
        "INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb) "
        "VALUES (%s, %s, %s) RETURNING id",
        (order_id, service_pid, 1),
    )
    service_line_id = cur.fetchone()["id"]

    return {
        "order_id": order_id,
        "physical_line_id": physical_line_id,
        "service_line_id": service_line_id,
    }


def _line_state(cur, line_id):
    cur.execute(
        "SELECT quantity_lb, quantity_shipped_lb, line_status "
        "FROM sales_order_lines WHERE id = %s",
        (line_id,),
    )
    return cur.fetchone()


def _order_status(cur, order_id):
    cur.execute("SELECT status FROM sales_orders WHERE id = %s", (order_id,))
    return cur.fetchone()["status"]


@pytest.mark.db
def test_full_stock_plus_service_ships_order(db_cursor, ship_order_db):
    """Scenario A: physical line fully stocked + service line → order='shipped',
    service line line_status='fulfilled'."""
    seeded = _seed(db_cursor, stock_lb=100, physical_qty_lb=100)

    resp = ship_order(
        order_id=seeded["order_id"],
        req=ShipOrderRequest(mode="commit", ship_all=True),
        _=True,
    )

    assert resp["order_status"] == "shipped", resp
    assert _order_status(db_cursor, seeded["order_id"]) == "shipped"

    svc = _line_state(db_cursor, seeded["service_line_id"])
    assert svc["line_status"] == "fulfilled"
    assert float(svc["quantity_shipped_lb"]) == float(svc["quantity_lb"])

    phys = _line_state(db_cursor, seeded["physical_line_id"])
    assert phys["line_status"] == "fulfilled"

    # Response should distinguish service vs physical.
    svc_result = next(r for r in resp["lines_shipped"] if r["line_id"] == seeded["service_line_id"])
    assert svc_result.get("is_service") is True
    phys_result = next(r for r in resp["lines_shipped"] if r["line_id"] == seeded["physical_line_id"])
    assert not phys_result.get("is_service")


@pytest.mark.db
def test_partial_stock_plus_service_still_fulfills_service(db_cursor, ship_order_db):
    """Scenario B: physical line under-stocked + service line → order='partial_ship'
    (because physical is short), service line still auto-fulfills."""
    seeded = _seed(db_cursor, stock_lb=50, physical_qty_lb=100)

    resp = ship_order(
        order_id=seeded["order_id"],
        req=ShipOrderRequest(mode="commit", ship_all=True),
        _=True,
    )

    assert resp["order_status"] == "partial_ship", resp
    assert _order_status(db_cursor, seeded["order_id"]) == "partial_ship"

    svc = _line_state(db_cursor, seeded["service_line_id"])
    assert svc["line_status"] == "fulfilled"
    assert float(svc["quantity_shipped_lb"]) == float(svc["quantity_lb"])

    phys = _line_state(db_cursor, seeded["physical_line_id"])
    assert phys["line_status"] == "partial"
    assert float(phys["quantity_shipped_lb"]) == 50.0


@pytest.mark.db
def test_service_only_order_raises_zero_shipment(db_cursor, ship_order_db):
    """Scenario C: service-only order → ZERO_SHIPMENT error, no state change.
    Service lines don't count toward the physical-shipped guard."""
    seeded = _seed(db_cursor, stock_lb=0, service_only=True)

    svc_before = _line_state(db_cursor, seeded["service_line_id"])
    status_before = _order_status(db_cursor, seeded["order_id"])

    with pytest.raises(HTTPException) as exc_info:
        ship_order(
            order_id=seeded["order_id"],
            req=ShipOrderRequest(mode="commit", ship_all=True),
            _=True,
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error_code"] == "ZERO_SHIPMENT"

    # Nothing should have changed.
    svc_after = _line_state(db_cursor, seeded["service_line_id"])
    assert svc_after["line_status"] == svc_before["line_status"]
    assert float(svc_after["quantity_shipped_lb"]) == float(svc_before["quantity_shipped_lb"])
    assert _order_status(db_cursor, seeded["order_id"]) == status_before

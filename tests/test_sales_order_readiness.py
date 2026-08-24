"""PR 2 read-only sales-order readiness and dispatch-blocker coverage."""

from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

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
        proxy = _ConnProxy(_db_connection, "readiness_api")
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


def _seed_customer(cur):
    token = uuid4().hex[:10]
    name = f"READINESS Customer {token}"
    cur.execute("INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id", (name,))
    return cur.fetchone()["id"], name, token


def _seed_product(
    cur,
    token,
    *,
    service=False,
    with_lot=False,
    incomplete=False,
    received_at_null=False,
    no_production=False,
    stock=0,
):
    suffix = uuid4().hex[:6]
    cur.execute(
        "INSERT INTO products "
        "(name, type, odoo_code, uom, is_service, no_production, active) "
        "VALUES (%s, %s, %s, %s, %s, %s, true) RETURNING id",
        (
            f"READINESS {'Service' if service else 'FG'} {token} {suffix}",
            "packaging" if service else "finished",
            f"RDY-{token}-{suffix}",
            "each" if service else "lb",
            service,
            no_production,
        ),
    )
    product_id = cur.fetchone()["id"]
    lot_id = None
    if with_lot:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code, entry_source, received_at) "
            "VALUES (%s, %s, %s, CASE WHEN %s THEN NULL ELSE NOW() END) RETURNING id",
            (
                product_id,
                f"STAGED-{token}-{suffix}" if incomplete else f"RDY-LOT-{token}-{suffix}",
                "found_inventory" if incomplete else "received",
                incomplete or received_at_null,
            ),
        )
        lot_id = cur.fetchone()["id"]
        if stock:
            cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
            transaction_id = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
                "VALUES (%s, %s, %s, %s)",
                (transaction_id, product_id, lot_id, stock),
            )
    return product_id, lot_id


def _seed_order(cur, customer_id, token, *, status="confirmed", ship_date=True, floor_ready=True):
    order_number = f"RDY-SO-{token}-{uuid4().hex[:5]}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status, requested_ship_date) "
        "VALUES (%s, %s, %s, CASE WHEN %s THEN CURRENT_DATE ELSE NULL END) RETURNING id",
        (customer_id, order_number, status, ship_date),
    )
    order_id = cur.fetchone()["id"]
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


def _allocate(cur, order_id, line_id, product_id, qty, *, lot_id=None, expires_at=None):
    cur.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, lot_id, quantity_lb, source, expires_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (
            order_id,
            line_id,
            product_id,
            lot_id,
            qty,
            "auto_fifo" if expires_at else ("staged_lot" if lot_id else "manual"),
            expires_at,
        ),
    )
    return cur.fetchone()["id"]


def _expected(cur, product_id, qty, *, status="open"):
    cur.execute(
        "INSERT INTO suppliers (name, active) VALUES (%s, true) RETURNING id",
        (f"READINESS Supplier {uuid4().hex[:8]}",),
    )
    supplier_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO expected_receipts (product_id, supplier_id, expected_qty, status) "
        "VALUES (%s, %s, %s, %s)",
        (product_id, supplier_id, qty, status),
    )


def _post_ship(cur, line_id, product_id, lot_id, qty, *, recorded=None):
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('ship', NOW()) RETURNING id")
    transaction_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)",
        (transaction_id, product_id, lot_id, -qty),
    )
    cur.execute(
        "INSERT INTO sales_order_shipments (sales_order_line_id, transaction_id, quantity_lb) "
        "VALUES (%s, %s, %s)",
        (line_id, transaction_id, round(qty, 2)),
    )
    if recorded is not None:
        cur.execute(
            "UPDATE sales_order_lines SET quantity_shipped_lb = %s, line_status = 'fulfilled' WHERE id = %s",
            (recorded, line_id),
        )
    return transaction_id


def _codes(payload):
    return {item["code"]: item["severity"] for item in payload["blockers"]}


@pytest.mark.db
def test_two_unallocated_coverable_orders_are_both_blocked_on_all_gets(db_cursor, client):
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_ids = []
    for _ in range(2):
        order_id, _ = _seed_order(db_cursor, customer_id, token)
        _add_line(db_cursor, order_id, product_id, 100)
        order_ids.append(order_id)

    dispatch = client.get("/sales/orders/fulfillment-check", params={"customer_name": customer_name})
    assert dispatch.status_code == 200, dispatch.text
    matching = [row for row in dispatch.json()["orders"] if row["order_id"] in order_ids]
    assert len(matching) == 2
    for order in matching:
        assert order["inventory_ready"] is False
        assert order["dispatch_ready"] is False
        assert _codes(order) == {"unallocated": "block"}
        assert order["shortage_lb"] == pytest.approx(0)

    listed = client.get("/sales/orders", params={"customer": customer_name, "limit": 10})
    assert listed.status_code == 200, listed.text
    assert {row["order_id"] for row in listed.json()["orders"]} == set(order_ids)
    assert all(_codes(row) == {"unallocated": "block"} for row in listed.json()["orders"])

    detail = client.get(f"/sales/orders/{order_ids[0]}")
    assert detail.status_code == 200, detail.text
    readiness = detail.json()["lines"][0]["readiness"]
    assert readiness["coverable_lb"] == pytest.approx(100)
    assert readiness["unallocated_need_lb"] == pytest.approx(100)
    assert _codes(readiness) == {"unallocated": "block"}


@pytest.mark.db
def test_shortage_and_inbound_cover_warn_never_fill_the_hole(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_id, product_id, 100)
    _expected(db_cursor, product_id, 100)

    body = client.get(f"/sales/orders/{order_id}").json()
    readiness = body["lines"][0]["readiness"]
    assert readiness["on_hand_lb"] == pytest.approx(0)
    assert readiness["coverable_lb"] == pytest.approx(0)
    assert readiness["shortage_lb"] == pytest.approx(100)
    assert readiness["inventory_ready"] is False
    assert _codes(readiness) == {
        "shortage": "block",
        "unallocated": "block",
        "inbound_cover": "warn",
    }


@pytest.mark.db
def test_fully_allocated_with_inbound_warn_stays_dispatch_ready(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)
    _expected(db_cursor, product_id, 50)

    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["inventory_ready"] is True
    assert body["dispatch_ready"] is True
    assert body["shortage_lb"] == pytest.approx(0)
    assert _codes(body) == {"inbound_cover": "warn"}


@pytest.mark.db
def test_partial_allocation_and_sibling_lines_compete_for_same_sku(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_a = _add_line(db_cursor, order_id, product_id, 80)
    line_b = _add_line(db_cursor, order_id, product_id, 80)
    _allocate(db_cursor, order_id, line_a, product_id, 80)

    body = client.get(f"/sales/orders/{order_id}").json()
    by_id = {line["line_id"]: line["readiness"] for line in body["lines"]}
    assert by_id[line_a]["inventory_ready"] is True
    assert by_id[line_b]["coverable_lb"] == pytest.approx(20)
    assert by_id[line_b]["shortage_lb"] == pytest.approx(60)
    assert _codes(by_id[line_b]) == {"shortage": "block", "unallocated": "block"}

    db_cursor.execute(
        "UPDATE sales_order_allocations SET quantity_lb = 40 WHERE sales_order_line_id = %s",
        (line_a,),
    )
    body = client.get(f"/sales/orders/{order_id}").json()
    line = next(item for item in body["lines"] if item["line_id"] == line_a)["readiness"]
    assert "partial_allocation" in _codes(line)
    assert "unallocated" not in _codes(line)


@pytest.mark.db
def test_service_line_excluded_and_voided_physical_ship_diverges(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=100)
    service_id, _ = _seed_product(db_cursor, token, service=True)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    physical_line = _add_line(db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled")
    _add_line(db_cursor, order_id, service_id, 1, shipped=1, status="fulfilled")
    ship_id = _post_ship(db_cursor, physical_line, product_id, lot_id, 100, recorded=100)

    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["shipped_recorded_lb"] == pytest.approx(100)
    assert body["shipped_effective_lb"] == pytest.approx(100)
    assert body["fulfillment_diverged"] is False
    assert _codes(body)["service_only"] == "info"
    service = next(line for line in body["lines"] if line["line_id"] != physical_line)
    assert service["readiness"] == {
        "inventory_ready": True,
        "fulfillment_diverged": False,
        "blockers": [],
    }

    voided = client.post(
        f"/records/transactions/{ship_id}/corrections",
        json={"event_type": "void", "reason": "readiness divergence test"},
    )
    assert voided.status_code == 200, voided.text
    dispatch = client.get("/sales/orders/fulfillment-check", params={"order_id": order_id})
    assert dispatch.status_code == 200, dispatch.text
    order = next(row for row in dispatch.json()["orders"] if row["order_id"] == order_id)
    assert order["status"] == "shipped"
    assert order["shipped_effective_lb"] == pytest.approx(0)
    assert order["shipped_recorded_lb"] == pytest.approx(100)
    assert order["fulfillment_diverged"] is True
    assert _codes(order)["fulfillment_diverged"] == "block"


@pytest.mark.db
def test_effective_ship_uses_ledger_precision_not_shipment_quantity(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=20)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 20, shipped=10.3333, status="partial")
    _post_ship(db_cursor, line_id, product_id, lot_id, 10.3333)

    readiness = client.get(f"/sales/orders/{order_id}").json()["lines"][0]["readiness"]
    assert readiness["shipped_recorded_lb"] == pytest.approx(10.3333)
    assert readiness["shipped_effective_lb"] == pytest.approx(10.3333)
    assert readiness["remaining_lb"] == pytest.approx(9.6667)
    assert readiness["fulfillment_diverged"] is False


@pytest.mark.db
def test_staging_date_floor_ready_and_no_ship_date_blockers(db_cursor, client, monkeypatch):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(
        db_cursor, token, with_lot=True, incomplete=True, stock=100
    )
    order_id, _ = _seed_order(
        db_cursor, customer_id, token, ship_date=False, floor_ready=False
    )
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    sku_allocation = _allocate(db_cursor, order_id, line_id, product_id, 100)

    body = client.get(f"/sales/orders/{order_id}").json()
    assert _codes(body["lines"][0]["readiness"])["unstaged"] == "block"
    assert _codes(body)["not_floor_ready"] == "block"
    assert _codes(body)["no_ship_date"] == "warn"
    assert body["dispatch_ready"] is False

    db_cursor.execute(
        "UPDATE sales_order_allocations SET status='released', released_at=NOW() WHERE id=%s",
        (sku_allocation,),
    )
    _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)
    body = client.get(f"/sales/orders/{order_id}").json()
    line_codes = _codes(body["lines"][0]["readiness"])
    assert "unstaged" not in line_codes
    assert line_codes["missing_lot_dates"] == "block"

    db_cursor.execute("UPDATE lots SET received_at=NOW() WHERE id=%s", (lot_id,))
    db_cursor.execute(
        "INSERT INTO sales_order_flags (so_number, ready, ready_at, ready_by) "
        "SELECT order_number, true, NOW(), 'test' FROM sales_orders WHERE id=%s",
        (order_id,),
    )
    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["dispatch_ready"] is True
    assert _codes(body) == {"no_ship_date": "warn"}

    db_cursor.execute("DELETE FROM sales_order_flags WHERE so_number=(SELECT order_number FROM sales_orders WHERE id=%s)", (order_id,))
    monkeypatch.setenv("FACTORY_READY_REQUIRED", "false")
    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["dispatch_ready"] is True
    assert _codes(body)["not_floor_ready"] == "warn"


@pytest.mark.db
def test_partial_pin_on_incomplete_lot_clears_unstaged(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(
        db_cursor, token, with_lot=True, incomplete=True, stock=100
    )
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 40, lot_id=lot_id)

    readiness = client.get(f"/sales/orders/{order_id}").json()["lines"][0]["readiness"]
    assert [item["code"] for item in readiness["blockers"]] == [
        "partial_allocation",
        "missing_lot_dates",
    ]


@pytest.mark.db
def test_normal_lot_without_received_at_has_no_staging_or_date_blockers(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(
        db_cursor, token, with_lot=True, received_at_null=True, stock=100
    )
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100, lot_id=lot_id)

    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["dispatch_ready"] is True
    assert _codes(body["lines"][0]["readiness"]) == {}


@pytest.mark.db
def test_no_production_physical_sku_uses_normal_readiness(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(
        db_cursor, token, with_lot=True, no_production=True, stock=100
    )
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    body = client.get(f"/sales/orders/{order_id}").json()
    readiness = body["lines"][0]["readiness"]
    assert readiness["inventory_ready"] is True
    assert readiness["remaining_lb"] == pytest.approx(100)
    assert _codes(readiness) == {}
    assert body["dispatch_ready"] is True


@pytest.mark.db
def test_cross_order_allocation_leaves_competing_order_short(db_cursor, client):
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_a, _ = _seed_order(db_cursor, customer_id, token)
    line_a = _add_line(db_cursor, order_a, product_id, 100)
    _allocate(db_cursor, order_a, line_a, product_id, 100)
    order_b, _ = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_b, product_id, 100)

    response = client.get(
        "/sales/orders/fulfillment-check", params={"customer_name": customer_name}
    )
    assert response.status_code == 200, response.text
    orders = {row["order_id"]: row for row in response.json()["orders"]}
    assert orders[order_a]["inventory_ready"] is True
    assert orders[order_b]["shortage_lb"] == pytest.approx(100)
    assert _codes(orders[order_b]) == {
        "shortage": "block",
        "unallocated": "block",
    }


@pytest.mark.db
def test_non_diverged_partial_ship_order_is_in_fulfillment_check(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="partial_ship")
    line_id = _add_line(db_cursor, order_id, product_id, 100, shipped=40, status="partial")
    _post_ship(db_cursor, line_id, product_id, lot_id, 40)
    _allocate(db_cursor, order_id, line_id, product_id, 60)

    response = client.get("/sales/orders/fulfillment-check", params={"order_id": order_id})
    assert response.status_code == 200, response.text
    order = next(row for row in response.json()["orders"] if row["order_id"] == order_id)
    assert order["status"] == "partial_ship"
    assert order["fulfillment_diverged"] is False
    assert order["shipped_effective_lb"] == pytest.approx(40)


@pytest.mark.db
def test_cancelled_lines_are_excluded_from_readiness(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    active_line = _add_line(db_cursor, order_id, product_id, 100)
    cancelled_line = _add_line(
        db_cursor, order_id, product_id, 100, status="cancelled"
    )
    _allocate(db_cursor, order_id, active_line, product_id, 100)

    body = client.get(f"/sales/orders/{order_id}").json()
    lines = {line["line_id"]: line for line in body["lines"]}
    assert lines[active_line]["readiness"]["inventory_ready"] is True
    assert lines[cancelled_line]["line_status"] == "cancelled"
    assert "readiness" not in lines[cancelled_line]
    assert body["ordered_lb"] == pytest.approx(100)
    assert body["dispatch_ready"] is True


@pytest.mark.db
def test_restore_of_voided_ship_counts_in_shipped_effective(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, lot_id = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token, status="shipped")
    line_id = _add_line(
        db_cursor, order_id, product_id, 100, shipped=100, status="fulfilled"
    )
    ship_id = _post_ship(db_cursor, line_id, product_id, lot_id, 100)

    voided = client.post(
        f"/records/transactions/{ship_id}/corrections",
        json={"event_type": "void", "reason": "readiness restore test"},
    )
    assert voided.status_code == 200, voided.text
    assert client.get(f"/sales/orders/{order_id}").json()["shipped_effective_lb"] == pytest.approx(0)

    restored = client.post(
        f"/records/transactions/{ship_id}/corrections",
        json={"event_type": "restore", "reason": "readiness restore test"},
    )
    assert restored.status_code == 200, restored.text
    body = client.get(f"/sales/orders/{order_id}").json()
    assert body["shipped_effective_lb"] == pytest.approx(100)
    assert body["fulfillment_diverged"] is False


@pytest.mark.db
def test_closed_and_cancelled_expected_receipts_do_not_warn_inbound_cover(db_cursor, client):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)
    _expected(db_cursor, product_id, 50, status="closed")
    _expected(db_cursor, product_id, 70, status="cancelled")

    body = client.get(f"/sales/orders/{order_id}").json()
    readiness = body["lines"][0]["readiness"]
    assert readiness["inbound_open_lb"] == pytest.approx(0)
    assert "inbound_cover" not in _codes(readiness)
    assert body["dispatch_ready"] is True


@pytest.mark.db
def test_expired_allocation_is_formula_only_and_all_gets_never_write(db_cursor, client):
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    allocation_id = _allocate(
        db_cursor, order_id, line_id, product_id, 100,
        expires_at="2000-01-01T00:00:00+00:00",
    )

    responses = [
        client.get(f"/sales/orders/{order_id}"),
        client.get("/sales/orders", params={"customer": customer_name}),
        client.get("/sales/orders/fulfillment-check", params={"order_id": order_id}),
    ]
    assert all(response.status_code == 200 for response in responses)
    assert _codes(responses[0].json()) == {"unallocated": "block"}

    db_cursor.execute(
        "SELECT status, released_at, release_reason FROM sales_order_allocations WHERE id=%s",
        (allocation_id,),
    )
    row = db_cursor.fetchone()
    assert row["status"] == "active"
    assert row["released_at"] is None
    assert row["release_reason"] is None
    assert "UPDATE" not in main.SALES_ORDER_READINESS_SQL.upper()


@pytest.mark.db
def test_readiness_cte_explains_for_a_page_of_orders(db_cursor):
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, with_lot=True, stock=10)
    order_id, _ = _seed_order(db_cursor, customer_id, token)
    _add_line(db_cursor, order_id, product_id, 10)
    db_cursor.execute("EXPLAIN (FORMAT TEXT) " + main.SALES_ORDER_READINESS_SQL, ([order_id],))
    plan = "\n".join(row["QUERY PLAN"] for row in db_cursor.fetchall())
    assert "sales_order_allocations" in plan
    assert "sales_order_shipments" in plan

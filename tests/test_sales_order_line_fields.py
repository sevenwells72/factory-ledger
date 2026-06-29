"""Regression test for the inline-expand line fields on GET /sales/orders/{id}.

Branch feat/sales-order-inline-expand adds `sku` (products.odoo_code) and
`uom` (products.uom) to each line in the sales-order detail response so the
dashboard's inline expand/collapse panel can show SKU, product, ordered qty,
unit of measure, and remaining qty without a second round-trip.

The change is purely additive — these tests assert the new keys are present
and correct, and that pre-existing line keys are untouched. DB writes are
rolled back via the same savepoint-proxy pattern as test_write_response_contract.py.
"""

from contextlib import contextmanager

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

try:
    import main
except Exception as e:  # pragma: no cover
    pytest.skip(f"cannot import main ({e})", allow_module_level=True)


class _ConnProxy:
    """Wrap a real psycopg2 connection so commit()/rollback() operate on an
    inner SAVEPOINT; the outer db fixtures roll everything back on teardown."""

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
        proxy = _ConnProxy(_db_connection, "line_fields_inner")
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


def _seed_customer(conn, name="Line Fields Test Customer"):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
            (name,),
        )
        return cur.fetchone()[0]


def _seed_product(conn, name, odoo_code, uom="lb", case_size_lb=None):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, odoo_code, case_size_lb, active) "
            "VALUES (%s, 'finished', %s, %s, %s, true) RETURNING id",
            (name, uom, odoo_code, case_size_lb),
        )
        return cur.fetchone()[0]


@pytest.mark.db
def test_order_lines_expose_sku_and_uom(client, _db_connection):
    _seed_customer(_db_connection)
    _seed_product(_db_connection, "LINE FIELDS GRANOLA ZQX", "70999", uom="lb")

    created = client.post("/sales/orders", json={
        "customer_name": "Line Fields Test Customer",
        "lines": [{"product_name": "LINE FIELDS GRANOLA ZQX", "quantity_lb": 100}],
    })
    assert created.status_code == 200, created.text
    order_id = created.json()["order_id"]

    resp = client.get(f"/sales/orders/{order_id}")
    assert resp.status_code == 200, resp.text
    line = resp.json()["lines"][0]

    # New additive fields for the inline-expand panel
    assert line["sku"] == "70999"
    assert line["uom"] == "lb"
    # Pre-existing fields untouched
    assert line["product"] == "LINE FIELDS GRANOLA ZQX"
    assert line["quantity_lb"] == 100
    assert "remaining_lb" in line


@pytest.mark.db
def test_uom_defaults_to_lb_when_null(client, _db_connection):
    """uom is coalesced to 'lb' so the panel never shows a blank unit column."""
    _seed_customer(_db_connection, "Line Fields Null UoM Customer")
    # Insert a product with NULL uom directly (bypassing the seed default).
    with _db_connection.cursor() as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, odoo_code, active) "
            "VALUES ('LINE FIELDS NULL UOM ZQX', 'finished', NULL, '70998', true) "
            "RETURNING id"
        )
        cur.fetchone()

    created = client.post("/sales/orders", json={
        "customer_name": "Line Fields Null UoM Customer",
        "lines": [{"product_name": "LINE FIELDS NULL UOM ZQX", "quantity_lb": 50}],
    })
    assert created.status_code == 200, created.text
    order_id = created.json()["order_id"]

    resp = client.get(f"/sales/orders/{order_id}")
    assert resp.status_code == 200, resp.text
    line = resp.json()["lines"][0]
    assert line["uom"] == "lb"
    assert line["sku"] == "70998"


@pytest.mark.db
def test_order_list_exposes_compact_case_line_data_for_pallet_display(client, _db_connection):
    """The list response carries case counts so the UI never derives pallets from pounds."""
    _seed_customer(_db_connection, "Pallet Summary Test Customer")
    _seed_product(
        _db_connection,
        "PALLET SUMMARY GRANOLA 25 LB ZQX",
        "70997",
        uom="25 lb case",
        case_size_lb=25,
    )

    created = client.post("/sales/orders", json={
        "customer_name": "Pallet Summary Test Customer",
        "lines": [{"product_name": "PALLET SUMMARY GRANOLA 25 LB ZQX", "quantity_lb": 600}],
    })
    assert created.status_code == 200, created.text
    order_id = created.json()["order_id"]

    resp = client.get("/sales/orders", params={"customer": "Pallet Summary Test Customer"})
    assert resp.status_code == 200, resp.text
    order = next(item for item in resp.json()["orders"] if item["order_id"] == order_id)
    assert order["pallet_lines"] == [{
        "line_id": order["pallet_lines"][0]["line_id"],
        "product": "PALLET SUMMARY GRANOLA 25 LB ZQX",
        "sku": "70997",
        "uom": "25 lb case",
        "case_size_lb": 25.0,
        "unit_count": 24,
        "shipped_units": 0,
        "remaining_units": 24,
        "is_non_weight": False,
    }]

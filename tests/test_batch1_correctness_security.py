"""Batch 1 security and effective-ledger regression coverage.

No test in this module may use production. Database-backed cases run only
through the guarded TEST_DATABASE_URL fixture from conftest.py and roll back.
"""

from contextlib import contextmanager
import inspect
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor
from starlette.routing import Match

import main


ROOT = Path(__file__).resolve().parents[1]


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
def remediation_client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "batch1_remediation_api")
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


def _insert_transaction(cur, txn_type, lines, **header):
    columns = ["type", "timestamp", "status"] + list(header)
    values = [txn_type, "NOW()", "'posted'"] + ["%s"] * len(header)
    params = list(header.values())
    cur.execute(
        f"INSERT INTO transactions ({', '.join(columns)}) "
        f"VALUES (%s, {', '.join(values[1:])}) RETURNING id",
        [txn_type, *params],
    )
    transaction_id = cur.fetchone()["id"]
    for product_id, lot_id, quantity_lb in lines:
        cur.execute(
            "INSERT INTO transaction_lines "
            "(transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, %s)",
            (transaction_id, product_id, lot_id, quantity_lb),
        )
    return transaction_id


def _seed_trace_graph(cur):
    token = uuid4().hex[:10]
    ingredient_name = f"B1 Ingredient {token}"
    active_batch_name = f"B1 Active Batch {token}"
    void_batch_name = f"B1 Void Batch {token}"
    supplier_lot = f"SUP-B1-{token}"

    product_ids = {}
    for key, name, product_type in (
        ("ingredient", ingredient_name, "ingredient"),
        ("active_batch", active_batch_name, "batch"),
        ("void_batch", void_batch_name, "batch"),
    ):
        cur.execute(
            "INSERT INTO products (name, type, odoo_code, active) "
            "VALUES (%s, %s, %s, true) RETURNING id",
            (name, product_type, f"B1-{key}-{token}"),
        )
        product_ids[key] = cur.fetchone()["id"]

    lots = {}
    for key, product_key, entry_source in (
        ("ingredient", "ingredient", "received"),
        ("active_batch", "active_batch", "production_output"),
        ("void_batch", "void_batch", "production_output"),
    ):
        lot_code = f"B1-{key.upper()}-{token}"
        cur.execute(
            "INSERT INTO lots (product_id, lot_code, entry_source, supplier_lot_code) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (
                product_ids[product_key],
                lot_code,
                entry_source,
                supplier_lot if key == "ingredient" else None,
            ),
        )
        lots[key] = {"id": cur.fetchone()["id"], "code": lot_code}

    receive_id = _insert_transaction(
        cur,
        "receive",
        [(product_ids["ingredient"], lots["ingredient"]["id"], 100)],
        shipper_name="B1 Supplier",
        bol_reference=f"BOL-{token}",
    )
    active_make_id = _insert_transaction(
        cur,
        "make",
        [
            (product_ids["ingredient"], lots["ingredient"]["id"], -10),
            (product_ids["active_batch"], lots["active_batch"]["id"], 10),
        ],
    )
    void_make_id = _insert_transaction(
        cur,
        "make",
        [
            (product_ids["ingredient"], lots["ingredient"]["id"], -20),
            (product_ids["void_batch"], lots["void_batch"]["id"], 20),
        ],
    )
    for transaction_id, quantity in ((active_make_id, 10), (void_make_id, 20)):
        cur.execute(
            "INSERT INTO ingredient_lot_consumption "
            "(transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, %s)",
            (transaction_id, product_ids["ingredient"], lots["ingredient"]["id"], quantity),
        )

    active_direct_ship_id = _insert_transaction(
        cur,
        "ship",
        [(product_ids["ingredient"], lots["ingredient"]["id"], -5)],
        customer_name="B1 Active Customer",
        order_reference=f"ACTIVE-{token}",
    )
    void_direct_ship_id = _insert_transaction(
        cur,
        "ship",
        [(product_ids["ingredient"], lots["ingredient"]["id"], -7)],
        customer_name="B1 Voided Customer",
        order_reference=f"VOID-{token}",
    )
    active_batch_ship_id = _insert_transaction(
        cur,
        "ship",
        [(product_ids["active_batch"], lots["active_batch"]["id"], -2)],
        customer_name="B1 Batch Customer",
        order_reference=f"BATCH-ACTIVE-{token}",
    )
    void_batch_ship_id = _insert_transaction(
        cur,
        "ship",
        [(product_ids["active_batch"], lots["active_batch"]["id"], -3)],
        customer_name="B1 Batch Void Customer",
        order_reference=f"BATCH-VOID-{token}",
    )

    return {
        "token": token,
        "supplier_lot": supplier_lot,
        "product_ids": product_ids,
        "lots": lots,
        "receive_id": receive_id,
        "active_make_id": active_make_id,
        "void_make_id": void_make_id,
        "active_direct_ship_id": active_direct_ship_id,
        "void_direct_ship_id": void_direct_ship_id,
        "active_batch_ship_id": active_batch_ship_id,
        "void_batch_ship_id": void_batch_ship_id,
    }


def _void(client, transaction_id):
    response = client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "void", "reason": "Batch 1 effective-state regression"},
    )
    assert response.status_code == 200, response.text


def test_admin_sql_route_is_removed():
    assert not any(route.path == "/admin/sql" for route in main.app.routes)
    assert "/admin/sql" not in main.app.openapi()["paths"]
    assert not hasattr(main, "admin_sql_query")
    assert not hasattr(main, "AdminSQLQuery")
    old_route_scope = {
        "type": "http",
        "path": "/admin/sql",
        "root_path": "",
        "method": "POST",
    }
    assert all(route.matches(old_route_scope)[0] is Match.NONE for route in main.app.routes)


def test_floor_void_schema_requires_reason_and_keeps_operation_count():
    schema = yaml.safe_load(
        (ROOT / "gpt-configs/schemas/openapi-floor.yaml").read_text(encoding="utf-8")
    )
    operation = schema["paths"]["/void/{transaction_id}"]["post"]
    assert operation["requestBody"]["required"] is True
    body_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert body_schema["$ref"] == "#/components/schemas/VoidRequest"
    void_request = schema["components"]["schemas"]["VoidRequest"]
    assert void_request["required"] == ["reason"]
    assert void_request["properties"]["reason"]["type"] == "string"
    assert void_request["properties"]["reason"]["minLength"] == 1
    assert "reason" in operation["description"].lower()

    methods = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
    operation_count = sum(
        1
        for path_item in schema["paths"].values()
        for method in path_item
        if method.lower() in methods
    )
    assert operation_count == 22


def test_critical_read_sql_uses_effective_views():
    for read_path in (
        main.trace_batch,
        main._trace_ingredient_backward,
        main.trace_ingredient,
        main.trace_supplier_lot,
        main.get_sales_order,
        main.generate_packing_slip,
        main.sales_dashboard,
        main.audit_integrity,
    ):
        source = inspect.getsource(read_path)
        assert "ledger_current_transactions" in source, read_path.__name__
        assert "effective_status = 'posted'" in source, read_path.__name__

    for line_sensitive_path in (
        main.trace_batch,
        main._trace_ingredient_backward,
        main.trace_ingredient,
        main.trace_supplier_lot,
        main.generate_packing_slip,
        main.audit_integrity,
    ):
        source = inspect.getsource(line_sensitive_path)
        assert "ledger_current_transaction_lines" in source, line_sensitive_path.__name__

    integrity_source = inspect.getsource(main.audit_integrity)
    assert "WHERE effective_status = 'voided'" in integrity_source


@pytest.mark.db
def test_effective_trace_directions_exclude_correction_voids_and_keep_originals(
    _db_connection, remediation_client
):
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        graph = _seed_trace_graph(cur)

    for transaction_id in (
        graph["void_make_id"],
        graph["void_direct_ship_id"],
        graph["void_batch_ship_id"],
    ):
        _void(remediation_client, transaction_id)

    active_batch = remediation_client.get(
        f"/trace/batch/{graph['lots']['active_batch']['code']}",
        params={"product_id": graph["product_ids"]["active_batch"]},
    )
    assert active_batch.status_code == 200, active_batch.text
    active_batch_data = active_batch.json()
    assert active_batch_data["trace_type"] == "batch"
    assert active_batch_data["output_lb"] == pytest.approx(10)
    assert [row["transaction_id"] for row in active_batch_data["customer_shipments"]] == [
        graph["active_batch_ship_id"]
    ]
    assert active_batch_data["on_hand_lb"] == pytest.approx(8)

    void_batch = remediation_client.get(
        f"/trace/batch/{graph['lots']['void_batch']['code']}",
        params={"product_id": graph["product_ids"]["void_batch"]},
    )
    assert void_batch.status_code == 200, void_batch.text
    assert void_batch.json()["trace_type"] == "ingredient"
    assert void_batch.json()["on_hand_lb"] == pytest.approx(0)

    ingredient = remediation_client.get(
        f"/trace/ingredient/{graph['lots']['ingredient']['code']}",
        params={"product_id": graph["product_ids"]["ingredient"]},
    )
    assert ingredient.status_code == 200, ingredient.text
    ingredient_data = ingredient.json()
    assert [row["batch_lot_code"] for row in ingredient_data["used_in_batches"]] == [
        graph["lots"]["active_batch"]["code"]
    ]
    assert [row["transaction_id"] for row in ingredient_data["direct_shipments"]] == [
        graph["active_direct_ship_id"]
    ]
    assert ingredient_data["on_hand_lb"] == pytest.approx(85)

    supplier = remediation_client.get(f"/trace/supplier-lot/{graph['supplier_lot']}")
    assert supplier.status_code == 200, supplier.text
    supplier_lot = supplier.json()["matched_internal_lots"][0]
    assert supplier_lot["total_received_lb"] == pytest.approx(100)
    assert [row["batch_lot_code"] for row in supplier_lot["production_usage"]] == [
        graph["lots"]["active_batch"]["code"]
    ]
    assert [row["transaction_id"] for row in supplier_lot["customer_shipments"]] == [
        graph["active_direct_ship_id"]
    ]
    assert supplier_lot["on_hand_lb"] == pytest.approx(85)

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, status FROM transactions WHERE id = ANY(%s) ORDER BY id",
            ([graph["void_make_id"], graph["void_direct_ship_id"], graph["void_batch_ship_id"]],),
        )
        originals = cur.fetchall()
        assert len(originals) == 3
        assert all(row["status"] == "posted" for row in originals)
        cur.execute(
            "SELECT COUNT(*) AS count FROM ledger_corrections "
            "WHERE target_table = 'transactions' AND target_id = ANY(%s)",
            ([row["id"] for row in originals],),
        )
        assert cur.fetchone()["count"] == 3

    integrity = remediation_client.get("/audit/integrity")
    assert integrity.status_code == 200, integrity.text
    production_missing_ilc = next(
        check for check in integrity.json()["checks"] if check["name"] == "production_missing_ilc"
    )
    assert all(
        row["transaction_id"] != graph["void_make_id"]
        for row in production_missing_ilc["details"]
    )


@pytest.mark.db
def test_sales_order_shipment_history_excludes_correction_voids(
    _db_connection, remediation_client
):
    token = uuid4().hex[:10]
    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
            (f"B1 Customer {token}",),
        )
        customer_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO products (name, type, odoo_code, active, case_size_lb) "
            "VALUES (%s, 'finished', %s, true, 10) RETURNING id",
            (f"B1 Finished {token}", f"B1-FG-{token}"),
        )
        product_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO lots (product_id, lot_code, entry_source) "
            "VALUES (%s, %s, 'pack_output') RETURNING id",
            (product_id, f"B1-FG-LOT-{token}"),
        )
        lot_id = cur.fetchone()["id"]
        _insert_transaction(cur, "adjust", [(product_id, lot_id, 100)])
        cur.execute(
            "INSERT INTO sales_orders (order_number, customer_id, status) "
            "VALUES (%s, %s, 'partial_ship') RETURNING id",
            (f"B1-SO-{token}", customer_id),
        )
        order_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO sales_order_lines "
            "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
            "VALUES (%s, %s, 20, 20, 'fulfilled') RETURNING id",
            (order_id, product_id),
        )
        line_id = cur.fetchone()["id"]

        active_ship_id = _insert_transaction(
            cur,
            "ship",
            [(product_id, lot_id, -10)],
            customer_name=f"B1 Customer {token}",
            order_reference=f"B1-SO-{token}",
        )
        void_ship_id = _insert_transaction(
            cur,
            "ship",
            [(product_id, lot_id, -10)],
            customer_name=f"B1 Customer {token}",
            order_reference=f"B1-SO-{token}",
        )
        for transaction_id in (active_ship_id, void_ship_id):
            cur.execute(
                "INSERT INTO sales_order_shipments "
                "(sales_order_line_id, transaction_id, quantity_lb) VALUES (%s, %s, 10)",
                (line_id, transaction_id),
            )
        cur.execute(
            "INSERT INTO shipments (sales_order_id, customer_id, shipped_at) "
            "VALUES (%s, %s, NOW()) RETURNING id",
            (order_id, customer_id),
        )
        shipment_id = cur.fetchone()["id"]
        for transaction_id in (active_ship_id, void_ship_id):
            cur.execute(
                "INSERT INTO shipment_lines "
                "(shipment_id, transaction_id, sales_order_line_id, product_id, quantity_lb) "
                "VALUES (%s, %s, %s, %s, 10)",
                (shipment_id, transaction_id, line_id, product_id),
            )

    _void(remediation_client, void_ship_id)

    order = remediation_client.get(f"/sales/orders/{order_id}")
    assert order.status_code == 200, order.text
    assert [row["transaction_id"] for row in order.json()["shipments"]] == [active_ship_id]

    sales_dashboard = remediation_client.get("/sales/dashboard")
    assert sales_dashboard.status_code == 200, sales_dashboard.text
    recent = next(
        row
        for row in sales_dashboard.json()["recent_shipments_7d"]
        if row["order_number"] == f"B1-SO-{token}"
    )
    assert recent["shipped_lb"] == pytest.approx(10)

    with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
        assert main.lot_on_hand(cur, lot_id) == pytest.approx(90)
        cur.execute("SELECT status FROM transactions WHERE id = %s", (void_ship_id,))
        assert cur.fetchone()["status"] == "posted"

"""Sales-order ship commit retry protection (migration 088, PR #146 review).

A GPT action that times out after the server committed resends the identical
body; that retry must return the first receipt, not record a second shipment.
"""
from pathlib import Path

import pytest

from tests.test_sales_order_state_model import (  # noqa: F401
    client, _seed_customer, _seed_product, _seed_order, _add_line, _void_shipment,
)

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "migrations/088_ship_order_idempotency.sql"
DOWN = ROOT / "migrations/down/088_ship_order_idempotency_down.sql"


@pytest.fixture
def receipts(db_cursor):
    db_cursor.execute(UP.read_text())


def _order(cur, qty=100, stock=500):
    customer_id, token = _seed_customer(cur)
    product_id, _ = _seed_product(cur, token, with_lot=True, stock=stock)
    order_id, _ = _seed_order(cur, customer_id, token, status="ready")
    line_id = _add_line(cur, order_id, product_id, qty)
    return order_id, line_id


def _shipments(cur, order_id):
    cur.execute("SELECT count(*) AS n FROM shipments WHERE sales_order_id = %s", (order_id,))
    return cur.fetchone()["n"]


def _shipped_lb(cur, line_id):
    cur.execute("SELECT quantity_shipped_lb FROM sales_order_lines WHERE id = %s", (line_id,))
    return float(cur.fetchone()["quantity_shipped_lb"])


def _commit(client, order_id, body):
    return client.post(f"/sales/orders/{order_id}/ship/commit", json=body)


@pytest.mark.db
def test_identical_retry_replays_the_first_receipt(db_cursor, client, receipts):
    order_id, line_id = _order(db_cursor)
    body = {"lines": [{"line_id": line_id, "quantity_lb": 40}]}
    first = _commit(client, order_id, body)
    assert first.status_code == 200, first.text
    again = _commit(client, order_id, body)
    assert again.status_code == 200, again.text
    assert again.json()["idempotent_replay"] is True
    assert again.json()["shipment_id"] == first.json()["shipment_id"]
    assert "nothing new was shipped" in again.json()["message"]
    assert f"shipment {first.json()['shipment_id']}" in again.json()["message"]
    assert _shipments(db_cursor, order_id) == 1
    assert _shipped_lb(db_cursor, line_id) == 40


@pytest.mark.db
def test_same_body_after_the_window_is_a_new_shipment(db_cursor, client, receipts):
    order_id, line_id = _order(db_cursor)
    body = {"lines": [{"line_id": line_id, "quantity_lb": 40}]}
    assert _commit(client, order_id, body).status_code == 200
    db_cursor.execute("UPDATE ship_order_commit_receipts SET created_at = created_at - interval '11 minutes' "
                      " WHERE sales_order_id = %s", (order_id,))
    second = _commit(client, order_id, body)
    assert second.status_code == 200, second.text
    assert "idempotent_replay" not in second.json()
    assert _shipments(db_cursor, order_id) == 2
    assert _shipped_lb(db_cursor, line_id) == 80


@pytest.mark.db
def test_different_lines_in_the_window_still_ship(db_cursor, client, receipts):
    order_id, line_id = _order(db_cursor)
    assert _commit(client, order_id, {"lines": [{"line_id": line_id, "quantity_lb": 40}]}).status_code == 200
    second = _commit(client, order_id, {"lines": [{"line_id": line_id, "quantity_lb": 30}]})
    assert second.status_code == 200, second.text
    assert "idempotent_replay" not in second.json()
    assert _shipped_lb(db_cursor, line_id) == 70


@pytest.mark.db
def test_ship_all_retry_replays_the_first_receipt(db_cursor, client, receipts):
    order_id, line_id = _order(db_cursor)
    assert _commit(client, order_id, {"ship_all": True}).status_code == 200
    again = _commit(client, order_id, {"ship_all": True})
    assert again.status_code == 200 and again.json()["idempotent_replay"] is True
    assert _shipments(db_cursor, order_id) == 1 and _shipped_lb(db_cursor, line_id) == 100


@pytest.mark.db
def test_a_voided_shipment_is_not_replayed(db_cursor, client, receipts):
    order_id, line_id = _order(db_cursor)
    body = {"lines": [{"line_id": line_id, "quantity_lb": 40}]}
    first = _commit(client, order_id, body)
    assert first.status_code == 200, first.text
    _void_shipment(client, first.json()["lines_shipped"][0]["transaction_id"], "test: shipment entered by mistake")
    after = _commit(client, order_id, body)
    assert after.status_code == 200, after.text
    assert "idempotent_replay" not in after.json()
    assert after.json()["shipment_id"] != first.json()["shipment_id"]


@pytest.mark.db
def test_refused_commit_stores_no_receipt(db_cursor, client, receipts):
    order_id, _ = _order(db_cursor, stock=0)
    resp = _commit(client, order_id, {"ship_all": True})
    assert resp.status_code == 409, resp.text
    db_cursor.execute("SELECT count(*) AS n FROM ship_order_commit_receipts WHERE sales_order_id = %s", (order_id,))
    assert db_cursor.fetchone()["n"] == 0


@pytest.mark.db
def test_mode_commit_route_is_guarded_too(db_cursor, client, receipts):
    order_id, line_id = _order(db_cursor)
    body = {"mode": "commit", "lines": [{"line_id": line_id, "quantity_lb": 25}]}
    assert client.post(f"/sales/orders/{order_id}/ship", json=body).status_code == 200
    again = client.post(f"/sales/orders/{order_id}/ship", json=body)
    assert again.status_code == 200 and again.json()["idempotent_replay"] is True
    assert _shipped_lb(db_cursor, line_id) == 25


@pytest.mark.db
def test_before_migration_088_the_commit_runs_unguarded(db_cursor, client):
    db_cursor.execute("SELECT to_regclass('public.ship_order_commit_receipts') IS NULL AS absent")
    assert db_cursor.fetchone()["absent"]
    order_id, line_id = _order(db_cursor)
    body = {"lines": [{"line_id": line_id, "quantity_lb": 40}]}
    assert _commit(client, order_id, body).status_code == 200
    assert _commit(client, order_id, body).status_code == 200
    assert _shipped_lb(db_cursor, line_id) == 80


@pytest.mark.db
def test_migration_088_is_rerunnable_and_reversible(db_cursor):
    db_cursor.execute(UP.read_text())
    db_cursor.execute(UP.read_text())
    db_cursor.execute("SELECT count(*) AS n FROM migration_markers WHERE name = '088_ship_order_idempotency'")
    assert db_cursor.fetchone()["n"] == 1
    db_cursor.execute(DOWN.read_text())
    db_cursor.execute("SELECT to_regclass('public.ship_order_commit_receipts') IS NULL AS absent")
    assert db_cursor.fetchone()["absent"]

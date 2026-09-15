"""B-2 dashboard fixes: batch families + yield, ingredient uom, floor case counts."""

from contextlib import contextmanager

import pytest
from psycopg2.extras import RealDictCursor

import main


def _insert_product(cur, *, sku, name, product_type, uom="lb",
                    default_batch_lb=None, yield_multiplier=1.0,
                    case_size_lb=None, on_hand_lb=0):
    cur.execute(
        """
        INSERT INTO products (
            odoo_code, name, type, active, uom, default_batch_lb,
            yield_multiplier, case_size_lb
        )
        VALUES (%s, %s, %s, true, %s, %s, %s, %s)
        RETURNING id
        """,
        (sku, name, product_type, uom, default_batch_lb, yield_multiplier, case_size_lb),
    )
    product_id = cur.fetchone()["id"]
    if on_hand_lb:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (product_id, f"B2-{sku}-LOT"),
        )
        lot_id = cur.fetchone()["id"]
        cur.execute(
            """
            INSERT INTO transactions (
                type, timestamp, status, occurred_at, business_date, operator_id
            )
            VALUES (
                'adjust',
                clock_timestamp() AT TIME ZONE 'UTC',
                'posted',
                clock_timestamp(),
                timezone('America/New_York', clock_timestamp())::date,
                'b2-test'
            )
            RETURNING id
            """
        )
        txn_id = cur.fetchone()["id"]
        cur.execute(
            """
            INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
            VALUES (%s, %s, %s, %s)
            """,
            (txn_id, product_id, lot_id, on_hand_lb),
        )
    return product_id


@pytest.fixture
def api_txn(_db_connection, monkeypatch):
    @contextmanager
    def _test_transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as api_cur:
            yield api_cur

    monkeypatch.setattr(main, "get_transaction", _test_transaction)
    return _db_connection


@pytest.mark.db
def test_batch_panel_includes_coconut_and_applies_yield(api_txn, db_cursor, monkeypatch):
    _insert_product(
        db_cursor, sku="B2-COCO", name="Batch Coconut Sweetened Flake B2",
        product_type="batch", default_batch_lb=360, yield_multiplier=1.0,
        on_hand_lb=4320,
    )
    _insert_product(
        db_cursor, sku="B2-GRA", name="Batch Classic Granola B2",
        product_type="batch", default_batch_lb=323, yield_multiplier=0.8,
        on_hand_lb=516.8,
    )
    _insert_product(
        db_cursor, sku="B2-SKIP", name="Batch Uncategorized Mixer B2",
        product_type="batch", default_batch_lb=100, on_hand_lb=50,
    )
    monkeypatch.setattr(main, "_load_dashboard_config", lambda: {
        "batch_skus": [{"name": "Batch Classic Granola #9", "standard_batch_size_lbs": None}],
    })

    response = main.dashboard_api_batches()
    by_name = {row["product_name"]: row for row in response["batches"]}

    assert "Batch Coconut Sweetened Flake B2" in by_name
    coco = by_name["Batch Coconut Sweetened Flake B2"]
    assert coco["production_family"] == "coconut"
    assert coco["yield_multiplier"] == 1.0
    assert coco["standard_batch_size_lbs"] == 360
    assert coco["made_unit_size_lbs"] == pytest.approx(360)
    # 4320 posted lb / 360 lb per pan = 12.0 pans.
    assert coco["batch_count"] == pytest.approx(12.0)
    assert coco["lots"][0]["batch_count"] == pytest.approx(12.0)

    granola = by_name["Batch Classic Granola B2"]
    assert granola["production_family"] == "granola"
    assert granola["batch_count"] == pytest.approx(2.0)

    assert "Batch Uncategorized Mixer B2" not in by_name


@pytest.mark.db
def test_ingredient_panel_returns_product_uom(api_txn, db_cursor, monkeypatch):
    _insert_product(
        db_cursor, sku="B2-BAG", name="B2 Printed Bag",
        product_type="packaging", uom="unit", on_hand_lb=40,
    )
    _insert_product(
        db_cursor, sku="B2-OATS", name="B2 Oats",
        product_type="ingredient", uom="lb", on_hand_lb=100,
    )
    monkeypatch.setattr(main, "_load_dashboard_config", lambda: {
        "ingredient_categories": [{
            "id": "b2_mixed",
            "title": "B2 Mixed",
            "unit": "lb",
            "items": ["B2 Printed Bag", "B2 Oats"],
        }]
    })

    response = main.dashboard_api_ingredients(category=None)
    cat = response["categories"][0]
    by_name = {item["name"]: item for item in cat["items"]}
    assert by_name["B2 Printed Bag"]["uom"] == "unit"
    assert by_name["B2 Printed Bag"]["lots"][0]["uom"] == "unit"
    assert by_name["B2 Oats"]["uom"] == "lb"


def test_floor_unit_count_never_overstates():
    assert main._floor_unit_count(99.9, 10) == 9
    assert main._floor_unit_count(100, 10) == 10
    assert main._floor_unit_count(25.4, 10) == 2
    assert main._floor_unit_count(0, 10) == 0
    assert main._floor_unit_count(50, None) is None


@pytest.mark.db
@pytest.mark.parametrize("sku", ["90003", "90004", "90005"])
def test_admin_coconut_yield_update_preserves_posted_pounds(
    api_txn, db_cursor, monkeypatch, sku
):
    product_id = _insert_product(
        db_cursor, sku=sku, name=f"Batch Coconut Sweetened {sku}",
        product_type="batch", default_batch_lb=360, yield_multiplier=1.11,
        on_hand_lb=4795.2,
    )
    result = main.admin_update_product(
        product_id, main.ProductUpdate(yield_multiplier=1.0), True,
    )
    assert result["changes"] == {"yield_multiplier": 1.0}
    monkeypatch.setattr(main, "_load_dashboard_config", lambda: {"batch_skus": []})
    row = next(r for r in main.dashboard_api_batches()["batches"]
               if r["product_name"] == f"Batch Coconut Sweetened {sku}")
    assert row["made_unit_size_lbs"] == 360
    assert row["batch_count"] == pytest.approx(13.3)
    assert row["on_hand_lbs"] == pytest.approx(4795.2)
    db_cursor.execute("SELECT quantity_lb FROM transaction_lines WHERE product_id = %s",
                      (product_id,))
    assert float(db_cursor.fetchone()["quantity_lb"]) == pytest.approx(4795.2)


@pytest.mark.parametrize("multiplier, expected", [(1.2, 120), (0.8, 80), (None, 100)])
def test_made_unit_retains_legitimate_yield_changes(multiplier, expected):
    assert main._made_unit_size_lbs(100, multiplier) == expected

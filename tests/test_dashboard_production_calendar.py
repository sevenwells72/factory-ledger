"""Production Calendar API contract for made-unit and pack-format rendering."""

from contextlib import contextmanager

import pytest
from psycopg2.extras import RealDictCursor

import main


def _insert_product(cur, *, sku, name, product_type, quantity_lb, transaction_type,
                    default_batch_lb=None, yield_multiplier=1.0,
                    case_size_lb=None, pack_format=None):
    cur.execute(
        """
        INSERT INTO products (
            odoo_code, name, type, active, default_batch_lb,
            yield_multiplier, case_size_lb, pack_format
        )
        VALUES (%s, %s, %s, true, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            sku, name, product_type, default_batch_lb,
            yield_multiplier, case_size_lb, pack_format,
        ),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, f"CALENDAR-{sku}-LOT"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute(
        """
        INSERT INTO transactions (
            type, timestamp, status, occurred_at, business_date, operator_id
        )
        VALUES (
            %s,
            clock_timestamp() AT TIME ZONE 'UTC',
            'posted',
            clock_timestamp(),
            timezone('America/New_York', clock_timestamp())::date,
            'calendar-test'
        )
        RETURNING id
        """,
        (transaction_type,),
    )
    transaction_id = cur.fetchone()["id"]
    cur.execute(
        """
        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
        VALUES (%s, %s, %s, %s)
        """,
        (transaction_id, product_id, lot_id, quantity_lb),
    )


@pytest.mark.db
def test_calendar_response_exposes_pack_format_and_accurate_made_counts(
    _db_connection, db_cursor, monkeypatch
):
    products = [
        dict(sku="CAL-M-C", name="Batch Coconut Calendar Test", product_type="batch",
             quantity_lb=4795.2, transaction_type="make", default_batch_lb=360,
             yield_multiplier=1.11),
        dict(sku="CAL-M-GR", name="Batch Granola Calendar Test", product_type="batch",
             quantity_lb=200, transaction_type="make", default_batch_lb=50),
        dict(sku="CAL-M-GH", name="Batch Graham Calendar Test", product_type="batch",
             quantity_lb=150, transaction_type="make", default_batch_lb=75),
        dict(sku="CAL-P-10", name="Granola Calendar Test 10 LB", product_type="finished",
             quantity_lb=20, transaction_type="pack", case_size_lb=10, pack_format="10lb"),
        dict(sku="CAL-P-25", name="Granola Calendar Test 25 LB", product_type="finished",
             quantity_lb=75, transaction_type="pack", case_size_lb=25, pack_format="25lb"),
        dict(sku="CAL-P-BAG", name="Granola Calendar Test 12x10 OZ", product_type="finished",
             quantity_lb=30, transaction_type="pack", case_size_lb=7.5, pack_format="bagged"),
        dict(sku="CAL-P-CO", name="Coconut Calendar Test 10 LB", product_type="finished",
             quantity_lb=50, transaction_type="pack", case_size_lb=10),
        dict(sku="CAL-P-GH", name="Graham Calendar Test 10 LB", product_type="finished",
             quantity_lb=60, transaction_type="pack", case_size_lb=10),
        dict(sku="CAL-P-BULK", name="Granola Calendar Test Bulk per/lb", product_type="finished",
             quantity_lb=10, transaction_type="pack", case_size_lb=1),
    ]

    for product in products:
        _insert_product(db_cursor, **product)

    @contextmanager
    def _test_transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as api_cur:
            yield api_cur

    monkeypatch.setattr(main, "get_transaction", _test_transaction)
    response = main.dashboard_api_production(days=1, month=None)

    day = next(
        item for item in response["days"]
        if any(row["sku"] == "CAL-M-C" for row in item["batches"])
    )
    batches = {row["sku"]: row for row in day["batches"] if row["sku"].startswith("CAL-")}
    packed = {
        row["sku"]: row
        for row in day["finished_goods"]
        if row["sku"].startswith("CAL-")
    }

    assert batches["CAL-M-C"]["transaction_type"] == "make"
    assert batches["CAL-M-C"]["standard_batch_size_lbs"] == 360
    assert batches["CAL-M-C"]["yield_multiplier"] == 1.11
    assert batches["CAL-M-C"]["made_unit_size_lbs"] == pytest.approx(399.6)
    assert batches["CAL-M-C"]["batch_count"] == 12
    assert batches["CAL-M-GR"]["batch_count"] == 4
    assert batches["CAL-M-GH"]["batch_count"] == 2

    assert packed["CAL-P-10"]["pack_format"] == "10lb"
    assert packed["CAL-P-10"]["unit_count"] == 2
    assert packed["CAL-P-25"]["pack_format"] == "25lb"
    assert packed["CAL-P-25"]["unit_count"] == 3
    assert packed["CAL-P-BAG"]["pack_format"] == "bagged"
    assert packed["CAL-P-BAG"]["unit_count"] == 4
    assert packed["CAL-P-CO"]["pack_format"] is None
    assert packed["CAL-P-CO"]["unit_count"] == 5
    assert packed["CAL-P-GH"]["pack_format"] is None
    assert packed["CAL-P-GH"]["unit_count"] == 6
    assert packed["CAL-P-BULK"]["pack_format"] is None

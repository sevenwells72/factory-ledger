"""Independent contract tests for the authenticated Today So Far production tile."""

from contextlib import contextmanager
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor

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


def _insert_row(cur, *, sku, name, kind, qty, day, status="posted", batch_lb=None,
                yield_multiplier=1.0, case_size=None, pack_format=None):
    cur.execute("""
        INSERT INTO products (odoo_code, name, type, active, default_batch_lb,
                              yield_multiplier, case_size_lb, pack_format)
        VALUES (%s, %s, %s, true, %s, %s, %s, %s) RETURNING id
    """, (sku, name, "batch" if kind == "make" else "finished", batch_lb,
          yield_multiplier, case_size, pack_format))
    product_id = cur.fetchone()["id"]
    cur.execute("INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
                (product_id, f"TODAY-TILE-{sku}"))
    lot_id = cur.fetchone()["id"]
    stamp = datetime.strptime(day + " 12:00:00", "%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO transactions (type, timestamp, status, occurred_at, business_date, operator_id)
        VALUES (%s, %s, %s, %s, %s, 'today-tile-test') RETURNING id
    """, (kind, stamp, status, stamp, day))
    transaction_id = cur.fetchone()["id"]
    cur.execute("""
        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
        VALUES (%s, %s, %s, %s)
    """, (transaction_id, product_id, lot_id, qty))


@pytest.fixture
def tile_call(_db_connection, monkeypatch):
    @contextmanager
    def _transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
    monkeypatch.setattr(main, "get_transaction", _transaction)
    return main.production_today_tile


@pytest.mark.db
def test_today_tile_requires_dashboard_key(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "today_tile_auth")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "today-tile-dashboard-key")
    with TestClient(main.app) as client:
        assert client.get("/production/today-tile").status_code == 401
        assert client.get("/production/today-tile", headers={"X-API-Key": "wrong-key"}).status_code == 403
        assert client.get("/production/today-tile", headers={"X-API-Key": "today-tile-dashboard-key"}).status_code == 200
    _db_connection.rollback()


@pytest.mark.db
def test_today_tile_validates_date_parameter(tile_call):
    with pytest.raises(main.HTTPException, match="date must be YYYY-MM-DD format"):
        tile_call(date="2026-99-99", _=True)


@pytest.mark.db
def test_today_tile_scopes_to_requested_plant_day(db_cursor, tile_call):
    _insert_row(db_cursor, sku="TT-DAY-IN", name="Batch Granola Today", kind="make", qty=100,
                day="2026-08-05", batch_lb=50)
    _insert_row(db_cursor, sku="TT-DAY-OUT", name="Batch Granola Yesterday", kind="make", qty=500,
                day="2026-08-04", batch_lb=50)
    result = tile_call(date="2026-08-05", _=True)
    assert result["date"] == "2026-08-05"
    assert result["made"]["granola_batches"] == pytest.approx(2.0)


@pytest.mark.db
def test_today_tile_coconut_pans_apply_yield_adjustment(db_cursor, tile_call):
    _insert_row(db_cursor, sku="TT-CO", name="Batch Coconut Tile", kind="make", qty=799.2,
                day="2026-08-05", batch_lb=360, yield_multiplier=1.11)
    result = tile_call(date="2026-08-05", _=True)
    assert result["made"]["coconut_pans"] == pytest.approx(2.0)  # not 799.2 / 360


@pytest.mark.db
def test_today_tile_excludes_voided_activity(db_cursor, tile_call):
    _insert_row(db_cursor, sku="TT-POSTED", name="Batch Granola Posted", kind="make", qty=100,
                day="2026-08-05", batch_lb=50)
    _insert_row(db_cursor, sku="TT-VOID", name="Batch Granola Voided", kind="make", qty=500,
                day="2026-08-05", status="voided", batch_lb=50)
    result = tile_call(date="2026-08-05", _=True)
    assert result["made"]["granola_batches"] == pytest.approx(2.0)


@pytest.mark.db
def test_today_tile_splits_pack_formats(db_cursor, tile_call):
    day = "2026-08-05"
    _insert_row(db_cursor, sku="TT-10", name="Granola Tile 10 LB", kind="pack", qty=25,
                day=day, case_size=10, pack_format="10lb")
    _insert_row(db_cursor, sku="TT-25", name="Granola Tile 25 LB", kind="pack", qty=75,
                day=day, case_size=25, pack_format="25lb")
    _insert_row(db_cursor, sku="TT-BAG", name="Granola Tile Retail", kind="pack", qty=15,
                day=day, case_size=7.5, pack_format="bagged")
    result = tile_call(date=day, _=True)
    assert result["packed"]["granola_bulk_10lb_cases"] == pytest.approx(2.5)
    assert result["packed"]["granola_bulk_25lb_cases"] == pytest.approx(3.0)
    assert result["packed"]["granola_retail_cases"] == pytest.approx(2.0)
    assert result["packed"]["granola_retail_bag_count_available"] is False


@pytest.mark.db
def test_today_tile_surfaces_other_products(db_cursor, tile_call):
    day = "2026-08-05"
    _insert_row(db_cursor, sku="TT-OTHER-M", name="Batch Chips Tile", kind="make", qty=80,
                day=day, batch_lb=40)
    _insert_row(db_cursor, sku="TT-OTHER-P", name="Sprinkles Tile", kind="pack", qty=20,
                day=day, case_size=10)
    _insert_row(db_cursor, sku="TT-MISSING-M", name="Batch Granola Missing Definition", kind="make", qty=80,
                day=day)
    _insert_row(db_cursor, sku="TT-MISSING-P", name="Coconut Missing Definition", kind="pack", qty=20,
                day=day)
    result = tile_call(date=day, _=True)
    assert result["made"]["other_batches"] == pytest.approx(2.0)
    assert result["made"]["other_products"][0]["product_name"] == "Batch Chips Tile"
    assert result["packed"]["other_cases"] == pytest.approx(2.0)
    assert {item["product_name"] for item in result["made"]["other_products"]} == {
        "Batch Chips Tile", "Batch Granola Missing Definition"
    }
    assert {item["product_name"] for item in result["packed"]["other_products"]} == {
        "Sprinkles Tile", "Coconut Missing Definition"
    }
    assert any(not item["count_available"] for item in result["packed"]["other_products"])

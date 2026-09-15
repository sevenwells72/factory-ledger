"""055 contracts, using rollback-only local DB and existing ledger test helpers.

Legacy helper predicates are deliberately preserved. Two require finished_good,
which the current catalog constraint forbids; only those historical-fixture tests
relax that constraint inside the test savepoint (never in the migration).
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from tests.test_void_semantics import void_db, _insert_txn, _seed_lot

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
UP = (ROOT / "migrations/055_void_aware_views.sql").read_text()
DOWN = (ROOT / "migrations/down/055_void_aware_views_down.sql").read_text()
VIEWS = (
    "inventory_summary", "lot_balances", "low_stock_alerts", "todays_transactions",
    "production_history", "v_lot_quantities", "v_batch_products_needing_setup",
    "v_products_missing_boms", "v_test_batches_for_review",
)


@pytest.fixture
def migrated(db_cursor, void_db):
    db_cursor.execute(UP)
    return db_cursor


def void(tid):
    result = main.void_transaction(tid, main.VoidRequest(reason="055 regression"), True)
    assert result["success"]


def value(cur, view, column, key, ident):
    assert view in VIEWS
    cur.execute(f'SELECT {column} FROM {view} WHERE {key} = %s', (ident,))
    row = cur.fetchone()
    return None if row is None else row[column]


@pytest.mark.parametrize("view,column,key,empty", [
    ("inventory_summary", "on_hand", "id", 0),
    ("lot_balances", "balance", "id", None),
    ("low_stock_alerts", "on_hand", "id", 0),
    ("todays_transactions", "quantity_lb", "id", None),
    ("production_history", "quantity_made", "id", None),
    ("v_lot_quantities", "quantity_on_hand", "lot_id", 0),
])
def test_make_and_receive_void(migrated, view, column, key, empty):
    cur = migrated
    pid, lid = _seed_lot(cur, product_type="ingredient" if view == "low_stock_alerts" else "batch")
    lot_view = view in {"lot_balances", "v_lot_quantities"}
    receive = _insert_txn(cur, "receive", [(pid, lid, 10)]) if lot_view else None
    tid = _insert_txn(cur, "make", [(pid, lid, 40)])
    ident = lid if lot_view else tid if view in {"todays_transactions", "production_history"} else pid
    assert value(cur, view, column, key, ident) == (50 if lot_view else 40)
    void(tid)
    assert value(cur, view, column, key, ident) == (10 if lot_view else empty)
    if receive:
        void(receive)
        assert value(cur, view, column, key, ident) == empty


@pytest.mark.parametrize("view", VIEWS[6:])
def test_legacy_helper_production_void_and_make_exclusion(migrated, view):
    cur = migrated
    # Historical type is unreachable through today's app/catalog. Preserve it.
    if view != "v_test_batches_for_review":
        cur.execute("ALTER TABLE products DROP CONSTRAINT products_type_check")
    pid, lid = _seed_lot(cur, product_type="batch" if view == "v_test_batches_for_review" else "finished_good")
    cur.execute("UPDATE products SET verification_status='unverified', has_bom=false, bom_status='none', production_context=%s WHERE id=%s",
                ("test_batch" if view == "v_test_batches_for_review" else "standard", pid))
    make = _insert_txn(cur, "make", [(pid, lid, 40)])
    empty = None if view == "v_products_missing_boms" else 0
    assert value(cur, view, "total_produced", "product_id", pid) == empty
    void(make)
    tid = _insert_txn(cur, "production", [(pid, lid, 60)])
    assert value(cur, view, "total_produced", "product_id", pid) == 60
    assert value(cur, view, "batch_count", "product_id", pid) == 1
    void(tid)
    assert value(cur, view, "total_produced", "product_id", pid) == empty
    assert value(cur, view, "batch_count", "product_id", pid) == empty
    if view == "v_test_batches_for_review":
        assert value(cur, view, "recommendation", "product_id", pid) == "No batches yet"


def test_inventory_summary_amended_quantity(migrated):
    cur = migrated
    pid, lid = _seed_lot(cur)
    tid = _insert_txn(cur, "make", [(pid, lid, 40)])
    assert value(cur, "inventory_summary", "on_hand", "id", pid) == 40
    cur.execute("SELECT id FROM transaction_lines WHERE transaction_id=%s", (tid,))
    main._append_transaction_line_correction(cur, cur.fetchone()["id"], {"quantity_lb": 25}, "055 amended quantity", "test")
    assert value(cur, "inventory_summary", "on_hand", "id", pid) == 25
    cur.execute("SELECT quantity_lb FROM transaction_lines WHERE transaction_id=%s", (tid,))
    assert cur.fetchone()["quantity_lb"] == 40
    void(tid)
    assert value(cur, "inventory_summary", "on_hand", "id", pid) == 0


def shape(cur):
    cur.execute("""SELECT table_name,column_name,ordinal_position,data_type,udt_name,numeric_precision,numeric_scale
                   FROM information_schema.columns WHERE table_schema='public'
                   AND table_name = ANY(%s) ORDER BY table_name,ordinal_position""", (list(VIEWS),))
    return [dict(r) for r in cur.fetchall()]


def definitions(cur):
    return {v: _definition(cur, v) for v in VIEWS}


def _definition(cur, view):
    cur.execute("SELECT pg_get_viewdef(%s::regclass) AS definition", (view,))
    return cur.fetchone()["definition"]


def test_reapply_shapes_and_exact_rollback(db_cursor, void_db):
    cur = db_cursor
    # Works both before and after the production schema is refreshed.
    cur.execute(DOWN.replace("BEGIN;", "", 1).rsplit("COMMIT;", 1)[0])
    previous, columns = definitions(cur), shape(cur)
    cur.execute(UP)
    once = definitions(cur)
    cur.execute(UP)
    assert definitions(cur) == once
    assert shape(cur) == columns
    cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name='055_void_aware_views'")
    assert cur.fetchone()["n"] == 1
    cur.execute(DOWN.replace("BEGIN;", "", 1).rsplit("COMMIT;", 1)[0])
    assert definitions(cur) == previous
    assert shape(cur) == columns
    cur.execute("SELECT 1 FROM migration_markers WHERE name='055_void_aware_views'")
    assert cur.fetchone() is None


def test_today_uses_et_business_date(migrated):
    cur = migrated
    pid, lid = _seed_lot(cur)
    # Explicitly disagree with legacy timestamp's date; no session timezone changes.
    cur.execute("""INSERT INTO transactions(type,timestamp,occurred_at,business_date)
                   VALUES ('make', '2000-01-01', now(),
                           (now() AT TIME ZONE 'America/New_York')::date) RETURNING id""")
    tid = cur.fetchone()["id"]
    cur.execute("INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,7)", (tid,pid,lid))
    assert value(cur, "todays_transactions", "quantity_lb", "id", tid) == 7
    void(tid)
    assert value(cur, "todays_transactions", "quantity_lb", "id", tid) is None


@pytest.mark.parametrize("route,view,keys", [
    ("inventory", "inventory_summary", {"id","name","type","on_hand"}),
    ("low-stock", "low_stock_alerts", {"id","name","on_hand"}),
    ("today", "todays_transactions", {"id","type","timestamp","notes","product","lot_code","quantity_lb"}),
    ("lots", "lot_balances", {"id","lot_code","created_at","product","type","balance"}),
    ("production", "production_history", {"id","timestamp","product","lot_code","quantity_made"}),
])
def test_legacy_endpoint_json_keys(migrated, monkeypatch, route, view, keys):
    pid, lid = _seed_lot(migrated, product_type="ingredient")
    _insert_txn(migrated, "make", [(pid,lid,40)])
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "test-dashboard-key")
    with TestClient(main.app) as client:
        response = client.get('/dashboard/'+route, headers={"X-API-Key":"test-dashboard-key"})
    assert response.status_code == 200, response.text
    rows = response.json()
    assert rows and all(set(row) == keys for row in rows)

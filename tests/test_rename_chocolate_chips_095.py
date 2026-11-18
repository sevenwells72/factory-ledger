"""Migration 095 — "Chocolate Chips – Real – N CT" loses the word "Real" (Michael, 2026-11-18).

Three ingredient rows, products.name only: 72 (4,000 CT), 73 (1,000 CT), 74 (10,000 CT).
Ids, odoo codes and every other column stay; no history table is read or written.

dashboard/dashboard_config.json lists the 4,000 CT and 1,000 CT chips by exact name for
the Inventory "Granola General Ingredients" panel, so the config change ships in the same
PR and the last tests here hold the two together: the config names and the names the
migration writes must be the same strings.

Every test seeds the production BEFORE state recorded read-only on 2026-11-18 (ids, codes
and names verbatim) inside the suite's rolled-back transaction, then runs the SQL files
exactly as an operator would.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import re

import psycopg2
import pytest
from psycopg2.extras import RealDictCursor

import main

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "migrations" / "095_rename_chocolate_chips.sql"
DOWN = ROOT / "migrations" / "down" / "095_rename_chocolate_chips_down.sql"
CONFIG = ROOT / "dashboard" / "dashboard_config.json"
MARKER = "095_rename_chocolate_chips"

# (id, odoo_code, BEFORE name, AFTER name) — production, read-only 2026-11-18.
CHIPS = [
    (72, "25010", "Chocolate Chips – Real – 4,000 CT", "Chocolate Chips – 4,000 CT"),
    (73, "25011", "Chocolate Chips – Real – 1,000 CT", "Chocolate Chips – 1,000 CT"),
    (74, "25012", "Chocolate Chips – Real – 10,000 CT", "Chocolate Chips – 10,000 CT"),
]
BEFORE = {pid: old for pid, _code, old, _new in CHIPS}
AFTER = {pid: new for pid, _code, _old, new in CHIPS}
# Neighbours that must keep their names (production ids and names; the codes of 6 and 53
# are stand-ins, the migration never reads them).
NEIGHBOURS = [
    (6, "11006", "Chocolate Chips – Sugar Free", "ingredient"),
    (53, "11053", "White Chocolate Chips", "ingredient"),
    (181, "25013", "Real Chocolate Chips 10000ct 50 LB", "finished"),
]
BATCH_CC9 = 108
HISTORY_TABLES = ("transactions", "transaction_lines", "lots", "ingredient_lot_consumption",
                  "transaction_substitutions", "lot_events", "write_tickets", "exceptions")

pytestmark = pytest.mark.db


def seed_before_state(cur):
    """Production rows as recorded, verbatim ids. The suite's transaction rolls them back."""
    for pid, code, old, _new in CHIPS:
        cur.execute("INSERT INTO products (id, odoo_code, name, type, uom, active) "
                    "VALUES (%s, %s, %s, 'ingredient', 'lb', true)", (pid, code, old))
    for pid, code, name, ptype in NEIGHBOURS:
        cur.execute("INSERT INTO products (id, odoo_code, name, type, uom, active) "
                    "VALUES (%s, %s, %s, %s, 'lb', true)", (pid, code, name, ptype))


def run_sql(cur, path):
    cur.execute(path.read_text())


def names(cur):
    cur.execute("SELECT id, name FROM products WHERE id IN (72, 73, 74)")
    return {r["id"]: r["name"] for r in cur.fetchall()}


def rows(cur, ids):
    """Every column of the given product rows except the name."""
    cur.execute("SELECT to_jsonb(p) - 'name' AS row FROM products p WHERE id = ANY(%s) ORDER BY id", (list(ids),))
    return [r["row"] for r in cur.fetchall()]


def marker(cur):
    cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name=%s", (MARKER,))
    return cur.fetchone()["n"]


def _refused(cur, path, fragment):
    cur.execute("SAVEPOINT refusal")
    with pytest.raises(psycopg2.errors.RaiseException) as err:
        run_sql(cur, path)
    cur.execute("ROLLBACK TO SAVEPOINT refusal")
    assert fragment in str(err.value)


def test_095_renames_the_three_chips_and_nothing_else(db_cursor):
    seed_before_state(db_cursor)
    everything_but_names = rows(db_cursor, [6, 53, 72, 73, 74, 181])
    cur = db_cursor
    cur.execute("SELECT id, name FROM products WHERE id IN (6, 53, 181) ORDER BY id")
    neighbours = [(r["id"], r["name"]) for r in cur.fetchall()]

    run_sql(db_cursor, UP)

    assert names(db_cursor) == AFTER
    assert rows(db_cursor, [6, 53, 72, 73, 74, 181]) == everything_but_names, "ids, codes and every other column stay"
    cur.execute("SELECT id, name FROM products WHERE id IN (6, 53, 181) ORDER BY id")
    assert [(r["id"], r["name"]) for r in cur.fetchall()] == neighbours
    assert marker(db_cursor) == 1


def test_095_is_rerunnable(db_cursor):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    run_sql(db_cursor, UP)
    assert names(db_cursor) == AFTER
    assert marker(db_cursor) == 1


def test_095_finishes_a_half_applied_catalog(db_cursor):
    """One row already at its target (a hand rename, or a partial earlier run) is accepted."""
    seed_before_state(db_cursor)
    db_cursor.execute("UPDATE products SET name=%s WHERE id=73", (AFTER[73],))
    run_sql(db_cursor, UP)
    assert names(db_cursor) == AFTER


def test_095_down_restores_recorded_names_and_up_reapplies(db_cursor):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    run_sql(db_cursor, DOWN)
    assert names(db_cursor) == BEFORE
    assert marker(db_cursor) == 0
    # A second down finds no marker and refuses rather than rewriting anything.
    _refused(db_cursor, DOWN, "095 down refused")
    assert names(db_cursor) == BEFORE
    run_sql(db_cursor, UP)
    assert names(db_cursor) == AFTER


def test_095_refuses_a_catalog_it_was_not_written_for(db_cursor):
    seed_before_state(db_cursor)

    # A different product sits on id 73 (another code).
    db_cursor.execute("SAVEPOINT s")
    db_cursor.execute("UPDATE products SET odoo_code='99973' WHERE id=73")
    _refused(db_cursor, UP, "095 refused: product 73 is not ingredient 25011")
    db_cursor.execute("ROLLBACK TO SAVEPOINT s")

    # The right product, but someone already gave it a third name.
    db_cursor.execute("SAVEPOINT s")
    db_cursor.execute("UPDATE products SET name='Chocolate Chips – Dark – 4,000 CT' WHERE id=72")
    _refused(db_cursor, UP, "neither the recorded name nor the target")
    db_cursor.execute("ROLLBACK TO SAVEPOINT s")

    # One of the three rows is missing altogether.
    db_cursor.execute("SAVEPOINT s")
    db_cursor.execute("DELETE FROM products WHERE id=74")
    _refused(db_cursor, UP, "095 refused: product 74 is not ingredient 25012")
    db_cursor.execute("ROLLBACK TO SAVEPOINT s")

    # The target name already belongs to another product.
    db_cursor.execute("SAVEPOINT s")
    db_cursor.execute("INSERT INTO products (odoo_code, name, type, uom, active) "
                      "VALUES ('99999', %s, 'ingredient', 'lb', true)", (AFTER[72],))
    _refused(db_cursor, UP, "is already used by another product")
    db_cursor.execute("ROLLBACK TO SAVEPOINT s")

    # Every refusal left all three names alone and wrote no marker.
    assert names(db_cursor) == BEFORE
    assert marker(db_cursor) == 0


def test_095_down_refuses_after_a_later_rename(db_cursor):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    db_cursor.execute("UPDATE products SET name='Chocolate Chips – 4,000 CT (Semi-Sweet)' WHERE id=72")
    _refused(db_cursor, DOWN, "do not all hold exactly the names 095 wrote")
    assert names(db_cursor)[72] == "Chocolate Chips – 4,000 CT (Semi-Sweet)"
    assert marker(db_cursor) == 1


def _sql_without_comments(path):
    text = re.sub(r"--[^\n]*", "", path.read_text())
    return re.sub(r"'(?:[^']|'')*'", "''", text)  # string literals (names, messages) do not count


def test_095_up_and_down_touch_only_the_catalog():
    for path in (UP, DOWN):
        sql = _sql_without_comments(path)
        for table in HISTORY_TABLES:
            assert not re.search(rf"\b{table}\b", sql), f"{path.name} references {table}"
        referenced = set(re.findall(r"\b(?:FROM|UPDATE|INSERT\s+INTO|JOIN)\s+([a-z_]+)", sql, re.I))
        assert referenced <= {"products", "migration_markers"}, (path.name, referenced)
        assert not re.search(r"\b(ALTER|CREATE|DROP|TRUNCATE)\b", sql, re.I), f"{path.name} carries DDL"


def test_095_leaves_recipes_lots_and_posted_history_untouched(db_cursor):
    """Recipes and stock link by product id: the same rows, byte for byte, before and after."""
    seed_before_state(db_cursor)
    cur = db_cursor
    cur.execute("INSERT INTO products (id, odoo_code, name, type, uom, active, default_batch_lb) "
                "VALUES (%s, '90001', 'Batch Classic Chocolate Chip Granola #9', 'batch', 'lb', true, 348)", (BATCH_CC9,))
    cur.execute("INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) VALUES (%s, 72, 25)", (BATCH_CC9,))
    cur.execute("INSERT INTO lots (product_id, lot_code, entry_source) VALUES (72, '26-11-16-ABAK-001', 'received') RETURNING id")
    lot = cur.fetchone()["id"]
    cur.execute("INSERT INTO transactions (type, timestamp, notes, occurred_at, operator_id) "
                "VALUES ('receive', now() - interval '2 days', %s, now() - interval '2 days', 'Arturo') RETURNING id",
                ("Received Chocolate Chips – Real – 4,000 CT",))
    txn = cur.fetchone()["id"]
    cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, 72, %s, 2000)", (txn, lot))

    def snapshot():
        snap = {}
        for table in ("transactions", "transaction_lines", "lots", "batch_formulas"):
            cur.execute(f"SELECT to_jsonb(t) AS row FROM {table} t ORDER BY 1")
            snap[table] = sorted(str(r["row"]) for r in cur.fetchall())
        return snap

    before = snapshot()
    run_sql(cur, UP)
    assert snapshot() == before
    # The old name written into a past transaction's note is history and stays as written.
    cur.execute("SELECT notes FROM transactions WHERE id=%s", (txn,))
    assert cur.fetchone()["notes"] == "Received Chocolate Chips – Real – 4,000 CT"
    run_sql(cur, DOWN)
    assert snapshot() == before


@pytest.mark.parametrize("typed, expected", [
    ("Chocolate Chips – 4,000 CT", 72),
    ("Chocolate Chips – 1,000 CT", 73),
    ("Chocolate Chips – 10,000 CT", 74),
])
def test_new_names_resolve_exactly_after_095(db_cursor, typed, expected):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    assert main.resolve_product_id(db_cursor, typed) == (expected, typed)


@pytest.mark.parametrize("typed, expected", [
    ("Chocolate Chips – Real – 4,000 CT", 72),
    ("Chocolate Chips Real 4,000 CT", 72),
    ("Chocolate Chips – Real – 1,000 CT", 73),
    ("Chocolate Chips – Real – 10,000 CT", 74),
])
def test_old_wording_never_resolves_to_a_different_product_after_095(db_cursor, typed, expected):
    """Someone who still types the old name gets the same product or a question, never
    a silent match on another chip."""
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    try:
        product_id, _name = main.resolve_product_id(db_cursor, typed)
    except main.HTTPException as refused:
        assert refused.status_code in (404, 409)
        detail = refused.detail
        if refused.status_code == 409:
            assert expected in [s["product_id"] for s in detail["suggestions"]]
    else:
        assert product_id == expected


# ── dashboard/dashboard_config.json travels with the migration ────────────────

def _granola_ingredient_names():
    config = json.loads(CONFIG.read_text())
    (category,) = [c for c in config["ingredient_categories"] if c["id"] == "granola_ingredients"]
    return category["items"]


def test_dashboard_config_lists_the_names_095_writes():
    items = _granola_ingredient_names()
    assert AFTER[72] in items and AFTER[73] in items
    assert not [name for name in items if "Real" in name and name.startswith("Chocolate Chips")]
    everything = CONFIG.read_text()
    for old in BEFORE.values():
        assert json.dumps(old)[1:-1] not in everything and old not in everything, f"config still names {old!r}"


@pytest.fixture
def api_txn(_db_connection, monkeypatch):
    @contextmanager
    def _test_transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as api_cur:
            yield api_cur

    monkeypatch.setattr(main, "get_transaction", _test_transaction)
    return _db_connection


def _chips_panel(monkeypatch):
    """The Inventory ingredients panel, driven by the repository's real config entries for chips."""
    listed = [name for name in _granola_ingredient_names() if name.startswith("Chocolate Chips")]
    monkeypatch.setattr(main, "_load_dashboard_config", lambda: {"ingredient_categories": [
        {"id": "granola_ingredients", "title": "Granola General Ingredients", "unit": "lb", "items": listed}]})
    response = main.dashboard_api_ingredients(category=None, include_archived=False)
    assert isinstance(response, dict), response
    (panel,) = response["categories"]
    return panel


def test_ingredients_panel_finds_both_chips_once_095_is_applied(api_txn, db_cursor, monkeypatch):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    panel = _chips_panel(monkeypatch)
    assert panel["missing_skus"] == []
    assert {item["product_id"]: item["name"] for item in panel["items"]} == {
        73: AFTER[73], 72: AFTER[72], 6: "Chocolate Chips – Sugar Free"}


def test_ingredients_panel_reports_the_chips_missing_until_095_is_applied(api_txn, db_cursor, monkeypatch):
    """Why the deploy and the migration go out back to back: config ahead of the catalog
    shows the two chips as Missing SKUs (never a wrong balance, never another product)."""
    seed_before_state(db_cursor)
    panel = _chips_panel(monkeypatch)
    assert sorted(panel["missing_skus"]) == sorted([AFTER[72], AFTER[73]])
    assert [item["product_id"] for item in panel["items"]] == [6]

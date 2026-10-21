"""Migration 078 — recipe corrections approved by Michael 2026-10-13.

179 Granola Fruit Nut Batch: make-recipe 20 + 1.25×4 = 25 lb → 276.3 + 13×4 = 328.3 lb,
default_batch_lb 384.52 → 328.3, the five planner rows (product_bom) rewritten in place to
the same formula, legacy yields 15/38 → 13/32. 284 (SS Classic Chocolate Chip #9, kosher):
chips line 73 (1,000 CT) → 72 (4,000 CT), still 25 lb, and never excluded from inventory.
Nothing historical changes, and neither SQL file reads a history table.

Every test seeds the production BEFORE state recorded read-only on 2026-10-13 (ids, codes
and quantities verbatim) inside the suite's rolled-back transaction, then runs the SQL
files exactly as an operator would. The make/pack tests drive the real endpoints through
the TestClient so the deductions and the output are the code's, not the test's.
"""

from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import re

import psycopg2
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed", allow_module_level=True)

import main

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "migrations" / "078_recipe_corrections.sql"
DOWN = ROOT / "migrations" / "down" / "078_recipe_corrections_down.sql"

FRUIT_NUT, FRUIT_NUT_FG, CLASSIC9, KOSHER_CC = 179, 142, 107, 284
ALMONDS, WALNUTS, RAISINS, CRANBERRIES = 3, 49, 36, 17
CHIPS_4000, CHIPS_1000 = 72, 73
INCLUSIONS = (ALMONDS, WALNUTS, RAISINS, CRANBERRIES)

# Production catalog rows the migration guards on (id, odoo_code, name, type).
PRODUCTS = [
    (CLASSIC9, "90002", "Batch Classic Granola #9", "batch", 323),
    (FRUIT_NUT, "90008", "Granola Fruit Nut Batch", "batch", Decimal("384.52")),
    (KOSHER_CC, "90026", "Batch SS Classic Chocolate Chip Granola #9 (Kosher Ignition)", "batch", 348),
    (FRUIT_NUT_FG, "70061", "Granola Fruit Nut 25 LB", "finished", None),
    (ALMONDS, "11003", "Almonds – Sliced", "ingredient", None),
    (CRANBERRIES, "11017", "Cranberries Dried", "ingredient", None),
    (RAISINS, "11036", "Raisins", "ingredient", None),
    (WALNUTS, "11049", "Walnuts", "ingredient", None),
    (CHIPS_4000, "25010", "Chocolate Chips – Real – 4,000 CT", "ingredient", None),
    (CHIPS_1000, "25011", "Chocolate Chips – Real – 1,000 CT", "ingredient", None),
    (13, "11013", "Coconut Macaroon Desiccated", "ingredient", None),
    (18, "11018", "Flavor – Almond", "ingredient", None),
    (30, "11030", "Honey", "ingredient", None),
    (32, "11032", "Oats", "ingredient", None),
    (34, "11034", "Oil – Canola", "ingredient", None),
    (39, "11039", "Salt", "ingredient", None),
    (44, "11044", "Sugar – Light Brown", "ingredient", None),
]
# batch_formulas BEFORE (id, product_id, ingredient_product_id, quantity_lb) — production 2026-10-13.
FORMULAS_BEFORE = [
    (179, FRUIT_NUT, CLASSIC9, 20), (180, FRUIT_NUT, WALNUTS, 1.25), (181, FRUIT_NUT, ALMONDS, 1.25),
    (182, FRUIT_NUT, RAISINS, 1.25), (183, FRUIT_NUT, CRANBERRIES, 1.25),
    (191, KOSHER_CC, 13, 12.5), (192, KOSHER_CC, 18, 1.8), (193, KOSHER_CC, 30, 1.6), (194, KOSHER_CC, 32, 225),
    (195, KOSHER_CC, 34, 30), (196, KOSHER_CC, 39, 1.7), (197, KOSHER_CC, 44, 50), (198, KOSHER_CC, CHIPS_1000, 25),
]
# product_bom BEFORE (id, finished_product_id, component_product_id, quantity) — production 2026-10-13.
PLANNER_BEFORE = [
    (116, FRUIT_NUT_FG, FRUIT_NUT, 25),
    (117, FRUIT_NUT, CLASSIC9, 323), (118, FRUIT_NUT, ALMONDS, 15.38), (119, FRUIT_NUT, WALNUTS, 15.38),
    (120, FRUIT_NUT, RAISINS, 15.38), (121, FRUIT_NUT, CRANBERRIES, 15.38),
]
FRUIT_NUT_AFTER = {CLASSIC9: Decimal("276.3"), **{i: Decimal("13") for i in INCLUSIONS}}
FRUIT_NUT_BEFORE_FORMULA = {CLASSIC9: Decimal("20"), **{i: Decimal("1.25") for i in INCLUSIONS}}
FRUIT_NUT_BEFORE_PLANNER = {CLASSIC9: Decimal("323"), **{i: Decimal("15.38") for i in INCLUSIONS}}


def seed_before_state(cur):
    """Production rows as recorded, verbatim ids. The suite's transaction rolls them back."""
    for pid, code, name, ptype, batch_lb in PRODUCTS:
        cur.execute(
            "INSERT INTO products (id, odoo_code, name, type, uom, active, default_batch_lb, yield_multiplier, "
            "case_size_lb, pack_format, parent_batch_product_id, yield_25lb_cases, yield_10lb_cases,is_service) "
            "VALUES (%s, %s, %s, %s, 'lb', true, %s, 1, %s, %s, %s, %s, %s,false)",
            (pid, code, name, ptype, batch_lb,
             25 if pid == FRUIT_NUT_FG else None, "25lb" if pid == FRUIT_NUT_FG else None,
             FRUIT_NUT if pid == FRUIT_NUT_FG else None,
             15 if pid == FRUIT_NUT else (13 if pid == CLASSIC9 else None),
             38 if pid == FRUIT_NUT else (32 if pid == CLASSIC9 else None)),
        )
    for row in FORMULAS_BEFORE:
        cur.execute("INSERT INTO batch_formulas (id, product_id, ingredient_product_id, quantity_lb) VALUES (%s, %s, %s, %s)", row)
    for row in PLANNER_BEFORE:
        cur.execute("INSERT INTO product_bom (id, finished_product_id, component_product_id, quantity, uom) VALUES (%s, %s, %s, %s, 'lb')", row)


def run_sql(cur, path):
    cur.execute(path.read_text())


def formula(cur, product_id):
    cur.execute("SELECT id, ingredient_product_id, quantity_lb, COALESCE(exclude_from_inventory,false) AS excl "
                "FROM batch_formulas WHERE product_id=%s ORDER BY id", (product_id,))
    return [dict(r) for r in cur.fetchall()]


def planner(cur, finished_id):
    cur.execute("SELECT id, component_product_id, quantity, uom FROM product_bom WHERE finished_product_id=%s ORDER BY id", (finished_id,))
    return [dict(r) for r in cur.fetchall()]


def batch_lb(cur, product_id):
    cur.execute("SELECT default_batch_lb FROM products WHERE id=%s", (product_id,))
    return cur.fetchone()["default_batch_lb"]


def yields(cur):
    cur.execute("SELECT yield_25lb_cases, yield_10lb_cases FROM products WHERE id=%s", (FRUIT_NUT,))
    row = cur.fetchone()
    return row["yield_25lb_cases"], row["yield_10lb_cases"]


def marker(cur):
    cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name='078_recipe_corrections'")
    return cur.fetchone()["n"]


def assert_after_state(cur):
    rows = formula(cur, FRUIT_NUT)
    assert [r["id"] for r in rows] == [179, 180, 181, 182, 183], "ids must survive — rows rewritten, not replaced"
    assert {r["ingredient_product_id"]: r["quantity_lb"] for r in rows} == FRUIT_NUT_AFTER
    assert not any(r["excl"] for r in rows)
    assert sum(r["quantity_lb"] for r in rows) == Decimal("328.3")
    assert batch_lb(cur, FRUIT_NUT) == Decimal("328.3")
    assert yields(cur) == (13, 32)
    plan = planner(cur, FRUIT_NUT)
    assert [r["id"] for r in plan] == [117, 118, 119, 120, 121], "planner rows rewritten in place, never deleted"
    assert {r["component_product_id"]: r["quantity"] for r in plan} == FRUIT_NUT_AFTER
    assert all(r["uom"] == "lb" for r in plan)
    kosher = formula(cur, KOSHER_CC)
    assert len(kosher) == 8 and [r["id"] for r in kosher] == [191, 192, 193, 194, 195, 196, 197, 198]
    chips = [r for r in kosher if r["id"] == 198][0]
    assert (chips["ingredient_product_id"], chips["quantity_lb"], chips["excl"]) == (CHIPS_4000, Decimal("25"), False)
    assert not any(r["ingredient_product_id"] == CHIPS_1000 for r in kosher)
    assert sum(r["quantity_lb"] for r in kosher) == Decimal("347.6")
    # Untouched on purpose: the finished-good pack path and 284's other lines.
    assert planner(cur, FRUIT_NUT_FG) == [{"id": 116, "component_product_id": FRUIT_NUT, "quantity": Decimal("25"), "uom": "lb"}]
    cur.execute("SELECT case_size_lb, parent_batch_product_id FROM products WHERE id=%s", (FRUIT_NUT_FG,))
    assert dict(cur.fetchone()) == {"case_size_lb": Decimal("25"), "parent_batch_product_id": FRUIT_NUT}
    assert batch_lb(cur, KOSHER_CC) == Decimal("348")
    assert marker(cur) == 1


def assert_before_state(cur):
    assert {r["ingredient_product_id"]: r["quantity_lb"] for r in formula(cur, FRUIT_NUT)} == FRUIT_NUT_BEFORE_FORMULA
    assert batch_lb(cur, FRUIT_NUT) == Decimal("384.52")
    assert yields(cur) == (15, 38)
    assert {r["component_product_id"]: r["quantity"] for r in planner(cur, FRUIT_NUT)} == FRUIT_NUT_BEFORE_PLANNER
    chips = [r for r in formula(cur, KOSHER_CC) if r["id"] == 198][0]
    assert (chips["ingredient_product_id"], chips["quantity_lb"], chips["excl"]) == (CHIPS_1000, Decimal("25"), False)
    assert marker(cur) == 0


# ─────────────────────────── migration file ───────────────────────────

def test_078_rewrites_fruit_nut_recipe_batch_size_planner_rows_and_284_chips(db_cursor):
    seed_before_state(db_cursor)
    assert_before_state(db_cursor)
    run_sql(db_cursor, UP)
    assert_after_state(db_cursor)


def test_078_is_rerunnable(db_cursor):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    run_sql(db_cursor, UP)
    assert_after_state(db_cursor)
    assert marker(db_cursor) == 1


def test_078_down_restores_recorded_before_values_and_up_reapplies(db_cursor):
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    run_sql(db_cursor, DOWN)
    assert_before_state(db_cursor)
    assert [r["id"] for r in planner(db_cursor, FRUIT_NUT)] == [117, 118, 119, 120, 121]
    # A second down finds before-values, not 078's, and refuses rather than rewriting them.
    _refused(db_cursor, DOWN, "078 down refused")
    assert_before_state(db_cursor)
    run_sql(db_cursor, UP)
    assert_after_state(db_cursor)


def _refused(cur, path, fragment):
    cur.execute("SAVEPOINT refusal")
    with pytest.raises(psycopg2.errors.RaiseException) as err:
        run_sql(cur, path)
    cur.execute("ROLLBACK TO SAVEPOINT refusal")
    assert fragment in str(err.value)


def test_078_refuses_a_catalog_it_was_not_written_for(db_cursor):
    # Fresh schema-only database: product 179 does not exist.
    _refused(db_cursor, UP, "078 refused: product 179")
    seed_before_state(db_cursor)
    # A Fruit Nut line that is neither the recorded before-value nor the target.
    db_cursor.execute("UPDATE batch_formulas SET quantity_lb=99 WHERE id=179")
    _refused(db_cursor, UP, "neither the recorded before-value nor the target")
    db_cursor.execute("UPDATE batch_formulas SET quantity_lb=20 WHERE id=179")
    # A sixth make-recipe line nobody approved.
    db_cursor.execute("INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) VALUES (%s, 32, 1)", (FRUIT_NUT,))
    _refused(db_cursor, UP, "not the recorded five-line recipe")
    db_cursor.execute("DELETE FROM batch_formulas WHERE product_id=%s AND ingredient_product_id=32", (FRUIT_NUT,))
    # Planner rows missing (as on a database that never had them).
    db_cursor.execute("DELETE FROM product_bom WHERE finished_product_id=%s", (FRUIT_NUT,))
    _refused(db_cursor, UP, "not the recorded five planner rows")
    for row in PLANNER_BEFORE[1:]:
        db_cursor.execute("INSERT INTO product_bom (id, finished_product_id, component_product_id, quantity, uom) VALUES (%s, %s, %s, %s, 'lb')", row)
    # 284 with both chip sizes, or a chips line that is not 25 lb.
    db_cursor.execute("INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) VALUES (%s, %s, 25)", (KOSHER_CC, CHIPS_4000))
    _refused(db_cursor, UP, "exactly one chips line")
    db_cursor.execute("DELETE FROM batch_formulas WHERE product_id=%s AND ingredient_product_id=%s", (KOSHER_CC, CHIPS_4000))
    db_cursor.execute("UPDATE batch_formulas SET quantity_lb=30 WHERE id=198")
    _refused(db_cursor, UP, "chips line is not 25 lb")
    db_cursor.execute("UPDATE batch_formulas SET quantity_lb=25 WHERE id=198")
    # Nothing was changed by the refused runs.
    assert_before_state(db_cursor)
    run_sql(db_cursor, UP)
    assert_after_state(db_cursor)


def test_078_refuses_an_excluded_284_chips_line(db_cursor):
    """Review fix 1: an excluded chips line would let 284 post 348 lb with zero chips deducted."""
    seed_before_state(db_cursor)
    # Before-state shape (73) flagged excluded: refused, nothing changes.
    db_cursor.execute("UPDATE batch_formulas SET exclude_from_inventory=true WHERE id=198")
    _refused(db_cursor, UP, "284 chips line is excluded from inventory")
    db_cursor.execute("UPDATE batch_formulas SET exclude_from_inventory=false WHERE id=198")
    assert_before_state(db_cursor)
    run_sql(db_cursor, UP)
    # After-state shape (72) flagged excluded on a rerun: refused as well.
    db_cursor.execute("UPDATE batch_formulas SET exclude_from_inventory=true WHERE id=198")
    _refused(db_cursor, UP, "284 chips line is excluded from inventory")
    db_cursor.execute("UPDATE batch_formulas SET exclude_from_inventory=NULL WHERE id=198")
    run_sql(db_cursor, UP)  # NULL counts as deducted, like the make path's COALESCE
    # Other products' exclusions (e.g. water lines) are not 078's business.
    db_cursor.execute("UPDATE batch_formulas SET exclude_from_inventory=true WHERE id=191")
    run_sql(db_cursor, UP)


def test_078_sets_fruit_nut_yields_13_32_and_refuses_unrecorded_yields(db_cursor):
    """Review fix 3: 179 yield_25lb_cases / yield_10lb_cases 15/38 → 13/32, guarded."""
    seed_before_state(db_cursor)
    for col, bad in (("yield_25lb_cases", 14), ("yield_10lb_cases", 33)):
        db_cursor.execute(f"SELECT {col} AS v FROM products WHERE id=%s", (FRUIT_NUT,))
        good = db_cursor.fetchone()["v"]
        db_cursor.execute(f"UPDATE products SET {col}=%s WHERE id=%s", (bad, FRUIT_NUT))
        _refused(db_cursor, UP, f"{col} for 179 is neither")
        db_cursor.execute(f"UPDATE products SET {col}=%s WHERE id=%s", (good, FRUIT_NUT))
    assert_before_state(db_cursor)
    run_sql(db_cursor, UP)
    assert yields(db_cursor) == (13, 32)
    run_sql(db_cursor, DOWN)
    assert yields(db_cursor) == (15, 38)


# Each tamper moves one row 078 wrote off its 078 value (column change, undo).
DOWN_TAMPERS = [
    ("UPDATE batch_formulas SET quantity_lb=280 WHERE id=179", "UPDATE batch_formulas SET quantity_lb=276.3 WHERE id=179", "batch_formulas for 179"),
    ("UPDATE batch_formulas SET quantity_lb=12 WHERE id=182", "UPDATE batch_formulas SET quantity_lb=13 WHERE id=182", "batch_formulas for 179"),
    ("UPDATE batch_formulas SET exclude_from_inventory=true WHERE id=180", "UPDATE batch_formulas SET exclude_from_inventory=false WHERE id=180", "batch_formulas for 179"),
    ("UPDATE products SET default_batch_lb=330 WHERE id=179", "UPDATE products SET default_batch_lb=328.3 WHERE id=179", "batch size / yields"),
    ("UPDATE products SET yield_25lb_cases=14 WHERE id=179", "UPDATE products SET yield_25lb_cases=13 WHERE id=179", "batch size / yields"),
    ("UPDATE products SET yield_10lb_cases=NULL WHERE id=179", "UPDATE products SET yield_10lb_cases=32 WHERE id=179", "batch size / yields"),
    ("UPDATE product_bom SET quantity=277 WHERE id=117", "UPDATE product_bom SET quantity=276.3 WHERE id=117", "product_bom for 179"),
    ("UPDATE product_bom SET quantity=15.38 WHERE id=121", "UPDATE product_bom SET quantity=13 WHERE id=121", "product_bom for 179"),
    ("UPDATE batch_formulas SET exclude_from_inventory=true WHERE id=198", "UPDATE batch_formulas SET exclude_from_inventory=false WHERE id=198", "284 chips line"),
    ("UPDATE batch_formulas SET quantity_lb=30 WHERE id=198", "UPDATE batch_formulas SET quantity_lb=25 WHERE id=198", "284 chips line"),
    ("UPDATE batch_formulas SET ingredient_product_id=73 WHERE id=198", "UPDATE batch_formulas SET ingredient_product_id=72 WHERE id=198", "284 chips line"),
    ("DELETE FROM migration_markers WHERE name='078_recipe_corrections'", "INSERT INTO migration_markers (name) VALUES ('078_recipe_corrections')", "marker"),
]


def test_078_down_refuses_unless_every_row_holds_exactly_the_078_value(db_cursor):
    """Review fix 2: the down file never blind-overwrites a value 078 did not write."""
    seed_before_state(db_cursor)
    run_sql(db_cursor, UP)
    for tamper, undo, fragment in DOWN_TAMPERS:
        db_cursor.execute(tamper)
        _refused(db_cursor, DOWN, "078 down refused")
        _refused(db_cursor, DOWN, fragment)
        db_cursor.execute(undo)
    assert_after_state(db_cursor)  # the refused runs changed nothing
    run_sql(db_cursor, DOWN)
    assert_before_state(db_cursor)


HISTORY_TABLES = ("transactions", "transaction_lines", "lots", "ingredient_lot_consumption",
                  "transaction_substitutions", "lot_events")


def _sql_without_comments(path):
    text = re.sub(r"--[^\n]*", "", path.read_text())
    return re.sub(r"'(?:[^']|'')*'", "''", text)  # string literals (messages) do not count


def test_078_up_and_down_never_reference_a_history_table():
    """Review fix 2: no history read (the old down counted posted makes); only catalog tables."""
    for path in (UP, DOWN):
        sql = _sql_without_comments(path)
        for table in HISTORY_TABLES:
            assert not re.search(rf"\b{table}\b", sql), f"{path.name} references {table}"
        referenced = set(re.findall(r"\b(?:FROM|UPDATE|INSERT\s+INTO|JOIN)\s+([a-z_]+)", sql, re.I))
        assert referenced == {"products", "batch_formulas", "product_bom", "migration_markers"}, (path.name, referenced)


def _post_make_history(cur, product_id, lot_code, output_lb, inputs):
    """A historical posted make with its ingredient lines, as the ledger holds them."""
    cur.execute("INSERT INTO transactions (type, timestamp, notes, occurred_at, operator_id) "
                "VALUES ('make', now() - interval '7 days', %s, now() - interval '7 days', 'legacy-shared-key') RETURNING id",
                (f"2 batch(es) history {lot_code}",))
    txn = cur.fetchone()["id"]
    cur.execute("INSERT INTO lots (product_id, lot_code, entry_source) VALUES (%s, %s, 'production_output') RETURNING id", (product_id, lot_code))
    out_lot = cur.fetchone()["id"]
    cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (txn, product_id, out_lot, output_lb))
    for ing, qty in inputs:
        cur.execute("INSERT INTO lots (product_id, lot_code, entry_source) VALUES (%s, %s, 'received') RETURNING id", (ing, f"{lot_code}-IN-{ing}"))
        lot = cur.fetchone()["id"]
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (txn, ing, lot, -qty))
        cur.execute("INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (txn, ing, lot, qty))
    return txn


def _ledger_snapshot(cur):
    """Every column of every history row (not a column subset)."""
    snap = {}
    for table in HISTORY_TABLES:
        cur.execute(f"SELECT to_jsonb(t) AS row FROM {table} t ORDER BY 1")
        snap[table] = sorted(str(r["row"]) for r in cur.fetchall())
    return snap


def test_078_leaves_posted_history_untouched(db_cursor):
    seed_before_state(db_cursor)
    # The Oct 6 shape: 2 batches booked as 769.04 lb for 50 lb of inputs under the old recipe.
    _post_make_history(db_cursor, FRUIT_NUT, "OCT 06 2026", Decimal("769.04"),
                       [(CLASSIC9, 40), (WALNUTS, 2.5), (ALMONDS, 2.5), (RAISINS, 2.5), (CRANBERRIES, 2.5)])
    before = _ledger_snapshot(db_cursor)
    run_sql(db_cursor, UP)
    assert _ledger_snapshot(db_cursor) == before
    # Balances derived from the ledger are unchanged as well (the ghost 769.04 lb stays until a true-up).
    db_cursor.execute(f"SELECT COALESCE(SUM(tl.quantity_lb),0) AS lb FROM lots l JOIN {main.POSTED_LINES} tl ON tl.lot_id=l.id WHERE l.product_id=%s", (FRUIT_NUT,))
    assert db_cursor.fetchone()["lb"] == Decimal("769.04")
    run_sql(db_cursor, DOWN)
    assert _ledger_snapshot(db_cursor) == before


# ─────────────────────────── behaviour through the endpoints ───────────────────────────

class _ConnProxy:
    """commit()/rollback() act on an inner SAVEPOINT; the session fixture rolls everything back."""

    def __init__(self, conn, sp):
        self._conn, self._sp = conn, sp
        with conn.cursor() as c:
            c.execute(f"SAVEPOINT {sp}")

    def cursor(self, *a, **k):
        return self._conn.cursor(*a, **k)

    def commit(self):
        with self._conn.cursor() as c:
            c.execute(f"RELEASE SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")

    def rollback(self):
        with self._conn.cursor() as c:
            c.execute(f"ROLLBACK TO SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")


@pytest.fixture
def api(_db_connection, monkeypatch):
    @contextmanager
    def _conn():
        proxy = _ConnProxy(_db_connection, "recipe_078_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _conn)
    from psycopg2.extras import RealDictCursor
    cur = _db_connection.cursor(cursor_factory=RealDictCursor)
    cur.execute("SAVEPOINT recipe_078_test")
    with TestClient(main.app) as client:
        client.headers["X-API-Key"] = main.API_KEY
        try:
            yield client, cur
        finally:
            cur.execute("ROLLBACK TO SAVEPOINT recipe_078_test")
            cur.close()
    _db_connection.rollback()


def _stock(cur, product_id, lot_code, lb):
    cur.execute("INSERT INTO lots (product_id, lot_code, entry_source, received_at) VALUES (%s, %s, 'received', now() - interval '1 day') RETURNING id", (product_id, lot_code))
    lot = cur.fetchone()["id"]
    cur.execute("INSERT INTO transactions (type, timestamp, notes, occurred_at) VALUES ('receive', now(), 'test stock 078', now()) RETURNING id")
    txn = cur.fetchone()["id"]
    cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (txn, product_id, lot, lb))
    return lot


def _on_hand(cur, product_id):
    cur.execute(f"SELECT COALESCE(SUM(tl.quantity_lb),0) AS lb FROM lots l JOIN {main.POSTED_LINES} tl ON tl.lot_id=l.id WHERE l.product_id=%s", (product_id,))
    return cur.fetchone()["lb"]


def _consumed(cur, txn_id):
    cur.execute("SELECT product_id, -SUM(quantity_lb) AS lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0 GROUP BY product_id", (txn_id,))
    lines = {r["product_id"]: r["lb"] for r in cur.fetchall()}
    cur.execute("SELECT ingredient_product_id, SUM(quantity_lb) AS lb FROM ingredient_lot_consumption WHERE transaction_id=%s GROUP BY 1", (txn_id,))
    assert {r["ingredient_product_id"]: r["lb"] for r in cur.fetchall()} == lines, "ILC must mirror the ledger lines"
    return lines


def test_make_fruit_nut_after_078_posts_328_3_and_deducts_the_corrected_formula(api):
    client, cur = api
    seed_before_state(cur)
    run_sql(cur, UP)
    _stock(cur, CLASSIC9, "B26-1001-001", 700)
    for ing in INCLUSIONS:
        _stock(cur, ing, f"INC-{ing}", 40)

    preview = client.post("/make", json={"mode": "preview", "product_name": "90008", "batches": 2}).json()
    assert preview["batch_size_lb"] == 328.3 and preview["total_output_lb"] == pytest.approx(656.6)
    assert {i["ingredient_id"]: i["needed_lb"] for i in preview["ingredients"]} == {CLASSIC9: pytest.approx(552.6), **{i: 26.0 for i in INCLUSIONS}}
    assert preview["all_ingredients_available"] is True

    r = client.post("/make", json={"mode": "commit", "product_name": "90008", "batches": 1, "lot_code": "FN-078-TEST", "confirmed_sku": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_lb"] == pytest.approx(328.3) and body["formula_weight_lb"] == pytest.approx(328.3)
    consumed = _consumed(cur, body["transaction_id"])
    assert consumed == {CLASSIC9: Decimal("276.3"), **{i: Decimal("13") for i in INCLUSIONS}}
    assert _on_hand(cur, FRUIT_NUT) == Decimal("328.3")
    assert _on_hand(cur, CLASSIC9) == Decimal("423.7")
    assert all(_on_hand(cur, i) == Decimal("27") for i in INCLUSIONS)
    assert body["pack_needed"]["finished_goods"] == [{"product_id": FRUIT_NUT_FG, "name": "Granola Fruit Nut 25 LB", "case_size_lb": 25.0}]


def test_pack_fruit_nut_25lb_cases_stays_consistent_after_078(api):
    client, cur = api
    seed_before_state(cur)
    run_sql(cur, UP)
    _stock(cur, CLASSIC9, "B26-1001-002", 300)
    for ing in INCLUSIONS:
        _stock(cur, ing, f"INC2-{ing}", 20)
    made = client.post("/make", json={"mode": "commit", "product_name": "90008", "batches": 1, "lot_code": "FN-078-PACK", "confirmed_sku": True}).json()
    assert made["output_lb"] == pytest.approx(328.3)

    # 13 full 25-lb cases fit in one 328.3 lb batch (the audit's 13-case yield); 14 do not.
    too_many = client.post("/pack", json={"mode": "preview", "source_product": "90008", "target_product": "70061", "cases": 14}).json()
    assert too_many["total_lb"] == 350 and too_many["all_lots_sufficient"] is False
    preview = client.post("/pack", json={"mode": "preview", "source_product": "90008", "target_product": "70061", "cases": 13}).json()
    assert preview["case_weight_lb"] == 25 and preview["total_lb"] == 325 and preview["all_lots_sufficient"] is True
    assert "add_in_ingredients" not in preview, "142 is packed straight from 179 — no add-ins"
    r = client.post("/pack", json={"mode": "commit", "source_product": "90008", "target_product": "70061", "cases": 13})
    assert r.status_code == 200, r.text
    assert r.json()["total_lb"] == 325 and r.json()["batch_lots_consumed"] == [{"lot_code": "FN-078-PACK", "consumed_lb": 325.0}]
    assert _on_hand(cur, FRUIT_NUT_FG) == Decimal("325")
    assert _on_hand(cur, FRUIT_NUT) == Decimal("3.3")

    # Planner / full-BOM loader sees one consistent recipe: 142 ← 179 @ 25 lb; 179 ← 328.3 lb of inputs = its batch size.
    fg_to_batch, batch_to_ingredients, batch_sizes = main._load_bom_structure(cur)
    assert [(c["component_product_id"], c["quantity"], c["default_batch_lb"]) for c in fg_to_batch[FRUIT_NUT_FG]] == [(FRUIT_NUT, 25.0, 328.3)]
    assert {c["component_product_id"]: c["quantity"] for c in fg_to_batch[FRUIT_NUT]} == {CLASSIC9: 276.3, **{i: 13.0 for i in INCLUSIONS}}
    assert sum(i["quantity_lb"] for i in batch_to_ingredients[FRUIT_NUT]) == pytest.approx(328.3) == batch_sizes[FRUIT_NUT]


def test_make_284_after_078_deducts_4000ct_chips_not_1000ct(api):
    client, cur = api
    seed_before_state(cur)
    # Stock for every 284 ingredient except 1,000 CT chips (production has 0 lb of 73 as well).
    for ing, lb in ((13, 50), (18, 10), (30, 10), (32, 500), (34, 60), (39, 10), (44, 100), (CHIPS_4000, 100)):
        _stock(cur, ing, f"K-{ing}", lb)
    before = client.post("/make", json={"mode": "preview", "product_name": "90026", "batches": 1}).json()
    chips = [i for i in before["ingredients"] if i["ingredient_id"] in (CHIPS_1000, CHIPS_4000)]
    assert [(c["ingredient_id"], c["sufficient"]) for c in chips] == [(CHIPS_1000, False)], "before 078 the recipe demands 1,000 CT"
    assert before["all_ingredients_available"] is False

    run_sql(cur, UP)
    after = client.post("/make", json={"mode": "preview", "product_name": "90026", "batches": 1}).json()
    chips = [i for i in after["ingredients"] if i["ingredient_id"] in (CHIPS_1000, CHIPS_4000)]
    assert [(c["ingredient_id"], c["needed_lb"], c["sufficient"]) for c in chips] == [(CHIPS_4000, 25.0, True)]
    assert after["all_ingredients_available"] is True and after["total_output_lb"] == 348

    r = client.post("/make", json={"mode": "commit", "product_name": "90026", "batches": 1, "lot_code": "K-078-TEST", "confirmed_sku": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_lb"] == 348
    consumed = _consumed(cur, body["transaction_id"])
    assert consumed[CHIPS_4000] == Decimal("25") and CHIPS_1000 not in consumed
    assert consumed == {13: Decimal("12.5"), 18: Decimal("1.8"), 30: Decimal("1.6"), 32: Decimal("225"), 34: Decimal("30"),
                        39: Decimal("1.7"), 44: Decimal("50"), CHIPS_4000: Decimal("25")}
    assert _on_hand(cur, CHIPS_4000) == Decimal("75") and _on_hand(cur, CHIPS_1000) == 0

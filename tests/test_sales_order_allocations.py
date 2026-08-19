"""Migration 044 — `sales_order_allocations` (FR-4, PR 1: schema only).

DDL / uniqueness / CHECK-constraint coverage ONLY. There is deliberately no
endpoint, helper, or ship-path behavior to test yet (design doc
docs/designs/044-so-allocations-design.md, PR plan). Later PRs add
allocate / readiness / consume-on-ship tests to this module.

Covers:
  * table + columns exist with the designed types / nullability / defaults
  * FKs to sales_orders, sales_order_lines, products, lots, transactions, self
  * CHECKs: quantity_lb > 0, status / source enums,
    shipped <=> ship_transaction_id, released => released_at
  * partial unique LIVE indexes: one active SKU-level row per line; one active
    pin per (line, lot); released/shipped/superseded rows never collide; two
    different lines may pin the same lot; SKU-level + lot-level coexist on a
    line
  * migration file is idempotent (re-run inside the test transaction = no-op)
  * regression guard: office schema still 30 ops, floor schema still 22
"""

import re
from pathlib import Path

import pytest

try:
    import psycopg2
    from psycopg2 import errors as pg_errors
except Exception:  # pragma: no cover
    pytest.skip("psycopg2 not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "migrations" / "044_sales_order_allocations.sql"


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def _seed_order(cur, n_lines=2, product_ids=None):
    """customer + SO + `n_lines` lines (each on its own product unless given).
    Returns (order_id, [line_ids], [product_ids])."""
    cur.execute(
        "INSERT INTO customers (name, active) VALUES ('TEST-044 Customer', true) RETURNING id"
    )
    customer_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status) "
        "VALUES (%s, 'TEST-044-SO', 'confirmed') RETURNING id",
        (customer_id,),
    )
    order_id = cur.fetchone()["id"]
    line_ids, pids = [], []
    for i in range(n_lines):
        if product_ids is not None:
            pid = product_ids[i]
        else:
            cur.execute(
                "INSERT INTO products (name, type, odoo_code, uom, active) "
                "VALUES (%s, 'finished', %s, 'lb', true) RETURNING id",
                (f"TEST-044 FG {i}", f"T044-{i}"),
            )
            pid = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb) "
            "VALUES (%s, %s, 100) RETURNING id",
            (order_id, pid),
        )
        line_ids.append(cur.fetchone()["id"])
        pids.append(pid)
    return order_id, line_ids, pids


def _seed_lot(cur, product_id, lot_code):
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, lot_code),
    )
    return cur.fetchone()["id"]


def _seed_txn(cur):
    cur.execute(
        "INSERT INTO transactions (type, timestamp, status) "
        "VALUES ('ship', now(), 'posted') RETURNING id"
    )
    return cur.fetchone()["id"]


def _insert(cur, order_id, line_id, product_id, lot_id=None, qty=10, status="active",
            source="manual", **extra):
    cols = ["sales_order_id", "sales_order_line_id", "product_id", "lot_id",
            "quantity_lb", "status", "source"] + list(extra)
    vals = [order_id, line_id, product_id, lot_id, qty, status, source] + list(extra.values())
    cur.execute(
        f"INSERT INTO sales_order_allocations ({', '.join(cols)}) "
        f"VALUES ({', '.join(['%s'] * len(vals))}) RETURNING id",
        vals,
    )
    return cur.fetchone()["id"]


def _expect_error(cur, exc_type, fn, *args, **kwargs):
    """Run `fn` inside its own savepoint, assert it raises `exc_type`, and roll
    back to the savepoint so the enclosing test transaction stays usable."""
    cur.execute("SAVEPOINT t044")
    with pytest.raises(exc_type):
        fn(*args, **kwargs)
    cur.execute("ROLLBACK TO SAVEPOINT t044")
    cur.execute("RELEASE SAVEPOINT t044")


# ─────────────────────────────────────────────────────────────────
# DDL shape
# ─────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = {
    # name: (data_type, is_nullable, default_contains)
    "id": ("bigint", "NO", None),
    "sales_order_id": ("integer", "NO", None),
    "sales_order_line_id": ("integer", "NO", None),
    "product_id": ("integer", "NO", None),
    "lot_id": ("integer", "YES", None),
    "quantity_lb": ("numeric", "NO", None),
    "status": ("text", "NO", "'active'"),
    "source": ("text", "NO", None),
    "ship_transaction_id": ("integer", "YES", None),
    "last_ship_transaction_id": ("integer", "YES", None),
    "split_from_id": ("bigint", "YES", None),
    "created_at": ("timestamp with time zone", "NO", "clock_timestamp()"),
    "created_at_source": ("text", "NO", "'database'"),
    "created_by": ("text", "YES", None),
    "released_at": ("timestamp with time zone", "YES", None),
    "released_by": ("text", "YES", None),
    "release_reason": ("text", "YES", None),
    "expires_at": ("timestamp with time zone", "YES", None),
    "note": ("text", "YES", None),
}


@pytest.mark.db
def test_table_columns_types_nullability_defaults(db_cursor):
    cur = db_cursor
    cur.execute(
        """SELECT column_name, data_type, is_nullable, column_default,
                  numeric_precision, numeric_scale, is_identity
             FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'sales_order_allocations'"""
    )
    cols = {r["column_name"]: r for r in cur.fetchall()}
    assert set(cols) == set(EXPECTED_COLUMNS), (
        f"unexpected column set: extra={set(cols) - set(EXPECTED_COLUMNS)} "
        f"missing={set(EXPECTED_COLUMNS) - set(cols)}"
    )
    for name, (dtype, nullable, default_sub) in EXPECTED_COLUMNS.items():
        c = cols[name]
        assert c["data_type"] == dtype, f"{name}: {c['data_type']} != {dtype}"
        assert c["is_nullable"] == nullable, f"{name}: nullable {c['is_nullable']} != {nullable}"
        if default_sub is None:
            assert c["column_default"] is None or name == "id", f"{name}: unexpected default {c['column_default']}"
        else:
            assert default_sub in (c["column_default"] or ""), f"{name}: default {c['column_default']}"
    assert cols["id"]["is_identity"] == "YES"
    assert (cols["quantity_lb"]["numeric_precision"], cols["quantity_lb"]["numeric_scale"]) == (14, 4)
    # created_by is the interim FR-15 source tag (nullable, no default) —
    # never a 'legacy-shared-key' placeholder default.
    assert cols["created_by"]["column_default"] is None


@pytest.mark.db
def test_foreign_keys(db_cursor):
    cur = db_cursor
    cur.execute(
        """SELECT kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_col
             FROM information_schema.table_constraints tc
             JOIN information_schema.key_column_usage kcu
               ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
             JOIN information_schema.constraint_column_usage ccu
               ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
            WHERE tc.table_schema = 'public' AND tc.table_name = 'sales_order_allocations'
              AND tc.constraint_type = 'FOREIGN KEY'"""
    )
    fks = {(r["column_name"], r["ref_table"], r["ref_col"]) for r in cur.fetchall()}
    assert fks == {
        ("sales_order_id", "sales_orders", "id"),
        ("sales_order_line_id", "sales_order_lines", "id"),
        ("product_id", "products", "id"),
        ("lot_id", "lots", "id"),
        ("ship_transaction_id", "transactions", "id"),
        ("last_ship_transaction_id", "transactions", "id"),
        ("split_from_id", "sales_order_allocations", "id"),
    }


@pytest.mark.db
def test_fk_rejects_unknown_lot_and_line(db_cursor):
    cur = db_cursor
    order_id, (line_a, _), (pid_a, _) = _seed_order(cur)
    _expect_error(cur, pg_errors.ForeignKeyViolation,
                  _insert, cur, order_id, line_a, pid_a, lot_id=2_000_000_000)
    _expect_error(cur, pg_errors.ForeignKeyViolation,
                  _insert, cur, order_id, 2_000_000_000, pid_a)
    _expect_error(cur, pg_errors.ForeignKeyViolation,
                  _insert, cur, order_id, line_a, pid_a, status="shipped",
                  ship_transaction_id=2_000_000_000)


@pytest.mark.db
def test_indexes_exist_with_live_predicates(db_cursor):
    cur = db_cursor
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = 'sales_order_allocations'"
    )
    idx = {r["indexname"]: r["indexdef"] for r in cur.fetchall()}
    assert set(idx) == {
        "sales_order_allocations_pkey",
        "soa_active_sku_uniq",
        "soa_active_lot_uniq",
        "soa_active_product_idx",
        "soa_active_lot_idx",
        "soa_order_idx",
        "soa_ship_txn_idx",
    }

    def norm(s):
        return re.sub(r"\s+", " ", s.replace("::text", "")).strip()

    sku = norm(idx["soa_active_sku_uniq"])
    assert sku.startswith("CREATE UNIQUE INDEX")
    assert "(sales_order_line_id)" in sku
    assert "WHERE ((status = 'active') AND (lot_id IS NULL))" in sku

    lot = norm(idx["soa_active_lot_uniq"])
    assert lot.startswith("CREATE UNIQUE INDEX")
    assert "(sales_order_line_id, lot_id)" in lot
    assert "WHERE ((status = 'active') AND (lot_id IS NOT NULL))" in lot

    assert "WHERE (status = 'active')" in norm(idx["soa_active_product_idx"])
    assert "WHERE ((status = 'active') AND (lot_id IS NOT NULL))" in norm(idx["soa_active_lot_idx"])
    assert "WHERE (ship_transaction_id IS NOT NULL)" in norm(idx["soa_ship_txn_idx"])
    # plain (non-unique) indexes must not accidentally be unique
    for name in ("soa_active_product_idx", "soa_active_lot_idx", "soa_order_idx", "soa_ship_txn_idx"):
        assert not idx[name].startswith("CREATE UNIQUE"), name


# ─────────────────────────────────────────────────────────────────
# CHECK constraints
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_quantity_must_be_positive(db_cursor):
    cur = db_cursor
    order_id, (line_a, _), (pid_a, _) = _seed_order(cur)
    for bad in (0, -1, -0.0001):
        _expect_error(cur, pg_errors.CheckViolation,
                      _insert, cur, order_id, line_a, pid_a, qty=bad)
    assert _insert(cur, order_id, line_a, pid_a, qty=0.0001)  # smallest positive at scale 4


@pytest.mark.db
def test_status_and_source_enums(db_cursor):
    cur = db_cursor
    order_id, (line_a, _), (pid_a, _) = _seed_order(cur)
    _expect_error(cur, pg_errors.CheckViolation,
                  _insert, cur, order_id, line_a, pid_a, status="pending")
    _expect_error(cur, pg_errors.CheckViolation,
                  _insert, cur, order_id, line_a, pid_a, source="fifo")
    _expect_error(cur, pg_errors.NotNullViolation,
                  _insert, cur, order_id, line_a, pid_a, source=None)
    # every designed value is accepted (distinct lines/lots so uniques don't interfere)
    txn = _seed_txn(cur)
    lots = [_seed_lot(cur, pid_a, f"T044-ENUM-{i}") for i in range(3)]
    for i, src in enumerate(("manual", "auto_fifo", "staged_lot")):
        assert _insert(cur, order_id, line_a, pid_a, lot_id=lots[i], source=src)
    assert _insert(cur, order_id, line_a, pid_a, status="released", released_at="now()")
    assert _insert(cur, order_id, line_a, pid_a, status="shipped", ship_transaction_id=txn)
    assert _insert(cur, order_id, line_a, pid_a, status="superseded")
    # default status is 'active'
    cur.execute(
        "INSERT INTO sales_order_allocations (sales_order_id, sales_order_line_id, product_id, quantity_lb, source) "
        "VALUES (%s, %s, %s, 5, 'manual') RETURNING status, created_at_source, created_by",
        (order_id, line_a, pid_a),
    )
    row = cur.fetchone()
    assert row["status"] == "active"
    assert row["created_at_source"] == "database"
    assert row["created_by"] is None


@pytest.mark.db
def test_shipped_iff_ship_transaction_id(db_cursor):
    cur = db_cursor
    order_id, (line_a, _), (pid_a, _) = _seed_order(cur)
    txn = _seed_txn(cur)
    # shipped without a txn → reject
    _expect_error(cur, pg_errors.CheckViolation,
                  _insert, cur, order_id, line_a, pid_a, status="shipped")
    # a txn on a non-shipped row → reject (both directions of the <=>)
    for st in ("active", "superseded"):
        _expect_error(cur, pg_errors.CheckViolation,
                      _insert, cur, order_id, line_a, pid_a, status=st,
                      ship_transaction_id=txn)
    _expect_error(cur, pg_errors.CheckViolation,
                  _insert, cur, order_id, line_a, pid_a, status="released",
                  released_at="now()", ship_transaction_id=txn)
    # last_ship_transaction_id is NOT part of the iff — it survives void/restore
    assert _insert(cur, order_id, line_a, pid_a, status="active", last_ship_transaction_id=txn)
    assert _insert(cur, order_id, line_a, pid_a, status="shipped",
                   ship_transaction_id=txn, last_ship_transaction_id=txn)


@pytest.mark.db
def test_released_requires_released_at(db_cursor):
    cur = db_cursor
    order_id, (line_a, _), (pid_a, _) = _seed_order(cur)
    _expect_error(cur, pg_errors.CheckViolation,
                  _insert, cur, order_id, line_a, pid_a, status="released")
    assert _insert(cur, order_id, line_a, pid_a, status="released",
                   released_at="now()", released_by="dashboard", release_reason="test")
    # released_at on a non-released row is allowed (no constraint the other way)
    assert _insert(cur, order_id, line_a, pid_a, status="superseded", released_at="now()",
                   release_reason="split_on_ship")


# ─────────────────────────────────────────────────────────────────
# Partial unique LIVE indexes
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_one_live_sku_level_row_per_line(db_cursor):
    cur = db_cursor
    order_id, (line_a, line_b), (pid_a, pid_b) = _seed_order(cur)
    first = _insert(cur, order_id, line_a, pid_a, qty=40)
    # second ACTIVE SKU-level row on the same line → unique violation
    _expect_error(cur, pg_errors.UniqueViolation,
                  _insert, cur, order_id, line_a, pid_a, qty=60)
    # another line is independent
    assert _insert(cur, order_id, line_b, pid_b, qty=60)
    # non-live rows on the same line never collide (any number of them)
    txn = _seed_txn(cur)
    assert _insert(cur, order_id, line_a, pid_a, status="released", released_at="now()")
    assert _insert(cur, order_id, line_a, pid_a, status="released", released_at="now()")
    assert _insert(cur, order_id, line_a, pid_a, status="shipped", ship_transaction_id=txn)
    assert _insert(cur, order_id, line_a, pid_a, status="superseded", split_from_id=first)
    # once the live row is released, the key is free again
    cur.execute(
        "UPDATE sales_order_allocations SET status='released', released_at=now() WHERE id=%s",
        (first,),
    )
    assert _insert(cur, order_id, line_a, pid_a, qty=60)
    # and flipping a released row back to active while another is live collides
    _expect_error(cur, pg_errors.UniqueViolation, cur.execute,
                  "UPDATE sales_order_allocations SET status='active' WHERE id=%s", (first,))


@pytest.mark.db
def test_one_live_pin_per_line_and_lot(db_cursor):
    cur = db_cursor
    order_id, (line_a, line_b), (pid_a, _) = _seed_order(cur, n_lines=2)
    # line_b is on a different product by default; re-point it at pid_a so both
    # lines legitimately compete for the same lot.
    cur.execute("UPDATE sales_order_lines SET product_id=%s WHERE id=%s", (pid_a, line_b))
    lot_1 = _seed_lot(cur, pid_a, "T044-LOT-1")
    lot_2 = _seed_lot(cur, pid_a, "T044-LOT-2")

    pin = _insert(cur, order_id, line_a, pid_a, lot_id=lot_1, qty=80, source="staged_lot")
    # same line, same lot, both active → violation
    _expect_error(cur, pg_errors.UniqueViolation,
                  _insert, cur, order_id, line_a, pid_a, lot_id=lot_1, qty=5, source="staged_lot")
    # same line, DIFFERENT lot → fine (multi-lot split)
    assert _insert(cur, order_id, line_a, pid_a, lot_id=lot_2, qty=20, source="auto_fifo")
    # DIFFERENT line, same lot → fine (partial lots across orders/lines)
    assert _insert(cur, order_id, line_b, pid_a, lot_id=lot_1, qty=30, source="staged_lot")
    # SKU-level (lot NULL) and lot-level rows coexist on the same line (hybrid)
    assert _insert(cur, order_id, line_a, pid_a, lot_id=None, qty=10)
    # non-live duplicate of the pin never collides
    txn = _seed_txn(cur)
    assert _insert(cur, order_id, line_a, pid_a, lot_id=lot_1, qty=80,
                   status="shipped", ship_transaction_id=txn, split_from_id=pin)
    assert _insert(cur, order_id, line_a, pid_a, lot_id=lot_1, qty=80,
                   status="superseded", release_reason="split_on_ship")
    # after the live pin is superseded the (line, lot) key is free again
    cur.execute(
        "UPDATE sales_order_allocations SET status='superseded', release_reason='lot_merged' WHERE id=%s",
        (pin,),
    )
    assert _insert(cur, order_id, line_a, pid_a, lot_id=lot_1, qty=80, source="staged_lot")


@pytest.mark.db
def test_void_coalesce_shape_holds_under_unique_index(db_cursor):
    """Worked example from the design (Q2): allocate 100 lot-level, ship 40 →
    leftover 60 active + 40 shipped + original superseded. A void must
    COALESCE onto the leftover (one live 100), because re-activating the
    shipped row would violate soa_active_lot_uniq. This test pins the
    index semantics the PR-3 handler will rely on."""
    cur = db_cursor
    order_id, (line_a,), (pid_a,) = _seed_order(cur, n_lines=1)
    lot = _seed_lot(cur, pid_a, "T044-COALESCE")
    txn = _seed_txn(cur)

    original = _insert(cur, order_id, line_a, pid_a, lot_id=lot, qty=100, source="staged_lot")
    # ship 40: original → superseded, leftover 60 active, 40 shipped
    cur.execute(
        "UPDATE sales_order_allocations SET status='superseded', release_reason='split_on_ship' WHERE id=%s",
        (original,),
    )
    leftover = _insert(cur, order_id, line_a, pid_a, lot_id=lot, qty=60, source="staged_lot",
                       split_from_id=original)
    shipped = _insert(cur, order_id, line_a, pid_a, lot_id=lot, qty=40, source="staged_lot",
                      status="shipped", ship_transaction_id=txn, last_ship_transaction_id=txn,
                      split_from_id=original)

    # naive void (flip shipped row back to active) → unique violation
    _expect_error(
        cur, pg_errors.UniqueViolation, cur.execute,
        "UPDATE sales_order_allocations SET status='active', ship_transaction_id=NULL WHERE id=%s",
        (shipped,),
    )
    # coalesce: leftover += 40, shipped → superseded/void_coalesced
    cur.execute(
        "UPDATE sales_order_allocations SET quantity_lb = quantity_lb + 40, last_ship_transaction_id=%s WHERE id=%s",
        (txn, leftover),
    )
    cur.execute(
        "UPDATE sales_order_allocations SET status='superseded', release_reason='void_coalesced', "
        "ship_transaction_id=NULL WHERE id=%s",
        (shipped,),
    )
    cur.execute(
        "SELECT id, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND lot_id=%s AND status='active'",
        (line_a, lot),
    )
    live = cur.fetchall()
    assert len(live) == 1 and live[0]["id"] == leftover
    assert live[0]["quantity_lb"] == 100


@pytest.mark.db
def test_full_void_then_restore_cycle_holds_under_unique_index(db_cursor):
    """Design Q2 worked example end-to-end (required test):
      allocate 100 lot-level → one live 100
      ship 40 (split)        → leftover 60 active + 40 shipped + original superseded
      void                   → ONE live 100 (coalesced), shipped row void_coalesced
      restore                → leftover 60 active + 40 shipped
    Every step must succeed without a soa_active_lot_uniq violation, and
    every intermediate state must have exactly one live row for (line, lot).
    This exercises the index/CHECK contract the PR-3 handler will rely on."""
    cur = db_cursor
    order_id, (line_a,), (pid_a,) = _seed_order(cur, n_lines=1)
    lot = _seed_lot(cur, pid_a, "T044-CYCLE")
    txn = _seed_txn(cur)

    def live_rows():
        cur.execute(
            "SELECT id, quantity_lb, last_ship_transaction_id FROM sales_order_allocations "
            "WHERE sales_order_line_id=%s AND lot_id=%s AND status='active' ORDER BY id",
            (line_a, lot),
        )
        return cur.fetchall()

    def row(rid):
        cur.execute(
            "SELECT status, quantity_lb, ship_transaction_id, last_ship_transaction_id, "
            "release_reason, split_from_id FROM sales_order_allocations WHERE id=%s",
            (rid,),
        )
        return cur.fetchone()

    # 1. allocate 100 lot-level → one live 100
    original = _insert(cur, order_id, line_a, pid_a, lot_id=lot, qty=100, source="staged_lot")
    assert [r["quantity_lb"] for r in live_rows()] == [100]

    # 2. ship 40 (split): original superseded; leftover 60 active; 40 shipped
    cur.execute(
        "UPDATE sales_order_allocations SET status='superseded', release_reason='split_on_ship' WHERE id=%s",
        (original,),
    )
    leftover = _insert(cur, order_id, line_a, pid_a, lot_id=lot, qty=60, source="staged_lot",
                       split_from_id=original)
    shipped = _insert(cur, order_id, line_a, pid_a, lot_id=lot, qty=40, source="staged_lot",
                      status="shipped", ship_transaction_id=txn, last_ship_transaction_id=txn,
                      split_from_id=original)
    live = live_rows()
    assert [r["id"] for r in live] == [leftover] and live[0]["quantity_lb"] == 60
    assert row(shipped)["status"] == "shipped" and row(shipped)["quantity_lb"] == 40
    assert row(original)["status"] == "superseded"

    # 3. void: re-activating the shipped row would collide with the leftover
    _expect_error(
        cur, pg_errors.UniqueViolation, cur.execute,
        "UPDATE sales_order_allocations SET status='active', ship_transaction_id=NULL WHERE id=%s",
        (shipped,),
    )
    #    ... so coalesce: leftover 60 → 100 (+ remember the txn), shipped → superseded/void_coalesced
    cur.execute(
        "UPDATE sales_order_allocations SET quantity_lb = quantity_lb + %s, last_ship_transaction_id=%s "
        "WHERE id=%s",
        (40, txn, leftover),
    )
    cur.execute(
        "UPDATE sales_order_allocations SET status='superseded', release_reason='void_coalesced', "
        "ship_transaction_id=NULL WHERE id=%s",
        (shipped,),
    )
    live = live_rows()
    assert len(live) == 1 and live[0]["id"] == leftover and live[0]["quantity_lb"] == 100
    assert live[0]["last_ship_transaction_id"] == txn
    s = row(shipped)
    assert s["status"] == "superseded" and s["release_reason"] == "void_coalesced"
    assert s["ship_transaction_id"] is None and s["last_ship_transaction_id"] == txn
    # source is never overwritten by void/restore
    cur.execute("SELECT DISTINCT source FROM sales_order_allocations WHERE sales_order_line_id=%s", (line_a,))
    assert [r["source"] for r in cur.fetchall()] == ["staged_lot"]

    # 4. restore: find the void_coalesced row S for this txn, shrink the live
    #    leftover L by S.quantity_lb (100 → 60), flip S back to shipped.
    cur.execute(
        "SELECT id, quantity_lb FROM sales_order_allocations "
        "WHERE last_ship_transaction_id=%s AND release_reason='void_coalesced' AND status='superseded'",
        (txn,),
    )
    coalesced = cur.fetchall()
    assert [c["id"] for c in coalesced] == [shipped]
    take = coalesced[0]["quantity_lb"]
    assert live_rows()[0]["quantity_lb"] >= take, "RESTORE_SPLIT_MISSING guard would fire"
    cur.execute(
        "UPDATE sales_order_allocations SET quantity_lb = quantity_lb - %s WHERE id=%s",
        (take, leftover),
    )
    cur.execute(
        "UPDATE sales_order_allocations SET status='shipped', ship_transaction_id=%s, release_reason=NULL "
        "WHERE id=%s",
        (txn, shipped),
    )

    # end state: one active 60 + one shipped 40 (+ the superseded original), no violations
    live = live_rows()
    assert len(live) == 1 and live[0]["id"] == leftover and live[0]["quantity_lb"] == 60
    s = row(shipped)
    assert s["status"] == "shipped" and s["quantity_lb"] == 40
    assert s["ship_transaction_id"] == txn and s["last_ship_transaction_id"] == txn
    assert s["split_from_id"] == original
    cur.execute(
        "SELECT status, count(*) AS n, sum(quantity_lb) AS lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s GROUP BY status ORDER BY status",
        (line_a,),
    )
    by_status = {r["status"]: (r["n"], r["lb"]) for r in cur.fetchall()}
    assert by_status == {"active": (1, 60), "shipped": (1, 40), "superseded": (1, 100)}

    # 5. a SECOND void/restore round still raises no unique violation (same shape)
    cur.execute(
        "UPDATE sales_order_allocations SET quantity_lb = quantity_lb + 40 WHERE id=%s", (leftover,)
    )
    cur.execute(
        "UPDATE sales_order_allocations SET status='superseded', release_reason='void_coalesced', "
        "ship_transaction_id=NULL WHERE id=%s",
        (shipped,),
    )
    assert [r["quantity_lb"] for r in live_rows()] == [100]
    cur.execute(
        "UPDATE sales_order_allocations SET quantity_lb = quantity_lb - 40 WHERE id=%s", (leftover,)
    )
    cur.execute(
        "UPDATE sales_order_allocations SET status='shipped', ship_transaction_id=%s, release_reason=NULL "
        "WHERE id=%s",
        (txn, shipped),
    )
    assert [r["quantity_lb"] for r in live_rows()] == [60]
    assert row(shipped)["status"] == "shipped"


# ─────────────────────────────────────────────────────────────────
# Migration file hygiene
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_migration_is_idempotent(db_cursor):
    """Re-running 044 against a DB that already has it is a no-op. The file
    wraps itself in BEGIN/COMMIT; strip those so the re-run stays inside the
    test's rollback-only transaction."""
    sql = MIGRATION.read_text()
    body = "\n".join(
        ln for ln in sql.splitlines()
        if ln.strip().upper() not in ("BEGIN;", "COMMIT;")
    )
    cur = db_cursor
    cur.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename='sales_order_allocations'"
    )
    before = cur.fetchone()["n"]
    cur.execute(body)  # must not raise
    cur.execute(
        "SELECT count(*) AS n FROM pg_indexes WHERE tablename='sales_order_allocations'"
    )
    assert cur.fetchone()["n"] == before == 7


def test_migration_file_has_no_session_guc_or_business_date():
    sql = MIGRATION.read_text()
    low = sql.lower()
    assert "set session" not in low
    assert "default_transaction_read_only" not in low
    assert "business_date" not in re.sub(r"--[^\n]*", "", low), "no business_date column by design"
    assert "create table if not exists public.sales_order_allocations" in low
    assert "legacy-shared-key" not in re.sub(r"--[^\n]*", "", low)


# ─────────────────────────────────────────────────────────────────
# Regression guard (design §Test plan): GPT schemas untouched
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path,expected",
    [("openapi-gpt-v3.yaml", 30), ("gpt-configs/schemas/openapi-floor.yaml", 22)],
)
def test_gpt_schema_operation_counts_unchanged(path, expected):
    text = (ROOT / path).read_text()
    ops = re.findall(r"^\s*operationId:\s*\S+", text, flags=re.M)
    assert len(ops) == expected, f"{path}: {len(ops)} operationIds (expected {expected})"

"""Sales-order allocation schema and lifecycle coverage (migrations 044/045).

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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


class _AllocationConnProxy:
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
def allocation_client(db_cursor, monkeypatch):
    conn = db_cursor.connection

    @contextmanager
    def _fake_get_conn():
        proxy = _AllocationConnProxy(conn, "allocation_api")
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

try:
    import psycopg2
    from psycopg2 import errors as pg_errors
except Exception:  # pragma: no cover
    pytest.skip("psycopg2 not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "migrations" / "044_sales_order_allocations.sql"
RESTORE_MIGRATION = ROOT / "migrations" / "045_sales_order_allocation_reactivations.sql"


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

def _seed_order(cur, n_lines=2, product_ids=None):
    """customer + SO + `n_lines` lines (each on its own product unless given).
    Returns (order_id, [line_ids], [product_ids])."""
    cur.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM customers")
    customer_name = f"TEST-044 Customer {cur.fetchone()['next_id']}"
    cur.execute(
        "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
        (customer_name,),
    )
    customer_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status) "
        "VALUES (%s, %s, 'confirmed') RETURNING id",
        (customer_id, f"TEST-044-SO-{customer_id}"),
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
    """Addendum §3 uniqueness cycle using consume-from-live restore:
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

    # 4. restore consumes the recorded 40 from the current live row.  The
    #    void_coalesced row is immutable history and is never flipped.
    first_restore = main._consume_allocation_row(cur, leftover, 40, txn)

    # end state: one active 60 + one newly shipped 40; historical rows stay superseded
    live = live_rows()
    assert len(live) == 1 and live[0]["quantity_lb"] == 60
    assert row(shipped)["status"] == "superseded"
    new_shipped = row(first_restore["shipped_id"])
    assert new_shipped["status"] == "shipped" and new_shipped["quantity_lb"] == 40
    assert new_shipped["ship_transaction_id"] == txn
    cur.execute(
        "SELECT status, count(*) AS n, sum(quantity_lb) AS lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s GROUP BY status ORDER BY status",
        (line_a,),
    )
    by_status = {r["status"]: (r["n"], r["lb"]) for r in cur.fetchall()}
    assert by_status["active"] == (1, 60)
    assert by_status["shipped"] == (1, 40)

    # 5. a SECOND void/restore round still raises no unique violation (same shape)
    main._void_ship_allocations(cur, txn, "test")
    assert [r["quantity_lb"] for r in live_rows()] == [100]
    second_live = live_rows()[0]["id"]
    second_restore = main._consume_allocation_row(cur, second_live, 40, txn)
    assert [r["quantity_lb"] for r in live_rows()] == [60]
    assert row(second_restore["shipped_id"])["status"] == "shipped"
    assert row(shipped)["status"] == "superseded"


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


@pytest.mark.db
def test_restore_reactivation_migration_shape_and_idempotency(db_cursor):
    sql = RESTORE_MIGRATION.read_text()
    body = "\n".join(
        line for line in sql.splitlines()
        if line.strip().upper() not in ("BEGIN;", "COMMIT;")
    )
    cur = db_cursor
    cur.execute(body)
    cur.execute(body)
    cur.execute(
        """SELECT column_name, data_type, is_nullable
             FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='sales_order_allocation_reactivations'
            ORDER BY ordinal_position"""
    )
    assert [(r["column_name"], r["data_type"], r["is_nullable"]) for r in cur.fetchall()] == [
        ("transaction_id", "integer", "NO"),
        ("sales_order_line_id", "integer", "NO"),
        ("quantity_lb", "numeric", "NO"),
        ("correction_id", "uuid", "NO"),
        ("created_at", "timestamp with time zone", "NO"),
    ]
    low = sql.lower()
    assert "numeric(14,4)" in low
    assert "check (quantity_lb >= 0)" in low
    assert "primary key (transaction_id, sales_order_line_id)" in low
    assert "set session" not in low
    assert "business_date" not in re.sub(r"--[^\n]*", "", low)


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


# ─────────────────────────────────────────────────────────────────
# PR 3 helper/state-machine behavior (HTTP wiring is tested below)
# ─────────────────────────────────────────────────────────────────

def _post_stock(cur, product_id, lot_id, qty):
    cur.execute(
        "INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id"
    )
    transaction_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)",
        (transaction_id, product_id, lot_id, qty),
    )
    return transaction_id


def _allocation_checkpoint(
    cur,
    line_id,
    product_id,
    *,
    live_lb,
    shipped_lb,
    on_hand_lb,
    originally_allocated_lb=100,
):
    cur.execute(
        """SELECT status, COALESCE(SUM(quantity_lb), 0) AS quantity_lb
             FROM sales_order_allocations
            WHERE sales_order_line_id = %s
              AND status IN ('active', 'shipped')
            GROUP BY status""",
        (line_id,),
    )
    totals = {row["status"]: float(row["quantity_lb"]) for row in cur.fetchall()}
    actual_live = totals.get("active", 0.0)
    actual_shipped = totals.get("shipped", 0.0)
    assert actual_live == pytest.approx(live_lb)
    assert actual_shipped == pytest.approx(shipped_lb)
    assert actual_live + actual_shipped == pytest.approx(originally_allocated_lb)

    # Invariant 2: every posted-effective ship's SOA attribution is capped by
    # that transaction's effective ledger deduction for this product.
    cur.execute(
        """SELECT soa.ship_transaction_id,
                  SUM(soa.quantity_lb) AS allocated_shipped_lb,
                  (SELECT COALESCE(SUM(ABS(tl.quantity_lb)), 0)
                     FROM ledger_current_transaction_lines tl
                    WHERE tl.transaction_id = soa.ship_transaction_id
                      AND tl.product_id = %s) AS ledger_shipped_lb
             FROM sales_order_allocations soa
             JOIN ledger_current_transactions ct
               ON ct.id = soa.ship_transaction_id
              AND ct.type = 'ship'
              AND ct.effective_status = 'posted'
            WHERE soa.product_id = %s AND soa.status = 'shipped'
            GROUP BY soa.ship_transaction_id""",
        (product_id, product_id),
    )
    for row in cur.fetchall():
        assert float(row["allocated_shipped_lb"]) <= (
            float(row["ledger_shipped_lb"]) + main.BALANCE_EPSILON
        )

    # Invariant 3: product and every physical lot remain nonnegative.
    product_on_hand = main._product_on_hand(cur, product_id)
    assert product_on_hand == pytest.approx(on_hand_lb)
    assert product_on_hand >= -main.BALANCE_EPSILON
    for lot in main.fifo_lot_balances(cur, product_id, include_empty=True):
        assert float(lot["available"]) >= -main.BALANCE_EPSILON

    # Invariant 4: rows made historical by a split/coalesce remain historical.
    cur.execute(
        """SELECT count(*) AS n
             FROM sales_order_allocations
            WHERE sales_order_line_id = %s
              AND release_reason IN ('split_on_ship', 'void_coalesced')
              AND status <> 'superseded'""",
        (line_id,),
    )
    assert cur.fetchone()["n"] == 0


def _reactivation_quantity(cur, transaction_id, line_id):
    cur.execute(
        """SELECT quantity_lb
             FROM sales_order_allocation_reactivations
            WHERE transaction_id=%s AND sales_order_line_id=%s""",
        (transaction_id, line_id),
    )
    row = cur.fetchone()
    return None if row is None else float(row["quantity_lb"])


def _restore_atomic_state(cur, transaction_id, line_id, product_id):
    cur.execute(
        "SELECT effective_status FROM ledger_current_transactions WHERE id=%s",
        (transaction_id,),
    )
    effective_status = cur.fetchone()["effective_status"]
    cur.execute(
        "SELECT count(*) AS n FROM ledger_corrections WHERE target_table='transactions' AND target_id=%s",
        (transaction_id,),
    )
    correction_count = cur.fetchone()["n"]
    return {
        "allocations": _allocation_row_state(cur, line_id),
        "reactivated_lb": _reactivation_quantity(cur, transaction_id, line_id),
        "effective_status": effective_status,
        "correction_count": correction_count,
        "on_hand_lb": main._product_on_hand(cur, product_id),
    }


def _allocation_row_state(cur, line_id):
    cur.execute(
        """SELECT id, status, quantity_lb, ship_transaction_id,
                  last_ship_transaction_id, split_from_id, release_reason
             FROM sales_order_allocations
            WHERE sales_order_line_id = %s
            ORDER BY id""",
        (line_id,),
    )
    return [dict(row) for row in cur.fetchall()]


@pytest.mark.db
def test_available_lots_subtracts_lot_pins_then_shadows_foreign_sku_fifo(db_cursor):
    cur = db_cursor
    order_id, (line_a, line_b), (product_id, _) = _seed_order(cur, n_lines=2)
    cur.execute("UPDATE sales_order_lines SET product_id=%s WHERE id=%s", (product_id, line_b))
    lot_a = _seed_lot(cur, product_id, "T044-TAKEABLE-A")
    lot_b = _seed_lot(cur, product_id, "T044-TAKEABLE-B")
    _post_stock(cur, product_id, lot_a, 100)
    _post_stock(cur, product_id, lot_b, 100)
    _insert(cur, order_id, line_a, product_id, lot_id=lot_a, qty=30)
    _insert(cur, order_id, line_a, product_id, qty=40)

    lots = main.available_lots_for_product(cur, product_id, line_b)
    by_id = {row["lot_id"]: row for row in lots}
    assert by_id[lot_a]["reserved_others_lot"] == pytest.approx(30)
    assert by_id[lot_a]["takeable_unpinned"] == pytest.approx(70)
    assert by_id[lot_a]["foreign_sku_shadow_lb"] == pytest.approx(40)
    assert by_id[lot_a]["takeable"] == pytest.approx(30)
    assert by_id[lot_b]["takeable"] == pytest.approx(100)


@pytest.mark.db
def test_helper_full_void_then_restore_cycle(db_cursor):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-HELPER-CYCLE")
    _post_stock(cur, product_id, lot_id, 100)
    allocation_id = _insert(
        cur,
        order_id,
        line_id,
        product_id,
        lot_id=lot_id,
        qty=100,
        source="staged_lot",
    )

    plan = main._sales_order_ship_plan(cur, product_id, line_id, 40)
    assert plan["actual_ship_lb"] == pytest.approx(40)
    assert plan["lots"] == [{"lot_id": lot_id, "lot_code": "T044-HELPER-CYCLE", "quantity_lb": 40.0}]
    transaction_id = _seed_txn(cur)
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, -40)",
        (transaction_id, product_id, lot_id),
    )
    cur.execute(
        "INSERT INTO sales_order_shipments "
        "(sales_order_line_id, transaction_id, quantity_lb) VALUES (%s, %s, 40)",
        (line_id, transaction_id),
    )
    consumed = main._consume_sales_order_allocations(cur, plan, transaction_id)
    assert consumed[0]["allocation_id"] == allocation_id

    cur.execute(
        "SELECT status, quantity_lb, release_reason, ship_transaction_id "
        "FROM sales_order_allocations WHERE sales_order_line_id=%s ORDER BY id",
        (line_id,),
    )
    shipped_state = cur.fetchall()
    assert [(r["status"], float(r["quantity_lb"])) for r in shipped_state] == [
        ("superseded", 100.0), ("active", 60.0), ("shipped", 40.0)
    ]

    void_event = main._append_transaction_correction(
        cur, transaction_id, "void", "helper cycle void", None, "test"
    )
    restored = void_event["allocations_restored"]
    assert restored == [{
        "allocation_id": consumed[0]["shipped_id"],
        "live_allocation_id": consumed[0]["leftover_id"],
        "sales_order_line_id": line_id,
        "quantity_lb": 40.0,
        "coalesced": True,
    }]
    assert _reactivation_quantity(cur, transaction_id, line_id) == pytest.approx(40)
    cur.execute(
        "SELECT id, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND lot_id=%s AND status='active'",
        (line_id, lot_id),
    )
    live = cur.fetchall()
    assert len(live) == 1 and float(live[0]["quantity_lb"]) == pytest.approx(100)

    restore_event = main._append_transaction_correction(
        cur, transaction_id, "restore", "helper cycle restore", None, "test"
    )
    reshipped = restore_event["allocations_reshipped"]
    assert len(reshipped) == 1
    assert reshipped[0]["allocation_id"] == consumed[0]["leftover_id"]
    assert reshipped[0]["quantity_lb"] == pytest.approx(40)
    assert reshipped[0]["sales_order_line_id"] == line_id
    cur.execute(
        "SELECT status, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status IN ('active', 'shipped') ORDER BY status",
        (line_id,),
    )
    assert [(r["status"], float(r["quantity_lb"])) for r in cur.fetchall()] == [
        ("active", 60.0), ("shipped", 40.0)
    ]


@pytest.mark.db
def test_next_product_write_persists_expired_auto_fifo_release(db_cursor):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-EXPIRED-WRITE")
    _post_stock(cur, product_id, lot_id, 100)
    allocation_id = _insert(
        cur,
        order_id,
        line_id,
        product_id,
        lot_id=lot_id,
        qty=25,
        source="auto_fifo",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    plan = main._sales_order_ship_plan(cur, product_id, line_id, 10, released_by="test")
    assert plan["actual_ship_lb"] == pytest.approx(10)
    assert plan["allocation_takes"] == []
    cur.execute(
        "SELECT status, release_reason, released_at FROM sales_order_allocations WHERE id=%s",
        (allocation_id,),
    )
    row = cur.fetchone()
    assert row["status"] == "released"
    assert row["release_reason"] == "expired"
    assert row["released_at"] is not None


@pytest.mark.db
def test_void_inventory_releases_uncovered_lot_pin_even_if_product_total_is_covered(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=50 WHERE id=%s", (line_id,))
    lot_a = _seed_lot(cur, product_id, "T044-VOID-STOCK-A")
    lot_b = _seed_lot(cur, product_id, "T044-VOID-STOCK-B")
    receive_a = _post_stock(cur, product_id, lot_a, 50)
    _post_stock(cur, product_id, lot_b, 50)
    allocation_id = _insert(
        cur, order_id, line_id, product_id, lot_id=lot_a, qty=50, source="staged_lot"
    )

    voided = allocation_client.post(
        f"/void/{receive_a}", json={"reason": "remove pinned lot stock"}
    )
    assert voided.status_code == 200, voided.text
    assert sum(
        float(row["quantity_lb"]) for row in voided.json()["allocations_released"]
    ) == pytest.approx(50)
    cur.execute(
        "SELECT status, release_reason FROM sales_order_allocations WHERE id=%s",
        (allocation_id,),
    )
    assert cur.fetchone() == {"status": "released", "release_reason": "inventory_voided"}


@pytest.mark.db
def test_http_manual_upsert_release_and_sibling_overallocation(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_a, line_b), (product_id, _) = _seed_order(cur, n_lines=2)
    cur.execute(
        "UPDATE sales_order_lines SET product_id=%s, quantity_lb=80 WHERE id = ANY(%s)",
        (product_id, [line_a, line_b]),
    )
    lot_id = _seed_lot(cur, product_id, "T044-HTTP-MANUAL")
    _post_stock(cur, product_id, lot_id, 100)

    first = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_a, "quantity_lb": 30},
    )
    assert first.status_code == 200, first.text
    allocation_id = first.json()["allocations"][0]["id"]
    second = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_a, "quantity_lb": 50},
    )
    assert second.status_code == 200, second.text
    assert second.json()["allocations"][0]["id"] == allocation_id
    assert second.json()["line_readiness"]["allocated_lb"] == pytest.approx(80)
    assert second.json()["line_readiness"]["inventory_ready"] is True
    cur.execute(
        "SELECT count(*) AS n, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status='active' GROUP BY quantity_lb",
        (line_a,),
    )
    live = cur.fetchone()
    assert live["n"] == 1 and float(live["quantity_lb"]) == pytest.approx(80)

    rejected = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_b, "quantity_lb": 30},
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["error_code"] == "OVER_ALLOCATION"
    assert rejected.json()["detail"]["coverable_lb"] == pytest.approx(20)

    listing = allocation_client.get(f"/sales/orders/{order_id}/allocations")
    assert listing.status_code == 200, listing.text
    assert [row["id"] for row in listing.json()["allocations"]] == [allocation_id]

    released = allocation_client.post(
        f"/sales/orders/{order_id}/allocations/{allocation_id}/release"
    )
    assert released.status_code == 200, released.text
    cur.execute(
        "SELECT status, release_reason FROM sales_order_allocations WHERE id=%s",
        (allocation_id,),
    )
    assert cur.fetchone() == {"status": "released", "release_reason": "manual_release"}


@pytest.mark.db
def test_dashboard_scoped_key_can_use_allocation_and_received_at_routes(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=25 WHERE id=%s", (line_id,))
    lot_id = _seed_lot(cur, product_id, "STAGED-T044-DASHBOARD-KEY")
    cur.execute("UPDATE lots SET entry_source='found_inventory' WHERE id=%s", (lot_id,))
    _post_stock(cur, product_id, lot_id, 25)
    allocation_client.headers["X-API-Key"] = main.DASHBOARD_API_KEY

    created = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 25, "lot_id": lot_id},
    )
    assert created.status_code == 200, created.text
    allocation_id = created.json()["allocations"][0]["id"]
    assert created.json()["allocations"][0]["created_by"] == "dashboard"
    assert created.json()["allocations"][0]["source"] == "staged_lot"
    assert allocation_client.get(f"/sales/orders/{order_id}/allocations").status_code == 200
    patched = allocation_client.patch(
        f"/lots/{lot_id}/received-at", json={"received_at": "2026-08-14T12:00:00-04:00"}
    )
    assert patched.status_code == 200, patched.text
    released = allocation_client.post(
        f"/sales/orders/{order_id}/allocations/{allocation_id}/release"
    )
    assert released.status_code == 200, released.text


@pytest.mark.db
def test_http_auto_fifo_splits_lots_and_sets_48_hour_ttl(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=150 WHERE id=%s", (line_id,))
    lot_a = _seed_lot(cur, product_id, "T044-AUTO-A")
    lot_b = _seed_lot(cur, product_id, "T044-AUTO-B")
    _post_stock(cur, product_id, lot_a, 60)
    _post_stock(cur, product_id, lot_b, 90)

    before = datetime.now(timezone.utc)
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "auto_fifo", "line_id": line_id},
    )
    assert response.status_code == 200, response.text
    rows = response.json()["allocations"]
    assert [(row["lot_id"], float(row["quantity_lb"])) for row in rows] == [
        (lot_a, 60.0), (lot_b, 90.0)
    ]
    cur.execute(
        "SELECT source, expires_at FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s ORDER BY lot_id",
        (line_id,),
    )
    persisted = cur.fetchall()
    assert {row["source"] for row in persisted} == {"auto_fifo"}
    for row in persisted:
        assert before + timedelta(hours=47, minutes=59) < row["expires_at"]
        assert row["expires_at"] < before + timedelta(hours=48, minutes=1)


@pytest.mark.db
def test_http_allocate_ship_40_void_restore_cycle(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-E2E-CYCLE")
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={
            "mode": "manual",
            "line_id": line_id,
            "quantity_lb": 100,
            "lot_id": lot_id,
            "source": "staged_lot",
        },
    )
    assert allocated.status_code == 200, allocated.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert shipped.status_code == 200, shipped.text
    ship_line = shipped.json()["lines_shipped"][0]
    transaction_id = ship_line["transaction_id"]
    assert ship_line["lots_used"] == [{"lot_code": "T044-E2E-CYCLE", "quantity_lb": 40.0}]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )

    voided = allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "allocation cycle test"}
    )
    assert voided.status_code == 200, voided.text
    assert voided.json()["allocations_restored"][0]["coalesced"] is True
    assert _reactivation_quantity(cur, transaction_id, line_id) == pytest.approx(40)
    cur.execute(
        "SELECT quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND lot_id=%s AND status='active'",
        (line_id, lot_id),
    )
    live = cur.fetchall()
    assert len(live) == 1 and float(live[0]["quantity_lb"]) == pytest.approx(100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "allocation cycle restore"},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["allocations_reshipped"][0]["quantity_lb"] == pytest.approx(40)
    cur.execute(
        "SELECT status, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status IN ('active', 'shipped') ORDER BY status",
        (line_id,),
    )
    assert [(row["status"], float(row["quantity_lb"])) for row in cur.fetchall()] == [
        ("active", 60.0), ("shipped", 40.0)
    ]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )

    # The same transaction may be voided again after a successful restore.
    # Its PK row is overwritten with this void's actual reactivation quantity.
    voided_again = allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "allocation cycle second void"}
    )
    assert voided_again.status_code == 200, voided_again.text
    assert _reactivation_quantity(cur, transaction_id, line_id) == pytest.approx(40)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )
    restored_again = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "allocation cycle second restore"},
    )
    assert restored_again.status_code == 200, restored_again.text
    assert sum(
        float(row["quantity_lb"])
        for row in restored_again.json()["allocations_reshipped"]
    ) == pytest.approx(40)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )


@pytest.mark.db
def test_sku_level_allocation_is_consumed_and_split_on_ship(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-SKU-CONSUME")
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100},
    )
    assert allocated.status_code == 200, allocated.text
    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert shipped.status_code == 200, shipped.text
    cur.execute(
        "SELECT status, lot_id, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status IN ('active', 'shipped') ORDER BY status",
        (line_id,),
    )
    rows = cur.fetchall()
    assert [(row["status"], row["lot_id"], float(row["quantity_lb"])) for row in rows] == [
        ("active", None, 60.0), ("shipped", None, 40.0)
    ]


@pytest.mark.db
def test_merge_coalesces_two_live_pins_onto_surviving_lot(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_a = _seed_lot(cur, product_id, "T044-MERGE-A")
    lot_b = _seed_lot(cur, product_id, "T044-MERGE-B")
    _post_stock(cur, product_id, lot_a, 40)
    _post_stock(cur, product_id, lot_b, 60)
    _insert(cur, order_id, line_id, product_id, lot_id=lot_a, qty=40, source="staged_lot")
    _insert(cur, order_id, line_id, product_id, lot_id=lot_b, qty=60, source="staged_lot")

    merged = allocation_client.post(
        "/admin/lots/merge",
        json={"source_lot_id": lot_a, "target_lot_id": lot_b, "reason": "allocation merge test"},
    )
    assert merged.status_code == 200, merged.text
    cur.execute(
        "SELECT id, lot_id, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status='active'",
        (line_id,),
    )
    live = cur.fetchall()
    assert len(live) == 1
    assert live[0]["lot_id"] == lot_b
    assert float(live[0]["quantity_lb"]) == pytest.approx(100)
    cur.execute(
        "SELECT release_reason FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status='superseded'",
        (line_id,),
    )
    assert cur.fetchone()["release_reason"] == "lot_merged"


@pytest.mark.db
def test_received_at_patch_clears_missing_date_and_validates(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "STAGED-T044-PATCH")
    cur.execute("UPDATE lots SET entry_source='found_inventory', received_at=NULL WHERE id=%s", (lot_id,))
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={
            "mode": "manual", "line_id": line_id, "quantity_lb": 100,
            "lot_id": lot_id, "source": "staged_lot",
        },
    )
    assert allocated.status_code == 200, allocated.text
    assert "missing_lot_dates" in {
        blocker["code"] for blocker in allocated.json()["line_readiness"]["blockers"]
    }

    naive = allocation_client.patch(
        f"/lots/{lot_id}/received-at", json={"received_at": "2026-08-14T12:00:00"}
    )
    assert naive.status_code == 422
    assert naive.json()["detail"]["error_code"] == "INVALID_RECEIVED_AT"
    null_value = allocation_client.patch(
        f"/lots/{lot_id}/received-at", json={"received_at": None}
    )
    assert null_value.status_code == 422
    assert null_value.json()["detail"]["error_code"] == "RECEIVED_AT_REQUIRED"
    future = allocation_client.patch(
        f"/lots/{lot_id}/received-at",
        json={"received_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
    )
    assert future.status_code == 422
    assert future.json()["detail"]["error_code"] == "RECEIVED_AT_IN_FUTURE"
    missing = allocation_client.patch(
        "/lots/2000000000/received-at", json={"received_at": "2026-08-14T12:00:00-04:00"}
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_code"] == "LOT_NOT_FOUND"

    patched = allocation_client.patch(
        f"/lots/{lot_id}/received-at", json={"received_at": "2026-08-14T12:00:00-04:00"}
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["lot_code"] == "STAGED-T044-PATCH"
    assert patched.json()["lot_is_incomplete"] is False
    detail = allocation_client.get(f"/sales/orders/{order_id}")
    codes = {blocker["code"] for blocker in detail.json()["lines"][0]["readiness"]["blockers"]}
    assert "missing_lot_dates" not in codes
    assert "unstaged" not in codes


@pytest.mark.db
def test_line_edit_shrinks_then_order_cancel_releases_allocations(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-EDIT-CANCEL")
    _post_stock(cur, product_id, lot_id, 100)
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100},
    )
    assert response.status_code == 200, response.text

    edited = allocation_client.patch(
        f"/sales/orders/{order_id}/lines/{line_id}/update?quantity_lb=60"
    )
    assert edited.status_code == 200, edited.text
    assert sum(float(row["quantity_lb"]) for row in edited.json()["allocations_released"]) == pytest.approx(40)
    cur.execute(
        "SELECT quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status='active'",
        (line_id,),
    )
    assert float(cur.fetchone()["quantity_lb"]) == pytest.approx(60)

    cancelled = allocation_client.patch(
        f"/sales/orders/{order_id}/status", json={"status": "cancelled"}
    )
    assert cancelled.status_code == 200, cancelled.text
    assert sum(float(row["quantity_lb"]) for row in cancelled.json()["allocations_released"]) == pytest.approx(60)
    cur.execute(
        "SELECT count(*) AS n FROM sales_order_allocations "
        "WHERE sales_order_id=%s AND status='active'",
        (order_id,),
    )
    assert cur.fetchone()["n"] == 0


@pytest.mark.db
def test_line_edit_rejects_quantity_below_effective_shipped(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-EDIT-SHIPPED-GUARD")
    _post_stock(cur, product_id, lot_id, 100)
    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert shipped.status_code == 200, shipped.text
    rejected = allocation_client.patch(
        f"/sales/orders/{order_id}/lines/{line_id}/update?quantity_lb=39"
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["error_code"] == "QTY_BELOW_SHIPPED_EFFECTIVE"


@pytest.mark.db
def test_line_cancel_releases_allocation(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-LINE-CANCEL")
    _post_stock(cur, product_id, lot_id, 50)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=50 WHERE id=%s", (line_id,))
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 50},
    )
    assert response.status_code == 200, response.text
    cancelled = allocation_client.patch(
        f"/sales/orders/{order_id}/lines/{line_id}/cancel"
    )
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["allocations_released"][0]["quantity_lb"] == 50.0
    cur.execute(
        "SELECT status, release_reason FROM sales_order_allocations WHERE sales_order_line_id=%s",
        (line_id,),
    )
    assert cur.fetchone() == {"status": "released", "release_reason": "line_cancelled"}


# ─────────────────────────────────────────────────────────────────
# PR 4 standalone ship / pack takeable and observe-only behavior
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_standalone_ship_preview_reports_lot_and_sku_reservations_and_ignores_expired(
    db_cursor, allocation_client
):
    cur = db_cursor
    lot_order, (lot_line,), (product_id,) = _seed_order(cur, n_lines=1)
    sku_order, (sku_line,), _ = _seed_order(
        cur, n_lines=1, product_ids=[product_id]
    )
    expired_order, (expired_line,), _ = _seed_order(
        cur, n_lines=1, product_ids=[product_id]
    )
    lot_a = _seed_lot(cur, product_id, "T044-PR4-SHIP-A")
    lot_b = _seed_lot(cur, product_id, "T044-PR4-SHIP-B")
    _post_stock(cur, product_id, lot_a, 100)
    _post_stock(cur, product_id, lot_b, 100)

    assert allocation_client.post(
        f"/sales/orders/{lot_order}/allocations",
        json={"mode": "manual", "line_id": lot_line, "quantity_lb": 30, "lot_id": lot_a},
    ).status_code == 200
    assert allocation_client.post(
        f"/sales/orders/{sku_order}/allocations",
        json={"mode": "manual", "line_id": sku_line, "quantity_lb": 40},
    ).status_code == 200
    expired_id = _insert(
        cur,
        expired_order,
        expired_line,
        product_id,
        lot_id=lot_b,
        qty=25,
        source="auto_fifo",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    cur.execute("SELECT name FROM products WHERE id=%s", (product_id,))
    product_name = cur.fetchone()["name"]
    cur.execute(
        "SELECT id, order_number FROM sales_orders WHERE id IN (%s, %s) ORDER BY id",
        (lot_order, sku_order),
    )
    expected_orders = [
        {"order_number": row["order_number"], "quantity_lb": qty}
        for row, qty in zip(cur.fetchall(), (30.0, 40.0))
    ]

    preview = allocation_client.post(
        "/ship/preview",
        json={
            "product_name": product_name,
            "quantity_lb": 200,
            "customer_name": "T044 PR4 Preview Customer",
            "order_reference": "T044-PR4-PREVIEW",
            "force_standalone": True,
        },
    )
    assert preview.status_code == 200, preview.text
    payload = preview.json()
    assert payload["total_available_lb"] == pytest.approx(200)
    assert payload["total_takeable_lb"] == pytest.approx(130)
    assert payload["can_ship_lb"] == pytest.approx(130)
    assert payload["reserved_others_lb"] == pytest.approx(70)
    assert payload["reserved_by_orders"] == expected_orders
    assert payload["allocation_warning"]["warning_code"] == "RESERVED_STOCK_OBSERVE_ONLY"
    assert payload["allocation_warning"]["reserved_taken_lb"] == pytest.approx(70)
    by_lot = {row["lot_code"]: row for row in payload["allocations"]}
    assert by_lot["T044-PR4-SHIP-A"]["takeable_lb"] == pytest.approx(30)
    assert by_lot["T044-PR4-SHIP-B"]["takeable_lb"] == pytest.approx(100)
    cur.execute(
        "SELECT status, release_reason FROM sales_order_allocations WHERE id=%s",
        (expired_id,),
    )
    assert cur.fetchone() == {"status": "active", "release_reason": None}


@pytest.mark.db
def test_order_ship_preview_excludes_own_allocation_but_counts_competing_order(
    db_cursor, allocation_client
):
    cur = db_cursor
    own_order, (own_line,), (product_id,) = _seed_order(cur, n_lines=1)
    other_order, (other_line,), _ = _seed_order(
        cur, n_lines=1, product_ids=[product_id]
    )
    lot_id = _seed_lot(cur, product_id, "T044-PR4-OWN-EXCLUSION")
    _post_stock(cur, product_id, lot_id, 100)
    assert allocation_client.post(
        f"/sales/orders/{own_order}/allocations",
        json={"mode": "manual", "line_id": own_line, "quantity_lb": 60, "lot_id": lot_id},
    ).status_code == 200
    assert allocation_client.post(
        f"/sales/orders/{other_order}/allocations",
        json={"mode": "manual", "line_id": other_line, "quantity_lb": 20},
    ).status_code == 200
    cur.execute("SELECT order_number FROM sales_orders WHERE id=%s", (other_order,))
    other_number = cur.fetchone()["order_number"]

    preview = allocation_client.post(
        f"/sales/orders/{own_order}/ship/preview",
        json={"ship_all": False, "lines": [{"line_id": own_line, "quantity_lb": 100}]},
    )
    assert preview.status_code == 200, preview.text
    line = preview.json()["lines"][0]
    assert line["can_ship_lb"] == pytest.approx(80)
    assert line["reserved_others_lb"] == pytest.approx(20)
    assert line["reserved_by_orders"] == [
        {"order_number": other_number, "quantity_lb": 20.0}
    ]


@pytest.mark.db
def test_standalone_ship_observe_mode_proceeds_warns_and_shrinks_stolen_pin(
    db_cursor, allocation_client
):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-PR4-SHIP-SHRINK")
    _post_stock(cur, product_id, lot_id, 100)
    assert allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 60, "lot_id": lot_id},
    ).status_code == 200
    cur.execute("SELECT name FROM products WHERE id=%s", (product_id,))
    product_name = cur.fetchone()["name"]

    shipped = allocation_client.post(
        "/ship/commit",
        json={
            "product_name": product_name,
            "quantity_lb": 80,
            "customer_name": f"T044 PR4 Standalone {order_id}",
            "order_reference": f"T044-PR4-SHIP-{order_id}",
            "force_standalone": True,
            "force_create_customer": True,
        },
    )
    assert shipped.status_code == 200, shipped.text
    payload = shipped.json()
    assert payload["can_ship_lb"] == pytest.approx(40)
    assert payload["reserved_others_lb"] == pytest.approx(60)
    assert payload["allocation_warning"]["reserved_taken_lb"] == pytest.approx(40)
    assert sum(row["quantity_lb"] for row in payload["allocations_released"]) == pytest.approx(40)
    assert {row["reason"] for row in payload["allocations_released"]} == {"inventory_shipped"}
    assert main._product_on_hand(cur, product_id) == pytest.approx(20)
    cur.execute(
        "SELECT quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status='active'",
        (line_id,),
    )
    assert float(cur.fetchone()["quantity_lb"]) == pytest.approx(20)


@pytest.mark.db
def test_pack_source_observe_mode_warns_and_target_allocation_does_not_block(
    db_cursor, allocation_client
):
    cur = db_cursor
    order_id, (source_line, target_line), (source_id, target_id) = _seed_order(
        cur, n_lines=2
    )
    source_lot = _seed_lot(cur, source_id, "T044-PR4-PACK-SOURCE")
    target_stock_lot = _seed_lot(cur, target_id, "T044-PR4-PACK-TARGET-STOCK")
    _post_stock(cur, source_id, source_lot, 100)
    _post_stock(cur, target_id, target_stock_lot, 50)
    cur.execute("UPDATE products SET case_size_lb=10 WHERE id=%s", (target_id,))
    assert allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": source_line, "quantity_lb": 60, "lot_id": source_lot},
    ).status_code == 200
    assert allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": target_line, "quantity_lb": 50, "lot_id": target_stock_lot},
    ).status_code == 200
    cur.execute("SELECT id, name FROM products WHERE id IN (%s, %s)", (source_id, target_id))
    names = {row["id"]: row["name"] for row in cur.fetchall()}
    request = {
        "source_product": names[source_id],
        "target_product": names[target_id],
        "cases": 8,
        "case_weight_lb": 10,
        "target_lot_code": "T044-PR4-PACK-OUTPUT",
    }

    preview = allocation_client.post("/pack/preview", json=request)
    assert preview.status_code == 200, preview.text
    preview_payload = preview.json()
    assert preview_payload["total_batch_available_lb"] == pytest.approx(100)
    assert preview_payload["can_pack_lb"] == pytest.approx(40)
    assert preview_payload["reserved_others_lb"] == pytest.approx(60)
    assert preview_payload["allocation_warning"]["reserved_taken_lb"] == pytest.approx(40)

    packed = allocation_client.post("/pack/commit", json=request)
    assert packed.status_code == 200, packed.text
    payload = packed.json()
    assert payload["can_pack_lb"] == pytest.approx(40)
    assert payload["reserved_others_lb"] == pytest.approx(60)
    assert payload["allocation_warning"]["reserved_taken_lb"] == pytest.approx(40)
    assert main._product_on_hand(cur, source_id) == pytest.approx(20)
    assert main._product_on_hand(cur, target_id) == pytest.approx(130)
    cur.execute(
        "SELECT quantity_lb, status FROM sales_order_allocations WHERE sales_order_line_id=%s",
        (source_line,),
    )
    source_allocation = cur.fetchone()
    assert source_allocation == {"quantity_lb": 60, "status": "active"}


@pytest.mark.db
def test_service_line_allocate_returns_service_line_not_allocatable(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE products SET is_service=true WHERE id=%s", (product_id,))
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 1},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "SERVICE_LINE_NOT_ALLOCATABLE"


@pytest.mark.db
@pytest.mark.parametrize("order_status", ["cancelled", "invoiced"])
def test_cancelled_and_invoiced_order_allocate_returns_order_not_allocatable(
    db_cursor, allocation_client, order_status
):
    cur = db_cursor
    order_id, (line_id,), _ = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_orders SET status=%s WHERE id=%s", (order_status, order_id))
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 1},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "ORDER_NOT_ALLOCATABLE"


@pytest.mark.db
def test_sku_and_lot_allocations_share_product_on_hand_limit(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=200 WHERE id=%s", (line_id,))
    lot_id = _seed_lot(cur, product_id, "T044-SKU-LOT-SUM")
    _post_stock(cur, product_id, lot_id, 150)
    assert allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100},
    ).status_code == 200
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 60, "lot_id": lot_id},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "OVER_ALLOCATION"
    assert response.json()["detail"]["coverable_lb"] == pytest.approx(50)


@pytest.mark.db
def test_allocate_above_remaining_effective_is_rejected_with_huge_on_hand(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-LINE-COVERABLE")
    _post_stock(cur, product_id, lot_id, 10000)
    response = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 101},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "OVER_ALLOCATION"
    assert response.json()["detail"]["coverable_lb"] == pytest.approx(100)


@pytest.mark.db
def test_lot_pin_on_one_order_blocks_second_order_from_same_stock(db_cursor, allocation_client):
    cur = db_cursor
    order_one, (line_one,), (product_id,) = _seed_order(cur, n_lines=1)
    order_two, (line_two,), _ = _seed_order(cur, n_lines=1, product_ids=[product_id])
    lot_id = _seed_lot(cur, product_id, "T044-CROSS-ORDER")
    _post_stock(cur, product_id, lot_id, 100)
    assert allocation_client.post(
        f"/sales/orders/{order_one}/allocations",
        json={"mode": "manual", "line_id": line_one, "quantity_lb": 100, "lot_id": lot_id},
    ).status_code == 200
    response = allocation_client.post(
        f"/sales/orders/{order_two}/allocations",
        json={"mode": "manual", "line_id": line_two, "quantity_lb": 1, "lot_id": lot_id},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "OVER_ALLOCATION"


@pytest.mark.db
def test_void_partial_reship_restore_keeps_leftover_active_and_reconciles_ledger(db_cursor, allocation_client):
    """S1: stock-first guard wins when both stock and coverage are only 60."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=200 WHERE id=%s", (line_id,))
    lot_id = _seed_lot(cur, product_id, "T044-VOID-RESHIP-RESTORE")
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100, "lot_id": lot_id},
    )
    assert allocated.status_code == 200, allocated.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    ship_a = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert ship_a.status_code == 200, ship_a.text
    txn_a = ship_a.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=100, on_hand_lb=0
    )

    void_a = allocation_client.post(f"/void/{txn_a}", json={"reason": "void A"})
    assert void_a.status_code == 200, void_a.text
    assert _reactivation_quantity(cur, txn_a, line_id) == pytest.approx(100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    ship_b = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert ship_b.status_code == 200, ship_b.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )
    before_restore = _restore_atomic_state(cur, txn_a, line_id, product_id)

    restore_a = allocation_client.post(
        f"/records/transactions/{txn_a}/corrections",
        json={"event_type": "restore", "reason": "restore A without coverage"},
    )
    assert restore_a.status_code == 409, restore_a.text
    detail = restore_a.json()["detail"]
    assert detail["error_code"] == "RESTORE_STOCK_MISSING"
    assert detail["lot_id"] == lot_id
    assert detail["lot_code"] == "T044-VOID-RESHIP-RESTORE"
    assert detail["required_lb"] == pytest.approx(100)
    assert detail["available_lb"] == pytest.approx(60)
    assert _restore_atomic_state(cur, txn_a, line_id, product_id) == before_restore
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )


@pytest.mark.db
def test_void_partial_reship_restore_reports_split_missing_when_stock_covers(db_cursor, allocation_client):
    """S1b: 200 stock leaves 160 on-hand, isolating live coverage 60 < 100."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=200 WHERE id=%s", (line_id,))
    lot_id = _seed_lot(cur, product_id, "T044-VOID-RESHIP-SPLIT-MISSING")
    _post_stock(cur, product_id, lot_id, 200)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100, "lot_id": lot_id},
    )
    assert allocated.status_code == 200, allocated.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=200
    )

    ship_a = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert ship_a.status_code == 200, ship_a.text
    txn_a = ship_a.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=100, on_hand_lb=100
    )

    void_a = allocation_client.post(f"/void/{txn_a}", json={"reason": "S1b void A"})
    assert void_a.status_code == 200, void_a.text
    assert _reactivation_quantity(cur, txn_a, line_id) == pytest.approx(100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=200
    )

    ship_b = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert ship_b.status_code == 200, ship_b.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=160
    )
    before_restore = _restore_atomic_state(cur, txn_a, line_id, product_id)

    restore_a = allocation_client.post(
        f"/records/transactions/{txn_a}/corrections",
        json={"event_type": "restore", "reason": "S1b isolate coverage"},
    )
    assert restore_a.status_code == 409, restore_a.text
    detail = restore_a.json()["detail"]
    assert detail["error_code"] == "RESTORE_SPLIT_MISSING"
    assert detail["sales_order_line_id"] == line_id
    assert detail["required_lb"] == pytest.approx(100)
    assert detail["available_lb"] == pytest.approx(60)
    assert _restore_atomic_state(cur, txn_a, line_id, product_id) == before_restore
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=160
    )


@pytest.mark.db
def test_void_b_then_restore_a_consumes_coalesced_live_quantity(db_cursor, allocation_client):
    """S2: once B is voided, restore A consumes the full live 100."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=200 WHERE id=%s", (line_id,))
    lot_id = _seed_lot(cur, product_id, "T044-VOID-B-RESTORE-A")
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100, "lot_id": lot_id},
    )
    assert allocated.status_code == 200, allocated.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    ship_a = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert ship_a.status_code == 200, ship_a.text
    txn_a = ship_a.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=100, on_hand_lb=0
    )
    void_a = allocation_client.post(f"/void/{txn_a}", json={"reason": "void A"})
    assert void_a.status_code == 200, void_a.text
    assert _reactivation_quantity(cur, txn_a, line_id) == pytest.approx(100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    ship_b = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert ship_b.status_code == 200, ship_b.text
    txn_b = ship_b.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )
    void_b = allocation_client.post(f"/void/{txn_b}", json={"reason": "void B"})
    assert void_b.status_code == 200, void_b.text
    assert _reactivation_quantity(cur, txn_b, line_id) == pytest.approx(40)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    restore_a = allocation_client.post(
        f"/records/transactions/{txn_a}/corrections",
        json={"event_type": "restore", "reason": "restore A with full coverage"},
    )
    assert restore_a.status_code == 200, restore_a.text
    assert sum(
        float(row["quantity_lb"])
        for row in restore_a.json()["allocations_reshipped"]
    ) == pytest.approx(100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=100, on_hand_lb=0
    )
    cur.execute(
        "SELECT ship_transaction_id, quantity_lb FROM sales_order_allocations "
        "WHERE sales_order_line_id=%s AND status='shipped'",
        (line_id,),
    )
    assert [(row["ship_transaction_id"], float(row["quantity_lb"])) for row in cur.fetchall()] == [
        (txn_a, 100.0)
    ]

    # Both reservation coverage and stock are now absent for B.  Handler order
    # is normative: the lot-level stock preflight must fail first.
    before_restore_b = _restore_atomic_state(cur, txn_b, line_id, product_id)
    restore_b = allocation_client.post(
        f"/records/transactions/{txn_b}/corrections",
        json={"event_type": "restore", "reason": "restore B after A"},
    )
    assert restore_b.status_code == 409, restore_b.text
    assert restore_b.json()["detail"]["error_code"] == "RESTORE_STOCK_MISSING"
    assert restore_b.json()["detail"]["required_lb"] == pytest.approx(40)
    assert restore_b.json()["detail"]["available_lb"] == pytest.approx(0)
    assert _restore_atomic_state(cur, txn_b, line_id, product_id) == before_restore_b
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=100, on_hand_lb=0
    )


@pytest.mark.db
def test_partial_void_reship_restore_consumes_recorded_reactivation_only(db_cursor, allocation_client):
    """S3: record 40 (not coalesced live 100); restore T consumes that 40."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    cur.execute("UPDATE sales_order_lines SET quantity_lb=200 WHERE id=%s", (line_id,))
    lot_id = _seed_lot(cur, product_id, "T044-PARTIAL-RESTORE-LEDGER-QTY")
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 100, "lot_id": lot_id},
    )
    assert allocated.status_code == 200, allocated.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    ship_t = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 40}]},
    )
    assert ship_t.status_code == 200, ship_t.text
    txn_t = ship_t.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=60, shipped_lb=40, on_hand_lb=60
    )
    void_t = allocation_client.post(f"/void/{txn_t}", json={"reason": "void T"})
    assert void_t.status_code == 200, void_t.text
    assert _reactivation_quantity(cur, txn_t, line_id) == pytest.approx(40)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=100, shipped_lb=0, on_hand_lb=100
    )

    ship_c = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 30}]},
    )
    assert ship_c.status_code == 200, ship_c.text
    txn_c = ship_c.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=70, shipped_lb=30, on_hand_lb=70
    )

    restore_t = allocation_client.post(
        f"/records/transactions/{txn_t}/corrections",
        json={"event_type": "restore", "reason": "restore T recorded quantity"},
    )
    assert restore_t.status_code == 200, restore_t.text
    assert sum(
        float(row["quantity_lb"])
        for row in restore_t.json()["allocations_reshipped"]
    ) == pytest.approx(40)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=30, shipped_lb=70, on_hand_lb=30
    )
    cur.execute(
        """SELECT ship_transaction_id, quantity_lb
             FROM sales_order_allocations
            WHERE sales_order_line_id=%s AND status='shipped'
            ORDER BY ship_transaction_id""",
        (line_id,),
    )
    attributed = [
        (row["ship_transaction_id"], float(row["quantity_lb"]))
        for row in cur.fetchall()
    ]
    assert sorted(attributed) == sorted([(txn_t, 40.0), (txn_c, 30.0)])
    assert max(qty for txn_id, qty in attributed if txn_id == txn_t) <= 40


@pytest.mark.db
def test_unallocated_ship_void_restore_records_zero_and_succeeds(db_cursor, allocation_client):
    """S4: stored zero is known and restores ledger stock with no SOA effect."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-UNALLOCATED-RESTORE")
    _post_stock(cur, product_id, lot_id, 100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )

    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert shipped.status_code == 200, shipped.text
    transaction_id = shipped.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )

    voided = allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "S4 unallocated void"}
    )
    assert voided.status_code == 200, voided.text
    assert _reactivation_quantity(cur, transaction_id, line_id) == pytest.approx(0)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "S4 unallocated restore"},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["allocations_reshipped"] == []
    assert "allocation_reactivation_unknown" not in restored.json()
    assert "allocation_reactivation_unknown_line_ids" not in restored.json()
    assert main._line_shipped_effective(cur, line_id, product_id) == pytest.approx(100)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )


@pytest.mark.db
def test_partial_allocation_ship_void_restore_consumes_recorded_fifty(db_cursor, allocation_client):
    """S5: ship 100 with 50 reserved; record and restore only those 50."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-PARTIAL-ALLOCATION-RESTORE")
    _post_stock(cur, product_id, lot_id, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 50, "lot_id": lot_id},
    )
    assert allocated.status_code == 200, allocated.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=50, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=50,
    )

    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert shipped.status_code == 200, shipped.text
    transaction_id = shipped.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=50,
        on_hand_lb=0, originally_allocated_lb=50,
    )

    voided = allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "S5 partial allocation void"}
    )
    assert voided.status_code == 200, voided.text
    assert _reactivation_quantity(cur, transaction_id, line_id) == pytest.approx(50)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=50, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=50,
    )

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "S5 partial allocation restore"},
    )
    assert restored.status_code == 200, restored.text
    assert sum(
        float(row["quantity_lb"])
        for row in restored.json()["allocations_reshipped"]
    ) == pytest.approx(50)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=50,
        on_hand_lb=0, originally_allocated_lb=50,
    )


@pytest.mark.db
def test_restore_missing_reactivation_record_warns_and_uses_zero(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-MISSING-REACTIVATION")
    _post_stock(cur, product_id, lot_id, 100)
    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert shipped.status_code == 200, shipped.text
    transaction_id = shipped.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    voided = allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "pre-mechanism void"}
    )
    assert voided.status_code == 200, voided.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )
    cur.execute(
        "DELETE FROM sales_order_allocation_reactivations WHERE transaction_id=%s",
        (transaction_id,),
    )
    assert _reactivation_quantity(cur, transaction_id, line_id) is None

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "pre-mechanism restore"},
    )
    assert restored.status_code == 200, restored.text
    payload = restored.json()
    assert payload["allocations_reshipped"] == []
    assert payload["allocation_reactivation_unknown"] is True
    assert payload["allocation_reactivation_unknown_line_ids"] == [line_id]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )


@pytest.mark.db
def test_restore_stock_missing_after_competing_ship_is_atomic(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-RESTORE-STOCK-MISSING")
    _post_stock(cur, product_id, lot_id, 100)
    cur.execute("SELECT name FROM products WHERE id=%s", (product_id,))
    product_name = cur.fetchone()["name"]

    shipped = allocation_client.post(
        f"/sales/orders/{order_id}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_id, "quantity_lb": 100}]},
    )
    assert shipped.status_code == 200, shipped.text
    transaction_id = shipped.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    assert allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "stock-miss void"}
    ).status_code == 200
    assert _reactivation_quantity(cur, transaction_id, line_id) == pytest.approx(0)
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )

    competing = allocation_client.post(
        "/ship/commit",
        json={
            "product_name": product_name,
            "quantity_lb": 100,
            "customer_name": f"TEST-044 Competing {order_id}",
            "order_reference": f"T044-COMPETE-{order_id}",
            "force_standalone": True,
            "force_create_customer": True,
        },
    )
    assert competing.status_code == 200, competing.text
    _allocation_checkpoint(
        cur, line_id, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    before_restore = _restore_atomic_state(cur, transaction_id, line_id, product_id)

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "stock should be missing"},
    )
    assert restored.status_code == 409, restored.text
    detail = restored.json()["detail"]
    assert detail["error_code"] == "RESTORE_STOCK_MISSING"
    assert detail["transaction_id"] == transaction_id
    assert detail["lot_id"] == lot_id
    assert detail["lot_code"] == "T044-RESTORE-STOCK-MISSING"
    assert detail["required_lb"] == pytest.approx(100)
    assert detail["available_lb"] == pytest.approx(0)
    assert _restore_atomic_state(cur, transaction_id, line_id, product_id) == before_restore


@pytest.mark.db
def test_restore_shrinks_other_order_pin_with_inventory_restored_reason(db_cursor, allocation_client):
    cur = db_cursor
    order_a, (line_a,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-RESTORE-SHRINK")
    _post_stock(cur, product_id, lot_id, 100)
    shipped = allocation_client.post(
        f"/sales/orders/{order_a}/ship/commit",
        json={"ship_all": False, "lines": [{"line_id": line_a, "quantity_lb": 100}]},
    )
    assert shipped.status_code == 200, shipped.text
    transaction_id = shipped.json()["lines_shipped"][0]["transaction_id"]
    _allocation_checkpoint(
        cur, line_a, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    assert allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "shrink setup void"}
    ).status_code == 200
    assert _reactivation_quantity(cur, transaction_id, line_a) == pytest.approx(0)
    _allocation_checkpoint(
        cur, line_a, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )

    order_b, (line_b,), _ = _seed_order(cur, n_lines=1, product_ids=[product_id])
    pin = allocation_client.post(
        f"/sales/orders/{order_b}/allocations",
        json={"mode": "manual", "line_id": line_b, "quantity_lb": 80, "lot_id": lot_id},
    )
    assert pin.status_code == 200, pin.text
    allocation_id = pin.json()["allocations"][0]["id"]
    _allocation_checkpoint(
        cur, line_a, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )
    _allocation_checkpoint(
        cur, line_b, product_id, live_lb=80, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=80,
    )

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "restore and shrink other order"},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["allocations_reshipped"] == []
    cur.execute(
        "SELECT status, release_reason FROM sales_order_allocations WHERE id=%s",
        (allocation_id,),
    )
    released = cur.fetchone()
    assert released["status"] == "released"
    assert released["release_reason"] == "inventory_restored"
    _allocation_checkpoint(
        cur, line_a, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    _allocation_checkpoint(
        cur, line_b, product_id, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )


@pytest.mark.db
def test_multi_line_transaction_records_zero_and_full_then_restores_together(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (zero_line, full_line), (zero_product, full_product) = _seed_order(
        cur, n_lines=2
    )
    zero_lot = _seed_lot(cur, zero_product, "T044-MULTI-ZERO")
    full_lot = _seed_lot(cur, full_product, "T044-MULTI-FULL")
    _post_stock(cur, zero_product, zero_lot, 100)
    _post_stock(cur, full_product, full_lot, 100)
    allocated = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": full_line, "quantity_lb": 100, "lot_id": full_lot},
    )
    assert allocated.status_code == 200, allocated.text
    allocation_id = allocated.json()["allocations"][0]["id"]

    # The current order API emits one transaction per physical line.  Seed the
    # supported multi-line transaction shape directly in this test fixture;
    # the behavior under test (void and restore) still runs through real APIs.
    cur.execute(
        "INSERT INTO transactions (type, timestamp, status) "
        "VALUES ('ship', now(), 'posted') RETURNING id"
    )
    transaction_id = cur.fetchone()["id"]
    cur.execute(
        """INSERT INTO transaction_lines
                   (transaction_id, product_id, lot_id, quantity_lb)
             VALUES (%s, %s, %s, -100), (%s, %s, %s, -100)""",
        (
            transaction_id, zero_product, zero_lot,
            transaction_id, full_product, full_lot,
        ),
    )
    cur.execute(
        """INSERT INTO sales_order_shipments
                   (sales_order_line_id, transaction_id, quantity_lb)
             VALUES (%s, %s, 100), (%s, %s, 100)""",
        (zero_line, transaction_id, full_line, transaction_id),
    )
    main._consume_allocation_row(cur, allocation_id, 100, transaction_id)
    _allocation_checkpoint(
        cur, zero_line, zero_product, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    _allocation_checkpoint(
        cur, full_line, full_product, live_lb=0, shipped_lb=100, on_hand_lb=0,
    )

    voided = allocation_client.post(
        f"/void/{transaction_id}", json={"reason": "multi-line void"}
    )
    assert voided.status_code == 200, voided.text
    assert _reactivation_quantity(cur, transaction_id, zero_line) == pytest.approx(0)
    assert _reactivation_quantity(cur, transaction_id, full_line) == pytest.approx(100)
    _allocation_checkpoint(
        cur, zero_line, zero_product, live_lb=0, shipped_lb=0,
        on_hand_lb=100, originally_allocated_lb=0,
    )
    _allocation_checkpoint(
        cur, full_line, full_product, live_lb=100, shipped_lb=0, on_hand_lb=100,
    )

    restored = allocation_client.post(
        f"/records/transactions/{transaction_id}/corrections",
        json={"event_type": "restore", "reason": "multi-line restore"},
    )
    assert restored.status_code == 200, restored.text
    assert sum(
        float(row["quantity_lb"])
        for row in restored.json()["allocations_reshipped"]
    ) == pytest.approx(100)
    assert {
        row["sales_order_line_id"]
        for row in restored.json()["allocations_reshipped"]
    } == {full_line}
    _allocation_checkpoint(
        cur, zero_line, zero_product, live_lb=0, shipped_lb=0,
        on_hand_lb=0, originally_allocated_lb=0,
    )
    _allocation_checkpoint(
        cur, full_line, full_product, live_lb=0, shipped_lb=100, on_hand_lb=0,
    )


@pytest.mark.db
def test_restore_split_missing_returns_409(db_cursor):
    """S2's second failure mode: coverage guard when stock preflight is isolated."""
    cur = db_cursor
    order_id, (line_id,), (product_id,) = _seed_order(cur, n_lines=1)
    lot_id = _seed_lot(cur, product_id, "T044-RESTORE-MISSING")
    _post_stock(cur, product_id, lot_id, 100)
    allocation_id = _insert(cur, order_id, line_id, product_id, lot_id=lot_id, qty=100)
    txn_id = _seed_txn(cur)
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, -40)",
        (txn_id, product_id, lot_id),
    )
    cur.execute(
        "INSERT INTO sales_order_shipments "
        "(sales_order_line_id, transaction_id, quantity_lb) VALUES (%s, %s, 40)",
        (line_id, txn_id),
    )
    main._consume_allocation_row(cur, allocation_id, 40, txn_id)
    main._append_transaction_correction(
        cur, txn_id, "void", "unit coverage void", None, "test"
    )
    assert _reactivation_quantity(cur, txn_id, line_id) == pytest.approx(40)
    cur.execute("UPDATE sales_order_allocations SET quantity_lb=20 WHERE status='active' AND sales_order_line_id=%s", (line_id,))
    with pytest.raises(main.HTTPException) as exc:
        main._prepare_restore_ship_allocations(cur, txn_id, "test")
    assert exc.value.status_code == 409
    assert exc.value.detail["error_code"] == "RESTORE_SPLIT_MISSING"
    assert exc.value.detail["required_lb"] == pytest.approx(40)
    assert exc.value.detail["available_lb"] == pytest.approx(20)


@pytest.mark.db
def test_lot_product_mismatch_manual_expiry_and_inactive_release_errors(db_cursor, allocation_client):
    cur = db_cursor
    order_id, (line_id, _), (product_id, other_product_id) = _seed_order(cur, n_lines=2)
    lot_id = _seed_lot(cur, other_product_id, "T044-WRONG-PRODUCT")
    mismatch = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 1, "lot_id": lot_id},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["error_code"] == "LOT_PRODUCT_MISMATCH"
    expiry = allocation_client.post(
        f"/sales/orders/{order_id}/allocations",
        json={"mode": "manual", "line_id": line_id, "quantity_lb": 1,
              "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
    )
    assert expiry.status_code == 422
    assert expiry.json()["detail"]["error_code"] == "MANUAL_ALLOCATION_CANNOT_EXPIRE"
    active_id = _insert(cur, order_id, line_id, product_id, qty=1, status="shipped",
                        ship_transaction_id=_seed_txn(cur))
    inactive = allocation_client.post(
        f"/sales/orders/{order_id}/allocations/{active_id}/release"
    )
    assert inactive.status_code == 409
    assert inactive.json()["detail"]["error_code"] == "ALLOCATION_NOT_ACTIVE"

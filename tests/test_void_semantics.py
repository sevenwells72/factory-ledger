"""Tests for unified void semantics.

Convention under test: status='posted' is the single source of truth for ALL
balance math. Voiding a transaction flips its status to 'voided' (its lines
drop out of every balance via POSTED_LINES) and does NOT post a reversal
transaction.

Regression context: the old POST /void/{id} marked the original voided AND
inserted a posted reversal transaction, while some balance queries counted
voided lines and others filtered status='posted' — so the two views disagreed
by exactly the voided amounts (prod lots 582 / 293, June 2026).

Uses the same `_ConnProxy` + monkeypatched `main.get_db_connection` pattern as
test_ship_order_service_line.py so every write the endpoints make — including
their internal commits — is rolled back with the test transaction.
"""

from contextlib import contextmanager

import pytest

try:
    from fastapi import HTTPException
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed", allow_module_level=True)

try:
    import main
    from main import (
        VoidRequest,
        get_current_inventory,
        get_lot,
        lot_on_hand,
        validate_lot_deduction,
        void_transaction,
    )
except Exception as e:  # pragma: no cover
    pytest.skip(f"cannot import main ({e})", allow_module_level=True)


class _ConnProxy:
    """Wrap the test connection so commit()/rollback() act on an inner
    SAVEPOINT; the outer db_cursor/_db_connection fixtures roll everything
    back on teardown."""

    def __init__(self, conn, sp_name):
        self._conn = conn
        self._sp = sp_name
        with self._conn.cursor() as c:
            c.execute(f"SAVEPOINT {self._sp}")

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        with self._conn.cursor() as c:
            c.execute(f"RELEASE SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")

    def rollback(self):
        with self._conn.cursor() as c:
            c.execute(f"ROLLBACK TO SAVEPOINT {self._sp}")
            c.execute(f"SAVEPOINT {self._sp}")


@pytest.fixture
def void_db(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "void_semantics_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    return _db_connection


def _insert_txn(cur, txn_type, lines, status="posted", notes=None):
    """Insert a transaction with [(product_id, lot_id, qty_lb), ...] lines."""
    cur.execute(
        "INSERT INTO transactions (type, timestamp, status, notes) "
        "VALUES (%s, NOW(), %s, %s) RETURNING id",
        (txn_type, status, notes),
    )
    txn_id = cur.fetchone()["id"]
    for product_id, lot_id, qty in lines:
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, %s)",
            (txn_id, product_id, lot_id, qty),
        )
    return txn_id


def _seed_lot(cur, product_name="VOIDTEST Synthetic Batch", product_type="batch"):
    cur.execute(
        "INSERT INTO products (name, type, active) VALUES (%s, %s, true) RETURNING id",
        (product_name, product_type),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, f"VOIDTEST-{product_id}"),
    )
    lot_id = cur.fetchone()["id"]
    return product_id, lot_id


def _txn_count(cur):
    cur.execute("SELECT COUNT(*) AS n FROM transactions")
    return cur.fetchone()["n"]


@pytest.mark.db
def test_void_restores_balance_with_no_new_transaction_row(db_cursor, void_db):
    cur = db_cursor
    product_id, lot_id = _seed_lot(cur)
    _insert_txn(cur, "receive", [(product_id, lot_id, 100)])
    ship_id = _insert_txn(cur, "ship", [(product_id, lot_id, -30)])
    assert lot_on_hand(cur, lot_id) == pytest.approx(70.0)

    before = _txn_count(cur)
    resp = void_transaction(ship_id, VoidRequest(reason="test void"), True)

    assert resp["success"] is True
    assert resp["voided_transaction_id"] == ship_id
    # Backward-compatible keys present but inert: no reversal is posted.
    assert resp["reversal_transaction_id"] is None
    assert resp["reversal_lines"] == []

    assert _txn_count(cur) == before, "void must not insert any transaction row"
    assert lot_on_hand(cur, lot_id) == pytest.approx(100.0)

    cur.execute("SELECT status, notes FROM transactions WHERE id = %s", (ship_id,))
    row = cur.fetchone()
    assert row["status"] == "voided"
    assert "test void" in row["notes"]


@pytest.mark.db
def test_all_balance_endpoints_agree_with_voided_rows_present(db_cursor, void_db):
    cur = db_cursor
    product_id, lot_id = _seed_lot(cur)
    _insert_txn(cur, "receive", [(product_id, lot_id, 100)])
    _insert_txn(cur, "ship", [(product_id, lot_id, -25)])
    # A voided transaction whose lines must count NOWHERE.
    _insert_txn(cur, "adjust", [(product_id, lot_id, 9999)], status="voided")

    expected = 75.0

    assert lot_on_hand(cur, lot_id) == pytest.approx(expected)
    assert validate_lot_deduction(cur, lot_id, "VOIDTEST", 0) == pytest.approx(expected)

    lot_detail = get_lot(lot_id, True)
    assert float(lot_detail["quantity_on_hand"]) == pytest.approx(expected)

    # Direct call bypasses FastAPI Query validation, so a large limit is fine
    # and keeps the assertion immune to how many real batch lots exist.
    inv = get_current_inventory(product_type="batch", limit=100000, _=True)
    rows = [r for r in inv["inventory"] if r["lot_id"] == lot_id]
    assert rows, "synthetic lot missing from /inventory/current"
    assert float(rows[0]["quantity_on_hand"]) == pytest.approx(expected)


@pytest.mark.db
def test_regression_lot_582_pattern_never_recreated(db_cursor, void_db):
    """Reproduce the lot-582 production pattern under the new code:
    make → void → later activity. The old code left a posted reversal that
    made posted-only views disagree with unfiltered views by -2907. The new
    code must produce identical numbers in both views and zero reversals."""
    cur = db_cursor
    product_id, lot_id = _seed_lot(cur)

    make_id = _insert_txn(cur, "make", [(product_id, lot_id, 2907)])
    void_transaction(make_id, VoidRequest(reason="wrong batch count"), True)

    # Later activity, as happened on prod lot 582
    _insert_txn(cur, "make", [(product_id, lot_id, 1938)])
    _insert_txn(cur, "pack", [(product_id, lot_id, -700)])
    _insert_txn(cur, "adjust", [(product_id, lot_id, -1238)])

    # No reversal transaction may exist for this lot
    cur.execute(
        """SELECT COUNT(*) AS n FROM transactions t
           JOIN transaction_lines tl ON tl.transaction_id = t.id
           WHERE tl.lot_id = %s AND t.notes LIKE 'Reversal of transaction%%'""",
        (lot_id,),
    )
    assert cur.fetchone()["n"] == 0

    # Posted-only view and "all rows minus voided" view must agree exactly
    assert lot_on_hand(cur, lot_id) == pytest.approx(0.0)
    cur.execute(
        """SELECT COALESCE(SUM(tl.quantity_lb), 0) AS bal
           FROM transaction_lines tl
           JOIN transactions t ON t.id = tl.transaction_id
           WHERE tl.lot_id = %s AND COALESCE(t.status, 'posted') != 'voided'""",
        (lot_id,),
    )
    assert float(cur.fetchone()["bal"]) == pytest.approx(0.0)


@pytest.mark.db
def test_double_void_fails_cleanly_and_changes_nothing(db_cursor, void_db):
    """The historical-cleanup script may be re-run; a second void of the same
    transaction must return a clear error and leave the ledger untouched."""
    cur = db_cursor
    product_id, lot_id = _seed_lot(cur)
    _insert_txn(cur, "receive", [(product_id, lot_id, 50)])
    adj_id = _insert_txn(cur, "adjust", [(product_id, lot_id, 10)])

    void_transaction(adj_id, VoidRequest(reason="first void"), True)
    assert lot_on_hand(cur, lot_id) == pytest.approx(50.0)
    before = _txn_count(cur)
    cur.execute("SELECT notes FROM transactions WHERE id = %s", (adj_id,))
    notes_before = cur.fetchone()["notes"]

    with pytest.raises(HTTPException) as exc_info:
        void_transaction(adj_id, VoidRequest(reason="second void"), True)
    assert exc_info.value.status_code == 400
    assert "already voided" in str(exc_info.value.detail)

    assert _txn_count(cur) == before
    assert lot_on_hand(cur, lot_id) == pytest.approx(50.0)
    cur.execute("SELECT status, notes FROM transactions WHERE id = %s", (adj_id,))
    row = cur.fetchone()
    assert row["status"] == "voided"
    assert row["notes"] == notes_before, "failed re-void must not touch notes"

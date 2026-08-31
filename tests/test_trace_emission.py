"""Trace emission (TRACEABILITY_DESIGN.md §4 — §9 step 3) coverage.

Every emission hook site is exercised through its endpoint:

1. Per endpoint (receive / make / pack / ship / SO-ship / adjust / both
   found-inventory paths): a commit produces exactly the expected
   trace_events row and trace_event_lots roles/signs/quantities matching
   transaction_lines and ingredient_lot_consumption; a preview produces
   nothing.
2. Fail-hard (§4): a forced emission failure on /receive and /make aborts
   the whole operational commit — error surfaced, no transaction, no lot,
   no silent success.
3. Flag gate (§9 rollback story): TRACE_EMIT_ENABLED=false skips the hooks
   entirely while the operational commit succeeds; default is ON when the
   variable is absent.
4. Void/restore/amend (§3.4): corrections append marker events
   (correction_id set, no lot rows); the original operational event is
   never touched.
5. Double-fire guard: a duplicate (transaction, event_type) emission is
   blocked by trace_events_txn_type_uniq (NULLS NOT DISTINCT).
6. 044 interplay (§10): an SO-ship commit that releases an expired
   auto-FIFO allocation and plans around a foreign lot pin still emits
   roles matching the final shipped lines.
7. Lot merge (§10 M1, Q2): one 'merge' marker per affected transaction, no
   lot rows, no trace rewrite; an already-marked transaction is skipped by
   a later merge instead of aborting it.

Table/constraint-level coverage lives in test_trace_tables_048.py.
"""

from contextlib import contextmanager
from uuid import uuid4

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed", allow_module_level=True)

import psycopg2
from psycopg2 import errors as pg_errors
from psycopg2.extras import RealDictCursor

import main


# ─────────────────────────────────────────────────────────────────
# Fixtures (savepoint-proxy pattern, same as test_lot_identity_047)
# ─────────────────────────────────────────────────────────────────

class _ConnProxy:
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
def client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "trace_emit_sp")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as c:
        c.headers["X-API-Key"] = main.API_KEY
        yield c
    _db_connection.rollback()


def _cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


# ─────────────────────────────────────────────────────────────────
# Seeding helpers (direct SQL: lot codes must be UPPERCASE — the 047
# format CHECK applies to new inserts and these bypass API normalization)
# ─────────────────────────────────────────────────────────────────

def _token():
    return uuid4().hex[:8].upper()


def _seed_product(conn, ptype="ingredient", name=None, **cols):
    token = _token()
    fields = {
        "name": name or f"TraceEmit {ptype} {token}",
        "type": ptype,
        "odoo_code": f"TE-{token}",
        "uom": "lb",
        "active": True,
    }
    fields.update(cols)
    with _cur(conn) as cur:
        cur.execute(
            f"INSERT INTO products ({', '.join(fields)}) "
            f"VALUES ({', '.join(['%s'] * len(fields))}) RETURNING id, name",
            list(fields.values()),
        )
        return cur.fetchone()


def _seed_lot(conn, product_id, lot_code=None):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) "
            "RETURNING id, lot_code",
            (product_id, lot_code or f"TE-{_token()}"),
        )
        return cur.fetchone()


def _seed_posted_lines(conn, lines, txn_type="receive"):
    """One posted transaction with `lines` = [(product_id, lot_id, qty), ...]."""
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status) "
            "VALUES (%s, NOW(), 'posted') RETURNING id",
            (txn_type,),
        )
        txn_id = cur.fetchone()["id"]
        for product_id, lot_id, qty in lines:
            cur.execute(
                "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
                "VALUES (%s, %s, %s, %s)",
                (txn_id, product_id, lot_id, qty),
            )
        return txn_id


def _seed_stocked_lot(conn, product_id, qty, lot_code=None):
    lot = _seed_lot(conn, product_id, lot_code)
    _seed_posted_lines(conn, [(product_id, lot["id"], qty)])
    return lot


def _events_for(conn, txn_id):
    with _cur(conn) as cur:
        cur.execute(
            "SELECT * FROM trace_events WHERE transaction_id = %s ORDER BY id",
            (txn_id,),
        )
        return cur.fetchall()


def _event_lots(conn, event_id):
    with _cur(conn) as cur:
        cur.execute(
            "SELECT lot_id, role, quantity_lb FROM trace_event_lots "
            "WHERE trace_event_id = %s ORDER BY lot_id, role",
            (event_id,),
        )
        return cur.fetchall()


def _lines_for(conn, txn_id):
    with _cur(conn) as cur:
        cur.execute(
            "SELECT product_id, lot_id, quantity_lb FROM transaction_lines "
            "WHERE transaction_id = %s ORDER BY id",
            (txn_id,),
        )
        return cur.fetchall()


def _ilc_for(conn, txn_id):
    with _cur(conn) as cur:
        cur.execute(
            "SELECT ingredient_lot_id, quantity_lb FROM ingredient_lot_consumption "
            "WHERE transaction_id = %s ORDER BY id",
            (txn_id,),
        )
        return cur.fetchall()


def _txn_row(conn, txn_id):
    with _cur(conn) as cur:
        cur.execute(
            "SELECT occurred_at, business_date FROM transactions WHERE id = %s",
            (txn_id,),
        )
        return cur.fetchone()


def _assert_single_event(conn, txn_id, event_type, epcis_type,
                         expected_roles, **party_cols):
    """The transaction has exactly one event; its roles (lot_id, role, qty)
    match `expected_roles` exactly; occurred_at/business_date mirror the
    transaction row; party columns match."""
    events = _events_for(conn, txn_id)
    assert len(events) == 1, events
    ev = events[0]
    assert ev["event_type"] == event_type
    assert ev["epcis_type"] == epcis_type
    assert ev["correction_id"] is None
    txn = _txn_row(conn, txn_id)
    assert ev["occurred_at"] == txn["occurred_at"]
    assert ev["business_date"] == txn["business_date"]
    for col, val in party_cols.items():
        assert ev[col] == val, f"{col}: {ev[col]!r} != {val!r}"
    got = {(r["lot_id"], r["role"]): float(r["quantity_lb"])
           for r in _event_lots(conn, ev["id"])}
    want = {(lot_id, role): float(qty) for lot_id, role, qty in expected_roles}
    assert got == want
    return ev


# ─────────────────────────────────────────────────────────────────
# 1. Per-endpoint emission (commit emits, preview doesn't)
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_receive_commit_emits_event(client, _db_connection):
    product = _seed_product(_db_connection)
    shipper = f"Trace Emit Supplier {_token()}"
    resp = client.post("/receive", json={
        "mode": "commit", "product_name": product["name"], "cases": 4,
        "case_size_lb": 25.0, "shipper_name": shipper, "bol_reference": "TE-BOL-1",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ev = _assert_single_event(
        _db_connection, body["transaction_id"], "receive", "object",
        [(body["lot_id"], "received", 100.0)],
        source_party=shipper,
    )
    # Quantity mirrors the transaction line exactly (R2 invariant).
    lines = _lines_for(_db_connection, body["transaction_id"])
    assert [(l["lot_id"], float(l["quantity_lb"])) for l in lines] == \
        [(body["lot_id"], 100.0)]
    assert ev["sales_order_id"] is None and ev["customer_id"] is None


@pytest.mark.db
def test_receive_preview_emits_nothing(client, _db_connection):
    product = _seed_product(_db_connection)
    resp = client.post("/receive", json={
        "mode": "preview", "product_name": product["name"], "cases": 4,
        "case_size_lb": 25.0, "shipper_name": "Preview Supplier",
        "bol_reference": "TE-BOL-2",
    })
    assert resp.status_code == 200, resp.text
    with _cur(_db_connection) as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM trace_events te "
            "JOIN transactions t ON t.id = te.transaction_id "
            "WHERE t.shipper_name = 'Preview Supplier'")
        assert cur.fetchone()["n"] == 0


def _seed_make_setup(conn, n_ingredients=2, per_batch=10.0, batch_lb=100.0):
    batch = _seed_product(conn, ptype="batch", default_batch_lb=batch_lb)
    ing_lots = []
    for _ in range(n_ingredients):
        ing = _seed_product(conn)
        lot = _seed_stocked_lot(conn, ing["id"], 500.0)
        with _cur(conn) as cur:
            cur.execute(
                "INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) "
                "VALUES (%s, %s, %s)",
                (batch["id"], ing["id"], per_batch),
            )
        ing_lots.append((ing["id"], lot["id"]))
    return batch, ing_lots


@pytest.mark.db
def test_make_commit_emits_transformation(client, _db_connection):
    batch, ing_lots = _seed_make_setup(_db_connection)
    resp = client.post("/make", json={
        "mode": "commit", "product_name": batch["name"], "batches": 2,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    txn_id = body["transaction_id"]
    # Each ILC row → input (negative); output lot → output (positive).
    expected = [(lot_id, "input", -20.0) for _, lot_id in ing_lots]
    expected.append((body["lot_id"], "output", 200.0))
    ev = _assert_single_event(
        _db_connection, txn_id, "make", "transformation", expected)
    # Inputs mirror ingredient_lot_consumption row for row.
    ilc = {(r["ingredient_lot_id"]): float(r["quantity_lb"])
           for r in _ilc_for(_db_connection, txn_id)}
    got_inputs = {r["lot_id"]: -float(r["quantity_lb"])
                  for r in _event_lots(_db_connection, ev["id"])
                  if r["role"] == "input"}
    assert got_inputs == ilc


@pytest.mark.db
def test_make_preview_emits_nothing(client, _db_connection):
    batch, _ = _seed_make_setup(_db_connection)
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        before = cur.fetchone()["n"]
    resp = client.post("/make", json={
        "mode": "preview", "product_name": batch["name"], "batches": 2,
    })
    assert resp.status_code == 200, resp.text
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        assert cur.fetchone()["n"] == before


def _seed_pack_setup(conn, batch_qty=400.0):
    batch = _seed_product(conn, ptype="batch")
    fg = _seed_product(conn, ptype="finished", case_size_lb=10.0)
    batch_lot = _seed_stocked_lot(conn, batch["id"], batch_qty)
    return batch, fg, batch_lot


@pytest.mark.db
def test_pack_commit_emits_transformation(client, _db_connection):
    batch, fg, batch_lot = _seed_pack_setup(_db_connection)
    resp = client.post("/pack", json={
        "mode": "commit", "source_product": batch["name"],
        "target_product": fg["name"], "cases": 5,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_single_event(
        _db_connection, body["transaction_id"], "pack", "transformation",
        [(batch_lot["id"], "input", -50.0),
         (body["output_lot_id"], "output", 50.0)],
    )


@pytest.mark.db
def test_pack_preview_emits_nothing(client, _db_connection):
    batch, fg, _ = _seed_pack_setup(_db_connection)
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        before = cur.fetchone()["n"]
    resp = client.post("/pack", json={
        "mode": "preview", "source_product": batch["name"],
        "target_product": fg["name"], "cases": 5,
    })
    assert resp.status_code == 200, resp.text
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        assert cur.fetchone()["n"] == before


@pytest.mark.db
def test_ship_commit_emits_event_multi_lot(client, _db_connection):
    fg = _seed_product(_db_connection, ptype="finished")
    lot_a = _seed_stocked_lot(_db_connection, fg["id"], 30.0, f"TE-A-{_token()}")
    lot_b = _seed_stocked_lot(_db_connection, fg["id"], 100.0, f"TE-B-{_token()}")
    customer = f"Trace Emit Customer {_token()}"
    resp = client.post("/ship", json={
        "mode": "commit", "product_name": fg["name"], "quantity_lb": 50.0,
        "customer_name": customer, "order_reference": "TE-REF-1",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    txn_id = body["transaction_id"]
    ev = _assert_single_event(
        _db_connection, txn_id, "ship", "object",
        [(lot_a["id"], "shipped", -30.0), (lot_b["id"], "shipped", -20.0)],
        destination_party=customer,
    )
    assert ev["customer_id"] is not None
    assert ev["sales_order_id"] is None
    # Roles mirror the ship transaction lines exactly.
    lines = {l["lot_id"]: float(l["quantity_lb"])
             for l in _lines_for(_db_connection, txn_id)}
    got = {r["lot_id"]: float(r["quantity_lb"])
           for r in _event_lots(_db_connection, ev["id"])}
    assert got == lines


@pytest.mark.db
def test_ship_preview_emits_nothing(client, _db_connection):
    fg = _seed_product(_db_connection, ptype="finished")
    _seed_stocked_lot(_db_connection, fg["id"], 100.0)
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        before = cur.fetchone()["n"]
    resp = client.post("/ship", json={
        "mode": "preview", "product_name": fg["name"], "quantity_lb": 50.0,
        "customer_name": f"Preview Customer {_token()}",
        "order_reference": "TE-REF-2",
    })
    assert resp.status_code == 200, resp.text
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        assert cur.fetchone()["n"] == before


def _seed_order(conn, product_qty_pairs, status="confirmed"):
    """customer + SO + one line per (product_id, quantity)."""
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
            (f"TraceEmit SO Customer {_token()}",),
        )
        customer_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO sales_orders (customer_id, order_number, status) "
            "VALUES (%s, %s, %s) RETURNING id",
            (customer_id, f"TE-SO-{_token()}", status),
        )
        order_id = cur.fetchone()["id"]
        line_ids = []
        for pid, qty in product_qty_pairs:
            cur.execute(
                "INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb) "
                "VALUES (%s, %s, %s) RETURNING id",
                (order_id, pid, qty),
            )
            line_ids.append(cur.fetchone()["id"])
    return order_id, line_ids, customer_id


@pytest.mark.db
def test_so_ship_commit_emits_event_per_line_transaction(client, _db_connection):
    fg1 = _seed_product(_db_connection, ptype="finished")
    fg2 = _seed_product(_db_connection, ptype="finished")
    lot1 = _seed_stocked_lot(_db_connection, fg1["id"], 100.0)
    lot2 = _seed_stocked_lot(_db_connection, fg2["id"], 100.0)
    order_id, _, customer_id = _seed_order(
        _db_connection, [(fg1["id"], 40.0), (fg2["id"], 25.0)])
    resp = client.post(f"/sales/orders/{order_id}/ship",
                       json={"mode": "commit", "ship_all": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    shipped = [r for r in body["lines_shipped"] if r.get("shipped_lb", 0) > 0]
    assert len(shipped) == 2
    for result, (lot, qty) in zip(shipped, [(lot1, 40.0), (lot2, 25.0)]):
        ev = _assert_single_event(
            _db_connection, result["transaction_id"], "ship", "object",
            [(lot["id"], "shipped", -qty)],
            customer_id=customer_id, sales_order_id=order_id,
        )
        assert ev["destination_party"] is not None


@pytest.mark.db
def test_so_ship_preview_emits_nothing(client, _db_connection):
    fg = _seed_product(_db_connection, ptype="finished")
    _seed_stocked_lot(_db_connection, fg["id"], 100.0)
    order_id, _, _ = _seed_order(_db_connection, [(fg["id"], 40.0)])
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        before = cur.fetchone()["n"]
    resp = client.post(f"/sales/orders/{order_id}/ship", json={"mode": "preview"})
    assert resp.status_code == 200, resp.text
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        assert cur.fetchone()["n"] == before


@pytest.mark.db
def test_adjust_commit_emits_signed_adjust(client, _db_connection):
    product = _seed_product(_db_connection)
    lot = _seed_stocked_lot(_db_connection, product["id"], 50.0)
    resp = client.post("/adjust", json={
        "mode": "commit", "product_name": product["name"],
        "lot_code": lot["lot_code"], "adjustment_lb": -7.5,
        "reason": "trace emission regression",
    })
    assert resp.status_code == 200, resp.text
    _assert_single_event(
        _db_connection, resp.json()["transaction_id"], "adjust", "object",
        [(lot["id"], "adjusted", -7.5)],
    )


@pytest.mark.db
def test_adjust_preview_emits_nothing(client, _db_connection):
    product = _seed_product(_db_connection)
    lot = _seed_stocked_lot(_db_connection, product["id"], 50.0)
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        before = cur.fetchone()["n"]
    resp = client.post("/adjust", json={
        "mode": "preview", "product_name": product["name"],
        "lot_code": lot["lot_code"], "adjustment_lb": -7.5,
        "reason": "trace emission regression",
    })
    assert resp.status_code == 200, resp.text
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM trace_events")
        assert cur.fetchone()["n"] == before


@pytest.mark.db
def test_found_inventory_emits_adjust(client, _db_connection):
    product = _seed_product(_db_connection)
    resp = client.post("/inventory/found", json={
        "product_id": product["id"], "quantity": 12.0,
        "reason_code": "found_count",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    with _cur(_db_connection) as cur:
        cur.execute(
            "SELECT transaction_id FROM transaction_lines WHERE lot_id = %s",
            (body["lot_id"],),
        )
        txn_id = cur.fetchone()["transaction_id"]
    _assert_single_event(
        _db_connection, txn_id, "adjust", "object",
        [(body["lot_id"], "adjusted", 12.0)],
    )


@pytest.mark.db
def test_found_with_new_product_emits_adjust(client, _db_connection):
    resp = client.post("/inventory/found-with-new-product", json={
        "product_name": f"TraceEmit Found {_token()}",
        "product_type": "ingredient", "quantity": 8.0,
        "reason_code": "found_count",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    with _cur(_db_connection) as cur:
        cur.execute(
            "SELECT transaction_id FROM transaction_lines WHERE lot_id = %s",
            (body["lot_id"],),
        )
        txn_id = cur.fetchone()["transaction_id"]
    _assert_single_event(
        _db_connection, txn_id, "adjust", "object",
        [(body["lot_id"], "adjusted", 8.0)],
    )


# ─────────────────────────────────────────────────────────────────
# 2. Fail-hard: an emission failure aborts the operational commit
# ─────────────────────────────────────────────────────────────────

def _breaking_emit(real_emit):
    """A wrapper that corrupts event_type so the trace INSERT violates the
    trace_events CHECK — a genuine mid-transaction DB failure, the §4
    scenario fail-hard exists for."""
    def _bad(cur, txn_id, event_type, *args, **kwargs):
        return real_emit(cur, txn_id, "not_a_real_event_type", *args, **kwargs)
    return _bad


@pytest.mark.db
def test_receive_fail_hard_rolls_back_whole_commit(client, _db_connection, monkeypatch):
    product = _seed_product(_db_connection)
    shipper = f"Fail Hard Supplier {_token()}"
    monkeypatch.setattr(main, "emit_trace_event",
                        _breaking_emit(main.emit_trace_event))
    resp = client.post("/receive", json={
        "mode": "commit", "product_name": product["name"], "cases": 2,
        "case_size_lb": 25.0, "shipper_name": shipper, "bol_reference": "TE-FH-1",
    })
    # Error surfaced — no silent success.
    assert resp.status_code == 500, resp.text
    assert resp.json().get("success") is not True
    # The operational transaction (and the minted lot) must NOT exist.
    with _cur(_db_connection) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM transactions WHERE shipper_name = %s",
                    (shipper,))
        assert cur.fetchone()["n"] == 0
        cur.execute("SELECT COUNT(*) AS n FROM lots WHERE product_id = %s",
                    (product["id"],))
        assert cur.fetchone()["n"] == 0


@pytest.mark.db
def test_make_fail_hard_rolls_back_whole_commit(client, _db_connection, monkeypatch):
    batch, ing_lots = _seed_make_setup(_db_connection)
    monkeypatch.setattr(main, "emit_trace_event",
                        _breaking_emit(main.emit_trace_event))
    resp = client.post("/make", json={
        "mode": "commit", "product_name": batch["name"], "batches": 1,
    })
    assert resp.status_code == 500, resp.text
    assert resp.json().get("success") is not True
    with _cur(_db_connection) as cur:
        # No output lot, no make transaction, no ILC rows survived.
        cur.execute("SELECT COUNT(*) AS n FROM lots WHERE product_id = %s",
                    (batch["id"],))
        assert cur.fetchone()["n"] == 0
        cur.execute(
            "SELECT COUNT(*) AS n FROM ingredient_lot_consumption "
            "WHERE ingredient_lot_id = ANY(%s)",
            ([lot_id for _, lot_id in ing_lots],))
        assert cur.fetchone()["n"] == 0


# ─────────────────────────────────────────────────────────────────
# 3. TRACE_EMIT_ENABLED flag gate
# ─────────────────────────────────────────────────────────────────

def test_trace_emit_enabled_parsing(monkeypatch):
    monkeypatch.delenv("TRACE_EMIT_ENABLED", raising=False)
    assert main.trace_emit_enabled() is True  # default ON when absent
    for off in ("false", "FALSE", "0", "off", "no", " False "):
        monkeypatch.setenv("TRACE_EMIT_ENABLED", off)
        assert main.trace_emit_enabled() is False, off
    for on in ("true", "TRUE", "1", "anything-unrecognized"):
        monkeypatch.setenv("TRACE_EMIT_ENABLED", on)
        assert main.trace_emit_enabled() is True, on


@pytest.mark.db
def test_flag_off_commit_succeeds_with_zero_trace_rows(client, _db_connection, monkeypatch):
    monkeypatch.setenv("TRACE_EMIT_ENABLED", "false")
    product = _seed_product(_db_connection)
    resp = client.post("/receive", json={
        "mode": "commit", "product_name": product["name"], "cases": 3,
        "case_size_lb": 20.0, "shipper_name": f"Flag Off Supplier {_token()}",
        "bol_reference": "TE-FLAG-1",
    })
    assert resp.status_code == 200, resp.text
    txn_id = resp.json()["transaction_id"]
    # Operational rows exist, trace rows don't (R1 will show the debt).
    assert len(_lines_for(_db_connection, txn_id)) == 1
    assert _events_for(_db_connection, txn_id) == []


# ─────────────────────────────────────────────────────────────────
# 4. Void/restore/amend markers (§3.4)
# ─────────────────────────────────────────────────────────────────

def _ship_and_void(client, conn):
    fg = _seed_product(conn, ptype="finished")
    _seed_stocked_lot(conn, fg["id"], 100.0)
    resp = client.post("/ship", json={
        "mode": "commit", "product_name": fg["name"], "quantity_lb": 20.0,
        "customer_name": f"Void Customer {_token()}", "order_reference": "TE-V-1",
    })
    assert resp.status_code == 200, resp.text
    txn_id = resp.json()["transaction_id"]
    void = client.post(f"/void/{txn_id}", json={"reason": "trace marker test"})
    assert void.status_code == 200, void.text
    return fg, txn_id, void.json()


@pytest.mark.db
def test_void_appends_marker_and_preserves_original(client, _db_connection):
    _, txn_id, void_body = _ship_and_void(client, _db_connection)
    events = _events_for(_db_connection, txn_id)
    assert [e["event_type"] for e in events] == ["ship", "void"]
    ship_ev, void_ev = events
    # Marker: correction_id set, no lot rows, nothing updated in place.
    assert str(void_ev["correction_id"]) == void_body["correction_id"]
    assert void_ev["epcis_type"] == "object"
    assert _event_lots(_db_connection, void_ev["id"]) == []
    # The original ship event remains intact, lot rows included.
    assert ship_ev["correction_id"] is None
    assert len(_event_lots(_db_connection, ship_ev["id"])) == 1


@pytest.mark.db
def test_restore_appends_second_marker(client, _db_connection):
    _, txn_id, _ = _ship_and_void(client, _db_connection)
    resp = client.post(f"/records/transactions/{txn_id}/corrections",
                       json={"event_type": "restore", "reason": "trace restore test"})
    assert resp.status_code == 200, resp.text
    events = _events_for(_db_connection, txn_id)
    assert [e["event_type"] for e in events] == ["ship", "void", "restore"]
    restore_ev = events[-1]
    assert restore_ev["correction_id"] is not None
    assert _event_lots(_db_connection, restore_ev["id"]) == []


@pytest.mark.db
def test_amend_appends_marker(client, _db_connection):
    product = _seed_product(_db_connection)
    resp = client.post("/receive", json={
        "mode": "commit", "product_name": product["name"], "cases": 1,
        "case_size_lb": 25.0, "shipper_name": f"Amend Supplier {_token()}",
        "bol_reference": "TE-AM-1",
    })
    assert resp.status_code == 200, resp.text
    txn_id = resp.json()["transaction_id"]
    amend = client.post(f"/records/transactions/{txn_id}/corrections",
                        json={"event_type": "amend", "reason": "trace amend test",
                              "replacement_values": {"notes": "amended by test"}})
    assert amend.status_code == 200, amend.text
    events = _events_for(_db_connection, txn_id)
    assert [e["event_type"] for e in events] == ["receive", "amend"]
    amend_ev = events[-1]
    assert str(amend_ev["correction_id"]) == amend.json()["correction_id"]
    assert _event_lots(_db_connection, amend_ev["id"]) == []


# ─────────────────────────────────────────────────────────────────
# 5. Double-fire guard
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_duplicate_emission_blocked_by_txn_type_uniq(_db_connection, db_cursor):
    product = _seed_product(_db_connection)
    lot = _seed_lot(_db_connection, product["id"])
    txn_id = _seed_posted_lines(
        _db_connection, [(product["id"], lot["id"], 5.0)], txn_type="adjust")
    times = main._txn_trace_times(db_cursor, txn_id)
    main.emit_trace_event(db_cursor, txn_id, "adjust", "object",
                          [(lot["id"], "adjusted", 5.0)], *times)
    db_cursor.execute("SAVEPOINT dup_fire")
    with pytest.raises(pg_errors.UniqueViolation) as excinfo:
        main.emit_trace_event(db_cursor, txn_id, "adjust", "object",
                              [(lot["id"], "adjusted", 5.0)], *times)
    assert excinfo.value.diag.constraint_name == "trace_events_txn_type_uniq"
    db_cursor.execute("ROLLBACK TO SAVEPOINT dup_fire")


# ─────────────────────────────────────────────────────────────────
# 6. 044 interplay: expire/shrink side-writes precede emission
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_so_ship_with_expired_allocation_and_foreign_pin(client, _db_connection):
    """A foreign lot pin steers the plan off lot A; an expired auto-FIFO
    allocation on lot B is released inside the same commit. The emitted
    roles must match the final shipped lines exactly."""
    fg = _seed_product(_db_connection, ptype="finished")
    lot_a = _seed_stocked_lot(_db_connection, fg["id"], 10.0, f"TE-IA-{_token()}")
    lot_b = _seed_stocked_lot(_db_connection, fg["id"], 90.0, f"TE-IB-{_token()}")

    # Foreign order pins all of lot A; its expired auto-FIFO slice on lot B
    # must be released (expire side-write) during the own-order commit.
    foreign_order, foreign_lines, _ = _seed_order(_db_connection, [(fg["id"], 100.0)])
    with _cur(_db_connection) as cur:
        cur.execute(
            "INSERT INTO sales_order_allocations "
            "(sales_order_id, sales_order_line_id, product_id, lot_id, "
            " quantity_lb, status, source) "
            "VALUES (%s, %s, %s, %s, 10, 'active', 'manual')",
            (foreign_order, foreign_lines[0], fg["id"], lot_a["id"]),
        )
        cur.execute(
            "INSERT INTO sales_order_allocations "
            "(sales_order_id, sales_order_line_id, product_id, lot_id, "
            " quantity_lb, status, source, expires_at) "
            "VALUES (%s, %s, %s, %s, 5, 'active', 'auto_fifo', "
            "        clock_timestamp() - interval '1 hour') RETURNING id",
            (foreign_order, foreign_lines[0], fg["id"], lot_b["id"]),
        )
        expired_id = cur.fetchone()["id"]

    own_order, _, customer_id = _seed_order(_db_connection, [(fg["id"], 50.0)])
    resp = client.post(f"/sales/orders/{own_order}/ship",
                       json={"mode": "commit", "ship_all": True})
    assert resp.status_code == 200, resp.text
    result = resp.json()["lines_shipped"][0]
    assert result["shipped_lb"] == 50.0
    txn_id = result["transaction_id"]

    # The expire side-write fired inside this commit.
    with _cur(_db_connection) as cur:
        cur.execute("SELECT status, release_reason FROM sales_order_allocations "
                    "WHERE id = %s", (expired_id,))
        row = cur.fetchone()
    assert row["status"] == "released" and row["release_reason"] == "expired"

    # Roles == final shipped lines: everything from lot B, nothing from the
    # pinned lot A.
    ev = _assert_single_event(
        _db_connection, txn_id, "ship", "object",
        [(lot_b["id"], "shipped", -50.0)],
        customer_id=customer_id, sales_order_id=own_order,
    )
    lines = {l["lot_id"]: float(l["quantity_lb"])
             for l in _lines_for(_db_connection, txn_id)}
    assert lines == {lot_b["id"]: -50.0}
    got = {r["lot_id"]: float(r["quantity_lb"])
           for r in _event_lots(_db_connection, ev["id"])}
    assert got == lines


# ─────────────────────────────────────────────────────────────────
# 7. Lot merge markers (§10 M1, Q2)
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_merge_emits_marker_per_affected_transaction(client, _db_connection):
    product = _seed_product(_db_connection)
    lot_a = _seed_lot(_db_connection, product["id"], f"TE-MA-{_token()}")
    lot_b = _seed_lot(_db_connection, product["id"], f"TE-MB-{_token()}")
    lot_c = _seed_lot(_db_connection, product["id"], f"TE-MC-{_token()}")
    # T1 touches lots A and B; T2 touches only A.
    t1 = _seed_posted_lines(_db_connection, [
        (product["id"], lot_a["id"], 5.0), (product["id"], lot_b["id"], 5.0)])
    t2 = _seed_posted_lines(_db_connection, [(product["id"], lot_a["id"], 3.0)])

    resp = client.post("/admin/lots/merge", json={
        "source_lot_id": lot_a["id"], "target_lot_id": lot_c["id"],
        "reason": "trace merge marker test",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["trace_merge_markers"] == 2
    for txn_id in (t1, t2):
        events = _events_for(_db_connection, txn_id)
        assert [e["event_type"] for e in events] == ["merge"]
        assert events[0]["correction_id"] is None
        assert events[0]["epcis_type"] == "object"
        assert _event_lots(_db_connection, events[0]["id"]) == []

    # A second merge whose lines land on an already-marked transaction
    # (B's line lives in T1) skips the marker instead of failing the merge.
    resp = client.post("/admin/lots/merge", json={
        "source_lot_id": lot_b["id"], "target_lot_id": lot_c["id"],
        "reason": "trace merge marker test 2",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["trace_merge_markers"] == 0
    assert len(_events_for(_db_connection, t1)) == 1

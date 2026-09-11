"""`sales_order_allocations.released_by` holds ONE kind of value.

Before this, it held two. The manual release endpoint wrote
`caller_source_tag(request)` — `'dashboard'` for the scoped key, NULL for the
master key — while the cancel and ship paths wrote `_operator_id(_)`, which
returns the constant `'legacy-shared-key'` on 100% of calls because
`verify_api_key` hands it a bare `True` (docs/design/so-state-model-findings.md,
"Follow-up: `_operator_id()` is a no-op placeholder"). Same column, two
incompatible vocabularies, decided by which code path happened to release the
row.

These tests pin each of the three paths the findings doc names to
`caller_source_tag`:

  * `PATCH /sales/orders/{id}/lines/{line_id}/cancel` — `cancel_order_line`
  * `POST  /sales/orders/{id}/ship/commit`            — `ship_order`
  * `PATCH /sales/orders/{id}/status` → `cancelled`   — the legacy order cancel

For each: the scoped dashboard key writes `'dashboard'`, the master key writes
NULL, and `'legacy-shared-key'` is never written. NULL is the correct answer for
a master-key call, not a gap to be filled — per-user attribution is FR-15 and
does not exist yet, and a placeholder that looks like an identity is worse than
an honest absence.

`_operator_id()` itself is deliberately untouched: other subsystems still call
it, and removing it is a separate change.
"""

from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import RealDictCursor

import main


PLACEHOLDER = "legacy-shared-key"


class _ConnProxy:
    """commit()/rollback() operate on an inner SAVEPOINT, so the endpoints'
    own commits are still undone when the test's savepoint rolls back."""

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
def client(db_cursor, monkeypatch):
    conn = db_cursor.connection

    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(conn, "released_by_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as test_client:
        yield test_client


# Both keys under test. The master key is the one the GPT presents; the scoped
# key is the dashboard's.
KEYS = [
    pytest.param("dashboard", "dashboard", id="dashboard-key"),
    pytest.param("master", None, id="master-key"),
]


def _headers(which):
    key = main.DASHBOARD_API_KEY if which == "dashboard" else main.API_KEY
    return {"X-API-Key": key}


# ─────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────

def _seed(cur, *, stock=500.0, qty=100.0):
    """One confirmed order, one physical line, one lot with stock on hand."""
    token = uuid4().hex[:8].upper()
    cur.execute(
        "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
        (f"RBATTR Customer {token}",),
    )
    customer_id = cur.fetchone()["id"]

    cur.execute(
        "INSERT INTO products (name, type, odoo_code, uom, is_service, active) "
        "VALUES (%s, 'finished', %s, 'lb', false, true) RETURNING id",
        (f"RBATTR FG {token}", f"RBA-{token}"),
    )
    product_id = cur.fetchone()["id"]

    cur.execute(
        "INSERT INTO lots (product_id, lot_code, entry_source, received_at) "
        "VALUES (%s, %s, 'received', NOW()) RETURNING id",
        (product_id, f"RBA-LOT-{token}"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
    txn = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)",
        (txn, product_id, lot_id, stock),
    )

    order_number = f"RBA-SO-{token}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status) "
        "VALUES (%s, %s, 'confirmed') RETURNING id",
        (customer_id, order_number),
    )
    order_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO sales_order_flags (so_number, ready, ready_at, ready_by) "
        "VALUES (%s, true, NOW(), 'test')",
        (order_number,),
    )
    cur.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, %s, 0, 'pending') RETURNING id",
        (order_id, product_id, qty),
    )
    line_id = cur.fetchone()["id"]
    return {
        "order_id": order_id,
        "order_number": order_number,
        "line_id": line_id,
        "product_id": product_id,
        "lot_id": lot_id,
    }


def _allocate(cur, seeded, qty=100.0, *, line_id=None, source="manual", expired=False):
    cur.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, quantity_lb, source, expires_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (seeded["order_id"], line_id or seeded["line_id"], seeded["product_id"], qty,
         source, None),
    )
    allocation_id = cur.fetchone()["id"]
    if expired:
        cur.execute(
            "UPDATE sales_order_allocations "
            "   SET expires_at = clock_timestamp() - interval '1 hour' WHERE id = %s",
            (allocation_id,),
        )
    return allocation_id


def _released(cur, allocation_id):
    cur.execute(
        "SELECT status, released_by, release_reason FROM sales_order_allocations "
        " WHERE id = %s",
        (allocation_id,),
    )
    return dict(cur.fetchone())


def _assert_attribution(row, expected_tag):
    assert row["status"] == "released", row
    assert row["released_by"] != PLACEHOLDER, (
        f"released_by must never be the {PLACEHOLDER!r} placeholder: {row}"
    )
    assert row["released_by"] == expected_tag, row


# ─────────────────────────────────────────────────────────────────
# Path 1 — line cancel
# ─────────────────────────────────────────────────────────────────
# PATCH .../lines/{line_id}/cancel is NOT on the dashboard allowlist — only
# .../lines/{line_id}/update is — so the master key is the only key that
# reaches it, and NULL is the only attribution it can honestly record. The
# 403 is asserted below so this stays a fact about the allowlist rather than
# an untested gap.

@pytest.mark.db
def test_line_cancel_records_the_caller_surface(client, db_cursor):
    seeded = _seed(db_cursor)
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/cancel",
        headers=_headers("master"),
    )
    assert resp.status_code == 200, resp.text

    row = _released(db_cursor, allocation_id)
    assert row["release_reason"] == "line_cancelled"
    _assert_attribution(row, None)


@pytest.mark.db
def test_line_cancel_is_master_key_only(client, db_cursor):
    """Why the test above has no 'dashboard' case."""
    seeded = _seed(db_cursor)
    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/cancel",
        headers=_headers("dashboard"),
    )
    assert resp.status_code == 403, resp.text


# ─────────────────────────────────────────────────────────────────
# Path 2 — ship commit
# ─────────────────────────────────────────────────────────────────
# The ship path's released_by reaches the column through
# _sales_order_ship_plan → available_lots_for_product(persist_expired=True) →
# _expire_auto_fifo_allocations, so the row that proves it is an auto-FIFO
# allocation whose TTL has elapsed. Shipping the product persists that expiry.

@pytest.mark.db
@pytest.mark.parametrize("key,expected", KEYS)
def test_ship_commit_records_the_caller_surface(client, db_cursor, key, expected):
    seeded = _seed(db_cursor)
    # A second line on the same product, holding a stale auto-FIFO reservation.
    db_cursor.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, 25, 0, 'pending') RETURNING id",
        (seeded["order_id"], seeded["product_id"]),
    )
    other_line = db_cursor.fetchone()["id"]
    stale = _allocate(db_cursor, seeded, 25.0, line_id=other_line,
                      source="auto_fifo", expired=True)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/ship/commit",
        json={"lines": [{"line_id": seeded["line_id"], "quantity_lb": 10}]},
        headers=_headers(key),
    )
    assert resp.status_code == 200, resp.text

    row = _released(db_cursor, stale)
    assert row["release_reason"] == "expired"
    _assert_attribution(row, expected)


# ─────────────────────────────────────────────────────────────────
# Path 3 — legacy order cancel via PATCH .../status
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
@pytest.mark.parametrize("key,expected", KEYS)
def test_legacy_status_cancel_records_the_caller_surface(client, db_cursor, key, expected):
    seeded = _seed(db_cursor)
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/status",
        json={"status": "cancelled"},
        headers=_headers(key),
    )
    assert resp.status_code == 200, resp.text

    row = _released(db_cursor, allocation_id)
    assert row["release_reason"] == "order_cancelled"
    _assert_attribution(row, expected)


# ─────────────────────────────────────────────────────────────────
# The column now speaks one language
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_manual_release_and_ship_agree(client, db_cursor):
    """The comparison that motivated the change: the manual release endpoint
    and a ship must stamp the SAME vocabulary for the same key. Both are on
    the dashboard allowlist, so both are reachable with the scoped key."""
    manual_seed = _seed(db_cursor)
    manual_alloc = _allocate(db_cursor, manual_seed)
    resp = client.post(
        f"/sales/orders/{manual_seed['order_id']}/allocations/{manual_alloc}/release",
        headers=_headers("dashboard"),
    )
    assert resp.status_code == 200, resp.text

    ship_seed = _seed(db_cursor)
    db_cursor.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, 25, 0, 'pending') RETURNING id",
        (ship_seed["order_id"], ship_seed["product_id"]),
    )
    other_line = db_cursor.fetchone()["id"]
    stale = _allocate(db_cursor, ship_seed, 25.0, line_id=other_line,
                      source="auto_fifo", expired=True)
    resp = client.post(
        f"/sales/orders/{ship_seed['order_id']}/ship/commit",
        json={"lines": [{"line_id": ship_seed["line_id"], "quantity_lb": 10}]},
        headers=_headers("dashboard"),
    )
    assert resp.status_code == 200, resp.text

    assert (_released(db_cursor, manual_alloc)["released_by"]
            == _released(db_cursor, stale)["released_by"]
            == "dashboard")


@pytest.mark.db
def test_the_three_fixed_paths_no_longer_reference_operator_id():
    """Source-level guard: a later edit must not quietly reinstate the
    placeholder on these handlers. Scoped to the three, because _operator_id()
    is deliberately still live elsewhere."""
    import inspect

    for fn in (main.cancel_order_line, main.ship_order, main.update_order_status):
        src = inspect.getsource(fn)
        assert "_operator_id(" not in src, (
            f"{fn.__name__} must attribute releases with caller_source_tag(request), "
            f"not the _operator_id placeholder"
        )

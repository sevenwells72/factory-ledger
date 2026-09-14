"""FR-15 step 5a: per-actor attribution for sales-order writes (migration 052).

A third kind of API key, resolved from the `actors` table by sha256, that names
a PERSON. When one authenticates a write, the attribution columns record that
person's name instead of the surface tag `'dashboard'` or an honest NULL.

What these tests are really defending is the FIRST half of the feature: that
adding a third key kind changed NOTHING for the two that already existed. Every
endpoint test below runs three times — master key, dashboard key, actor key —
and the first two assert the pre-FR-15 answer verbatim. If actor resolution
ever leaks into a legacy-key call, one of them fails.

Covered:
  * legacy keys unchanged on every touched path (master -> NULL,
    dashboard -> 'dashboard')
  * an actor key attributes close / cancel / reopen / ready / allocate /
    release / ship / update_line to the actor's name
  * the resolved actor outranks a caller-supplied `changed_by`
  * a deactivated actor is rejected exactly like a key that never existed
  * an unknown key is rejected exactly as before FR-15
  * GET /auth/whoami for all three key kinds
  * the 60-second actor cache: one load per TTL, not one per request, and a
    newly minted key is invisible until the cache turns over
  * migrations/052_actors.sql re-runs as a no-op

NOT here: the lock-sequence assertion. FR-15 touched eleven sales-order write
paths and must not have moved a single lock, which is asserted by
test_so_write_paths_take_the_same_locks_in_the_same_order in
tests/test_sales_order_state_model.py — appended to the existing concurrency
harness rather than restated here, so there is one place that knows the
normative lock order.

Same TestClient + savepoint-proxy pattern as test_released_by_attribution.py.
"""

import logging
import re
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

import main


ROOT = Path(__file__).resolve().parent.parent
MIGRATION_052 = ROOT / "migrations" / "052_actors.sql"

PLACEHOLDER = "legacy-shared-key"


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

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


def _apply_052(cur):
    """Apply migration 052 inside the test transaction.

    Idempotent, so it does not matter whether the local test database already
    has the table from a prod schema dump. Executed with NO parameters:
    psycopg2 only interprets %-sequences when args are passed, and the
    migration's RAISE NOTICE contains one.
    """
    cur.execute(MIGRATION_052.read_text())


@pytest.fixture
def actors(db_cursor):
    """Four actors, one per role plus a deactivated one, on the test
    transaction's connection so the app (proxied onto the same connection)
    can see them. All rolled back on teardown."""
    _apply_052(db_cursor)
    token = uuid4().hex[:8].upper()
    made = {}
    seeds = [
        ("owner", "owner", True),
        ("floor", "floor", True),
        ("office", "office", True),
        ("retired", "floor", False),
    ]
    for label, role, active in seeds:
        name = f"ACT {label} {token}"
        key = f"actor-key-{label}-{token}"
        db_cursor.execute(
            "INSERT INTO actors (name, role, key_hash, active) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (name, role, sha256(key.encode()).hexdigest(), active),
        )
        made[label] = {
            "id": db_cursor.fetchone()["id"],
            "name": name,
            "role": role,
            "key": key,
            "active": active,
        }
    # The cache is module state and survives between tests; a stale entry from
    # a previous test's rolled-back rows would make this one nondeterministic.
    main._reset_actor_cache()
    yield made
    main._reset_actor_cache()


@pytest.fixture
def client(db_cursor, monkeypatch):
    conn = db_cursor.connection

    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(conn, "actor_attr_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as test_client:
        yield test_client


# ─────────────────────────────────────────────────────────────────
# Key matrix
#
# Every endpoint test runs all three. `expected` is the attribution the write
# must record; the two legacy rows are the pre-FR-15 answers, unchanged.
# ─────────────────────────────────────────────────────────────────

MASTER, DASHBOARD, ACTOR = "master", "dashboard", "actor"

ALL_KEYS = [
    pytest.param(MASTER, id="master-key"),
    pytest.param(DASHBOARD, id="dashboard-key"),
    pytest.param(ACTOR, id="actor-key"),
]


def _headers(which, actors):
    if which == MASTER:
        return {"X-API-Key": main.API_KEY}
    if which == DASHBOARD:
        return {"X-API-Key": main.DASHBOARD_API_KEY}
    return {"X-API-Key": actors["floor"]["key"]}


def _expected(which, actors):
    if which == MASTER:
        return None
    if which == DASHBOARD:
        return "dashboard"
    return actors["floor"]["name"]


def _assert_attribution(value, which, actors):
    expected = _expected(which, actors)
    assert value != PLACEHOLDER, (
        f"attribution must never be the {PLACEHOLDER!r} placeholder (got {value!r})"
    )
    assert value == expected, f"{which} key: expected {expected!r}, got {value!r}"


# ─────────────────────────────────────────────────────────────────
# Seeding (mirrors test_released_by_attribution.py)
# ─────────────────────────────────────────────────────────────────

def _seed(cur, *, stock=500.0, qty=100.0, status="confirmed"):
    token = uuid4().hex[:8].upper()
    cur.execute(
        "INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
        (f"ACTATTR Customer {token}",),
    )
    customer_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, uom, is_service, active) "
        "VALUES (%s, 'finished', %s, 'lb', false, true) RETURNING id",
        (f"ACTATTR FG {token}", f"ACTA-{token}"),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code, entry_source, received_at) "
        "VALUES (%s, %s, 'received', NOW()) RETURNING id",
        (product_id, f"ACTA-LOT-{token}"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
    txn = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)",
        (txn, product_id, lot_id, stock),
    )
    order_number = f"ACTA-SO-{token}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status) "
        "VALUES (%s, %s, %s) RETURNING id",
        (customer_id, order_number, status),
    )
    order_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, %s, 0, 'pending') RETURNING id",
        (order_id, product_id, qty),
    )
    line_id = cur.fetchone()["id"]
    return {"order_id": order_id, "order_number": order_number, "line_id": line_id,
            "product_id": product_id, "lot_id": lot_id, "customer_id": customer_id}


def _allocate(cur, seeded, qty=100.0, *, line_id=None, source="manual", expired=False):
    cur.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, quantity_lb, source) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (seeded["order_id"], line_id or seeded["line_id"], seeded["product_id"],
         qty, source),
    )
    allocation_id = cur.fetchone()["id"]
    if expired:
        cur.execute(
            "UPDATE sales_order_allocations "
            "   SET expires_at = clock_timestamp() - interval '1 hour' WHERE id = %s",
            (allocation_id,),
        )
    return allocation_id


def _state_changed_by(cur, order_id):
    cur.execute("SELECT state, state_changed_by FROM sales_orders WHERE id = %s", (order_id,))
    return dict(cur.fetchone())


def _persisted_exit(cur, order_id):
    """Both halves of the mirrored model, read back from the row rather than
    from the response body — an exit that returns the right JSON and persists
    the wrong attribution is exactly the failure these tests exist to catch."""
    cur.execute(
        "SELECT status, state, state_reason, state_note, state_changed_by "
        "  FROM sales_orders WHERE id = %s",
        (order_id,),
    )
    return dict(cur.fetchone())


def _allocation(cur, allocation_id):
    cur.execute(
        "SELECT status, created_by, released_by, release_reason "
        "  FROM sales_order_allocations WHERE id = %s",
        (allocation_id,),
    )
    return dict(cur.fetchone())


def _released_for_line(cur, line_id):
    """Every released row for a line, however it got released.

    _shrink_active_allocations either releases the row in place or splits off
    a new released row, depending on whether the excess consumes the whole
    allocation. Both carry released_by, and which branch runs is not the point
    of these tests.
    """
    cur.execute(
        "SELECT id, released_by FROM sales_order_allocations "
        " WHERE sales_order_line_id = %s AND status = 'released' ORDER BY id",
        (line_id,),
    )
    return [dict(r) for r in cur.fetchall()]


# ═════════════════════════════════════════════════════════════════
# The write paths
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_close_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor)
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/close",
        json={"reason": "short_closed", "mode": "commit"},
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state"] == "closed"
    _assert_attribution(order["state_changed_by"], which, actors)

    released = _allocation(db_cursor, allocation_id)
    assert released["status"] == "released"
    _assert_attribution(released["released_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_cancel_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor)
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/cancel",
        json={"reason": "customer_cancelled", "mode": "commit"},
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state"] == "cancelled"
    _assert_attribution(order["state_changed_by"], which, actors)
    _assert_attribution(_allocation(db_cursor, allocation_id)["released_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_reopen_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor)
    db_cursor.execute(
        "UPDATE sales_orders SET state = 'cancelled', state_reason = 'other', "
        "       state_changed_by = 'seed', state_changed_at = NOW() WHERE id = %s",
        (seeded["order_id"],),
    )

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/reopen",
        json={"mode": "commit"},
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state"] == "open"
    _assert_attribution(order["state_changed_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_ready_flag_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales-orders/{seeded['order_number']}/ready",
        json={"ready": True},
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    db_cursor.execute(
        "SELECT ready, ready_by FROM sales_order_flags WHERE so_number = %s",
        (seeded["order_number"],),
    )
    row = dict(db_cursor.fetchone())
    assert row["ready"] is True
    if which == ACTOR:
        assert row["ready_by"] == actors["floor"]["name"]
    else:
        # Untouched legacy behaviour: the body's `by`, defaulting to 'floor'.
        assert row["ready_by"] == "floor"


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_allocation_create_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/allocations",
        json={"line_id": seeded["line_id"], "mode": "manual", "quantity_lb": 25},
        headers=_headers(which, actors),
    )
    assert resp.status_code in (200, 201), resp.text

    db_cursor.execute(
        "SELECT created_by FROM sales_order_allocations "
        " WHERE sales_order_line_id = %s AND status = 'active' ORDER BY id DESC LIMIT 1",
        (seeded["line_id"],),
    )
    _assert_attribution(db_cursor.fetchone()["created_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_manual_release_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor)
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/allocations/{allocation_id}/release",
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    row = _allocation(db_cursor, allocation_id)
    assert row["release_reason"] == "manual_release"
    _assert_attribution(row["released_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_ship_commit_records_the_actor(client, db_cursor, actors, which):
    """Shipping stamps released_by through _expire_auto_fifo_allocations, so
    the row that proves it is an auto-FIFO reservation whose TTL has elapsed —
    the same lever test_released_by_attribution.py pulls."""
    seeded = _seed(db_cursor)
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
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    row = _allocation(db_cursor, stale)
    assert row["release_reason"] == "expired"
    _assert_attribution(row["released_by"], which, actors)


# ─────────────────────────────────────────────────────────────────
# The legacy status endpoint's two administrative exits
#
# PATCH .../status is not just an operational status write: 'cancelled' and
# 'invoiced' route through the same exit logic as POST /cancel and /close,
# and so write BOTH sales_orders.state_changed_by and released_by on every
# reservation they release. Two columns, two writers, one request — and the
# state model's whole point is that the two halves never disagree, so both
# are read back from the row rather than from the response body.
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_legacy_status_cancelled_records_the_actor(client, db_cursor, actors, which):
    seeded = _seed(db_cursor, status="confirmed")
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/status",
        json={"status": "cancelled"},
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    row = _persisted_exit(db_cursor, seeded["order_id"])
    assert (row["status"], row["state"]) == ("cancelled", "cancelled")
    assert row["state_note"] == "via legacy status endpoint"
    _assert_attribution(row["state_changed_by"], which, actors)

    released = _allocation(db_cursor, allocation_id)
    assert released["status"] == "released"
    assert released["release_reason"] == "order_cancelled"
    _assert_attribution(released["released_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_legacy_status_invoiced_records_the_actor(client, db_cursor, actors, which):
    """'invoiced' routes through close but keeps its own mirror, so the
    persisted pair is ('invoiced', 'closed') — and the attribution has to be
    right on both the order row and the reservations close releases."""
    seeded = _seed(db_cursor, status="shipped")
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/status",
        json={"status": "invoiced"},
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    row = _persisted_exit(db_cursor, seeded["order_id"])
    assert (row["status"], row["state"]) == ("invoiced", "closed")
    assert row["state_note"] == "via legacy status endpoint (invoiced)"
    _assert_attribution(row["state_changed_by"], which, actors)

    released = _allocation(db_cursor, allocation_id)
    assert released["status"] == "released"
    assert released["release_reason"] == "order_closed"
    _assert_attribution(released["released_by"], which, actors)


# update_order_line has TWO sites that write released_by, and they used to
# write the placeholder at both. Each is pinned on its own, with the full
# three-key matrix, because they release different rows for different reasons
# and a fix that reached only one of them would still leave the column holding
# two vocabularies.
#
# INTENTIONAL BEHAVIOUR CHANGE — OWNER RULING. Before this PR both sites wrote
# the constant 'legacy-shared-key' on 100% of calls, for every key kind. They
# now write what every other allocation writer writes: the actor's name for an
# actor key, 'dashboard' for the scoped dashboard key, and NULL for the master
# key. 'legacy-shared-key' was a placeholder, and fixing this writer was a
# logged follow-up in docs/design/so-state-model-findings.md, not a regression
# introduced here. The NULL for the master key is the point: it is the honest
# answer for a key that names a surface, not a person, and it is what the other
# three writers have recorded since they were fixed.

@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_update_line_records_the_actor_at_the_shrink_site(client, db_cursor, actors, which):
    """Site 1 of 2: _shrink_active_allocations, reached when the new quantity
    is smaller than what is already reserved against the line."""
    seeded = _seed(db_cursor)
    _allocate(db_cursor, seeded, 100.0)

    # 100 lb reserved, the line cut to 10: the 90 lb of excess is shed by
    # _shrink_active_allocations. Nothing here is auto-FIFO or expired, so the
    # expiry site below releases nothing and this test sees the shrink site
    # alone.
    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/update"
        f"?quantity_lb=10",
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    released = _released_for_line(db_cursor, seeded["line_id"])
    assert released, "cutting the line must shed the excess reservation"
    for row in released:
        assert row["released_by"] != PLACEHOLDER
        _assert_attribution(row["released_by"], which, actors)


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_update_line_records_the_actor_at_the_expiry_site(client, db_cursor, actors, which):
    """Site 2 of 2: _expire_auto_fifo_allocations, which persists elapsed
    auto-FIFO TTLs for the whole product on any product-locking write.

    The row it releases belongs to a DIFFERENT line, and the update here
    RAISES the quantity, so no excess exists and the shrink site does not run.
    Whatever released_by this row carries came from the expiry site.
    """
    seeded = _seed(db_cursor)
    db_cursor.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, 25, 0, 'pending') RETURNING id",
        (seeded["order_id"], seeded["product_id"]),
    )
    other_line = db_cursor.fetchone()["id"]
    stale = _allocate(db_cursor, seeded, 25.0, line_id=other_line,
                      source="auto_fifo", expired=True)

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/update"
        f"?quantity_lb=150",
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    row = _allocation(db_cursor, stale)
    assert row["status"] == "released"
    assert row["release_reason"] == "expired"
    assert row["released_by"] != PLACEHOLDER
    _assert_attribution(row["released_by"], which, actors)

    assert not _released_for_line(db_cursor, seeded["line_id"]), (
        "raising the quantity must not shed anything from the line itself — "
        "this test would otherwise be reading the shrink site's write"
    )


@pytest.mark.db
def test_update_line_no_longer_references_the_operator_placeholder():
    """Source-level guard, matching the one test_released_by_attribution.py
    keeps over the other three handlers. _operator_id() is deliberately still
    live in other subsystems, so this is scoped to this handler.

    Also pins that BOTH write sites are fed from the shared helper, and that
    neither one is fed a literal: a future edit that reintroduced a constant at
    either call site would pass the tests above only if it happened to equal
    the expected value for all three key kinds, which no constant can.
    """
    import inspect

    src = inspect.getsource(main.update_order_line)
    assert "_operator_id(" not in src, (
        "update_order_line must attribute releases with caller_source_tag(request), "
        "not the _operator_id placeholder"
    )
    code = "\n".join(
        "" if ln.strip().startswith("#") else ln.split("#")[0]
        for ln in src.splitlines()
    )
    assert PLACEHOLDER not in code, (
        f"update_order_line must not WRITE the {PLACEHOLDER!r} placeholder. "
        f"(Comments explaining that it used to are fine and are stripped here.)"
    )
    assert "released_by = caller_source_tag(request)" in src, (
        "both write sites must be fed from one caller_source_tag(request) call"
    )
    assert "_expire_auto_fifo_allocations(cur, product_id, released_by)" in src, (
        "the expiry site must pass the resolved released_by"
    )
    shrink = src[src.index("_shrink_active_allocations("):]
    shrink = shrink[:shrink.index(")")]
    assert "released_by," in shrink, (
        "the shrink site must pass the resolved released_by"
    )


# ═════════════════════════════════════════════════════════════════
# The actor outranks a self-reported identity
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_resolved_actor_beats_body_changed_by(client, db_cursor, actors):
    """`changed_by` is whatever the caller typed. The actor is authenticated.
    Letting the body win would let anyone holding a personal key sign someone
    else's name to an administrative exit."""
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/close",
        json={"reason": "short_closed", "mode": "commit", "changed_by": "Somebody Else"},
        headers=_headers(ACTOR, actors),
    )
    assert resp.status_code == 200, resp.text

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state_changed_by"] == actors["floor"]["name"]


@pytest.mark.db
def test_body_changed_by_still_wins_for_the_master_key(client, db_cursor, actors):
    """Unchanged legacy behaviour: the office GPT holds the master key,
    resolves no actor, and its self-reported identity is still stored."""
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/close",
        json={"reason": "short_closed", "mode": "commit", "changed_by": "gpt-sales-admin"},
        headers=_headers(MASTER, actors),
    )
    assert resp.status_code == 200, resp.text

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state_changed_by"] == "gpt-sales-admin"


@pytest.mark.db
def test_over_long_changed_by_is_still_rejected_for_an_actor(client, db_cursor, actors):
    """The length check runs BEFORE the actor branch, on every key kind. An
    over-long changed_by must stay a 400 rather than becoming a value silently
    discarded because the caller happened to hold a personal key."""
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/close",
        json={"reason": "short_closed", "mode": "commit",
              "changed_by": "x" * (main.SO_CHANGED_BY_MAX + 1)},
        headers=_headers(ACTOR, actors),
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "CHANGED_BY_TOO_LONG"


# ═════════════════════════════════════════════════════════════════
# Rejection
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_inactive_actor_is_rejected(client, db_cursor, actors):
    """A deactivated actor's key is indistinguishable from one that was never
    minted — which is what deactivation is supposed to mean.

    NOTE ON THE STATUS CODE. This asserts 403, not 401, because 403 is what an
    unrecognised key has returned on the header dependency since the scoped
    dashboard key shipped (tests/test_dashboard_api_key.py pins it). FR-15's
    requirement was that unknown keys are rejected "exactly as today"; today is
    403 here and 401 on the packing-slip query-param dependency, and preserving
    that mattered more than making this one route differ from its neighbours.
    """
    assert actors["retired"]["active"] is False
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/close",
        json={"reason": "short_closed", "mode": "commit"},
        headers={"X-API-Key": actors["retired"]["key"]},
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Invalid API key"

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state"] == "open", "a rejected call must not have written anything"


@pytest.mark.db
def test_unknown_key_is_rejected_exactly_as_before(client, actors):
    """Both dependencies keep their historical codes: 403 on the header path,
    401 on the packing-slip query-param path."""
    resp = client.get("/sales/orders?limit=1", headers={"X-API-Key": "no-such-key"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Invalid API key"

    resp = client.get("/sales/orders?limit=1")
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "API key required"


@pytest.mark.db
def test_actor_key_is_scoped_to_the_dashboard_allowlist(client, actors):
    """An actor key replaces the shared dashboard key; it does not upgrade it.
    A named person does not thereby get master-key reach."""
    for label, resp in [
        ("POST /make", client.post("/make", json={}, headers=_headers(ACTOR, actors))),
        ("POST /adjust", client.post("/adjust", json={}, headers=_headers(ACTOR, actors))),
        ("POST /void/1", client.post("/void/1", json={}, headers=_headers(ACTOR, actors))),
        ("POST /sales/orders", client.post("/sales/orders", json={},
                                           headers=_headers(ACTOR, actors))),
        ("GET /admin/lots/duplicates",
         client.get("/admin/lots/duplicates", headers=_headers(ACTOR, actors))),
    ]:
        assert resp.status_code == 403, f"{label}: {resp.status_code} {resp.text}"
        assert resp.json()["detail"] == "API key not authorized for this endpoint", label


# ─────────────────────────────────────────────────────────────────
# The two sales-order line writers an actor key must NOT reach
#
# Both are master-key only, and both are next door to
# PATCH .../lines/{id}/update, which IS allowlisted — so "the line endpoints"
# is not a thing anyone can reason about as a group. Each is pinned
# separately, with the row read back, because a 403 that still wrote is the
# failure that matters.
#
# POST .../lines has a second reason to stay out: sales_order_lines carries
# no attribution column at all, so an actor key reaching it would be reach
# without a record — the one combination this feature exists to prevent.
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_actor_key_is_rejected_on_add_lines(client, db_cursor, actors):
    seeded = _seed(db_cursor)

    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/lines",
        json={"lines": [{"product_id": seeded["product_id"], "quantity_lb": 5}]},
        headers=_headers(ACTOR, actors),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "API key not authorized for this endpoint"

    db_cursor.execute(
        "SELECT count(*) AS n FROM sales_order_lines WHERE sales_order_id = %s",
        (seeded["order_id"],),
    )
    assert db_cursor.fetchone()["n"] == 1, "a rejected call must not have added a line"

    # The scoped dashboard key is refused identically — an actor key replaces
    # that key, so the two must agree on every route.
    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/lines",
        json={"lines": [{"product_id": seeded["product_id"], "quantity_lb": 5}]},
        headers=_headers(DASHBOARD, actors),
    )
    assert resp.status_code == 403, resp.text

    # ...and the route is genuinely master-only, not simply broken.
    assert ("POST", "/sales/orders/{order_id}/lines") not in main.DASHBOARD_KEY_ALLOWLIST
    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/lines",
        json={"lines": [{"product_id": seeded["product_id"], "quantity_lb": 5}]},
        headers=_headers(MASTER, actors),
    )
    assert resp.status_code != 403, resp.text


@pytest.mark.db
def test_actor_key_is_rejected_on_line_cancel(client, db_cursor, actors):
    seeded = _seed(db_cursor)
    allocation_id = _allocate(db_cursor, seeded)

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/cancel",
        headers=_headers(ACTOR, actors),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "API key not authorized for this endpoint"

    db_cursor.execute(
        "SELECT line_status FROM sales_order_lines WHERE id = %s", (seeded["line_id"],)
    )
    assert db_cursor.fetchone()["line_status"] == "pending", (
        "a rejected call must not have cancelled the line"
    )
    assert _allocation(db_cursor, allocation_id)["status"] == "active", (
        "a rejected call must not have released the line's reservation"
    )

    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/cancel",
        headers=_headers(DASHBOARD, actors),
    )
    assert resp.status_code == 403, resp.text

    assert ("PATCH", "/sales/orders/{order_id}/lines/{line_id}/cancel") \
        not in main.DASHBOARD_KEY_ALLOWLIST
    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/cancel",
        headers=_headers(MASTER, actors),
    )
    assert resp.status_code != 403, resp.text


# ─────────────────────────────────────────────────────────────────
# Scope, stated exhaustively rather than by example
#
# The tests above name five endpoints an actor key must not reach. Five is a
# sample, and a sample cannot support a claim about a surface. This walks
# EVERY operation the floor GPT's schema declares and asserts the allowlist
# decides each one — which is the honest form of the scope claim:
#
#     floor-EXCLUSIVE endpoints denied; SHARED allowlisted endpoints
#     accepted.
#
# Not "actor keys are denied the floor schema". The floor GPT's schema and
# the dashboard's allowlist overlap on the read endpoints and on two sales
# order writes, and an actor key reaches the overlap by design — it replaces
# the shared dashboard key, which already reaches exactly those routes.
#
# Every operation is resolved to the FastAPI route template first, because
# that template — not the YAML path, and not the request URL — is what
# _route_key() matches against. A schema path that no longer names a real
# route is itself a failure worth hearing about.
# ─────────────────────────────────────────────────────────────────

FLOOR_SCHEMA = ROOT / "gpt-configs" / "schemas" / "openapi-floor.yaml"

_HTTP_METHODS = ("get", "post", "put", "patch", "delete")


def _floor_schema_operations():
    """[(METHOD, yaml path)] for every operation in the floor GPT's schema."""
    import yaml

    spec = yaml.safe_load(FLOOR_SCHEMA.read_text())
    return sorted(
        (method.upper(), path)
        for path, item in (spec.get("paths") or {}).items()
        for method in item
        if method.lower() in _HTTP_METHODS
    )


def _route_template_for(method, path):
    """The FastAPI route template `path` resolves to, or None."""
    for route in main.app.routes:
        if getattr(route, "path", None) == path and method in (getattr(route, "methods", None) or ()):
            return route.path
    return None


FLOOR_OPERATIONS = _floor_schema_operations()


def test_the_floor_schema_operations_all_resolve_to_real_routes():
    """A guard on the guard below: if a schema path stopped matching a route,
    every membership check under it would silently compare against nothing."""
    assert FLOOR_OPERATIONS, "the floor schema declares no operations"
    missing = [(m, p) for m, p in FLOOR_OPERATIONS if _route_template_for(m, p) is None]
    assert not missing, f"floor schema operations with no matching route: {missing}"


@pytest.mark.db
@pytest.mark.parametrize(
    "method,path",
    FLOOR_OPERATIONS,
    ids=[f"{m} {p}" for m, p in FLOOR_OPERATIONS],
)
def test_every_floor_schema_operation_obeys_the_allowlist(client, actors, method, path):
    """In DASHBOARD_KEY_ALLOWLIST -> an actor key is accepted (not 403).
    Not in it -> 403, with the authorization detail, every time.

    Path parameters are filled with '1' so resolve_order_id takes its integer
    branch and no lookup can 404 ahead of the auth dependency, and bodies are
    left empty: a request that gets past auth should fail validation rather
    than perform work. The assertion is about the auth decision only.
    """
    template = _route_template_for(method, path)
    assert template is not None, f"{method} {path} matches no route"

    url = re.sub(r"\{[^}]+\}", "1", path)
    kwargs = {"headers": _headers(ACTOR, actors)}
    if method in ("POST", "PUT", "PATCH"):
        kwargs["json"] = {}
    resp = client.request(method, url, **kwargs)

    allowlisted = (method, template) in main.DASHBOARD_KEY_ALLOWLIST
    if allowlisted:
        assert resp.status_code != 403, (
            f"{method} {template} is on DASHBOARD_KEY_ALLOWLIST, so an actor "
            f"key must reach it: {resp.status_code} {resp.text[:200]}"
        )
    else:
        assert resp.status_code == 403, (
            f"{method} {template} is NOT on DASHBOARD_KEY_ALLOWLIST, so an "
            f"actor key must be refused: {resp.status_code} {resp.text[:200]}"
        )
        assert resp.json()["detail"] == "API key not authorized for this endpoint"


def test_the_shared_and_floor_exclusive_halves_are_both_non_empty():
    """The scope claim has two halves and is only interesting if both are
    populated — an empty 'accepted' half would make the exhaustive test above
    pass while meaning 'actor keys reach nothing'.

    The accepted set is also written down in
    docs/design/so-state-model-findings.md; this is the copy that cannot go
    stale, because the test fails if the two halves change shape.
    """
    shared = [(m, p) for m, p in FLOOR_OPERATIONS
              if (m, _route_template_for(m, p)) in main.DASHBOARD_KEY_ALLOWLIST]
    exclusive = [(m, p) for m, p in FLOOR_OPERATIONS if (m, p) not in shared]

    assert shared, "no floor-schema operation is shared with the dashboard allowlist"
    assert exclusive, "no floor-schema operation is floor-exclusive"

    # The floor's inventory verbs are the point of the exclusive half: they
    # move stock and are not part of the dashboard's surface.
    for op in [("POST", "/make"), ("POST", "/adjust"), ("POST", "/pack"),
               ("POST", "/receive"), ("POST", "/ship"),
               ("POST", "/void/{transaction_id}")]:
        assert op in exclusive, f"{op} must stay floor-exclusive"

    # ...and the shared half is read endpoints plus the two sales-order
    # writes the dashboard already performs with its own key.
    assert ("GET", "/sales/orders") in shared
    assert ("PATCH", "/sales/orders/{order_id}/status") in shared


# ═════════════════════════════════════════════════════════════════
# GET /auth/whoami
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_whoami_for_an_actor_key(client, actors):
    resp = client.get("/auth/whoami", headers=_headers(ACTOR, actors))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "actor": {"name": actors["floor"]["name"], "role": "floor"},
        "key_kind": "actor",
    }


@pytest.mark.db
def test_whoami_for_the_dashboard_key(client, actors):
    resp = client.get("/auth/whoami", headers=_headers(DASHBOARD, actors))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"actor": None, "key_kind": "legacy_dashboard"}


@pytest.mark.db
def test_whoami_for_the_master_key(client, actors):
    resp = client.get("/auth/whoami", headers=_headers(MASTER, actors))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"actor": None, "key_kind": "legacy_ledger"}


@pytest.mark.db
def test_whoami_reports_each_role(client, actors):
    for label in ("owner", "floor", "office"):
        resp = client.get("/auth/whoami", headers={"X-API-Key": actors[label]["key"]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["actor"] == {"name": actors[label]["name"],
                                        "role": actors[label]["role"]}


def test_whoami_is_allowlisted_but_not_in_the_gpt_schema():
    assert ("GET", "/auth/whoami") in main.DASHBOARD_KEY_ALLOWLIST
    yaml = (ROOT / "openapi-gpt-v3.yaml").read_text()
    assert "/auth/whoami" not in yaml, (
        "whoami must not be added to openapi-gpt-v3.yaml — it is at its hard "
        "30-operation ceiling and no GPT needs this route"
    )


# ═════════════════════════════════════════════════════════════════
# The cache
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_authentication_costs_one_load_per_ttl_not_one_per_request(client, actors):
    """The point of the cache: a burst of authenticated requests must not be a
    burst of queries."""
    headers = _headers(ACTOR, actors)
    for _ in range(6):
        assert client.get("/auth/whoami", headers=headers).status_code == 200
    assert main._actor_cache_loads == 1, (
        f"expected exactly one actor-table load, got {main._actor_cache_loads}"
    )


@pytest.mark.db
def test_a_new_key_is_invisible_until_the_ttl_elapses(client, db_cursor, actors, monkeypatch):
    """A key minted seconds ago may not work yet — mint, wait a minute, hand it
    out. The alternative is letting an unknown key force a DB round-trip, which
    hands an unauthenticated caller a free denial-of-service lever."""
    # Warm the cache so the new row below lands behind it.
    assert client.get("/auth/whoami", headers=_headers(ACTOR, actors)).status_code == 200
    assert main._actor_cache_loads == 1

    fresh_key = f"actor-key-fresh-{uuid4().hex[:8]}"
    db_cursor.execute(
        "INSERT INTO actors (name, role, key_hash, active) VALUES (%s, 'office', %s, true)",
        (f"ACT fresh {uuid4().hex[:8]}", sha256(fresh_key.encode()).hexdigest()),
    )

    resp = client.get("/auth/whoami", headers={"X-API-Key": fresh_key})
    assert resp.status_code == 403, "the cache is still warm; the new key is not visible yet"
    assert main._actor_cache_loads == 1, "an unknown key must not force a refresh"

    # Age the cache past its TTL.
    import time as _time
    monkeypatch.setattr(
        main, "_actor_cache_loaded_at",
        _time.monotonic() - main.ACTOR_CACHE_TTL_S - 1,
    )

    resp = client.get("/auth/whoami", headers={"X-API-Key": fresh_key})
    assert resp.status_code == 200, resp.text
    assert resp.json()["key_kind"] == "actor"
    assert main._actor_cache_loads == 2


@pytest.mark.db
def test_a_deactivated_key_stops_working_after_the_ttl(client, db_cursor, actors, monkeypatch):
    """Deactivation is eventually-consistent within one TTL, and this pins the
    'eventually' so nobody mistakes it for an incident-response control."""
    headers = _headers(ACTOR, actors)
    assert client.get("/auth/whoami", headers=headers).status_code == 200

    db_cursor.execute("UPDATE actors SET active = false WHERE id = %s",
                      (actors["floor"]["id"],))
    assert client.get("/auth/whoami", headers=headers).status_code == 200, (
        "still cached — deactivation is not instant, by design"
    )

    import time as _time
    monkeypatch.setattr(
        main, "_actor_cache_loaded_at",
        _time.monotonic() - main.ACTOR_CACHE_TTL_S - 1,
    )
    assert client.get("/auth/whoami", headers=headers).status_code == 403


# ─────────────────────────────────────────────────────────────────
# The TTL is a revocation BOUND, not a hint
#
# "Deactivation takes effect within 60 seconds" is only true if the two
# clocks a refresh has — when it started and when it landed — are both
# accounted for. Each test below is one interleaving where using the wrong
# clock keeps a deactivated key alive PAST the TTL, which is the only kind
# of staleness that matters here.
#
# Both drive _load_actors through a stub rather than the database: the point
# is the cache's own bookkeeping under a specific ordering of loads, and a
# real query cannot be made to take three TTLs or to finish out of order on
# demand.
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_a_refresh_slower_than_the_ttl_is_never_served(client, actors, monkeypatch):
    """A load that outlives the TTL has produced an answer that is ALREADY
    expired when it arrives — the rows are as old as the moment the query
    began. Publishing it and calling it fresh would restart the clock on
    rows read before the deactivation, making the real staleness bound
    TTL + load duration instead of TTL.

    The TTL is shrunk rather than the load being made to take a real minute.
    """
    import time as _time

    monkeypatch.setattr(main, "ACTOR_CACHE_TTL_S", 0.2)
    key = actors["floor"]["key"]
    key_hash = sha256(key.encode()).hexdigest()
    still_active = {
        key_hash: {
            "id": actors["floor"]["id"],
            "name": actors["floor"]["name"],
            "role": "floor",
            "key_hash": key_hash,
        }
    }

    loads = []

    def _slow_first_load():
        loads.append(len(loads))
        if len(loads) == 1:
            # Outlives its own TTL. Its rows were read BEFORE the actor was
            # deactivated, so they still carry the key.
            _time.sleep(main.ACTOR_CACHE_TTL_S * 3)
            return still_active
        # By now the actor is gone.
        return {}

    main._reset_actor_cache()
    monkeypatch.setattr(main, "_load_actors", _slow_first_load)

    assert main._resolve_actor(key) is None, (
        "the slow load's snapshot was older than the TTL the moment it "
        "arrived; serving it would have authenticated a deactivated key"
    )
    assert len(loads) == 2, (
        f"the expired-on-arrival snapshot must be refreshed, not served "
        f"(loads: {len(loads)})"
    )

    resp = client.get("/auth/whoami", headers=_headers(ACTOR, actors))
    assert resp.status_code == 403, (
        f"the next request after the TTL must fail auth: {resp.status_code} {resp.text}"
    )


@pytest.mark.db
def test_an_older_refresh_never_overwrites_a_newer_snapshot(client, actors, monkeypatch):
    """Two refreshes overlap and the one that STARTED first FINISHES last. It
    is holding the staler rows, so publishing it on the strength of having
    landed most recently would walk a deactivation back — and the cache would
    then serve the revoked key for another full TTL.

    Ordering is forced with events, not timing, so there is no race to lose.
    """
    import threading

    key = actors["floor"]["key"]
    key_hash = sha256(key.encode()).hexdigest()
    before_deactivation = {
        key_hash: {
            "id": actors["floor"]["id"],
            "name": actors["floor"]["name"],
            "role": "floor",
            "key_hash": key_hash,
        }
    }

    entered = threading.Event()
    release = threading.Event()
    started = []
    counter_lock = threading.Lock()

    def _fake_load():
        with counter_lock:
            nth = len(started)
            started.append(nth)
        if nth == 0:
            entered.set()
            assert release.wait(10), "the older refresh was never released"
            return before_deactivation      # stale rows, returned last
        return {}                           # fresh rows: the actor is gone

    main._reset_actor_cache()
    monkeypatch.setattr(main, "_load_actors", _fake_load)

    older_result = {}

    def _older_refresh():
        older_result["snapshot"] = main._actors_by_hash()

    older = threading.Thread(target=_older_refresh, name="older-refresh")
    older.start()
    try:
        assert entered.wait(10), "the older refresh never reached its load"

        # Starts second, finishes first.
        assert main._actors_by_hash() == {}
    finally:
        release.set()
        older.join(timeout=10)
    assert not older.is_alive(), "the older refresh never finished"

    assert main._actor_cache == {}, (
        "the older refresh overwrote the newer snapshot with its staler rows"
    )
    assert older_result["snapshot"] == {}, (
        "a refresh must return the published snapshot, not the stale rows it "
        "happens to be holding"
    )
    assert main._resolve_actor(key) is None

    resp = client.get("/auth/whoami", headers=_headers(ACTOR, actors))
    assert resp.status_code == 403, (
        f"the next request must fail auth: {resp.status_code} {resp.text}"
    )


@pytest.mark.db
def test_a_refresh_that_cannot_beat_the_ttl_fails_closed(client, actors, monkeypatch):
    """When every attempt outlives the TTL there is no fresh snapshot to be
    had. The choice is between serving a stale one and authenticating nobody,
    and for a revocation bound the second is the only safe answer. Bounded, so
    a permanently slow database degrades instead of spinning forever."""
    import time as _time

    monkeypatch.setattr(main, "ACTOR_CACHE_TTL_S", 0.05)
    key = actors["floor"]["key"]
    key_hash = sha256(key.encode()).hexdigest()

    loads = []

    def _always_slow():
        loads.append(len(loads))
        _time.sleep(main.ACTOR_CACHE_TTL_S * 3)
        return {
            key_hash: {
                "id": actors["floor"]["id"],
                "name": actors["floor"]["name"],
                "role": "floor",
                "key_hash": key_hash,
            }
        }

    main._reset_actor_cache()
    monkeypatch.setattr(main, "_load_actors", _always_slow)

    assert main._resolve_actor(key) is None, "must fail closed, not serve stale rows"
    assert len(loads) == main.ACTOR_CACHE_REFRESH_ATTEMPTS, (
        f"retries must be bounded at {main.ACTOR_CACHE_REFRESH_ATTEMPTS} "
        f"(got {len(loads)})"
    )


@pytest.mark.db
def test_a_slow_FAILING_load_is_not_retried(client, actors, monkeypatch):
    """The retry above exists for a load that was slow. A load that FAILED
    slowly must be taken at its word and left alone: a connection timeout is
    the likely shape of a failure here, and three of those inside one request
    is a multi-minute hang bolted onto an auth check.

    The empty snapshot such a load publishes is already the documented
    degradation — every key falls through to pre-FR-15 behaviour — so there is
    nothing a retry could improve.
    """
    import time as _time

    monkeypatch.setattr(main, "ACTOR_CACHE_TTL_S", 0.05)
    loads = []

    def _slow_boom():
        loads.append(len(loads))
        _time.sleep(main.ACTOR_CACHE_TTL_S * 3)
        raise RuntimeError("connection timed out")

    main._reset_actor_cache()
    monkeypatch.setattr(main, "_load_actors", _slow_boom)

    assert main._resolve_actor(actors["floor"]["key"]) is None
    assert loads == [0], f"a failed load must not be retried (attempts: {len(loads)})"

    # ...and the legacy keys are untouched by any of it.
    assert client.get("/auth/whoami", headers=_headers(MASTER, actors)).status_code == 200
    assert client.get("/auth/whoami", headers=_headers(DASHBOARD, actors)).status_code == 200


@pytest.mark.db
def test_last_used_at_is_stamped_and_then_throttled(client, db_cursor, actors):
    """One write per key per 10 minutes, not one per request."""
    headers = _headers(ACTOR, actors)
    actor_id = actors["floor"]["id"]

    db_cursor.execute("SELECT last_used_at FROM actors WHERE id = %s", (actor_id,))
    assert db_cursor.fetchone()["last_used_at"] is None

    assert client.get("/auth/whoami", headers=headers).status_code == 200
    db_cursor.execute("SELECT last_used_at FROM actors WHERE id = %s", (actor_id,))
    first = db_cursor.fetchone()["last_used_at"]
    assert first is not None, "the first authenticated request stamps last_used_at"

    for _ in range(4):
        assert client.get("/auth/whoami", headers=headers).status_code == 200
    db_cursor.execute("SELECT last_used_at FROM actors WHERE id = %s", (actor_id,))
    assert db_cursor.fetchone()["last_used_at"] == first, (
        "last_used_at must be throttled, not written on every request"
    )


# ─────────────────────────────────────────────────────────────────
# The last_used_at throttle: a shortcut in memory, a BOUND in the database
#
# The in-memory map is per PROCESS. Railway runs more than one worker,
# workers restart, and a fresh one starts with an empty map — so an
# in-memory throttle alone actually means "one write per key per worker per
# 10 minutes, plus one per restart". The predicate in the UPDATE's WHERE
# clause is what makes the documented bound true across every connection
# that can reach the row; the map's remaining job is to keep that bound from
# costing a query on every authenticated request.
# ─────────────────────────────────────────────────────────────────

class _PerThreadSeen:
    """Stands in for main._actor_last_used_seen, giving each thread a map of
    its own — which is what two worker PROCESSES have.

    With the real shared map the in-memory shortcut alone suppresses the
    second caller and the database predicate is never reached, so the test
    below would pass whether or not that predicate exists.
    """

    def __init__(self):
        import threading
        self._local = threading.local()

    @property
    def _map(self):
        if not hasattr(self._local, "map"):
            self._local.map = {}
        return self._local.map

    def get(self, key, default=None):
        return self._map.get(key, default)

    def __setitem__(self, key, value):
        self._map[key] = value

    def __contains__(self, key):
        return key in self._map

    def pop(self, key, default=None):
        return self._map.pop(key, default)

    def clear(self):
        self._map.clear()


@pytest.mark.db
def test_the_last_used_throttle_is_enforced_by_the_database(monkeypatch):
    """Two connections, two throttle maps, one actor row, concurrently:
    exactly one write.

    Real committed rows on real connections, following the concurrency
    harness in tests/test_sales_order_state_model.py, because the claim is
    about what TWO transactions do to one row — which a single rolled-back
    fixture transaction cannot show. Cleanup propagates rather than being
    swallowed: a leaked actor row is a symptom, not a tidiness problem.
    """
    import os
    import threading

    import psycopg2
    from psycopg2.extras import RealDictCursor

    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")

    setup = psycopg2.connect(url)
    setup.autocommit = True
    conns = {}
    actor_id = None
    try:
        with setup.cursor(cursor_factory=RealDictCursor) as cur:
            # Idempotent by construction — see test_migration_052_rerun_is_a_no_op.
            _apply_052(cur)
            token = uuid4().hex[:8].upper()
            key = f"actor-key-throttle-{token}"
            key_hash = sha256(key.encode()).hexdigest()
            cur.execute(
                "INSERT INTO actors (name, role, key_hash, active) "
                "VALUES (%s, 'office', %s, true) RETURNING id",
                (f"ACT throttle {token}", key_hash),
            )
            actor_id = cur.fetchone()["id"]

        actor = {"id": actor_id, "name": f"ACT throttle {token}",
                 "role": "office", "key_hash": key_hash}

        names = ("worker-a", "worker-b")
        for name in names:
            conn = psycopg2.connect(url)
            conn.autocommit = False
            conns[name] = conn

        @contextmanager
        def _per_thread_connection():
            conn = conns[threading.current_thread().name]
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        monkeypatch.setattr(main, "get_db_connection", _per_thread_connection)
        monkeypatch.setattr(main, "_actor_last_used_seen", _PerThreadSeen())

        start = threading.Barrier(len(names))
        wrote = {}

        def _touch():
            start.wait(timeout=10)
            wrote[threading.current_thread().name] = main._touch_actor_last_used(actor)

        threads = [threading.Thread(target=_touch, name=n) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), f"{t.name} never finished"

        assert sorted(wrote.values()) == [False, True], (
            f"exactly one of two concurrent connections may write "
            f"last_used_at, got {wrote}"
        )

        with setup.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT last_used_at FROM actors WHERE id = %s", (actor_id,))
            assert cur.fetchone()["last_used_at"] is not None, (
                "the one write that was allowed must have landed"
            )
    finally:
        for conn in conns.values():
            conn.close()
        try:
            if actor_id is not None:
                with setup.cursor() as cur:
                    cur.execute("DELETE FROM actors WHERE id = %s", (actor_id,))
        finally:
            setup.close()


@pytest.mark.db
def test_the_sql_throttle_interval_matches_the_in_memory_one(db_cursor):
    """The two throttles express one interval and must not drift apart. The
    database itself is asked what the SQL literal means, so this cannot be
    satisfied by two constants that merely look alike."""
    assert "interval '10 minutes'" in main._ACTOR_LAST_USED_SQL
    assert "last_used_at IS NULL" in main._ACTOR_LAST_USED_SQL, (
        "a never-used key must still get its first stamp"
    )
    db_cursor.execute("SELECT EXTRACT(EPOCH FROM interval '10 minutes') AS secs")
    assert float(db_cursor.fetchone()["secs"]) == float(main.ACTOR_LAST_USED_THROTTLE_S)


@pytest.mark.db
def test_the_in_memory_shortcut_keeps_the_hot_path_query_free(client, actors, monkeypatch):
    """The database predicate is the bound; the map is what keeps the bound
    from putting an UPDATE in front of every authenticated read."""
    real = main._write_actor_last_used
    issued = []

    def _counting(actor_id):
        issued.append(actor_id)
        return real(actor_id)

    monkeypatch.setattr(main, "_write_actor_last_used", _counting)

    headers = _headers(ACTOR, actors)
    for _ in range(6):
        assert client.get("/auth/whoami", headers=headers).status_code == 200
    assert issued == [actors["floor"]["id"]], (
        f"six authenticated requests must cost exactly one last_used_at "
        f"query, not {len(issued)}"
    )


@pytest.mark.db
def test_a_failing_last_used_write_never_fails_the_request(client, db_cursor, actors, monkeypatch):
    """Failure injection at the database write itself.

    last_used_at is telemetry bolted onto an auth check. A pool exhaustion or
    a lock timeout on that UPDATE must never be the reason a valid read — or,
    worse, a valid administrative exit — returns 500.
    """
    attempts = []

    def _explode(actor_id):
        attempts.append(actor_id)
        raise RuntimeError("connection pool exhausted")

    monkeypatch.setattr(main, "_write_actor_last_used", _explode)

    headers = _headers(ACTOR, actors)
    key_hash = sha256(actors["floor"]["key"].encode()).hexdigest()

    resp = client.get("/auth/whoami", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["key_kind"] == "actor"
    assert attempts == [actors["floor"]["id"]]

    # The in-memory stamp is given back, so one transient failure does not
    # take the column dark for ten minutes.
    assert key_hash not in main._actor_last_used_seen

    # And a WRITE path, which is where a 500 would actually cost something.
    seeded = _seed(db_cursor)
    resp = client.post(
        f"/sales/orders/{seeded['order_id']}/close",
        json={"reason": "short_closed", "mode": "commit"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert len(attempts) == 2, "the next request must retry, not stay throttled"

    order = _state_changed_by(db_cursor, seeded["order_id"])
    assert order["state"] == "closed"
    _assert_attribution(order["state_changed_by"], ACTOR, actors)


@pytest.mark.db
def test_legacy_keys_never_touch_the_actors_table(client, actors):
    """The resolution order is load-bearing: a legacy key must return before
    the actor lookup, so it cannot be affected by the table's contents, its
    absence, or a failed load."""
    main._reset_actor_cache()
    for which in (MASTER, DASHBOARD):
        resp = client.get("/auth/whoami", headers=_headers(which, actors))
        assert resp.status_code == 200, resp.text
        assert resp.json()["actor"] is None
    assert main._actor_cache_loads == 0, (
        "a legacy key must not trigger an actors load"
    )


@pytest.mark.db
def test_a_failed_actor_load_degrades_to_pre_fr15_behaviour(client, actors, monkeypatch):
    """If migration 052 has not been applied on this database, the load raises.
    That must cache empty and leave the legacy keys working, not 500."""
    def _boom():
        raise RuntimeError('relation "actors" does not exist')

    main._reset_actor_cache()
    monkeypatch.setattr(main, "_load_actors", _boom)

    assert client.get("/auth/whoami", headers=_headers(MASTER, actors)).status_code == 200
    assert client.get("/auth/whoami", headers=_headers(DASHBOARD, actors)).status_code == 200
    resp = client.get("/auth/whoami", headers=_headers(ACTOR, actors))
    assert resp.status_code == 403, "no actor is resolvable, so the key is unknown"


# ═════════════════════════════════════════════════════════════════
# The packing slip's ?key= must not reach the access log
#
# GET /sales/orders/{id}/packing-slip takes its key as a QUERY parameter,
# because a browser following a printable link cannot set a header. Uvicorn's
# access logger writes the request line including the query string, so
# without redaction every fetch deposits a live credential in the platform
# log — and a REJECTED fetch deposits the key someone tried, which is the
# worse half: it turns log access into key material.
# ═════════════════════════════════════════════════════════════════

SLIP_KEY = "packing-slip-secret-key-4f2a9c"


def _uvicorn_access_record(path_with_query):
    """The access line Uvicorn actually emits, argument for argument.

    Taken from uvicorn/protocols/http/*_impl.py: the format string is
    '%s - "%s %s HTTP/%s" %d' and the third argument is
    get_path_with_query_string(scope). Reproducing the CALL rather than the
    already-formatted string is the point — the filter has to reach into
    record.args, which is where the credential actually sits.
    """
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:54321", "GET", path_with_query, "1.1", 401),
        exc_info=None,
    )


def test_uvicorn_still_logs_the_query_string_it_is_being_filtered_for():
    """Pins the assumption the filter is built on. If Uvicorn ever stopped
    putting the query string in the access line, this test would say so
    instead of the redaction test quietly passing on a string that no longer
    carries a key."""
    from uvicorn.protocols.utils import get_path_with_query_string

    scope = {
        "path": "/sales/orders/12/packing-slip",
        "raw_path": None,
        "query_string": f"key={SLIP_KEY}".encode(),
    }
    assert SLIP_KEY in get_path_with_query_string(scope)


def test_the_redaction_filter_is_installed_on_the_access_logger():
    installed = [f for f in logging.getLogger("uvicorn.access").filters
                 if isinstance(f, main.RedactKeyQueryParam)]
    assert len(installed) == 1, (
        f"expected exactly one redaction filter on uvicorn.access, got {len(installed)}"
    )
    # Importing main twice must not stack duplicates.
    main._install_access_log_redaction()
    assert len([f for f in logging.getLogger("uvicorn.access").filters
                if isinstance(f, main.RedactKeyQueryParam)]) == 1


def test_the_packing_slip_query_key_never_reaches_the_access_log():
    """Captures the access logger and asserts the key string is absent.

    Both directions are checked: the credential is gone, and the rest of the
    line — path, method, status — survives, because a redaction that ate the
    log line would be traded one problem for another.
    """
    access_logger = logging.getLogger("uvicorn.access")
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Capture()
    access_logger.addHandler(handler)
    previous_level, previous_propagate = access_logger.level, access_logger.propagate
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    try:
        access_logger.handle(_uvicorn_access_record(
            f"/sales/orders/12/packing-slip?key={SLIP_KEY}"
        ))
        access_logger.handle(_uvicorn_access_record(
            f"/sales/orders/12/packing-slip?key={SLIP_KEY}&format=pdf"
        ))
    finally:
        access_logger.removeHandler(handler)
        access_logger.setLevel(previous_level)
        access_logger.propagate = previous_propagate

    assert len(captured) == 2
    for line in captured:
        assert SLIP_KEY not in line, f"the plaintext key reached the access log: {line}"
        assert "key=[REDACTED]" in line
        assert "/sales/orders/12/packing-slip" in line, "the path must survive"
        assert "GET" in line and "401" in line, "method and status must survive"
    assert "format=pdf" in captured[1], "other query parameters must survive"


def test_the_real_keys_are_redacted_too():
    """The rejection case is what put a key in the log, but the ACCEPTED case
    logs a working credential — so the filter is checked against every key
    kind this app has, not just an invented string."""
    for label, key in (("master", main.API_KEY), ("dashboard", main.DASHBOARD_API_KEY)):
        if not key:
            continue
        redacted = main._redact_key_query_param(
            f"/sales/orders/12/packing-slip?key={key}"
        )
        assert key not in redacted, label
        assert redacted == "/sales/orders/12/packing-slip?key=[REDACTED]", label


def test_redaction_leaves_everything_that_is_not_a_key_parameter_alone():
    """A filter that over-matches would quietly destroy log lines. `monkey=`
    and `keyword=` both contain 'key=' and must survive intact."""
    for untouched in (
        "/lots/by-code/ABC?monkey=1",
        "/products/search?keyword=key=notaparam",
        "/sales/orders",
    ):
        assert main._redact_key_query_param(untouched) == untouched, untouched


@pytest.mark.db
def test_the_packing_slip_query_key_path_still_works_as_before(client, actors, db_cursor):
    """Redaction is a logging change and must not have moved the endpoint's
    own behaviour: the query-param dependency still accepts a valid key and
    still rejects an unknown one with 401, not 403."""
    seeded = _seed(db_cursor)

    resp = client.get(
        f"/sales/orders/{seeded['order_id']}/packing-slip?key={SLIP_KEY}"
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid API key"

    resp = client.get(
        f"/sales/orders/{seeded['order_id']}/packing-slip?key={main.API_KEY}"
    )
    assert resp.status_code == 200, resp.text[:300]


# ═════════════════════════════════════════════════════════════════
# Migration 052
# ═════════════════════════════════════════════════════════════════

def test_migration_052_has_no_transaction_control():
    """It is applied by pasting into the Supabase SQL editor, which runs each
    execution in its own transaction. A COMMIT inside the pasted text aborts
    that run."""
    sql = MIGRATION_052.read_text()
    statements = [ln.strip().upper() for ln in sql.splitlines()]
    assert "BEGIN;" not in statements
    assert "COMMIT;" not in statements
    assert "ROLLBACK;" not in statements


@pytest.mark.db
def test_migration_052_rerun_is_a_no_op(db_cursor, actors):
    """Re-running against a database that already has it changes nothing:
    no error, no duplicate marker, no lost or added rows, same columns and
    same constraints."""
    cur = db_cursor

    def snapshot():
        cur.execute(
            "SELECT column_name, data_type, is_nullable, column_default "
            "  FROM information_schema.columns "
            " WHERE table_schema = 'public' AND table_name = 'actors' "
            " ORDER BY column_name"
        )
        columns = [tuple(r.values()) for r in cur.fetchall()]
        cur.execute(
            "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
            " WHERE conrelid = 'public.actors'::regclass ORDER BY conname"
        )
        constraints = [tuple(r.values()) for r in cur.fetchall()]
        cur.execute("SELECT count(*) AS n FROM actors")
        rows = cur.fetchone()["n"]
        cur.execute(
            "SELECT count(*) AS n FROM migration_markers WHERE name = '052_actors'"
        )
        markers = cur.fetchone()["n"]
        return columns, constraints, rows, markers

    before = snapshot()
    assert before[3] == 1, "the marker row is written on the first apply"

    _apply_052(cur)   # must not raise
    _apply_052(cur)   # twice, for good measure

    assert snapshot() == before


@pytest.mark.db
def test_actors_table_shape(db_cursor, actors):
    """The columns FR-15 actually depends on, and the role vocabulary."""
    cur = db_cursor
    cur.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        " WHERE table_schema = 'public' AND table_name = 'actors'"
    )
    cols = {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in cur.fetchall()}
    assert cols["name"] == ("text", "NO")
    assert cols["role"] == ("text", "NO")
    assert cols["key_hash"] == ("text", "NO")
    assert cols["active"] == ("boolean", "NO")
    assert cols["created_at"] == ("timestamp with time zone", "NO")
    assert cols["last_used_at"] == ("timestamp with time zone", "YES")

    cur.execute("SAVEPOINT bad_role")
    with pytest.raises(Exception):
        cur.execute(
            "INSERT INTO actors (name, role, key_hash) VALUES ('bad', 'admin', 'h')"
        )
    cur.execute("ROLLBACK TO SAVEPOINT bad_role")

    cur.execute("SAVEPOINT dup_hash")
    with pytest.raises(Exception):
        cur.execute(
            "INSERT INTO actors (name, role, key_hash) VALUES (%s, 'floor', %s)",
            (f"dup {uuid4().hex[:6]}",
             sha256(actors["floor"]["key"].encode()).hexdigest()),
        )
    cur.execute("ROLLBACK TO SAVEPOINT dup_hash")


def test_mint_script_roles_match_the_migration_check():
    """The script validates roles before printing keys. If its vocabulary ever
    drifted from the CHECK constraint, the failure would land at paste time —
    after the keys had been generated and handed out."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mint_actor_keys", ROOT / "scripts" / "mint_actor_keys.py"
    )
    mint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mint)

    sql = MIGRATION_052.read_text()
    for role in mint.ROLES:
        assert f"'{role}'" in sql, f"role {role!r} is not in the 052 CHECK constraint"
    assert set(mint.ROLES) == {"owner", "floor", "office"}


def test_mint_script_writes_hashes_and_never_plaintext(tmp_path, capsys):
    """The security property, asserted rather than asserted-in-a-comment: the
    file it writes contains the sha256 of each key and not the key itself."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mint_actor_keys", ROOT / "scripts" / "mint_actor_keys.py"
    )
    mint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mint)

    out = tmp_path / "actors_insert.sql"
    assert mint.main(["Blubber:owner", "Arturo:floor", "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    sql = out.read_text()

    keys = [line.split()[-1] for line in printed.splitlines()
            if line.startswith("  ") and len(line.split()) == 3]
    assert len(keys) == 2, printed
    for key in keys:
        assert key not in sql, "a plaintext key was written to disk"
        assert sha256(key.encode()).hexdigest() in sql, "the hash is missing"

    assert "ON CONFLICT (name) DO UPDATE" in sql
    assert "active   = true" in sql

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


@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_update_line_records_the_actor(client, db_cursor, actors, which):
    """The one-liner this PR fixes.

    update_order_line wrote released_by from the operator-id placeholder — the
    constant 'legacy-shared-key' on every call — at both its expire and its
    shrink site. It is the fourth and last of the allocation writers to be routed
    through caller_source_tag; the other three were fixed earlier and are
    guarded by test_released_by_attribution.py.
    """
    seeded = _seed(db_cursor)
    _allocate(db_cursor, seeded, 100.0)

    # 100 lb reserved, the line cut to 10: the 90 lb of excess is shed by
    # _shrink_active_allocations, which is one of the two sites that used to
    # write the placeholder.
    resp = client.patch(
        f"/sales/orders/{seeded['order_id']}/lines/{seeded['line_id']}/update"
        f"?quantity_lb=10",
        headers=_headers(which, actors),
    )
    assert resp.status_code == 200, resp.text

    released = _released_for_line(db_cursor, seeded["line_id"])
    assert released, "cutting the line must shed the excess reservation"
    for row in released:
        _assert_attribution(row["released_by"], which, actors)


@pytest.mark.db
def test_update_line_no_longer_references_the_operator_placeholder():
    """Source-level guard, matching the one test_released_by_attribution.py
    keeps over the other three handlers. _operator_id() is deliberately still
    live in other subsystems, so this is scoped to this handler."""
    import inspect

    src = inspect.getsource(main.update_order_line)
    assert "_operator_id(" not in src, (
        "update_order_line must attribute releases with caller_source_tag(request), "
        "not the _operator_id placeholder"
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

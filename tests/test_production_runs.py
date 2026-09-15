"""Scheduling S1 — production_runs + run_coverage (migration 053).

A run is one finished SKU, one planned quantity, on one day. Coverage links
pounds of a run to sales_order_lines. Nothing here is read by Health,
readiness or availability until S2, so what these tests defend is narrower
than it looks:

  * the seven routes exist, are on DASHBOARD_KEY_ALLOWLIST, and are in no
    GPT yaml (openapi-gpt-v3.yaml still has exactly 30 operations)
  * migrations/053_production_runs.sql has no transaction control and
    re-runs as a no-op
  * units: planned_qty_lb is canonical and is products.case_size_lb ×
    planned_qty for 'cases', with case_size_lb_used recorded; a SKU with no
    case size must be planned in lb
  * CRUD + validation, over-coverage rejected, coverage on a closed or
    cancelled line / order rejected, coverage capped at the line's EFFECTIVE
    remaining pounds
  * evidence: looks_complete / partial / none from the POSTED ledger only,
    inside the ±1-day business_date window
  * completion stamps completed_by from the actor and touches NO sales-order
    row, flag, allocation or ledger line
  * the three-key attribution matrix (actor name / 'dashboard' / NULL) on
    every write
  * no new path locks a products, lots or sales_order_allocations row

The lock SEQUENCE of the five write paths is pinned in
tests/test_sales_order_state_model.py (EXPECTED_LOCK_SEQUENCE), next to the
eleven sales-order writers, so there is one place that knows the normative
order.

Same TestClient + savepoint-proxy pattern as test_actor_attribution.py.
"""

import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

import main

ROOT = Path(__file__).resolve().parent.parent
MIGRATION_052 = ROOT / "migrations" / "052_actors.sql"
MIGRATION_053 = ROOT / "migrations" / "053_production_runs.sql"
GPT_SCHEMA = ROOT / "openapi-gpt-v3.yaml"
FLOOR_SCHEMA = ROOT / "gpt-configs" / "schemas" / "openapi-floor.yaml"

ET = ZoneInfo("America/New_York")
PLANNED = date(2026, 9, 16)

RUN_ROUTES = {
    ("GET", "/production/runs"),
    ("POST", "/production/runs"),
    ("PATCH", "/production/runs/{run_id}"),
    ("POST", "/production/runs/{run_id}/cancel"),
    ("POST", "/production/runs/{run_id}/complete"),
    ("GET", "/production/runs/{run_id}/evidence"),
    ("PUT", "/production/runs/{run_id}/coverage"),
}

RUN_WRITE_HANDLERS = (
    "create_production_run",
    "update_production_run",
    "cancel_production_run",
    "complete_production_run",
    "put_production_run_coverage",
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

class _ConnProxy:
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


def _apply(cur, path):
    """Apply a migration inside the test transaction, with NO parameters:
    psycopg2 only interprets %-sequences when args are passed, and the
    RAISE NOTICE lines contain one."""
    cur.execute(path.read_text())


@pytest.fixture
def schema(db_cursor):
    """052 (actors) and 053 (runs) applied on the test transaction, rolled
    back on teardown. Both are idempotent, so a test database that already
    carries them from a prod schema dump is fine."""
    _apply(db_cursor, MIGRATION_052)
    _apply(db_cursor, MIGRATION_053)
    _apply(db_cursor, ROOT / "migrations/054_run_type.sql")
    return db_cursor


@pytest.fixture
def actors(schema):
    token = uuid4().hex[:8].upper()
    name = f"RUN actor {token}"
    key = f"run-actor-key-{token}"
    schema.execute(
        "INSERT INTO actors (name, role, key_hash, active) VALUES (%s, 'floor', %s, true)",
        (name, sha256(key.encode()).hexdigest()),
    )
    main._reset_actor_cache()
    yield {"name": name, "key": key}
    main._reset_actor_cache()


@pytest.fixture
def client(schema, monkeypatch):
    conn = schema.connection

    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(conn, "prod_runs_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as test_client:
        yield test_client


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
    return {"X-API-Key": actors["key"]}


def _expected(which, actors):
    return {MASTER: None, DASHBOARD: "dashboard"}.get(which, actors["name"])


DASH = {"X-API-Key": "test-dashboard-key"}


# ─────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────

def _product(cur, token, *, label, ptype="finished", case_size_lb=7.5,
             active=True, no_production=False, is_service=False):
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, uom, case_size_lb, is_service, "
        "                      active, no_production) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (f"RUN {label} {token}", ptype, f"RUN-{label}-{token}",
         "12x10 oz case" if case_size_lb else "lb", case_size_lb, is_service,
         active, no_production),
    )
    return cur.fetchone()["id"]


def _seed(cur, *, qty=100.0, case_size_lb=7.5, state="open", status="confirmed"):
    """A finished SKU, a customer, one open order with one line of `qty` lb."""
    token = uuid4().hex[:8].upper()
    product_id = _product(cur, token, label="FG", case_size_lb=case_size_lb)
    cur.execute("INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id",
                (f"RUN Customer {token}",))
    customer_id = cur.fetchone()["id"]
    order_number = f"RUN-SO-{token}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status, requested_ship_date) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (customer_id, order_number, status, PLANNED + timedelta(days=3)),
    )
    order_id = cur.fetchone()["id"]
    if state != "open":
        reason = "short_closed" if state == "closed" else "customer_cancelled"
        cur.execute(
            "UPDATE sales_orders SET state = %s, state_reason = %s WHERE id = %s",
            (state, reason, order_id),
        )
    line_id = _line(cur, order_id, product_id, qty)
    return {"token": token, "product_id": product_id, "customer_id": customer_id,
            "order_id": order_id, "order_number": order_number, "line_id": line_id}


def _line(cur, order_id, product_id, qty, line_status="pending"):
    cur.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, %s, 0, %s) RETURNING id",
        (order_id, product_id, qty, line_status),
    )
    return cur.fetchone()["id"]


def _production_line(cur, token, code):
    cur.execute(
        "INSERT INTO production_lines (name, line_code, active) VALUES (%s, %s, true) RETURNING id",
        (f"RUN line {code} {token}", f"run_{code}_{token}".lower()),
    )
    return cur.fetchone()["id"]


def _assign(cur, product_id, line_id):
    cur.execute(
        "INSERT INTO product_line_assignments (product_id, line_id) VALUES (%s, %s)",
        (product_id, line_id),
    )


def _post_output(cur, product_id, qty_lb, on: date, ttype="pack"):
    """A posted make/pack whose output line is `qty_lb` of the product, at
    noon plant time on `on` (so business_date == on)."""
    token = uuid4().hex[:8].upper()
    cur.execute(
        "INSERT INTO lots (product_id, lot_code, entry_source) VALUES (%s, %s, 'pack_output') RETURNING id",
        (product_id, f"RUN-OUT-{token}"),
    )
    lot_id = cur.fetchone()["id"]
    at = datetime(on.year, on.month, on.day, 12, 0, tzinfo=ET)
    cur.execute(
        "INSERT INTO transactions (type, timestamp, occurred_at) VALUES (%s, %s, %s) RETURNING id",
        (ttype, at, at),
    )
    txn = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)",
        (txn, product_id, lot_id, qty_lb),
    )
    return txn


def _void(cur, txn_id):
    cur.execute(
        "INSERT INTO ledger_corrections (target_table, target_id, event_type, "
        " previous_values, replacement_values, reason) "
        "VALUES ('transactions', %s, 'void', '{}', '{}', 'test void')",
        (txn_id,),
    )


def _ship_effective(cur, seeded, qty_lb):
    """A posted ship transaction linked to the line through
    sales_order_shipments — the only thing the readiness formula counts as
    shipped. quantity_shipped_lb on the line is deliberately NOT updated,
    so the test proves the cap uses effective pounds, not the recorded
    mirror."""
    token = uuid4().hex[:8].upper()
    cur.execute(
        "INSERT INTO lots (product_id, lot_code, entry_source, received_at) "
        "VALUES (%s, %s, 'received', NOW()) RETURNING id",
        (seeded["product_id"], f"RUN-SHIP-{token}"),
    )
    lot_id = cur.fetchone()["id"]
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
    rcv = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)", (rcv, seeded["product_id"], lot_id, qty_lb),
    )
    cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('ship', NOW()) RETURNING id")
    ship = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
        "VALUES (%s, %s, %s, %s)", (ship, seeded["product_id"], lot_id, -qty_lb),
    )
    cur.execute(
        "INSERT INTO sales_order_shipments (sales_order_line_id, transaction_id, quantity_lb) "
        "VALUES (%s, %s, %s)", (seeded["line_id"], ship, qty_lb),
    )
    return ship


def _create(client, seeded, *, qty=10, unit="cases", headers=DASH, **extra):
    body = {"run_type": "pack", "product_id": seeded["product_id"], "planned_qty": qty,
            "planned_unit": unit, "planned_date": PLANNED.isoformat(), **extra}
    resp = client.post("/production/runs", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["run"]


def _cover(client, run_id, coverage, headers=DASH):
    return client.put(f"/production/runs/{run_id}/coverage",
                      json={"coverage": coverage}, headers=headers)


def _run_row(cur, run_id):
    cur.execute("SELECT * FROM production_runs WHERE id = %s", (run_id,))
    return dict(cur.fetchone())


def _coverage_rows(cur, run_id):
    cur.execute("SELECT * FROM run_coverage WHERE run_id = %s ORDER BY sales_order_line_id", (run_id,))
    return [dict(r) for r in cur.fetchall()]


# ═════════════════════════════════════════════════════════════════
# Allowlist shape and schema hygiene (no DB)
# ═════════════════════════════════════════════════════════════════

def test_the_seven_routes_are_allowlisted_and_real():
    registered = {(m, r.path) for r in main.app.routes for m in getattr(r, "methods", [])}
    assert RUN_ROUTES <= main.DASHBOARD_KEY_ALLOWLIST
    assert RUN_ROUTES <= registered
    extra = {k for k in main.DASHBOARD_KEY_ALLOWLIST if k[1].startswith("/production/runs")}
    assert extra == RUN_ROUTES, f"unexpected run routes on the allowlist: {extra - RUN_ROUTES}"


def test_no_run_route_is_in_any_gpt_schema():
    for schema_path in (GPT_SCHEMA, FLOOR_SCHEMA):
        assert "/production/runs" not in schema_path.read_text(), schema_path


def test_gpt_schema_still_has_exactly_30_operations():
    text = GPT_SCHEMA.read_text()
    assert len(re.findall(r"^\s*operationId:", text, re.M)) == 30


def test_no_run_write_path_locks_products_lots_or_allocations():
    """Order → lines → run, never a step-3 lock. Docstrings and comments are
    stripped so prose about a lock does not count as one."""
    import inspect

    for name in RUN_WRITE_HANDLERS:
        src = inspect.getsource(getattr(main, name))
        src = re.sub(r'""".*?"""', "", src, flags=re.S)
        src = "\n".join(line.split("#")[0] for line in src.splitlines())
        for forbidden in ("_lock_allocation_product", "_lock_allocation_products",
                          "FROM lots", "sales_order_allocations", "FOR UPDATE\n",
                          "FOR UPDATE\"", "FOR UPDATE "):
            assert forbidden not in src, (name, forbidden)


# ═════════════════════════════════════════════════════════════════
# Migration 053
# ═════════════════════════════════════════════════════════════════

def test_migration_053_has_no_transaction_control():
    statements = [ln.strip().upper() for ln in MIGRATION_053.read_text().splitlines()]
    assert "BEGIN;" not in statements
    assert "COMMIT;" not in statements
    assert "ROLLBACK;" not in statements


def test_migration_053_has_no_speculative_stage_columns():
    sql = MIGRATION_053.read_text().lower()
    for col in ("run_kind", "stage", "parent_run_id", "actual_transaction_id"):
        assert not re.search(rf"^\s*{col}\s", sql, re.M), col


@pytest.mark.db
def test_migration_053_rerun_is_a_no_op(schema):
    cur = schema

    def snapshot():
        out = []
        for table in ("production_runs", "run_coverage"):
            cur.execute(
                "SELECT column_name, data_type, is_nullable, column_default "
                "  FROM information_schema.columns "
                " WHERE table_schema = 'public' AND table_name = %s ORDER BY column_name",
                (table,),
            )
            out.append([tuple(r.values()) for r in cur.fetchall()])
            cur.execute(
                "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
                " WHERE conrelid = %s::regclass ORDER BY conname",
                (f"public.{table}",),
            )
            out.append([tuple(r.values()) for r in cur.fetchall()])
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                " WHERE schemaname = 'public' AND tablename = %s ORDER BY indexname",
                (table,),
            )
            out.append([tuple(r.values()) for r in cur.fetchall()])
            cur.execute(f"SELECT count(*) AS n FROM {table}")
            out.append(cur.fetchone()["n"])
        cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name = '053_production_runs'")
        out.append(cur.fetchone()["n"])
        return out

    before = snapshot()
    assert before[-1] == 1, "the marker row is written on the first apply"
    assert before[3] == 0 and before[7] == 0, "no seed rows"
    _apply(cur, MIGRATION_053)
    _apply(cur, MIGRATION_053)
    assert snapshot() == before


@pytest.mark.db
def test_migration_053_touches_nothing_in_production_schedule(schema):
    cur = schema
    sql = MIGRATION_053.read_text().lower()
    assert "production_schedule" not in sql.replace("production_schedule /", "")
    for table in ("production_schedule", "production_lines", "product_line_assignments"):
        cur.execute("SELECT to_regclass(%s) AS r", (f"public.{table}",))
        assert cur.fetchone()["r"] is not None, table


# ═════════════════════════════════════════════════════════════════
# Create — units, product eligibility, line default
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_create_in_cases_stores_canonical_pounds_and_the_native_quantity(client, schema):
    seeded = _seed(schema, case_size_lb=7.5)
    run = _create(client, seeded, qty=1800, unit="cases", notes="  SS Choc Chip   12x10 ")
    assert run["planned_qty_lb"] == 13500.0
    assert run["planned_qty"] == 1800.0
    assert run["planned_unit"] == "cases"
    assert run["case_size_lb_used"] == 7.5
    assert run["status"] == "planned"
    assert run["planned_date"] == PLANNED.isoformat()
    assert run["notes"] == "SS Choc Chip 12x10"
    assert run["covered_lb"] == 0 and run["coverage"] == []
    row = _run_row(schema, run["id"])
    assert float(row["planned_qty_lb"]) == 13500.0
    assert float(row["case_size_lb_used"]) == 7.5


@pytest.mark.db
def test_create_in_lb_records_no_case_size(client, schema):
    seeded = _seed(schema)
    run = _create(client, seeded, qty=250, unit="lb")
    assert run["planned_qty_lb"] == 250.0
    assert run["planned_qty"] == 250.0
    assert run["planned_unit"] == "lb"
    assert run["case_size_lb_used"] is None


@pytest.mark.db
def test_bulk_sku_with_no_case_size_must_be_planned_in_lb(client, schema):
    seeded = _seed(schema, case_size_lb=None)
    body = {"run_type": "pack", "product_id": seeded["product_id"], "planned_qty": 5,
            "planned_unit": "cases", "planned_date": PLANNED.isoformat()}
    resp = client.post("/production/runs", json=body, headers=DASH)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error_code"] == "CASE_WEIGHT_REQUIRED"
    run = _create(client, seeded, qty=500, unit="lb")
    assert run["planned_qty_lb"] == 500.0


@pytest.mark.db
@pytest.mark.parametrize("qty, unit, code", [
    (0, "lb", "INVALID_QUANTITY"),
    (-3, "cases", "INVALID_QUANTITY"),
    (10, "bags", "INVALID_UNIT"),
])
def test_create_rejects_bad_quantity_or_unit(client, schema, qty, unit, code):
    seeded = _seed(schema)
    body = {"run_type": "pack", "product_id": seeded["product_id"], "planned_qty": qty,
            "planned_unit": unit, "planned_date": PLANNED.isoformat()}
    resp = client.post("/production/runs", json=body, headers=DASH)
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error_code"] == code


@pytest.mark.db
@pytest.mark.parametrize("kwargs, code, status", [
    ({"ptype": "batch"}, "PRODUCT_NOT_SCHEDULABLE", 400),
    ({"no_production": True}, "PRODUCT_NOT_SCHEDULABLE", 400),
    ({"active": False}, "PRODUCT_NOT_SCHEDULABLE", 400),
    ({"is_service": True}, "PRODUCT_NOT_SCHEDULABLE", 400),
])
def test_create_rejects_products_that_are_not_finished_skus_made_here(client, schema, kwargs, code, status):
    product_id = _product(schema, uuid4().hex[:8], label="X", **kwargs)
    body = {"run_type": "pack", "product_id": product_id, "planned_qty": 10, "planned_unit": "lb",
            "planned_date": PLANNED.isoformat()}
    resp = client.post("/production/runs", json=body, headers=DASH)
    assert resp.status_code == status, resp.text
    assert resp.json()["detail"]["error_code"] == code


@pytest.mark.db
def test_create_unknown_product_is_404(client, schema):
    body = {"run_type": "pack", "product_id": 999999999, "planned_qty": 10, "planned_unit": "lb",
            "planned_date": PLANNED.isoformat()}
    resp = client.post("/production/runs", json=body, headers=DASH)
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "PRODUCT_NOT_FOUND"


@pytest.mark.db
def test_line_defaults_from_the_single_assignment_and_only_then(client, schema):
    seeded = _seed(schema)
    pouch = _production_line(schema, seeded["token"], "pouch")
    bulk = _production_line(schema, seeded["token"], "bulk")

    # No assignment → NULL.
    assert _create(client, seeded, qty=1, unit="lb")["line_id"] is None
    # Exactly one → that line, with its label.
    _assign(schema, seeded["product_id"], pouch)
    run = _create(client, seeded, qty=1, unit="lb")
    assert run["line_id"] == pouch
    assert run["line_code"].startswith("run_pouch_")
    # Two → ambiguous → NULL again.
    _assign(schema, seeded["product_id"], bulk)
    assert _create(client, seeded, qty=1, unit="lb")["line_id"] is None
    # Explicit wins, and must exist.
    assert _create(client, seeded, qty=1, unit="lb", line_id=bulk)["line_id"] == bulk
    resp = client.post("/production/runs", json={
        "run_type": "pack", "product_id": seeded["product_id"], "planned_qty": 1, "planned_unit": "lb",
        "planned_date": PLANNED.isoformat(), "line_id": 999999999}, headers=DASH)
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_code"] == "PRODUCTION_LINE_NOT_FOUND"


# ═════════════════════════════════════════════════════════════════
# List
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_list_filters_by_date_status_and_product(client, schema):
    a = _seed(schema)
    b = _seed(schema)
    r1 = _create(client, a, qty=1, unit="lb")
    r2 = _create(client, a, qty=2, unit="lb", planned_date=(PLANNED + timedelta(days=5)).isoformat())
    r3 = _create(client, b, qty=3, unit="lb")
    client.post(f"/production/runs/{r3['id']}/cancel", json={}, headers=DASH)

    def ids(**params):
        resp = client.get("/production/runs", params=params, headers=DASH)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["count"] == len(body["runs"])
        return [r["id"] for r in body["runs"]]

    assert ids(product=a["product_id"]) == [r1["id"], r2["id"]]
    assert ids(**{"from": PLANNED.isoformat(), "to": PLANNED.isoformat(),
                  "product": a["product_id"]}) == [r1["id"]]
    assert ids(status="cancelled", product=b["product_id"]) == [r3["id"]]
    assert ids(status="planned", product=b["product_id"]) == []
    resp = client.get("/production/runs", params={"status": "bogus"}, headers=DASH)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "INVALID_STATUS"


# ═════════════════════════════════════════════════════════════════
# PATCH / cancel / complete
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_patch_edits_fields_and_reconverts_quantity(client, schema):
    seeded = _seed(schema, case_size_lb=7.5)
    run = _create(client, seeded, qty=10, unit="cases")
    resp = client.patch(f"/production/runs/{run['id']}", json={
        "planned_qty": 20, "planned_date": (PLANNED + timedelta(days=1)).isoformat(),
        "notes": "moved", "status": "in_progress"}, headers=DASH)
    assert resp.status_code == 200, resp.text
    out = resp.json()["run"]
    assert out["planned_qty_lb"] == 150.0 and out["planned_qty"] == 20.0
    assert out["planned_unit"] == "cases" and out["case_size_lb_used"] == 7.5
    assert out["planned_date"] == (PLANNED + timedelta(days=1)).isoformat()
    assert out["notes"] == "moved" and out["status"] == "in_progress"
    assert sorted(resp.json()["changed_fields"]) == ["notes", "planned_date", "planned_qty", "status"]
    # Switching the unit keeps the number and re-derives the pounds.
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_unit": "lb"}, headers=DASH)
    assert resp.status_code == 200, resp.text
    out = resp.json()["run"]
    assert out["planned_qty_lb"] == 20.0 and out["planned_unit"] == "lb"
    assert out["case_size_lb_used"] is None
    # Clearing notes / line with null.
    resp = client.patch(f"/production/runs/{run['id']}", json={"notes": None, "line_id": None}, headers=DASH)
    assert resp.status_code == 200, resp.text
    assert resp.json()["run"]["notes"] is None


@pytest.mark.db
def test_patch_rejects_empty_body_and_terminal_statuses(client, schema):
    seeded = _seed(schema)
    run = _create(client, seeded, qty=1, unit="lb")
    resp = client.patch(f"/production/runs/{run['id']}", json={}, headers=DASH)
    assert resp.status_code == 422 and resp.json()["detail"]["error_code"] == "NO_FIELDS"
    for status in ("done", "cancelled", "bogus"):
        resp = client.patch(f"/production/runs/{run['id']}", json={"status": status}, headers=DASH)
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["error_code"] == "INVALID_STATUS_TRANSITION"
        assert "/complete" in resp.json()["detail"]["message"]
    assert _run_row(schema, run["id"])["status"] == "planned"


@pytest.mark.db
def test_patch_cannot_shrink_below_coverage(client, schema):
    seeded = _seed(schema, qty=100)
    run = _create(client, seeded, qty=100, unit="lb")
    assert _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 80}]).status_code == 200
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 79.9}, headers=DASH)
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "RUN_OVERCOVERED"
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) == 100.0
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 80}, headers=DASH)
    assert resp.status_code == 200, resp.text


@pytest.mark.db
def test_cancel_is_terminal_and_leaves_coverage_rows_in_place(client, schema):
    seeded = _seed(schema, qty=100)
    run = _create(client, seeded, qty=100, unit="lb")
    assert _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 50}]).status_code == 200
    resp = client.post(f"/production/runs/{run['id']}/cancel", json={"reason": "oven down"}, headers=DASH)
    assert resp.status_code == 200, resp.text
    out = resp.json()["run"]
    assert out["status"] == "cancelled"
    assert out["notes"] == "Cancelled: oven down"
    assert len(_coverage_rows(schema, run["id"])) == 1, "coverage is filtered at read time, not deleted"
    for path, body in ((f"/production/runs/{run['id']}/cancel", {}),
                       (f"/production/runs/{run['id']}/complete", {})):
        resp = client.post(path, json=body, headers=DASH)
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["error_code"] == "RUN_NOT_ACTIVE"
    resp = client.patch(f"/production/runs/{run['id']}", json={"notes": "x"}, headers=DASH)
    assert resp.status_code == 409 and resp.json()["detail"]["error_code"] == "RUN_NOT_EDITABLE"
    resp = _cover(client, run["id"], [])
    assert resp.status_code == 409 and resp.json()["detail"]["error_code"] == "RUN_NOT_ACTIVE"


@pytest.mark.db
def test_unknown_run_is_404_everywhere(client, schema):
    for method, path, body in (
        ("patch", "/production/runs/999999999", {"notes": "x"}),
        ("post", "/production/runs/999999999/cancel", {}),
        ("post", "/production/runs/999999999/complete", {}),
        ("put", "/production/runs/999999999/coverage", {"coverage": []}),
        ("get", "/production/runs/999999999/evidence", None),
    ):
        resp = getattr(client, method)(path, headers=DASH, **({"json": body} if body is not None else {}))
        assert resp.status_code == 404, (method, path, resp.text)
        assert resp.json()["detail"]["error_code"] == "RUN_NOT_FOUND"


# ═════════════════════════════════════════════════════════════════
# Coverage
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_coverage_is_a_full_replace(client, schema):
    seeded = _seed(schema, qty=100)
    other_line = _line(schema, seeded["order_id"], seeded["product_id"], 60)
    run = _create(client, seeded, qty=200, unit="lb")

    resp = _cover(client, run["id"], [
        {"sales_order_line_id": seeded["line_id"], "qty_lb": 100},
        {"sales_order_line_id": other_line, "qty_lb": 60},
    ])
    assert resp.status_code == 200, resp.text
    out = resp.json()["run"]
    assert out["covered_lb"] == 160.0
    assert [(c["sales_order_line_id"], c["qty_lb"]) for c in out["coverage"]] == [
        (seeded["line_id"], 100.0), (other_line, 60.0)]
    assert out["coverage"][0]["order_number"] == seeded["order_number"]
    assert out["coverage"][0]["sales_order_id"] == seeded["order_id"]

    resp = _cover(client, run["id"], [{"sales_order_line_id": other_line, "qty_lb": 30}])
    assert resp.status_code == 200, resp.text
    assert [(c["sales_order_line_id"], float(c["qty_lb"])) for c in _coverage_rows(schema, run["id"])] == [(other_line, 30.0)]

    resp = _cover(client, run["id"], [])
    assert resp.status_code == 200, resp.text
    assert _coverage_rows(schema, run["id"]) == []


@pytest.mark.db
def test_over_coverage_is_rejected(client, schema):
    seeded = _seed(schema, qty=100)
    other_line = _line(schema, seeded["order_id"], seeded["product_id"], 100)
    run = _create(client, seeded, qty=150, unit="lb")
    resp = _cover(client, run["id"], [
        {"sales_order_line_id": seeded["line_id"], "qty_lb": 100},
        {"sales_order_line_id": other_line, "qty_lb": 51},
    ])
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "RUN_OVERCOVERED"
    assert detail["covered_lb"] == 151.0 and detail["planned_qty_lb"] == 150.0
    assert _coverage_rows(schema, run["id"]) == [], "a rejected PUT changes nothing"


@pytest.mark.db
def test_over_coverage_is_judged_at_database_precision(client, schema):
    """Codex cross-review P2 on PR #52. The sum was validated on the unrounded
    floats while each row was stored as numeric(14,4): a 100.0006 lb run
    accepted ten lines of 10.00006 lb (raw Σ 100.0006), which landed as
    10.0001 each = 100.0010 lb, over the plan by 0.0004. Every quantity is now
    quantized to 4 places (ROUND_HALF_UP) BEFORE validation and the same
    values are inserted, so Σ stored coverage ≤ stored plan holds exactly."""
    seeded = _seed(schema, qty=20)
    lines = [seeded["line_id"]] + [
        _line(schema, seeded["order_id"], seeded["product_id"], 20) for _ in range(9)]
    run = _create(client, seeded, qty=100.0006, unit="lb")
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) == 100.0006

    # Codex's exact case: ten open lines at 10.00006 lb each.
    resp = _cover(client, run["id"], [{"sales_order_line_id": lid, "qty_lb": 10.00006} for lid in lines])
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "RUN_OVERCOVERED"
    assert detail["covered_lb"] == 100.001 and detail["planned_qty_lb"] == 100.0006
    assert _coverage_rows(schema, run["id"]) == [], "a rejected PUT changes nothing"

    # The rounded sum exactly equals the plan: ten × 10.0001 = 100.0010 on a
    # plan whose input 100.00095 quantizes (half up) to 100.0010.
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 100.00095}, headers=DASH)
    assert resp.status_code == 200, resp.text
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) == 100.001
    resp = _cover(client, run["id"], [{"sales_order_line_id": lid, "qty_lb": 10.00006} for lid in lines])
    assert resp.status_code == 200, resp.text
    rows = _coverage_rows(schema, run["id"])
    assert [float(r["qty_lb"]) for r in rows] == [10.0001] * 10
    assert sum(float(r["qty_lb"]) for r in rows) == pytest.approx(100.001)
    assert resp.json()["run"]["covered_lb"] == pytest.approx(100.001)
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) >= sum(
        r["qty_lb"] for r in rows), "Σ stored coverage ≤ stored plan, exactly"


@pytest.mark.db
def test_patch_planned_quantity_is_judged_at_database_precision(client, schema):
    """Same policy on PATCH: the plan is quantized before it is compared to
    the stored coverage, so a value that rounds to exactly the covered total
    is accepted and one that rounds below it is not — in both units."""
    seeded = _seed(schema, qty=200, case_size_lb=7.5)
    run = _create(client, seeded, qty=200, unit="lb")
    assert _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 100.001}]).status_code == 200
    # 100.00095 → 100.0010 == covered: accepted, stored at exactly the plan validated.
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 100.00095}, headers=DASH)
    assert resp.status_code == 200, resp.text
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) == 100.001
    # 100.00094 → 100.0009 < covered: rejected, plan unchanged.
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 100.00094}, headers=DASH)
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "RUN_OVERCOVERED"
    assert resp.json()["detail"]["planned_qty_lb"] == 100.0009
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) == 100.001
    # Cases convert in Decimal and quantize the same way: 13.33346 × 7.5 =
    # 100.00095 → 100.0010 (accepted); 13.33345 × 7.5 = 100.001875 → 100.0019.
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 13.33346, "planned_unit": "cases"}, headers=DASH)
    assert resp.status_code == 200, resp.text
    assert float(_run_row(schema, run["id"])["planned_qty_lb"]) == 100.001
    # A quantity that rounds to 0.0000 lb cannot be stored (CHECK > 0): 422, not 500.
    resp = client.patch(f"/production/runs/{run['id']}", json={"planned_qty": 0.00004, "planned_unit": "lb"}, headers=DASH)
    assert resp.status_code == 422 and resp.json()["detail"]["error_code"] == "INVALID_QUANTITY"


@pytest.mark.db
def test_coverage_cannot_exceed_the_lines_effective_remaining(client, schema):
    seeded = _seed(schema, qty=100)
    run = _create(client, seeded, qty=500, unit="lb")
    resp = _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 100.5}])
    assert resp.status_code == 409 and resp.json()["detail"]["error_code"] == "COVERAGE_EXCEEDS_REMAINING"
    # Ship 40 lb effectively (posted ship linked to the line; the recorded
    # mirror is left at 0): remaining is now 60, not 100.
    _ship_effective(schema, seeded, 40)
    resp = _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 61}])
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["remaining_lb"] == 60.0
    resp = _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 60}])
    assert resp.status_code == 200, resp.text


@pytest.mark.db
@pytest.mark.parametrize("line_status", ["cancelled", "fulfilled"])
def test_coverage_on_a_closed_line_is_rejected(client, schema, line_status):
    seeded = _seed(schema, qty=100)
    closed_line = _line(schema, seeded["order_id"], seeded["product_id"], 50, line_status=line_status)
    run = _create(client, seeded, qty=100, unit="lb")
    resp = _cover(client, run["id"], [{"sales_order_line_id": closed_line, "qty_lb": 10}])
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "LINE_NOT_OPEN"
    assert resp.json()["detail"]["line_status"] == line_status


@pytest.mark.db
@pytest.mark.parametrize("state", ["closed", "cancelled"])
def test_coverage_on_a_line_of_an_exited_order_is_rejected(client, schema, state):
    seeded = _seed(schema, qty=100, state=state)
    run = _create(client, seeded, qty=100, unit="lb")
    resp = _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 10}])
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error_code"] == "ORDER_NOT_OPEN" and detail["state"] == state


@pytest.mark.db
def test_coverage_line_must_yield_the_runs_product(client, schema):
    a = _seed(schema, qty=100)
    b = _seed(schema, qty=100)
    run = _create(client, a, qty=100, unit="lb")
    resp = _cover(client, run["id"], [{"sales_order_line_id": b["line_id"], "qty_lb": 10}])
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_code"] == "LINE_PRODUCT_MISMATCH"


@pytest.mark.db
def test_coverage_input_validation(client, schema):
    seeded = _seed(schema, qty=100)
    run = _create(client, seeded, qty=100, unit="lb")
    resp = _cover(client, run["id"], [{"sales_order_line_id": 999999999, "qty_lb": 10}])
    assert resp.status_code == 404 and resp.json()["detail"]["error_code"] == "LINE_NOT_FOUND"
    resp = _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 0}])
    assert resp.status_code == 422 and resp.json()["detail"]["error_code"] == "INVALID_QUANTITY"
    resp = _cover(client, run["id"], [
        {"sales_order_line_id": seeded["line_id"], "qty_lb": 10},
        {"sales_order_line_id": seeded["line_id"], "qty_lb": 10}])
    assert resp.status_code == 422 and resp.json()["detail"]["error_code"] == "DUPLICATE_LINE"
    service = _product(schema, seeded["token"], label="SVC", is_service=True, case_size_lb=None)
    svc_line = _line(schema, seeded["order_id"], service, 0)
    resp = _cover(client, run["id"], [{"sales_order_line_id": svc_line, "qty_lb": 1}])
    assert resp.status_code == 422 and resp.json()["detail"]["error_code"] == "SERVICE_LINE_NOT_COVERABLE"


@pytest.mark.db
def test_coverage_across_two_orders_is_accepted_and_ordered(client, schema):
    a = _seed(schema, qty=100)
    b_line = _line(schema, _seed(schema, qty=100)["order_id"], a["product_id"], 100)
    run = _create(client, a, qty=200, unit="lb")
    resp = _cover(client, run["id"], [
        {"sales_order_line_id": b_line, "qty_lb": 70},
        {"sales_order_line_id": a["line_id"], "qty_lb": 30},
    ])
    assert resp.status_code == 200, resp.text
    assert [c["qty_lb"] for c in resp.json()["run"]["coverage"]] == [30.0, 70.0]


# ═════════════════════════════════════════════════════════════════
# Evidence and completion
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_evidence_none_partial_and_looks_complete(client, schema):
    seeded = _seed(schema, case_size_lb=7.5)
    run = _create(client, seeded, qty=100, unit="cases")     # 750 lb
    url = f"/production/runs/{run['id']}/evidence"

    ev = client.get(url, headers=DASH).json()
    assert ev["suggested_state"] == "none"
    assert ev["recorded_lb"] == 0 and ev["transactions"] == []
    assert ev["window"] == {"from": (PLANNED - timedelta(days=1)).isoformat(),
                            "to": (PLANNED + timedelta(days=1)).isoformat()}
    assert ev["planned_qty_lb"] == 750.0 and ev["planned_qty"] == 100.0

    t1 = _post_output(schema, seeded["product_id"], 300, PLANNED)
    ev = client.get(url, headers=DASH).json()
    assert ev["suggested_state"] == "partial"
    assert ev["recorded_lb"] == 300.0 and ev["recorded_qty"] == 40.0
    assert [t["transaction_id"] for t in ev["transactions"]] == [t1]

    t2 = _post_output(schema, seeded["product_id"], 450, PLANNED + timedelta(days=1), ttype="pack")
    ev = client.get(url, headers=DASH).json()
    assert ev["suggested_state"] == "looks_complete"
    assert ev["recorded_lb"] == 750.0
    assert [(t["transaction_id"], t["type"]) for t in ev["transactions"]] == [(t1, "pack"), (t2, "pack")]


@pytest.mark.db
def test_evidence_ignores_outside_the_window_voids_and_other_products(client, schema):
    seeded = _seed(schema)
    other = _seed(schema)
    run = _create(client, seeded, qty=100, unit="lb")
    url = f"/production/runs/{run['id']}/evidence"
    _post_output(schema, seeded["product_id"], 100, PLANNED - timedelta(days=2))   # outside
    _post_output(schema, seeded["product_id"], 100, PLANNED + timedelta(days=2))   # outside
    _post_output(schema, other["product_id"], 100, PLANNED)                        # other SKU
    voided = _post_output(schema, seeded["product_id"], 100, PLANNED)
    assert client.get(url, headers=DASH).json()["suggested_state"] == "looks_complete"
    _void(schema, voided)
    ev = client.get(url, headers=DASH).json()
    assert ev["suggested_state"] == "none", ev
    assert ev["recorded_lb"] == 0


@pytest.mark.db
def test_completion_stamps_the_run_and_touches_nothing_else(client, schema, actors):
    seeded = _seed(schema, qty=100)
    run = _create(client, seeded, qty=100, unit="lb")
    assert _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 100}]).status_code == 200
    schema.execute("INSERT INTO sales_order_flags (so_number, ready) VALUES (%s, false)", (seeded["order_number"],))

    def snapshot():
        cur = schema
        cur.execute("SELECT * FROM sales_orders WHERE id = %s", (seeded["order_id"],))
        order = dict(cur.fetchone())
        cur.execute("SELECT * FROM sales_order_lines WHERE sales_order_id = %s ORDER BY id", (seeded["order_id"],))
        lines = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM sales_order_flags WHERE so_number = %s", (seeded["order_number"],))
        flags = dict(cur.fetchone())
        cur.execute("SELECT count(*) AS n FROM sales_order_allocations WHERE sales_order_id = %s", (seeded["order_id"],))
        allocations = cur.fetchone()["n"]
        on_hand = main._product_on_hand(cur, seeded["product_id"])
        cur.execute("SELECT count(*) AS n FROM transactions")
        txns = cur.fetchone()["n"]
        cur.execute("SELECT count(*) AS n FROM lots WHERE product_id = %s", (seeded["product_id"],))
        lots = cur.fetchone()["n"]
        return order, lines, flags, allocations, on_hand, txns, lots, _coverage_rows(cur, run["id"])

    before = snapshot()
    assert before[4] == 0.0, "no stock before completion"

    resp = client.post(f"/production/runs/{run['id']}/complete", json={"note": "confirmed by floor"},
                       headers={"X-API-Key": actors["key"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run"]["status"] == "done"
    assert body["run"]["completed_by"] == actors["name"]
    assert body["run"]["completed_at"] is not None
    assert body["run"]["notes"] == "confirmed by floor"
    assert body["evidence"]["suggested_state"] == "none", "completion is a statement, not evidence"

    row = _run_row(schema, run["id"])
    assert row["status"] == "done" and row["completed_by"] == actors["name"]
    assert row["completed_at"] is not None and row["updated_by"] == actors["name"]
    assert snapshot() == before, "completing a run creates no inventory, satisfies no SO, sets no flag"

    # And the sales-order surfaces still say what they said: no Ready flag,
    # line still pending, nothing shipped.
    detail = client.get(f"/sales/orders/{seeded['order_id']}", headers=DASH).json()
    assert detail["fulfillment"] == "unshipped"
    assert detail.get("ready") in (False, None)
    keys = set(detail) | {k for line in detail.get("lines", []) for k in line}
    keys |= set(detail.get("health", {})) | set(detail.get("readiness", {}) or {})
    assert not {k for k in keys if "run" in k.lower() or "coverage" in k.lower()}, \
        "S1 adds nothing to the sales-order detail response"


@pytest.mark.db
def test_completion_from_in_progress_and_evidence_after_done(client, schema):
    seeded = _seed(schema)
    run = _create(client, seeded, qty=100, unit="lb")
    assert client.patch(f"/production/runs/{run['id']}", json={"status": "in_progress"}, headers=DASH).status_code == 200
    _post_output(schema, seeded["product_id"], 100, PLANNED)
    resp = client.post(f"/production/runs/{run['id']}/complete", json={}, headers=DASH)
    assert resp.status_code == 200, resp.text
    assert resp.json()["evidence"]["suggested_state"] == "looks_complete"
    ev = client.get(f"/production/runs/{run['id']}/evidence", headers=DASH).json()
    assert ev["status"] == "done" and ev["suggested_state"] == "looks_complete"


# ═════════════════════════════════════════════════════════════════
# Attribution matrix — every write, three keys
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
@pytest.mark.parametrize("which", ALL_KEYS)
def test_every_write_records_the_actor(client, schema, actors, which):
    headers = _headers(which, actors)
    expected = _expected(which, actors)
    seeded = _seed(schema, qty=100)

    run = _create(client, seeded, qty=100, unit="lb", headers=headers)
    row = _run_row(schema, run["id"])
    assert row["created_by"] == expected and row["updated_by"] == expected

    # PATCH: updated_by, created_by untouched (stamp it differently first).
    schema.execute("UPDATE production_runs SET updated_by = 'someone-else' WHERE id = %s", (run["id"],))
    assert client.patch(f"/production/runs/{run['id']}", json={"notes": "n"}, headers=headers).status_code == 200
    row = _run_row(schema, run["id"])
    assert row["updated_by"] == expected and row["created_by"] == expected

    # PUT coverage: run_coverage.created_by and the run's updated_by.
    schema.execute("UPDATE production_runs SET updated_by = 'someone-else' WHERE id = %s", (run["id"],))
    assert _cover(client, run["id"], [{"sales_order_line_id": seeded["line_id"], "qty_lb": 10}],
                  headers=headers).status_code == 200
    assert [c["created_by"] for c in _coverage_rows(schema, run["id"])] == [expected]
    assert _run_row(schema, run["id"])["updated_by"] == expected

    # complete: completed_by and updated_by.
    schema.execute("UPDATE production_runs SET updated_by = 'someone-else' WHERE id = %s", (run["id"],))
    assert client.post(f"/production/runs/{run['id']}/complete", json={}, headers=headers).status_code == 200
    row = _run_row(schema, run["id"])
    assert row["completed_by"] == expected and row["updated_by"] == expected

    # cancel: updated_by, on a fresh run.
    run2 = _create(client, seeded, qty=1, unit="lb", headers=headers)
    schema.execute("UPDATE production_runs SET updated_by = 'someone-else' WHERE id = %s", (run2["id"],))
    assert client.post(f"/production/runs/{run2['id']}/cancel", json={"reason": "r"}, headers=headers).status_code == 200
    assert _run_row(schema, run2["id"])["updated_by"] == expected


@pytest.mark.db
def test_all_three_key_kinds_reach_every_route_and_an_unknown_key_does_not(client, schema, actors):
    seeded = _seed(schema)
    run = _create(client, seeded, qty=1, unit="lb")
    for headers in ({"X-API-Key": main.API_KEY}, DASH, {"X-API-Key": actors["key"]}):
        assert client.get("/production/runs", headers=headers).status_code == 200
        assert client.get(f"/production/runs/{run['id']}/evidence", headers=headers).status_code == 200
    for method, path in (("get", "/production/runs"), ("post", "/production/runs"),
                         ("patch", f"/production/runs/{run['id']}"),
                         ("put", f"/production/runs/{run['id']}/coverage")):
        resp = client.request(method.upper(), path, headers={"X-API-Key": "nope"}, json={})
        assert resp.status_code == 403, (method, path, resp.text)
        assert client.request(method.upper(), path, json={}).status_code == 401

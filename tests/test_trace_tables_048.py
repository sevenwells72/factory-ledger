"""Migration 048 (trace tables — TRACEABILITY_DESIGN.md §3.2/§3.3, §9 step 2)
coverage. Schema only: no emission code exists yet, so every test drives the
tables directly with SQL.

1. Migration application: applies cleanly on a fresh DB (post-047 schema.sql
   base) and re-runs as a no-op.
2. Constraint behavior: correction-pairing CHECK (void/restore/amend require
   correction_id, operational events forbid it), tel_role_sign role/sign
   combinations, the NULLS NOT DISTINCT (transaction_id, event_type,
   correction_id) unique, the (event, lot, role) unique, and the generated
   late_recorded flag around the 24 h boundary.
3. 039 protections: UPDATE/DELETE rejected on both tables; created_at and
   created_at_source are database-owned except the 'trace_backfill_048'
   provenance carve-out (which also — and only which — may supply a
   historical recorded_at on trace_events); 046's transactions/api_backfill
   carve-out is unchanged, and the two carve-outs don't cross tables.
4. FK integrity: events need a real transaction (and correction), event-lot
   rows need a real event and lot.

No test uses production; the migration tests build scratch databases on the
guarded TEST_DATABASE_URL host and drop them (same pattern as
test_lot_identity_047.py).
"""

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import pytest

import psycopg2
from psycopg2 import errors as pg_errors
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "tests" / "schema" / "schema.sql"
MIGRATION_048 = ROOT / "migrations" / "048_trace_tables.sql"

# schema.sql is a post-047 (pre-048) prod dump, so a scratch build needs no
# revert step: load the schema, then apply 048 on top.


def _psql():
    for candidate in ("/opt/homebrew/opt/postgresql@17/bin/psql", "psql"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def _scratch_setup():
    test_url = os.environ.get("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL not set — DB-backed tests skipped")
    psql = _psql()
    if psql is None:
        pytest.skip("psql binary not found")
    parsed = urlparse(test_url)
    admin_url = urlunparse(parsed._replace(path="/postgres"))

    def _run(url, *args):
        return subprocess.run(
            [psql, url, "-q", "-v", "ON_ERROR_STOP=1", *args],
            capture_output=True, text=True,
        )

    def build():
        db_name = f"factory_ledger_test_048_{uuid4().hex[:8]}"
        proc = _run(admin_url, "-c", f'CREATE DATABASE "{db_name}"')
        assert proc.returncode == 0, proc.stderr
        url = urlunparse(parsed._replace(path=f"/{db_name}"))
        proc = _run(url, "-c", "CREATE EXTENSION IF NOT EXISTS pg_trgm")
        assert proc.returncode == 0, proc.stderr
        proc = _run(url, "-f", str(SCHEMA))
        assert proc.returncode == 0, f"schema load failed:\n{proc.stderr}"
        return db_name, url

    def apply_048(url):
        return _run(url, "-f", str(MIGRATION_048))

    def drop(db_name):
        _run(admin_url, "-c", f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')

    return build, apply_048, drop


@pytest.fixture
def scratch_db_factory():
    """Per-test builder for migration-application tests; drops its DBs."""
    build, apply_048, drop = _scratch_setup()
    created = []

    def _build():
        db_name, url = build()
        created.append(db_name)
        return url

    yield _build, apply_048

    for db_name in created:
        drop(db_name)


@pytest.fixture(scope="module")
def db048_url():
    """One scratch DB with 048 applied, shared by the constraint/trigger/FK
    tests below (each test runs in its own rolled-back transaction)."""
    build, apply_048, drop = _scratch_setup()
    db_name, url = build()
    proc = apply_048(url)
    assert proc.returncode == 0, f"048 failed:\n{proc.stderr}"
    yield url
    drop(db_name)


@pytest.fixture
def conn(db048_url):
    c = psycopg2.connect(db048_url)
    c.autocommit = False
    yield c
    c.rollback()
    c.close()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


def _seed_txn(conn, txn_type="receive"):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status) "
            "VALUES (%s, NOW(), 'posted') RETURNING id",
            (txn_type,),
        )
        return cur.fetchone()["id"]


def _seed_lot(conn):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, 'ingredient', 'lb', true) RETURNING id",
            (f"Trace048 {uuid4().hex[:10]}",),
        )
        pid = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (pid, f"ZZ048-{uuid4().hex[:8].upper()}"),
        )
        return cur.fetchone()["id"]


def _seed_correction(conn, txn_id, event_type="void"):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO ledger_corrections (target_table, target_id, event_type, "
            "previous_values, replacement_values, reason) "
            "VALUES ('transactions', %s, %s, '{}', '{}', 'test 048') RETURNING id",
            (txn_id, event_type),
        )
        return cur.fetchone()["id"]


_EVENT_COLS = (
    "event_type, epcis_type, transaction_id, correction_id, occurred_at, "
    "business_date"
)


def _insert_event(conn, txn_id, event_type="receive", epcis_type="object",
                  correction_id=None, occurred_at=None, extra_cols=None):
    cols = {
        "event_type": event_type,
        "epcis_type": epcis_type,
        "transaction_id": txn_id,
        "correction_id": correction_id,
        "occurred_at": occurred_at or "2026-08-31T12:00:00+00",
        "business_date": "2026-08-31",
    }
    cols.update(extra_cols or {})
    with _cur(conn) as cur:
        cur.execute(
            f"INSERT INTO trace_events ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))}) RETURNING *",
            list(cols.values()),
        )
        return cur.fetchone()


def _insert_tel(conn, event_id, lot_id, role, qty, extra_cols=None):
    cols = {
        "trace_event_id": event_id,
        "lot_id": lot_id,
        "role": role,
        "quantity_lb": qty,
    }
    cols.update(extra_cols or {})
    with _cur(conn) as cur:
        cur.execute(
            f"INSERT INTO trace_event_lots ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))}) RETURNING *",
            list(cols.values()),
        )
        return cur.fetchone()


def _expect_error(conn, exc_class, fn, constraint=None):
    with conn.cursor() as c:
        c.execute("SAVEPOINT expect_048")
    with pytest.raises(exc_class) as excinfo:
        fn()
    if constraint is not None:
        assert excinfo.value.diag.constraint_name == constraint
    with conn.cursor() as c:
        c.execute("ROLLBACK TO SAVEPOINT expect_048")
        c.execute("RELEASE SAVEPOINT expect_048")
    return excinfo


# ─────────────────────────────────────────────────────────────────
# 1. Migration application (scratch databases)
# ─────────────────────────────────────────────────────────────────

def _assert_048_objects(url):
    with psycopg2.connect(url) as conn, _cur(conn) as cur:
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename IN ('trace_events', 'trace_event_lots')
            ORDER BY indexname
        """)
        names = {r["indexname"] for r in cur.fetchall()}
        assert {
            "trace_events_txn_idx", "trace_events_occurred_idx",
            "trace_events_late_idx", "tel_lot_idx", "tel_lot_role_idx",
        } <= names
        cur.execute("""
            SELECT conname FROM pg_constraint
            WHERE conrelid IN ('public.trace_events'::regclass,
                               'public.trace_event_lots'::regclass)
        """)
        cons = {r["conname"] for r in cur.fetchall()}
        assert {
            "trace_events_correction_pairing", "trace_events_txn_type_uniq",
            "tel_role_sign", "tel_event_lot_role_uniq",
        } <= cons
        # The txn/type unique must treat NULL correction_ids as equal.
        cur.execute("""
            SELECT indnullsnotdistinct FROM pg_index
            WHERE indexrelid = 'public.trace_events_txn_type_uniq'::regclass
        """)
        assert cur.fetchone()["indnullsnotdistinct"] is True
        # 039 protections attached to both tables.
        cur.execute("""
            SELECT tgname FROM pg_trigger
            WHERE tgrelid IN ('public.trace_events'::regclass,
                              'public.trace_event_lots'::regclass)
              AND NOT tgisinternal
        """)
        trgs = {r["tgname"] for r in cur.fetchall()}
        assert trgs == {
            "trg_trace_events_created_at", "trg_trace_events_append_only",
            "trg_trace_event_lots_created_at", "trg_trace_event_lots_append_only",
        }
        # Both tables start empty — the migration is inert schema.
        cur.execute("SELECT (SELECT count(*) FROM trace_events) "
                    "+ (SELECT count(*) FROM trace_event_lots) AS n")
        assert cur.fetchone()["n"] == 0


def test_048_applies_on_fresh_db_and_reruns_as_noop(scratch_db_factory):
    build, apply_048 = scratch_db_factory
    url = build()

    proc = apply_048(url)
    assert proc.returncode == 0, f"048 failed on fresh DB:\n{proc.stderr}"
    _assert_048_objects(url)

    # Idempotent guards: a re-run must also succeed and change nothing.
    proc = apply_048(url)
    assert proc.returncode == 0, f"048 re-run failed:\n{proc.stderr}"
    _assert_048_objects(url)


# ─────────────────────────────────────────────────────────────────
# 2a. Correction-pairing CHECK
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("event_type", ["void", "restore", "amend"])
def test_correction_marker_requires_correction_id(conn, event_type):
    txn = _seed_txn(conn)
    _expect_error(
        conn, pg_errors.CheckViolation,
        lambda: _insert_event(conn, txn, event_type=event_type),
        constraint="trace_events_correction_pairing",
    )
    # With a real correction it lands (no trace_event_lots rows — §3.4 marker).
    corr = _seed_correction(conn, txn, event_type=event_type)
    row = _insert_event(conn, txn, event_type=event_type, correction_id=corr)
    assert str(row["correction_id"]) == str(corr)


@pytest.mark.parametrize(
    "event_type,epcis_type",
    [("receive", "object"), ("make", "transformation"), ("pack", "transformation"),
     ("ship", "object"), ("adjust", "object"), ("merge", "object")],
)
def test_operational_event_forbids_correction_id(conn, event_type, epcis_type):
    txn = _seed_txn(conn)
    corr = _seed_correction(conn, txn)
    _expect_error(
        conn, pg_errors.CheckViolation,
        lambda: _insert_event(conn, txn, event_type=event_type,
                              epcis_type=epcis_type, correction_id=corr),
        constraint="trace_events_correction_pairing",
    )
    row = _insert_event(conn, txn, event_type=event_type, epcis_type=epcis_type)
    assert row["correction_id"] is None
    assert row["event_uuid"] is not None


# ─────────────────────────────────────────────────────────────────
# 2b. (transaction_id, event_type, correction_id) unique
# ─────────────────────────────────────────────────────────────────

def test_duplicate_operational_event_rejected(conn):
    """NULLS NOT DISTINCT: a double-fired emission hook can't insert the same
    (txn, event_type) twice even though correction_id is NULL on both."""
    txn = _seed_txn(conn)
    _insert_event(conn, txn, event_type="receive")
    _expect_error(
        conn, pg_errors.UniqueViolation,
        lambda: _insert_event(conn, txn, event_type="receive"),
        constraint="trace_events_txn_type_uniq",
    )
    # A different event type on the same transaction is fine.
    _insert_event(conn, txn, event_type="adjust")


def test_correction_events_unique_per_correction(conn):
    txn = _seed_txn(conn)
    corr_1 = _seed_correction(conn, txn, "void")
    corr_2 = _seed_correction(conn, txn, "void")
    _insert_event(conn, txn, event_type="void", correction_id=corr_1)
    # Same (txn, 'void') under a DIFFERENT correction: allowed (re-void after
    # restore appends a new marker).
    _insert_event(conn, txn, event_type="void", correction_id=corr_2)
    # The same correction can't emit the same marker twice.
    _expect_error(
        conn, pg_errors.UniqueViolation,
        lambda: _insert_event(conn, txn, event_type="void", correction_id=corr_2),
        constraint="trace_events_txn_type_uniq",
    )


# ─────────────────────────────────────────────────────────────────
# 2c. tel_role_sign + (event, lot, role) unique
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,qty,ok", [
    ("output", 5, True), ("output", -5, False),
    ("received", 5, True), ("received", -5, False),
    ("input", -5, True), ("input", 5, False),
    ("shipped", -5, True), ("shipped", 5, False),
    ("adjusted", 5, True), ("adjusted", -5, True),
    ("moved", 5, True), ("moved", -5, True),
])
def test_tel_role_sign_convention(conn, role, qty, ok):
    txn = _seed_txn(conn)
    event = _insert_event(conn, txn, event_type="adjust")
    lot = _seed_lot(conn)
    if ok:
        row = _insert_tel(conn, event["id"], lot, role, qty)
        assert float(row["quantity_lb"]) == qty
    else:
        _expect_error(
            conn, pg_errors.CheckViolation,
            lambda: _insert_tel(conn, event["id"], lot, role, qty),
            constraint="tel_role_sign",
        )


def test_tel_zero_quantity_rejected(conn):
    txn = _seed_txn(conn)
    event = _insert_event(conn, txn, event_type="adjust")
    lot = _seed_lot(conn)
    _expect_error(
        conn, pg_errors.CheckViolation,
        lambda: _insert_tel(conn, event["id"], lot, "adjusted", 0),
    )


def test_tel_duplicate_event_lot_role_rejected(conn):
    txn = _seed_txn(conn)
    event = _insert_event(conn, txn, event_type="adjust")
    lot = _seed_lot(conn)
    _insert_tel(conn, event["id"], lot, "adjusted", 5)
    _expect_error(
        conn, pg_errors.UniqueViolation,
        lambda: _insert_tel(conn, event["id"], lot, "adjusted", 7),
        constraint="tel_event_lot_role_uniq",
    )
    # Same event+lot under a different role is a different fact — allowed.
    _insert_tel(conn, event["id"], lot, "moved", 5)


# ─────────────────────────────────────────────────────────────────
# 2d. late_recorded around the 24 h boundary
# ─────────────────────────────────────────────────────────────────

OCC = "2026-08-01T12:00:00+00"


def _backfilled_event(conn, txn, recorded_at):
    return _insert_event(
        conn, txn, event_type="receive", occurred_at=OCC,
        extra_cols={"recorded_at": recorded_at,
                    "created_at_source": "trace_backfill_048"},
    )


@pytest.mark.parametrize("recorded_at,late", [
    ("2026-08-01T13:00:00+00", False),           # 1 h after
    ("2026-08-02T11:59:59+00", False),           # 24 h − 1 s
    ("2026-08-02T12:00:00+00", False),           # exactly 24 h — strict >
    ("2026-08-02T12:00:01+00", True),            # 24 h + 1 s
    ("2026-08-05T12:00:00+00", True),            # days later
])
def test_late_recorded_boundary(conn, recorded_at, late):
    txn = _seed_txn(conn)
    row = _backfilled_event(conn, txn, recorded_at)
    assert row["late_recorded"] is late


def test_late_recorded_on_live_insert(conn):
    """Normal (non-backfill) inserts get recorded_at=now from the trigger, so
    an event claimed to have occurred days ago flags late; a fresh one
    doesn't."""
    txn = _seed_txn(conn)
    stale = _insert_event(conn, txn, event_type="receive", occurred_at=OCC)
    assert stale["late_recorded"] is True
    fresh = _insert_event(conn, txn, event_type="adjust",
                          occurred_at="now()")
    assert fresh["late_recorded"] is False


# ─────────────────────────────────────────────────────────────────
# 3a. Append-only
# ─────────────────────────────────────────────────────────────────

def test_trace_events_append_only(conn):
    txn = _seed_txn(conn)
    event = _insert_event(conn, txn, event_type="receive")
    for stmt in (
        "UPDATE trace_events SET operator_id = 'x' WHERE id = %s",
        "DELETE FROM trace_events WHERE id = %s",
    ):
        def _attempt(stmt=stmt):
            with conn.cursor() as c:
                c.execute(stmt, (event["id"],))
        exc = _expect_error(conn, psycopg2.errors.lookup("23000"), _attempt)
        assert "append-only" in str(exc.value)


def test_trace_event_lots_append_only(conn):
    txn = _seed_txn(conn)
    event = _insert_event(conn, txn, event_type="adjust")
    lot = _seed_lot(conn)
    row = _insert_tel(conn, event["id"], lot, "adjusted", 5)
    for stmt in (
        "UPDATE trace_event_lots SET quantity_lb = 6 WHERE id = %s",
        "DELETE FROM trace_event_lots WHERE id = %s",
    ):
        def _attempt(stmt=stmt):
            with conn.cursor() as c:
                c.execute(stmt, (row["id"],))
        exc = _expect_error(conn, psycopg2.errors.lookup("23000"), _attempt)
        assert "append-only" in str(exc.value)


# ─────────────────────────────────────────────────────────────────
# 3b. created_at ownership + the trace_backfill_048 carve-out
# ─────────────────────────────────────────────────────────────────

def test_caller_supplied_entry_times_are_overridden(conn):
    """Without the marker, created_at, created_at_source AND recorded_at are
    all database-owned no matter what the caller sends."""
    txn = _seed_txn(conn)
    row = _insert_event(
        conn, txn, event_type="receive", occurred_at=OCC,
        extra_cols={"created_at": "2020-01-01T00:00:00+00",
                    "created_at_source": "sneaky_caller",
                    "recorded_at": "2020-01-01T00:00:00+00"},
    )
    assert row["created_at_source"] == "database"
    with _cur(conn) as cur:
        cur.execute(
            "SELECT created_at > clock_timestamp() - interval '5 minutes' AS ca_now, "
            "recorded_at > clock_timestamp() - interval '5 minutes' AS ra_now "
            "FROM trace_events WHERE id = %s", (row["id"],))
        got = cur.fetchone()
    assert got["ca_now"] and got["ra_now"]


def test_trace_backfill_carveout_accepted_on_both_tables(conn):
    """created_at_source='trace_backfill_048' survives, and on trace_events it
    unlocks a caller-supplied historical recorded_at. created_at itself stays
    database-owned even for the backfill (matches 046: api_backfill never
    caller-sets created_at either)."""
    txn = _seed_txn(conn)
    row = _backfilled_event(conn, txn, "2026-08-01T13:00:00+00")
    assert row["created_at_source"] == "trace_backfill_048"
    with _cur(conn) as cur:
        cur.execute(
            "SELECT recorded_at = timestamptz '2026-08-01T13:00:00+00' AS ra_kept, "
            "created_at > clock_timestamp() - interval '5 minutes' AS ca_now "
            "FROM trace_events WHERE id = %s", (row["id"],))
        got = cur.fetchone()
    assert got["ra_kept"] and got["ca_now"]

    lot = _seed_lot(conn)
    tel = _insert_tel(conn, row["id"], lot, "received", 5,
                      extra_cols={"created_at_source": "trace_backfill_048"})
    assert tel["created_at_source"] == "trace_backfill_048"


def test_carveouts_do_not_cross_tables(conn):
    txn = _seed_txn(conn)
    # 046's transactions marker buys nothing on the trace tables …
    row = _insert_event(
        conn, txn, event_type="receive", occurred_at=OCC,
        extra_cols={"created_at_source": "api_backfill",
                    "recorded_at": "2026-08-01T13:00:00+00"},
    )
    assert row["created_at_source"] == "database"
    assert row["late_recorded"] is True  # recorded_at forced to now
    # … and the trace marker buys nothing on transactions.
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status, created_at_source) "
            "VALUES ('receive', NOW(), 'posted', 'trace_backfill_048') "
            "RETURNING created_at_source, entry_backfilled")
        got = cur.fetchone()
    assert got["created_at_source"] == "database"
    assert got["entry_backfilled"] is False


def test_046_api_backfill_carveout_unchanged(conn):
    """Regression guard: replacing ledger_enforce_created_at must keep the
    046 transactions behavior byte-identical."""
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status, created_at_source) "
            "VALUES ('receive', NOW(), 'posted', 'api_backfill') "
            "RETURNING created_at_source, entry_backfilled")
        marked = cur.fetchone()
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status) "
            "VALUES ('receive', NOW(), 'posted') "
            "RETURNING created_at_source, entry_backfilled")
        plain = cur.fetchone()
    assert marked["created_at_source"] == "api_backfill"
    assert marked["entry_backfilled"] is True
    assert plain["created_at_source"] == "database"
    assert plain["entry_backfilled"] is False


# ─────────────────────────────────────────────────────────────────
# 4. FK integrity
# ─────────────────────────────────────────────────────────────────

def test_event_requires_real_transaction_and_correction(conn):
    _expect_error(
        conn, pg_errors.ForeignKeyViolation,
        lambda: _insert_event(conn, 999999999, event_type="receive"),
    )
    txn = _seed_txn(conn)
    _expect_error(
        conn, pg_errors.ForeignKeyViolation,
        lambda: _insert_event(conn, txn, event_type="void",
                              correction_id=str(uuid4())),
    )


def test_event_lot_requires_real_event_and_lot(conn):
    txn = _seed_txn(conn)
    event = _insert_event(conn, txn, event_type="adjust")
    lot = _seed_lot(conn)
    _expect_error(
        conn, pg_errors.ForeignKeyViolation,
        lambda: _insert_tel(conn, 999999999, lot, "adjusted", 5),
    )
    _expect_error(
        conn, pg_errors.ForeignKeyViolation,
        lambda: _insert_tel(conn, event["id"], 999999999, "adjusted", 5),
    )

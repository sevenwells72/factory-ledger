"""The in-app startup migrations run once per DATABASE, not once per BOOT.

`main.py` still applies a dozen `ALTER TABLE … ADD COLUMN IF NOT EXISTS` /
`CREATE TABLE IF NOT EXISTS` blocks to itself at startup. IF NOT EXISTS makes
them no-ops in effect but not in cost: every ALTER TABLE takes ACCESS EXCLUSIVE
on its target for the duration of the statement, on every process start. That
is what wedged this suite (docs/design/so-state-model-findings.md, harness
fragility), and it is what `migration_markers` now prevents.

What these tests pin down:

  * a first boot against a database with no markers really does run the DDL,
    and records one marker per gated block;
  * a second boot issues NO `ALTER TABLE` at all — asserted from a statement
    log recorded off the connection itself, not inferred from the schema;
  * the two data sweeps that were deliberately left ungated (Migration 007 and
    the 051 cutover reconciliation) still run on EVERY boot, so a row that
    arrives after the first boot is still swept;
  * two sequential boots are idempotent — no duplicate markers, no re-seeded
    rows.

Everything runs in a throwaway schema on its own autocommit connection, for
the same reason `test_sales_order_state_model.py` does it: the suite's shared
`db_cursor` sits in one long-lived, never-committed transaction, and issuing
ACCESS EXCLUSIVE DDL through it — or against the real `public` tables while it
is open — is precisely the deadlock this change exists to stop.
"""

import logging
import threading
from collections import Counter
from contextlib import contextmanager
from uuid import uuid4

import pytest

import main


# Every gated block, by marker name. A block added to main.py without a marker
# — or a marker renamed — fails the first-boot test rather than silently
# reverting to per-boot DDL.
GATED_MARKERS = {
    "startup_migration_label_type",
    "startup_migration_004",
    "startup_migration_005",
    "startup_migration_006",
    "startup_migration_008",
    "startup_migration_010",
    "startup_migration_011",
    "startup_migration_012",
}


# A database as it looked BEFORE the startup migrations ever ran: the tables
# they touch, without the columns they add. If a column below already existed,
# the corresponding ALTER would be a no-op and the first-boot test would prove
# nothing.
PRE_MIGRATION_SCHEMA_DDL = """
CREATE TABLE products (
    id          serial PRIMARY KEY,
    name        text NOT NULL,
    odoo_code   text
);
CREATE TABLE batch_formulas (
    id                    serial PRIMARY KEY,
    ingredient_product_id integer REFERENCES products(id)
);
CREATE TABLE lots (
    id         serial PRIMARY KEY,
    product_id integer REFERENCES products(id),
    lot_code   text
);
CREATE TABLE customers (
    id   serial PRIMARY KEY,
    name text NOT NULL
);
CREATE TABLE sales_orders (
    id                 serial PRIMARY KEY,
    order_number       text NOT NULL,
    status             text NOT NULL,
    state              text NOT NULL DEFAULT 'open',
    state_reason       text,
    state_note         text,
    state_changed_at   timestamptz,
    state_changed_by   text,
    status_before_exit text
);
CREATE TABLE transactions (
    id            serial PRIMARY KEY,
    type          text,
    customer_name text
);
"""

# Columns and relations the gated blocks are supposed to create.
EXPECTED_COLUMNS = [
    ("products", "label_type"),
    ("products", "yield_multiplier"),
    ("products", "case_size_lb"),
    ("products", "parent_batch_product_id"),
    ("batch_formulas", "exclude_from_inventory"),
    ("lots", "status"),
    ("lots", "merged_into_lot_id"),
    ("lots", "merged_at"),
    ("lots", "merge_reason"),
    ("lots", "supplier_lot_code"),
    ("lots", "lot_type"),
    ("lots", "received_at"),
]
EXPECTED_TABLES = ["customer_aliases", "lot_supplier_codes", "migration_markers"]


# ─────────────────────────────────────────────────────────────────
# A pool that records every statement it is asked to run
# ─────────────────────────────────────────────────────────────────

class _RecordingCursor:
    """Delegates to a real cursor, appending each statement to a shared log."""

    def __init__(self, cur, log):
        self._cur = cur
        self._log = log

    def execute(self, sql, params=None):
        self._log.append(sql if isinstance(sql, str) else sql.decode())
        return self._cur.execute(sql, params) if params is not None else self._cur.execute(sql)

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __enter__(self):
        self._cur.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cur.__exit__(*exc)


class _RecordingConnection:
    def __init__(self, conn, log):
        self._conn = conn
        self._log = log

    def cursor(self, *args, **kwargs):
        return _RecordingCursor(self._conn.cursor(*args, **kwargs), self._log)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _SpyPool:
    """One connection, handed out repeatedly, with a statement log.

    ``putconn`` mirrors psycopg2's own pool: a connection returned mid-
    transaction is rolled back, so a block that failed cannot poison the next
    one.
    """

    def __init__(self, conn):
        self.log = []
        self._conn = _RecordingConnection(conn, self.log)
        self._raw = conn

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        import psycopg2.extensions as ext
        if self._raw.info.transaction_status != ext.TRANSACTION_STATUS_IDLE:
            self._raw.rollback()

    def clear_log(self):
        self.log.clear()

    def ddl_statements(self):
        """Statements that would take a heavy lock on an existing relation.

        `migration_markers`' own CREATE TABLE IF NOT EXISTS is excluded: it is
        the guard's bookkeeping, it targets a table with no other readers, and
        on an existing table it takes no lock beyond the catalog lookup.
        """
        out = []
        for sql in self.log:
            upper = " ".join(sql.split()).upper()
            if "MIGRATION_MARKERS" in upper:
                continue
            if upper.startswith(("ALTER TABLE", "CREATE TABLE", "CREATE INDEX",
                                 "CREATE UNIQUE INDEX")):
                out.append(" ".join(sql.split()))
        return out

    def alter_table_statements(self):
        return [s for s in self.ddl_statements() if s.upper().startswith("ALTER TABLE")]


def _set_autocommit(conn, on):
    """Flip autocommit safely — psycopg2 refuses it inside a transaction."""
    if conn.autocommit == on:
        return
    conn.rollback()
    conn.autocommit = on


@contextmanager
def _pre_051_database(monkeypatch):
    """A throwaway schema shaped like a database that never booted this app.

    Its own autocommit connection, for the reason spelled out in the module
    docstring. `main.db_pool` is pointed at it for the duration, so
    `_run_startup_migrations()` runs against this schema and nothing else.
    """
    import os
    import psycopg2 as pg

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")

    schema = f"startup_mig_{uuid4().hex[:8]}"
    conn = pg.connect(url, application_name="startup-migrations-test")
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
        try:
            with conn.cursor() as cur:
                # SET while autocommit is on, so it survives the rollbacks the
                # gated blocks perform on their skip path.
                cur.execute(f"SET search_path TO {schema}")
                cur.execute(PRE_MIGRATION_SCHEMA_DDL)
            _set_autocommit(conn, False)
            spy = _SpyPool(conn)
            monkeypatch.setattr(main, "db_pool", spy)
            yield spy, conn
        finally:
            _set_autocommit(conn, True)
            with conn.cursor() as cur:
                cur.execute("SET search_path TO public")
                cur.execute(f"DROP SCHEMA {schema} CASCADE")
    finally:
        conn.close()


def _query(conn, sql, params=None):
    from psycopg2.extras import RealDictCursor
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _markers(conn):
    return {r["name"] for r in _query(conn, "SELECT name FROM migration_markers")}


# ─────────────────────────────────────────────────────────────────
# First boot
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_first_boot_runs_the_ddl(monkeypatch):
    """No markers ⇒ the schema work actually happens."""
    with _pre_051_database(monkeypatch) as (spy, conn):
        assert not spy.alter_table_statements(), "precondition: nothing run yet"

        main._run_startup_migrations()

        assert spy.alter_table_statements(), (
            "a first boot must issue the ALTER TABLEs — otherwise the guard has "
            "locked the schema out rather than merely skipping repeat work"
        )
        for table, column in EXPECTED_COLUMNS:
            rows = _query(
                conn,
                "SELECT 1 FROM information_schema.columns "
                " WHERE table_schema = current_schema() "
                "   AND table_name = %s AND column_name = %s",
                (table, column),
            )
            assert rows, f"{table}.{column} was not created by the first boot"
        for table in EXPECTED_TABLES:
            assert _query(conn, "SELECT to_regclass(%s) AS r", (table,))[0]["r"], \
                f"{table} was not created by the first boot"


@pytest.mark.db
def test_first_boot_writes_one_marker_per_gated_block(monkeypatch):
    with _pre_051_database(monkeypatch) as (spy, conn):
        main._run_startup_migrations()
        assert _markers(conn) == GATED_MARKERS


@pytest.mark.db
def test_first_boot_still_applies_the_block_seed_data(monkeypatch):
    """The gate is per BLOCK, so a block's seeding runs with its DDL."""
    with _pre_051_database(monkeypatch) as (spy, conn):
        _set_autocommit(conn, True)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO customers (name) VALUES ('Setton Farms')")
            cur.execute(
                "INSERT INTO products (name, odoo_code) VALUES ('Batch BS Test', 'BS-T')")
        _set_autocommit(conn, False)

        main._run_startup_migrations()

        aliases = _query(conn, "SELECT alias FROM customer_aliases")
        assert sorted(a["alias"] for a in aliases) == sorted(
            ["Setton International", "Setton Intl"])


# ─────────────────────────────────────────────────────────────────
# Second boot — the whole point
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_second_boot_issues_no_alter_table(monkeypatch):
    """Asserted from the statement log, not from the resulting schema.

    A schema-shaped assertion cannot tell "the ALTER was skipped" from "the
    ALTER ran and was a no-op", and the no-op is the expensive case.
    """
    with _pre_051_database(monkeypatch) as (spy, conn):
        main._run_startup_migrations()
        spy.clear_log()

        main._run_startup_migrations()

        assert spy.alter_table_statements() == [], (
            "a booted database must issue no ALTER TABLE: "
            f"{spy.alter_table_statements()}"
        )
        assert spy.ddl_statements() == [], (
            "nor any CREATE TABLE / CREATE INDEX: " f"{spy.ddl_statements()}"
        )


@pytest.mark.db
def test_second_boot_adds_no_duplicate_markers(monkeypatch):
    with _pre_051_database(monkeypatch) as (spy, conn):
        main._run_startup_migrations()
        first = _query(conn, "SELECT name, applied_at FROM migration_markers ORDER BY name")

        main._run_startup_migrations()
        second = _query(conn, "SELECT name, applied_at FROM migration_markers ORDER BY name")

        assert first == second, "markers must be untouched by a repeat boot"
        assert len(first) == len(GATED_MARKERS)


@pytest.mark.db
def test_two_sequential_boots_do_not_re_seed(monkeypatch):
    with _pre_051_database(monkeypatch) as (spy, conn):
        _set_autocommit(conn, True)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO customers (name) VALUES ('QUALI-PACK USA')")
        _set_autocommit(conn, False)

        main._run_startup_migrations()
        after_first = _query(conn, "SELECT count(*) AS n FROM customer_aliases")[0]["n"]

        main._run_startup_migrations()
        after_second = _query(conn, "SELECT count(*) AS n FROM customer_aliases")[0]["n"]

        assert after_first == 3, "the QUALI-PACK aliases seed on the first boot"
        assert after_second == after_first, "the second boot must seed nothing"


# ─────────────────────────────────────────────────────────────────
# The sweeps that are deliberately NOT gated
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_migration_007_sweep_runs_on_every_boot(monkeypatch):
    """A data fix, not DDL. An order that arrives AFTER the first boot must
    still be migrated — which a marker gate would prevent."""
    with _pre_051_database(monkeypatch) as (spy, conn):
        main._run_startup_migrations()

        _set_autocommit(conn, True)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sales_orders (order_number, status) "
                        "VALUES ('LATE-NEW', 'new')")
        _set_autocommit(conn, False)

        main._run_startup_migrations()

        row = _query(conn, "SELECT status FROM sales_orders WHERE order_number = 'LATE-NEW'")
        assert row[0]["status"] == "confirmed"


@pytest.mark.db
def test_051_reconcile_sweep_runs_on_every_boot(monkeypatch):
    with _pre_051_database(monkeypatch) as (spy, conn):
        main._run_startup_migrations()

        _set_autocommit(conn, True)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sales_orders (order_number, status, state) "
                        "VALUES ('LATE-CANCELLED', 'cancelled', 'open')")
        _set_autocommit(conn, False)

        main._run_startup_migrations()

        row = _query(
            conn,
            "SELECT state, state_reason, state_changed_by FROM sales_orders "
            " WHERE order_number = 'LATE-CANCELLED'",
        )[0]
        assert row["state"] == "cancelled"
        assert row["state_reason"] == "other"
        assert row["state_changed_by"] == "startup-reconcile"


@pytest.mark.db
def test_the_sweeps_are_not_marker_gated(monkeypatch):
    """Belt and braces on the two exemptions: no marker may exist that would
    let a future edit quietly gate them."""
    with _pre_051_database(monkeypatch) as (spy, conn):
        main._run_startup_migrations()
        names = _markers(conn)
        assert not any("007" in n or "reconcile" in n or "009" in n for n in names), names


# ─────────────────────────────────────────────────────────────────
# Concurrency
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_a_racing_instance_that_already_wrote_the_marker_is_not_an_error(monkeypatch):
    """Two instances booting together can both see no marker. The DDL is all
    IF NOT EXISTS and the insert is ON CONFLICT DO NOTHING, so the loser is a
    no-op rather than a crash."""
    with _pre_051_database(monkeypatch) as (spy, conn):
        assert main._ensure_migration_markers()
        ran = {"n": 0}

        def _body(cur):
            ran["n"] += 1
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS racy_col text")
            # The other instance commits its marker while this one is working.
            conn2_sql = ("INSERT INTO migration_markers (name) VALUES ('racy') "
                         "ON CONFLICT (name) DO NOTHING")
            cur.execute(conn2_sql)

        main._run_once_startup_migration("racy", _body)
        main._run_once_startup_migration("racy", _body)

        assert ran["n"] == 1, "the second call must find the marker and skip"
        assert _query(conn, "SELECT count(*) AS n FROM migration_markers "
                            " WHERE name = 'racy'")[0]["n"] == 1


@pytest.mark.db
def test_a_failing_block_writes_no_marker_and_is_retried(monkeypatch):
    """Non-fatal, as it has always been — but not silently permanent."""
    with _pre_051_database(monkeypatch) as (spy, conn):
        assert main._ensure_migration_markers()
        calls = {"n": 0}

        def _body(cur):
            calls["n"] += 1
            if calls["n"] == 1:
                cur.execute("SELECT 1 FROM a_table_that_does_not_exist")
            else:
                cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS retried_col text")

        main._run_once_startup_migration("retryable", _body)
        assert "retryable" not in _markers(conn), "a failed block must not be marked done"

        main._run_once_startup_migration("retryable", _body)
        assert "retryable" in _markers(conn)
        assert calls["n"] == 2
        assert _query(
            conn,
            "SELECT 1 FROM information_schema.columns "
            " WHERE table_schema = current_schema() AND table_name = 'products' "
            "   AND column_name = 'retried_col'",
        )


@pytest.mark.db
def test_missing_marker_table_falls_back_to_ungated(monkeypatch):
    """If the bookkeeping table cannot be established, run the DDL as before.

    Skipping schema work because a marker table is missing would be strictly
    worse than the lock cost the guard exists to avoid.
    """
    with _pre_051_database(monkeypatch) as (spy, conn):
        assert main._ensure_migration_markers()
        ran = {"n": 0}

        def _body(cur):
            ran["n"] += 1
            cur.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS ungated_col text")

        main._run_once_startup_migration("ungated", _body, gated=False)
        main._run_once_startup_migration("ungated", _body, gated=False)

        assert ran["n"] == 2, "ungated blocks run every time, as they always did"
        assert "ungated" not in _markers(conn), "and record nothing"


# ─────────────────────────────────────────────────────────────────
# Two instances booting at the same instant
# ─────────────────────────────────────────────────────────────────
# The tests above drive one connection. They cannot see the race the advisory
# lock exists to close, because a single connection cannot observe "no marker"
# while it is itself between its own SELECT and its own COMMIT. These do it for
# real: two connections, two threads, released together on a barrier, against a
# database with no markers AND no `migration_markers` table at all — the state a
# brand-new database is in when a deploy starts two instances at once.

class _ThreadSpyPool:
    """`main.db_pool` for a concurrency test: one `_SpyPool` per thread.

    `main.db_pool` is a module global, so two threads booting at once share one
    pool object. This dispatches on the calling thread, so each "instance" gets
    its own real connection — and its own statement log to be counted later.
    """

    def __init__(self, pools_by_thread):
        self._pools = pools_by_thread

    def _pool(self):
        return self._pools[threading.get_ident()]

    def getconn(self):
        return self._pool().getconn()

    def putconn(self, conn):
        return self._pool().putconn(conn)


@contextmanager
def _concurrent_pre_051_database(monkeypatch, instances=2):
    """A throwaway schema with N independent connections pointed at it.

    Deliberately does NOT create `migration_markers`: two instances issuing
    `CREATE TABLE IF NOT EXISTS` at the same instant is itself a documented
    Postgres race, and it is the first thing the lock has to survive.
    """
    import os
    import psycopg2 as pg

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")

    schema = f"startup_mig_conc_{uuid4().hex[:8]}"
    admin = pg.connect(url, application_name="startup-migrations-conc-admin")
    admin.autocommit = True
    conns = []
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"SET search_path TO {schema}")
            cur.execute(PRE_MIGRATION_SCHEMA_DDL)
        for i in range(instances):
            conn = pg.connect(url, application_name=f"startup-migrations-conc-{i}")
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {schema}")
            conn.autocommit = False
            conns.append(conn)
        registry = {}
        monkeypatch.setattr(main, "db_pool", _ThreadSpyPool(registry))
        yield registry, conns, admin
    finally:
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass
        with admin.cursor() as cur:
            cur.execute("SET search_path TO public")
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        admin.close()


def _single_boot_ddl_baseline(monkeypatch):
    """What one uncontended boot issues — the yardstick for "exactly once"."""
    with _pre_051_database(monkeypatch) as (spy, _conn):
        main._run_startup_migrations()
        return Counter(spy.ddl_statements())


@pytest.mark.db
def test_two_instances_booting_at_once_apply_every_ddl_exactly_once(monkeypatch):
    """The real race, with real connections — not a simulated one.

    Without `pg_advisory_xact_lock` both instances read "no marker" inside the
    same window and both run the whole DDL sequence; the ALTERs are IF NOT
    EXISTS so nothing errors, but every one of them takes ACCESS EXCLUSIVE a
    second time, which is the entire cost this change exists to remove. With
    the lock the loser waits, then finds the marker and issues nothing.
    """
    baseline = _single_boot_ddl_baseline(monkeypatch)
    assert baseline, "precondition: an uncontended boot issues DDL"

    with _concurrent_pre_051_database(monkeypatch) as (registry, conns, admin):
        assert _query(admin, "SELECT to_regclass('migration_markers') AS r")[0]["r"] is None, \
            "precondition: no marker table yet — both instances must create it"

        barrier = threading.Barrier(len(conns))
        results = [{"spy": None, "error": None} for _ in conns]

        def _boot(index, conn):
            spy = _SpyPool(conn)
            registry[threading.get_ident()] = spy
            results[index]["spy"] = spy
            barrier.wait(timeout=30)
            try:
                main._run_startup_migrations()
            except BaseException as exc:      # noqa: BLE001 — recorded, re-raised below
                results[index]["error"] = exc

        threads = [threading.Thread(target=_boot, args=(i, c), name=f"boot-{i}")
                   for i, c in enumerate(conns)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads), (
            "a booting instance never finished — the advisory lock is meant to "
            "make the loser WAIT, not hang"
        )

        for i, result in enumerate(results):
            assert result["error"] is None, f"instance {i} failed to boot: {result['error']!r}"

        # The marker table itself: created once, by whichever instance won.
        assert _query(admin, "SELECT to_regclass('migration_markers') AS r")[0]["r"]
        assert _query(
            admin,
            "SELECT count(*) AS n FROM information_schema.tables "
            " WHERE table_schema = current_schema() AND table_name = 'migration_markers'",
        )[0]["n"] == 1

        # One marker per gated block, one row each.
        rows = _query(admin, "SELECT name, count(*) AS n FROM migration_markers GROUP BY name")
        assert {r["name"] for r in rows} == GATED_MARKERS
        assert all(r["n"] == 1 for r in rows), rows

        # And the point of the whole exercise: no statement ran twice.
        combined = Counter()
        for result in results:
            combined.update(result["spy"].ddl_statements())
        assert combined == baseline, (
            "every DDL statement must be issued exactly once across BOTH "
            "instances; extra copies mean the second instance ran the sequence "
            f"too. Extra: {combined - baseline}, missing: {baseline - combined}"
        )


# ─────────────────────────────────────────────────────────────────
# When migration_markers cannot be used at all
# ─────────────────────────────────────────────────────────────────
# A marker table the app cannot read or write must degrade to the pre-guard
# every-boot behaviour WITH A WARNING — never to "no marker found, and also no
# marker written", and never to schema that silently stops being applied.
#
# These use a second, non-superuser role that OWNS the schema and every table
# the startup blocks touch — so the DDL is within its rights — while
# `migration_markers` is owned by someone else. That is the prod shape: RLS is
# enabled on the public tables with no policies attached, and the owner bypasses
# RLS while a non-owner does not.

@contextmanager
def _marker_table_denied_database(monkeypatch, mode):
    """A pre-051 database whose `migration_markers` is out of the app's reach.

    ``mode`` picks how:

      ``"select"``  INSERT granted, SELECT not  → the marker READ is denied
      ``"insert"``  SELECT granted, INSERT not  → the marker WRITE is denied
      ``"rls"``     both granted, but RLS is enabled with no policies → the
                    read silently returns nothing and the write is refused.
                    The silent read is why this case needs its own test: it
                    looks exactly like "not applied yet".
    """
    import os
    import psycopg2 as pg

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")

    tag = uuid4().hex[:8]
    schema = f"startup_mig_deny_{tag}"
    role = f"startup_mig_role_{tag}"

    admin = pg.connect(url, application_name="startup-migrations-deny-admin")
    admin.autocommit = True
    app = None
    try:
        with admin.cursor() as cur:
            cur.execute(f"CREATE ROLE {role} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE")
            cur.execute(f"CREATE SCHEMA {schema} AUTHORIZATION {role}")

        # The app role builds — and therefore owns — everything the blocks touch.
        app = pg.connect(url, user=role, application_name="startup-migrations-deny-app")
        app.autocommit = True
        with app.cursor() as cur:
            cur.execute(f"SET search_path TO {schema}")
            cur.execute(PRE_MIGRATION_SCHEMA_DDL)

        # …but NOT migration_markers, which admin owns. Same DDL the app would
        # have used, so the shape cannot drift from main.py's.
        with admin.cursor() as cur:
            cur.execute(f"SET search_path TO {schema}")
            cur.execute(main.STARTUP_MIGRATION_MARKERS_DDL)
            if mode == "select":
                cur.execute(f"GRANT INSERT ON {schema}.migration_markers TO {role}")
            elif mode == "insert":
                cur.execute(f"GRANT SELECT ON {schema}.migration_markers TO {role}")
            elif mode == "rls":
                cur.execute(f"GRANT SELECT, INSERT ON {schema}.migration_markers TO {role}")
                cur.execute(f"ALTER TABLE {schema}.migration_markers "
                            f"ENABLE ROW LEVEL SECURITY")
            else:  # pragma: no cover — programming error in the test itself
                raise AssertionError(f"unknown mode {mode!r}")

        _set_autocommit(app, False)
        spy = _SpyPool(app)
        monkeypatch.setattr(main, "db_pool", spy)
        yield spy, admin
    finally:
        if app is not None:
            try:
                app.close()
            except Exception:
                pass
        with admin.cursor() as cur:
            cur.execute("SET search_path TO public")
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"DROP OWNED BY {role}")
            cur.execute(f"DROP ROLE IF EXISTS {role}")
        admin.close()


def _ungated_fallback_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING and "UNGATED" in r.getMessage()]


def _assert_degraded_to_ungated(caplog, spy, admin):
    """Boot succeeded, applied the schema anyway, and said so out loud.

    The schema is checked FIRST on purpose: an unreachable marker table must
    never cost applied schema, and that — not the wording of a log line — is
    what a regression here would take away.
    """
    assert spy.alter_table_statements(), (
        "the fallback must actually issue the DDL — a marker table the app "
        "cannot use must not stop the schema work"
    )
    for table, column in EXPECTED_COLUMNS:
        assert _query(
            admin,
            "SELECT 1 FROM information_schema.columns "
            " WHERE table_schema = current_schema() "
            "   AND table_name = %s AND column_name = %s",
            (table, column),
        ), f"{table}.{column} was not applied by the ungated fallback"
    for table in EXPECTED_TABLES:
        assert _query(admin, "SELECT to_regclass(%s) AS r", (table,))[0]["r"], \
            f"{table} was not created by the ungated fallback"

    # Nothing was recorded, so the next boot does the same thing — loudly.
    assert _query(admin, "SELECT count(*) AS n FROM migration_markers")[0]["n"] == 0

    # And it is loud: one warning per block, naming the block and the cause.
    warnings = _ungated_fallback_warnings(caplog)
    for marker in sorted(GATED_MARKERS):
        assert any(marker in w for w in warnings), (
            f"{marker} must warn by name that it fell back to the ungated path; "
            f"got: {warnings}"
        )
        assert any(marker in w and "migration_markers" in w for w in warnings), (
            f"{marker}'s warning must name the cause, not just the block"
        )


@pytest.mark.db
def test_marker_select_denied_falls_back_to_ungated(monkeypatch, caplog):
    """Permission denied on the marker READ ⇒ run the block, don't skip it."""
    caplog.set_level(logging.WARNING, logger=main.logger.name)
    with _marker_table_denied_database(monkeypatch, "select") as (spy, admin):
        main._run_startup_migrations()
        _assert_degraded_to_ungated(caplog, spy, admin)


@pytest.mark.db
def test_marker_insert_denied_falls_back_to_ungated(monkeypatch, caplog):
    """Permission denied on the marker WRITE ⇒ the block's work is rolled back
    and re-run ungated, not left half-applied and unrecorded."""
    caplog.set_level(logging.WARNING, logger=main.logger.name)
    with _marker_table_denied_database(monkeypatch, "insert") as (spy, admin):
        main._run_startup_migrations()
        _assert_degraded_to_ungated(caplog, spy, admin)


@pytest.mark.db
def test_marker_blocked_by_rls_falls_back_to_ungated(monkeypatch, caplog):
    """RLS on, no policies, non-owner: the read returns nothing and the write
    is refused. The empty read is indistinguishable from "not applied yet", so
    the refused write is what has to catch it — and it does."""
    caplog.set_level(logging.WARNING, logger=main.logger.name)
    with _marker_table_denied_database(monkeypatch, "rls") as (spy, admin):
        main._run_startup_migrations()
        _assert_degraded_to_ungated(caplog, spy, admin)

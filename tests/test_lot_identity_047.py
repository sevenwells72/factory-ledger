"""Migration 047 (lot identity — TRACEABILITY_DESIGN.md §3.1) coverage.

1. Migration application: applies cleanly on a fresh empty DB and on a DB
   carrying the historical data shapes from
   docs/trace-preclean-worklist-2026-08.md — merged twins that RETAIN their
   lot_code ('JUL 15 2026' merged into 'JUL15 2026', 'BB041327 Lot' merged
   into 'BB041327') plus legacy codes that violate the new format CHECK —
   with zero tier-1 index violations at build time, and re-runs as a no-op.
2. Twin minting per worklist shape: exact twin (existing
   lots_product_id_lot_code_key), whitespace variant (allowed + tier-2
   suspicious_code_similarity warning), trailing-'Lot' variant (tier-1 hard
   409), punctuation variant (allowed + warning), same normalized code on a
   different product (allowed, no warning), merged lot's code reusable by a
   new lot (index predicate).
3. lot_uuid: minted at INSERT, unique, survives rename, present in
   receive/make/pack commit responses and lot lookups.

No test uses production; API tests run through the guarded TEST_DATABASE_URL
fixtures and roll back, and the migration tests build scratch databases on
the same guarded host and drop them.
"""

import os
import subprocess
import uuid as uuidlib
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse, urlunparse
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

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "tests" / "schema" / "schema.sql"
MIGRATION_047 = ROOT / "migrations" / "047_lot_identity.sql"
# Marker of the pending-migration include appended to schema.sql; the scratch
# builds strip it so the migration can be applied ON TOP of seeded data.
PENDING_047_MARKER = "PENDING MIGRATION 047"

# The tier-1 index expression (must mirror migrations/047_lot_identity.sql).
T1_SQL = ("regexp_replace(regexp_replace(upper(btrim(lot_code)), '\\s+', ' ', 'g'), "
          "'\\s+LOT$', '')")


# ─────────────────────────────────────────────────────────────────
# Fixtures (same savepoint-proxy pattern as test_audit_savepoints_lotcodes)
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
        proxy = _ConnProxy(_db_connection, "lot047_sp_inner")
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


def _seed_product(conn, name=None, product_type="ingredient"):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, %s, 'lb', true) RETURNING id",
            (name or f"Lot047 {uuid4().hex[:10]}", product_type),
        )
        return cur.fetchone()["id"]


def _seed_lot(conn, product_id, lot_code, **cols):
    with _cur(conn) as cur:
        keys = ["product_id", "lot_code"] + list(cols)
        vals = [product_id, lot_code] + list(cols.values())
        cur.execute(
            f"INSERT INTO lots ({', '.join(keys)}) "
            f"VALUES ({', '.join(['%s'] * len(vals))}) RETURNING id, lot_uuid",
            vals,
        )
        return cur.fetchone()


def _found(client, product_id, lot_code, qty=5):
    return client.post("/inventory/found", json={
        "product_id": product_id,
        "quantity": qty,
        "reason_code": "found_count",
        "lot_code": lot_code,
    })


def _similarity_warnings(body):
    return [w for w in body.get("warnings", [])
            if w.get("warning") == "suspicious_code_similarity"]


def _lot_count(conn, product_id):
    with _cur(conn) as cur:
        cur.execute("SELECT COUNT(*) AS n FROM lots WHERE product_id = %s", (product_id,))
        return cur.fetchone()["n"]


@contextmanager
def _savepoint(conn, name="lot047_expect"):
    with conn.cursor() as c:
        c.execute(f"SAVEPOINT {name}")
    try:
        yield
    finally:
        with conn.cursor() as c:
            c.execute(f"ROLLBACK TO SAVEPOINT {name}")
            c.execute(f"RELEASE SAVEPOINT {name}")


# ─────────────────────────────────────────────────────────────────
# 1. Migration application (scratch databases)
# ─────────────────────────────────────────────────────────────────

def _psql():
    for candidate in ("/opt/homebrew/opt/postgresql@17/bin/psql", "psql"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def _stripped_schema(tmp_path):
    """schema.sql with the pending-047 include removed, so scratch builds can
    seed pre-047 data before applying the migration."""
    text = SCHEMA.read_text()
    marker_pos = text.find(PENDING_047_MARKER)
    if marker_pos != -1:
        # Cut from the start of the marker's comment line.
        cut = text.rfind("\n--", 0, marker_pos)
        text = text[:cut] if cut != -1 else text
    out = tmp_path / "schema_no_047.sql"
    out.write_text(text)
    return out


@pytest.fixture
def scratch_db_factory(tmp_path):
    """Yields a builder: fresh scratch DB loaded with the pre-047 schema.
    Drops every scratch DB it created on teardown."""
    test_url = os.environ.get("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL not set — DB-backed tests skipped")
    psql = _psql()
    if psql is None:
        pytest.skip("psql binary not found")

    parsed = urlparse(test_url)
    admin_url = urlunparse(parsed._replace(path="/postgres"))
    schema_file = _stripped_schema(tmp_path)
    created = []

    def _run(url, *args):
        proc = subprocess.run(
            [psql, url, "-q", "-v", "ON_ERROR_STOP=1", *args],
            capture_output=True, text=True,
        )
        return proc

    def build():
        db_name = f"factory_ledger_test_047_{uuid4().hex[:8]}"
        proc = _run(admin_url, "-c", f'CREATE DATABASE "{db_name}"')
        assert proc.returncode == 0, proc.stderr
        created.append(db_name)
        url = urlunparse(parsed._replace(path=f"/{db_name}"))
        proc = _run(url, "-c", "CREATE EXTENSION IF NOT EXISTS pg_trgm")
        assert proc.returncode == 0, proc.stderr
        proc = _run(url, "-f", str(schema_file))
        assert proc.returncode == 0, f"pre-047 schema load failed:\n{proc.stderr}"
        return url

    def apply_047(url):
        return _run(url, "-f", str(MIGRATION_047))

    yield build, apply_047

    for db_name in created:
        _run(admin_url, "-c", f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')


def _assert_047_objects(url):
    with psycopg2.connect(url) as conn, _cur(conn) as cur:
        cur.execute("""
            SELECT count(*) AS n FROM pg_indexes
            WHERE tablename = 'lots' AND indexname = 'lots_product_code_norm_uniq'
        """)
        assert cur.fetchone()["n"] == 1
        cur.execute("""
            SELECT conname, convalidated FROM pg_constraint
            WHERE conrelid = 'public.lots'::regclass
              AND conname IN ('lots_lot_uuid_key', 'lots_code_format_chk')
            ORDER BY conname
        """)
        rows = {r["conname"]: r["convalidated"] for r in cur.fetchall()}
        assert rows == {"lots_code_format_chk": False, "lots_lot_uuid_key": True}


def test_047_applies_cleanly_on_fresh_db_and_reruns_as_noop(scratch_db_factory):
    build, apply_047 = scratch_db_factory
    url = build()

    proc = apply_047(url)
    assert proc.returncode == 0, f"047 failed on fresh DB:\n{proc.stderr}"
    _assert_047_objects(url)

    # Idempotent guards: a re-run must also succeed.
    proc = apply_047(url)
    assert proc.returncode == 0, f"047 re-run failed:\n{proc.stderr}"
    _assert_047_objects(url)


def test_047_applies_cleanly_on_historical_shapes_with_merged_twins(scratch_db_factory):
    build, apply_047 = scratch_db_factory
    url = build()

    # Seed the post-step-0 historical shapes from the worklist BEFORE 047:
    # merged twins retain their lot_code, plus legacy codes the new format
    # CHECK would reject.
    with psycopg2.connect(url) as conn:
        with _cur(conn) as cur:
            cur.execute(
                "INSERT INTO products (name, type, uom, active) "
                "VALUES ('Batch Classic Granola #9 (test)', 'batch', 'lb', true) "
                "RETURNING id")
            p_batch = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO products (name, type, uom, active) "
                "VALUES ('Granola SS Choc Chip Case (test)', 'finished', 'lb', true) "
                "RETURNING id")
            p_fg = cur.fetchone()["id"]

            # Group A: 'JUL 15 2026' merged into surviving twin 'JUL15 2026'.
            cur.execute(
                "INSERT INTO lots (product_id, lot_code, entry_source) "
                "VALUES (%s, 'JUL15 2026', 'production_output') RETURNING id",
                (p_batch,))
            survivor_a = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO lots (product_id, lot_code, entry_source, status, "
                "merged_into_lot_id, merged_at, merge_reason) "
                "VALUES (%s, 'JUL 15 2026', 'production_output', 'merged', %s, "
                "now(), 'step-0 twin merge (test seed)')",
                (p_batch, survivor_a))

            # Group B: 'BB041327 Lot' merged into surviving twin 'BB041327'.
            cur.execute(
                "INSERT INTO lots (product_id, lot_code, entry_source) "
                "VALUES (%s, 'BB041327', 'pack_output') RETURNING id",
                (p_fg,))
            survivor_b = cur.fetchone()["id"]
            cur.execute(
                "INSERT INTO lots (product_id, lot_code, entry_source, status, "
                "merged_into_lot_id, merged_at, merge_reason) "
                "VALUES (%s, 'BB041327 Lot', 'pack_output', 'merged', %s, "
                "now(), 'step-0 twin merge (test seed)')",
                (p_fg, survivor_b))

            # Legacy codes the NOT VALID format CHECK must NOT reject:
            # lowercase entry and a Spanish month token.
            cur.execute(
                "INSERT INTO lots (product_id, lot_code) VALUES (%s, 'feb-03-2026 legacy')",
                (p_batch,))
            cur.execute(
                "INSERT INTO lots (product_id, lot_code) VALUES (%s, 'DIC 15 2025')",
                (p_fg,))
        conn.commit()

    proc = apply_047(url)
    assert proc.returncode == 0, f"047 failed on historical shapes:\n{proc.stderr}"
    _assert_047_objects(url)

    with psycopg2.connect(url) as conn, _cur(conn) as cur:
        # Zero violations of the tier-1 index at build time (dry-run query).
        cur.execute(f"""
            WITH norm AS (
                SELECT product_id, {T1_SQL} AS k
                FROM lots WHERE status IS DISTINCT FROM 'merged')
            SELECT count(*) AS n FROM (
                SELECT product_id, k FROM norm GROUP BY 1, 2 HAVING count(*) > 1
            ) twins
        """)
        assert cur.fetchone()["n"] == 0

        # Merged twins coexist with their survivors (predicate excludes them).
        cur.execute("SELECT count(*) AS n FROM lots")
        assert cur.fetchone()["n"] == 6

        # Every historical row got its own lot_uuid.
        cur.execute("""
            SELECT count(*) AS total,
                   count(lot_uuid) AS filled,
                   count(DISTINCT lot_uuid) AS distinct_uuids
            FROM lots
        """)
        row = cur.fetchone()
        assert row["total"] == row["filled"] == row["distinct_uuids"] == 6

        # The index is live: a NEW tier-1 twin of the surviving lot is blocked
        # ('JUL15  2026' — double space — normalizes onto 'JUL15 2026').
        cur.execute("SAVEPOINT twin")
        with pytest.raises(pg_errors.UniqueViolation) as exc:
            cur.execute(
                "INSERT INTO lots (product_id, lot_code) "
                "SELECT id, 'JUL15  2026' FROM products WHERE type = 'batch'")
        assert exc.value.diag.constraint_name == "lots_product_code_norm_uniq"
        cur.execute("ROLLBACK TO SAVEPOINT twin")

        # The format CHECK disciplines new rows only.
        with pytest.raises(pg_errors.CheckViolation):
            cur.execute(
                "INSERT INTO lots (product_id, lot_code) "
                "SELECT id, 'new lowercase code' FROM products WHERE type = 'batch'")


# ─────────────────────────────────────────────────────────────────
# 2. Twin minting per worklist shape
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_exact_twin_row_blocked_by_existing_constraint(db_cursor, _db_connection):
    """Same code, same product: a second lot ROW is impossible — the
    pre-existing lots_product_id_lot_code_key raises. (At the API, the same
    exact code is find-or-create by design: it adds to the existing lot and
    never creates a twin — see test below.)"""
    pid = _seed_product(_db_connection)
    _seed_lot(_db_connection, pid, "ZZ047-EXACT-TWIN")
    with _savepoint(_db_connection):
        with pytest.raises(pg_errors.UniqueViolation) as exc:
            _seed_lot(_db_connection, pid, "ZZ047-EXACT-TWIN")
        assert exc.value.diag.constraint_name == "lots_product_id_lot_code_key"


def test_exact_code_api_mint_reuses_existing_lot(client, _db_connection):
    pid = _seed_product(_db_connection)
    first = _found(client, pid, "ZZ047-EXACT-API")
    assert first.status_code == 200, first.text
    second = _found(client, pid, "ZZ047-EXACT-API")
    assert second.status_code == 200, second.text
    assert second.json()["lot_id"] == first.json()["lot_id"]
    assert _lot_count(_db_connection, pid) == 1
    assert _similarity_warnings(second.json()) == []


def test_whitespace_variant_allowed_with_t2_warning(client, _db_connection):
    """'JUL 15 2026' vs 'JUL15 2026' (worklist Group A): tier-1 keys differ,
    so the hard index allows it — but the aggressive tier-2 key matches and
    the soft warning fires. The write is never blocked."""
    pid = _seed_product(_db_connection, product_type="batch")
    first = _found(client, pid, "JUL15 2026")
    assert first.status_code == 200, first.text
    second = _found(client, pid, "JUL 15 2026")
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["lot_id"] != first.json()["lot_id"]
    warnings = _similarity_warnings(body)
    assert len(warnings) == 1
    assert warnings[0]["similar_lot_id"] == first.json()["lot_id"]
    assert warnings[0]["similar_lot_code"] == "JUL15 2026"
    assert _lot_count(_db_connection, pid) == 2  # both rows exist


def test_trailing_lot_variant_hard_409(client, _db_connection):
    """'BB041327 Lot' vs 'BB041327' (worklist Group B): tier-1 keys collide —
    the mint is hard-blocked with a 409 and no second row is created."""
    pid = _seed_product(_db_connection, product_type="finished")
    first = _found(client, pid, "BB041327")
    assert first.status_code == 200, first.text
    second = _found(client, pid, "BB041327 Lot")
    assert second.status_code == 409, second.text
    assert "BB041327" in second.text
    assert _lot_count(_db_connection, pid) == 1

    # The index itself (not just the API pre-check) enforces it.
    with _savepoint(_db_connection):
        with pytest.raises(pg_errors.UniqueViolation) as exc:
            _seed_lot(_db_connection, pid, "BB041327 LOT")
        assert exc.value.diag.constraint_name == "lots_product_code_norm_uniq"


def test_punctuation_variant_allowed_with_t2_warning(client, _db_connection):
    pid = _seed_product(_db_connection)
    first = _found(client, pid, "ABC-123")
    assert first.status_code == 200, first.text
    second = _found(client, pid, "ABC123")
    assert second.status_code == 200, second.text
    warnings = _similarity_warnings(second.json())
    assert len(warnings) == 1
    assert warnings[0]["similar_lot_id"] == first.json()["lot_id"]
    assert warnings[0]["similar_lot_code"] == "ABC-123"
    assert _lot_count(_db_connection, pid) == 2


def test_same_normalized_code_different_product_allowed_no_warning(client, _db_connection):
    pid_1 = _seed_product(_db_connection)
    pid_2 = _seed_product(_db_connection)
    first = _found(client, pid_1, "ABC-123")
    assert first.status_code == 200, first.text
    second = _found(client, pid_2, "ABC123")
    assert second.status_code == 200, second.text
    assert _similarity_warnings(second.json()) == []
    assert _lot_count(_db_connection, pid_1) == 1
    assert _lot_count(_db_connection, pid_2) == 1


def test_merged_lot_code_reusable_by_new_lot(client, _db_connection):
    """The index predicate (status IS DISTINCT FROM 'merged') lets a new lot
    take a tier-1 variant of a merged lot's retained code; the tier-2 warning
    also ignores merged lots. (The EXACT retained code is still taken by
    lots_product_id_lot_code_key — the worklist §1 rename caveat.)"""
    pid = _seed_product(_db_connection, product_type="batch")
    merged = _seed_lot(_db_connection, pid, "ZZ047 REUSE LOT", status="merged")
    resp = _found(client, pid, "ZZ047 REUSE")
    assert resp.status_code == 200, resp.text
    assert resp.json()["lot_id"] != merged["id"]
    assert _similarity_warnings(resp.json()) == []
    assert _lot_count(_db_connection, pid) == 2

    with _savepoint(_db_connection):
        with pytest.raises(pg_errors.UniqueViolation) as exc:
            _seed_lot(_db_connection, pid, "ZZ047 REUSE LOT")
        assert exc.value.diag.constraint_name == "lots_product_id_lot_code_key"


# ─────────────────────────────────────────────────────────────────
# 3. lot_uuid + commit-response / warning wiring per endpoint
# ─────────────────────────────────────────────────────────────────

def _db_lot_uuid(conn, lot_id):
    with _cur(conn) as cur:
        cur.execute("SELECT lot_uuid FROM lots WHERE id = %s", (lot_id,))
        return str(cur.fetchone()["lot_uuid"])


@pytest.mark.db
def test_lot_uuid_minted_on_insert_and_unique(db_cursor, _db_connection):
    pid = _seed_product(_db_connection)
    lot_a = _seed_lot(_db_connection, pid, "ZZ047-UUID-A")
    lot_b = _seed_lot(_db_connection, pid, "ZZ047-UUID-B")
    assert lot_a["lot_uuid"] and lot_b["lot_uuid"]
    assert lot_a["lot_uuid"] != lot_b["lot_uuid"]
    uuidlib.UUID(str(lot_a["lot_uuid"]))  # parses as a UUID

    with _savepoint(_db_connection):
        with pytest.raises(pg_errors.UniqueViolation) as exc:
            with _cur(_db_connection) as cur:
                cur.execute(
                    "INSERT INTO lots (product_id, lot_code, lot_uuid) "
                    "VALUES (%s, 'ZZ047-UUID-C', %s)",
                    (pid, str(lot_a["lot_uuid"])),
                )
        assert exc.value.diag.constraint_name == "lots_lot_uuid_key"


def test_found_response_has_lot_uuid(client, _db_connection):
    pid = _seed_product(_db_connection)
    resp = _found(client, pid, "ZZ047-FOUND-UUID")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    uuidlib.UUID(body["lot_uuid"])
    assert body["lot_uuid"] == _db_lot_uuid(_db_connection, body["lot_id"])


def test_receive_commit_lot_uuid_and_t2_warning(client, _db_connection):
    pid = _seed_product(_db_connection)
    with _cur(_db_connection) as cur:
        cur.execute("SELECT name FROM products WHERE id = %s", (pid,))
        name = cur.fetchone()["name"]

    base = {"mode": "commit", "product_name": name, "cases": 2,
            "case_size_lb": 50, "shipper_name": "ZZ047 Foods",
            "bol_reference": "BOL-047"}
    first = client.post("/receive", json={**base, "lot_code": "ZZ047-RCV-1"})
    assert first.status_code == 200, first.text
    body = first.json()
    uuidlib.UUID(body["lot_uuid"])
    assert body["lot_uuid"] == _db_lot_uuid(_db_connection, body["lot_id"])
    assert _similarity_warnings(body) == []

    second = client.post("/receive", json={**base, "lot_code": "ZZ047 RCV 1"})
    assert second.status_code == 200, second.text
    warnings = _similarity_warnings(second.json())
    assert len(warnings) == 1
    assert warnings[0]["similar_lot_code"] == "ZZ047-RCV-1"


def _seed_makeable_product(conn, name):
    with _cur(conn) as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, default_batch_lb, active) "
            "VALUES (%s, 'batch', 'lb', 50, true) RETURNING id",
            (name,),
        )
        batch_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, 'ingredient', 'lb', true) RETURNING id",
            (f"{name} ING",),
        )
        ing_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) "
            "VALUES (%s, %s, 50)",
            (batch_id, ing_id),
        )
        lot = _seed_lot(conn, ing_id, f"ZZ047-MAKE-ING-{uuid4().hex[:8].upper()}")
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status) "
            "VALUES ('receive', NOW(), 'posted') RETURNING id",
        )
        txn_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, 200)",
            (txn_id, ing_id, lot["id"]),
        )
    return batch_id


def test_make_commit_lot_uuid_and_t2_warning(client, _db_connection):
    name = f"ZZ047 Make Batch {uuid4().hex[:8].upper()}"
    _seed_makeable_product(_db_connection, name)

    first = client.post("/make", json={
        "mode": "commit", "product_name": name, "batches": 1,
        "lot_code": "ZZ047-MK-1",
    })
    assert first.status_code == 200, first.text
    body = first.json()
    uuidlib.UUID(body["lot_uuid"])
    assert body["lot_uuid"] == _db_lot_uuid(_db_connection, body["lot_id"])
    assert _similarity_warnings(body) == []

    second = client.post("/make", json={
        "mode": "commit", "product_name": name, "batches": 1,
        "lot_code": "zz047mk1",  # lowercase input is normalized before mint
    })
    assert second.status_code == 200, second.text
    assert second.json()["lot_code"] == "ZZ047MK1"
    warnings = _similarity_warnings(second.json())
    assert len(warnings) == 1
    assert warnings[0]["similar_lot_code"] == "ZZ047-MK-1"


def test_pack_commit_lot_uuid_and_t2_warning(client, _db_connection):
    source_name = f"ZZ047 Pack Source {uuid4().hex[:8].upper()}"
    target_name = f"ZZ047 Pack Target {uuid4().hex[:8].upper()}"
    source_id = _seed_product(_db_connection, name=source_name, product_type="batch")
    target_id = _seed_product(_db_connection, name=target_name, product_type="finished")
    with _cur(_db_connection) as cur:
        cur.execute("UPDATE products SET case_size_lb = 10 WHERE id = %s", (target_id,))
        source_lot = _seed_lot(_db_connection, source_id, "ZZ047-PACK-SRC")
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status) "
            "VALUES ('receive', NOW(), 'posted') RETURNING id")
        txn_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, 200)",
            (txn_id, source_id, source_lot["id"]),
        )

    base = {"mode": "commit", "source_product": source_name,
            "target_product": target_name, "cases": 2, "case_weight_lb": 10}
    first = client.post("/pack", json={**base, "target_lot_code": "ZZ047-PK-1"})
    assert first.status_code == 200, first.text
    body = first.json()
    uuidlib.UUID(body["output_lot_uuid"])
    assert body["output_lot_uuid"] == _db_lot_uuid(_db_connection, body["output_lot_id"])
    assert _similarity_warnings(body) == []

    second = client.post("/pack", json={**base, "target_lot_code": "zz047pk1"})
    assert second.status_code == 200, second.text
    assert second.json()["output_lot_code"] == "ZZ047PK1"  # input normalized
    warnings = _similarity_warnings(second.json())
    assert len(warnings) == 1
    assert warnings[0]["similar_lot_code"] == "ZZ047-PK-1"


def test_lot_uuid_survives_rename_and_appears_in_lookups(client, _db_connection):
    pid = _seed_product(_db_connection)
    minted = _found(client, pid, "ZZ047-RENAME-A")
    assert minted.status_code == 200, minted.text
    lot_id = minted.json()["lot_id"]
    lot_uuid = minted.json()["lot_uuid"]

    renamed = client.patch(f"/lots/{lot_id}/rename",
                           json={"new_lot_code": "ZZ047-RENAME-B"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["renamed"] is True
    assert renamed.json()["lot_uuid"] == lot_uuid

    got = client.get(f"/lots/{lot_id}")
    assert got.status_code == 200, got.text
    assert got.json()["lot_code"] == "ZZ047-RENAME-B"
    assert got.json()["lot_uuid"] == lot_uuid

    by_code = client.get("/lots/by-code/ZZ047-RENAME-B",
                         params={"product_id": pid})
    assert by_code.status_code == 200, by_code.text
    assert by_code.json()["lot_uuid"] == lot_uuid


# ─────────────────────────────────────────────────────────────────
# 4. Caller-supplied lot-code input normalization (upper + trim +
#    collapse whitespace; punctuation/'LOT' tokens untouched)
# ─────────────────────────────────────────────────────────────────

def test_lowercase_receive_code_lands_upcased_with_uuid(client, _db_connection):
    pid = _seed_product(_db_connection)
    with _cur(_db_connection) as cur:
        cur.execute("SELECT name FROM products WHERE id = %s", (pid,))
        name = cur.fetchone()["name"]
    resp = client.post("/receive", json={
        "mode": "commit", "product_name": name, "cases": 1,
        "case_size_lb": 50, "shipper_name": "ZZ047 Foods",
        "bol_reference": "BOL-047-LC", "lot_code": "zz047-lc-rcv-1",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["lot_code"] == "ZZ047-LC-RCV-1"
    uuidlib.UUID(body["lot_uuid"])
    with _cur(_db_connection) as cur:
        cur.execute("SELECT lot_code, lot_uuid FROM lots WHERE id = %s", (body["lot_id"],))
        row = cur.fetchone()
    assert row["lot_code"] == "ZZ047-LC-RCV-1"
    assert str(row["lot_uuid"]) == body["lot_uuid"]


def test_casing_variants_resolve_to_same_lot(client, _db_connection):
    pid = _seed_product(_db_connection)
    first = _found(client, pid, "BB041327")
    assert first.status_code == 200, first.text
    second = _found(client, pid, "bb041327")
    assert second.status_code == 200, second.text
    assert second.json()["lot_id"] == first.json()["lot_id"]
    assert second.json()["lot_code"] == "BB041327"
    assert _similarity_warnings(second.json()) == []
    assert _lot_count(_db_connection, pid) == 1


def test_whitespace_variants_resolve_to_same_lot(client, _db_connection):
    pid = _seed_product(_db_connection)
    first = _found(client, pid, "ZZ047 WS X")
    assert first.status_code == 200, first.text
    for variant in ("  ZZ047 WS X  ", "ZZ047  WS   X", "zz047 ws x"):
        resp = _found(client, pid, variant)
        assert resp.status_code == 200, resp.text
        assert resp.json()["lot_id"] == first.json()["lot_id"], variant
        assert resp.json()["lot_code"] == "ZZ047 WS X", variant
        assert _similarity_warnings(resp.json()) == [], variant
    assert _lot_count(_db_connection, pid) == 1


def test_rename_input_is_normalized(client, _db_connection):
    pid = _seed_product(_db_connection)
    minted = _found(client, pid, "ZZ047-RN-LC-A")
    lot_id = minted.json()["lot_id"]
    resp = client.patch(f"/lots/{lot_id}/rename",
                        json={"new_lot_code": "  zz047-rn-lc  b "})
    assert resp.status_code == 200, resp.text
    assert resp.json()["lot_code"] == "ZZ047-RN-LC B"
    with _cur(_db_connection) as cur:
        cur.execute("SELECT lot_code FROM lots WHERE id = %s", (lot_id,))
        assert cur.fetchone()["lot_code"] == "ZZ047-RN-LC B"


def test_rename_onto_tier1_variant_is_409(client, _db_connection):
    pid = _seed_product(_db_connection)
    _found(client, pid, "ZZ047-RN-ONE")
    minted = _found(client, pid, "ZZ047-RN-TWO")
    lot_id = minted.json()["lot_id"]
    resp = client.patch(f"/lots/{lot_id}/rename",
                        json={"new_lot_code": "ZZ047-RN-ONE Lot"})
    assert resp.status_code == 409, resp.text
    # Unchanged after the refusal.
    with _cur(_db_connection) as cur:
        cur.execute("SELECT lot_code FROM lots WHERE id = %s", (lot_id,))
        assert cur.fetchone()["lot_code"] == "ZZ047-RN-TWO"

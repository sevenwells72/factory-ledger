"""AI-assisted sales-order intake — Phase 1 coverage
(docs/designs/sales-order-intake.md).

1. Migration 050 schema behavior, driven with SQL through the rolled-back
   db_cursor fixture: purchase_documents.document_kind default + CHECK;
   customer_product_aliases FKs, case_size_lb CHECK, the code-or-description
   CHECK, alias_key generation/normalization, the (customer_id, alias_key)
   unique index, and the exact latest-wins ON CONFLICT upsert the approve
   path will use; sales_orders.customer_po + source_document_id and the
   normalized dedupe lookup.
2. Migration re-apply is a no-op (idempotence via psql, exit 0).
3. extraction.py kind='sales' unit tests — no DB, no network (client faked):
   sales tool/prompt selection, strict-schema validation incl. unit_price
   rules, nullable pass-through, unknown kind, and the guarantee that the
   default kind still runs the byte-identical purchase path.

Phase 2 extends this module with endpoint coverage: shared upload
(document_kind='sales', per-kind sha256 dedupe), kind-aware extraction retry,
/sales/orders/match (alias → exact → restricted fuzzy chain, private-label
leak guard, prior-sales pool per owner ruling 4), /sales/orders/extract/approve
(atomicity, dedupe warn+force, alias learning, private-label first-sale
warnings, quantity_lb authority), the (customer, normalized PO) advisory lock
with a two-real-connection race, allowlist membership (and POST /sales/orders
staying OFF it), and the readonly tripwire.
"""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import anthropic
from psycopg2 import errors as pg_errors

import extraction
from extraction import ExtractionError, extract_purchase_document

ROOT = Path(__file__).resolve().parent.parent
MIGRATION_050 = ROOT / "migrations" / "050_sales_doc_intake.sql"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _seed_product(cur, name, label_type=None, case_size_lb=None, is_service=False):
    cur.execute(
        """INSERT INTO products (name, type, uom, active, label_type, case_size_lb, is_service)
           VALUES (%s, 'finished', 'lb', true, COALESCE(%s, 'house'), %s, %s) RETURNING id""",
        (name, label_type, case_size_lb, is_service),
    )
    return cur.fetchone()["id"]


def _seed_customer(cur, name):
    cur.execute("INSERT INTO customers (name) VALUES (%s) RETURNING id", (name,))
    return cur.fetchone()["id"]


def _insert_document(cur, path="2026/09/test-050.png", kind=None, status=None, sha="def456"):
    cols, vals, params = ["storage_path", "mime_type", "file_sha256", "byte_size"], \
        ["%s", "'image/png'", "%s", "10"], [path, sha]
    if kind is not None:
        cols.append("document_kind")
        vals.append("%s")
        params.append(kind)
    if status is not None:
        cols.append("status")
        vals.append("%s")
        params.append(status)
    cur.execute(
        f"INSERT INTO purchase_documents ({', '.join(cols)}) VALUES ({', '.join(vals)}) "
        "RETURNING id, document_kind, status",
        params,
    )
    return cur.fetchone()


def _insert_sales_order(cur, customer_id, customer_po=None, source_document_id=None):
    cur.execute(
        """INSERT INTO sales_orders (customer_id, order_number, status, customer_po, source_document_id)
           VALUES (%s, '', 'confirmed', %s, %s)
           RETURNING id, order_number, customer_po, source_document_id""",
        (customer_id, customer_po, source_document_id),
    )
    return cur.fetchone()


def _expect_error(cur, error_cls, sql, params=None):
    cur.execute("SAVEPOINT expect_error_050")
    with pytest.raises(error_cls):
        cur.execute(sql, params)
    cur.execute("ROLLBACK TO SAVEPOINT expect_error_050")


# The exact upsert the Phase-2 approve path will run (latest wins).
ALIAS_UPSERT = """
    INSERT INTO customer_product_aliases
        (customer_id, customer_item_code, customer_description, product_id, case_size_lb, created_by)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (customer_id, alias_key)
    DO UPDATE SET customer_item_code = EXCLUDED.customer_item_code,
                  customer_description = EXCLUDED.customer_description,
                  product_id = EXCLUDED.product_id,
                  case_size_lb = EXCLUDED.case_size_lb,
                  created_by = EXCLUDED.created_by,
                  updated_at = clock_timestamp()
"""


# ─────────────────────────────────────────────────────────────────
# 1. Migration 050 schema behavior
# ─────────────────────────────────────────────────────────────────

class TestDocumentKind:
    def test_default_is_purchase(self, db_cursor):
        row = _insert_document(db_cursor)
        assert row["document_kind"] == "purchase"

    def test_sales_kind_accepted(self, db_cursor):
        row = _insert_document(db_cursor, kind="sales")
        assert row["document_kind"] == "sales"

    def test_kind_check(self, db_cursor):
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size, document_kind)
               VALUES ('2026/09/badkind.png', 'image/png', 'x', 10, 'invoice')""",
        )


class TestCustomerProductAliases:
    def test_alias_key_prefers_item_code(self, db_cursor):
        cust = _seed_customer(db_cursor, "Alias Key Cust")
        prod = _seed_product(db_cursor, "Alias Key Prod")
        db_cursor.execute(
            """INSERT INTO customer_product_aliases
                   (customer_id, customer_item_code, customer_description, product_id, case_size_lb)
               VALUES (%s, '  ITEM   123 ', 'Some   Granola', %s, 10)
               RETURNING alias_key""",
            (cust, prod),
        )
        assert db_cursor.fetchone()["alias_key"] == "item 123"

    def test_alias_key_falls_back_to_description(self, db_cursor):
        cust = _seed_customer(db_cursor, "Alias Fallback Cust")
        prod = _seed_product(db_cursor, "Alias Fallback Prod")
        db_cursor.execute(
            """INSERT INTO customer_product_aliases
                   (customer_id, customer_description, product_id)
               VALUES (%s, '  GRANOLA  ORIGINAL   10LB ', %s)
               RETURNING alias_key, case_size_lb""",
            (cust, prod),
        )
        row = db_cursor.fetchone()
        assert row["alias_key"] == "granola original 10lb"
        assert row["case_size_lb"] is None  # nullable: teaches product mapping only

    def test_code_or_description_required(self, db_cursor):
        cust = _seed_customer(db_cursor, "No Key Cust")
        prod = _seed_product(db_cursor, "No Key Prod")
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            "INSERT INTO customer_product_aliases (customer_id, product_id) VALUES (%s, %s)",
            (cust, prod),
        )

    def test_case_size_lb_check(self, db_cursor):
        cust = _seed_customer(db_cursor, "Bad Case Cust")
        prod = _seed_product(db_cursor, "Bad Case Prod")
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            """INSERT INTO customer_product_aliases
                   (customer_id, customer_item_code, product_id, case_size_lb)
               VALUES (%s, 'X1', %s, 0)""",
            (cust, prod),
        )

    def test_fk_integrity(self, db_cursor):
        prod = _seed_product(db_cursor, "FK Prod 050")
        _expect_error(
            db_cursor, pg_errors.ForeignKeyViolation,
            """INSERT INTO customer_product_aliases (customer_id, customer_item_code, product_id)
               VALUES (999999999, 'X1', %s)""",
            (prod,),
        )
        cust = _seed_customer(db_cursor, "FK Cust 050")
        _expect_error(
            db_cursor, pg_errors.ForeignKeyViolation,
            """INSERT INTO customer_product_aliases (customer_id, customer_item_code, product_id)
               VALUES (%s, 'X1', 999999999)""",
            (cust,),
        )

    def test_same_key_different_customer_ok(self, db_cursor):
        c1 = _seed_customer(db_cursor, "Cust A 050")
        c2 = _seed_customer(db_cursor, "Cust B 050")
        prod = _seed_product(db_cursor, "Shared Key Prod")
        for cust in (c1, c2):
            db_cursor.execute(
                """INSERT INTO customer_product_aliases (customer_id, customer_item_code, product_id)
                   VALUES (%s, 'GR-10', %s)""",
                (cust, prod),
            )
        db_cursor.execute(
            "SELECT count(*) AS n FROM customer_product_aliases WHERE customer_item_code = 'GR-10'"
        )
        assert db_cursor.fetchone()["n"] == 2

    def test_latest_wins_upsert(self, db_cursor):
        cust = _seed_customer(db_cursor, "Upsert Cust 050")
        p1 = _seed_product(db_cursor, "Upsert Prod 1")
        p2 = _seed_product(db_cursor, "Upsert Prod 2")
        db_cursor.execute(ALIAS_UPSERT, (cust, "gr-10", "Granola 10lb", p1, 10, "dashboard"))
        # Same normalized item code (case/whitespace differ) → update, not a row.
        db_cursor.execute(ALIAS_UPSERT, (cust, "  GR-10 ", "Granola 10 lb case", p2, 12.5, "dashboard"))
        db_cursor.execute(
            """SELECT product_id, case_size_lb, customer_description, updated_at > created_at AS bumped,
                      count(*) OVER () AS n
               FROM customer_product_aliases WHERE customer_id = %s""",
            (cust,),
        )
        row = db_cursor.fetchone()
        assert row["n"] == 1
        assert row["product_id"] == p2
        assert float(row["case_size_lb"]) == 12.5
        assert row["customer_description"] == "Granola 10 lb case"
        assert row["bumped"] is True

    def test_description_lookup_finds_code_keyed_rows(self, db_cursor):
        # A row keyed by item code still carries its description; the
        # description lookup (alias tier 2) must find it.
        cust = _seed_customer(db_cursor, "Desc Lookup Cust")
        prod = _seed_product(db_cursor, "Desc Lookup Prod")
        db_cursor.execute(
            """INSERT INTO customer_product_aliases
                   (customer_id, customer_item_code, customer_description, product_id)
               VALUES (%s, 'ZZ-1', 'Chocolate  Granola   Case', %s)""",
            (cust, prod),
        )
        db_cursor.execute(
            """SELECT product_id FROM customer_product_aliases
               WHERE customer_id = %s
                 AND lower(regexp_replace(btrim(customer_description), '\\s+', ' ', 'g')) =
                     lower(regexp_replace(btrim(%s), '\\s+', ' ', 'g'))
               ORDER BY updated_at DESC LIMIT 1""",
            (cust, " chocolate granola  case "),
        )
        assert db_cursor.fetchone()["product_id"] == prod


class TestSalesOrderColumns:
    def test_customer_po_and_source_document(self, db_cursor):
        cust = _seed_customer(db_cursor, "PO Cust 050")
        doc = _insert_document(db_cursor, path="2026/09/so-doc.png", kind="sales")
        row = _insert_sales_order(db_cursor, cust, customer_po="PO-1001", source_document_id=doc["id"])
        assert row["customer_po"] == "PO-1001"
        assert row["source_document_id"] == doc["id"]
        assert row["order_number"].startswith("SO-")  # trigger still fires

    def test_columns_nullable_for_legacy_rows(self, db_cursor):
        cust = _seed_customer(db_cursor, "Legacy Cust 050")
        row = _insert_sales_order(db_cursor, cust)
        assert row["customer_po"] is None
        assert row["source_document_id"] is None

    def test_source_document_fk(self, db_cursor):
        cust = _seed_customer(db_cursor, "Bad Doc Cust 050")
        _expect_error(
            db_cursor, pg_errors.ForeignKeyViolation,
            """INSERT INTO sales_orders (customer_id, order_number, status, source_document_id)
               VALUES (%s, '', 'confirmed', 999999999)""",
            (cust,),
        )

    def test_normalized_po_dedupe_lookup(self, db_cursor):
        # The exact warn-only dedupe query shape the approve path will use.
        cust = _seed_customer(db_cursor, "Dedupe Cust 050")
        other = _seed_customer(db_cursor, "Other Cust 050")
        _insert_sales_order(db_cursor, cust, customer_po="  PO  2002 ")
        _insert_sales_order(db_cursor, other, customer_po="PO 2002")  # other customer: no hit
        db_cursor.execute(
            """SELECT id FROM sales_orders
               WHERE customer_id = %s
                 AND lower(regexp_replace(btrim(customer_po), '\\s+', ' ', 'g')) = %s""",
            (cust, "po 2002"),
        )
        assert len(db_cursor.fetchall()) == 1


class TestMigrationIdempotence:
    def test_reapply_is_noop(self):
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL not set")
        psql = "/opt/homebrew/opt/postgresql@17/bin/psql"
        if not Path(psql).exists():
            psql = "psql"
        result = subprocess.run(
            [psql, "-X", "-v", "ON_ERROR_STOP=1", url, "-f", str(MIGRATION_050)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "ERROR" not in result.stderr


# ─────────────────────────────────────────────────────────────────
# 3. extraction.py kind='sales' unit tests (no DB, no network)
# ─────────────────────────────────────────────────────────────────

GOOD_SALES_EXTRACTION = {
    "customer_name": "Chef Quality Foods",
    "po_number": "CQ-88412",
    "document_date": "2026-09-05",
    "requested_ship_date": "2026-09-15",
    "lines": [
        {"customer_item_code": "1614", "description": "CQ GRANOLA 10 LB",
         "quantity": 20, "unit": "CASE", "unit_price": 32.5},
        {"customer_item_code": None, "description": "Coconut Sweetened Flake",
         "quantity": 500, "unit": "LB", "unit_price": None},
    ],
}


class FakeMessages:
    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return self.response


def _fake_client(monkeypatch, tool_input=GOOD_SALES_EXTRACTION, blocks=None, raises=None):
    if blocks is None:
        blocks = [SimpleNamespace(type="tool_use", input=tool_input)]
    messages = FakeMessages(response=SimpleNamespace(content=blocks), raises=raises)
    monkeypatch.setattr(extraction, "_client", lambda: SimpleNamespace(messages=messages))
    return messages


class TestExtractSalesDocument:
    def test_happy_path(self, monkeypatch):
        monkeypatch.delenv("EXTRACTION_MODEL", raising=False)
        messages = _fake_client(monkeypatch)
        out = extract_purchase_document(b"png-bytes", "image/png", kind="sales")
        assert out["extraction"] == GOOD_SALES_EXTRACTION
        assert out["extraction_model"] == extraction.DEFAULT_EXTRACTION_MODEL
        call = messages.calls[0]
        assert call["tool_choice"] == {"type": "tool", "name": "record_customer_purchase_order"}
        assert call["tools"] == [extraction.SALES_EXTRACTION_TOOL]
        prompt = call["messages"][0]["content"][1]["text"]
        assert prompt == extraction.SALES_EXTRACTION_PROMPT
        assert "BUYER" in prompt  # the we-are-the-vendor inversion is stated

    def test_default_kind_is_purchase_path(self, monkeypatch):
        # Guard for owner ruling 9 without touching the ER test module: the
        # default kind still sends the exact purchase tool/prompt.
        messages = _fake_client(monkeypatch, tool_input={
            "supplier_name": "S", "reference_number": None, "document_date": None,
            "expected_delivery_date": None,
            "lines": [{"vendor_description": "X", "quantity": 1, "unit": None}],
        })
        extract_purchase_document(b"png-bytes", "image/png")
        call = messages.calls[0]
        assert call["tool_choice"] == {"type": "tool", "name": "record_purchase_document"}
        assert call["tools"] == [extraction.EXTRACTION_TOOL]
        assert call["messages"][0]["content"][1]["text"] == extraction.EXTRACTION_PROMPT

    def test_unknown_kind_rejected(self, monkeypatch):
        _fake_client(monkeypatch)
        with pytest.raises(ExtractionError, match="Unknown document kind"):
            extract_purchase_document(b"png-bytes", "image/png", kind="invoice")

    def test_nullable_fields_pass_through(self, monkeypatch):
        payload = {
            "customer_name": "Setton Farms",
            "po_number": None,
            "document_date": None,
            "requested_ship_date": None,
            "lines": [{"customer_item_code": None, "description": "Granola 25 LB",
                       "quantity": 3, "unit": None, "unit_price": None}],
        }
        _fake_client(monkeypatch, tool_input=payload)
        out = extract_purchase_document(b"png-bytes", "image/png", kind="sales")
        assert out["extraction"] == payload

    def test_zero_unit_price_allowed(self, monkeypatch):
        payload = dict(GOOD_SALES_EXTRACTION, lines=[
            {"customer_item_code": None, "description": "Sample - no charge",
             "quantity": 1, "unit": "CASE", "unit_price": 0.0},
        ])
        _fake_client(monkeypatch, tool_input=payload)
        out = extract_purchase_document(b"png-bytes", "image/png", kind="sales")
        assert out["extraction"]["lines"][0]["unit_price"] == 0.0

    @pytest.mark.parametrize("price", [float("nan"), float("inf"), -1.0])
    def test_bad_unit_price_rejected(self, monkeypatch, price):
        payload = dict(GOOD_SALES_EXTRACTION, lines=[
            {"customer_item_code": None, "description": "X",
             "quantity": 1, "unit": "CASE", "unit_price": price},
        ])
        _fake_client(monkeypatch, tool_input=payload)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"png-bytes", "image/png", kind="sales")

    @pytest.mark.parametrize("qty", [0, -2, float("nan"), float("inf")])
    def test_non_finite_or_non_positive_quantity_rejected(self, monkeypatch, qty):
        payload = dict(GOOD_SALES_EXTRACTION, lines=[
            {"customer_item_code": None, "description": "X",
             "quantity": qty, "unit": "CASE", "unit_price": None},
        ])
        _fake_client(monkeypatch, tool_input=payload)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"png-bytes", "image/png", kind="sales")

    @pytest.mark.parametrize("field", ["document_date", "requested_ship_date"])
    @pytest.mark.parametrize("value", ["09/05/2026", "2026-9-5", "next week", "2026-13-40"])
    def test_invalid_date_string_rejected(self, monkeypatch, field, value):
        payload = dict(GOOD_SALES_EXTRACTION)
        payload[field] = value
        _fake_client(monkeypatch, tool_input=payload)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"png-bytes", "image/png", kind="sales")

    def test_missing_customer_name_rejected(self, monkeypatch):
        payload = {k: v for k, v in GOOD_SALES_EXTRACTION.items() if k != "customer_name"}
        _fake_client(monkeypatch, tool_input=payload)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"png-bytes", "image/png", kind="sales")

    def test_empty_lines(self, monkeypatch):
        _fake_client(monkeypatch, tool_input=dict(GOOD_SALES_EXTRACTION, lines=[]))
        with pytest.raises(ExtractionError, match="No product lines"):
            extract_purchase_document(b"png-bytes", "image/png", kind="sales")


# ─────────────────────────────────────────────────────────────────
# Phase 2: endpoint coverage (upload / retry / match / approve)
# ─────────────────────────────────────────────────────────────────

from contextlib import contextmanager

from fastapi.testclient import TestClient

import main


class _ConnProxy:
    """Savepoint proxy so endpoint 'commits' stay inside the rolled-back outer
    transaction (same pattern as tests/test_expected_receipt_extract.py)."""
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
        proxy = _ConnProxy(_db_connection, "so_intake_inner")
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


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """The intake rate limiter is module-global state — isolate every test."""
    main._rate_buckets.clear()
    yield
    main._rate_buckets.clear()


@pytest.fixture
def cur(_db_connection):
    from psycopg2.extras import RealDictCursor
    c = _db_connection.cursor(cursor_factory=RealDictCursor)
    yield c
    c.close()


@pytest.fixture
def mock_storage(monkeypatch):
    """No real Supabase calls (same shape as the ER module's fixture)."""
    calls = {"uploads": [], "downloads": [], "signed": [], "upload_error": None, "exists": set()}

    def _upload(path, content, mime, upsert=False):
        if calls["upload_error"] is not None:
            raise calls["upload_error"]
        calls["exists"].add(path)
        calls["uploads"].append({"path": path, "bytes": len(content), "mime": mime, "upsert": upsert})

    def _download(path):
        calls["downloads"].append(path)
        return b"stored-bytes"

    def _sign(path, expires_in=600):
        calls["signed"].append(path)
        return f"https://storage.test/signed/{path}"

    monkeypatch.setattr(main, "storage_upload_purchase_document", _upload)
    monkeypatch.setattr(main, "storage_download_purchase_document", _download)
    monkeypatch.setattr(main, "storage_signed_purchase_document_url", _sign)
    return calls


@pytest.fixture
def mock_extractor(monkeypatch):
    """Deterministic stand-in for the vision model. Records the kind each call
    arrived with — 'purchase-2arg' proves the ER path still uses the exact
    pre-refactor two-argument call (owner ruling 9)."""
    state = {"result": {"extraction": dict(GOOD_SALES_EXTRACTION), "extraction_model": "claude-sonnet-5"},
             "error": None, "calls": 0, "kinds": []}

    def _extract(content, mime, *args):
        state["calls"] += 1
        state["kinds"].append(args[0] if args else "purchase-2arg")
        if state["error"]:
            raise state["error"]
        return state["result"]

    monkeypatch.setattr(extraction, "extract_purchase_document", _extract)
    return state


def _seed_prior_sale(cur, customer_id, product_id, qty=100,
                     so_status="shipped", line_status="fulfilled"):
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status) VALUES (%s, '', %s) RETURNING id",
        (customer_id, so_status),
    )
    so_id = cur.fetchone()["id"]
    cur.execute(
        """INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb, line_status)
           VALUES (%s, %s, %s, %s)""",
        (so_id, product_id, qty, line_status),
    )
    return so_id


def _doc_row(cur, document_id):
    cur.execute("SELECT * FROM purchase_documents WHERE id = %s", (document_id,))
    return cur.fetchone()


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"so-png-body"


def _post_so_file(client, name="cust-po.png", content=PNG_BYTES):
    return client.post("/sales/orders/extract", files={"file": (name, content, "image/png")})


def _match(client, customer_name, lines, po_number=None):
    return client.post("/sales/orders/match", json={"extraction": {
        "customer_name": customer_name, "po_number": po_number,
        "document_date": None, "requested_ship_date": None, "lines": lines}})


def _line(description, quantity, unit="CASE", code=None, price=None):
    return {"customer_item_code": code, "description": description,
            "quantity": quantity, "unit": unit, "unit_price": price}


class TestSalesUploadEndpoint:
    def test_upload_creates_sales_kind_row(self, client, cur, mock_storage, mock_extractor):
        r = _post_so_file(client)
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["status"] == "uploaded"
        assert mock_extractor["calls"] == 0, "upload must not invoke the model"
        row = _doc_row(cur, data["document_id"])
        assert row["document_kind"] == "sales"
        assert row["status"] == "uploaded"

    def test_sha_dedupe_is_scoped_per_kind(self, client, cur, mock_storage, mock_extractor):
        """The same bytes may exist once as a vendor PO and once as a customer
        PO; within a kind, identical bytes resume the in-flight row."""
        r_purchase = client.post("/expected-receipts/extract",
                                 files={"file": ("po.png", PNG_BYTES, "image/png")})
        assert r_purchase.status_code == 201, r_purchase.text
        r_sales = _post_so_file(client)
        assert r_sales.status_code == 201, r_sales.text  # NOT a resume of the purchase row
        assert r_sales.json()["document_id"] != r_purchase.json()["document_id"]
        r_sales2 = _post_so_file(client)
        assert r_sales2.status_code == 200, r_sales2.text  # same-kind resume
        assert r_sales2.json()["document_id"] == r_sales.json()["document_id"]
        assert r_sales2.json()["already_seen"] is True

    def test_upload_then_extract_runs_sales_kind(self, client, cur, mock_storage, mock_extractor):
        doc_id = _post_so_file(client).json()["document_id"]
        r = client.post(f"/purchase-documents/{doc_id}/extract")
        assert r.status_code == 200, r.text
        assert r.json()["extraction"] == GOOD_SALES_EXTRACTION
        assert mock_extractor["kinds"] == ["sales"]
        assert _doc_row(cur, doc_id)["status"] == "extracted"

    def test_purchase_retry_still_uses_two_arg_call(self, client, cur, mock_storage, mock_extractor):
        """Owner ruling 9: the purchase path calls the extractor with the exact
        pre-refactor 2-arg signature (the ER module's 2-arg fakes keep working)."""
        doc_id = client.post("/expected-receipts/extract",
                             files={"file": ("po.png", PNG_BYTES, "image/png")}).json()["document_id"]
        r = client.post(f"/purchase-documents/{doc_id}/extract")
        assert r.status_code == 200, r.text
        assert mock_extractor["kinds"] == ["purchase-2arg"]

    def test_handlers_run_in_threadpool(self):
        import inspect
        assert not inspect.iscoroutinefunction(main.extract_sales_order_document)
        assert not inspect.iscoroutinefunction(main.extract_expected_receipt_document)


class TestSalesMatchEndpoint:
    def test_alias_by_item_code_wins_and_converts(self, client, cur):
        cust = _seed_customer(cur, "Match Cust A")
        decoy = _seed_product(cur, "Decoy Granola 10 LB")
        target = _seed_product(cur, "Real Granola Product", case_size_lb=25)
        cur.execute(
            """INSERT INTO customer_product_aliases
                   (customer_id, customer_item_code, customer_description, product_id, case_size_lb)
               VALUES (%s, 'CQ-77', 'whatever they call it', %s, 10)""",
            (cust, target),
        )
        r = _match(client, "Match Cust A", [_line("Decoy Granola 10 LB", 3, code="cq-77")])
        assert r.status_code == 200, r.text
        line = r.json()["lines"][0]
        assert line["match_source"] == "alias"
        assert line["product"]["product_id"] == target
        assert line["case_size_lb"] == 10.0          # alias case size beats product master
        assert line["case_size_source"] == "alias"
        assert line["quantity_lb"] == 30.0
        assert decoy != target

    def test_alias_by_description(self, client, cur):
        cust = _seed_customer(cur, "Match Cust B")
        target = _seed_product(cur, "Desc Alias Target")
        cur.execute(
            """INSERT INTO customer_product_aliases (customer_id, customer_description, product_id, case_size_lb)
               VALUES (%s, 'Original  Crunch Case', %s, 12)""",
            (cust, target),
        )
        r = _match(client, "Match Cust B", [_line("  original crunch   case ", 2)])
        line = r.json()["lines"][0]
        assert line["match_source"] == "alias"
        assert line["quantity_lb"] == 24.0

    def test_exact_name_match_uses_product_case_size(self, client, cur):
        cust = _seed_customer(cur, "Match Cust C")
        prod = _seed_product(cur, "Exact House Granola", case_size_lb=20)
        r = _match(client, "Match Cust C", [_line("Exact House Granola", 4)])
        line = r.json()["lines"][0]
        assert line["match_source"] == "exact"
        assert line["product"]["product_id"] == prod
        assert line["product"]["prior_sales"] is False
        assert line["case_size_source"] == "product"
        assert line["quantity_lb"] == 80.0

    def test_exact_odoo_code_match_via_item_code(self, client, cur):
        cust = _seed_customer(cur, "Match Cust C2")
        cur.execute(
            "INSERT INTO products (name, type, uom, active, odoo_code) VALUES ('Odoo Coded Prod', 'finished', 'lb', true, '424242') RETURNING id")
        prod = cur.fetchone()["id"]
        r = _match(client, "Match Cust C2", [_line("their own wording", 1, code="424242")])
        line = r.json()["lines"][0]
        assert line["match_source"] == "exact"
        assert line["product"]["product_id"] == prod

    def test_private_label_exact_leak_guard(self, client, cur):
        """An exact text collision with another customer's private-label SKU
        must NOT match or appear anywhere in the response."""
        cust = _seed_customer(cur, "Match Cust D")
        pl = _seed_product(cur, "Setton Secret Granola 25 LB", label_type="private_label", case_size_lb=25)
        r = _match(client, "Match Cust D", [_line("Setton Secret Granola 25 LB", 2)])
        line = r.json()["lines"][0]
        assert line["match_source"] == "none"
        assert line["product"] is None
        assert line["quantity_lb"] is None
        assert all(c["product_id"] != pl for c in line["candidates"])

    def test_private_label_exact_allowed_with_prior_sales(self, client, cur):
        cust = _seed_customer(cur, "Match Cust E")
        pl = _seed_product(cur, "Blue Secret Granola 25 LB", label_type="private_label", case_size_lb=25)
        _seed_prior_sale(cur, cust, pl)
        r = _match(client, "Match Cust E", [_line("Blue Secret Granola 25 LB", 2)])
        line = r.json()["lines"][0]
        assert line["match_source"] == "exact"
        assert line["product"]["product_id"] == pl
        assert line["product"]["prior_sales"] is True
        assert line["quantity_lb"] == 50.0

    def test_fuzzy_restricted_to_prior_sales_pool(self, client, cur):
        cust = _seed_customer(cur, "Match Cust F")
        in_pool = _seed_product(cur, "Bagged Original Granola", case_size_lb=15)
        out_pool = _seed_product(cur, "Bagged Original Granola Deluxe", case_size_lb=15)
        _seed_prior_sale(cur, cust, in_pool)
        r = _match(client, "Match Cust F", [_line("original granola bagged", 2)])
        line = r.json()["lines"][0]
        assert line["match_source"] == "fuzzy"
        assert line["product"]["product_id"] == in_pool
        assert line["quantity_lb"] is None, "fuzzy never computes lb"
        assert line["case_size_source"] == "none"
        assert all(c["product_id"] != out_pool for c in line["candidates"])

    def test_no_fuzzy_with_empty_pool(self, client, cur):
        cust = _seed_customer(cur, "Match Cust G")
        _seed_product(cur, "Some Fuzzy Target Granola")
        r = _match(client, "Match Cust G", [_line("fuzzy target granolla", 1)])
        line = r.json()["lines"][0]
        assert line["match_source"] == "none"
        assert line["candidates"] == []

    def test_cancelled_history_excluded_from_pool(self, client, cur):
        """Owner ruling 4: cancelled orders AND cancelled lines don't count."""
        cust = _seed_customer(cur, "Match Cust H")
        prod = _seed_product(cur, "Cancelled History Granola")
        _seed_prior_sale(cur, cust, prod, so_status="cancelled")
        _seed_prior_sale(cur, cust, prod, line_status="cancelled")
        r = _match(client, "Match Cust H", [_line("cancelled history granola x", 1)])
        assert r.json()["lines"][0]["match_source"] == "none"

    def test_lb_unit_is_pounds(self, client, cur):
        cust = _seed_customer(cur, "Match Cust I")
        r = _match(client, "Match Cust I", [_line("Anything At All", 500, unit="LB")])
        line = r.json()["lines"][0]
        assert line["case_size_source"] == "unit_is_lb"
        assert line["quantity_lb"] == 500.0

    def test_unknown_customer_gives_candidates(self, client, cur):
        _seed_customer(cur, "Chef Quality Foods LLC")
        r = _match(client, "Chef Quality", [_line("x", 1)])
        data = r.json()
        assert data["customer"]["match"] is None
        assert any(c["name"] == "Chef Quality Foods LLC" for c in data["customer"]["candidates"])
        assert data["lines"][0]["match_source"] == "none"

    def test_customer_alias_resolves(self, client, cur):
        cust = _seed_customer(cur, "Canonical Foods Inc")
        cur.execute("INSERT INTO customer_aliases (customer_id, alias) VALUES (%s, 'CanFoods 050')", (cust,))
        r = _match(client, "CanFoods 050", [_line("x", 1)])
        assert r.json()["customer"]["match"]["customer_id"] == cust

    def test_duplicate_po_warning(self, client, cur):
        cust = _seed_customer(cur, "Match Cust J")
        _insert_sales_order(cur, cust, customer_po="  PO 9001 ")
        r = _match(client, "Match Cust J", [_line("x", 1)], po_number="po  9001")
        dup = r.json()["duplicate_warning"]
        assert dup and len(dup["existing"]) == 1
        assert dup["existing"][0]["order_number"].startswith("SO-")

    def test_no_writes(self, client, cur):
        cust = _seed_customer(cur, "Match Cust K")
        cur.execute("SELECT count(*) AS n FROM sales_orders")
        before = cur.fetchone()["n"]
        _match(client, "Match Cust K", [_line("x", 1)])
        cur.execute("SELECT count(*) AS n FROM sales_orders")
        assert cur.fetchone()["n"] == before


def _approve_payload(doc_id, cust_id, lines, po="PO-5000", **kw):
    return dict({"document_id": doc_id, "customer_id": cust_id,
                 "customer_po": po, "lines": lines}, **kw)


def _approve_line(product_id, quantity_lb, quantity=None, unit=None, case_size_lb=None,
                  price=None, code=None, description="CUSTOMER WORDING", save_alias=False):
    return {"product_id": product_id, "quantity_lb": quantity_lb, "quantity": quantity,
            "unit": unit, "case_size_lb": case_size_lb, "unit_price": price,
            "customer_item_code": code, "customer_description": description,
            "save_alias": save_alias}


class TestSalesApproveEndpoint:
    def _setup(self, cur, name="Approve Cust", case_size=10):
        cust = _seed_customer(cur, name)
        prod = _seed_product(cur, f"{name} Product", case_size_lb=case_size)
        doc = _insert_document(cur, path=f"x/{name.replace(' ', '-')}.png",
                               kind="sales", status="extracted", sha=f"sha-{name}")
        return cust, prod, doc["id"]

    def test_happy_path_two_lines(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust A")
        prod2 = _seed_product(cur, "Approve Cust A Bulk")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust,
            [_approve_line(prod, 30, quantity=3, unit="cases", case_size_lb=10,
                           price=32.5, code="CQ-1", description="THEIR GRANOLA", save_alias=True),
             _approve_line(prod2, 500, quantity=500, unit="lb", description="BULK FLAKE")],
            po="PO-A-1", requested_ship_date="2026-09-15", order_date="2026-09-08"))
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["order_number"].startswith("SO-")
        assert data["total_lb"] == 530.0
        assert data["aliases_saved"] == 1
        assert data["customer_po"] == "PO-A-1"
        cur.execute("SELECT * FROM sales_orders WHERE id = %s", (data["order_id"],))
        so = cur.fetchone()
        assert so["customer_po"] == "PO-A-1"
        assert so["source_document_id"] == doc_id
        assert so["status"] == "confirmed"
        assert str(so["order_date"]) == "2026-09-08"
        cur.execute("SELECT product_id, quantity_lb, unit_price, notes FROM sales_order_lines WHERE sales_order_id = %s ORDER BY id", (data["order_id"],))
        rows = cur.fetchall()
        assert [float(x["quantity_lb"]) for x in rows] == [30.0, 500.0]
        assert float(rows[0]["unit_price"]) == 32.5
        assert rows[0]["notes"] == "PO PO-A-1: THEIR GRANOLA — 3 cases"
        cur.execute("SELECT product_id, case_size_lb FROM customer_product_aliases WHERE customer_id = %s", (cust,))
        alias = cur.fetchone()
        assert alias["product_id"] == prod and float(alias["case_size_lb"]) == 10.0
        assert _doc_row(cur, doc_id)["status"] == "approved"

    def test_po_required(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust B")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10)], po="   "))
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "PO_REQUIRED"

    def test_duplicate_blocks_without_force(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust C")
        _insert_sales_order(cur, cust, customer_po="po c-9")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10)], po="  PO  C-9 "))
        assert r.status_code == 409, r.text
        detail = r.json()["detail"]
        assert detail["error_code"] == "DUPLICATE_PO"
        assert len(detail["existing"]) == 1
        r2 = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10)], po="  PO  C-9 ", force=True))
        assert r2.status_code == 201, r2.text
        assert r2.json()["duplicate_overridden"] is True

    def test_atomic_rollback_on_bad_line(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust D")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10), _approve_line(999999999, 5)]))
        assert r.status_code == 404
        cur.execute("SELECT count(*) AS n FROM sales_orders WHERE customer_id = %s", (cust,))
        assert cur.fetchone()["n"] == 0, "the good line must roll back too"
        assert _doc_row(cur, doc_id)["status"] == "extracted"

    @pytest.mark.parametrize("status", ["uploaded", "extraction_failed", "upload_failed"])
    def test_approve_requires_extracted_status(self, client, cur, status):
        cust = _seed_customer(cur, f"Approve Cust E {status}")
        prod = _seed_product(cur, f"Approve Prod E {status}")
        doc = _insert_document(cur, path=f"x/e-{status}.png", kind="sales", status=status, sha=f"sha-e-{status}")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc["id"], cust, [_approve_line(prod, 10)]))
        assert r.status_code == 409
        assert r.json()["detail"]["error_code"] == "DOCUMENT_NOT_EXTRACTED"

    def test_double_approve_rejected(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust F")
        r1 = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10)]))
        assert r1.status_code == 201, r1.text
        r2 = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10)], force=True))
        assert r2.status_code == 409
        assert r2.json()["detail"]["error_code"] == "DOCUMENT_ALREADY_APPROVED"

    def test_purchase_document_rejected(self, client, cur):
        """A vendor-PO document can't be approved into a sales order."""
        cust = _seed_customer(cur, "Approve Cust G")
        prod = _seed_product(cur, "Approve Prod G")
        doc = _insert_document(cur, path="x/g.png", kind="purchase", status="extracted", sha="sha-g")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc["id"], cust, [_approve_line(prod, 10)]))
        assert r.status_code == 409
        assert r.json()["detail"]["error_code"] == "DOCUMENT_KIND_MISMATCH"

    def test_unknown_and_inactive_customer(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust H")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, 999999999, [_approve_line(prod, 10)]))
        assert r.status_code == 404
        assert r.json()["detail"]["error_code"] == "CUSTOMER_NOT_FOUND"
        cur.execute("UPDATE customers SET active = false WHERE id = %s", (cust,))
        r2 = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 10)]))
        assert r2.status_code == 422
        assert r2.json()["detail"]["error_code"] == "CUSTOMER_INACTIVE"

    def test_alias_conversion_mismatch(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust I")
        bad = _approve_line(prod, 35, quantity=3, unit="cases", case_size_lb=10, save_alias=True)
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(doc_id, cust, [bad]))
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "ALIAS_CONVERSION_MISMATCH"
        # Same numbers without save_alias: allowed (nothing is learned).
        ok = dict(bad, save_alias=False)
        r2 = client.post("/sales/orders/extract/approve", json=_approve_payload(doc_id, cust, [ok]))
        assert r2.status_code == 201, r2.text

    def test_save_alias_without_conversion_teaches_product_only(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust J")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust,
            [_approve_line(prod, 42, unit="lb", code="J-1", save_alias=True)]))
        assert r.status_code == 201, r.text
        cur.execute("SELECT case_size_lb FROM customer_product_aliases WHERE customer_id = %s", (cust,))
        assert cur.fetchone()["case_size_lb"] is None

    def test_alias_latest_wins_via_endpoint(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust K")
        prod2 = _seed_product(cur, "Approve Cust K Better Product")
        cur.execute(
            """INSERT INTO customer_product_aliases (customer_id, customer_item_code, product_id, case_size_lb)
               VALUES (%s, 'K-9', %s, 5)""", (cust, prod))
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust,
            [_approve_line(prod2, 24, quantity=2, unit="cases", case_size_lb=12,
                           code="  k-9 ", save_alias=True)]))
        assert r.status_code == 201, r.text
        cur.execute("SELECT product_id, case_size_lb, count(*) OVER () AS n FROM customer_product_aliases WHERE customer_id = %s", (cust,))
        row = cur.fetchone()
        assert row["n"] == 1
        assert row["product_id"] == prod2
        assert float(row["case_size_lb"]) == 12.0

    @pytest.mark.parametrize("qty", [0, -5, float("nan"), float("inf")])
    def test_non_finite_quantity_lb_rejected(self, client, cur, qty):
        cust, prod, doc_id = self._setup(cur, f"Approve Cust L {qty}")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, qty)]))
        assert r.status_code == 422, r.text
        cur.execute("SELECT count(*) AS n FROM sales_orders WHERE source_document_id = %s", (doc_id,))
        assert cur.fetchone()["n"] == 0

    def test_private_label_first_sale_warns_never_blocks(self, client, cur):
        cust, _, doc_id = self._setup(cur, "Approve Cust M")
        pl = _seed_product(cur, "Approve PL Granola M", label_type="private_label", case_size_lb=25)
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(pl, 25, quantity=1, unit="cases", case_size_lb=25)]))
        assert r.status_code == 201, r.text
        warnings = r.json()["warnings"] or []
        assert any("private-label" in w for w in warnings)

    def test_private_label_with_history_no_warning(self, client, cur):
        cust, _, doc_id = self._setup(cur, "Approve Cust N")
        pl = _seed_product(cur, "Approve PL Granola N", label_type="private_label", case_size_lb=25)
        _seed_prior_sale(cur, cust, pl)
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(pl, 25, quantity=1, unit="cases", case_size_lb=25)]))
        assert r.status_code == 201, r.text
        assert not any("private-label" in w for w in (r.json()["warnings"] or []))

    def test_manual_lb_override_never_recomputed(self, client, cur):
        """A reviewed lb value with no case size must be stored verbatim —
        the core's case-weight auto-lookup must not overwrite it even though
        the product master has a (different) case size."""
        cust, prod, doc_id = self._setup(cur, "Approve Cust O", case_size=50)
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc_id, cust, [_approve_line(prod, 123.4, quantity=3, unit="cases")]))
        assert r.status_code == 201, r.text
        cur.execute("SELECT quantity_lb FROM sales_order_lines WHERE sales_order_id = %s", (r.json()["order_id"],))
        assert float(cur.fetchone()["quantity_lb"]) == 123.4

    def test_no_lines_rejected(self, client, cur):
        cust, prod, doc_id = self._setup(cur, "Approve Cust P")
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(doc_id, cust, []))
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "NO_LINES"

    def test_manual_endpoint_still_works_via_core(self, client, cur):
        """POST /sales/orders (GPT path) now runs through _create_sales_order_core
        — same response shape as before the refactor."""
        _seed_customer(cur, "Manual Core Cust")
        _seed_product(cur, "Manual Core Prod", case_size_lb=10)
        r = client.post("/sales/orders", json={
            "customer_name": "Manual Core Cust",
            "lines": [{"product_name": "Manual Core Prod", "quantity": 2, "unit": "cases",
                       "case_weight_lb": 10}]})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["order_number"].startswith("SO-")
        assert data["status"] == "confirmed"
        assert data["total_lb"] == 20.0
        assert data["lines"][0]["case_weight_lb"] == 10.0


class TestSalesApprovalAdvisoryLock:
    """Same guarantees as the ER approve lock (audit fix 10), on the
    (customer, normalized PO) pair."""

    KEY = "so-intake-po:42:po 777"

    def _try_lock(self, cur, key):
        cur.execute("SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS ok", (key,))
        return cur.fetchone()["ok"]

    def test_lock_serializes_the_normalized_pair(self):
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL not set")
        import psycopg2 as pg
        from psycopg2.extras import RealDictCursor
        conn_a, conn_b = pg.connect(url), pg.connect(url)
        try:
            with conn_a.cursor(cursor_factory=RealDictCursor) as ca, \
                 conn_b.cursor(cursor_factory=RealDictCursor) as cb:
                main._lock_customer_po(ca, 42, "  PO   777 ")
                assert self._try_lock(cb, self.KEY) is False, "same pair must block"
                assert self._try_lock(cb, "so-intake-po:42:po 778") is True
                assert self._try_lock(cb, "so-intake-po:43:po 777") is True
            conn_b.rollback()
            conn_a.rollback()
            with conn_b.cursor(cursor_factory=RealDictCursor) as cb:
                assert self._try_lock(cb, self.KEY) is True
            conn_b.rollback()
        finally:
            conn_a.close()
            conn_b.close()

    def test_approve_takes_the_pair_lock(self, client, cur, monkeypatch):
        cust, prod, doc_id = (None, None, None)
        cust = _seed_customer(cur, "Lock Cust 050")
        prod = _seed_product(cur, "Lock Prod 050")
        doc = _insert_document(cur, path="x/lock-050.png", kind="sales", status="extracted", sha="sha-lock")
        calls = []
        orig = main._lock_customer_po

        def _spy(c, customer_id, po):
            calls.append((customer_id, po))
            return orig(c, customer_id, po)

        monkeypatch.setattr(main, "_lock_customer_po", _spy)
        r = client.post("/sales/orders/extract/approve", json=_approve_payload(
            doc["id"], cust, [_approve_line(prod, 10)], po="PO-LOCK-050"))
        assert r.status_code == 201, r.text
        assert calls == [(cust, "PO-LOCK-050")]

    def test_two_full_approvals_same_pair_loser_gets_409(self, monkeypatch):
        """Two COMPLETE approvals of two different sales documents with the
        same (customer, PO), each on its own real DB connection, racing
        through the real endpoint — exactly one 201; the loser re-checks
        under the pair lock and gets the 409 (ER audit-2 finding 10 pattern)."""
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL not set")
        import threading
        import time
        from contextlib import contextmanager as _ctx
        import psycopg2 as pg
        from psycopg2.extras import RealDictCursor
        from fastapi.testclient import TestClient as TC

        PO = "PO-RACE-050"
        LOCK_KEY = "so-intake-po:%d:po-race-050"

        seed = pg.connect(url)
        seed.autocommit = True
        cust_id = pid = None
        doc_ids = []
        so_ids = []
        try:
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("INSERT INTO customers (name) VALUES ('Race Cust 050') RETURNING id")
                cust_id = sc.fetchone()["id"]
                sc.execute("INSERT INTO products (name, type, uom, active) VALUES ('Race Prod 050', 'finished', 'lb', true) RETURNING id")
                pid = sc.fetchone()["id"]
                for n in (1, 2):
                    sc.execute(
                        """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size, status, document_kind)
                           VALUES (%s, 'image/png', %s, 10, 'extracted', 'sales') RETURNING id""",
                        (f"x/race-050-{n}.png", f"race-050-sha-{n}"))
                    doc_ids.append(sc.fetchone()["id"])

            @_ctx
            def _real_conn():
                conn = pg.connect(url)
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            monkeypatch.setattr(main, "get_db_connection", _real_conn)

            holder = pg.connect(url)
            results = []
            try:
                with holder.cursor() as hc:
                    hc.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (LOCK_KEY % cust_id,))

                with TC(main.app) as tc:
                    tc.headers["X-API-Key"] = main.API_KEY

                    def _approve(doc_id):
                        r = tc.post("/sales/orders/extract/approve", json=_approve_payload(
                            doc_id, cust_id,
                            [_approve_line(pid, 100, description=f"RACE LINE {doc_id}")],
                            po=PO))
                        results.append(r)

                    threads = [threading.Thread(target=_approve, args=(d,)) for d in doc_ids]
                    for t in threads:
                        t.start()
                    deadline = time.time() + 15
                    with seed.cursor(cursor_factory=RealDictCursor) as sc:
                        while time.time() < deadline:
                            sc.execute("SELECT count(*) AS n FROM pg_locks WHERE locktype = 'advisory' AND NOT granted")
                            if sc.fetchone()["n"] >= 2:
                                break
                            time.sleep(0.05)
                        else:
                            pytest.fail("both approvals never queued on the advisory lock")
                    holder.rollback()
                    for t in threads:
                        t.join(timeout=30)
                    assert not any(t.is_alive() for t in threads), "an approval hung"
            finally:
                holder.close()

            assert sorted(r.status_code for r in results) == [201, 409], \
                [(r.status_code, r.text[:200]) for r in results]
            loser = next(r for r in results if r.status_code == 409)
            assert loser.json()["detail"]["error_code"] == "DUPLICATE_PO"
            with seed.cursor(cursor_factory=RealDictCursor) as sc:
                sc.execute("SELECT id FROM sales_orders WHERE customer_id = %s", (cust_id,))
                so_ids = [r["id"] for r in sc.fetchall()]
                assert len(so_ids) == 1, "exactly the winner's order exists"
                sc.execute("SELECT count(*) AS n FROM purchase_documents WHERE id = ANY(%s) AND status = 'approved'", (doc_ids,))
                assert sc.fetchone()["n"] == 1, "exactly one document approved"
        finally:
            with seed.cursor() as sc:
                if so_ids or cust_id:
                    sc.execute("DELETE FROM sales_order_lines WHERE sales_order_id IN (SELECT id FROM sales_orders WHERE customer_id = %s)", (cust_id,))
                    sc.execute("DELETE FROM sales_orders WHERE customer_id = %s", (cust_id,))
                if doc_ids:
                    sc.execute("DELETE FROM purchase_documents WHERE id = ANY(%s)", (doc_ids,))
                if cust_id:
                    sc.execute("DELETE FROM customer_product_aliases WHERE customer_id = %s", (cust_id,))
                if pid:
                    sc.execute("DELETE FROM products WHERE id = %s", (pid,))
                if cust_id:
                    sc.execute("DELETE FROM customers WHERE id = %s", (cust_id,))
            seed.close()


class TestSalesAllowlistAndTripwire:
    SO_INTAKE_ROUTES = [
        ("POST", "/sales/orders/extract"),
        ("POST", "/sales/orders/match"),
        ("POST", "/sales/orders/extract/approve"),
    ]

    def test_routes_on_dashboard_allowlist(self):
        for pair in self.SO_INTAKE_ROUTES:
            assert pair in main.DASHBOARD_KEY_ALLOWLIST, pair

    def test_manual_create_stays_off_the_allowlist(self):
        """Owner ruling 1: the dashboard key never reaches POST /sales/orders
        (it auto-creates customers)."""
        assert ("POST", "/sales/orders") not in main.DASHBOARD_KEY_ALLOWLIST

    def test_dashboard_key_accepted_on_match(self, client, cur):
        _seed_customer(cur, "Dash Key Cust 050")
        r = client.post("/sales/orders/match",
                        headers={"X-API-Key": main.DASHBOARD_API_KEY},
                        json={"extraction": {"customer_name": "Dash Key Cust 050",
                                             "po_number": None, "document_date": None,
                                             "requested_ship_date": None,
                                             "lines": [{"customer_item_code": None, "description": "x",
                                                        "quantity": 1, "unit": None, "unit_price": None}]}})
        assert r.status_code == 200, r.text

    def test_dashboard_key_rejected_on_manual_create(self, client):
        r = client.post("/sales/orders",
                        headers={"X-API-Key": main.DASHBOARD_API_KEY},
                        json={"customer_name": "X", "lines": [{"product_name": "Y", "quantity_lb": 1}]})
        assert r.status_code == 403, r.text

    @pytest.fixture
    def readonly_client(self, monkeypatch):
        """Every DB connection raises a real psycopg2 readonly error on first
        execute — same fixture shape as the ER module."""
        import psycopg2.errors

        exc = psycopg2.errors.ReadOnlySqlTransaction(
            "cannot execute INSERT in a read-only transaction")

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **kw):
                raise exc

        class _Conn:
            def cursor(self, *a, **kw):
                return _Cur()

            def commit(self):
                pass

            def rollback(self):
                pass

        @contextmanager
        def _fake_get_conn():
            yield _Conn()

        monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
        monkeypatch.setattr(main, "_capture_readonly_diagnostics", lambda: {"stub": True})
        uploads = []
        monkeypatch.setattr(main, "storage_upload_purchase_document",
                            lambda *a, **kw: uploads.append(a))
        with TestClient(main.app, raise_server_exceptions=False) as c:
            c.headers["X-API-Key"] = main.API_KEY
            c.storage_uploads = uploads
            yield c

    def test_sales_upload_trips_readonly_tripwire(self, readonly_client):
        r = _post_so_file(readonly_client)
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error_code"] == "READONLY_TRANSACTION"
        assert body["success"] is False
        assert "error_detail" in body
        # Row-first ordering: nothing reached Storage.
        assert readonly_client.storage_uploads == []

    def test_sales_approve_trips_readonly_tripwire(self, readonly_client):
        r = readonly_client.post("/sales/orders/extract/approve", json=_approve_payload(
            1, 1, [_approve_line(1, 10)]))
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error_code"] == "READONLY_TRANSACTION"
        assert "error_detail" in body

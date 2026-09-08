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

Endpoint coverage (extract/match/approve routes) is Phase 2 and will extend
this module.
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

def _seed_product(cur, name):
    cur.execute(
        "INSERT INTO products (name, type, uom, active) VALUES (%s, 'finished', 'lb', true) RETURNING id",
        (name,),
    )
    return cur.fetchone()["id"]


def _seed_customer(cur, name):
    cur.execute("INSERT INTO customers (name) VALUES (%s) RETURNING id", (name,))
    return cur.fetchone()["id"]


def _insert_document(cur, path="2026/09/test-050.png", kind=None):
    cols, vals, params = ["storage_path", "mime_type", "file_sha256", "byte_size"], \
        ["%s", "'image/png'", "'def456'", "10"], [path]
    if kind is not None:
        cols.append("document_kind")
        vals.append("%s")
        params.append(kind)
    cur.execute(
        f"INSERT INTO purchase_documents ({', '.join(cols)}) VALUES ({', '.join(vals)}) "
        "RETURNING id, document_kind",
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

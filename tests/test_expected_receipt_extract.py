"""AI-assisted expected-receipt intake — Phase 1 coverage
(docs/designs/expected-receipt-intake.md).

1. Migration 049 schema behavior, driven with SQL through the rolled-back
   db_cursor fixture: purchase_documents CHECKs (mime, status, byte_size) and
   storage_path uniqueness; supplier_product_aliases FKs, lb_per_unit CHECK,
   the normalized (supplier, vendor_description) unique index, and the exact
   latest-wins ON CONFLICT upsert the approve path will use;
   expected_receipts.source_document_id FK.
2. Migration re-apply is a no-op (idempotence against the already-migrated
   test DB via psql, exit 0).
3. extraction.py unit tests — no DB, no network: the Anthropic client is
   faked. Verifies the strict-schema validation, nullable fields, verbatim
   pass-through, the vendor/model override, block types per mime, and that
   every failure mode raises ExtractionError (never a guessed result).

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
MIGRATION_049 = ROOT / "migrations" / "049_purchase_doc_intake.sql"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _seed_product(cur, name):
    cur.execute(
        "INSERT INTO products (name, type, uom, active) VALUES (%s, 'ingredient', 'lb', true) RETURNING id",
        (name,),
    )
    return cur.fetchone()["id"]


def _seed_supplier(cur, name):
    cur.execute("INSERT INTO suppliers (name) VALUES (%s) RETURNING id", (name,))
    return cur.fetchone()["id"]


def _insert_document(cur, path="2026/09/test-049.png"):
    cur.execute(
        """INSERT INTO purchase_documents
               (storage_path, mime_type, file_sha256, byte_size)
           VALUES (%s, 'image/png', 'abc123', 10)
           RETURNING id, status, storage_bucket""",
        (path,),
    )
    return cur.fetchone()


def _expect_error(cur, error_cls, sql, params=None):
    cur.execute("SAVEPOINT expect_error_049")
    with pytest.raises(error_cls):
        cur.execute(sql, params)
    cur.execute("ROLLBACK TO SAVEPOINT expect_error_049")


# ─────────────────────────────────────────────────────────────────
# 1. Migration 049 schema behavior
# ─────────────────────────────────────────────────────────────────

class TestPurchaseDocuments:
    def test_defaults(self, db_cursor):
        row = _insert_document(db_cursor)
        assert row["status"] == "uploaded"
        assert row["storage_bucket"] == "purchase-documents"

    def test_mime_type_check(self, db_cursor):
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size)
               VALUES ('x/1', 'image/gif', 'a', 10)""",
        )

    def test_status_check(self, db_cursor):
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size, status)
               VALUES ('x/2', 'image/png', 'a', 10, 'bogus')""",
        )

    def test_byte_size_check(self, db_cursor):
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size)
               VALUES ('x/3', 'image/png', 'a', 0)""",
        )

    def test_storage_path_unique(self, db_cursor):
        _insert_document(db_cursor, path="x/dup")
        _expect_error(
            db_cursor, pg_errors.UniqueViolation,
            """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size)
               VALUES ('x/dup', 'application/pdf', 'b', 20)""",
        )

    def test_all_allowed_mimes_and_statuses(self, db_cursor):
        for i, mime in enumerate(("image/png", "image/jpeg", "application/pdf")):
            db_cursor.execute(
                """INSERT INTO purchase_documents (storage_path, mime_type, file_sha256, byte_size)
                   VALUES (%s, %s, 'a', 10)""",
                (f"x/mime-{i}", mime),
            )
        doc = _insert_document(db_cursor, path="x/status")
        for status in ("extracted", "extraction_failed", "approved"):
            db_cursor.execute(
                "UPDATE purchase_documents SET status = %s WHERE id = %s",
                (status, doc["id"]),
            )


class TestSupplierProductAliases:
    def test_insert_and_normalized_unique(self, db_cursor):
        supplier_id = _seed_supplier(db_cursor, "Alias Norm Sup 049")
        product_id = _seed_product(db_cursor, "Alias Norm Prod 049")
        db_cursor.execute(
            """INSERT INTO supplier_product_aliases
                   (supplier_id, vendor_description, product_id, lb_per_unit, unit)
               VALUES (%s, 'Coconut Flake  Desiccated 50#', %s, 50, 'BAG')""",
            (supplier_id, product_id),
        )
        # Same wording modulo case/whitespace → the normalized unique index fires.
        _expect_error(
            db_cursor, pg_errors.UniqueViolation,
            """INSERT INTO supplier_product_aliases
                   (supplier_id, vendor_description, product_id)
               VALUES (%s, '  coconut FLAKE desiccated 50#  ', %s)""",
            (supplier_id, product_id),
        )

    def test_same_description_different_supplier_ok(self, db_cursor):
        product_id = _seed_product(db_cursor, "Alias Shared Prod 049")
        sup_a = _seed_supplier(db_cursor, "Alias Sup A 049")
        sup_b = _seed_supplier(db_cursor, "Alias Sup B 049")
        for sup in (sup_a, sup_b):
            db_cursor.execute(
                """INSERT INTO supplier_product_aliases
                       (supplier_id, vendor_description, product_id)
                   VALUES (%s, 'GRAHAM CRUMBS 50 LB', %s)""",
                (sup, product_id),
            )

    def test_lb_per_unit_check(self, db_cursor):
        supplier_id = _seed_supplier(db_cursor, "Alias Norm Sup 049")
        product_id = _seed_product(db_cursor, "Alias Norm Prod 049")
        _expect_error(
            db_cursor, pg_errors.CheckViolation,
            """INSERT INTO supplier_product_aliases
                   (supplier_id, vendor_description, product_id, lb_per_unit)
               VALUES (%s, 'zero lb alias', %s, 0)""",
            (supplier_id, product_id),
        )

    def test_fk_integrity(self, db_cursor):
        product_id = _seed_product(db_cursor, "Alias FK Prod 049")
        supplier_id = _seed_supplier(db_cursor, "Alias FK Sup 049")
        _expect_error(
            db_cursor, pg_errors.ForeignKeyViolation,
            """INSERT INTO supplier_product_aliases (supplier_id, vendor_description, product_id)
               VALUES (999999999, 'fk test', %s)""",
            (product_id,),
        )
        _expect_error(
            db_cursor, pg_errors.ForeignKeyViolation,
            """INSERT INTO supplier_product_aliases (supplier_id, vendor_description, product_id)
               VALUES (%s, 'fk test', 999999999)""",
            (supplier_id,),
        )

    def test_latest_wins_upsert(self, db_cursor):
        """The exact ON CONFLICT statement the Phase 2 approve path will run:
        a re-approval with a different product/conversion overwrites the alias."""
        supplier_id = _seed_supplier(db_cursor, "Alias Upsert Sup 049")
        prod_1 = _seed_product(db_cursor, "Alias Upsert Prod A 049")
        prod_2 = _seed_product(db_cursor, "Alias Upsert Prod B 049")
        upsert = """
            INSERT INTO supplier_product_aliases
                (supplier_id, vendor_description, product_id, lb_per_unit, unit, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (supplier_id,
                         lower(regexp_replace(btrim(vendor_description), '\\s+', ' ', 'g')))
            DO UPDATE SET product_id = EXCLUDED.product_id,
                          lb_per_unit = EXCLUDED.lb_per_unit,
                          unit = EXCLUDED.unit,
                          created_by = EXCLUDED.created_by,
                          updated_at = clock_timestamp()
            RETURNING id, product_id, lb_per_unit
        """
        db_cursor.execute(upsert, (supplier_id, "SS Classic #9 Bulk", prod_1, 25, "CASE", "dashboard"))
        first = db_cursor.fetchone()
        db_cursor.execute(upsert, (supplier_id, "  ss classic #9  BULK ", prod_2, 30, "CASE", "dashboard"))
        second = db_cursor.fetchone()
        assert second["id"] == first["id"], "normalized-equal wording must hit the same row"
        assert second["product_id"] == prod_2
        assert float(second["lb_per_unit"]) == 30
        db_cursor.execute("SELECT count(*) AS n FROM supplier_product_aliases WHERE supplier_id = %s", (supplier_id,))
        # exactly one row for the wording, not two
        db_cursor.execute(
            """SELECT count(*) AS n FROM supplier_product_aliases
               WHERE supplier_id = %s
                 AND lower(regexp_replace(btrim(vendor_description), '\\s+', ' ', 'g'))
                     = 'ss classic #9 bulk'""",
            (supplier_id,),
        )
        assert db_cursor.fetchone()["n"] == 1


class TestSourceDocumentLink:
    def test_source_document_fk(self, db_cursor):
        product_id = _seed_product(db_cursor, "Doc Link Prod 049")
        supplier_id = _seed_supplier(db_cursor, "Doc Link Sup 049")
        _expect_error(
            db_cursor, pg_errors.ForeignKeyViolation,
            """INSERT INTO expected_receipts (product_id, supplier_id, expected_qty, source_document_id)
               VALUES (%s, %s, 100, 999999999)""",
            (product_id, supplier_id),
        )
        doc = _insert_document(db_cursor, path="x/link")
        db_cursor.execute(
            """INSERT INTO expected_receipts (product_id, supplier_id, expected_qty, source_document_id)
               VALUES (%s, %s, 100, %s)
               RETURNING source_document_id""",
            (product_id, supplier_id, doc["id"]),
        )
        assert db_cursor.fetchone()["source_document_id"] == doc["id"]

    def test_source_document_nullable(self, db_cursor):
        """Manual creation keeps working with no document."""
        product_id = _seed_product(db_cursor, "Doc Null Prod 049")
        supplier_id = _seed_supplier(db_cursor, "Doc Null Sup 049")
        db_cursor.execute(
            """INSERT INTO expected_receipts (product_id, supplier_id, expected_qty)
               VALUES (%s, %s, 100) RETURNING source_document_id""",
            (product_id, supplier_id),
        )
        assert db_cursor.fetchone()["source_document_id"] is None


class TestMigrationIdempotence:
    def test_reapply_is_noop(self):
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL not set")
        psql = "/opt/homebrew/opt/postgresql@17/bin/psql"
        if not Path(psql).exists():
            psql = "psql"
        result = subprocess.run(
            [psql, "-X", "-v", "ON_ERROR_STOP=1", url, "-f", str(MIGRATION_049)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "ERROR" not in result.stderr


# ─────────────────────────────────────────────────────────────────
# 3. extraction.py unit tests (no DB, no network)
# ─────────────────────────────────────────────────────────────────

GOOD_EXTRACTION = {
    "supplier_name": "Blue Stripes Cacao",
    "reference_number": "PO-26791",
    "document_date": "2026-09-05",
    "expected_delivery_date": "2026-09-12",
    "lines": [
        {"vendor_description": "CACAO NIBS ORGANIC 25KG", "quantity": 4, "unit": "BAG"},
        {"vendor_description": "Graham Cracker Crumbs - 50 LB", "quantity": 10, "unit": "CASE"},
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


def _fake_client(monkeypatch, tool_input=GOOD_EXTRACTION, blocks=None, raises=None):
    if blocks is None:
        blocks = [SimpleNamespace(type="tool_use", input=tool_input)]
    messages = FakeMessages(response=SimpleNamespace(content=blocks), raises=raises)
    monkeypatch.setattr(extraction, "_client", lambda: SimpleNamespace(messages=messages))
    return messages


class FakeAPIError(anthropic.APIError):
    def __init__(self):  # anthropic.APIError needs request/body; bypass
        Exception.__init__(self, "boom")


class TestExtractPurchaseDocument:
    def test_happy_path(self, monkeypatch):
        monkeypatch.delenv("EXTRACTION_MODEL", raising=False)
        messages = _fake_client(monkeypatch)
        out = extract_purchase_document(b"png-bytes", "image/png")
        assert out["extraction"] == GOOD_EXTRACTION
        assert out["extraction_model"] == extraction.DEFAULT_EXTRACTION_MODEL
        call = messages.calls[0]
        assert call["model"] == extraction.DEFAULT_EXTRACTION_MODEL
        assert call["tool_choice"] == {"type": "tool", "name": "record_purchase_document"}
        # image mime → image block
        assert call["messages"][0]["content"][0]["type"] == "image"
        assert call["messages"][0]["content"][0]["source"]["media_type"] == "image/png"

    def test_pdf_uses_document_block(self, monkeypatch):
        messages = _fake_client(monkeypatch)
        extract_purchase_document(b"%PDF-", "application/pdf")
        block = messages.calls[0]["messages"][0]["content"][0]
        assert block["type"] == "document"
        assert block["source"]["media_type"] == "application/pdf"

    def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_MODEL", "claude-opus-5")
        messages = _fake_client(monkeypatch)
        out = extract_purchase_document(b"x", "image/jpeg")
        assert out["extraction_model"] == "claude-opus-5"
        assert messages.calls[0]["model"] == "claude-opus-5"

    def test_nullable_fields_pass_through(self, monkeypatch):
        payload = {
            "supplier_name": "Somebody",
            "reference_number": None,
            "document_date": None,
            "expected_delivery_date": None,
            "lines": [{"vendor_description": "THING", "quantity": 1, "unit": None}],
        }
        _fake_client(monkeypatch, tool_input=payload)
        out = extract_purchase_document(b"x", "image/png")
        assert out["extraction"]["reference_number"] is None
        assert out["extraction"]["lines"][0]["unit"] is None

    def test_unsupported_mime(self, monkeypatch):
        _fake_client(monkeypatch)
        with pytest.raises(ExtractionError, match="Unsupported mime"):
            extract_purchase_document(b"x", "image/gif")

    def test_oversize_file(self, monkeypatch):
        _fake_client(monkeypatch)
        with pytest.raises(ExtractionError, match="15 MB"):
            extract_purchase_document(b"x" * (extraction.MAX_FILE_BYTES + 1), "image/png")

    def test_no_tool_use_block(self, monkeypatch):
        _fake_client(monkeypatch, blocks=[SimpleNamespace(type="text", text="I can't read this")])
        with pytest.raises(ExtractionError, match="no structured extraction"):
            extract_purchase_document(b"x", "image/png")

    def test_schema_mismatch(self, monkeypatch):
        bad = dict(GOOD_EXTRACTION)
        del bad["supplier_name"]
        _fake_client(monkeypatch, tool_input=bad)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"x", "image/png")

    def test_non_numeric_quantity_rejected(self, monkeypatch):
        bad = {
            "supplier_name": "S",
            "reference_number": None,
            "document_date": None,
            "expected_delivery_date": None,
            "lines": [{"vendor_description": "THING", "quantity": "ten", "unit": None}],
        }
        _fake_client(monkeypatch, tool_input=bad)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"x", "image/png")

    def test_empty_lines(self, monkeypatch):
        empty = dict(GOOD_EXTRACTION, lines=[])
        _fake_client(monkeypatch, tool_input=empty)
        with pytest.raises(ExtractionError, match="No product lines"):
            extract_purchase_document(b"x", "image/png")

    def test_api_error_wrapped(self, monkeypatch):
        _fake_client(monkeypatch, raises=FakeAPIError())
        with pytest.raises(ExtractionError, match="Vision API call failed"):
            extract_purchase_document(b"x", "image/png")

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ExtractionError, match="ANTHROPIC_API_KEY"):
            extraction._client()

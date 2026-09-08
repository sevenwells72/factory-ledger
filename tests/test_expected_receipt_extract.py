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

    # Audit fix 11: strict means strict — non-finite/non-positive quantities
    # and invalid date strings are schema violations.

    @pytest.mark.parametrize("qty", [float("nan"), float("inf"), float("-inf"), 0, -3])
    def test_non_finite_or_non_positive_quantity_rejected(self, monkeypatch, qty):
        bad = dict(GOOD_EXTRACTION,
                   lines=[{"vendor_description": "THING", "quantity": qty, "unit": None}])
        _fake_client(monkeypatch, tool_input=bad)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"x", "image/png")

    @pytest.mark.parametrize("field", ["document_date", "expected_delivery_date"])
    @pytest.mark.parametrize("value", ["Sept 5 2026", "2026-13-45", "2026-02-30", "tomorrow", "26-09-05"])
    def test_invalid_date_string_rejected(self, monkeypatch, field, value):
        bad = dict(GOOD_EXTRACTION)
        bad[field] = value
        _fake_client(monkeypatch, tool_input=bad)
        with pytest.raises(ExtractionError, match="did not match schema"):
            extract_purchase_document(b"x", "image/png")

    def test_valid_dates_still_pass(self, monkeypatch):
        ok = dict(GOOD_EXTRACTION, document_date="2026-09-05", expected_delivery_date=None)
        _fake_client(monkeypatch, tool_input=ok)
        out = extract_purchase_document(b"x", "image/png")
        assert out["extraction"]["document_date"] == "2026-09-05"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ExtractionError, match="ANTHROPIC_API_KEY"):
            extraction._client()


# ═════════════════════════════════════════════════════════════════
# Phase 2 — endpoint coverage (extract / match / approve / retry / url)
# ═════════════════════════════════════════════════════════════════

from contextlib import contextmanager
from datetime import date

from fastapi.testclient import TestClient

import main


class _ConnProxy:
    """Savepoint proxy so endpoint 'commits' stay inside the rolled-back outer
    transaction (same pattern as tests/test_expected_receipts.py)."""
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
        proxy = _ConnProxy(_db_connection, "intake_inner")
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


@pytest.fixture
def cur(_db_connection):
    from psycopg2.extras import RealDictCursor
    c = _db_connection.cursor(cursor_factory=RealDictCursor)
    yield c
    c.close()


@pytest.fixture
def mock_storage(monkeypatch):
    """No real Supabase calls: record uploads, serve downloads, sign URLs."""
    calls = {"uploads": [], "downloads": [], "signed": []}

    def _upload(path, content, mime):
        calls["uploads"].append({"path": path, "bytes": len(content), "mime": mime})

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
    """Deterministic stand-in for the vision model."""
    state = {"result": {"extraction": dict(GOOD_EXTRACTION), "extraction_model": "claude-sonnet-5"},
             "error": None, "calls": 0}

    def _extract(content, mime):
        state["calls"] += 1
        if state["error"]:
            raise state["error"]
        return state["result"]

    monkeypatch.setattr(extraction, "extract_purchase_document", _extract)
    return state


def _doc_row(cur, document_id):
    cur.execute("SELECT * FROM purchase_documents WHERE id = %s", (document_id,))
    return cur.fetchone()


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"png-body"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"jpeg-body"


def _make_pdf(pages: int) -> bytes:
    from reportlab.pdfgen import canvas
    buf = __import__("io").BytesIO()
    c = canvas.Canvas(buf)
    for i in range(pages):
        c.drawString(72, 720, f"page {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


def _post_file(client, name="po.png", mime="image/png", content=PNG_BYTES):
    return client.post("/expected-receipts/extract", files={"file": (name, content, mime)})


class TestExtractEndpoint:
    def test_upload_is_upload_only(self, client, cur, mock_storage, mock_extractor):
        """Audit fix 8: POST /expected-receipts/extract stores the file and
        returns the document_id WITHOUT calling the vision model."""
        r = _post_file(client)
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["success"] is True
        assert data["already_seen"] is False
        assert data["status"] == "uploaded"
        assert "extraction" not in data
        assert mock_extractor["calls"] == 0, "upload must not invoke the model"
        assert len(mock_storage["uploads"]) == 1
        assert mock_storage["uploads"][0]["path"] == data["storage_path"]
        row = _doc_row(cur, data["document_id"])
        assert row["status"] == "uploaded"
        assert row["mime_type"] == "image/png"
        assert row["file_sha256"]

    def test_upload_then_extract_happy_path(self, client, cur, mock_storage, mock_extractor):
        doc_id = _post_file(client).json()["document_id"]
        r = client.post(f"/purchase-documents/{doc_id}/extract")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["extraction"] == GOOD_EXTRACTION
        assert data["extraction_model"] == "claude-sonnet-5"
        assert mock_extractor["calls"] == 1
        assert mock_storage["downloads"], "extraction reads the stored file"
        row = _doc_row(cur, doc_id)
        assert row["status"] == "extracted"
        assert row["extraction"]["supplier_name"] == GOOD_EXTRACTION["supplier_name"]

    def test_handlers_run_in_threadpool(self):
        """Audit fix 8: both handlers are plain def — FastAPI runs them in the
        threadpool, so the sync DB/Storage/model calls can't block the loop."""
        import inspect
        assert not inspect.iscoroutinefunction(main.extract_expected_receipt_document)
        assert not inspect.iscoroutinefunction(main.run_purchase_document_extraction)

    def test_already_seen_on_same_bytes(self, client, mock_storage, mock_extractor):
        assert _post_file(client).json()["already_seen"] is False
        r2 = _post_file(client)
        assert r2.status_code == 201
        assert r2.json()["already_seen"] is True
        # The extract call reports it too (server-side, not trusted from the client).
        r3 = client.post(f"/purchase-documents/{r2.json()['document_id']}/extract")
        assert r3.status_code == 200
        assert r3.json()["already_seen"] is True

    def test_unsupported_type(self, client, mock_storage, mock_extractor):
        r = _post_file(client, name="po.gif", mime="image/gif", content=b"GIF89a-not-supported")
        assert r.status_code == 415
        assert r.json()["detail"]["error_code"] == "UNSUPPORTED_FILE_TYPE"
        assert mock_storage["uploads"] == []

    def test_jpg_alias_mime(self, client, cur, mock_storage, mock_extractor):
        r = _post_file(client, name="po.jpg", mime="image/jpg", content=JPEG_BYTES)
        assert r.status_code == 201
        assert _doc_row(cur, r.json()["document_id"])["mime_type"] == "image/jpeg"

    # Audit fix 9: magic bytes are authoritative; PDFs are page-capped
    # before anything reaches Storage.

    def test_mime_spoof_rejected(self, client, cur, mock_storage, mock_extractor):
        r = _post_file(client, name="po.png", mime="image/png", content=b"<script>not an image</script>")
        assert r.status_code == 415
        assert r.json()["detail"]["error_code"] == "UNSUPPORTED_FILE_TYPE"
        assert mock_storage["uploads"] == []
        cur.execute("SELECT count(*) AS n FROM purchase_documents")
        assert cur.fetchone()["n"] == 0, "no row for rejected content"

    def test_magic_bytes_override_claimed_mime(self, client, cur, mock_storage, mock_extractor):
        r = _post_file(client, name="po.png", mime="image/png", content=_make_pdf(1))
        assert r.status_code == 201, r.text
        assert _doc_row(cur, r.json()["document_id"])["mime_type"] == "application/pdf"

    def test_pdf_within_page_cap_accepted(self, client, mock_storage, mock_extractor):
        r = _post_file(client, name="po.pdf", mime="application/pdf", content=_make_pdf(2))
        assert r.status_code == 201, r.text

    def test_pdf_over_page_cap_rejected_before_storage(self, client, cur, mock_storage, mock_extractor):
        r = _post_file(client, name="po.pdf", mime="application/pdf", content=_make_pdf(21))
        assert r.status_code == 413
        assert r.json()["detail"]["error_code"] == "PDF_TOO_MANY_PAGES"
        assert mock_storage["uploads"] == [], "rejected before storage"
        cur.execute("SELECT count(*) AS n FROM purchase_documents")
        assert cur.fetchone()["n"] == 0

    def test_pdf_unreadable_rejected(self, client, mock_storage, mock_extractor):
        r = _post_file(client, name="po.pdf", mime="application/pdf", content=b"%PDF-1.4 truncated garbage")
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "PDF_UNREADABLE"
        assert mock_storage["uploads"] == []

    def test_empty_file(self, client, mock_storage, mock_extractor):
        r = _post_file(client, content=b"")
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "EMPTY_FILE"

    def test_too_large(self, client, mock_storage, mock_extractor, monkeypatch):
        monkeypatch.setattr(extraction, "MAX_FILE_BYTES", 10)
        r = _post_file(client, content=b"x" * 11)
        assert r.status_code == 413
        assert r.json()["detail"]["error_code"] == "FILE_TOO_LARGE"

    def test_extraction_failure_keeps_file_and_row(self, client, cur, mock_storage, mock_extractor):
        doc_id = _post_file(client).json()["document_id"]
        mock_extractor["error"] = ExtractionError("model refused")
        r = client.post(f"/purchase-documents/{doc_id}/extract")
        assert r.status_code == 502
        detail = r.json()["detail"]
        assert detail["error_code"] == "EXTRACTION_FAILED"
        assert detail["document_id"] == doc_id
        row = _doc_row(cur, doc_id)
        assert row["status"] == "extraction_failed"
        assert len(mock_storage["uploads"]) == 1  # file kept, no re-upload needed

    def test_retry_after_failure(self, client, cur, mock_storage, mock_extractor):
        doc_id = _post_file(client).json()["document_id"]
        mock_extractor["error"] = ExtractionError("model refused")
        assert client.post(f"/purchase-documents/{doc_id}/extract").status_code == 502
        mock_extractor["error"] = None
        r = client.post(f"/purchase-documents/{doc_id}/extract")
        assert r.status_code == 200, r.text
        assert r.json()["extraction"] == GOOD_EXTRACTION
        assert mock_storage["downloads"], "retry must download the stored file"
        assert _doc_row(cur, doc_id)["status"] == "extracted"

    def test_retry_unknown_document(self, client, mock_storage, mock_extractor):
        r = client.post("/purchase-documents/999999999/extract")
        assert r.status_code == 404
        assert r.json()["detail"]["error_code"] == "DOCUMENT_NOT_FOUND"

    # Audit fix 5: an extraction retry must never reopen an approved document.

    def test_retry_on_approved_document_409(self, client, cur, mock_storage, mock_extractor):
        doc = _insert_document(cur, path="x/retry-approved-049.png")
        cur.execute("UPDATE purchase_documents SET status = 'approved', approved_at = clock_timestamp() WHERE id = %s",
                    (doc["id"],))
        r = client.post(f"/purchase-documents/{doc['id']}/extract")
        assert r.status_code == 409
        assert r.json()["detail"]["error_code"] == "DOCUMENT_ALREADY_APPROVED"
        assert mock_extractor["calls"] == 0, "no paid model call for an approved document"

    def test_approval_during_retry_model_call_wins(self, client, cur, mock_storage, monkeypatch):
        """The race the audit reproduced: retry passes its status pre-check,
        approval commits while the model call runs, then the retry's status
        write must be a conditional no-op + 409 — never a reopen."""
        doc = _insert_document(cur, path="x/retry-race-049.png")

        def _extract(content, mime):
            cur.execute(
                "UPDATE purchase_documents SET status = 'approved', approved_at = clock_timestamp() WHERE id = %s",
                (doc["id"],))
            return {"extraction": dict(GOOD_EXTRACTION), "extraction_model": "race-model"}

        monkeypatch.setattr(extraction, "extract_purchase_document", _extract)
        r = client.post(f"/purchase-documents/{doc['id']}/extract")
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["error_code"] == "DOCUMENT_ALREADY_APPROVED"
        assert _doc_row(cur, doc["id"])["status"] == "approved"

    def test_approval_during_failed_retry_model_call_wins(self, client, cur, mock_storage, monkeypatch):
        doc = _insert_document(cur, path="x/retry-race-fail-049.png")

        def _extract(content, mime):
            cur.execute(
                "UPDATE purchase_documents SET status = 'approved', approved_at = clock_timestamp() WHERE id = %s",
                (doc["id"],))
            raise ExtractionError("model refused")

        monkeypatch.setattr(extraction, "extract_purchase_document", _extract)
        r = client.post(f"/purchase-documents/{doc['id']}/extract")
        assert r.status_code == 409, r.text
        assert _doc_row(cur, doc["id"])["status"] == "approved", "failure path must not flip an approved doc"


class TestMatchEndpoint:
    def _seed(self, cur):
        sup = _seed_supplier(cur, "Vendor Match Co 049")
        cur.execute(
            """INSERT INTO products (name, type, uom, active, case_size_lb)
               VALUES ('Match Oats Rolled 049', 'ingredient', 'lb', true, 25) RETURNING id""")
        exact_pid = cur.fetchone()["id"]
        alias_pid = _seed_product(cur, "Match Alias Target 049")
        cur.execute(
            """INSERT INTO supplier_product_aliases
                   (supplier_id, vendor_description, product_id, lb_per_unit, unit)
               VALUES (%s, 'VNDR OATS SPECIAL 22.68KG', %s, 50, 'BAG')""",
            (sup, alias_pid))
        return sup, exact_pid, alias_pid

    def _match(self, client, supplier_name, lines, reference=None):
        return client.post("/expected-receipts/match", json={"extraction": {
            "supplier_name": supplier_name, "reference_number": reference,
            "document_date": None, "expected_delivery_date": None, "lines": lines}})

    def test_alias_beats_fuzzy_and_converts(self, client, cur):
        sup, exact_pid, alias_pid = self._seed(cur)
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "vndr oats  SPECIAL 22.68kg", "quantity": 4, "unit": "BAG"}])
        assert r.status_code == 200, r.text
        line = r.json()["lines"][0]
        assert line["match_source"] == "alias"
        assert line["product"]["product_id"] == alias_pid
        assert line["lb_per_unit"] == 50 and line["lb_source"] == "alias"
        assert line["expected_qty_lb"] == 200

    def test_exact_match_uses_case_size(self, client, cur):
        sup, exact_pid, _ = self._seed(cur)
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "Match Oats Rolled 049", "quantity": 3, "unit": "CASE"}])
        line = r.json()["lines"][0]
        assert line["match_source"] == "exact"
        assert line["product"]["product_id"] == exact_pid
        assert line["lb_source"] == "case_size" and line["lb_per_unit"] == 25
        assert line["expected_qty_lb"] == 75

    def test_parsed_description_weight(self, client, cur):
        sup, exact_pid, _ = self._seed(cur)
        cur.execute("UPDATE products SET name = 'Graham Crumbs Fine 049' WHERE id = %s", (exact_pid,))
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "GRAHAM CRUMBS FINE 049 50 LB", "quantity": 2, "unit": "BAG"}])
        line = r.json()["lines"][0]
        assert line["lb_source"] == "parsed_description" and line["lb_per_unit"] == 50
        # tiered search: keyword-or-weaker on this wording → product-master rule
        # doesn't apply, but parsed weight is text-derived and allowed…
        if line["match_source"] in ("alias", "exact"):
            assert line["expected_qty_lb"] == 100
        else:
            assert line["expected_qty_lb"] is None  # …while qty stays null until confirm

    # Audit fix 2: alias conversions are lb-per-ALIAS-UNIT and apply only when
    # the line's unit matches the unit the alias was saved with.

    def test_alias_unit_mismatch_falls_back_to_line_unit(self, client, cur):
        """A 50-lb/BAG alias on a '100 LB' line must yield 100 lb, not 5,000."""
        sup, exact_pid, alias_pid = self._seed(cur)
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "VNDR OATS SPECIAL 22.68KG", "quantity": 100, "unit": "LB"}])
        line = r.json()["lines"][0]
        assert line["match_source"] == "alias", "product association still reuses"
        assert line["product"]["product_id"] == alias_pid
        assert line["lb_source"] == "unit_is_lb" and line["lb_per_unit"] == 1.0
        assert line["expected_qty_lb"] == 100

    def test_alias_unit_mismatch_container_requires_manual_lb(self, client, cur):
        sup, exact_pid, alias_pid = self._seed(cur)
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "VNDR OATS SPECIAL 22.68KG", "quantity": 3, "unit": "CASE"}])
        line = r.json()["lines"][0]
        assert line["match_source"] == "alias"
        assert line["lb_per_unit"] is None and line["lb_source"] == "none"
        assert line["expected_qty_lb"] is None

    def test_alias_unit_match_is_normalized(self, client, cur):
        sup, exact_pid, alias_pid = self._seed(cur)
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "vndr oats special 22.68kg", "quantity": 4, "unit": " bag. "}])
        line = r.json()["lines"][0]
        assert line["lb_source"] == "alias" and line["lb_per_unit"] == 50
        assert line["expected_qty_lb"] == 200

    def test_alias_with_null_unit_applies_to_unitless_line(self, client, cur):
        sup, exact_pid, alias_pid = self._seed(cur)
        cur.execute(
            """INSERT INTO supplier_product_aliases
                   (supplier_id, vendor_description, product_id, lb_per_unit, unit)
               VALUES (%s, 'VNDR UNITLESS THING', %s, 30, NULL)""",
            (sup, alias_pid))
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "VNDR UNITLESS THING", "quantity": 2, "unit": None}])
        line = r.json()["lines"][0]
        assert line["lb_source"] == "alias" and line["lb_per_unit"] == 30
        assert line["expected_qty_lb"] == 60

    def test_fuzzy_never_computes_lb(self, client, cur):
        self._seed(cur)
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "Match Oats Roled 049 extra wordage", "quantity": 7, "unit": "lb"}])
        line = r.json()["lines"][0]
        assert line["match_source"] in ("fuzzy", "none")
        assert line["expected_qty_lb"] is None
        assert line["lb_per_unit"] == 1.0 and line["lb_source"] == "unit_is_lb"

    def test_unknown_supplier_gives_candidates(self, client, cur):
        self._seed(cur)
        r = self._match(client, "Vendor Mach Co 049",
                        [{"vendor_description": "whatever", "quantity": 1, "unit": None}])
        data = r.json()
        assert data["supplier"]["match"] is None
        assert data["supplier"]["confidence"] == "none"
        assert any("Vendor Match Co 049" == c["name"] for c in data["supplier"]["candidates"])

    def test_duplicate_warning_includes_cancelled(self, client, cur):
        sup, exact_pid, _ = self._seed(cur)
        cur.execute(
            """INSERT INTO expected_receipts (product_id, supplier_id, expected_qty, reference_number, status)
               VALUES (%s, %s, 500, 'PO-777', 'cancelled')""", (exact_pid, sup))
        r = self._match(client, "Vendor Match Co 049",
                        [{"vendor_description": "x", "quantity": 1, "unit": None}],
                        reference="  po-777 ")
        warn = r.json()["duplicate_warning"]
        assert warn and warn["existing"][0]["status"] == "cancelled"

    def test_no_writes(self, client, cur):
        sup, exact_pid, _ = self._seed(cur)
        cur.execute("SELECT count(*) AS n FROM expected_receipts")
        before = cur.fetchone()["n"]
        self._match(client, "Vendor Match Co 049",
                    [{"vendor_description": "Match Oats Rolled 049", "quantity": 3, "unit": "CASE"}])
        cur.execute("SELECT count(*) AS n FROM expected_receipts")
        assert cur.fetchone()["n"] == before


class TestWeightTokenParser:
    """Audit fix 7: _parse_weight_lb_from_description — numeric boundaries,
    N × M packs, and null on any ambiguity."""

    @pytest.mark.parametrize("desc,expected", [
        ("Graham Cracker Crumbs - 50 LB", 50.0),
        ("CHOC CHIPS 50LB", 50.0),
        ("HONEY 60# PAIL", 60.0),
        ("BUTTER .5 LB bag", 0.5),          # audit: was misread as 5 lb
        ("BUTTER 0.5 LB bag", 0.5),
        ("PECANS 4 x 5 LB case", 20.0),     # audit: was misread as 5 lb/unit
        ("PECANS 4x5lb", 20.0),
        ("PECANS 2 X 2.5# case", 5.0),
        ("PECANS 4 x 5 LB (20 LB total)", 20.0),  # pack math and total agree
        ("SS Classic #9 Bulk", None),       # '#9' is an item number
        ("no weight printed here", None),
        ("", None),
        ("MIX 50 LB or 25 LB", None),       # conflicting tokens → manual
        ("WEIRD 1.5.5 LB", None),           # malformed number → manual
        ("ZERO 0 LB", None),
        ("ZERO 0.0 LB pail", None),
    ])
    def test_parser(self, desc, expected):
        assert main._parse_weight_lb_from_description(desc) == expected

    def test_pack_weight_via_match_endpoint(self, client, cur):
        sup = _seed_supplier(cur, "Vendor Pack Co 049")
        r = client.post("/expected-receipts/match", json={"extraction": {
            "supplier_name": "Vendor Pack Co 049", "reference_number": None,
            "document_date": None, "expected_delivery_date": None,
            "lines": [{"vendor_description": "PECANS 4 x 5 LB", "quantity": 3, "unit": "CASE"}]}})
        line = r.json()["lines"][0]
        assert line["lb_per_unit"] == 20 and line["lb_source"] == "parsed_description"


class TestApproveEndpoint:
    def _seed(self, cur):
        sup = _seed_supplier(cur, "Vendor Approve Co 049")
        p1 = _seed_product(cur, "Approve Prod A 049")
        p2 = _seed_product(cur, "Approve Prod B 049")
        doc = _insert_document(cur, path="x/approve-049.png")
        return sup, p1, p2, doc["id"]

    def _approve(self, client, doc_id, sup, lines, reference="PO-049", force=False):
        return client.post("/expected-receipts/extract/approve", json={
            "document_id": doc_id, "supplier_id": sup, "reference_number": reference,
            "expected_date": "2026-09-15", "lines": lines, "force": force})

    @staticmethod
    def _line(pid, qty_lb=100, desc="VNDR THING 50LB", save_alias=True, **kw):
        return {"product_id": pid, "expected_qty_lb": qty_lb, "vendor_description": desc,
                "quantity": 2, "unit": "BAG", "lb_per_unit": 50, "save_alias": save_alias, **kw}

    def test_happy_path_two_lines(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        r = self._approve(client, doc_id, sup,
                          [self._line(p1, desc="VNDR A 50LB"), self._line(p2, qty_lb=60, desc="VNDR B 30LB")])
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["success"] is True and data["created_count"] == 2 and data["aliases_saved"] == 2
        for rec in data["created"]:
            assert rec["source_document_id"] == doc_id
            assert rec["reference_number"] == "PO-049"
            assert rec["status"] == "open"
        assert "VNDR A 50LB" in data["created"][0]["notes"]
        assert data["created"][0]["expected_date"] == "2026-09-15"
        row = _doc_row(cur, doc_id)
        assert row["status"] == "approved" and row["approved_at"] is not None
        cur.execute("SELECT product_id, lb_per_unit FROM supplier_product_aliases WHERE supplier_id = %s ORDER BY id", (sup,))
        aliases = cur.fetchall()
        assert [a["product_id"] for a in aliases] == [p1, p2]

    def test_alias_latest_wins_via_endpoint(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        self._approve(client, doc_id, sup, [self._line(p1, desc="SAME WORDING")])
        doc2 = _insert_document(cur, path="x/approve-049b.png")["id"]
        r = self._approve(client, doc2, sup, [self._line(p2, desc="  same   wording ")], force=True)
        assert r.status_code == 201, r.text
        cur.execute(
            """SELECT product_id FROM supplier_product_aliases
               WHERE supplier_id = %s AND lower(regexp_replace(btrim(vendor_description), '\\s+', ' ', 'g')) = 'same wording'""",
            (sup,))
        rows = cur.fetchall()
        assert len(rows) == 1 and rows[0]["product_id"] == p2

    def test_duplicate_blocks_without_force(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        cur.execute(
            """INSERT INTO expected_receipts (product_id, supplier_id, expected_qty, reference_number, status)
               VALUES (%s, %s, 10, 'PO-049', 'closed')""", (p1, sup))
        r = self._approve(client, doc_id, sup, [self._line(p1)])
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert detail["error_code"] == "DUPLICATE_REFERENCE"
        assert detail["existing"][0]["status"] == "closed"
        r2 = self._approve(client, doc_id, sup, [self._line(p1)], force=True)
        assert r2.status_code == 201
        assert r2.json()["duplicate_overridden"] is True

    def test_atomic_rollback_on_bad_line(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        r = self._approve(client, doc_id, sup, [self._line(p1), self._line(999999999)])
        assert r.status_code == 404
        cur.execute("SELECT count(*) AS n FROM expected_receipts WHERE source_document_id = %s", (doc_id,))
        assert cur.fetchone()["n"] == 0, "first line must roll back with the second"
        assert _doc_row(cur, doc_id)["status"] != "approved"

    def test_double_approve_rejected(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        assert self._approve(client, doc_id, sup, [self._line(p1)]).status_code == 201
        r = self._approve(client, doc_id, sup, [self._line(p1)], force=True)
        assert r.status_code == 409
        assert r.json()["detail"]["error_code"] == "DOCUMENT_ALREADY_APPROVED"

    def test_qty_must_be_positive(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        r = self._approve(client, doc_id, sup, [self._line(p1, qty_lb=0)])
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "INVALID_QUANTITY"

    def test_inactive_supplier_rejected(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        cur.execute("UPDATE suppliers SET active = false WHERE id = %s", (sup,))
        r = self._approve(client, doc_id, sup, [self._line(p1)])
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "SUPPLIER_INACTIVE"

    def test_no_lines_rejected(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        r = self._approve(client, doc_id, sup, [])
        assert r.status_code == 422
        assert r.json()["detail"]["error_code"] == "NO_LINES"

    # Audit fix 11: NaN/±inf weights are 422s on approve AND on the manual path.

    @pytest.mark.parametrize("qty", [float("nan"), float("inf"), 0, -10])
    def test_non_finite_expected_qty_lb_rejected(self, client, cur, qty):
        sup, p1, p2, doc_id = self._seed(cur)
        r = self._approve(client, doc_id, sup, [self._line(p1, qty_lb=qty)])
        assert r.status_code == 422, r.text
        cur.execute("SELECT count(*) AS n FROM expected_receipts WHERE source_document_id = %s", (doc_id,))
        assert cur.fetchone()["n"] == 0

    def test_non_finite_lb_per_unit_rejected(self, client, cur):
        sup, p1, p2, doc_id = self._seed(cur)
        line = self._line(p1)
        line["lb_per_unit"] = float("nan")
        r = self._approve(client, doc_id, sup, [line])
        assert r.status_code == 422, r.text

    def test_manual_endpoint_rejects_nan_expected_qty(self, client, cur):
        """The core-level finite check covers the pre-existing manual gap:
        NaN passes `<= 0` but must still land as INVALID_QUANTITY."""
        sup = _seed_supplier(cur, "Vendor NaN Co 049")
        pid = _seed_product(cur, "NaN Prod 049")
        r = client.post("/expected-receipts", json={
            "product_id": pid, "supplier_name": "Vendor NaN Co 049",
            "expected_qty": float("nan")})
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["error_code"] == "INVALID_QUANTITY"
        cur.execute("SELECT count(*) AS n FROM expected_receipts WHERE supplier_id = %s", (sup,))
        assert cur.fetchone()["n"] == 0

    def test_created_receipts_settle_like_manual_ones(self, client, cur):
        """The core-refactor guarantee: intake-created rows behave identically
        in the list view (remaining = ledger SUM, floored)."""
        sup, p1, p2, doc_id = self._seed(cur)
        self._approve(client, doc_id, sup, [self._line(p1, qty_lb=100)])
        r = client.get(f"/expected-receipts?supplier_id={sup}")
        items = r.json()["expected_receipts"]
        assert len(items) == 1
        assert items[0]["remaining"] == 100 and items[0]["received_qty"] == 0
        assert items[0]["source_document_id"] == doc_id


class TestApprovalAdvisoryLock:
    """Audit fix 10: approve serializes on hash(supplier_id, normalized
    reference), then re-checks duplicates under the lock."""

    KEY = "er-intake-ref:42:po-777"

    def _try_lock(self, cur, key):
        cur.execute("SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS ok", (key,))
        return cur.fetchone()["ok"]

    def test_lock_serializes_the_normalized_pair(self):
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL not set")
        import psycopg2 as pg
        from psycopg2.extras import RealDictCursor
        import main as main_mod
        conn_a, conn_b = pg.connect(url), pg.connect(url)
        try:
            with conn_a.cursor(cursor_factory=RealDictCursor) as ca, \
                 conn_b.cursor(cursor_factory=RealDictCursor) as cb:
                # Whitespace/case in the reference must land on the same key.
                main_mod._lock_supplier_reference(ca, 42, "  PO-777 ")
                assert self._try_lock(cb, self.KEY) is False, "same pair must block"
                assert self._try_lock(cb, "er-intake-ref:42:po-778") is True
                assert self._try_lock(cb, "er-intake-ref:43:po-777") is True
            conn_b.rollback()
            conn_a.rollback()  # transaction-scoped: rollback releases it
            with conn_b.cursor(cursor_factory=RealDictCursor) as cb:
                assert self._try_lock(cb, self.KEY) is True
            conn_b.rollback()
        finally:
            conn_a.close()
            conn_b.close()

    def test_no_lock_without_reference(self):
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("TEST_DATABASE_URL not set")
        import psycopg2 as pg
        from psycopg2.extras import RealDictCursor
        import main as main_mod
        conn = pg.connect(url)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as c:
                main_mod._lock_supplier_reference(c, 42, None)
                main_mod._lock_supplier_reference(c, 42, "   ")
                c.execute("SELECT count(*) AS n FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid()")
                assert c.fetchone()["n"] == 0
            conn.rollback()
        finally:
            conn.close()

    def test_approve_takes_the_pair_lock(self, client, cur, monkeypatch):
        sup = _seed_supplier(cur, "Vendor Lock Co 049")
        pid = _seed_product(cur, "Lock Prod 049")
        doc = _insert_document(cur, path="x/lock-049.png")
        calls = []
        orig = main._lock_supplier_reference

        def _spy(c, supplier_id, reference):
            calls.append((supplier_id, reference))
            return orig(c, supplier_id, reference)

        monkeypatch.setattr(main, "_lock_supplier_reference", _spy)
        r = client.post("/expected-receipts/extract/approve", json={
            "document_id": doc["id"], "supplier_id": sup, "reference_number": "PO-LOCK",
            "lines": [{"product_id": pid, "expected_qty_lb": 10, "vendor_description": "x"}]})
        assert r.status_code == 201, r.text
        assert calls == [(sup, "PO-LOCK")]


class TestSignedUrlEndpoint:
    def test_signed_url(self, client, cur, mock_storage):
        doc = _insert_document(cur, path="x/url-049.png")
        r = client.get(f"/purchase-documents/{doc['id']}/url")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["url"].endswith("x/url-049.png")
        assert data["document"]["status"] == "uploaded"

    def test_unknown_document(self, client, mock_storage):
        r = client.get("/purchase-documents/999999999/url")
        assert r.status_code == 404


class TestAllowlistAndTripwire:
    INTAKE_ROUTES = [
        ("POST", "/expected-receipts/extract"),
        ("POST", "/expected-receipts/match"),
        ("POST", "/expected-receipts/extract/approve"),
        ("POST", "/purchase-documents/{document_id}/extract"),
        ("GET", "/purchase-documents/{document_id}/url"),
    ]

    def test_routes_on_dashboard_allowlist(self):
        for pair in self.INTAKE_ROUTES:
            assert pair in main.DASHBOARD_KEY_ALLOWLIST, pair

    def test_dashboard_key_accepted_on_match(self, client, cur):
        _seed_supplier(cur, "Dash Key Sup 049")
        r = client.post("/expected-receipts/match",
                        headers={"X-API-Key": main.DASHBOARD_API_KEY},
                        json={"extraction": {"supplier_name": "Dash Key Sup 049",
                                             "reference_number": None, "document_date": None,
                                             "expected_delivery_date": None,
                                             "lines": [{"vendor_description": "x", "quantity": 1, "unit": None}]}})
        assert r.status_code == 200, r.text

    @pytest.fixture
    def readonly_client(self, monkeypatch):
        """Every DB connection raises a real psycopg2 readonly error on first
        execute — the 'readonly armed' state the global tripwire exists for."""
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
        # storage/extractor must not be the failure here — and the upload spy
        # lets tests assert NOTHING reached Storage (row-first ordering).
        uploads = []
        monkeypatch.setattr(main, "storage_upload_purchase_document",
                            lambda *a, **kw: uploads.append(a))
        monkeypatch.setattr(extraction, "extract_purchase_document",
                            lambda *a, **kw: {"extraction": dict(GOOD_EXTRACTION), "extraction_model": "m"})
        with TestClient(main.app, raise_server_exceptions=False) as c:
            c.headers["X-API-Key"] = main.API_KEY
            c.storage_uploads = uploads
            yield c

    def test_extract_trips_readonly_tripwire(self, readonly_client):
        r = _post_file(readonly_client)
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error_code"] == "READONLY_TRANSACTION"
        assert body["success"] is False
        assert body["retryable"] is True
        assert "error_detail" in body  # write_response_envelope post-processed it
        # Row-first ordering (owner ruling): the readonly 503 fires on the
        # purchase_documents INSERT, so no orphan object lands in Storage.
        assert readonly_client.storage_uploads == []

    def test_approve_trips_readonly_tripwire(self, readonly_client):
        r = readonly_client.post("/expected-receipts/extract/approve", json={
            "document_id": 1, "supplier_id": 1, "reference_number": "PO-1",
            "lines": [{"product_id": 1, "expected_qty_lb": 10, "vendor_description": "x"}]})
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error_code"] == "READONLY_TRANSACTION"
        assert body["success"] is False
        assert "error_detail" in body

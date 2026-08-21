"""Tests for Kosher Ignition visibility (FACTORY_LEDGER_CHANGELOG row 51 follow-up).

Products whose `verification_notes` is non-empty (e.g. the SS Classic #9
batches 90025/90026, which require the oven to be lit by the owner or his
messenger) must surface that note wherever production is initiated:

  - GET /bom/batches/{id}/formula returns verification_notes(+_es) and a
    production_warning block.
  - POST /make (preview and commit) returns a production_warning block the
    Floor GPT relays to the operator verbatim before committing.

Products without notes must NOT gain any of these keys.
"""

from contextlib import contextmanager

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi not installed", allow_module_level=True)

import main


NOTE_EN = (
    "Kosher Ignition: oven must be turned on by owner (Blubber) or his "
    "designated messenger. Do not start production without this."
)
NOTE_ES = "Ignición kosher: el horno debe ser encendido por el dueño o su mensajero."


class _ConnProxy:
    """Wrap the test connection so commit()/rollback() act on an inner
    SAVEPOINT; the outer _db_connection fixture rolls everything back."""

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
        proxy = _ConnProxy(_db_connection, "prod_warning_inner")
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


def _seed_batch(conn, name, notes=None, notes_es=None):
    """Batch product + 1-ingredient formula + 100 lb of posted ingredient stock."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO products (name, type, uom, default_batch_lb, active, "
            "verification_notes, verification_notes_es) "
            "VALUES (%s, 'batch', 'lb', 50, true, %s, %s) RETURNING id",
            (name, notes, notes_es),
        )
        batch_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO products (name, type, uom, active) "
            "VALUES (%s, 'ingredient', 'lb', true) RETURNING id",
            (f"{name} OATS ZQX",),
        )
        ing_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) "
            "VALUES (%s, %s, 50)",
            (batch_id, ing_id),
        )
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (ing_id, f"PRODWARN-{ing_id}"),
        )
        lot_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO transactions (type, timestamp, status, notes) "
            "VALUES ('receive', NOW(), 'posted', 'prod-warning test seed') RETURNING id",
            (),
        )
        txn_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, 100)",
            (txn_id, ing_id, lot_id),
        )
    return batch_id


def test_bom_formula_includes_verification_notes(client, _db_connection):
    batch_id = _seed_batch(
        _db_connection, "PRODWARN Kosher Batch ZQX", notes=NOTE_EN, notes_es=NOTE_ES
    )
    body = client.get(f"/bom/batches/{batch_id}/formula").json()
    assert body["verification_notes"] == NOTE_EN
    assert body["verification_notes_es"] == NOTE_ES
    warning = body["production_warning"]
    assert warning["verification_notes"] == NOTE_EN
    assert warning["verification_notes_es"] == NOTE_ES
    assert NOTE_EN in warning["message"]


def test_bom_formula_without_notes_has_no_warning(client, _db_connection):
    batch_id = _seed_batch(_db_connection, "PRODWARN Plain Batch ZQX")
    body = client.get(f"/bom/batches/{batch_id}/formula").json()
    assert "production_warning" not in body
    assert "verification_notes" not in body


def test_make_preview_includes_production_warning(client, _db_connection):
    _seed_batch(_db_connection, "PRODWARN Kosher Batch ZQX", notes=NOTE_EN, notes_es=NOTE_ES)
    body = client.post(
        "/make",
        json={"mode": "preview", "product_name": "PRODWARN Kosher Batch ZQX", "batches": 1},
    ).json()
    warning = body["production_warning"]
    assert warning["verification_notes"] == NOTE_EN
    assert warning["verification_notes_es"] == NOTE_ES
    assert NOTE_EN in body["preview_message"]


def test_make_commit_includes_production_warning(client, _db_connection):
    _seed_batch(_db_connection, "PRODWARN Kosher Batch ZQX", notes=NOTE_EN)
    body = client.post(
        "/make",
        json={"mode": "commit", "product_name": "PRODWARN Kosher Batch ZQX", "batches": 1},
    ).json()
    assert body["success"] is True
    warning = body["production_warning"]
    assert warning["verification_notes"] == NOTE_EN
    assert "verification_notes_es" not in warning  # ES note not set on this product


def test_make_without_notes_has_no_warning(client, _db_connection):
    _seed_batch(_db_connection, "PRODWARN Plain Batch ZQX")
    preview = client.post(
        "/make",
        json={"mode": "preview", "product_name": "PRODWARN Plain Batch ZQX", "batches": 1},
    ).json()
    assert "production_warning" not in preview
    commit = client.post(
        "/make",
        json={"mode": "commit", "product_name": "PRODWARN Plain Batch ZQX", "batches": 1},
    ).json()
    assert commit["success"] is True
    assert "production_warning" not in commit


def test_whitespace_only_notes_do_not_warn(client, _db_connection):
    batch_id = _seed_batch(_db_connection, "PRODWARN Blank Notes ZQX", notes="   ")
    body = client.get(f"/bom/batches/{batch_id}/formula").json()
    assert "production_warning" not in body

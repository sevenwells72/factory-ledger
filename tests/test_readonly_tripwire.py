"""Tests for the global @app.exception_handler(Exception) readonly tripwire.

Covers the three cases from proposal v2 §4a:
  1. Positive — ship_commit returns 503 with diagnostics; READONLY_TRIPWIRE
     log line emitted with expected JSON shape.
  2. Negative (regression guard) — a non-readonly RuntimeError in ship_commit
     follows the per-route 500 path, body is exactly {"error": "boom"}, and
     NO READONLY_TRIPWIRE log line is emitted. Guards against future removal
     of the `if _is_readonly_error(e): raise` line silently routing every
     error through the global handler.
  3. Parametrized — readonly errors on 4 routes spanning categories
     (commit-branch, lot rename PATCH, admin BOM POST, dashboard notes POST)
     all return 503 with READONLY_TRANSACTION error_code via the global
     handler. Proves the wiring is per-route, not just on ship.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import httpx
import pytest

try:
    import main
except Exception as e:  # pragma: no cover — PEP 604 requires py3.10+
    pytest.skip(f"cannot import main ({e})", allow_module_level=True)


# ─── Fake conn/cursor that raises a given exception on the first execute ───

class _FakeCursor:
    def __init__(self, exc: Exception):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *args, **kwargs):
        raise self._exc

    def fetchone(self):  # pragma: no cover — should never be reached
        raise AssertionError("fetchone() should not be reached; execute() raised first")

    def fetchall(self):  # pragma: no cover
        raise AssertionError("fetchall() should not be reached; execute() raised first")


class _FakeConn:
    def __init__(self, exc: Exception):
        self._exc = exc

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self._exc)

    def commit(self):  # pragma: no cover — execute raised, commit shouldn't be called
        pass

    def rollback(self):
        pass


def _make_raising_get_db_connection(exc: Exception):
    """Return a context manager that yields a fake conn whose cursor.execute
    raises `exc` on first call. Drop-in replacement for `main.get_db_connection`."""

    @contextmanager
    def _get_db():
        yield _FakeConn(exc)

    return _get_db


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def anyio_backend():
    """anyio (3.7.x in this venv) requires this fixture to pick an async backend."""
    return "asyncio"


@pytest.fixture
async def client():
    """ASGI-in-process HTTP client.

    Replaces starlette's TestClient because the container's starlette↔httpx
    versions are incompatible: starlette TestClient calls
    `super().__init__(app=...)` but httpx>=0.28 removed the `app=` kwarg
    from `Client.__init__`. The modern documented pattern is
    `AsyncClient(transport=ASGITransport(app=app))`, with `raise_app_exceptions=False`
    matching the original `TestClient(..., raise_server_exceptions=False)`.
    """
    transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def api_key(monkeypatch):
    """Ensure main.API_KEY is set (it normally comes from env at startup)."""
    if not main.API_KEY:
        monkeypatch.setattr(main, "API_KEY", "test-key")
    return main.API_KEY or "test-key"


@pytest.fixture
def stub_diagnostics(monkeypatch):
    """Replace the DB probe so tests don't need a live pool. Returns the dict
    the test should expect to see echoed in the response/log."""
    fake = {
        "default_ro": "on",
        "txn_ro": "on",
        "server_ip": "10.0.0.1/32",
        "db": "postgres",
        "usr": "postgres",
        "is_replica": True,
        "pg_version": "PostgreSQL 17.6 (test fixture)",
    }
    monkeypatch.setattr(main, "_capture_readonly_diagnostics", lambda: dict(fake))
    return fake


READONLY_ERROR_STR = "cannot execute INSERT in a read-only transaction"


class _ReadOnlyError(Exception):
    """Stand-in for psycopg2.errors.ReadOnlySqlTransaction. The tripwire
    matches on the string content, not the exception class."""

    def __str__(self) -> str:
        return READONLY_ERROR_STR


# ─── Case 1: positive — ship_commit returns 503 with diagnostics ─────────────

@pytest.mark.anyio
async def test_readonly_ship_commit_returns_503_with_diagnostics(
    client, api_key, stub_diagnostics, monkeypatch, caplog
):
    monkeypatch.setattr(
        main, "get_db_connection", _make_raising_get_db_connection(_ReadOnlyError())
    )

    with caplog.at_level(logging.ERROR, logger="main"):
        resp = await client.post(
            "/sales/orders/1/ship",
            json={"mode": "commit", "ship_all": True},
            headers={"X-API-Key": api_key},
        )

    # Response envelope
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "READONLY_TRANSACTION"
    assert body["retryable"] is True
    assert body["error"] == READONLY_ERROR_STR
    assert body["diagnostics"] == stub_diagnostics
    assert "message" in body and "read-only" in body["message"].lower()

    # Log shape — exactly one READONLY_TRIPWIRE line
    tripwire = [r for r in caplog.records if r.getMessage().startswith("READONLY_TRIPWIRE: ")]
    assert len(tripwire) == 1, f"expected 1 READONLY_TRIPWIRE log, got {len(tripwire)}"

    payload = json.loads(tripwire[0].getMessage()[len("READONLY_TRIPWIRE: "):])
    assert payload["path"] == "/sales/orders/1/ship"
    assert payload["method"] == "POST"
    assert payload["error"] == READONLY_ERROR_STR
    assert payload["diagnostics"] == stub_diagnostics


# ─── Case 2: NEGATIVE regression guard — RuntimeError takes per-route path ───

@pytest.mark.anyio
async def test_generic_runtimeerror_in_ship_commit_uses_per_route_500(
    client, api_key, monkeypatch, caplog
):
    """If anyone ever removes `if _is_readonly_error(e): raise` from
    ship_order's commit-branch except, ALL exceptions (including this
    RuntimeError) would bubble to the global handler, which returns
    `{"success": False, "error_code": "INTERNAL_SERVER_ERROR", "error": "boom"}`.

    This test asserts the OLD per-route shape exactly, so removal of the
    re-raise line would fail the equality assertion below.
    """
    monkeypatch.setattr(
        main, "get_db_connection", _make_raising_get_db_connection(RuntimeError("boom"))
    )

    with caplog.at_level(logging.ERROR, logger="main"):
        resp = await client.post(
            "/sales/orders/1/ship",
            json={"mode": "commit", "ship_all": True},
            headers={"X-API-Key": api_key},
        )

    # Per-route legacy shape — no `success`, `error_code`, or `diagnostics`
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body == {"error": "boom"}, (
        "Body must match the legacy per-route shape exactly. If you see "
        "{'success': False, 'error_code': 'INTERNAL_SERVER_ERROR', ...} here, "
        "someone removed the `if _is_readonly_error(e): raise` line from "
        "ship_order's commit-branch except. Restore it."
    )

    # Global tripwire must NOT have fired
    tripwire = [r for r in caplog.records if r.getMessage().startswith("READONLY_TRIPWIRE: ")]
    assert tripwire == [], (
        f"Expected 0 READONLY_TRIPWIRE log lines for a generic RuntimeError, "
        f"got {len(tripwire)}: {[r.getMessage() for r in tripwire]}"
    )

    # Per-route logger.error should still fire
    per_route = [
        r for r in caplog.records if "Ship order commit failed: boom" in r.getMessage()
    ]
    assert len(per_route) == 1


# ─── Case 3: parametrized — wiring works on routes across categories ─────────

@pytest.mark.parametrize(
    "method, path, body, label",
    [
        # commit-branch route (the original incident class)
        ("POST", "/sales/orders/1/ship", {"mode": "commit", "ship_all": True},
         "POST /sales/orders/{id}/ship (commit)"),
        # PATCH route with a single top-level except (rename)
        # NOTE: lot_id is int in the path; body strip happens before the DB call,
        # so any non-empty new_lot_code reaches get_transaction() → first
        # cur.execute() → _ReadOnlyError.
        ("PATCH", "/lots/1/rename", {"new_lot_code": "L260101-002"},
         "PATCH /lots/{lot_id}/rename"),
        # admin BOM POST (admin surface)
        # NOTE: BomLineCreate requires `ingredient_product_id` (not ingredient_id).
        # Handler enters get_transaction() and immediately runs
        # `SELECT id, name FROM products WHERE id = %s` — that's where the fake
        # cursor raises.
        ("POST", "/admin/bom/1/lines",
         {"ingredient_product_id": 2, "quantity_lb": 1.0},
         "POST /admin/bom/{product_id}/lines"),
        # dashboard notes POST (different code region; no auth)
        # NOTE: NoteCreate requires `category` (∈ {'note','todo','reminder'}) and
        # `title`. Handler enters get_transaction() and runs an INSERT — fake
        # cursor raises on .execute().
        ("POST", "/dashboard/api/notes",
         {"category": "note", "title": "tripwire test"},
         "POST /dashboard/api/notes"),
    ],
)
@pytest.mark.anyio
async def test_global_handler_fires_for_each_route(
    client, api_key, stub_diagnostics, monkeypatch, caplog, method, path, body, label
):
    """Readonly errors on each of these routes should produce the same 503
    envelope and the same READONLY_TRIPWIRE log shape — proving the wiring
    is present at each per-route except, not just on ship_order."""
    monkeypatch.setattr(
        main, "get_db_connection", _make_raising_get_db_connection(_ReadOnlyError())
    )

    with caplog.at_level(logging.ERROR, logger="main"):
        resp = await client.request(method, path, json=body, headers={"X-API-Key": api_key})

    assert resp.status_code == 503, f"[{label}] body: {resp.text}"
    body_json = resp.json()
    assert body_json["error_code"] == "READONLY_TRANSACTION", f"[{label}] {body_json}"
    assert body_json["retryable"] is True, f"[{label}] {body_json}"
    assert "diagnostics" in body_json, f"[{label}] {body_json}"

    tripwire = [r for r in caplog.records if r.getMessage().startswith("READONLY_TRIPWIRE: ")]
    assert len(tripwire) == 1, f"[{label}] got {len(tripwire)} READONLY_TRIPWIRE lines"
    payload = json.loads(tripwire[0].getMessage()[len("READONLY_TRIPWIRE: "):])
    assert payload["path"] == path, f"[{label}] path mismatch: {payload['path']}"
    assert payload["method"] == method, f"[{label}] method mismatch: {payload['method']}"

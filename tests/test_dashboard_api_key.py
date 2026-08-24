"""Auth tests for the scoped DASHBOARD_API_KEY (branch feat/dashboard-scoped-api-key).

Two keys are accepted by verify_api_key / verify_api_key_flexible:
  * API_KEY (master)          -> full access, unchanged behaviour
  * DASHBOARD_API_KEY (scoped) -> only (METHOD, route-template) pairs in
    main.DASHBOARD_KEY_ALLOWLIST; every other route -> 403
    "API key not authorized for this endpoint"
  * missing key -> 401, unknown key -> 403 (header) / 401 (packing-slip ?key=)

Same TestClient + savepoint-proxy pattern as test_notes_auth.py.
"""

from contextlib import contextmanager

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

import main


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
        proxy = _ConnProxy(_db_connection, "dash_key_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "test-dashboard-key")
    with TestClient(main.app) as c:
        yield c
    _db_connection.rollback()


MASTER = {"X-API-Key": "test-api-key"}
DASH = {"X-API-Key": "test-dashboard-key"}
WRONG = {"X-API-Key": "nope"}

NOT_AUTHORIZED = "API key not authorized for this endpoint"


# ─────────────────────────────────────────────────────────────────
# Allowlist shape (no DB needed)
# ─────────────────────────────────────────────────────────────────

def _registered():
    return {(m, r.path) for r in main.app.routes for m in getattr(r, "methods", [])}


def test_allowlist_entries_are_real_routes():
    unmatched = [k for k in main.DASHBOARD_KEY_ALLOWLIST if k not in _registered()]
    assert not unmatched, f"allowlist entries with no matching route: {unmatched}"


def test_allowlist_never_grants_admin_or_dangerous_routes():
    for method, path in main.DASHBOARD_KEY_ALLOWLIST:
        assert not path.startswith("/admin"), (method, path)
        assert path not in {"/ship", "/receive", "/make", "/pack", "/adjust", "/schedule"}, (method, path)
        assert not path.startswith("/void"), (method, path)
        assert not path.startswith("/make"), (method, path)
        assert not path.startswith("/pack"), (method, path)
        assert not path.startswith("/adjust"), (method, path)
        if method == "DELETE":
            assert path.startswith("/dashboard/api/notes"), (method, path)


def test_admin_sql_is_master_only():
    assert ("POST", "/admin/sql") not in main.DASHBOARD_KEY_ALLOWLIST


# ─────────────────────────────────────────────────────────────────
# Dashboard key on allowlisted routes -> passes auth
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_dashboard_key_allowed_on_read_routes(client):
    for path in ["/dashboard/inventory", "/sales/orders?limit=1", "/products/search?q=granola",
                 "/transactions/history?limit=1", "/reason-codes", "/customers"]:
        resp = client.get(path, headers=DASH)
        assert resp.status_code == 200, f"{path}: {resp.status_code} {resp.text[:200]}"


@pytest.mark.db
def test_dashboard_key_matches_route_template_not_raw_url(client):
    # /lots/{lot_id} is allowlisted; a nonexistent id must reach the handler (404),
    # not be rejected by auth (403).
    resp = client.get("/lots/999999999", headers=DASH)
    assert resp.status_code == 404, resp.text


# ─────────────────────────────────────────────────────────────────
# Dashboard key on NON-allowlisted routes -> 403, before any handler logic
# ─────────────────────────────────────────────────────────────────

def _forbidden_calls(client):
    return [
        ("POST /make", client.post("/make", json={}, headers=DASH)),
        ("POST /pack", client.post("/pack", json={}, headers=DASH)),
        ("POST /adjust", client.post("/adjust", json={}, headers=DASH)),
        ("POST /ship", client.post("/ship", json={}, headers=DASH)),
        ("POST /ship/commit", client.post("/ship/commit", json={}, headers=DASH)),
        ("POST /receive", client.post("/receive", json={}, headers=DASH)),
        ("POST /void/{id}", client.post("/void/1", json={}, headers=DASH)),
        ("POST /admin/lots/merge", client.post("/admin/lots/merge", json={}, headers=DASH)),
        ("DELETE /admin/bom/lines/{id}", client.delete("/admin/bom/lines/1", headers=DASH)),
        ("PUT /admin/products/{id}", client.put("/admin/products/1", json={}, headers=DASH)),
        ("POST /schedule", client.post("/schedule", json={}, headers=DASH)),
        ("GET /admin/lots/duplicates", client.get("/admin/lots/duplicates", headers=DASH)),
        ("POST /sales/orders", client.post("/sales/orders", json={}, headers=DASH)),
        ("POST /lots/{id}/reassign", client.post("/lots/1/reassign", json={}, headers=DASH)),
    ]


@pytest.mark.db
def test_dashboard_key_403_on_non_allowlisted_routes(client):
    for label, resp in _forbidden_calls(client):
        assert resp.status_code == 403, f"{label}: expected 403, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["detail"] == NOT_AUTHORIZED, label


@pytest.mark.db
def test_dashboard_key_403_on_packing_slip_via_query_param(client):
    resp = client.get("/sales/orders/1/packing-slip?key=test-dashboard-key")
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == NOT_AUTHORIZED


# ─────────────────────────────────────────────────────────────────
# Master key: unchanged full access
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_master_key_still_reaches_master_only_routes(client):
    # Auth passes -> handler runs (200 for reads; 422 body validation for empty writes)
    assert client.get("/admin/lots/duplicates", headers=MASTER).status_code == 200
    assert client.get("/products/unverified", headers=MASTER).status_code == 200
    assert client.post("/make", json={}, headers=MASTER).status_code == 422


@pytest.mark.db
def test_master_key_on_allowlisted_routes(client):
    assert client.get("/dashboard/inventory", headers=MASTER).status_code == 200


# ─────────────────────────────────────────────────────────────────
# Missing / wrong key: existing semantics unchanged
# ─────────────────────────────────────────────────────────────────

@pytest.mark.db
def test_missing_key_401_and_wrong_key_403(client):
    for path in ["/dashboard/inventory", "/admin/lots/duplicates"]:
        r = client.get(path)
        assert r.status_code == 401 and r.json()["detail"] == "API key required", (path, r.text)
        r = client.get(path, headers=WRONG)
        assert r.status_code == 403 and r.json()["detail"] == "Invalid API key", (path, r.text)
    # packing-slip flexible dep keeps its historical 401-on-wrong-key
    r = client.get("/sales/orders/1/packing-slip?key=nope")
    assert r.status_code == 401 and r.json()["detail"] == "Invalid API key", r.text

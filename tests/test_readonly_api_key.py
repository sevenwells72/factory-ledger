"""G1: the read-only GPT key (READONLY_API_KEY, key_kind='readonly').

Design §0.6, §10 row G1, §10.1 steps 3–4. The key is accepted ONLY on
main.READONLY_KEY_ALLOWLIST — exactly the operations left in the trimmed GPT
schemas under gpt-configs/readonly/ — and every other authenticated route
answers 403 "API key not authorized for this endpoint". Unset (the default),
the key does not exist and the master / dashboard / actor keys behave as before.

Same TestClient + savepoint-proxy pattern as test_dashboard_api_key.py.
"""

import re
from contextlib import contextmanager
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
    from fastapi.routing import APIRoute
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

import yaml

import main

ROOT = Path(__file__).resolve().parent.parent
READONLY_DIR = ROOT / "gpt-configs" / "readonly"
OFFICE_SOURCE = ROOT / "openapi-gpt-v3.yaml"
FLOOR_SOURCE = ROOT / "gpt-configs" / "schemas" / "openapi-floor.yaml"
OFFICE_READONLY = READONLY_DIR / "openapi-office-readonly.yaml"
FLOOR_READONLY = READONLY_DIR / "openapi-floor-readonly.yaml"
OFFICE_INSTRUCTIONS = READONLY_DIR / "office-instructions.md"
FLOOR_INSTRUCTIONS = READONLY_DIR / "floor-instructions.md"

READONLY_TEST_KEY = "test-readonly-key"
MASTER = {"X-API-Key": "test-api-key"}
DASH = {"X-API-Key": "test-dashboard-key"}
RO = {"X-API-Key": READONLY_TEST_KEY}
NOT_AUTHORIZED = "API key not authorized for this endpoint"

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
# The one POST the read-only key reaches: a product lookup that writes nothing.
READ_POSTS = {("POST", "/products/resolve")}
# The only non-GET route without the API-key dependency: the key-to-session
# exchange, which authenticates the body's actor key itself (key_sessions.py).
UNAUTHENTICATED_WRITES = {("POST", "/auth/session")}


def _operations(path):
    spec = yaml.safe_load(path.read_text())
    return {
        (method.upper(), p): op["operationId"]
        for p, item in (spec.get("paths") or {}).items()
        for method, op in item.items()
        if method.lower() in _HTTP_METHODS
    }


def _registered():
    return {(m, r.path) for r in main.app.routes for m in getattr(r, "methods", None) or ()}


def _uses_api_key(route):
    auth = {main.verify_api_key, main.verify_api_key_flexible}

    def calls(dependant):
        for sub in dependant.dependencies:
            yield sub.call
            yield from calls(sub)

    return any(c in auth for c in calls(route.dependant))


def _authenticated_routes():
    return sorted(
        (m, r.path)
        for r in main.app.routes
        if isinstance(r, APIRoute) and _uses_api_key(r)
        for m in r.methods
    )


# ─────────────────────────────────────────────────────────────────
# Allowlist shape (no DB needed)
# ─────────────────────────────────────────────────────────────────

def test_allowlist_is_exactly_the_trimmed_gpt_schemas():
    schema_ops = set(_operations(OFFICE_READONLY)) | set(_operations(FLOOR_READONLY))
    assert main.READONLY_KEY_ALLOWLIST == schema_ops


def test_allowlist_is_reads_only():
    for method, path in main.READONLY_KEY_ALLOWLIST:
        assert method == "GET" or (method, path) in READ_POSTS, (method, path)


def test_allowlist_entries_are_real_routes_reachable_by_existing_scoped_keys():
    registered = _registered()
    for key in main.READONLY_KEY_ALLOWLIST:
        assert key in registered, key
        assert key in (main.DASHBOARD_KEY_ALLOWLIST | main.ACTOR_WRITE_ALLOWLIST), key


@pytest.mark.parametrize("trimmed,source,cap", [
    (OFFICE_READONLY, OFFICE_SOURCE, 14),
    (FLOOR_READONLY, FLOOR_SOURCE, 11),
], ids=["office", "floor"])
def test_trimmed_schema_is_the_read_subset_of_its_source(trimmed, source, cap):
    trimmed_ops, source_ops = _operations(trimmed), _operations(source)
    expected = {k: v for k, v in source_ops.items() if k[0] == "GET" or k in READ_POSTS}
    assert trimmed_ops == expected
    assert len(trimmed_ops) == cap <= 30  # HARD RULE: 30-operation cap
    assert not [k for k in trimmed_ops if k[0] != "GET" and k not in READ_POSTS]


@pytest.mark.parametrize("trimmed", [OFFICE_READONLY, FLOOR_READONLY], ids=["office", "floor"])
def test_trimmed_schema_is_self_contained_and_pasteable(trimmed):
    text = trimmed.read_text()
    spec = yaml.safe_load(text)
    assert spec["openapi"] == "3.1.0"
    assert spec["servers"] == [{"url": "https://fastapi-production-b73a.up.railway.app"}]
    assert spec["components"]["securitySchemes"]["ApiKeyAuth"] == {
        "type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert "READ-ONLY" in spec["info"]["description"]
    defined = set((spec["components"].get("schemas") or {}))
    referenced = set(re.findall(r"#/components/schemas/([A-Za-z0-9_]+)", text))
    assert referenced <= defined, referenced - defined
    assert defined <= referenced, f"unused schemas: {defined - referenced}"
    for (method, path), op_id in _operations(trimmed).items():
        op = spec["paths"][path][method.lower()]
        assert len(op.get("summary", "")) <= 300, op_id
        assert len(op.get("description", "")) <= 300, op_id
    assert "?key=" not in text


@pytest.mark.parametrize("instructions,schema", [
    (OFFICE_INSTRUCTIONS, OFFICE_READONLY),
    (FLOOR_INSTRUCTIONS, FLOOR_READONLY),
], ids=["office", "floor"])
def test_instructions_fit_and_carry_no_write_guidance(instructions, schema):
    text = instructions.read_text(encoding="utf-8")
    assert len(text) <= 8000  # GPT editor limit, in characters
    assert "read-only" in text.lower()
    assert "use the FL Assistant in the FL dashboard" in text
    # Every action the instructions name exists in the paired read-only schema.
    kept = set(_operations(schema).values())
    removed = (set(_operations(OFFICE_SOURCE).values()) | set(_operations(FLOOR_SOURCE).values())) - kept
    named = set(re.findall(r"\b([a-z]+[A-Z][A-Za-z]+)\b", text))
    assert not (named & removed), named & removed
    for action in named & (set(_operations(OFFICE_SOURCE).values()) | set(_operations(FLOOR_SOURCE).values())):
        assert action in kept, action
    for write_hint in ('mode: "preview"', 'mode: "commit"', "confirmation_code", "?key=", "/void/"):
        assert write_hint not in text, write_hint


def test_floor_instructions_are_bilingual():
    text = FLOOR_INSTRUCTIONS.read_text(encoding="utf-8")
    assert "Solo puedo consultar" in text
    assert "usa el FL Assistant en el dashboard de FL" in text
    assert "Spanish input" in text


# ─────────────────────────────────────────────────────────────────
# Startup refuses a read-only key that equals a legacy key
# ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("clash", ["API_KEY", "DASHBOARD_API_KEY"])
def test_startup_refuses_readonly_key_equal_to_a_legacy_key(monkeypatch, clash):
    import asyncio
    monkeypatch.setattr(main, "DATABASE_URL", main.DATABASE_URL or "postgresql://localhost/none")
    monkeypatch.setattr(main, "API_KEY", "k-master")
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "k-dash")
    monkeypatch.setattr(main, "READONLY_API_KEY", "k-master" if clash == "API_KEY" else "k-dash")
    monkeypatch.setattr(main, "assert_staging_database", lambda url: None)
    with pytest.raises(RuntimeError, match="READONLY_API_KEY must differ"):
        asyncio.run(main.startup())


# ─────────────────────────────────────────────────────────────────
# Live auth decisions
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


def _client(_db_connection, monkeypatch, readonly_key):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "readonly_key_inner")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    monkeypatch.setattr(main, "DASHBOARD_API_KEY", "test-dashboard-key")
    monkeypatch.setattr(main, "READONLY_API_KEY", readonly_key)
    # Feature-flagged routes 404 before auth when off; switch them on so the
    # sweep below tests the key decision, not the flag.
    monkeypatch.setenv("KEY_SESSIONS_ENABLED", "1")
    return TestClient(main.app)


@pytest.fixture
def client(_db_connection, monkeypatch):
    with _client(_db_connection, monkeypatch, READONLY_TEST_KEY) as c:
        yield c
    _db_connection.rollback()


@pytest.fixture
def client_unset(_db_connection, monkeypatch):
    with _client(_db_connection, monkeypatch, "") as c:
        yield c
    _db_connection.rollback()


def _call(client, method, path, headers):
    url = re.sub(r"\{[^}]+\}", "1", path)
    kwargs = {"headers": headers}
    if method in ("POST", "PUT", "PATCH"):
        kwargs["json"] = {}
    return client.request(method, url, **kwargs)


# Real requests for every GPT read: a 200 proves the key got through auth AND
# the handler ran (the parameter values only need to be valid, not to match).
GPT_READ_CALLS = {
    ("POST", "/products/resolve"): ("POST", "/products/resolve", {"names": ["granola"]}),
    ("GET", "/products/search"): ("GET", "/products/search?q=granola", None),
    ("GET", "/bom/products"): ("GET", "/bom/products?limit=1", None),
    ("GET", "/bom/batches/{batch_id}/formula"): ("GET", "/bom/batches/999999999/formula", None),
    ("GET", "/inventory/lookup"): ("GET", "/inventory/lookup?q=granola", None),
    ("GET", "/lots/by-code/{lot_code}"): ("GET", "/lots/by-code/NO-SUCH-LOT", None),
    ("GET", "/lots/by-supplier-lot/{supplier_lot_code}"): ("GET", "/lots/by-supplier-lot/NO-SUCH-LOT", None),
    ("GET", "/trace/batch/{lot_code}"): ("GET", "/trace/batch/NO-SUCH-LOT", None),
    ("GET", "/trace/ingredient/{lot_code}"): ("GET", "/trace/ingredient/NO-SUCH-LOT", None),
    ("GET", "/trace/supplier-lot/{supplier_lot_code}"): ("GET", "/trace/supplier-lot/NO-SUCH-LOT", None),
    ("GET", "/transactions/history"): ("GET", "/transactions/history?limit=1", None),
    ("GET", "/production/day-summary"): ("GET", "/production/day-summary", None),
    ("GET", "/customers"): ("GET", "/customers", None),
    ("GET", "/customers/search"): ("GET", "/customers/search?q=acme", None),
    ("GET", "/sales/orders"): ("GET", "/sales/orders?limit=1", None),
    ("GET", "/sales/orders/{order_id}"): ("GET", "/sales/orders/999999999", None),
}


def test_every_gpt_read_has_a_live_call():
    assert set(GPT_READ_CALLS) == main.READONLY_KEY_ALLOWLIST


@pytest.mark.db
@pytest.mark.parametrize("route", sorted(GPT_READ_CALLS), ids=[f"{m} {p}" for m, p in sorted(GPT_READ_CALLS)])
def test_readonly_key_reaches_every_gpt_read(client, route):
    method, url, body = GPT_READ_CALLS[route]
    resp = client.request(method, url, headers=RO, json=body)
    # Lookups of a nonexistent lot/order/batch answer 404 from the handler —
    # past auth; only the route-scope 403 would mean the key was refused.
    assert resp.status_code in (200, 404), f"{route}: {resp.status_code} {resp.text[:200]}"
    if resp.status_code == 404:
        assert resp.json().get("detail") != NOT_AUTHORIZED
    # The same request gets the same status with the master key.
    assert client.request(method, url, headers=MASTER, json=body).status_code == resp.status_code


WRITE_ROUTES = [k for k in _authenticated_routes() if k[0] != "GET" and k not in READ_POSTS]
OTHER_GETS = [k for k in _authenticated_routes() if k[0] == "GET" and k not in main.READONLY_KEY_ALLOWLIST]


def test_route_inventory_is_complete():
    """Every non-GET route is either covered by the 403 sweep below, the one
    read POST, or the self-authenticating session exchange."""
    non_get = {k for k in _registered() if k[0] not in ("GET", "HEAD")}
    assert non_get == set(WRITE_ROUTES) | READ_POSTS | UNAUTHENTICATED_WRITES
    assert len(WRITE_ROUTES) > 50


@pytest.mark.db
@pytest.mark.parametrize("route", WRITE_ROUTES, ids=[f"{m} {p}" for m, p in WRITE_ROUTES])
def test_readonly_key_refused_on_every_write_route(client, route):
    method, path = route
    resp = _call(client, method, path, RO)
    assert resp.status_code == 403, f"{method} {path}: {resp.status_code} {resp.text[:200]}"
    assert resp.json()["detail"] == NOT_AUTHORIZED


@pytest.mark.db
@pytest.mark.parametrize("route", OTHER_GETS, ids=[f"{m} {p}" for m, p in OTHER_GETS])
def test_readonly_key_refused_on_every_other_get(client, route):
    method, path = route
    resp = _call(client, method, path, RO)
    assert resp.status_code == 403, f"{method} {path}: {resp.status_code} {resp.text[:200]}"
    assert resp.json()["detail"] == NOT_AUTHORIZED


@pytest.mark.db
def test_readonly_key_cannot_open_a_session(client):
    resp = client.post("/auth/session", json={"actor_key": READONLY_TEST_KEY})
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"]["error_code"] == "ACTOR_KEY_INVALID"


@pytest.mark.db
def test_readonly_key_refused_on_packing_slip_query_param(client):
    resp = client.get(f"/sales/orders/1/packing-slip?key={READONLY_TEST_KEY}")
    assert resp.status_code == 403 and resp.json()["detail"] == NOT_AUTHORIZED, resp.text


@pytest.mark.db
def test_shared_keys_unchanged_with_readonly_key_set(client):
    # Master: still everything (auth passes; empty write body -> 422).
    assert client.post("/make", json={}, headers=MASTER).status_code == 422
    assert client.get("/admin/lots/duplicates", headers=MASTER).status_code == 200
    # Dashboard: still its allowlist, still 403 outside it.
    assert client.get("/dashboard/inventory", headers=DASH).status_code == 200
    assert client.post("/receive/commit", json={}, headers=DASH).status_code == 422
    r = client.post("/make", json={}, headers=DASH)
    assert r.status_code == 403 and r.json()["detail"] == NOT_AUTHORIZED
    # Missing / unknown keys keep their codes.
    assert client.get("/customers").status_code == 401
    r = client.get("/customers", headers={"X-API-Key": "nope"})
    assert r.status_code == 403 and r.json()["detail"] == "Invalid API key"


@pytest.mark.db
def test_unset_readonly_key_is_just_an_unknown_key(client_unset):
    r = client_unset.get("/customers", headers=RO)
    assert r.status_code == 403 and r.json()["detail"] == "Invalid API key"
    # An empty X-API-Key never matches the empty unset value.
    r = client_unset.get("/customers", headers={"X-API-Key": ""})
    assert r.status_code == 401
    assert client_unset.get("/customers", headers=MASTER).status_code == 200
    assert client_unset.get("/customers", headers=DASH).status_code == 200

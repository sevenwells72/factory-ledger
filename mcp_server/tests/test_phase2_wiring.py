"""Phase 2 integration: one Authenticator in server.py and per-caller backend credentials.

Everything here runs through `create_app`, not a test-local mirror of the wiring. The
ledger is a recording fake that answers with the actor name bound to the key it saw,
so a request served with the wrong user's key is visible in the response itself.
"""

import asyncio
import random
from contextlib import asynccontextmanager

import httpx
import pytest

from factory_ledger_mcp import identity
from factory_ledger_mcp.adapter import LedgerReader, ToolFailure
from factory_ledger_mcp.config import Settings
from factory_ledger_mcp.server import create_app, transport_security

from .conftest import MCP_HEADERS_BASE, rpc
from .test_auth import (
    ADMIN,
    ADMIN_KEY,
    FLOOR,
    FLOOR_KEY,
    PUBLIC_URL,
    bearer,
    fake_google,  # noqa: F401  (fixture re-exported for this module)
    google_settings,
    register,
    sign_in,
)

SHARED_KEY = "synthetic-shared-master-key"  # what this service must never send
ACTOR_NAMES = {ADMIN_KEY: "Admin Actor", FLOOR_KEY: "Floor Actor"}
ORDER_IDS = {"SO-ADMIN": 11, "SO-FLOOR": 22}


def order_detail(number, key):
    return {
        "order_id": ORDER_IDS.get(number, 1),
        "order_number": number,
        "customer": f"Customer {number}",
        "status": "new",
        "fulfillment": "unshipped",
        "lines": [],
        "shipments": [],
        "served_for": ACTOR_NAMES[key],
    }


def fake_ledger(record, *, jitter=0.0):
    """Records (method, url, X-API-Key) and answers as the actor that key names."""

    async def handle(request):
        key = request.headers.get("x-api-key")
        record.append((request.method, str(request.url), key))
        if jitter:
            await asyncio.sleep(random.uniform(0, jitter))
        if key not in ACTOR_NAMES:
            return httpx.Response(401 if not key else 403, json={"detail": "Invalid API key"})
        path = request.url.path
        if path == "/auth/whoami":
            actor = {"name": ACTOR_NAMES[key], "role": "floor"}
            return httpx.Response(200, json={"key_kind": "actor", "actor": actor})
        if path.endswith("/allocations"):
            return httpx.Response(200, json={"allocations": []})
        if path == "/sales/orders":
            number = request.url.params.get("customer", "Customer SO-X").removeprefix("Customer ")
            row = {"order_number": number, "order_id": ORDER_IDS.get(number, 1)}
            row |= {"customer": f"Customer {number}", "customer_po": "PO-1"}
            return httpx.Response(200, json={"orders": [row]})
        if path.startswith("/sales/orders/"):
            return httpx.Response(200, json=order_detail(path.rsplit("/", 1)[1], key))
        return httpx.Response(200, json={"path": path, "served_for": ACTOR_NAMES[key]})

    return httpx.MockTransport(handle)


@pytest.fixture
def google_app(fake_google, tmp_path):  # noqa: F811
    """The real create_app in google mode with a fake Google and a recording ledger."""

    @asynccontextmanager
    async def open_app(settings=None, *, jitter=0.0, base_url=PUBLIC_URL):
        record = []
        settings = settings or google_settings()
        app = create_app(
            settings,
            backend_transport=fake_ledger(record, jitter=jitter),
            outbound_transport=httpx.ASGITransport(app=fake_google.app()),
            confirmation_path=tmp_path / "confirmations.sqlite3",
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=("203.0.113.9", 4321)),
                base_url=base_url,
                headers=MCP_HEADERS_BASE,
            ) as client:
                yield client, record, app

    return open_app


async def test_create_app_mounts_oauth_routes_and_protects_both_groups(google_app):
    async with google_app() as (client, record, app):
        health = (await client.get("/health")).json()
        assert health["auth"] == "google" and health["google_oauth_ready"] is True
        assert health["backend_credential"] == "per-user named-actor key"
        metadata = (await client.get("/.well-known/oauth-authorization-server")).json()
        assert metadata["token_endpoint"] == PUBLIC_URL + "/token"
        for group in ("office", "floor"):
            resource = await client.get(f"/.well-known/oauth-protected-resource/{group}/mcp")
            assert resource.json()["resource"] == f"{PUBLIC_URL}/{group}/mcp"
            denied = await client.post(f"/{group}/mcp", json={})
            assert denied.status_code == 401
            assert "resource_metadata=" in denied.headers["www-authenticate"]
        assert record == []

        info = await register(client)
        tokens = await sign_in(client, info["client_id"], ADMIN)
        client.headers.update(bearer(tokens["access_token"]))
        assert len((await rpc(client, "office", "tools/list"))["result"]["tools"]) == 30
        result = (
            await rpc(client, "office", "tools/call", {"name": "listCustomers", "arguments": {}})
        )["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["data"]["served_for"] == ACTOR_NAMES[ADMIN_KEY]
        assert record == [("GET", "http://127.0.0.1:8100/customers", ADMIN_KEY)]
        assert app.state.authenticator.google_oauth_ready
        with pytest.raises(PermissionError):
            identity.current_identity()


async def test_locked_mode_mounts_no_oauth_routes(tmp_path):
    app = create_app(
        Settings(environment="test", auth_mode="locked"),
        confirmation_path=tmp_path / "confirmations.sqlite3",
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            assert (await client.get("/health")).json()["google_oauth_ready"] is False
            assert (await client.get("/.well-known/oauth-authorization-server")).status_code == 404
            assert (await client.post("/register", json={})).status_code == 404
            for group in ("office", "floor"):
                assert (await client.post(f"/{group}/mcp", json={})).status_code == 401


def test_transport_allowlists_include_the_public_origin():
    hosted = google_settings(public_url="https://mcp.example.invalid")
    security = transport_security(hosted)
    assert "mcp.example.invalid" in security.allowed_hosts
    assert "https://mcp.example.invalid" in security.allowed_origins
    assert "127.0.0.1:*" in security.allowed_hosts  # loopback development still works
    local = transport_security(Settings(environment="test", auth_mode="locked"))
    assert local.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]


async def test_public_host_is_accepted_and_foreign_hosts_rejected_with_421(google_app):
    hosted = google_settings(public_url="https://mcp.example.invalid")
    async with google_app(hosted, base_url="https://mcp.example.invalid") as (client, record, app):
        # Mint a valid token directly; the sign-in flow itself is covered in test_auth.
        token = app.state.authenticator.provider.signer.sign(
            "at", {"sub": ADMIN, "cid": "cl.test", "scp": ["office.read"], "res": None}, ttl=300
        )
        client.headers.update(bearer(token))
        listed = await rpc(client, "office", "tools/list")
        assert len(listed["result"]["tools"]) == 30
        for host in ("evil.example.invalid", "mcp.example.invalid.evil.example"):
            response = await client.post(
                "/office/mcp",
                headers={"Host": host},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
            assert response.status_code == 421, (host, response.text)
        assert record == []
    # Without MCP_PUBLIC_URL naming that host, the same request is rejected: the allowlist
    # entry is what makes the Railway domain usable.
    async with google_app(base_url="https://mcp.example.invalid") as (client, record, app):
        token = app.state.authenticator.provider.signer.sign(
            "at", {"sub": ADMIN, "cid": "cl.test", "scp": ["office.read"], "res": None}, ttl=300
        )
        response = await client.post(
            "/office/mcp",
            headers=bearer(token),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert response.status_code == 421


async def test_shared_key_on_the_client_is_never_sent_reads_or_writes(tmp_path):
    record = []
    # Deliberately poisonous: a service-level shared key configured on the HTTP client.
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8100",
        headers={"X-API-Key": SHARED_KEY},
        transport=fake_ledger(record),
        trust_env=False,
    ) as client:
        from factory_ledger_mcp.adapter import ConfirmationStore

        reader = LedgerReader(client, ConfirmationStore(tmp_path / "c.sqlite3"))
        # No identity bound: no credential at all, so the backend itself denies the read.
        with pytest.raises(ToolFailure) as denied:
            await reader.call("office", "listCustomers", {})
        assert denied.value.payload["status"] == 401
        assert record[-1][2] is None
        # Identity bound: only that user's named-actor key, for reads...
        with identity.identity_scope(identity.Identity(FLOOR, "floor", FLOOR_KEY)):
            data = await reader.call("floor", "getOrder", {"order_id": "SO-FLOOR"})
            assert data["served_for"] == ACTOR_NAMES[FLOOR_KEY]
            # ...and for every request inside a write preview (whoami, order, list, allocations).
            preview = await reader.call(
                "floor", "updateOrderStatus", {"order_id": "SO-FLOOR", "status": "confirmed"}
            )
            assert preview["phase"] == "preview"
            assert preview["proposal"]["order_number"] == "SO-FLOOR"
        assert len(record) >= 6
        assert {key for _, _, key in record[1:]} == {FLOOR_KEY}
        assert SHARED_KEY not in {key for _, _, key in record}
        assert client.headers["X-API-Key"] == SHARED_KEY  # untouched, and still never used


def test_settings_have_no_shared_key_and_refuse_a_leftover_variable(monkeypatch):
    assert not hasattr(Settings(), "test_api_key")
    monkeypatch.setenv("MCP_TEST_API_KEY", "leftover-shared-key")
    with pytest.raises(ValueError, match="MCP_TEST_API_KEY is no longer supported"):
        Settings.from_env()


async def test_two_simultaneous_users_never_get_each_others_actor_key(google_app):
    async with google_app(jitter=0.02) as (client, record, _):
        info = await register(client)
        admin = (await sign_in(client, info["client_id"], ADMIN))["access_token"]
        floor = (await sign_in(client, info["client_id"], FLOOR))["access_token"]
        users = [(admin, ADMIN_KEY, "SO-ADMIN"), (floor, FLOOR_KEY, "SO-FLOOR")]
        del record[:]

        async def read(token, marker, index):
            params = {"name": "getOrder", "arguments": {"order_id": f"{marker}-{index}"}}
            response = await client.post(
                "/floor/mcp",
                headers=bearer(token),
                json={"jsonrpc": "2.0", "id": index, "method": "tools/call", "params": params},
            )
            assert response.status_code == 200, response.text
            return response.json()["result"]["structuredContent"]["data"]["served_for"]

        async def preview(token, marker, index):
            params = {
                "name": "updateOrderStatus",
                "arguments": {"order_id": marker, "status": "confirmed"},
            }
            response = await client.post(
                "/floor/mcp",
                headers=bearer(token),
                json={"jsonrpc": "2.0", "id": index, "method": "tools/call", "params": params},
            )
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            assert result["isError"] is False, result
            return result["structuredContent"]["data"]["proposal"]["order_number"]

        jobs, expected = [], []
        for index in range(12):
            token, key, marker = users[index % 2]
            jobs.append(read(token, marker, index))
            expected.append(ACTOR_NAMES[key])
        for index in range(12, 24):
            token, key, marker = users[index % 2]
            jobs.append(preview(token, marker, index))
            expected.append(marker)
        results = await asyncio.gather(*jobs)
        assert results == expected

        # Every backend request is attributable to exactly one user by its order marker
        # (path, numeric id or customer filter) and carried only that user's key.
        whoami = [key for method, url, key in record if url.endswith("/auth/whoami")]
        assert sorted(whoami) == sorted([ADMIN_KEY] * 6 + [FLOOR_KEY] * 6)
        for method, url, key in record:
            if url.endswith("/auth/whoami"):
                continue
            if "SO-ADMIN" in url or "/sales/orders/11/" in url:
                assert key == ADMIN_KEY, (method, url, key)
            elif "SO-FLOOR" in url or "/sales/orders/22/" in url:
                assert key == FLOOR_KEY, (method, url, key)
            else:
                raise AssertionError(f"unattributable request {method} {url}")
        assert {key for _, _, key in record} == {ADMIN_KEY, FLOOR_KEY}
        with pytest.raises(PermissionError):
            identity.current_identity()

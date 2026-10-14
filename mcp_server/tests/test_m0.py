"""M0 acceptance: real app startup, OAuth, refusal and credential isolation."""

import json
from dataclasses import replace

import httpx
import pytest

from factory_ledger_mcp import identity
from factory_ledger_mcp.adapter import WRITE_NAMES, LedgerReader, ToolFailure
from factory_ledger_mcp.auth import Authenticator
from factory_ledger_mcp.config import Settings
from factory_ledger_mcp.identity import AllowedUser, parse_allowlist
from factory_ledger_mcp.server import create_app

from .conftest import rpc
from .test_auth import (
    ADMIN,
    ADMIN_KEY,
    FLOOR,
    FLOOR_KEY,
    MCP_HEADERS,
    PUBLIC_URL,
    FakeGoogle,
    allowlist,
    bearer,
    google_settings,
    register,
    sign_in,
)


def whoami_handler(calls, overrides=None):
    actors = {
        ADMIN_KEY: {"id": 1, "name": "Synthetic admin", "role": "owner"},
        FLOOR_KEY: {"id": 2, "name": "Synthetic floor", "role": "floor"},
    }

    def handle(request):
        calls.append(request)
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        if request.url.path == "/auth/whoami":
            actor = actors.get(request.headers.get("x-api-key"))
            payload = {"key_kind": "actor", "actor": actor, "permissions": {}}
            return httpx.Response(200, json=payload if overrides is None else overrides)
        return httpx.Response(200, json={"products": [{"id": 17, "name": "From FL"}]})

    return handle


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "", "falsee"])
def test_writes_switch_cannot_enable_unshipped_writes(monkeypatch, value):
    monkeypatch.setenv("MCP_WRITES_ENABLED", value)
    with pytest.raises(ValueError, match="MCP_WRITES_ENABLED=false"):
        Settings.from_env()
    with pytest.raises(ValueError, match="MCP_WRITES_ENABLED=false"):
        Settings(writes_enabled=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"key_kind": "master", "actor": None},
        {"key_kind": "dashboard", "actor": None},
        {"key_kind": "readonly", "actor": None},
        {"key_kind": "actor", "actor": {"id": 2, "name": "Synthetic admin", "role": "owner"}},
        {"key_kind": "actor", "actor": {"id": 1, "name": "Other person", "role": "owner"}},
        {"key_kind": "actor", "actor": {"id": 1, "name": "Synthetic admin", "role": "floor"}},
        {"key_kind": "actor", "actor": None},
        [],
    ],
)
async def test_wrong_or_shared_actor_mapping_refuses_real_app_startup(bad):
    calls = []
    app = create_app(
        google_settings(), backend_transport=httpx.MockTransport(whoami_handler(calls, bad))
    )
    with pytest.raises(ValueError, match="expected actor") as error:
        async with app.router.lifespan_context(app):
            pytest.fail("wrong actor mapping started")
    assert ADMIN_KEY not in str(error.value) and FLOOR_KEY not in str(error.value)
    assert [r.url.path for r in calls] == ["/auth/whoami"]


@pytest.mark.parametrize("status", [401, 403, 404, 500])
async def test_inactive_invalid_or_unverifiable_actor_fails_closed(status):
    app = create_app(
        google_settings(),
        backend_transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"private": ADMIN_KEY})
        ),
    )
    with pytest.raises(ValueError, match="actor verification failed") as error:
        async with app.router.lifespan_context(app):
            pytest.fail("unverifiable actor started")
    assert ADMIN_KEY not in str(error.value)


async def test_swapped_personal_keys_are_detected_before_serving():
    users = allowlist()
    users[ADMIN] = replace(users[ADMIN], actor_key=FLOOR_KEY)
    users[FLOOR] = replace(users[FLOOR], actor_key=ADMIN_KEY)
    app = create_app(
        google_settings(allowed_users=users),
        backend_transport=httpx.MockTransport(whoami_handler([])),
    )
    with pytest.raises(ValueError, match="expected actor"):
        async with app.router.lifespan_context(app):
            pytest.fail("swapped keys started")


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor_id", None),
        ("actor_id", True),
        ("actor_id", 0),
        ("actor_name", ""),
        ("actor_role", "admin"),
        ("actor_role", "floor"),
    ],
)
def test_allowlist_requires_independent_expected_actor_pins(field, value):
    entry = {
        "email": ADMIN,
        "role": "admin_office_floor",
        "actor_key_env": "MCP_ACTOR_KEY_A",
        "actor_id": 1,
        "actor_name": "Synthetic admin",
        "actor_role": "owner",
    }
    entry[field] = value
    with pytest.raises(ValueError):
        parse_allowlist(json.dumps([entry]), {"MCP_ACTOR_KEY_A": ADMIN_KEY})


async def test_google_to_all_17_reads_and_write_refusal_over_real_app():
    calls = []
    google = FakeGoogle()
    app = create_app(
        google_settings(),
        backend_transport=httpx.MockTransport(whoami_handler(calls)),
        outbound_transport=httpx.ASGITransport(app=google.app()),
    )
    async with app.router.lifespan_context(app):
        assert [r.url.path for r in calls] == ["/auth/whoami", "/auth/whoami"]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=PUBLIC_URL, headers=MCP_HEADERS
        ) as client:
            denied = await client.post("/mcp", json={})
            assert denied.status_code == 401
            assert "/.well-known/oauth-protected-resource/mcp" in denied.headers["www-authenticate"]
            resource = (await client.get("/.well-known/oauth-protected-resource/mcp")).json()
            assert resource["scopes_supported"] == ["all.read"]
            registered = await register(client)
            for email, key in [(ADMIN, ADMIN_KEY), (FLOOR, FLOOR_KEY)]:
                tokens = await sign_in(
                    client,
                    registered["client_id"],
                    email,
                    resource=PUBLIC_URL + "/mcp",
                    scope="all.read",
                )
                client.headers.update(bearer(tokens["access_token"]))
                tools = (await rpc(client, "all", "tools/list"))["result"]["tools"]
                assert len(tools) == 17
                assert all(t["annotations"]["readOnlyHint"] for t in tools)
                assert all(
                    t["_meta"]["securitySchemes"] == [{"type": "oauth2", "scopes": ["all.read"]}]
                    for t in tools
                )
                for name in WRITE_NAMES:
                    before = len(calls)
                    result = (
                        await rpc(
                            client,
                            "all",
                            "tools/call",
                            {"name": name, "arguments": {"phase": "commit", "mode": "commit"}},
                        )
                    )["result"]
                    assert (
                        result["isError"]
                        and result["structuredContent"]["error"] == "write_blocked"
                    )
                    assert len(calls) == before
                result = (
                    await rpc(
                        client,
                        "all",
                        "tools/call",
                        {"name": "searchProducts", "arguments": {"q": "From FL"}},
                    )
                )["result"]
                assert result["structuredContent"]["data"] == {
                    "products": [{"id": 17, "name": "From FL"}]
                }
                assert calls[-1].headers["x-api-key"] == key
                assert result["isError"] is False


@pytest.mark.parametrize(
    "uri",
    [
        "https://evil.example/callback",
        "https://chatgpt.com.evil.example/connector/oauth/a",
        "http://chatgpt.com/connector/oauth/a",
        "https://chatgpt.com/other",
        "https://chatgpt.com/connector/oauth/a?redirect=https://evil.example",
    ],
)
async def test_dcr_refuses_non_chatgpt_callbacks(uri):
    from mcp.server.auth.provider import RegistrationError
    from mcp.shared.auth import OAuthClientInformationFull

    auth = Authenticator(google_settings())
    async with auth:
        with pytest.raises(RegistrationError):
            await auth.provider.register_client(
                OAuthClientInformationFull(redirect_uris=[uri], token_endpoint_auth_method="none")
            )


async def test_client_defaults_and_redirects_cannot_leak_credentials():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://evil.example/"})

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8100",
        follow_redirects=True,
        headers={"X-API-Key": "shared-default", "Authorization": "secret", "Cookie": "secret"},
        transport=httpx.MockTransport(handler),
    ) as client:
        with identity.identity_scope(AllowedUser(ADMIN, "floor", ADMIN_KEY).identity()):
            with pytest.raises(ToolFailure, match="redirects"):
                await LedgerReader(client).call("all", "searchProducts", {"q": "x"})
    assert len(requests) == 1
    assert requests[0].headers["x-api-key"] == ADMIN_KEY
    assert "authorization" not in requests[0].headers and "cookie" not in requests[0].headers

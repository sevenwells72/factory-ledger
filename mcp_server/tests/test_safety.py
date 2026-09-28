from pathlib import Path

import httpx
import pytest
import yaml

from factory_ledger_mcp.adapter import CATALOG, LedgerReader, ToolFailure
from factory_ledger_mcp.config import Settings

from .conftest import rpc

ROOT = Path(__file__).resolve().parents[2]


def write_cases():
    writes = []
    for group, source in [
        ("office", "openapi-gpt-v3.yaml"),
        ("floor", "gpt-configs/schemas/openapi-floor.yaml"),
    ]:
        doc = yaml.safe_load((ROOT / source).read_text())
        names = {item["name"] for item in CATALOG[group]}
        for methods in doc["paths"].values():
            for method, operation in methods.items():
                if method in {"post", "patch", "put", "delete"}:
                    if operation["operationId"] not in names:
                        writes.append((group, operation["operationId"]))
    assert len(writes) == 26
    return writes


@pytest.mark.parametrize("group,name", write_cases())
async def test_every_write_rejected_even_for_admin(stack, group, name):
    async with stack(role="admin") as (client, backend):
        result = (
            await rpc(
                client,
                group,
                "tools/call",
                {
                    "name": name,
                    "arguments": {"mode": "commit"},
                },
            )
        )["result"]
        assert result["isError"] is True
        assert backend.state.requests == []


@pytest.mark.parametrize(
    "name,args",
    [
        ("shipOrder", {"order_id": "1", "mode": "commit"}),
        ("shipOrder", {"order_id": "1", "mode": "preview", "phase": "commit"}),
        (
            "shipOrder",
            {"order_id": "1", "mode": "preview", "lines": [{"line_id": 1, "quantity_lb": -1}]},
        ),
        ("inventoryLookup", {"q": "Demo", "limit": 1000}),
        ("getDaySummary", {"date": "2026-02-30"}),
        ("getOrder", {"order_id": "../ship/commit"}),
        ("getOrder", {"order_id": "%2e%2e%2fship"}),
        ("getOrder", {"order_id": ".."}),
        ("getOrder", {"order_id": "1?mode=commit"}),
        ("getOrder", {"order_id": "1", "actor": "Michael Gross"}),
    ],
)
async def test_invalid_or_dangerous_arguments_never_reach_backend(stack, name, args):
    async with stack() as (client, backend):
        result = (await rpc(client, "floor", "tools/call", {"name": name, "arguments": args}))[
            "result"
        ]
        assert result["isError"]
        assert backend.state.requests == []


@pytest.mark.parametrize("header", ["", "Bearer wrong-token"])
async def test_missing_bad_tokens_cannot_list_or_call(stack, header):
    async with stack() as (client, backend):
        for group in CATALOG:
            response = await client.post(
                f"/{group}/mcp", headers={"Authorization": header}, json={}
            )
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
        assert backend.state.requests == []


@pytest.mark.parametrize("kwargs", [{"auth_mode": "locked"}, {"peer": "192.0.2.1"}])
async def test_locked_mode_and_nonlocal_peers_blocked(stack, kwargs):
    async with stack(**kwargs) as (client, backend):
        assert (await client.post("/office/mcp", json={})).status_code == 401
        assert backend.state.requests == []


async def test_floor_role_can_use_floor_but_not_office(stack):
    async with stack(role="floor") as (client, backend):
        response = await client.post("/office/mcp", json={})
        assert response.status_code == 403
        tools = (await rpc(client, "floor", "tools/list"))["result"]["tools"]
        assert len(tools) == 12
        # A name from the office-only catalog cannot be called via the floor endpoint.
        result = (
            await rpc(
                client,
                "floor",
                "tools/call",
                {
                    "name": "listCustomers",
                    "arguments": {},
                },
            )
        )["result"]
        assert result["isError"]
        assert backend.state.requests == []


async def test_health_is_available_when_locked(stack):
    async with stack(auth_mode="locked") as (client, _):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["google_oauth_ready"] is False


@pytest.mark.parametrize(
    "url",
    [
        "https://production.example.com",
        "http://production.example.com",
        "http://localhost:8100",
        "http://127.0.0.1.example.com",
        "http://127.0.0.1@evil.example",
        "http://127.0.0.1:8100/proxy",
        "http://127.0.0.1:8100?target=production",
    ],
)
def test_remote_or_ambiguous_api_urls_rejected(url):
    with pytest.raises(ValueError, match="loopback"):
        Settings(ledger_url=url)


def test_oauth_stub_cannot_run_in_production_or_railway(monkeypatch):
    with pytest.raises(ValueError, match="production"):
        Settings(environment="production", auth_mode="local_stub", dev_token="test-only-token-123")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "test-platform-indicator")
    with pytest.raises(ValueError, match="Railway"):
        Settings(auth_mode="local_stub", dev_token="test-only-token-123")


@pytest.mark.parametrize(
    "response,code",
    [
        (
            httpx.Response(302, headers={"location": "https://production.example.com"}),
            "redirect_blocked",
        ),
        (httpx.Response(503, text="private backend internals"), "upstream_failure"),
        (httpx.Response(200, text="<html>not JSON</html>"), "invalid_response"),
        (httpx.Response(200, json={"success": False, "error": "not found"}), "ledger_error"),
        (httpx.Response(200, content=b"x" * 2_000_001), "response_too_large"),
    ],
)
async def test_upstream_errors_are_not_successes_or_silent_truncation(response, code):
    calls = []

    def handler(request):
        calls.append(request)
        return response

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8100",
        follow_redirects=False,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(ToolFailure) as error:
            await LedgerReader(client).call("office", "searchProducts", {"q": "Demo"})
        assert error.value.payload["error"] == code
        assert "private backend internals" not in str(error.value.payload)
        assert len(calls) == 1


async def test_timeout_does_not_retry():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("timeout")

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8100", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ToolFailure) as error:
            await LedgerReader(client).call("office", "searchProducts", {"q": "Demo"})
        assert error.value.payload["error"] == "timeout"
        assert len(calls) == 1


async def test_test_backend_key_is_distinct_from_client_identity(stack):
    async with stack(test_api_key="synthetic-test-api-key") as (client, backend):
        result = await rpc(
            client,
            "office",
            "tools/call",
            {
                "name": "searchProducts",
                "arguments": {"q": "Demo"},
            },
        )
        assert not result["result"]["isError"]
        assert backend.state.requests[0]["headers"]["x-api-key"] == "synthetic-test-api-key"
        assert "authorization" not in backend.state.requests[0]["headers"]


async def test_floor_lot_disambiguation_and_boolean_query(stack):
    async with stack() as (client, backend):
        await rpc(
            client,
            "floor",
            "tools/call",
            {
                "name": "getLotByCode",
                "arguments": {"lot_code": "DEMO-LOT", "product_id": 1},
            },
        )
        assert backend.state.requests[-1]["params"] == {"product_id": "1"}
        await rpc(
            client,
            "office",
            "tools/call",
            {
                "name": "listOrders",
                "arguments": {"overdue_only": False, "status": "open"},
            },
        )
        assert backend.state.requests[-1]["params"] == {"overdue_only": "false", "status": "open"}

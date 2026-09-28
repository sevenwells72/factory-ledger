from contextlib import asynccontextmanager

import httpx
import pytest

from factory_ledger_mcp.config import Settings
from factory_ledger_mcp.demo import create_demo_app
from factory_ledger_mcp.server import create_app

TOKEN = "test-only-local-token"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}


@pytest.fixture
def stack():
    @asynccontextmanager
    async def open_stack(role="reader", auth_mode="local_stub", peer="127.0.0.1", test_api_key=""):
        backend = create_demo_app()
        settings = Settings(
            environment="test",
            auth_mode=auth_mode,
            dev_token=TOKEN,
            dev_role=role,
            test_api_key=test_api_key,
        )
        app = create_app(settings, backend_transport=httpx.ASGITransport(app=backend))
        async with backend.router.lifespan_context(backend), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=(peer, 12345)),
                base_url="http://127.0.0.1:8000",
                headers=HEADERS,
            ) as client:
                yield client, backend

    return open_stack


async def rpc(client, group, method, params=None):
    response = await client.post(
        f"/{group}/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()

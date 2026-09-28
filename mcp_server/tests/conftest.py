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


@pytest.fixture(scope="session")
def postgres():
    from .ledger_harness import postgres_cluster

    with postgres_cluster() as admin:
        yield admin


@pytest.fixture
def ledger(postgres, tmp_path):
    """Fresh database + real main.py/auth for each test, safe across parallel worktrees."""
    import uuid

    from .ledger_harness import LedgerDatabase, ledger_process, seed_database

    name = "mcp_test_" + uuid.uuid4().hex
    postgres.sql(f'CREATE DATABASE "{name}" TEMPLATE mcp_template;')
    db = LedgerDatabase(postgres.pg_bin, postgres.socket_dir, name)
    try:
        with ledger_process(db, tmp_path / "ledger.log") as url:
            seed_database(db)
            yield db, url
    finally:
        postgres.sql(f'DROP DATABASE "{name}" WITH (FORCE);')


@pytest.fixture
def real_stack(ledger):
    from .ledger_harness import ACTOR_KEY

    db, url = ledger

    @asynccontextmanager
    async def open_stack(test_api_key=ACTOR_KEY):
        settings = Settings(
            environment="test",
            auth_mode="local_stub",
            dev_token=TOKEN,
            test_api_key=test_api_key,
            ledger_url=url,
        )
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers=HEADERS,
            ) as client:
                yield client, db

    return open_stack

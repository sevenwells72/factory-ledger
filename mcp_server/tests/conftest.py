from contextlib import asynccontextmanager

import httpx
import pytest

from factory_ledger_mcp.config import Settings
from factory_ledger_mcp.demo import create_demo_app
from factory_ledger_mcp.identity import AllowedUser
from factory_ledger_mcp.server import create_app

TOKEN = "test-only-local-token"
MCP_HEADERS_BASE = {
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}
HEADERS = {"Authorization": f"Bearer {TOKEN}", **MCP_HEADERS_BASE}
# Synthetic allowlisted users for the loopback stub: local_stub + MCP_DEV_EMAIL binds one of
# them for the whole request, exactly as a Google sign-in would. Keys are synthetic.
STUB_ADMIN = "admin.stub@example.invalid"
STUB_FLOOR = "floor.stub@example.invalid"
STUB_ADMIN_KEY = "stub-actor-key-admin"
STUB_FLOOR_KEY = "stub-actor-key-floor"


def stub_users(admin_key=STUB_ADMIN_KEY, floor_key=STUB_FLOOR_KEY):
    """Direct AllowedUser mapping; bypasses env parsing so tests may bind empty/wrong keys."""
    return {
        STUB_ADMIN: AllowedUser(STUB_ADMIN, "admin_office_floor", admin_key),
        STUB_FLOOR: AllowedUser(STUB_FLOOR, "floor", floor_key),
    }


@pytest.fixture
def stack(tmp_path):
    @asynccontextmanager
    async def open_stack(
        role="reader", auth_mode="local_stub", peer="127.0.0.1", dev_email="", allowed_users=None
    ):
        backend = create_demo_app()
        if allowed_users is None:
            allowed_users = stub_users() if dev_email else {}
        settings = Settings(
            environment="test",
            auth_mode=auth_mode,
            dev_token=TOKEN,
            dev_role=role,
            dev_email=dev_email,
            allowed_users=allowed_users,
        )
        app = create_app(
            settings,
            backend_transport=httpx.ASGITransport(app=backend),
            confirmation_path=tmp_path / "confirmations.sqlite3",
        )
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
def real_stack(ledger, tmp_path):
    """The full request path against the real ledger: stub sign-in binds one identity whose
    named-actor key is `actor_key`; the adapter sends that key on every ledger call."""
    from .ledger_harness import ACTOR_KEY

    db, url = ledger

    @asynccontextmanager
    async def open_stack(actor_key=ACTOR_KEY, email=STUB_ADMIN, role="admin_office_floor"):
        settings = Settings(
            environment="test",
            auth_mode="local_stub",
            dev_token=TOKEN,
            ledger_url=url,
            dev_email=email,
            allowed_users={email: AllowedUser(email, role, actor_key)},
        )
        app = create_app(settings, confirmation_path=tmp_path / "confirmations.sqlite3")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers=HEADERS,
            ) as client:
                yield client, db

    return open_stack

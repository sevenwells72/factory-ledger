"""One service, two independent Streamable HTTP tool catalogs."""

import json
import os
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import identity
from .adapter import CATALOG, WRITE_CATALOG, ConfirmationStore, LedgerReader, ToolFailure
from .auth import DevelopmentAuth
from .config import Settings


def group_server(group, reader):
    server = Server(
        f"Factory Ledger — {group}",
        version="0.1.0",
        instructions=(
            "Report ledger results faithfully. Every write requires two calls: preview, then "
            "explicit operator approval of the summary and warnings, then commit with the "
            "unchanged arguments and one-time token. Never auto-approve, retry writes, or claim "
            "a blocked or uncertain write succeeded. Always report the internal SO for orders."
        ),
    )

    @server.list_tools()
    async def list_tools():
        try:
            person = identity.current_identity()
        except PermissionError:
            person = None
        permission = identity.can_write_office if group == "office" else identity.can_write_floor
        entries = [(item, True) for item in CATALOG[group]]
        if permission(person):
            entries.extend((item, False) for item in WRITE_CATALOG[group])
        return [
            types.Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["input_schema"],
                annotations=types.ToolAnnotations(
                    readOnlyHint=read_only,
                    destructiveHint=not read_only
                    and item["name"]
                    not in {"createCustomer", "createOrder", "createExpectedReceipt"},
                    idempotentHint=read_only,
                    openWorldHint=False,
                ),
            )
            for item, read_only in entries
        ]

    async def call_tool(name, arguments):
        try:
            data = await reader.call(group, name, arguments)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
                structuredContent={"data": data},
                isError=False,
            )
        except ToolFailure as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(exc.payload))],
                structuredContent=exc.payload,
                isError=True,
            )

    # The SDK call_tool decorator fills its cache by calling list_tools on a miss.
    # Register directly so a write resolves identity exactly once, and validation
    # always uses the adapter's sanitized errors instead of echoing tool inputs.
    async def call_tool_request(request):
        return types.ServerResult(
            await call_tool(request.params.name, request.params.arguments or {})
        )

    server.request_handlers[types.CallToolRequest] = call_tool_request
    return server


def create_app(
    settings=None, *, backend_transport=None, confirmation_path="work/mcp-confirmations.sqlite3"
):
    settings = settings or Settings.from_env()
    client = httpx.AsyncClient(
        base_url=settings.ledger_url,
        timeout=httpx.Timeout(15.0, connect=3.0),
        follow_redirects=False,
        trust_env=False,
        transport=backend_transport,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={"X-API-Key": settings.test_api_key} if settings.test_api_key else {},
    )
    reader = LedgerReader(
        client, ConfirmationStore(confirmation_path), environment=settings.environment
    )
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
    )
    managers = {
        group: StreamableHTTPSessionManager(
            app=group_server(group, reader),
            json_response=True,
            stateless=True,
            security_settings=security,
        )
        for group in CATALOG
    }

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(client)
            for manager in managers.values():
                await stack.enter_async_context(manager.run())
            yield

    async def health(request):
        return JSONResponse(
            {
                "status": "ok",
                "phase": 2,
                "write_confirmation": "two-call-token",
                "backend_idempotency_ready": False,
                "auth": settings.auth_mode,
                "google_oauth_ready": False,
                "read_tool_counts": {group: len(specs) for group, specs in CATALOG.items()},
                "write_tool_counts": {group: len(specs) for group, specs in WRITE_CATALOG.items()},
            }
        )

    routes = [Route("/health", health)]
    for group, manager in managers.items():
        routes.append(
            Route(
                f"/{group}/mcp",
                DevelopmentAuth(manager.handle_request, settings, group),
                methods=["GET", "POST", "DELETE"],
            )
        )
    return Starlette(routes=routes, lifespan=lifespan)


def main():
    settings = Settings.from_env()
    # A Railway container may boot in locked mode for health checks; stub access stays loopback.
    host = "127.0.0.1" if settings.auth_mode == "local_stub" else "0.0.0.0"
    uvicorn.run(create_app(settings), host=host, port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()

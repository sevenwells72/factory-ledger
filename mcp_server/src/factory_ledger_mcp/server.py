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

from .adapter import CATALOG, LedgerReader, ToolFailure
from .auth import DevelopmentAuth
from .config import Settings


def group_server(group, reader):
    server = Server(
        f"Factory Ledger — {group}",
        version="0.1.0",
        instructions="Phase 1: read-only. Report API results faithfully. No changes available.",
    )

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["input_schema"],
                annotations=types.ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
            for item in CATALOG[group]
        ]

    @server.call_tool()
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

    return server


def create_app(settings=None, *, backend_transport=None):
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
    reader = LedgerReader(client)
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
                "phase": 1,
                "read_only": True,
                "auth": settings.auth_mode,
                "google_oauth_ready": False,
                "tool_counts": {group: len(specs) for group, specs in CATALOG.items()},
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

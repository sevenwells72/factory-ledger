"""One service, two independent Streamable HTTP tool catalogs behind one authenticator.

Phase 2 wiring: a single `Authenticator` owns the OAuth provider (google mode) and
gates both `/office/mcp` and `/floor/mcp`. Its discovery, authorization, token,
registration, revocation and Google callback routes are mounted beside the MCP
routes. Every ledger request carries the calling user's own named-actor key, which
the adapter reads from the request-scoped identity; the service holds no shared key.
"""

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
from .auth import Authenticator
from .config import Settings

LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


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


def transport_security(settings):
    """DNS-rebinding protection: loopback for development plus the public origin, if any.

    Without the public host the SDK answers 421 to every request that arrives through
    the Railway domain, even after a successful sign-in.
    """
    hosts, origins = list(LOOPBACK_HOSTS), list(LOOPBACK_ORIGINS)
    if settings.public_host:
        hosts += [settings.public_host, f"{settings.public_host}:*"]
        origins.append(settings.public_url)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


def create_app(
    settings=None,
    *,
    backend_transport=None,
    outbound_transport=None,
    confirmation_path="work/mcp-confirmations.sqlite3",
):
    settings = settings or Settings.from_env()
    # No default credential: the adapter sets X-API-Key per request from the caller's identity.
    client = httpx.AsyncClient(
        base_url=settings.ledger_url,
        timeout=httpx.Timeout(15.0, connect=3.0),
        follow_redirects=False,
        trust_env=False,
        transport=backend_transport,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )
    reader = LedgerReader(
        client, ConfirmationStore(confirmation_path), environment=settings.environment
    )
    authenticator = Authenticator(settings, outbound_transport=outbound_transport)
    security = transport_security(settings)
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
            await stack.enter_async_context(authenticator)
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
                "google_oauth_ready": authenticator.google_oauth_ready,
                "backend_credential": "per-user named-actor key",
                "read_tool_counts": {group: len(specs) for group, specs in CATALOG.items()},
                "write_tool_counts": {group: len(specs) for group, specs in WRITE_CATALOG.items()},
            }
        )

    routes = [Route("/health", health)]
    for group, manager in managers.items():
        routes.append(
            Route(
                f"/{group}/mcp",
                authenticator.protect(manager.handle_request, group),
                methods=["GET", "POST", "DELETE"],
            )
        )
    routes += authenticator.routes()
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.authenticator = authenticator
    return app


def main():
    settings = Settings.from_env()
    # A Railway container may boot in locked mode for health checks; stub access stays loopback.
    host = "127.0.0.1" if settings.auth_mode == "local_stub" else "0.0.0.0"
    uvicorn.run(create_app(settings), host=host, port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()

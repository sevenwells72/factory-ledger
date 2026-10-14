"""M0: authenticated read-only Streamable HTTP over the real staging FL API."""

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
from .adapter import CATALOG, LedgerReader, ToolFailure
from .auth import Authenticator
from .config import Settings

LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def group_server(group, reader, settings=None):
    server = Server(
        f"Factory Ledger — {group}",
        version="0.1.0",
        instructions=(
            "Factory Ledger STAGING, read-only. Report FL results faithfully; never invent "
            "quantities, order status or business decisions. No write tools are available. "
            "shipOrder only previews a shipment and cannot dispatch or save it. For entries, "
            "use the FL Assistant or dashboard. Always identify this data as staging."
        ),
    )

    @server.list_tools()
    async def list_tools():
        entries = CATALOG[group]
        return [
            types.Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["input_schema"],
                **(
                    {
                        "_meta": {
                            "securitySchemes": [{"type": "oauth2", "scopes": [f"{group}.read"]}]
                        }
                    }
                    if settings and settings.auth_mode == "google"
                    else {}
                ),
                annotations=types.ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
            for item in entries
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
    reader = LedgerReader(client)
    authenticator = Authenticator(settings, outbound_transport=outbound_transport)
    security = transport_security(settings)
    managers = {
        group: StreamableHTTPSessionManager(
            app=group_server(group, reader, settings),
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
            await identity.validate_actor_mappings(settings.allowed_users, reader)
            app.state.actor_mappings_verified = bool(settings.allowed_users)
            for manager in managers.values():
                await stack.enter_async_context(manager.run())
            yield

    async def health(request):
        return JSONResponse(
            {
                "status": "ok",
                "milestone": "M0",
                "environment": settings.environment,
                "writes_enabled": False,
                "actor_mappings_verified": getattr(
                    request.app.state, "actor_mappings_verified", False
                ),
                "auth": settings.auth_mode,
                "google_oauth_ready": authenticator.google_oauth_ready,
                "backend_credential": "per-user named-actor key",
                "read_tool_counts": {group: len(specs) for group, specs in CATALOG.items()},
                "write_tool_counts": {group: 0 for group in CATALOG},
            }
        )

    routes = [Route("/health", health)]
    for group, manager in managers.items():
        routes.append(
            Route(
                "/mcp" if group == "all" else f"/{group}/mcp",
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
    # OAuth codes arrive in query strings; never put them in access logs.
    uvicorn.run(
        create_app(settings), host=host, port=int(os.getenv("PORT", "8000")), access_log=False
    )


if __name__ == "__main__":
    main()

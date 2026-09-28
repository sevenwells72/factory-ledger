"""Local-only stand-in for Google OAuth. Never authenticates a real CNS user.

TODO before hosted access: Google sign-in through an MCP-compatible OAuth 2.1
authorization server; discovery, PKCE, refresh/revocation, issuer/audience/expiry
verification, verified CNS domain and membership, and per-user group scopes.
Map Michael Gross, Luz and Miriam by verified Google subject, never display name.
Reject outside-CNS identities. Other CNS members are readers. Floor users receive
floor access and no office access until the owner resolves the read-only option.
This stub exposes no writes regardless of the simulated role.
"""

import hmac

from starlette.responses import JSONResponse

from .config import Settings


class DevelopmentAuth:
    def __init__(self, app, settings: Settings, group: str):
        self.app, self.settings, self.group = app, settings, group

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        settings = self.settings
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"authorization", b"")
        peer = (scope.get("client") or ("",))[0]
        valid = (
            settings.auth_mode == "local_stub"
            and peer in {"127.0.0.1", "::1"}
            and hmac.compare_digest(supplied, f"Bearer {settings.dev_token}".encode())
        )
        if not valid:
            await JSONResponse(
                {"error": "Authentication required; Google OAuth is not implemented in Phase 1."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
            return
        if self.group == "office" and settings.dev_role == "floor":
            await JSONResponse({"error": "Office access denied for floor role"}, status_code=403)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)

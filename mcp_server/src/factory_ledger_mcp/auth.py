"""Google sign-in through the standard MCP OAuth 2.1 authorization flow.

The service is its own OAuth authorization server for ChatGPT plugins and Claude
custom connectors (discovery, PKCE S256, dynamic client registration and client ID
metadata documents, refresh and revocation). Google is only the identity source:

    client --/authorize--> this service --302--> Google sign-in
    Google --/oauth/google/callback--> this service --302 code--> client redirect_uri
    client --/token (PKCE)--> this service ==> access + refresh tokens

The Google ID token is verified against Google's JWKS (issuer, audience, expiry,
nonce, verified email). The verified email is then looked up in the env-managed
allowlist; unlisted accounts are denied even after a successful Google sign-in.
Tokens, authorization codes and registered client IDs are HMAC-signed with
MCP_TOKEN_SECRET, so they survive restarts without a database; single-use codes,
pending logins and revocations are bounded in-memory state. Roles and actor keys
are re-resolved from the allowlist on every request, so removing a user takes
effect immediately. The backend actor key is never placed in a token or a log.

Integration (server.py): build one `Authenticator(settings)` per app, mount
`authenticator.routes()` beside the MCP routes, wrap each group endpoint with
`authenticator.protect(handler, group)`, enter it as an async context manager in
the lifespan and add `settings.public_host` to the transport host allowlist.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit

import httpx
import jwt
from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.handlers.metadata import MetadataHandler
from mcp.server.auth.handlers.register import RegistrationHandler
from mcp.server.auth.handlers.revoke import RevocationHandler
from mcp.server.auth.handlers.token import TokenHandler
from mcp.server.auth.middleware.client_auth import ClientAuthenticator
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import build_metadata, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.server.streamable_http import MCP_PROTOCOL_VERSION_HEADER
from mcp.server.transport_security import DEFAULT_MAX_REQUEST_BODY_SIZE, RequestBodyLimitMiddleware
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, AnyUrl, ValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route, request_response

from .config import Settings
from .identity import (
    ALL_SCOPES,
    AllowedUser,
    Identity,
    can_read_group,
    identity_scope,
    resolve_allowed_user,
    scopes_for,
)

logger = logging.getLogger("factory_ledger_mcp.auth")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
CALLBACK_PATH = "/oauth/google/callback"
AUTHORIZATION_PATH = "/authorize"
TOKEN_PATH = "/token"
REGISTRATION_PATH = "/register"
REVOCATION_PATH = "/revoke"
AUTH_CODE_TTL = 300
PENDING_LOGIN_TTL = 600
MAX_PENDING_LOGINS = 500
MAX_TRACKED_TOKENS = 10_000
JWKS_TTL = 6 * 3600
JWKS_MIN_REFRESH_INTERVAL = 60
MAX_METADATA_BYTES = 64_000
CLIENT_ID_PREFIX = "cl."
TOKEN_AUTH_METHODS = ["none", "client_secret_post", "client_secret_basic"]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenSigner:
    """Compact HMAC-SHA256 tokens: `<kind>.<payload>.<mac>`; kind is domain separation."""

    def __init__(self, secret: str):
        self._key = hashlib.sha256(secret.encode()).digest()

    def sign(self, kind: str, payload: dict[str, Any], ttl: int | None) -> str:
        now = int(time.time())
        body = dict(payload, k=kind, iat=now, jti=secrets.token_urlsafe(24))
        if ttl is not None:
            body["exp"] = now + ttl
        encoded = _b64(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
        mac = hmac.new(self._key, f"{kind}.{encoded}".encode(), hashlib.sha256).digest()
        return f"{kind}.{encoded}.{_b64(mac)}"

    def verify(self, kind: str, token: str | None) -> dict[str, Any] | None:
        if not token or token.count(".") != 2 or len(token) > 8192:
            return None
        got_kind, encoded, mac = token.split(".")
        if got_kind != kind:
            return None
        expected = hmac.new(self._key, f"{kind}.{encoded}".encode(), hashlib.sha256).digest()
        try:
            if not hmac.compare_digest(_unb64(mac), expected):
                return None
            body = json.loads(_unb64(encoded))
        except (ValueError, TypeError):
            return None
        if not isinstance(body, dict) or body.get("k") != kind or "jti" not in body:
            return None
        if "exp" in body and body["exp"] < time.time():
            return None
        return body


class SignedAccessToken(AccessToken):
    jti: str


class SignedRefreshToken(RefreshToken):
    jti: str


@dataclass
class PendingLogin:
    client_id: str
    params: AuthorizationParams
    code_verifier: str
    nonce: str
    expires_at: float


@dataclass(frozen=True)
class BearerResult:
    status: Literal["ok", "invalid", "forbidden"]
    identity: Identity | None = None
    token: SignedAccessToken | None = None


class _Expiring(dict[str, float]):
    """Bounded jti -> expiry map; expired entries are pruned on insert."""

    def add(self, key: str, expires_at: float) -> None:
        now = time.time()
        if len(self) >= MAX_TRACKED_TOKENS:
            for stale in [k for k, exp in self.items() if exp < now]:
                del self[stale]
            while len(self) >= MAX_TRACKED_TOKENS:
                del self[next(iter(self))]
        self[key] = expires_at

    def active(self, key: str) -> bool:
        exp = self.get(key)
        return exp is not None and exp >= time.time()


class GoogleOAuthProvider:
    """`OAuthAuthorizationServerProvider` backed by Google sign-in and the allowlist."""

    def __init__(
        self, settings: Settings, *, outbound_transport: httpx.AsyncBaseTransport | None = None
    ):
        if settings.auth_mode != "google":
            raise ValueError("GoogleOAuthProvider requires MCP_AUTH_MODE=google")
        self.settings = settings
        self.signer = TokenSigner(settings.token_secret)
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            trust_env=False,
            follow_redirects=False,
            transport=outbound_transport,
            headers={"Accept": "application/json"},
        )
        self.pending: dict[str, PendingLogin] = {}
        self.consumed_codes = _Expiring()
        self.revoked = _Expiring()
        self._jwks: dict[str, Any] = {}
        self._jwks_fetched_at = 0.0
        self._cimd_cache: dict[str, tuple[float, OAuthClientInformationFull]] = {}

    async def aclose(self) -> None:
        await self.http.aclose()

    # ----- clients: signed self-contained registrations and CIMD documents -----

    def _client_secret_for(self, client_id: str) -> str:
        key = hashlib.sha256(("client-secret:" + self.settings.token_secret).encode()).digest()
        return hmac.new(key, client_id.encode(), hashlib.sha256).hexdigest()

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        if not client_id:
            return None
        if client_id.startswith(CLIENT_ID_PREFIX):
            body = self.signer.verify("cl", client_id[len(CLIENT_ID_PREFIX) :])
            if body is None:
                return None
            try:
                client = OAuthClientInformationFull.model_validate(body["c"])
            except (KeyError, ValidationError):
                return None
            client.client_id = client_id
            if client.token_endpoint_auth_method != "none":
                client.client_secret = self._client_secret_for(client_id)
            return client
        if client_id.startswith("https://"):
            return await self._client_from_metadata_document(client_id)
        return None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        method = client_info.token_endpoint_auth_method or "client_secret_post"
        if method not in TOKEN_AUTH_METHODS:
            raise RegistrationError(
                "invalid_client_metadata", "Unsupported token endpoint auth method"
            )
        registration = {
            "redirect_uris": [str(uri) for uri in client_info.redirect_uris or []],
            "token_endpoint_auth_method": method,
            "grant_types": client_info.grant_types,
            "response_types": client_info.response_types,
            "scope": client_info.scope or " ".join(ALL_SCOPES),
            "client_name": (client_info.client_name or "")[:120] or None,
        }
        client_id = CLIENT_ID_PREFIX + self.signer.sign("cl", {"c": registration}, ttl=None)
        client_info.client_id = client_id
        client_info.scope = registration["scope"]
        client_info.client_secret = self._client_secret_for(client_id) if method != "none" else None
        client_info.client_secret_expires_at = None

    async def _client_from_metadata_document(self, url: str) -> OAuthClientInformationFull | None:
        parts = urlsplit(url)
        if not parts.hostname or parts.fragment or parts.username or parts.path in {"", "/"}:
            return None
        cached = self._cimd_cache.get(url)
        if cached and cached[0] > time.time():
            return cached[1]
        try:
            async with self.http.stream("GET", url) as response:
                if response.status_code != 200:
                    return None
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_METADATA_BYTES:
                        return None
            document = json.loads(data)
        except (httpx.HTTPError, ValueError):
            logger.info("client metadata document unavailable for %s", parts.hostname)
            return None
        if not isinstance(document, dict) or document.get("client_id") != url:
            return None
        if document.get("token_endpoint_auth_method") not in (None, "none"):
            return None
        try:
            client = OAuthClientInformationFull.model_validate(
                {
                    **{
                        k: v for k, v in document.items() if k not in {"client_id", "client_secret"}
                    },
                    "token_endpoint_auth_method": "none",
                    "scope": document.get("scope") or " ".join(ALL_SCOPES),
                }
            )
        except ValidationError:
            return None
        client.client_id = url
        if len(self._cimd_cache) >= 100:
            self._cimd_cache.clear()
        self._cimd_cache[url] = (time.time() + 3600, client)
        return client

    # ----- authorization: hand off to Google, then mint a bound code -----

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if params.resource and not self._resource_is_ours(params.resource):
            raise AuthorizeError("invalid_request", "resource is not served by this MCP service")
        now = time.time()
        for state in [s for s, p in self.pending.items() if p.expires_at < now]:
            del self.pending[state]
        while len(self.pending) >= MAX_PENDING_LOGINS:
            del self.pending[next(iter(self.pending))]
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        nonce = secrets.token_urlsafe(24)
        self.pending[state] = PendingLogin(
            client_id=client.client_id or "",
            params=params,
            code_verifier=code_verifier,
            nonce=nonce,
            expires_at=now + PENDING_LOGIN_TTL,
        )
        query = {
            "client_id": self.settings.google_client_id,
            "redirect_uri": self.settings.public_url + CALLBACK_PATH,
            "response_type": "code",
            "scope": "openid email",
            "state": state,
            "nonce": nonce,
            "code_challenge": _b64(hashlib.sha256(code_verifier.encode()).digest()),
            "code_challenge_method": "S256",
            "prompt": "select_account",
            "access_type": "online",
        }
        if self.settings.google_hosted_domain:
            query["hd"] = self.settings.google_hosted_domain
        return f"{GOOGLE_AUTH_URL}?{urlencode(query)}"

    async def handle_google_callback(self, request: Request) -> Response:
        query = request.query_params
        pending = self.pending.pop(query.get("state", ""), None)
        if pending is None or pending.expires_at < time.time():
            return JSONResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Sign-in expired; retry from the client",
                },
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )

        def back_to_client(error: str, description: str) -> Response:
            logger.info("google sign-in %s for client %s", error, pending.client_id[:24])
            return RedirectResponse(
                construct_redirect_uri(
                    str(pending.params.redirect_uri),
                    error=error,
                    error_description=description,
                    state=pending.params.state,
                ),
                status_code=302,
                headers={"Cache-Control": "no-store"},
            )

        if query.get("error") or not query.get("code"):
            return back_to_client("access_denied", "Google sign-in was cancelled or failed")
        claims = await self._verified_google_claims(query["code"], pending)
        if claims is None:
            return back_to_client("access_denied", "Google identity could not be verified")
        user = resolve_allowed_user(self.settings.allowed_users, claims.get("email"))
        if user is None:
            logger.warning(
                "sign-in denied: %s is not an allowed Factory Ledger user", claims.get("email")
            )
            return back_to_client(
                "access_denied", "This Google account is not allowed to use Factory Ledger"
            )
        code = self.signer.sign(
            "ac",
            {
                "sub": user.email,
                "cid": pending.client_id,
                "ru": str(pending.params.redirect_uri),
                "rx": pending.params.redirect_uri_provided_explicitly,
                "cc": pending.params.code_challenge,
                "scp": pending.params.scopes,
                "res": pending.params.resource,
            },
            ttl=AUTH_CODE_TTL,
        )
        logger.info("sign-in accepted for %s (%s)", user.email, user.role)
        return RedirectResponse(
            construct_redirect_uri(
                str(pending.params.redirect_uri), code=code, state=pending.params.state
            ),
            status_code=302,
            headers={"Cache-Control": "no-store"},
        )

    async def _verified_google_claims(
        self, code: str, pending: PendingLogin
    ) -> dict[str, Any] | None:
        try:
            response = await self.http.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self.settings.google_client_id,
                    "client_secret": self.settings.google_client_secret,
                    "redirect_uri": self.settings.public_url + CALLBACK_PATH,
                    "grant_type": "authorization_code",
                    "code_verifier": pending.code_verifier,
                },
            )
            payload = response.json() if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            payload = None
        id_token = payload.get("id_token") if isinstance(payload, dict) else None
        if not isinstance(id_token, str):
            logger.warning("google token exchange failed")
            return None
        try:
            kid = jwt.get_unverified_header(id_token).get("kid")
            key = await self._google_key(kid)
            if key is None:
                return None
            claims = jwt.decode(
                id_token,
                key,
                algorithms=["RS256"],
                audience=self.settings.google_client_id,
                issuer=list(GOOGLE_ISSUERS),
                leeway=30,
                options={"require": ["exp", "iat", "aud", "iss", "sub"]},
            )
        except jwt.PyJWTError:
            logger.warning("google id token rejected")
            return None
        if not hmac.compare_digest(str(claims.get("nonce", "")), pending.nonce):
            return None
        if claims.get("email_verified") is not True or not isinstance(claims.get("email"), str):
            return None
        if (
            self.settings.google_hosted_domain
            and claims.get("hd") != self.settings.google_hosted_domain
        ):
            return None
        return claims

    async def _google_key(self, kid: str | None):
        if not kid:
            return None
        now = time.time()
        stale = now - self._jwks_fetched_at > JWKS_TTL
        if (
            stale or kid not in self._jwks
        ) and now - self._jwks_fetched_at > JWKS_MIN_REFRESH_INTERVAL:
            try:
                response = await self.http.get(GOOGLE_JWKS_URL)
                keys = response.json().get("keys", []) if response.status_code == 200 else []
                self._jwks = {
                    k["kid"]: jwt.PyJWK.from_dict(k).key
                    for k in keys
                    if isinstance(k, dict) and k.get("kid") and k.get("kty") == "RSA"
                }
                self._jwks_fetched_at = now
            except (httpx.HTTPError, ValueError, jwt.PyJWKError):
                logger.warning("google signing keys unavailable")
        return self._jwks.get(kid)

    # ----- codes and tokens -----

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        body = self.signer.verify("ac", authorization_code)
        if (
            body is None
            or self.consumed_codes.active(body["jti"])
            or body.get("cid") != client.client_id
        ):
            return None
        return AuthorizationCode(
            code=authorization_code,
            scopes=body.get("scp") or [],
            expires_at=body["exp"],
            client_id=body["cid"],
            code_challenge=body["cc"],
            redirect_uri=AnyUrl(body["ru"]),
            redirect_uri_provided_explicitly=bool(body.get("rx")),
            resource=body.get("res"),
            subject=body["sub"],
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        body = self.signer.verify("ac", authorization_code.code)
        if body is None or self.consumed_codes.active(body["jti"]):
            raise TokenError("invalid_grant", "authorization code is invalid or already used")
        self.consumed_codes.add(body["jti"], body["exp"])
        user = resolve_allowed_user(self.settings.allowed_users, authorization_code.subject)
        if user is None:
            raise TokenError("invalid_grant", "account is no longer allowed")
        return self._issue(
            client.client_id or "", user, authorization_code.scopes, authorization_code.resource
        )

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> SignedRefreshToken | None:
        body = self.signer.verify("rt", refresh_token)
        if body is None or self.revoked.active(body["jti"]) or body.get("cid") != client.client_id:
            return None
        return SignedRefreshToken(
            token=refresh_token,
            client_id=body["cid"],
            scopes=body.get("scp") or [],
            expires_at=body["exp"],
            resource=body.get("res"),
            subject=body["sub"],
            jti=body["jti"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: SignedRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        user = resolve_allowed_user(self.settings.allowed_users, refresh_token.subject)
        if user is None:
            raise TokenError("invalid_grant", "account is no longer allowed")
        self.revoked.add(refresh_token.jti, refresh_token.expires_at or time.time() + 1)
        return self._issue(client.client_id or "", user, scopes, refresh_token.resource)

    def _issue(
        self, client_id: str, user: AllowedUser, requested: list[str], resource: str | None
    ) -> OAuthToken:
        granted = scopes_for(user.identity())
        scopes = [s for s in requested if s in granted] if requested else granted
        if requested and not scopes:
            raise TokenError("invalid_scope", "requested scopes are not granted to this user")
        claims = {"sub": user.email, "cid": client_id, "scp": scopes, "res": resource}
        access = self.signer.sign("at", claims, ttl=self.settings.access_token_ttl)
        refresh = self.signer.sign("rt", claims, ttl=self.settings.refresh_token_ttl)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self.settings.access_token_ttl,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    def inspect_bearer(self, token: str | None) -> BearerResult:
        """Verify an access token and re-resolve the user; never trusts token role claims."""
        body = self.signer.verify("at", token)
        if body is None or self.revoked.active(body["jti"]):
            return BearerResult("invalid")
        user = resolve_allowed_user(self.settings.allowed_users, body.get("sub"))
        if user is None:
            return BearerResult("forbidden")
        identity = user.identity()
        granted = scopes_for(identity)
        scopes = [s for s in body.get("scp") or granted if s in granted]
        access = SignedAccessToken(
            token=token or "",
            client_id=body.get("cid", ""),
            scopes=scopes,
            expires_at=body.get("exp"),
            resource=body.get("res"),
            subject=identity.email,
            claims={"iss": self.settings.public_url},
            jti=body["jti"],
        )
        return BearerResult("ok", identity, access)

    async def load_access_token(self, token: str) -> SignedAccessToken | None:
        result = self.inspect_bearer(token)
        return result.token if result.status == "ok" else None

    async def revoke_token(self, token: SignedAccessToken | SignedRefreshToken) -> None:
        jti = getattr(token, "jti", None)
        if jti:
            self.revoked.add(jti, token.expires_at or time.time() + self.settings.refresh_token_ttl)

    def _resource_is_ours(self, resource: str | None) -> bool:
        if resource is None:
            return True
        normalized = resource.rstrip("/")
        return normalized == self.settings.public_url or any(
            normalized == self.settings.resource_url(group) for group in ("office", "floor")
        )


# ----- request gate -----


class AuthGate:
    """ASGI wrapper for one group endpoint: authenticate, authorize, bind identity."""

    def __init__(
        self, app, settings: Settings, group: str, provider: GoogleOAuthProvider | None = None
    ):
        if group not in {"office", "floor"}:
            raise ValueError("group must be office or floor")
        if settings.auth_mode == "google" and provider is None:
            raise ValueError(
                "google mode needs a shared Authenticator: use Authenticator(settings).protect()"
            )
        self.app, self.settings, self.group, self.provider = app, settings, group, provider

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"authorization", b"")
        if self.settings.auth_mode == "google":
            await self._google(scope, receive, send, supplied)
        elif self.settings.auth_mode == "local_stub":
            await self._local_stub(scope, receive, send, supplied)
        else:
            await self._deny(scope, receive, send, 401, "Authentication required", "Bearer")

    async def _deny(self, scope, receive, send, status, message, www_authenticate=None, error=None):
        body = {
            "error": error or ("unauthorized" if status == 401 else "forbidden"),
            "error_description": message,
        }
        headers = {"Cache-Control": "no-store"}
        if www_authenticate:
            headers["WWW-Authenticate"] = www_authenticate
        await JSONResponse(body, status_code=status, headers=headers)(scope, receive, send)

    def _challenge(self, error: str | None = None) -> str:
        metadata = (
            f"{self.settings.public_url}/.well-known/oauth-protected-resource/{self.group}/mcp"
        )
        parts = [f'resource_metadata="{metadata}"']
        if error:
            parts.insert(0, f'error="{error}"')
        return "Bearer " + ", ".join(parts)

    async def _google(self, scope, receive, send, supplied: bytes):
        assert self.provider is not None
        if not supplied.lower().startswith(b"bearer "):
            await self._deny(
                scope, receive, send, 401, "Authentication required", self._challenge()
            )
            return
        result = self.provider.inspect_bearer(supplied[7:].decode("latin-1").strip())
        if result.status == "invalid":
            await self._deny(
                scope,
                receive,
                send,
                401,
                "Invalid or expired token",
                self._challenge("invalid_token"),
                "invalid_token",
            )
            return
        if result.status == "forbidden" or result.identity is None or result.token is None:
            await self._deny(
                scope, receive, send, 403, "This account is not allowed to use Factory Ledger"
            )
            return
        resource = result.token.resource
        if resource is not None and resource.rstrip("/") not in {
            self.settings.public_url,
            self.settings.resource_url(self.group),
        }:
            await self._deny(
                scope,
                receive,
                send,
                401,
                "Token was issued for another resource",
                self._challenge("invalid_token"),
                "invalid_token",
            )
            return
        if (
            not can_read_group(result.identity, self.group)
            or f"{self.group}.read" not in result.token.scopes
        ):
            await self._deny(scope, receive, send, 403, f"No {self.group} access for this account")
            return
        with identity_scope(result.identity):
            await self.app(scope, receive, send)

    async def _local_stub(self, scope, receive, send, supplied: bytes):
        settings = self.settings
        peer = (scope.get("client") or ("",))[0]
        valid = peer in {"127.0.0.1", "::1"} and hmac.compare_digest(
            supplied, f"Bearer {settings.dev_token}".encode()
        )
        if not valid:
            await self._deny(
                scope,
                receive,
                send,
                401,
                "Authentication required; loopback development token only",
                "Bearer",
            )
            return
        if settings.dev_email:
            user = settings.allowed_users.get(settings.dev_email)
            if user is None or not can_read_group(user.identity(), self.group):
                await self._deny(
                    scope, receive, send, 403, f"No {self.group} access for this account"
                )
                return
            with identity_scope(user.identity()):
                await self.app(scope, receive, send)
            return
        # Legacy demo roles: no identity is bound, so current_identity() keeps raising.
        if self.group == "office" and settings.dev_role == "floor":
            await self._deny(scope, receive, send, 403, "Office access denied for floor role")
            return
        await self.app(scope, receive, send)


class DevelopmentAuth(AuthGate):
    """Backward-compatible gate for locked/local_stub; google mode needs `Authenticator`."""

    def __init__(self, app, settings: Settings, group: str):
        super().__init__(app, settings, group, provider=None)


# ----- routes -----


def _cors(app, methods: list[str]):
    return CORSMiddleware(
        app=app,
        allow_origins="*",
        allow_methods=methods,
        allow_headers=[MCP_PROTOCOL_VERSION_HEADER],
    )


def _limited(handler):
    return RequestBodyLimitMiddleware(request_response(handler), DEFAULT_MAX_REQUEST_BODY_SIZE)


class Authenticator:
    """Owns the OAuth provider (google mode) and builds gates and discovery routes."""

    def __init__(
        self, settings: Settings, *, outbound_transport: httpx.AsyncBaseTransport | None = None
    ):
        self.settings = settings
        self.provider = (
            GoogleOAuthProvider(settings, outbound_transport=outbound_transport)
            if settings.auth_mode == "google"
            else None
        )

    @property
    def google_oauth_ready(self) -> bool:
        return self.provider is not None

    def protect(self, app, group: str) -> AuthGate:
        return AuthGate(app, self.settings, group, self.provider)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        if self.provider is not None:
            await self.provider.aclose()

    def routes(self) -> list[Route]:
        if self.provider is None:
            return []
        settings, provider = self.settings, self.provider
        issuer = AnyHttpUrl(settings.public_url)
        registration = ClientRegistrationOptions(
            enabled=True, valid_scopes=list(ALL_SCOPES), default_scopes=list(ALL_SCOPES)
        )
        revocation = RevocationOptions(enabled=True)
        metadata = build_metadata(issuer, None, registration, revocation)
        metadata.token_endpoint_auth_methods_supported = list(TOKEN_AUTH_METHODS)
        metadata.revocation_endpoint_auth_methods_supported = list(TOKEN_AUTH_METHODS)
        metadata.client_id_metadata_document_supported = True
        metadata.scopes_supported = list(ALL_SCOPES)
        metadata_endpoint = _cors(
            request_response(MetadataHandler(metadata).handle), ["GET", "OPTIONS"]
        )
        client_auth = ClientAuthenticator(provider)
        routes = [
            Route(
                "/.well-known/oauth-authorization-server",
                metadata_endpoint,
                methods=["GET", "OPTIONS"],
            ),
            Route(
                "/.well-known/openid-configuration", metadata_endpoint, methods=["GET", "OPTIONS"]
            ),
            Route(
                AUTHORIZATION_PATH,
                _limited(AuthorizationHandler(provider).handle),
                methods=["GET", "POST"],
            ),
            Route(
                TOKEN_PATH,
                _cors(_limited(TokenHandler(provider, client_auth).handle), ["POST", "OPTIONS"]),
                methods=["POST", "OPTIONS"],
            ),
            Route(
                REGISTRATION_PATH,
                _cors(
                    _limited(RegistrationHandler(provider, options=registration).handle),
                    ["POST", "OPTIONS"],
                ),
                methods=["POST", "OPTIONS"],
            ),
            Route(
                REVOCATION_PATH,
                _cors(
                    _limited(RevocationHandler(provider, client_auth).handle), ["POST", "OPTIONS"]
                ),
                methods=["POST", "OPTIONS"],
            ),
            Route(CALLBACK_PATH, provider.handle_google_callback, methods=["GET"]),
        ]
        for group in ("office", "floor"):
            routes += create_protected_resource_routes(
                resource_url=AnyHttpUrl(settings.resource_url(group)),
                authorization_servers=[issuer],
                scopes_supported=[f"{group}.read", f"{group}.write"],
                resource_name=f"Factory Ledger — {group}",
            )
        routes += create_protected_resource_routes(
            resource_url=AnyHttpUrl(settings.public_url + "/"),
            authorization_servers=[issuer],
            scopes_supported=list(ALL_SCOPES),
            resource_name="Factory Ledger",
        )
        return routes

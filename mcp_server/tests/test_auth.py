"""Google OAuth / allowlist auth: denial paths, role permissions, isolation, config safety.

Google is replaced by an in-process fake that signs ID tokens with a throwaway RSA
key; every email, key and secret here is synthetic (`example.invalid`).
"""

import asyncio
import hashlib
import json
import re
import secrets
import subprocess
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from factory_ledger_mcp import identity
from factory_ledger_mcp.adapter import LedgerReader
from factory_ledger_mcp.auth import (
    CALLBACK_PATH,
    GOOGLE_AUTH_URL,
    Authenticator,
    DevelopmentAuth,
    TokenSigner,
)
from factory_ledger_mcp.config import Settings
from factory_ledger_mcp.demo import create_demo_app
from factory_ledger_mcp.identity import (
    AllowedUser,
    AllowlistError,
    can_read_floor,
    can_read_office,
    can_write_floor,
    can_write_office,
    parse_allowlist,
    resolve_allowed_user,
)
from factory_ledger_mcp.server import create_app, group_server

from .conftest import TOKEN, rpc

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_URL = "http://127.0.0.1:8000"
GOOGLE_CLIENT_ID = "synthetic-google-client-id.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "synthetic-google-client-secret"
TOKEN_SECRET = "synthetic-token-secret-with-at-least-32-characters"
ADMIN = "admin.synthetic@example.invalid"
FLOOR = "floor.synthetic@example.invalid"
OUTSIDER = "outsider.synthetic@example.invalid"
ADMIN_KEY = "synthetic-actor-key-admin"
FLOOR_KEY = "synthetic-actor-key-floor"
ACTOR_ENV = {"MCP_ACTOR_KEY_ADMIN": ADMIN_KEY, "MCP_ACTOR_KEY_FLOOR": FLOOR_KEY}
ALLOWLIST_JSON = json.dumps(
    [
        {"email": ADMIN, "role": "admin_office_floor", "actor_key_env": "MCP_ACTOR_KEY_ADMIN"},
        {"email": FLOOR, "role": "floor", "actor_key_env": "MCP_ACTOR_KEY_FLOOR"},
    ]
)
REDIRECT_URI = "https://client.example/callback"
CIMD_CLIENT_ID = "https://client.example/cimd/client.json"
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}


def allowlist():
    return parse_allowlist(ALLOWLIST_JSON, ACTOR_ENV)


def google_settings(**overrides):
    values = dict(
        environment="test",
        auth_mode="google",
        public_url=PUBLIC_URL,
        google_client_id=GOOGLE_CLIENT_ID,
        google_client_secret=GOOGLE_CLIENT_SECRET,
        token_secret=TOKEN_SECRET,
        allowed_users=allowlist(),
    )
    values.update(overrides)
    return Settings(**values)


class FakeGoogle:
    """Token endpoint + JWKS + a client metadata document, all synthetic."""

    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "synthetic-kid"
        self.issuer = "https://accounts.google.com"
        self.email_verified = True
        self.token_requests = []

    def id_token(self, email, nonce, **overrides):
        now = int(time.time())
        claims = {
            "iss": self.issuer,
            "aud": GOOGLE_CLIENT_ID,
            "sub": "google-subject-" + hashlib.sha256(email.encode()).hexdigest()[:12],
            "email": email,
            "email_verified": self.email_verified,
            "iat": now,
            "exp": now + 300,
            "nonce": nonce,
        }
        claims.update(overrides)
        pem = self.private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        return jwt.encode(claims, pem, algorithm="RS256", headers={"kid": self.kid})

    def app(self):
        async def token(request):
            form = dict(await request.form())
            self.token_requests.append(form)
            if "|" not in form.get("code", ""):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            email, nonce = form["code"].split("|", 1)
            return JSONResponse(
                {
                    "access_token": "synthetic-google-access-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "id_token": self.id_token(email, nonce),
                }
            )

        async def certs(request):
            jwk = RSAAlgorithm.to_jwk(self.private_key.public_key(), as_dict=True)
            jwk.update({"kid": self.kid, "use": "sig", "alg": "RS256"})
            return JSONResponse({"keys": [jwk]})

        async def cimd(request):
            return JSONResponse(
                {
                    "client_id": CIMD_CLIENT_ID,
                    "client_name": "Synthetic CIMD client",
                    "redirect_uris": [REDIRECT_URI],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                }
            )

        return Starlette(
            routes=[
                Route("/token", token, methods=["POST"]),
                Route("/oauth2/v3/certs", certs),
                Route("/cimd/client.json", cimd),
            ]
        )


@pytest.fixture
def fake_google():
    return FakeGoogle()


def build_google_app(settings, auth, backend_transport):
    """Mirror the documented server.py integration: routes + protected group endpoints."""
    client = httpx.AsyncClient(
        base_url=settings.ledger_url,
        transport=backend_transport,
        trust_env=False,
        follow_redirects=False,
    )
    reader = LedgerReader(client)
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*"],
        allowed_origins=["http://127.0.0.1:*"],
    )
    managers = {
        group: StreamableHTTPSessionManager(
            app=group_server(group, reader),
            json_response=True,
            stateless=True,
            security_settings=security,
        )
        for group in ("office", "floor")
    }

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(client)
            await stack.enter_async_context(auth)
            for manager in managers.values():
                await stack.enter_async_context(manager.run())
            yield

    routes = [
        Route(f"/{group}/mcp", auth.protect(manager.handle_request, group), methods=["GET", "POST"])
        for group, manager in managers.items()
    ]
    return Starlette(routes=routes + auth.routes(), lifespan=lifespan)


@pytest.fixture
def google_stack(fake_google):
    @asynccontextmanager
    async def open_stack(settings=None, backend_transport=None):
        settings = settings or google_settings()
        backend = create_demo_app()
        auth = Authenticator(
            settings, outbound_transport=httpx.ASGITransport(app=fake_google.app())
        )
        app = build_google_app(
            settings, auth, backend_transport or httpx.ASGITransport(app=backend)
        )
        async with backend.router.lifespan_context(backend), app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=("203.0.113.9", 4321)),
                base_url=PUBLIC_URL,
                headers=MCP_HEADERS,
            ) as client:
                yield client, backend, auth

    return open_stack


def pkce():
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, jwt.utils.base64url_encode(digest).decode()


async def register(client, method="none"):
    response = await client.post(
        "/register",
        json={
            "redirect_uris": [REDIRECT_URI],
            "token_endpoint_auth_method": method,
            "client_name": "Synthetic connector",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def start_sign_in(client, client_id, email, *, resource=None, scope=None, state="s1"):
    verifier, challenge = pkce()
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if resource:
        params["resource"] = resource
    if scope:
        params["scope"] = scope
    response = await client.get("/authorize", params=params)
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    assert location.startswith(GOOGLE_AUTH_URL + "?")
    google = parse_qs(urlsplit(location).query)
    assert google["redirect_uri"] == [PUBLIC_URL + CALLBACK_PATH]
    assert google["code_challenge_method"] == ["S256"]
    assert google["client_id"] == [GOOGLE_CLIENT_ID]
    callback = await client.get(
        CALLBACK_PATH, params={"state": google["state"][0], "code": f"{email}|{google['nonce'][0]}"}
    )
    assert callback.status_code == 302, callback.text
    target = callback.headers["location"]
    assert target.startswith(REDIRECT_URI + "?")
    return parse_qs(urlsplit(target).query), verifier


async def exchange(client, client_id, code, verifier, *, secret=None, resource=None):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    if secret:
        data["client_secret"] = secret
    if resource:
        data["resource"] = resource
    return await client.post("/token", data=data)


async def sign_in(client, client_id, email, **kwargs):
    secret = kwargs.pop("secret", None)
    query, verifier = await start_sign_in(client, client_id, email, **kwargs)
    assert "code" in query, query
    response = await exchange(
        client,
        client_id,
        query["code"][0],
        verifier,
        secret=secret,
        resource=kwargs.get("resource"),
    )
    assert response.status_code == 200, response.text
    return response.json()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def observe(monkeypatch):
    """Replace the backend call with one that reports the identity bound to the request."""
    seen = []

    async def call(self, group, name, arguments):
        await asyncio.sleep(0.01)
        try:
            who = identity.current_identity()
        except PermissionError as exc:
            seen.append(exc)
            return {"identity": None}
        seen.append(who)
        return {"identity": who.email, "role": who.role, "group": group, "tool": name}

    monkeypatch.setattr(LedgerReader, "call", call)
    return seen


# ----- discovery and denial -----


async def test_discovery_metadata_matches_mcp_authorization_profile(google_stack):
    async with google_stack() as (client, _, _):
        metadata = (await client.get("/.well-known/oauth-authorization-server")).json()
        assert metadata["issuer"].rstrip("/") == PUBLIC_URL
        assert metadata["authorization_endpoint"] == PUBLIC_URL + "/authorize"
        assert metadata["token_endpoint"] == PUBLIC_URL + "/token"
        assert metadata["registration_endpoint"] == PUBLIC_URL + "/register"
        assert metadata["revocation_endpoint"] == PUBLIC_URL + "/revoke"
        assert metadata["code_challenge_methods_supported"] == ["S256"]
        assert "none" in metadata["token_endpoint_auth_methods_supported"]
        assert metadata["client_id_metadata_document_supported"] is True
        assert set(metadata["scopes_supported"]) == {
            "office.read",
            "office.write",
            "floor.read",
            "floor.write",
        }
        assert (await client.get("/.well-known/openid-configuration")).json() == metadata
        for group in ("office", "floor"):
            resource = (
                await client.get(f"/.well-known/oauth-protected-resource/{group}/mcp")
            ).json()
            assert resource["resource"] == f"{PUBLIC_URL}/{group}/mcp"
            assert [s.rstrip("/") for s in resource["authorization_servers"]] == [PUBLIC_URL]
            assert resource["bearer_methods_supported"] == ["header"]


@pytest.mark.parametrize("header", [None, "Bearer not-a-token", "Basic abc", "Bearer "])
async def test_unauthenticated_requests_denied_with_discovery_hint(google_stack, header):
    async with google_stack() as (client, backend, _):
        for group in ("office", "floor"):
            headers = {"Authorization": header} if header is not None else {}
            for method in ("tools/list", "tools/call"):
                response = await client.post(
                    f"/{group}/mcp",
                    headers=headers,
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": {}},
                )
                assert response.status_code == 401
                challenge = response.headers["www-authenticate"]
                assert challenge.startswith("Bearer ")
                assert (
                    f'resource_metadata="{PUBLIC_URL}/.well-known/oauth-protected-resource/{group}/mcp"'
                    in challenge
                )
                if header and header.startswith("Bearer") and header != "Bearer ":
                    assert 'error="invalid_token"' in challenge
        assert backend.state.requests == []


async def test_forged_expired_and_wrong_kind_tokens_rejected(google_stack):
    async with google_stack() as (client, backend, auth):
        claims = {"sub": ADMIN, "cid": "cl.x", "scp": ["office.read"], "res": None}
        forged = TokenSigner("another-synthetic-secret-of-32-characters!").sign("at", claims, 3600)
        expired = auth.provider.signer.sign("at", claims, ttl=-5)
        refresh_as_bearer = auth.provider.signer.sign("rt", claims, 3600)
        tampered = auth.provider.signer.sign("at", claims, 3600)
        head, body, mac = tampered.split(".")
        tampered = ".".join([head, body[:-2] + ("AA" if body[-2:] != "AA" else "BB"), mac])
        for token in (forged, expired, refresh_as_bearer, tampered):
            response = await client.post("/office/mcp", headers=bearer(token), json={})
            assert response.status_code == 401, token[:12]
        assert backend.state.requests == []


async def test_unlisted_google_account_denied_after_successful_sign_in(google_stack, fake_google):
    async with google_stack() as (client, backend, _):
        info = await register(client)
        query, _ = await start_sign_in(client, info["client_id"], OUTSIDER, state="keep-me")
        assert "code" not in query
        assert query["error"] == ["access_denied"]
        assert query["state"] == ["keep-me"]
        assert len(fake_google.token_requests) == 1  # Google sign-in itself succeeded.

        fake_google.email_verified = False
        query, _ = await start_sign_in(client, info["client_id"], ADMIN)
        assert query["error"] == ["access_denied"]
        fake_google.email_verified = True

        fake_google.issuer = "https://evil.example"
        query, _ = await start_sign_in(client, info["client_id"], ADMIN)
        assert query["error"] == ["access_denied"]
        assert backend.state.requests == []


async def test_callback_rejects_unknown_or_replayed_state_and_cancelled_login(google_stack):
    async with google_stack() as (client, _, _):
        info = await register(client)
        response = await client.get(CALLBACK_PATH, params={"state": "unknown", "code": "x|y"})
        assert response.status_code == 400
        _, challenge = pkce()
        authorize = await client.get(
            "/authorize",
            params={
                "client_id": info["client_id"],
                "redirect_uri": REDIRECT_URI,
                "response_type": "code",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "abc",
            },
        )
        state = parse_qs(urlsplit(authorize.headers["location"]).query)["state"][0]
        cancelled = await client.get(
            CALLBACK_PATH, params={"state": state, "error": "access_denied"}
        )
        assert cancelled.status_code == 302
        assert parse_qs(urlsplit(cancelled.headers["location"]).query)["error"] == ["access_denied"]
        replay = await client.get(CALLBACK_PATH, params={"state": state, "code": f"{ADMIN}|nonce"})
        assert replay.status_code == 400


# ----- roles -----


async def test_admin_signs_in_and_identity_is_bound_inside_tool_calls(
    google_stack, fake_google, observe
):
    async with google_stack() as (client, _, _):
        info = await register(client)
        tokens = await sign_in(
            client, info["client_id"], ADMIN, resource=f"{PUBLIC_URL}/office/mcp"
        )
        assert set(tokens["scope"].split()) == {
            "office.read",
            "office.write",
            "floor.read",
            "floor.write",
        }
        assert tokens["token_type"] == "Bearer" and tokens["refresh_token"]
        google_exchange = fake_google.token_requests[-1]
        assert google_exchange["client_secret"] == GOOGLE_CLIENT_SECRET
        assert (
            google_exchange["code_verifier"]
            and google_exchange["grant_type"] == "authorization_code"
        )

        client.headers.update(bearer(tokens["access_token"]))
        listed = (await rpc(client, "office", "tools/list"))["result"]["tools"]
        assert len(listed) == 30  # 14 reads + 16 office write proposals for admin_office_floor
        result = (
            await rpc(client, "office", "tools/call", {"name": "listCustomers", "arguments": {}})
        )["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["data"]["identity"] == ADMIN
        who = observe[-1]
        assert isinstance(who, identity.Identity)
        assert (who.email, who.role, who.actor_key) == (ADMIN, "admin_office_floor", ADMIN_KEY)
        assert ADMIN_KEY not in repr(who)
        assert ADMIN_KEY not in tokens["access_token"] and ADMIN_KEY not in tokens["refresh_token"]

        # Audience binding: an office token is not accepted at the floor endpoint.
        denied = await client.post("/floor/mcp", json={})
        assert denied.status_code == 401
        assert 'error="invalid_token"' in denied.headers["www-authenticate"]

        floor_tokens = await sign_in(
            client, info["client_id"], ADMIN, resource=f"{PUBLIC_URL}/floor/mcp"
        )
        client.headers.update(bearer(floor_tokens["access_token"]))
        assert len((await rpc(client, "floor", "tools/list"))["result"]["tools"]) == 22
        # No identity leaks outside a request.
        with pytest.raises(PermissionError):
            identity.current_identity()


async def test_floor_role_reads_both_groups_but_holds_no_office_write(google_stack, observe):
    async with google_stack() as (client, _, _):
        info = await register(client)
        tokens = await sign_in(client, info["client_id"], FLOOR)
        assert set(tokens["scope"].split()) == {"office.read", "floor.read", "floor.write"}
        client.headers.update(bearer(tokens["access_token"]))
        # Floor role: office reads only (14); floor reads plus the 10 floor write proposals.
        for group, count in (("office", 14), ("floor", 22)):
            assert len((await rpc(client, group, "tools/list"))["result"]["tools"]) == count
            result = (
                await rpc(
                    client,
                    group,
                    "tools/call",
                    {"name": "getOrder", "arguments": {"order_id": "SO-1"}},
                )
            )["result"]
            assert result["structuredContent"]["data"] == {
                "identity": FLOOR,
                "role": "floor",
                "group": group,
                "tool": "getOrder",
            }
        who = observe[-1]
        assert who.actor_key == FLOOR_KEY
        assert (can_read_office(who), can_write_office(who)) == (True, False)
        assert (can_read_floor(who), can_write_floor(who)) == (True, True)

        # Asking for a scope the role does not grant is refused at the token endpoint.
        query, verifier = await start_sign_in(
            client, info["client_id"], FLOOR, scope="office.write"
        )
        response = await exchange(client, info["client_id"], query["code"][0], verifier)
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_scope"

        # A narrowed office-only token cannot be used at the floor endpoint.
        narrow = await sign_in(client, info["client_id"], FLOOR, scope="office.read")
        assert narrow["scope"] == "office.read"
        denied = await client.post("/floor/mcp", headers=bearer(narrow["access_token"]), json={})
        assert denied.status_code == 403


async def test_removed_user_is_forbidden_immediately_and_cannot_refresh(google_stack):
    users = allowlist()
    async with google_stack(google_settings(allowed_users=users)) as (client, backend, _):
        info = await register(client)
        tokens = await sign_in(client, info["client_id"], ADMIN)
        client.headers.update(bearer(tokens["access_token"]))
        assert (
            await client.post(
                "/office/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
        ).status_code == 200
        del users[ADMIN]
        response = await client.post("/office/mcp", json={})
        assert response.status_code == 403
        refreshed = await client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": info["client_id"],
            },
        )
        assert refreshed.status_code == 400
        assert refreshed.json()["error"] == "invalid_grant"
        assert backend.state.requests == []


# ----- OAuth mechanics -----


async def test_pkce_client_binding_and_single_use_codes(google_stack):
    async with google_stack() as (client, _, _):
        info = await register(client)
        other = await register(client)
        query, verifier = await start_sign_in(client, info["client_id"], ADMIN)
        code = query["code"][0]
        wrong = await exchange(client, info["client_id"], code, "wrong-verifier-" + verifier)
        assert wrong.status_code == 400 and wrong.json()["error"] == "invalid_grant"
        stolen = await exchange(client, other["client_id"], code, verifier)
        assert stolen.status_code == 400 and stolen.json()["error"] == "invalid_grant"
        good = await exchange(client, info["client_id"], code, verifier)
        assert good.status_code == 200
        replay = await exchange(client, info["client_id"], code, verifier)
        assert replay.status_code == 400 and replay.json()["error"] == "invalid_grant"


async def test_refresh_rotation_and_revocation(google_stack):
    async with google_stack() as (client, _, _):
        info = await register(client, method="client_secret_post")
        assert info["client_secret"]
        tokens = await sign_in(client, info["client_id"], ADMIN, secret=info["client_secret"])
        form = {"client_id": info["client_id"], "client_secret": info["client_secret"]}
        refreshed = await client.post(
            "/token",
            data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"], **form},
        )
        assert refreshed.status_code == 200, refreshed.text
        rotated = refreshed.json()
        assert rotated["access_token"] != tokens["access_token"]
        assert rotated["refresh_token"] != tokens["refresh_token"]
        reused = await client.post(
            "/token",
            data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"], **form},
        )
        assert reused.status_code == 400
        assert (
            await client.post(
                "/office/mcp",
                headers=bearer(rotated["access_token"]),
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
        ).status_code == 200

        revoked = await client.post("/revoke", data={"token": rotated["access_token"], **form})
        assert revoked.status_code == 200
        assert (
            await client.post("/office/mcp", headers=bearer(rotated["access_token"]), json={})
        ).status_code == 401
        await client.post(
            "/revoke",
            data={"token": rotated["refresh_token"], "token_type_hint": "refresh_token", **form},
        )
        after = await client.post(
            "/token",
            data={"grant_type": "refresh_token", "refresh_token": rotated["refresh_token"], **form},
        )
        assert after.status_code == 400

        # The wrong client secret never reaches the grant.
        bad = await client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": rotated["refresh_token"],
                "client_id": info["client_id"],
                "client_secret": "wrong",
            },
        )
        assert bad.status_code == 401


async def test_client_id_metadata_document_clients_need_no_registration(google_stack):
    async with google_stack() as (client, _, _):
        tokens = await sign_in(client, CIMD_CLIENT_ID, ADMIN)
        assert tokens["access_token"]
        unknown = await client.get(
            "/authorize",
            params={
                "client_id": "https://client.example/missing.json",
                "redirect_uri": REDIRECT_URI,
                "response_type": "code",
                "code_challenge": pkce()[1],
                "code_challenge_method": "S256",
            },
        )
        assert unknown.status_code == 400


async def test_registrations_and_tokens_survive_a_restart(google_stack, fake_google):
    async with google_stack() as (client, _, first):
        info = await register(client, method="client_secret_post")
        tokens = await sign_in(client, info["client_id"], ADMIN, secret=info["client_secret"])
    restarted = Authenticator(
        google_settings(), outbound_transport=httpx.ASGITransport(app=fake_google.app())
    )
    reloaded = await restarted.provider.get_client(info["client_id"])
    assert reloaded is not None
    assert [str(u) for u in reloaded.redirect_uris] == [REDIRECT_URI]
    assert reloaded.client_secret == info["client_secret"]
    assert restarted.provider.inspect_bearer(tokens["access_token"]).status == "ok"
    other_secret = Authenticator(
        google_settings(token_secret="a-different-synthetic-secret-32-chars!!")
    )
    assert other_secret.provider.inspect_bearer(tokens["access_token"]).status == "invalid"
    assert await other_secret.provider.get_client(info["client_id"]) is None


async def test_concurrent_requests_keep_identities_isolated(google_stack, observe):
    async with google_stack() as (client, _, _):
        info = await register(client)
        admin = (await sign_in(client, info["client_id"], ADMIN))["access_token"]
        floor = (await sign_in(client, info["client_id"], FLOOR))["access_token"]

        async def call(token, index):
            response = await client.post(
                "/floor/mcp",
                headers=bearer(token),
                json={
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "tools/call",
                    "params": {"name": "getOrder", "arguments": {"order_id": f"SO-{index}"}},
                },
            )
            assert response.status_code == 200, response.text
            return response.json()["result"]["structuredContent"]["data"]["identity"]

        expected = [(admin, ADMIN), (floor, FLOOR)] * 8
        results = await asyncio.gather(*(call(token, i) for i, (token, _) in enumerate(expected)))
        assert results == [email for _, email in expected]
        assert {who.email for who in observe} == {ADMIN, FLOOR}


# ----- local test mode -----


async def test_local_stub_dev_email_binds_a_real_allowlisted_identity(observe):
    settings = Settings(
        environment="test",
        auth_mode="local_stub",
        dev_token=TOKEN,
        allowed_users=allowlist(),
        dev_email=FLOOR.upper(),
    )
    backend = create_demo_app()
    app = create_app(settings, backend_transport=httpx.ASGITransport(app=backend))
    async with backend.router.lifespan_context(backend), app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url=PUBLIC_URL,
            headers={**MCP_HEADERS, **bearer(TOKEN)},
        ) as client:
            # The decided matrix applies: floor users read office, unlike the legacy demo role.
            result = (
                await rpc(
                    client, "office", "tools/call", {"name": "listCustomers", "arguments": {}}
                )
            )["result"]
            assert result["structuredContent"]["data"]["identity"] == FLOOR
            assert observe[-1].actor_key == FLOOR_KEY
            wrong = await client.post("/office/mcp", headers=bearer("nope"), json={})
            assert wrong.status_code == 401 and wrong.headers["www-authenticate"] == "Bearer"


async def test_local_stub_without_dev_email_binds_nothing(observe):
    settings = Settings(
        environment="test", auth_mode="local_stub", dev_token=TOKEN, dev_role="admin"
    )
    backend = create_demo_app()
    app = create_app(settings, backend_transport=httpx.ASGITransport(app=backend))
    async with backend.router.lifespan_context(backend), app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url=PUBLIC_URL,
            headers={**MCP_HEADERS, **bearer(TOKEN)},
        ) as client:
            result = (
                await rpc(
                    client, "office", "tools/call", {"name": "listCustomers", "arguments": {}}
                )
            )["result"]
            assert result["structuredContent"]["data"] == {"identity": None}
            assert isinstance(observe[-1], PermissionError)


def test_dev_email_must_be_allowlisted_and_stub_only():
    with pytest.raises(ValueError, match="MCP_ALLOWED_USERS"):
        Settings(environment="test", auth_mode="local_stub", dev_token=TOKEN, dev_email=OUTSIDER)
    with pytest.raises(ValueError, match="local_stub"):
        google_settings(dev_email=ADMIN)
    with pytest.raises(ValueError, match="Authenticator"):
        DevelopmentAuth(lambda *_: None, google_settings(), "office")


# ----- production safety -----


def production_google(**overrides):
    values = dict(
        environment="production",
        auth_mode="google",
        public_url="https://mcp.example.invalid",
        google_client_id=GOOGLE_CLIENT_ID,
        google_client_secret=GOOGLE_CLIENT_SECRET,
        token_secret=TOKEN_SECRET,
        allowed_users=allowlist(),
        ledger_url="https://ledger.example.invalid",
    )
    values.update(overrides)
    return Settings(**values)


def test_production_accepts_only_locked_or_google_without_test_settings(monkeypatch):
    assert production_google().hosted
    assert production_google(ledger_url="http://ledger.railway.internal:8080").hosted
    assert Settings(environment="production", auth_mode="locked").hosted
    for overrides, match in [
        ({"auth_mode": "local_stub", "dev_token": TOKEN}, "production"),
        ({"dev_token": TOKEN}, "refused in production"),
        ({"dev_email": ADMIN}, "refused in production"),
        ({"dev_role": "admin"}, "refused in production"),
        ({"public_url": "http://mcp.example.invalid"}, "https"),
        ({"ledger_url": "http://ledger.example.invalid"}, "railway.internal"),
        ({"ledger_url": "https://user:pw@ledger.example.invalid"}, "credentials"),
        ({"ledger_url": "https://ledger.example.invalid/proxy"}, "path"),
    ]:
        with pytest.raises(ValueError, match=match):
            production_google(**overrides)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "synthetic-platform-indicator")
    with pytest.raises(ValueError, match="Railway"):
        Settings(environment="local", auth_mode="local_stub", dev_token=TOKEN)
    # The shared test key no longer exists as a setting; a leftover variable is refused
    # everywhere, not only when hosted, so it can never be silently ignored.
    assert not hasattr(Settings(environment="local", auth_mode="locked"), "test_api_key")
    monkeypatch.setenv("MCP_TEST_API_KEY", "shared-key")
    with pytest.raises(ValueError, match="MCP_TEST_API_KEY"):
        Settings.from_env()
    monkeypatch.delenv("MCP_TEST_API_KEY")
    # Locked/google modes never accept remote ledgers outside a hosted google deployment.
    with pytest.raises(ValueError, match="loopback"):
        google_settings(ledger_url="https://ledger.example.invalid")


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"public_url": ""}, "MCP_PUBLIC_URL"),
        ({"public_url": "http://127.0.0.1:8000/base"}, "path"),
        ({"google_client_id": ""}, "MCP_GOOGLE_CLIENT_ID"),
        ({"google_client_secret": " "}, "MCP_GOOGLE_CLIENT_SECRET"),
        ({"token_secret": "short"}, "MCP_TOKEN_SECRET"),
        ({"allowed_users": {}}, "MCP_ALLOWED_USERS"),
        ({"access_token_ttl": 10}, "lifetimes"),
    ],
)
def test_google_mode_requires_complete_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        google_settings(**overrides)


def test_settings_from_env_loads_allowlist_and_actor_keys(monkeypatch):
    for name, value in {
        "MCP_ENV": "test",
        "MCP_AUTH_MODE": "google",
        "MCP_PUBLIC_URL": PUBLIC_URL + "/",
        "MCP_GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
        "MCP_GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
        "MCP_TOKEN_SECRET": TOKEN_SECRET,
        "MCP_ALLOWED_USERS": ALLOWLIST_JSON,
        **ACTOR_ENV,
    }.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert settings.public_url == PUBLIC_URL and settings.public_host == "127.0.0.1:8000"
    assert settings.allowed_users[ADMIN].actor_key == ADMIN_KEY
    assert ADMIN_KEY not in repr(settings) and TOKEN_SECRET not in repr(settings)
    assert GOOGLE_CLIENT_SECRET not in repr(settings)


# ----- allowlist parsing -----


def users(*entries):
    return json.dumps(
        [
            {"email": email, "role": role, "actor_key_env": key_env, **extra}
            for email, role, key_env, extra in entries
        ]
    )


X = ("x@example.invalid", "floor", "MCP_ACTOR_KEY_X", {})


@pytest.mark.parametrize(
    "raw,env,match",
    [
        ("{not json", ACTOR_ENV, "valid JSON"),
        ('{"email": "x@example.invalid"}', ACTOR_ENV, "JSON list"),
        (
            '[{"email": "x@example.invalid", "role": "admin", "actor_key_env": "MCP_ACTOR_KEY_X"}]',
            {"MCP_ACTOR_KEY_X": "k"},
            "role",
        ),
        (
            '[{"email": "x@example.invalid", "role": "floor", "actor_key_env": "MCP_ACTOR_KEY_X"}]',
            {},
            "missing or empty",
        ),
        (
            '[{"email": "x@example.invalid", "role": "floor", "actor_key_env": "API_KEY"}]',
            {"API_KEY": "k"},
            "MCP_ACTOR_KEY_",
        ),
        (
            '[{"email": "x@example.invalid", "role": "floor", "actor_key_env": "MCP_ACTOR_KEY_"}]',
            {"MCP_ACTOR_KEY_": "k"},
            "MCP_ACTOR_KEY_",
        ),
        (
            '[{"email": "x@example.invalid", "role": "floor", "actor_key_env": "MCP_ACTOR_KEY_X"}]',
            {"MCP_ACTOR_KEY_X": "shared", "API_KEY": "shared"},
            "shared key",
        ),
        (
            '[{"email": "x@example.invalid", "role": "floor", "actor_key_env": "MCP_ACTOR_KEY_X"}]',
            {"MCP_ACTOR_KEY_X": "shared", "MCP_TEST_API_KEY": "shared"},
            "shared key",
        ),
        (
            users(X, ("X@example.invalid", "floor", "MCP_ACTOR_KEY_Y", {})),
            {"MCP_ACTOR_KEY_X": "k1", "MCP_ACTOR_KEY_Y": "k2"},
            "more than once",
        ),
        (
            users(X, ("y@example.invalid", "floor", "MCP_ACTOR_KEY_Y", {})),
            {"MCP_ACTOR_KEY_X": "same", "MCP_ACTOR_KEY_Y": "same"},
            "duplicates",
        ),
        (
            users(("x@example.invalid", "floor", "MCP_ACTOR_KEY_X", {"actor_key": "inline"})),
            {"MCP_ACTOR_KEY_X": "k"},
            "unsupported fields",
        ),
        (
            '[{"email": "not-an-email", "role": "floor", "actor_key_env": "MCP_ACTOR_KEY_X"}]',
            {"MCP_ACTOR_KEY_X": "k"},
            "email",
        ),
        ('["x@example.invalid"]', ACTOR_ENV, "object"),
    ],
)
def test_allowlist_parsing_fails_closed(raw, env, match):
    with pytest.raises(AllowlistError, match=match):
        parse_allowlist(raw, env)


def test_allowlist_normalizes_and_resolves_case_insensitively():
    assert parse_allowlist("", {}) == {} and parse_allowlist(None, {}) == {}
    users = parse_allowlist(ALLOWLIST_JSON.replace(ADMIN, "  " + ADMIN.upper() + " "), ACTOR_ENV)
    assert set(users) == {ADMIN, FLOOR}
    assert resolve_allowed_user(users, ADMIN.upper()) == users[ADMIN]
    assert resolve_allowed_user(users, OUTSIDER) is None
    assert resolve_allowed_user(users, None) is None
    assert users[ADMIN].identity() == identity.Identity(ADMIN, "admin_office_floor", ADMIN_KEY)
    assert ADMIN_KEY not in repr(users[ADMIN]) and ADMIN_KEY not in repr(users[ADMIN].identity())
    assert isinstance(AllowedUser(ADMIN, "floor", "k"), AllowedUser)


# ----- repository hygiene -----


def test_no_real_emails_keys_or_secrets_tracked_in_repo():
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "mcp_server",
            ".railway",
            "docs/mcp-migration-plan.md",
            "docs/mcp-auth-contract.md",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked and not any(name.endswith(".env") for name in tracked)
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    secret_shapes = [
        re.compile(r"GOCSPX-[A-Za-z0-9_-]{10,}"),  # Google client secret prefix
        re.compile(r"MCP_ACTOR_KEY_[A-Z0-9_]+\s*[=:]\s*['\"]?[A-Za-z0-9+/=_-]{16,}"),
        re.compile(r"MCP_(TOKEN_SECRET|GOOGLE_CLIENT_SECRET)\s*=\s*['\"]?[A-Za-z0-9+/=_-]{12,}"),
        re.compile(r"\d{6,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com"),
    ]
    offenders = []
    for name in tracked:
        if name.endswith((".lock", ".png", ".jpg")):
            continue
        text = (ROOT / name).read_text(errors="ignore")
        for found in email.findall(text):
            domain = found.rsplit("@", 1)[1].lower()
            # RFC 2606 reserved names and the SDK/model vendor domains are placeholders.
            reserved = domain.endswith((".invalid", ".example", ".test", "example.com"))
            if not reserved and domain != "anthropic.com":
                offenders.append((name, found))
        for pattern in secret_shapes:
            for found in pattern.findall(text):
                offenders.append((name, str(found)[:40]))
    assert offenders == []


# ----- real ledger: the mapped actor key names a real actor and stays route-restricted -----


async def test_identity_actor_key_is_a_named_ledger_actor_without_master_reach(
    ledger, fake_google, monkeypatch
):
    from .ledger_harness import ACTOR_KEY

    db, url = ledger
    users = parse_allowlist(
        json.dumps(
            [{"email": ADMIN, "role": "admin_office_floor", "actor_key_env": "MCP_ACTOR_KEY_ADMIN"}]
        ),
        {"MCP_ACTOR_KEY_ADMIN": ACTOR_KEY},
    )
    settings = google_settings(allowed_users=users, ledger_url=url)
    probes = []

    async def call(self, group, name, arguments):
        key = identity.current_identity().actor_key
        async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=10) as ledger_client:
            whoami = await ledger_client.get("/auth/whoami", headers={"X-API-Key": key})
            resolve = await ledger_client.post(
                "/products/resolve",
                headers={"X-API-Key": key},
                json={"names": ["MCP Test Almonds"]},
            )
            admin = await ledger_client.get("/admin/lots/duplicates", headers={"X-API-Key": key})
        probes.append((whoami.status_code, whoami.json(), resolve.status_code, admin.status_code))
        return {"ok": True}

    monkeypatch.setattr(LedgerReader, "call", call)
    auth = Authenticator(settings, outbound_transport=httpx.ASGITransport(app=fake_google.app()))
    app = build_google_app(settings, auth, None)
    before = db.snapshot()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=PUBLIC_URL, headers=MCP_HEADERS
        ) as client:
            info = await register(client)
            tokens = await sign_in(client, info["client_id"], ADMIN)
            client.headers.update(bearer(tokens["access_token"]))
            result = (
                await rpc(
                    client, "office", "tools/call", {"name": "listCustomers", "arguments": {}}
                )
            )["result"]
            assert result["isError"] is False
    status, whoami, resolve_status, admin_status = probes[0]
    assert status == 200
    assert whoami == {
        "actor": {"name": "Synthetic MCP actor", "role": "floor"},
        "key_kind": "actor",
    }
    assert resolve_status == 200  # PR #66 allows named-actor product resolution.
    assert admin_status == 403  # The actor still has no master-key reach or fallback.
    after = db.snapshot()
    changed = {table for table in before if before[table] != after[table]}
    assert changed <= {"actors"}
    if changed:

        def strip(rows):
            return [{k: v for k, v in row.items() if k != "last_used_at"} for row in rows]

        assert strip(before["actors"]) == strip(after["actors"])

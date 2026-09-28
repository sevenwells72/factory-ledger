"""Configuration: locked by default; loopback-only test stub; Google OAuth for hosting.

Auth modes:
  locked      every MCP request is 401; /health works (container default).
  local_stub  loopback-only bearer token for local development and tests. Refused in
              production and on Railway. With MCP_DEV_EMAIL it resolves that
              allowlisted user's real identity; without it, no identity exists.
  google      MCP OAuth 2.1 (PKCE, discovery, DCR/CIMD) with Google as the identity
              source, an email allowlist + roles from MCP_ALLOWED_USERS and one
              named-actor key per user from MCP_ACTOR_KEY_* variables.

Hosted (production or Railway) configuration fails closed: no dev token, dev email,
demo role or shared test key may be present, and only locked/google modes load.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .identity import AllowedUser, normalize_email, parse_allowlist

AUTH_MODES = ("locked", "local_stub", "google")
MIN_TOKEN_SECRET_LENGTH = 32
LOOPBACK_HOSTS = {"127.0.0.1", "::1"}
RAILWAY_INTERNAL_SUFFIX = ".railway.internal"


def _railway_detected() -> bool:
    return bool(os.getenv("RAILWAY_ENVIRONMENT_ID"))


def _reject_url_extras(url, label: str) -> None:
    if url.username or url.password or url.query or url.fragment:
        raise ValueError(f"{label} must not contain credentials, a query string or a fragment")
    if url.path not in {"", "/"}:
        raise ValueError(f"{label} must be a bare origin without a path")


@dataclass(frozen=True)
class Settings:
    environment: str = "local"
    auth_mode: str = "locked"
    dev_token: str = field(default="", repr=False)
    test_api_key: str = field(default="", repr=False)
    dev_role: str = "reader"
    ledger_url: str = "http://127.0.0.1:8100"
    # Auth workstream (google mode; dev_email is local_stub only).
    public_url: str = ""
    google_client_id: str = ""
    google_client_secret: str = field(default="", repr=False)
    google_hosted_domain: str = ""
    token_secret: str = field(default="", repr=False)
    allowed_users: Mapping[str, AllowedUser] = field(default_factory=dict, repr=False)
    dev_email: str = ""
    access_token_ttl: int = 3600
    refresh_token_ttl: int = 30 * 24 * 3600

    def __post_init__(self):
        if self.environment not in {"local", "test", "production"}:
            raise ValueError("MCP_ENV must be local, test, or production")
        if self.auth_mode not in AUTH_MODES:
            raise ValueError(f"MCP_AUTH_MODE must be one of {', '.join(AUTH_MODES)}")
        if self.dev_role not in {"admin", "floor", "reader"}:
            raise ValueError("MCP_DEV_ROLE must be admin, floor, or reader")
        object.__setattr__(self, "dev_email", normalize_email(self.dev_email))
        object.__setattr__(self, "public_url", self.public_url.strip().rstrip("/"))

        hosted = self.hosted
        if self.auth_mode == "local_stub":
            if hosted:
                raise ValueError("Local auth stub cannot run in production or on Railway")
            if len(self.dev_token) < 16:
                raise ValueError(
                    "Set MCP_DEV_TOKEN to a local-only token of at least 16 characters"
                )
        if hosted:
            # Fail closed: nothing test-only may be configured where real users connect.
            if self.dev_token or self.dev_email or self.dev_role != "reader":
                raise ValueError(
                    "MCP_DEV_TOKEN, MCP_DEV_EMAIL and MCP_DEV_ROLE are refused in production "
                    "or on Railway"
                )
            if self.test_api_key:
                raise ValueError(
                    "MCP_TEST_API_KEY is refused in production or on Railway; users are "
                    "mapped to named-actor keys and there is no shared-key fallback"
                )
        if self.dev_email:
            if self.auth_mode != "local_stub":
                raise ValueError("MCP_DEV_EMAIL only applies to MCP_AUTH_MODE=local_stub")
            if self.dev_email not in self.allowed_users:
                raise ValueError("MCP_DEV_EMAIL must be listed in MCP_ALLOWED_USERS")
        if self.access_token_ttl < 60 or self.refresh_token_ttl < self.access_token_ttl:
            raise ValueError("Token lifetimes must be at least 60 seconds and refresh >= access")
        if self.auth_mode == "google":
            self._validate_google(hosted)
        self._validate_ledger_url(hosted)

    def _validate_google(self, hosted: bool) -> None:
        if not self.public_url:
            raise ValueError("MCP_PUBLIC_URL is required for Google OAuth")
        url = urlsplit(self.public_url)
        loopback = url.hostname in LOOPBACK_HOSTS or url.hostname == "localhost"
        if url.scheme != "https" and not (url.scheme == "http" and loopback and not hosted):
            raise ValueError("MCP_PUBLIC_URL must be an https origin (http only on loopback)")
        if not url.hostname:
            raise ValueError("MCP_PUBLIC_URL must include a host")
        _reject_url_extras(url, "MCP_PUBLIC_URL")
        if not self.google_client_id.strip() or not self.google_client_secret.strip():
            raise ValueError("MCP_GOOGLE_CLIENT_ID and MCP_GOOGLE_CLIENT_SECRET are required")
        if len(self.token_secret) < MIN_TOKEN_SECRET_LENGTH:
            raise ValueError(
                f"MCP_TOKEN_SECRET must be at least {MIN_TOKEN_SECRET_LENGTH} random characters"
            )
        if not self.allowed_users:
            raise ValueError("MCP_ALLOWED_USERS must list at least one user for Google OAuth")

    def _validate_ledger_url(self, hosted: bool) -> None:
        url = urlsplit(self.ledger_url)
        remote_ok = hosted and self.auth_mode == "google"
        if remote_ok and url.hostname and url.hostname not in LOOPBACK_HOSTS:
            # Hosted service reaching the real ledger: public HTTPS or Railway private network.
            private = url.scheme == "http" and url.hostname.endswith(RAILWAY_INTERNAL_SUFFIX)
            if url.scheme != "https" and not private:
                raise ValueError(
                    "MCP_LEDGER_API_URL must be https:// or an http://*.railway.internal origin"
                )
            _reject_url_extras(url, "MCP_LEDGER_API_URL")
            return
        if (
            url.scheme != "http"
            or url.hostname not in LOOPBACK_HOSTS
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Phase 1 requires a numeric loopback HTTP ledger URL; remote APIs blocked"
            )

    @property
    def hosted(self) -> bool:
        return self.environment == "production" or _railway_detected()

    @property
    def public_host(self) -> str:
        """Host[:port] of MCP_PUBLIC_URL, for transport host allowlists."""
        return urlsplit(self.public_url).netloc if self.public_url else ""

    def resource_url(self, group: str) -> str:
        return f"{self.public_url}/{group}/mcp"

    @classmethod
    def from_env(cls):
        env = os.environ
        return cls(
            environment=env.get("MCP_ENV", "local"),
            auth_mode=env.get("MCP_AUTH_MODE", "locked"),
            dev_token=env.get("MCP_DEV_TOKEN", ""),
            test_api_key=env.get("MCP_TEST_API_KEY", ""),
            dev_role=env.get("MCP_DEV_ROLE", "reader"),
            ledger_url=env.get("MCP_LEDGER_API_URL", "http://127.0.0.1:8100"),
            public_url=env.get("MCP_PUBLIC_URL", ""),
            google_client_id=env.get("MCP_GOOGLE_CLIENT_ID", ""),
            google_client_secret=env.get("MCP_GOOGLE_CLIENT_SECRET", ""),
            google_hosted_domain=env.get("MCP_GOOGLE_HOSTED_DOMAIN", "").strip().lower(),
            token_secret=env.get("MCP_TOKEN_SECRET", ""),
            allowed_users=parse_allowlist(env.get("MCP_ALLOWED_USERS"), env),
            dev_email=env.get("MCP_DEV_EMAIL", ""),
            access_token_ttl=int(env.get("MCP_ACCESS_TOKEN_TTL", "3600")),
            refresh_token_ttl=int(env.get("MCP_REFRESH_TOKEN_TTL", str(30 * 24 * 3600))),
        )

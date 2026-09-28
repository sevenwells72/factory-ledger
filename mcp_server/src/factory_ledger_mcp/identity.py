"""Shared auth contract: resolved identity, role matrix and the env-managed allowlist.

`current_identity()` reads a request-scoped context variable that only the auth
gate in `auth.py` sets, and only after a verified Google sign-in (or the loopback
test stub) resolved an email against the allowlist. No environment value, tool
argument or demo role authenticates a user. Tests may monkeypatch
`factory_ledger_mcp.identity.current_identity`; call it through the module.

Allowlist format (`MCP_ALLOWED_USERS`, JSON, no emails in code):

    [
      {"email": "person@example.com",
       "role": "admin_office_floor",
       "actor_key_env": "MCP_ACTOR_KEY_PERSON"}
    ]

Each `actor_key_env` names a separate environment variable holding that person's
existing Factory Ledger named-actor API key (actors table, FR-15). The variable
name must start with `MCP_ACTOR_KEY_` so an entry can never point at the shared
`API_KEY`/`DASHBOARD_API_KEY`. Any malformed entry fails the whole load (closed).
"""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["admin_office_floor", "floor"]
ROLES: tuple[str, ...] = ("admin_office_floor", "floor")
ACTOR_KEY_ENV_PREFIX = "MCP_ACTOR_KEY_"
SHARED_KEY_ENV_NAMES = ("API_KEY", "DASHBOARD_API_KEY", "MCP_TEST_API_KEY")
MAX_ALLOWLIST_BYTES = 64_000

# Scopes advertised to OAuth clients. Permission checks use the role helpers below;
# scopes are derived from the role at verification time, never stored in a token.
SCOPES_BY_ROLE: dict[str, tuple[str, ...]] = {
    "admin_office_floor": ("office.read", "office.write", "floor.read", "floor.write"),
    "floor": ("office.read", "floor.read", "floor.write"),
}
ALL_SCOPES: tuple[str, ...] = SCOPES_BY_ROLE["admin_office_floor"]


@dataclass(frozen=True)
class Identity:
    email: str
    role: Literal["admin_office_floor", "floor"]
    actor_key: str = field(repr=False)


@dataclass(frozen=True)
class AllowedUser:
    email: str
    role: Role
    actor_key: str = field(repr=False)

    def identity(self) -> Identity:
        return Identity(self.email, self.role, self.actor_key)


def normalize_email(email: str) -> str:
    return email.strip().lower()


class AllowlistError(ValueError):
    """Malformed allowlist or actor-key configuration; the service must not start."""


def parse_allowlist(raw: str | None, environ: Mapping[str, str]) -> dict[str, AllowedUser]:
    """Parse MCP_ALLOWED_USERS and resolve every actor key from `environ`, or raise.

    Returns a mapping keyed by normalized email. An empty/blank value yields an
    empty allowlist (every sign-in denied); the caller decides whether that is
    acceptable for its auth mode.
    """
    if raw is None or not raw.strip():
        return {}
    if len(raw.encode()) > MAX_ALLOWLIST_BYTES:
        raise AllowlistError("MCP_ALLOWED_USERS is unreasonably large")
    try:
        entries = json.loads(raw)
    except ValueError as exc:
        raise AllowlistError("MCP_ALLOWED_USERS is not valid JSON") from exc
    if not isinstance(entries, list):
        raise AllowlistError("MCP_ALLOWED_USERS must be a JSON list of user objects")

    shared_keys = {
        environ[name].strip() for name in SHARED_KEY_ENV_NAMES if environ.get(name, "").strip()
    }
    users: dict[str, AllowedUser] = {}
    seen_keys: dict[str, str] = {}
    for index, entry in enumerate(entries):
        where = f"MCP_ALLOWED_USERS[{index}]"
        if not isinstance(entry, dict):
            raise AllowlistError(f"{where} must be an object")
        extra = set(entry) - {"email", "role", "actor_key_env"}
        if extra:
            raise AllowlistError(f"{where} has unsupported fields: {', '.join(sorted(extra))}")
        email = entry.get("email")
        role = entry.get("role")
        key_env = entry.get("actor_key_env")
        if (
            not isinstance(email, str)
            or "@" not in email
            or any(c.isspace() for c in email.strip())
        ):
            raise AllowlistError(f"{where}.email must be an email address")
        email = normalize_email(email)
        if email in users:
            raise AllowlistError(f"{where}.email is listed more than once")
        if role not in ROLES:
            raise AllowlistError(f"{where}.role must be one of {', '.join(ROLES)}")
        if (
            not isinstance(key_env, str)
            or not key_env.startswith(ACTOR_KEY_ENV_PREFIX)
            or len(key_env) <= len(ACTOR_KEY_ENV_PREFIX)
            or not key_env.isupper()
            or not key_env.replace("_", "").isalnum()
        ):
            raise AllowlistError(
                f"{where}.actor_key_env must name a variable starting with {ACTOR_KEY_ENV_PREFIX}"
            )
        actor_key = environ.get(key_env, "").strip()
        if not actor_key:
            raise AllowlistError(f"{where}: environment variable {key_env} is missing or empty")
        if actor_key in shared_keys:
            raise AllowlistError(f"{where}: {key_env} must be a named-actor key, not a shared key")
        if actor_key in seen_keys:
            raise AllowlistError(f"{where}: {key_env} duplicates {seen_keys[actor_key]}")
        seen_keys[actor_key] = key_env
        users[email] = AllowedUser(email=email, role=role, actor_key=actor_key)
    return users


def resolve_allowed_user(
    allowlist: Mapping[str, AllowedUser], email: str | None
) -> AllowedUser | None:
    """Look a verified email up in the allowlist. Unlisted or blank means None."""
    if not email:
        return None
    return allowlist.get(normalize_email(email))


def scopes_for(identity: Identity | None) -> list[str]:
    if identity is None:
        return []
    return list(SCOPES_BY_ROLE.get(identity.role, ()))


_current: ContextVar[Identity | None] = ContextVar("factory_ledger_identity", default=None)


def current_identity() -> Identity:
    """The verified, allowlisted identity for the request being served.

    Raises PermissionError outside an authenticated request, including for
    unauthenticated or unlisted callers. There is no default user and TEST-ONLY
    demo settings never authenticate anyone.
    """
    identity = _current.get()
    if identity is None:
        raise PermissionError(
            "No authenticated, allowlisted identity in this request scope "
            "(TEST-ONLY demo settings never authenticate)"
        )
    return identity


@contextmanager
def identity_scope(identity: Identity) -> Iterator[Identity]:
    """Bind `identity` to the current async context for the duration of one request."""
    if not isinstance(identity, Identity):
        raise TypeError("identity_scope requires a resolved Identity")
    token = _current.set(identity)
    try:
        yield identity
    finally:
        _current.reset(token)


def can_write_office(identity: Identity | None) -> bool:
    return identity is not None and identity.role == "admin_office_floor"


def can_write_floor(identity: Identity | None) -> bool:
    return identity is not None and identity.role in {"admin_office_floor", "floor"}


def can_read_office(identity: Identity | None) -> bool:
    return identity is not None and identity.role in {"admin_office_floor", "floor"}


def can_read_floor(identity: Identity | None) -> bool:
    return identity is not None and identity.role in {"admin_office_floor", "floor"}


def can_read_group(identity: Identity | None, group: str) -> bool:
    if group == "office":
        return can_read_office(identity)
    if group == "floor":
        return can_read_floor(identity)
    return False

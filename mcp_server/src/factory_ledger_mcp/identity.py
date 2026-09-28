"""Shared auth contract. Identity resolution is a TEST-ONLY stub, not Google auth.

The auth workstream must replace current_identity with verified, request-scoped
resolution. Tests may monkeypatch it; deployed code must never install a test user.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Identity:
    email: str
    role: Literal["admin_office_floor", "floor"]
    actor_key: str = field(repr=False)


def current_identity() -> Identity:
    """TEST-ONLY STUB: deny until authenticated, allowlisted identity is implemented.

    Future implementation raises PermissionError for unauthenticated or unlisted
    callers. No environment value, tool argument or demo role authenticates a user.
    """
    raise PermissionError("TEST-ONLY identity stub: no authenticated, allowlisted user")


def can_write_office(identity: Identity | None) -> bool:
    return identity is not None and identity.role == "admin_office_floor"


def can_write_floor(identity: Identity | None) -> bool:
    return identity is not None and identity.role in {"admin_office_floor", "floor"}


def can_read_office(identity: Identity | None) -> bool:
    return identity is not None and identity.role in {"admin_office_floor", "floor"}


def can_read_floor(identity: Identity | None) -> bool:
    return identity is not None and identity.role in {"admin_office_floor", "floor"}

import pytest

from factory_ledger_mcp.identity import (
    Identity,
    can_read_floor,
    can_read_office,
    can_write_floor,
    can_write_office,
    current_identity,
)


@pytest.mark.parametrize(
    "role,permissions",
    [
        ("admin_office_floor", (True, True, True, True)),
        ("floor", (False, True, True, True)),
        (None, (False, False, False, False)),
        ("reader", (False, False, False, False)),
        ("admin", (False, False, False, False)),
    ],
)
def test_decided_role_matrix_fails_closed_for_missing_or_legacy_roles(role, permissions):
    # Invalid legacy roles deliberately probe the runtime fail-closed boundary.
    identity = Identity("synthetic@example.invalid", role, "test-only-key") if role else None
    assert (
        can_write_office(identity),
        can_write_floor(identity),
        can_read_office(identity),
        can_read_floor(identity),
    ) == permissions


def test_identity_stub_never_trusts_demo_configuration(monkeypatch):
    monkeypatch.setenv("MCP_ENV", "test")
    monkeypatch.setenv("MCP_AUTH_MODE", "local_stub")
    monkeypatch.setenv("MCP_DEV_ROLE", "admin")
    monkeypatch.setenv("MCP_TEST_API_KEY", "test-only-key")
    with pytest.raises(PermissionError, match="TEST-ONLY"):
        current_identity()


def test_identity_repr_does_not_expose_backend_key():
    identity = Identity("synthetic@example.invalid", "floor", "private-test-actor-key")
    assert identity.actor_key == "private-test-actor-key"
    assert "private-test-actor-key" not in repr(identity)

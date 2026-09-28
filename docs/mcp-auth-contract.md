# MCP identity contract — 2026-09-28

This is the shared interface for `feature/mcp-auth` and `feature/mcp-writes`.
It is **local only — not deployed**. Google sign-in and write tools are not implemented
by this contract commit. `main.py` and existing OpenAPI YAML remain unchanged.

## Interface

`mcp_server/src/factory_ledger_mcp/identity.py` defines:

```python
@dataclass(frozen=True)
class Identity:
    email: str
    role: Literal["admin_office_floor", "floor"]
    actor_key: str

def current_identity() -> Identity: ...

def can_write_office(identity: Identity | None) -> bool: ...
def can_write_floor(identity: Identity | None) -> bool: ...
def can_read_office(identity: Identity | None) -> bool: ...
def can_read_floor(identity: Identity | None) -> bool: ...
```

| Identity role | Office read | Office write | Floor read | Floor write |
|---|---|---|---|---|
| `admin_office_floor` | Yes | Yes | Yes | Yes |
| `floor` | Yes, including office orders | No | Yes | Yes |
| Missing / unlisted / unknown role | No | No | No | No |

Michael, Luz and Miriam are assigned `admin_office_floor`; allowlisted floor users
are assigned `floor`. No real email addresses or credentials are stored in code.
The dataclass describes a resolved, trusted identity; constructing it is not an
authentication step. Helpers only check role permissions and grant nothing for
`None` or unknown legacy labels such as `admin`/`reader`.

## TEST-ONLY stub — deliberately denies every call

`current_identity()` is clearly marked **TEST-ONLY** and currently always raises
`PermissionError`. It reads no environment variable, header or tool argument and
returns no default user. Tests in the write workstream may monkeypatch
`factory_ledger_mcp.identity.current_identity` to return an explicit synthetic
identity; always restore the patch after the test. Prefer importing the module
(`from factory_ledger_mcp import identity`) and calling `identity.current_identity()`
so the resolver has one patchable seam.

There is no runtime test-user setter, fallback identity, global mutable principal,
or production bypass. The existing `DevelopmentAuth` bearer stub remains separate;
its older demo labels and group boundary do not implement this decided role matrix.
The helpers express future permissions and do not register or enable write tools.

## Auth workstream responsibilities

Replace only the resolver implementation while preserving the interface above:

1. Verify Google sign-in through the chosen MCP-compatible OAuth flow and validate
   issuer, audience, expiry and verified email. Never trust caller-supplied email,
   role or actor fields as authentication.
2. Resolve the verified email using the **email allowlist + role mapping from a
   Railway environment variable**. Unlisted users are denied even if Google sign-in
   succeeds or they belong to the company Workspace. Missing/malformed role settings
   fail closed. The variable name/format is to be defined and documented by the auth
   workstream; no deployment variable is created by this commit.
3. Resolve that user's server-side backend actor key without logging it or sending
   it to the model. It is excluded from the dataclass representation. The key is an
   upstream credential, not proof of Google identity; backend route restrictions
   still apply. Phase 1's integration tests show actor keys cannot call
   `resolveProducts`; do not silently fall back to a broader master key.
4. Store the resolved identity in request-scoped context, isolate concurrent calls
   and clear it on completion. `current_identity()` returns `Identity` only for a
   verified, allowlisted caller with a valid role/key mapping, otherwise it raises
   `PermissionError` (including unauthenticated or unlisted callers).
5. Map missing/invalid authentication to HTTP 401 and authenticated-but-forbidden
   access to 403 at the transport boundary. Enforce permissions on both tool listing
   and invocation. Floor users must receive office reads; office writes require
   `admin_office_floor`.

## Write workstream responsibilities

Resolve the identity once per request, check the relevant helper and use its
server-side actor key. Do not trust tool arguments for identity. Every proposed write
returns a plain-English summary and a one-time confirmation token. Save only on a
second call with that token, after verifying identity, role, expiry, exact payload,
record versions and non-reuse. Display **"No PO"** for allowed orders without a PO,
leaving the dedicated `customer_po` field empty. The full decisions and receipt
requirements are in `mcp-migration-plan.md`.

## Shared real-ledger test harness

Reuse `mcp_server/tests/ledger_harness.py` and the `ledger` / `real_stack` fixtures in
`mcp_server/tests/conftest.py`. Setup instructions are in `mcp_server/README.md`.
Each session owns its private PostgreSQL cluster; each test gets a fresh database
from `tests/schema/schema.sql`, real `main.py` handlers/auth and synthetic keys.
The backend runs in its separate pinned dependency environment. Parallel worktrees
must create their own `.venv-ledger-test` or set `MCP_TEST_LEDGER_PYTHON` explicitly;
no shared database URL or production secret is needed or accepted.

Snapshots compare all public tables and sequences. For reads, only
`actors.last_used_at` usage metadata may change; ledger, order, inventory, lot and
product data must not. Auth and write tests should preserve this baseline while
adding request isolation, role denial, approval expiry/replay and business receipts.

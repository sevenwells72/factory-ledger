# Factory Ledger M0: read-only ChatGPT connector

This is a separate Python service. It translates 17 MCP read tools into calls to
Factory Ledger's existing HTTP endpoints. FL supplies the results and business
rules. The adapter has no database credentials, ledger implementation, write
execution code, confirmation store or production mode.

Michael's setup and activation checklist is in
[the staging setup guide](../docs/mcp-staging-setup.md).

## Safety contract

- `MCP_WRITES_ENABLED=false` is the default and only accepted value. All write
  tools are absent; calls to historical write names return `write_blocked` before
  any FL request. Unknown names never reach FL. Enabling writes requires a future
  reviewed implementation; changing an environment variable cannot enable them.
- `/mcp` exposes all **17** unique read tools. `/office/mcp` (14) and `/floor/mcp`
  (12) retain the reusable catalogs; they overlap. Every tool has `readOnlyHint`
  and only read scopes are issued or advertised.
- The only POST reads are `/products/resolve` and the dedicated
  `/sales/orders/{order_id}/ship/preview`. `shipOrder` means **preview only**;
  `mode=commit`, write approval fields and unknown arguments are rejected. The
  server also sets `mode=preview` before calling FL's preview wrapper.
- Google verifies the user's identity, then the exact email must be allowlisted.
  Each entry pins an independently checked FL actor ID, exact name and role and
  points to that person's own `MCP_ACTOR_KEY_*` secret. Before serving requests,
  every key must pass `GET /auth/whoami` with exactly those actor fields and
  `key_kind=actor`. Wrong, shared, inactive, duplicate or unavailable mappings
  stop startup. FL currently has no email field: the operator must verify the
  email-to-person association when setting these pins.
- Every FL call carries only the signed-in person's key. The MCP bearer token,
  client default headers, cookies and shared API keys are never forwarded.
- Hosted settings require `MCP_ENV=staging` and the exact FL staging origin
  `https://fastapi-staging-production-dd7b.up.railway.app`. Other remote origins,
  local stubs on Railway, and `MCP_ENV=production` are refused.
- Locked mode serves health only and denies every MCP request. There is no
  anonymous or shared-key fallback if Google or actor setup is incomplete.
- Responses are bounded at 2 MB. Redirects, non-JSON responses and upstream 5xx
  details are refused. Timeouts do not retry. Valid FL results and business
  refusals are relayed without interpretation or new business rules.

## OAuth and runtime

The inherited Google OAuth provider is ported from `feature/mcp-auth` through
`integration/mcp-with-66`; read routing/catalogs come from `feature/mcp-server`.
The old write implementation and stale write tests are deliberately excluded.
Catalogs are regenerated from current main's existing GPT schemas, with the
explicit lot disambiguation and shipment-preview overrides in `build_catalog.py`.
No backend routes, migrations, dashboard or GPT configuration were changed.

The service is ChatGPT's authorization server; Google is its identity provider.
It supports OAuth authorization code + PKCE S256, dynamic client registration
(DCR), issuer/resource discovery, refresh rotation and revocation. Google ID
JWTs are checked for signature, issuer, audience, expiry, nonce and verified
email. DCR accepts only documented HTTPS ChatGPT callback paths; arbitrary
client metadata URL fetches are disabled. The Google callback is
`/oauth/google/callback`, distinct from ChatGPT's callback.

Run **one process / one replica**. Login and replay/revocation state is bounded
in memory. Client registration survives restarts while access/refresh tokens
and codes do not: users must reconnect after a deploy/restart. This deliberate
M0 limitation avoids accepting replayed/revoked tokens after state loss without
introducing another database. Rotate `MCP_TOKEN_SECRET` to invalidate client
registrations too. Removing an allowlist entry and redeploying removes access.
Uvicorn access logging is disabled because OAuth callbacks contain secret codes.
Do not enable HTTP debug logging or copy Railway variable values into logs.

## Development and tests

Python 3.12 and PostgreSQL 17 are required. MCP uses its own locked modern
runtime; the unchanged FL backend uses its existing legacy dependency versions.
Do not install these two environments into each other.

```sh
uv sync --project mcp_server --python python3.12 --frozen
uv venv --python python3.12 .venv-test
uv pip install --python .venv-test/bin/python -r requirements.txt -r tests/requirements-test.txt
mcp_server/.venv/bin/ruff check mcp_server
mcp_server/.venv/bin/python mcp_server/scripts/full_suite.py
```

The runner checks for at least 5 GB free, runs the full backend suite on a fresh
private local cluster with explicit America/New_York session time, drops its
regular DB before migration/race tests allocate scratch DBs, then runs MCP tests
and Node tests. It finishes independent phases after a test failure and exits
nonzero if any phase failed. No tests are skipped by the partition: both regular
and scratch partitions must pass. Existing main fixtures have time-sensitive
failures near plant midnight; the staging report records the actual results.
There is at most one populated throwaway DB at a time; it is removed immediately
when its phase/test finishes, and the private cluster is removed on exit. Child
processes inherit no API keys, database URLs or integration credentials.

The separate offline count-tool suite is outside root `pytest.ini`'s `tests/`
collection. Run it explicitly with `--suite audits`. At the M0 base `d5e9fea`, it
has the existing UTC-versus-New-York fixture date failure tracked in
[PR #112](https://github.com/sevenwells72/factory-ledger/pull/112). This branch does
not incorporate that independent pending change; see the staging report for the
actual audit result.

MCP integration tests launch the **real current-main FL app** in a separate
process, use synthetic named actors, exercise all 17 reads and compare every
public table and sequence before/after. Only FL's `actors.last_used_at` audit
stamp may change. OAuth tests use a local fake Google signing key, never live
Google accounts. Other tests cover swapped/shared actor keys, read scopes,
write refusal even for owners, path injection, invalid shipment quantities,
credential isolation, redirects, token replay, revocation and restart behavior.

Regenerate the read catalog after inspecting backend/schema changes:

```sh
mcp_server/.venv/bin/python mcp_server/scripts/build_catalog.py
```

For a synthetic demonstration only, use `.env.example`'s commented local settings
and run `factory-ledger-demo` plus `factory-ledger-mcp` from the MCP environment.
The demo is never mounted or imported by the hosted server.

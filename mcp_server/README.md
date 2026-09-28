# Factory Ledger MCP — Phase 1

One independent service with two read-only MCP endpoints:

| Group | Local URL | Tools |
|---|---|---:|
| Office | `http://127.0.0.1:8000/office/mcp` | 14 |
| Floor | `http://127.0.0.1:8000/floor/mcp` | 12 |

This replaces the read surfaces of **Factory-Ledger 2.0** and **Factory Ledger - Floor 2.0**.
The Floor count includes shipping preview. No write operation is registered, even for admin.
The complete inventories, retired schemas, decisions, estimate and rollout are in
[`../docs/mcp-migration-plan.md`](../docs/mcp-migration-plan.md).

## Run locally with synthetic data

Prerequisites: Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).
Run commands from this directory (`mcp_server/`), in a separate environment from the legacy app.

```sh
uv sync --locked --no-editable
uv run --locked --no-editable factory-ledger-demo
```

The first terminal serves **synthetic response fixtures in an in-memory SQLite database**
on port 8100. It is a routing/contract demonstration, not a replacement ledger or a
simulation of production queries, FIFO, calculations or fuzzy matching. Its data never
persists, its database is set query-only after seeding, and it has no write routes.

In a second terminal, also in `mcp_server/`:

```sh
set -a
. ./.env.example
set +a
uv run --locked --no-editable factory-ledger-mcp
```

This explicitly opts into the loopback-only development auth stub. It does not sign in
as Michael or any other real person. The sample token is synthetic, not a production secret.
Use `MCP_DEV_ROLE=floor` to simulate no office access; `reader` and `admin` can read both
groups. All three roles remain read-only in Phase 1.

In a third terminal:

```sh
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/floor/mcp \
  -H 'Authorization: Bearer local-only-demo-token' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
curl -s http://127.0.0.1:8000/office/mcp \
  -H 'Authorization: Bearer local-only-demo-token' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"getOrder","arguments":{"order_id":"SO-DEMO-001"}}}'
```

Normal MCP clients should initialize before listing/calling tools; tests exercise that
handshake. For a local MCP Inspector, use either endpoint above and the same bearer token.
Stop each service with Ctrl-C. No OpenAI/Anthropic API key is needed.

To use a separately prepared **local ledger with a test database**, set
`MCP_LEDGER_API_URL=http://127.0.0.1:<test-port>` and bind one synthetic allowlisted user
for the stub session: `MCP_ALLOWED_USERS` (one entry whose `actor_key_env` names an
`MCP_ACTOR_KEY_*` variable holding that test ledger's named-actor key) plus
`MCP_DEV_EMAIL` set to that entry's email. **There is no shared backend key.** Every
ledger request, read or write, carries only the calling user's own named-actor key;
`MCP_TEST_API_KEY` was removed in Phase 2 and the service refuses to start if it is set.
Never start the root app against a production environment: its startup can run
migrations. The integration harness below starts it only inside a disposable test database.
Remote URLs, host aliases, redirects and environment proxies are blocked. No production
URL or credential is packaged; the generator ignores the schemas' server/auth settings.

## Verification

The suite keeps the synthetic routing/validation tests and adds real ledger integration
checks. The latter run unmodified `main.py` handlers, startup and authentication in a
separate process using the root application's pinned requirements, with a disposable
PostgreSQL database loaded from `../tests/schema/schema.sql`. Missing prerequisites
fail the integration tests; they are not silently skipped.

```sh
# One-time test-only backend environment, separate from the MCP environment:
uv venv --python 3.12 .venv-ledger-test
uv pip install --python .venv-ledger-test/bin/python -r ../requirements.txt
# PostgreSQL 17 binaries must be on PATH (Homebrew's standard path is also detected).
uv run --locked --no-editable pytest -q
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
uv build
```

`tests/ledger_harness.py` / `conftest.py` expose reusable `ledger` and `real_stack`
fixtures for the auth and write worktrees. Each session creates its own private cluster
with a Unix socket and no PostgreSQL TCP listener. Each test clones a fresh database,
starts the real ledger on a dynamically bound loopback socket, seeds synthetic data,
and stops the process/drops the database afterward. Inherited database URLs and secrets
are not forwarded. `MCP_TEST_PG_BIN` can select PostgreSQL's binary directory;
`MCP_TEST_LEDGER_PYTHON` can select the isolated backend interpreter. Neither option
selects a database. Concurrent worktrees have separate clusters and ports.

Integration checks exercise blank/wrong mapped actor keys (401/403), real shipment
quantities (10 lb on one line versus 100 lb on each of two), conflicting-input rejection,
409 lot ambiguity and `product_id` disambiguation in both groups. A fixed set of 26 read
scenarios compares every public table, row, column and sequence before/after calls.
Actor-key reads may update only `actors.last_used_at`. The real backend still forbids
actor keys on `resolveProducts` (403) and this service never falls back to a shared key,
so that office tool stays unavailable until the backend allowlists the route for actors.
No auth dependency or business handler is replaced.

Phase 2 wiring tests (`tests/test_phase2_wiring.py`) run the real `create_app` in google
mode with an in-process fake Google and a recording fake ledger: OAuth discovery and
protected-resource routes are mounted, both group endpoints challenge with
`resource_metadata`, the `MCP_PUBLIC_URL` host is accepted while foreign `Host` headers get
421, a shared key configured on the HTTP client is never sent (reads or writes), and two
users calling concurrently never receive each other's actor key.

`tests/test_named_actor_pending.py` probes the 14 backend write routes that still reject
named-actor keys. They are marked expected-fail with the reason
"blocked until PR #66 (fix/named-actor-writes) is merged; flips to pass after rebasing
onto main"; nothing from that branch is merged here. After the rebase, drop the markers
and flip `named_actor_allowed` in `catalog.json` for the same routes.
The existing synthetic tests still cover MCP handshake, registration/routing, validation,
group boundaries, unavailable writes, error preservation, size limits and retry policy.

Google OAuth, live client installation, production data, container execution and hosted
Railway behavior remain untested. This is **local only — not deployed**.

`--no-editable` also avoids a macOS hidden `.pth` file issue in this Documents workspace.
Source files are cache keys so `uv` rebuilds the local package after edits. Production
dependencies are locked separately from the old FastAPI application. The SDK is deliberately
bounded to its maintained 1.x API; the lock resolves `mcp==1.30.0`.

## Tool contracts

`scripts/build_catalog.py` reads the two existing YAML files, expands local schema references,
and writes `src/factory_ledger_mcp/catalog.json`. It never edits an existing schema. Runtime
loads only this read catalog; new writes cannot become tools merely by appearing in YAML.
To deliberately refresh it, run `uv run --locked --no-editable python scripts/build_catalog.py`
and review the diff. The source-plus-overrides parity test detects stale catalogs.

Shared names stay unchanged within each group. Optional filters, enum/limit bounds and
bilingual responses are retained. Floor `shipOrder` requires `mode: preview` and calls
the existing `/sales/orders/{order_id}/ship/preview` backend wrapper, which also forces
preview. Office `resolveProducts` is the only other allowed POST. Mixed preview/commit
operations such as `make`, `receive` and office `shipOrder` are excluded entirely.
MCP-only contract overrides are deliberate and leave all OpenAPI YAML unchanged:

- Both `getLotByCode` tools accept optional integer `product_id`, passed as a query
  parameter to resolve the backend's `409 ambiguous_lot_code` response. This fills the
  office schema omission and documents the existing floor argument.
- Floor `shipOrder` rejects `ship_all=true` whenever `lines` is supplied. Explicit
  `lines` must be nonempty; otherwise the backend treats an empty list as all lines.
  Omit `ship_all` or set it false for explicit quantities. Parameter descriptions do
  not direct callers to an unavailable commit tool.

All tool annotations are read-only, non-destructive and limited to the ledger.
Here **read-only** means no ledger, order, inventory, lot, or product changes;
`actors.last_used_at` usage-metadata updates are an owner-approved exception.

## Railway scaffold — not deployed

The Dockerfile, independent lockfile, `PORT` start command and `/health` endpoint are ready
for a separate service build. Docker image execution has not been verified here because
Docker is unavailable. No Railway project command or deployment was run (only local CLI help/version).

The project-level [`../.railway/railway.ts`](../.railway/railway.ts) declares only
`factory-ledger-mcp`, using a named IaC partial so existing services remain outside
its ownership. No existing service settings are recreated or guessed; root
`railway.json` is byte-identical. The new service has root `/mcp_server`, a Dockerfile
build, start command `/app/.venv/bin/factory-ledger-mcp`, `/health` with a 30-second
timeout, restart on failure with at most three retries, and repository-absolute watch
paths `/mcp_server/**` and `/.railway/railway.ts`. The scaffold defines no source
repository/branch, domain, database or secrets. A later authorized setup must supply
those and review the scoped plan before applying anything.

Railway [deprecated JSON/TOML Config as Code](https://docs.railway.com/config-as-code):
new services cannot opt in and existing users face a December 1, 2026 hard cutoff.
The obsolete MCP-specific JSON scaffold was removed. The legacy service's migration
is separate work; it is not changed by this branch. The new declaration follows the
[official IaC API](https://docs.railway.com/infrastructure-as-code/reference) and was
locally type-checked/evaluated with the official `railway@3.11.0` SDK. No Railway
project was queried, planned, changed or deployed; remote reconciliation is unverified.

The container image defaults to `MCP_ENV=production` and `MCP_AUTH_MODE=locked`: health
checks work, but both MCP endpoints return 401 until the dashboard sets
`MCP_AUTH_MODE=google`. The stub refuses production and Railway even if explicitly
requested. This is build scaffolding, **not a deployed connector**.

Phase 2 declares the service's variables in `railway.ts` as dashboard-managed
`preserve()` references, so the file never carries a value: `MCP_AUTH_MODE`,
`MCP_PUBLIC_URL`, `MCP_GOOGLE_CLIENT_ID`, `MCP_GOOGLE_CLIENT_SECRET`, `MCP_TOKEN_SECRET`,
`MCP_ALLOWED_USERS`, `MCP_LEDGER_API_URL`, one `MCP_ACTOR_KEY_*` per allowlisted person
and the optional `MCP_GOOGLE_HOSTED_DOMAIN`. The former hard-coded `MCP_AUTH_MODE=locked`
literal is gone so it can no longer override the dashboard. The file also refuses any
linked project other than the ledger's, and remains a named partial: it can only create,
change or delete `factory-ledger-mcp`. Removing the `partial` export would turn it into a
whole-project definition whose plan proposes deleting every undeclared service; keep it.
Before any apply: run `railway config plan`, confirm the plan names only
`factory-ledger-mcp`, and never pass `--yes --confirm-destructive`.

`server.py` now builds one `Authenticator`, mounts its OAuth routes, protects both group
endpoints and adds `MCP_PUBLIC_URL`'s host to the transport allowlists. Still pending
before hosted access: creating the Google OAuth client, setting the variables above,
running the plan, and publishing two ChatGPT plugins / two Claude connector entries after
their authenticated acceptance checks. None are published or installed.

References: [OpenAI MCP servers](https://developers.openai.com/plugins/build/mcp-server),
[OpenAI authentication](https://developers.openai.com/plugins/build/auth),
[MCP Python SDK 1.x](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x),
[Railway IaC reference](https://docs.railway.com/infrastructure-as-code/reference).

Cross-review validation (2026-09-28): **98 tests passed** (91 routing/validation, 7 real-ledger integration cases); Ruff lint/format and package builds passed.

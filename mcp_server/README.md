# Factory Ledger MCP — Phase 2 integration

One independent service with two MCP endpoints and per-user backend actor keys:

| Group | Local URL | Reads/previews | Confirmed-write tools |
|---|---|---:|---:|
| Office | `http://127.0.0.1:8000/office/mcp` | 14 | 16 |
| Floor | `http://127.0.0.1:8000/floor/mcp` | 12 | 10 |

Branch `integration/mcp-with-66` combines the Phase 2 adapter at `d74f747` with
PR #66 through `1337fdd`. Named actors can use all catalog routes, including
`resolveProducts`, `createCustomer`, and `renameLot`. All 20 stale write permission
flags across 14 routes are enabled. Administrative routes remain unavailable to actors.
Every save still needs a separate preview and a one-time confirmation token, with MCP
role checks. **`createOrder` saves remain blocked by the backend contract**;
`updateOrderHeader` also blocks changes to `customer_po`. See the next backend PR below.

This replaces the surfaces of **Factory-Ledger 2.0** and **Factory Ledger - Floor 2.0**.
The Floor read count includes shipping preview; `commitShipOrder` is a separate write.
The inventories, decisions and rollout are in
[`../docs/mcp-migration-plan.md`](../docs/mcp-migration-plan.md).
This integration is **test-only, not deployed**, and has not been merged to main.

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
groups. These unbound synthetic demo roles remain read-only. A mapped test user can exercise
confirmed writes only against the disposable real ledger described below.

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
Actor-key reads may update only `actors.last_used_at`. All 26 read scenarios, including
successful `resolveProducts`, use the synthetic named-actor key and preserve business
data. No auth dependency or business handler is replaced.

Phase 2 wiring tests (`tests/test_phase2_wiring.py`) run the real `create_app` in google
mode with an in-process fake Google and a recording fake ledger: OAuth discovery and
protected-resource routes are mounted, both group endpoints challenge with
`resource_metadata`, the `MCP_PUBLIC_URL` host is accepted while foreign `Host` headers get
421, a shared key configured on the HTTP client is never sent (reads or writes), and two
users calling concurrently never receive each other's actor key.

`tests/test_named_actor_routes.py` requires the 14 PR #66 route probes and the
`createCustomer`/`renameLot` adapter commits to pass normally, with no expected-fail
markers. Invalid integer line paths and an invalid ship mode keep authorization probes
free of business mutations. Additional checks verify all 11 admin routes return 403
for floor, office, and owner actors, and that the disposable schema includes 056's table,
marker, and append-only trigger. Customer and lot commits must persist actor audit rows.
The full suite includes this real-ledger harness; no separate database URL is accepted.
Migration 056 is loaded only by `tests/schema/schema.sql` in that disposable cluster;
no standalone migration command or external database is used.
The existing synthetic tests still cover MCP handshake, registration/routing, validation,
group boundaries, unavailable writes, error preservation, size limits and retry policy.

Google OAuth, live client installation, production data, container execution and hosted
Railway behavior remain untested. This is **local only — not deployed**.

`--no-editable` also avoids a macOS hidden `.pth` file issue in this Documents workspace.
Source files are cache keys so `uv` rebuilds the local package after edits. Production
dependencies are locked separately from the old FastAPI application. The SDK is deliberately
bounded to its maintained 1.x API; the lock resolves `mcp==1.30.0`.

Integration gate (2026-09-29): **205 passed; zero failures, skips or expected-fails**. Includes 55 disposable-PostgreSQL tests (54 real-ledger HTTP tests plus the schema-056 check) and all 16 former expected-fail cases. Ruff lint/format passed. Wheel and sdist rebuilt; packaged and installed source/catalog files match the checkout. `createOrder` remains blocked and is covered by passing rejection tests.

## Tool contracts

`scripts/build_catalog.py` reads the two existing YAML files, expands local schema references,
and writes `src/factory_ledger_mcp/catalog.json`. It never edits an existing schema. Runtime
loads the reviewed read sections plus the separately reviewed `_writes` sections.
New writes cannot become tools merely by appearing in YAML. The Phase 1 generator
produces only read sections: do not overwrite the combined catalog with its CLI output.
Use its `build()` result to review read-section updates while preserving `_writes`;
the source-plus-overrides parity test detects stale read catalogs.

Shared names stay unchanged within each group. Optional filters, enum/limit bounds and
bilingual responses are retained. Floor `shipOrder` requires `mode: preview` and calls
the existing `/sales/orders/{order_id}/ship/preview` backend wrapper, which also forces
preview. Office `resolveProducts` is the other read-only POST. Write operations such
as `make`, `receive` and office `shipOrder` use the two-call confirmation flow. Floor
saves use the separate `commitShipOrder` tool.
MCP-only contract overrides are deliberate and leave all OpenAPI YAML unchanged:

- Both `getLotByCode` tools accept optional integer `product_id`, passed as a query
  parameter to resolve the backend's `409 ambiguous_lot_code` response. This fills the
  office schema omission and documents the existing floor argument.
- Floor `shipOrder` rejects `ship_all=true` whenever `lines` is supplied. Explicit
  `lines` must be nonempty; otherwise the backend treats an empty list as all lines.
  Omit `ship_all` or set it false for explicit quantities. Saving uses the separate
  confirmed `commitShipOrder` flow.

Read-tool annotations are read-only and non-destructive; write tools are annotated
separately and restricted by user role and confirmation. Here **read-only** means no
ledger, order, inventory, lot, or product changes;
`actors.last_used_at` usage-metadata updates are an owner-approved exception.

## Next backend PR: keep `createOrder` saves blocked

PR #66 fixes authorization; it does not complete `POST /sales/orders`. The adapter
continues to return an unsaved proposal and rejects commit. The next backend PR needs:

- **Dedicated PO on create:** expose `customer_po` in `OrderCreate` and pass it into
  the existing `_create_sales_order_core(customer_po=...)`. The database column already
  exists. Preserve leading zeros/punctuation, allow an empty PO, and report "No PO".
- **External reference and duplicate protection:** accept and persist
  `external_order_ref`; enforce its agreed uniqueness and normalized customer-plus-PO
  conflicts transactionally, including concurrent requests. The current endpoint does
  neither. The adapter's scan of at most 200 orders/notes cannot provide this guarantee.
  Add durable request-id/idempotency protection so a retry returns the original order
  instead of creating another one.
- **Resolved customer and products:** accept the selected `customer_id` and line
  `product_id`, validate active records, and use those exact identities. The current
  endpoint requires names and resolves them again, which cannot bind the approved IDs.
- **Pallet Charge in the same create call:** accept product **176** with `unit: each`,
  its count and unit price alongside physical products. `OrderLineInput` currently
  rejects `each` unless callers bypass quantity validation with `quantity_lb`; that
  legacy escape hatch is not an acceptable service-count contract. Persist the count
  and amount without treating pallets as physical pounds or inventory.
- **Consistent quantities and receipts:** support cases using the selected product's
  case weight (validation currently demands `case_weight_lb` before the core can look
  it up); exclude service counts from physical `total_lb`. Return/read back PO,
  external reference, product IDs, service counts, prices and amounts alongside the
  existing internal SO number, order ID, line IDs and status. Create currently omits
  several of these fields, and `getOrder` omits `customer_po` and line `product_id`.
  Preserve a zero unit price/amount as zero; current detail truthiness checks turn
  zero prices and a zero total into missing values.
- **Atomic persistence and verification:** save the header, PO/reference, physical and
  service lines, actor audit and idempotency receipt together. Roll back on any failure
  and verify the saved PO/product-176 line before reporting success. Existing order/line
  inserts already share a transaction; extend it to cover the new contract.

Related remaining restriction: `OrderHeaderUpdate` also omits `customer_po`, so MCP
continues to reject a PO edit. Other header edits remain available.

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

Historical Phase 1 cross-review validation (2026-09-28): **98 tests passed** (91 routing/validation, 7 real-ledger integration cases); Ruff lint/format and package builds passed.

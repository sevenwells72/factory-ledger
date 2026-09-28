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
`MCP_LEDGER_API_URL=http://127.0.0.1:<test-port>` and `MCP_TEST_API_KEY` to its synthetic
test key. Never start the root app against a production environment: its startup can
run migrations. This task did not import/start that app or connect to PostgreSQL.
Remote URLs, host aliases, redirects and environment proxies are blocked. No production
URL or credential is packaged; the generator ignores the schemas' server/auth settings.

## Verification

September 28, 2026: **89 tests passed**, Ruff lint/format checks passed, wheel and source
distribution built successfully. A separate local TCP smoke test used the SDK client
against both running services: initialization, 14/12 tool catalogs, office order read,
Floor shipping preview and rejected commit all passed. Both processes were stopped.

```sh
uv run --locked --no-editable pytest -q
uv run --locked --no-editable ruff check .
uv run --locked --no-editable ruff format --check .
uv build
```

Tests drive real JSON-RPC/Streamable HTTP handlers using in-process HTTP transports and
the synthetic SQLite fixture API; they do not contact the network. They cover all 26 read
registrations, all 26 deferred write registrations, authentication/group boundaries,
argument validation, preview-only routing, error preservation, response limits and
no-retry/no-redirect behavior. Fixture data is checked for changes after calls.
Backend PostgreSQL business logic and live ChatGPT/Claude installation remain untested.

`--no-editable` also avoids a macOS hidden `.pth` file issue in this Documents workspace.
Source files are cache keys so `uv` rebuilds the local package after edits. Production
dependencies are locked separately from the old FastAPI application. The SDK is deliberately
bounded to its maintained 1.x API; the lock resolves `mcp==1.30.0`.

## Tool contracts

`scripts/build_catalog.py` reads the two existing YAML files, expands local schema references,
and writes `src/factory_ledger_mcp/catalog.json`. It never edits an existing schema. Runtime
loads only this read catalog; new writes cannot become tools merely by appearing in YAML.
To deliberately refresh it, run `uv run --locked --no-editable python scripts/build_catalog.py`
and review the diff. The source-parity test detects stale catalogs.

Shared names stay unchanged within each group. Optional filters, enum/limit bounds and
bilingual responses are retained. Floor `shipOrder` requires `mode: preview` and calls
the existing `/sales/orders/{order_id}/ship/preview` backend wrapper, which also forces
preview. Office `resolveProducts` is the only other allowed POST. Mixed preview/commit
operations such as `make`, `receive` and office `shipOrder` are excluded entirely.
All tool annotations are read-only, non-destructive and limited to the ledger.

## Railway scaffold — not deployed

The Dockerfile, independent lockfile, `PORT` start command and `/health` endpoint are ready
for a separate service build. Docker image execution has not been verified here because
Docker is unavailable. No Railway command or deployment was run.

For a **future authorized** service setup, set root directory to `/mcp_server`, select
configuration file `/mcp_server/railway.json`, and keep its environment separate from
the existing ledger service. Watch patterns are repository-absolute. The root Railway
configuration, root dependencies and `main.py` are unchanged.

The container defaults to `MCP_ENV=production` and `MCP_AUTH_MODE=locked`: health checks
work, but both MCP endpoints return 401. The stub refuses production and Railway even
if explicitly requested. This is build scaffolding, **not a production-ready connector**.

Before hosted access, complete the Google OAuth TODOs in `auth.py`: compatible OAuth 2.1
authorization server, PKCE/discovery, verified CNS membership, issuer/audience/expiry,
revocation, per-user group scopes and real user assignments. Configure trusted hosted
origins/hosts and a scoped backend credential only in that later authorized phase.
Publish two ChatGPT plugins / two Claude connector entries at the respective group URLs
after their authenticated acceptance checks. None are published or installed by Phase 1.

References: [OpenAI MCP servers](https://developers.openai.com/plugins/build/mcp-server),
[OpenAI authentication](https://developers.openai.com/plugins/build/auth),
[MCP Python SDK 1.x](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x),
[Railway configuration](https://docs.railway.com/config-as-code/reference).

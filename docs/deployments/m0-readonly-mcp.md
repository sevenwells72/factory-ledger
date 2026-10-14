# M0 read-only MCP: builder handoff

Builder: Codex. Reviewer requested: **Claude Code**. Draft only; no merge or
production deployment is authorized.

## What shipped to staging

A new standalone Railway service serves the M0 adapter at
https://factory-ledger-mcp-staging-production.up.railway.app/mcp.
It is **locked pending owner setup**; it is not yet usable for authenticated
ChatGPT reads. The code and tests cover Google sign-in through the real MCP app,
but a real Google/ChatGPT login has not been claimed or performed.

- Project: `2206e070-d160-4528-a4f1-86a587ad88c3` (gleaming-solace).
- Service: `ef21f85c-e0a1-44b3-a6f0-49a238d34be1` (factory-ledger-mcp-staging).
- Environment: `f4d219df-2fea-45e8-85de-466b36a86c07`, platform label `production`;
  this is the existing staging service namespace documented in `docs/staging.md`.
  The new service has `MCP_ENV=staging` / `ENVIRONMENT=staging` and can call only
  the pinned FastAPI-staging origin. No existing service was deployed or changed.
- Deployment: `8fd85ed2-41d1-47ad-8710-429583f7b970`, **SUCCESS**.
- Source: explicit CLI upload of only `mcp_server` with `--path-as-root` and
  `--service factory-ledger-mcp-staging`; no GitHub auto-deploy link.
- Runtime source digest (sorted SHA-256 file manifest):
  `d0711a023ab112f6e0873efe1ee1c22491cf9dd58ae96dc4e5783ab2bed931eb`.
  Runtime sources were checked unchanged after the successful upload.
- Variables: only eight deliberately set non-secret settings plus Railway's own
  metadata. No Google, database, actor, shared API, or production credentials
  were copied into the new service.
- Hosted checks: `/health` 200; `writes_enabled=false`; read counts 17 combined,
  14 office, 12 floor; zero writes in all catalogs. All three MCP endpoints deny
  unauthenticated tool listing and attempted receive calls with 401. Discovery
  is intentionally 404 while locked. Evidence:
  [smoke result](m0-readonly-mcp-smoke.json).

## Implementation and provenance

Worktree: `~/dev/fl-m0-wt`, branch `feat/mcp-readonly`, based on current main
`d5e9fea` (PR #111). Ported additive source from:

- `feature/mcp-server` `d74f747d2766393c75d07e2ca295f6ada49b6c74`;
- Google auth ancestry `feature/mcp-auth` `1daa88df5299ed0c68c36697576a6802f4a9f882`;
- integrated source `integration/mcp-with-66` `2c6d625e10840959fe88ef6a443d6f4f38979c70`.

M0 contains only the 17 reusable reads, regenerated against current schemas and
verified against current FL routes/parameter names and the actual running FL
server. The server relays FL results; no ledger business logic was added. Old
write preparation/confirmation/business logic, its SQLite store and stale write
tests were omitted. A refusal list preserves safe errors for historical write
names; unknown tools cannot route anywhere. `shipOrder` is strictly the existing
read-only preview; both schema validation and FL's dedicated wrapper prohibit
commit. `MCP_WRITES_ENABLED=true` fails startup because writes are not shipped.

Google claims are verified, the email must be explicitly allowlisted, and every
personal key must pass `/auth/whoami` against its independently configured actor
ID, exact name and role **before startup succeeds**. There is no shared-key
fallback. All roles have read scopes only. OAuth supports PKCE S256, DCR,
issuer/resource discovery, refresh rotation and revocation; client redirect URIs
are restricted to ChatGPT. URL client metadata fetching is disabled. One replica
is required; tokens/codes are invalidated on restart to preserve replay/refusal
semantics with process-local state. Registered client IDs survive with the same
signing secret. OAuth query codes are excluded from access logs.

No migrations, backend application routes, dashboard, GPT configuration or
repository-root Railway configuration changed. Two existing test fixtures were
adjusted: assistant race seeding targets its own isolated DB (one-DB constraint),
and the shipment future-time test pins plant noon so the office same-day rule
does not mask its intended assertion in the six minutes before midnight.

## Validation

On 2026-10-14, the full runner completed **every configured phase**:

| Check | Final result |
|---|---|
| Backend regular partition, fresh `fl_m0_test` | 2,972 passed, 6 failed; 81 allocated to scratch |
| Backend scratch partition, one DB at a time | 81 passed; 2,978 allocated to regular |
| MCP, including real FL / all-table-and-sequence checks | 197 passed |
| Node | 107 passed, 0 failed, 0 skipped |
| Ruff and Git whitespace checks | Passed |
| Extra offline count audit (outside root collection) | 190 passed, 23 subtests passed, 2 failed |

The final configured runner exits **nonzero** for the six baseline backend
failures; this is not an all-green or merge-ready claim. An earlier fresh run
before plant midnight passed all 3,059 backend tests across the same two
partitions. The final overnight result above takes precedence. All disposable
cluster directories were removed; no test DB remains.
Tests use a private PostgreSQL 17 cluster with synthetic data and stripped child
credentials. Normal backend tests and scratch migration/race tests run in two
complementary partitions; no test is skipped. The normal DB is dropped before
scratch fixtures run. MCP integration tests create/load/drop one database each,
launch the real unchanged FL server, exercise all 17 reads, and compare every
public table and sequence; only actor `last_used_at` audit stamps may change.
Disk checks remained above 5 GB (about 16 GB free after implementation).

The separate `audits/fresh-start` suite's final two failures are
`V3Tests.test_R12_shortage_set_change_after_preview_stops_before_any_write` and
`V3Tests.test_drift_refuses_write`. Both stop at the existing “Approval date is in
the future” fixture issue, before their intended assertions. The independent
fixture fix is already open as
[PR #112](https://github.com/sevenwells72/factory-ledger/pull/112); its files were
not copied into this branch.

The overnight regular backend rerun exposes six existing same-day fixture failures
in `test_named_actor_writes` (Arturo/Luz void), `test_roles_a2` (floor void),
`test_phase1_ledger_integrity` (production calendar), and `test_void_aware_views`
(today rows, two cases). These test files, `main.py`, the schema and their shared
fixtures match the base commit. The private cluster uses explicit
`America/New_York`, as the existing receipt assertions require. Its legacy
`timestamp DEFAULT now()` fallback is interpreted as UTC by the business-time
trigger: after plant midnight this shifts synthetic rows to the previous day.
A diagnostic run with UTC sessions instead failed eight existing comparisons
of timestamp strings/offsets. No application change or skipped test masks either
issue; Claude Code should review this existing test-environment limitation.
The runner continues the remaining phases and retains a nonzero result.

## Exact remaining owner steps and choices

1. Create or supply the staging Google Web application OAuth client and enter its
   client ID/secret privately in the new service. Google callback:
   `https://factory-ledger-mcp-staging-production.up.railway.app/oauth/google/callback`.
2. Independently establish each allowed Google email's real **staging** actor ID,
   exact name, role and personal actor key. No synthetic existing key was assumed
   to identify Michael. Supply `MCP_ALLOWED_USERS` and separate `MCP_ACTOR_KEY_*`
   values; generate a fresh random `MCP_TOKEN_SECRET` (at least 32 characters).
3. Change **this service only** from `MCP_AUTH_MODE=locked` to `google` and
   redeploy one replica. Startup itself refuses incorrect/shared/swapped mappings.
4. Follow [Michael's click-by-click guide](../mcp-staging-setup.md): Business
   workspace → Plugins → **+ → Add custom MCP server** → URL → **OAuth** →
   **Create as a plugin** → allowlisted Google sign-in → install → new chat →
   **@** the staging plugin. Real sign-in and live named-actor tool acceptance
   remain to be performed after prerequisites exist.

Safest choices: locked hosting instead of invented OAuth credentials; hard
staging origin pin and no production mode; physically omit writes instead of
shipping dormant mutation code; require explicit actor pins because FL has no
email field; use DCR with ChatGPT callbacks instead of fetching arbitrary client
metadata; reconnect after restart instead of introducing durable auth storage.
Initial full checkout hit macOS `mmap: Operation canceled`; the requested
worktree was recovered using sparse checkout excluding archived screenshot and
spike directories. Only this builder used this worktree; the helper agent
researched public documentation and touched no worktrees.

Claude Code should review read refusal, startup mapping and OAuth boundaries,
current-route fidelity and the staging-only deployment evidence. Please use the
repository's prescribed review evidence template and label after reviewing the
pushed head. This builder report is **not** a review verdict.

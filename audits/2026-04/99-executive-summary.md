# Executive Summary — Factory Ledger Full-System Audit

**Date:** 2026-04-19 · **Auditor:** Opus 4.7 · **Scope:** 9,293-line monolith, 99 endpoints, 23 migrations

## Stop-the-presses issues

**None.** No active data-corruption bug, no live security exposure beyond trusted scope, no unshippable regression. The system is operationally sound; the findings are hygiene, performance, and risk-hardening. Two findings below rise above "hygiene" — they are not emergencies but should not ship another feature before being addressed.

## Top 3 to address this week

### 1. Fix `ship_order` missing `is_service` guard → [F05-04](15-traceability-gaps.md)
**Effort: small (~1 hour).** Orders containing a Pallet Charge (or any `is_service=true` line) can never transition to `status='shipped'` via `shipOrder`. They stick at `partial_ship` forever. Every other order endpoint handles service items correctly; only the primary ship commit path is missing the guard ([main.py:5711–5787](../../main.py#L5711)). This is the root cause behind the pattern of one-shot "close SO-XXX" migrations (022, 024, 017-like fixes) — operators write SQL to force-close orders the API won't close.

Fix: when selecting lines to ship, either auto-fulfill service lines (set `quantity_shipped_lb = quantity_lb`, `line_status='fulfilled'` without any inventory lookup) or exclude them from `all_fully_shipped`. Matches the `is_service` filtering already in `create_sales_order` at [main.py:4883–4908](../../main.py#L4883).

### 2. Extract startup-inline DDL → proper migration files → [F17-01](17-migration-integrity.md)
**Effort: medium (half day).** "Migrations 004–012" currently exist as inline DDL inside `startup()` at [main.py:170–480](../../main.py#L170). They run on every Railway cold start. They are schema changes (ALTER TABLE, CREATE TABLE, CREATE INDEX) that are NOT in `migrations/`, so the repo's migration directory silently lies about the current schema. There's also a numbering collision: inline "Migration 012" (parent_batch_product_id) conflicts with `migrations/012_pg_trgm_product_search.sql` — two different things share the number.

Why this week: any schema review, audit, or onboarding engineer must know to read 310 lines of Python startup as migration history. Each inline block is wrapped in `try/except logger.warning`, meaning a schema failure in production silently downgrades to a warning — the app keeps booting with the wrong schema. This is the single riskiest structural issue in the codebase.

Fix: move each block to a numbered SQL file; add a `schema_migrations` tracking table; run migrations via a deploy hook, not app startup.

### 3. Add performance indexes → [F04-10](14-performance.md)
**Effort: small (single migration).** ~14 hot-path columns have no visible index in `migrations/`. The most impactful: `transaction_lines(lot_id)` — used by >15 call-sites for lot-balance SUMs, including the `validate_lot_deduction` helper that gates every write. As the ledger grows, every ship/make/pack/trace gets slower linearly because of seq scans.

Why this week: Before fulfillment-check (F04-01) or ship_order (F04-02) start hitting dashboard-visible latency. First **verify via `pg_indexes`** — Supabase console may have added some indexes outside of the repo's migrations/ folder. Then deploy the delta.

A draft of the migration is in [14-performance.md](14-performance.md#f04-10-14-likely-missing-indexes-on-hot-path-columns).

## Other high-impact items (not this week, but near-term)

| Finding | Category | Severity | Effort |
|---|---|---|---|
| [F06-07](16-schema-openapi.md) `GPT_INSTRUCTIONS.md` references phantom `createSalesOrder` | OpenAPI | high | 5 min |
| [F04-01](14-performance.md) `fulfillment_check` fires ~480 queries worst case | Performance | critical | medium |
| [F05-05](15-traceability-gaps.md) GAP-5 void doesn't cascade | Traceability | high | medium |
| [F04-02](14-performance.md) `ship_order` commit = 6+8N queries | Performance | critical | medium |
| [F02-07](11-dead-code.md) 14 `/dashboard/api/*` endpoints lack auth | Auth hygiene | medium | small |
| [F03-03](13-endpoint-audit.md) 25 UNKNOWN-category endpoints need triage | Endpoint audit | medium | medium (2wk telemetry) |
| [F02-06](11-dead-code.md) `/audit/integrity` unauthenticated | Auth hygiene | medium | small |

## Category highlights

**Category 1 (Monolith structure):** main.py is navigable — each subsystem already has banner comments. Proposed [module split](10-monolith-structure.md#proposed-module-split-plan--do-not-implement-yet) would produce ~15 routers, none >1000 lines. ~500 lines of duplication realistically DRY-able (~5% of file) — not a fire but freezes progress on FIFO bugs until factored.

**Category 2 (Dead code):** impressively clean. **One** genuinely dead function (`bilingual_response`), zero unused imports, zero commented-out blocks, zero TODO/FIXME markers. The real waste is in 2 stale OpenAPI files and 12 alias stubs.

**Category 3 (99-vs-30 audit):** Categorized every endpoint. 30 GPT, 21 dashboard, 12 stubs (delete candidates), 5 legacy dashboard-view endpoints (delete candidates), 25 UNKNOWN needing human disambiguation. Post-cleanup target: ~70 total endpoints.

**Category 4 (Performance):** 6 endpoints exceed 5 roundtrips/request. `fulfillment_check` peaks at ~480. 14 missing indexes on hot-path columns. No connection-pool leaks.

**Category 5 (Traceability):** Prior audit's GAPs 1/2/3 are FIXED. GAP-4 PARTIAL. **GAP-5, 6 (partial), 7, 8, 10, 11, 12, 13** still open. Newly found: F05-04 `ship_order` missing service-item guard — a functional bug with recall implications.

**Category 6 (OpenAPI):** 30/30 confirmed. 7 slots reclaimable via unification. Phantom `createSalesOrder` in GPT instructions is actionable today. 2 stale schema files should be deleted.

**Category 7 (Migration integrity):** Migration gaps resolved — 001/002 = Supabase console bootstrap; 006–011 = inline startup DDL (F17-01). Numbering collision at 012. GAP-12 (SO-260213-001) still needs a reconciliation migration.

## What's going right

- Ledger-append discipline mostly holds (except reassign/merge, GAP-10/11)
- `format_timestamp`, `get_plant_now`, `resolve_order_id`, `resolve_customer_id` are well-factored single-source helpers
- `audit_integrity` endpoint exists with 8 checks — exactly the right shape for ongoing detection
- Recent FDA-relevant gaps (GAP-1/2/3) have been addressed in the last 45 days
- CHANGELOG regression-guard discipline is exemplary; every row cites what breaks if reverted
- No backdoors, no hard-coded secrets, no password-logging, no SQL-injection in the ORM paths

## Recommended sequencing

1. **This week**: F05-04 (ship_order service guard) + F06-07 (phantom operationId fix) — both <1 hr.
2. **Next week**: F17-01 (extract startup DDL) + F04-10 (performance indexes).
3. **Following 2 weeks**: F02-07 / F02-06 (add auth to 15 dashboard+audit endpoints); F03-03 endpoint instrumentation to triage UNKNOWN endpoints.
4. **Next quarter**: F05-05 (void cascade) + F05-08 (shipment_lines.lot_id) — the remaining traceability gaps that matter for FDA recall integrity.
5. **Before any new feature that needs a new GPT action**: reclaim 1–2 slots via F06-05 unification so you're not racing the 30-cap.

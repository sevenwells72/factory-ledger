# Category 7 — Migration & Data Integrity

Summary: the "migration gap" mystery resolves to **inline DDL hidden in app startup** plus a **numbering collision at 012**. No nullability drift on columns used as required. The GAP-12 unreconciled order remains open; no newer orphaned-order patterns surfaced, but the `audit_integrity` check exists.

---

### [F17-01] CRITICAL FINDING — "Migrations 004–012" are inline DDL in `startup()`, not migration files
**Severity**: high (hidden deploy-time schema changes outside version control)
**Files**: [main.py:170–480](../../main.py#L170) — 9 blocks labeled "Migration 004" through "Migration 012" executing as `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + backfill `UPDATE` on every app boot.

**Current behavior**: The FastAPI `@app.on_event("startup")` handler runs the following inline migrations:
| Block | Lines | Purpose |
|---|---|---|
| *(SKU protection)* | [170–211](../../main.py#L170) | `ALTER TABLE products ADD COLUMN label_type`; flag private-label SKUs by `odoo_code` and name pattern |
| Migration 004 | [213–244](../../main.py#L213) | `ALTER TABLE batch_formulas ADD exclude_from_inventory`; flag Water |
| Migration 005 | [246–262](../../main.py#L246) | `ALTER TABLE products ADD yield_multiplier` |
| Migration 006 | [264–304](../../main.py#L264) | `ALTER TABLE products ADD case_size_lb`; regex-populate from name |
| Migration 007 | [306–322](../../main.py#L306) | `UPDATE sales_orders SET status='confirmed' WHERE status='new'` |
| Migration 008 | [324–340](../../main.py#L324) | `ALTER TABLE lots ADD status, merged_into_lot_id, merged_at, merge_reason` |
| Migration 009 | [342–364](../../main.py#L342) | Reclassify `customer_name='Internal Packaging'` ship→pack |
| Migration 010 | [366–409](../../main.py#L366) | `CREATE TABLE customer_aliases`, seed Setton + QUALI-PACK aliases |
| Migration 011 | [411–450](../../main.py#L411) | `ALTER TABLE lots ADD supplier_lot_code, lot_type, received_at`; `CREATE TABLE lot_supplier_codes` |
| Migration 012 | [452–480](../../main.py#L452) | `ALTER TABLE products ADD parent_batch_product_id`; seed 8 FG→batch mappings |

**Implications**:
1. **Migration gaps in `migrations/` are explained.** 001/002 are bootstrap schema applied via Supabase console before migrations/ was version-controlled (confirmed by DEPLOYMENT.md phase 1). 004–012 exist BUT they are inline in `startup()`, not as SQL files. The "migration gap" at 006–011 in `migrations/` is the shadow of these inline migrations.
2. **Numbering collision: inline 012 vs migrations/012.** Inline "Migration 012" adds `parent_batch_product_id`; `migrations/012_pg_trgm_product_search.sql` enables pg_trgm. Two unrelated changes share the number "012".
3. **Schema changes aren't reviewable in a diff that only looks at `migrations/`.** Anyone auditing the DB state from the repo must know to read lines 170–480 of main.py as well.
4. **Every Railway deploy executes these DDL statements.** Idempotent guards (`IF NOT EXISTS`, self-limiting WHERE clauses) mean 0 rows change on subsequent boots, but the DDL itself runs each time — adds ~1–3s to cold start per F04-12.
5. **Silent failures.** Each block is wrapped in `try/except` logging a warning (e.g. [main.py:211](../../main.py#L211), [main.py:244](../../main.py#L244)). A schema change that fails in production is logged as a warning and the app keeps running with the wrong schema.

**Risk**: This is the single most impactful finding in this category. It means:
- **Reviewability**: Schema history is split across two codebases.
- **Auditability**: A regulator auditing FDA-relevant schema (lots, transactions, shipments) must know to read two places.
- **Deploy safety**: A typo in a startup migration silently warns but doesn't fail the deploy; the next endpoint that references the missing column crashes.
- **Rollback**: If a deploy needs to roll back, these inline migrations have no rollback path; they already committed the ALTER TABLE.

**Recommended fix** (ordered by value):
1. **Immediate (tonight)**: Extract each inline block to a numbered SQL file in `migrations/`. Suggested numbering — use 001a/002a to avoid collisions with the "initial schema" and with the existing 012:
   - `migrations/001a_sku_protection.sql`
   - `migrations/004a_exclude_from_inventory.sql` (keep 004_production_scheduling as-is)
   - `migrations/005a_yield_multiplier.sql`
   - `migrations/006a_case_size_lb.sql` through `migrations/012a_parent_batch_product_id.sql`
   Better: renumber the whole set once to keep `migrations/` monotonic. E.g. shift `012_pg_trgm_product_search.sql` → `013` and relabel downstream. Document the renumber in `FACTORY_LEDGER_CHANGELOG.md`.
2. **Short-term**: Add a `schema_migrations` tracking table; the app records which migrations have been applied (by filename). A migration runner invoked from a CI/deploy hook (or a manual `python run_migrations.py`) applies new ones and skips the rest.
3. **Long-term**: Remove the inline block from `startup()` entirely. App boot should never run DDL.
**Effort**: medium (half day to extract + numbering decision) + medium (schema_migrations runner)

---

### [F17-02] Migration numbering collision at 012
**Severity**: medium
**Files**: [main.py:452–480](../../main.py#L452) labels its block "Migration 012". [migrations/012_pg_trgm_product_search.sql](../../migrations/012_pg_trgm_product_search.sql) is a different, unrelated migration also numbered 012.
**Risk**: A newcomer reading commit messages or CHANGELOGs sees "Migration 012" with two different meanings depending on context. Future references are ambiguous.
**Recommended fix**: Part of F17-01 — assign unique numbers when extracting inline blocks. Prefer renaming inline "Migration 012" to a new number (e.g. `013_parent_batch_product_id.sql`) and renumbering the existing `012_pg_trgm_product_search` if needed.
**Effort**: small (once F17-01 is underway)

---

### [F17-03] "Migration 006–011" gap in `migrations/` is benign once F17-01 is understood
**Severity**: informational (resolved)
**Files**: `migrations/` jumps from 005 to 012. Per F17-01, numbers 006–011 are reserved for the inline startup DDL. This was deliberate — commits during that window (pre-`012_pg_trgm_product_search.sql` commit `964c24b`) applied schema via startup inline or via Supabase console, not via migration files.
**Status**: Benign, but directly tied to F17-01. Resolving F17-01 (moving inline DDL to files) also resolves this.

---

### [F17-04] 001 / 002 gap — bootstrap schema applied via Supabase console
**Severity**: informational (resolved)
**Files**: [DEPLOYMENT.md:25–35](../../DEPLOYMENT.md#L25) documents Phase 1 as "Database Setup (Supabase) — Database created and schema initialized with 5 tables: products, lots, transactions, transaction_lines, batch_formulas" — applied via Supabase console before `migrations/` was a tracked folder.
**Status**: Benign.

---

### [F17-05] No orphaned-order signal in code; `audit_integrity` check #3 is the detection
**Severity**: informational
**Files**: [main.py:9172–9190](../../main.py#L9172) — check "ship_missing_shipment_lines": every `ship` transaction after 2026-02-27 with no `shipment_lines` row is flagged MAJOR.
**Current behavior**: This check catches the orphaned-order pattern described by the user's `SO-260326-002` example — a shipment transaction that exists in `transactions`/`transaction_lines` but never got the `shipments`/`shipment_lines` rows. The endpoint is live at [/audit/integrity](../../main.py#L9122). **It's currently unauthenticated** (see [F02-06](11-dead-code.md)) — any operator can hit it and see the current orphan list.
**Recommended fix**: (1) Add auth to `/audit/integrity` per F02-06. (2) Add the check for the complementary direction — `shipment_lines` rows whose parent `transactions.status = 'voided'` (per [F05-14](15-traceability-gaps.md)).
**Effort**: small

---

### [F17-06] GAP-12: SO-260213-001 never posted inventory-deducting transactions
**Severity**: medium (historical data)
**Files**: [migrations/024_close_so260213001_juliette.sql](../../migrations/024_close_so260213001_juliette.sql).
**Current behavior**: Only `UPDATE sales_order_lines SET quantity_shipped_lb = quantity_lb, line_status='fulfilled'` and `UPDATE sales_orders SET status='fulfilled'`. No inserts to `transactions`, `transaction_lines`, `shipments`, or `shipment_lines`. Grep of migrations for "260213" returns only this file.
**Risk**: 2,402 lb of 4 granola products "shipped" without any ledger record. Lot balances for those products are inflated. FDA recall by lot code for that shipment is impossible.
**Recommended fix**: New migration `033_reconcile_so260213001.sql` that:
1. Inserts a `ship` transaction dated 2026-02-26 with BOL `28106-I` and customer `Juliette Food LLC`.
2. Uses FIFO against lot state as of 2026-02-26 to insert deducting `transaction_lines`.
3. Creates `shipments` row + `shipment_lines` rows for the 4 granola products.
4. Does NOT touch the 2 pallet charge lines (they're service items).
**Effort**: medium

---

### [F17-07] Migrations 014/021/022/023 are historical reconciliations
**Severity**: informational (inventory is a ledger of fixes)
**Files**:
- [014_fix_so260217001_flake_overshipment.sql](../../migrations/014_fix_so260217001_flake_overshipment.sql)
- [021_fix_so260217008_undershipment.sql](../../migrations/021_fix_so260217008_undershipment.sql)
- [022_close_so260217001.sql](../../migrations/022_close_so260217001.sql)
- [023_reconcile_so260312005_dicarlo.sql](../../migrations/023_reconcile_so260312005_dicarlo.sql)

**Pattern**: Each is a one-shot SQL fix correcting a specific order-level discrepancy between physical shipment and ledger. This means: **the orphaned-shipment pattern recurs**. Every time an order ships off-system (manual, no GPT call, no dashboard) a reconciliation migration is eventually needed.

**Implication** — operational:
- Most of these reconciliations map to known root causes that are NOW fixed (standalone `/ship` writing shipments — GAP-3 / changelog #15) or still open (ship_order's missing `is_service` guard — [F05-04](15-traceability-gaps.md), GAP-12 not yet reconciled — F17-06).
- As long as [F05-04](15-traceability-gaps.md) (orders with pallet charges stuck) remains open, expect new reconciliation migrations in this pattern.

**Recommended fix** (strategic, not a code change): fix F05-04 to cut off the need for future `close_soNNN.sql` migrations. Add integrity check F05-15 to surface the population at risk.
**Effort**: (see F05-04 — small)

---

### [F17-08] Columns that should be NOT NULL but aren't (nullability hygiene)
**Severity**: low
**Files**: Inferred from how columns are used in main.py.

| Column | Where used | Inferred requirement | Status |
|---|---|---|---|
| `lots.entry_source` | All trace endpoints filter on it; audit_integrity check #5 filters on it | Always set on create | **Actual: nullable** — no `NOT NULL` in the ALTER TABLE statements; handlers set it but the DB doesn't enforce |
| `lots.lot_code` | Every lookup, every trace | Required | Nullable in DB, but every insert path sets it |
| `transactions.type` | Every trace/history/audit filter | Required | Nullable; always set by handlers |
| `transactions.timestamp` | Every ORDER BY / date-range filter | Required | Nullable; always set |
| `transaction_lines.quantity_lb` | Every balance SUM | Required numeric | Set from migration 020 to NUMERIC(14,4) — nullability not specified |
| `sales_order_lines.line_status` | Every order query | Required | Set by handlers; DB column allows NULL |
| `customers.name` | resolve_customer_id primary key by name | Required, unique | Not declared unique; `LOWER(name)` uniqueness enforced only via `customer_aliases` uniqueness constraint |

**Risk**: Mostly theoretical — handlers always populate these. But a direct SQL insert (e.g., from `/admin/sql`) could bypass and create a row the rest of the app can't handle.
**Recommended fix**: Migration to add `NOT NULL` constraints to the above, with a pre-check for existing nulls. For `customers.name`, add a unique index on `LOWER(name)`.
**Effort**: small (one migration, no code change)

---

### [F17-09] Foreign keys that should exist but don't
**Severity**: low
**Files**: Inferred from cross-table references:
- `shipment_lines.transaction_id` — references `transactions(id)` per logic, but migration 013 defines it as `INTEGER REFERENCES transactions(id)` ✓ (FK exists)
- `shipment_lines.product_id` — references `products(id)` — verify via `\d shipment_lines` (migration 013 declares it; confirm)
- `ingredient_lot_consumption.ingredient_lot_id` — references `lots(id)` — not inspected in-file; verify
- `sales_order_shipments.sales_order_line_id` — references `sales_order_lines(id)` — confirm

Without DB introspection (out of scope for this audit), FK existence can't be conclusively verified. Most appear to be declared in migration 013 and 005.
**Recommended fix**: Run `\d <table>` for each cross-referenced table in Supabase and confirm all FK constraints exist. Add any missing ones.
**Effort**: small

---

### [F17-10] `migrations/` contains 8 one-shot data-reconciliation files and 15 schema-changing files
**Severity**: informational (hygiene)
**Files**: Data-reconciliation (one-shot): 014, 015, 021, 022, 023, 024, 025, 029. Schema-changing: the rest (003, 004, 005, 012, 013, 016–020, 026–028, 030, 031) + the 9 hidden inline blocks from F17-01.
**Risk**: Repository navigation confusion; see [F02-12](11-dead-code.md).
**Recommended fix**: Move one-shot reconciliations to `migrations/archive/reconciliations/`; keep schema-changing files in `migrations/`.
**Effort**: small

---

### What I could NOT determine without DB access
- Whether each of the F04-10 indexes actually exists in Supabase (Supabase console may have added some outside of the migrations/ folder)
- Whether any other orders besides SO-260213-001 have the "shipped IRL but no ledger" pattern. The `audit_integrity` check #3 detects this; recommend running it against prod and filing tickets for each orphan found.
- Whether the inline-migration safeguards (`try/except WARN`) have ever silently failed in production — requires Railway log review.

**Recommended follow-up commands** (for the user, not this audit):
```sql
-- Confirm indexes
SELECT schemaname, tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename, indexname;

-- Orphaned ships (live run of audit_integrity check #3)
SELECT t.id, t.timestamp, t.customer_name
FROM transactions t LEFT JOIN shipment_lines sl ON sl.transaction_id = t.id
WHERE t.type='ship' AND COALESCE(t.status,'posted')='posted'
  AND t.timestamp >= '2026-02-27' AND sl.id IS NULL;

-- Unreconciled pallet-charge-stuck orders (see F05-04 / F05-15)
SELECT so.order_number, so.status, array_agg(DISTINCT p.name) as stuck_products
FROM sales_orders so
JOIN sales_order_lines sol ON sol.sales_order_id = so.id
JOIN products p ON p.id = sol.product_id
WHERE so.status = 'partial_ship'
  AND sol.line_status NOT IN ('fulfilled','cancelled')
GROUP BY so.id
HAVING bool_and(COALESCE(p.is_service, false));
```

# Category 4 — N+1 Queries & Performance Risks

Scope: per-endpoint DB roundtrip counts, missing indexes, unbounded queries, cold-start cost.

Summary: **6 endpoints exceed 5 DB roundtrips per request**, one endpoint fires up to **~500 queries per call**, and **~14 likely-missing indexes** on hot-path columns. No connection-pool leaks found; `/health` and `/` are clean.

---

### [F04-01] `fulfillment_check` fires ~480 queries in the worst case
**Severity**: critical
**Files**: [main.py:5059–5170](../../main.py#L5059)
**Current behavior**: 1 query for open orders (no LIMIT — [main.py:5104](../../main.py#L5104)). For each order, 1 query for its lines ([main.py:5112](../../main.py#L5112)). For each line, 1 inventory SUM over `lots ⋈ transaction_lines` ([main.py:5137–5143](../../main.py#L5137)). With 80 open orders × 5 lines = 1 + 80 + 400 = **481 queries**.
**Risk**: The endpoint is unauthenticated by the OpenAPI contract (not in gpt-v3), so it's easy to miss in perf testing; any GPT or dashboard call walks away with a 1–3 second wait. Under load it's a db-connection hog (connection pool is `minconn=2, maxconn=20` per [main.py:164](../../main.py#L164)).
**Recommended fix**: Bulk-fetch all lines with `sales_order_id = ANY(%s)` (one query). Bulk-SUM inventory with `l.product_id = ANY(%s) GROUP BY l.product_id` (one query). Total goes from ~481 to 3.
**Effort**: medium

---

### [F04-02] `ship_order` commit fires `6 + 8N` queries where N = lines shipped
**Severity**: critical (hot write path)
**Files**: [main.py:5646–5855](../../main.py#L5646) — especially the commit loop at [main.py:5751–5815](../../main.py#L5751). Per line: inventory SUM ([main.py:5753](../../main.py#L5753)), `FOR UPDATE` re-SUM ([main.py:5761](../../main.py#L5761)), per-lot INSERT transaction_lines ([main.py:5785](../../main.py#L5785)), UPDATE `sales_order_lines.quantity_shipped_lb` ([main.py:5789](../../main.py#L5789)), UPDATE `line_status` ([main.py:5801](../../main.py#L5801)), INSERT `sales_order_shipments` ([main.py:5804](../../main.py#L5804)), INSERT `shipment_lines` ([main.py:5807](../../main.py#L5807)), SELECT `case_size_lb` ([main.py:5813](../../main.py#L5813)).
**Risk**: A 20-line order = ~160 roundtrips inside a single transaction. Holds a pool connection for the duration. Under concurrency this is the most likely endpoint to serialize behind the other.
**Recommended fix**: (1) `execute_values` for `transaction_lines` inserts. (2) Collapse the two `sales_order_lines` UPDATEs into one with a CASE expression. (3) Join `products p` into the initial line-fetch so `case_size_lb` is already in hand. (4) `execute_values` for `shipment_lines`.
**Effort**: medium

---

### [F04-03] `production_requirements` is O(ingredients × sub-ingredients)
**Severity**: high
**Files**: [main.py:7891–8077](../../main.py#L7891) — inner loops at [main.py:7972–8028](../../main.py#L7972).
**Current behavior**: For each BOM ingredient: COUNT sub-BOM ([main.py:7977](../../main.py#L7977)), inventory SUM ([main.py:7981](../../main.py#L7981)); if sub-BOM exists, nested per-sub-ingredient inventory SUM ([main.py:8023](../../main.py#L8023)). 12-ingredient batch with 2 sub-BOMs × 10 sub-ingredients ≈ **60 queries**.
**Recommended fix**: Prefetch all `batch_formulas` by `product_id = ANY(%s)`; prefetch inventory with `product_id = ANY(%s) GROUP BY product_id`. Reduces to ~4 queries.
**Effort**: medium

---

### [F04-04] `make` commit ingredient consumption loop is N+3M
**Severity**: high
**Files**: [main.py:2597–2963](../../main.py#L2597) — consumption loop at [main.py:2831–2890](../../main.py#L2831).
**Current behavior**: Per formula ingredient: SELECT override lot ([main.py:2850](../../main.py#L2850)), `FOR UPDATE` lock ([main.py:2853](../../main.py#L2853)), `validate_lot_deduction` (internal SELECT), per-lot INSERT `transaction_lines` + INSERT `ingredient_lot_consumption` ([main.py:2884–2885](../../main.py#L2884)). Typical batch ≈ 40–60 roundtrips.
**Recommended fix**: `execute_values` for `ingredient_lot_consumption` and `transaction_lines`. Single `FOR UPDATE ... WHERE id = ANY(%s)` lock.
**Effort**: medium

---

### [F04-05] `pack` commit loop is N+4M (similar pattern)
**Severity**: high
**Files**: [main.py:3070–3311](../../main.py#L3070) — commit block at [main.py:3182–3264](../../main.py#L3182). Same pattern as `make`. Add-in FIFO loop [main.py:3239–3264](../../main.py#L3239) compounds the issue.
**Recommended fix**: Same — `execute_values`.
**Effort**: medium

---

### [F04-06] `trace_supplier_lot` fires 4 queries per matched lot
**Severity**: medium
**Files**: [main.py:3918–4073](../../main.py#L3918) — inner loops at [main.py:3964–4046](../../main.py#L3964).
**Current behavior**: For each matched internal lot: on_hand SUM, received metadata, production usage, shipments. 10 matched lots = 40 queries.
**Recommended fix**: Aggregate per `lot_id = ANY(%s)` with `GROUP BY lot_id` → 4 queries total regardless of lot count.
**Effort**: medium

---

### [F04-07] `search_products` N+1 on match-detail fetches
**Severity**: medium
**Files**: [main.py:1347–1375](../../main.py#L1347).
**Current behavior**: Tiered search returns matches, then re-queries each id individually ([main.py:1360–1366](../../main.py#L1360)). With `limit=20`, that's 21 queries.
**Recommended fix**: Single `SELECT ... WHERE id = ANY(%s)`; preserve tier-order in Python.
**Effort**: small

---

### [F04-08] `create_sales_order` fires ~4N queries per line
**Severity**: medium
**Files**: [main.py:4857–4969](../../main.py#L4857) — line loop at [main.py:4879–4940](../../main.py#L4879).
**Current behavior**: Per line: `resolve_product_id` (1–4 queries depending on tier), SELECT product details ([main.py:4883](../../main.py#L4883)), INSERT line ([main.py:4917](../../main.py#L4917)), AVG customer-history sanity check ([main.py:4927](../../main.py#L4927)). 10-line order ≈ 40 queries.
**Risk**: The AVG subquery scans `sales_order_lines ⋈ sales_orders` with no dedicated index on `(customer_id, product_id)` (see F04-10).
**Recommended fix**: Bulk-resolve all products upfront; one `execute_values` for INSERT lines; single windowed AVG across all `(customer_id, product_id)` pairs.
**Effort**: medium

---

### [F04-09] 8 FIFO loops share the same shape but are not factored
**Severity**: high (correctness + perf)
**Files**: See [F01-D1 in 10-monolith-structure.md](10-monolith-structure.md).
**Current behavior**: Every allocation does `SELECT ... ORDER BY received_at ASC ... FOR UPDATE`, collects ids, re-selects, loops deducting. Each loop INSERTs one tx_line per lot consumed instead of batching.
**Risk**: Twofold — any perf regression (e.g. adding an index on `received_at`) has to be re-measured on 8 queries; and bulk-insert is blocked by the scattered shape.
**Recommended fix**: Same as F01-D1. After extracting the helper, batch the INSERTs via `execute_values` inside it.
**Effort**: large

---

### [F04-10] ~14 likely-missing indexes on hot-path columns
**Severity**: high
**Files**: migrations/*.sql (index declarations); [main.py](../../main.py) for query shapes.
**Current behavior**: Declared indexes in migrations/ (across all files) and in startup inline DDL:
- `idx_customer_aliases_lower_alias` (startup, [main.py:380](../../main.py#L380))
- `idx_lot_supplier_codes_lot_id`, `idx_lot_supplier_codes_supplier_lot`, `idx_lots_supplier_lot_code` (startup, [main.py:434](../../main.py#L434))
- `idx_products_name_trgm` (012)
- `idx_shipments_sales_order_id`, `idx_shipment_lines_shipment_id`, etc. (013)
- (017) `idx_transactions_status`

Hot columns with no visible index:
| Table.Column(s) | Hot uses |
|---|---|
| `transaction_lines.lot_id` | Every lot balance SUM — used in >15 places |
| `transaction_lines.transaction_id` | `get_transaction_history`, `audit_integrity`, dashboard shipments/receipts |
| `transaction_lines.product_id` | `audit_integrity` check #6 |
| `transactions(type, timestamp)` | Every history/trace filter; timestamp DESC ORDER BY |
| `lots.product_id` | Every "lots for product" SELECT |
| `lots` `LOWER(lot_code)` | Lot lookup by code; disambiguation in trace endpoints |
| `sales_order_lines.sales_order_id` | get_sales_order, ship_order, fulfillment_check |
| `sales_order_lines.product_id` | create_sales_order AVG |
| `customer_aliases.customer_id` | Used but may have implicit FK index; verify |
| `products` `LOWER(name)` | resolve_product, quick-create |
| `ingredient_lot_consumption.ingredient_lot_id` | trace backward |
| `ingredient_lot_consumption.transaction_id` | trace forward |
| `shipment_lines.transaction_id` | `audit_integrity` check #3, trace |
| `sales_orders(status, requested_ship_date)` | `sales_dashboard` overdue/due-this-week |

**Risk**: Lot balance SUM over `transaction_lines.lot_id` is the single hottest query — Supabase likely does this with a seq scan today (no index). As `transaction_lines` grows, every ship/make/pack/trace gets slower linearly.
**Recommended fix**: New migration `032_add_performance_indexes.sql` with all `CREATE INDEX IF NOT EXISTS` statements. **Verify first** with `SELECT indexname FROM pg_indexes WHERE schemaname='public'` — some may already exist via Supabase console (which wouldn't appear in /migrations/).
**Effort**: small (write migration) + small (verify against prod)

Draft migration:
```sql
CREATE INDEX IF NOT EXISTS idx_tl_lot_id            ON transaction_lines(lot_id);
CREATE INDEX IF NOT EXISTS idx_tl_transaction_id    ON transaction_lines(transaction_id);
CREATE INDEX IF NOT EXISTS idx_tl_product_id        ON transaction_lines(product_id);
CREATE INDEX IF NOT EXISTS idx_txn_type_ts          ON transactions(type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_lots_product_id      ON lots(product_id);
CREATE INDEX IF NOT EXISTS idx_lots_lot_code_lower  ON lots(LOWER(lot_code));
CREATE INDEX IF NOT EXISTS idx_sol_sales_order      ON sales_order_lines(sales_order_id);
CREATE INDEX IF NOT EXISTS idx_sol_product_id       ON sales_order_lines(product_id);
CREATE INDEX IF NOT EXISTS idx_ca_customer_id       ON customer_aliases(customer_id);
CREATE INDEX IF NOT EXISTS idx_products_name_lower  ON products(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_ilc_ingredient_lot   ON ingredient_lot_consumption(ingredient_lot_id);
CREATE INDEX IF NOT EXISTS idx_ilc_transaction_id   ON ingredient_lot_consumption(transaction_id);
CREATE INDEX IF NOT EXISTS idx_sl_transaction_id    ON shipment_lines(transaction_id);
CREATE INDEX IF NOT EXISTS idx_so_status_ship_date  ON sales_orders(status, requested_ship_date);
```

---

### [F04-11] Unbounded list queries (missing LIMIT clauses)
**Severity**: medium
**Files**:
- [main.py:5104](../../main.py#L5104) `fulfillment_check` — open orders, no LIMIT
- [main.py:6418](../../main.py#L6418), [main.py:6438](../../main.py#L6438) `sales_dashboard` overdue + due-this-week — no LIMIT
- [main.py:1498](../../main.py#L1498) `get_current_inventory` — not paginated (spot-check suggests this was the old endpoint, superseded by /inventory/lookup which has `limit`)
- [main.py:9122](../../main.py#L9122) `audit_integrity` — intentional (diagnostic)
- [main.py:7716](../../main.py#L7716) `admin_sql_query` — no server-side LIMIT enforcement on arbitrary SELECTs

**Risk**: Single request returns thousands of rows → spikes memory, saturates network. `audit_integrity` being slow is tolerable; the sales-dashboard / fulfillment-check routes are called from the Netlify dashboard on page load.
**Recommended fix**: Add `LIMIT 200` defaults with an override parameter on each listed endpoint. `admin_sql_query` should wrap the user SQL in a subquery with `LIMIT 1000`.
**Effort**: small

---

### [F04-12] Startup-path inline DDL runs on every cold start
**Severity**: medium
**Files**: [main.py:170–480](../../main.py#L170) — 9 blocks of `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + backfill `UPDATE`s, labeled "Migration 004" through "Migration 012".
**Current behavior**: Each block acquires its own pool connection, runs idempotent DDL, commits. On repeat boots, the `UPDATE`s touch 0 rows (all predicates are self-limiting, e.g. `WHERE label_type != 'private_label'` or `WHERE case_size_lb IS NULL`). Migration 006 still runs a regex UPDATE on the full `products` table each boot ([main.py:285–292](../../main.py#L285)) — the full scan executes even when 0 rows match.
**Risk**: Railway cold-start time +1–3s depending on product count; also, any change to these inline blocks is a deploy-time DDL that's not peer-reviewable as a migration. Most importantly, this is the real answer to "migration gaps 006–011 mystery" (see [F17-01](17-migration-integrity.md)) — they're hiding here.
**Recommended fix**: Move each block to `migrations/006_*.sql` through `migrations/012_*.sql` (numbering collides with existing `012_pg_trgm_product_search.sql` — renumber needed). Add a `schema_migrations` tracking table. Run migrations via a separate runner script at deploy time, not in the app startup.
**Effort**: medium (half day to convert; add runner + CI step)

---

### [F04-13] `/admin/sql` has only a prefix check
**Severity**: medium (requires API key but still a footgun)
**Files**: [main.py:7715–7727](../../main.py#L7715)
**Current behavior**: Guards `sql.upper().startswith("SELECT")`. Psycopg2 by default rejects multi-statement inputs, so a chained `DROP TABLE` injection is blocked at the driver level — but a lone `SELECT pg_sleep(60)`, `SELECT * FROM transactions` (exfiltrate everything), or `SELECT set_config('role','superuser', false)` is allowed.
**Risk**: API-key gated, so reduced — but this is still easy to misuse, and DB credentials run with write permissions (startup does ALTER TABLE). A server-side auto-LIMIT or read-only role would tighten it.
**Recommended fix**: (1) Wrap user SQL: `cur.execute("SELECT * FROM (" + user_sql + ") _sub LIMIT 1000")`. (2) If possible, use a read-only DB role for this endpoint's connection. (3) Log every admin-sql call with the SQL + caller for audit.
**Effort**: small

---

### [F04-14] Top-10 endpoint DB-roundtrip counts
**Severity**: informational
Per-endpoint cur.execute() counts (worst case):
1. `fulfillment_check` — `1 + O(orders) + O(lines)` → up to ~500
2. `ship_order` commit — `6 + 8N` — 166 for a 20-line order
3. `make` commit — `7 + 3–5M` — 40–60 typical
4. `pack` commit — `5 + 4–6M` — 30–50 typical
5. `create_sales_order` — `3 + 4N` — 40 for 10-line order
6. `production_requirements` — `4 + 2M + 2S` — ~60 worst
7. `trace_supplier_lot` — `1 + 4L`
8. `receive` commit — `6 + E` — low
9. `trace_batch` — ~6 fixed
10. `search_products` — `1 + N` up to 21

---

### What's NOT a problem
- Connection pool is sized (2–20) and every endpoint uses the context managers; no connection leaks observed.
- `/health` runs a single `SELECT 1`; `/` is a static dict.
- `format_timestamp` is used consistently; no timezone re-conversion churn.
- Auth check (`secrets.compare_digest`) is constant-time.

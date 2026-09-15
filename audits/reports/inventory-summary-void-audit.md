# Inventory-summary void audit — 2026-09-15

Read-only audit of every place that adds up ledger lines without excluding voided (or amended) events. No code, data, or migration changes were made. Database queries ran on the session pooler (port 5432) inside `BEGIN; SET TRANSACTION READ ONLY;`.

## 1. Summary (plain words)

Nine database views still add up the raw `transaction_lines` table. That table keeps every line ever written, including lines that were later voided or amended, so those views show the old numbers. The one Blubber noticed is `inventory_summary`, which today shows 3,600 lb of Toasted Sweetened Flake batch stock when the correct answer is 0 lb. The 3,600 lb is voided make #1699.

Nobody normally sees these numbers. The Netlify dashboard, the ChatGPT actions, the exports and the health checks all read the correction-aware views (`ledger_current_transactions` and `ledger_current_transaction_lines`, filtered to `effective_status = 'posted'`) since the void-semantics fix in June 2026. The stale views only reach the outside world through five legacy `GET /dashboard/*` endpoints that no screen or GPT action calls, and through anyone querying the views directly, which is how the coconut investigation was misled.

Across all products, 10 product balances in `inventory_summary` differ from the correct posted-only balance. The differences run both ways: voided makes inflate batch stock, and the ingredient deductions on those same voided makes deflate ingredient stock.

## 2. Where and when

| Item | Value |
|---|---|
| Repo | `/Users/cns/dev/factory-ledger`, branch `main` at `656d18d` (equal to `origin/main`; only uncommitted change is changelog row #153) |
| Schema reference | `tests/schema/schema.sql`, re-dumped from production 2026-09-15 12:44 ET |
| Live DB snapshot | 2026-09-15 15:30 ET, read-only |

## 3. Every reader of the unfiltered views

| Location | What it is | Route | Dashboard screen | GPT operationId | User-facing? | Void-filtered? |
|---|---|---|---|---|---|---|
| `tests/schema/schema.sql:3978` (prod) | view `inventory_summary` (SUM over `transaction_lines`) | — | — | — | INDIRECT: only via the endpoint below or direct SQL | NO |
| `main.py:4308` | `GET /dashboard/inventory` → `SELECT * FROM inventory_summary WHERE on_hand > 0` | `/dashboard/inventory` | none (not called by `dashboard/*.js`) | none (not in `openapi-gpt-v3.yaml`) | NO today; reachable with the dashboard or master key | NO |
| `tests/schema/schema.sql:4014` (prod) | view `low_stock_alerts` (ingredients under 100 lb, SUM over `transaction_lines`) | — | — | — | INDIRECT | NO |
| `main.py:4318` | `GET /dashboard/low-stock` → `low_stock_alerts` | `/dashboard/low-stock` | none | none | NO today | NO |
| `tests/schema/schema.sql:2237` (prod) | view `todays_transactions` (raw `transactions` + `transaction_lines`, `timestamp::date = CURRENT_DATE`) | — | — | — | INDIRECT | NO |
| `main.py:4328` | `GET /dashboard/today` → `todays_transactions` | `/dashboard/today` | none | none | NO today | NO |
| `tests/schema/schema.sql:3995` (prod) | view `lot_balances` (per-lot SUM over `transaction_lines`, HAVING > 0) | — | — | — | INDIRECT | NO |
| `main.py:4338` | `GET /dashboard/lots` → `lot_balances LIMIT 100` | `/dashboard/lots` | none | none | NO today | NO |
| `tests/schema/schema.sql:4058` (prod) | view `production_history` (raw `transactions` type = 'make', SUM of positive lines) | — | — | — | INDIRECT | NO |
| `main.py:4348` | `GET /dashboard/production` → `production_history LIMIT 50` | `/dashboard/production` | none | none | NO today | NO |
| `tests/schema/schema.sql:2472` (prod) | view `v_lot_quantities` (per-lot SUM over `transaction_lines`) | — | — | — | NO: no reader in `main.py`, dashboard, or GPT spec | NO |
| `tests/schema/schema.sql:2392` (prod) | view `v_batch_products_needing_setup` (count/SUM over `transaction_lines`) | — | — | — | NO: no reader in code | NO |
| `tests/schema/schema.sql:2489` (prod) | view `v_products_missing_boms` (count/SUM over `transaction_lines`) | — | — | — | NO: no reader in code | NO |
| `tests/schema/schema.sql:2554` (prod) | view `v_test_batches_for_review` (count/SUM over `transaction_lines`) | — | — | — | NO: no reader in code | NO |
| `main.py:2674-2678` | `DASHBOARD_KEY_ALLOWLIST` entries for the five legacy routes | — | — | — | — | n/a |
| `tests/test_dashboard_api_key.py:107,173,182` | auth tests that call `/dashboard/inventory` and only assert the status code | — | — | — | — | n/a |
| `FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:403` | already documents these six views and five endpoints as "legacy ... not correction-aware" | — | — | — | — | n/a |
| `audits/reports/DASHBOARD_ACTIONABILITY_AUDIT.md:47-51`, `docs/design/scheduling-spec-draft.md:66,164`, `docs/designs/044-so-allocations-design.md:205` | prior docs naming the legacy endpoints/views | — | — | — | — | n/a |

Raw `transaction_lines` references in `main.py` that are **correct** and out of scope: `main.py:9206` (row lock by id), `:10097` and `:10237` and `:10409` (correction-log joins by line id, not balance math). The `lot_balances` at `main.py:11891-11978` is a CTE inside `GET /export/orders-matrix.xlsx` built on `ledger_current_*` with `effective_status = 'posted'`; it shadows the view and is correct.

Everything user-facing (all `/dashboard/api/*` routes, `/inventory/*`, `/trace/*`, `/transactions/history`, Health v3, exports) already reads `ledger_current_transaction_lines` joined to `ledger_current_transactions` with `effective_status = 'posted'`.

## 4. Live view definitions and the missing filter

Live `pg_get_viewdef('inventory_summary')` matches `tests/schema/schema.sql:3978`:

```sql
SELECT p.id, p.name, p.type,
       COALESCE(sum(tl.quantity_lb), 0::numeric) AS on_hand
  FROM products p
  LEFT JOIN lots l ON l.product_id = p.id
  LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
 WHERE COALESCE(p.active, true) = true
 GROUP BY p.id
 ORDER BY p.type, p.name;
```

`lot_balances`, `low_stock_alerts`, `v_lot_quantities`, `v_batch_products_needing_setup`, `v_products_missing_boms` and `v_test_batches_for_review` use the same `LEFT JOIN transaction_lines tl ON tl.lot_id = l.id` shape. `todays_transactions` and `production_history` join `transactions t JOIN transaction_lines tl` directly.

Live check of every view in `public` whose definition mentions `transaction_lines`: 10 views, none contain `effective_status`, `void`, or `ledger_current_` (the tenth is `ledger_current_transaction_lines` itself, which is the correct base and carries amendments; void status lives on `ledger_current_transactions`).

The missing filter, for the `lot`-joined views:

```sql
  LEFT JOIN ledger_current_transaction_lines tl ON tl.lot_id = l.id
  LEFT JOIN ledger_current_transactions ct
         ON ct.id = tl.transaction_id AND ct.effective_status = 'posted'
  -- and sum ct-qualified lines only:
  COALESCE(sum(tl.quantity_lb) FILTER (WHERE ct.id IS NOT NULL), 0)
```

or, simpler and equivalent, a subquery of posted lines:

```sql
  LEFT JOIN (
      SELECT tl.lot_id, tl.product_id, tl.quantity_lb
        FROM ledger_current_transaction_lines tl
        JOIN ledger_current_transactions ct ON ct.id = tl.transaction_id
       WHERE ct.effective_status = 'posted'
  ) tl ON tl.lot_id = l.id
```

For `todays_transactions` and `production_history`: replace `transactions t` with `ledger_current_transactions t`, `transaction_lines tl` with `ledger_current_transaction_lines tl`, and add `AND t.effective_status = 'posted'` to the WHERE clause. Using `ledger_current_transaction_lines` also picks up amended quantities, product ids and lot ids, which raw `transaction_lines` ignores.

## 5. Side-by-side numbers (live, 2026-09-15 15:30 ET)

Toasted Sweetened Flake batch (product 128):

| Source | lb |
|---|---:|
| `inventory_summary.on_hand` | 3,600.0 |
| posted-only (`ledger_current_*`, `effective_status = 'posted'`) | 0.0 |
| gap | 3,600.0 = voided make #1699, lot 1026, +3,600 lb |

Voided make #1699 also carried ingredient deductions (lot 615 −2,400; lot 797 −1,320; lot 461 −84; lot 441 −60; lot 48 −30), so the raw views understate those ingredients by the same amounts.

All products where `inventory_summary` disagrees with posted-only:

| Product | `inventory_summary` | posted-only | diff |
|---|---:|---:|---:|
| 128 Batch Coconut Toasted Sweetened Flake | 3,600.00 | 0.00 | +3,600.00 |
| 12 Coconut Flake Desiccated | 6,850.00 | 9,250.00 | −2,400.00 |
| 40 Sugar – 6X | 26,351.51 | 27,671.51 | −1,320.00 |
| 4 Almonds – Slivered | 553.00 | 53.00 | +500.00 |
| 29 Glycol | 2,082.00 | 2,166.00 | −84.00 |
| 33 Oats – Gluten Free | 2,075.00 | 2,000.00 | +75.00 |
| 16 Corn Starch | 1,445.00 | 1,505.00 | −60.00 |
| 39 Salt | 8,552.74 | 8,581.74 | −29.00 |
| 165 Graham Cracker Crumbs – 10 LB | −1.00 | 0.00 | −1.00 |
| 190 Desiccated Flake 50 LB | 101,300.01 | 101,300.00 | +0.01 |

Overall: 57 voided transactions exist; their lines still net +281.01 lb into the raw views. After the row #153 adjustments, `inventory_summary` reads 0.0 for products 125, 126 and 127 and 3,600 for 128; posted-only reads 0.0 for all four. Lot 518 (the May 1 Toasted lot) reads 0.0 in `v_lot_quantities` and posted-only, and is absent from `lot_balances` (HAVING > 0), all consistent.

## 6. What a fix would change, ranked by user impact (nothing implemented)

1. **Migration 055: redefine the six legacy views** (`inventory_summary`, `lot_balances`, `low_stock_alerts`, `todays_transactions`, `production_history`, `v_lot_quantities`) on `ledger_current_transaction_lines` + `ledger_current_transactions` with `effective_status = 'posted'`. Keeps every column name and type so the five legacy endpoints and their auth tests keep working. Idempotent `CREATE OR REPLACE VIEW`, marker row in `migration_markers`, down file restoring the raw definitions. Apply on port 5432 only.
2. **Same treatment for the three helper views** `v_batch_products_needing_setup`, `v_products_missing_boms`, `v_test_batches_for_review`. No code reads them; fixing them costs nothing extra and stops anyone being misled by them in the SQL editor.
3. **Re-dump `tests/schema/schema.sql`** with `scripts/dump_prod_schema.sh` after applying, so the local test DB matches.
4. **Tests**: one test per redefined view that posts a make, voids it, and asserts the view no longer counts it (and one amend case for `inventory_summary`).
5. **Docs**: update `FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:403` to say the legacy views are now correction-aware; changelog row #154.
6. **Optional, owner call**: delete the five legacy `/dashboard/*` endpoints, their `DASHBOARD_KEY_ALLOWLIST` entries and the auth-test references. Nothing calls them. Not needed for correctness once the views are fixed.

Not touched by any of this: `/make`, `/pack`, `/adjust`, void/amend logic, trace paths, `openapi-gpt-v3.yaml` (stays at 30 operations), allocation, Health.

## 7. Surprising or uncertain

- This was already known and documented: `FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:403` says these views "are not correction-aware". The June 2026 void-semantics fix (`fix/void-semantics`, changelog row 30) moved the application onto posted-only math but left the views as they were.
- `todays_transactions` filters on `timestamp::date = CURRENT_DATE` in the database's time zone (UTC), not `business_date` in ET. That is a separate day-boundary bug in the same legacy view; worth fixing in the same migration by using `business_date = (now() AT TIME ZONE 'America/New_York')::date`, but it is a behaviour change beyond voids and should be called out in the PR.
- `production_history` filters `t.type = 'make'`; `SYSTEM_KNOWLEDGE` says some legacy views use obsolete enum values. In the live definitions only the `type = 'make'` and `type = 'ingredient'` checks appear, and both are current values, so no obsolete-enum problem was found in these nine views.
- The `lot_balances` CTE name inside the orders-matrix export shadows the view of the same name. Harmless, but easy to misread.

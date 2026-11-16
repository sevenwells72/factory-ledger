# Customer merge: Hialeah / New Urban Farms (#33, #61 → #57)

- **Executed:** 2026-11-16 14:08 ET (transaction `now()` 2026-11-16 19:08:08 UTC), production, direct SQL via port 5432 in one transaction with in-transaction count checks.
- **Approved by:** Michael (owner), in session, 2026-11-16: true merge approved, then "Yes, move 16 rows", then "Yes, move 7 orders" after SO-261116-004 appeared on #33.
- **Attributed to:** actor #1, operator "Michael". `actor_write_audit` rows 91–100, route `manual-sql: customer merge 2026-11-16 (#33,#61 -> #57)`.
- **Backup first:** `~/Backups/fl/2026-11-16`, `BACKUP OK`, key checks all match.
- **Why:** one real customer had three active records (#33 "Hialeah / New Urban Farms", #57 "Hialea/New Urban Farms", #61 "Hialeah/New Urban Farms"), so the Office picker offered three choices.

## What changed (customer link only)

No quantities, dates, lots, BOLs, prices, statuses or order lines changed. The `trg_sales_order_updated` trigger set `sales_orders.updated_at` on the 7 moved orders, and the update to customers set `customers.updated_at` the same way.

### `sales_orders.customer_id` (7 rows)

| Row ID | Order | Status | From | To |
|---|---|---|---|---|
| 36 | SO-260217-005 | cancelled | 33 | 57 |
| 37 | SO-260217-006 | shipped | 33 | 57 |
| 79 | SO-260312-003 | cancelled | 33 | 57 |
| 86 | SO-260318-002 | shipped | 33 | 57 |
| 95 | SO-260320-002 | shipped | 61 | 57 |
| 300 | SO-260818-001 | shipped | 61 | 57 |
| 406 | SO-261116-004 | confirmed (PO 4356, created 2026-11-16 13:58 ET) | 33 | 57 |

### `shipments.customer_id` (10 rows)

| Row IDs | Order | From | To |
|---|---|---|---|
| 17, 18, 19 | SO-260217-006 | 33 | 57 |
| 30 | SO-260312-003 | 33 | 57 |
| 33 | SO-260318-002 | 33 | 57 |
| 66, 67, 68 | (no sales order; backfilled shipment records) | 33 | 57 |
| 141 | SO-260320-002 | 61 | 57 |
| 357 | SO-260818-001 | 61 | 57 |

(`actor_write_audit` only accepts customers / lots / sales_orders / sales_order_lines / exceptions / notes, so the shipment moves are recorded here and in the changelog only.)

### `customers`

| ID | Name before | Name after | Active before → after |
|---|---|---|---|
| 33 | Hialeah / New Urban Farms | Hialeah / New Urban Farms (merged into #57) | true → false |
| 57 | Hialea/New Urban Farms | Hialeah / New Urban Farms | true → true |
| 61 | Hialeah/New Urban Farms | Hialeah/New Urban Farms (merged into #57) | true → false |

### `customer_aliases` (new rows)

| Alias row | Customer | Alias |
|---|---|---|
| 6 | 57 | Hialea/New Urban Farms |
| 7 | 57 | Hialeah/New Urban Farms |

Before the merge, none of the three customers had any aliases.

## Left unchanged on purpose

- **`trace_events` row 24** (`ship`, 2026-09-01, SO-260818-001) still points at #61. The table is append-only, and its trigger `trg_trace_events_append_only` blocks updates. #61 still exists, archived, so the trace still resolves.
- **`transactions.customer_name`** text is unchanged on 8 posted ledger entries (6 say "Hialeah / New Urban Farms", 2 say "Hialeah/New Urban Farms"). Ledger entries are never edited.
- **#33 and #61 were not deleted.**

## Verification after commit

- Read-only DB check: #57 has 17 orders and 18 shipments, and is active. #33 and #61 each have 0 orders and 0 shipments, and are inactive.
- Live API `POST /resolve` (kind customer, "hia"): one candidate, #57 "Hialeah / New Urban Farms". The old spellings "Hialea/New Urban Farms" and "Hialeah/New Urban Farms" resolve to #57 through the aliases.
- Live API `GET /customers/search?q=hia`: only #57.
- Live API `GET /sales/orders/{370,402,406}`: SO-261016-001, SO-261116-001 and SO-261116-004 show customer "Hialeah / New Urban Farms", status confirmed, 1 line each.

## Open question (not acted on)

SO-261116-004 (#33 originally, PO 4356, Granola Classic 25 LB 3,000 lb, ship by Nov 26) and SO-261116-001 (#57, no PO, Granola Classic 25 LB 3,000 lb, ship by Nov 17) may be the same order entered twice. Michael to confirm. If one is a duplicate, cancel it as a separate, approved step.

## How to reverse

Run as one transaction, after a fresh verified backup and owner approval. Rename #57 first so the original names are free again:

```sql
BEGIN;
DELETE FROM customer_aliases WHERE id IN (6, 7) AND customer_id = 57;
UPDATE customers SET name = 'Hialea/New Urban Farms' WHERE id = 57 AND name = 'Hialeah / New Urban Farms';
UPDATE customers SET name = 'Hialeah / New Urban Farms', active = true WHERE id = 33;
UPDATE customers SET name = 'Hialeah/New Urban Farms',  active = true WHERE id = 61;
UPDATE sales_orders SET customer_id = 33 WHERE customer_id = 57 AND id IN (36, 37, 79, 86, 406);
UPDATE sales_orders SET customer_id = 61 WHERE customer_id = 57 AND id IN (95, 300);
UPDATE shipments    SET customer_id = 33 WHERE customer_id = 57 AND id IN (17, 18, 19, 30, 33, 66, 67, 68);
UPDATE shipments    SET customer_id = 61 WHERE customer_id = 57 AND id IN (141, 357);
-- expect 5 + 2 orders and 8 + 2 shipments; ROLLBACK if not
COMMIT;
```

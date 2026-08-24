# Traceability Data-Entry Audit — August 2026 (Passes 1–2)

**Date:** 2026-08-24 · **Window:** 2026-05-27 → 2026-08-24 (90 New York calendar days)
**Method:** read-only queries via `scripts/traceability_audit.sql`, run against production through the port-6543 pooler with every query inside `BEGIN TRANSACTION READ ONLY … COMMIT` (no session GUCs). Ledger reads use `ledger_current_transactions` with `effective_status = 'posted'`. Entry-lag metrics use only rows with `created_at_source = 'database'` — the 2026-08-11 migration 039 backfill stamped older rows with the migration timestamp, so their `created_at` is not a real entry time.

No application code or data was modified.

---

## 1. Coverage matrix

"Event-date column?" asks whether the schema stores an event date distinct from `created_at`. Note that **no write endpoint accepts a caller-supplied event date**: for ledger transactions, `occurred_at` defaults to the server clock at entry and `business_date` is derived from it (migration 039 trigger), so a distinct event date only diverges from `created_at` when rows are posted late by hand or amended via `ledger_corrections`.

| Event type | Table(s) | Required fields (NOT NULL / API-required) | Event-date column distinct from `created_at`? | Lot / traceability linkage (FKs) |
|---|---|---|---|---|
| Receipts | `transactions` (`type='receive'`) + `transaction_lines` | API: product_name, cases, case_size_lb, shipper_name, bol_reference. DB: type, occurred_at, business_date, operator_id; line product_id, lot_id, quantity_lb | Yes — `business_date` / `occurred_at` (server-derived at entry) | `transaction_lines.lot_id→lots`, `.product_id→products`; `transactions.expected_receipt_id→expected_receipts`; supplier lot codes via `lot_supplier_codes.lot_id→lots`. **No supplier FK** — `shipper_name` is free text (suppliers resolved by name only for ER matching) |
| Expected receipts | `expected_receipts` | product_id, supplier_id, expected_qty, status | Yes — `expected_date` (caller-supplied planned date, nullable) | `product_id→products`, `supplier_id→suppliers`; back-linked from `transactions.expected_receipt_id` |
| Shipments | `transactions` (`type='ship'`) + `shipments` + `shipment_lines` + `sales_order_shipments` | API: product_name, quantity_lb, customer_name, order_reference. DB: `shipments.shipped_at` NOT NULL; `shipment_lines` shipment_id, transaction_id, product_id, quantity_lb | Yes — `business_date` / `shipped_at` (both server-set at entry) | Lot linkage **only** via `transaction_lines.lot_id` (`shipment_lines` has no lot_id); `shipment_lines.sales_order_line_id→sales_order_lines` (nullable); `shipments.sales_order_id/customer_id/transaction_id`; `sales_order_shipments.transaction_id→transactions`. `transactions.customer_name` is free text |
| SO allocations | `sales_order_allocations` (+ `sales_order_allocation_reactivations`) | sales_order_id, sales_order_line_id, product_id, quantity_lb, status, source | **No** — `created_at` only (lifecycle stamps `released_at`/`expires_at`, no event date) | FKs to `sales_orders`, `sales_order_lines`, `products`, `lots` (lot_id nullable), `transactions` (ship_transaction_id, last_ship_transaction_id), self (`split_from_id`); reactivations → `ledger_corrections` |
| Production batches (make) | `transactions` (`type='make'`) + `transaction_lines` (output) + `ingredient_lot_consumption` (inputs) | API: product_name, batches. DB: ILC quantity_lb; line product_id, lot_id, quantity_lb | Yes — `business_date` / `occurred_at` (server-derived) | Output lot via `transaction_lines.lot_id→lots`; inputs via `ingredient_lot_consumption.ingredient_lot_id→lots`, `.transaction_id→transactions`; output lot stamped `lots.entry_source='production_output'` |
| Finished-goods lots | `lots` | product_id, lot_code, entry_source, status | **No** for produced lots — `received_at` exists but is meant for received lots (and is patchable after the fact); produced lots carry only `created_at` | `product_id→products`; genealogy through the creating make/pack transaction's `transaction_lines` + `ingredient_lot_consumption`; `merged_into_lot_id→lots` (self) |
| Inventory adjustments | `transactions` (`type='adjust'`); found-inventory paths also write `inventory_adjustments` | API: product_name, lot_code, adjustment_lb, reason. `inventory_adjustments`: product_id, adjustment_type, quantity_adjustment, quantity_after, uom, reason_code | Yes — `business_date` / `occurred_at` (server-derived; backdated only by direct posts/amends) | `transaction_lines.lot_id→lots`; **`inventory_adjustments` has no FK constraints at all** (lot_id, product_id, inventory_count_id are bare integers) |
| Packaging entries (pack) | `transactions` (`type='pack'`) + `transaction_lines` (batch debit + FG credit) + `ingredient_lot_consumption` (batch-lot draw) | API: source_product, target_product, cases. DB: same as make | Yes — `business_date` / `occurred_at` (server-derived) | Batch→FG genealogy via `transaction_lines.lot_id` and `ingredient_lot_consumption.ingredient_lot_id→lots`; FG lot stamped `lots.entry_source='pack_output'` |

---

## 2. Raw metrics (last 90 days)

### 2.1 Volume, active days, zero days

| Event type | Total entries | Active days | Zero days (of 90) | Zero **week**days (of 64) | Max/day |
|---|---:|---:|---:|---:|---:|
| receive | 57 | 31 | 59 | 34 | 5 |
| expected_receipt | **0** | 0 | 90 | 64 | 0 |
| ship | 254 | 44 | 46 | 20 | 22 |
| so_allocation | **1** | 1 | 89 | 63 | 1 |
| make | 160 | 65 | 25 | 4 | 6 |
| pack | 286 | 67 | 23 | 2 | 10 |
| fg_lot | 352 | 66 | 24 | 3 | 13 |
| adjust | 185 | 40 | 50 | 24 | 66 |

Adjustments arrive in bursts, not daily discipline: 41 on 2026-06-08 (batch-granola zero-out) and 66 on 2026-08-14 (inventory reconciliation) account for 58% of the window's volume. Makes/packs run ~7 days a week; receipts and ships are episodic (multi-weekday gaps are real business cadence, e.g. no ships 7/17–7/21).

### 2.2 Null / blank rates

| Event type | Field | Null/blank | Rate |
|---|---|---:|---:|
| receive (57 txns) | shipper_name, bol_reference, cases_received, case_size_lb | 0 | 0% |
| receive | lots without any supplier lot code | 0 / 57 lots | 0% |
| receive | expected_receipt_id (ER link) | 57 | 100% (feature live only since 8/18; 1 receipt since, unlinked) |
| expected_receipt | — | 0 rows ever created | — |
| ship (254 txns) | customer_name | 0 | 0% |
| ship | order_reference | 229 | 90% (the SO-ship path never writes it; linkage lives in `sales_order_shipments`) |
| ship | bol_reference / shipper_name (dispatch proof) | 254 | 100% (no dispatch-proof columns are populated anywhere) |
| ship | txns without lot lines / without shipment_lines | 0 / 0 | 0% |
| ship | txns with no `sales_order_shipments` link (standalone) | 25 | 10% |
| shipment_lines (616) | sales_order_line_id | 161 | 26% |
| so_allocation (1 row) | lot_id | 1 | 100% (n=1; status `released`, reason present) |
| make (160) | missing output lines / missing ingredient-lot consumption | 0 / 0 | 0% |
| pack (286) | missing lines / missing ILC | 0 / 2 | 0% / 0.7% |
| ingredient_lot_consumption (1,360 rows) | ingredient_lot_id, ingredient_product_id, transaction_id | 0 | 0% |
| fg_lot (352: 194 pack, 158 production) | lot_code blank / non-active | 0 / 0 | 0% |
| adjust (185 txns) | adjust_reason | 1 | 0.5% |
| inventory_adjustments (130 rows) | lot_id, reason_code, adjusted_by | 0 | 0% |
| inventory_adjustments | suspected_supplier (optional, found-inventory attribution) | 95 | 73% |

### 2.3 Entry lag (event `business_date` vs `created_at`, NY days; trustworthy rows only)

| Type | n measurable | Entered late (>0d) | p50 | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| receive | 5 | 0 | 0 | 0 | 0 | 0 |
| make | 20 | 0 | 0 | 0 | 0 | 0 |
| pack | 48 | 2 | 0 | 0 | 3 | 3 |
| ship | 47 | 7 | 0 | 3 | 11.1 | 18 |
| adjust | 79 | **66 (84%)** | **3** | 3 | 20.1 | 24 |

Excluded as unmeasurable (`created_at_source='migration_backfill_039'`): receive 52, ship 207, make 140, pack 238, adjust 106 — everything entered before 2026-08-11 carries the backfill timestamp, so timeliness is only measurable for the last ~2 weeks.

### 2.4 Other

- Expected receipts: 0 rows ever → lead-time metrics empty.
- SO allocations: 1 row ever — the allocate+release smoke test from the 044-series deploy earlier today (2026-08-24 14:17 ET, enforcement flag OFF). No organic usage yet.
- Ledger corrections in window: 1 void, 0 amends.

---

## 3. Grades

Scale: A = reliable for a recall trace, B = minor gaps, C = material gaps, D = cannot be relied on.

### Receipts
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **A** | 0% nulls on supplier, BOL, cases, case size; every received lot carries a supplier lot code. |
| Timeliness | **B** | All 5 measurable entries were same-day, but 52 of 57 predate trustworthy `created_at`, so the record is too thin to prove the habit. |
| Linkage | **B** | Lot and product FKs 100%; but supplier is free text with no FK, and 0 receipts link to expected receipts (feature is 6 days old). |

### Expected receipts
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **D** | Zero rows have ever been created — the main entry channel (office GPT schema v3.5.0) is still not pasted, so the feature is dead weight. |
| Timeliness | **D** | No data to measure. |
| Linkage | **D** | The `transactions.expected_receipt_id` link has never been exercised (1 receipt since go-live, unlinked). |

### Shipments
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **C** | Customer always present, but order_reference is blank on 90% of ship txns and no dispatch proof (BOL, shipper, signature, POD) is captured anywhere. |
| Timeliness | **B** | 85% same-day, p90 = 3 days, but a tail to 18 days late. |
| Linkage | **B** | 100% of ships have lot-level lines and shipment_lines; 90% tie to a sales order, but 26% of shipment lines lack a sales_order_line link and lot linkage depends entirely on `transaction_lines` (no lot on `shipment_lines`). |

### Sales-order allocations
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **D** | 1 allocation ever, and it is the deploy smoke test from earlier today (2026-08-24, enforcement OFF) — no organic usage to audit. |
| Timeliness | **D** | No event-date column exists and there is no volume to measure. |
| Linkage | **D** | FK design is strong on paper (SO, line, lot, ship txn), but unexercised as of this audit. |

### Production batches (make)
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **A** | 0 of 160 makes missing output lines or ingredient-lot consumption; 1,360 ILC rows with zero nulls. |
| Timeliness | **A** | All 20 measurable makes entered same-day; production runs ~7 days/week with no unexplained gaps. |
| Linkage | **A** | Full two-way genealogy: input lots via ILC FKs, output lots via transaction_lines, `entry_source='production_output'` stamped. |

### Finished-goods lots
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **A** | 352 lots, zero blank lot codes, all active, entry_source stamped on every one. |
| Timeliness | **B** | Lots are created atomically with their make/pack txn (good), but they have no event-date column of their own, so their timeliness is only as good as the parent transaction's. |
| Linkage | **A** | Product FK plus full genealogy through the creating transaction; recall trace lot→batch→ingredient lots works end-to-end. |

### Inventory adjustments
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **A−** | Only 1 of 185 missing a reason; found-inventory rows fully attributed (suspected_supplier 73% blank, but it is genuinely unknown for found stock). |
| Timeliness | **D** | 84% of measurable adjustments entered late (p50 = 3 days, max 24) and volume arrives in bursts (66 on 8/14, 41 on 6/8) — adjustments are reconciliation catch-up, not day-of recording. |
| Linkage | **B** | Ledger side has mandatory lot FKs, but the `inventory_adjustments` audit table has no FK constraints at all, so its lot/product references are unenforced. |

### Packaging entries (pack)
| Dimension | Grade | Justification |
|---|---|---|
| Completeness | **A−** | 0 of 286 missing lines; only 2 missing batch-lot consumption (0.7%). |
| Timeliness | **A−** | 46 of 48 measurable packs same-day, worst case 3 days. |
| Linkage | **A−** | Batch→FG lot genealogy complete except the 2 packs with no ILC rows, which break the ingredient-level trace for their FG lots. |

---

## 4. Appendix — daily entry matrix (raw psql output)

```
    day     | dow | receive | exp_rcpt | ship | so_alloc | make | pack | fg_lot | adjust
------------+-----+---------+----------+------+----------+------+------+--------+--------
 2026-05-27 | Wed |       3 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-05-28 | Thu |       1 |        0 |    4 |        0 |    6 |    6 |     11 |      0
 2026-05-29 | Fri |       0 |        0 |    0 |        0 |    1 |    2 |      2 |      0
 2026-05-30 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-05-31 | Sun |       0 |        0 |    0 |        0 |    1 |    2 |      2 |      0
 2026-06-01 | Mon |       1 |        0 |    8 |        0 |    2 |   10 |      9 |      1
 2026-06-02 | Tue |       2 |        0 |    3 |        0 |    4 |    3 |      7 |      0
 2026-06-03 | Wed |       0 |        0 |    7 |        0 |    3 |    2 |      4 |      1
 2026-06-04 | Thu |       0 |        0 |    9 |        0 |    1 |    8 |      7 |      1
 2026-06-05 | Fri |       2 |        0 |    0 |        0 |    1 |    2 |      2 |      0
 2026-06-06 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-06-07 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-06-08 | Mon |       1 |        0 |    0 |        0 |    5 |    6 |     11 |     41
 2026-06-09 | Tue |       2 |        0 |    2 |        0 |    3 |    2 |      5 |      1
 2026-06-10 | Wed |       3 |        0 |    8 |        0 |    2 |    3 |      5 |      0
 2026-06-11 | Thu |       0 |        0 |   22 |        0 |    1 |    6 |      4 |      0
 2026-06-12 | Fri |       1 |        0 |    5 |        0 |    2 |    3 |      3 |      0
 2026-06-13 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-06-14 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-06-15 | Mon |       0 |        0 |    6 |        0 |    1 |    4 |      4 |      0
 2026-06-16 | Tue |       0 |        0 |    2 |        0 |    2 |    2 |      3 |      1
 2026-06-17 | Wed |       0 |        0 |    0 |        0 |    2 |    3 |      3 |      1
 2026-06-18 | Thu |       0 |        0 |    5 |        0 |    3 |    4 |      6 |      2
 2026-06-19 | Fri |       0 |        0 |    2 |        0 |    2 |    6 |      6 |      0
 2026-06-20 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-06-21 | Sun |       0 |        0 |    0 |        0 |    1 |    1 |      2 |      0
 2026-06-22 | Mon |       3 |        0 |    0 |        0 |    3 |    5 |      6 |      2
 2026-06-23 | Tue |       0 |        0 |    5 |        0 |    3 |    5 |      5 |      1
 2026-06-24 | Wed |       1 |        0 |    3 |        0 |    4 |    4 |      6 |      3
 2026-06-25 | Thu |       1 |        0 |    4 |        0 |    3 |    2 |      4 |      1
 2026-06-26 | Fri |       0 |        0 |    1 |        0 |    2 |    2 |      3 |      0
 2026-06-27 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-06-28 | Sun |       1 |        0 |    0 |        0 |    2 |    1 |      3 |      0
 2026-06-29 | Mon |       2 |        0 |    6 |        0 |    6 |    8 |     12 |      4
 2026-06-30 | Tue |       2 |        0 |    4 |        0 |    5 |    8 |     12 |      1
 2026-07-01 | Wed |       1 |        0 |   10 |        0 |    2 |    8 |      8 |      1
 2026-07-02 | Thu |       0 |        0 |    1 |        0 |    2 |    5 |      5 |      1
 2026-07-03 | Fri |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-04 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-05 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-06 | Mon |       2 |        0 |    0 |        0 |    2 |    5 |      4 |      1
 2026-07-07 | Tue |       0 |        0 |    3 |        0 |    3 |    6 |      7 |      2
 2026-07-08 | Wed |       0 |        0 |    8 |        0 |    4 |    5 |      9 |      3
 2026-07-09 | Thu |       2 |        0 |    5 |        0 |    2 |    5 |      6 |      3
 2026-07-10 | Fri |       1 |        0 |    0 |        0 |    2 |    3 |      5 |      1
 2026-07-11 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-12 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-13 | Mon |       2 |        0 |   14 |        0 |    2 |    6 |      7 |      1
 2026-07-14 | Tue |       0 |        0 |    0 |        0 |    2 |    3 |      3 |      2
 2026-07-15 | Wed |       1 |        0 |    4 |        0 |    2 |    4 |      4 |      2
 2026-07-16 | Thu |       0 |        0 |    2 |        0 |    2 |    3 |      2 |      5
 2026-07-17 | Fri |       0 |        0 |    0 |        0 |    2 |    1 |      2 |      0
 2026-07-18 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-19 | Sun |       0 |        0 |    0 |        0 |    1 |    2 |      2 |      0
 2026-07-20 | Mon |       0 |        0 |    0 |        0 |    0 |    1 |      1 |      0
 2026-07-21 | Tue |       3 |        0 |    1 |        0 |    5 |    9 |     13 |      7
 2026-07-22 | Wed |       0 |        0 |    7 |        0 |    4 |    7 |      8 |      2
 2026-07-23 | Thu |       0 |        0 |    0 |        0 |    2 |    3 |      4 |      1
 2026-07-24 | Fri |       2 |        0 |    4 |        0 |    2 |    3 |      4 |      4
 2026-07-25 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-07-26 | Sun |       0 |        0 |    0 |        0 |    1 |    2 |      3 |      0
 2026-07-27 | Mon |       0 |        0 |    0 |        0 |    2 |    6 |      6 |      0
 2026-07-28 | Tue |       1 |        0 |   16 |        0 |    2 |    3 |      4 |      0
 2026-07-29 | Wed |       0 |        0 |    0 |        0 |    3 |    4 |      5 |      1
 2026-07-30 | Thu |       0 |        0 |    1 |        0 |    2 |    6 |      5 |      0
 2026-07-31 | Fri |       0 |        0 |    0 |        0 |    1 |    2 |      2 |      0
 2026-08-01 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-02 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-03 | Mon |       4 |        0 |    6 |        0 |    4 |    8 |     10 |      2
 2026-08-04 | Tue |       5 |        0 |    6 |        0 |    4 |    6 |      9 |      1
 2026-08-05 | Wed |       1 |        0 |    5 |        0 |    3 |    3 |      5 |      3
 2026-08-06 | Thu |       1 |        0 |    5 |        0 |    3 |    3 |      5 |      3
 2026-08-07 | Fri |       0 |        0 |    0 |        0 |    1 |    3 |      3 |      1
 2026-08-08 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-09 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-10 | Mon |       0 |        0 |    4 |        0 |    2 |    3 |      4 |      0
 2026-08-11 | Tue |       0 |        0 |    0 |        0 |    2 |    7 |      6 |      0
 2026-08-12 | Wed |       0 |        0 |    4 |        0 |    2 |    5 |      6 |      2
 2026-08-13 | Thu |       0 |        0 |    3 |        0 |    4 |    5 |      7 |      0
 2026-08-14 | Fri |       1 |        0 |   10 |        0 |    2 |    6 |      5 |     66
 2026-08-15 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-16 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-17 | Mon |       3 |        0 |   11 |        0 |    2 |    3 |      6 |      3
 2026-08-18 | Tue |       0 |        0 |    0 |        0 |    3 |    3 |      5 |      4
 2026-08-19 | Wed |       0 |        0 |    7 |        0 |    2 |    6 |      6 |      2
 2026-08-20 | Thu |       0 |        0 |    2 |        0 |    2 |    7 |      5 |      0
 2026-08-21 | Fri |       1 |        0 |    0 |        0 |    1 |    5 |      4 |      0
 2026-08-22 | Sat |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-23 | Sun |       0 |        0 |    0 |        0 |    0 |    0 |      0 |      0
 2026-08-24 | Mon |       0 |        0 |    9 |        1 |    0 |    1 |      0 |      0
(90 rows)
```

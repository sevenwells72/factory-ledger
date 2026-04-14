# Factory Ledger Traceability Audit

**Date:** 2026-03-24
**Scope:** Read-only audit of trace endpoints, ship flow, lot lifecycle, and data integrity
**Trigger:** Broken backward traceability on SO-260318-005 (Rainbow Sprinkles 25 LB to American Classic Specialties)

---

## 1. Architecture Summary

### Core Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `products` | Product registry | id, name, type (ingredient/batch/finished), case_size_lb, parent_batch_product_id, is_service |
| `lots` | Lot tracking | id, product_id, lot_code, entry_source, supplier_lot_code, lot_type, received_at, status |
| `transactions` | Immutable event log | id, type (receive/ship/make/pack/adjust), timestamp, status (posted/voided), customer_name, shipper_name, bol_reference |
| `transaction_lines` | Double-entry ledger lines | id, transaction_id, product_id, lot_id, quantity_lb (+in / -out) |
| `ingredient_lot_consumption` | Production traceability | transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb |
| `batch_formulas` | BOM recipes | product_id, ingredient_product_id, quantity_lb |

### Sales/Shipping Tables

| Table | Purpose |
|-------|---------|
| `sales_orders` / `sales_order_lines` | Order headers and line items |
| `shipments` | Shipment header (one per ship/commit) |
| `shipment_lines` | Per-product detail within a shipment |
| `sales_order_shipments` | Links ship transactions to SO lines |

### Auxiliary Tables

| Table | Purpose |
|-------|---------|
| `lot_supplier_codes` | Commingled receipt supplier lot breakdown |
| `lot_reassignments` | Audit trail for product reassignment |
| `inventory_adjustments` | Found inventory audit trail |

### Lot Flow Diagram

```
Supplier ──receive──▸ Ingredient Lot ──make──▸ Batch Lot ──pack──▸ FG Lot ──ship──▸ Customer
                            │                                                  ▲
                            └──────────── direct ship (buy-resell) ────────────┘
                                     ↑ THIS PATH IS NOT TRACED ↑
```

### Inventory Model

There is **no balance column** on `lots`. On-hand is computed as `SUM(transaction_lines.quantity_lb) WHERE lot_id = X`. All movements are append-only transaction_lines rows (positive = inflow, negative = outflow).

---

## 2. Trace Endpoint Coverage Matrix

### Dedicated Trace Endpoints

| Endpoint | Direction | Chain Covered | Chain NOT Covered |
|----------|-----------|---------------|-------------------|
| `GET /trace/batch/{lot_code}` | Backward | batch → ingredient lots (via `ingredient_lot_consumption`) with supplier origin (shipper, BOL, received date) | batch → forward to shipment/customer |
| `GET /trace/ingredient/{lot_code}` | Forward | ingredient → downstream batches (via `ingredient_lot_consumption`) | ingredient → direct shipment to customer; supplier origin info (shipper, BOL) |
| Helper: `_trace_ingredient_backward()` | Backward | receive transaction → supplier origin + downstream batches | Forward to shipment/customer |

### Trace-Adjacent Endpoints

| Endpoint | Purpose | Trace Value |
|----------|---------|-------------|
| `GET /lots/by-supplier-lot/{code}` | Recall lookup: supplier lot → internal lots | Searches both `lots.supplier_lot_code` and `lot_supplier_codes` table |
| `GET /lots/by-code/{lot_code}` | Lot metadata lookup | No forward/backward trace |
| `GET /dashboard/api/lot/{lot_code}` | Full transaction timeline for a lot | Shows all transaction types (receive, ship, make, adjust) as a flat timeline — the closest thing to a complete lifecycle view |
| `GET /dashboard/api/activity/shipments` | Recent shipping log | Lists ship transactions with lots/quantities, no backward trace |

### Integrity Check (`GET /audit/integrity`)

| Check | Severity | What It Catches |
|-------|----------|-----------------|
| `production_missing_ilc` | CRITICAL | Make transactions with no ingredient_lot_consumption rows (trace dead ends) |
| `ship_missing_shipment_lines` | MAJOR | Ship transactions after 2026-02-27 with no shipment_lines |
| `lots_missing_supplier_lot_code` | MAJOR | Received lots with blank supplier_lot_code |

---

## 3. Identified Gaps

### GAP-1: Direct-ship-from-ingredient is invisible to trace endpoints
**Severity: CRITICAL**

When a product is bought and resold (e.g., Rainbow Sprinkles 25 LB), it skips make/pack and ships directly from ingredient inventory. The ship transaction creates negative `transaction_lines` against the ingredient lot, but:

- `/trace/ingredient/{lot_code}` only queries `ingredient_lot_consumption` for downstream **batches** — it never looks for ship transactions against the lot
- `/trace/batch/{lot_code}` falls back to `_trace_ingredient_backward()` for received lots, which also only looks at `ingredient_lot_consumption`
- No trace endpoint queries `transaction_lines` for `type='ship'` transactions against an ingredient lot

**Impact:** FDA recall scenario — cannot answer "which customers received lot X of ingredient Y?" for any buy-resell product.

**Files:** `main.py` lines 3203-3420 (`trace_batch`, `trace_ingredient`, `_trace_ingredient_backward`)

---

### GAP-2: No forward trace from any lot to customer/shipment
**Severity: CRITICAL**

Neither trace endpoint follows the chain forward to identify which customer received a lot. The trace stops at:
- For ingredients: "which batches consumed this lot"
- For batches: "which ingredient lots went into this batch"

There is no endpoint answering: "Given lot X, which customers received it, when, and in what quantity?"

The dashboard lot timeline (`/dashboard/api/lot/{lot_code}`) shows ship transactions in a flat list, but this is not a structured trace — it doesn't return customer names, order numbers, or shipment dates in a recall-ready format.

**Impact:** Cannot satisfy FDA 24-hour recall requirement for forward traceability without manual timeline parsing.

**Files:** `main.py` lines 3203-3420

---

### GAP-3: Standalone `/ship` does not create `shipments` or `shipment_lines`
**Severity: HIGH**

The standalone `POST /ship` endpoint (line 2049) only creates `transactions` + `transaction_lines`. It never inserts into `shipments` or `shipment_lines`. Only order-based `POST /sales/orders/{id}/ship` creates those records.

**Impact:**
- Standalone shipments are invisible to any query relying on `shipments`/`shipment_lines`
- Packing slips cannot be generated for standalone shipments
- The integrity check `ship_missing_shipment_lines` flags these as failures
- Migration 018 manually backfilled 11 pre-migration standalone shipments, confirming this was a known issue — but the gap persists in current code

**Files:** `main.py` line 2049 (`ship_inventory`), compare with line 4945 (`ship_sales_order`)

---

### GAP-4: `/trace/ingredient` omits supplier origin info
**Severity: HIGH**

The `/trace/ingredient/{lot_code}` endpoint returns downstream batches but does NOT return supplier origin information (shipper_name, bol_reference, received date). The `_trace_ingredient_backward()` helper (used internally by `/trace/batch`) does include this info, but it's not exposed through `/trace/ingredient`.

**Impact:** Calling `/trace/ingredient` for a recall gives you forward consumption data but not where the ingredient came from. Must make additional API calls.

**Files:** `main.py` line 3352 (`trace_ingredient`) vs line 3288 (`_trace_ingredient_backward`)

---

### GAP-5: Void does not cascade to shipment/sales-order records
**Severity: HIGH**

`POST /void/{transaction_id}` creates reversal transaction_lines and marks the original as voided, but does NOT:
- Delete or void `shipment_lines` rows
- Delete or void `sales_order_shipments` rows
- Reduce `sales_order_lines.quantity_shipped_lb`
- Update `sales_order_lines.line_status` or `sales_orders.status`

**Impact:** Voiding a ship transaction restores lot inventory but leaves the sales order showing the quantity as still shipped. Order status is now inconsistent with actual inventory.

**Files:** `main.py` void endpoint (search for `/void/`)

---

### GAP-6: Supplier lot is never set automatically during ship or receive-to-ship
**Severity: HIGH**

`supplier_lot_code` on the `lots` table is set during `/receive` (required field) and can be updated via `PATCH /lots/{lot_code}/supplier-lot`. It is **never** set or propagated during any ship operation. For the Rainbow Sprinkles case, the lot was received but the supplier lot field appears blank — this means either:
- The receive didn't set it (pre-requirement enforcement)
- The lot was created by a path that doesn't require it (found inventory, adjustment)

The `PATCH /lots/{lot_code}/supplier-lot` endpoint exists but is purely manual — no automated flow calls it.

**Files:** `main.py` line 1852 (receive commit), PATCH endpoint for supplier-lot

---

### GAP-7: FIFO bypass on standalone `/ship`
**Severity: MEDIUM**

The standalone `/ship` endpoint accepts an optional `lot_code` parameter. If provided, it pins the shipment to that specific lot, bypassing FIFO. The order-based `/sales/orders/{id}/ship` has no lot override — FIFO is mandatory.

**Impact:** Users can manually select lots on standalone shipments, potentially shipping newer inventory before older. Not necessarily a bug, but inconsistent with the FIFO guarantee and could cause traceability confusion if the wrong lot is specified.

**Files:** `main.py` line 2049, `ShipRequest` model at line 664

---

### GAP-8: `shipment_lines` lacks `lot_id` column
**Severity: MEDIUM**

The `shipment_lines` table stores `transaction_id` and `product_id` but not `lot_id`. Lot-level traceability requires joining through `transaction_lines` to get `lot_id`. If a single transaction shipped from multiple lots (FIFO spanning lots), the join produces multiple rows, making it harder to determine exactly how much came from each lot for a given shipment line.

**Files:** `migrations/013_shipment_tables.sql` line 47

---

### GAP-9: "UNKNOWN" lot code in pack preview
**Severity: MEDIUM**

The string `"UNKNOWN"` is used as a fallback lot code in the `/pack` preview (line 2840) when no source lots exist and no `target_lot_code` was provided. If a user commits a pack operation in a state where this fallback fires, a lot with code "UNKNOWN" could be created in the database. This lot would then be shippable with no meaningful traceability.

**Impact:** Any shipment line referencing lot code "UNKNOWN" is a traceability dead end.

**Files:** `main.py` line 2840

---

### GAP-10: Lot reassignment mutates historical transaction_lines
**Severity: MEDIUM**

`POST /lots/{lot_id}/reassign` changes `product_id` on the lot AND all its `transaction_lines` in place via UPDATE. While logged in `lot_reassignments`, the actual ledger entries are mutated, breaking the append-only principle. Historical transaction_lines no longer reflect what product they originally recorded.

**Files:** `main.py` reassign endpoint

---

### GAP-11: Lot merge mutates historical records
**Severity: MEDIUM**

`POST /admin/lots/merge` moves `transaction_lines` and `ingredient_lot_consumption` from source to target lot via UPDATE. No transfer transaction is created. The source lot is marked `status='merged'` but the mutation is invisible in the ledger.

**Files:** `main.py` merge endpoint

---

### GAP-12: Migration 024 closed SO-260213-001 without inventory deductions
**Severity: MEDIUM (historical)**

Migration 024 marks 4 order lines (2,402 lb total) as fulfilled by directly setting `quantity_shipped_lb = quantity_lb` and `line_status = 'fulfilled'` — but no `transactions`, `transaction_lines`, `shipments`, or `shipment_lines` records were created. The lot balances for those products are overstated.

**Files:** `migrations/024_close_so260213001_juliette.sql`

---

### GAP-13: No end-to-end recall trace endpoint
**Severity: HIGH**

A full recall trace requires: supplier lot → internal ingredient lots → batches → finished goods → customers. Currently this requires chaining 3-4 API calls, and the final link (lot → customer) has no dedicated endpoint (see GAP-2). There is no single endpoint that provides a complete forward or backward recall trace.

**Files:** All trace endpoints in `main.py` lines 3203-3420

---

## 4. Specific Finding: SO-260318-005 Rainbow Sprinkles

**What happened:** 1,250 lb of Rainbow Sprinkles 25 LB shipped to American Classic Specialties. The shipment references lot `26-03-20-INVE-001`.

**Why trace is broken:**
1. Lot `26-03-20-INVE-001` was likely created via `/receive` or `/inventory/found` (entry_source = 'received' or 'found_inventory')
2. It was shipped directly — no make/pack step
3. `/trace/ingredient` only looks at `ingredient_lot_consumption` for downstream batches → finds nothing (no batches consumed this lot)
4. The ship transaction created negative `transaction_lines` against this lot, but no trace endpoint queries for ship-type transaction_lines on ingredient lots
5. The supplier lot field is blank → either it was received before the requirement was enforced, or it was created via found-inventory

**Root cause confirmed:** The trace logic only follows ingredient → batch (via `ingredient_lot_consumption`). A direct ingredient → ship path via `transaction_lines` is never queried by any trace endpoint.

---

## 5. Recommended Fixes

### FIX-1: Add forward shipment trace to `/trace/ingredient` and `/trace/batch`
**Addresses:** GAP-1, GAP-2, GAP-13
**Complexity:** Medium
**Files:** `main.py` (trace endpoints)

After querying `ingredient_lot_consumption` for downstream batches, also query:
```sql
SELECT t.id, t.timestamp, t.customer_name, t.order_reference,
       tl.quantity_lb, tl.lot_id
FROM transaction_lines tl
JOIN transactions t ON t.id = tl.transaction_id
WHERE tl.lot_id = {lot_id}
  AND tl.quantity_lb < 0
  AND t.type = 'ship'
  AND t.status = 'posted'
```
Add the results as a `shipments` array in the trace response. For `/trace/batch`, also trace forward from the batch lot through pack outputs to ship transactions.

### FIX-2: Create `/trace/recall/forward` and `/trace/recall/backward` endpoints
**Addresses:** GAP-13
**Complexity:** High
**Files:** `main.py`

- **Forward:** Given a supplier lot code, return: internal lots → batches → FG lots → customers with quantities and dates
- **Backward:** Given a customer + date range, return: shipment lots → batches → ingredient lots → suppliers

### FIX-3: Make standalone `/ship` create `shipments` + `shipment_lines`
**Addresses:** GAP-3
**Complexity:** Medium
**Files:** `main.py` line 2049

Mirror the shipment record creation from the order-based ship endpoint. Create a `shipments` row (with `sales_order_id = NULL` for standalone) and `shipment_lines` rows for each product shipped. The `shipments` table already allows nullable `sales_order_id` per migration 018 alterations.

### FIX-4: Add `lot_id` column to `shipment_lines`
**Addresses:** GAP-8
**Complexity:** Low
**Files:** New migration + `main.py` (ship endpoints)

```sql
ALTER TABLE shipment_lines ADD COLUMN lot_id INTEGER REFERENCES lots(id);
```
Populate it during ship commits. Backfill existing rows by joining through `transaction_lines`.

### FIX-5: Cascade void to shipment and sales order records
**Addresses:** GAP-5
**Complexity:** Medium
**Files:** `main.py` (void endpoint)

When voiding a ship transaction:
1. Delete or mark void the associated `shipment_lines` rows
2. Delete or mark void `sales_order_shipments` rows
3. Recompute `sales_order_lines.quantity_shipped_lb` and `line_status`
4. Recompute `sales_orders.status`

### FIX-6: Add supplier origin info to `/trace/ingredient` response
**Addresses:** GAP-4
**Complexity:** Low
**Files:** `main.py` line 3352

Reuse the receive-transaction query from `_trace_ingredient_backward()` to include shipper_name, bol_reference, and received date in the `/trace/ingredient` response.

### FIX-7: Backfill supplier lot for lot 26-03-20-INVE-001
**Addresses:** The specific incident
**Complexity:** Low
**Files:** New migration

```sql
UPDATE lots
SET supplier_lot_code = '550075853'
WHERE lot_code = '26-03-20-INVE-001'
  AND product_id = (SELECT id FROM products WHERE name = 'Rainbow Sprinkles 25 LB');
```

### FIX-8: Prevent "UNKNOWN" lot codes from being committed
**Addresses:** GAP-9
**Complexity:** Low
**Files:** `main.py` line 2840 (pack endpoint)

Add validation in the pack commit path to reject or auto-generate a valid lot code instead of allowing "UNKNOWN" to persist.

---

## 6. Priority Matrix

| Priority | Gap | Fix | Impact |
|----------|-----|-----|--------|
| P0 | GAP-1, GAP-2 | FIX-1 | FDA recall compliance — cannot trace lots to customers |
| P0 | GAP-3 | FIX-3 | Standalone shipments have no structured records |
| P1 | GAP-13 | FIX-2 | No single-call recall trace endpoint |
| P1 | GAP-5 | FIX-5 | Voided shipments leave orphaned SO records |
| P1 | GAP-4 | FIX-6 | Trace missing supplier origin |
| P1 | GAP-6 | FIX-7 | Specific incident remediation |
| P2 | GAP-8 | FIX-4 | Denormalized lot traceability on shipment_lines |
| P2 | GAP-9 | FIX-8 | UNKNOWN lot prevention |
| P2 | GAP-7 | — | FIFO bypass is by design but worth documenting |
| P3 | GAP-10, GAP-11 | — | Ledger mutation — consider transfer transactions in future |
| P3 | GAP-12 | — | Historical data; manual inventory reconciliation needed |

---

## 7. Files Examined

| File | Lines | Areas Audited |
|------|-------|---------------|
| `main.py` | 8,455 | All trace, ship, receive, make, pack, void, merge, reassign, adjust, packing slip, dashboard endpoints |
| `migrations/013_shipment_tables.sql` | 64 | shipments, shipment_lines, sales_order_shipments schema |
| `migrations/018_backfill_pre_migration_shipments.sql` | — | Historical shipment backfill |
| `migrations/024_close_so260213001_juliette.sql` | — | Order closed without inventory deduction |
| `migrations/025_set_supplier_lot_sprinkles_26-03-10-FOUN-001.sql` | — | Prior supplier lot backfill |
| `GPT_INSTRUCTIONS.md` | — | Intended behavior documentation |
| `CONTEXT.md` | — | Schema documentation |

---

*Audit performed by Claude Code. No files were modified.*

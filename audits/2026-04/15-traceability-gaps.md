# Category 5 — Traceability Gaps (FDA Recall Risk)

Any finding affecting the ability to answer FDA recall questions is CRITICAL regardless of other factors.

This audit cross-references [TRACEABILITY_AUDIT_2026-03-24.md](../../TRACEABILITY_AUDIT_2026-03-24.md) (the prior audit that identified GAPs 1–13) against current code. Status assigned: **FIXED | PARTIAL | STILL OPEN**.

Summary table:

| Gap | 2026-03-24 severity | Current status | This audit finding |
|---|---|---|---|
| GAP-1 direct-ship invisible | CRITICAL | FIXED | F05-01 |
| GAP-2 no forward trace | CRITICAL | FIXED | F05-01 |
| GAP-3 standalone ship w/o shipments | HIGH | FIXED | F05-02 |
| GAP-4 trace/ingredient no supplier origin | HIGH | PARTIAL | F05-03 |
| GAP-5 void doesn't cascade | HIGH | **STILL OPEN** | F05-05 |
| GAP-6 auto supplier lot on ship/receive/found | HIGH | PARTIAL | F05-06 |
| GAP-7 FIFO bypass on standalone ship | MEDIUM | STILL OPEN | F05-07 |
| GAP-8 shipment_lines lacks lot_id | MEDIUM | STILL OPEN | F05-08 |
| GAP-9 "UNKNOWN" pack fallback | MEDIUM | PARTIAL (preview still leaks) | F05-09 |
| GAP-10 reassign mutates history | MEDIUM | STILL OPEN | F05-10 |
| GAP-11 merge mutates history | MEDIUM | STILL OPEN | F05-11 |
| GAP-12 SO-260213-001 unreconciled | MEDIUM historical | STILL OPEN | F05-12 |
| GAP-13 end-to-end recall endpoint | HIGH | PARTIAL | F05-13 |
| **NEW** ship_order missing `is_service` guard | — | **NEW** | F05-04 |
| **NEW** /inventory/found skips supplier_lot_code | — | **NEW** | F05-06 (sub) |

---

### [F05-01] GAP-1 / GAP-2 Direct-ship trace + forward trace — FIXED
**Severity**: resolved
**Files**: [main.py:3693–3730](../../main.py#L3693) `_trace_ingredient_backward` now queries `ship` transactions directly against `tl.lot_id` and returns `direct_shipments`; `trace_ingredient` at [main.py:3838–3874](../../main.py#L3838) and `trace_supplier_lot` at [main.py:4007–4023](../../main.py#L4007) do the same. `trace_supplier_lot` returns a forward chain (internal lots → production usage → customer shipments) in one call.
**Verification**: Changelog entries #12 and #18 confirm these land.

---

### [F05-02] GAP-3 Standalone `/ship` creates shipments + shipment_lines — FIXED
**Severity**: resolved
**Files**: [main.py:2552–2562](../../main.py#L2552) — INSERT `shipments(transaction_id, shipped_at, customer_id)` + INSERT `shipment_lines(shipment_id, transaction_id, product_id, quantity_lb)` in a loop. Changelog #15 + migration 030 backfilled pre-existing standalone ships.
**Verification**: `audit_integrity` check #3 ("ship_missing_shipment_lines", [main.py:9172](../../main.py#L9172)) would catch any regression.

---

### [F05-03] GAP-4 `/trace/ingredient` still omits supplier origin — PARTIAL
**Severity**: high
**Files**: [main.py:3757–3915](../../main.py#L3757).
**Current behavior**: The response includes `supplier_lot_code` at [main.py:3888](../../main.py#L3888) and, for output lots, `upstream_ingredients` with their supplier_lot_code ([main.py:3817](../../main.py#L3817)). But unlike `trace_batch` (which returns a full `supplier` block with `shipper_name`, `bol_reference`, `received_date` via `_trace_ingredient_backward`), the non-output branch of `/trace/ingredient` never looks up the `receive` transaction and never returns a `supplier` dict.
**Risk**: Recall workflow for an ingredient lot must issue a second call to `/trace/batch` or the dashboard `/dashboard/api/lot/...` to get supplier metadata. During an FDA 24-hour window, every extra call is friction.
**Recommended fix**: In the non-output branch (after [main.py:3822](../../main.py#L3822)), run the same supplier-origin query from `_trace_ingredient_backward` ([main.py:3646–3666](../../main.py#L3646)) and add `supplier` to the response dict at [main.py:3886](../../main.py#L3886).
**Effort**: small

---

### [F05-04] NEW — `ship_order` lacks the `is_service` guard
**Severity**: CRITICAL (FDA recall: ship_order is the primary ship path for any invoiced order; if it can't close orders with a pallet charge, operators revert to the standalone `/ship` which is lossier)
**Files**: [main.py:5646–5855](../../main.py#L5646).
**Current behavior**: The line-selection query at [main.py:5711–5719](../../main.py#L5711) / [main.py:5720–5734](../../main.py#L5720) pulls every `sales_order_lines` row with `line_status NOT IN ('fulfilled', 'cancelled')` — no `is_service` filter. The FIFO loop at [main.py:5753–5787](../../main.py#L5753) then tries to find lot inventory for each line. Service products (Pallet Charge, freight) have no lots, so `candidates` is empty, `available=0`, and the inner `actual_ship <= BALANCE_EPSILON` check at [main.py:5771](../../main.py#L5771) sets `"status": "no_stock"` for that line and leaves `all_fully_shipped = False`. Net effect: an order that contains a pallet charge can never transition to `status='shipped'`; it's stuck at `partial_ship` forever.
**Related**: `OrderLineInput` validator and `create_sales_order` + `add_order_lines` ([main.py:4883–4908](../../main.py#L4883) and [main.py:5516–5540](../../main.py#L5516)) DO handle `is_service` (they set `quantity_lb=0` or skip case-weight). `is_service` is also filtered out of weight-totals (changelog entry #1, migration 028). The one path that doesn't is the ship commit.
**Risk**: Any customer invoice with a Pallet Charge can't be closed through the GPT's shipOrder action. Migration 022 and 024 were operator-driven reconciliation migrations that likely stemmed from this exact issue. Operators will keep writing SQL migrations to force-close orders.
**Recommended fix**: In the ship_order line-selection, either (a) auto-fulfill `is_service` lines at ship-time by setting `quantity_shipped_lb = quantity_lb`, `line_status = 'fulfilled'` without any inventory lookup, or (b) exclude `is_service` lines from `all_fully_shipped` calculation entirely and treat their status as derived from the ship transaction. Option (a) is simpler and matches how changelog #13 described the intended behavior.
**Effort**: small (30–60 min)

---

### [F05-05] GAP-5 Void doesn't cascade to shipment/sales-order records — STILL OPEN
**Severity**: high
**Files**: [main.py:3411–3486](../../main.py#L3411) `void_transaction`.
**Current behavior**: Updates `transactions.status = 'voided'` ([main.py:3435–3438](../../main.py#L3435)); inserts reversal `transaction_lines` ([main.py:3452](../../main.py#L3452)). It does NOT:
- Delete or mark void the `shipment_lines` rows for this txn
- Delete or mark void the `sales_order_shipments` rows
- Reduce `sales_order_lines.quantity_shipped_lb`
- Recompute `sales_order_lines.line_status`
- Recompute `sales_orders.status`
**Risk**: Voiding a ship transaction restores lot inventory but leaves the sales order showing the quantity as still shipped. Order status drifts from reality. A subsequent `ship_order` preview uses the stale `quantity_shipped_lb` ([F01-D5](10-monolith-structure.md)) and may refuse to ship the now-un-shipped lines. An FDA recall query against `shipment_lines` returns voided rows as if they were real shipments.
**Recommended fix**: When `txn['type'] == 'ship'`, look up `sales_order_shipments` and `shipment_lines` by `transaction_id`:
1. `UPDATE sales_order_lines SET quantity_shipped_lb = quantity_shipped_lb - X` per line
2. Recompute `line_status` using the same helper proposed in [F01-D5](10-monolith-structure.md)
3. Recompute `sales_orders.status`
4. Either DELETE `shipment_lines`+`sales_order_shipments` rows, or add a `voided_at` column and mark them
5. Add integrity check: `shipment_lines` whose txn is voided but row isn't
**Effort**: medium (half day + migration for optional voided_at column)

---

### [F05-06] GAP-6 Auto supplier lot — PARTIAL (receive FIXED, found-inventory STILL OPEN)
**Severity**: high (FDA recall)
**Files**:
- `/receive` at [main.py:2117–2192](../../main.py#L2117) — FIXED. Defaults `supplier_lot_code` to `lot_code` then `"N/A"` and UPDATEs the lot (changelog #20). 
- `/inventory/found` at [main.py:4357–4438](../../main.py#L4357) — **STILL OPEN**. `find_or_create_lot` is called at [main.py:4392–4396](../../main.py#L4392) without any `supplier_lot_code` parameter, and no subsequent UPDATE of `lots.supplier_lot_code` occurs.
- `/inventory/found-with-new-product` at [main.py:4441–4515](../../main.py#L4441) — **STILL OPEN** for the same reason (INSERT at [main.py:4479–4483](../../main.py#L4479)).

**Current behavior**: Lots created via found-inventory have NULL `supplier_lot_code`. The `audit_integrity` check #5 ([main.py:9202–9217](../../main.py#L9202)) only flags received lots (`WHERE entry_source = 'received'`), so it doesn't catch this.

**Risk**: Any lot created through the found-inventory workflow has no supplier lot identifier. For buy-resell products, this recreates the exact GAP-1 incident that the Rainbow Sprinkles 26-03-20-INVE-001 case triggered — the supplier_lot is blank, so a recall against a supplier batch can't find the internal lot.
**Recommended fix**: (1) Add optional `supplier_lot_code` to `AddFoundInventoryRequest` / `AddFoundInventoryWithNewProductRequest`. (2) On INSERT/UPDATE of the lot, default to `suspected_supplier + '-' + estimated_age` or to `'N/A'` if nothing was provided, matching receive's fallback. (3) Broaden `audit_integrity` check #5 to include `entry_source IN ('received', 'found_inventory')`.
**Effort**: small

---

### [F05-07] GAP-7 FIFO bypass on standalone `/ship` (lot_code override) — STILL OPEN
**Severity**: medium
**Files**: [main.py:2437–2447](../../main.py#L2437). If `req.lot_code` is supplied, the endpoint pins to that lot and bypasses FIFO. No reason code, no audit trail, no flag in the response.
**Risk**: Intentional escape hatch, but no visibility. An operator could unintentionally ship newer stock before older with zero record of override.
**Recommended fix**: Require `ship_override_reason` in the request when `lot_code` is pinned and force-standalone is false; log it into `transactions.notes` + a new boolean column `fifo_override` on `transactions`; surface it in `/trace/*` responses so recall investigators see why a non-FIFO lot was consumed.
**Effort**: small

---

### [F05-08] GAP-8 `shipment_lines` lacks `lot_id` column — STILL OPEN
**Severity**: medium (FDA recall friction, not a correctness gap)
**Files**: [migrations/013_shipment_tables.sql:47–54](../../migrations/013_shipment_tables.sql#L47). `shipment_lines` defines `id, shipment_id, transaction_id, sales_order_line_id, product_id, quantity_lb` — no `lot_id`. INSERTs at [main.py:2559–2562](../../main.py#L2559) and [main.py:5807–5810](../../main.py#L5807) don't set it.
**Current behavior**: Recall query "which customer received lot X?" requires joining through `transaction_lines` on `transaction_id`. When a shipment spans multiple lots (FIFO), the join explodes — one shipment_line row but multiple tx_line rows; allocation per-lot must be re-derived.
**Recommended fix**: New migration:
```sql
ALTER TABLE shipment_lines ADD COLUMN IF NOT EXISTS lot_id INTEGER REFERENCES lots(id);
```
Update `/ship` and `/ship_order` to insert one `shipment_lines` row per lot split (not per product). Backfill existing rows by joining through `transaction_lines`.
**Effort**: medium (migration + code change + backfill + one-time validation)

---

### [F05-09] GAP-9 "UNKNOWN" pack fallback — PARTIAL (commit safe, preview still leaks)
**Severity**: medium
**Files**: [main.py:3120](../../main.py#L3120) (preview branch) vs [main.py:3207–3210](../../main.py#L3207) (commit branch).
**Current behavior**: Commit can't produce "UNKNOWN" — guard at [main.py:3167](../../main.py#L3167) and [main.py:3204–3205](../../main.py#L3204) raises 400 when no inventory or under-allocation. Preview still returns `output_lot_code = "UNKNOWN"` in the no-allocation path ([main.py:3120](../../main.py#L3120)).
**Risk**: A user seeing "UNKNOWN" in preview might assume the system auto-creates a placeholder lot (it doesn't — commit would fail). Mostly confusion, not data corruption.
**Recommended fix**: In preview, either (a) raise the same 400 as commit when no allocations, or (b) return `output_lot_code = null` + a warning message.
**Effort**: small

---

### [F05-10] GAP-10 Lot reassignment mutates history — STILL OPEN
**Severity**: medium
**Files**: [main.py:4246–4356](../../main.py#L4246) `reassign_lot`. At [main.py:4313–4315](../../main.py#L4313): `UPDATE transaction_lines SET product_id = %s WHERE lot_id = %s`. Audit row written to `lot_reassignments` at [main.py:4319–4325](../../main.py#L4319) but the ledger is mutated.
**Risk**: Historical queries silently change answer. "How much product X was received last month" depends on when you run the query.
**Recommended fix**: Never UPDATE historical `transaction_lines`. Instead: post a compensating pair of adjust transactions (negative on old product_id, positive on new product_id, same lot_id, same timestamps as now). The append-only invariant of the ledger is preserved.
**Effort**: medium

---

### [F05-11] GAP-11 Lot merge mutates history — STILL OPEN
**Severity**: medium
**Files**: [main.py:7781–7884](../../main.py#L7781) `merge_lots`. At [main.py:7829–7832](../../main.py#L7829): `UPDATE transaction_lines SET lot_id = %s WHERE lot_id = %s`, and [main.py:7836–7839](../../main.py#L7836) same for `ingredient_lot_consumption`. Source lot is marked `status='merged'` with `merged_into_lot_id` at [main.py:7844–7851](../../main.py#L7844).
**Risk**: Same as F05-10. Trace by source lot_code after merge loses all references.
**Recommended fix**: Keep `transaction_lines.lot_id` immutable. Resolve merges at query time by following the `merged_into_lot_id` chain in trace endpoints. Or post offsetting adjust transactions (preferred — preserves the ledger semantics).
**Effort**: medium

---

### [F05-12] GAP-12 SO-260213-001 never reconciled in the ledger — STILL OPEN (historical)
**Severity**: medium (historical data gap)
**Files**: [migrations/024_close_so260213001_juliette.sql](../../migrations/024_close_so260213001_juliette.sql) updates only `sales_order_lines.quantity_shipped_lb` + `sales_orders.status`. No `transactions`, `transaction_lines`, `shipments`, or `shipment_lines` rows ever created. Grep of migrations for "260213" returns only this file — no follow-up reconcile migration exists.
**Risk**: The 2,402 lb of inventory marked "shipped" on this order was never deducted from any lot balance. Current lot balances overstate on-hand for those 4 granola products. If the overstated lots have since gone to zero in practice, the discrepancy has been absorbed silently by later adjusts — but lot traceability for that shipment is permanently lost.
**Recommended fix**: New migration `033_reconcile_so260213001.sql` that inserts a `ship` transaction + deducting `transaction_lines` per product using FIFO against the lots available at 2026-02-26, plus `shipments` + `shipment_lines` rows. Tag the transaction with `notes = 'historical reconciliation: originally recorded via migration 024'`.
**Effort**: medium (migration + manual FIFO allocation based on lot state at 2026-02-26)

---

### [F05-13] GAP-13 End-to-end recall chain — PARTIAL
**Severity**: high
**Files**: [main.py:3918–4073](../../main.py#L3918) `trace_supplier_lot`.
**Current behavior**: `trace_supplier_lot` is effectively the recall endpoint — it resolves internal lots (including commingled via `lot_supplier_codes` at [main.py:3937–3945](../../main.py#L3937)), enumerates `production_usage` + `customer_shipments`, and returns `total_exposure_summary`. But: the `production_usage` loop at [main.py:3987–4003](../../main.py#L3987) identifies batches that consumed a recalled ingredient, but DOES NOT chain forward from those batches to their own ship transactions. The caller must separately call `/trace/batch/{batch_lot_code}` for each returned batch to get customers.
**Risk**: Multi-hop recall (supplier lot → ingredient lot → batch → FG pack → customer ship) requires 2–3 API roundtrips. Under an FDA 24-hour window this is manageable but fragile.
**Recommended fix**: In the `production_usage` loop, also query ship transactions against each batch_lot_id and attach `downstream_shipments` to each batch entry. Then `trace_supplier_lot` becomes a complete one-call recall trace.
**Effort**: medium

---

### NEW findings this audit

### [F05-14] NEW — `audit_integrity` check for voided-but-uncascaded shipments
**Severity**: medium
**Files**: [main.py:9122](../../main.py#L9122).
**Current behavior**: No check flags `shipment_lines` rows whose parent transaction is voided. Given [F05-05](#f05-05-gap-5-void-doesnt-cascade-to-shipmentsales-order-records--still-open), this blind spot is real.
**Recommended fix**: Add check #9: `ship_lines_with_voided_txn`:
```sql
SELECT sl.id, sl.transaction_id
FROM shipment_lines sl
JOIN transactions t ON t.id = sl.transaction_id
WHERE t.status = 'voided'
```
Severity: MAJOR.
**Effort**: small

### [F05-15] NEW — `audit_integrity` check for service-line orders stuck partial
**Severity**: medium
**Files**: [main.py:9122](../../main.py#L9122).
**Current behavior**: Given [F05-04](#f05-04-new--ship_order-lacks-the-is_service-guard), orders with pallet charges are systematically stuck at `status='partial_ship'`. A check would surface the population at risk.
**Recommended fix**: Add check #10: any `sales_orders.status='partial_ship'` whose only non-fulfilled lines have `products.is_service = true`.
**Effort**: small

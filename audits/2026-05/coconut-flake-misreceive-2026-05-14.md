# Coconut Flake Desiccated — Misreceive Audit (2026-05-14)

## Summary

Five lots of 50 lb bags of desiccated coconut flake were physically on site and
needed to be recorded into Factory Ledger:

| Lot  | Bags | lb (intended) |
|------|------|----------------|
| 6012 | 288  | 14,400         |
| 6013 | 256  | 12,800         |
| 6020 | 896  | 44,800         |
| 6036 | 98   | 4,900          |
| 6037 | 64   | 3,200          |
| **Total** | **1,602** | **80,100** |

All five are raw ingredient lots from supplier Franklin Baker, intended to be
booked against product **id 12 "Coconut Flake Desiccated"** (`type=ingredient`,
`uom=lb`). They are *not* finished goods.

## What actually happened in the prior session (2026-05-14, 17:18–17:37 ET)

A prior automated session (`adjusted_by=system`, GPT-driven) ran the same
physical-inventory reconciliation but mis-routed the data:

* **Lot 6013 was never entered at all.**
* **Lot 6012** was split across **two products**: id 10 "Coconut Fancy
  Desiccated" (10,000 lb) and id 12 "Coconut Flake Desiccated" (1,900 lb,
  posted twice).
* **Lot 6020** was split across **two products**: id 12 (1,075 lb, posted
  twice) and id 14 "Coconut Medium Desiccated" (10,000 lb).
* **Lot 6036** was posted under id 12 at 3,300 lb, then posted again at 3,300
  lb (doubled).
* **Lot 6037** was posted under id 12 at 3,400 lb, then posted again at 3,400
  lb (doubled).

Net result: **39,350 lb** spread across **6 (lot, product) combinations** and
**10 transactions** (6 `/inventory/found` calls + 4 follow-up `/adjust`
duplicates) — none of which matches the intended 80,100 lb on five
single-product lots.

The 17:37 batch appears to be an attempted "re-run" that landed *additively*
on top of the 17:18 batch rather than as a correction.

## Discrepancy table

| Lot  | System on-hand (current)                                                            | Intended (image)           | Discrepancy        |
|------|-------------------------------------------------------------------------------------|----------------------------|--------------------|
| 6012 | id 10 @ 10,000 lb + id 12 @ 3,800 lb (1,900 + 1,900 dup)                            | id 12 @ 14,400 lb          | wrong product + qty |
| 6013 | (no rows)                                                                            | id 12 @ 12,800 lb          | missing entirely    |
| 6020 | id 12 @ 2,150 lb (1,075 + 1,075 dup) + id 14 @ 10,000 lb                            | id 12 @ 44,800 lb          | wrong product + qty |
| 6036 | id 12 @ 6,600 lb (3,300 + 3,300 dup)                                                | id 12 @ 4,900 lb           | right product, doubled, off by +1,700 vs single 3,300 entry |
| 6037 | id 12 @ 6,800 lb (3,400 + 3,400 dup)                                                | id 12 @ 3,200 lb           | right product, doubled, off by +3,600 vs single 3,400 entry |

## Bad transactions to reverse (10 rows)

All `txn_type=adjust`, all touch a lot in `{6012, 6013, 6020, 6036, 6037}`,
all created 2026-05-14 between 17:18 and 17:37 ET.

| transaction_id | timestamp                  | lot_code | lot_id | product_id | product_name              | quantity_lb | txn_notes                                  |
|----------------|----------------------------|----------|--------|------------|---------------------------|-------------|---------------------------------------------|
| 911            | 2026-05-14 17:18:52.644105 | 6012     | 610    | 10         | Coconut Fancy Desiccated  | +10,000.0   | Found inventory: found_during_count         |
| 913            | 2026-05-14 17:18:54.425830 | 6012     | 612    | 12         | Coconut Flake Desiccated  | +1,900.0    | Found inventory: found_during_count         |
| 924            | 2026-05-14 17:37:14.379979 | 6012     | 612    | 12         | Coconut Flake Desiccated  | +1,900.0    | Adjustment: 1900.0 lb                       |
| 914            | 2026-05-14 17:18:55.187433 | 6020     | 613    | 12         | Coconut Flake Desiccated  | +1,075.0    | Found inventory: found_during_count         |
| 918            | 2026-05-14 17:18:58.422917 | 6020     | 617    | 14         | Coconut Medium Desiccated | +10,000.0   | Found inventory: found_during_count         |
| 925            | 2026-05-14 17:37:15.212947 | 6020     | 613    | 12         | Coconut Flake Desiccated  | +1,075.0    | Adjustment: 1075.0 lb                       |
| 915            | 2026-05-14 17:18:55.959437 | 6036     | 614    | 12         | Coconut Flake Desiccated  | +3,300.0    | Found inventory: found_during_count         |
| 926            | 2026-05-14 17:37:15.969200 | 6036     | 614    | 12         | Coconut Flake Desiccated  | +3,300.0    | Adjustment: 3300.0 lb                       |
| 916            | 2026-05-14 17:18:56.782138 | 6037     | 615    | 12         | Coconut Flake Desiccated  | +3,400.0    | Found inventory: found_during_count         |
| 927            | 2026-05-14 17:37:16.820297 | 6037     | 615    | 12         | Coconut Flake Desiccated  | +3,400.0    | Adjustment: 3400.0 lb                       |

All ten have `adjust_reason = NULL` (none filled in `transactions.adjust_reason`).
The six `/inventory/found` calls carry `adjust_reason` via the related
`inventory_adjustments` row (see next section); the four `/adjust` calls have
no structured reason.

## Associated inventory_adjustments rows (6 rows)

These were written by the `/inventory/found` endpoint and are the structured
audit metadata for the six 17:18 transactions. The four 17:37 `/adjust`
duplicates did **not** write to `inventory_adjustments`.

| inventory_adjustment_id | lot_code | product_id | adjustment_type | quantity_before | quantity_adjustment | quantity_after | reason_code         | reason_notes                                    | suspected_supplier | estimated_age | adjusted_by |
|-------------------------|----------|------------|-----------------|-----------------|---------------------|----------------|---------------------|--------------------------------------------------|--------------------|---------------|-------------|
| 110                     | 6012     | 10         | found           | 0               | +10,000             | 10,000         | found_during_count  | Physical inventory reconciliation 2026-05-14    | Franklin Baker     | unknown       | system      |
| 112                     | 6012     | 12         | found           | 0               | +1,900              | 1,900          | found_during_count  | Physical inventory reconciliation 2026-05-14    | Franklin Baker     | unknown       | system      |
| 113                     | 6020     | 12         | found           | 0               | +1,075              | 1,075          | found_during_count  | Physical inventory reconciliation 2026-05-14    | Franklin Baker     | unknown       | system      |
| 117                     | 6020     | 14         | found           | 0               | +10,000             | 10,000         | found_during_count  | Physical inventory reconciliation 2026-05-14    | Franklin Baker     | unknown       | system      |
| 114                     | 6036     | 12         | found           | 0               | +3,300              | 3,300          | found_during_count  | Physical inventory reconciliation 2026-05-14    | Franklin Baker     | unknown       | system      |
| 115                     | 6037     | 12         | found           | 0               | +3,400              | 3,400          | found_during_count  | Physical inventory reconciliation 2026-05-14    | Franklin Baker     | unknown       | system      |

## Lot rows created by the prior session (6 rows)

| lot_id | lot_code | product_id | entry_source     | entry_source_notes                              | created_at                  |
|--------|----------|------------|------------------|--------------------------------------------------|-----------------------------|
| 610    | 6012     | 10         | found_inventory  | Physical inventory reconciliation 2026-05-14    | 2026-05-14 17:18:52.492806  |
| 612    | 6012     | 12         | found_inventory  | Physical inventory reconciliation 2026-05-14    | 2026-05-14 17:18:54.274471  |
| 613    | 6020     | 12         | found_inventory  | Physical inventory reconciliation 2026-05-14    | 2026-05-14 17:18:55.036100  |
| 617    | 6020     | 14         | found_inventory  | Physical inventory reconciliation 2026-05-14    | 2026-05-14 17:18:58.271335  |
| 614    | 6036     | 12         | found_inventory  | Physical inventory reconciliation 2026-05-14    | 2026-05-14 17:18:55.808313  |
| 615    | 6037     | 12         | found_inventory  | Physical inventory reconciliation 2026-05-14    | 2026-05-14 17:18:56.630807  |

All six have `supplier_lot_code = NULL` and `entry_source = 'found_inventory'`.

## Remediation — Path A (void + re-enter)

Execution will:

1. Reverse the 10 bad transactions via offsetting `/adjust` calls (one
   reversal per bad transaction line), `reason` prefixed with
   `count_correction:` and referencing this audit file plus the original
   `transaction_id`.
2. After reversal, on-hand will be **0 lb** for every (lot_id, product_id)
   pair in `{(610, 10), (612, 12), (613, 12), (617, 14), (614, 12), (615, 12)}`.
3. The 6 empty lot rows above will be left in place as zero-balance,
   audit-trail rows. They are not deleted. They remain physically tied to
   their (wrong) product_id but carry no inventory.
4. Five fresh `/inventory/found` calls will be made against product id 12
   with `lot_code` matching the image, `performed_by="blubber"`,
   `reason_code="found_during_count"`, notes referencing this audit file as
   the corrected entry that supersedes the prior bad data.

After remediation, the on-hand state for product 12 will reflect the
intended 80,100 lb across the five lots above. Products 10 and 14 will have
zero on-hand under lot codes 6012 and 6020 (and will not have any lot 6012 /
6020 at all *outside* the orphan rows recorded here).

## Why this approach (not lot reassignment, not row deletion)

* **No reassignment**: the bad data isn't a single-product clerical error
  that lot-reassignment can fix; it's a fan-out across three different
  products with doubled quantities. Reversing-adjusts are the cleaner
  ledger move and preserve full history.
* **No row deletion**: deleting `transactions`, `transaction_lines`,
  `inventory_adjustments`, or `lots` rows would destroy the audit trail and
  risk FK damage elsewhere. Zero-balance orphans are intentional.

## Abbreviation note — "FB" on count sheets = Franklin Baker

The "FB" abbreviation that appears on the source count sheet for these five
lots refers to **Franklin Baker** (the supplier), not "Fancy Bakers" — and
not a product attribute at all. Prior sessions (and any future
investigation pass) need to interpret it that way. It is plausible that the
2026-05-14 13:18 ET session's mis-routing of lot 6012 to product id 10
"Coconut **Fancy** Desiccated" was driven in part by misreading "FB" as
"Fancy Bakers" → "Fancy", which then anchored the wrong product
disambiguation for the whole reconciliation.

Confirmed corroborating evidence:
- All six prior `inventory_adjustments` rows already record
  `suspected_supplier = "Franklin Baker"`.
- The pre-existing receive against product id 190 from 2026-04-22 uses
  shipper-code `FRAN` (Franklin Baker) and supplier_lot_code `6011` — same
  numeric series as 6012/6013/6020/6036/6037, supporting that all six lots
  came from one supplier.

Convention going forward: on Coconut counts, "FB" = Franklin Baker
(supplier). Use product `id 12 Coconut Flake Desiccated` (ingredient, lb)
for receipts from this supplier of 50 lb bags of plain desiccated flake.

## Execution result (2026-05-14, post-remediation)

Executed Path A / Option B (10 `/void/{txn_id}` calls + 5 fresh
`/inventory/found` calls) — all 15 API calls returned HTTP 200.

### Voids posted

All 10 originals updated to `status='voided'` with notes suffixed
`| Voided at 2026-05-14 16:18` (server clock; container TZ differs from
plant ET — see Gotchas in MEMORY). Each created an offsetting reversal:

| Voided txn | Reversal txn | (lot_id, product_id) | Reversal qty (lb) |
|------------|--------------|----------------------|-------------------|
| 911 | 932 | (610, 10) | -10,000 |
| 913 | 933 | (612, 12) | -1,900  |
| 924 | 934 | (612, 12) | -1,900  |
| 914 | 935 | (613, 12) | -1,075  |
| 918 | 936 | (617, 14) | -10,000 |
| 925 | 937 | (613, 12) | -1,075  |
| 915 | 938 | (614, 12) | -3,300  |
| 926 | 939 | (614, 12) | -3,300  |
| 916 | 940 | (615, 12) | -3,400  |
| 927 | 941 | (615, 12) | -3,400  |

### Re-entries

| lot_code | product_id | quantity (lb) | lot_id | new lot row? |
|----------|------------|---------------|--------|--------------|
| 6012     | 12         | 14,400        | 612    | no (rode on existing zero-balance row) |
| 6013     | 12         | 12,800        | 624    | **yes** (first time entered) |
| 6020     | 12         | 44,800        | 613    | no |
| 6036     | 12         |  4,900        | 614    | no |
| 6037     | 12         |  3,200        | 615    | no |

All five used `performed_by="blubber-via-claude-code"` (column allows up to
varchar(100)), `reason_code="found_during_count"`,
`suspected_supplier="Franklin Baker"`,
`notes="Physical inventory reconciliation 2026-05-14 — corrected entry,
supersedes prior bad entries voided this session. See
audits/2026-05/coconut-flake-misreceive-2026-05-14.md"`.

### Final on-hand state

Product 12 "Coconut Flake Desiccated":

| lot_code | on_hand_lb |
|----------|------------|
| 6012     | 14,400     |
| 6013     | 12,800     |
| 6020     | 44,800     |
| 6036     | 4,900      |
| 6037     | 3,200      |
| **Sum**  | **80,100** |

Matches the count-sheet total to the pound.

Products 10 and 14 — orphan zero-balance lot rows remain (lot_id 610 on
pid 10, lot_id 617 on pid 14), both at **0 lb**. No phantom inventory.

### Trace verification

`GET /trace/ingredient/{lot}?product_id=12` returned HTTP 200 for all 5
lots with `on_hand_lb` matching the targets above, `used_in_batches=[]`,
`direct_shipments=[]`, `total_shipped_lb=0`.

`GET /trace/ingredient/{lot}` without the disambiguator returns HTTP 409
`ambiguous_lot_code` for **6012** and **6020** because the orphan
zero-balance lot rows on products 10 and 14 still match by lot code alone.
This is correct endpoint behavior and surfaces both matches in the
response, but UI surfaces will need to pass `?product_id=12` (or the
lot_id) for clean traces on these two codes until/unless the orphans are
soft-deleted. Tracked as a follow-up.

## Out of scope for this remediation

* Product id 190 "Desiccated Flake 50 LB" (type=finished, case_size=50.0).
  It has live sales-order history and is not part of this incident.
* Why the prior session hallucinated the splits and quantities. Worth
  investigating as a separate ticket — most likely candidates: GPT
  instruction gap on disambiguating Coconut Fancy / Flake / Medium
  Desiccated, or a /inventory/found flow that didn't surface "lot already
  exists for this product" warnings.

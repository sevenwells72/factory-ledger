The original Q2 restore (flip by last_ship / reverse-coalesce) is superseded. Void-coalesce stays. Restore quantity is whatever that void actually put back on live rows, recorded at void time — not ledger shipped pounds.

# Addendum — Restore-of-ship allocation semantics

| Field | Value |
|---|---|
| **Parent** | `docs/designs/044-so-allocations-design.md` |
| **Author** | Auditor / design writer (read-only pass) |
| **Date** | 2026-08-20 |
| **Status** | Normative. Supersedes parent Q2 restore procedure, Q4 restore row, and Key Decision 10's use of `last_ship_transaction_id` as a restore selector. Does **not** reopen settled decisions: no auto-allocate at SO create, 48h TTL on `auto_fifo` only, warn-only inbound cover, sibling competition, nullable `created_by` with source tags. |
| **Scope** | How a voided SO ship is restored onto `sales_order_allocations`, how that quantity is recorded, two parked pre-PR-5 holes in the same handler, backfill, and what to keep from the WIP restore rewrite. |
| **Constraint** | Plan only. No code, no migrations applied, no SQL executed. |
| **Owner decisions (2026-08-20)** | Addendum accepted as normative. Migration number **045 is assigned to this work**; write-foundation shifts to 046+. |

This addendum exists because three implementation rounds failed the same state machine. The parent design's restore procedure cannot be implemented as written once split leftovers stop inheriting `last_ship_transaction_id` (correct, kept) and once unallocated shipping remains the majority path (Key Decision 7 / 11, settled).

---

## 0. Failure history (do not re-implement)

Three rounds, three wrong restore quantities. The void half is not on trial: coalesce-onto-leftover, unique live indexes, and `source` never overwritten all hold.

### Round 0 — `ef709e5` (PR 3 as reviewed)

`_consume_allocation_row` copied `last_ship_transaction_id` onto the split leftover. `_restore_ship_allocations` selected `status='active' AND last_ship_transaction_id = txn`. After allocate 100 → ship 100 A → void A → ship 40 B → restore A, the never-shipped 60 lb leftover still carried A's marker and flipped to `shipped` on A. A's genuine 100 lb row stayed `superseded/split_on_ship`. Silent mis-attribution. No 409, no unique violation. Reproduced in `docs/reviews/ef709e5-pr3-review.md` §1.

### Round 1 — `e87219c` (selector extension)

Correct piece, **kept:** leftovers are inserted with `last_ship_transaction_id=NULL`.

Defective piece, **do not restore:** the restore selector was extended to `superseded/split_on_ship` rows that still carried an old transaction marker, and those rows were flipped to `shipped` at **full historical quantity** without consuming any live pounds. Phantom allocation accounting (200 lb shipped-looking from a 100 lb reservation). Negative on-hand reachable via `POST /records/transactions/{id}/corrections` `event_type=restore`. Three reproductions on file (S1 / S2 / S3 in the re-review). Changelog row 85 records this as superseded.

### Round 2 — WIP on `feat/044-so-allocations` (changelog row 86)

Restore demand taken from effective-posted `ledger_current_transaction_lines` joined through `sales_order_shipments`. Preflight: live unexpired coverage must be ≥ that ledger quantity, else atomic 409 `RESTORE_SPLIT_MISSING`. Consume through `_consume_allocation_row`. Superseded rows never flipped.

This is correct **only** when the line was fully allocated at ship time, because then allocation-consumed lb = ledger-shipped lb.

It is wrong in general:

| Case | Ledger ship lb | What void actually reactivated | WIP restore |
|---|---|---|---|
| Unallocated (majority; no auto-allocate at create) | 100 | **0** | 409 `RESTORE_SPLIT_MISSING` |
| Partial allocation (allocate 50, ship 100) | 100 | **50** | 409 unless live happens to be ≥ 100 |
| Fully allocated | 100 | 100 | 409 only when a later ship has already taken the live remainder (S1) — this case is the only one the WIP tests |

`tests/test_sales_order_readiness.py::test_restore_of_voided_ship_counts_in_shipped_effective` is an unallocated ship → void → restore. It correctly fails under the WIP: restore refuses because live coverage is 0 against a 100 lb ledger line. That test is a readiness invariant, not an allocation-consume test, and it must pass.

### Why the quantity cannot be recovered after the fact

1. The ledger shows shipped lb, not allocation-consumed lb. Unallocated and partially-allocated ships are first-class (KD 11).
2. `last_ship_transaction_id` inheritance is gone by design (round 1 keep). After void of a full consume and a later split, no live row carries the voided txn marker, and the historical row is `superseded/split_on_ship` — using it as attribution is the round-1 phantom-pound bug.
3. Void-coalesce **merges** quantities across origins. S3: a 40 lb shipped slice coalesces onto a 60 lb leftover → one live 100. The live row's post-void quantity is 100; the quantity this void reactivated is 40. Any restore that reads the leftover reads the wrong number.

**Normative insight:** restore must re-consume exactly what the void of that txn reactivated into live rows — 0 unallocated, partial if partial, full if full. That number is a void-time fact. It must be recorded at void time.

Parent Q2 restore steps (reverse-coalesce by `last_ship` + `void_coalesced`, full-consume flip of `active AND last_ship=txn`) are **withdrawn**. Parent Q2 **void-coalesce** is unchanged.

---

## 1. Recording mechanism

Record **per (voided ship transaction, sales-order line)** the pounds that `_void_ship_allocations` actually returned to live rows. Call this `reactivated_lb`.

Definition (normative):

    reactivated_lb(txn, line) =
        SUM(quantity_lb of SOA rows with ship_transaction_id = txn
            AND status = 'shipped'
            AND sales_order_line_id = line)
        immediately before those rows are coalesced or reactivated

That sum is exactly the `quantity_lb` values `_void_ship_allocations` already returns, grouped by line. Coalesce does not change it: the leftover's post-update quantity is **not** the recorded value.

- Unallocated line: no shipped SOA → `reactivated_lb = 0`. **Write the 0.** Absence means "never recorded," not "zero."
- S3: 40 lb slice onto 60 lb leftover → record **40**, not 100.
- S5: allocate 50, ship 100 → only 50 was SOA-shipped → record **50**, not 100.
- Multi-row consume on one line (lot pin + SKU remainder): SUM them onto one (txn, line) record.
- `quantity_lb = 0` is a legal stored value. Do not use a `> 0` check that would drop unallocated lines.

### 1.1 Option (a) — column(s) / JSON on the void correction

`ledger_corrections` is the void event. Candidates: a new jsonb column, or stuffing an array into `replacement_values` / `previous_values`.

**Rejected as the primary store.**

- `trg_ledger_corrections_append_only` forbids UPDATE/DELETE. The current handler INSERTs the correction **before** `_void_ship_allocations` runs. A column on that row cannot be filled after coalesce unless the INSERT is reordered and the quantity is computed first. That is possible, but it couples SOA lifecycle onto a Phase 1 append-only table whose `replacement_values` is the transaction-header snapshot (`status: voided`, amendable header fields). Mixing `allocation_reactivations` into `effective_record` pollutes every consumer of `ledger_current_transactions`.
- One correction per transaction; multi-line ships need an array. Queryability and CHECKs are poor.
- Write-foundation (`docs/designs/045-write-foundation-design.md`) already owns this table's attribution story. A third concern on the same append-only surface is the wrong layer.

Using `replacement_values` with **no** migration is worse: it changes the meaning of a header snapshot without a schema signal.

### 1.2 Option (b) — small side table (recommended)

A dedicated table keyed `(transaction_id, sales_order_line_id)` holding `quantity_lb numeric(14,4)` with `quantity_lb >= 0`.

Written in the **same database transaction** as the void, after `_void_ship_allocations` returns (or equivalently: walk `sales_order_shipments` for the txn and SUM matching shipped-SOA quantities, including zeros). One row per shipped SO line on that txn, always.

Suggested shape (prose, not DDL):

| Column | Role |
|---|---|
| `transaction_id` | The voided ship. PK part. FK `transactions(id)`. |
| `sales_order_line_id` | PK part. FK `sales_order_lines(id)`. |
| `quantity_lb` | `reactivated_lb`. `numeric(14,4)`, `>= 0`. **0 is stored.** |
| `correction_id` | The void `ledger_corrections.id`. Audit join; not used by restore math. |
| `created_at` | `timestamptz`, `clock_timestamp()`. |

PK `(transaction_id, sales_order_line_id)`. Secondary index on `transaction_id` if the PK does not already serve restore's point lookup.

**UPSERT on a later void of the same txn after a restore.** Double-void of a still-voided txn remains 400 and changes nothing. Void after a successful restore is a new void and must overwrite the recorded quantity with whatever *that* void reactivated (it can differ if allocations changed between restore and the second void).

Restore **does not delete** the row. A still-voided txn's record is the demand; a posted txn's record is history until the next void overwrites it.

Why this survives S3: the writer persists the shipped-slice SUM (40), never the leftover's coalesced quantity (100). Coalesce, later splits, and `last_ship` rewrites cannot change the record.

### 1.3 Option (c) — column on `sales_order_shipments` (close second, not chosen)

That table already has one row per `(sales_order_line_id, transaction_id)` and already stores ledger `quantity_lb` (`numeric(12,2)` — parent design forbids summing it for readiness). A sibling `allocation_reactivated_lb numeric(14,4) NULL` with NULL = unknown (pre-mechanism) and 0 = recorded unallocated is a workable backfill story and avoids a new relation.

**Not recommended.** Parent Q3 already has to warn readers off `sales_order_shipments.quantity_lb`. Putting the restore demand next to the forbidden-to-sum ship quantity is a query footgun (S5: ship 100 vs reactivated 50 on the same row). The table means "this line was shipped by this txn," not "this void returned this many reserved pounds." No `correction_id`. An UPDATE of ship-time linkage on void overloads a table that is not the allocation journal.

### 1.4 Recommendation

**Option (b), the side table.** It is the only option whose key, units, and lifecycle match the fact being stored, that can hold explicit 0 without overloading ledger snapshots, and that can be written *after* coalesce in the same transaction without fighting append-only triggers.

### 1.5 Migration numbering (resolved by owner decision 2026-08-20)

- `044` is `sales_order_allocations` (applied; objects in the production dump).
- **Owner decision: this work takes migration `045`. Write-foundation migrations start at `046+`.** The write-foundation design doc (`docs/designs/045-write-foundation-design.md`) keeps its filename; a note at the top of that doc records the shift.
- The migration must be idempotent (house style — re-runnable, guards on existing objects).
- Rollback of the new table is safe only while no restore path depends on it. After restore reads it, dropping the table reintroduces the WIP 409-on-unallocated bug if anyone falls back to ledger quantity — do not fall back (see §5).

No change to `soa_active_sku_uniq` / `soa_active_lot_uniq`. No `business_date`. No session GUCs. No GPT operations. Office stays 30, Floor stays 22.

---

## 2. Restore algorithm

Applies only to `event_type='restore'` of a `transactions.type='ship'` that is currently `effective_status='voided'`. Non-ship restores are unchanged (no SOA consume). Amend still does not rewrite SOA.

`last_ship_transaction_id` is **audit only**. It is never a restore selector. Split leftovers continue to be inserted with `last_ship_transaction_id=NULL` (round 1 keep). Void still stamps it on coalesced leftovers and reactivated full-consume rows as today.

### 2.1 Order of operations (same DB transaction as the correction)

The current handler inserts the restore correction first (`main.py` `_append_transaction_correction`), then calls `_restore_ship_allocations`. A 409 rolls the whole transaction back, so it is not corrupt — but the stock guard in §4.b **must** observe POSTED_LINES *before* the ship is effective again. Normative order:

1. Lock the transaction row (`FOR UPDATE`, already done).
2. Reject restore unless currently voided (existing 400).
3. **Stock preflight** (§4.b). Fail 409 `RESTORE_STOCK_MISSING` before any correction INSERT.
4. Load `reactivated_lb` per line (§2.2).
5. Lock products (`_lock_allocation_product`), persist elapsed `auto_fifo` (`_expire_auto_fifo_allocations`).
6. **Coverage preflight** (§2.3). Fail 409 `RESTORE_SPLIT_MISSING` before any correction INSERT and before any consume.
7. INSERT the restore correction (ship becomes posted-effective; on-hand drops).
8. Consume recorded quantities (§2.4).
9. **Shrink** (§4.a) on the ship's product ids.
10. Return `allocations_reshipped` plus any backfill warning (§5).

All-or-nothing. No partial consume. No mid-handler commit.

### 2.2 Load recorded quantity

For every `sales_order_shipments` row of this `transaction_id` (physical SO lines only; services never have shipments):

| Side-table row | `reactivated_lb` used by restore |
|---|---|
| Present | `quantity_lb` as stored, including 0 |
| Absent (pre-mechanism void) | **0**, and set the warning in §5 |

Do **not** substitute ledger `ABS(transaction_lines.quantity_lb)`. That is the round-2 bug.

If the side table has extra keys not in `sales_order_shipments`, ignore them. If a shipment line is missing from the side table, it is the absent/backfill case, not an implicit ledger quantity.

### 2.3 Coverage preflight

For each line with `reactivated_lb > ε`:

- Sum `quantity_lb` of that line's live, unexpired rows (`status='active' AND (expires_at IS NULL OR expires_at > clock_timestamp())`) under `FOR UPDATE`.
- If `available + ε < reactivated_lb` → 409 `RESTORE_SPLIT_MISSING` with `transaction_id`, `sales_order_line_id`, `required_lb=reactivated_lb`, `available_lb=available`. Same envelope family as today.

For each line with `reactivated_lb ≤ ε` (explicit 0 or backfill-as-0): **do not 409.** That line contributes nothing to consume. Restore of the ledger ship still proceeds.

`RESTORE_SPLIT_MISSING` means "this void reactivated reserved pounds and those pounds are no longer live." It does **not** mean "the ledger ship is larger than live allocations."

### 2.4 Consume (same split mechanics as ship, not the full ship plan)

Walk **this line's** live unexpired rows only, in the same order consume-on-ship uses for *this order's* covering rows:

1. Own lot-level rows, `created_at ASC, id ASC`.
2. Own SKU-level row (at most one).

Take `min(remaining, row.quantity_lb)` until `reactivated_lb` is exhausted. Each take goes through `_consume_allocation_row` with `transaction_id` = the restored ship. Full take flips the row to `shipped`; partial take supersedes the live row, inserts leftover `active` with `last_ship_transaction_id=NULL`, inserts shipped slice stamped with this txn.

Do **not** call `_sales_order_ship_plan` for the remainder. That helper FIFO-takes physical lots for unreserved pounds and would re-ship inventory a second time. The ledger restore already re-posts the stock move. Restore only converts reservations.

Do **not** flip any `superseded` row to `shipped`. `void_coalesced` and `split_on_ship` rows are history. Restore mints new shipped slices from current live rows, the same way a new ship would.

Expired auto_fifo rows are not live (step 5 persisted them `released/expired`). If expiry drops coverage below `reactivated_lb`, that is `RESTORE_SPLIT_MISSING`.

`source` is never overwritten.

### 2.5 What restore does not do

- Does not rewrite `quantity_shipped_lb` / line status / order header (parent: full SO unwind is out of scope).
- Does not 409 solely because the line was never allocated.
- Does not use `last_ship_transaction_id`, `release_reason='void_coalesced'`, or `split_from_id` as demand or as a row selector.
- Does not steal coverage from a different SO line.
- Does not run when `reactivated_lb = 0` except as a no-op on SOA.

---

## 3. Acceptance scenarios

Shared setup unless noted: one physical line, originally ordered ≥ the shipped total, one lot unless the case needs otherwise, `BALANCE_EPSILON = 0.0001`. Checkpoints after every step. `allocated` means the line's original exclusive reservation (live + shipped SOA; released/superseded historical rows are not in the sum).

**Invariants every scenario must satisfy at every committed checkpoint:**

1. **Conservation:** `SUM(live SOA.quantity_lb) + SUM(shipped SOA.quantity_lb) == allocated` for that line. (Unallocated lines: both sums 0.)
2. **Attribution cap:** for each ship txn T that is currently posted-effective, `SUM(SOA.quantity_lb WHERE status='shipped' AND ship_transaction_id=T) ≤ ABS(ledger qty of T on that product)`. Never greater.
3. **On-hand:** product (and each lot) posted on-hand `>= -ε`. Restore must not be a path to negative inventory.
4. **Superseded immunity:** restore never sets `status='shipped'` on a row that entered the call as `superseded`.

Row identity of leftover vs original is not part of the contract; aggregates, attribution, and recorded quantities are.

### S1 — allocate 100, ship 100 A, void A, ship 40 B, restore A

Fully allocated, later partial re-ship blocks restore. Same 409 as the WIP S1 test; the `required_lb` source is the **record (100)**, coincidentally equal to the ledger.

| Step | live | shipped SOA | on-hand | record(A) | notes |
|---|---|---|---|---|---|
| allocate 100 | 100 | 0 | 100 | — | |
| ship 100 A | 0 | 100 on A | 0 | — | |
| void A | 100 | 0 | 100 | **100** | explicit record |
| ship 40 B | 60 | 40 on B | 60 | 100 | leftover does **not** carry last_ship=A |
| restore A | **409 `RESTORE_SPLIT_MISSING`** `required_lb=100`, `available_lb=60` | (unchanged) | (unchanged) | 100 | A stays voided; no SOA attributed to A; leftover stays live |

End state (after 409, state unchanged from post-B): live 60 + shipped 40(B) = 100. Attribution A = 0 ≤ 100. On-hand 60. No leftover flipped to shipped-on-A (the `ef709e5` bug). No 100 lb superseded row flipped to shipped (the `e87219c` bug).

### S2 — S1 + void B before restore A

| Step | live | shipped SOA | on-hand | records |
|---|---|---|---|---|
| … through ship 40 B | 60 | 40 on B | 60 | A=100 |
| void B | 100 | 0 | 100 | A=100, **B=40** |
| restore A | 0 | 100 on A | 0 | A=100 consumed |

End: live 0 + shipped 100(A) = 100. Attribution A = 100 ≤ 100. On-hand 0. B remains voided. A subsequent restore B is **two** 409s in principle: `RESTORE_SPLIT_MISSING` (recorded 40, live 0) and `RESTORE_STOCK_MISSING` (on-hand 0, B's ledger 40). Either may fire first; both are correct; tests should preflight stock then coverage as §2.1 and assert at least the first failing code, with the other covered by a dedicated unit test if the handler short-circuits.

### S3 — partial ship 40 T, void T, partial ship 30 C, restore T

The coalesce recording case.

| Step | live | shipped SOA | on-hand | record(T) |
|---|---|---|---|---|
| allocate 100 | 100 | 0 | 100 | — |
| ship 40 T | 60 | 40 on T | 60 | — |
| void T | 100 | 0 | 100 | **40** (not 100) |
| ship 30 C | 70 | 30 on C | 70 | 40 |
| restore T | 30 | 40 on T + 30 on C | 30 | 40 consumed from live |

End: live 30 + shipped 70 = 100. Attribution T = 40 ≤ 40; C = 30 ≤ 30. On-hand 30. The `void_coalesced` 40 lb row stays superseded; restore creates a **new** 40 lb shipped slice on T from the live 70. If restore had recorded 100, this would 409 against live 70 — that is the wrong quantity.

### S4 — unallocated ship, void, restore (majority path)

No `POST .../allocations` at any time. Matches `test_restore_of_voided_ship_counts_in_shipped_effective`.

| Step | SOA rows | on-hand | record |
|---|---|---|---|
| seed stock 100 | none | 100 | — |
| ship 100 A | none | 0 | — |
| void A | none | 100 | **0** (row written) |
| restore A | none | 0 | 0 → **zero allocation effect**, HTTP 200 |

`allocations_reshipped = []`. No `RESTORE_SPLIT_MISSING`. No warning field (§5 warning is for **absent** records, not stored zeros). `shipped_effective_lb` returns to 100. Conservation 0+0 == 0 allocated. Attribution 0 ≤ 100. On-hand 0.

This is the case the WIP restore breaks, and the case production will hit constantly until operators opt into reservation.

### S5 — partial allocation: allocate 50, ship 100 A, void A, restore A

| Step | live | shipped SOA | on-hand | record(A) |
|---|---|---|---|---|
| allocate 50 | 50 | 0 | 100 | — |
| ship 100 A | 0 | 50 on A | 0 | — (consume-on-ship takes 50; other 50 is unreserved FIFO) |
| void A | 50 | 0 | 100 | **50** |
| restore A | 0 | 50 on A | 0 | 50 consumed |

End: live 0 + shipped 50 = 50 allocated. Attribution 50 ≤ 100 ledger. On-hand 0. HTTP 200. WIP would have demanded 100 from live 50 and 409'd a legitimate restore.

### Additional must-pass (cheap, same neighborhood)

- **Q2 uniqueness still holds:** allocate 100, ship 40, void → **one** live 100 (`void_coalesced` on the shipped slice); never a second live key. Restore then consume-from-live 40 → live 60 + shipped 40. Rewrite `test_helper_full_void_then_restore_cycle` / HTTP cycle to assert aggregates under consume-from-live, not reverse-flip of the `void_coalesced` row. The PR-1 hand-simulated unique-index test may keep its *void* half as an index contract; its restore half must not be read as the restore algorithm.
- **Atomicity:** S1's 409 leaves SOA, side table, ledger status, and on-hand identical to pre-restore (existing `before_restore` snapshot pattern).
- **Superseded immunity:** after S1's ship B, the `split_on_ship` 100 lb row is still superseded after the failed restore, and after a successful S2 restore no superseded row has `status='shipped'`.
- **Multi-line ship:** two lines, one txn; records independent; one line unallocated (0) and one allocated (full) restore together without 409 on the zero line.

---

## 4. Parked pre-PR-5 holes (same handler)

Both were named in `docs/reviews/ef709e5-pr3-review.md` §4 and are still open in the WIP. They stay in PR 3's restore path; they are not deferred to PR 5's steal flag.

### 4.a Restore-of-ship never ran `_shrink_overallocated_products`

**Fact.** `_append_transaction_correction` runs `_shrink_overallocated_products` on void of a **non-ship**. Restore of a ship re-posts a stock reduction and today only calls `_restore_ship_allocations`. Live reservations on other orders (or leftover live on this line, if restore consumed less than live — it won't, it consumes exactly recorded — or siblings) can be left with `SUM(active) > on_hand`.

**Decision:** shrink **does** run on restore of a ship.

**Where:** step 9 of §2.1, **after** a successful consume (step 8), on `DISTINCT product_id` of the restored ship's effective lines. Reuse `_shrink_overallocated_products` (lot deficit first, then product-level, `requested_ship_date DESC NULLS LAST`). `release_reason='inventory_restored'` (free text, no CHECK; do not reuse `inventory_voided` so logs distinguish void-of-receive from restore-of-ship).

**Why after consume, not before:** shrinking first could release *this* line's live coverage and then 409 `RESTORE_SPLIT_MISSING` on a restore that would have been fully covered. Consume this line's recorded pounds into `shipped` first; those pounds are no longer live and are not shrinkable. Then repair everyone else's live claims against the new on-hand.

**S4:** no live SOA → shrink is a no-op. **S1 409:** shrink does not run (transaction rolled back before INSERT). **Observe-mode steal:** with `ALLOCATIONS_ENFORCED` off, an unallocated restore may re-take pounds another order reserved; shrink then releases the uncovered remainder. That is the same observe-mode hole parent Q4 already documents for pack; restore must not make it worse by skipping the repair.

**PR 5:** steal-409 (`STOCK_ALLOCATED`) on restore-of-unallocated-pounds that are now reserved by others is **not** specified here. Flag-off behavior is: allow the restore (if stock preflight passes), then shrink. PR 5 may add a 409 in front of that; it must not remove the shrink.

### 4.b Corrections handler posts restores with no stock guard

**Fact.** `validate_lot_deduction` (400, string detail) guards new ship/pack deductions after the lot is locked. Restore of a voided ship re-posts the original negative lines via `ledger_corrections` and never calls it. If the pounds were consumed after the void, restore drives posted on-hand negative. That is reachable today through `POST /records/transactions/{id}/corrections`.

**Decision:** lot-level stock preflight, always on, flag-independent. Negative on-hand is not an observe-mode privilege.

**Where:** step 3 of §2.1, **before** the restore INSERT. The ship is still voided, so POSTED_LINES does not include it. For each effective (product, lot, `ABS(quantity_lb)`) of the original ship, lock the lot (`FOR UPDATE`, via the existing product lock order: lots by `id ASC` then active SOA) and require `lot_on_hand + ε >= qty`.

**Error:** 409 `RESTORE_STOCK_MISSING` with the allocation envelope:

```json
{
  "error_code": "RESTORE_STOCK_MISSING",
  "message": "Cannot restore ship transaction #…: posted on-hand cannot cover the restored deduction",
  "transaction_id": 0,
  "lot_id": 0,
  "lot_code": "",
  "required_lb": 0.0,
  "available_lb": 0.0
}
```

Do not reuse `validate_lot_deduction`'s 400 string. Do not use `RESTORE_SPLIT_MISSING` (that code is reservation coverage, not inventory). Product-level-only checks are insufficient: stock may sit on a different lot than the original ship lines.

If stock preflight fails, no correction is written, no SOA consume, no shrink, no side-table write.

---

## 5. Backfill / compatibility

Voids committed before the recording mechanism exists have no side-table row. Absence ≠ 0. Stored 0 means "we looked; nothing was reactivated."

Recommended behavior: treat missing as `reactivated_lb = 0` and proceed (unallocated restore works; allocated historical restore does not re-pin). Surface a warning, do not 409.

Response field (additive, on the restore payload, alongside `allocations_reshipped`):

```json
{
  "allocation_reactivation_unknown": true,
  "allocation_reactivation_unknown_line_ids": [123]
}
```

Omit both fields when every shipment line had a stored row (including stored zeros). Dashboard / GPT can ignore unknown fields.

Why 0, not ledger qty: substituting ledger qty is the WIP restore. It 409s S4 and S5 for every historical void, including the readiness test and the majority unallocated fleet. Factory Ledger ships without allocating today.

Why a warning, not silence: a pre-mechanism void of a fully allocated ship will restore inventory without converting the live 100 back to shipped. Conservation still holds (live 100 + shipped 0 = 100) but SUM(live) may exceed post-restore on-hand. §4.a shrink then releases the uncovered live pounds (`inventory_restored`). Operators who still want the reservation re-allocate. The warning is how we refuse to guess.

Alternatives (not recommended):

| Alternative | Behavior | Reject because |
|---|---|---|
| Treat missing as ledger qty | WIP S4/S5 409 | Majority path, readiness test |
| 409 `RESTORE_REACTIVATION_UNKNOWN` | Safest, blocks all pre-mechanism restores | Blocks unallocated restores that must work; there is no production allocation fleet yet, but the code path is already tested and is the default ship |
| `min(ledger_qty, live_qty)` | S4 → 0 (luck), S5 → 50 (luck), S1 → consume 60 as A | Silent mis-attribution; same family as ef709e5 |
| Refuse only if live > 0 and record missing | Heuristic | Still guesses; hard to explain |

No backfill job. Do not invent `reactivated_lb` from ledger or from `last_ship`. Optional operator-approved backfill is out of scope.

Going forward, every ship void writes the side-table row, including zeros, so the warning path is only for voids that landed before this ships.

---

## 6. WIP commit: keep vs discard

The WIP is changelog row 86 / CHANGE_LOG.md 2026-08-20 round 2. It is local-only, not deployed.

### Keep

| Piece | Why |
|---|---|
| Leftover `last_ship_transaction_id=NULL` on split (`_consume_allocation_row`) | Round 1 correct fix; parent restore-by-marker is withdrawn, but inheritance remains a lie |
| `POST /sales/orders/{id}/allocations/{allocation_id}/release` and dashboard DELETE-guard restoration | Round 1, unrelated and correct |
| Named guard tests (`LOT_PRODUCT_MISMATCH`, `MANUAL_ALLOCATION_CANNOT_EXPIRE`, `ALLOCATION_NOT_ACTIVE`, service/cancelled allocate, sibling/line coverable, cross-order pin) | Round 1 coverage the review asked for |
| `_void_ship_allocations` coalesce (leftover `quantity_lb +=` / else flip shipped → active; never two live keys) | Parent Q2 void; S3's 40-not-100 depends on it |
| Never flipping superseded rows in restore | Round 2 correct constraint |
| Consume through `_consume_allocation_row` (split leftover + shipped slice) rather than UPDATE historical rows to shipped | Round 2 shape; this addendum keeps it and changes only the quantity source |
| Product lock + persist expired auto_fifo before restore consume | Round 2 |
| Atomic 409 `RESTORE_SPLIT_MISSING` with `required_lb` / `available_lb`, no partial consume | Round 2 shape; trigger becomes recorded>0 ∧ live < recorded |
| Conservation checkpoints (`_allocation_checkpoint`: live+shipped == allocated, on-hand ≥ 0) | Keep; S5 must pass `originally_allocated_lb=50` |
| S1 / S2 / S3 narrative and HTTP wiring | Keep S1 as 409 against recorded 100 vs live 60; keep S2 consume 100 after void B; keep S3 consume 40 not ledger-confused leftover 100 |
| Unique live indexes, consume-on-ship always on, `ALLOCATIONS_ENFORCED` still off | Settled |

### Discard

| Piece | Why |
|---|---|
| Restore demand from `ledger_current_transaction_lines` / `sales_order_shipments` | Core WIP bug; unallocated and partial-allocation 409 |
| Preflight "every shipment line of this txn must have live coverage ≥ ledger qty" | Same bug; S4/S5 |
| Changelog row 86 "Permanent rule: restore demand comes from effective ledger lines" | Replace with: restore demand comes from the void-time reactivation record; ledger qty is an upper bound on attribution (invariant 2), not the consume target |
| Any 409 `RESTORE_SPLIT_MISSING` when `reactivated_lb = 0` | S4 must 200 |
| Reverse-coalesce restore (flip `void_coalesced` back to shipped, shrink leftover by S.quantity_lb) as the implementation | Parent Q2 restore; withdrawn. Void-coalesce itself stays |
| Selector on `last_ship_transaction_id` / `release_reason IN ('void_coalesced','split_on_ship')` | ef709e5 and e87219c |
| Reading leftover `quantity_lb` after coalesce as the restore target | S3 records 40, leftover is 100 |

### Rewrite / add

- `_restore_ship_allocations`: drop the ledger SUM; load the side table; skip zeros; consume recorded; never touch superseded.
- `_void_ship_allocations` (or its caller): after the existing loop, UPSERT one side-table row per `sales_order_shipments` line, `quantity_lb` = SUM(returned quantity for that line) or 0.
- `_append_transaction_correction` restore-of-ship: stock preflight → coverage preflight → INSERT → consume → shrink. Void-of-ship: INSERT void (or void-SOA then INSERT; either is fine) + write records. Do not UPDATE `ledger_corrections`.
- Tests: keep S1–S3 with recorded-qty assertions; add S4 (HTTP 200, zero SOA, readiness `shipped_effective` restored — this is the currently-failing readiness test, which must go green); add S5; add "missing record → 0 + warning"; add `RESTORE_STOCK_MISSING` (restore after a competing ship emptied the lot); add shrink-on-restore (second order live pin uncovered by restoring an unallocated ship).
- `test_full_void_then_restore_cycle_holds_under_unique_index`: keep as unique-index documentation of void coalesce; do not treat its hand-written restore UPDATE as the restore spec.

---

## Key Decisions (addendum)

1. Restore consume target is void-time `reactivated_lb`, not ledger shipped lb and not leftover quantity. Ledger qty remains an attribution ceiling.
2. Record that quantity in a dedicated `(transaction_id, sales_order_line_id, quantity_lb)` table, including explicit zeros. Not on `ledger_corrections` (append-only, wrong layer). Not on `sales_order_shipments` (wrong units beside a column we already forbid summing).
3. `RESTORE_SPLIT_MISSING` only when recorded > 0 and live cannot cover. Recorded 0 restores with zero SOA effect.
4. Superseded rows are never flipped by restore. New shipped slices come from `_consume_allocation_row` on current live rows.
5. `last_ship_transaction_id` is not a restore selector. Round-1 NULL on leftovers stays. Void may still stamp it for audit.
6. Parent Q2 restore procedure is withdrawn; parent Q2 void-coalesce is not.
7. Restore-of-ship runs `_shrink_overallocated_products` after consume, reason `inventory_restored`.
8. Restore-of-ship is 409 `RESTORE_STOCK_MISSING` at lot level before the correction INSERT. Always; not behind `ALLOCATIONS_ENFORCED`.
9. Pre-mechanism voids: missing record → treat as 0 + `allocation_reactivation_unknown`. No silent ledger fallback.
10. Migration number: **045, assigned by owner decision 2026-08-20** (this supersedes the flag-only status of the auditor draft). Write-foundation starts at 046+.

Settled, not reopened: no auto-allocate at create; 48h TTL auto_fifo only; inbound cover is warn-only; sibling lines compete; created_by nullable source tags; consume-on-ship always on; unique live indexes stay; no new GPT ops.

---

## What this addendum changes in the parent

| Parent location | Change |
|---|---|
| Q2 "Restore of txn" steps 1–2 | Replaced by this document §2 |
| Q4 table row "Restore of that voided ship" | Replaced: consume recorded qty from live rows; 409 only if recorded > 0 and uncovered; 0 is a no-op |
| Key Decision 10 "last_ship_transaction_id remembers the txn" (as restore machinery) | Weakened to audit stamp only |
| Test plan "Restore of a full-consume void: allocation returns to shipped" | Still true when recorded = live = ledger (fully allocated, no intervening ship). Unallocated restore must also succeed with zero SOA |
| Sequence diagram "reverse coalesce → leftover active + shipped slice" | Restore is consume-from-live of recorded qty; leftover identity is not guaranteed to be the original shipped slice flipped back |
| PR 3 description | Still the landing PR; this is fix round 3 of that PR, plus migration 045 |

---

## PR Plan

Still on branch `feat/044-so-allocations`. Not a new numbered PR in the parent six. Does not merge to main without per-action approval.

**PR 3 fix round 3 — record-at-void + consume-recorded-on-restore**

- Title: `fix(allocations): record void reactivation qty; restore consumes that, not ledger`
- Files: migration 045 (idempotent); `main.py` (`_void_ship_allocations` writer or its caller, `_restore_ship_allocations`, `_append_transaction_correction` restore/void-of-ship order, error helpers); `tests/test_sales_order_allocations.py` (S1–S5, stock miss, shrink-on-restore, missing-record warning); `tests/test_sales_order_readiness.py::test_restore_of_voided_ship_counts_in_shipped_effective` must pass **unmodified**; `FACTORY_LEDGER_CHANGELOG.md` (new row superseding 86's "ledger is restore demand" rule); no GPT schema changes
- Depends on: PR 1 (table + unique indexes), PR 2 (readiness), PR 3 writes as of round 1 keep-list
- Does not include: PR 4 takeable on /ship+/pack; PR 5 `STOCK_ALLOCATED`; write-foundation (046+)
- Acceptance: S1–S5 end-states in §3; readiness unallocated restore 200; no superseded flip; conservation + attribution cap + on-hand ≥ 0 at every checkpoint; migration 045 is this work's

---

End of addendum. Parent document otherwise unchanged.

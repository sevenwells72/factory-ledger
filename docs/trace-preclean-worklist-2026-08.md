# Trace pre-clean worklist — §9 step 0 dry run (2026-08-31)

**Scope:** `docs/designs/TRACEABILITY_DESIGN.md` §9 step 0 — pre-clean dry run before migration 047.
**Method:** production Postgres, READ-ONLY only — `scripts/psql_ro.sh` (session-mode pooler, `:5432`, per the hard rule; never session GUCs on 6543) with every query inside `BEGIN TRANSACTION READ ONLY; … COMMIT;`. No writes of any kind were made. Data as of 2026-08-31 ~13:35 ET. `lots` table: **1,048 rows, 0 merged** (no merge has ever run in prod).

---

## 1. Normalized-code twins (dry run of `lots_product_code_norm_uniq`)

Dry-run query (violations of the proposed unique index):

```sql
WITH norm AS (
  SELECT id, product_id, lot_code, status,
         upper(regexp_replace(btrim(lot_code), '\s+', ' ', 'g')) AS norm_code
  FROM lots)
SELECT product_id, norm_code, count(*), array_agg(id ORDER BY id)
FROM norm GROUP BY 1, 2 HAVING count(*) > 1;
```

### Result: **0 violations.** The index can be created today with no prior merges.

The design's "expected ≈3" is **stale/incorrect**:

- The cited `APR 10 2026 Lot` twins (lots 391, 392, 394 — plus 513, 515) all belong to **different products** (128, 125, 114, 161, 154). The proposed index is per-product, so they don't violate it. The baseline §3i table they come from lists *cross-product* code collisions, which the design explicitly tolerates.
- Under the index's exact normalization (case-fold + whitespace-collapse + trim) there is no within-product pair anywhere in the table.

### Near-twins the proposed normalization does NOT catch (found under a stronger probe: strip all non-alphanumerics + trailing `LOT` token)

Two within-product groups are real physical twins but survive the design's expression. Neither blocks migration 047; both are recommended hygiene merges.

**Group A — product 107 `Batch Classic Granola #9`: lot 999 `JUL15 2026` vs lot 1003 `JUL 15 2026`** (missing internal space — normalization can't collapse it)

| lot | exact code | created (UTC) | entry_source | on-hand (`lot_on_hand`) | posted txns | activity | ILC rows as input | open SO allocs |
|---|---|---|---|---|---|---|---|---|
| 999 | `JUL15 2026` | 2026-07-15 20:54 | production_output | **0.0** | 4 | make 1657 +7,106; packs 1660 −1,400, 1671 −5,700, 1964 −6 | 2 | 0 (0 any status) |
| 1003 | `JUL 15 2026` | 2026-07-15 23:46 | production_output | **0.0** | 2 | make 1662 +1,938; pack 1964 −1,938 | 0 | 0 (0 any status) |

These are two same-day makes of the same batch product that got separate lot rows only because of the typo'd code. **Recommend: merge 1003 → 999 (999 survives).** Why: 999 has the most activity (4 txns vs 2), is the older row, and has downstream `ingredient_lot_consumption` references (2 rows) — surviving it moves the fewest rows (2 line corrections, 0 ILC repoints; the reverse direction would move 4+2). On-hand is zero on both sides so no balance question. Cosmetic caveat: the survivor keeps the typo'd code `JUL15 2026`; a follow-up `PATCH /lots/999/rename` to `JUL 15 2026` is **blocked** by the existing `UNIQUE (product_id, lot_code)` because merged lot 1003 retains that exact code — optional, only if 1003's code is renamed aside first. **Safety: obviously safe** (both empty, no open allocations, all touching txns posted).

**Group B — product 145 `Granola SS Chocolate Chip 12x10 OZ Case`: lot 401 `BB041327 Lot` vs lot 410 `BB041327`** (trailing ` Lot` noise word)

| lot | exact code | created (UTC) | entry_source | on-hand | posted txns | activity | ILC rows as input | open SO allocs |
|---|---|---|---|---|---|---|---|---|
| 401 | `BB041327 Lot` | 2026-04-14 20:18 | pack_output | **0.0** | 2 | pack 616 +2,250; ship 1330 −2,250 | 0 | 0 (0 any status) |
| 410 | `BB041327` | 2026-04-15 13:05 | pack_output | **0.0** | 4 | packs 625 +3,375, 638 +1,125, 644 +2,250; ship 1329 −6,750 | 0 | 0 (0 any status) |

Same best-by-coded FG lot family, split by the ` Lot` suffix on day one. **Recommend: merge 401 → 410 (410 survives).** Why: 410 has the most activity (4 txns vs 2) and the clean code; both are fully shipped out (zero on-hand), so the "oldest" tiebreak (401) is outweighed. Moves 2 line corrections, 0 ILC. **Safety: obviously safe** (both empty, no open allocations, both ships posted 2026-06-11).

**Flagged not-obviously-safe: none.** No group has on-hand on both lots, allocations on either lot, or voided-transaction complications.

### Design feedback for §3.1 (from this dry run)

1. **The index DDL needs a predicate for merges to ever fix a violation.** `/admin/lots/merge` marks the source `status='merged'` but leaves its `lot_code` unchanged, so a merged twin pair would still collide under the unqualified index. As written this doesn't matter today (0 violations, 0 merged rows), but the index should be `WHERE status IS DISTINCT FROM 'merged'` (or merge must rename the source code) or the first real twin-merge will make the index impossible to build and future merges impossible to complete.
2. If the intent was to also catch `JUL15 2026`-style and ` Lot`-suffix twins, the normalization expression needs strengthening; as specified it only folds case and whitespace. Alternatively accept the residue — both live cases are handled by the merges above.

---

## 2. ILC-less packs (pack transactions with zero `ingredient_lot_consumption` rows)

The design's "2 known ILC-less packs" is the **90-day-windowed** audit figure (`traceability_audit.sql` §8a limits to `business_date >= CURRENT_DATE - 89`). Full history has **8 posted + 1 voided**:

| txn | business_date | effective_status | shape | lines (lot / qty lb) | FG lot on-hand now |
|---|---|---|---|---|---|
| 77 | 2026-02-05 | posted | batch debit only, **no FG output line** | lot 71 `FEB-03-2026` (Batch Coconut Sw. Flake) −1,030 | n/a |
| 78 | 2026-02-05 | posted | batch debit only, no output | lot 71 `FEB-03-2026` −2,030 | n/a |
| 94 | 2026-02-05 | posted | batch debit only, no output | lot 57 `JAN 30 2026` (Batch Classic #9) −1,292 | n/a |
| 95 | 2026-02-05 | posted | batch debit only, no output | lot 84 `FEB-02-26` (Batch Classic #9) −208 | n/a |
| 97 | 2026-02-05 | posted | batch debit only, no output | lot 68 `FEB-03-2026` (Batch Classic #9) −3,000 | n/a |
| 98 | 2026-02-05 | posted | batch debit only, no output | lot 81 `FEB-04-26` (Batch Classic #9) −3,000 | n/a |
| 470 | 2026-03-18 | **voided** | reversal of txn 469 (voided in the 2026-06-10 void-semantics cleanup) | lot 314 −10 / lot 293 +10 | lot 293: 0.0 |
| 1964 | 2026-08-14 | posted | INV-RECON-2026-08-17 YTD bulk pack (`entry_backfilled`, operator `inv-recon-2026-08-17`): 14 batch-lot debits in lines, **no ILC mirror** | 14 Batch Classic #9 lots (912…1063) totaling −12,115; output lot 1207 `BULK-#9-YTD` +12,115 | lot 1207: 0.0 |
| 1966 | 2026-08-14 | posted | same recon shape | 4 Batch SS Original #1 lots −1,875; output lot 1223 `BULK-#1-YTD` +1,875 | lot 1223: 0.0 |

The audit's "2" = **1964 and 1966** (only ones inside the 90-day window; 470 is voided-effective and excluded).

**Annotation / grandfathering marker (per §8 O2 + §9 step 4):** the step-4 backfill synthesizes these packs' trace rows with **`created_at_source = 'trace_backfill_048'`** (using the `api_backfill`-style trigger carve-out from 046). That marker is the grandfather: the O2 check ("transformation event with an `output` row and zero `input` rows") exempts backfill-marked events and **fails on any occurrence without the marker** — so these legacy rows report informationally while any post-deploy ILC-less pack is a hard failure.

**Backfill-spec gap to resolve before step 4:** the population has two shapes, and O2 as written only sees one of them.

- **1964/1966** — batch debits exist in `transaction_lines` (just no ILC mirror). If the backfill synthesizes pack `input` rows *only* from ILC (§4's table), these two become the O2 seed population; better: synthesize their inputs from the negative lines (quantities are right there), mark them backfilled, and O2/R2/M2 all go green with real genealogy preserved.
- **77/78/94/95/97/98** — single negative batch line, **no output line at all** (early-Feb entry practice; the packed FG was never credited). Backfilled events will have an input-side (if taken from lines) but **no `output` row** — O2 ("output but zero inputs") never fires on them, but **R2** (quantity mirror) and **M2/M3** (batch/FG mass balance) will. They need the same grandfather marker treatment plus an explicit note in the §8 gate ("the 2 legacy packs grandfathered" → should read "the 8 legacy packs, two shapes").

Step-0 action for the owner: no DB write needed now — the "annotation" is applied by the backfill itself; this worklist is the approval record enumerating exactly which transactions get grandfathered.

---

## 3. New violations since the 8/24 baseline

**None — no live bug indicated.**

- Twin check: zero within-product normalized-code groups exist at all, and zero twin-group lots were created on/after 2026-08-24.
- ILC check: 27 pack transactions created on/after 2026-08-24 (of 536 total) — **all 27 have ILC rows.** The newest ILC-less pack is txn 1966 (created 2026-08-17, the inventory recon), before the baseline.

---

## Bottom line

- **Merges needing approval to unblock migration 047: 0.** The proposed unique index has zero violations today.
- **Recommended optional hygiene merges: 2** (approve individually per house rules):
  1. lot **1003 → 999** (`JUL 15 2026` into `JUL15 2026`, Batch Classic Granola #9) — obviously safe.
  2. lot **401 → 410** (`BB041327 Lot` into `BB041327`, Granola SS Chocolate Chip 12x10 OZ Case) — obviously safe.
- **Flagged not-obviously-safe: none** (every twin lot has zero on-hand, zero open SO allocations, and only posted-effective history).
- Carry-forward items: §3.1 index needs a `status IS DISTINCT FROM 'merged'` predicate; §8/§9 must widen "2 ILC-less packs" to the 8-posted/two-shapes population above. *(Both applied to `docs/designs/TRACEABILITY_DESIGN.md` in commit `39ed8c1`, 2026-08-31.)*

---

## 4. Merge receipts (2026-08-31, owner-approved, executed one at a time)

Both hygiene merges were run via `POST /admin/lots/merge` against production (master key), each followed by read-only verification (`scripts/psql_ro.sh`, session pooler :5432, `BEGIN TRANSACTION READ ONLY`). Design-doc fixes from §1's feedback were committed first as `39ed8c1` on `fix/audit-insert-savepoints`.

### Receipt 1 — lot 1003 → 999 (product 107, Batch Classic Granola #9)

- **API response:** `merged: true`; `rows_moved: {transaction_lines: 2, ingredient_lot_consumption: 0}`; `line_correction_ids: [0f144ec0-907f-4bb4-bdbf-e0ed6883bd15, 8b994421-99af-40a9-b333-fc626f8ad218]`; `allocation_moves: []`; `target_lot_new_balance: 0.0`.
- **Merged at:** `2026-08-31 17:49:06.81103+00` (`lots.merged_at`).
- **Verification (read-only, post-merge):**
  - Lot 1003: `status='merged'`, `merged_into_lot_id=999`, `merge_reason` recorded; **0** effective lines remain on 1003.
  - Survivor 999 effective history intact + absorbed: original make 1657 +7,106 / packs 1660 −1,400, 1671 −5,700, 1964 −6 all present, plus moved make 1662 +1,938 and pack 1964 −1,938 (6 lines, all posted-effective).
  - `lot_on_hand()` replica: 999 = **0.0000**, 1003 = **0** (no posted lines) — unchanged, both zero.
  - ILC: 999 keeps its **2** rows as ingredient input; 1003 has 0 (had 0).

### Receipt 2 — lot 401 → 410 (product 145, Granola SS Chocolate Chip 12x10 OZ Case)

- **API response:** `merged: true`; `rows_moved: {transaction_lines: 2, ingredient_lot_consumption: 0}`; `line_correction_ids: [822671b9-65d6-4269-987c-ca387efe28ca, b8f77428-73f7-4d41-8ed7-476fea51aa5d]`; `allocation_moves: []`; `target_lot_new_balance: 0.0`.
- **Merged at:** `2026-08-31 17:49:25.833942+00` (`lots.merged_at`).
- **Verification (read-only, post-merge):**
  - Lot 401: `status='merged'`, `merged_into_lot_id=410`, `merge_reason` recorded; **0** effective lines remain on 401.
  - Survivor 410 effective history intact + absorbed: original packs 625 +3,375, 638 +1,125, 644 +2,250 / ship 1329 −6,750 all present, plus moved pack 616 +2,250 and ship 1330 −2,250 (6 lines, all posted-effective).
  - `lot_on_hand()` replica: 410 = **0.0000**, 401 = **0** (no posted lines) — unchanged, both zero.
  - ILC: 0 rows either lot (unchanged).

### Post-merge strengthened dry run

Re-ran the strengthened normalization probe (upper + strip trailing `LOT` token + strip all non-alphanumerics) with the index's `WHERE status IS DISTINCT FROM 'merged'` predicate, all products:

**0 remaining within-product twin groups.** The §3.1 index (strengthened expression + merged-exclusion predicate) can be created in migration 047 with no further pre-cleaning.

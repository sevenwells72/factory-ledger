# Recipe corrections — 179 Fruit Nut and 284 chips (migration 078, PR #104 `fix/recipe-corrections`)

Built in `~/dev/fl-recipes-wt` as BUILDER for Codex review; base `origin/main` **7fec510** (PR #105 post-A12 housekeeping; originally built on c195a39, rebased in review fix round 1). **Staging only** — no production writes, no deploy, no merge. Approved by Michael on 2026-10-13 from the read-only recipe audit `~/Documents/fl-audits/recipe-check-2026-10-13/recipe-check.md` (rows 179 and 284) and `catalog-cleanup.csv` row 29 (284 chip size, APPROVED 2026-10-07).

## What changes (data only)

| | Before (production, read-only 2026-10-13 17:43 UTC, re-read unchanged 17:57 UTC after 075) | After |
|---|---|---|
| **179 Granola Fruit Nut Batch (90008) — make-recipe** `batch_formulas` ids 179–183 | 20 lb Classic #9 (107) + 1.25 lb each sliced almonds (3), walnuts (49), raisins (36), dried cranberries (17) = **25 lb** | **276.3 lb** Classic #9 + **13 lb** each of the four = **328.3 lb** |
| **179 batch size** `products.default_batch_lb` | **384.52** | **328.3** |
| **179 planner rows** `product_bom` ids 117–121 (finished_product_id 179) | 323 lb Classic #9 + 15.38 lb × 4 = 384.52 | rewritten in place to 276.3 + 13 × 4 = 328.3 (same ids, uom `lb`) |
| **179 legacy case yields** `products.yield_25lb_cases` / `yield_10lb_cases` (added in review fix round 1, Michael approved) | **15 / 38** | **13 / 32** (328.3 / 25 = 13.1; 328.3 / 10 = 32.8) |
| **284 SS Classic Chocolate Chip #9 kosher (90026) — chips line** `batch_formulas` id 198 | ingredient **73** Chocolate Chips – Real – 1,000 CT, 25 lb | ingredient **72** Chocolate Chips – Real – 4,000 CT, 25 lb (other 7 lines untouched; total 347.6 lb, batch 348 unchanged). The line must stay **deducted** (`exclude_from_inventory` false/NULL) — the guard and post-check refuse an excluded chips line, which would post 348 lb with zero chips taken. Using 1,000 CT on a given make is a per-make A5 substitution (`transaction_substitutions`, reason-coded), never a formula exclusion |

Unchanged on purpose: product **142 Granola Fruit Nut 25 LB** (70061; `case_size_lb` 25, `parent_batch_product_id` 179) and its planner row `product_bom` 116 (142 → 179 @ 25 lb); every transaction, transaction line, lot, lot balance and `ingredient_lot_consumption` row (the migration never reads them). 075 (merged while this was being built) renamed 284 to "Batch Classic Chocolate Chip Granola #9 — EXTRA-KOSHER (SS)" and added `kosher_tier`; 078 identifies rows by id + `odoo_code`, so the rename is irrelevant to it.

Why the make-recipe matters: `/make` deducts `batch_formulas` × batches and posts `default_batch_lb` × batches × `yield_multiplier` (1.0). With the old rows each 2-batch make booked 769.04 lb of Fruit Nut for 50 lb of inputs. The planner / full-BOM loader (`_load_bom_structure`) and the plan endpoint read `product_bom`, which carried a second, different recipe; after 078 all three agree at 328.3 lb.

## Files

- `migrations/078_recipe_corrections.sql` — guard (every row must be in its recorded BEFORE or AFTER state, catalog identity by id + odoo_code, 284 chips line not excluded; otherwise `078 refused: …`), five UPDATE groups, post-checks (recipe = batch size = planner sum = 328.3; yields 13/32; 284 has one deducted 72 line and no 73 line), marker `078_recipe_corrections`. Rerunnable.
- `migrations/down/078_recipe_corrections_down.sql` — strict: refuses (`078 down refused: …`) unless the marker is present and every row 078 touches (formula ids 179–183, planner ids 117–121, 179 batch size + yields, 284 line 198) holds exactly its 078 value, so it never overwrites a later edit; then restores the recorded BEFORE values (yields 15/38) and deletes the marker. A second run refuses. It reads and writes only `products`, `batch_formulas`, `product_bom`, `migration_markers` — no transaction/lot/ILC access (the old NOTICE count of posted makes was removed in fix round 1).
- `tests/test_recipe_corrections_078.py` — 12 cases (fix round 1 added: excluded 284 chips refused before and after, yields 13/32 + refusals, twelve down tampers each refused with nothing changed, static check that neither file references a history table; history snapshots now compare every column of transactions, transaction_lines, lots, ingredient_lot_consumption, transaction_substitutions and lot_events): exact after-state incl. surviving ids, rerun, down → before → up, six refusals (fresh schema, off-value line, sixth line, missing planner rows, two chip sizes, chips ≠ 25 lb) with nothing changed, posted history + derived balances untouched by up and down, `/make` 179 (preview 2 batches = 656.6, commit 1 batch posts 328.3 and deducts 276.3 + 4 × 13 with ILC parity), `/pack` 142 (13 × 25 lb fits one batch, 14 do not, straight repack, planner loader consistent), `/make` 284 (before 078 the preview demands 73 and is short; after, 25 lb of 72 is deducted and 73 is untouched).
- `scripts/check_recipe_corrections_staging.py` — staging acceptance (A6/A7/A9 pattern: guarded staging URI, local TestClient on the staging DB, no deploy). `--seed` / `--restock --round N` / `--apply-migration` / `--make [--round N]` (a round whose Fruit Nut make already posted is re-verified, not made again; a 284 commit refused by A12 is recorded, not raised).
- `docs/deployments/recipe-corrections-staging-receipt.json` — sanitized receipt of the round-1 run below; `recipe-corrections-staging-receipt-r3.json` — fix round 1.

Numbering: 073 (#96 PIN), 075 (A12, merged + on prod 17:44:52 UTC), 076 (A9, live), 077 (#103 key sessions, staging) are taken; **078 is free** on every remote branch.

## Apply recipe

Table owner (`postgres`) on port 5432, one transaction, STAGING FIRST:

```
BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public;
\i migrations/078_recipe_corrections.sql
COMMIT;
```

No code change rides with it (the make/pack/planner code already reads these tables), so there is no deploy order; apply at a moment with no Fruit Nut or 284 make in flight. Production only with Michael's explicit approval, after the Codex review. Verify afterwards with `data/q1.sql`–`q5.sql` from the audit folder through the read-only wrapper. Rollback: `migrations/down/078_recipe_corrections_down.sql`, same recipe.

## Staging acceptance — 2026-10-13

Staging had **no** `batch_formulas` rows for 179/284, no `product_bom` rows at all and 0 lb of every ingredient involved, so the script seeded the recorded production BEFORE state (13 formula rows, 6 planner rows) plus clearly labelled test stock — lot codes `STG-078-*`, transaction 1000000073 `adjust` "Found inventory: STAGING FIXTURE 078 — recipe-correction test stock, not real inventory", operator `staging-fixture-078`, marker `staging_recipe_fixture_078_v1` (17:53:33 UTC). Deliberately **no stock of 73** (1,000 CT), so a 284 make can only succeed if the chips line moved to 72.

- **078 applied to staging 17:53:50 UTC** (`postgres`, port 5432, single transaction); after-state read back: 179 formula 276.3 + 13 × 4, `default_batch_lb` 328.3, planner rows 276.3 + 13 × 4 (ids kept), 284 chips = 72 @ 25 lb, marker present.
- **Make 179, 1 batch → transaction 1000000074, lot `STG-078-FRUIT-NUT-001`: +328.3 lb Fruit Nut; −276.3 lb Classic #9 (lot STG-078-CLASSIC9), −13 lb each almonds, walnuts, raisins, cranberries.** On-hand before → after: Fruit Nut 0 → 328.3; Classic #9 400 → 123.7; each inclusion 30 → 17. ILC mirrors the five lines (sum 328.3). Preview for the same make showed batch size 328.3, needs 276.3 / 13 × 4, all available.
- **Pack consistency for 142 (25-lb case), preview only:** 13 cases = 325 lb → sufficient from the 328.3 lb lot, no add-ins (straight repack); 14 cases = 350 lb → insufficient (328.3 available). Nothing packed.
- **Make 284, 1 batch → transaction 1000000075, lot `STG-078-KOSHER-CC-001`: +348 lb; −25 lb Chocolate Chips – Real – 4,000 CT (72, lot STG-078-CHIPS-4000CT), 1,000 CT (73) untouched at 0 lb;** the other seven lines deducted at formula (oats 225, brown sugar 50, canola 30, coconut 12.5, almond flavor 1.8, salt 1.7, honey 1.6; ILC 8 rows, 347.6 lb). The Kosher Ignition `production_warning` was returned as on production.
- Independent read-only check afterwards: markers present, the three transactions and 15 lines as above, no rows with ids < 1e9 touched. The two staging makes stay as labelled staging history (same practice as the A6/A7/A9 receipts).

### Fix round 1 re-run — 2026-10-13 18:35 UTC

- `--restock --round 2 --apply-migration`: labelled top-up (txn 1000000077, lots `STG-078-R2-*`, marker `staging_recipe_fixture_078_v2`), revised 078 re-applied — 179 yields 15/38 → 13/32, every other row already at its target (no-op), chips line 72 @ 25 lb deducted; the 078 marker keeps its 17:53:50 timestamp.
- Round 2 make: **transaction 1000000078, lot `STG-078-FRUIT-NUT-002`, +328.3; −276.3 Classic #9 (123.7 + 152.6 from two lots), −13 × 4.** The run was then killed by a closed output pipe before the 284 step and before writing a receipt (operator error: output piped to `head`).
- Round 3 (`--restock --round 3`, txn 1000000079, lots `STG-078-R3-*`): **transaction 1000000080, lot `STG-078-FRUIT-NUT-003`, +328.3; −276.3 Classic #9, −13 × 4.** 142 pack preview: 13 cases = 325 lb sufficient; 14 cases = 350 lb now also sufficient because three Fruit Nut test lots exist (984.9 lb) — preview only.
- **284 preview** after the revised 078: 348 lb out, needs 25 lb of 72 (sufficient), nothing of 73. **284 commit refused** by the A12 gate (403 `KOSHER_ATTESTATION_REQUIRED`): 284 is `extra_kosher` on staging (075), so a make needs a ticket approved from Michael's own signed-in session. Not simulated; the round-1 commit (1000000075) and the fresh-DB `/make` 284 test cover the deduction. Receipt `recipe-corrections-staging-receipt-r3.json` (the resumed round-3 run).

## Verification (builder, before review)

Fresh database `factory_ledger_test_recipes2` dropped and rebuilt from `tests/schema/schema.sql` (markers 067/068/072/074/075/076) immediately before the run, on base c195a39: **2,699 Python passed, 0 failed, 0 skipped** (48.5 s); **100 Node passed, 0 failed** (`node --test tests/test_*.js`). The eight 078 cases are included. GPT yaml untouched (30 ops).

Fix round 1 (rebased on 7fec510): `factory_ledger_test_recipes2` dropped and rebuilt from the 7fec510 dump — **2,703 Python passed, 0 failed, 0 skipped; 100 Node passed**. The four new regression tests fail against the pre-fix 078 files and pass after.


## Open items for Michael (not in the approved list; not changed)

1. **Stock true-ups** from the audit remain to be approved separately: lot 1560 "OCT 06 2026" still carries 219.04 lb of ghost Fruit Nut; Classic #9 is overstated by roughly 1,538 lb and each inclusion by about 70 lb from the six under-deducted makes; 4,000 CT chips are probably about 950 lb high (five 108 makes consumed 1,000 CT that only existed via same-day adjustments). 078 corrects the recipes only.
2. ~~**`products.yield_25lb_cases` / `yield_10lb_cases` on 179**~~ — done in 078 fix round 1 (13 / 32, Michael approved).
3. **Product 115 (SS Chocolate Chip #5, pending deactivation)** still points at 73 — moot once hidden; left alone.
4. After the production apply, re-run the audit's `q1`–`q5` and refresh `recipe-check.md`; the oat bran, flax and coconut questions still wait on Arturo's sheets.

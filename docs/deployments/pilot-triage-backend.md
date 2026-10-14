# October 14 pilot backend triage — staging only

Builder: Codex. Reviewer: Claude Code. Base: origin/main `611d0c4`.
No merge or production write is authorized. Production was queried only inside explicit
read-only transactions on port 5432, ending in rollback. No historical rows are changed.

## Changes

- Make prepare and both legacy preview/commit paths refuse an empty, zero-only or entirely excluded recipe with `RECIPE_MISSING`. A recipe removed after preparation also blocks commit. Pack refuses a missing required intermediate recipe, including legacy routes; ordinary packing still consumes its source stock.
- Order-detail `shipments[].shipment_id` is the shipment header used by the attachment route. The old sales-order-shipment row ID remains as `legacy_shipment_row_id`. A missing header is null, never an unrelated ID.
- Shift summary loads receipt transactions, evidence, referenced lots, balances and catalog in one SQL statement, then renders from that request-local snapshot. Effective ledger views, original voided quantities, stored kosher evidence, metadata receipt separation and confirmation hashes are preserved. Six action-specific parity tests compare the new loader with the individual receipt reader.
- Owner-only exception refusals no longer claim every decision is a held correction. Held corrections return `success:false`, `posted:false`, `held:true`, `status:awaiting_approval`.
- Future order dates default to the New York plant day. Migration 082 changes the existing atomic daily order-number allocator and the column default to that day. Existing orders and counters are not rewritten; explicit order dates remain supported.

## Oat bran: migration 081

Michael approved restoring the actual quantity on October 14. Current rows and the March
backup were read directly from production; posted historical consumption independently
corroborates them. Full sanitized evidence is in `pilot-triage-production-recipe-read.json`.

| Product | Formula row in production | Before | After | Unchanged batch size | Historical evidence |
|---|---:|---:|---:|---:|---|
| SS Original #1 (116), 90016 | 69 | 0 lb | 28 lb | 350 lb | TX 2267: 56 lb / 2 batches |
| SS Chocolate Chip #2 (114), 90011 | 56 | 0 lb | 29 lb | 393 lb | TX 2242: 290 lb / 10 batches |
| SS Cranberry #3 (118), 90013 | 82 | 0 lb | 28 lb | 379 lb | TX 2057: 140 lb / 5 batches |

The up file locks and guards product identities, active status, unchanged batch sizes,
all 24 formula rows and inventory-exclusion flags. It accepts only the recorded before
or corrected state and is rerunnable. Formula IDs are preserved. The strict down file
requires the applied marker and the complete exact after-state before restoring the three
zeros. Neither file reads or writes historical transactions, lots or consumption.
Migration 081 is data-only and deliberately not in the schema-only test dump; its tests
seed the recorded catalog. Migration 082 is the sole pending schema include.

Both were applied to the guarded staging database on October 14. Staging already contained
the recipes, so no recipe seeding was necessary. See `pilot-triage-staging-migrations.json`.

## Validation

- Baseline: 20 backend regression cases fail on the original backend (including the corrected missing-recipe substitution input), plus the oat-bran regression fails at 0 vs 28/29/28 without migration 081. The original 30-receipt summary executes 518 queries.
- New tests cover legacy routes, prepare/commit recipe removal, photo attachment using order-detail IDs, held response shape, owner refusal text, opposing database/plant calendar days, bounded summary queries, exact migration values, rerun, guard drift, strict down and historical immutability.
- The full suite is run serially on fresh local UTF-8 PostgreSQL 17. Existing race fixtures clean up their temporary databases. The main disposable database is removed at completion. No unrelated database is removed.
- The staging acceptance runner creates labelled test stock and exactly 30 numbered receipts through the real routes, verifies substitution ledger lines and replay, confirms the shift, and deactivates its temporary actor. These staging receipts remain as audit evidence.
- Initial staging database check: `MK-261014-004`, ticket 593, transaction 1000000471; 350 lb output, 213 lb oats consumed (185 + 28 substituted), zero oat bran. Summary 0.499/0.409/0.320 s; confirmation 0.428 s. Evidence: `pilot-triage-staging.json`.

Final full suite: **3,100 Python and 126 Node passed**, zero failures/skips; fresh UTF-8
PostgreSQL 17 database removed. Runtime `cf1ae6d` deployed to FastAPI-staging as
`9116e3a2-736a-4fc3-a811-941c96074ea7` (SUCCESS); health HTTP 200.
Hosted HTTPS acceptance: **MK-261014-005**, ticket 623, transaction 1000000501:
350 lb output, 213 lb rolled oats (185 + 28 substituted), zero oat bran; replay one post.
Thirty-receipt summary **1.229 / 1.089 / 1.008 s**; confirmation **1.454 s**.
Both temporary actors were deactivated. See `pilot-triage-staging-hosted.json` and
`pilot-triage-validation.json`. Production remains unchanged; draft review only.

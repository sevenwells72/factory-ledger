# S1 run types — implementation and rollout

Implements `S1-amendment-bake-vs-pack.md` with the owner's September 15 decisions.

## Behavior

- Create requires `run_type`: bake, coconut, pack or other. PATCH rejects changing it or the product; cancel and recreate instead.
- Bake/coconut accept a batch recipe and whole pans or pounds. Pans multiply the stored `default_batch_lb`, without `yield_multiplier`. The run saves that yield. Editing notes/date/status preserves it; changing quantity/unit uses current metadata.
- Pack stays creatable in cases/pounds and is labelled “does not consume WIP yet.” Other accepts active non-service products in pounds or cases when a case weight exists.
- Bake/coconut coverage follows the finished SKU's parent batch or batch BOM component; add-ins are ignored. Existing remaining-pound and total-coverage limits and lock order are preserved.
- One assignment wins; otherwise infer from run type and pack format by line code. Optional API `line_id` overrides remain. The form sends no line id.
- The daily sections of the existing Runs week board show the recipe, pans and expected pounds. Mixed bake/coconut plus pack coverage is allowed and warned on screen, including runs outside the displayed week. Health calculations remain unchanged.
- Run responses add `coverage_product_ids` for the form's routed SKU choices and per-coverage `mixed_bake_pack` for board warnings. These are derived, not stored. No new API operations.

## Applied data and schema

Migration `054_run_type` applied to production **2026-09-15 16:43:44 UTC** over port **5432**, before code merge. No existing runs were present. Schema-only snapshot refreshed with `scripts/dump_prod_schema.sh`.

The separate, idempotent `scripts/s1_batch_line_assignments.sql` inserted and verified exactly three granola assignments: 90008, 90025, 90026. This stays outside 054 so the migration contains only the specified schema changes. The fallback remains.

The guarded down file refuses any non-pack or pans row, printing blockers and deleting nothing. It restores the old unit check and removes the new columns/marker only when safe. Existing pack coverage survives. Code rollback can continue inserting pack rows using the database default.

## Interpretation and deferred work

“Daily board” means the daily sections of the live Runs week board, not the standalone scheduler prototype. Batch eligibility follows §2a (`type = batch`); no additional recipe-family taxonomy was invented. The Q8 rows are a separate data fix, as allowed by that question.

Q3 uses stored yields. Q6 Health wording, Q10 matrix yield cleanup and Q11 coconut correction remain outside this change. No WIP consumption links, ledger writes, inventory effects or changes to `production_schedule` or `feat/planner-v2`.

## Validation

- Full suite: **1,118 passed** (`scripts/run_tests.sh -o addopts='' -q`), including run types, migration/reapply/down guards, Q8 idempotency, Health and existing lock sequences.
- Existing Runs interaction/design audit: all four 1440/390 × light/dark variants pass, zero targeted STATUS failures.
- New run-type browser suite: all four variants pass; whole-pan rejection, bake/coconut expected pounds, immutable edit payloads, saved/current yields, routed choices, cross-week mixed coverage warning and daily rows verified.
- Python/JavaScript syntax and `git diff --check` pass. GPT schema remains exactly 30 operations and unchanged.
- Production marker, zero run rows, all four fallback lines and exactly three requested granola assignments verified after commit.

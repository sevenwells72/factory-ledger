# Daily inventory reconciliation — scoping (read-only)

Date: 2026-11-09 · Code audited: `origin/main` `4068c6a` (read from `~/dev/factory-ledger-home`; no checkout made) · Data: production, read-only (port 5432, every query inside `BEGIN; SET TRANSACTION READ ONLY; … COMMIT;`).
No code changes, no commits, no migrations, no database writes. `feat/planner-v2` not touched.

> This file sits in `~/Documents/factory-ledger`, which is an old checkout (Oct 15). Everything below describes current main, not that checkout.
> Related earlier scope: `docs/verification/daily-verification-scope.md` on branch `wip/home-mac-2026-11-02-daily-verification-scope` (blind spot counts, closeout, owner email). This check would feed the same email.

## The short version

1. **Everything needed to build this exists.** Void-aware balances and movements come from two views; there is no snapshot table yet; every make type has a recipe.
2. **Run by hand for Nov 5, 6 and 9, the balance comparison found zero differences.** Every make posted exactly its recipe quantities, so "expected" and "actual" usage are the same numbers.
3. **That is a limit of the check, not proof the stock is right.** Makes are posted *from* the recipe, and the "actual ending balance" is the book balance. The roll-forward can only catch changes to the books (edits to past days, voids, late entries, off-recipe makes). It cannot see what is physically on the shelf. Only counts can do that.
4. **The pack check is where real findings are.** One flag over tolerance in the three days (Nov 9), and a pattern of coconut batches packed out for slightly more than was made, leaving six lots below zero today.
5. **Recommended home:** one plain Python script, run from the existing nightly `exceptions-sweep` cron on Railway, moved to a Render Cron Job unchanged later. Email through an HTTPS email API (Resend) using the `httpx` library the app already has.

---

## 1. Where balances and movements come from

| Need | Source | Notes |
| --- | --- | --- |
| Posted, void-aware transactions | view `ledger_current_transactions`, filter `effective_status = 'posted'` | Applies the latest row in `ledger_corrections` (void / restore / amend) on top of `transactions.status`. Last redefined in migration 069. |
| Posted, void-aware lines | view `ledger_current_transaction_lines` joined to the view above | Applies line-level amendments (`product_id`, `lot_id`, `quantity_lb`). |
| On-hand per lot | `SUM(quantity_lb)` of the above grouped by `lot_id` | Ready-made: view `v_lot_quantities` (all lots) and `lot_balances` (balance > 0 only). Per product: `inventory_summary` (active products only). All three were made void-aware in migration 055. |
| Movement type | `ledger_current_transactions.type`: `receive`, `make`, `pack`, `ship`, `adjust` | Found/opening stock is an `adjust`. |
| Voids | `ledger_corrections` (`target_table='transactions'`, `event_type='void'`, `created_at`, `operator_id`, `reason`) | Only 9 correction rows exist; 3 since Oct 28. Older voids are just `transactions.status='voided'` with no timestamp. |
| Recipes | `batch_formulas` (`product_id`, `ingredient_product_id`, `quantity_lb` per batch, `exclude_from_inventory`) + `products.default_batch_lb`, `products.yield_multiplier` | This is what `/make` deducts. `boms` / `bom_lines` are empty. `product_bom` (114 rows) is the planner's copy and is not used for deductions. |
| Recorded substitutions | `transaction_substitutions` (per make: original ingredient, substitute or none, reason, note) | 8 rows, all the automatic oat bran → oats swap. |

**Snapshot table: none exists.** Nothing in the database is named snapshot, reconciliation or verification. (`shift_confirmations.summary_snapshot` is a per-person end-of-shift record, not an inventory snapshot.)

**Check on the method:** rebuilding balances from raw `transactions` + `ledger_corrections` "as of a moment" gives exactly the same numbers as the views today (0 of 160 lots differ). So past snapshots can be reconstructed for testing, and the nightly snapshot can simply copy the view.

Current size: 154 lots with stock across 80 products; 6 lots below zero (see §3). A nightly snapshot is about 160 rows.

## 2. Are recipes complete enough?

**Yes for every make type in use.** Only `batch` products are ever made (470 posted makes, all with a batch product as output). All 25 batch products have a recipe and a batch size.

| Group | Products | State |
| --- | --- | --- |
| Granola (current) | #9 Classic (107), #9 Choc Chip (108), Setton Cinnamon Almond (109), Vanilla Crisp #16 (112), Vanilla Almond (113), SS #1 (116), SS #2 (114), SS #3 Cranberry (118), SS Low Carb (119, 120), extra-kosher copies (283, 284), Setton Cocoa Crunch (111) | Complete. Ingredient pounds add up to the batch size within 0.4 lb. |
| Sweetened coconut | Fancy (125), Flake (126), Medium (127) | Complete. 324.5 lb of stock ingredients + 65 lb water (not tracked) → 360 lb booked output. |
| Toasted coconut | Toasted Sweetened Flake (128) | Complete. 324.5 lb in → 300 lb out (toasting loss built in). |
| Fruit Nut | Granola Fruit Nut Batch (179) | Complete. Uses 276.3 lb of **Batch Classic #9** as an ingredient (corrected in migration 078). |

Gaps and things the builder must handle:

1. **Recipes have no history.** `batch_formulas` holds only today's recipe. Expected usage must be computed the same night as the make. A later re-run for an old day would use the new recipe. Fix: store the expected quantities the checker used in its own results.
2. **Batch count is not stored.** It has to be derived: output lb ÷ (`default_batch_lb` × `yield_multiplier`). It came out a whole number for every make since Oct 28. The number also appears in the notes text ("6 batch(es) of …") but notes are not reliable data.
3. **Two recipes use another batch as an ingredient** (179 uses 107; inactive 122 uses 121). Expected usage for those is drawn from WIP lots, not ingredient lots. The checker must not assume ingredients are always `type='ingredient'`.
4. **Water** (ingredient 50) is flagged `exclude_from_inventory` on the three sweetened-coconut recipes and must be skipped.
5. **Yield is fixed, not measured.** Coconut output is booked at 360 lb per batch, but real pack-out runs about 1% higher (see §3). The recipe is complete; the yield assumption is slightly low.
6. **Never made, recipes look thin:** Setton Coffee #12 (110), SS Choc Chip #5 (115, 5 ingredients vs 8 for #2), SS Original #4 (117, 4 ingredients vs 7 for #1). Worth a look before first use; no effect on the checker today.
7. **Inactive Barely Sweet recipes** (121–124) point at inactive ingredients; 121 adds up to 346.2 lb against a 350 lb batch. Not in production; only matters if they come back.
8. **Packing has no recipe at all.** A pack moves pounds from a batch lot to a finished lot. Packaging materials are not deducted, and add-ins are deducted only when the finished product has a different parent batch. So "expected usage" applies to makes only; packs are checked by provenance (§3, check 3).
9. **Recipe-posted makes.** Since the reset, every make's ingredient lines equal recipe × batches exactly (the worker confirms the recipe; pounds are not weighed in). So the "make vs recipe" check will stay silent until formula-driven entry (`formula_entry.py`, 10% warning) is used with real weights. Note the checker's 2% is much tighter than that 10% warning.

## 3. Hand run: Nov 5, Nov 6, Nov 9 (the last three production days)

Method (a local script over a read-only export; nothing written to the database):

- **Opening** = balances as they stood at 23:59 ET the night before (rebuilt from entry and void timestamps).
- **Expected ending** = opening + receipts + make output − *recipe* usage (recorded substitutions accepted as correct) + packs + ships + adjustments ± voids, per lot. Recipe usage is spread over lots in the same proportion the make actually drew them.
- **Actual ending** = balances at 23:59 ET that night.
- **Tolerance** = greater of 2% or 5 lb.

### Result

| Day | Makes | Lots that moved | Balance differences | Make vs recipe | Over tolerance |
| --- | --- | --- | --- | --- | --- |
| Thu Nov 5 | 3 | 23 | **0** | all exact | **none** |
| Fri Nov 6 | 3 | 24 | **0** | all exact (232 lb oat bran → oats, recorded) | **none** |
| Mon Nov 9 | 7 | 35 | **0** | all exact (28 lb and 112 lb oat bran → oats, recorded) | **1** |

Makes: Nov 5 — Classic Choc Chip #9 ×4, Classic #9 ×3, Coconut Flake ×14. Nov 6 — SS Choc Chip #2 ×8, Coconut Fancy ×3, Coconut Medium ×3. Nov 9 — Classic #9 ×2 and ×2, Vanilla Almond ×1, Setton Cinnamon Almond ×1, Classic Choc Chip #9 ×1, SS Original #1 ×1 and ×4.

### What would have been flagged

**Over tolerance (would be emailed):**

- **Nov 9 — packed more than was made.** TX2973 packed 375 lb of Granola Honey Nut 25 LB from Batch SS Original #1 lot "OCT 28 2026". Only 350 lb of that lot was ever made. The missing 25 lb (7.1%) was added two minutes earlier by adjustment TX2972: reason "unknown", shared key, no named person, no note.

**Under tolerance but worth knowing (shown here; emailed only if you choose):**

- **Nov 5 — coconut over-pack.** Batch Coconut Sweetened Flake lot "NOV 05 2026": 5,040 lb made, 5,095 lb packed (+55 lb, 1.1%). The lot now sits at **−55 lb**.
- **Nov 6 — coconut over-pack.** Batch Coconut Sweetened Medium lot "NOV 06 2026": 1,080 lb made, 1,090 lb packed (+10 lb, 0.9%). Lot at **−10 lb**.
- **Nov 9 — entry voided the same day.** TX2971 (+375 lb adjustment, shared key) was entered 09:23 and voided 09:25 as "Adjustment entered by mistake". Net effect zero.

**Nothing flagged for:** receipts (Almonds 1,500 lb, Light Brown Sugar 5,000 lb on Nov 5), shipments, the Nov 5 physical-count adjustment (+1,400 lb, named, reason given), or packs with no make behind them (every batch lot packed on these days has a posted make).

### What the same check shows for the other days since the reset (context)

| Day | Finding |
| --- | --- |
| Oct 28 | TX2892 SS Original #1: recipe calls for 28 lb oat bran, 0 used, **no substitution recorded** (before the automatic swap existed). This is the one true "make vs recipe" catch. |
| Oct 28, 29, Nov 2 | Coconut Flake packed above make by 50–70 lb, each time covered by a small "unknown" adjustment from the shared key. |
| Nov 2 | Two makes voided and re-entered by the owner for Oct 28/29 (kosher reclassification) — shows as "void of an earlier day" + "late entry". Canola Oil lot 26-09-17-CBSF-001 went to −14 lb. |
| Nov 3, 4 | Coconut Flake lots left at −70 and −60 lb. Sugar 6X lot 26-08-06-SWEE-001 went to −68.5 lb. |
| Nov 4 | Five adjustments on CQ Coconut Flake 10 LB (+1,200, +990, −1,400, −940, +940 lb), all reason "unknown", shared key, no named person. Not a reconciliation difference, but exactly what an owner email should surface. |

**Six lots are below zero today** (posted lines only): Canola Oil 26-09-17-CBSF-001 (−14), Sugar 6X 26-08-06-SWEE-001 (−68.5), Batch Coconut Flake NOV 03 / NOV 04 / NOV 05 (−70, −60, −55), Batch Coconut Medium NOV 06 (−10). The Oct 28 handoff recorded zero.

### What this means for the design

- The roll-forward is a **books-integrity check**: it proves nothing moved a balance except entries the checker saw. It will fire on edits to earlier days, voids, late entries and off-recipe makes. It will not fire on physical loss, miscounts or unrecorded use.
- **2% of a balance is too loose for big lots.** Pooled Coconut Flake holds about 150,000 lb, so 2% is 3,000 lb — more than a full day's use. Recommend measuring the percentage against the **day's expected movement** for that lot, not its balance.
- **Add three cheap rules** the hand run shows are needed: (a) any lot below zero is always flagged; (b) any adjustment with reason "unknown" or no named person is listed; (c) any void or late entry touching an earlier day is listed.
- Coconut will flag almost daily until the booked yield (360 lb per batch) matches real pack-out, or the over-pack is recorded as a yield adjustment with a reason (`hydration_yield` exists).

## 4. Where to run it, and how to send email

| Option | Verdict |
| --- | --- |
| **Railway cron (existing `exceptions-sweep`, 02:30 ET)** | **Use now.** Already runs nightly with the production database guards. Add the new script to the same run; no new service. |
| **Render Cron Job** | **Same script later.** No Render plan exists in the repo (only old placeholder text in `DEPLOYMENT.md`). If the job is a plain `python scripts/…` command reading environment variables, the move is a copy of the command and its variables. |
| **Supabase pg_cron** | **No.** Not installed (available, version 1.6.4, would need enabling). The recipe and substitution rules would have to be rewritten in SQL and kept in step with the Python make code; it cannot send email without also enabling `pg_net`; and it cannot be tested with the existing pytest setup. |

Notes:

- Railway "config as code" stops working Dec 1 (handoff item 16). The cron's start command is set in the Railway dashboard, not in `railway.json`, so this job adds nothing to that problem. Changing the command is a production change needing Michael's approval.
- The job should compare against **the most recent earlier snapshot**, whatever its date, not "yesterday". A missed night or a weekend then needs no special handling, and the move between hosts cannot leave a gap.
- 02:30 ET works: all entries on these days were made between 09:00 and 17:30.

**Email — simplest reliable route:** an HTTPS email API. Recommend **Resend**: one POST with `httpx` (already in `requirements.txt`), one secret (`RESEND_API_KEY`), plus `RECON_EMAIL_TO` / `RECON_EMAIL_FROM`. One-time setup: verify the sending domain (three DNS records for sevenwellsgranola.com). There is no email code anywhere in the app today.
Not recommended: SMTP through the Google mailbox. It needs no DNS work, but hosting providers commonly block outbound mail ports on lower plans — check before relying on it on either Railway or Render.

Always send, even when clean ("No differences — 3 makes, 24 lots checked"), so a silent night means the job failed, not that all is well.

## 5. PR plan for Codex

One PR, built in a fresh worktree off `origin/main`. Nothing here adds a ChatGPT action (office schema is at 30 of 30).

### Migration (next free number — 094 today)

Three new tables, all append-only (same trigger as migration 039/076), row-level security on, no changes to existing tables, with a down file.

- `inventory_snapshots` — `id`, `taken_at`, `plant_date`, `job_run_id`, `lot_id`, `product_id`, `quantity_lb`. Unique on (`job_run_id`, `lot_id`). Every lot with a non-zero posted balance.
- `reconciliation_runs` — `id`, `started_at`, `finished_at`, `previous_run_id`, `window_start`, `window_end`, counts (makes, lots checked, flags), `email_sent_at`, `email_error`, `status`.
- `reconciliation_flags` — `id`, `run_id`, `kind`, `severity`, `product_id`, `lot_id`, `transaction_id`, `expected_lb`, `actual_lb`, `difference_lb`, `tolerance_lb`, `detail` (jsonb: the recipe quantities used, batches, substitutions, human-readable message), `created_at`.
  Kinds: `BALANCE_DIFFERENCE`, `MAKE_OFF_RECIPE`, `MAKE_INGREDIENT_MISSING`, `MAKE_UNLISTED_INGREDIENT`, `MAKE_NO_RECIPE`, `PACK_NO_MAKE`, `PACK_EXCEEDS_MAKE`, `NEGATIVE_LOT`, `VOID_OF_EARLIER_ENTRY`, `LATE_ENTRY`, `UNEXPLAINED_ADJUSTMENT`.

### Code

- `reconciliation.py` (new module, read-only except its own three tables):
  - `take_snapshot(cur, run_id)` — copies posted balances per lot from `ledger_current_*`.
  - `expected_usage(cur, make_id)` — batches from output ÷ batch size; recipe × batches; skip `exclude_from_inventory`; apply `transaction_substitutions` (shortfall on the original moves to the substitute; "not used" with a reason means expected 0).
  - `reconcile(cur, previous_run, this_run)` — window is by entry time (`created_at`) and void time (`ledger_corrections.created_at`) between the two snapshots; per-lot roll-forward; make-vs-recipe; pack provenance; the three extra rules. Tolerance in one function: greater of 2% or 5 lb.
  - `render_email(run, flags)` — plain text and simple HTML, plain language, flags first.
- `scripts/daily_reconciliation.py` — entry point modelled on `scripts/exceptions_sweep.py` (same database guards via `staging_safety.py`, `lock_timeout`, `statement_timeout`). Order: snapshot → compare with previous snapshot → save flags → send email → record send. `--dry-run` prints the email and writes nothing. If email fails, flags are still saved and the error is recorded.
- The checker **never** posts, voids or adjusts anything, and never writes to `transactions`, `transaction_lines`, `ledger_corrections`, `exceptions` or any existing table. A test enforces this.

### Endpoints (owner/office only, `include_in_schema=False`, not in either GPT schema)

- `GET /verification/reconciliation/runs?limit=` — recent runs with flag counts.
- `GET /verification/reconciliation/runs/{id}` — one run with its flags.
- `GET /verification/reconciliation/preview` — runs the comparison against the latest snapshot without saving (owner only).

No dashboard page in this PR; the email is the interface. A page can follow.

### Tests (pytest against the local test database, as existing suites do)

1. Snapshot equals `v_lot_quantities` and excludes voided transactions.
2. Clean day: receipt + make at recipe + pack + ship → zero flags.
3. Make 3% over recipe on one ingredient → `MAKE_OFF_RECIPE`; 1% over → no flag; 4 lb over on a small line → no flag (5 lb floor).
4. Recorded oat bran → oats substitution → no flag. Same make with no substitution row → `MAKE_INGREDIENT_MISSING`.
5. Water / `exclude_from_inventory` line ignored. Recipe using a batch as an ingredient (Fruit Nut) handled.
6. Void of an earlier day's entry → `VOID_OF_EARLIER_ENTRY`, and the balance still reconciles. Entry created and voided inside the window → no balance difference.
7. Backdated entry (`business_date` earlier than entry date) → `LATE_ENTRY`.
8. Pack from a batch lot with no make → `PACK_NO_MAKE`. Pack above make by 7% → `PACK_EXCEEDS_MAKE`; by 1% → no flag. Lot below zero → `NEGATIVE_LOT`.
9. Missed night: snapshots two days apart still reconcile across the gap.
10. Re-running the job for the same window does not send a second email.
11. Email failure leaves flags saved and `email_error` set.
12. Guard test: after a run, row counts and checksums of `transactions`, `transaction_lines`, `ledger_corrections` and `exceptions` are unchanged.
13. Replay fixture built from this hand run (Nov 5, 6, 9) reproduces the results in §3.
14. OpenAPI guard: office schema still 30 operations, floor schema unchanged.

### Rollout (each step needs Michael's approval)

1. Staging: apply the migration, run with `--dry-run`, then a real run sending to a test address.
2. Production: verified backup → apply migration → first run (snapshot only; nothing to compare yet) → second night produces the first email.
3. Add the script to the `exceptions-sweep` cron command.
4. Update the three change logs; add the regression row to `FACTORY_LEDGER_CHANGELOG.md` after deploy.

### Decisions needed from Michael before the build

1. Tolerance base: 2% of the **day's movement** (recommended) or 2% of the balance (as written)?
2. Should a lot below zero always be flagged, even when the amount is inside tolerance? (Recommended: yes.)
3. Should "unknown"-reason or unnamed adjustments be listed in the email? (Recommended: yes.)
4. Coconut yield: raise the booked yield, or record over-pack as a yield adjustment? Until one is chosen the email will mention coconut most days.
5. Email provider and the address to send to.

# Formula-Driven Production Entry — spec (draft)

**Status:** Phase A built on `feat/formula-driven-entry` (not deployed), 2026-11-01. Revision 3: D1 decided — every decision is now closed (§5). Revision 2 recorded D2–D7 and the phasing. Phase B (recipe versions) is not started.
**Branch:** `docs/formula-driven-entry` (from `origin/main` `87f500f`).
**Screen:** Entry → **Make** (`dashboard/entry.html` + `dashboard/entry.js`, draft card `dashboard/ledger-ticket.js`).

---

## 1. How make/bake entry works today

### 1.1 Screen and endpoints

| Step | Where | What happens |
|---|---|---|
| Form | `dashboard/entry.js:39-48` (action `make`) | Worker fills the form (below) |
| Prepare | `POST /make/prepare` → `write_tickets.py:869` → `ticket_actions.validate` (`ticket_actions.py:122-160`) → `main._make_preview_core` (`main.py:8752`) | FL builds a draft: recipe pounds, suggested lots (oldest first), shortages, oat-bran swap, lot-confirmation card. Nothing posts. |
| Confirm lots | `ledger-ticket.js:30-62` | One row per suggested lot: worker types the last 4 / full code / pallet code from the bag; "Check confirmations → new draft" re-prepares |
| Record | `POST /tickets/{ticket}/commit` (`write_tickets.py:873`) | Re-checks everything under row locks, posts the transaction, writes `ingredient_lot_consumption`, lot evidence, substitutions, shortage flags |

The same make logic also serves the FL Assistant (`fl_assistant.py:1057`, `assistant_tools.py:21`) and the legacy `POST /make`, `/make/preview`, `/make/commit` routes (`main.py:9719`, `16444`).

### 1.2 What the worker fills in by hand

1. **Product** (search picker)
2. **Number of batches** (whole number)
3. Output lot code (optional; FL makes `BYY-MMDD-NNN` if blank)
4. Note (optional)
5. "I confirm the selected output product/SKU" checkbox (needed when sibling SKUs share a formula)
6. **Substitutions** (optional, per row): recipe ingredient, substitute product, substitute lot, **free-text** reason
7. **"Choose a different ingredient lot"** (optional, per row): ingredient + one lot
8. On the draft: **typed confirmation of every lot** (last 4 characters, full code, or pallet)

### 1.3 Are ingredients pre-filled from the recipe?

**Yes, already.** The worker never types ingredient pounds. FL computes `needed_lb = batch_formulas.quantity_lb × batches` for every recipe line (`main.py:8803`). Lines with `exclude_from_inventory` (e.g. water in sweetened coconut) are skipped. The draft lists each ingredient with its suggested lot(s) and pounds.

What the worker **cannot** do today:
- change an ingredient's pounds (only the batch count scales everything);
- split one ingredient across lots with chosen pounds: overrides allow **one lot per ingredient** (`DUPLICATE_LOT_SELECTION`, `ticket_actions.py:128`), and FL decides the pounds;
- pick the lot from a list of lots in stock on the form: the override row is a search box, while the draft's candidate list only appears to disambiguate a 4-character suffix;
- give a reason from a fixed list for a substitution (the reason is free text; FL only checks it is not blank, `lot_confirmation.py:252`).

---

## 2. Where the data lives

| Thing | Location | Notes |
|---|---|---|
| Recipe (what make deducts) | table `batch_formulas` (`product_id`, `ingredient_product_id`, `quantity_lb`, `exclude_from_inventory`) | Pounds **per one batch**. Edited in place (e.g. `migrations/078_recipe_corrections.sql`); **no versions, no history**. |
| Planner BOM | table `product_bom` | Separate copy used by the planner/requirements (`/production/requirements`, `/admin/product-bom`). Not what make deducts. `boms`/`bom_lines` are empty. |
| Pan / batch weight | `products.default_batch_lb` (+ `yield_multiplier`) | Output lb = `default_batch_lb × batches × yield_multiplier` (`main.py:8756`). Coconut 90003/4/5: owner ruling 2026-09-15, 1 pan = 360 lb. FL stores no separate "pan" record; pans are displays of lb ÷ `default_batch_lb`. |
| Lot balances | sum of `transaction_lines.quantity_lb` over **posted** lines (`api.POSTED_LINES`), per `lots.id` | Matches the posted-only rule. |
| Lot dates | `lots.received_at`, fallback `lots.created_at` | No expiry / best-by column exists. |
| Oldest-lot (FIFO) helper | `ticket_actions.choose_inputs` (`ticket_actions.py:50`) | Orders lots by `COALESCE(received_at, created_at), id`; takes from the oldest, **spills into the next** when one runs out; the shortfall is pinned on one lot (`exceptions_enforcement.assign_shortfall`). Also: `lot_confirmation.confirmation_group` lists every in-stock lot oldest-first as `candidates`. |
| Lot confirmation (R4, 2026-10-06 rules) | `lot_confirmation.py` (`validate_plan`, `verify`, `record_confirmations`) | "Prepare never pre-confirms a suggestion": commit fails `LOT_NOT_CONFIRMED` unless every consumed lot was typed/scanned. "Different lot" creates an exception trace lot. |
| Oat bran → rolled oats | `oat_bran_substitution.py` | Only SS batch products `90016`, `90011`, `90013`; bran `11031` → oats `11032`; pound-for-pound for the **missing** part only; recorded with reason `oat_bran_auto_substitute`; recipe never edited. |
| Shortage policy (R3) | `exceptions_enforcement.py` (`shortfalls`, `assign_shortfall`) | Short stock does not block; the post goes negative on one lot and opens a SHORTAGE exception owned by Arturo. Only "no lot exists at all" blocks. |
| Substitutions record | table `transaction_substitutions` (`migrations/063`) | Append-only. |
| Runs | table `production_runs` (`migrations/053`, `054`: `run_type` bake/pack/coconut, `planned_unit` incl. `pans`, `pan_yield_lb_used`) | `/production/runs/{id}/complete` writes the run row only, never inventory. |

---

## 3. The feature

### 3.1 Goal

The worker picks a product and types **how many pans**. FL fills in every ingredient, its pounds and a suggested lot. The worker taps to confirm each lot, edits only what really differed, and records one entry for the run. Inventory changes only on the final confirm.

### 3.2 Worker flow

1. **Pick product** (same picker as today; batch products only).
2. **Enter number of pans** (whole number ≥ 1). **Decided (D1):** one pan = one FL batch = the product's existing `default_batch_lb` for all granola; the form says "Pans" and FL multiplies the recipe by the pans.
3. FL shows an **ingredient list**, one card per recipe line:
   - ingredient name, **standard pounds** = recipe qty × pans;
   - **suggested lot** pre-filled (see 3.3), shown as a pick-list of every in-stock lot, oldest first, each with code, received date and lb on hand;
   - the lot row starts **unconfirmed** (grey, "Tap to confirm");
   - **pounds field**, pre-filled with the standard pounds, editable;
   - **"Add another lot"** to split the ingredient: a second lot row with its own pick-list and pounds;
   - **"Not used / substituted"** to set the line to 0 or swap the product (needs a reason, 3.5).
4. **Confirm each lot.** Tapping the lot opens the existing confirmation (last 4 characters, full code, or pallet). No lot is ever pre-confirmed. Record stays disabled until every lot row with pounds > 0 is confirmed.
5. **Review** card: warnings (over 10%, shortages, substitutions) listed for acknowledgement.
6. **Record.** FL re-checks and posts once. Receipt shown.

### 3.3 Lot suggestion

- **Decided (D2): keep today's rule**, `ticket_actions.choose_inputs`: take from the oldest lot first (by `COALESCE(received_at, created_at), id`), and when it runs out continue with the next-oldest, and so on.
- The form shows **every lot the spill uses** as its own row, pre-filled with the pounds FL plans to take from it. Each row's pounds are editable and each row needs its own confirmation.
- If total stock is short, pre-fill as above and show the shortage (3.7).
- A suggestion is only a suggestion: `confirmed:false` until the worker taps and types the code (2026-10-06 rule R4).
- Pooled opening-stock lots are shown by FL lot code only, never as a supplier lot.

### 3.4 Editing pounds and splitting lots

- Any line's pounds may be edited. The split rows must add up to the pounds the worker entered for that ingredient; FL shows the running total.
- Each lot row carries its own pounds. FL posts exactly those pounds per lot.
- Removing a split row puts its pounds back on the row above.
- **Decided (D7):** changing a row's **pounds** keeps its lot confirmation. Changing a row's **lot** clears the confirmation; the new lot must be confirmed before Record is enabled.

### 3.5 Zero or substituted ingredient: reason required

- Setting a line to **0** or **substituting** a product requires a reason picked from a **fixed list** (no free text for the reason; optional note).
- **Decided list (D3):**

  | Code | English | Note |
  |---|---|---|
  | `out_of_stock` | Out of stock | |
  | `not_used` | Not used | |
  | `substituted` | Substituted | |
  | `quality_problem` | Quality problem | |
  | `owner_instruction` | Owner instruction | |
  | `other` | Other | note required |

  Spanish labels to be written at build time and checked by Liz/Arturo.
- `oat_bran_auto_substitute` stays a system reason, not on the worker's list.
- The **master recipe is never changed** by an entry. Substitutions and zeroes are stored on this run only (`transaction_substitutions` / excluded lines).

### 3.6 Oat bran

- When oat bran is short, FL auto-substitutes rolled oats **pound-for-pound for the missing part**, recorded with `oat_bran_auto_substitute`.
- **Decided (D4): apply to every recipe containing oat bran (11031)**, not only the SS products 90016 / 90011 / 90013. This replaces the earlier SS-only rule written in `oat_bran_substitution.py` (`SS_CODES`, `eligible()`), so that rule changes in the build PR and its tests (`tests/test_oat_bran_*`) are updated to match.
- Because the swap lives in the shared make code, it then applies to every make (Entry screen and FL Assistant), not only this screen.
- The swapped oats line still needs its lot confirmed like any other line.
- **Which products this affects (read-only check, 2026-11-02):** `batch_formulas` rows with oat bran 11031, run in a `BEGIN; SET TRANSACTION READ ONLY;` transaction on staging and on production. Both return the same three products, all active, none with the bran line excluded from inventory:

  | Product | Code | Oat bran lb / pan | Rolled oats lb / pan | Pan lb |
  |---|---|---|---|---|
  | Batch SS Original Granola #1 (id 116) | 90016 | 28 | 185 | 350 |
  | Batch SS Chocolate Chip Granola #2 (id 114) | 90011 | 29 | 200 | 393 |
  | Batch SS Cranberry Granola #3 (id 118) | 90013 | 28 | 185 | 379 |

  So D4 starts no new product swapping today; it applies automatically to any recipe that adds oat bran later.

### 3.7 Short stock never blocks

- If confirmed lots do not hold enough, the entry still records. FL posts the pounds, the lot goes negative, and a SHORTAGE flag opens for Arturo (existing R3 behaviour).
- The draft says plainly: "Only X lb of Y on hand — this will be recorded as a shortage."
- Only blocker kept: the ingredient has **no lot at all** in FL (nothing to record the shortage against), as today.

### 3.8 The 10% warning

- For each ingredient, compare the pounds entered with the standard (recipe × pans).
- **Warn only when the difference is more than 10%** of the standard, in either direction. At 10% or less: no warning.
- A zeroed line (−100%) and a substituted line count as "more than 10%" and already carry a reason; they show in the warning list without a second acknowledgement.
- A warning does not block; it needs one tap to acknowledge (existing `acknowledged_warnings` mechanism) and is stored on the record.
- **Decided (D5): per ingredient only.** No check on total input weight.
- **Decided (D6):** every ingredient difference over 10% appears on the **owner's weekly review list**, next to corrections over 10% (R2): product, run receipt, ingredient, standard lb, used lb, % difference, reason if any, who recorded it.

### 3.9 What is saved

- **One record per run** (one make transaction, one output lot), never one per pan. Pans are stored on the record as the count entered.
- **Recipe used**: Phase A stores the standard pounds per line as read at save time, so later recipe changes never rewrite what this run "should" have used. Phase B adds the recipe **version** reference (4.2 item 6).
- Per ingredient: standard lb, entered lb, lot rows (lot, lb, confirmation method), reason if 0/substituted, warning acknowledged.
- **Lots re-checked at save**: on Record, FL re-reads every chosen lot under lock (exists, not merged, right product, balance) and re-validates confirmations. If something changed since the draft (lot merged, recipe changed), the save stops with "Prepare again". A balance drop only changes the shortage amount; it never blocks.
- **Inventory deducted only on final confirm.** Draft, edits and confirmations post nothing. Abandoning the draft leaves no trace in inventory.

### 3.10 Out of scope

- A pre-bake preparation stage (weighing out / staging ingredients before baking).
- Starting this entry from the **Runs** screen ("Record this run" pre-filling product and pans from a planned bake run). Planned as a later hook; this spec keeps the entry independent of `production_runs`.
- Editing master recipes (stays in admin).
- Pack entries.

---

## 4. Reuse vs new

### 4.1 Reuse as-is

| Piece | Where |
|---|---|
| Prepare/commit ticket lifecycle, payload hash, re-check under locks at commit | `write_tickets.py` |
| Recipe × batches math, excluded lines, SKU confirmation, output lot code | `main._make_preview_core` |
| Lot confirmation (last 4 / full code / pallet / different lot), evidence records | `lot_confirmation.py`, `ledger-ticket.js` |
| Oldest-first lot order and the in-stock candidate list | `ticket_actions.choose_inputs`, `lot_confirmation.confirmation_group` |
| Shortage post-and-flag | `exceptions_enforcement.py` |
| Oat bran auto-substitute | `oat_bran_substitution.py` |
| Substitution storage | `transaction_substitutions` |
| Warning acknowledgement | `acknowledged_warnings` at commit |
| Posting exact per-lot pounds | `_post_prepared_inputs` (`main.py:8719`) already posts a lot/pounds plan |
| Product and lot pickers, `/entry/lots` | `entry.js`, `resolution.py:920` |

### 4.2 New

1. **Per-lot pounds and splits in the request**: `MakePrepareRequest` gains per-ingredient `{ingredient_product_id, lines:[{lot_id, quantity_lb}]}`; replaces the one-lot-per-ingredient override for this screen. Validation: lots belong to the ingredient, pounds > 0.
2. **Edited pounds**: requirements come from what the worker entered, not only recipe × batches; `_post_prepared_inputs` checks against entered totals.
3. **Lot rows from the existing spill plan**: no change to `choose_inputs` (D2); the form renders each planned lot as an editable, separately confirmed row.
4. **Fixed substitution/zero reason list** (D3): kept as a code constant in Phase A (no DB change), validated at prepare and commit; the current free-text reason input becomes a select.
5. **10% deviation warning** per ingredient (D5) with `requires_ack`, and the same facts fed to the owner's weekly view (D6), using the review-facts pattern corrections already use (`exceptions_enforcement.py:227`). Build-time check: confirm where that weekly view is rendered today; if it has no screen yet, list these rows wherever the correction highlights appear.
5a. **Oat bran for every recipe** (D4): widen `oat_bran_substitution.eligible()` from `SS_CODES` to any batch whose recipe has an inventory-deducted oat bran 11031 line; update its tests.
5b. **Lot confirmation follows the lot, not the pounds** (D7).
6. **Recipe versions (Phase B, separate PR)**: new migration, e.g. `recipe_versions (id, product_id, version, created_at, created_by, note)` + `recipe_version_lines`, or a `version`/`valid_from` on `batch_formulas`, and a `recipe_version_id` + standard-lb snapshot on the make record. Admin recipe edits (`/admin/bom/...`) must create a new version instead of editing rows in place. **Production migration: needs Michael's approval, backup first.**
7. **Pans label and count stored on the record** (the "batches" field relabelled "Pans", with the count kept explicit).
8. **New Make form UI**: ingredient cards, tap-to-confirm lot pick-list, pounds fields, add-lot, reason select, running totals, bilingual EN/ES, phone-width layout.
9. **Tests**: split lots, edited pounds, 10% boundary (exactly 10% = no warning), zero/substitute needs reason, oat bran, shortage posts, re-check-at-save failures, recipe-version snapshot, no posting before commit.

### 4.3 Size estimate

**Medium-to-large.** About 60% of the hard parts (ticket lifecycle, lot confirmation, shortage policy, oat bran, posting exact lots) already exist. The new work is concentrated in:
- the Make form UI rewrite (largest single piece),
- per-lot/pound request + validation changes in `ticket_actions` / `write_tickets` / `main`,
- recipe versioning (migration + admin edit path + snapshot), which is what pushes it from medium toward large.

**Phasing (decided):**
- **Phase A first (medium), one PR, no DB change:** pans form, editable pounds, lot rows from the spill plan with add/remove, tap-to-confirm, reason list (code constant), per-ingredient 10% warning + weekly view, oat bran for every recipe. Per-ingredient standard vs used pounds and reasons are kept in the ticket's stored payload/specification and the existing `transaction_substitutions`, so no new table or column is needed. Until Phase B, the record shows the recipe as it was read at save time, not a version number.
- **Phase B later, separate PR (small-medium, migration):** recipe versions, admin edits creating a new version, and the version reference on each record. Needs Michael's approval and a backup before the migration.

Staging first; production only with Michael's per-step approval, backup and verification.

---

## 5. Decisions

### 5.1 Decided (Blubber, 2026-11-01)

| # | Decision |
|---|---|
| D2 | Keep today's lot rule: oldest first, then the next (`choose_inputs`). The form shows every lot the spill uses, pre-filled with its pounds, each editable. |
| D3 | Reason list: Out of stock, Not used, Substituted, Quality problem, Owner instruction, Other (note required). |
| D4 | Oat bran → rolled oats pound-for-pound applies to **every recipe containing oat bran**, not only SS. |
| D5 | 10% warning is per ingredient only. |
| D6 | Ingredient differences over 10% appear on the owner's weekly review list. |
| D7 | Changing pounds does not require re-confirming the lot; changing the lot does. |
| Phasing | Phase A (form + rules, no DB change) first; Phase B (recipe versions) later, in a separate PR. |

### 5.2 Decided later (2026-11-01)

| # | Decision |
|---|---|
| D1 | 1 pan = 1 FL batch for all granola, using each product's existing `default_batch_lb` (read-only production check the same day: every active granola batch weight matches its recipe total and recent makes, except a pre-2026-10-13 Fruit Nut gap and Dark Chocolate 350 vs 346.2 lb recipe). |

No decisions are open.

## 6. Phase A as built (`feat/formula-driven-entry`)

- **Screen:** Entry → Make asks for **Number of pans**; substitutions take a reason from the fixed list (note required for Other). The old "Choose a different ingredient lot" row is gone: bags are chosen per ingredient on the draft. The draft shows one card per ingredient (`dashboard/make-formula.js`): recipe lb, bags pre-filled from the oldest-first-then-next plan, editable lb per bag, Add another bag, Remove bag (its lb move to the bag above), Not used (0 lb) with the reason list, a typed code per bag, and "Check bags and pounds → new draft". Changing lb keeps a bag's confirmation; changing the bag clears it (D7).
- **Server:** `formula_entry.py`; `/make/prepare` accepts `formula_entry: true` and `ingredient_lines[]` (`{ingredient_product_id, bags:[{lot_id, quantity_lb}], reason_code, note}`). Only the Make screen sends them; FL Assistant and the legacy routes keep their payload and draft unchanged (the fields are excluded from the Assistant's tool schema). Ingredients sent as lines use exactly those bags and pounds; others keep the FIFO plan. The ticket freezes the entered pounds and the commit posts them (`required_lb` in `main._make_commit_core`). Output stays pans × `default_batch_lb`.
- **Reasons:** 0 lb or a swap without a fixed-list reason is a draft blocker (and refused at commit). Not-used reasons are saved in `transaction_substitutions` (substitute NULL); the recipe is never changed.
- **10% rule:** an ingredient more than 10% from recipe × pans (exactly 10% is fine) adds one `INGREDIENT_OFF_RECIPE` warning that must be ticked before Record. Not-used, substituted and oat-bran-swap lines are listed, with their reason, in a second `INGREDIENT_CHANGED` warning that needs no tick (spec 3.8). Each such row is saved in `ingredient_review` (`weekly_view: true`) on the ticket receipt.
- **Where the owner sees it (D6, Blubber 2026-11-02):** on the **shift summary**, where correction highlights already appear: each make row gets an `INGREDIENT_OFF_RECIPE` (warn), `INGREDIENT_NOT_USED`, `INGREDIENT_SUBSTITUTED` or `INGREDIENT_SWAP` (info) flag with ingredient, recipe lb, used lb, %, reason and who recorded it.
- **Oat bran:** any batch recipe that deducts oat bran 11031 now gets the pound-for-pound rolled oats (11032) swap, in every make path. On the Make screen (fixes from review of f8ba30d, Blubber 2026-11-02):
  - The card's "Recipe" lb, the 10% check and the saved `standard_requirements` are the **original recipe × pans**, never FL's stock-adjusted split. When oats replace bran, oat bran + the swapped oats are compared with the recipe bran (and oats less the swapped lb with the recipe oats); the bran card reads "oat bran → rolled oats (swap)". Using exactly the recipe gives no warning.
  - The swap that posts is the one the entered pounds make (bran short of the recipe, oats over it), not FL's proposal. If the worker uses the real bran, no swap is recorded.
  - **Never blocks:** if the worker enters more oat bran than FL shows on hand, Record posts what they confirmed and flags the shortage, like any other short ingredient. Prepare and Record agree. (Callers that do not send a bran line keep the old "prepare again" refusal when bran disappears between prepare and Record.)
- **Unchanged:** prepare → confirm → Record ticket flow, lot confirmations and evidence, shortage post-and-flag, one make transaction per run, nothing posts before Record, lots re-locked and re-checked at Record. No migration, no OpenAPI GPT operation added.
- **Not in Phase A:** recipe versions (Phase B), Runs-screen hook, a separate weekly review screen (the shift summary carries the flags until then).

## 7. Make-entry simplification, PR-1 (`feat/make-entry-simplify`, 2026-11-02)

Goal: Arturo's Make screen faster and more reliable, with no new guessed inventory.

**Screen (390 × 844 first).** Product + **Number of pans** stay on top; output lot code, note and
substitutions fold under "More options". "Show ingredients and lots" prepares the draft. Each
ingredient shows recipe lb and **one big lot button** ("Tap to confirm: lot · supplier lot ·
received date · lb"). One tap confirms that lot; nothing is ever pre-confirmed. "Change lot or
pounds" (folded) holds the lot list (radio buttons, oldest marked), lb per lot, Add another lot,
Remove this lot and Not used. The sticky Record bar shows "N lots still need a tap". The typed
last-4 / full-code / pallet fields are gone from this screen (FL Assistant and the GPT keep them).

**Lot evidence.** A tap is stored as `transaction_lot_confirmations.method='selected'`,
`value` = the FL lot code shown, `actor_id` = who tapped (signed-in person only; shared keys
get `ACTOR_REQUIRED`). FL checks the lot belongs to the ingredient, is not merged and still
has that code, at prepare and again under lock at Record. `selected` is refused outside the
Make screen (`LOT_SELECTION_MAKE_SCREEN_ONLY`). Needs **migration 090** (adds the method + an
index); until it is applied FL answers `LOT_SELECTION_UNAVAILABLE` and records nothing.
Taps that only confirm the suggested lot are sent with Record (no extra re-check); a different
lot or changed pounds still needs "Check lots and pounds" (D7 unchanged).

**One suggested lot (replaces D2's spill on this screen only).** FL suggests the oldest lot
*with book stock* for the whole amount and never spills into a second lot on book balances.
If that lot is short on the books, the card says so: add the real second lot with its pounds,
or Record and the rest posts as a shortage on that lot (R3 unchanged: warning, not blocker;
SHORTAGE flag for Arturo). Real multi-lot use with known pounds posts exactly. No new
fallback deduction exists; zero-stock ingredients keep today's shortage pinning.

**Never NOT RECORDED vs RECORDED confusion; never twice.**
- The draft always opens with "NOT RECORDED YET — this is a draft"; Record success shows
  "RECORDED ✓ MK-…"; any failure shows "NOT RECORDED — reason. Your lots and pounds are kept."
  inline at the top of the draft (not only in the page notice).
- Entries (lots, pounds, taps) survive Edit entry, Cancel, a failed prepare, a failed
  re-check and an expired/stale draft ("Check again (keeps my entries)").
- Each Make form carries an `entry_attempt_id`. A retry of the same ticket replays the saved
  receipt (unchanged). A **new** draft or ticket of a Make already committed (or held for the
  owner) gets `MAKE_ALREADY_RECORDED` with the receipt: blocked at prepare, refused at Record
  (checked under the make posting lock, so two open drafts cannot both post).
- No answer from FL at Record → "NOT CONFIRMED — … Tap Record again: FL will not record the
  same Make twice." Record stays on and retries the same ticket.
- Record ignores double taps while a request is open.

**Legacy routes.** `POST /make` (mode commit) and `/make/commit` post FIFO lots with no lot
confirmation. Read-only production check 2026-11-02: 7 such makes after cutover (shared key,
last 2026-10-30); all makes since 2026-11-02 are tickets. So they stay **allowed** by default;
each post is logged and its response says `lot_verification: "none"` with
`LEGACY_MAKE_UNVERIFIED_LOTS`. `LEGACY_MAKE_COMMIT=block` (Railway variable) refuses them with
`LEGACY_MAKE_DISABLED` (preview still works) — set it only after Arturo records every Make on
the Make screen and the Custom GPT no longer calls `make`. Owner decision; not done here.

**Unchanged:** extra-kosher hold (owner approval before anything posts), oat bran → oats
swap, fixed reasons, 10% review, shift summary flags, ticket re-check under lock, payload hash,
append-only evidence. No mixed-bin / unresolved-consumption model.

**Deploy order (needs Michael's approval per step):** backup → apply 090 on production →
deploy code → verify `/health`, entry.html serves `make-formula.js?v=5`, one staging Make.

### 7.1 Staging acceptance fixes (ChatGPT Work report, 2026-11-02; Makes MK-261102-003…006, XR-261102-001)

- **Negative pounds:** a negative or unreadable amount is kept exactly as typed or pasted, shown in red next to the lot ("…lb is not allowed: pounds cannot be negative"), and the lot row, the chosen lot and its supplier lot stay. It is never turned into 0 or "Not used". Check and Record stay off until it is fixed ("Record is off: 1 amount is negative"). The server already refuses any lot amount ≤ 0 (`quantity_lb > 0`, 422, no ticket, no ledger line); now covered by tests.
- **0 lb vs Not used:** typing 0 on every lot no longer means "Not used". FL asks for the pounds or the explicit **Not used (0 lb)** button; only that button (or a draft FL already marked not used) shows the reason list.
- **Lots at zero or negative book stock:** a lot already in the draft keeps FL lot, supplier lot, received date and its **signed book balance** ("on books -10 lb (none available)"); the draft sends it as `unstocked_lots`. The pick list shows "on books N lb" instead of a bare number. No lot is offered or switched to automatically; shortage posting unchanged.
- **Wording:** "1 pan" / "1 bandeja"; "Falta tocar 1 lote".
- **Owner hold:** after "waiting for owner approval", Record stays off but **Edit entry** still works; re-preparing the same entry says it is already waiting / already RECORDED with the receipt.
- **Retry scenarios now tested** (pytest + Playwright with DB counts): (A) lost answer after a successful commit → NOT CONFIRMED → retry returns the same receipt; a new draft of the same entry is refused with that receipt; exactly one Make, one deduction. (B) extra-kosher held → Michael approves → retrying the original ticket replays the receipt; re-preparing the entry and a second draft opened before approval are both refused with the receipt; no new transaction, line, finished lot or ingredient deduction.

### 7.2 Final acceptance fixes (ChatGPT Work, staging deployment 386c9693)

- **Negative pounds:** while any amount is negative or unreadable, **both** Record and **Check lots and pounds** are disabled (and a forced click sends nothing). The row reads "Lot confirmed — pounds "-25" must be fixed", never "Confirmed — 0 lb"; the raw value, the lot and every tap stay. Fixing the value re-enables Check; Record waits for the re-check.
- **Shortage wording** (FL warning and the card hint, EN/ES) states book balance before, pounds used, the NEW shortage and the book balance after. Example: lot at -25 lb, Make uses 25 lb → "is already at -25 lb on the books (none available); this entry uses 25 lb, adding 25 lb of shortage. Book balance after: -50 lb." Posting rules unchanged (the flag records the new 25 lb). The warning is shared by Make and Pack.
- **Release candidate check:** `a787587` → `08e5bd2` changed only `CHANGE_LOG.md`. The tested staging build (`037cdad`) differed from the PR only by the staging-only test sign-in (`staging_test_signin.py`, its two hooks in `main.py`/`key_sessions.py`, `dashboard/key-session.js` and its `?v=7` tags).

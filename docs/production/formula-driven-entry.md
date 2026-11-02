# Formula-Driven Production Entry — spec (draft)

**Status:** draft for owner review, 2026-11-01. Docs only: no code, no migration, no deploy.
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
2. **Enter number of pans** (whole number ≥ 1). One pan = one FL batch = `default_batch_lb` (see decision D1).
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

- Suggest the **oldest lot that alone has enough stock** for the line's pounds (by `COALESCE(received_at, created_at), id`).
- If no single lot has enough, suggest the oldest lot and add a second row for the remainder from the next-oldest lot, and so on, the same as `choose_inputs` today. (Decision D2: this is a change from today's pure oldest-first spill.)
- If total stock is short, pre-fill as above and show the shortage (3.6).
- A suggestion is only a suggestion: `confirmed:false` until the worker taps and types the code (2026-10-06 rule R4).
- Pooled opening-stock lots are shown by FL lot code only, never as a supplier lot.

### 3.4 Editing pounds and splitting lots

- Any line's pounds may be edited. The split rows must add up to the pounds the worker entered for that ingredient; FL shows the running total.
- Each lot row carries its own pounds. FL posts exactly those pounds per lot.
- Removing a split row puts its pounds back on the row above.

### 3.5 Zero or substituted ingredient: reason required

- Setting a line to **0** or **substituting** a product requires a reason picked from a **fixed list** (no free text for the reason; optional note).
- Proposed list (decision D3): `out_of_stock`, `quality_hold`, `customer_request`, `equivalent_spec` (e.g. 1,000 CT ↔ 4,000 CT chips), `owner_instruction`, `other` (note required).
- `oat_bran_auto_substitute` stays a system reason, not on the worker's list.
- The **master recipe is never changed** by an entry. Substitutions and zeroes are stored on this run only (`transaction_substitutions` / excluded lines).

### 3.6 Oat bran

- Unchanged rule: when oat bran is short, FL auto-substitutes rolled oats **pound-for-pound for the missing part**, recorded with `oat_bran_auto_substitute`.
- Today this applies only to SS products 90016 / 90011 / 90013. Decision D4: keep it SS-only, or apply to every recipe containing oat bran 11031.
- The swapped oats line still needs its lot confirmed like any other line.

### 3.7 Short stock never blocks

- If confirmed lots do not hold enough, the entry still records. FL posts the pounds, the lot goes negative, and a SHORTAGE flag opens for Arturo (existing R3 behaviour).
- The draft says plainly: "Only X lb of Y on hand — this will be recorded as a shortage."
- Only blocker kept: the ingredient has **no lot at all** in FL (nothing to record the shortage against), as today.

### 3.8 The 10% warning

- For each ingredient, compare the pounds entered with the standard (recipe × pans).
- **Warn only when the difference is more than 10%** of the standard, in either direction. At 10% or less: no warning.
- A zeroed line (−100%) and a substituted line count as "more than 10%" and already carry a reason; they show in the warning list without a second acknowledgement.
- A warning does not block; it needs one tap to acknowledge (existing `acknowledged_warnings` mechanism) and is stored on the record.
- Decision D5: should the warning compare per ingredient only, or also flag total input weight off by more than 10%?

### 3.9 What is saved

- **One record per run** (one make transaction, one output lot), never one per pan. Pans are stored on the record as the count entered.
- **Recipe version used**: the record stores which recipe version it was made against (new, 4.2), plus the standard pounds per line, so later recipe changes never rewrite what this run "should" have used.
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
3. **"Oldest lot with enough stock" suggestion** (small change to `choose_inputs`, behind a flag so pack/assistant keep today's behaviour unless decided otherwise).
4. **Fixed substitution/zero reason list**: a lookup table (or `correction_reasons` rows with `applies_to` including `make`) + validation; the current free-text reason input becomes a select.
5. **10% deviation warning** with `requires_ack`.
6. **Recipe versions**: new migration, e.g. `recipe_versions (id, product_id, version, created_at, created_by, note)` + `recipe_version_lines`, or a `version`/`valid_from` on `batch_formulas`, and a `recipe_version_id` + standard-lb snapshot on the make record. Admin recipe edits (`/admin/bom/...`) must create a new version instead of editing rows in place. **Production migration: needs Michael's approval, backup first.**
7. **Pans label and count stored on the record** (the "batches" field relabelled "Pans", with the count kept explicit).
8. **New Make form UI**: ingredient cards, tap-to-confirm lot pick-list, pounds fields, add-lot, reason select, running totals, bilingual EN/ES, phone-width layout.
9. **Tests**: split lots, edited pounds, 10% boundary (exactly 10% = no warning), zero/substitute needs reason, oat bran, shortage posts, re-check-at-save failures, recipe-version snapshot, no posting before commit.

### 4.3 Size estimate

**Medium-to-large.** About 60% of the hard parts (ticket lifecycle, lot confirmation, shortage policy, oat bran, posting exact lots) already exist. The new work is concentrated in:
- the Make form UI rewrite (largest single piece),
- per-lot/pound request + validation changes in `ticket_actions` / `write_tickets` / `main`,
- recipe versioning (migration + admin edit path + snapshot), which is what pushes it from medium toward large.

Suggested phasing:
- **Phase A (medium):** pans + editable pounds + split lots + tap-to-confirm pick-list + reason list + 10% warning. No migration if the reason list reuses `correction_reasons`.
- **Phase B (small-medium, migration):** recipe versions and the snapshot on each record.

Staging first; production only with Michael's per-step approval, backup and verification.

---

## 5. Decisions for the owner

| # | Question | Default proposed |
|---|---|---|
| D1 | Is **1 pan = 1 FL batch** (`default_batch_lb`) for every product, granola included? Today only coconut (360 lb) is confirmed as a pan. | Yes; check each granola batch weight before launch |
| D2 | Suggest the **oldest lot that alone has enough** (new), or keep today's **oldest first, then the next** (spill)? | Oldest with enough; spill only if none has enough |
| D3 | Approve the **fixed reason list** for 0/substituted ingredients | List in 3.5 |
| D4 | Oat bran → rolled oats: **SS products only** (today) or every recipe with oat bran? | Keep SS-only |
| D5 | 10% warning: per ingredient only, or also on total input weight? | Per ingredient only |
| D6 | Should the 10% warning go on the owner's weekly view like corrections over 10% (R2)? | Yes, list it there |
| D7 | Can the worker change the pounds of a lot after confirming it, or must they re-confirm? | Re-confirm only if the lot changes; pounds may change freely |

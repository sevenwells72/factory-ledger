# Cross-review: f8ba30d — Formula-driven Make entry, Phase A

- **Branch:** `feat/formula-driven-entry` (worktree `~/dev/fl-formula-entry-wt`), parent `bd1dc4f`, base `origin/main` `87f500f`
- **Built by:** Codex. **Reviewed by:** Claude (independent; did not build it), 2026-11-02
- **Spec:** `docs/production/formula-driven-entry.md` (D1–D7, Phase A)
- **Scope of this review:** read-only. No push, merge, deploy, or staging/production database access. Only local throwaway databases were used (`fl_review_f8ba_off`, `fl_review_f8ba_on`, `fl_review_f8ba_probe`, `fl_formula_browser_review`).

## Verdict: REQUEST CHANGES

The core is sound. FL posts exactly the bags and pounds the worker confirmed, splits trace forwards and backwards, and no bag gets through without a typed code. Re-checks at Record work, and the master recipe is never written. All suites pass.

Two things need fixing before production, and one decision needs Blubber:

1. **(High)** On an oat-bran recipe, a worker who enters the bran they really used, when that is more than FL's book stock, gets a draft that says it can be recorded. Record then fails every time.
2. **(Medium)** For oat-bran recipes, the "Recipe" pounds that are shown, saved and used for the 10% check are not recipe × pans.
3. **(Medium, decision)** The over-10% facts are saved but appear on no screen. The spec says to show them where correction highlights already appear.

None of these can corrupt inventory. Finding 1 can push a worker to record a different bran/oats split from the one they used, which goes against the "accurate inventory going forward" principle.

## Findings

### 1. HIGH — Oat bran: editing bran above book stock gives a recordable draft that can never be recorded

- **Where:** `ticket_actions.py:311-317` (the lock-time refusal "The reviewed oat bran is no longer available") together with `formula_entry.py:83-88`, which only guards the 0 lb case for an auto-swapped bran line.
- **Reproduced** with a local probe. Recipe per pan: 30 lb bran (11031), 50 lb oats (11032), 20 lb honey. The bran bag shows 10 lb on hand. The worker enters bran 30 lb from that bag, oats 50 lb and honey 20 lb, then confirms every bag:
  - Prepare: `can_commit: true`, no blockers; warnings `WILL_CREATE_SHORTAGE` and `INGREDIENT_OFF_RECIPE`.
  - Record: `409 TICKET_STALE / INPUT_PLAN_CHANGED`: "The reviewed oat bran is no longer available; prepare again to use extra oats."
  - Preparing again produces the same draft and the same refusal, so there is no way forward. The only way to record is to accept FL's split (bran 10, oats 70), which is not what was used.
- **Why it matters:** the screen invites the worker to type the real bran pounds, and then refuses them only at the last step. That is a dead end on the floor, and it pushes the worker to record a split that didn't happen. The bran row is the only ingredient where short stock blocks instead of posting and flagging (R3, spec 3.7).
- **Fix (needs a one-line owner choice):** do one of these:
  - (a) Make it a **prepare-time blocker** with a plain message, e.g. "FL shows only 10 lb oat bran. Count or receive the bran first, or use oats for the rest." The draft then never says it can be recorded.
  - (b) When the bran pounds come from the worker's own `ingredient_lines` entry, **skip the lock-time refusal** and let the normal shortage flag (R3) handle it.

  Option (b) matches "short stock never blocks". Option (a) keeps the D4 swap strict. Add a test for whichever is chosen. With zero bran in stock the behaviour is already consistent: the bran card is absent and a bran line is refused at prepare (`UNUSED_INGREDIENT_LINE`), so that case doesn't dead-end.

### 2. MEDIUM — For oat-bran recipes, the saved "standard" is the stock-adjusted swap, not recipe × pans

- **Where:** `formula_entry.py:52`: `standard = dict(requirements)` is taken after `oat_bran.apply_preview` has already moved the missing bran onto oats (`ticket_actions.py:145-153`). The same value is saved as `specification['standard_requirements']` (`ticket_actions.py:171`) and drives `review()` and `warnings()`.
- **Seen in the probe:** the cards say "Recipe: 10 lb" for bran and "Recipe: 70 lb" for oats, but the recipe is 30 and 50. The worker who used exactly the recipe is flagged +200% and −28.6% for the owner, and the weekly-review rows carry `standard_lb` 10 and 70.
- **Spec:** 3.2 (standard pounds = recipe qty × pans), 3.8 (compare with recipe × pans), 3.9 (Phase A stores the standard pounds as read at save time).
- **Fix:** keep the recipe × pans figure (`draft['original_ingredients']`) as `standard_lb`, and show the automatic swap separately, e.g. "recipe 30 lb · FL swap: 10 lb bran + 20 lb extra oats". Decide on purpose whether the 10% check compares against the recipe or against the swap-adjusted plan. Comparing against the swap avoids flagging every short-bran run, but the saved record should still carry the true recipe pounds. Manual substitutions are fine as built: the substitute takes the original's pounds.

### 3. MEDIUM (owner decision) — Over-10% facts (D6) are saved but appear nowhere

- **Where:** `ticket_actions.py:403-406` writes `ingredient_review` into the saved ticket response only. Nothing in the backend or dashboard reads it. A grep for `ingredient_review` and `INGREDIENT_OFF_RECIPE` finds only the writer and the draft warning.
- **Spec 4.2 item 5:** "Build-time check: confirm where that weekly view is rendered today; if it has no screen yet, list these rows wherever the correction highlights appear." Correction highlights appear on the **shift summary**: `shift_summary.py:493-512` (`LARGE_CORRECTION`) and `:518-520` ("highlighted on the weekly view"). Make rows there get no flag for an off-recipe ingredient. Spec §6 as amended by Codex says "No review screen in this PR", which departs from the decided spec without a recorded owner OK.
- **Fix:** add an `INGREDIENT_OFF_RECIPE` warning flag to make rows in `shift_summary.py`, built from `response.ingredient_review`, with product, ingredient, standard lb, used lb, %, reason and who recorded it. If Blubber prefers to defer it, record that decision in spec §5 instead. Until one of these happens, D6 is not delivered. Not-used and swap reasons are queryable in `transaction_substitutions`, but off-recipe pounds are not.

### 4. LOW — Bag lots are locked in screen order before the ID-ordered lock pass

- **Where:** `formula_entry.py:76`: `actions.lot(api, cur, bag['lot_id'], pid, lock)` takes `FOR UPDATE` at Record, in `ingredient_lines` order, before `ticket_actions.py:283-286` locks every plan lot in `lot_id` order.
- **Effect:** two Records at the same moment that share two lots listed in opposite orders can deadlock. Postgres aborts one: the worker sees an error and nothing posts, so this is not a data problem. The same pattern already exists in `lot_confirmation.prepare_substitutions` (`lot_confirmation.py:262`).
- **Fix:** call `actions.lot(..., lock=False)` in `formula.prepare`. The identity and merged checks still run, and the ID-ordered loop below takes the locks.

### 5. LOW — Warning list omits zeroed and swapped lines

- **Where:** `formula_entry.py:179-181`: `warnings()` lists only `off_recipe`.
- **Spec 3.8:** zeroed and substituted lines "show in the warning list without a second acknowledgement". They do show on their cards and in `ingredient_review`, so this is cosmetic. Either add them as non-ack lines or adjust the spec wording.

### 6. INFO — Before staging: list which recipes the D4 widening newly affects

- `oat_bran_substitution.eligible()` now matches any batch recipe that deducts 11031 (`oat_bran_substitution.py:17-24`), in every make path. That includes direct `/make`, tickets, and the FL Assistant through tickets: confirmed at `main.py:8895`, `:8966`, `:9145` and `ticket_actions.py:145`, `:172`, `:311`.
- In the repo only `migrations/081` names 11031, which is the SS restore. Run a **read-only** query on staging (and later production) for `batch_formulas` rows with 11031 and `exclude_from_inventory = false`, so Blubber sees exactly which products start swapping.
- A ticket prepared just before deploy for a newly eligible product will fail at Record with `INPUT_PLAN_CHANGED`, because the specification now carries `auto_substitutions`. That is harmless: the worker prepares again.

### 7. NIT — Dead branch in `ledger-ticket.js`

- `dashboard/ledger-ticket.js:44-66`: the ambiguous last-4 handler that writes `payload.ingredient_lots` can no longer run for Make. Every Make draft now takes the formula path, and the lot map is empty when `formula` is set. It still serves Pack, where it only touches `substitutions` and `ingredient_lots` when those exist. It can stay, but a comment would help the next reader.

## Checklist (items 1–10 from the request)

| # | Check | Result |
|---|---|---|
| 1 | Traceability: exact bags and lb, splits, 0 lb, no drop or double deduction, trace works for a split | **Pass.** `_post_prepared_inputs` posts exactly the frozen plan per ingredient (`main.py:8719`), and `required_lb` replaces recipe × batches (`main.py:9042-9049`). Probe: split oats 100 + 60, honey 40 → `transaction_lines` and `ingredient_lot_consumption` match exactly. `GET /trace/batch/{output}` lists both oats bags, and `GET /trace/ingredient/{bag}` returns the output lot for each bag. A 0 lb ingredient posts nothing and gets a `transaction_substitutions` row with a NULL substitute and its reason. Oat bran with oats already in the recipe: oats deducted once (70 lb), with no duplicate row (`oat_bran_substitution.formula` merges, `substituted_formula` combines). The totals check (`ticket_actions.py:287-288`) refuses any plan that doesn't add up to the entered pounds. |
| 2 | Commit re-check | **Pass.** At Record, lots are re-read `FOR UPDATE`, a merged lot is refused (`LOT_MERGED`, nothing posted), and a recipe change is refused because `standard_requirements` sits in the signed specification. Probe: 45 lb taken from a planned bag after prepare → Record posts with `state_changed: true` and a 45 lb SHORTAGE flag. That is R3 / spec 3.9 ("a balance drop only changes the shortage amount"), not a blind post. |
| 3 | D7: bag change needs a code, pounds change doesn't | **Pass.** Confirmations are keyed by `lot_id` (`lot_confirmation.validate_plan`), so new pounds keep the code and a new bag needs its own. Probe: a swapped-in bag without a code → Record `422 LOT_NOT_CONFIRMED`; late evidence with the wrong code at Record → `422 LOT_CONFIRMATION_MISMATCH`; nothing posted either time. UI: `lot.onchange` clears the code (`make-formula.js:47`), and any edit disables Record until a fresh draft. |
| 4 | Reasons; Other needs a note; recipe never modified | **Pass.** `Literal[formula.CODES]` refuses free text (422). A blank or Other-without-note reason is a draft blocker and is refused again at Record. Swaps use the same list. The diff never writes to `batch_formulas`, and a test asserts the recipe is unchanged. |
| 5 | >10% needs a tick and is saved for review; exactly 10% passes | **Pass for the rule** (176 and 144 lb on a 160 lb standard pass; 176.5 and 143 are flagged; the tick is enforced at Record with `WARNING_NOT_ACKNOWLEDGED`). **Gaps:** the "standard" for oat-bran recipes (finding 2) and visibility (finding 3). |
| 6 | Oat bran widened (D4, reverses row 208) in direct, ticket and Assistant paths | **Pass**, with findings 1, 2 and 6. Changelog row 243 records the reversal of row 208. |
| 7 | Old paths and Assistant unchanged except via `formula_entry` | **Pass.** Non-formula payloads have `formula_entry`/`ingredient_lines` popped, so their hashes are unchanged (`write_tickets.py:811-813`). The Assistant tool schema excludes both fields (`assistant_tools.py:30-31`). Draft and warnings are unchanged without the flag. The only shared behaviour change is D4. |
| 8 | Removing "Choose a different ingredient lot" | **Pass.** No other page, test, staff doc or GPT text depends on it. `ingredient_lots` remains accepted by the API (tests and the Assistant still use it). The old picker and the new bag list both show only in-stock lots, so nothing is lost. See nit 7. |
| 9 | Standing rules | **Pass.** `openapi-gpt-v3.yaml` is untouched with 30 operations. `origin/feat/planner-v2` is untouched (last commit 2026-06-09). No migration. Cache-bust versions are bumped on every page loading a changed asset: `ledger-forms.css` v13 on all 14 pages, `ledger-ui.js` v21 on 6, `ledger-ticket.js` v11 on 4, `entry.js` v11, and `make-formula.js` v1 on `entry.html`, its only loader. All three changelogs are updated: `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md` row 243, and the global log, written at the resolved iCloud path. `~/change-log.md` is still a symlink. |
| 10 | Tests | See below. |

## Test runs (this review, fresh local databases)

- **Python, key sessions off** (`KEY_SESSIONS_ENABLED=0`, new DB `fl_review_f8ba_off` from `tests/schema/schema.sql`): **7,267 passed**, 0 failed.
- **Python, key sessions on** (`KEY_SESSIONS_ENABLED=1`, new DB `fl_review_f8ba_on`): **7,267 passed**, 0 failed. The two modes ran one after the other.
- **Node** (`node --test tests/*.js`): **611/611 passed**.
- **Playwright** `tests/visual/run-formula-entry.mjs` against a local fixture server (`d12_fixture.py`, `FORMULA_BROWSER=1`, DB `fl_formula_browser_review`, 127.0.0.1 only, external requests aborted): **pass**, `{"ok":true,"receipt":"MK-261102-001"}`. Fresh screenshots in `.venv-test/formula-browser/` (390 px EN/ES, desktop, receipt). `failure.png` there is left over from Codex's 2026-11-01 run, not this one. All four throwaway DBs were dropped afterwards.
- **Reviewer probes** (temporary file, deleted after use; separate DB `fl_review_f8ba_probe`): split trace forwards and backwards ✓; stock drop between prepare and Record → posts and flags ✓; unconfirmed bag and wrong late code refused ✓; oat bran default plan with oats in the recipe, no double deduction ✓; **bran above book stock → Record dead-ends ✗ (finding 1)**.

**Suggested tests to add with the fixes:** formula entry on an oat-bran recipe (default plan and edited bran); split ingredient forward and backward trace; balance drop between prepare and Record.

## Plain-English summary for Blubber

**Is it safe to put on staging? Yes.**

- FL records exactly the bags and pounds your people confirm, including splits, and an ingredient marked "not used" keeps its reason.
- No bag can be recorded without typing its code.
- Nothing changes inventory until Record, the master recipe is never touched, and recalls trace both ways for split bags.
- Every automated test passes, and staging is the place to find what's left.

**Not ready for production yet.** Two things to fix first:

- **The oat bran dead end.** If a worker types more oat bran than FL thinks is on the shelf, the screen lets them finish but Record refuses every time. Pick one fix: (a) FL stops them earlier with a clear message ("count or receive the bran first"), or (b) FL records it and flags the shortage, like every other ingredient.
- **Oat bran "recipe" numbers.** On oat-bran products the card shows the recipe after FL's automatic oats swap, not the real recipe. That makes the 10% flag and the saved record misleading for those products.

**One decision for you.** You asked for over-10% differences on your weekly review. Right now they are saved but don't appear on any screen. Codex can add them to the shift summary next to the large-correction flags, which is what the spec said, or you can say "later".

Before staging, also run a read-only check to see which products besides SS #1/#2/#3 contain oat bran. With this change, those products will start swapping in rolled oats when bran runs short.

---

# Re-review: 130c15a — fixes for this review

- **Commit:** `130c15a` on `feat/formula-driven-entry` (parent `f8ba30d`), worktree `~/dev/fl-formula-entry-wt`. Reviewed 2026-11-02.
- **Independence caveat:** 130c15a was written in the same Claude session that writes this re-review. To compensate, a fresh agent that had not seen the fix reasoning also read the diff cold. Its findings are folded in below and marked "(independent read)".
- **Scope:** read-only. No push, merge, deploy, or staging/production access in this re-review. Only local throwaway DBs were used (`fl_rr130_off`, `fl_rr130_on`, `fl_rr130_probe`, `fl_formula_browser_rr130`), all dropped afterwards. The temporary probe file was deleted and never committed.

## Verdict: APPROVE (for staging)

All seven findings are fixed. The probes that dead-ended before now post. Traceability, the Record re-check and the bag-confirmation rule still hold. One new latent issue (N1) and two low items are listed below. None blocks staging. N1 should be fixed before production, and it is a small change.

## The seven findings

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1 | Oat bran above book stock dead-ends at Record | **Fixed** | Probe (recipe/pan 30 bran, 50 oats, 20 honey; bran bag 10 lb; entered 30/50/20): Prepare `can_commit: true` with `WILL_CREATE_SHORTAGE` and no `INGREDIENT_OFF_RECIPE`. Record **200**; posts bran 30, oats 50, honey 20; SHORTAGE 20 lb on the bran bag; no swap row. Preparing again gives the same draft. Also: a screen-shaped draft (all lines sent) where bran drops to 0 after Prepare posts and flags 10 lb short. The lock-time refusal is skipped only for worker-entered bran lines (`ticket_actions.py` near the `entered =` check). |
| 2 | "Recipe"/standard is the stock-adjusted swap | **Fixed** | Same recipe, default draft: bran card standard 30 / entered 10 / checked 30, oats 50 / 70 / 50, no warning. The swap row posted is 20 lb, and the transaction note reads "Used 10 lb oat bran and 20 lb extra oats…". SS #1 at 2 pans with bran stock 0, 10 and 100: standard 56 / 370, no `INGREDIENT_OFF_RECIPE`. With 100 lb bran there are no warnings at all, and nothing is off-recipe on the receipt. The signed `standard_requirements` holds recipe × pans. |
| 3 | D6 facts on no screen | **Fixed** | Shift-summary make rows carry `INGREDIENT_OFF_RECIPE` (warn) and `INGREDIENT_NOT_USED` / `_SUBSTITUTED` / `_SWAP` (info), each with recipe lb, used lb, %, reason and who recorded it. Covered by new tests `test_finding5_and_3_*`. |
| 4 | Bag lots locked in screen order | **Fixed** | `formula_entry.prepare` calls `actions.lot` without a lock. The first locks taken at Record are in lot-ID order (`test_finding4_*` spies the calls). The same pattern still exists in `lot_confirmation.prepare_substitutions`, which predates this branch and is unchanged. |
| 5 | Warning list omits 0 lb / swapped lines | **Fixed** | A non-ack `INGREDIENT_CHANGED` warning lists not-used, substituted and swap lines with their reason, in English and Spanish. |
| 6 | Which recipes D4 affects | **Fixed** | Spec §3.6 lists the read-only staging and production result: SS #1/#2/#3 (116/114/118) only. |
| 7 | Dead Make lot-picker in `ledger-ticket.js` | **Fixed** | Branch removed. Its 4 tests were replaced by one asserting no picker and no `ingredient_lots`. No remaining references. |

## Regression checks

- **Traceability:** split bags 100 + 60 → `GET /trace/batch/{output}` lists both oats bags, and `GET /trace/ingredient/{bag}` returns the output lot for each.
  - Oats entered beyond the swap, on a recipe without oats (bran 10, oats 30): all 30 lb post. The new append in `main._make_commit_core` covers ingredients the reshaped recipe dropped.
  - Bran entered from a zero-stock bag with a full-swap proposal: posts bran 5, oats 75, honey 20 exactly.
  - (Independent read) Nothing posts twice or to the wrong ingredient. `oat_bran.record`'s `consumed[0]` is safe because a swap > 0 implies oats were consumed.
- **Record re-check:** a stock drop between Prepare and Record posts and flags (45 lb SHORTAGE). A lot merged after Prepare is refused at Record and nothing posts.
  - (Independent read) The signed specification is stable between Prepare and Record. The specification keeps FL's proposal; the swap that actually posts is outside it.
- **Bag-confirmation rule (D7):** an unconfirmed new bag is refused at Record (`422 LOT_NOT_CONFIRMED`). A wrong late code is refused (`422 LOT_CONFIRMATION_MISMATCH`), and nothing posts. A pounds-only change keeps the confirmations (`can_commit: true`).

## New findings

**N1. LOW–MEDIUM (latent today): a reason is demanded for a 0 lb rolled-oats card whose recipe is 0 lb.** Found by the independent read; confirmed against the code.
- **Where:** `formula_entry.prepare` sends every 0 lb line to `_reason_blocker`, including the oats line when `standard[oats]` is 0. That happens when the recipe has bran but no oats, or when oats were manually swapped out while bran is short. `make-formula.js` shows the reason box in the same case. The reason is then never saved, because `review()` and `record()` skip rows with a standard of 0.
- **Effect:** the worker uses the real bran, leaves oats at 0, and is asked to "Choose a reason for Rolled oats". Picking any reason lets them record, so this is not a dead end.
- **When it happens:** none of today's three oat-bran products hit it on a normal run, because they all contain oats. It needs a manual oats swap while bran is short.
- **Fix:** skip the reason blocker, and hide the reason box, when the card's standard is 0 lb. Consider hiding a 0-standard oats card that has no bags. Fix before production.

**N2. LOW (as built in f8ba30d, not changed here): manual substitutes over 10% never need the tick.**
- A `substituted` row takes precedence over `off_recipe`, so a swapped-in ingredient used at 2.5× its original's pounds is only an info line, with no % shown.
- This matches "manual substitutions are fine as built" above. Blubber may still want the % on the shift summary later.

**N3. INFO (design): bran + oats are compared together only when FL proposed a swap at Prepare.**
- When bran stock is sufficient and the worker uses bran 20 + oats 60 against a 30/50 recipe, both lines are flagged and need the tick.
- That is consistent with the spec, since no swap happened.

## Edge case: callers without a bran line (FL Assistant, direct API, formula entry with no lines)

**Acceptable for staging and production; no fix required now.**
- **Reproduced:** a non-formula ticket was prepared with bran 10 lb. Bran dropped to 4 lb before Record, and Record returned `409 TICKET_STALE / INPUT_PLAN_CHANGED` ("prepare again to use extra oats").
- **Not a dead end:** preparing again gives a fresh, recordable draft (bran 4 + 26 lb extra oats). This only happens when stock changes between Prepare and Record. The f8ba30d dead end was different: it repeated on every prepare.
- **Make screen not affected:** the screen always sends every card, the bran card included, once codes are typed. A screen-shaped draft with a bran drop posts and flags (probe above).
- **Fix later** if it shows up for FL Assistant users on the floor. The fix would be to treat the Assistant's frozen bran pounds as entered, then post and flag.

## Test runs (re-review, fresh local DBs)

- **Python, key sessions off** (`fl_rr130_off`, `pg_trgm` + `tests/schema/schema.sql` with `ON_ERROR_STOP`): **7,274 passed**, 0 failed, 0 skipped (JUnit count).
- **Python, key sessions on** (`fl_rr130_on`, run after the off run): **7,274 passed**, 0 failed, 0 skipped.
- **Node** (`node --test tests/*.js`): **609/609**.
- **Playwright** `tests/visual/run-formula-entry.mjs` (local fixture server on 127.0.0.1, external requests aborted): **pass**, receipt MK-261102-001.
- **Re-review probes** (temporary file, 14 cases): **13 pass.** The one failure is a formula-entry request that sent no ingredient lines while bran dropped between Prepare and Record. It got `409 INPUT_PLAN_CHANGED`. That is the API-only edge case above, not a screen path; the screen-shaped version of the same scenario passes.
- **Standing rules re-checked:** `openapi-gpt-v3.yaml` is unchanged vs `origin/main` (30 operations). `origin/feat/planner-v2` is untouched (last commit 2026-06-09). No migration. Cache-bust is bumped on every page loading a changed asset (`ledger-ticket.js` v12 on 4 pages, `make-formula.js` v2 on `entry.html`).

## Plain-English answer for Blubber

**OK to push and put on staging? Yes.**

- The oat bran dead end is gone. If someone uses more bran than FL shows, it records what they used and flags the shortage.
- Oat bran products now show the real recipe. Using exactly the recipe gives no warning, and a rolled-oats swap is labelled as a swap.
- Over-10%, not-used and swapped ingredients show on the shift summary with the reason and who recorded it.
- Bag codes, the double-check at Record and recall tracing all still work. Every automated test passes.

One small fix should land before production. In a rare case FL asks for a reason on a rolled-oats card the recipe doesn't use. It happens when a recipe has bran but no oats, or someone manually swaps the oats out while bran is short. None of today's products hits it on a normal run.

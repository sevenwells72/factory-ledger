# Formula Make (PR #173): staging test follow-ups, 2026-11-02

Owner staging test of PR #173 (FastAPI-staging deployment `21290fdd`) passed product search,
autocomplete, sign-in, recipe math, lot confirmation and attribution. This note covers what
that test left open. Branch `feat/make-mobile-ux` (on top of PR #173 `3cd6574`) holds the code
changes; nothing here was deployed, merged or written to either database. Staging facts below
come from read-only queries (`BEGIN TRANSACTION READ ONLY`).

## 1. Staging coconut: why the normal batch could not be recorded

**Facts (staging, posted lines only):**

- Batch Classic Granola #9 (product 107, code 90002) uses **Coconut Macaroon Desiccated**
  (product 13, code 11013), 12.5 lb per pan.
- Every product-13 lot is at **0 lb**: `OPEN261014-P616` (opening stock 782.5 lb, used by
  October staging makes) plus five older STG test lots. No receipt has ever been posted for
  product 13 on staging.
- Other coconut items do have stock (e.g. Coconut Medium 11014, Flake 11012), but they are
  different products and are not in the #9 recipe.
- So the exception path was the only way to record #9: MK-261102-001 posted coconut
  "not used — out of stock" (Arturo, note "PR173 staging exception test").

**Proposal — a clearly labeled test lot, through the app (needs your approval):**

Record one ordinary **Receive** on the staging Record-an-entry screen, signed in as
**Michael (Owner)** with the TEST MODE picker:

| Field | Value |
|---|---|
| Product | Coconut Macaroon Desiccated (11013) |
| Supplier | STAGING Test Supplier |
| Cases × lb per case | 2 × 25 lb = 50 lb (enough for 2 pans = 25 lb, plus one repeat) |
| BOL / delivery reference | `STAGING-TEST-261102` |
| Supplier lot | `STAGING TEST — NOT A REAL LOT` |
| FL lot code | `STG-TEST-COCONUT-261102` |
| Note | `STAGING TEST LOT for PR #173 normal-batch test. Not real inventory.` |

Why this way: it goes through the same receive checks as real stock, is attributed to a
named person, gets a receipt (`RCV-…`), and the lot name says it is a test everywhere it
appears (bag dropdown, receipt, shift summary). No SQL, no edits to existing transactions.
Because every other product-13 lot is empty, the oldest-first plan will pick this lot.
Staging only; production is not touched. Afterwards it can be used up by the test or
zeroed with a normal staging adjustment carrying a reason.

## 2. Mobile Make screen (390 × 844): what changed

Measured with `tests/visual/run-formula-entry.mjs` (6-ingredient fixture): the first draft
went from **6,094 px to 5,005 px** tall (−18%). All changes are in the browser; FL's checks,
blockers, required ticks, codes and reasons are unchanged.

| Reported problem | Change |
|---|---|
| Repeated blockers | On a formula Make draft, the per-bag "Confirm … lot …: type last four characters…" blockers (one red box per bag) become **one** line: "5 bags still need the code from their label — type it on each bag marked Not confirmed below." Each bag still shows its own state. Any other blocker is listed as before. Other screens are unchanged. |
| Repeated blockers / warnings | Warnings FL marks `requires_ack: false` (shortage, "not used or swapped") show as notes, **not** checkboxes. Warnings that need a tick (e.g. more than 10% off recipe) keep their checkbox; Record still sends only those. FL's own shortage note replaces the duplicate client "Only N lb on hand" line when FL sent one. |
| Truncated dropdowns | The bag choice takes the full width (lot code, date and lb are readable). The code field sits next to the pounds; the confirmation method has its own full-width row with shorter names: "Last 4 (with hyphen)", "Full code", "Pallet moved (full code)" (Spanish too). |
| Excessive scrolling | Bag state and Remove share one line; "Add another bag" and "Not used" share one line; tighter padding on phones. |
| Extra review step after changing quantities | FL must re-check edited pounds and codes before Record (the ticket is bound to the exact payload), so the step stays. It is now cheaper: the **Check bags and pounds** button lives in a **sticky bar at the bottom** next to Record, appears only after an edit (with "You changed bags, pounds or codes. Check them first, then Record."), and the draft **stays on screen** (dimmed) while FL checks, then returns to the same ingredient instead of jumping to the top. A failed check keeps the edits and Record stays off. |

Not changed on purpose: Record is never automatic after a re-check (new warnings must be
seen and ticked); every bag still needs its typed code; the reason list and 10% rule are as
specified.

Cache-bust: `ledger-forms.css` v14, `ledger-ticket.js` v13, `make-formula.js` v4, `entry.js` v13.

## 3. Search failures and timeouts

New automated coverage (no behavior change except the 5xx wording below):

- **Browser** (`tests/test_pilot_entry_ui.js`): every entry-screen picker (Make product,
  Receive product, Receive supplier, Adjust lot) is capped at 25 s and ends in its own
  field with Try again, never "Searching…" and never a page-level error; lot-search failure
  then retry; server message in Spanish; autocomplete failure while typing; a late answer
  after the time limit is ignored; typing during a search abandons it quietly.
- **Server** (`tests/test_entry_search_failures.py`): a real PostgreSQL statement timeout and
  a dropped connection inside `/entry/make-products`, `/entry/lots` and `/resolve` return a
  5xx JSON failure (never a partial list), and the next search works.

**Finding:** for the two GET searches, the app-wide error handler puts the raw database text
in the response (`"error": "canceling statement due to statement timeout"`); `/resolve`
wraps it. Workers could see that text in the search field. The Make screen now shows
"FL had a problem answering. Tap Try again; if it keeps failing, tell the office." for any
5xx. The server handler is app-wide and was left unchanged; tidying it is a separate change.

## 4. Exception batches: owner approval vs warning-only (decision for you — nothing changed)

**What happens today (as specified in PR #173, D5/D6):** any floor worker can record a batch
with an ingredient not used, swapped, or more than 10% off recipe. Not used/swapped need a
reason from the fixed list; more than 10% off needs one tick. It posts immediately and shows
as a flag on the shift summary. MK-261102-001 confirms it: coconut 0 lb (reason "out of
stock"), oats 500 lb vs 450 lb recipe (+11.1%, ticked); no owner-queue item was opened.

FL already has two "held until the owner approves" patterns to reuse: corrections
(`awaiting_approval`, approve/reject on the Exceptions page) and Extra-Kosher (`HELD until
Michael approves from his own signed-in device; nothing posts before approval`).

| Option | What the floor sees | Inventory / traceability | Cost |
|---|---|---|---|
| **A. Warning-only (today)** | Tick + Record; posts now | Accurate at once; owner sees it after the fact on the shift summary | None |
| **B. Post now, owner sign-off required after** | Same as today | Posts now; also opens an Exceptions-page item the owner must accept/correct within N days (like SHORTAGE) | Small: one exception kind + resolver |
| **C. Hold above a threshold** | "HELD — waiting for Michael" for: an ingredient not used, a swap other than the automatic oat bran rule, or more than ±25%; smaller differences post as today | Nothing posts until approved; batch WIP not available until then | Medium: reuse the held-ticket path; owner must approve promptly or the floor waits |
| **D. Hold everything flagged** | Every >10% / not used / swap waits for approval | Same as C, more often | Medium, and likely daily friction |

Recommendation for discussion: **B**, possibly with **C** for "recipe ingredient not used".
B keeps the floor moving and inventory accurate in real time, and still forces an owner decision on
every exception. C fits ingredients whose absence changes the product (a batch without
coconut is arguably a different product). Please decide; nothing is built yet.

## 5. Classic #9: 323 lb recipe input vs 320 lb packing yield (for review)

Staging values for Batch Classic Granola #9 (product 107, 90002); production uses the same
master data unless noted otherwise — please confirm before acting on it.

| Source | lb per pan | Where it comes from |
|---|---|---|
| Recipe (sum of `batch_formulas`) | **322.6** | Oats 225 + brown sugar 50 + canola oil 30 + coconut 12.5 + almond flavor 1.8 + salt 1.7 + honey 1.6 |
| Pan weight FL posts (`products.default_batch_lb`) | **323** | Used by the Make screen (pans × 323) and by scheduling runs |
| 10-lb case yield (`yield_10lb_cases`) | **320** | 32 cases × 10 lb |
| 25-lb case yield (`yield_25lb_cases`) | **325** | 13 cases × 25 lb |

What this means:

- Each pan posts **323 lb** of batch WIP, but packing it into 10-lb cases expects **320 lb**:
  about **3 lb per pan (≈0.9%)** stays in the batch lot as an unexplained remainder (or shows
  as a pack shortfall), every pan, unless it is recorded as loss.
- Packing into 25-lb cases expects 325 lb, **2 lb more** than posted: the pack would run
  short against the batch lot.
- 322.6 vs 323 is a rounding difference (already open as question 3 in
  `docs/scheduling/S1-amendment-bake-vs-pack.md`).
- Separately, the Make output is **pans × 323 regardless of the pounds entered**:
  MK-261102-001 posted 646 lb out while 670.2 lb of inputs left stock (oats +50 lb, coconut
  −25 lb). Formula entry records actual inputs but not an actual output weight.

Questions for you:

1. What does one pan really weigh after baking: 320, 322.6, 323 or 325 lb?
2. Should the 3 lb/pan difference to 10-lb packing be expected loss (bake/cooling/fines), or
   should one of the stored numbers change?
3. Should the Make screen ask for (or let the worker correct) the actual output weight when
   inputs differ from the recipe, or keep output = pans × pan weight?

Nothing changes until you decide; changing `default_batch_lb` or the yields is a production
master-data change needing your approval and a backup.

## 6. Recommended staging test plan (after approvals)

1. **Approve** (separately): the staging test lot (§1) and a staging-only deploy of
   PR #173 + `feat/make-mobile-ux` + the test sign-in (one combined branch, as before).
2. Sign in as **Michael (Owner)**, record the §1 receive; check the receipt and lot name.
3. Sign in as **Arturo (Floor)** on a phone (or 390 × 844): Make → Classic #9 → 2 pans.
   - Coconut card offers `STG-TEST-COCONUT-261102` with 50 lb; type each bag's code.
   - Exactly one "N bags still need the code" line until codes are typed.
   - Change one pound value: the sticky bar shows **Check bags and pounds**; the screen
     stays on that ingredient after the check; Record turns on only after it.
   - Record a **normal batch** (all within 10%): no tick needed; receipt `MK-…`, entered by
     Arturo; shift summary shows no flags for it.
4. Repeat with oats +15%: one tick required; shift summary shows the flag.
5. Search failure (manual): with the phone offline, search a product → "Could not reach
   FL… Try again"; back online → Try again works.
6. Spanish (ES) pass of steps 3–4 at 390 px: no clipped dropdowns, no sideways scrolling.

# Count-tool readiness for the Nov 20 cutover — 2026-10-13

**Plain English first.** The count sheets and the review logic still work for the live count
while production runs. The part that posts the reset does **not** fit the system as it is
live today:

1. It posts through the old direct `/adjust` route. Those entries get **no receipt number**,
   and that route skips the new rules (shortage refusal, the over-500 lb photo hold). That is
   the "silent bypass" we must not use.
2. When it is moved onto the normal ticket path (prepare → commit), three new rules will
   stop it on cutover morning unless there is an owner-approved path:
   - corrections **over 500 lb** are **held** for the owner;
   - a **positive** correction on a lot with an **open shortage** is **refused**;
   - **extra-kosher** stock (283–290) has no lots yet; the tools refuse to create lots, and
     extra-kosher stock can only enter through an **owner-approved found ticket**.
3. The printed sheets still carry the **old names** of 283–290 (A12 renamed them), so every
   filled 283–290 row would be rejected; and **285/288 (bulk per lb)** are still treated as
   "expected zero cases", so any count of them is held.
4. There is no simple way to apply **one area at a time** (Michael's option A), and the
   tools can only talk to production, so they cannot be rehearsed on staging.

Everything else — per-row count times, the movement review, duplicate ticks, coverage,
multi-area holds, the signed approval, the journal and no-blind-retry — still works.

Checked against `origin/main` f4e51b5 (A1, A2, A3b, A5, A6, A7, A9, A12, key sessions,
078 recipes all merged). Read-only catalog check of production on 2026-10-13: 210 products
(25 batch, 75 finished, 78 ingredient, 32 packaging incl. billing 102/176) → **178 reset
products**, unchanged from the issued packet; **0 extra-kosher lots**; 0 open shortages;
owner actor = `Blubber`. Since the packet was issued (2026-10-06) only the names of
283–290 changed (A12, migration 075). The recipe corrections (078) do not appear on the
sheets.

## Tool by tool

Legend: **works** = still fine; **breaks** = would fail or violate a live rule;
**needs change** = works but not for the stated requirement.

### Count packet — `build_v3_materials.py`, `v3/count-sheet-v3.csv/.pdf`, register, coverage, move log

| Step | Verdict | Why |
|---|---|---|
| All 210 products on blind sheets, 178 in reset scope (e) | works | Catalog count and split unchanged (178 / 32). |
| Area, counter, start/end time per sheet; time per row (a) | works | Already in the v3 columns; reconciler uses each row time, falls back to sheet end. |
| Moved-during-count log (a) | works | Cross-area moves; FL movements are logged in FL itself. |
| 283–290 as their own lines (d) | **breaks** | Lines exist (4 rows each) but carry the **pre-A12 names** (e.g. "Batch SS Classic Granola #9 (Kosher Ignition)"). The reconciler requires the exact catalog name, so every filled row would be held as "Product/SKU/name mismatch". Regenerate from the current catalog. |
| 285/288 bulk per lb (b, d) | **breaks** | Printed as `cases`, and `fresh_start_common.EXPECTED_ZERO_PRODUCTS` still lists 285/288 (an older Sunshine decision), so any positive count is held "Legacy package identity requires owner decision". Both are lb products (`uom='lb'`, no case size). |
| Corrected recipes (078) (d) | works | Recipes are not on the sheets. |
| Duplicate tick column | needs change | The builder still writes the old column set; only the template command adds `duplicate`. |
| Per-area pause instructions (a) | needs change | README/guides describe "approved products only", not the 15–30 min per-area pause with receipt read-back. |

### Preview — `reset_preview.py --v3` → `fresh_start_v3.analyze`

| Step | Verdict | Why |
|---|---|---|
| Per-row cutoffs, sheet-window/30-min movement review, late-entry rules, multi-area hold, moves, duplicates, coverage | works | New ticket-era movements (A1 receipts, A3b short postings, A6 shipments, A12 packs/downgrades) are still ordinary ledger lines and are reviewed like any other. A2 LATE_ENTRY exceptions do not change lines. |
| Reading the ledger | needs change | Production credential is hard-wired; no staging target for a rehearsal. Snapshot lacks `receipt_number`, `ticket_id`, `reason_code`, lot `kosher_tier`, open exceptions. |
| Recognising earlier reset entries | needs change | Detects them by `adjust_reason` starting `OPENING BALANCE … v3 … L{lot}`. Ticket posts store the reason **label** ("Physical count") in `adjust_reason`; the v3 text must travel in the ticket note. |
| Counted lot not in FL | works for regular stock (held, 7-day follow-up, never created); **breaks** for extra-kosher (b) | 283–290 have no lots at all; P1.21 requires an owner-approved new-lot load. |
| One area at a time (a) | needs change | Approval must cover every proposed change in a preview, so a per-area apply needs hand-cut input copies on cutover morning. Add an area-group scope. |
| Open shortages | needs change | Not visible. A lot with an open SHORTAGE and a positive count will be refused at posting. |
| Over-500 lb corrections | needs change | Not flagged in advance; they will be held at posting. |

### Approval — `approval.md`

| Step | Verdict | Why |
|---|---|---|
| Signed preview hash, product IDs, named actor, owner signature | works | |
| Owner-approved held entries | needs change | No field records the owner's approval of >500 lb holds, extra-kosher attestation or shortage closure. |

### Apply — `apply_reset.py`

| Step | Verdict | Why |
|---|---|---|
| Personal key, `/auth/whoami`, actor-hash lookup | works | A2 keeps owner on direct and ticket routes; key sessions (PR #103) do not change raw personal keys. Named actor must be `Blubber` in production. |
| Posting route | **breaks** (c) | Direct `POST /adjust`: no receipt number; skips A3b shortage refusal and the 500 lb hold. Must post `POST /adjust/prepare` (reason `physical_count`, note = v3 reason, `occurred_at` = count cutoff) → `POST /tickets/{t}/commit`, giving `ADJ-YYMMDD-NNN`. |
| >500 lb correction | **breaks** after the switch | Commit returns **202 held** + LARGE_CORRECTION exception. Needs an owner-approved release (owner `POST /exceptions/{id}/approve`) bound to the signed approval. |
| Positive count on a lot with an open shortage | **breaks** after the switch | Prepare returns **409 SHORTAGE_OPEN_RESOLVE_INSTEAD**. Owner decision needed (see below). |
| Extra-kosher new lots (283–290) (b) | **breaks** | Executor never creates lots. FL's only allowed path is a **found ticket** (new lot, reason `physical_count`) that is **always held** for the owner's kosher attestation, then posted on approval (FND receipt, tier + attester stamped). Receive of 283/284 is refused for everyone and must never be used. |
| Back-dating to the count time | works | Owner is exempt from LATE_ENTRY; >14 days stays owner-only; `backfill=true` kept. |
| Journal, no blind retry, readback | needs change | Proof matches by `adjust_reason`; must match by `ticket_id`/receipt instead. Tickets add server idempotency (commit replays the same receipt). |
| Duplicate warning (POSSIBLE_DUPLICATE) | works | Executor must stop, not auto-acknowledge. |
| A3b 10 % highlight, pre-make-adjust tag | works | Highlight is information for the weekly view; pre-make tag only for the same actor. |
| Per-area read-back | needs change | Print the receipts per lot/area so they can be checked against the sheet. |
| Production-only URL | needs change | Add an explicit staging target for the rehearsal; production stays the default. |

### Verify — `verify_reset.py --v3`

| Step | Verdict | Why |
|---|---|---|
| Read-only full-scope verdict | works | |
| Proof of executor posts | needs change | Same `adjust_reason` matching as apply; must prove ticket receipts, extra-kosher tier and owner attestation. |

### Other live features

- **A5 lot/supplier rules:** apply to receive (supplier) and make/pack (lot confirmation). A found of finished/batch stock opens no unidentified-lot clock (ingredients only). No change.
- **A6 ship gate, A7 orders, A9 shift summary:** reset receipts appear on Michael's shift summary for that day; nothing blocks.
- **Key sessions:** independent; the executor keeps using the raw personal key file (`~/.config/factory-ledger/blubber_key`, mode 600).

## Michael's decisions (asked and answered 2026-10-13)

1. **Open shortage + count says more stock → the ticket closes it.** Small API change: an adjust ticket with reason `physical_count` may name the lot's open SHORTAGE exceptions; its post closes them as *counted* in the same transaction (receipt, count time, his key). Why an API change was unavoidable: without it the only ways through were waiving (records "waived", two steps) or the existing *counted* resolution (posts "now" with **no receipt number**, and over 500 lb it refuses without a photo — no owner-hold path). Both break requirement (c).
2. **285/288 are counted in lb** and loaded with the other extra-kosher stock. 183–186 stay expected-zero.
3. **The executor approves, per the signed list.** With `Owner approves held entries: Y` and every new extra-kosher lot listed in `approval.md`, the executor approves exactly those FL holds with Michael's own key (note = preview hash) and checks FL recorded him as approver/attester. Anything not on the list stops and stays held.

## Part 2 — what was fixed

| Was | Now |
|---|---|
| Direct `/adjust`, no receipts, skips A3b | `apply_reset.py` v3 posts `/adjust/prepare` or `/inventory/found/prepare` → `/tickets/{t}/commit`, reason `physical_count`, `occurred_at` = count cutoff, `backfill=true`, named key. Receipts `ADJ-…` / `FND-…`. Journal v2 records the ticket per write; proof is by `ticket_id` + receipt, so an interrupted run is proved or provably not posted (no blind retry). |
| >500 lb would be held | Preview marks the row `LARGE_CORRECTION`; the signed `Owner approves held entries: Y` lets the executor approve that one hold (`/exceptions/{id}/approve`), then the read-back proves it was released as approved. |
| Open shortage refused (409) | **API (smallest change):** `resolves_shortage_ids` on `AdjustPrepareRequest`; `physical_count` only; must equal the lot's open set (re-read `FOR UPDATE` at commit, before the lot lock); owner/floor only; the post closes each as `counted` with the receipt. Without naming them the old 409 still applies. Preview lists the IDs; apply stops if they changed. A negative lot is no longer held when an open shortage explains it (A3b posts short makes negative on purpose). |
| Extra-kosher stock could not be loaded | A counted lot code that FL does not have, on an **extra-kosher** product, becomes a NEW lot (FL's canonical upper-case code) loaded by a found ticket. FL always holds it for kosher attestation; the executor approves only lots listed under `Extra-kosher lots`, then checks tier + attester on the lot and the FND receipt. Regular unmatched lots are still held and never created; nothing is retagged; receive of 283/284 is never used. |
| 283–290 names stale; 285/288 held | Packet reissued from a read-only catalog export (`v3/raw/catalog-2026-10-13.json`): new names, 285/288 printed in lb, **Dup?** box on every row, step 9 "pause posting per area". 210 products / 850 rows / 110 pages; only 6 products changed sheet number. |
| No per-area apply, production only | `--area` (repeatable) on preview/verify; `--target staging|production` on every v3 tool (apply requires it explicitly; preview records it; mismatch refused; staging must be the staging host and production must not be). |
| Duplicate warning would stall | A `POSSIBLE_DUPLICATE` is acknowledged only when the matching posting is a ledger line the signed preview was computed from (baseline re-proved before every write) and is journaled; any other warning stops. |
| No read-back | Per-lot `Read back ADJ-…` lines, `readback.csv` and `timings.json` per approved preview. |

Tests: 183 offline tool tests (`test_fresh_start_v3.py`, `test_apply_reset.py`; 22 new) and 7 new DB tests (`tests/test_reset_shortage_close.py`). Full suite on a freshly built PG17 DB from `tests/schema/schema.sql`: **2,750 Python passed, 0 failed, 0 skipped; 107 Node passed.**

**Observation for review (not changed here):** A3b's 500 lb rule converts a correction by the product's ledger unit. For the five finished products whose uom is exactly `case`, it multiplies the (pound) delta by the case weight. A 40 lb reset on a 15 lb-case item therefore counts as "600 lb" and is held. The preview mirrors FL exactly, so the hold is predicted, signed and approved rather than stalling. The A3b conversion itself probably deserves a separate look.

## Part 3 — staging dress rehearsal (2026-10-13 ~19:47 EDT)

Script: `scripts/count_rehearsal_staging.py`; receipt: `docs/deployments/count-rehearsal-staging-receipt.json` (reference `STG-CNT-9F7F22`). **This branch's API ran in-process on 127.0.0.1 against the guarded staging database.** Hosted FastAPI-staging runs `main`, which lacks the shortage change; redeploying it would have displaced the key-sessions build. No production read or write except one timed read-only snapshot. Synthetic rehearsal owner/floor actors were deactivated afterwards, and the owner is **not** Michael.

Fake filled sheets, three areas, each run as on cutover morning: fresh preview `--area` → owner review (scripted stand-in) → signed approval → dry run → `--apply` with the phrase typed into a real terminal → verify.

| Area | Lots | What it exercised | Receipts | Machine time of the pause |
|---|---|---|---|---|
| Dry storage | 8 | estimated row; **−680 lb** (LARGE_CORRECTION hold approved under the signed list); 6 ordinary lots | ADJ-261013-016 … -023 | preview 2.1 s, dry run 1.6 s, apply + read-back 24.7 s (2.0–3.3 s per lot), verify 0.9 s → **29 s** |
| Production floor | 1 | coconut short-posted by a make (book −20, SHORTAGE #32 open), counted 25 lb → +45 closes it as *counted* | ADJ-261013-024 | 1.2 / 1.3 / 7.7 / 0.9 s → **11 s** |
| Warehouse | 2 | 39 cases granola with a floor damage write-off **inside** the window (owner review: after count) and the POSSIBLE_DUPLICATE it caused; **extra-kosher 283** on a lot FL had never seen (written lower-case on the sheet) → NEW lot, kosher hold approved | FND-261013-004, ADJ-261013-026 | 0.9 + 0.9 (two previews) / 1.5 / 11.3 / 1.0 s → **15 s** |

After each apply, verification shows every lot in the area READY with remaining difference **0**. Its exit code is 2 only because the rest of the staging catalog was not counted (full-scope sign-off). Shortage #32 is `resolved/counted` with `resolution_ticket_id` = the reset ticket. Lot `STG-CNT-9F7F22-EK-1` is `extra_kosher`, `found_inventory`, attested by the rehearsal owner. Both owner holds are `resolved/approved` by the owner actor.

**What this means for the real pause (production).** A read-only production snapshot takes **4.0 s** (6,543 lines, 1,235 lots; measured 2026-10-13); staging took about 1 s. The executor reads one snapshot per lot plus about three per run, and preview/dry run/verify read one each. Expect roughly **35 s fixed + about 6 s per lot**: a 40-lot area needs **about 4–5 minutes of machine time**. The rest of the 15–30 minutes is people: the owner checks the fresh preview against the sheet and signs. Review the area's first preview **before** the pause; at the pause, re-run it, confirm nothing changed, sign and apply. Areas over about 100 lots, or with many held/kosher rows, should be split.

Earlier attempts (kept honest): run `A575D9` stopped at the typed confirmation (pty quirk, nothing posted). `3A3FEE` stopped in area 2 on the old negative-lot hold (fixed above). `C5CDEB` stopped in area 3 on the duplicate warning (rule added above) after loading a synthetic extra-kosher lot on staging **285**. `EA2A30` completed on **288**. Those synthetic lots remain on staging and are labelled with their reference.

## Still owed before Nov 20

- After merge, repeat one area against **hosted** staging (unset `FL_STAGING_API_URL`) once Railway staging has this code; then a production **dry run** with Michael's real key (`--target production`, no `--apply`).
- Michael answers the real owner review (movement classifications) per area; the scripted stand-in is not a substitute.
- Staging carries synthetic rehearsal lots on 283/285/288; harmless, but any later staging test of those products must expect them.

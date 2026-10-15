# Factory Ledger pilot plan

## Key dates

| Milestone | Date (2026, New York) | What happens |
|---|---|---|
| Pilot start | Wed Oct 21 | Arturo and Luz practice on staging for part of each day. |
| Clean-summary and dry-run window | Oct 26–30 | Five clean shift summaries plus the production dry run. |
| Packet printing | Fri Oct 30 | Print approved count packets. |
| Go / no-go | Mon Nov 2 | Michael decides from the evidence. |
| Physical count and signing | Thu Nov 5 | Count while production runs; Michael signs the approval list that evening. |
| Reset and cutover | Fri Nov 6 | Each area pauses 15–30 minutes; whole-reset proof before release; rotate master key; GPTs become read-only. |
| A10 stage 1 | Mon Nov 9 | Retire the approved old staff write paths. |
| First fallback | Fri Nov 13 | Cutover only after renewed readiness approval. |
| Second fallback | Fri Nov 20 | Cutover only after renewed readiness approval. |
| M0 and GPT retirement deadline | Before Fri Dec 11 | M0 read-only plugin live; old GPTs retired. |

**For Michael.** [Floor guide](ARTURO-GUIDE.md) · [Office guide](OFFICE-GUIDE.md) · [Daily check](MICHAEL-DAILY-CHECK.md) · [Printable cards](cards/).

Use the **Key dates** table for the schedule. This replaces the old ten-day plan. Arturo and Luz practice on staging for **part of each day**. The exact daily practice slot, selected events and opening stock remain Michael's decisions (§8); this plan does not assume double entry of every production event or authorize staging stock creation.

## 1. Operating rules

- **Sign in once per shift with your personal key.** Use an FL session with a **180-minute idle timeout**. Background auto-refresh does not extend it. Sign out at handover; sign in again after expiry. **PIN login stays OFF.** A second routine sign-in at shift summary is a pilot defect to resolve, not the planned staff workflow.
- **No receipt = not recorded.** Every write follows **prepare → Record → receipt**. Read the prepared details before Record. Only a receipt number proves the entry is in the ledger; a draft, chat reply, hold or button press does not.
- **FL Assistant:** unsupported requests get a **neutral English/Spanish card**, never a raw error. **Red is only for a failed Record.** Missing details are listed in plain words. Adjust/found without a reason shows reason buttons. Staff supply the missing facts; FL must not guess.
- **Extra-kosher:** receive is refused; found stock is held until **Michael approves from his phone**. Extra-kosher makes also need his attestation. Pre-cutover stock belongs in the physical count under **283/284 batches and 285–290 finished goods**. Never receive it to bypass a hold or relabel a regular lot.
- **Live count:** production keeps running during the physical count. Record movements and count times. Each area pauses **15–30 minutes at reset**. Michael signs the exact approval list **the evening before**; changed evidence requires renewed approval.
- **Whole reset:** proof must say **FULL RESET SCOPE RECONCILED (exit 0)** before **any release step**, including key rotation or flags. Any gap stops release. **O-3 closed**; acceptance still requires actual cutover proof.
- **Master key rotates on cutover day**, after that proof. GPTs become read-only at cutover and never become the write fallback.
- **Close/reopen an order as “shipped, not recorded”: owner-only, note required, pre-cutover only.** Every path enforces the same rule, including requests using the master key, which cannot impersonate the owner. **O-4 closed.**
- **Sunshine bulk:** use the paper bin log until after cutover. **O-2 closed; A8 deferred.** No new invoicing trigger is assumed.

### Recipe reference

| Recipe / product | Settled rule |
|---|---|
| SS #1 / #2 / #3 | Oat bran stays in the recipes: **29 / 28 / 28 lb**, respectively. |
| Rolled oats substitution | Rolled oats may replace oat bran **pound-for-pound without approval**. Record the substitute and its lot; this does not change the base recipe. |
| Fruit Nut | **328.3 lb = 276.3 lb Classic #9 + 13 lb each of four nuts/fruits**. |
| Product 284 | **4,000-count chips**. Do not teach a different chip standard. |

## 2. People and daily routine

| Person | Responsibility |
|---|---|
| Michael | Selects practice scope and accepts evidence; approves holds from his phone; reviews daily; decides go/no-go and each production action. |
| Arturo | Floor practice, physical lot confirmation, receipt checks and shift summary; directs the count. |
| Luz | Office practice and receipt/paper reconciliation; coordinates fallback within office permissions. |
| Miriam | Office guide applies to her too; Michael confirms her acceptance slot rather than assuming extra pilot days. |
| Engineering | Fixes and verifies reported defects in separately reviewed work; no production action is authorized by this docs update. |

At practice start, verify the **staging** label and the signed-in name. Use the FL Assistant at `https://fastapi-staging-production-dd7b.up.railway.app/dash/fl-assistant` and staging dashboard at `https://fastapi-staging-production-dd7b.up.railway.app/dashboard/index.html`. Staging entries do not update real inventory. Michael must state how selected practice entries relate to the real recording process before practice begins (P-1).

During the agreed practice slot, run the relevant checks below. At shift end, Arturo compares the summary with the agreed practice record and chooses **Confirm**, **Confirm with notes** or **Something is missing**. Luz reconciles office entries. Michael does the [daily check](MICHAEL-DAILY-CHECK.md) next morning. A partial-day pilot still needs an accurate summary of its agreed scope; do not claim an unpracticed full shift was tested.

## 3. Practice checks

Use selected staging examples approved by Michael. Repeat failed checks after the fix. Checks are grouped by skill, not invented calendar days.

| Check | Arturo / Luz do | Pass evidence |
|---|---|---|
| Sign-in and handover | Personal sign-in; verify own name; sign out at handover. Engineering verifies 180-minute real inactivity while background refresh runs. | Correct actor on receipts; expired/signed-out Record refused; no routine second key entry for the summary; PIN off. |
| Receive and receipts | Receive with real supplier and supplier lot; look up the returned receipt. | Supplier and quantities match; exactly one receipt for one event; no receipt is treated as not recorded. |
| Make and pack | Arturo checks physical tags and confirms ingredient/source lots, including a wrong-last-four refusal. Luz attempts a floor-only action on staging. | Correct lots on receipts; wrong lot and office make refused without a raw error. |
| Recipes and substitution | Verify the recipe reference; record an oats-for-bran example through the accepted interface. | Correct recipe pounds; substitute and lot visible before Record; no approval demanded for the settled substitution. If missing, stop and report P-4. |
| Shipping | BOL number; physical lot confirmation; BOL photo same day by 23:59 plant time. Exercise missing-photo follow-up on staging. | Receipt plus BOL proof; a deliberately missing photo is visible for follow-up. A packing-slip/truck photo is not BOL proof. |
| Missing facts and reasons | Omit a quantity or lot; try adjust/found without a reason; try an unsupported request, in both languages. | Missing details listed plainly; reason buttons appear; neutral bilingual unsupported card; no raw errors or red pre-Record card. |
| Correction and failed Record | Use a stated reason; Unknown needs a note. Exercise a large correction hold and a failed Record. | Over 500 lb needs photo or Michael's approval; held ≠ recorded; red only after failed Record; use same draft/receipt check before retry. |
| Extra-kosher | Attempt receive; prepare found stock and make; Michael approves from his phone; pack from correct lot; attempt regular-source pack. | Receive refused; found/make held until owner approval; receipt after approval; regular-source extra-kosher pack refused. |
| Shortage and identification | Record an approved shortage example and resolve from actual count/missing movement; identify an unreadable supplier lot from evidence. | No invented stock to force a make through; shortage within 2 business days; unidentified received lot within 7 business days. |
| Office work and close guard | Use **Office / Oficina → Office action / Acción de oficina** for order/expected-receipt work: prepare → Record → receipt. Check “shipped, not recorded” owner/note/pre-cutover restrictions. | ORD/ER receipts; planned receipt does not add stock; all close/reopen paths enforce the rule, including master-key paths. O-8 and O-4 are closed decisions; verify the accepted deployment. |
| Paper fallback | Write a chosen event on numbered paper and enter once with original time through the permitted form. | Receipt number written back to paper; duplicate warning resolved; Luz does only office-authorized actions. Arturo/Michael enter floor-only work. |
| Summary and unaided use | Arturo and Luz finish the agreed practice scope without coaching; compare summary, receipts and paper. | Correct attribution and quantities; confirmed summary; every discrepancy explained. |

## 4. Clean-summary and dry-run window

During the **clean-summary and dry-run window** in Key dates, collect **five clean shift summaries** with zero unexplained discrepancies. Every receive needs a real supplier. Include the extra-kosher approval/pack/refusal evidence. A missing event, false confirmation or unexplained difference prevents that summary from counting.

The **production dry run** is a separate owner-approved, no-apply operation with real count evidence, correct production owner identity and measured timings. Staging proof does not replace it. Michael supplies evidence/approval, engineering executes the accepted runbook, and the reviewer checks output. **O-10 stays open until performed.** This docs task performs none of those operations.

If five clean summaries or the dry run are incomplete, report the gap at **go/no-go**. Do not invent a spare day or silently compress acceptance. Use the listed fallbacks only with renewed readiness review; Michael must set the revised count/signing/rehearsal schedule (P-7).

## 5. Count and cutover handoff

At **packet printing**, Luz prints the approved packet and Arturo checks readability. At **physical count and signing**, count while production runs, capture every movement and preserve private evidence. Extra-kosher pre-cutover stock goes under products **283/284/285–290**, with Michael's attestation; uncertain identity/ownership remains held. Michael signs each exact reset approval list that evening.

At **reset and cutover**, each area pauses 15–30 minutes. Use the signed plan, current movements, one receipt per posted entry and independent readback. The master runbook controls reset execution. **FULL RESET SCOPE RECONCILED (exit 0) must precede any release step.** Rotate the master key on this cutover day, then make the GPTs read-only and release accepted personal-session workflows in the runbook order.

A10 stage 1 and the M0/GPT retirement deadline are in Key dates. A10 stage 2 still requires its accepted rollout; closed O-8 does not mean an unverified rollout already occurred. Sunshine stays on its paper bin log until after cutover; A8 remains deferred.

## 6. Go / no-go checklist

Michael signs GO only from evidence. The later physical count and signed reset remain release conditions; do not pretend they happened before the go/no-go meeting.

- [ ] Current staging software and required flows accepted by Michael, Arturo and an office user; each personal sign-in has correct attribution and idle/sign-out behavior.
- [ ] Five clean summaries from the agreed practice scope; every receive has a real supplier; owner-approved extra-kosher make/found/pack and refusals demonstrated.
- [ ] Neutral bilingual unsupported/missing-detail/reason behavior accepted; red only for failed Record; no raw error taught as normal staff workflow.
- [ ] Current packet, area grouping, staffing, live count, evening signing and 15–30 minute pauses ready; production dry run completed (O-10).
- [ ] Whole-reset proof procedure accepted (O-3 closed); proof wording required before any release. All-path order close/reopen rule accepted (O-4 closed); production deployment evidence checked.
- [ ] Production sessions/proxy plan, complete private key-consumer inventory and dashboard-key delivery ready (O-5/O-6/O-7); master rotation scheduled for cutover day.
- [ ] Dedicated OpenAI project/key, $20 cap, 50%/80% alerts and test-key retirement evidenced (O-13); owner must confirm readiness, not assume from an old deadline.
- [ ] Office order/expected-receipt workflow accepted (O-8 closed); supported fallback and staffing cover required actions; open P-items resolved or explicitly deferred without weakening the gate.
- [ ] No open STOP problem. Safety defects are repaired and rechecked; a date is not a waiver.

If any mandatory check fails, NO-GO. Michael selects the appropriate fallback from Key dates and schedules fresh acceptance, count evidence and signatures.

## 7. Problem reporting

Staff call **Michael** when stuck. They record the time, attempted action and receipt number or **NOT recorded**, plus a screen photo with no key visible. Keep the event on paper; check for an existing receipt before entering it again.

| Class | Examples | Action |
|---|---|---|
| STOP | False success/missing receipt, duplicate event, wrong actor, changed quantity after draft, exposed key or a rule bypass | Stop that action, preserve evidence, tell Michael and engineering; repair and reverify before resuming. |
| FIX BEFORE CUTOVER | Missing details unclear, unsupported request/raw error, wrong language, missing required screen or failed fallback | Assign owner; resolve before go/no-go for required work. |
| LATER | Safe wording or speed improvement | Track for later without hiding a safety defect. |

| Item | Milestone / time | Who | What happened | Receipt / NOT recorded | Class | Owner | Resolution |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Record actual dated operational evidence in the private pilot log. Do not add competing schedule dates to this document.

## 8. Open questions for Michael

| ID | Question | Status / owner |
|---|---|---|
| P-1 | Which part-day practice slots and events should Arturo/Luz use, and exactly how should those selected events relate to the current real recording process? | OPEN — Michael. No all-day or blanket double-entry assumption. |
| P-2 | What opening stock/examples should staging use, and who approves/prepares them? | OPEN — Michael with engineering. No fixture load authorized; shortages need an explained disposition. |
| P-3 | When should Miriam do her acceptance checks, and will Arturo/Luz confirm the Spanish and printed-card readability? | OPEN — Michael schedules; staff validate. |
| P-4 | Which accepted screen should staff use if a substitution is not fully shown, and who demonstrates it during practice? | OPEN — Michael with engineering. Do not guess a substitute lot or teach a no-receipt shortcut. |
| P-5 | Who backs up Michael for urgent help, and what phone/contact details should appear on the printed cards? | OPEN — Michael. Current guides say call Michael; no invented number. |
| P-6 | Where is the approved numbered paper form/private evidence folder, and who covers floor-only work if Arturo is unavailable? | OPEN — Michael with Luz. Office permissions do not expand during fallback. |
| P-7 | If a fallback is used, what are the revised go/no-go, clean-summary/dry-run, printing, count/signing and A10 slots? | OPEN — Michael. Only fallback cutover dates are settled. |
| P-8 | What is the accepted M0/GPT retirement method and plan if M0 acceptance is late? | OPEN — Michael; tracked as O-16/O-17 in the runbook. Deadline remains as in Key dates. |

The accepted office screens and shared shift-summary session are from [PR #115](https://github.com/sevenwells72/factory-ledger/pull/115) (merged `d962a1f`); the all-path close/reopen guard is [PR #121](https://github.com/sevenwells72/factory-ledger/pull/121) (merged `9bd30be`). Reopen requires a new note and the order’s immutable New York creation date before the configured cutoff, even if it was later shipped. Master-key requests are refused for this owner-only action.

For an erroneous entry, an authorized floor/owner opens **Receipt # / Recibo # → Look up / Buscar → Prepare void for entry #**, checks the Data-entry error draft, then **Record → VD receipt**. Office users call Arturo/Michael. Confirm the affected entry before voiding; never guess a reversal quantity.

O-2, O-3, O-4 and O-8 are closed settled decisions. The runbook retains the owned open-item register for the remaining operational work. Closed decisions do not certify production deployment or authorize action in this docs task.

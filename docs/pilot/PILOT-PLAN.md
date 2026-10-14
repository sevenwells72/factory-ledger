# Factory Ledger pilot plan — Nov 2 to Nov 20, 2026

**For:** Michael (owner). **Staff guides:** [Arturo](ARTURO-GUIDE.md) · [Office](OFFICE-GUIDE.md) · [Michael's daily check](MICHAEL-DAILY-CHECK.md) · [quick cards (PDF)](cards/).
**Based on:** `origin/main` @ `d5e9fea` (PR #111 merged), design `docs/design/phase1-safe-operating-system.md` rev 3.8 + the 2026-10-13 decisions, `FOLLOWUPS.md`, `docs/contracts/`, `docs/deployments/`. Written 2026-10-13.

This plan only describes what is merged on main today. Anything else is labelled:

- **🟡 Coming before the pilot (PR #107):** waits on PR #107 (D1 + D2 dashboard: receipt page, Exceptions page, "Record an entry" forms). It is a draft today. It must be merged and on staging before Nov 2, or that day's test moves.
- **⛔ Not built (no PR):** nothing is in progress for it. The plan works around it.

---

## 1. Dates

| Date | What happens |
|---|---|
| **by Wed Oct 28** | Production OpenAI key (its own project, $20/month hard cap, alerts at 50 % and 80 %) set on FastAPI-staging (decision 38). |
| **by Fri Oct 30** | Pilot stack on staging; PR #107 merged and on staging. Staging personal keys for Arturo, Luz and Miriam handed over in person. |
| **Mon Nov 2** | **Staging pilot starts** (day 1 of the 10-day script below). |
| **Fri Nov 6** | **Checkpoint** (day 5). Old OpenAI test key revoked before today. Go / fix / use the "simplify" lever (design §10.2 item 9). |
| **Mon Nov 9 – Fri Nov 13** | **Five confirmed shift summaries in a row** (days 6–10), with zero unexplained discrepancies (gate (d)). Gate (e) is tested Tue Nov 10. |
| Mon Nov 16 | Spare day, used only if one day of Nov 9–13 has to be redone. The latest clean five-day run that still meets the gate starts Wed Nov 11. |
| **Tue Nov 17** | **Go / no-go** against the readiness gate (§6). |
| **Fri Nov 20** | **Cutover** (count, reset, key rotation, GPTs read-only, FL Assistant live on production; design §10.1). |
| Mon Nov 23 | A10 stage 1 (old direct write routes closed). |
| **Fri Dec 4** | **Fallback cutover date** if the gate is not green on Nov 17. The date never overrides the gate. |

## 2. How the pilot works (read this first)

1. **The pilot runs on staging, not production.** Staging is a separate copy of Factory Ledger. Nothing entered there changes real inventory.
2. **Until cutover, the real record stays exactly as today.** Arturo keeps recording real events the current way (the Floor GPT). He *also* enters the same real events in the FL Assistant on staging. The staging shift summary is then checked against what really happened on the floor. *(Safest-option choice made for this plan. If you want only some events on staging, cut the script, not the real recording.)*
3. **Staging stock does not match the floor.** Unless staging is given opening stock, almost every make or pack will create a shortage flag. See decision P-2 in §8.
4. **Pages (staging):**
   - FL Assistant: `https://fastapi-staging-production-dd7b.up.railway.app/dash/fl-assistant`
   - Shift summary: `https://fastapi-staging-production-dd7b.up.railway.app/dashboard/shift-summary.html`
   - Dashboard: `https://fastapi-staging-production-dd7b.up.railway.app/dashboard/index.html`

   Use the Railway addresses, not the Netlify site. The Netlify site is production and does not show the sign-in bar.
5. **Sign-in:** each person types their **personal key** once per shift. FL swaps it for a session (180 min idle limit) and every entry is recorded to that person. **Sign out** before handing over a shared tablet. PIN login stays off. A QR badge is built only if typing the key fails in week 1 (decision 33).
6. **One rule for everyone:** *no receipt number = not recorded.* Only the green "✓ RECORDED IN FACTORY LEDGER" card with a number like `MK-261102-003` counts.

## 3. Who does what

| Person | Role in FL | Pilot job |
|---|---|---|
| **Michael** | owner | Runs the pilot. Daily 10-minute check ([MICHAEL-DAILY-CHECK.md](MICHAEL-DAILY-CHECK.md)). Approves held entries (large corrections, extra-kosher makes) **from his own signed-in device only**. Keeps the problem log (§7). Makes the go/no-go call. |
| **Arturo** | floor | Enters real floor events on staging. Confirms his shift summary every day. Owns shortages and unidentified lots (2 and 7 business days). |
| **Luz** | office | Office tests (receipts, ships, lookups, exceptions list). **Fallback operator** on the dashboard forms (decision 10). |
| **Miriam** | office | Same tests as Luz on at least two days, so the gate has a second office user. |
| **Engineering** (Claude Code / Codex) | — | Fixes problems from the log. Staging only. Every production step and merge needs Michael's per-action approval. |

### Every pilot day

| When | Who | What |
|---|---|---|
| Start of shift | Arturo | Sign in to the FL Assistant on the floor tablet. Check the header says "Signed in as Arturo · floor". |
| During the shift | Arturo | Enter each real event in the Assistant right after it happens. Wait for the green card. Record a NOT recorded card on the problem sheet. |
| During the day | Luz (Miriam on her days) | The office test for that day (§4). |
| End of shift | Arturo | Open the shift summary, check it against the floor, press **Confirm**, **Confirm with notes** or **Something is missing**. Sign out. |
| Next morning, 10 min | Michael | Daily check: approvals, exceptions, yesterday's summary, late entries, the problem log. |
| Next morning | Engineering | Fixes from the log, on staging only. |

## 4. The 10-day test script

Each day still includes **all normal real entries** for that day. The script adds the special tests. "✓ day passes" means every item in that row is true.

### Week 1: learn each entry type (Nov 2–6)

**Day 1 · Mon Nov 2 — sign in, receive, shift summary**
- **Arturo:**
  - Sign in once.
  - Record every real receive. Choose the real supplier. Type the supplier lot from the bag or pallet.
  - Ask "what did I enter today?".
  - Look up one receipt by number.
  - Confirm the shift summary.
  - Sign out. Sign in again. Check the name.
- **Luz:**
  - Sign in on her own device.
  - Record one receive from paperwork in the Assistant (office may receive).
  - Look up Arturo's receipt by number in the Assistant.
  - Open the shift summary with "View all actors (read-only)".
- **✓ day passes when:**
  - Every receive has a receipt number **and** a real supplier.
  - Every receipt shows the right person.
  - The summary is confirmed.
  - Sign out worked.

**Day 2 · Tue Nov 3 — make and pack with lot confirmation**
- **Arturo:**
  - Record each make and pack.
  - Confirm each ingredient lot from the physical tag: **last 4 characters including the hyphen** (e.g. `-004`).
  - Once, type a wrong last-4 on purpose and see FL refuse it.
  - When FL asks, use **Confirm this output SKU**.
- **Luz:** Try to record a make in her Assistant. FL must refuse it (office cannot make).
- **✓ day passes when:**
  - Every make/pack has a receipt.
  - No make posted without a confirmed lot.
  - The wrong last-4 was refused.
  - The office make was refused.

**Day 3 · Wed Nov 4 — substitutions, voice and text**
- **Arturo:**
  - One make with a recorded substitution: **rolled oats for oat bran, pound for pound**, with the reason said out loud.
  - On a chocolate-chip batch, **1,000 CT chips in place of 4,000 CT** as a substitution (4,000 CT is the standard; decision 19).
  - Record at least three entries by **voice** (hold to talk, read the transcript, fix it, Send) and three by typing.
- **Michael:** Read the substitution receipts.
- **✓ day passes when:**
  - Each substitution receipt names the substitute product, its lot and the reason.
  - Pounds are the same as the recipe line.
  - Nothing was sent without Arturo pressing Send.
- ⚠ **Not yet proven:** FL accepts substitutions and the Assistant can pass them, but the substitution has never been tested end to end through the Assistant. **If the draft card does not show the substitute and its lot, do not press Record.** Use the 🟡 "Record an entry" make form (PR #107), which has substitution fields. Log it as a problem.

**Day 4 · Thu Nov 5 — shipping with BOL and photo**
- **Arturo:**
  - Ship one real order: BOL number in the draft, BOL photo attached, Record.
  - Ship a second one with the BOL number but no photo, then attach the BOL photo later the same day.
- **Luz:**
  - Ship one order from the office (office may ship).
  - Try a ship dated yesterday. FL must refuse it: only owner or floor may confirm a "not same day" ship.
- **✓ day passes when:**
  - No ship posted without a BOL number.
  - Every ship has its BOL photo by 23:59.
  - Next morning Michael sees **no** "BOL photo missing" item. If a photo was left out on purpose, he sees exactly one.

**Day 5 · Fri Nov 6 — corrections with reasons + CHECKPOINT**
- **Arturo:**
  - One small correction with a reason from the fixed list of 8 (e.g. Damage/disposal).
  - One "found" stock with a reason.
  - One correction with reason **Unknown**: FL must ask for a note.
  - One correction **over 500 lb with a photo**: it posts.
  - One correction **over 500 lb without a photo**: it shows "Waiting for owner approval".
- **Michael:** Approve or reject the held correction from his own device (🟡 Exceptions page, PR #107). Arturo then taps **Check approval** and sees the receipt.
- **Checkpoint (Michael, end of day):**
  - Days 1–5 all passed or have a fix date.
  - No problem in the "stop" class is open (§7).
  - PR #107 is on staging.
  - The OpenAI test key is revoked.
  - If red: apply the §10.2 item 9 "simplify" set and/or move A8, and write down what moved.
- **✓ day passes when:**
  - Every correction has one of the 8 reasons.
  - Unknown has a note.
  - The held correction posted only after approval, under the same receipt number.

### Week 2: five confirmed shift summaries in a row (Nov 9–13)

From today every day must end with a **confirmed** shift summary, and every discrepancy must have a written explanation. A day with "Something is missing" that nobody can explain breaks the run of five.

**Day 6 · Mon Nov 9 — shortages**
- **Arturo:**
  - Record a make that uses more than staging has. The draft warns of a shortage; Record anyway: the make posts and a shortage opens, due in 2 business days.
  - Try to "fix" it by adding stock with a correction. FL must refuse it ("resolve the shortage instead").
  - Resolve the shortage (counted, or the missing receive entered and linked) in the 🟡 Exceptions page (PR #107).
- **✓ day passes when:**
  - The make was not blocked.
  - The add-stock attempt was refused.
  - The shortage was closed with a reason inside 2 business days.
  - The summary is confirmed.

**Day 7 · Tue Nov 10 — extra-kosher hold (gate (e))**
- **Arturo:**
  - Record one **Extra-Kosher Classic #9** make (after Michael turns on the oven). It shows "Waiting for owner approval".
  - Then pack an extra-kosher finished product from that lot.
  - Try to pack an extra-kosher finished product from a **regular** #9 lot. FL must refuse it.
  - Keep the paper extra-kosher log (date + lot) as today.
- **Michael:** Approve the held make from **his own** signed-in device (🟡 Exceptions page, PR #107). Never on Arturo's tablet.
- **✓ day passes when:**
  - The make posted only after Michael's approval, with Michael shown as the kosher approver.
  - The pack shows EXTRA-KOSHER.
  - The regular-lot pack was refused.
  - Every receive this day has a real supplier.
  - The summary is confirmed.

**Day 8 · Wed Nov 11 — unidentified lots**
- **Arturo:** Receive one real delivery whose bag has no readable supplier lot (supplier lot "N/A"). It posts, flagged UNIDENTIFIED, due in 7 business days.
- **Luz** (or Arturo): Find the real supplier lot from the packing slip or supplier and enter it on the 🟡 Exceptions page (PR #107) → supplier lot, reason **Data-entry error** or **Wrong lot used**.
- **✓ day passes when:**
  - The flag shows on the summary.
  - Entering the real code closed it.
  - The summary is confirmed.

**Day 9 · Thu Nov 12 — paper and fallback drill**
- **Arturo:**
  - For one hour, write events on paper as if FL were down (§9 of his guide). Enter them the same shift with the time from the paper.
  - Look up one entry from yesterday by asking the Assistant.
- **Luz:**
  - Enter one of Arturo's paper events on the 🟡 "Record an entry" dashboard form (PR #107), as the fallback operator.
  - Check its receipt.
- **✓ day passes when:**
  - Every paper event got a receipt number written back on the paper.
  - Paper times were used.
  - Nothing was entered twice (a "possible duplicate" warning was read, not just ticked).
  - The summary is confirmed.

**Day 10 · Fri Nov 13 — a normal day, no help**
- **Arturo:** The whole shift with no script and no one standing by.
- **Miriam:** Repeats Luz's day-1 and day-4 office tests on her own key.
- **✓ day passes when:**
  - Every event has a receipt.
  - The summary is confirmed with zero unexplained discrepancies.
  - This is the 5th confirmed day in a row.

### Tests that run every day
- **Sign out on the shared tablet** at every hand-over and at end of shift.
- **NOT recorded drill:** whenever a red card appears, follow Arturo's guide §6. Log every red card.
- **FL Assistant by voice and text:** at least a few of each, every day.

## 5. Success criteria summary

| Area | Pass when |
|---|---|
| Identity | Every receipt in the pilot shows the person who entered it. Sign out ends the session, and a Record after sign-out is refused. A 180-min idle session asks to sign in again. |
| Recorded = receipt | No event counted as recorded without a receipt number. No entry posted twice (retry shows "Same entry, same receipt"). |
| Lots | Every make/pack/ship consumed lot was confirmed from the physical tag. |
| Suppliers | 100 % of receives in Nov 9–13 carry a real supplier (gate (e)). |
| Corrections | Every correction has one of the 8 reasons. Over 500 lb has a photo or was approved by Michael. |
| Shortages | None open past its 2-business-day deadline without Michael knowing. |
| Shipping | Every ship has a BOL number. Every BOL photo arrives the same day, or the miss is visible the next morning. |
| Kosher | The extra-kosher make and pack were tested with Michael's approval, and the regular-source refusal was seen (gate (e)). |
| Shift summary | Five consecutive confirmed days Nov 9–13, zero unexplained discrepancies (gate (d)). |

## 6. Readiness gate checklist — Tue Nov 17 (go / no-go)

From design §11 item 10, as amended by rev 3, 3.2 and 3.8. **All must be ✅ to go. If (d) is not met, cutover moves to Dec 4.**

**(a) Master-key rotation**
- [ ] Not a Nov 17 item since rev 3.8: rotation happens **on cutover day** (§10.1 step 2). Confirm the cutover-day runbook is ready.

**(b) Software on staging, accepted by Michael, Arturo and one office user**
- [ ] A1–A6, A7, A9, A12 and F1 on staging (merged on main today).
- [ ] **PR #107 (D1 + D2) merged and on staging** — receipt lookup, Exceptions page, "Record an entry" forms.
- [ ] G1 read-only GPT build ready for cutover day (merged as PR #111; key issue and editor pastes are cutover-day steps).
- [ ] Pilot sign-in checks passed for all four people: key exchange works, the 180-min idle end asks again, Sign out → a later Record is refused, the right person is on every receipt. (PIN items are "at PIN activation", not gate items.)
- [ ] Resolution tests (design §3.1) pass on staging, e.g. "Classic" and "chocolate chip" ask which one, never auto-pick.
- [ ] Michael, Arturo and Luz or Miriam each signed off their part of the 10-day script.

**(c) Physical count reviewed and reset approved**
- [ ] Count packet and per-area procedure ready (option A, live count, 15–30 min pause per area).
- [ ] Count-tool PR merged and the hosted-staging rehearsal repeated (FOLLOWUPS P1.21/P1.19).
- [ ] Reset approval plan ready, including the owner-approved extra-kosher stock loading.

**(d) Five consecutive confirmed shift summaries on staging, zero unexplained discrepancies**
- [ ] Nov 9–13, or a clean run of five ending by Nov 17.

**(e) Suppliers and kosher**
- [ ] Every RCV receipt in those five days has a real supplier.
- [ ] At least one extra-kosher make + pack ran with Michael's approval.
- [ ] The "regular source" pack refusal was seen.

**Also needed (from FOLLOWUPS, not in item 10)**
- [ ] Production OpenAI key on staging since Oct 28; test key revoked before Nov 6 (decision 38).
- [ ] Nightly jobs proven: the first `expire-tickets` run logged, and one `BOL_PHOTO_MISSING` opened by `exceptions-sweep` for a deliberately photo-less staging ship (P1.9, P1.14).
- [ ] No open "stop" problem in the log (§7). Every "fix before cutover" problem is fixed or has Michael's written waiver.
- [ ] Spanish: every message Arturo saw on a red card during the pilot was understood (several server messages are English-only today; see P-4).
- [ ] Decision made on how office creates **orders and expected receipts** after cutover (P-3).

## 7. How problems are reported

**On the floor (Arturo) or in the office (Luz/Miriam):**
1. **Do not try to fix it by entering something else.** A wrong entry fixed by guessing is worse than a missing entry.
2. Take a photo of the screen. **Never photograph the sign-in screen while a key is typed.**
3. Write on the problem sheet (or text Michael): the time, what you were trying to record, the receipt number, or the words "NOT recorded".
4. If it is about a real event, the real record is unchanged (the old way). Keep going.

**Michael** copies each report into the problem log below every morning and sorts it:

| Class | Meaning | Examples | Action |
|---|---|---|---|
| **STOP** | The ledger may be wrong or attributed to the wrong person | Green card but the receipt doesn't exist; one event posted twice; a receipt shows the wrong person; posted values differ from the draft card; a key or token visible in a screen, log or photo | Pause that entry type for everyone. Engineering fixes the same day. The day does not count toward the five. |
| **FIX BEFORE CUTOVER** | Blocks a real event with no workaround, or confuses staff | A refusal Arturo can't understand; Spanish missing; a summary row is wrong; a needed screen is missing | Fix before Nov 17, or written waiver from Michael. |
| **LATER** | Annoying but safe | Wording, speed, extra taps | After cutover. |

**Problem log** (keep in the PR thread or a copy of this table):

| # | Date/time | Who | What happened | Receipt # / NOT recorded | Class | Owner | Fixed (date / PR) |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |

## 8. Open points for Michael (safest option chosen; change if you disagree)

| # | Question | What this plan assumes |
|---|---|---|
| P-1 | Do real events still go through the current process during the pilot? | **Yes.** The Floor GPT stays the real record until cutover. Staging entries are practice copies of the same events. |
| P-2 | Should staging get opening stock so makes don't always create shortages? | **Recommended:** engineering loads a clearly labelled staging stock fixture (the same `STG-` labelling used in earlier acceptance) before Nov 2, staging only, with your approval. Never entered by Arturo. Until then a staging shortage is expected and is noted in the summary, not counted as a discrepancy. |
| P-3 | How does the office create orders and expected receipts *with tickets*? | FL has the ticket routes (A7), but **⛔ no screen uses them**: the dashboard Sales Orders and Expected Receipts tabs still use the old direct path, and PR #107 has no order or expected-receipt form. **For the pilot the office keeps using the existing tabs.** A screen, or a decision to keep the old tabs until A10 stage 2, is needed before cutover. |
| P-4 | Spanish | The Assistant and shift summary are bilingual. **Several server messages are English-only** (session ended, wrong key, too many sign-in tries, output-SKU confirmation) and the dashboard sign-in bar is English-only. The guides translate them. Fix or accept before Nov 17. |
| P-5 | Undo a wrong entry (void) | FL has a void ticket route, but **⛔ no screen** uses it. For the pilot: tell Michael the receipt number. A wrong quantity can be corrected with reason *Data-entry error*. |
| P-6 | Shift summary page sign-in | That page still takes the **personal key** directly (kept in that browser tab only), not the session. Arturo types his key a second time at end of shift (FOLLOWUPS P1.21(e)). |
| P-7 | Weekly view | **⛔ Not built** (planned Nov 23–27). During the pilot, Michael's daily check replaces it. |

## 9. What is not part of the pilot

- Sunshine bulk bins and invoicing (A8): not built. Sunshine ships stay standalone ship + paper bin log.
- Orders/expected receipts via tickets: see P-3.
- MCP plugin, PIN login, QR badges: off.
- Production: no pilot entry touches production. Cutover-day steps are in design §10.1 and `gpt-configs/readonly/CUTOVER_CHECKLIST.md`.

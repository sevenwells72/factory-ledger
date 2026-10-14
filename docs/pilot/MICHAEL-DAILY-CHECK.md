# Michael — 10-minute daily check (pilot, Nov 2 – Nov 20)

Do this every morning, about yesterday, on **your own** device, signed in as **you**. Never use Arturo's tablet: an approval records the signed-in person.
Staging pages are listed in [PILOT-PLAN.md §2](PILOT-PLAN.md#2-how-the-pilot-works-read-this-first).
🟡 = needs PR #107 (Exceptions page, receipt page) on staging. Until it is there, the fallback is in *italics*.

---

## 1. Approvals — 2 min

🟡 **Exceptions page** → filter on items waiting for you.

| Item | What to do |
|---|---|
| **Large correction** (over 500 lb without a photo) | Look at the draft. Ask Arturo for the photo or the reason. **Approve** → it posts under the same receipt number, recorded as entered by Arturo and approved by you. **Reject** → note required, nothing posts. |
| **Kosher approval** (held Extra-Kosher make or found) | Approve only if you turned on the oven for that batch (or the found stock is truly extra-kosher). Your approval **is** the kosher attestation. The hold expires after 7 days but stays on the list until you reject it. |
| Late entry (48 h – 14 days late) | Acknowledge, or ask why. |
| BOL photo missing | Get the photo uploaded, or waive with a note (owner only). |

*Without #107: there is no approve screen on main. Approval is possible only through the API (`POST /exceptions/{id}/approve` or `/reject`, owner key or session). Do not ask engineering to approve for you: the approval must be your own identity. Leave the item held and move the test day.*

## 2. Open exceptions — 2 min

- **Shortages:** any open past **2 business days**? Make sure Arturo has a plan today. Overdue items are marked escalated by the nightly sweep (02:30 ET).
- **Unidentified lots:** any past **7 business days**?
- **Shift discrepancy:** read Arturo's note. Explained → write the explanation in the problem log. Not explained → that day does **not** count toward the five.
- **"Pre-make adjust"** (stock added shortly before a make): ask Arturo why. This is the "never add stock to make it go through" check.

*Without #107: shift summary page → **View all actors (read-only)** → OPEN EXCEPTIONS shows every open item, read-only.*

## 3. Yesterday's shift summary — 2 min

Shift summary page → date = yesterday → **View all actors (read-only)**, or sign in with your key to see your own.

- [ ] Arturo confirmed (banner shows who, when and the outcome). Not confirmed = the run of five breaks (gate (d)).
- [ ] Every **RCV** has a real supplier (no "supplier missing" flag; gate (e)).
- [ ] **NOT ENTERED YET?** is empty or explained.
- [ ] Any red line (short or negative lot, overdue) is known.
- [ ] Count toward the gate: **day __ of 5**.

## 4. Late entries and corrections — 2 min

- **LATE ENTRIES** section in the summary: what was entered late, and why? Over 48 hours late opens a Late entry item for you.
- Corrections on yesterday's receipts: every one has one of the 8 reasons; **Unknown** always has a note; anything over 500 lb has a photo or your approval.
- Spot-check **two receipts** against reality (paper, the bag tag, the BOL). 🟡 Receipt box → receipt page. *Without #107: ask the FL Assistant "receipt MK-…".*

## 5. Problem log — 2 min

Copy the floor and office reports into the log ([PILOT-PLAN.md §7](PILOT-PLAN.md#7-how-problems-are-reported)) and class each one **STOP / FIX BEFORE CUTOVER / LATER**.

## What to escalate (same day, to engineering)

**Stop that entry type and escalate at once if:**
- A green receipt number does not exist when you look it up.
- One real event appears twice (two receipts), or a retry made a second post.
- A receipt shows the wrong person.
- Posted values differ from what the draft card showed.
- Something posted that should have been refused: no BOL, unconfirmed lot, an office make, an Extra-Kosher make without your approval, a regular-lot pack into an Extra-Kosher product.
- A key or session token is visible anywhere: screen, log, photo, PR. Then **revoke that key** (a new key for that person ends all their sessions at once).

**Escalate the same day (fix before cutover) if:**
- Arturo or Luz cannot record a real event and there is no workaround.
- A message they could not understand (record the exact text; several server messages are English-only).
- The Assistant is down for more than one hour (OpenAI outage or the $20 cap reached: check the OpenAI alert e-mails at 50 % / 80 %).

**Do not escalate** wording or speed issues during the run: log them as LATER.

## Friday add-ons

- **Fri Nov 6 — checkpoint:** see [PILOT-PLAN.md day 5](PILOT-PLAN.md#4-the-10-day-test-script). Confirm the OpenAI test key is revoked.
- **Fri Nov 13:** count the confirmed days. If five are clean, prepare the Nov 17 gate ([PILOT-PLAN.md §6](PILOT-PLAN.md#6-readiness-gate-checklist--tue-nov-17-go--no-go)). If not, plan the Nov 16 spare day or the Dec 4 fallback.

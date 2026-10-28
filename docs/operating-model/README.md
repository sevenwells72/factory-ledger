# Factory Ledger operating model review (2026-10-28)

Read-only review of how Factory Ledger is built, what production actually runs today, and how Arturo, Luz/Liz and Michael should use it. Nothing in production, the code, the GPTs or any setting was changed to produce this.

## Documents

| | Document | What it answers |
|---|---|---|
| A | This page | Executive summary and recommended operating model |
| B | [01-architecture.md](01-architecture.md) | Every component, what it does, how data flows (diagram) |
| C | [02-capability-matrix.md](02-capability-matrix.md) | Which interface can do each task, today and after switch-on |
| D | [03-guide-arturo.md](03-guide-arturo.md) | Plant manager daily guide |
| D | [04-guide-liz.md](04-guide-liz.md) | Office daily guide |
| D | [05-guide-michael.md](05-guide-michael.md) | Owner daily and weekly guide |
| E | [06-access-reference.md](06-access-reference.md) | Production and staging links, accounts, sign-in |
| F | [07-issues-and-recommendations.md](07-issues-and-recommendations.md) | Prioritised gaps, risks, duplication, interface overlap analysis |
| G | [08-standard-operating-procedures.md](08-standard-operating-procedures.md) | House rules for using Factory Ledger consistently |
| H | [09-switch-on-plan.md](09-switch-on-plan.md) | Step-by-step plan to switch on personal sign-in, FL Assistant and read-only GPTs (draft for approval) |

## How this was verified

- **Code:** the exact code production runs (GitHub `main` at commit `9a9a9536`, Railway deployment `9a01f022`, 2026-10-28 12:39 ET). The local checkout in this folder is older (`0fa518f6`) and was not used as the source of truth.
- **Live production (read-only):** health checks, page loads and feature switches on the production and staging servers. Read-only database queries (each wrapped in `BEGIN TRANSACTION READ ONLY`) covered accounts, roles, recent entries, exceptions, drafts and open orders. No passwords, keys or PINs were read out or written down.
- **Documents:** the change logs, the deployment plans (including `FREEZE-LIFT-PLAN.md` in the count-day worktree), the design documents and the open pull requests.
- **Not verified:** anything that needs a person signed in to a GPT editor, to ChatGPT, or to Netlify or Railway settings screens. That covers what is pasted in each GPT and the exact production environment-variable values beyond what the servers report about themselves. These are marked **Unverified** wherever they come up.

---

## A. Executive summary

### How Factory Ledger is organised

Factory Ledger is **one database with one back-end server**. Everything else is a window onto it:

- **Database (Supabase):** the single record of inventory, lots, orders, receipts and who did what.
- **Back-end server (Railway):** the only thing that talks to the database. It applies every business rule: lot tracking, supplier-lot requirements, kosher rules, roles, duplicate checks and owner approvals.
- **Four ways in:**
  1. **Dashboard** (Netlify website): screens for reviewing everything, plus entry and office forms.
  2. **FL Assistant:** a chat page inside the dashboard. You describe what happened; it drafts the entry; you press **Record**.
  3. **Floor GPT** ("Factory Ledger - Floor 2.0" in ChatGPT): chat for floor work.
  4. **Office GPT** ("Factory-Ledger 2.0" in ChatGPT): chat for office work, including sales orders.

So yes, they are **different windows onto the same records**. A shipment recorded in any of them lands in the same ledger and shows on the same dashboard. But they are **not equal**. They have different capabilities, different safety checks and, most importantly, **different identity**:

| | Who it records as | Safety level |
|---|---|---|
| Dashboard forms and FL Assistant | **The named person** (personal key) | Full: draft → Record, role limits, 500 lb photo or owner-approval rule, duplicate warnings |
| Floor GPT and Office GPT | **"legacy-shared-key"**, which names no one | Partial: core lot, kosher and supplier-lot rules apply, but role limits, the large-adjustment approval rule and most duplicate warnings do not |
| Dashboard tabs while sign-in is off (production today) | Shared dashboard key, which names no one | Order and expected-receipt edits only; no inventory entry buttons shown |

### The single most important finding

**In production today, only the two GPTs can record inventory entries (receive, make, pack, ship, adjust, void).** Two switches were never turned on in production (they are on in staging):

- **FL Assistant is off:** production answers "FL Assistant is not enabled" to every Assistant request.
- **Personal sign-in ("key sessions") is off:** with it off, the dashboard hides its **Record entry**, **Office** and **Exceptions** screens.

The main dashboard tabs **can still save office changes**, and they save them under a shared "dashboard" identity that names no one. That covers new sales orders from a PO photo, order edits, close and cancel, expected receipts, production runs, notes and supply requests.

The production database agrees with this. **Arturo's and Luz's accounts have never been used.** Every routine receive, make, pack and ship entry since August was recorded under the shared GPT identity. The only named entries are Michael's ("Blubber") reset and count corrections. There is also **no "Liz" account; the office accounts are "Luz" and "Miriam"**. I assume Liz is Luz, but please confirm.

### Recommended operating model (my judgement)

One primary tool per person, all on personal sign-in:

| Person | Primary | Backup | Why |
|---|---|---|---|
| **Arturo** (plant) | **FL Assistant** on his phone (Spanish or English, voice or typing, photos) | **Record entry** screen in the dashboard (same sign-in, same records) | Fastest on a phone; every entry carries his name; nothing posts until he presses Record |
| **Luz/Liz** (office) | **Dashboard**: Sales Orders, Expected Receipts and Activity tabs, plus the **Office** screen for changes | FL Assistant for creating an order from a quick message | Office work is review-heavy and needs tables, filters and exports, not chat |
| **Michael** (owner) | **Dashboard**: Operations "Needs Attention", **Exceptions** for approvals, History and Traceability | Office GPT or FL Assistant for ad-hoc questions | Monitoring and approving; almost no routine entry |

The GPTs become **read-only, then retire**, as the October design already planned. That design was never executed: the read-only versions are written and the server supports them, but they were never pasted in. Today both GPTs are fully write-capable and share one master key, which has been exposed publicly.

**This model needs a short, deliberate switch-on.** Until then, use the interim model in each guide: Arturo uses the Floor GPT, Luz the Office GPT plus the read-only dashboard. Switch-on is a handful of configuration steps, each needing your approval (see [07](07-issues-and-recommendations.md) §1). It is not new development.

### Top next steps

1. **Decide the switch-on date.** Then, as approved steps, in this order:
   1. turn on personal sign-in and the FL Assistant in production;
   2. confirm the OpenAI key and spending cap;
   3. hand Arturo and Luz their personal keys;
   4. run a 1-day supervised pilot with one entry of each type.
2. **The same week, make the GPTs read-only:** paste the prepared read-only versions and issue the read-only key. **Then rotate the master key**, which has been exposed publicly.
3. **Fix four small things that block the guides** (each a small change, see [07](07-issues-and-recommendations.md)):
   - a visible "PRODUCTION / STAGING" banner;
   - a menu link to FL Assistant and the Shift Summary page;
   - a "Packing slip" button on the sales order;
   - showing the packing-slip link on the Assistant's shipment receipt.
4. **Settle the open count corrections:** #9 Regular lot split, and confirming Almond Vanilla and PB Banana. Then resolve open exceptions 66 and 67.
5. **Confirm one oddity:** production has **no receive, make, pack or ship entries after 2026-10-23**. If the plant has produced or shipped since then, that work is not yet in Factory Ledger.

# F. Issues, overlap analysis and recommendations

Priority:

- **P1** blocks the recommended workflows, or risks wrong or unattributed inventory.
- **P2** causes confusion or extra work.
- **P3** is nice to have.

Every fix marked "config" or "small" is a change that needs your separate approval. None of it was done in this review.

---

## 1. What it takes to switch on the recommended model

No new development is needed for the core switch-on. In order, each step approved separately:

1. **Staging dress rehearsal with the real people.** Arturo and Luz each do one entry of every type on staging, on their own phone or PC.
   - There is no record of staff ever using staging themselves. The October walkthroughs and the "Round E" checks were run by Michael or by scripts.
2. **Production readiness check.**
   - Run the sign-in settings checker from draft PR #127. On Oct 15 it reported a missing proxy rule and a missing OpenAI credential for production.
   - Confirm the Assistant's database tables are present.
   - Confirm a **separate OpenAI key with a monthly cap** (FOLLOWUPS P1.18), and revoke the old proof-of-concept key before Nov 6.
3. **Turn on personal sign-in** in production. This reveals Record entry, Office, Exceptions, Receipt void and Shift Summary confirm.
4. **Turn on the FL Assistant** in production.
5. **Hand out personal keys** to Arturo and Luz, plus Liz if she is a different person.
6. **Supervised first day:** one receive, one make, one pack, one ship (this also checks the #146 slip link and the #153 one-row-per-shipment guard), one Found and one Adjust under 500 lb. Michael approves one held correction.
7. **Same week: make both GPTs read-only.**
   - Set the read-only key.
   - Paste `gpt-configs/readonly/` into both GPTs (click-by-click instructions in `CUTOVER_CHECKLIST.md`).
8. **Rotate the master key** once nothing depends on it.

---

## 2. Issues (prioritised)

### P1: blocks the workflows or risks the records

| # | Issue | Evidence | Recommendation |
|---|---|---|---|
| F-1 | **Personal sign-in and FL Assistant are off in production.** The dashboard's entry, office and exception screens are hidden, and the Assistant refuses every request. | Production `/auth/session/config` → `key_sessions_enabled:false`; Assistant routes refuse; 0 sign-in sessions, 0 Assistant chats; Arturo, Luz and Miriam have never been used | Follow §1 above. **Config, not development.** |
| F-2 | **Most entries name no one.** Both GPTs use the shared master key and record "legacy-shared-key". Dashboard tab edits use the shared dashboard key. | 235 shared-key entries in 30 days; only Michael's 142 are named | The switch-on, then read-only GPTs (§1 steps 3–7) |
| F-3 | **The GPTs bypass safeguards.** They use the older direct entry routes, which do not enforce: role limits, the 500 lb photo or owner-approval rule, the late-entry exception, or the must-acknowledge duplicate warnings. "Preview before commit" depends only on the GPT following its instructions. The Office GPT's instructions even say "no reconfirmation" for orders. | Backend permission code (shared keys are not role-gated); GPT instructions | Read-only GPTs (step 7). Later, retire the direct routes (planned as "A10"). |
| F-4 | **The master key has been exposed** (Aug 13 GPT instructions and repository history), and both GPTs still use it. The live Office GPT instructions may still contain a key in the static packing-slip link (repo copy redacted; live copy **unverified**). | FACTORY_LEDGER_CHANGELOG row 230; design §10 | Rotate right after step 7. Don't wait "two weeks after stable", because stable operation never depends on this key once the GPTs are read-only. |
| F-5 | **No receive, make, pack or ship entries in production after Oct 23.** The only entries since are Michael's count and reset corrections. | Database, read-only | Confirm whether the plant ran Oct 26–28. If it did, enter the missing work (with late-entry rules) before anything else, or inventory is already off. |
| F-6 | **Packing slips have no reliable path.** The dashboard has no Packing slip button. The Assistant creates a link but doesn't show it. The Floor GPT's 10-minute link is live but has never been seen on a real shipment. The Office GPT's link is static and contains a key. | Dashboard and Assistant code; #146 notes | **Small:** add a "Packing slip" button on the sales-order detail (the back end already has the page), and show the link on the Assistant's shipment receipt. |
| F-7 | **No production/staging banner.** Staging is reached by any Netlify preview link and looks identical. | Dashboard code | **Small:** a red "STAGING — test data" bar whenever the page is not on the production address. |
| F-8 | **Found inventory is impossible through either GPT**, although both GPTs' instructions describe it. Today a Found entry needs Michael and an engineering session. | Both GPT schemas | Goes away with the switch-on (Assistant and Record entry both support Found). Interim: tell staff to send Found items to Michael. |

### P2: confusion, extra work, accountability

| # | Issue | Recommendation |
|---|---|---|
| F-9 | **"Liz" vs "Luz".** There is no Liz account; Luz and Miriam are office accounts. | Confirm. If Liz is a separate person, create her an office account. Deactivate any account nobody uses (Miriam?). |
| F-10 | **Hidden pages.** FL Assistant is reachable only from a small "FL" link on Record entry and Office. Shift Summary is linked from nowhere. The Assistant's fallback buttons point at `/dashboard/entry.html`, which likely doesn't exist on Netlify (**unverified**). | **Small:** add FL Assistant and Shift Summary to the menu and the mobile bar; fix the fallback links. Meanwhile, use phone home-screen bookmarks. |
| F-11 | **No physical count feature.** Counts are entered as Adjust "physical count", one lot at a time. Count day needed a Google Form plus owner-loaded corrections. | Fine for spot counts. For the next full count, decide in advance (FOLLOWUPS P1.21 asks for an owner-approved count-loading path). Don't build it until a second full count is planned. |
| F-12 | **The production plan is not in Factory Ledger.** The schedule is empty and the Runs page has never been used. The "production board" scheduler is an offline simulator with July sample data. | Decide: either plan in **Runs** (live, links runs to order lines) or keep planning outside Factory Ledger. Hide the offline scheduler to avoid confusion. |
| F-13 | **Owner alerts are pull-only.** Nothing emails or texts you; you must open Exceptions. Corrections over 500 lb that include a photo post without approval (only highlighted). | Daily habit (guide D3). Decide whether over-500 lb-with-photo should also wait for you. |
| F-14 | **Training material is unmerged.** Bilingual staff guides and quick cards are in draft PR #113. The cutover runbook is in draft PR #117. | Review and merge #113 (docs only) after updating it to this model. |
| F-15 | **Open data questions.** #9 Regular (7,883 vs 1,682 lb), Almond Vanilla / PB Banana product confirmation, Found lots 66/67 unidentified, K&K 5 cases unexplained, sprinkles lot split. Coconut Flake FOUND-011 is pooled (159,250 lb). | Owner decisions listed in guide D3. |
| F-16 | **Database connection reliability.** #138 merged with open review blockers; follow-up draft #141 addresses them. | Review #141 before more users are on the dashboard at once. |
| F-17 | **Stale documents can mislead.** STATUS.md lists old GPT versions; the regression changelog still marks #142–#154 "not merged / not deployed"; CONTEXT.md and DEPLOYMENT.md are from August. | Update the regression changelog rows (required by CLAUDE.md after deploys). Mark STATUS, CONTEXT and DEPLOYMENT as historical, or refresh them. |

### P3

| # | Issue | Recommendation |
|---|---|---|
| F-18 | **QuickBooks is manual.** There is no connection; EULA and Privacy pages exist for a possible Intuit app. | Weekly export-and-compare is enough at this volume. Revisit only if invoicing errors appear. |
| F-19 | **Office GPT is at its 30-action cap**, and order writes have no preview. | Irrelevant once it is read-only. |
| F-20 | **PIN sign-in** is dormant (PR #96). | Turn it on only if pasting a key once per shift proves too hard on the floor. |

---

## 3. Overlap analysis (Part 6)

**Why both custom GPTs and FL Assistant?**
- **Historically:** the GPTs came first (spring 2026).
- **Why the Assistant was built** (Sept to Oct, F1): the GPTs cannot know who is talking, cannot be held to "nothing posts until you press Record", sometimes claimed success without recording (the April fabrication audit), are capped at 30 actions and 8,000 characters of instructions, and live outside your control in ChatGPT.
- **The plan was always replacement:** the GPTs become read-only at cutover, then retire.
- There is **no lasting reason to keep GPTs for writing**.

**What FL Assistant can do that the GPTs can't:**
- name the person;
- enforce Record;
- enforce roles and owner approval;
- record Found inventory;
- attach BOL, lot-tag and slip photos;
- take Spanish voice dictation;
- show must-acknowledge duplicate warnings;
- create customers and orders with duplicate-PO checks;
- show "Today's entries".

**What the GPTs can do that FL Assistant can't:**
- void from chat (the Assistant hands off to the Receipt page);
- create expected receipts from chat (the Assistant hands off to Office);
- forward and backward lot trace in chat (the Office GPT);
- answer free-form questions across data.

All of these have a dashboard equivalent. The last one is what a **read-only** GPT or connector is for.

**The dashboard's unique purpose:**
- **seeing the whole picture**: tables, tabs, filters, exports, traceability, history, exceptions;
- **office forms** (orders, expected receipts, document intake);
- **owner approvals**.

Chat is poor at all three.

**Is there unnecessary duplication?** Yes:
- **Writing through the GPTs duplicates the Assistant and dashboard, with weaker controls.** That is the duplication to remove.
- The **Assistant and the Record entry form** overlap on purpose: same sign-in, same drafts, same receipts. One is the fast chat, the other a dependable form fallback. Keep both.

**Can the same transaction be recorded twice through two interfaces?** Yes, today:
- Arturo records a receipt in the Floor GPT and Luz records it again in the Office GPT: nothing stops it. Direct-route entries only have a duplicate guard for order shipments (#153).
- On the draft route, a second similar receive, make, pack, ship or found triggers a must-acknowledge "possible duplicate" warning. That warning compares against the ledger, so it should also catch an earlier GPT entry (**likely, not tested across interfaces ❓**).
- **Safeguard in practice:** one owner per kind of entry (see the SOPs in [08](08-standard-operating-procedures.md)), plus read-only GPTs.

**Do all interfaces enforce the same validation, permissions, lot tracking and approvals?**
- **Lot tracking and core validation:** yes. That includes supplier lot required, kosher, pooled lot, no shipping beyond stock, shortages and unidentified-lot exceptions.
- **Permissions and approvals:** no, the GPTs bypass them (F-3).

**Start in one interface, finish in another?**
- **Not possible, and that is safe.**
- A GPT "preview" records nothing.
- An Assistant or dashboard draft belongs to the person who made it and expires after 10 minutes (30 on the dashboard), so it can't be finished elsewhere or by someone else.
- If you switch interfaces mid-entry, simply start again in the new one.
- The only risk is the opposite: finishing in both. Hence one interface per person.

**Same person, several interfaces?**
- Using the **Assistant and the dashboard together is safe**: same identity, same drafts, same duplicate warnings.
- Mixing them with a **writing GPT** loses the name and the warnings. After the GPTs are read-only, this problem disappears.

---

## 4. Plugin / connector migration (the GPT replacement)

- **Planned:** a Factory Ledger connector ("M0", draft PR #114) for ChatGPT and Claude, **read-only**, with each person signing in with Google. Writes through it come later (M1–M3), if ever. Target: before the GPTs retire (design date Dec 11).
- **Effect on the recommended model: small**, provided **staff entries already go through the FL Assistant and dashboard** by then. Then retiring or migrating the GPTs removes only a lookup convenience; no floor or office workflow depends on it.
- **The risk is the opposite order:** if staff are still writing through the GPTs when they are retired, there is a hard deadline with no tested replacement in use.
- **Recommendation:**
  - Do **not** build connector writes for staff.
  - Use the read-only connector for Michael's ad-hoc questions.
  - Treat the FL Assistant as the only chat that writes.

---

## 5. Recommended future operating model (Part 9, my judgement)

**Keep:**
1. **Dashboard**: everyone's place to look; Luz's and Michael's primary tool.
2. **FL Assistant**: Arturo's primary tool, with the **Record entry form** as its built-in fallback.
3. **One read-only chat** (read-only GPT now, the connector later), for questions only.

**Retire as write tools:** both custom GPTs, as soon as switch-on is proven. Don't keep "both, just in case": that is exactly what creates unnamed duplicates.

**Primary and backup per person:**

| Person | Primary | Backup | Never |
|---|---|---|---|
| Arturo | FL Assistant (phone) | Record entry form | Writing GPTs after cutover |
| Luz/Liz | Dashboard + Office screen | FL Assistant for a quick order | Office GPT for adjustments, ever |
| Michael | Dashboard (Needs Attention, Exceptions, History) | Read-only chat | Routine entries |

**Does the current architecture create unnecessary complexity?**
- The **back end and database are sound and simple**: one ledger with strong rules.
- **The complexity is in the doors:** five ways in (two GPTs, the Assistant, dashboard forms, dashboard tabs on a shared key) with three different identity models.
- Collapsing to **one identity model (personal sign-in) and two doors** (the Assistant for floor chat, the dashboard for everything else) removes most of it.

**Standardise before full staff adoption:**
- one account per person;
- the switch-on (§1);
- read-only GPTs and the master-key rotation;
- a staging banner;
- menu links for the Assistant and Shift Summary;
- a packing-slip button;
- the SOP rules in [08](08-standard-operating-procedures.md);
- a one-page bilingual card for Arturo (adapt PR #113).

**Greatest benefit for least work:**
1. Switch-on plus read-only GPTs. This gives names on every entry, owner approvals that really apply, and duplicate warnings.
2. Packing-slip button and Assistant slip link.
3. Staging banner and menu links.
4. Daily Shift Summary sign-off, with a 15-minute owner review.

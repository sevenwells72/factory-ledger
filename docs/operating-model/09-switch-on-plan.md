# H. Switch-on plan — personal sign-in, FL Assistant, read-only GPTs (DRAFT for owner approval)

Drafted 2026-10-28. **Nothing in this plan has been done.**

- Every step marked ★ changes production. It needs Michael's explicit approval for that step, at the time it runs.
- One step at a time: the next starts only after the previous one is checked.

**Builds on:**
- the Oct 15 sign-in settings plan (draft PR #127, `docs/O5-PROD-SIGNIN.md`);
- the read-only GPT checklist (`gpt-configs/readonly/CUTOVER_CHECKLIST.md`);
- the design's cutover list (`docs/design/phase1-safe-operating-system.md` §10.1).

This plan puts them in one order and adds what changed since Oct 15.

## What this achieves

| Before | After |
|---|---|
| Staff entries go through two writing GPTs under one shared, publicly exposed key; nobody is named | Arturo records in **FL Assistant** (backup: Record entry form); Luz uses the **dashboard + Office**; every entry names the person |
| Role limits, the 500 lb owner-approval rule and duplicate warnings are skipped by GPT entries | They apply to every staff entry |
| The Exceptions page is hidden; owner approvals need an engineering session | Michael approves on the Exceptions page |
| Both GPTs can write | Both GPTs answer questions only |

## Already verified (read-only, 2026-10-28)

| Prerequisite | Status |
|---|---|
| Production back end is current `main` (9a9a9536), healthy | ✅ |
| Sign-in tables, Assistant tables and correction reasons exist in the production database | ✅ |
| Accounts Blubber (owner), Arturo (floor), Luz (office), Miriam (office) are active and each has a key on file | ✅ (whether Arturo and Luz *have* their keys is unknown) |
| Duplicate-shipment guard table (#153) exists | ✅ |
| Personal sign-in switch | Off (to turn on in step G2) |
| FL Assistant switch | Off (step G3) |
| OpenAI key for the Assistant on production | ❌ missing on Oct 15 (step P3); not re-checked |
| Proxy setting so each phone gets its own sign-in limit (`FORWARDED_ALLOW_IPS`) | ❌ missing on Oct 15 (step P4); not re-checked |
| Packing-slip path for staff once the GPTs are read-only | ❌ none (step P2) |

## Decisions Michael makes first

| # | Decision | My recommendation |
|---|---|---|
| D1 | Go-live day | A normal mid-week production day with a shipment, and Michael available all morning. Not a Friday. |
| D2 | Is Liz the same person as "Luz"? | If not: mint her own office account in step P6. Never share Luz's key. |
| D3 | Keep Miriam's office account? | Deactivate it if she won't use Factory Ledger (step P6). Unused active accounts are risk with no benefit. |
| D4 | Order of the GPT switch | **Turn on sign-in and the Assistant first (Day 1).** Make the GPTs read-only and rotate the master key **within two working days**, once Day 1 is clean. This keeps a familiar fallback for the first day. The alternative, all on one day, is simpler but leaves no fallback if Day 1 goes badly. |
| D5 | OpenAI spending | A separate OpenAI project for the Assistant with a hard $20/month cap and 50% / 80% alerts (as already decided Oct 12). |
| D6 | Corrections over 500 lb **with** a photo | Leave as is for go-live (highlighted, not held). Revisit after two weeks of real use. |

---

## Phase P — Preparation (no change for staff)

| Step | What | Who | Check | Rollback |
|---|---|---|---|---|
| **P1** | **Confirm the Oct 24–28 gap.** Production has no receive, make, pack or ship entries after Oct 23. If the plant worked, list the missing work and enter it **before** go-live, through the GPTs as today. Michael approves the late-entry exceptions afterwards. | Arturo, Luz → Michael | History shows those days complete | n/a |
| **P2** ★ | **Small code change, one PR, staging first:** (a) a **Packing slip** button on the dashboard sales order (the back end already generates the PDF; it needs adding to the routes the dashboard may call); (b) show the packing-slip link on the Assistant's shipment receipt; (c) a red **STAGING** banner on any non-production address; (d) **FL Assistant** and **Shift Summary** in the menu and the mobile bar; fix the Assistant's fallback links. Full test suite, staging check, then merge. **Merging deploys production automatically**, so this is the approval. | Engineering → Michael approves the merge | On production: a real order's Packing slip opens a PDF; no banner on `cns-factory-ledger.netlify.app`; banner on a preview address | Revert the PR (auto-deploys) |
| **P3** ★ | **OpenAI key.** Michael creates the separate OpenAI project with its cap and alerts, and makes a key. On Railway → *gleaming-solace* → **FastAPI** (not FastAPI-staging) → Variables, add `OPENAI_API_KEY` and deploy. The Assistant stays **off**. Revoke the old proof-of-concept key (due before Nov 6). | Michael (OpenAI console, Railway) | Deploy succeeds; `/health` healthy; Assistant still refuses (still off) | Remove the variable and deploy |
| **P4** ★ | **Proxy setting for sign-in limits.** Sign-in allows 5 failed attempts per 15 minutes per device. Without this setting every phone looks like the same device, so one person mistyping could lock everyone out. Complete the short ingress review in `O5-PROD-SIGNIN.md` ("Proxy trust"), then set `FORWARDED_ALLOW_IPS` on FastAPI as reviewed and deploy. | Engineering → Michael | Deploy succeeds; `/health` healthy | Restore the previous value and deploy |
| **P5** | **Run the read-only readiness checker** (`scripts/check_prod_signin_settings.py production --proxy-trust-reviewed`, from PR #127's branch). It prints PASS/FAIL only, never values. It needs the Railway CLI and permission to read FastAPI's variables, so Michael runs it or approves it. | Engineering | Every line PASS **except** `KEY_SESSIONS_ENABLED`, `ASSISTANT_ENABLED` and the two "deployed flag" lines (still off by design) | n/a (read-only) |
| **P6** ★ | **Personal keys.** Keys can't be recovered, and nothing shows that the Sept 14 keys ever reached Arturo or Luz. So **mint fresh keys** for Arturo and Luz (plus Liz if D2 says she is separate) with `scripts/mint_actor_keys.py`. Apply the hash-only SQL it writes in Supabase (this replaces their old keys; their account history is kept). Deactivate Miriam if D3 says so. **Do not re-mint Blubber**: engineering sessions use that key. Hand each key over **in person**, straight into the person's phone or browser password manager. | Michael | For each person, `/auth/whoami` with their key shows the right name and role (owner runs it without printing the key) | Re-mint, or `active = false` for that account |
| **P7** | **Staging rehearsal with the real people, on their own devices**, using their **staging** test keys on a preview address. The new banner must show. **Arturo, in the FL Assistant:** one receive with a lot photo, one make, one pack, one ship against `SO-STAGING-TEST` with a BOL photo, one Found, one Adjust under 500 lb, and one same-day void from the Receipt page. **Luz:** create an order from a PO PDF, edit a line, add an expected receipt, print a packing slip. **Michael:** approve one held correction (an Adjust over 500 lb with no photo) on Exceptions; confirm Shift Summary. | Arturo, Luz, Michael | Every entry has a receipt number and the right name in History. Each person can sign in and sign out unaided. | n/a (staging) |

**Gate to go live:** P1–P7 done, and P5 shows only the expected "off" failures.

---

## Phase G — Go-live day

| Step | What | Check | Rollback (one step) |
|---|---|---|---|
| **G1** ★ | **Fresh production backup**: `~/Backups/fl/fl-backup-and-verify.sh`, the same procedure used for the Oct 27–28 deploys. | Backup verifies (key checks match) | n/a |
| **G2** ★ | **Turn on personal sign-in.** Railway → FastAPI → Variables: `KEY_SESSIONS_ENABLED` = `1` (exactly) → deploy → wait for Success. | `https://fastapi-production-b73a.up.railway.app/auth/session/config` shows `"key_sessions_enabled":true`. The dashboard tool bar shows Record entry, Office and Exceptions. | Set it to `0` and deploy. Entry screens hide again; entries already recorded stay. |
| **G3** ★ | **Turn on FL Assistant.** `ASSISTANT_ENABLED` = `1` → deploy → Success. | `https://cns-factory-ledger.netlify.app/fl-assistant.html` opens. After sign-in, a question like "¿Cuánta avena tenemos?" gets an answer. | Set it to `0` and deploy |
| **G4** | **Readiness checker again:** every line PASS, exit 0. | — | If anything fails, set both switches back to `0` in one deploy |
| **G5** | **Each person signs in on their own device**, on the production address. The name and role shown must be right. Sign out, then sign back in. | Arturo = floor, Luz = office, Blubber = owner | Fix the key for that person (P6) |
| **G6** | **Supervised real entries**, recorded as they happen (not test data): the first receive, make, pack and ship of the day by Arturo in the Assistant; Luz prints that shipment's packing slip from the dashboard; any Found or Adjust as it comes up. Michael looks up each receipt number. | History shows each entry with **Arturo** or **Luz**, not "legacy-shared-key"; stock moves as expected; the packing slip prints | If the Assistant misbehaves, Arturo switches to the Record entry form (same sign-in). If sign-in itself fails, roll back G2/G3; the GPTs still work. |
| **G7** | **Announce the rule** (SOP-1, SOP-2): from now on, entries go **only** through the FL Assistant or the dashboard. The GPTs are for questions only, even though they can technically still write for the next 1–2 days. | No new "legacy-shared-key" entries in History after the announcement | — |
| **G8** | **End of Day 1:** Arturo does **Shift Summary → Confirm**; Michael reviews Exceptions and the day's History. | Shift Summary confirmed; no unexplained entries | — |

**Stop and roll back (G2/G3 to `0`)** if:
- any entry posts under the wrong name;
- a person can do something their role forbids;
- the server becomes unhealthy;
- the Assistant claims success without a receipt number.

---

## Phase R — Within two working days: GPTs read-only, master key rotated

Only after Day 1 and Day 2 are clean (no rollback, no wrong names, packing slips printed).

| Step | What | Check | Rollback |
|---|---|---|---|
| **R1** | **List everything that uses the master key** besides the two GPTs, for example engineering scripts and any automation. **Not verified in this review.** Each one must either move to a personal or owner key or be accepted as stopping. | Written list, agreed by Michael | n/a |
| **R2** ★ | **Rotate the master key and the dashboard key** (O-7 procedure, `docs/cutover/O-7-dashboard-key-rotation.md`): new values on FastAPI → deploy. The writing GPTs stop working at this moment. That is intended. | Dashboard still loads (it fetches its key from the server, no Netlify rebuild); a GPT write attempt fails | Restore the previous values **only** if the dashboard breaks. Never restore an exposed key otherwise. |
| **R3** ★ | **Issue the read-only key** (checklist part A): generate it, add `READONLY_API_KEY` on FastAPI, deploy, then run the six-line proof. | Reads 200; writes 403; session exchange 401 | Remove the variable and deploy |
| **R4** ★ | **Switch both GPTs to read-only** (checklist part B, Office then Floor): paste the read-only instructions and schema, and set the action's key to the read-only key. | Office shows 14 actions, Floor 11, none of them writes | Re-paste the read-only files. There is no return to writing GPTs, because their old key is dead. |
| **R5** | **Smoke test** (checklist part C): every read action once, plus one write attempt per GPT. | 25 reads pass; both write attempts are refused politely; Recent Entries shows nothing new | Per the checklist |
| **R6** | **Tell staff:** "ChatGPT answers questions; entries go through the FL Assistant or the dashboard." Update the bookmarks. Record the step in both change logs and the regression changelog. | — | — |

---

## Phase L — Later (separate decisions, not needed to go live)

- **Retire the old direct entry routes ("A10")**, so nothing can bypass the draft → Record step even with an engineering key. The design put this one working day after cutover; there is a prepared plan in draft PR #125.
- **Order legacy-close cutoff ("O-4")**: set the cutover date so post-cutover orders can't be closed without a recorded shipment.
- **Read-only ChatGPT/Claude connector (M0, draft PR #114)** for Michael's questions, before the GPTs are retired entirely.
- **PIN sign-in (PR #96)**, only if pasting a key once per shift proves too hard on the floor.
- **Merge the staff guides (PR #113)** after updating them to this model; refresh STATUS.md and the regression changelog rows for #142–#154.

## Time and people

| Phase | Effort | Who |
|---|---|---|
| P1 | 1 hour | Arturo, Luz, Michael |
| P2 | About 1–2 days of engineering, then review and staging check | Engineering, Michael approves |
| P3–P6 | About 2 hours in total, mostly Michael in the OpenAI, Railway and Supabase screens | Michael (+ engineering) |
| P7 | About 1 hour each for Arturo and Luz | All three |
| G1–G8 | About 1 hour of switching, then a supervised morning | Michael + engineering on call |
| R1–R6 | About 2 hours | Michael (ChatGPT editor) + engineering |

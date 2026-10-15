# FL Assistant fix round A: page & display

Branch: `fix/assistant-page-ux`. Worktree: `/Users/cns/dev/fl-fixA-wt`. Based on `origin/main` at `de8c1f2`. No deployment or merge.

| Item | Status | What staff now see |
|---|---|---|
| 1. Today’s entries | Done | All staff’s entries, newest recorded first, paged in batches of 500. Each plain wrapping line shows time, what, receipt number and person. Direct sidebar/phone controls and common English/Spanish day questions avoid the model round trip. A single 15-second deadline produces an error and Retry. |
| 2. Read answers | Done | A plain sentence followed by a readable table, correct lb/cases, and one copy of each identical read card. No raw JSON disclosure. |
| 3. Draft wording | Done | Product and quantity always have visible fields; requested quantity is explicitly identified when an incomplete SKU confirmation has no prepared quantity. New York times, plain reasons and lot-identity notes, and adjustments such as 2,000 → 1,995 lb. Shipment products come from their actual lines. |
| 4. Refusals | Done | Clear refusal card with an explanation; no disabled Record button posing as a draft. |
| 5. Reason buttons | Done | One click submits the reason immediately; no extra Send. Staff see the human label while FL receives the canonical code. |
| 6. Duplicate checkbox | Done | “Yes, this is a separate entry, record it anyway.” Recording remains gated by the acknowledgment. |
| 7. Waiting feedback | Done | Working text/spinner during Send, SKU confirmation, Record and other waiting actions; repeated writes remain disabled. |
| 8. Desktop/phone header | Done | Header and receipt tools stay contained at 1440 and 390 px, including all links. ES and Sign out stay on one line. Phone Today button stays accessible after the welcome screen. |
| 9. Short URL assets | Done | `/dash/fl-assistant` uses absolute `/dashboard/` paths for every script and stylesheet. |
| 10. Sign-in flash | Done | Session-check state appears first; a valid restored session goes directly to chat. |
| 11. Void hand-off | Done | Explicit void requests with a receipt number show a link to that receipt’s screen, where the existing authorized Prepare void action remains available. |

## Validation correction and final recheck

**Correction:** At the original reviewed commit `8385c5d`, the existing Assistant browser runner **failed with sessions off and on**. The earlier README and PR-body claim that it passed was incorrect. The original [off failure log](existing-browser-sessions-off.log) and [on failure log](existing-browser-sessions-on.log) remain unchanged. Its ship-product locator and session-expiry fixtures are now fixed; the new passing logs below come from the corrected runner.

| Review fix | Behavior and regression coverage |
|---|---|
| 1. Existing runner / duplicate ship product | Ship cards identify each line product once, preserving quantities, lots and service lines. Node covers single/bulk shipments; the existing browser scopes the shipment product and explicitly expires both `/receipts` and `/assistant/turn`, clears credentials/drafts and signs in again. Both runner modes pass. |
| 2. Personal Today wording | Legacy `today_entries` says “You've recorded 0 entries today. Team total: see Today's entries.” Direct team reads retain their all-staff count. Node covers zero/one and EN/ES; browser case 12 checks zero own entries beside six entries by another person, at desktop/phone in both session modes and languages. Server tool remains round B's scope. |
| 3. Finished stock cases | Known case counts appear alongside pounds, e.g. `1,600 lb (80 cases)`. Node covers supplied/zero/absent/partial counts and mixed units; browser case 13 checks EN/ES and the pound lot table. |
| 4. Today errors | Only actual deadline/abort timeouts show the 15-second sentence. HTTP, connection and pagination failures retain plain messages. Node covers these plus session expiry; browser cases 1/14 cover the 15-second stalled read, immediate 403/503/network errors and Retry recovery. |
| 5. Recorded receipt identity | The confirmed receipt number identifies the card; the internal Transaction field is removed. Node covers live and restored EN/ES cards; browser case 15 records a receive and checks the receipt number and absence of the internal ID. |

Final local checks (October 15, 2026):

- **Full Python: 5,818 passed**, zero failed/errors/skips, across 21 fresh schema groups using the single-DB runner and local PostgreSQL 17. Only `fl_fixA2_suite_20261015` was allocated, then dropped in each group's `finally`. [Group results](fix-a2/python/summary.json), [run output](fix-a2/python-run.log), [exact adapted runner](fix-a2/run-fresh-suite.py).
- **Node: 295 passed**, zero failures/skips: `node --test tests/test_*.js`. [Output](fix-a2/node-results.log).
- **Expanded Round A browser: 30/30 sessions off and 30/30 sessions on**, including all original 11 items plus review cases 12–15 at 1440/390 px. EN/ES checks are within the relevant cases. [Off results](fix-a2/sessions-off/results.json), [on results](fix-a2/sessions-on/results.json); 104 screenshots.
- **Existing Assistant browser: passed in both modes.** [Off output](fix-a2/existing-browser-sessions-off.log), [on output](fix-a2/existing-browser-sessions-on.log). Pilot mode proves both Today and Assistant session expiry; legacy mode retains the ordinary flow, receipt/retry, held corrections, private attachments, lot confirmation, EN/ES and 390 px light/dark.
- `git diff --check` passed. [Final database query](fix-a2/database-cleanup.txt): **zero `fl_fixA2_` databases remain**.

Browser commands used `PLAYWRIGHT_MODULE=/Users/cns/Documents/factory-ledger/node_modules/playwright/index.mjs`, `PILOT_SESSIONS=0` / `1`, and output directories under `/private/tmp/fl_fixA2_`. Python used `/Users/cns/Documents/factory-ledger/.venv-test/bin/python` (3.12), `PYTHONPATH=/private/tmp/fl_fixA_pydeps`, and `DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`. Both browser runners use real Chromium with deterministic local API fixtures; no hosted data or model calls.

Evidence paths in the browser JSON are relative to `fix-a2/`. Examples: [own/team wording on phone](fix-a2/sessions-on/390-12-own-and-team-entries.png), [finished stock](fix-a2/sessions-on/390-13-finished-lb-cases.png), [permission error](fix-a2/sessions-on/390-14-today-permission.png), [recorded receipt](fix-a2/sessions-on/390-15-recorded-receipt-number.png).

## Session B hand-off and limits

No `assistant_tools.py`, `fl_assistant.py`, resolver, prompt or commit logic changed. The browser’s explicit Today controls and common day questions use `GET /receipts?summary=true&limit=500&offset=…` without an actor filter. Session B should use this same summary path, with pagination and the intended all-staff scope, for other wording that still reaches its `today_entries` tool; its current tool uses the older detailed, self-filtered read. This is outside Round A’s server-understanding scope.

The five-second target was verified on fresh local data and browser fixtures. Hosted staging performance and real-model wording need a recheck after the approved staging deployment.

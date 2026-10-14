# D1 / D2 dashboard review evidence

Builder: Codex. Reviewer: Claude Code. Branch `feat/dash-d1-d2`, rebased onto `901ad5c` (PR #109).
No production changes, schema migrations, or edits to parallel worktrees.

## Part 1 — D1

Receipt lookup is shared by dashboard HTML pages. Receipt details read `GET /receipts/{number}`; History, Recent Entries and all Activity tables show the stored receipt as the primary identifier. Legacy rows are explicitly labeled Legacy TX; no receipt is fabricated. The additive backend change projects stored receipt identities onto existing read responses.

Exceptions are fetched from `GET /exceptions?status=open`, which includes escalated items, grouped by kind and ordered by FL's overdue flag and due date. Rows name the owner and link to their source receipt when FL has issued one. Resolve reads the individual exception before showing counted adjustment, A6 photo, supplier-lot, or owner decision forms. Reserved extra-kosher receipts and other held drafts are never labeled recorded.

### Current API contract (PR #109)

- Exception list and detail return actor-specific `allowed_actions`. These are the only offered controls; absent/empty lists offer none. `approve` maps to the approve form (`approved`), `reject` to reject (`declined`); other resolution kinds are unchanged. There is no copied permission/kind fallback.
- Explicit resolve/approve/reject routes are used after the review card. FL also provides resolve prepare/commit. Direct responses carry `resolution_receipt_number` (XR), which is shown as a linked, green **recorded decision** only when returned. Source `receipt_number` never substitutes for XR. Approve's posting receipt and resolve's fixing ADJ are labeled separately. A partial resolution can record XR while the exception stays open; the UI says so.
- `GET /receipts/{number}` supplies live `effective_status` and `void_receipt_number`. A voided source page is grey VOIDED / PARTLY VOIDED and links to VD; a committed ticket alone does not imply effective stock. Legacy rows without a number remain explicit.
- A5 supplier-lot is live on this base. The form uses `/lots/{lot_id}/supplier-lot/prepare` then ticket commit, displaying the LOT receipt and `resolved_exception_ids`, with no direct PATCH or second resolve call.

### Verification

- Dedicated fresh local PostgreSQL databases; never the shared test database.
- Node assertions for receipt/hold truthfulness, verbatim FL messages, escaping and queue ordering.
- Python tests execute receive prepare/commit and verify receipt identity on History, Recent Entries, Activity and receipt detail.
- `tests/visual/run-d12.mjs` runs Chrome at 390 px against real local FastAPI routes. It asserts session auth headers, token-only storage, receipt lookup, live supplier-lot draft without fallback, owner approval, logout and EN/ES. No mocked happy-path write routes.
- Browser fixtures are restricted to a named local database by `tests/visual/d12_fixture.py`; runtime evidence is under `.venv-test/d12-visual/`.

Part 1 final checks: 2,745 Python tests on fresh `fl_d12_20261013_p1_final`, 110 Node tests, zero failures/errors/skips; real-route Chrome passed. Part 2 and staging evidence are below.


## Part 2 — D2

`entry.html` provides receive, adjust, found, make, pack, order shipping and standalone shipping. Search uses the real `/resolve` candidates; the operator explicitly picks products, suppliers, lots and orders. Receive includes the real supplier and supplier-lot field; FL's unidentified warning/deadline is displayed in the draft. Corrections use the fixed EN/ES reason list. Make supports multiple substitutions (original ingredient, substitute product, actual lot and reason) and explicit ingredient lot overrides; pack supports chosen source lots and quantities. Every consumed lot requires entered confirmation, with last-four including the hyphen, full code, or a pallet's full code. FL validates the code and pallet move. Ambiguous/incorrect evidence retains editable controls so the person can correct it and request a fresh draft.

All seven ledger flows use their actual `*/prepare` route then `/tickets/{ticket}/commit`, carrying the exact hash and only explicitly acknowledged warnings. Form edits invalidate the draft. The draft shows the signed-in person, plant-time occurrence/expiry and FL's quantities, warnings and blockers; the entry form folds away on a phone while the person reviews. A6 uploads carry photo bytes and returned attachment IDs; BOL photo is attached at ship commit. Record retries keep the same ticket, held/failed/no-receipt responses never show green, and only a returned receipt produces a green receipt link. No stock, threshold, permission, kosher, or substitution policy was implemented in the browser.

With `KEY_SESSIONS_ENABLED` on, only the FL session token is kept in sessionStorage and sent as `X-API-Key`. Sign out revokes it. Expiry clears it and asks for sign-in; the same person can retry the same unexpired ticket. With the flag off, the new forms hold a personal key in memory for the current tab; existing dashboard behavior is retained. Configuration failure fails closed. Explicit `FL_SESSION_API_BASE` hosting configuration is honored, including staging previews. No credentials appear in screenshots, fixtures or committed evidence.

### Part 2 verification

- Fresh local `fl_d12_20261013_p2_final`: **2,745 Python passed**, no failures/errors/skips (71.20 s). Existing deprecation warnings remain.
- Node suite: **112 passed**, including held/reserved receipt truthfulness, malformed/missing receipt outcomes, message escaping, queue ordering and New York dates.
- Fresh browser DB `fl_d12_browser_final`; real local FastAPI + Chrome at 390 px: D1, receive, make with rolled-oats substitution, pack, adjust, found, order ship with BOL photo, standalone ship with BOL photo. Successful routes are real; assertions inspect request methods, payload IDs/hash and session headers. Negative commit-only fault injection covers HTTP error, invalid JSON, missing receipt and network failure; none shows green, and same-ticket retry succeeds.
- Real expired-session rejection, token clearing, same-person retry, server logout rejection and flag-off raw-key-in-memory operation pass. Configuration-outage fault injection confirms there is no raw-key fallback. English/Spanish and no horizontal overflow pass.
- A5 part 2's published contract was checked read-only from `origin/feat/supplier-lot-ticket` at `0e7c62f`, including both reason values, previous/new supplier lot in the draft and atomic exception closure. The form matches it. The later #106 rebase and review-fix runner verify the live supplier-lot round trip. No parallel checkout was read or modified.

### Hosted staging walkthrough (latest #109 integration)

Chrome overlays only this branch's dashboard static assets at the hosted staging origin; all resolver, session, prepare, commit and exception requests use the real hosted staging API. No deployment, schema or service settings changed. Protected credentials stay in memory; synthetic actors are deactivated in `finally`.

1. Receive **1 container × 10 lb of synthetic rolled oats** with its supplier/BOL/supplier lot → **RCV-261013-016** (transaction 1000000157).
2. Make **two 10 lb synthetic batches**, substituting rolled oats for oat bran, with a reason and typed **-001** lot confirmation → **MK-261013-016** (transaction 1000000158). FL opens shortage **#39** for the 10 lb deficit.
3. Open Exceptions filtered to that make receipt. FL supplies exactly `counted`, `missing_movement`, `voided`. Choose counted, enter zero and a note, review and confirm → **XR-261013-004**, separately labeled correction **ADJ-261013-031**. The green decision link uses XR, not source MK. The closed row disappears; opening XR shows its actual receipt page.
4. Verify session auth headers, no retained personal key, 390 px layout, then Sign out. Actor `STG-D12-815187CF Floor` is inactive.

Evidence: [receive draft](d12-staging/receive-draft-390.png), [receive recorded](d12-staging/receive-recorded-390.png), [make with substitution](d12-staging/make-substitution-draft-390.png), [make recorded](d12-staging/make-substitution-recorded-390.png), [resolution review](d12-staging/exceptions-review-390.png), [XR recorded and closed queue](d12-staging/exceptions-xr-recorded-390.png), [XR receipt detail](d12-staging/resolution-receipt-390.png), [sanitized requests and receipt IDs](d12-staging/evidence.json).

Earlier checks remain immutable: receive/make 014 reached the shortage before a browser assertion raced the detail load; [its synthetic shortage was closed](d12-staging/interrupted-check-cleanup.json). Receive/make 015 and XR-261013-003 passed; the final 016/004 pass waits for the refreshed queue before taking screenshots.

### Remaining acceptance steps

- Claude Code independent review; PR stays draft and unmerged.
- Final hosted dashboard deployment and broad owner acceptance are separate from browser-overlay staging checks. No production action is included.

## Review fix round 1 (Claude Code review, comment 6157664298)

Rebased onto `origin/main` 1a423f7 (A5 part 2, #106). Builder for this round: Claude Code. Presentation only — FL still enforces every permission and rule.

| # | Fix | Node regression | Browser regression (`run-d12-fixes.mjs`, real local FastAPI) |
|---|-----|-----------------|------|
| 0 | "Identify supplier lot" runs live `POST /lots/{lot_id}/supplier-lot/prepare` → `/tickets/{t}/commit`, shows `resolved_exception_ids` | route + field wiring | floor, 390 px: LOT receipt green; "Resolved exception #N"; exception now `resolved`; no second `/resolve` call |
| 1 | Updated for #109: resolve/reject show their XR decision receipt in green; source receipt is never used; fixing/posting ADJ is labeled separately | `decisionOutcome` | counted SHORTAGE returns XR + ADJ with working XR link; owner rejection returns XR without a stock post |
| 2 | Updated for #109: only `allowed_actions`, mapping approve → approved and reject → declined; missing/empty means no buttons | `resolutionChoices` | floor: no LARGE_CORRECTION action button; UNIDENTIFIED_LOT only Identify; owner held correction offers approve/reject |
| 3 | Owner approve note optional (blank omitted → `{}`); reject, write-off and waive require a note | `decisionRequest`, `noteRequired` | owner, 1280 px, Spanish: reject with blank note blocked by the form; blank approve sends `{}`, shows green XR and a separately labeled ADJ |
| 4 | Receipt page: VOIDED / PARTLY VOIDED (grey) from FL's per-entry `effective_status`, never green; shows the VD receipt when FL returns one | `receiptState` | `ADJ-…` voided via legacy `/void`: `VOIDED —` / `ANULADO —`, no green; a live `MK-…` still green |
| 5 | Plain timeout / lost-connection wording; red banner cleared when the same-ticket retry records | `transportText` | commit answered after 21 s: plain text (no "signal"/"abort"), never green; retry replays the receipt, banner hidden, exactly 2 commit requests |
| 6 | EN/ES relabels in place; half-filled form and open draft survive | `draftHtml` carries both labels; no re-render on switch | adjust form keeps lot, −1 lb, reason, note across EN→ES; open draft survives ES→EN with English labels |

Before/after: the 7 new Node tests fail on 71125d9's dashboard files and pass now. The browser runner, run block by block against 71125d9's dashboard files on a fresh DB, fails in all four blocks and passes on this head.

Existing runners updated for the new truth: `run-d12` now expects the live supplier-lot draft (was the pre-#106 404); `run-d2` expects the open draft to survive the language switch and the plain lost-connection wording.

### Verification (2026-10-13)

- Fresh local DB `fl_d12_fix1_final2` (pg_trgm + `tests/schema/schema.sql`): **2,796 Python passed, 1 failed** — `test_shift_summary.py::test_hash_is_deterministic_and_covers_only_receipt_rows`, which moves the plant clock +3 h and fails when run after 21:00 ET (the run was at 21:17 ET). Neither that test nor `shift_summary.py` differs from `origin/main`.
- Node: **119 passed**, 0 failed/skipped.
- Browser, fresh `fl_d12_browser_*` DBs, Chrome: `run-d12`, `run-d2`, `run-d12-auth`, `run-d12-fixes` (sessions on) and `run-d12-auth-off` (sessions off) all pass.
- Hosted staging (branch assets over the staging API, no deploy): receive + make substitution passed — **RCV-261013-013**, **MK-261013-013**; synthetic actor deactivated. That historical run did not exercise Exceptions; the #109 integration check below does.

## PR #109 integration (2026-10-13)

The previous backend dependencies are now available on `901ad5c`: `allowed_actions`, XR decision receipts, VD backlinks and optional blank approve notes. Both changelog histories were preserved during rebase. No backend logic, migrations or other worktrees were changed in this update.

- Fresh UTF-8 PostgreSQL DB **`fl_d12_xr_20261013_2233`**, loaded from `tests/schema/schema.sql` plus pg_trgm: **2,835 Python passed**, 0 errors/failures/skips; 81.60 s. Started at **22:38 ET**; the shift-summary test passes at this hour on merged main. Existing deprecation warnings remain.
- **121 Node passed**, 0 failures/skips. Regressions cover server-only `allowed_actions`, approve/reject mappings including kosher holds, missing/malformed XR, distinct source/ADJ/XR identities, rejection without stock, partial resolution status and live VD state.
- All five local real-route Chrome runners passed on dedicated fresh `fl_d12_browser_xr_*` databases: `run-d12`, `run-d2`, `run-d12-auth`, `run-d12-auth-off`, `run-d12-fixes`. The fix runner verifies counted XR/ADJ links, actual reject XR, optional owner approval, allowed buttons, disappearance of closed rows, and a real VD backlink; existing EN/ES, 390 px, seven D2 flows, session headers, expiry/logout, flag-off and negative commit checks remain green.
- The sixth runner, `run-d12-staging`, passed against hosted staging as described above. Credentials and ticket tokens are excluded from saved evidence. All synthetic actors are inactive.
- The only initial failures were browser assertions racing asynchronous exception detail/refresh. Tests now wait for the selected exception ID and closed-row removal; final reruns pass. The implementation also removes completed decision controls immediately.
- No new migration or backend business logic in this update. No production, other-worktree, deployment or merge action.

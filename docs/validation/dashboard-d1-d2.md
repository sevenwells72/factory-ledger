# D1 / D2 dashboard review evidence

Builder: Codex. Reviewer: Claude Code. Branch `feat/dash-d1-d2`, based on `f4e51b5`.
No production changes, schema migrations, or edits to parallel worktrees.

## Part 1 — D1

Receipt lookup is shared by dashboard HTML pages. Receipt details read `GET /receipts/{number}`; History, Recent Entries and all Activity tables show the stored receipt as the primary identifier. Legacy rows are explicitly labeled Legacy TX; no receipt is fabricated. The additive backend change projects stored receipt identities onto existing read responses.

Exceptions are fetched from `GET /exceptions?status=open`, which includes escalated items, grouped by kind and ordered by FL's overdue flag and due date. Rows name the owner and link to their source receipt when FL has issued one. Resolve reads the individual exception before showing counted adjustment, A6 photo, supplier-lot, or owner decision forms. Reserved extra-kosher receipts and other held drafts are never labeled recorded.

### Current API differences from the design

- A3b ships `/exceptions/{id}/resolve`, `/approve`, and `/reject`, not `/resolve/prepare`. D1 uses these actual operations after an explicit review card. FL performs the count calculation and all approval/evidence checks. No client-side rule or stock calculation was added.
- A3b counted resolution may post a correction without a receipt; metadata resolution and rejection also do not issue XR receipts. The UI reports the server's exception status without green success and instructs users to check History before retrying a counted adjustment. This backend gap is not silently represented as a ticket or a fabricated receipt.
- Large-correction holds may have no receipt number yet. Such rows say receipt unavailable. The current backend also has no ticketed void producer; the read projection can show a VD receipt if a committed void ticket links its `result_ref.correction_id`, and otherwise preserves the legacy correction trail.
- A5 part 2 is absent on this base. Supplier-lot forms use `POST /lots/{lot_id}/supplier-lot/prepare` with `supplier_lot_code`, `reason_code`, `note`, `client_source: dashboard`, followed by the normal ticket commit. There is no PATCH fallback. Check its final contract and run the identification/auto-resolution round trip once that endpoint reaches main.

### Verification

- Dedicated fresh local PostgreSQL databases; never the shared test database.
- Node assertions for receipt/hold truthfulness, verbatim FL messages, escaping and queue ordering.
- Python tests execute receive prepare/commit and verify receipt identity on History, Recent Entries, Activity and receipt detail.
- `tests/visual/run-d12.mjs` runs Chrome at 390 px against real local FastAPI routes. It asserts session auth headers, token-only storage, receipt lookup, supplier-lot prepare 404 without fallback, owner approval, logout and EN/ES. No mocked happy-path write routes.
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
- A5 part 2's published contract was checked read-only from `origin/feat/supplier-lot-ticket` at `0e7c62f`, including both reason values, previous/new supplier lot in the draft and atomic exception closure. The form matches it. Main remains `f4e51b5`; the real supplier-lot round trip remains owed after that API lands on main. No parallel checkout was read or modified.

### Hosted staging walkthrough

The home Mac has the protected staging URL/key files. A guarded connection verified the staging project and attachment/session migration markers. **No staging deployment was replaced:** Chrome overlaid only this branch's dashboard static assets at the hosted staging origin; all resolver, session, prepare and commit requests went to the real hosted staging API. No schema or service settings changed. The synthetic actor was deactivated in `finally`; the immutable test receipts remain labeled `STG-D12-*`.

1. Receive **1 container × 10 lb of synthetic rolled oats**, choose its real synthetic supplier, enter BOL and supplier lot, review FL's draft and Record → **RCV-261013-010** (transaction 1000000086).
2. Make **one 10 lb synthetic batch**; substitute **rolled oats for oat bran**, choose the received lot, enter the reason, type **-001**, check the new draft and Record → **MK-261013-006** (transaction 1000000087).
3. Verify token auth headers and no retained personal key, phone width, then Sign out. First acceptance pass also produced RCV-261013-009 / MK-261013-005; the second pass corrected screenshot timing and is the captured evidence.

Evidence: [receive draft](d12-staging/receive-draft-390.png), [receive recorded](d12-staging/receive-recorded-390.png), [make with substitution](d12-staging/make-substitution-draft-390.png), [make recorded](d12-staging/make-substitution-recorded-390.png), [sanitized requests and receipt IDs](d12-staging/evidence.json).

### Remaining integration/acceptance steps

- Claude Code independent review of both part commits; PR stays draft and unmerged.
- After A5 supplier-lot reaches main: rebase, rerun actual identify/commit/automatic exception closure and LOT receipt lookup. No direct-route workaround exists.
- A3b's non-ticketed counted-resolution/XR and missing held/legacy-void receipt gaps remain upstream, as described under Part 1.
- Hosted deployment of the final branch and broad owner acceptance are separate from this browser-overlay staging walkthrough. No production action is included.

## Review fix round 1 (Claude Code review, comment 6157664298)

Rebased onto `origin/main` 1a423f7 (A5 part 2, #106). Builder for this round: Claude Code. Presentation only — FL still enforces every permission and rule.

| # | Fix | Node regression | Browser regression (`run-d12-fixes.mjs`, real local FastAPI) |
|---|-----|-----------------|------|
| 0 | "Identify supplier lot" runs live `POST /lots/{lot_id}/supplier-lot/prepare` → `/tickets/{t}/commit`, shows `resolved_exception_ids` | route + field wiring | floor, 390 px: LOT receipt green; "Resolved exception #N"; exception now `resolved`; no second `/resolve` call |
| 1 | `/resolve` and `/reject` never green, never the source receipt; neutral "Resolved / Waived / Rejected — no new receipt"; only an `/approve` ticket receipt is green | `decisionOutcome` | real SHORTAGE from a make (`MK-…`), resolved *counted 0*: neutral notice, no `MK-…`, no green anywhere |
| 2 | Choices from `allowed_resolutions` when FL returns it; otherwise owner-only choices hidden via `/auth/whoami` `permissions.approve_exception` / `resolve_exception` | `resolutionChoices` | floor: LARGE_CORRECTION shows "The owner decides it" (no approve/reject); UNIDENTIFIED_LOT offers only "Identify supplier lot" |
| 3 | Owner approve note optional (blank omitted → `{}`); reject, write-off and waive require a note | `decisionRequest`, `noteRequired` | owner, 1280 px, Spanish: reject with blank note blocked by the form; blank approve sends `{}` and shows green `ADJ-…` |
| 4 | Receipt page: VOIDED / PARTLY VOIDED (grey) from FL's per-entry `effective_status`, never green; shows the VD receipt when FL returns one | `receiptState` | `ADJ-…` voided via legacy `/void`: `VOIDED —` / `ANULADO —`, no green; a live `MK-…` still green |
| 5 | Plain timeout / lost-connection wording; red banner cleared when the same-ticket retry records | `transportText` | commit answered after 21 s: plain text (no "signal"/"abort"), never green; retry replays the receipt, banner hidden, exactly 2 commit requests |
| 6 | EN/ES relabels in place; half-filled form and open draft survive | `draftHtml` carries both labels; no re-render on switch | adjust form keeps lot, −1 lb, reason, note across EN→ES; open draft survives ES→EN with English labels |

Before/after: the 7 new Node tests fail on 71125d9's dashboard files and pass now. The browser runner, run block by block against 71125d9's dashboard files on a fresh DB, fails in all four blocks and passes on this head.

Existing runners updated for the new truth: `run-d12` now expects the live supplier-lot draft (was the pre-#106 404); `run-d2` expects the open draft to survive the language switch and the plain lost-connection wording.

### Verification (2026-10-13)

- Fresh local DB `fl_d12_fix1_final2` (pg_trgm + `tests/schema/schema.sql`): **2,796 Python passed, 1 failed** — `test_shift_summary.py::test_hash_is_deterministic_and_covers_only_receipt_rows`, which moves the plant clock +3 h and fails when run after 21:00 ET (the run was at 21:17 ET). Neither that test nor `shift_summary.py` differs from `origin/main`.
- Node: **119 passed**, 0 failed/skipped.
- Browser, fresh `fl_d12_browser_*` DBs, Chrome: `run-d12`, `run-d2`, `run-d12-auth`, `run-d12-fixes` (sessions on) and `run-d12-auth-off` (sessions off) all pass.
- Hosted staging (branch assets over the staging API, no deploy): receive + make substitution passed — **RCV-261013-013**, **MK-261013-013**; synthetic actor deactivated. The staging runner does not exercise the Exceptions page.

### Waiting on the backend PR (minimal fields)

- `GET /exceptions/{id}` → `allowed_resolutions: [kind, …]` for the signed-in person (computed in FL with `permissions.allowed`). The page already prefers it. Until then the browser keeps FL's kind list and hides only what `/auth/whoami` denies, so floor can still see e.g. "Acknowledge" on LATE_ENTRY or "Attach BOL photo" (owner-only kinds) and FL answers 403.
- Ticketed exception resolution with an `XR-…` receipt (the counted adjust currently posts with no receipt). When `/resolve` returns a receipt for that action, the page will show it green; until then it shows the neutral "no new receipt" state.
- `void_receipt_number` (a `VD-…` from a ticketed void) on `GET /receipts/{n}`; and ideally a receipt-level `effective_status`. The page already renders both when present.
- `DecisionRequest` accepting a blank approve note — not needed by this page (it omits a blank note), but keeps other clients consistent.

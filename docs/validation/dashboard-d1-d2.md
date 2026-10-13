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

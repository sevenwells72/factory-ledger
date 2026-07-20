# Change Log

## 2026-07-20 14:39 — Keep toasted board cells unchanged
- **File(s) changed:** `dashboard/scheduler/seven-wells-production-board.html`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Limited pan-product cell rendering to Granola Bake and Coconut (sweet); Coconut (toasted) board cells continue using the prior SKU presentation, while toasted clipboard text still uses the requested coconut pan format.
- **Why:** Match the Phase 2 board-cell scope exactly and avoid an unrequested visual change to the toasted station.

---

## 2026-07-20 14:34 — Preserve coconut detail cases and single-SKU bake text
- **File(s) changed:** `dashboard/scheduler/seven-wells-production-board.html`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Pan-presentation aggregation now carries the scheduler's existing coconut `cases` values instead of recalculating them; copied baking schedules retain today's wording whenever a pan group has one SKU that day.
- **Why:** Keep case counts byte-for-byte tied to scheduled cell data and preserve the required Vanilla Crisp/single-SKU clipboard behavior.

---

## 2026-07-20 14:31 — Phase 2 pan-product schedule presentation
- **File(s) changed:** `dashboard/scheduler/seven-wells-production-board.html`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Grouped Bake and Coconut board details and copied daily schedules by pan product, retained SKU/destination sub-lines and customer allocations, left packing sections SKU-based, and bumped the board to `v1.9-dev`.
- **Why:** Present shared bake/mix runs as the pan products the floor actually makes without changing schedule calculations, station data, totals, or changeovers.

---

## 2026-07-20 14:25 — Merge pan-product mapping with dashboard order-fetch fix
- **File(s) changed:** `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`, `dashboard/dashboard.js`, `dashboard/index.html`, `dashboard/scheduler/seven-wells-production-board.html`
- **What changed:** Merged the independently completed pan-product mapping and dashboard order-fetch histories; resolved the two changelog conflicts by preserving both original entries in newest-first order.
- **Why:** Bring both production-board scheduling behavior and the deployed dashboard order-fetch correction together on `main` without altering either code change.

---

## 2026-07-20 14:10 — Phase 1 pan-product mapping
- **File(s) changed:** `dashboard/scheduler/seven-wells-production-board.html`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Added the explicit pan-product catalog mapping and `panOf()` fallback/override model; bake changeovers now count and deduct capacity by distinct pan product rather than distinct SKU; bumped the board to `v1.8-dev`. Rendering and SKU-based pack/repack changeovers are unchanged.
- **Why:** Model SKUs that share a baked or mixed batch as one pan product so the scheduler does not report false bake changeovers.

---

## 2026-07-20 11:37 — Fetch sales orders by selected server-side status
- **File(s) changed:** `dashboard/dashboard.js`, `dashboard/index.html`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Changed the Sales Orders list request to send the selected status to `GET /sales/orders` with the existing 200-row safety cap. The default request now uses `status=open`; changing the status dropdown refetches that status from the API, while All Orders omits the status parameter. Bumped the dashboard JavaScript cache-buster to v23.
- **Why:** Prevent closed historical orders from consuming the 200-row response before later-dated open orders are returned, while keeping every status dropdown option backed by an appropriate server request.

---

## 2026-07-14 13:29 — Reorder SO detail sections
- **File(s) changed:** `dashboard/dashboard.js`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Moved the unchanged TOTAL ORDERED / SHIPPED / REMAINING and PALLETS summary markup below the product line-items table and before NOTES in the sales order detail view.
- **Why:** Put product details directly beneath the order header while keeping NOTES last.

---

## 2026-07-09 20:16 — Mark dashboard order editing tested
- **File(s) changed:** `dashboard/index.html`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Bumped dashboard cache-busters to `dashboard.css?v=13` and `dashboard.js?v=19`. Recorded production manual test results for dashboard order editing: server rejection gate verified on cancelled `SO-260709-003`; dashboard header and line edits verified on confirmed `SO-260710-001`, including derived cases and `line_value` recalculation. Backlog notes: (a) `GET /sales/orders` silently ignores the `?search=` query param; (b) `cancelled` is terminal — document the allowed status transition graph, including whether `shipped` → `cancelled` is blocked.
- **Why:** Prepare the tested dashboard order-editing branch for merge while preserving known follow-ups.

---

## 2026-07-09 19:51 — Tighten order line edit save scan
- **File(s) changed:** `dashboard/dashboard.js`
- **What changed:** Limited the line-save scan to rows with editable quantity/price inputs so fulfilled or cancelled lines shown read-only in edit mode are skipped safely.
- **Why:** Prevent read-only line rows from causing a client-side save error.

---

## 2026-07-09 19:50 — Add dashboard sales order editing controls
- **File(s) changed:** `dashboard/dashboard.js`, `dashboard/dashboard.css`, `dashboard/index.html`
- **What changed:** Added order-detail edit mode for ship-by date, notes, status, and editable line quantity/price fields. Writes use the existing authenticated Railway API helper, wait for 2xx responses, show API rejection text, and re-fetch order detail after successful edits; status changes require confirmation. Bumped dashboard JS/CSS cache-busters.
- **Why:** Let dashboard users edit eligible sales orders from the order detail view without API or OpenAPI schema changes.

---

## 2026-07-09 12:08 — Bump GPT instructions to v3.7.0
- **File(s) changed:** `gpt-instructions-v3.md`, `CHANGE_LOG.md`, `~/change-log.md`
- **What changed:** Updated the Factory Ledger GPT instruction header from v3.6.0 to v3.7.0; no instruction body text changed.
- **Why:** Prepare the Custom GPT instructions for the v3.7.0 release while preserving the under-8,000-character budget.

---

## 2026-07-09 11:57 — Bring GPT instructions under 8,000 chars
- **File(s) changed:** `gpt-instructions-v3.md`, `CHANGE_LOG.md`, `~/change-log.md`
- **What changed:** Condensed the Factory Ledger intro and CRITICAL RULES wording for NEVER INSTRUCT, BE CONCISE, and NEVER CLAIM UNAVAILABILITY; final full instruction character count is 7,991.
- **Why:** Keep the fabricated-unavailability guard while fitting the Custom GPT 8,000-character instruction limit.

---

## 2026-07-09 11:52 — Condense unavailability rule error handling
- **File(s) changed:** `gpt-instructions-v3.md`, `CHANGE_LOG.md`, `~/change-log.md`
- **What changed:** Removed the separate SURFACE API ERRORS DIRECTLY line and folded verbatim API-error handling into NEVER CLAIM UNAVAILABILITY; new full instruction character count is 8,153.
- **Why:** Preserve the fabricated-unavailability guard while recovering instruction budget.

---

## 2026-07-09 11:47 — Add NEVER CLAIM UNAVAILABILITY rule
- **File(s) changed:** `gpt-instructions-v3.md`, `CHANGE_LOG.md`, `~/change-log.md`
- **What changed:** Added a CRITICAL RULE preventing the GPT from claiming services, APIs, or actions are unavailable in the chat; recorded the exact post-change instruction character count as 8,257.
- **Why:** Prevent recurrence of the 2026-07-09 fabricated-unavailability incident.

---

## 2026-07-08 16:18 — Deploy orders matrix pan-note enhancement
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** Fast-forwarded feature commit `7127468` to `main`. Railway deployment `cc49789b-3e4b-4607-a7e3-21e4c2e5d328` completed successfully; production `/health` returned HTTP 200 with database connected and pool active. Matrix tests passed 2/2 before merge.
- **Why:** Record reproducible deployment evidence and the regression guard for the production release.

---

## 2026-07-08 16:15 — Clarify sub-tenth pan requirements in matrix notes
- **File(s) changed:** `main.py`, `tests/test_orders_matrix_export.py`, `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** Positive pan calculations that would render as 0.0 at one decimal now display `<0.1 pans`; added Cases and Pounds assertions for the fractional 70073 seed.
- **Why:** Avoid presenting a small positive production requirement as zero.

---

## 2026-07-08 16:10 — Verify serialized matrix comment dimensions
- **File(s) changed:** `tests/test_orders_matrix_export.py`, `CHANGE_LOG.md`
- **What changed:** Changed the comment-size assertion to inspect the generated XLSX VML shape data for 260×80 boxes because openpyxl intentionally reloads legacy comments with default dimensions even when the serialized workbook contains the requested size.
- **Why:** Test the actual dimensions Excel receives instead of openpyxl's lossy comment reload representation.

---

## 2026-07-08 16:07 — Add per-cell pan notes to orders matrix export
- **File(s) changed:** `main.py`, `tests/test_orders_matrix_export.py`, `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** Added 260×80 Factory Ledger comments to every nonzero quantity cell on the Cases and Pounds sheets with case, pound, pan-yield, documented batch, 70073 finished-weight, and 31012 repack details. Added BATCHES / PANS calculation comments without replacing the existing source-citation comments, plus regression coverage for both sheets and special routing.
- **Why:** Put production conversion math and exceptions directly on the matrix cells planners use.

---

## 2026-07-08 14:44 — Deploy styled open-orders matrix export
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** Fast-forwarded feature commit `5aec509` to `main`. Railway deployment `369dfadb-2fc5-43d7-a51a-12a06f6f423e` completed successfully at `fastapi-production-b73a.up.railway.app`; authenticated production smoke returned HTTP 200 and the XLSX media type. Netlify served dashboard JS v17 with the matrix button and export handler. No OpenAPI change; GPT schema remains at 30 operations.
- **Why:** Record reproducible deployment evidence and the regression guard for the production release.

---

## 2026-07-08 14:40 — Show fractional case quantities in matrix exports
- **File(s) changed:** `main.py`, `tests/test_orders_matrix_export.py`, `CHANGE_LOG.md`
- **What changed:** Detects any non-integer quantity per SKU and applies `#,##0.#;(#,##0.#);"—"` to that product column on both Cases and Pounds sheets, including its numeric summary cells. Added regression assertions for a 1.5-case 6x7 oz line and for retaining the integer format on whole-case columns. Confirmed the dashboard handler already checks `response.ok`, reads failure text, and sends it through the existing Sales Orders error alert before calling `blob()`.
- **Why:** Prevent partial cases from appearing silently rounded while preserving current whole-case formatting and safe dashboard error handling.

---

## 2026-07-08 14:31 — Isolate matrix endpoint tests from database startup
- **File(s) changed:** `tests/test_orders_matrix_export.py`, `CHANGE_LOG.md`
- **What changed:** Changed the seeded endpoint tests to issue ASGI requests without entering the application lifespan, preventing the real database startup hook from running while the endpoint's transaction source is replaced by deterministic seeded rows.
- **Why:** Keep the regression test self-contained and ensure it exercises the HTTP route/workbook response without requiring or risking any database connection.

---

## 2026-07-08 14:28 — Preserve undated open orders in matrix export
- **File(s) changed:** `main.py`, `CHANGE_LOG.md`
- **What changed:** Kept open order lines with no due date in the export (sorted after dated orders with blank date/weekday) while retaining 422 rejection for unknown UoMs or unusable case quantities; simplified grouped-SKU accumulation without changing totals.
- **Why:** The matrix must query the same open-order population as the existing CSV rather than silently dropping or rejecting otherwise valid undated orders.

---

## 2026-07-08 14:25 — Add Sales Orders matrix download and regression tests
- **File(s) changed:** `dashboard/index.html`, `dashboard/dashboard.js`, `tests/test_orders_matrix_export.py`, `CHANGE_LOG.md`
- **What changed:** Added the Sales Orders "Export Matrix (xlsx)" button with authenticated binary download handling and a dashboard cache-buster bump. Added endpoint tests covering both sheets, hidden order ID, header typography, case/pound conversion, grand totals, Monday week separators, autofilter scope, and 422 handling for unknown UoMs. Not deployed or committed.
- **Why:** Complete the dashboard workflow and protect the workbook contract with executable regression coverage.

---

## 2026-07-08 14:25 — Begin styled open-orders matrix Excel export
- **File(s) changed:** `main.py`, `requirements.txt`, `CHANGE_LOG.md`
- **What changed:** Added the authenticated dashboard-only `/export/orders-matrix.xlsx` endpoint and openpyxl workbook builder with Cases/Pounds sheets, UoM validation, family styling, formulas, week banding/separators, past-due emphasis, and production-planning summary rows; added the openpyxl runtime dependency. Work remains under review on `feat/orders-matrix-export`; not deployed.
- **Why:** Give Sales Orders users a production-ready matrix export without adding an operation to the capped GPT OpenAPI schema.

---

## 2026-06-29 13:17 — Deployed Sales Order pallet display
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md` (plus git: pushed `fa091e6` to `main`, triggering Railway and Netlify CD)
- **What changed:** Confirmed Railway deployment `d0c13a45-9ec6-4ff6-8388-0994716afa20` and Netlify deploy `6a42a80a626e1c0008f39d53` succeeded for commit `fa091e6`. Live checks confirmed the additive `pallet_lines` API field, the published helper/cache-busters, `26 pallets` for Restaurant Depot's 3,640-case order, `2.8 pallets / 3 physical mixed pallets` for Juliette's seven-line order, per-line `0.4 pallet`, remaining pallet values on detail, quiet `—` for unknown mappings, working expand/detail navigation, preserved Factory Ready controls, and no browser-console or Railway error-log matches.
- **Why:** Close the release with reproducible production evidence and a regression guard.

---

## 2026-06-29 13:09 — Add Sales Order pallet calculations and display
- **File(s) changed:** `main.py`, `dashboard/pallet-calculations.js`, `dashboard/dashboard.js`, `dashboard/dashboard.css`, `dashboard/index.html`, `tests/test_pallet_calculations.js`, `tests/test_sales_order_line_fields.py`, `tests/requirements-test.txt`
- **What changed:** Added a reusable browser/Node pallet utility that maps 10-lb cases to 140 cases/pallet and 25-lb cases to 60 cases/pallet, calculates exact and rounded-up physical pallets, formats singular/plural labels, totals mixed orders, and returns a quiet em dash for unknown products. Added compact case-count line summaries to the Sales Orders list read response so the UI calculates pallets from cases rather than pounds. Added pallet totals to the list, expanded drawer, and detail KPI area, plus per-line ordered/remaining pallet information. Added Node helper tests and an API regression test. Pinned the test-only httpx dependency below 0.28 to match the repo's FastAPI/Starlette TestClient. No order, inventory, shipment, lot, or production database data changes.
- **Why:** Make full and mixed-pallet Sales Orders immediately legible while preserving every existing order workflow and quantity display.

---

## 2026-06-23 14:42 — Factory Ready annotation flag for open Sales Orders (branch feat/so-floor-ready)
- **File(s) changed:** `main.py`, `migrations/037_sales_order_flags.sql`, `dashboard/dashboard.js`, `dashboard/dashboard.css`, `dashboard/index.html`, `tests/schema/schema.sql`, `tests/test_write_response_contract.py`, `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** Added a dashboard-only `sales_order_flags` annotation table migration (not run) and a `POST /sales-orders/{so_number}/ready` API that upserts `ready`, `ready_at`, `ready_by`, and `note` without touching sales order data or any inventory/FIFO/lot/production/shipment logic. The endpoint is modeled on `POST /dashboard/api/notes`: it uses `Depends(verify_api_key)`, `get_transaction()`, returns a plain dict for the global `write_response_envelope`, and re-raises readonly errors through `_is_readonly_error(e)` so the global readonly tripwire handles them. `/sales/orders` now LEFT JOINs `sales_order_flags` and adds only `ready`, `ready_at`, `ready_by`, and `note`. Sales Orders dashboard only: added a ready checkbox after the expand caret with optimistic POST, faded `.so-ready` rows with green left border, a READY pill near status, a drawer note input/display, and a "Hide ready" filter beside "Overdue only"; cache-busters bumped `dashboard.css?v=7→v=8`, `dashboard.js?v=10→v=11`. Tests/schema updated for local test DB coverage; focused contract tests added. NOT DEPLOYED; migration NOT RUN.
- **Why:** Let the floor mark open orders as factory-ready on the dashboard as a reversible UI annotation, while keeping all order, inventory, FIFO, lot, production, and shipment records unchanged.

---

## 2026-06-23 12:56 — Sales Orders inline expand/collapse of line items (branch feat/sales-order-inline-expand)
- **File(s) changed:** `main.py`, `dashboard/dashboard.js`, `dashboard/dashboard.css`, `dashboard/index.html`, `tests/test_sales_order_line_fields.py`
- **What changed:** Added a per-row expand/collapse control to the Sales Orders table that reveals that order's line items inline, loaded on demand. (a) `main.py` `get_sales_order`: added `p.odoo_code` and `p.uom` to the line query and `sku`/`uom` (uom coalesced to `'lb'`) to each line dict — purely additive, no new endpoint (op count unchanged). (b) `dashboard.js`: new leading expand column with a caret toggle button per row; clicking it (stopPropagation) toggles a hidden detail row and fetches `/sales/orders/{id}` once, caching the result in `state.orderLinesCache` so re-expand and filter re-renders need no refetch; renders a sub-table of SKU / Product / Ordered / UoM / Remaining via new `renderOrderLinesContent` + `bindOrderExpandToggles`. Row click (incl. the SO number) still opens the full detail page unchanged. Multiple rows can be open at once. (c) `dashboard.css`: dark-theme styles for the toggle, caret rotation, and inline line-items table matching existing `.orders-table` styling. (d) `index.html`: cache-busters bumped `dashboard.css?v=6→v=7`, `dashboard.js?v=9→v=10`. (e) New test file (2 tests) asserting lines expose `sku`/`uom` and that uom defaults to `'lb'` when NULL. Full suite 53/53 vs local TEST_DATABASE_URL. NOT YET DEPLOYED.
- **Why:** Operators wanted to see an order's line items (SKU, product, ordered qty, UoM, remaining) without leaving the list view, while keeping the existing full detail page for the SO-number click.

---

## 2026-06-10 15:05 — Deployed notes-write auth (merged fix/auth-unauthenticated-writes to main)
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md` (plus git: pushed branch `fix/auth-unauthenticated-writes` da9ad73 to origin, merged into `main` as `--no-ff` merge commit `60c6f62`, pushed 14:31:58 ET → Railway + Netlify deploys)
- **What changed:** Deployed the API-key requirement on the 4 notes write endpoints plus the paired dashboard.js/key + index.html v=9 changes. Deploy probe positive: `/openapi.json` `info.version` flipped to "3.1.1" at 14:33:10 ET. CD race resolved benignly — Netlify served v=9 at 14:32:29, BEFORE Railway went live, so new JS sent the key to the old API for ~40s (harmless); notes UI had zero downtime. Post-deploy smoke checks all green, replayable (`BASE=https://fastapi-production-b73a.up.railway.app`, key redacted as `<key>`):
  - (a) `curl -X POST $BASE/dashboard/api/notes -H "Content-Type: application/json" -d '{"category":"note","title":"x"}'` → HTTP 401 `{"detail":"API key required","success":false,"error_detail":{"code":"HTTP_401","message":"API key required"}}`
  - (b) `curl -X POST $BASE/dashboard/api/notes -H "X-API-Key: <key>" -H "Content-Type: application/json" -d '{"category":"note","title":"auth smoke - delete me"}'` → id 21 + `success:true`; `curl -X DELETE $BASE/dashboard/api/notes/21 -H "X-API-Key: <key>"` → `{"deleted":true,"id":21,"success":true}`; `curl $BASE/dashboard/api/notes` → 0 matches for id 21/title (18 notes total)
  - (c) `curl -X DELETE $BASE/dashboard/api/notes/999999` (no key) → HTTP 401 same envelope — auth fires before the domain 404
  - (d) `curl -X PATCH $BASE/sales/orders/999999/status -H "X-API-Key: <key>" -H "Content-Type: application/json" -d '{"status":"confirmed"}'` → HTTP 404 `{"detail":"Order #999999 not found","success":false,"error_detail":{"code":"HTTP_404","message":"Order #999999 not found"}}` — byte-shape unchanged from the 12:34 deploy record
  - (e) `railway logs --service FastAPI --since 30m` (both `-d` deploy and runtime streams): clean startup through "Application startup complete", `grep -c READONLY_TRIPWIRE` → 0, `grep -ciE 'error|exception|traceback'` → 0; access log shows exactly the smoke-check requests with expected statuses
  - Netlify verified directly: live index.html references `dashboard.js?v=9`; live dashboard.js shows all four notes fetches sending `X-API-Key`
  - Marked FACTORY_LEDGER_CHANGELOG.md row 32 DEPLOYED (Breaks-If-Reverted intact).
- **Why:** Deploy the June 9 audit's unauthenticated-write fix per the approved plan (consistency, not security — key still published in dashboard.js; this closes the keyless side door so a future rotation covers the full write surface).

---

## 2026-06-10 14:48 — Bump app version 3.1.0→3.1.1 (deploy probe for notes-auth deploy)
- **File(s) changed:** `main.py`, `CHANGE_LOG.md`
- **What changed:** FastAPI `version` string 3.1.0→3.1.1 (main.py:51), no functional change.
- **Why:** Deploy probe — `/openapi.json` `info.version` flipping to 3.1.1 positively confirms the Railway build serving the notes-auth change, same protocol as the 3.1.0 tripwire deploy.

---

## 2026-06-10 14:26 — Required API key on the 4 notes write endpoints (branch fix/auth-unauthenticated-writes, NOT merged/deployed)
- **File(s) changed:** `main.py`, `dashboard/dashboard.js`, `dashboard/index.html`, `tests/test_notes_auth.py` (new), `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** Added the standard `Depends(verify_api_key)` to the only four mutating routes without auth — `POST /dashboard/api/notes`, `PUT /dashboard/api/notes/{note_id}`, `DELETE /dashboard/api/notes/{note_id}`, `PUT /dashboard/api/notes/{note_id}/toggle` (missing key → 401, wrong key → 403; same mechanism as every other write route, no new auth scheme). The four notes fetches in dashboard.js now send `X-API-Key` via the existing in-file `SALES_API_KEY` const; index.html cache-buster bumped `dashboard.js?v=8`→`v=9` (v=8 was already taken by commit 1ff7efa, so not v=7→v=8 as originally planned). Read endpoints (incl. `GET /dashboard/api/notes`) untouched; no OpenAPI changes (gpt-v3 stays at 30 ops). Tests: new `tests/test_notes_auth.py` with 8 tests (401 without key on all four, 403 wrong key, rejected create leaves no row, with-key happy path per endpoint, with-key 404 domain errors preserved); confirmed first that the existing tripwire/contract tests pass unmodified under the new requirement (they already send the key — 43/43 before adding the new file); full suite 51/51 green against local TEST_DATABASE_URL. **Plainly: this change adds consistency, not security — the API key remains published in dashboard.js (served to anyone by Netlify). Its purpose is to make a future key rotation actually cover the full write surface; until that rotation, anyone who reads the dashboard source can still call every endpoint.** **CD race note:** one merge to main triggers both Railway (API) and Netlify (dashboard); if Railway finishes first there is a brief window where the live notes UI 401s on writes until Netlify serves the new JS. Expected, self-healing, no action needed.
- **Why:** June 9 audit's unauthenticated-write exposure: the notes CRUD endpoints were the only mutating routes callable with no key, so a key rotation alone would have left a keyless write door open. Caller analysis showed the four dashboard.js fetches are the only callers (no GPT schemas, no scripts), making this the complete blast radius.

---

## 2026-06-10 13:56 — Housekeeping: gitignore .DS_Store, untrack stray copy; removed merged tripwire branch/worktree
- **File(s) changed:** `.gitignore`, `.DS_Store` (untracked, file kept on disk), `CHANGE_LOG.md`
- **What changed:** Added `.DS_Store` to `.gitignore` and ran `git rm --cached .DS_Store` to stop tracking the stray root copy (the only tracked one). Separately (no file change): removed the now-merged `fix/readonly-tripwire-global-handler` worktree and deleted the branch local (`git branch -d`, merge-safe) + remote.
- **Why:** Post-deploy cleanup of the readonly-tripwire work; keep Finder metadata out of the repo.

---

## 2026-06-10 13:30 — Deployed global readonly tripwire + recovery (merged fix/readonly-tripwire-global-handler to main)
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md` (plus git: pushed branch `fix/readonly-tripwire-global-handler` to origin, merged into `main` as `--no-ff` merge commit `0777df7`, pushed → Railway deploy)
- **What changed:** Deployed the global readonly tripwire (503 + diagnostics + `READONLY_TRIPWIRE` log on every mutating route), the poisoned-connection discard / idempotent-write recovery path, and the rework that composes the tripwire handler with the live `write_response_envelope` (a `psycopg2.Error` handler runs inside ExceptionMiddleware so the envelope post-processes its 503 and adds `error_detail`; the bare-`Exception` handler stays as an outermost safety net). Deploy probe: app version bumped 3.0.0→3.1.0; `/openapi.json` `info.version` flipped to "3.1.0" at 13:24:34 ET (was 3.0.0 through 13:24:18), confirming the new build is serving. Post-deploy smoke checks, all green:
  - **(a) 404 envelope, normal HTTPException path unchanged by the handler rework** — `curl -sS -X PATCH 'https://fastapi-production-b73a.up.railway.app/sales/orders/999999/status' -H 'X-API-Key: <key>' -H 'Content-Type: application/json' -d '{"status":"confirmed"}'` → 404 `{"detail":"Order #999999 not found","success":false,"error_detail":{"code":"HTTP_404","message":"Order #999999 not found"}}`. NO `error_code`/`diagnostics` keys — proves the tripwire handlers did not intercept a normal error path.
  - **(b) reassign 400 already-assigned** — `curl -sS -X POST 'https://fastapi-production-b73a.up.railway.app/lots/473/reassign' -H 'X-API-Key: <key>' -H 'Content-Type: application/json' -d '{"to_product_id":2,"reason_code":"correction"}'` → 400 `{"error":"Lot is already assigned to Almonds – Diced","success":false,"error_detail":{"code":"HTTP_400","message":"Lot is already assigned to Almonds – Diced"}}`. NO `error_code`/`diagnostics`. **Honest note on this probe:** the first attempt omitted the required `reason_code` field (sent only `{"to_product_id":2}`, copying the 12:38 entry's *abbreviated* payload note) and got a `422 VALIDATION_ERROR` — a false deviation caused by the malformed probe, NOT a regression. Confirmed via `git show ddf2a32:main.py` / `f5423b8:main.py` that `LotReassignmentRequest.reason_code` is a required field (`reason_code: str`, no default) and byte-identical in both pre-merge deploys, so the merge changed nothing about this contract. Re-ran with `reason_code` included → the expected 400 above. The full reassign payload is recorded here verbatim so this can't recur.
  - **(c) notes round-trip** — POST `{"category":"note","title":"smoke test - delete me"}` → 200 `{"id":20,...,"success":true}`; `DELETE /dashboard/api/notes/20` → 200 `{"deleted":true,"id":20,"success":true}`; `GET /dashboard/api/notes` confirms id 20 absent.
  - **(d) Railway deployment logs** (`railway logs -d`) — clean startup (pool created, migrations 004–012 up to date, "Application startup complete", uvicorn on :8080); the smoke window shows exactly the expected probe traffic incl. the (b) 422 and both reassign probes; `grep -iE 'READONLY_TRIPWIRE|Unhandled exception|Traceback|ERROR|global_exception|CRITICAL|Application startup failed'` → **0 matches**. No tripwire, no safety-net/global-handler activity, no startup errors.
- **Marked rows 28a + 28b DEPLOYED** in FACTORY_LEDGER_CHANGELOG.md (Breaks-if-Reverted cells verified intact, 7 cells each).
- **Why:** Ship the global readonly observability + recovery work (SO-260514-001 incident follow-up) and prove it composes additively with the already-deployed write-response envelope: a readonly 503 now carries both the tripwire diagnostics and the uniform `error_detail` contract, while normal error paths are byte-for-byte unchanged.

---

## 2026-06-10 13:06 — Fixed FACTORY_LEDGER_CHANGELOG row-31 placement
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Row 31 (write-response contract deploy) had been appended at the end of the file, below the "Permanent Rules" section, instead of in the main change table; moved it to its correct position directly after row 30. Content unchanged.
- **Why:** Keep the regression-guard table scannable — a row outside the table is invisible to the "Breaks If Reverted" pre-change check.

---

## 2026-06-10 13:02 — Bumped app version 3.0.0 → 3.1.0 (deploy probe)
- **File(s) changed:** `main.py`
- **What changed:** FastAPI app version string bumped to "3.1.0" (surfaces as `info.version` in /openapi.json).
- **Why:** Positive deploy confirmation probe for the readonly-tripwire merge — polling /openapi.json for "3.1.0" proves the new build is serving.

---

## 2026-06-10 12:52 — Rebased readonly-tripwire branch onto post-envelope main; handler reworked to compose with write_response_envelope
- **File(s) changed:** `main.py`, `tests/test_readonly_tripwire.py`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md` (branch `fix/readonly-tripwire-global-handler`, NOT merged/deployed)
- **What changed:** (1) Rebased the branch (was a1da52e+a8752ca off 06b6f4a) onto main ddf2a32. Conflict resolutions: combined `Request`+`Response` fastapi imports; kept main's `write_response_envelope` middleware block and the branch's global-handler block side by side; changelogs merged chronologically — the branch's two FACTORY_LEDGER rows renumbered **29→28a, 30→28b** (main had taken 29–31; follows the row-28 chronological-replay precedent). (2) VERIFIED, not assumed, the handler/middleware ordering: in Starlette, a bare-`Exception` handler is hoisted into ServerErrorMiddleware — the OUTERMOST layer, beyond `write_response_envelope` — so its 503 would have SKIPPED the envelope. Reworked: shared `_exception_receipt_response()` now registered twice — `@app.exception_handler(psycopg2.Error)` (runs in ExceptionMiddleware, innermost, so the envelope post-processes the 503 and adds `error_detail`; this is the path real Supabase readonly errors take) and `@app.exception_handler(Exception)` (safety net for non-psycopg2 escapes; replicates `error_detail` itself since the envelope never sees its output, using its own `error_code` as the code so the two paths are distinguishable). (3) Test updates: the negative regression guard's exact-equality assertion `body == {"error":"boom"}` was invalidated by the envelope (which adds `success`/`error_detail` to per-route 500s) — now asserts `error`=="boom" + envelope keys + ABSENCE of `error_code`/`diagnostics`; new Case-5 coexistence test raises a real `psycopg2.errors.ReadOnlySqlTransaction` on ship-commit and asserts 503 + tripwire diagnostics + `success:false` + `error_detail{code:"HTTP_503",message}` — HTTP_503 proves the envelope (not the safety net) added it. (4) Full suite vs local TEST_DATABASE_URL: **43/43 green** (34 existing + 8 branch tripwire + 1 new). Branch logic re-validated against new main: `_is_readonly_error`/`_capture_readonly_diagnostics` unchanged, `putconn(close=True)` discard and idempotent-retry helpers intact, all ~30 per-route re-raise lines merged cleanly into the POSTED_LINES/void-semantics rewrites, `update_customer` retry wiring intact.
- **Why:** Bring the readonly-tripwire branch up to date with the deployed write-response envelope and prove the two compose: a readonly 503 must carry both the tripwire diagnostics and the uniform `error_detail` contract.

---

## 2026-06-10 12:38 — Deployed write-response contract (merged fix/write-response-contract to main)
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md` (plus git: pushed branch `fix/write-response-contract` e9e7482 to origin, merged into `main` as `--no-ff` merge commit f5423b8, pushed → Railway deploy)
- **What changed:** Deployed the uniform write-response envelope + reassign FOR UPDATE bug fix to production. Positively verified the new build live at ~12:34 ET: PATCH /sales/orders/999999/status (status=confirmed to reach the lookup path) returned 404 with both `success:false` and `error_detail:{code:"HTTP_404",...}` — keys the old build lacked. Post-deploy smoke checks, all green: (a) POST /lots/473/reassign with to_product_id=2 (its current product, Almonds – Diced) → 400 `{"error":"Lot is already assigned to Almonds – Diced","success":false,"error_detail":{...}}` — the formerly always-500 endpoint now works; (b) notes round-trip: POST "smoke test - delete me" → id 19 + success:true, DELETE → `{"deleted":true,"id":19,"success":true}`, GET confirms gone; (c) invalid transition on shipped order 3 (SO-260206-003) → 400 with success:false, error_detail, and original detail text ("Allowed transitions from 'shipped': ['invoiced']") intact; (d) GET /inventory/current top-level keys remain `count`/`inventory` — no success key injected. Added row 31 to FACTORY_LEDGER_CHANGELOG.md with the reassign regression called out in Breaks-If-Reverted.
- **Why:** Deploy the June 9 audit's write-contract work and the prod-breaking reassign bug fix (GROUP BY + FOR UPDATE 500'd on every call).

---

## 2026-06-10 12:12 — Uniform write-response contract + local test database (branch fix/write-response-contract, NOT merged/deployed)
- **File(s) changed:** `main.py`, `tests/conftest.py`, `tests/test_write_response_contract.py` (new), `tests/schema/schema.sql` (new), `tests/requirements-test.txt` (new), `scripts/setup_test_db.sh` (new), `scripts/dump_prod_schema.sh` (new), `openapi-gpt-v3.yaml`, `gpt-configs/schemas/openapi-floor.yaml`
- **What changed:** (1) Additive write-response envelope via new `write_response_envelope` HTTP middleware (outermost, registered after CORS): every JSON response to POST/PUT/PATCH/DELETE gains `success: bool` (setdefault — endpoints already returning it are untouched) and, on ≥400, `error_detail: {code, message}` derived from the existing `detail`/`error` keys, which are preserved verbatim. (2) Record ids added where missing: `order_id` on PATCH status / POST lines / PATCH cancel responses; nullable `reassignment_id` (RETURNING id, still best-effort) on POST /lots/{lot_id}/reassign. (3) BUG FIX: reassign's lot query used `GROUP BY ... FOR UPDATE OF l`, which Postgres rejects — the endpoint 500'd on every call in prod; split into row-lock + separate POSTED_LINES aggregate (same pattern as 89e15ae's ship-path fix). (4) Test DB: schema-only pg_dump of prod (zero data rows, verified) committed as tests/schema/schema.sql; setup_test_db.sh builds local factory_ledger_test on Homebrew postgresql@17 (prod is PG 17.6); conftest now uses TEST_DATABASE_URL ONLY (DATABASE_URL fallback removed), hard-refuses (pytest.exit) any URL matching prod hosts BEFORE copying it into DATABASE_URL/importing main, and scrubs DATABASE_URL when unset. (5) 23 new contract tests (happy-path success+id, failure envelopes on both raised-HTTPException and returned-JSONResponse paths, GET-not-enveloped); full suite 34/34 green against local test DB. (6) Response docs updated in openapi-gpt-v3.yaml (op count held at 30) and openapi-floor.yaml (21).
- **Why:** June 9 audit: 12 write endpoints lacked the success+id contract the 9 core inventory ops have; test suite previously ran against the production DATABASE_URL with rollback fixtures.

---

## 2026-06-10 11:31 — Deployed void-semantics migration and executed historical reversal cleanup
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md` (plus git: merged `fix/void-semantics` 2f280b2 into `main` as merge commit c2d8017, pushed; production data change via API)
- **What changed:** Pushed branch `fix/void-semantics` to origin; merged into `main` (regular `--no-ff` merge, c2d8017) and pushed, triggering the Railway deploy. Positively verified the new posted-only build live at 11:21 ET (lot 610 flipped 6,600 → −3,400, lot 617 → −1,000, lot 582 → −2,907 — the runbook's expected interim values). Ran `scripts/cleanup_void_reversals.sh`: all 12 historical reversal transactions (470, 864, 932–941) voided successfully. Verification: 13 of 16 lots matched the expected end state exactly; lots 613 (33,750 vs 36,150), 34 (794.6 vs 801.8), and 48 (1,335.32 vs 1,372.12) mismatched. **Resolution — benign:** all three deltas are fully explained by two real posted production transactions on the evening of 2026-06-09, after the read-only simulation that produced the script's expected values: #1295 (20:13, 4 batches Batch Classic Granola #9: lot 34 −7.2 lb, lot 48 −6.8 lb) and #1299 (20:47, 12 batches Batch Coconut Sweetened Flake: lot 613 −2,400 lb, lot 48 −30.0 lb). Live balances recomputed independently from the posted ledger and confirmed arithmetically correct; **no inventory adjustment made or needed.** Post-cleanup sanity pass (read-only): lots 582/293/612 = 0, 610 = 6,600, 617 = 9,000; `/trace/batch` for both JUN 09 2026 batch lots shows #1295/#1299 ingredients and outputs correctly; `/transactions/history` renders both with full lines; dashboard finished-goods on-hand (Batch #9 6,160 lb; Coconut Flake 14,349 lb) matches posted-only lot sums exactly. Updated `FACTORY_LEDGER_CHANGELOG.md` row 30: branch now DEPLOYED, cleanup COMPLETE, do-not-revert warning active.
- **Why:** Execute the void-semantics deploy per `VOID_SEMANTICS_RUNBOOK.md` after floor close; the 12 historical posted reversals had to be voided immediately post-deploy to close the interim negative-balance window.

---

## 2026-06-09 15:39 — Unified void semantics: posted-only balance math, void no longer posts reversals
- **File(s) changed:** `main.py`, `tests/test_void_semantics.py`, `scripts/cleanup_void_reversals.sh`, `VOID_SEMANTICS_RUNBOOK.md`, `gpt-configs/schemas/openapi-floor.yaml`, `gpt-configs/sources/floor-specific.md`, `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md`, `FACTORY_LEDGER_CHANGELOG.md`, `.gitignore`
- **What changed:** Added `POSTED_LINES` subquery constant + `lot_on_hand()` helper as the single source of truth for balance math; all ~50 balance/on-hand/availability queries (inventory, lots, ship/make/pack/adjust availability, sales-order ship, dashboard panels, production requirements, day summary, planner loaders, audit checks, trace on-hand scalars, lot timeline) now read transaction lines through a `status='posted'` filter. Reworked `POST /void/{id}`: flips status to 'voided' with optional `{"reason"}` body appended to notes, no reversal transaction inserted; response keeps `reversal_transaction_id`/`reversal_lines` keys (now always null/empty) for backward compatibility. Added 4 tests (void restores balance with no new row; all balance endpoints agree with voided rows present; lot-582 regression pattern; double-void fails cleanly). Added post-deploy cleanup script for the 12 historical reversal transactions (470, 864, 932–941) with expected-balance verification, plus runbook with deploy-timing warnings. Floor GPT schema/instructions prose updated to stop describing reversals (contract/keys unchanged, 21 ops; openapi-gpt-v3.yaml untouched at 30 ops). No schema migration needed.
- **Why:** The old void endpoint marked the original voided AND posted a reversal, while some balance queries counted voided lines and others filtered posted — so views disagreed by exactly the voided amounts (lots 582/293/612). Branch `fix/void-semantics`; NOT deployed.

---

## 2026-06-08 15:32 — Bumped dashboard asset cache versions for SO Inventory UI
- **File(s) changed:** `dashboard/index.html`, `CHANGE_LOG.md`
- **What changed:** Updated dashboard asset query strings to `dashboard.css?v=5` and `dashboard.js?v=7` so browsers fetch the new Sales Order Inventory UI after deploy.
- **Why:** The feature branch changed both dashboard CSS and JS; without a cache-buster bump, the deployed page could keep serving stale assets.

---

## 2026-06-08 15:02 — Added SO line Inventory summaries and active-lot FG filter
- **File(s) changed:** `main.py`, `dashboard/dashboard.js`, `dashboard/dashboard.css`, `CHANGE_LOG.md`
- **What changed:** Extended the existing `/dashboard/api/inventory/finished-goods` query to include only active lots. Added lazy per-line Inventory toggles on Sales Order detail lines that render On Hand, Remaining, and Delta using the existing case/pallet display math.
- **Why:** Operators need a fast on-hand vs. remaining comparison directly on Sales Order detail lines without showing lot-level or allocation detail.

---

## 2026-06-01 12:38 — Cleaned stale 25 LB Bulk Cases SKUs to clear dashboard "Missing SKUs" alert
- **File(s) changed:** `dashboard/dashboard_config.json`, `FACTORY_LEDGER_CHANGELOG.md`, `CHANGE_LOG.md`
- **What changed:** In the `bulk_25lb` panel's `skus` list: renamed `"Granola Vanilla Crisp 25 LB"` → `"Granola Vanilla Crisp 25 LB (French Vanilla)"` (matches active product id=134); removed `"Granola Setton Morning Latte Crunch 25 LB"` (retired, id=130 archived in migration 035); removed `"Classic Granola 25 LB"` (dead duplicate of active id=136 `"Granola Classic 25 LB"`; the id=171 row is deactivated) and stripped the resulting dangling trailing comma off `"Granola Setton French Vanilla 25 LB"`. List count 15→13; file re-validated with `python3 -m json.tool`. Added regression-guard row #29 to `FACTORY_LEDGER_CHANGELOG.md`.
- **Why:** The Factory Dashboard Operations tab rendered a red "Missing SKUs" alert on the "25 LB Bulk Cases" group. The backend (`main.py` `dashboard_api_finished_goods`) flags any config SKU with no exact case-insensitive name match against an `active=true` product (main.py:6878-6879). The three flagged names were a name-format mismatch (French Vanilla suffix) plus two retired/duplicate product rows — config staleness, not missing inventory. Landed via an isolated git worktree off `origin/main` (branch `chore/fg-config-cleanup`) and pushed direct to `main`; the dirty `feat/planner-v2` working tree was left untouched.

---

## 2026-05-28 13:30 — Idempotency key plan (design artifact, no implementation)
- **File(s) changed:** `IDEMPOTENCY_KEY_PLAN.md` (new)
- **What changed:** Created design document covering idempotency keys for ship, make, pack, adjust, and ship_order commit endpoints. Covers: what makes each request uniquely identifiable, shared `idempotency_keys` table schema, client-supplied `Idempotency-Key` header contract, replay detection flow, per-endpoint write inventory (what gets skipped on replay), no-double-write test plans, implementation sequence, and open questions.
- **Why:** Layer 3 of the readonly-transaction incident fix stack. Layers 1 (tripwire) and 2 (connection discard) are committed; this plan makes client retry safe by detecting and refusing duplicate writes.

---

## 2026-05-28 13:05 — Readonly connection discard + safe idempotent-write recovery path
- **File(s) changed:** `main.py`, `tests/test_readonly_tripwire.py`, `FACTORY_LEDGER_CHANGELOG.md` (row #30)
- **What changed:** `get_db_connection()` now detects readonly errors and calls `putconn(conn, close=True)` to evict the poisoned connection from the psycopg2 pool instead of returning it. New helpers: `_discard_readonly_connection` (rollback + close), `_verify_connection_writable` (SELECT checks `transaction_read_only` + `pg_is_in_recovery()`), `_run_db_write` (conn lifecycle with discard-on-readonly), `run_idempotent_write_with_readonly_retry` (retry once after discarding bad conn + verifying fresh conn is writable). `PATCH /customers/{customer_id}` rewired through the retry helper as proof-of-concept. Ship/make/pack/adjust intentionally NOT wired — they fail loud with 503 via the global handler (#29). Two new tests added to `tests/test_readonly_tripwire.py`: discard test verifies `putconn(close=True)`, retry test verifies one retry on fresh writable connection. All 8 tests pass (6 existing + 2 new).
- **Why:** After a Supabase failover window, poisoned readonly connections remained in the psycopg2 pool and failed subsequent requests even after the primary was restored. Discarding bad connections and providing a safe retry path for idempotent writes prevents cascading failures post-failover.

---

## 2026-05-28 11:48 — Global readonly tripwire (completes #27 follow-up)
- **File(s) changed:** `main.py`, `tests/test_readonly_tripwire.py`, `FACTORY_LEDGER_CHANGELOG.md` (row #29), `CHANGE_LOG.md` (this entry)
- **What changed:** Added `@app.exception_handler(Exception)` that detects readonly errors via the existing `_is_readonly_error(exc)` helper, runs `_capture_readonly_diagnostics()` on a fresh pooled connection, returns HTTP 503 with `{"success": false, "error_code": "READONLY_TRANSACTION", "retryable": true, "error": <str(exc)>, "diagnostics": {...}, "message": "..."}`, and emits a single `READONLY_TRIPWIRE:` ERROR log line (grep-friendly JSON). Added 33 `if _is_readonly_error(e): raise` lines across per-route `except Exception` blocks (ship_commit, `PATCH /lots/{lot_id}/rename`, `POST /admin/bom/{product_id}/lines`, `POST /dashboard/api/notes`, pack, make, adjust, receive, et al.) so the readonly case bubbles to the global handler instead of being swallowed into a per-route 500. New test file `tests/test_readonly_tripwire.py` (6 tests, 6 pass): positive ship_commit 503+log, negative regression guard (`RuntimeError("boom")` keeps per-route 500/`{"error":"boom"}`/no tripwire log), parametrized 4-route coverage (ship/lots-rename/admin-bom/dashboard-notes). Tests use `httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False))` + `@pytest.mark.anyio` — replaces starlette TestClient (incompatible with container's httpx 0.28.1). Existing 7 tests (`test_resolve_customer`, `test_ship_order_service_line`) unchanged. FACTORY_LEDGER_CHANGELOG row #29 added.
- **Why:** SO-260514-001 ship_order incident: PR #6's tripwire (#27) only covered `update_order_header`; ship_commit and every other mutating route swallowed readonly errors into a silent per-route 500 with no diagnostic receipt and no `READONLY_TRIPWIRE` log line. Completes the global-handler follow-up flagged in #27.

---

## 2026-05-19 15:42 — Executed STATUS.md `git mv` plan in four chore commits + reflected post-move paths
- **File(s) changed:** `STATUS.md` (paths updated to post-move locations + executed-note added), `CHANGE_LOG.md` (this entry). Commits adding the moves: `947fdd5`, `cc0b380`, `d22977e`, `ba5baab`.
- **What changed:** Ran the proposed `git mv` plan from STATUS.md in four separate commits on branch `claude/serene-panini-dc449d`, one per category, using `git mv` so history is preserved. (1) `chore: archive superseded OpenAPI schemas` — moved `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `openapi-schema.yaml` → `archive/superseded-schemas/`. (2) `chore: archive historical one-shot migrations` — moved the entire `migrations/` directory (29 files: 003-036) → `archive/migrations-applied/sql/` and `factory_ledger_reconciliation.sql` → `archive/migrations-applied/`. (3) `chore: relocate audit snapshots` — moved `AUDIT_GPT_FABRICATION_2026-04-21.md`, `TRACEABILITY_AUDIT_2026-03-24.md`, `DASHBOARD_ACTIONABILITY_AUDIT.md` → `audits/reports/` (alongside existing `audits/2026-05/`). (4) `chore: archive superseded instructions and v2.5.0-era docs` — moved `GPT_INSTRUCTIONS.md`, `GUIDE.md`, `SALES_API.md` → `archive/superseded-instructions/`. Each commit staged only the files in its category. STATUS.md updated post-hoc to swap original paths for post-move paths in every table row that changed location, and a notice was added near the top recording the four commit SHAs. `FOLLOWUPS.md` was deliberately NOT moved (the four-commit directive scoped the work to schemas/migrations/audits/instructions only). The LIVE files at the repo root (`main.py`, `openapi-gpt-v3.yaml`, `gpt-instructions-v3.md`, `gpt-configs/`, `dashboard/`, `tests/`, `scripts/`, deploy configs, `CLAUDE.md`, `CHANGE_LOG.md`, `FACTORY_LEDGER_CHANGELOG.md`, `DEPLOYMENT.md`, `CONTEXT.md`, `KEEPALIVE.md`) were all left in place — moving them would have broken Railway entrypoint, Netlify publish path, Custom-GPT Actions URL, or the CLAUDE.md HARD RULE pinning `openapi-gpt-v3.yaml`.
- **Why:** User asked for the proposed plan in STATUS.md to be executed in four category-scoped commits on this branch, then pushed. Granular commits keep each archival decision independently revertable; `git mv` preserves rename history so historical `git log --follow` against any of the moved files still works.

---

## 2026-05-19 15:30 — Added STATUS.md classifying repo files vs deployed GPT configs
- **File(s) changed:** `STATUS.md` (new)
- **What changed:** Read `~/Downloads/Factory_Ledger_GPT_Config.docx` (main GPT v3.6.0 / API 3.4.0 / 30 ops) and `~/Downloads/Factory_Ledger_Floor_GPT_Config.docx` (Floor / API 4.0.0 / 21 ops) as ground truth. Classified every file in this worktree as LIVE / HISTORICAL / SUPERSEDED / AUDIT / WORKING. Verdicts: `openapi-gpt-v3.yaml` and `gpt-configs/schemas/openapi-floor.yaml` are LIVE (exact title/version/op-list match); `openapi-v3.yaml` (3.3.0), `openapi-schema-gpt.yaml` (self-DEPRECATED 2.7.0), `openapi-schema.yaml` (2.7.0) → SUPERSEDED. `gpt-instructions-v3.md` (8175 B) is LIVE for the main GPT (has PRE-FLIGHT — CUSTOMER + 4xx detail.error_code lines that match deployed v3.6.0); `GPT_INSTRUCTIONS.md` (7831 B) → SUPERSEDED. `gpt-configs/sources/{shared-rules,floor-specific}.md` + `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` + `build_gpt_instructions.py` → LIVE. All 27 files in `migrations/` are unreferenced one-time SQL → HISTORICAL. `factory_ledger_reconciliation.sql`, `keepalive.log` → HISTORICAL. `GUIDE.md` + `SALES_API.md` (both v2.5.0-era) → SUPERSEDED. `AUDIT_GPT_FABRICATION_2026-04-21.md` + `TRACEABILITY_AUDIT_2026-03-24.md` + `DASHBOARD_ACTIONABILITY_AUDIT.md` → AUDIT. `FOLLOWUPS.md` → WORKING. Proposed (NOT executed) `git mv` plan moves SUPERSEDED schemas, SUPERSEDED instructions, applied migrations, and audits into `/archive/superseded-schemas`, `/archive/superseded-instructions`, `/archive/migrations-applied`, `/audits/reports`, and FOLLOWUPS into `/working`. `/live` directory intentionally NOT created — LIVE files are deploy-pinned (Railway entrypoint, Netlify publish path, `CLAUDE.md` HARD RULE on `openapi-gpt-v3.yaml` filename) so moving them would break production.
- **Why:** User requested a classification + proposed-only move plan to tidy the repo against the actual deployed GPT configs.

---

## 2026-05-19 12:08 — Dropped products.bake_line column + resolved 035 slot conflict (renumbered to 036)
- **File(s) changed:** `migrations/036_drop_bake_line.sql` (new — originally authored as `035_drop_bake_line.sql` in sibling worktree `optimistic-dijkstra-b5d5ae`, renumbered on import), `migrations/034a_fruit_nut_bom_bake_line_copack.sql` (header references updated 035→036), production `products` table (column dropped — already applied 2026-05-19 11:47 ET)
- **What changed:** Brought `drop_bake_line` migration into this worktree under the new number 036. Body of the file is unchanged from the sibling-worktree original: single `ALTER TABLE products DROP COLUMN bake_line` wrapped in `BEGIN; SET TRANSACTION READ WRITE; … COMMIT;`. Added a "Slot history" section to the file header documenting the renumber. Updated `034a_fruit_nut_bom_bake_line_copack.sql` header (lines 5 and 12) to point at the new filename and to reflect the surrounding sequence `…, 034, 034a, 035 (planner-v2's backfill_parent_batch), 036 (this file)`. The on-prod DROP was already executed 2026-05-19 11:47 ET from the sibling worktree — this file is now the on-repo record of that production change.
- **Why:** Two migrations were both claiming slot 035: (a) `feat/planner-v2`'s `035_backfill_parent_batch_product_id_and_archive_retired_fgs.sql` (committed 2026-04-30, applied to prod 2026-04-30 ~11:22 ET, unmerged to main), and (b) sibling worktree `optimistic-dijkstra-b5d5ae`'s `035_drop_bake_line.sql` (created and applied to prod today, 2026-05-19 11:47 ET). Conflict resolved by **chronological-replay rule: production apply order is canonical, regardless of which branch a migration was committed to first.** Earlier-applied migration keeps the contested slot; later-applied migration renumbers to the next free slot. `backfill_parent_batch` (applied 2026-04-30, 19 days earlier) keeps 035; `drop_bake_line` (applied today) becomes 036. This rule also keeps numeric-sort replay aligned with the real production timeline — `… 034 < 034a < 035 < 036` matches the actual apply sequence `parent_batch → 034a (bake_line ADD) → drop_bake_line`. **Precedent for future conflicts: when two branches independently claim the same migration number, neither commit date nor first-merged-to-main wins — the migration that ran against production first keeps the slot.**

---

## 2026-05-19 11:57 — Backfilled migration 034a for 2026-05-18 direct-SQL session
- **File(s) changed:** `migrations/034a_fruit_nut_bom_bake_line_copack.sql` (new)
- **What changed:** Created historical/backfill migration that captures, verbatim, the BEGIN…COMMIT transaction applied via Supabase SQL Editor on 2026-05-18 15:23 ET: (1) UPDATE products id=179 (Granola Fruit Nut Batch — name, brand='CNS', default_batch_lb 25→384.52, yields 15/38, verified); (2) clear-and-replace product_bom for finished_product_id=179 with 5 rows (Classic 323 lb + 4 inclusions @ 15.38 lb); (3) ALTER TABLE products ADD COLUMN bake_line text + UPDATE 7 Sunshine SKUs to 'SS'; (4) ALTER TABLE products ADD COLUMN is_copack boolean NOT NULL DEFAULT false + UPDATE 9 co-pack SKUs to true. File slotted as 034a (between existing 034_force_close_so260414003_hannas.sql and the planned 035_drop_bake_line.sql) to preserve real chronology for replay parity. Header comment marks it as already-applied and NOT to run against live DB.
- **Why:** Today's 035_drop_bake_line.sql (in sibling worktree) references a column never added by any prior migration in this branch's history. Backfilling 034a fixes the replay asymmetry so a fresh-schema replay of 030→031→…→034→034a→035 produces the correct current production state (Fruit Nut BOM present, is_copack present, bake_line absent).

---

## 2026-05-18 21:10 — Read-only transaction tripwire + baseline capture (uncommitted)
- **File(s) changed:** `main.py`, `audits/2026-05/readonly-baseline-20260518T210745Z.json` (new)
- **What changed:** (1) Captured baseline diagnostic against prod via `POST /admin/sql`: `default_ro=off`, `txn_ro=off`, `is_replica=false`, `usr=postgres`, `db=postgres`, PG 17.6 — saved verbatim to `audits/2026-05/readonly-baseline-20260518T210745Z.json` as the "healthy state" receipt. (2) Added module-level `READONLY_PROBE_SQL` constant + `_is_readonly_error(exc)` + `_capture_readonly_diagnostics()` helpers right after `get_transaction()` (around line 511). The diagnostics helper grabs a fresh pooled connection (not the failing request's poisoned one), runs the probe, and returns the row as a dict — wrapped in its own try/except so a probe failure yields `{"probe_error": "..."}` instead of masking the original error. (3) Enriched the per-route exception handler at the end of `PUT /sales/orders/{order_id}` (now at line 5719): only when `_is_readonly_error(e)` is true, attach the probe result under a `diagnostics` key in the JSON response AND emit a single-line ERROR log prefixed `READONLY_TRIPWIRE:` followed by JSON for grep-friendly Railway logs. Non-read-only errors return the original `{"error": str(e)}` shape unchanged. No retry logic, no new dependencies, status code stays 500.
- **Why:** Prod intermittently fails UPDATEs with `cannot execute UPDATE in a read-only transaction`, currently not reproducible (30/30 UPDATEs succeed). Most likely a transient Supabase failover window. Tripwire ensures the next occurrence leaves a receipt — session state (`default_ro`, `txn_ro`, `inet_server_addr`, `current_user`, `pg_is_in_recovery`) captured at the moment of failure so we can name the root cause without guessing. Change is uncommitted working-tree only; not yet deployed.

---

## 2026-04-22 12:10 — Reorganize GPT instruction + schema files into gpt-configs/
- **File(s) changed:** `gpt-configs/sources/shared-rules.md` (new), `gpt-configs/sources/floor-specific.md` (new), `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md` (new), `gpt-configs/schemas/openapi-floor.yaml` (new), `gpt-configs/README.md` (new), `build_gpt_instructions.py` (new, moved from ~/Downloads + path update)
- **What changed:** Created `gpt-configs/{sources,dist,schemas,archive}/` layout. Moved today's Floor & Fulfillment artifacts (shared-rules, floor-specific, GPT_FLOOR_INSTRUCTIONS, openapi-floor) from `~/Downloads` into the new structure. Placed `build_gpt_instructions.py` at repo root and updated its path constants to read from `gpt-configs/sources/` and write to `gpt-configs/dist/`. Added `parents=True` to the DIST_DIR mkdir. Wrote `gpt-configs/README.md` explaining the layout and edit-build-paste workflow. Verified script: `python build_gpt_instructions.py floor` → exit 0, 6,623 chars, generated header intact.
- **Why:** Scales cleanly from today's single Floor GPT to the planned 3-GPT split (Floor & Fulfillment, Sales & Admin, Trace & Recall). Keeps canonical sources separate from generated outputs so drift stops being possible. Legacy root-level GPT files (`GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md`, `openapi-gpt-v3.yaml`) left in place per instructions — they migrate to `archive/` once the three new GPTs replace the current production one.

---

## 2026-04-22 — Let HTTPException pass through in /receive preview
- **File(s) changed:** `main.py`
- **What changed:** Added `except HTTPException: raise` clause before the existing `except Exception as e` block at the end of the `/receive` preview branch (around line 2304). Other endpoints' preview branches already have this pattern.
- **Why:** The preview branch was catching `HTTPException` via the broad `except Exception`, swallowing legitimate 4xx responses (notably the 409 from `resolve_product_full` with disambiguation suggestions) and converting them to empty 500 errors with just `{"error": str(e)}`. Now HTTPException passes through to FastAPI's default handler with its original status and detail body intact.

---

## 2026-04-21 — Close print-fabrication gap in NEVER CLAIM SUCCESS rule
- **File(s) changed:** `gpt-instructions-v3.md`, `GPT_INSTRUCTIONS.md`
- **What changed:** Appended `You can't print.` to the NEVER CLAIM SUCCESS CRITICAL RULE in both files. Line 103 PACKING SLIP rule `NEVER say "Printing."` left as-is (token-level belt-and-suspenders). Final char counts: `gpt-instructions-v3.md` 7,966 → **7,983 / 8,000** (17 headroom); `GPT_INSTRUCTIONS.md` 7,632 → **7,649 / 8,000** (351 headroom).
- **Why:** Follow-up verification caught that the earlier sprint's removal of the `NEVER FAKE PRINTING` CRITICAL RULE was not fully covered by the new `NEVER CLAIM SUCCESS` rule. Two gaps: (1) `Printed` is absent from the `Done/Updated/Created/Cancelled/Shipped` verb list; (2) the rule's scope clause `only after a successful mutation response` does not apply to packing slips, which are read-path (listOrders to get order_id, then a static link is returned — no mutation endpoint is ever called). A strict reading of NEVER CLAIM SUCCESS therefore does not forbid `"Printed ✅"` fabrications. The retained `NEVER say "Printing."` on line 103 is literal-match only and does not cover `Printed`/`Sent to printer`/other variants. The removal + restoration within the same sprint is intentional, not a reversal: the initial offset assumed the new rule covered print; this follow-up analysis showed it doesn't, so one capability disclaimer (`You can't print.`) is restored in the semantically correct place — the CRITICAL RULES block alongside NEVER CLAIM SUCCESS — rather than re-adding a full NEVER FAKE PRINTING line.

---

## 2026-04-21 — Ship audit fix: anti-fabrication rule + status transition graph in GPT instructions
- **File(s) changed:** `gpt-instructions-v3.md`, `GPT_INSTRUCTIONS.md`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** (a) Added new CRITICAL RULE `NEVER CLAIM SUCCESS — "Done/Updated/Created/Cancelled/Shipped" only after a successful mutation response this turn. Never fake a tool call.` (b) Replaced ambiguous `Status: new→confirmed→...→invoiced` one-liner in SALES ORDERS with explicit one-step transition block including the `ready↔in_production` reverse edge, `shipped/partial_ship auto-only via shipOrder` callout, and the worked `confirmed→ready = confirmed→in_production→ready` example. (c) Added `Status → updateOrderStatus (one-step only; see SALES ORDERS)` bullet to ORDER EDITING — CALL API IMMEDIATELY so the GPT knows the Action exists and routes to it on "mark ready"/"move to production"/"cancel". Offsets to stay under 8,000-char ceiling: removed `Max 1 emoji per message.` from ACT DON'T LOOP; removed `Never restate what you're about to do.` from ACT DON'T LOOP (redundant with `No reconfirmation`); removed `- NEVER FAKE PRINTING — You CANNOT print. Clickable links only.` CRITICAL RULE (functionality retained in PACKING SLIP section `NEVER say "Printing."`); removed `- When in doubt → inventoryLookup first (fast, useful while you plan next call)` from ROUTING RULES. Final counts: `gpt-instructions-v3.md` 7,966 / 8,000 chars; `GPT_INSTRUCTIONS.md` 7,632 / 8,000 chars. Synced identical changes to both files since both claim v3.6.0 and which is deployed to the Custom GPT admin is ambiguous (dual-file drift flagged in audit as out-of-scope follow-up). Added FACTORY_LEDGER_CHANGELOG row #27.
- **Why:** Shipping the Finding 1 + Finding 2 instruction fixes from [AUDIT_GPT_FABRICATION_2026-04-21.md](AUDIT_GPT_FABRICATION_2026-04-21.md) alone, per user direction (don't bundle with Pass-1 4xx normalization). Every day the current instructions are live is another day ship_order / updateSupplierLot / shipOrder could fabricate an FDA-recall-tier success message without a tool call. OpenAPI enrichment and optional main.py 4xx normalization remain deferred to a separate change.

---

## 2026-04-21 — Audit: GPT fabricates mutation confirmations without calling the API
- **File(s) changed:** `AUDIT_GPT_FABRICATION_2026-04-21.md` (new)
- **What changed:** Created findings doc for the incident where the GPT replied `"Done. All 5 orders are now set to ready ✅"` without invoking `updateOrderStatus`. Scoped Step 0 design-intent review (changelog rows #7, #11, #13, #21), GPT instruction audit (`gpt-instructions-v3.md` 7,987 chars / 8,000 ceiling), OpenAPI audit (`updateOrderStatus` at line 807, op count 30/30 at cap), `main.py:5528-5583` handler review with full `MANUAL_TRANSITIONS` graph extracted from [main.py:4991](main.py#L4991), and a cross-cut risk table for all 15 GPT-exposed mutation endpoints. Two findings: (1) no CRITICAL RULE forbids claiming mutation success without a tool call — `NEVER HALLUCINATE` is parsed as "don't invent rows," doesn't cover "don't invent outcomes"; (2) transition graph is shown as an ambiguous display-order one-liner rather than a one-step-transitions table, and omits `ready→in_production` reverse edge + `shipped`/`partial_ship` auto-only flag. Proposed instruction text, OpenAPI enrichment, and optional 4xx normalization (bundled with FOLLOWUPS #2).
- **Why:** Operator reported the GPT fabricating `"Done ✅"` responses across 3 turns before actually invoking the API. Incident transcript pasted into audit prompt. Audit-only — no code changes made this session per prompt constraints.

---

## 2026-04-20 15:40 — FOLLOWUPS #6 resolved: deleted legacy `factory-ledger` Railway service
- **File(s) changed:** `FOLLOWUPS.md`
- **What changed:** Marked #6 as RESOLVED with investigation findings. Both services (`factory-ledger` and `FastAPI` in Railway project `gleaming-solace`) were deploying the same `main.py` from the same repo with identical env-var key sets; only `DATABASE_URL` differed (factory-ledger had a stale Supabase pooler password, no `sslmode`). `factory-ledger-production.up.railway.app` appeared nowhere in the repo (GPT instructions, OpenAPI, dashboard config, Netlify — all clean). Live check confirmed `factory-ledger` at HTTP 502, `FastAPI` at HTTP 200. Deleted the `factory-ledger` service via the Railway dashboard (CLI has no service-delete verb). Relinked local `railway` CLI to `FastAPI` so subsequent `railway ssh` / `railway logs` hit the live service.
- **Why:** Dead service was burning restart budget and polluting deployment logs. Investigation (compared env keys via `railway variables --kv`, pinged both public URLs, grepped repo for `factory-ledger-production`) confirmed zero external dependency. Most likely origin: `factory-ledger` was the original service (repo-named); when the Supabase pooler password rotated, a fresh `FastAPI` service was created with new creds instead of updating the old one, and the original was left orphaned.

---

## 2026-04-20 14:50 — F05-04 fix: ship_order auto-fulfills is_service lines
- **File(s) changed:** `main.py`, `tests/test_ship_order_service_line.py`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** `ship_order` commit path now detects `is_service=true` lines (pallet charges, freight, etc.) and auto-fulfills them without inventory lookup — sets `line_status='fulfilled'`, bumps `quantity_shipped_lb` by the remaining quantity, skips the `transactions`/`transaction_lines`/`shipment_lines`/`sales_order_shipments` inserts. Zero-shipment guard updated to only count physical (non-service) shipped_lb, so a pallet-only auto-fulfill can't bypass it. Preview path now also skips the stock lookup for service lines and returns `is_service: true` with `on_hand_lb: null, short: 0`. Added `tests/test_ship_order_service_line.py` with three DB-backed scenarios (A: full-stock+service → shipped; B: partial-stock+service → partial_ship with service fulfilled; C: service-only → ZERO_SHIPMENT, no state change). Response `results[]` dicts include `is_service: true` on service rows.
- **Pattern:** Matches existing `is_service` handling in `create_sales_order` ([main.py:5029-5067](main.py#L5029)) and `add_order_lines` ([main.py:5722-5733](main.py#L5722)), and FACTORY_LEDGER_CHANGELOG rows 1 and 13 which filter service lines out of weight totals and relax the `quantity_lb > 0` check. `ship_order` was the one remaining path that lacked the guard.
- **Why:** Audit finding `audits/2026-04/15-traceability-gaps.md` F05-04. Orders containing a Pallet Charge could never transition to `status='shipped'` because FIFO lookup returned zero for service products, tripping `actual_ship <= BALANCE_EPSILON` and leaving `all_fully_shipped=False` forever. Migrations 022 and 024 were operator SQL workarounds for this exact bug — further reverts would resume forcing operators to write one-off close-order migrations per stuck order. Per `audits/2026-04/17-migration-integrity.md` F17-06, F05-04 is one of the last remaining root causes of the recurring close-order migration pattern (the other was standalone `/ship` not writing shipments, fixed by FACTORY_LEDGER_CHANGELOG row 15 / GAP-3).
- **Breaks if reverted:** `ship_order` goes back to marking any order with a service line as `partial_ship` permanently; operators return to closing such orders via raw SQL (see migrations 022, 024); the `/sales/orders/{id}/ship/preview` response also resumes showing misleading "no stock / short: N" warnings for pallet-charge lines.

---

## 2026-04-20 12:55 — FOLLOWUPS #3 & #4 enriched + #6 added (factory-ledger service crashlooping)
- **File(s) changed:** `FOLLOWUPS.md`
- **What changed:** Enriched #3 (`_pick_by_address` threshold tuning) with the Setton address specifics, instrumentation guidance (log fallthrough-to-409 calls with top-two `addr_sim` scores), 0.5/0.15 loosening proposal, and "tune from data, not intuition" framing. Enriched #4 (GPT instruction headroom) with Pass 1 estimate-calibration detail (estimated ~52 chars, actual 91, 39-char delta unaccounted). Added #6 — `factory-ledger` service in Railway project `gleaming-solace` is crashlooping with "password authentication failed for user postgres"; prod traffic unaffected (FastAPI service serves GPT), but dead service consumes restart budget.
- **Why:** Pass 1 merge close-out. Reviewer asked for Setton-specific threshold context and estimate-calibration ask so next pass has concrete levers. Crashloop was stumbled into during pytest verification via `railway ssh`; capturing now so it's not lost.

---

## 2026-04-20 12:14 — Gitignore .env + archive migrations 032-034
- **File(s) changed:** `.gitignore`, `migrations/032_backfill_skus_and_merge_bs_cocoa.sql`, `migrations/033_force_close_so260326002_ace_endico.sql`, `migrations/034_force_close_so260414003_hannas.sql`
- **What changed:** Appended `.env` to `.gitignore` (history confirmed clean — `git log --all --full-history -- .env` returned empty, so no credential rotation needed). Committed migrations 032-034 into the repo; all three were already applied to prod, this aligns the checked-in history with deployed schema.
- **Why:** Pre-smoke-test housekeeping. `.env` was previously untracked but unignored, one `git add .` away from leaking Supabase/Railway/Google OAuth creds. Migrations 032-034 had been sitting untracked since application; committing them keeps `migrations/` in sync with prod.

---

## 2026-04-20 12:20 — FOLLOWUPS #5 added (createOrder auto-create audit) + Pass 1 merge
- **File(s) changed:** `FOLLOWUPS.md`
- **What changed:** Added section #5 — deferred audit item flagging that `create_sales_order` (main.py:5010) calls `resolve_customer_id` without `auto_create=False`, so a typo'd name with no address silently creates a duplicate customer row. Proposed fix: trigram-similarity guard (>0.7) before the auto-create branch fires.
- **Why:** Known gap in Pass 1's address-tiebreaker coverage. Address is optional on OrderCreate, so typos-without-address still slip through. Not blocking Pass 1 merge — recorded for a follow-up PR.

---

## 2026-04-20 12:05 — Pre-merge verification: Pass 1 Setton tests + FOLLOWUPS additions
- **File(s) changed:** `FOLLOWUPS.md`
- **What changed:** Ran pytest `tests/` inside the Railway fastapi container (Python 3.11, Supabase prod DB via SAVEPOINT+ROLLBACK fixture) against the Pass 1 local `main.py` — 4/4 tests PASSED. Added two items to FOLLOWUPS: (3) threshold tuning for `_pick_by_address` after 2–4 weeks of traffic, (4) GPT instruction headroom note (7,987/8,000 chars).
- **Why:** Four pre-merge checklist items for Pass 1: actual test run (not just compile check), audit of `resolve_customer_id` call sites, FOLLOWUPS additions, and CUSTOMER_NOT_FOUND reachability trace from createOrder. Audit and trace reported in-chat; no code changes required.

---

## 2026-04-20 — Pass 1: customer address tiebreaker + 4xx error shape normalization
- **File(s) changed:** `main.py`, `gpt-instructions-v3.md`, `openapi-gpt-v3.yaml`, `tests/__init__.py` (new), `tests/conftest.py` (new), `tests/test_resolve_customer.py` (new), `pytest.ini` (new), `FOLLOWUPS.md` (new)
- **What changed:**
  - `resolve_customer_id` now accepts an optional `address` kwarg; when the fuzzy name match returns >1 candidate and an address is supplied, a trigram-similarity tiebreaker on `customers.address` (thresholds 0.6 absolute + 0.2 gap) collapses to a single match. Fully additive: no address → behavior unchanged. New helper `_pick_by_address` in main.py.
  - `customer_address` field added to `OrderCreate` (POST /sales/orders) and `ShipRequest` (POST /ship), threaded through both `resolve_customer_id` call sites.
  - `resolve_customer_id` final 404 "Customer not found" now uses the standard dict error shape (error_code=CUSTOMER_NOT_FOUND). Auto-insert now persists supplied address.
  - `resolve_product_id` and `resolve_product_full` normalized to dict shape. Status-code shift 400 → 409 for PRODUCT_AMBIGUOUS and PRODUCT_UNCERTAIN (matches CUSTOMER_AMBIGUOUS). 404 PRODUCT_NOT_FOUND unchanged.
  - `resolve_order_id` 404 normalized to dict shape (error_code=ORDER_NOT_FOUND).
  - Sales-order endpoints `createOrder`, `getOrder`, `updateOrderHeader`, `addOrderLines`, `updateOrderLine` — all string-detail 4xx raises converted to dict shape (error_codes: ORDER_NOT_FOUND, CUSTOMER_NOT_FOUND, ORDER_HEADER_LOCKED, NO_FIELDS_TO_UPDATE, CASE_WEIGHT_REQUIRED, ORDER_LINES_LOCKED, LINE_NOT_FOUND).
  - `gpt-instructions-v3.md`: added PRE-FLIGHT — CUSTOMER section (searchCustomers before resolveProducts on PO entry; batched disambiguation notation); removed duplicate DAY SUMMARY routing line; expanded ERRORS section with "4xx with detail.error_code + detail.suggestions → show suggestions" rule. 7,643 → 7,987 chars (under 8,000 cap).
  - `openapi-gpt-v3.yaml`: added `components.schemas.ErrorResponse`; added `customer_address` field to OrderCreate/ShipRequest; added 400/404/409 response blocks referencing ErrorResponse on createOrder, getOrder, updateOrderHeader, addOrderLines, updateOrderLine (only status codes each operation actually raises). Operation count held at 30/30.
  - `tests/` scaffolding: pytest.ini, conftest.py with `db_cursor` fixture (SAVEPOINT + ROLLBACK per test), test_resolve_customer.py with 4 tests covering Setton-style tiebreaker (seeds temp rows in a rolled-back txn).
  - `FOLLOWUPS.md`: (a) NULL-address backfill for recurring customers (Setton Farms address is currently NULL — tiebreaker can't fire); (b) full list of the ~25 remaining 4xx raise sites to normalize in a follow-up PR.
- **Why:** Two real-world failures: GPT was asking for customer disambiguation even when a PO address uniquely identified the right row; plain-string 4xx error responses forced the GPT into a generic "something went wrong" fallback instead of surfacing the structured suggestions to the operator.

---

## 2026-04-16 22:00 — Applied migration 034: SO-260414-003 force-closed to shipped
- **File(s) changed:** `migrations/034_force_close_so260414003_hannas.sql` (applied in Supabase SQL Editor), `FACTORY_LEDGER_CHANGELOG.md` (added row #24)
- **What changed:** Migration ran successfully — `sales_orders.status` for SO-260414-003 is now `shipped`; paper-trail note (Hannas Gourmet ship-to, 03/11/2026 customer pickup, invoice 28123-I dated 03/02/2026, PO 2026-0099-SW, lot JAN 20 2026 on 10 × Granola Vanilla Almond 25 LB + 1 pallet charge) appended to notes. Post-flight assertions passed.
- **Why:** Same as #033 — paper-only close for an order that shipped before being entered into the ledger.

---

## 2026-04-16 21:45 — Draft migration 034: force-close SO-260414-003 (Hannas Gourmet) to shipped
- **File(s) changed:** `migrations/034_force_close_so260414003_hannas.sql`
- **What changed:** New migration (same pattern as 033) flips `sales_orders.status` from `confirmed` → `shipped` for SO-260414-003 and appends paper-trail note: Hannas Gourmet ship-to (1330-14 Lincoln Ave, Holbrook NY 11741), 03/11/2026 customer pickup, invoice 28123-I dated 03/02/2026, PO 2026-0099-SW, lot JAN 20 2026 on 10 units of Granola Vanilla Almond 25 LB, + 1 pallet charge. Pre- and post-flight asserts included.
- **Why:** Order physically shipped in March but entered retroactively on 04/14/2026 without underlying ledger transactions. User chose paper-only close over transaction backfill — same trade-offs as migration 033 (dashboard `Shipped` stays 0, no on-hand decrement, no shipment rows). Migration not yet applied — awaiting user run.

---

## 2026-04-16 21:25 — Applied migration 033: SO-260326-002 force-closed to shipped
- **File(s) changed:** `migrations/033_force_close_so260326002_ace_endico.sql` (applied in Supabase SQL Editor), `FACTORY_LEDGER_CHANGELOG.md` (added row #23)
- **What changed:** Migration ran successfully — `sales_orders.status` for SO-260326-002 is now `shipped` and paper-trail note (Ace Endico ship-to, 04/13/2026 customer pickup, invoice 28159-I, PO 624249, lot breakdown) is appended to `sales_orders.notes`. Post-flight assertions passed.
- **Why:** Order physically shipped but never entered as ledger transactions. User elected paper-only close over a transaction backfill. Bypassed the main.py:5375 manual-'shipped' guard via direct SQL; dashboard `Shipped` remains 0 lb for this order by design.

---

## 2026-04-16 21:05 — Draft migration 033: force-close SO-260326-002 (Ace Endico) to shipped
- **File(s) changed:** `migrations/033_force_close_so260326002_ace_endico.sql`
- **What changed:** New migration flips `sales_orders.status` from `confirmed` → `shipped` for SO-260326-002 and appends a paper-trail note (Ace Endico ship-to address, 04/13/2026 customer pickup, invoice 28159-I, PO 624249, lot/production-date breakdown for 140 cases Fancy + 20 cases Medium). Includes pre- and post-flight assertion blocks.
- **Why:** Order was physically shipped but never entered as shipment/transactions in the ledger. User chose to bypass the `/status` endpoint's manual-'shipped' guard (main.py:5375) rather than backfill real shipment records. Dashboard `Shipped` column will remain 0 lb for this order because no transaction_lines or sales_order_shipments rows are written. Migration not yet applied — awaiting user to run in Supabase SQL Editor.

---

## 2026-04-16 13:22 — Fix stale dashboard config SKU name for Vanilla Crisp #16
- **File(s) changed:** `dashboard/dashboard_config.json`
- **What changed:** Removed one space in `batch_skus` entry at line 104: `"Batch Vanilla Crisp Granola #16 (no almonds)"` → `"Batch Vanilla Crisp Granola #16(no almonds)"`, matching the canonical product name in the DB (products.id=112, odoo_code=90024).
- **Why:** Dashboard Operations tab flagged this SKU under "Missing SKUs". Root cause: `/dashboard/api/inventory/batches` does `LOWER(p.name) = ANY(unnest(%s::text[]))` exact match (main.py:6709), and the single extra space in the config prevented a match. Product exists and is active — config was stale.

---

## 2026-04-16 20:10 — Product catalog cleanup (migration 032 applied)
- **File(s) changed:** `migrations/032_backfill_skus_and_merge_bs_cocoa.sql`, `FACTORY_LEDGER_CHANGELOG.md`
- **What changed:** Migration 032 applied via Supabase SQL Editor (MyFirstProject). (1) Backfilled odoo_code='70089' on `Classic Granola 25 LB` (id=171). (2) Merged SKU-less duplicates `BS Cocoa Chips` (id=167) and `BS Cocoa Liquor` (id=168) into canonical 15008 `BS Cocoa Liquor – Chips` (id=61): repointed 2 rows in `lots`, 3 in `transaction_lines`, 2 in `inventory_adjustments`, updated 2 rows in `product_verification_history` to action='merged' + merged_into_product_id=61, deleted the two duplicate product rows. (3) Idempotently asserted label_type='private_label' on odoo_code 70082 (already correct; no-op UPDATE). Products count 203 → 201. All 8 post-flight SELECT assertions PASS (verified via /admin/sql). Changelog row #22 added to FACTORY_LEDGER_CHANGELOG.md.
- **Why:** Product catalog had 3 SKU-less rows (noticed during SKU printout). Two of the three (BS Cocoa Chips, BS Cocoa Liquor) were semantic duplicates of 15008; the third (Classic Granola 25 LB) just needed a code assigned from the next 700xx slot (70089). 70082 guard is defensive — it's already correct but a stale-backup restore could silently flip it to 'house'.

---

## 2026-04-14 17:30 — Fix GPT refusing to call API endpoints
- **File(s) changed:** `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md`, `openapi-gpt-v3.yaml`
- **What changed:** Restored missing SEARCH FIRST rule; added NEVER INSTRUCT rule (prevents GPT from telling users to run GET requests); added lot-code and supplier-lot routing rules; fixed 4 phantom endpoints in instructions that referenced API paths not in the OpenAPI schema (/lots/by-supplier-lot, /customers/search, /production/day-summary, /inventory/{product}); swapped getCurrentInventory and checkFulfillment for searchCustomers and getDaySummary in schema to match instructions; bumped instructions to v3.6.0 and schema to v3.4.0.
- **Why:** GPT was refusing to call API endpoints for lot lookups and inventory queries, instead telling users to manually run GET requests and paste results. Root cause: missing "SEARCH FIRST" directive dropped in v3.5.0, no lot-code routing rule, and phantom endpoints creating confusion about available Actions.

---

## 2026-04-14 14:00 — Fix /receive rejecting missing supplier_lot_code
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-gpt-v3.yaml`
- **What changed:** Replaced hard 400 rejection on missing supplier_lot_code with fallback logic: defaults to lot_code if provided, then 'N/A'. Added description to supplier_lot_code field in both OpenAPI schemas documenting the fallback.
- **Why:** GPT couldn't complete receives — schema said supplier_lot_code was optional but the API hard-rejected when it was omitted.

---

## 2026-03-26 — Trim GPT instructions to fit 8000-char limit
- **File(s) changed:** `GPT_INSTRUCTIONS.md`
- **What changed:** Compressed ROUTING RULES, PRE-FLIGHT INTENT, QUERIES, and LOT MERGES sections to fit under 8,000 chars (now 7,986). Updated QUERIES to list /inventory/lookup as primary. Removed /inventory/{item} and packing-slip line from QUERIES (covered elsewhere).
- **Why:** Adding ROUTING RULES pushed instructions to 8,279 chars, exceeding the 8,000-char GPT limit.

---

## 2026-03-26 — Add ROUTING RULES section to GPT instructions
- **File(s) changed:** `GPT_INSTRUCTIONS.md`
- **What changed:** Added ROUTING RULES section before PRE-FLIGHT — INTENT. Bare product names route directly to inventoryLookup; product + "orders" routes to listOrders; product + "trace"/"lot" routes to trace endpoints; default fallback is inventoryLookup.
- **Why:** GPT was asking unnecessary clarification questions instead of immediately calling the right endpoint.

---

## 2026-03-26 — Performance fix for /inventory/lookup bulk queries
- **File(s) changed:** `main.py`
- **What changed:** Added `_inventory_detail_for_products()` that fetches product info and lot inventory for multiple product_ids in 2 SQL queries (using `ANY(%s)`) instead of 2N. Refactored `/inventory/lookup` to call the bulk function. Changed default limit from 10 to 5 to reduce payload size for broad queries.
- **Why:** Queries like "coconut" matching 17 products caused 34+ database round trips, making the endpoint slow.

---

## 2026-03-26 — Remove temporary debug endpoint and logging from inventory lookup
- **File(s) changed:** `main.py`
- **What changed:** Removed `/inventory/debug/{product_id}` endpoint and all diagnostic logging from `_inventory_detail_for_product()`. The 0-lb bug was caused by uncommitted code not being deployed, not a query issue.
- **Why:** Cleanup after confirming the inventory lookup works correctly once deployed.

---

## 2026-03-26 — Add debug logging and /inventory/debug/{product_id} endpoint for inventory lookup bug
- **File(s) changed:** `main.py`
- **What changed:** Added detailed debug logging to `_inventory_detail_for_product()` (logs all lots before HAVING, transaction_lines by product_id, and product_id mismatches). Added temporary `/inventory/debug/{product_id}` endpoint that returns raw diagnostic data (lots, txn_lines via lot join vs product_id, grouped results without HAVING, inventory_summary view).
- **Why:** `/inventory/lookup` returns 0 lb on hand for product 10305 (Sprinkles Rainbow 25 LB) despite 3,125 lb existing in lot 26-03-20-INVE-001. Dashboard shows correct balance. Debugging whether HAVING clause, lot_id join, or product_id mismatch is the root cause.

---

## 2026-03-26 — Remove getInventoryItem from GPT schema to stay at 30-op limit
- **File(s) changed:** `openapi-gpt-v3.yaml`
- **What changed:** Removed /inventory/{item_name} (getInventoryItem) from GPT schema since /inventory/lookup fully replaces it for GPT use. Brings operation count from 31 back to 30.
- **Why:** ChatGPT GPT actions enforce a 30-operation maximum

---

## 2026-03-26 — Add /inventory/lookup unified endpoint + fuzzy fallback on getInventoryItem
- **File(s) changed:** `main.py`, `openapi-gpt-v3.yaml`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`
- **What changed:** New GET /inventory/lookup?q= endpoint that combines fuzzy product search with lot-level inventory detail in a single call. Added _inventory_detail_for_product helper. Updated /inventory/{item_name} with fuzzy fallback (tiered search if no LIKE match; returns 404/300 for zero/multiple matches). Updated all three OpenAPI schemas with the new endpoint and revised summaries.
- **Why:** Replace the two-step searchProducts → getInventoryItem flow with a single unified lookup for the GPT

---

## 2026-03-26 — Create openapi-gpt-v3.yaml (unified schema for ChatGPT GPT action)
- **File(s) changed:** `openapi-gpt-v3.yaml`
- **What changed:** Created trimmed GPT action schema from openapi-v3.yaml (30 operations, the ChatGPT limit). Removed productsMissingCaseSize, searchCustomers, productionDaySummary. Added back getBatchFormula (needed for /make lot overrides). Uses unified mode-in-body endpoints instead of deprecated split /preview /commit paths.
- **Why:** The GPT is currently using the deprecated openapi-schema-gpt.yaml with split paths. This new schema uses the correct unified endpoints and fits within the 30-operation ChatGPT limit.

---

## 2026-03-26 — Fix GPT shipping 404: add split-path route aliases
- **File(s) changed:** `main.py`, `openapi-schema-gpt.yaml`
- **What changed:** Added thin wrapper routes for /preview and /commit sub-paths on all transactional endpoints (/receive, /ship, /make, /pack, /adjust, /sales/orders/{id}/ship). The GPT schema uses these split paths but the API only had the combined endpoint with mode in the body, causing 404s. Also added the missing shipOrderPreview endpoint to the GPT schema.
- **Why:** GPT was getting "not found" when trying to ship SO-260325-003 because it called /sales/orders/SO-260325-003/ship/commit which didn't exist as a route.

---

## 2026-03-26 09:00 — Update GPT instructions to v3.5.0
- **File(s) changed:** `GPT_INSTRUCTIONS.md`
- **What changed:** Complete rewrite of GPT instructions from developer-assistant format to operator-facing GPT v3.5.0. New structure includes: critical rules, pre-flight intent/product resolution, universal disambiguation format, batched disambiguation, order entry from confirmations, transaction workflow, order editing, shipping, FIFO override, receive, supplier lot cross-reference, found inventory, make, pack (with smart resolve and source batch mismatch), adjust, sales orders, ingredient lots, pack add-ins, post-commit, qty display, day summary, FG identity, lot merges, packing slip link-only, queries, bilingual, and error codes.
- **Why:** User requested update to new v3.5.0 instruction set

---

## 2026-03-25 16:00 — Add product_id disambiguation to all three trace endpoints
- **File(s) changed:** `main.py`, `openapi-schema-gpt.yaml`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added optional `product_id` query param to `/trace/supplier-lot` in main.py (batch and ingredient already had it). Added `product_id` param to all three trace endpoints in both OpenAPI schemas. Added `/trace/supplier-lot` endpoint to both OpenAPI schemas (was missing). Updated GPT instructions trace line to include supplier-lot and note the `?product_id=` param. Removed `updateNote` operation from GPT schema to stay within 30-operation limit.
- **Why:** GPT could detect ambiguous lot codes and ask the user to pick, but had no way to pass the chosen product_id back to any trace endpoint — the OpenAPI schemas didn't expose the parameter.

---

## 2026-03-25 15:30 — Allow service items (Pallets, freight) on sales orders with zero quantity_lb
- **File(s) changed:** `main.py`, `migrations/031_relax_quantity_lb_check_for_service_items.sql`
- **What changed:** Added is_service detection to create_sales_order and add_order_lines endpoints. Service items now skip case weight lookup, unit-defaulting warning, and quantity sanity check. quantity_lb defaults to 0 for service items. Migration relaxes CHECK constraint from > 0 to >= 0.
- **Why:** Service items like Pallets have no physical weight, causing the quantity_lb > 0 constraint to reject them.

---

## 2026-03-25 12:00 — Unit Display Implementation: dual lb · units format everywhere
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `dashboard/dashboard.js`, `dashboard/traceability.html`, `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md`
- **What changed:** Added unit counts (derived from case_size_lb / default_batch_lb) to all operator-facing endpoints and dashboard views. Backend: added case_size_lb/default_batch_lb to /products/search, /bom/products; new /products/missing-case-size endpoint; unit totals on sales order list/detail/ship/update-line; unit_count on production calendar FG rows, batch_count on batch rows; unit_count per line on shipments/receipts/inventory/lots/trace endpoints; packing slip qty_display changed from "N cs" to "X lb · Y units". Frontend: new fmtQty() helper; dual format in production calendar, sales orders (list/detail/KPI/lines), shipping/receiving activity, FG lot rows, batch lot rows, lot detail panel, traceability nodes. GPT instructions: added QTY DISPLAY section. OpenAPI schemas updated.
- **Why:** Operators need to see both weight (lb) and unit count simultaneously for all product quantities to reduce manual conversion and improve operational visibility.

---

## 2026-03-25 00:15 — Accept order_number strings in all /sales/orders/{order_id} endpoints
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `resolve_order_id` dependency that accepts either integer DB id or order_number string (e.g. 'SO-260323-001') and resolves to the integer id. Applied to all 8 sales order endpoints. Updated both OpenAPI schemas to declare order_id as `type: string`. Added note to GPT instructions ORDER EDITING section. Compressed GPT instructions to stay under 8,000 char limit (ORDER ENTRY, SUPPLIER LOT, PACKING SLIP sections).
- **Why:** GPT only knows the order_number (not the DB integer id), so FastAPI rejected order number strings with validation errors. GPT fell back to showing curl commands instead of executing.

---

## 2026-03-24 23:45 — Fix GPT "show don't tell" bug for ship date edits
- **File(s) changed:** `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `format: date` and explicit description with date conversion examples to `requested_ship_date` in both OpenAPI schemas. Strengthened ORDER EDITING section in GPT instructions to say "CALL API IMMEDIATELY" and include date format conversion rules. Added "NEVER show curl, API docs, or payloads" and "NEVER suggest cancel/recreate for edits" directives.
- **Why:** GPT was responding with curl commands and API documentation instead of calling updateOrderHeader when asked to change a ship date. Root cause: the v3 schema had no description/format on requested_ship_date, and GPT instructions mapped the operation but didn't explicitly direct execution.

---

## 2026-03-24 22:30 — GAP-3: Standalone /ship writes shipments + shipment_lines
- **File(s) changed:** `main.py`, `migrations/030_backfill_standalone_shipment_records.sql`
- **What changed:**
  - Standalone `/ship` commit path now creates a `shipments` row (with `transaction_id`, `shipped_at`, `customer_id`) and `shipment_lines` rows (one per lot shipped, with positive quantity) after writing transaction_lines. No sales_order_id or sales_order_line_id (left NULL).
  - Response now includes `shipment_id` field.
  - New migration 030: backfills `shipments` + `shipment_lines` for all existing standalone ship transactions that were missing them. Wrapped in a transaction, uses ABS() on negative transaction_line quantities.
  - Integrity checker (ship_missing_shipment_lines) unchanged — date filter `>= 2026-02-27` already correct; backfill covers all existing orphans.
- **Why:** Standalone shipments were invisible to anything querying shipment tables (packing slips, integrity checker, reports). Unifies shipping model so both standalone and order-linked paths write to shipments/shipment_lines.

---

## 2026-03-24 21:00 — Rename UNKNOWN lot to 25216 + add lot rename endpoint
- **File(s) changed:** `migrations/029_rename_unknown_lot_to_25216.sql`, `main.py`
- **What changed:**
  - New migration 029: renames lot 324 (Chocolate Sprinkles 25 LB, product_id 203) from lot_code='UNKNOWN' to '25216'. Includes pre-flight/post-flight checks, conflict guard, and idempotency (skips if already renamed).
  - New `PATCH /lots/{lot_id}/rename` endpoint in main.py: accepts `{"new_lot_code": "..."}`, validates no duplicate for same product, updates `lots.lot_code`, returns old/new codes. Schema audit confirmed only `lots` table stores lot_code as text — all other tables use integer FKs.
  - New `LotRenameRequest` Pydantic model.
- **Why:** Lot 324 was created with lot_code='UNKNOWN' during receive; real code from packing slip is 25216. Supplier lot was already backfilled but internal lot_code was still 'UNKNOWN'. Rename endpoint prevents future cases from needing raw SQL.

---

## 2026-03-24 19:30 — Option C trace: full upstream+downstream from any trace endpoint, no dead ends
- **File(s) changed:** `main.py`
- **What changed:**
  - `/trace/ingredient` no longer hard-rejects output lots (pack_output, production_output). Instead returns lot_origin, origin_note, upstream_ingredients (via ingredient_lot_consumption), plus direct_shipments/on_hand as normal
  - `/trace/batch` now includes customer_shipments, total_shipped_lb, and on_hand_lb — queries ship transactions that deducted from the batch lot
  - supplier_lot_code was already present on batch trace ingredient rows (no change needed)
- **Why:** Lot 601141 (Graham Cracker Crumbs NTF 10 LB) has entry_source=pack_output but ships directly to customers. The Fix 4 guard rail blocked it from /trace/ingredient, making shipment data invisible. Now both endpoints give the full picture regardless of which one is called.

---

## 2026-03-24 17:00 — Extend trace endpoints for direct-ship and supplier-lot exposure (FDA recall compliance)
- **File(s) changed:** `main.py`
- **What changed:**
  - Added `INGREDIENT_ENTRY_SOURCES` and `OUTPUT_ENTRY_SOURCES` constants; updated `/trace/batch` and `/trace/ingredient` to use them (fixes stale enum values like 'receive'/'make'/'pack')
  - Added `direct_shipments`, `total_shipped_lb`, and `on_hand_lb` to both `/trace/ingredient` and `_trace_ingredient_backward()` — queries ship transactions that deducted from ingredient lots
  - Added new `GET /trace/supplier-lot/{supplier_lot_code}` endpoint — recall-ready: finds all internal lots for a supplier lot (via `lots.supplier_lot_code` and `lot_supplier_codes` table), returns production usage, customer shipments, and exposure summary
- **Why:** Ingredient/resale lots shipped directly to customers (sprinkles, graham crumbs) were invisible to trace endpoints because trace only followed `ingredient_lot_consumption`. FDA recall compliance requires answering "where did supplier lot X end up?" for all products.

---

## 2026-03-24 15:45 — Backfill supplier lots for SO-260318-005 (American Classic Specialties)
- **File(s) changed:** API patches only (no file changes)
- **What changed:** Linked supplier lot codes for 3 lots from the American Classic Specialties shipment:
  - Rainbow Sprinkles lot `26-03-20-INVE-001` → supplier lot `550075853` (was missing)
  - Chocolate Sprinkles lot `UNKNOWN` (product_id 203) → supplier lot `25216` (was set to "UNKNOWN")
  - Graham Crumbs lot `601141` → supplier lot `601141` (was null)
- **Why:** Physical packing slips had supplier lot numbers handwritten but system records were missing the cross-references. Note: Chocolate Sprinkles lot code remains "UNKNOWN" — no API endpoint exists to rename lot codes; a manual DB UPDATE would be needed to rename it.

---

## 2026-03-24 14:30 — Factory Ledger Traceability Audit Report
- **File(s) changed:** `TRACEABILITY_AUDIT_2026-03-24.md`
- **What changed:** Created comprehensive traceability audit report identifying 13 gaps in the trace, ship, and lot lifecycle flows. Key findings: direct-ship-from-ingredient is invisible to trace endpoints (CRITICAL), no forward trace from lot to customer (CRITICAL), standalone /ship doesn't create shipment records (HIGH). Includes priority matrix and concrete fix proposals.
- **Why:** SO-260318-005 (Rainbow Sprinkles to American Classic Specialties) exposed broken backward traceability — lot 26-03-20-INVE-001 shows as never used despite being shipped, and supplier lot 550075853 was never linked.

---

## 2026-03-24 10:00 — Fix order entry behavior — act don't explain, minimize exchanges
- **File(s) changed:** `gpt-instructions-v3.md`
- **What changed:** Added ORDER ENTRY FROM CONFIRMATIONS section with silent extract→resolve→create flow; strengthened BE CONCISE (4 lines max for order confirmations, no unprompted next steps) and ACT DON'T LOOP (no payload display, no step headers, max 1 emoji); tightened DISAMBIGUATION for order entry (one tight question only); compressed SUPPLIER LOT, PACK, and QUERIES sections to stay under 8K char limit. Version bumped to v3.4.0.
- **Why:** GPT was acting as consultant during order entry — dumping SOPs, over-confirming, showing payloads, taking 6-7 exchanges for what should be 1-2.

---

## 2026-03-23 17:15 — Extend lot collision disambiguation to main dashboard
- **File(s) changed:** `main.py`, `dashboard/dashboard.js`
- **What changed:** Added `product_id` to `json_build_object` in shipments and receipts API responses so lot links in the Activity tab have product context. Added `product_id` to search API lots query. In dashboard.js: added `data-product-id` to shipping, receiving, and product panel lot links; added `data-search-lot-product-id` to search result lot items; pass `product_id` through all `openLotPanel` calls. Rewrote `openLotPanel` to handle HTTP 409 (ambiguous lot code) by rendering a disambiguation picker instead of a raw error. Added `renderLotDisambiguation()` helper.
- **Why:** Main dashboard lot links (shipping, receiving, search, product panel) didn't pass `product_id` to the lot detail endpoint, causing raw 409 errors after the collision fix was deployed.

---

## 2026-03-23 16:30 — Fix lot code collision bug + trace type misclassification
- **File(s) changed:** `main.py`, `dashboard/traceability.html`
- **What changed:** Added optional `product_id` query parameter to 5 endpoints (`/lots/by-code`, `/lots/{lot_code}/supplier-lot`, `/trace/batch`, `/trace/ingredient`, `/dashboard/api/lot`) for disambiguation when lot codes collide across products. When `product_id` is omitted and multiple lots match, endpoints return HTTP 409 with a list of matches. Added type validation to `/trace/ingredient` — returns 400 `wrong_trace_type` if called with a finished-goods lot. Fixed `/trace/batch` production query to join on `lot_id` (integer) instead of `lot_code` (string). Fixed `/trace/ingredient` downstream query to join on `lot_id`. Frontend: fixed `buildLotIndex` dedup to key on `lot_code|product_id`, added `selectedProductId` tracking, pass `product_id` in all API calls, added disambiguation modal for 409 responses, auto-route forward trace to batch endpoint on `wrong_trace_type`, escaped lot codes in onclick handlers (XSS fix), persist `product_id` in URL state.
- **Why:** Lot codes like "MAR 12 2026" are shared across products (ingredient + batch). Endpoints using `WHERE LOWER(lot_code) = LOWER(...)` with `fetchone()` returned whichever row Postgres found first, causing wrong lot data, wrong traces, and wrong supplier-lot updates.

---

## 2026-03-23 14:00 — Fix backward trace for ingredient lots
- **File(s) changed:** `main.py`, `dashboard/traceability.html`
- **What changed:** Updated `/trace/batch/{lot_code}` to detect ingredient lots (entry_source = receive/adjusted/found) and return supplier origin + downstream batches instead of 404. Added `_trace_ingredient_backward()` helper. Updated traceability.html to render ingredient backward traces with supplier → ingredient → batches → customers flow.
- **Why:** Backward trace failed with "Batch not found" for ingredient lots because the endpoint only looked for make/pack transactions.

---

## 2026-03-23 12:30 — Dashboard actionability audit report
- **File(s) changed:** `DASHBOARD_ACTIONABILITY_AUDIT.md`
- **What changed:** Created comprehensive audit report covering: complete endpoint inventory (85+ endpoints), dashboard page inventory (4 tabs, 4 standalone pages), gap analysis mapping write endpoints to dashboard locations, auth/CORS security assessment, and prioritized implementation plan with top 5 action sketches (ship order, update status, adjust inventory, receive, edit order header)
- **Why:** Planning phase for making the Netlify dashboard actionable (write operations) instead of read-only

---

## 2026-03-23 — "Ready to Ship" display label and reverse transition
- **File(s) changed:** `dashboard/dashboard.js`, `dashboard/index.html`, `main.py`, `gpt-instructions-v3.md`, `GUIDE.md`, `CONTEXT.md`
- **What changed:** Renamed dashboard display label from "Ready" to "Ready to Ship" in status label mapping and filter dropdown. Added `ready → in_production` reverse transition in both VALID_TRANSITIONS and MANUAL_TRANSITIONS. Updated GPT instructions to suggest marking orders as "Ready to Ship" after production completes, and documented the reverse transition. Updated GUIDE.md and CONTEXT.md to reflect the new display name and reverse transition.
- **Why:** Improve clarity of the "ready" status (now displayed as "Ready to Ship") and allow orders to move back to in_production if production falls short or inventory is consumed elsewhere.

---

## 2026-03-19 — Fix /make commit crash: is_ingredient column does not exist
- **File(s) changed:** `main.py`
- **What changed:** Replaced `COALESCE(is_ingredient, false) = false` with `type != 'ingredient'` in the post-commit auto-prompt /pack query (line ~2622). The `is_ingredient` column never existed in the products table; the correct column is `type` with value `'ingredient'`.
- **Why:** /make commit failed with "column is_ingredient does not exist" when trying to query finished goods for pack prompting after batch creation.

---

## 2026-03-19 15:20 — Add regression guard changelog and update CLAUDE.md
- **File(s) changed:** `FACTORY_LEDGER_CHANGELOG.md`, `CLAUDE.md`
- **What changed:** Created FACTORY_LEDGER_CHANGELOG.md documenting all major fixes with "Breaks If Reverted" column, known root causes, and 10 permanent rules. Added Regression Guard section to CLAUDE.md so future sessions check the changelog before making changes.
- **Why:** Prevent regressions by ensuring every Claude Code session is aware of past fixes and their dependencies.

---

## 2026-03-19 — Fix pallet charges counted as weight in sales order totals
- **File(s) changed:** `main.py`, `migrations/028_add_is_service_to_products.sql`
- **What changed:** Added `is_service` boolean column to products table (migration 028) to flag service/charge items like pallets, freight, surcharges. Updated 4 locations in main.py to exclude service products from weight totals: order detail Python loop (lines ~4441-4442), order list SQL SUM (lines ~4171-4172), dashboard overdue query (line ~5488), dashboard due-this-week query (line ~5507). Uses PostgreSQL FILTER clause in SQL queries and DB flag + keyword fallback in Python.
- **Why:** Pallet Charge lines (e.g., 1 unit stored as 1 lb) were inflating order weight totals. SO-260305-003 showed 251 lb instead of 250 lb due to a pallet charge line.

---

## 2026-03-19 — Swap GPT schema to openapi-v3.yaml; add sliced almonds products; update GPT instructions
- **File(s) changed:** `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `GPT_INSTRUCTIONS.md`, `migrations/027_add_sliced_almonds_products.sql`
- **What changed:** Added "ACTIVE GPT SCHEMA" header comment to openapi-v3.yaml confirming it's the canonical spec (30/30 ops). Added large deprecation warning box to openapi-schema-gpt.yaml. Updated GPT_INSTRUCTIONS.md to reference openapi-v3.yaml instead of deprecated schema. Created migration 027 to add two missing sliced almonds products: "Almonds – Sliced" (general ingredient) and "BS Almonds – Sliced – Raw" (Blue Stripes ingredient), both with idempotent NOT EXISTS guards.
- **Why:** GPT was configured with deprecated openapi-schema-gpt.yaml causing 404s on all receive operations (split /preview /commit endpoints no longer exist). Arturo couldn't receive "Almonds Slice" because no sliced almonds product existed in the DB.

---

## 2026-03-19 — Trim GPT instructions to fit 8,000 char limit
- **File(s) changed:** `gpt-instructions-v3.md`
- **What changed:** Condensed SOURCE BATCH MISMATCH WARNING and QUERIES product lookup lines to save ~300 characters. Final count: 7,869 chars (131 under limit). No behavioral rules removed.
- **Why:** GPT instructions were 8,167 chars after adding pack_needed/batch_hint rules, exceeding OpenAI's 8,000 char limit.

---

## 2026-03-19 — Update GPT instructions for pack_needed and batch_hint fields
- **File(s) changed:** `gpt-instructions-v3.md`, `GPT_INSTRUCTIONS.md`
- **What changed:** Added behavioral rule in MAKE section: when /make commit returns `pack_needed`, GPT must surface FG SKUs, ask operator to pack, and execute /pack calls. Added batch_hint note in PACKING SLIP section: when INSUFFICIENT lines have a `batch_hint`, GPT explains unpacked batch inventory exists and offers to run /pack. Added corresponding sections to developer-facing GPT_INSTRUCTIONS.md.
- **Why:** Ensures the GPT actually acts on the new `pack_needed` and `batch_hint` fields added to the API, closing the loop on forgotten /pack steps.

---

## 2026-03-19 — Add batch-inventory hints for INSUFFICIENT packing slips and /pack prompt after /make
- **File(s) changed:** `main.py`
- **What changed:** Fix 1: When packing slip shows INSUFFICIENT for a FG product, cross-references `parent_batch_product_id` to check if unpacked batch inventory exists and displays a hint (e.g., "Note: 500 lb of Batch BS Dark Chocolate is available — run /pack to convert to finished goods") both in the JSON data and rendered on the PDF. Fix 2: After `/make` commit, the response now includes a `pack_needed` object listing all FG SKUs linked via `parent_batch_product_id` that can be packed from the batch, prompting the operator to run `/pack`.
- **Why:** Recurring issue where production happens (/make) but the /pack step is forgotten, leaving batch inventory idle while packing slips show INSUFFICIENT for finished goods. FOUND lots were being used as a workaround.

---

## 2026-03-19 — Add automatic add-in ingredient deduction to /pack
- **File(s) changed:** `main.py`, `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md`, `openapi-schema-gpt.yaml`, `openapi-v3.yaml`
- **What changed:** Replaced `check_pack_source_mismatch()` with `resolve_pack_add_ins()` that detects when packing from a base batch into an FG with an intermediate batch BOM containing add-in ingredients. Preview now shows add-in quantities needed with FIFO availability. Commit automatically deducts add-in ingredients via FIFO with row-level locking. Falls back to mismatch warning when intermediate BOM is missing or source batch not in BOM. Updated GPT instructions and OpenAPI schemas.
- **Why:** Floor process blends add-in ingredients (PB Chips, Banana Bites) at the packing hopper — there is no separate /make step for flavor variants. System now matches the actual workflow.

---

## 2026-03-19 — Add pack source batch mismatch warning safeguard
- **File(s) changed:** `main.py`, `gpt-instructions-v3.md`, `GPT_INSTRUCTIONS.md`, `openapi-schema-gpt.yaml`, `openapi-v3.yaml`
- **What changed:** Added `parent_batch_product_id` column to products table (inline migration 012) linking FG products to their expected source batch. Added `check_pack_source_mismatch()` helper that returns warning fields when the /pack source batch doesn't match the target FG's expected parent batch. Warning (English + Spanish) included in both preview and commit responses. Populated mappings for all 8 BS Granola FG→batch pairs. Updated GPT instructions to display mismatch warnings prominently and suggest running /make first.
- **Why:** Arturo packed PB Banana FG cases directly from Dark Chocolate Granola base batch, skipping the required intermediate /make step that deducts add-in ingredients (PB Chips, Banana Bites).

---

## 2026-03-18 — Expose listProducts endpoint to GPT; drop getProduct
- **File(s) changed:** `openapi-v3.yaml`, `main.py`, `gpt-instructions-v3.md`
- **What changed:** Replaced `getProduct` (GET /products/{product_id}) with `listProducts` (GET /bom/products) in OpenAPI schema to stay at 30-operation ChatGPT cap. Bumped /bom/products default limit from 50→200 and max from 200→500. Updated GPT instructions to use listProducts for catalog queries and searchProducts for single lookups; removed /products/{id} reference.
- **Why:** GPT "list finished goods" queries only returned 7 SKUs because no catalog endpoint was exposed — it fell back to getCurrentInventory which only returns products with positive on-hand stock.

---

## 2026-03-18 — Add 8oz BS panel to dashboard; fix case weight display rounding
- **File(s) changed:** `dashboard/dashboard_config.json`, `dashboard/dashboard.js`
- **What changed:** Added new "6x8 OZ Retail Cases (BS Line)" panel with SKUs 70085-70088 to dashboard config. Fixed case weight display: `fmtInt()` was truncating 2.63 to 2 for 7oz BS products — added `fmtWt()` formatter that preserves decimals for non-integer weights while showing whole numbers cleanly (e.g., "25" not "25.0").
- **Why:** New 8oz BS products were missing from dashboard; 7oz products showed "× 2 lb" instead of "× 2.63 lb"

---

## 2026-03-18 — Add Blue Stripes 8 OZ Granola product SKUs
- **File(s) changed:** `migrations/026_add_bs_8oz_granola_products.sql`
- **What changed:** Created migration to insert 4 new BS Granola finished goods (Hazelnut Butter, Almond Butter, Dark Chocolate, Peanut Butter Banana — all 6x8 OZ Case, 3.0 lb, private_label). odoo_codes 70085–70088 (70081–70084 were already taken by Setton products). Product IDs: 206–209. Type is `finished` (not `finished_good` — check constraint).
- **Why:** New SKUs needed in the products table

---

## 2026-03-18 — Fix day summary to show pack consumption from older batch lots; void test transaction 469
- **File(s) changed:** `main.py`
- **What changed:** Fixed `/production/day-summary` endpoint to include pack consumption (and adjustments) from batch lots produced on previous days. Previously, only lots with a `make` transaction on the same day were included in `batch_lots`, so pack consumption from older lots was silently dropped. Now, when a lot_id is not already in the dict, it looks up the lot's product info and adds it with `produced_lb: 0.0`. Also voided test transaction 469 (1-case test pack of CQ Granola 10 LB, TXN-8EB734).
- **Why:** Production team reported that the day summary was missing pack consumption from batch lots made on prior days (e.g., Batch Classic Granola #9 made across MAR 10-11, with MAR 10 lot consumption missing from MAR 11 summary).

---

## 2026-03-18 — Add PATCH /lots/{lot_code}/supplier-lot endpoint and GPT integration
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added new `PATCH /lots/{lot_code}/supplier-lot` endpoint to attach or update supplier lot cross-references on existing lots after receiving. Updated OpenAPI schema (v3.3.0) with the new endpoint. Added "SUPPLIER LOT CROSS-REFERENCE" section to GPT instructions so the GPT auto-records mismatches when Arturo reports packing slip lot differs from system lot. Added new endpoint to QUERIES reference.
- **Why:** Arturo frequently encounters supplier lot numbers on packing slips (e.g., 550078168 for sprinkles) that don't match system lot codes. Previously there was no way to record this post-receive — only at receive time. This endpoint closes that gap.

---

## 2026-03-18 — Add supplier lot cross-reference for lot 26-03-10-FOUN-001
- **File(s) changed:** `migrations/025_set_supplier_lot_sprinkles_26-03-10-FOUN-001.sql`
- **What changed:** Created migration to set supplier_lot_code = '550078168' on lot 26-03-10-FOUN-001 (Sprinkles Rainbow 10 LB). This lot was used to ship 23 cases to International Gourmet Foods (SO-260318-001, Shipment #32).
- **Why:** Packing slip shows supplier lot 550078168; adding cross-reference for traceability.

---

## 2026-03-16 17:20 — Deploy to Netlify and connect GitHub auto-deploy
- **File(s) changed:** (no code changes — deployment config only)
- **What changed:** Linked Netlify site (cns-factory-ledger) to GitHub repo sevenwells72/factory-ledger for auto-deploy from main branch with publish dir `dashboard/`. Deployed latest dashboard.js with Railway API URL fix to production.
- **Why:** Site was using Netlify Drop (manual uploads) and wasn't picking up git pushes. Now auto-deploys on every push to main.

---

## 2026-03-16 17:15 — Fix dashboard API calls for Netlify hosting
- **File(s) changed:** `dashboard/dashboard.js`
- **What changed:** Changed `API_BASE` from relative `/dashboard/api` to absolute `https://fastapi-production-b73a.up.railway.app/dashboard/api`. Also updated the `/audit/integrity` health badge fetch to use the full Railway URL. Relative paths were resolving to Netlify (returning 404 HTML) instead of the Railway backend.
- **Why:** Dashboard deployed on Netlify was showing "Failed to load" errors with 404 HTML responses for all API calls.

---

## 2026-03-16 17:00 — Add top navigation bar across all dashboard pages
- **File(s) changed:** `dashboard/index.html`, `dashboard/sankey.html`, `dashboard/process-flow.html`, `dashboard/traceability.html`, `dashboard/dashboard.css`
- **What changed:** Added a fixed 48px top navigation bar to all four dashboard HTML files. "CNS Factory Ledger" branding on the left, pill-style nav links (Dashboard, Material Flow, Production Lines, Traceability) on the right with active page highlighted in blue. Collapses to hamburger menu below 768px. Adjusted sticky header and tab-bar positions in all files to account for the new 48px offset. Removed redundant "back to Dashboard" links from sub-pages.
- **Why:** Users needed a way to navigate between all dashboard pages without returning to the main dashboard first.

---

## 2026-03-16 16:30 — Add lot traceability network graph page
- **File(s) changed:** `dashboard/traceability.html`
- **What changed:** Built a new audit-critical lot traceability page with D3.js network graph. Supports forward trace (ingredient -> batches -> customers) and backward trace (batch -> ingredients -> suppliers). Features include: search with autocomplete from lot index, completeness badges (complete/partial/incomplete), confirmed vs legacy/unknown link distinction (solid vs dashed lines), fan-out collapse for large traces (>8 children), zoom/pan controls, detail audit table, print and text export for auditors, URL deep-linking (?lot=X&direction=Y), and proper data gap handling with warning icons.
- **Why:** Required for food safety compliance (Setton Farms audits, HACCP, FDA recall readiness). Traces ingredient lots through production batches to customer shipments with honest data completeness indicators.

---

## 2026-03-16 15:00 — Increase transaction history limit and Sankey fetch size
- **File(s) changed:** `main.py`, `dashboard/sankey.html`
- **What changed:** Raised `/transactions/history` max limit from 100 to 1000 (line 2992 in main.py). Updated Sankey diagram fetch constant from 100 to 500 for both make and ship transaction fetches.
- **Why:** 100-record cap truncated data in Sankey diagram; 500 captures fuller production/shipment history

---

## 2026-03-16 14:30 — Add manufacturing process flow dashboard
- **File(s) changed:** `dashboard/process-flow.html`
- **What changed:** New single-file HTML dashboard showing 4 production lines (Granola Baking, Coconut Sweetened, Bulk Packing, Pouch Line) as cards with visual stage pipelines. Features: product classification by output name, today's production metrics (input lb, batch count, output lb, cases), yield calculation with coconut hydration handling, summary strip (lines active, produced today, avg yield, workers placeholder), auto-refresh every 60s with stale-data detection, dark industrial theme, fallback sample data when API is offline.
- **Why:** Need a real-time production floor view showing each line's status and throughput derived from make/pack transactions

---

## 2026-03-16 14:00 — Add Sankey diagram page for product flow visualization
- **File(s) changed:** `dashboard/sankey.html`
- **What changed:** New single-file HTML page with D3.js Sankey diagram showing raw materials → production lines → finished goods → customers. Features: date range selector, top-N controls for ingredients/products/customers, dark industrial theme, API integration with fallback sample data, hover tooltips, 50 lb minimum flow threshold.
- **Why:** Visualize end-to-end product flow through the factory operation

---

## 2026-03-12 — Close SO-260213-001 unrecorded shipment (Juliette Food LLC)
- **File(s) changed:** `migrations/024_close_so260213001_juliette.sql`
- **What changed:** Marked all 5 order lines (4 granola products + pallets) as fulfilled and set order to shipped. The physical shipment (BOL 28106-I, 02/26/2026, customer pick up) was never recorded as ship transactions in the system.
- **Why:** Order was stuck in confirmed status with 0 shipped despite physical shipment being complete per packing slip

---

## 2026-03-12 — Reconcile SO-260312-005 with pre-existing shipments (DiCarlo Food Service)
- **File(s) changed:** `migrations/023_reconcile_so260312005_dicarlo.sql`
- **What changed:** Linked 4 standalone ship transactions (Graham Cracker Crumbs 400 lb, Fancy UNIPRO 400 lb, Sprinkles Chocolate 600 lb, Sprinkles Rainbow 1,800 lb) to sales order SO-260312-005. Created shipments, sales_order_shipments, and shipment_lines records; updated line shipped quantities and statuses to fulfilled; set order status to shipped.
- **Why:** Shipments were recorded before the sales order was entered, so they weren't linked

---

## 2026-03-12 — Close SO-260217-001 partial ship (Feeser's Food Distributors)
- **File(s) changed:** `migrations/022_close_so260217001.sql`
- **What changed:** Closed order SO-260217-001 by reducing Flake UNIPRO 10 LB ordered qty from 300 → 200 lb to match what was shipped, marked line as fulfilled, updated order status to shipped. Remaining 100 lb will not be shipped per business decision.
- **Why:** Order was stuck in partial_ship status; business decided to close it as-is

---

## 2026-03-12 — Fix SO-260217-008 under-shipment (Curtze Food Service)
- **File(s) changed:** `migrations/021_fix_so260217008_undershipment.sql`
- **What changed:** Created correction migration to fix order SO-260217-008 which shows "Partial Ship" but was fully shipped per packing slip (invoice 28108-I, 02/24/2026). Medium UNIPRO 10 LB was recorded as 1,080 lb shipped instead of 1,200 lb (120 lb short due to insufficient on-hand at ship time). Pallets line (qty 1) was not recorded at all. Migration updates sales_order_lines, sales_order_shipments, shipment_lines, and order status.
- **Why:** Physical shipment (per BOL) was complete but system under-recorded due to inventory cap at ship time

---

## 2026-03-09 13:43 — Add /production/day-summary to schema, trim GPT instructions
- **File(s) changed:** `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `/production/day-summary` GET endpoint to OpenAPI schema (was in main.py but missing from schema). Removed `/lots/{lot_id}` (getLot) to stay within GPT's 30-operation limit — `getLotByCode` covers all real usage. Bumped schema version to 3.2.0. Trimmed GPT instructions from ~9,400 chars to ~5,600 chars by removing redundant field-by-field listings for RECEIVE, MAKE, PACK, and ADJUST sections (the schema already provides these).
- **Why:** GPT was slow due to ~500-800 wasted tokens per turn from duplicated field listings. Missing day-summary endpoint caused GPT to hallucinate when users said "wrap up" or "daily summary". Hit 30-operation GPT limit after adding day-summary.

---

## 2026-03-09 18:45 — Add since/until date filters to /transactions/history
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md`
- **What changed:** Added `since` and `until` (YYYY-MM-DD, both inclusive) query parameters to the `/transactions/history` endpoint. Filters on `t.timestamp`. Updated both OpenAPI specs and GPT instructions to document the new parameters.
- **Why:** GPT was hanging when users asked for transactions since a specific date — the endpoint had no date filtering, causing either no results or too much data.

---

## 2026-03-09 11:19 — Fix reconciliation SQL script: p.sku → p.odoo_code
- **File(s) changed:** `factory_ledger_reconciliation.sql` (also updated in Supabase SQL Editor)
- **What changed:** Replaced `p.sku` with `p.odoo_code` on lines 116 (SELECT) and 123 (GROUP BY) in Test 5 (Negative Balance). The `products` table has no `sku` column; `odoo_code` is the equivalent. All other column references across all 6 tests and the summary block were verified correct against the production schema via `information_schema.columns` inspection and EXPLAIN dry-run.
- **Why:** Script failed on Supabase production with "column so.ship_date does not exist" (original report) / "column p.sku does not exist" (actual remaining error after inspection).

---

## 2026-03-09 11:05 — Patch three critical bugs: transaction safety, zero-shipment guard, force_standalone audit trail
- **File(s) changed:** `main.py`
- **What changed:**
  - Fix 1: Added `except HTTPException: raise` to 4 write endpoints missing it (`/products/quick-create`, `/products/quick-create-batch`, `/inventory/found-with-new-product`, `/customers` POST). All other commit endpoints already had correct outer try/except pattern.
  - Fix 2: Added zero-shipment guard in `POST /sales/orders/{id}/ship` commit — if all line items hit the `no_stock` path (total shipped = 0), the pre-created shipments row is deleted and a 409 is returned instead of committing an empty `partial_ship` status update.
  - Fix 3: In `POST /ship` commit, when `force_standalone=true` is used and customer has open orders: logs `logger.warning` with customer name and order count, records `standalone_override=true` in transaction notes, returns `warning` and `standalone_override` fields in response JSON.
- **Why:** Code audit found these three patterns causing inventory/order drift in production.

---

## 2026-03-05 15:00 — Pre-merge review fixes (Decimal serialization, void safety, migration 015)
- **File(s) changed:** `main.py`, `migrations/015_fix_lot131_negative_balance.sql`
- **What changed:** Added DecimalSafeJSONResponse as default FastAPI response class for NUMERIC column compatibility; added FOR UPDATE lock + COALESCE(status) on /void endpoint; made migration 015 note text dynamic using actual balance
- **Why:** Pre-merge review identified psycopg2 Decimal serialization risk from migration 020, potential double-void race condition, and hardcoded balance in migration 015

---

## 2026-03-05 14:30 — Factory Ledger Data Integrity Remediation (Priorities 0–7)
- **File(s) changed:** `main.py`, `dashboard/index.html`, `dashboard/dashboard.css`, `dashboard/dashboard.js`, `migrations/015_fix_lot131_negative_balance.sql`, `migrations/016_backfill_received_at.sql`, `migrations/017_transaction_status_and_void_ghosts.sql`, `migrations/018_backfill_pre_migration_shipments.sql`, `migrations/019_populate_case_size_lb.sql`, `migrations/020_numeric_precision.sql`
- **What changed:**
  - P0: Added `validate_lot_deduction()` guard with BALANCE_EPSILON=0.0001 to /pack, /ship, /make, /sales/orders/{id}/ship
  - P1: Migration 015 posts adjustment to fix lot 131 -60 lb negative balance
  - P2: Migration 016 backfills received_at; updated all FIFO ORDER BY to use COALESCE(received_at, created_at)
  - P3: Migration 017 adds status column, voids ghost makes 80/83/84/177; added POST /void/{id} endpoint; 0-lb make guardrail; dashboard queries filter voided transactions
  - P3B: Migration 018 backfills shipments/shipment_lines for 11 pre-migration ship transactions
  - P4: Migration 019 auto-populates case_size_lb from product names; improved /pack error messages
  - P5: Migration 020 converts quantity columns to NUMERIC(14,4); added Decimal-based calculations
  - P6: Added supplier_lot_code required validation on /receive
  - P7: New GET /audit/integrity endpoint with 8 automated checks + dashboard health badge
- **Why:** Data integrity audit found negative lot balances, missing metadata, ghost transactions, floating point dust, and other issues requiring systematic remediation.

---

## 2026-03-05 11:30 — Fix SO-260217-001 Flake UNIPRO over-shipment (300→200 lb)
- **File(s) changed:** `migrations/014_fix_so260217001_flake_overshipment.sql`, `main.py`
- **What changed:** Created migration 014 to correct the over-shipment: zeroes out the 100 lb over-deduction on lot "FEB 24 2026" (lot 242) in transaction_lines, updates shipment_lines/sales_order_shipments from 300→200 lb, reduces sales_order_line 48 shipped qty from 300→200 and status from fulfilled→partial, and sets order 32 status to partial_ship. Also added `AND tl.quantity_lb != 0` filter to the packing slip query to skip zeroed-out correction rows.
- **Why:** System recorded 300 lb shipped for Coconut Sweetened Flake UNIPRO 10 LB on SO-260217-001 but only 200 lb physically left the warehouse. The extra 100 lb was a data entry error at ship time.

---

## 2026-03-05 09:50 — Fix packing slip to use actual shipment records instead of live FIFO
- **File(s) changed:** `main.py`
- **What changed:** Patched the packing slip endpoint (`/sales/orders/{id}/packing-slip`) to check for committed shipment records in the `shipments` table. If shipments exist, lot allocations are pulled from `shipment_lines → transaction_lines → lots` (reflecting what was actually shipped). If no shipments exist yet (pre-shipment), the original FIFO inventory preview is preserved for pick-list use.
- **Why:** After a shipment was committed, the packing slip was recalculating FIFO against current (post-shipment) inventory, causing it to show less than what was actually shipped (e.g., showing 200 lb instead of 300 lb for Coconut Sweetened Flake UNIPRO 10 LB on SO-260217-001).

---

## 2026-03-04 12:55 — Clarify lot_allocations schema description: source overrides only
- **File(s) changed:** `openapi-v3.yaml`
- **What changed:** Updated lot_allocations description in PackRequest to explicitly state it's for SOURCE lot overrides only, and to use target_lot_code for the output/FG lot.
- **Why:** GPT was misrouting operator-provided lot codes into lot_allocations (source) instead of target_lot_code (output).

---

## 2026-03-04 12:50 — Fix GPT pack lot code misrouting: lot code = output lot, not source
- **File(s) changed:** `gpt-instructions-v3.md`
- **What changed:** Added explicit instruction that when operator provides a lot code during packing, it is the target_lot_code (FG output lot), NOT a source lot allocation. Source lots use FIFO unless operator explicitly says "pull from lot X."
- **Why:** After schema fix, GPT successfully called /pack but misinterpreted lot code 602271 as a source lot allocation instead of the output lot, causing a "0 lb available" error.

---

## 2026-03-04 12:30 — Fix GPT pack workflow: schema descriptions + instruction conflict
- **File(s) changed:** `openapi-v3.yaml`, `gpt-instructions-v3.md`, `openapi-schema-gpt.yaml`
- **What changed:** Added field descriptions (source_product, target_product, case_weight_lb, lot_allocations, target_lot_code) to PackRequest in v3 OpenAPI schema so GPT knows it can pass raw SKU codes. Fixed instruction "Always ask which FG SKU" → only ask if FG SKU not already provided. Marked old openapi-schema-gpt.yaml (v2.7.0) as deprecated — its split /preview and /commit endpoints don't exist in the API.
- **Why:** GPT was stuck in a verify/resolve loop during pack operations. Root cause: old schema's /pack/preview endpoint returns 404 (only /pack with mode param exists), and missing field descriptions left GPT uncertain what values to pass.

---

## 2026-03-02 17:35 — Auto-populate case_size_lb for OZ-pattern products
- **File(s) changed:** `main.py`
- **What changed:** Extended Migration 006 startup logic to handle OZ-pattern product names (e.g., "12x10 OZ", "6x7 OZ"). Parses count × unit-oz from the name, converts to lb (÷16), and sets `case_size_lb`. Also ran UPDATE directly against production DB for 9 existing products: 5 × 12x10 OZ → 7.50 lb, 4 × 6x7 OZ → 2.63 lb.
- **Why:** Products with OZ-based case formats had `case_size_lb = NULL`, which blocked the `/pack` endpoint with HTTP 400. SKU 70003 (Granola SS Chocolate Chip 12x10 OZ Case) was specifically reported as failing during packing.

---

## 2026-03-02 17:15 — Migration 013: Create shipment tables
- **File(s) changed:** `migrations/013_shipment_tables.sql`
- **What changed:** Created migration for three missing tables: `shipments` (one record per ship/commit), `sales_order_shipments` (links transactions to order lines), and `shipment_lines` (per-product detail within a shipment). All columns derived from existing INSERT/SELECT statements in main.py. Includes foreign keys to sales_orders, customers, sales_order_lines, transactions, and products, plus indexes on all FK and timestamp columns.
- **Why:** The ship/commit endpoint (`POST /sales/orders/{id}/ship`) and sales dashboard were referencing these tables but no CREATE TABLE migration existed, causing "relation does not exist" errors on any dispatch operation.

---

## 2026-02-27 09:48 — Add /products/resolve to v3 OpenAPI schema
- **File(s) changed:** `openapi-v3.yaml`
- **What changed:** Added `/products/resolve` endpoint and updated `/products/search` description in the v3 schema (the one the GPT uses for actions). Bumped version 3.0.0 → 3.1.0.
- **Why:** The GPT needs to see the new resolve endpoint in its action schema to use it.

---

## 2026-02-27 09:40 — 3-tier fuzzy product search + bulk resolve endpoint
- **File(s) changed:** `main.py`, `openapi-schema.yaml`, `openapi-schema-gpt.yaml`, `migrations/012_pg_trgm_product_search.sql`
- **What changed:**
  - Added `_tiered_product_search()` helper: exact match → keyword (word-order independent via ILIKE ALL) → trigram similarity fallback
  - Upgraded `GET /products/search` to use 3-tier search; returns `match_tier` per result
  - Added `POST /products/resolve` endpoint for bulk product name resolution with confidence levels (high/medium/low/none)
  - Rewrote `resolve_product_id()` and `resolve_product_full()` to use 3-tier search instead of simple ILIKE
  - Added migration 012: enables `pg_trgm` extension and GIN index on `products.name`
  - Updated both OpenAPI schemas (v2.6.0 → v2.7.0) with new `/products/resolve` endpoint and improved `/products/search` docs
- **Why:** OCR'd order confirmations send product names in different word orders (e.g. "Chocolate Sprinkles 10 lb" vs DB name "Sprinkles Chocolate 10 LB"), causing sales order creation to fail. This makes product matching word-order and typo tolerant.

---

## 2026-02-26 — Add Sprinkles 25 LB finished goods (Rainbow & Chocolate)
- **File(s) changed:** `main.py`, `dashboard/dashboard_config.json`
- **What changed:**
  - Added Migration 012 in main.py: inserts two new finished goods — Sprinkles Rainbow 25 LB (Odoo 10305) and Sprinkles Chocolate 25 LB (Odoo 10306)
  - Added both SKUs to the "25 LB Bulk Cases" panel in dashboard_config.json
- **Why:** 25 LB versions of existing 10 LB sprinkles products (10302 Rainbow, 10303 Chocolate) were needed.

---

## 2026-02-26 15:10 — Supplier lot field: complete read integration
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Added Migration 011: `supplier_lot_code`, `lot_type`, `received_at` columns on `lots` table; `lot_supplier_codes` table with indexes for commingled receipt breakdowns
  - GET `/lots/by-code/{code}`: added `supplier_lot_code`, `lot_type`, `supplier_lot_entries` to response
  - GET `/lots/{id}`: added `supplier_lot_entries` array (columns already came through via `l.*`)
  - NEW endpoint GET `/lots/by-supplier-lot/{code}`: searches both `lots.supplier_lot_code` and `lot_supplier_codes` table for recall tracing
  - GET `/trace/batch/{lot}`: ingredient lots now include `supplier_lot_code`
  - GET `/trace/ingredient/{lot}`: response now includes `supplier_lot_code`
  - Packing slip PDF: FIFO lots now include supplier lot reference in small gray text below internal lot code
  - OpenAPI: added `getLotsBySupplierLot` operation (now 32 operations)
  - GPT instructions: added supplier lot search to QUERIES section
- **Why:** Supplier lot data was write-only (captured during receive but never surfaced in any query). Needed for recall tracing and full chain traceability.

---

## 2026-02-26 14:40 — Packing slip: stronger GPT instructions to output link only
- **File(s) changed:** `gpt-instructions-v3.md`, `openapi-v3.yaml`
- **What changed:**
  - Rewrote PACKING SLIP section with explicit prohibitions: NEVER generate inline slip, NEVER list items, NEVER say "Sent to dock printer"
  - Specified exact markdown link format: `📎 [Packing Slip – {order_number}](URL)`
  - Strengthened OpenAPI description to say "do NOT call as API action" and "never recreate in chat text"
- **Why:** GPT was hallucinating an inline text packing slip instead of providing the clickable PDF link

---

## 2026-02-26 14:25 — Packing slip: query param auth for browser-clickable links
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Added `verify_api_key_flexible` dependency that accepts API key from either `X-API-Key` header or `?key=` query parameter
  - Updated packing slip endpoint to use flexible auth so browser link clicks work
  - Updated GPT instructions to present clickable URL instead of calling API directly
  - Updated OpenAPI spec with `key` query parameter on getPackingSlip
- **Why:** ChatGPT can't display binary PDF inline — secretary needs a clickable link to open in browser and print

---

## 2026-02-26 14:09 — Added packing slip PDF endpoint
- **File(s) changed:** `main.py`, `requirements.txt`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Added `GET /sales/orders/{order_id}/packing-slip` endpoint that generates a printable PDF packing slip
  - PDF includes: company header, BILL TO / SHIP TO, ship date, SO#, items table with FIFO lot allocation (read-only), warehouse confirmation signature lines, traceability note, footer with timestamp
  - Non-weight items (pallets, freight) shown with "N/A" lot; insufficient stock flagged as "INSUFFICIENT"
  - Added `reportlab==4.1.0` to requirements.txt
  - Added `StreamingResponse` import and `io` import to main.py
  - Added `getPackingSlip` operation to openapi-v3.yaml (now 31 operations)
  - Added PACKING SLIP section to gpt-instructions-v3.md
- **Why:** Office secretary needs to request printable packing slips through the GPT for warehouse use

---

## 2026-02-26 — Upgrade to v3.0.0: Merged endpoints, LAT Code Policy v1.1, shipment records
- **File(s) changed:** `main.py`, `openapi-v3.yaml`, `gpt-instructions-v3.md`
- **What changed:**
  - Merged 6 preview/commit endpoint pairs into single endpoints with `mode` parameter: `/receive`, `/ship`, `/make`, `/pack`, `/adjust`, `/sales/orders/{id}/ship`
  - Added `Literal` import and `mode` field to all 6 request models (ReceiveRequest, ShipRequest, MakeRequest, PackRequest, AdjustRequest, ShipOrderRequest)
  - Added LAT Code Policy v1.1 fields to ReceiveRequest: `supplier_lot_code`, `lot_type`, `supplier_lot_entries`
  - Integrated `lot_supplier_codes` table for commingled receipt breakdowns in `/receive` commit path
  - Updated `/receive` commit to write `received_at`, `supplier_lot_code`, `lot_type` to lots table
  - Integrated `shipments` and `shipment_lines` tables in `/sales/orders/{id}/ship` commit path — auto-creates shipment record
  - Updated `check_open_orders_for_ship` URL reference from `/ship/commit` to `/ship`
  - Version bumped from 2.5.0 to 3.0.0
  - Generated OpenAPI YAML with exactly 30 operations (excludes dashboard, admin, system, scheduler endpoints)
  - Generated updated GPT instructions for v3.0.0
- **Why:** Upgrade to v3.0.0 — merging preview/commit stays under ChatGPT's 30-operation OpenAPI limit; LAT Code Policy v1.1 compliance; shipment tracking for sales orders; commingled receipt support

---

## 2026-02-25 14:42 — Moved global change log to iCloud Drive
- **File(s) changed:** `~/change-log.md` (now symlink)
- **What changed:** Moved `~/change-log.md` to `~/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md` (iCloud Drive) and created a symlink at `~/change-log.md` pointing to the new location
- **Why:** Enable remote sync of the global change log via iCloud without syncing all files

---

## 2026-02-25 14:36 — Added CLAUDE.md with change log protocol instructions
- **File(s) changed:** `CLAUDE.md`, `~/.claude/CLAUDE.md`
- **What changed:** Created project-level and global CLAUDE.md files containing instructions for maintaining dual change logs (project-level CHANGE_LOG.md and global ~/change-log.md) on every file change
- **Why:** User requested standardized change log protocol across all projects

---

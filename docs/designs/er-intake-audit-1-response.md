# ER intake — response to audit #1 (2026-09-08)

Fixes for all 12 findings of `er-intake-audit-1.md`, on branch `feat/er-intake`,
one commit per finding, each with regression tests. Head: `2ec5567`.
Suite after fixes: **466 passed** + 1 known pre-existing failure
(`tests/test_recent_ledger.py` badge test — also fails on clean main; stale
`dashboard.css?v=27` assertion, unrelated to this branch).

| # | Finding | Fix | Commit | Regression tests |
|---|---------|-----|--------|------------------|
| 1 (P1) | Fuzzy products treated as confirmed | Review state separates `suggested` from `chosen` in the new pure-logic module `dashboard/er-intake-logic.js`; only alias/exact matches arrive chosen, fuzzy renders as a pre-highlighted candidate. lb computation and Approve both require an explicit human pick; typing expected lb never activates a product-less line. | `38f4440` | `tests/test_er_intake_logic.js` (node:test, run inside pytest via `tests/test_er_intake_logic_js.py`): fuzzy-is-suggestion, qty/lb edits never compute for unconfirmed lines, direct-lb entry not approvable, explicit pick confirms |
| 2 (P1) | Alias conversion ignores current unit | Alias `lb_per_unit` applies only when the line's normalized unit equals the alias's unit (`_normalize_unit`, None == None counts); product association still reuses. Mismatch falls through to line-unit rules or stays null for manual lb. The audited 50-lb/BAG-alias-on-100-LB case now yields 100 lb. | `e65f873` | `TestMatchEndpoint::test_alias_unit_mismatch_falls_back_to_line_unit`, `…_container_requires_manual_lb`, `…_match_is_normalized`, `…null_unit_applies_to_unitless_line` |
| 3 (P1) | Corrections retain stale pounds / teach wrong aliases | Unit change (normalized) invalidates `lb_per_unit` + expected lb; un-choosing a product drops product-derived conversions (alias/case-weight) and always drops computed pounds (text-derived and manual conversions survive for the next pick). Per-line **Save alias** checkbox: default on, auto-off when expected lb is overridden, explicit choice wins. Approve sends the conversion only when save_alias is on AND quantity × lb/unit equals the approved pounds — an overridden line can still teach the product mapping, never a wrong conversion. | `9544c93` | logic tests: unit-change invalidation, cosmetic-unit no-op, product-change stale-conversion drop, save_alias defaults/override/toggle, `approveLinePayload` consistency gate |
| 4 (P1) | Rerenders discard review; stale rematches submittable | Reference, expected date, supplier and all line edits live in intake state; every render reads state. Supplier re-match goes through `ERIntake.mergeRematch`: exclusions and explicit save_alias choices persist; a user-chosen product persists only where the new alias/exact result names the same product, otherwise the line returns to unconfirmed. Match requests carry the current (edited) reference and qty/unit, are sequence-numbered (stale responses discarded), and Approve is disabled while a match is in flight. The duplicate override is armed per `forceKey(supplier_id, normalized reference)` and disarms when either edits. | `6c6bb6e` | logic tests: mergeRematch exclusion/agreeing-pick/reset/save_alias cases, forceKey normalization + pair binding |
| 5 (P1) | Retry can reopen an approved document | Both status writes in `_run_extraction_and_store` are conditional on `status <> 'approved'` with a rowcount check; rowcount 0 → 409 `DOCUMENT_ALREADY_APPROVED`, on the success and failure paths alike. Pre-call 409 (no paid model call) kept. | `9b9a500` | `test_retry_on_approved_document_409`, `test_approval_during_retry_model_call_wins`, `test_approval_during_failed_retry_model_call_wins` (approval flipped from inside the mocked model call) |
| 6 (P1, partial by design) | Browser-shipped key grants document access + paid extraction | Interim mitigation: per-key sliding-window rate limits — extraction (upload + model runs) 30/hour, signed URLs 120/hour — returning 429 `RATE_LIMITED` with `Retry-After` and a warning log line; buckets independent per key. Full fix (per-user sessions) deliberately deferred to the key-rotation work — recorded in the design doc's new **Deferred** section. | `7536231` | `TestRateLimits`: 429 + log line, shared extraction bucket across upload/extract, separate signed-URL bucket, per-key isolation |
| 7 (P2) | Weight-token parser misreads pack descriptions | Rewritten: numeric boundaries (`.5 LB` → 0.5, `1.5.5 LB` → nothing), `N x M LB` → N×M lb per unit, and None whenever tokens are absent or disagree (`50 LB or 25 LB`) — ambiguity means manual entry, never a guess. | `0ceaab9` | 16-case parametrized parser test incl. both audited misreads, plus a match-endpoint pack-weight test |
| 8 (P2) | Event-loop blocking; 15 s budget; lost document id | `POST /expected-receipts/extract` is upload-only (validate → row → Storage → 201 with `document_id`); the dashboard immediately calls `POST /purchase-documents/{id}/extract` with a 90 s per-call timeout override (upload gets 60 s). Both handlers are plain `def` → FastAPI threadpool. A timed-out extraction keeps the id and offers Retry. | `dc817e0` | `test_upload_is_upload_only`, `test_upload_then_extract_happy_path`, `test_handlers_run_in_threadpool` (`iscoroutinefunction`), reworked failure/retry tests |
| 9 (P2) | File validation doesn't enforce promised limits | Bounded read (`limit+1` bytes → 413 without buffering the rest); stored mime derived from magic bytes (client Content-Type advisory; unknown magic → 415); PDFs parsed with `pypdf` (new pinned dep 5.1.0, pure-python) — > 20 pages → 413 `PDF_TOO_MANY_PAGES`, unparseable → 422 `PDF_UNREADABLE` — all before any row or Storage write. | `8fbb0a9` | spoofed-mime rejection, magic-overrides-claim, page-cap accept/reject (real reportlab-generated PDFs), unreadable PDF, no row/no upload on rejection |
| 10 (P2) | Concurrent approvals bypass duplicate warnings | Approve takes `pg_advisory_xact_lock(hashtextextended('er-intake-ref:<supplier_id>:<normalized ref>'))` inside the transaction before the duplicate check, so same-pair approvals of different documents serialize and the loser re-checks under the lock and gets the 409. No-op without a reference. | `0e4e052` | two-connection lock tests (same key blocks, other pairs don't, released on rollback; no lock without ref), call-site spy asserting the approve path takes it |
| 11 (P2) | Numeric/date validation weaker than "strict" contract | `extraction.py`: quantity finite and > 0; dates exact `YYYY-MM-DD` calendar dates or null (pydantic field_validators → `ExtractionError`). Approve endpoint rejects NaN/±inf/non-positive `expected_qty_lb`/`quantity`/`lb_per_unit` with 422 `INVALID_QUANTITY` (endpoint-level: FastAPI echoes non-finite input into its 422 JSON, which starlette can't serialize). `_create_expected_receipt_core` now requires `math.isfinite`, closing the pre-existing NaN gap on the manual `POST /expected-receipts` path too. | `b738700` | parametrized NaN/inf/zero/negative + invalid-date extractor tests; approve NaN/inf 422s; manual-endpoint NaN → `INVALID_QUANTITY` end-to-end |
| 12 (P2) | Failed uploads leave misleading rows; approve accepts any status | New `upload_failed` status (migration 049 edited in place — still applied only to the local test DB, never prod — via named-constraint replacement so a re-run upgrades an existing table). A Storage failure after the row INSERT marks the row `upload_failed` before re-raising (helper 502s and transport errors alike); extraction answers 409 `UPLOAD_FAILED` for such rows (nothing to download — re-upload); approval requires `status = 'extracted'` (409 `DOCUMENT_NOT_EXTRACTED` otherwise). | `37d2afb` | storage-failure → `upload_failed` (both error shapes), extract-refuses-upload_failed, approve 409 for uploaded/upload_failed/extraction_failed |

## Verification items (no changes — matched)

* **RLS/grants on `purchase_documents` / `supplier_product_aliases` vs
  `expected_receipts`:** identical posture. None of the three has RLS; grants
  come from the same Supabase default privileges that `expected_receipts`
  (migration 041) received — confirmed identical on the local prod-schema DB,
  and confirmed read-only against prod that `expected_receipts` carries the
  standard Supabase default grants (`anon`/`authenticated`/`service_role`),
  which 049's tables will inherit the same way when applied. That broad
  default grant is a pre-existing platform-wide posture, noted but unchanged
  per the "match expected_receipts" criterion.
* **Bucket privacy:** `purchase-documents` does not exist in prod yet
  (checked `storage.buckets` read-only), so nothing pre-existing to fix; the
  first-use auto-create path sends `"public": false`.

## Notes

* The audit report itself is preserved verbatim at
  `docs/designs/er-intake-audit-1.md` (commit `0237b82`).
* New runtime dependency for deploy: `pypdf==5.1.0` (requirements.txt).
* The audit's "Additional" observations not in the fix scope (readonly mock
  granularity, duplicated schema validation, redundant per-line document
  lookup) remain open as cleanups.

## Round 2 — response to the re-audit (2026-09-08)

The re-audit left findings **3, 4, 8, 11, 12** open (plus an optional
concurrency-test ask on 10). One commit per item, each with a regression test
on the auditor's exact reproduction. Suite after fixes: **482 passed** + the
same known pre-existing `tests/test_recent_ledger.py` failure. Cache-busts:
`dashboard.js?v=50`, `er-intake-logic.js?v=5`.

| # | Reopened as | Fix | Commit | Regression tests |
|---|-------------|-----|--------|------------------|
| 3 | Client-only consistency gate | The alias-consistency rule is now a SERVER invariant on `/extract/approve`: `save_alias` with a non-null `lb_per_unit` where \|quantity × lb_per_unit − expected_qty_lb\| > 0.01 rejects the whole request with 422 `ALIAS_CONVERSION_MISMATCH` before anything is written. `lb_per_unit=null` + `save_alias` still teaches the product-only mapping (the dashboard's audit-fix-3 behavior stays legal). | `8bed0ba` | Auditor's case: qty 4 × 50 lb/BAG vs expected 175 with save_alias=true → 422 + nothing created; identical line with save_alias=false → 201, zero alias rows; product-only alias still 201 |
| 4a | Failed supplier re-match leaves stale review approvable | The newly selected supplier is kept; `ERIntake.applyRematchFailure` resets every line to unconfirmed (`chosen`, `lb_per_unit`, expected lb cleared; exclusions and explicit save_alias choices survive) and `intake.matchStale` holds Approve disabled — even against hand re-picked products — until a re-match succeeds (new Retry-matching button). | `d3e9413` | logic tests: alias-confirmed old product with 200 lb is not approvable after `applyRematchFailure`; exclusion + explicit save_alias survival |
| 4b | Only Approve was disabled while matching | `lockLines()`/`unlockLines()` in the logic module; every line mutator refuses edits on a locked line, and the review renders every input, picker, and header control disabled while `intake.matching`. Success unlocks via fresh merged lines; failure unlocks via `applyRematchFailure`. | `1536f1c` | logic tests: all seven mutators are no-ops while matching=true (deep-equal state unchanged); unlock re-enables edits; a failed rematch never leaves lines locked |
| 8 | Upload retry after timeout mints sibling documents | `POST /expected-receipts/extract` dedupes on sha256 before inserting: an existing row in status uploaded/extracted/extraction_failed/upload_failed is returned with **200** + `already_seen=true` instead of a new row; an `upload_failed` row is healed in place (object re-uploaded to its original path, status back to `uploaded`) since the bytes are in hand again. `approved` documents still start a fresh row. Dashboard upload-timeout copy: "Upload timed out. Drop the same file again to resume." | `a81f9b8` | second upload of identical bytes → 200 with the FIRST document_id, no sibling row, no re-upload; upload_failed resume heals the row and extraction then succeeds; approved doc does not block a fresh 201 |
| 11 | `strptime` accepts unpadded dates | `extraction.py` dates must match `^\d{4}-\d{2}-\d{2}$` (re.fullmatch) BEFORE the calendar check. | `a4ca5c3` | parametrized rejects incl. the auditor's `2026-2-3`, plus `2026-02-3`, `2026-2-03`, leading space, trailing newline; valid dates still pass |
| 12 | Migration 049 comment claims re-runs are no-ops | Header corrected: re-runs drop/recreate and re-VALIDATE the status CHECK constraint (brief ACCESS EXCLUSIVE lock); only the resulting schema is idempotent. Comment-only, DDL unchanged, still applied to the local test DB only. | `f996dac` | existing `test_reapply_is_noop` (re-run converges, exit 0) unchanged |
| 10 (optional) | Lock tested only as a primitive | End-to-end race test: two complete approvals of different documents with the same (supplier, reference), each endpoint transaction on its own real connection, provably in flight together (pre-held advisory lock, pg_locks-verified two waiters), then released. | `d254780` | exactly one 201; the loser gets 409 `DUPLICATE_REFERENCE`; exactly one receipt and one approved document exist |

The re-audit report is preserved verbatim at
`docs/designs/er-intake-audit-2.md`.

## Round 3 — response to audit #3 (2026-09-08)

Audit #3 (verbatim at `docs/designs/er-intake-audit-3.md`) confirms 3, 4, 11,
12 and the retry-matching flow fixed, leaving only finding **8** open, in two
sub-items. Suite after fixes: **485 passed** + the same known pre-existing
`tests/test_recent_ledger.py` failure.

| # | Remaining gap | Fix | Commit | Regression tests |
|---|---------------|-----|--------|------------------|
| 8a | A landed object whose success response was lost leaves the row `upload_failed`; healing retried with `x-upsert: false`, hit "already exists", and stayed stuck | The heal path uploads with `x-upsert: true`, so an object that is already there reads as success — legitimate because the bytes-match is proven via `file_sha256`, now also asserted explicitly (409 `SHA_MISMATCH`, unreachable by construction). Fresh uploads keep `x-upsert: false` (uuid-suffixed paths never legitimately exist). | `69614be` | mock Storage rejects non-upsert uploads onto an existing path with "already exists": re-dropping the bytes heals the row to `uploaded`, 200, existing document_id, and extraction then succeeds; header unit test — the helper sends `x-upsert: true` only when asked |
| 8b | Dedupe SELECT and INSERT not serialized — concurrent identical uploads mint siblings | Both now run in ONE transaction that first takes `pg_advisory_xact_lock(hashtext(file_sha256))` and holds it through the INSERT; the loser waits, sees the winner's committed row, and resumes it. Row-first/Storage-second ordering unchanged. | `50f6bf7` | two connections upload identical bytes provably in flight together (pre-held hash lock, pg_locks-verified two waiters): exactly one row, one 201 + one 200, both responses carry the same document_id |

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

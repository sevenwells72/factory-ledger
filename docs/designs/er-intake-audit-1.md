# ER intake — independent audit #1 (received 2026-09-08, verbatim)

Recommendation: no merge. The branch has paths to unconfirmed products, incorrect pounds, lost review corrections, and duplicate approval.

1. P1 — Fuzzy products are treated as confirmed. dashboard/dashboard.js:4000, :4181. `chosen: l.product || null` includes fuzzy suggestions. Although the backend initially leaves their pounds null, changing quantity or lb/unit computes pounds because `chosen` is already populated. Entering expected pounds directly also enables approval without selecting a product.

2. P1 — Alias conversions ignore the current unit. main.py:5362. The alias lookup returns `unit`, but conversion never checks it. A saved 50-lb/BAG alias applied to a subsequent quantity of 100 LB produces 5,000 lb instead of 100 lb; reproduced with the actual matching function.

3. P1 — Product/unit corrections retain stale pounds and teach incorrect aliases. dashboard.js:4184, :4201, :4278. Changing units updates only the label. Changing products preserves the previous product's conversion and expected pounds. Overriding expected pounds leaves `lb_per_unit` unchanged, yet approval always saves that conversion as an alias.

4. P1 — Rerenders discard reviewed values; supplier rematching can submit stale matches. dashboard.js:4101, :3997, :4142. Reference/date edits exist only in DOM inputs. Any line edit, selection, inclusion change, or duplicate-response rerender restores extracted values. Supplier rematching replaces all line edits and resets excluded lines to included. Approval remains available while rematching.

5. P1 — Extraction retry can reopen an approved document. main.py:5540, :5455, :5465. Retry checks status before the slow model call. Approval can then commit, after which retry unconditionally changes status to `extracted` or `extraction_failed`. Another approval becomes possible.

6. P1 — The browser-distributed dashboard key now grants document access and paid extraction. main.py:2306, dashboard.js:2092, main.py:5635. Holders can enumerate document IDs, mint signed URLs, upload files, and invoke/retry paid extraction.

7. P2 — Weight-token parsing silently misinterprets pack descriptions. main.py:5277. `.5 LB bag` becomes 5 lb; `4 x 5 LB case` becomes 5 lb/unit.

8. P2 — Extraction blocks the event loop and exceeds the browser's timeout budget. main.py:5474, dashboard.js:3942, dashboard/fetch-timeout.js:16. Async handler calls synchronous DB/HTTP/model ops. Both extraction requests inherit the 15-second browser timeout. A timed-out initial request loses the document ID.

9. P2 — File validation does not enforce the promised limits. main.py:5482. Entire file read before the 15 MB check. MIME accepted from client. No PDF page-count check.

10. P2 — Concurrent approvals of different documents bypass duplicate warnings. main.py:5592. Approval locks the document, not the supplier/reference pair.

11. P2 — Numeric/date validation is weaker than the "strict" contract. extraction.py:33, main.py:2608, main.py:4912. Extractor accepts invalid date strings and non-finite quantities. `expected_qty <= 0` does not reject NaN. The manual endpoint already had the finite-number gap.

12. P2 — Failed uploads leave misleading, unrecoverable document rows. main.py:5506, :5524. Upload failure leaves status `uploaded` with no object. Approval accepts any status except `approved`.

Additional: migration 049:26 adds public-schema tables without explicit RLS/revokes — verify anon/authenticated access. Bucket private setting unverified. Tests miss: edit preservation, conversion invalidation, unit-changing aliases, malformed pack weights, concurrent approval/retry, storage adapter failures, PDF page validation. Readonly mock fails on first SQL statement rather than specifically on a write. Duplicated extraction-schema/quantity validation and a redundant document lookup per approved line.

Manual POST /expected-receipts parity: creation behavior preserved; API contract gains `source_document_id` in request/response.

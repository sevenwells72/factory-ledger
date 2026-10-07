# V3 validation

**Plain-English summary:** All **172 offline tests passed**, including the existing safety suite, live-count reconciliation and 21 new packaging/billing exclusion regressions. Preview, apply and verification enforce the October 6 owner decision. The existing 110-page blind packet and 850-row CSV are unchanged; their recorded hashes and prior render checks still match. All 210 products remain on the floor packet; only 178 enter the reset. No production reset, application API call, branch, commit or PR was made. No real personal actor key was read.

## Evidence

- `all-tests.txt`: {'test_apply_reset': 65, 'test_count_pdf': 6, 'test_fresh_start_v2': 38, 'test_fresh_start_v3': 63}. Tests use temporary synthetic snapshots, counts, keys and mocked HTTP only; live sockets and subprocess/database access are blocked in the test harness. Temporary fake files were removed, including keys, journals and generated reports/PDFs.
- `pdf-checks.json`: 110 pages, 210 catalog products, 850 unique blank rows, every row printed, no filled lot/quantity fields. All pages rendered; the overview and representative instruction, count, unidentified-stock and movement-log pages were visually checked for clipping and legibility.
- `raw/01-scope.json` and its log: read-only scope snapshot dated 2026-10-06T18:37:34.872652+00:00, with `transaction_read_only=on` and successful ROLLBACK. Initial sandbox DNS failure was resolved by the authorized network-enabled read; no write query was attempted.
- `readiness/`: actual issued blank-sheet preview and verification against the saved snapshot. Zero count observations, zero proposed adjustments, 178 uncounted/held reset products, and 32 NOT COUNTED packaging/billing entries in `packaging-information-only.md` / `.csv`. Preview exits 0 because it produced reports; verification exits 2 because no physical count has been supplied. These are not fake or assumed physical quantities.
- Syntax checks passed for all fresh-start Python files. Tracked application, migration, script and dashboard files are unchanged. An API-client scan finds the production client only in `apply_reset.py`; preview/verification/builders do not import it. Historical product-185 archive SQL/evidence remains separate and was not executed.

## New regression coverage

Sheet-window and 30-minute movement holds; before/after classification; all post-cutoff movements; late entries inside and between area sessions; back-dated entries; multi-area signed allocation and prevention of invented opposing allocations; counted-to-uncounted duplicate removal; reverse-direction omission recovery; unresolved/mismatched moves; duplicate tags and unrelated-move rejection; changed evidence invalidating answers; per-row time and end-time fallback; distinct per-lot apply timestamps; packaging quantities retained in their entered units in a separate information-only report; Estimates in reasons; unknown-lot seven-day follow-up; incomplete product coverage; absence only after certified all-location search; inactive/service and correction holds; Sunshine ownership evidence; old reviewed counts without age deadline; changed source/ledger rejection; named actor enforcement; uncertain-write reconciliation without retry; standalone read-only journal verification; refusal of corrected reset events.

Existing tests retain exact signatures/hashes, inactive restrictions, decimal rounding, personal-key permissions, actor identity, API identity validation, redirects/proxy restrictions, durable journal integrity, interrupted-write recovery, duplicate/voided/wrong-quantity/wrong-actor refusal, fresh balance checks, excluded-scope behavior for historical v2, and the original package/lot rules. The two superseded assertions about an age deadline and the old floor README were updated to the owner's new timing policy and the relocated first-live-use safety guide.

## October 6 packaging decision checks

The 21 added tests cover positive packaging ledger balances without reset or inferred zero; missing and explicit-zero counts; unknown packaging lots, duplicate tags and unresolved moves without reset holds; fixed billing IDs despite changed catalog flags; future packaging products; explicit uncatalogued-label information rows; refusal to disguise known food as packaging; continued holds for unidentified non-packaging stock; packaging-only observations never zeroing food; finished food cases retaining normal conversion; information reports and reset-only verification sign-off; mocked apply making no packaging requests; refusal of obsolete signed scope policies, packaging rows and disguised billing rows; current-catalog apply guards; standalone read-only journal guards; and the real saved catalog's 178 reset / 32 excluded split with all 850 blank floor rows retained. The two earlier packaging-adjustment tests now check information-only behavior instead.

Both preview and verification were regenerated offline using the unchanged October 6 saved snapshot. No new database connection or credential read was needed for this decision. Expected exit codes were 0 for preview and 2 for incomplete verification. The separate information report currently has no physical quantities because completed floor counts have not been supplied.

## Limits to the conclusion

This validates the preparation tools and saved evidence, not production deployment or a real reset. FL has no per-location balances and its API lacks an atomic inventory guard; owner-reviewed allocation and a brief quiet posting interval for approved products remain necessary. Unknown lots are never created. Packaging counts are raw informational observations, not reconciled totals or adjustment targets. No packaging balance is loaded or zeroed. Billing IDs 102/176 remain excluded even if their catalog type changes. New catalog products of type packaging are excluded dynamically; known food sold in cases remains in scope. Count and Sunshine ownership facts still need the owner's real records.


## 2026-10-07 10:32 — Review and standalone verification

Clean temporary detached worktree of 805d471: test_fresh_start_v3.py and test_apply_reset.py ran 161 tests, all passed; fresh_start_v3.py --write-count-template generated 850 blank rows with the duplicate column. Worktree removed; no stash used. Full checkout Python suite: 533 passed, 1023 skipped, 27 warnings, exactly 4 expected DATABASE_URL errors, and 22 subtests passed; test_count_pdf.py excluded as requested. Node suite: 67 passed, no failures/skips. Both DB environment variables unset; no DB/application API calls or migrations. All 23 requested additions present; no prohibited additions; all 28 delivered fresh-start files scanned including extracted PDF content with no secrets (one synthetic FAKE.invalid URL), and zero hardcoded user-home paths. Manifest verifies all 27 companion files. origin/main 521330c maximum row 158 verified; row 159 used. main.py/migrations unchanged. Whitespace check passes for text with CSV CRLF allowed; PDF bytes retained.

Commands used (Python 3.12, with DATABASE_URL and TEST_DATABASE_URL unset):

```sh
# In audits/fresh-start/ of a temporary worktree created from committed HEAD:
python -B -m unittest -q test_fresh_start_v3.py test_apply_reset.py
python -B fresh_start_v3.py --write-count-template <temporary-path-under-audits/fresh-start>
# Back in the original checkout:
python -m pytest . --ignore=work --ignore=audits/fresh-start/test_count_pdf.py --continue-on-collection-errors
node --test tests/*.js
```

The four expected errors are the receipt extract/approve and sales upload/approve read-only-tripwire setup tests in `tests/test_expected_receipt_extract.py` and `tests/test_sales_order_extract.py`; all report `DATABASE_URL env var required`. DB-dependent tests remain unvalidated because database access was prohibited.

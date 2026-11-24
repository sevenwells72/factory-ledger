# Owner-directed review fixes — 2026-11-20

Implementation response to Claude Code comment [7026524246](https://github.com/sevenwells72/factory-ledger/pull/232#issuecomment-7026524246). This is test evidence, not a new cross-review. No Claude Code invocation was made in this task.

## Changes and regression evidence

| Owner item | Change | Proving test |
|---|---|---|
| a | Forward the raw note with 098 on; the shared posting core composes the SO destination once. | `tests/test_no_stock_review_fixes.py::test_sales_order_note_is_exactly_once` asserts the entire stored ledger note in both marker states. |
| b | Carry the held reason into Found posting; allow that retired reason only for the existing hold, without bypassing source applicability. | `test_held_found_retired_reason_can_be_approved_with_marker`, reused `test_096_held_found_with_098`, and `test_held_reason_does_not_bypass_source_applicability`. |
| c | Refuse a pair field before 098 with HTTP 422 `PAIRED_MOVE_UNAVAILABLE`, including a clear bilingual explanation. | `test_pair_field_is_refused_when_marker_off`. |
| d | Delivery redirect carries identity but leaves cases and case size blank. | New carry test in `tests/test_pilot_entry_ui.js`; real-page `tests/visual/run-no-stock-by-hand.mjs` checks both blank fields. |
| e | Exercise refused addition sources on ticket and direct Adjust; rerun 096 removal/destination/SO/large-removal/retired-Found contracts with 098 active. | `test_ingredient_addition_source_refusals` (10 cases) and `test_096_*` in the regression module. Existing direct Found, receiving-supplier, rule-lock, migration and normal floor proofs remain in `tests/test_no_stock_by_hand.py`. |
| f | Count only effective posted count transactions toward pack usage. A void frees its allowance without deleting the audit row. | `test_voided_under_pack_count_releases_allowance`; cumulative strict receipt-pack / max-100-lb tests remain. |

Before fixes, the Python regression run failed the marker-on exact-note, retired-held-Found, marker-off pair and void-allowance cases; the new JavaScript carry test also failed. Logs: `/private/tmp/fl232-r3-before.log` and `/private/tmp/fl232-r3-js-before.log`. New coverage for already-correct source rules is additional coverage rather than a claim those rules were broken.

## Independent shutdown

Draft [#234](https://github.com/sevenwells72/factory-ledger/pull/234), branch `feat/shared-key-write-shutdown`, starts from current main `ad07d80`. Its runtime diff is the original auth commit and its tests have no 098 dependency. Merge it before #232. `SHARED_KEY_WRITES=allow` restores shared-key writes; reverting that PR also removes the named-owner extension. No scheduled job needs the shared API key; the two manual scripts are documented in `docs/named-actor-writes.md`.

#232 is rebased onto that branch with original `b18a6c4` removed from its own series. Migration 098 stays on #232. Changelog rows: main 313, shutdown 314, inventory fixes 315. Both titles keep `[skip netlify]` to suppress previews.

## Test method and results

All database work uses localhost PostgreSQL 17 or guarded throwaway databases. Fresh primary database for each complete Python run. `FL_TEST_STOCK_RULES=on` seeds 098 into both the primary and framework-created scratch databases. Old-policy contracts explicitly select a pre-098 context; generic workflow fixtures use valid current inputs. No blanket opt-out, skip or xfail was added. Current 098 proofs and reused 096 contracts remain active.

| Complete local run | Passed | Failed | Errors / skipped |
|---|---:|---:|---:|
| Current main `ad07d807` Python | 7689 | 7 | 0 / 0 |
| Standalone #234 Python | 7693 | 7 | 0 / 0 |
| #232 Python, 098 marker off | 7790 | 7 | 0 / 0 |
| #232 Python, 098 marker on | 7790 | 7 | 0 / 0 |
| Standalone #234 JavaScript, including real-browser tests | 744 | 0 | 0 / 0 |
| #232 JavaScript, including real-browser tests | 747 | 0 | 0 / 0 |

The Python failure-ID sets were compared programmatically and match current main exactly. The intentional auth Request route snapshot is corrected. The two expected-receipt historical tests now use their native cursor; the added savepoint teardown conflict is fixed. Focused marker-on regressions: **186 passed**. No new failure is classified as baseline.

Real localhost browser checks passed for Receive, paired Adjust (two ledger legs, net zero), existing-lot Found refusal, voided-receipt Receive again, 096 Adjust, lot-history and Office navigation. Inspected the four 390px receiving-flow screenshots: paired draft, blank carried quantity, Found refusal and Receive again preserving identity/BOL with blank quantity/photo. The fixture displays a red STAGING label but connects only to localhost throwaway databases. Existing lot-history coverage checks 270 authenticated HTTP requests and six phone/tablet/desktop light/dark layouts.

Commands: `scripts/setup_test_db.sh --fresh` before each complete Python run; `scripts/run_tests.sh -p no:cacheprovider -o addopts='' -q --disable-warnings --junitxml=...`, with `FL_TEST_STOCK_RULES=on` only for the marker-on run; `PLAYWRIGHT_BROWSERS_PATH=/private/tmp/fl232-browsers node --test --test-concurrency=4 tests/*.js` in each branch. Runs are sequential.

Final evidence: `/private/tmp/fl-auth-r3-final.{log,xml}`, `/private/tmp/fl232-r3-{on-verified,off-verified,main}.{log,xml}`, `/private/tmp/fl-auth-r3-js.log`, `/private/tmp/fl232-r3-js.log`, and `/private/tmp/fl-098-browser/`. Generated lot-history output is retained outside the worktrees in `/private/tmp/fl-auth-r3-browser-work/` and `/private/tmp/fl232-r3-browser-work/`. `git diff --check` passes.

The seven baseline IDs are:

- `test_order_create_review.py::test_legacy_stuck_partial_order_finishes_without_new_shipment[False]`
- `test_order_create_review.py::test_legacy_stuck_partial_order_finishes_without_new_shipment[True]`
- `test_sales_order_state_model.py::TestNoDeadlock::test_multi_product_ship_and_exit_do_not_deadlock`
- `test_ship_order_idempotency.py::test_before_migration_088_the_commit_runs_unguarded`
- `test_ship_order_service_line.py::test_commit_route_preserves_already_fulfilled_409s`
- `test_trusted_browsers.py::test_flag_on_without_migration_reports_mode_on_but_unavailable_and_blocks`
- `test_write_tickets.py::test_migration_up_down_up_and_marker_stability`

No merge, deployment, remote migration, credential setting change, or staging/production write was made. The down/up reactivation window continues to be anchored to the marker's applied_at; down retains the audit rows, as documented in its comment.

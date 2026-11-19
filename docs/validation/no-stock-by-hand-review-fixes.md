# PR 232 — review fixes and local verification (2026-11-19)

Branch: `feat/no-stock-by-hand`. Draft only. No merge, deploy, staging/production query or hosted inventory write was performed. The owner-approved cumulative under-one-pack cap is retained unchanged.

## Five review findings

1. **Reason catalog compatibility:** migration 098 and its down file never edit or add columns to `correction_reasons`. `exceptions_enforcement._valid_reasons` supplies addition flags/labels in code; `resolve_reason` enforces that policy only with marker 098. The down file removes the marker and preserves lifetime count/move audit tables.
2. **Direct Found attribution and identity:** `main.add_found_inventory` passes its authenticated Request to `_found_commit_core`. Direct new-product Found uses the same core after creating the product, in the same transaction. Named ledger/trace attribution and `lot_confirmation.record_identity` now cover direct posts, including the unidentified-lot clock and one follow-up. Ticket identity recording stays once per ticket.
3. **Direct Found reason validation:** new-product Found uses the fixed-list/sign/note/shortage checks in the shared core. Invalid reasons and missing required notes roll back product, lot and ledger creation. `stock_by_hand.require_paired_move` also gives every standalone ingredient wrong-lot Found request the required paired-move instruction, including new lots. Other Found reasons keep their own rules.
4. **Every receiving supplier party:** `main._receive_supplier_parties` is called by both receive cores; every commingled party resolves to an active registered real supplier. Commit rechecks with supplier row locks. Unknown, inactive and sentinel secondary parties are refused with no ledger post.
5. **Verification and startup:** Ship, Make and Pack do not fetch stock-rule metadata at startup. Receive/Adjust/Found fetch it as a background read, including when switching from Ship; cached metadata is reused. Shipping retains its original two background reads. Tests now await asynchronous workflow switches and retain the pending-photo isolation assertion.

## Completed suites

| Check | Result |
|---|---|
| Final complete Python suite | **7,761 passed, 7 failed; 0 errors, 0 skipped** (7,768 collected) |
| Fresh complete `origin/main` Python suite, `29b64ffc8c03d3c8952be70db231ea3194792404` | **7,689 passed, the same 7 failed; 0 errors, 0 skipped** (7,696 collected) |
| New receiving-integrity HTTP regression file | **72 passed**, included in the final complete run |
| Complete JS suite | **746 passed, 0 failed, 0 skipped** |
| Architecture map | **0 problems** |
| Key-literal guard | **0 findings** (no live values supplied locally) |
| Whitespace check | `git diff --check` clean |

The final Python failure-ID set equals the main failure-ID set exactly: **zero new failing tests**. The 72 additional Python cases account for the complete collection increase; omitted master-key order-guard cases have been restored.

Commands (pinned Python 3.12 and repository Node/Playwright dependencies):

```sh
scripts/setup_test_db.sh --fresh
scripts/run_tests.sh -p no:cacheprovider -o addopts='' -q --disable-warnings --junitxml=/private/tmp/fl232-fix-final3-full.xml
scripts/run_tests.sh tests/test_no_stock_by_hand.py -p no:cacheprovider -o addopts='' -q --disable-warnings
PLAYWRIGHT_BROWSERS_PATH=/private/tmp/fl232-browsers node --test --test-concurrency=4 tests/*.js
python3 scripts/check_architecture_map.py
python3 scripts/check_no_key_literals.py
```

The main suite was run separately on a freshly rebuilt local `factory_ledger_test`, using an archive of the exact main SHA above. The final branch suite used an archive plus the changed files in `/private/tmp/fl232-fix-validation` to avoid macOS file hydration stalls. All **46 changed runtime/test files** were byte-compared and SHA-256 matched to the PR working tree before recording results. No Python/browser database suites ran concurrently.

The repository's pinned Playwright Chromium (revision 1243) was installed in a temporary local cache. The full JS run includes the real local HTTP/PostgreSQL browser checks:

- Receive, Adjust, Found and voided receipt screens; paired move has two opposite legs; Receive again preserves lot/supplier identity and BOL with quantity/photos blank.
- Existing adjustment-destination form behavior, including recording a 600 lb removal with no hold.
- Lot history: **270 actual authenticated HTTP requests**, text/PDF exports, and responsive 1440/820/390 layouts in both themes.
- Office/Assistant dashboard navigation checks.

## Remaining failures — each reproduced on main

| Test ID | Final branch | Fresh main |
|---|---|---|
| `tests/test_order_create_review.py::test_legacy_stuck_partial_order_finishes_without_new_shipment[False]` | fails | fails |
| `tests/test_order_create_review.py::test_legacy_stuck_partial_order_finishes_without_new_shipment[True]` | fails | fails |
| `tests/test_sales_order_state_model.py::TestNoDeadlock::test_multi_product_ship_and_exit_do_not_deadlock` | fails | fails |
| `tests/test_ship_order_idempotency.py::test_before_migration_088_the_commit_runs_unguarded` | fails | fails |
| `tests/test_ship_order_service_line.py::test_commit_route_preserves_already_fulfilled_409s` | fails | fails |
| `tests/test_trusted_browsers.py::test_flag_on_without_migration_reports_mode_on_but_unavailable_and_blocks` | fails | fails |
| `tests/test_write_tickets.py::test_migration_up_down_up_and_marker_stability` | fails | fails |

These failures predate this branch: the historical partial-shipment expectations, shipment fixture cleanup foreign key, pre-088/service-line expectations, trusted-browser feature/migration diagnostic, and the old write-ticket down migration dependency test. They remain visible and are not skipped or marked expected-failure.

## Specific policy proofs

- `test_marker_off_and_migration_leave_old_reason_policy_unchanged`: full reason rows remain unchanged, marker-off answers survive up/down, and marker-on source choices/USE_RECEIVE are code gated.
- A separate rollback-only probe loaded the **actual main `exceptions_enforcement.py` module** from the SHA above, applied 096, captured every reason row/column and old positive-side answers, applied 098, and verified all were unchanged and old `missing_receipt` still resolved. After the down file, old answers were unchanged and the marker absent. Every probe DDL/marker change was rolled back; no ledger posting occurred.
- `test_direct_found_named_identity_and_trace` (both direct routes), `test_direct_new_product_found_refusal_is_atomic`, and `test_found_wrong_lot_always_requires_net_zero_pair` prove the Found fixes.
- `test_direct_commingled_receive_refuses_unregistered_secondary`: preview and commit, unknown/inactive/sentinel parties.
- `test_under_pack_both_count_paths`, `test_cumulative_cap`, amended receipt/no-pack/above-zero cases: strict receipt-based limit, max 100 lb and cumulative per-lot usage retained.
- `test_two_concurrent_givebacks_cannot_both_post`: only one of two racing additions posts; the other is stale/refused under the commit lot lock.
- Paired move tests cover atomic lines, invalid inputs, shortage evidence, both-leg voiding and one owner email. The give-back/pair test also checks both legs in lot history, correct shift before/after and lot roll-forward, and one nightly move review.
- `test_nightly_review_includes_unnamed_and_keeps_existing_checks` proves unnamed ingredient additions are included while the existing unexplained-adjustment check remains present.
- Normal floor controls: marker-on Make, Pack, count-down, order shipping, standalone shipping and bulk shipping pass.
- Shared-key route refusal tests exercise enabled routes with default shutdown; disabled session routing retains its existing 404. Owner maintenance reach and floor refusal pass.

Historical shared-key contract/fault-injection tests explicitly request `legacy_shared_key_compat`, exercising the supported `SHARED_KEY_WRITES=allow` fallback. It is **not a suite default**. Ordinary workflow fixtures remain named; the default-shutdown matrix never opts into fallback. This preserves old attribution, transaction and fault-injection coverage without substituting a named actor while asserting anonymous behavior.

Shared-key shutdown remains isolated in commit `b18a6c4`, independently revertible. Checked-in scheduled reconciliation, exception sweep and expiry jobs use database connections, not the shared HTTP key. No live scheduler configuration was queried.

## Independent review

Claude Code cross-review is the next step after publishing these fixes. This document is implementation/verification evidence, not a Codex approval. Keep PR 232 draft. Any future staging/production action requires its own owner authorization; none is authorized here.

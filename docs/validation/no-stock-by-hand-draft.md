# No-stock-by-hand: blocked draft review record

This is an unfinished, blocked draft of section A, not a release or an approval request. Nothing was merged, deployed or applied to staging/production. Migration 098 exists only as code and in rollback-only/local disposable test databases. The owner-approved cumulative cap is retained exactly: every allowance addition ever recorded on the lot since go-live, including the proposed entry, must total strictly less than the smaller of the lot's effective receipt pack weight and 100 lb. Voids do not erase lifetime usage; paired moves do not earn or consume give-back.

## Reproduced specification conflict

Section A requires migration 098 to remove `adjust` from `missing_receipt.applies_to`, and also says applying the marker/migration with old code changes nothing. At origin/main `29b64ff`, `exceptions_enforcement.resolve_reason` directly rejects a reason when its `applies_to` excludes the source. It does not gate that check on 098.

A local `factory_ledger_test` transaction imported that unchanged origin/main module, applied 096, called `resolve_reason(cur, 'adjust', 'missing_receipt', delta_lb=10)`, then applied this draft's 098 and called the same old function again:

```text
ORIGIN/MAIN before 098: missing_receipt
ORIGIN/MAIN after 098: 422 REASON_CODE_INVALID
All compatibility-check changes rolled back; no ledger rows touched.
```

The migration keeps the explicitly requested reason-list change. The old-code compatibility guarantee is not met. No alternate rule or release order was silently selected. Section A says to stop and identify a requirement that cannot be built as written.

Claude Code cross-review could not run: its saved login expired and cannot renew. `claude auth login` is needed on this Mac. No approval is requested and the PR must remain draft.

## Enforcement and test map for the required review comment

| Requirement | Enforcement / display | HTTP or browser evidence in this draft |
|---|---|---|
| 1: one delivery per ingredient lot, preview and commit | `stock_by_hand.receive`; `main._receive_preview_core`, `_receive_commit_core`; ticket adapter `write_tickets.validate_receive` | `test_existing_receive_and_positive_adjust`; `test_direct_receive_validation`; `test_voided_receive_can_be_received_again_after_consumption` |
| 2: missing receipt goes to Receive; Found needs a new lot | `exceptions_enforcement.resolve_reason`; Found adapter in `ticket_actions.validate`; `main._found_commit_core`; `dashboard/entry.js` | `test_existing_receive_and_positive_adjust`; `test_found_new_lot_and_count_born_topups` (unfinished assertion: prepare returns a blocked 200 draft, with LOT_CODE_TAKEN, rather than the asserted 409); real-browser Receive redirect |
| 3: effective give-back, zero tolerance and no owner override | `stock_by_hand.addition`, called under posting lot locks from cores and ticket validation | `test_give_back_and_pair`; `test_go_live_does_not_credit_old_hand_removals`; `test_typing_refusal_is_bilingual_and_direct_reasons_enforced` |
| Both count paths; exact zero; effective receipt pack; strict boundary; lifetime cap | `stock_by_hand.addition`; `under_pack_counts`; `exceptions_enforcement.shortage_evidence`; owner count options in `ticket_actions.validate` | `test_under_pack_both_count_paths`; `test_cumulative_cap`; `test_no_pack_and_above_zero_count_refused`; `test_small_shortage_cannot_be_counted_above_zero`; `test_amended_receipt_pack_weight_is_effective` |
| Count alternatives, unverified and noningredient behavior | Bilingual messages in `stock_by_hand.addition`, `dashboard/exceptions-guide.js`; existing evidence resolver unchanged | `test_no_pack_and_above_zero_count_refused`; `test_noningredient_count_and_addition_unchanged` |
| 4: every posting route; fixed source, sign, note, destination; real supplier and positive quantity/BOL | Main shared posting cores, `exceptions_enforcement.resolve_reason`, `lot_confirmation.real_supplier` | Direct receive route matrix; direct typing/reason test; existing destination suite; full route/source matrix still needs completion |
| Addition source list and bilingual labels | 098 reason flags; `exceptions_enforcement._valid_reasons`; `dashboard/ledger-ui.js`, `dashboard/entry.js` | Existing destination tests plus new real-browser source list; broader source matrix unfinished |
| Paired wrong-lot move, same product, two locks in id order, one transaction/receipt, both trace legs | `stock_by_hand.validate_pair`; `ticket_actions.validate`; `main._adjust_commit_core`; `paired_lot_moves` | `test_give_back_and_pair`; `test_pair_validation_posts_nothing`; `test_pair_resolves_shortage_but_earns_no_giveback` |
| Paired shortage evidence, B below zero, void both legs, no give-back, one owner email | Existing `shortage_evidence` check retained; `record_shortages`; one `adjustment_alerts.queue_if_large` call | Pair evidence/void test; `test_large_pair_queues_one_owner_email` |
| Receipt, lot history, shift roll-forward and reconciliation show one move/two legs | `dashboard/receipt.js`, `dashboard/lot-history.js`; `lot_trace.history`; `shift_summary` adjust branch; `reconciliation.load_window/analyse` | New browser draft checks both lots; history and shift integration coverage still needs completion |
| 5: nightly review of every effective ingredient hand addition, named or not | `reconciliation.load_window`, `analyse`, marker-gated | `test_nightly_review_lists_named_ingredient_additions_only` |
| Shared key read-only, two lookup exceptions, emergency switch, API_KEY still required | Independent auth commit: `main._authenticate_incoming_api_key` | `test_shared_key_default_and_owner_scope`; route-enumeration test remains failing on disabled `/auth/session/logout` returning 404 before authentication, rather than asserted 403 |
| Named owner key gets owner route reach; no other role widened | `main._authenticate_incoming_api_key` uses full authenticated identity, including key_kind; append-only maintenance correction receives Request attribution | `test_named_owner_reaches_maintenance_but_floor_does_not`; amended receipt actor assertion; existing identity/role fixtures are still being migrated |
| Receive again after void, with same identity/BOL/FL code, blank quantity/photos | `dashboard/receipt.js`, carry handling in `dashboard/entry.js`; no new posting action | `test_voided_receive_can_be_received_again_after_consumption`; `tests/visual/run-no-stock-by-hand.mjs` |
| Marker off, rerunnable migration, down file | `stock_by_hand.live_at`; marker-off UI flag on SELECT-only `entry/lots?rules_only=true`; 098 and down | `test_migration_rerunnable_and_down_restores_list`; broader marker-off assertions unfinished; old-code conflict above |
| Two additions cannot both spend give-back | Ordered lot locks in posting core; guard re-evaluated at commit | `test_two_concurrent_givebacks_cannot_both_post` exists but has not completed successfully |
| Docs, no GPT changes, changelog, architecture map | Deployment guide, Arturo guide, named-actor guide, architecture map, one regression row | Architecture map check: 0 problems. GPT schemas/instructions unchanged. |

## Validation state

- Most recent focused HTTP run excluding the unfinished race: **47 passed, 2 failed, 1 deselected**. The two remaining failures are identified in the table above. The 50/49.99 lb and 100/99 lb boundaries, both count paths, the 55 lb cumulative cap, amended receipt weights, give-back, paired postings/evidence/void, named-owner maintenance reach and nightly review are among the passing checks.
- Full Python attempt rebuilt the scratch database first but was interrupted at about 60% after stalled local file reads: **179 failed, 4,556 passed** at that point. This was before subsequent fixture/auth corrections. A later selected run before the final owner-scope correction had **43 failed, 2,660 passed, 1 error**. The full suite is not green and has not been represented as complete.
- Full JavaScript run: **738 passed, 7 failed**. Four real-browser wrappers could not start the missing default Playwright Chromium binary; two picker-search assertions counted the new metadata lookup; the lot-history browser check also failed. After separating the metadata response in the test harness, the picker suite is **123 passed, 0 failed**. Browser reruns and baseline comparison are recorded in the PR as available.
- **Real Chrome workflow/visual rerun passed**: Receive redirect, paired Adjust with two legs, Found refusal, and the voided receipt/Receive again screen. It preserves product, supplier/no-lot identity, BOL and FL lot code while leaving corrected quantity and photos blank. The synthetic void is performed by Michael.
- Full origin/main Python baseline completed earlier with seven failures: `test_order_create_review` legacy stuck-partial (both parameter cases), `test_sales_order_state_model::TestNoDeadlock::test_multi_product_ship_and_exit_do_not_deadlock`, `test_ship_order_idempotency::test_before_migration_088_the_commit_runs_unguarded`, `test_ship_order_service_line::test_commit_route_preserves_already_fulfilled_409s`, `test_trusted_browsers::test_flag_on_without_migration_reports_mode_on_but_unavailable_and_blocks`, and `test_write_tickets::test_migration_up_down_up_and_marker_stability`. These are baseline observations, **not an explanation for this draft's other failures**.

Do not mark this PR ready. Cross-review must list remaining gaps, and each later staging/production step still requires separate owner approval and the prescribed backup/verification.

Completed origin/main JavaScript baseline: **744 tests, 741 passed, 3 failed**. The three failures are the Adjust destination, back-to-dashboard navigation and lot-history browser wrappers, each missing the default Playwright Chromium executable. This independently proves those environmental baseline failures; it does not explain this draft’s other failures.

## Final JavaScript rerun

`PLAYWRIGHT_CHANNEL=chrome node --test --test-concurrency=1 tests/*.js` completed: **745 tests, 743 passed, 2 failed**. The failures are the shipping-preload/background-activity assertion (new metadata-read fixture work remains) and the lot-history browser wrapper (its runner calls `chromium.launch()` without honoring the Chrome override, so it still requires the missing default Playwright executable; the same failure was reproduced on origin/main). The four requested screens pass in real Chrome, including the receipt correction flow. This does not make the incomplete Python suite or the blocked specification/review ready for approval.

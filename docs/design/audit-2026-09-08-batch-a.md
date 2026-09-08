# Batch A implementation — September 8, 2026

Status: implemented and verified locally on `fix/design-audit-1`. Not pushed or deployed.

Working checkout: `/Users/cns/Documents/Codex/2026-09-08/ca/work/design-audit-1`.
The shared `/Users/cns/Documents/factory-ledger` checkout was left in place on its existing branch.

## Approved scope and results

| Finding | Implemented behavior | Verification |
|---|---|---|
| 1 — Quantity/unit mismatch | Weight-labelled cases and bags show ledger pounds in Recent Entries, Supplies inventory/lots and ingredient panels. Signed quantities and correction events stay intact. Count-based unit/each/container products keep their native units. Supply requests retain purchasing units via a separate `request_unit` response field. | Production SELECT trace; DB regression cases for negative shipments, positive and negative packing, counted supplies, 500 lb on hand versus a 10-bag request. |
| 8 — Accessible controls | Factory Ready checkboxes identify the order/customer; the column has a visible heading. Notes edit/delete name the record. Trace zoom controls have names and tooltips. | Browser accessibility snapshots and trace interaction. |
| 9 — Receipt gating | Manual Save requires an explicitly selected product, supplier and finite positive pounds. Inline text identifies missing information. Editing an existing receipt keeps product/supplier fixed; saves in flight cannot be duplicated. | Browser blank/incomplete/valid/zero checks and Node gate tests. Existing intake-review logic tests all pass. |
| 10 — Readiness semantics | Preparation is explicitly “Factory Ready.” Advisory dispatch states are “Checks passed,” “Needs review,” or “Not checked”; explanations appear in the list and details. Divergent closed shipments retain review status and blockers. | Browser order list; Node readiness cases; existing database readiness suite. No shipping policy changes. |
| 11 — Dispatch attention | Fetches the full fulfillment-check set independently of the visible list filter. Unknown counts offer “Check” and a load/retry destination. A previously verified count is identified as stale after a failed refresh. | Initial browser load and forced outage; Node complete/empty/malformed response cases. |
| 12 — Health score | Labeled score opens a native dialog with plain check names, supplied record identifiers, review steps and retry. Missing IDs/units are identified rather than invented. | Browser score 80/100 and failure states; Escape closes the dialog and restores focus to its button. |
| 28 — Operational wording | Maps raw directions, product types and lot sources to readable names; uses “Left to ship” and “Related record name or number”; removes storage implementation wording. | Browser recent feed, notes/order labels and trace controls; Node label checks. |
| 33 — Production values | Idle output/stages show zero, idle yield is not applicable, active runs lacking input have unavailable yield, and failed initial loads show unavailable data. | Browser idle/outage screens and Node actual-render tests for idle, missing input and calculated yield. |

## Quantity trace against production

The read-only trace used port 5432 and an explicit read-only transaction. No production records were changed.

| Transaction / line | Product | Raw and effective ledger delta | Activity representation |
|---|---|---:|---|
| TX-2207 / 6851 | SKU 70060, 25 lb case | −1,000 lb | 1,000 lb / 40 units |
| TX-2206 / 6850 | SKU 10304, 25 lb case | −3,000 lb | 3,000 lb / 120 units |
| TX-2205 / 6849 | SKU 10300, 10 lb case | −100 lb | 100 lb / 10 units |

The Recent Entries API combined ledger pounds with the product packaging UoM. Activity was correct for these checked shipments. Corrections therefore belong in presentation/serialization, not the stored data.

The Supplies cross-check found ingredient 75 (Graham Cracker Crumbs) also stores pounds under `50 lb bag`. Ingredient 291 has a count-based `container` adjustment, and packaging uses `unit`/`each`. The fix explicitly preserves these count units. Supply-request quantities remain in purchasing units; their values are not converted.

## Validation

- 63 Python tests passed: `test_recent_ledger.py`, `test_supplies.py`, `test_sales_order_readiness.py`, `test_expected_receipts.py`, and `test_dashboard_b2.py`.
- 40 Node tests passed: `test_batch_a_ui.js` and `test_er_intake_logic.js`.
- Dashboard JS and both modified inline-page scripts passed syntax checks; `git diff --check` passed.
- Browser testing used a localhost fixture server which intercepts operational API calls and rejects writes. These are local implementation checks, not a post-deployment live smoke test.
- Python tests used a separate local database, `factory_ledger_design_audit_a`. Existing deprecation warnings remain; there were no test failures or skips.
- The old cache assertions pinned to CSS v27 / JS v40 were updated to the new versions.

Re-run from the worktree with the pinned Python 3.12 environment and `TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_design_audit_a`. Run `node --test tests/test_batch_a_ui.js tests/test_er_intake_logic.js` for the UI logic checks.

## Versions and remaining scope

Dashboard JavaScript: v52 → v53. Dashboard CSS: v35 → v36. Navigation links to the modified inline Production Lines and Traceability pages use v2. Material Flow's only change is those navigation URLs.

Batch B remains deferred: findings 2, 3, 4, 5, 6, 7, 15, 16, 20, 22, 31.
Batch C remains deferred: findings 13, 14, 17, 18, 19, 21, 23, 24, 25, 26, 27, 29, 30, 32, 34.

The health-check findings themselves are not repaired by this UI work. Counts-only checks still need administrator lookup for record IDs. No migrations, inventory corrections, shipping gates, or production writes were introduced. Deployment remains pending explicit authorization.

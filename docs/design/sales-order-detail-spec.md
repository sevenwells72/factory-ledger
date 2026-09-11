# Sales Order detail redesign and list Step 3.1

Date: 2026-09-11. Branch: `feat/so-detail-redesign`. Frozen baseline: `12b01b6`.

This specification is the first commit, before application changes. References: [Sales Orders list specification](sales-orders-redesign-spec.md), [MASTER Category 17, STATUS-001–014](FL-Design-Standards-MASTER.md#17-status--data-display), and [state model and Health v2.1 findings](so-state-model-findings.md).

## Acceptance and boundaries

The full visual audit must report zero STATUS-002/004/005/006/007/008/010/011 failures for Sales Orders list and detail screens. No new screen/variant/rule failure cells may appear elsewhere. LAYOUT-020 refresh-timing noise is excluded only when reproduced on the frozen baseline, following PR #45. At least 294 Python and 62 JavaScript tests must pass. Blubber must approve the Netlify PR preview before merge; human approval will not be inferred from automated checks.

Application changes are confined to `dashboard/`. Tests, fixtures, this specification, audit evidence, and changelogs accompany the implementation. Do not change `main.py`, `migrations/`, or `openapi-gpt-v3.yaml`. Use small commits and at most two review passes. If the harness still fails after pass two, stop corrections and report the unresolved failures.

## Case summary and independent dimensions

The detail header leads with SO number, customer, total ordered physical pounds, and ship-by date with factory-local relative wording. Missing optional copy is omitted; a missing date reads “Ship-by date not set.” Relative dates remain muted, including overdue dates. The desktop summary is compact; mobile wraps into readable fields without page overflow.

State, Ready to ship, Fulfillment, and Health occupy separate labelled elements (STATUS-001/013). State is a neutral Open, Closed, or Cancelled chip. Its explanation defines administrative state and includes the recorded reason and note. Exited orders visibly include the change date, actor when present, and a navigable related SO when supplied. Do not manufacture missing attribution.

Ready to ship is the existing floor flag, represented by a neutral checkbox, with the same gate as the list: authoritative open orders may change it, including open/shipped orders; closed/cancelled orders cannot, and an in-flight request disables repeat submission. Its popover gives the production/packing/staging definition and recorded actor/time or “Not marked.”

Fulfillment is neutral Unshipped, Partial, or Shipped, with the explanation “Ledger shows X of Y lb shipped” and the exclusion of voided shipments and cancelled lines. Read effective ledger quantities, never the mutable recorded shipment counter.

Health alone carries severity colour: critical uses danger, warning uses warning, quiet has no alarm or positive badge. A quiet Health field retains a neutral explanation affordance so its information remains accessible. Render server `reasons[]` followed by `info[]`, with `info_detail[]` expandable. Do not recompute the server's time-window severity in the browser. Show the allocations-not-enforced caveat once at its first relevant page location and strip repeated copies from explanations.

Every condition chip uses the existing `SOList` explanation behavior: `data-explain`, matching `aria-describedby`, and a focus stop. Hover, keyboard focus, tap, Escape, and outside dismissal work; only one explanation opens at once. Extend shared exports as needed rather than adding a second popover implementation (STATUS-005).

## Lines and secondary content

The lines table shows Product, Ordered lb, Shipped effective lb, Remaining, Allocated, Unallocated, Health, and Line status. Preserve inventory expansion and existing line/header editing where permitted. Non-weight/service lines use an explicit units label and never enter physical-weight totals. Cancelled lines stay visible in muted text with a neutral explained line-status chip; do not hide them or lower contrast using opacity.

Line Health information is matched by `line_id` from the order's `health.info_detail[]`. These rows currently describe unallocated pounds and are informational, not an independent severity: a matching line gets a neutral “Allocation details” chip and explanation naming product and unallocated pounds; unmatched lines have an empty Health cell. Do not assign the order-wide alarm to every line or reintroduce legacy dispatch blockers. Line status is a separate neutral explained condition. Numeric values, allocation/reservation tables, inventory expansion, exit previews, and pallet quantities all use `SOList.number`, whose rounding is half away from zero (ROUND_HALF_UP), pounds whole with separators and pallets one decimal. No secondary numeric renderer.

Keep desktop unexpanded table rows at most 56px, with secondary facts in explanations/expanded rows. Remove dangling dash subtitles and redundant summaries. Existing allocation/shipping actions stay usable. Replace “Auto FIFO expires after 48 hours unless the API returns a different expiry.” with plain copy describing first-in, first-out allocation and the recorded expiry shown in reservations. Errors use actionable office/factory language rather than raw responses (STATUS-006–011).

## Exits, refreshes, and legacy removal

Header Close and Cancel buttons appear on open orders, Reopen on exited orders. Reuse `SOListActions.open` from the already imported `so-list-actions.js`; do not duplicate its dialog. Keep preview then commit, release counts/weights, required notes and related SO validation, the 409 suggested-close offer, and preview invalidation on every field change. Never send `changed_by`. On commit refresh detail and list data/counts and restore focus to an available header action. Related SO links navigate to the corresponding detail.

Remove the legacy order-status selector and transition path. All order gates use authoritative state and effective fulfillment. Header editing follows the server's current open/unshipped gate; allocations and shipment previews require an open order. Eliminate legacy sales-order status reads everywhere in `dashboard/`, including calendar consumers. Allocation statuses, HTTP statuses, and unrelated domains are separate concepts and remain valid.

## List Step 3.1

Remove Hide ready to ship markup, listeners, filter logic, and CSS. Keep Filter by customer and the six counted tabs. Place Sort and Resize columns controls in the table header area, preserving sticky positioning, column sorting/resizing, keyboard access, and mobile usability. Delete CSS specifically belonging to removed controls. Update the existing list layout and interaction assertions to this control contract without weakening general audit rules.

## Evidence and delivery

Run the unchanged full audit on frozen baseline and final implementation with the same six default variants (1440/390, light/dark, plus existing zoom captures). Preserve the #45 evidence layout under `docs/design/audit/pr-screenshots/feat-so-detail-redesign/`: before/after raw results, browser-check reports, logs, selected viewport PNGs, comparison JSON, and an audit summary. Keep list and detail counts separate. Any fixture modernization must preserve the legacy baseline facts and be documented so improvements cannot be attributed to silently easier inputs.

Add `tests/visual/run-so-detail-interactions.mjs` for keyboard popover open/close, the real shared exit dialog's preview/commit on stub orders, field-edit preview invalidation, the 409 offer, Ready checkbox gating, and detail/list-count refreshes. Run existing Python and JavaScript suites plus list layout/interactions and shared exit-dialog checks. Two review passes comprise complete rendered audits and review of the accompanying interaction/test results; corrections occur only between the two passes.

Bump each changed browser asset's `?v=` in `dashboard/index.html`. Add newest-first FACTORY_LEDGER_CHANGELOG row 141 marked **NOT DEPLOYED**, and a dated CHANGE_LOG entry. Push the branch, create a PR with before/after failure tables and the verified Netlify preview link, and report the PR URL. No merge or production deployment is authorized. Keep human-review instructions concrete and free of angle-bracket placeholders.

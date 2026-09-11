# Sales Orders list redesign — Step 3

Date: 2026-09-11. Branch: `feat/so-list-redesign`, based on `origin/main` (`e772e36`).

This specification is committed before implementation. Scope is the dashboard Sales Orders list and its expanded-row Close, Cancel and Reopen actions. The detail page remains Step 4; STATUS-013 is therefore out of scope.

References: [MASTER STATUS-001–014 and explanation hook](FL-Design-Standards-MASTER.md#status-001--state-fulfillment-readiness-and-health-are-four-independent-dimensions-and-always-render-separately), [state model, Health v2.1 and info_detail](so-state-model-findings.md), and [visual harness](../../tests/visual/README.md).

## Requested specification


Tabs (STATUS-014), replacing the "All Open Orders" dropdown and the "Overdue only" checkbox, counts from GET /sales/orders/counts, refreshed with the list:
  Open (default) · Ready to ship · Overdue · Shipped · Closed · Cancelled
  Open = state=open. Ready to ship = open ∧ flag set (client filter). Overdue = ?overdue_only=true. Shipped = open ∧ fulfillment=shipped — the "ready to close" set. Closed / Cancelled = state= filter. Keep the customer filter and the ready-to-ship hide toggle as secondary refinements; relabel the toggle "Hide ready to ship".

Columns, in order (STATUS-001 — one dimension per column; STATUS-008 — single line, ≤56px at ≥1200px):
  1. Ready to ship — the existing Factory Ready checkbox, relabeled, still interactive. Popover: "Everything on this order is produced, packed, and staged for pickup." + "Marked by floor · Aug 26, 10:58 PM" or "Not marked".
  2. SO #  3. Customer
  4. Ship by — date on one line; relative in muted text on the same line ("in 3 days" / "15 days overdue"); danger tone only when overdue. Order date moves into the SO # popover.
  5. Fulfillment — unshipped renders as "—" (STATUS-002), partial as "Partial · 1,200 / 2,400 lb", shipped as "Shipped". Popover: "Ledger shows X of Y lb shipped; excludes voided shipments and cancelled lines."
  6. Health — critical: one danger-toned ⚠; warning: one warn-toned ⚠; quiet: nothing. Never text in the cell. Popover: each health.reasons line, then health.info lines with info_detail expandable. (STATUS-004: this is the only alarm in the row; STATUS-012: exactly these two tones.)
  7. Left to ship  8. Pallets — one line, "1 mixed (0.3)"; breakdown in the popover.
  Remove: Status column ("Confirmed" is implied by the Open tab — STATUS-003), Dispatch checks column, Blockers/Warnings column, the "Factory Ready not set" chip, the "Ascending" label under the Ship By header, the second Refresh button, and the disclaimer sentence at the top (its content lives in the popovers — STATUS-011). Do not read the legacy dispatch-check fields anywhere in this view.
  In the Closed and Cancelled tabs, Health and Ready to ship are blank and a single quiet chip shows the reason: "Closed · shipped, not recorded".

Popovers (STATUS-005): use the MASTER-defined hook exactly (data-explain + aria-describedby + a focus stop). Hover on pointer devices, tap on touch, Esc closes, one open at a time. Generic definition line first, this-record specifics second. Every chip, ⚠, checkbox and abbreviated cell listed above has one.

Actions (row expand panel):
  Close → dialog: reason (shipped-recorded / shipped-not-recorded / short-closed), note. Cancel → dialog: reason (customer cancelled / CNS declined / duplicate / superseded / other), note required for other, related SO required for duplicate/superseded. Both call the endpoint with mode=preview first and show "Will release N reservations (X lb)" before the commit button; then mode=commit. Cancel on a non-unshipped order: render the 409's suggested_action/suggested_reason as an inline offer to close instead. Reopen in the Closed and Cancelled tabs. Do not send changed_by. On any API error, show a plain-language message, never raw JSON (fixes the HTTP 503 leak the harness caught under STATUS-010).

Numbers (STATUS-006/007): one formatter for every quantity in this view — thousands separators, whole pounds, pallets to one decimal, tabular numerals; no dangling dashes on second lines.

Mobile (390px captures): tabs scroll horizontally, popovers open on tap, touch targets per the TOUCH rules.

Cache-bust: bump dashboard.css and dashboard.js versions in index.html (the test reads them from there).

Harness: run tests/visual/run-visual-audit.mjs and run-sales-orders-layout.mjs BEFORE changes and record per-rule STATUS counts for the sales-orders screens; run again AFTER. Target: zero STATUS-002/004/005/006/007/008/010/011 failures on the sales-orders screens; no new failures on any other screen. Put both tables in the PR body. Save before/after captures at 1440 and 390 under docs/design/audit/pr-screenshots/feat-so-list-redesign/.

Changelogs: FACTORY_LEDGER_CHANGELOG.md row, CHANGE_LOG.md, and /Users/michaelgross/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md.

Open PR "feat(dashboard): Sales Orders list — tabs, orthogonal columns, popovers, exits". Report: the before/after harness tables, the Netlify preview URL, and anything in this spec you could not do as written and what you did instead.
## Implementation decisions and API reconciliation

- STATUS-001/003: retain eight independent columns. In exited tabs place the quiet state/reason chip with the order identifier; Ready to ship and Health remain empty. Each exceptional chip explains its definition and record facts without redundant subtext (STATUS-009).
- STATUS-004/012 versus the requested overdue date tone: Health is explicitly the only alarm. Owner confirmed: keep overdue date wording muted and leave severity exclusively to the server-provided Health icon. Do not recalculate Health v2.1 in the browser.
- STATUS-005: `data-explain="explanation-id"`, `aria-describedby="explanation-id"`, and a native focusable control or `tabindex="0"`. Explanation content is hidden until opened. One at a time, with hover, keyboard focus, touch tap, Escape and outside dismissal. Popovers that include expandable info support keyboard and touch interaction.
- STATUS-006: one list formatter handles pounds/counts (whole numbers), pallet equivalents (one decimal) and separators, including row expansion, exit previews and health detail quantities. Numeric text uses tabular figures. Factory-local dates determine relative ship-by wording.
- The existing list API calculates but omits `ordered_lb` and `shipped_effective_lb`. Expose those additive fields so fulfillment and left-to-ship explanations use effective ledger totals, without one detail request per row. Include `line_status` in pallet lines so cancelled cases are excluded. Keep the ready-flag endpoint interactive for authoritative open orders, including physically shipped ones, and reject exited orders under the order row lock. Align the overdue list filter/flag with the counts endpoint's authoritative `state=open` and non-shipped fulfillment.
- STATUS-014: counts are the global API buckets and refresh alongside the list. Customer and hide-ready refinements remain secondary. The API's 200-row limit must be disclosed when reached, including that counts cover all orders; do not silently imply all matching orders were loaded.
- API reason values use underscores; render the specified human-readable labels. Preview release totals are computed from `reservations_to_release`. Reopen also previews before commit and explains that old reservations are not restored. Never send `changed_by`.
- Health `info` for quiet orders remains accessible from the SO # explanation while the Health cell stays blank. Health reasons precede info, with `info_detail` expandable.
- STATUS-010: map request failures to plain-language messages. Preserve structured error metadata internally only for the cancel-to-close offer and validation handling.
- Mobile horizontal tab scrolling is explicitly requested and takes precedence over MASTER NAV-008's generic no-scroll recommendation. Use TOUCH-003/004 target sizes and spacing, with tap explanations.
- Baseline and after audit tables distinguish list screens from detail screens, so existing detail-page STATUS findings are recorded without redesigning Step 4. Update obsolete layout assertions to inspect the requested columns rather than removed Status/Blockers, keeping substantive fit and reachability checks.

## Validation and delivery notes

- The existing backend must receive the additive list fields and Ready state guard before the redesigned dashboard is used against production; opening the PR creates a Netlify preview but does not deploy the Railway backend.
- The old table layout regression asserted Status width and Blockers reachability. Its replacement retains fit/sticky/scroll/reachability assertions for all eight requested columns and adds the 56 px bound and the 1200 px boundary. Legacy fixtures remain for unrelated/detail screens; a separate list fixture supplies the new API shape.
- Row clicks are now explicit explanation or expansion controls. The expansion provides **Open order details**, leaving the detail layout untouched. Returning from detail refreshes list state, Health and counts.
- Old saved sorting/column-width preferences are scoped to the previous column schema; the new list starts with Ship by sorting and stores preferences by tab under a fresh key.

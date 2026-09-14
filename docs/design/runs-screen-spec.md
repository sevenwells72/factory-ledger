# Production Runs screen — S1 scheduling

Date: 2026-09-14. Branch: `feat/runs-screen`, based on `origin/main` (`7c183e0`, PR #52).

This specification is committed before implementation. Scope is a standalone `/runs.html`; the dashboard navigation link ships separately. References: [scheduling Part 4](scheduling-spec-draft.md#part-4--s1-as-built-2026-09-14--normative), [MASTER Category 17](FL-Design-Standards-MASTER.md#17-status--data-display), [Sales Orders specification](sales-orders-redesign-spec.md), and [visual harness](../../tests/visual/README.md).

## Requested specification

A Monday–Sunday week, defaulting to today in America/New_York, with Previous, This week and Next controls. Each day contains one compact row per run: product; native planned quantity/unit; optional line; quiet status; coverage summary; actions. Display all seven days, including empty days. Tabs: This week, Planned, In progress, Done, Cancelled, with client-derived counts for the disclosed loaded date range. When navigating weeks, This week means the all-status bucket for the displayed week; the date range is explicit.

Quantities and counts use SOList.number exclusively (whole pounds, thousands separators, tabular figures). Each quantity/status has SOList.trigger/explanation hooks: data-explain, matching aria-describedby and a focusable control; hover, focus, tap, Escape, outside dismissal, one open at a time. Quantity explanation contains canonical pounds and the saved case-weight conversion. Coverage reads “covers N lines · X lb of Y lb” and lists SO numbers, line identifiers and pounds. Done and Cancelled stay muted, never positive or warning colors. Health is absent. Desktop list rows stay at most 56px high; mobile uses compact stacked content and 44px targets.

New run: searchable active finished product, quantity, cases/lb, date, optional production line and notes. Disable cases with an explanation when current case weight is missing. Edit only planned/in-progress runs with PATCH, including movement between those states. Cancel has a confirmation dialog and reason note. Complete first loads evidence: looks_complete → “Looks complete — Confirm”; partial → recorded/planned pounds and “Confirm anyway”; none → “No production recorded” and “Confirm anyway”. Evidence shows the date window and explains once that completion creates no inventory, changes no sales order, and does not set Ready to Ship.

Coverage editor: load open sales orders, filter for the product, show each eligible line's exact effective remaining pounds, editable covered pounds, running total against the plan, and one full-replacement Save. Zero removes a line; all zero clears coverage. Preserve entered values on failure. RUN_OVERCOVERED and per-line conflicts appear inline in plain language. Terminal runs have no write controls.

All requests use the same dashboard service and X-API-Key (`dashboard-key-2026`) as existing pages. No key-entry field. New assets use `?v=1`; existing assets retain current versions.

## Implementation decisions and API reconciliation

- Main.py is read-only. The seven S1 routes are authoritative: GET list has inclusive from/to; create/PATCH return `{run}`; evidence returns the summary directly; coverage accepts `{coverage:[{sales_order_line_id,qty_lb}]}`. Cancel sends `{reason}` and Complete sends `{note}`. Never write to an SO endpoint.
- `/products/search` only accepts q/limit; finished and active filtering is client-side. The server remains authoritative for service/resale restrictions. Search and week loads use generation guards against stale responses.
- Product search currently returns no line assignment. Leave a new run's line optional, allowing the server's single-assignment default; display the returned line on the row and prefill subsequent edits. Product search refreshes the current case-weight choices on edit; unchanged quantity/unit fields are omitted from PATCH to preserve the original conversion. If product data supplies a default line, prefill it. Accept a production-line number because S1 exposes no dashboard-allowlisted line catalog.
- `/sales/orders?state=open&limit=200` exposes pallet_lines without product ids or exact effective remaining pounds. Identify candidate orders by SKU/name (or additive product_id), then read their existing `/sales/orders/{id}` detail, match the stable SKU (product_id when supplied), non-service and non-cancelled lines and use readiness.remaining_lb. Never derive pounds from rounded cases. Disclose the 200-order cap and offer customer narrowing; preserve existing coverage outside the loaded candidates in every full replacement, with an explicit removal control so saving never silently drops links.
- Request errors are mapped from error_code to factory language, including line-specific metadata, with a safe generic fallback; never render response messages/JSON. Failed reads show retry and cannot enable a write based on incomplete data. Prevent duplicate submissions and retain dialog values. Inputs keep exact editable decimal values; display quantities follow the shared formatter.
- Requested status chips remain on rows even inside filtered tabs, an explicit scope choice relative to STATUS-003. No normal/healthy badges, row alarms or Health recomputation. Popover explanations provide record details without repeated visible disclaimers.

## Validation and delivery notes

Only new runs HTML/JS/CSS, new interaction runner/fixture, this spec, evidence, the two requested changelog entries, and evidence ignore rules may change. Shared screen registration stays in the new runner. All existing screen assets and backend files remain byte-identical to the base.

Capture the absent page before implementation (404, marked not applicable rather than a passing audit), then the implemented week, dialogs, evidence states, coverage errors, and tabs at 1440/390 in light/dark. Run the shared eight STATUS checks plus meaningful stateful interaction assertions for all seven routes, authentication, native quantities, current factory week, stale reads, errors, terminal gating and full replacement. Commit PNG, log and Markdown evidence only; exclude result/summary JSON and artifacts over 1 MB. Run existing JavaScript tests. At most two review passes; if the second still fails, stop and report.

Deliver a pushed PR with the before/after failure table and verified Netlify preview `/runs.html`. Per owner clarification, put the deploy-preview URL ending in /runs.html in the PR body and stop after opening the PR. Blubber reviews independently and merges manually; no reviewer outreach. Changelog next free row is 144, NOT DEPLOYED, subject to renumbering on merge.

Environment note: the requested local checkout stalled reading tracked files during git checkout; implementation uses a fresh isolated clone of the same remote, synced with checkout main, fetch --prune and pull --ff-only and verified at 7c183e0. The original checkout is preserved.

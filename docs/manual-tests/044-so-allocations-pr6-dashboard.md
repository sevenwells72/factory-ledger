# PR 6 Dashboard Manual Test — Sales-Order Allocations

Run this procedure only against a local API using the `factory_ledger_test` database, or an explicitly authorized staging environment. Never point these steps at production. The committed dashboard uses its normal absolute API constant; for local testing, use an uncommitted local override of `SALES_API_BASE` and restore it before committing.

## Prerequisites

1. Start the API against `postgresql://localhost:5432/factory_ledger_test` with `DASHBOARD_API_KEY=dashboard-key-2026`.
2. Serve `dashboard/` as static files and open the Sales Orders tab.
3. Prepare two open, physical sales-order lines for the same SKU and enough positive posted lot inventory to test both covered and competing demand. At least one lot should be eligible for a manual pin. Keep the order and line IDs available for API/database fixture setup.
4. For the auto-FIFO case, have two positive lots so the allocation can show one or more FIFO rows with `source=auto_fifo` and a future expiry.

## Test steps

1. Open an order that already has a mix of manual, lot-level, auto-FIFO, released, or shipped allocation history.

   Expected: the header shows computed Dispatch Ready or Blocked state and blocker chips with visibly distinct block, warning, and information treatments. Each physical line shows Ordered, Shipped Effective, Remaining, Allocated, and Shortage. Service lines show Service rather than false weight readiness. The reservation table identifies Manual, Staged lot, or Auto FIFO source; distinguishes SKU-level from a named lot; and offers Release only for a live allocation. The page states that reservations do not move physical inventory and that Dispatch Ready is not a shipping gate.

2. On an open physical line with coverable demand, select `Manual · SKU level`, enter a quantity within both effective remaining need and `coverable_lb`, and click Allocate.

   Expected: the request succeeds, a live SKU-level Manual row appears, Allocated increases, Remaining Effective does not change, and blocker chips update. No text claims that stock moved or shipped.

3. On an open physical line with a positive eligible lot, select `Manual · specific lot`, choose the lot, enter a coverable quantity, and click Allocate.

   Expected: the request succeeds and the reservation table shows the lot code and a lot-level source (`Manual` or `Staged lot` when the API classifies a STAGED/found-inventory lot). Physical on-hand is unchanged.

4. Submit a manual quantity greater than the line's returned `coverable_lb` or effective remaining need.

   Expected: the API returns structured 409 `OVER_ALLOCATION` (or the line-demand guard), and the UI shows an actionable sentence such as “Only N lb is coverable. Reduce the request to N lb or release a competing reservation.” Raw JSON, the HTTP response body, and an undefined value are not displayed. No reservation row is added.

5. Click Release on a live reservation and confirm the prompt.

   Expected: the row becomes released (or is reloaded in released history), its Release button disappears, Allocated and blockers recompute, and the confirmation copy explains that release makes pounds available to other orders without moving physical inventory.

6. Select `Auto FIFO · 48h TTL` on a line with unallocated need and click Allocate. Leave the prefilled quantity to allocate that need, or enter a smaller coverable quantity.

   Expected: one or more lot-level Auto FIFO rows appear in FIFO order. Every Auto FIFO row shows a live countdown such as `1d 23h remaining`; the countdown updates without a reload. An elapsed allocation displays Expired and cannot be released as though it were live. Manual and staged rows display No expiry.

7. Return to the orders list and select `Dispatch Queue`.

   Expected: the dashboard calls the read-only fulfillment-check route, shows a ready/blocked count, and allows All Dispatch States, Dispatch Ready, or Blocked filtering. Dispatch-ready orders sort before blocked orders, then by ship date. Blocker reasons are visible as severity chips, including block reasons such as shortage/unallocated/not-floor-ready and warning reasons such as no-ship-date/inbound-cover. A fulfillment-diverged shipped or invoiced order remains visible when returned by the API.

8. With `ALLOCATIONS_ENFORCED=false`, open an eligible order and click Preview all remaining lines.

   Expected: requested pounds, `can_ship_lb`, `reserved_others_lb`, and owning order numbers render when supplied. There is no allocation-warning block anywhere in the preview because PR 5 V-2 intentionally omits `allocation_warning` while the flag is off. There must be no empty box, `undefined`, blank warning heading, or other warning artifact.

9. With `ALLOCATIONS_ENFORCED=true`, create a scenario where another order reserves pounds for the same SKU (`reserved_others_lb > 0`), then run the same preview.

   Expected: the returned `allocation_warning` renders as an amber warning block with its structured warning code/message and the competing reservation details remain visible. This is a preview only and does not ship. If no authorized flag-on staging/local server is available, mark this step **deferred until staging**; do not mock the warning and do not run it against production.

10. Reload the page and reopen the tested order.

    Expected: allocation history, current computed readiness, effective remaining pounds, blockers, and TTL state all come from the API and remain consistent after reload. The browser console has no uncaught exception, and all dashboard requests use `X-API-Key: dashboard-key-2026` rather than the master key.

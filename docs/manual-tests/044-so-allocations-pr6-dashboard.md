# PR 6 Dashboard Manual Test — Sales-Order Allocations

Run this procedure only against a local API using the `factory_ledger_test` database, or an explicitly authorized staging environment. Never point these steps at production. The committed dashboard uses its normal absolute API constant; for local testing, use an uncommitted local override of `SALES_API_BASE` and restore it before committing.

## Prerequisites

1. Start the API against `postgresql://localhost:5432/factory_ledger_test` with `DASHBOARD_API_KEY=dashboard-key-2026`.
2. In `dashboard/dashboard.js` line 1855, temporarily replace the committed `const SALES_API_BASE = 'https://fastapi-production-b73a.up.railway.app';` with the exact uncommitted localhost override `const SALES_API_BASE = 'http://127.0.0.1:8000';`. Serve `dashboard/` as static files, open the Sales Orders tab, and restore the committed constant before committing.
3. Prepare two open, physical sales-order lines for the same SKU and enough positive posted lot inventory to test both covered and competing demand. At least one lot should be eligible for a manual pin. Keep the order and line IDs available for API/database fixture setup.
4. For the auto-FIFO case, have two positive lots so the allocation can show one or more FIFO rows with `source=auto_fifo` and a future expiry.

## Test steps

1. Open an order that already has a mix of manual, lot-level, auto-FIFO, released, or shipped allocation history. If released and shipped rows are not already present, use an open physical line with at least 10 lb of posted stock and create them through the local API (replace the IDs, but do not change the localhost base or database):

   ```bash
   API_BASE=http://127.0.0.1:8000
   ORDER_ID=123
   LINE_ID=456

   RELEASE_JSON=$(curl -fsS -X POST "$API_BASE/sales/orders/$ORDER_ID/allocations" \
     -H 'X-API-Key: dashboard-key-2026' -H 'Content-Type: application/json' \
     -d "{\"mode\":\"manual\",\"line_id\":$LINE_ID,\"quantity_lb\":5,\"note\":\"PR6 released fixture\"}")
   RELEASE_ID=$(printf '%s' "$RELEASE_JSON" | jq -r '.allocations[0].id')
   curl -fsS -X POST "$API_BASE/sales/orders/$ORDER_ID/allocations/$RELEASE_ID/release" \
     -H 'X-API-Key: dashboard-key-2026'

   curl -fsS -X POST "$API_BASE/sales/orders/$ORDER_ID/allocations" \
     -H 'X-API-Key: dashboard-key-2026' -H 'Content-Type: application/json' \
     -d "{\"mode\":\"manual\",\"line_id\":$LINE_ID,\"quantity_lb\":5,\"note\":\"PR6 shipped fixture\"}"
   curl -fsS -X POST "$API_BASE/sales/orders/$ORDER_ID/ship/commit" \
     -H 'X-API-Key: dashboard-key-2026' -H 'Content-Type: application/json' \
     -d "{\"ship_all\":false,\"lines\":[{\"line_id\":$LINE_ID,\"quantity_lb\":5}]}"
   ```

   Confirm both states directly in the permitted test database:

   ```bash
   psql postgresql://localhost:5432/factory_ledger_test -v order_id="$ORDER_ID" -c \
     "SELECT id, status, source, quantity_lb, release_reason, ship_transaction_id FROM sales_order_allocations WHERE sales_order_id = :'order_id' AND status IN ('released', 'shipped') ORDER BY id;"
   ```

   Expected: the header shows computed Dispatch Ready or Blocked state and blocker chips with visibly distinct block, warning, and information treatments. Each physical line shows Ordered, Shipped Effective, Remaining, Allocated, and Shortage. Service lines show Service rather than false weight readiness, while a cancelled line always shows Cancelled even if its readiness has no ordered pounds. The reservation table identifies Manual, Staged lot, or Auto FIFO source; distinguishes SKU-level from a named lot; and offers Release only for a live allocation. The page states that reservations do not move physical inventory and that Dispatch Ready is not a shipping gate.

2. On an open physical line with coverable demand, select `Manual · SKU level`, enter a quantity within both effective remaining need and `coverable_lb`, and click Allocate.

   Expected: the request succeeds, a live SKU-level Manual row appears, Allocated increases, Remaining Effective does not change, and blocker chips update. No text claims that stock moved or shipped.

3. On an open physical line with a positive eligible lot, select `Manual · specific lot`, choose the lot, enter a coverable quantity, and click Allocate.

   Expected: the request succeeds and the reservation table shows the lot code and a lot-level source (`Manual` or `Staged lot` when the API classifies a STAGED/found-inventory lot). Physical on-hand is unchanged. If `/inventory/current?limit=500` returns exactly 500 inventory rows, the picker visibly says the lot list may be incomplete; if the selected product has no match in those loaded rows, it must not say `No positive on-hand lots found`.

4. Submit a manual quantity greater than the line's returned `coverable_lb` or effective remaining need.

   Expected: the API returns structured 409 `OVER_ALLOCATION`, and the UI shows an actionable sentence such as “Only N lb is coverable. Reduce the request to N lb or release a competing reservation.” Raw JSON, the HTTP response body, and an undefined value are not displayed. No reservation row is added.

5. Click Release on a live reservation and confirm the prompt.

   Expected: the row becomes released (or is reloaded in released history), its Release button disappears, Allocated and blockers recompute, and the confirmation copy explains that release makes pounds available to other orders without moving physical inventory.

6. Select `Auto FIFO · 48h TTL` on a line with unallocated need and click Allocate. Leave the prefilled quantity to allocate that need, or enter a smaller coverable quantity.

   Expected: one or more lot-level Auto FIFO rows appear in FIFO order. Every Auto FIFO row shows a live countdown such as `1d 23h remaining`; the countdown refreshes on the dashboard's **60-second tick** without a page reload. An elapsed allocation displays Expired and cannot be released as though it were live. Manual and staged rows display No expiry.

7. Return to the orders list and select `Dispatch Queue`.

   Expected: the dashboard calls the read-only fulfillment-check route, shows a ready/blocked count, and allows All Dispatch States, Dispatch Ready, or Blocked filtering. Dispatch-ready orders sort before blocked orders, then by ship date. Blocker reasons are visible as severity chips, including block reasons such as shortage/unallocated/not-floor-ready and warning reasons such as no-ship-date/inbound-cover. A fulfillment-diverged shipped or invoiced order remains visible when returned by the API, but its Dispatch and Blockers columns both show `—` instead of pills.

8. While still in `Dispatch Queue`, inspect the Factory Ready checkbox and expand an order to inspect its Factory Ready note drawer.

   Expected: the checkbox, note input, and note Save button are disabled. Hovering the checkbox or note drawer shows the tooltip `Toggle Factory Ready from All Open Orders`. Interacting with these controls sends no `POST /sales-orders/{so_number}/ready` request. Switch to `All Open Orders` before changing Factory Ready or its floor note.

9. With `ALLOCATIONS_ENFORCED=false`, open an eligible order and click Preview all remaining lines.

   Expected: requested pounds, `can_ship_lb`, `reserved_others_lb`, and owning order numbers render when supplied. There is no allocation-warning block anywhere in the preview because PR 5 V-2 intentionally omits `allocation_warning` while the flag is off. There must be no empty box, `undefined`, blank warning heading, or other warning artifact.

10. With `ALLOCATIONS_ENFORCED=true`, create a scenario where another order reserves pounds for the same SKU (`reserved_others_lb > 0`), then run the same preview.

   Expected: the returned `allocation_warning` renders as an amber warning block with its structured warning code/message and the competing reservation details remain visible. This is a preview only and does not ship. If no authorized flag-on staging/local server is available, mark this step **deferred until staging**; do not mock the warning and do not run it against production.

11. Reload the page and reopen the tested order.

    Expected: allocation history, current computed readiness, effective remaining pounds, blockers, and TTL state all come from the API and remain consistent after reload. The browser console has no uncaught exception, and all dashboard requests use `X-API-Key: dashboard-key-2026` rather than the master key.

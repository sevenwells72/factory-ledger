# 00 — Screen Inventory (Dashboard)

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) (v1.0, 178 rules)
**Status:** Inventory only. **No audit findings, no scoring, no evaluation.** No application code was modified.

## How to read this

Every user-facing screen, view, tab, dialog, form, and overlay in the dashboard is listed once, grouped by the
surface it lives in. Rows are numbered `S-nn` so later audit files can cite a stable ID.

Column definitions:

| Column | Meaning |
|---|---|
| **Screen** | The name a user or the code would call it. Rendered-in-JS views are named for their render function's output, not the function. |
| **File(s)** | Markup file, plus the JS render/bind site where the view is produced dynamically. |
| **Purpose** | What the screen is for, in one line. |
| **Primary user** | Who it is built for — **Luz** (office/admin), **Arturo** (floor lead), **Blubber** (owner). See "Role assignment" below. |
| **Primary action** | The single most likely thing a user does here. `Read-only` where the screen commits nothing. |
| **Mobile** | Whether the screen is used/usable at phone widths. **Yes** = has explicit compact rules and no fixed-width blocker. **Partial** = renders but has a known width or density constraint. **No** = fixed desktop layout with no compact path. |

### Role assignment — basis and caveat

Role attribution is **inferred**, not documented. The repo has no written statement of who uses which dashboard
screen. It is derived from: (a) the persona examples in the standards doc (`FL-Design-Standards-MASTER.md`
NOTIFY-012: "Arturo: receipts/pickups; Luz: receiving and expected receipts; Blubber: exceptions and
escalations"), (b) `docs/designs/045-write-foundation-design.md`, which defines the dashboard actor enum as
`meir|luz|arturo` and the Supplies requester picker as Arturo / Luz / MG / Other, and (c) what each screen
actually does. **Open question for the audit:** the standards call the owner *Blubber*, while design doc 045
calls the same seat *Meir/MG*; whether these are one persona is unresolved in the repo. Treat every
"Primary user" cell as a proposal to confirm with the user, not a finding.

### Mobile support — basis

`dashboard.css` carries compact rules at `max-width: 768px` (L121, L2181, L2426) and `max-width: 480px`
(L318, L382); `mini-calendar.css` at 1050/768/520px; `sankey.html`, `process-flow.html`, and
`traceability.html` each carry a 768px nav breakpoint. The scheduler board has **no** responsive breakpoint —
only `@media print` (L185) — and a fixed `240px / 1fr / 340px` main grid (L54). Where a screen is marked
**Partial**, the constraint is named in the row.

---

## A. Shared chrome (present on every top-level page)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | `index.html:12-25`, `sankey.html`, `process-flow.html`, `traceability.html`; `dashboard.css:100-130` | Move between the four top-level pages (Dashboard · Material Flow · Production Lines · Traceability) | All | Navigate to a page | Yes — collapses to hamburger at ≤768px |
| S-02 | Mini-calendar strip (3-month) | `mini-calendar.js`, `mini-calendar.css`; mounted via `[data-mini-calendar]` in all four page headers | Show the current month ±1 with a dot on days that have open SOs shipping | Luz | Read ship-date load; page months | Partial — shrinks at 1050/768/520px; day cells are display-only |

---

## B. Factory Dashboard — page chrome (`index.html`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-03 | App header | `index.html:27-44`; `dashboard.css:131-160` | Page title, global search, health badge, theme toggle, manual refresh, last-refreshed timestamp | Luz | Refresh data / toggle theme | Yes — wraps to 3 rows at ≤768px |
| S-04 | Global search field + results dropdown | `index.html:32-35`; `dashboard.js:1507-1521` (`performSearch`), `1522-1615` (`renderSearchResults`) | One search across products, lots, sales orders, and customers | Luz | Open a matched lot / product / SO / customer | Yes — search moves to its own full-width row at ≤768px |
| S-05 | Tab bar (7 tabs) | `index.html:47-55`; `dashboard.js:494-510` (`initTabs`); `dashboard.css:250-263` | Switch between the seven dashboard sections | Luz | Switch section | Partial — 7 tabs exceed phone width; the bar scrolls horizontally (`overflow-x:auto`) |

---

## C. Tab 1 — Operations (`#tab-operations`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-06 | Today So Far tile | `index.html:60-70`; `dashboard.js:238-300` (`renderTodayTile`, `refreshTodayTile`) | Made / Packed counts for the current plant date, by product family | Blubber | Read today's output at a glance | Yes — 2-col grid → 1-col at ≤480px |
| S-07 | Today So Far — error/retry state | `dashboard.js:290-299` | Explain a failed load and offer Retry | Blubber | Retry the load | Yes |
| S-08 | Production Calendar (rolling 5-day / month) | `index.html:72-86`; `dashboard.js:644-704` (params/label), `705-803` (`renderProductionCalendar`) | Batches and finished goods produced per day, over a rolling window or a month | Blubber | Select a day to expand it | Partial — grid drops to 2 cols (rolling) / 3 cols (month) at ≤768px |
| S-09 | Production Calendar — day detail panel | `index.html:83`; `dashboard.js:804-903` (`renderMadeDetailRows`, `renderPackedDetailRows`, `renderProductionDetailFamily`, `updateProductionDaySelection`) | Per-family made/packed breakdown for the selected day | Blubber | Read the day's production detail | Partial — inherits the calendar's compact grid; no dedicated rules |
| S-10 | Finished Goods On-Hand — collapsible panels | `index.html:88-93`; `dashboard.js:904-983` (`renderFinishedGoodsPanels`) | On-hand finished goods by pack family, with case and pallet counts | Luz | Expand a family panel | Partial — `.inv-table` has no horizontal-scroll wrapper |
| S-11 | Finished Goods — per-product lot breakdown (expandable row) | `dashboard.js:930-983`; `bindExpandableRows` `1624-1639` | Lot-level on-hand under a product row | Luz | Open a lot | Partial — same table constraint |
| S-12 | Batch Inventory On-Hand | `index.html:95-100`; `dashboard.js:984-1044` (`estimatedBatchesOnHand`, `renderBatchFamilyTable`), `1045-1089` (`renderBatchInventory`) | Bulk/WIP batch inventory with estimated batches on hand | Blubber | Read WIP position | Partial — same table constraint |
| S-13 | On-Hand Ingredients | `index.html:102-107`; `dashboard.js:1090-1147` (`renderIngredients`) | Ingredient stock with lot detail and a "show more" overflow cut | Luz | Check an ingredient's stock | Partial — same table constraint |

---

## D. Tab 2 — Recent Entries (`#tab-recent`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-14 | Recent Entries feed | `index.html:111-124`; `dashboard.js:568-625` (`renderRecentEntries`), `626-643` (polling) | Newest-first audit feed of ledger transactions and corrections, with occurred-vs-entered timing flags | Blubber | Read what was posted and when | Yes — dedicated ≤480px card rules (`dashboard.css:318-327`) |
| S-15 | Recent Entries — loading / empty / error states | `index.html:120-121`; `dashboard.js:569-572`; `dashboard.css:310-317` | Cover the three non-data states, with a retry control on error | Blubber | Retry | Yes |

---

## E. Tab 3 — Activity (`#tab-activity`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-16 | Daily Entries (collapsible) + day/mode toolbar | `index.html:128-146`; `dashboard.js:1279-1302` (`dailyEntriesDate`), `1309-1371` (`renderDailyEntries`) | All ledger entries for one day, switchable between event date and entry date | Luz | Pick a day and read its entries | Partial — table has no scroll wrapper |
| S-17 | Shipping log (collapsible) | `index.html:148-158`; `dashboard.js:1185-1237` (`renderShipments`) | Posted shipments with customer, SO, lot, quantity | Luz | Verify a shipment posted | Partial — same |
| S-18 | Receiving log (collapsible) | `index.html:160-171`; `dashboard.js:1238-1278` (`renderReceipts`) | Posted receipts with supplier, lot, quantity | Luz | Verify a receipt posted | Partial — same |

---

## F. Tab 4 — Notes / To-Dos / Reminders (`#tab-notes`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-19 | Notes toolbar (category filters + show-completed + New) | `index.html:178-191` | Filter the list to All / Notes / To-Dos / Reminders and create a new item | Luz | Filter, or start a new item | Yes — stacks at ≤768px |
| S-20 | Notes list (cards) | `index.html:193`; `dashboard.js:1676-1784` (`renderNotes`) | Show notes/to-dos/reminders with priority, due date, and pinned entity | Luz | Tick an item done | Yes — row actions forced visible at ≤768px |
| S-21 | Notes — empty state | `dashboard.js:1677-1684` | Tell the user the list is empty and how to add | Luz | Create the first item | Yes |
| S-22 | **Dialog** — Note create/edit modal | `index.html:446-506`; `dashboard.js:1785-1848` (`openNoteModal`, `closeNoteModal`, save) | Create or edit a note/to-do/reminder: category, title, details, priority, due date, pinned entity | Luz | Save the item | Yes — modal goes full-bleed, `.form-row` → 1 col at ≤768px |
| S-23 | **Dialog** — Delete note confirmation | `dashboard.js:1771` (native `confirm`) | Confirm destruction of a note | Luz | Confirm or cancel deletion | Yes (browser-native) |

---

## G. Tab 5 — Sales Orders (`#tab-orders`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-24 | Orders toolbar (status filter, dispatch filter, customer search, overdue / hide-ready toggles, exports, refresh) | `index.html:203-237`; `dashboard.js:2031-2074` (`getFilteredOrders`), `2276-2290` (`updateDispatchQueueControls`) | Narrow the order list and export it as CSV or an Excel matrix | Luz | Filter to the orders that need work | Yes — filters stack, inputs go full-width at ≤768px |
| S-25 | Orders list table | `index.html:239-243`; `dashboard.js:2300-2353` (`renderOrdersList`), readiness helpers `1927-2029`, `2229-2275` | All sales orders with status, ship-by, readiness pill, and blockers | Luz | Open an order | Partial — wide table, `#orders-table-container` has no horizontal-scroll wrapper |
| S-26 | Orders list — Dispatch Queue mode | `dashboard.js:1967-1970` (`isDispatchQueueMode`), `2276-2290`, `1938-1966` (blocker/dispatch chips) | Alternate list mode showing only dispatch-ready vs blocked orders, with a summary | Luz | Find the next shippable order | Partial — same |
| S-27 | Orders list — expandable line rows | `dashboard.js:2354-2417` (`renderOrderLinesContent`), `2490-2578` (`bindOrderExpandToggles`) | Show an order's lines inline without leaving the list | Luz | Inspect lines in place | Partial — same |
| S-28 | Orders list — Factory Ready toggle + note | `dashboard.js:2418-2426` (`updateCachedOrderReady`), `2427-2451` (`bindOrderReadyNoteControls`), `2452-2489` (`bindOrderReadyToggles`) | Mark an order factory-ready from the list and attach a note | Luz | Flag an order ready | Partial — same |
| S-29 | Orders list — empty state | `dashboard.js` (orders-empty block in `renderOrdersList`) | Say no orders match the current filters | Luz | Relax a filter | Yes |
| S-30 | Order Detail view — header, status, dates, KPI row | `index.html:246-255`; `dashboard.js:3180-3307` (`renderOrderDetail`), `2893-2901` (`renderOrderReadinessSummary`) | Everything about one sales order: identity, status, ship-by, ordered/shipped/remaining/pallets | Luz | Read the order's position | Yes — KPI grid → 2 cols, dates stack at ≤768px |
| S-31 | Order Detail — line readiness table | `dashboard.js:3234-3286` | Per-line ordered / shipped / remaining / allocated / shortage / blockers / status | Luz | Spot the line that is short | Yes — wrapped in `.order-detail-table-wrap` with `overflow-x:auto` |
| S-32 | Order Detail — per-line inventory expander | `dashboard.js:2579-2610` (`flattenFinishedGoodsInventory`), `2611-2637` (`renderOrderInventoryContent`), `2638-2663` (`bindOrderInventoryToggles`) | Show on-hand inventory available for one order line | Luz | Check stock against a line | Yes |
| S-33 | **Form** — Order Detail edit mode (header + lines + notes) | `dashboard.js:2664-2667` (`canEditOrderHeader`), `2701-2713` (`renderOrderEditActions`), `2714-2841` (save header / save lines), `2842-2864` (`bindOrderDetailEditControls`) | Edit ship-by date, notes, line quantities and prices; change order status | Luz | Save header / save lines | Yes — inputs inherit the stacked detail layout |
| S-34 | Order Detail — edit-locked notice | `dashboard.js:2702-2704` | Explain that editing is only open while the order is New or Confirmed | Luz | Read-only | Yes |
| S-35 | **Dialog** — Order status change confirmation | `dashboard.js:2819` (native `confirm`) | Confirm a status transition before it is written | Luz | Confirm or cancel | Yes (browser-native) |
| S-36 | **Form** — Reservations / allocation section | `dashboard.js:2909-2960` (`renderAllocationSection`), `3031-3114` (`updateAllocationForm`), `3159-3179` (`bindAllocationControls`) | Reserve finished goods against a line: SKU-level, specific lot, or auto-FIFO with TTL | Luz | Allocate stock to a line | Yes — allocation form → 1 col at ≤768px |
| S-37 | Reservations — allocation history table | `dashboard.js:2936-2957`; expiry countdown `2879-2892`, `3153-3158` | Existing reservations with level, quantity, source, status/TTL, and Release | Luz | Release a reservation | Yes — `.allocation-table-wrap` scrolls horizontally |
| S-38 | **Dialog** — Release reservation confirmation | `dashboard.js:3096` (native `confirm`) | Confirm releasing reserved stock back to the pool | Luz | Confirm or cancel | Yes (browser-native) |
| S-39 | Shipping capacity preview | `dashboard.js:2961-2965` (`renderShippingPreviewSection`), `3115-3152` (`renderShippingPreview`) | Read-only preview of what is takeable now for the remaining lines | Luz | Run the preview | Yes |
| S-40 | Order Detail — notes card | `dashboard.js:3288-3299` | Show or edit free-text order notes | Luz | Read / edit notes | Yes |

---

## H. Tab 6 — Expected Receipts (`#tab-expected`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-41 | Expected Receipts toolbar (status filter, text filter, overdue toggle, summary, New, Refresh) | `index.html:259-277`; `dashboard.js:3385-3396` (`getFilteredExpectedReceipts`) | Narrow the incoming-delivery list and start a new expected receipt | Luz | Filter to open/overdue deliveries | Yes — shares the orders-toolbar stacking rules |
| S-42 | Expected Receipts table | `index.html:278-285`; `dashboard.js:3397-3406` (`erStatusBadge`), `3407-3508` (`renderExpectedReceipts`) | Incoming deliveries with expected / received / remaining, date, reference, status | Luz | Reconcile what has arrived | Partial — 9-column table, `#er-table-container` has no scroll wrapper |
| S-43 | Expected Receipts — row actions (Edit / Close / Cancel) | `dashboard.js:3438-3450`, `erSetStatus` | Edit, close, or cancel one expected receipt from its row | Luz | Close a fulfilled receipt | Partial — three buttons in one row cell |
| S-44 | Expected Receipts — empty state | `dashboard.js:3416-3421` | Say nothing matches the current filters | Luz | Relax a filter | Yes |
| S-45 | **Dialog / Form** — Expected Receipt create/edit modal | `index.html:398-444`; `dashboard.js:3509-3572` (`setErProduct`, product type-ahead), `3573-3627` (`closeErModal`, save) | Create or edit an expected delivery: product (type-ahead), supplier, qty, date, reference, notes | Luz | Save the expected receipt | Yes — modal full-bleed, `.form-row` → 1 col at ≤768px |

---

## I. Tab 7 — Supplies (`#tab-supplies`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-46 | Supplies page header + Request Supply | `index.html:290-296` | Introduce the tab and open the request form | Arturo | Open the request form | Yes — header left-aligns at ≤768px |
| S-47 | Supplies inventory sub-tabs (Ingredients / Packaging / All) with low-stock count badges | `index.html:300-310`; `dashboard.js:3656-3666` (`supplyItemsForTab`), `3667-3680` (`updateSupplyLowStockBadges`) | Switch the inventory table between supply categories, showing low-stock counts | Luz | Switch category | Yes — sub-tabs scroll horizontally at ≤768px |
| S-48 | Supplies search field | `index.html:311-314` | Filter the inventory table by product name | Luz | Find a product | Yes — goes full width at ≤768px |
| S-49 | Supplies inventory table | `index.html:316`; `dashboard.js:3827-3871` (`renderSuppliesInventory`) | On-hand, unit, incoming, and low-stock status per supply item | Luz | Spot a low-stock item | Yes — `.supplies-table-scroll` scrolls; `min-width:700px` |
| S-50 | Supplies — expanded lot / incoming detail row | `dashboard.js:3729-3739` (lot sort), `3740-3751` (field list, date fmt), `3752-3773` (`renderSupplyExpectedReceiptDetail`), `3774-3815` (`renderSupplyLotDetail`), `3816-3826` (`toggleSupplyProduct`) | FIFO lot detail and linked incoming expected receipts for one supply item | Luz | Read FIFO lot position | Yes — detail fields → 1 col at ≤768px |
| S-51 | Supply Requests list | `index.html:320-335`; `dashboard.js:3872-3919` (time fmt, error, feedback), `3920-3953` (`renderSupplyRequests`) | Open and completed supply requests from the floor, with a Done action | Luz | Mark a request done | Yes — `.supplies-table-scroll` |
| S-52 | Supply Requests — feedback / empty states | `index.html:330`; `dashboard.js:3893-3919`, `3929-3932` | Confirm a save, or say there are no requests | Luz | Read confirmation | Yes |
| S-53 | **Dialog / Form** — Request Supply modal | `index.html:337-396` (product select, unlisted item + unit, qty + unit, note, requested-by, Other name); `dashboard.js:3681-3728` (`populateSupplyProductSelector`), `3974-3987` / `3988-3992` (conditional fields), `3993-4004` / `4005-4077` (open/close/submit) | Let the floor request a supply, including items not in the catalogue | **Arturo** | Submit the request | Yes — modal full-bleed, `.form-row` → 1 col at ≤768px |

---

## J. Overlays shared across tabs (`index.html`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-54 | Lot Detail side panel | `index.html:508-516`; `dashboard.js:1394-1399` (`fmtQtyCases`), `1400-1433` (`renderLotPanel`), `1434-1436` (`closeLotPanel`), `1640-1675` (`bindLotLinks`) | Lot identity, original vs on-hand quantity, and full transaction timeline | Arturo | Trace what happened to a lot | Yes — panel goes 100vw at ≤768px |
| S-55 | Lot disambiguation view (same lot code, multiple products) | `dashboard.js:1372-1393` (`renderLotDisambiguation`) | Ask which product's lot was meant when a code matches more than one | Arturo | Pick the right lot | Yes — inside the lot panel |
| S-56 | Product Detail panel (active + depleted lots) | `dashboard.js:1439-1506` (`openProductPanel`) | All lots for one product, split active vs depleted, with totals | Luz | Open a specific lot | Yes — reuses the lot panel shell |

---

## K. Material Flow — Sankey (`sankey.html`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-57 | Sankey controls bar (period, custom range, top-N products / customers / ingredients) | `sankey.html` (controls block) | Scope the flow diagram by date range and node counts | Blubber | Change the period | Partial — only the nav has a 768px rule; the control row itself has none |
| S-58 | Material-flow Sankey chart + legend + column labels | `sankey.html` (chart container, D3 render) | Show material moving Ingredients → Production Lines → Finished Goods → Customers | Blubber | Read where volume goes | Partial — wide 4-column diagram, no compact layout |
| S-59 | Sankey banner + loading overlay | `sankey.html:197`, `764-778` | Report staleness/errors and cover the loading wait | Blubber | Read status | Partial |

---

## L. Production Lines — Process Flow (`process-flow.html`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-60 | Summary strip (Lines Active · Produced Today · Avg Yield) | `process-flow.html` (summary-strip) | Three headline production numbers | Blubber | Read today's line status | Yes — breakpoints at 900px and 550px (`process-flow.html:161,165`) |
| S-61 | Production lines grid | `process-flow.html` (`#lines-grid`, JS render) | Per-line current state and yield | Blubber | Read a line's state | Yes — same breakpoints |
| S-62 | Error / stale banners + auto-refresh footer | `process-flow.html` (`#error-banner`, `#stale-banner`, `#footer`) | Warn that data failed or is stale; state the 60s refresh cadence | Blubber | Read status | Yes |

---

## M. Traceability (`traceability.html`)

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-63 | Lot search field + type-ahead dropdown | `traceability.html` (search-area, `#lotSearch`, `#searchDropdown`) | Find a lot by lot code, supplier lot, or product name | Arturo | Select a lot to trace | Partial — only the nav has a 768px rule; the search row does not stack |
| S-64 | Trace direction switch (Forward / Backward) + Trace button | `traceability.html` (`#btn-forward`, `#btn-backward`, `#traceBtn`) | Choose trace direction and run the trace | Arturo | Run the trace | Partial — same |
| S-65 | Recent lots strip | `traceability.html` (`#recentLots`) | Re-open a recently traced lot without retyping | Arturo | Re-trace a recent lot | Partial |
| S-66 | Trace legend | `traceability.html` (legend block) | Key for node types (Supplier, Ingredient Lot, Batch, Finished Goods, Customer, Data Gap) and edge types | Arturo | Read the key | Partial |
| S-67 | Status bar | `traceability.html` (`#statusBar`) | State what to do next, or what the last trace returned | Arturo | Read status | Partial |
| S-68 | Trace graph (D3) + zoom controls (Fit / + / − / Show full chain) | `traceability.html` (`#graphContainer`, `#graphControls`, `#graphSvg`) | The supply-chain graph for the traced lot | Arturo | Explore the chain | **No** — fixed 500px SVG, pan/zoom only; no compact layout |
| S-69 | Node tooltip | `traceability.html` (`#tooltip`) | Detail for the hovered graph node | Arturo | Read node detail | No — hover-dependent |
| S-70 | Trace detail panel + export buttons | `traceability.html` (`#detailPanel`, `#traceDetail`, export-btns) | Text detail for the selected node, with Print Audit Report and Download Text Report | Blubber | Export the audit report | Partial |
| S-71 | Print audit report view | `traceability.html:240` (`@media print`) | Print-formatted recall/audit report | Blubber | Print | N/A (print) |

---

## N. Production Board / Scheduler (`scheduler/seven-wells-production-board.html`)

Standalone planning tool: **not linked from the site nav**, **no API calls**, state held in `localStorage`
(`:376-378`) and fed by CSV import or sample data. Treat it as a separate surface with its own conventions.

| # | Screen | File(s) | Purpose | Primary user | Primary action | Mobile |
|---|---|---|---|---|---|---|
| S-72 | Topbar KPI strip (Everything done by · Orders at risk · Dominant bottleneck · Labor utilization · Late days) | `:3-9` | Five headline planning outcomes for the current plan | Blubber | Read whether the plan is feasible | **No** — fixed-width KPI columns, no breakpoint |
| S-73 | Topbar actions (Set as baseline · Print · Copy week) | `:11-13`, `:948` (Clear baseline) | Snapshot a plan, print it, or copy the week's schedule text | Blubber | Set a baseline | No |
| S-74 | Legend bar | `:15` | Key for board cell states | Blubber | Read the key | No |
| S-75 | Delta panel (vs baseline) | `:25`, `:948` | Show what changed against the saved baseline | Blubber | Compare to baseline | No |
| S-76 | **Form** — Left settings panel: plan horizon, labor, granola bake, coconut line, pack lines, repack | `:27`, `:956-992` | Every scheduling constraint the optimizer runs on | Blubber | Change a constraint and replan | No |
| S-77 | Left panel — Finished goods on hand (disclosure) | `:993`, `:1026` | Stock deducted from demand before scheduling | Blubber | Review/adjust on-hand | No |
| S-78 | Left panel — Bulk-bin WIP (disclosure) | `:994`, `:1036` | Baked granola packable without a bake step | Blubber | Review/adjust WIP | No |
| S-79 | **Form** — Left panel: Product catalog (disclosure) | `:995`, `:1051` | Edit pan yields, cases-per-pan, bake tier; flags seeded assumptions | Blubber | Correct a product assumption | No |
| S-80 | Left panel — Orders in / plans out (CSV import, sample orders, save/load scenario, export CSV, reset) | `:996-1004`, `:1077-1101` | Get orders in and plans out; scenario persistence | Blubber | Import the orders CSV | No |
| S-81 | Left panel — CSV import result message (imported / unknown SKUs / skipped rows) | `:1079-1083`, `:1101` | Report exactly what the import accepted and dropped | Blubber | Read what was skipped | No |
| S-82 | Left panel — "How this works" (disclosure) | `:1005` | Explain the scheduling model | Blubber | Read the explanation | No |
| S-83 | Production board table (days × stations) | `:29`, `:1135-1185`; styles `:102-126` | The schedule itself: per-day, per-station quantities, crew badges, pins, bottleneck dots | Blubber | Open a cell to pin it | **No** — sticky-header table, `min-width:124px` per day column |
| S-84 | Board cell — day copy control + "more" toggle | `:1135`, `:1173-1176` | Copy one day's schedule; expand a truncated item list | Blubber | Copy a day | No |
| S-85 | Schedule panel (side) — Copy / Print / Close | `:30-37` | Plain-text schedule for one day or the week, ready to send | Blubber | Copy the schedule text | No |
| S-86 | **Dialog / Form** — Pin modal (force quantity, crew, feasibility) | `:1435-1506` (`openPin`, feasibility, `closeModal`); styles `:172-182` | Lock a day/station to a quantity and crew, then replan around it | Blubber | Pin and replan | Partial — `#modal` is `width:400px; max-width:94vw` |
| S-87 | Right panel — Order book (included / excluded orders) | `:40`, `:1308-1339` | Every order line in the plan, with ready vs due dates and include/exclude/delete | Blubber | Exclude an order from the plan | No |
| S-88 | **Form** — Add order line | `:1518`, `:1555-1580` | Add a single order line by hand (Order ID, Customer, SKU, Qty, Due) | Blubber | Add the line | No |
| S-89 | **Dialog** — Add-order validation alert | `:1570` (native `alert`) | Block a save missing SKU, qty, or due date | Blubber | Acknowledge | Partial (browser-native) |
| S-90 | Order book — empty state | `:1339` | Tell the user to import a CSV or load sample orders | Blubber | Import orders | No |
| S-91 | Print view | `:185` (`@media print`) | Printable board and schedule | Blubber | Print | N/A (print) |

---

## Summary

| Surface | Screens | Mobile: Yes | Partial | No |
|---|---|---|---|---|
| Shared chrome | 2 | 1 | 1 | 0 |
| Dashboard page chrome | 3 | 2 | 1 | 0 |
| Tab 1 — Operations | 8 | 2 | 6 | 0 |
| Tab 2 — Recent Entries | 2 | 2 | 0 | 0 |
| Tab 3 — Activity | 3 | 0 | 3 | 0 |
| Tab 4 — Notes | 5 | 5 | 0 | 0 |
| Tab 5 — Sales Orders | 17 | 11 | 6 | 0 |
| Tab 6 — Expected Receipts | 5 | 3 | 2 | 0 |
| Tab 7 — Supplies | 8 | 8 | 0 | 0 |
| Shared overlays | 3 | 3 | 0 | 0 |
| Material Flow (Sankey) | 3 | 0 | 3 | 0 |
| Production Lines | 3 | 3 | 0 | 0 |
| Traceability | 9 | 0 | 6 | 2 (+1 print) |
| Production Board / Scheduler | 20 | 0 | 2 | 17 (+1 print) |
| **Total** | **91** | **40** | **30** | **19** (+2 print-only) |

**By type:** 7 top-level tabs · 3 sub-tab/segment groups (Supplies categories, calendar rolling/month, trace
direction) · 8 dialogs and modals (4 custom: Note, Expected Receipt, Supply Request, Pin; 4 browser-native:
delete note, SO status change, release reservation, add-order validation) · 10 forms · 2 side panels · 2
print views · the rest are lists, tables, charts, and states.

**By primary user (inferred):** Luz 32 · Blubber 39 · Arturo 8 · All 2 · N/A 2. Arturo's dashboard footprint is
narrow — Request Supply, lot lookup, and Traceability — consistent with the floor working mainly through the
ChatGPT Floor GPT rather than this dashboard. **Confirm this with the user before the audit weights any
mobile rule.**

## Notes carried forward to the audit (observations, not findings)

1. **No dedicated mobile surface exists.** Every dashboard screen is the desktop layout reflowed by
   `max-width` rules. `dashboard/` contains no phone-specific view, no bottom tab bar, and no
   Receive/Pack/Ship entry screens — those live in the ChatGPT Floor GPT, outside this directory.
2. **Six wide tables have no horizontal-scroll wrapper** (`#orders-table-container`, `#er-table-container`,
   `.inv-table` in Finished Goods / Batch / Ingredients, Daily Entries / Shipping / Receiving).
   Order Detail, Allocations, and Supplies tables do have one.
3. **The scheduler is disconnected** — no site-nav link, no API, `localStorage` only, no responsive
   breakpoint. Whether it is in scope as a "dashboard screen" for standards conformance is a scoping
   question for the user.
4. **Four browser-native `confirm`/`alert` dialogs** coexist with four custom modals.
5. **The role model is undocumented** — see "Role assignment" above. Every Primary-user cell needs
   confirmation.

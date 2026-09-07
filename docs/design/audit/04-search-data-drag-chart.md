# 04 — Audit: Search, Tables, Direct Manipulation & Charts (SEARCH, DATA, DRAG, CHART)

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) §8 (SEARCH-001…008), §9 (DATA-001…012), §13 (DRAG-001…011), §14 (CHART-001…008) — 39 rules
**Screens:** [00-screen-inventory.md](00-screen-inventory.md) — S-01…S-91
**Status:** Audit findings. **No application code was modified.**

---

## How to read this

Statuses and tiering are as defined in [01-nav-layout.md](01-nav-layout.md): **P** PASS · **PA** PARTIAL · **F** FAIL · **N** N/A · **U** UNVERIFIABLE-FROM-CODE · **·** out of tier scope.

### Rule importance (drives Tier B scope)

- **Critical:** SEARCH-003; DATA-004; DRAG-001 (non-drag alternative clause), 003, 006
- **High:** SEARCH-001, 002, 004; DATA-001, 003, 008; DRAG-001 (support clause), 002, 005, 008; CHART-001, 002, 004
- **Medium:** SEARCH-005, 007, 008; DATA-002, 005, 006, 007, 009, 010, 011, 012; DRAG-004, 007, 009; CHART-003, 005, 006, 007
- **Low/Contextual:** SEARCH-006; DRAG-010, 011; CHART-008

### There is no drag and drop in this product

Verified by grep across `dashboard/*.html`, `dashboard/*.js`, and `dashboard/scheduler/*.html`: no `draggable` attribute, no `dragstart` / `dragover` / `drop` handler, no `DataTransfer`, no sortable library, no pointer-based reorder.

Consequently **DRAG-002 through DRAG-011 are N/A across all 91 screens**, with the reason *"no drag-and-drop interaction exists in this product."* They are recorded once in the N/A register rather than repeated in a matrix.

**DRAG-001 has two separable clauses and is audited in full:**
- the **Critical hard-rule clause** — *"every drag operation is also achievable via a button, menu, or keyboard"* — is vacuously satisfied everywhere, since there is no drag operation to have an alternative to;
- the **High strong-recommendation clause** — *"support drag and drop where users will instinctively try it"* — is assessed against the two surfaces where a user reaches for drag: the scheduler production board and the allocation flow.

Because DRAG is effectively a single-rule section here, no separate DRAG matrix is produced; the per-screen result is in the findings.

### There is no column sorting in this product

Verified by grep: no `aria-sort`, no header click handler, no sort control on any table. This drives **DATA-008** to FAIL on every table screen — see the findings.

---

## Matrix A — Search & Discovery (SEARCH-001…008)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 |
|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | PA | N | N | N | N | N | N | N |
| S-02 | Mini-calendar strip | N | N | N | N | N | N | N | N |
| S-03 | App header | **P** | N | N | N | N | N | N | **F** |
| S-04 | Global search + results | **P** | **F** | PA | PA | PA | N | N | **F** |
| S-05 | Tab bar (7 tabs) | N | N | N | N | N | N | N | **F** |
| S-06 | Today So Far tile | P | N | N | N | N | N | N | N |
| S-07 | Today So Far — error/retry | N | N | N | N | N | N | N | N |
| S-08 | Production Calendar | P | N | N | N | PA | N | N | **F** |
| S-09 | Calendar day detail panel | P | N | N | N | N | N | N | **F** |
| S-10 | Finished Goods panels | P | N | N | N | N | N | PA | **F** |
| S-11 | FG per-product lot rows | P | N | N | N | N | N | PA | **F** |
| S-12 | Batch Inventory | P | N | N | N | N | N | PA | **F** |
| S-13 | On-Hand Ingredients | P | N | N | N | N | N | PA | **F** |
| S-14 | Recent Entries feed | P | N | N | N | N | N | **F** | **F** |
| S-15 | Recent Entries states | N | N | N | N | N | N | N | N |
| S-16 | Daily Entries + toolbar | P | N | N | N | PA | N | PA | **F** |
| S-17 | Shipping log | P | N | N | N | N | N | **F** | **F** |
| S-18 | Receiving log | P | N | N | N | N | N | **F** | **F** |
| S-19 | Notes toolbar | P | N | N | N | P | N | N | **F** |
| S-20 | Notes list (cards) | P | N | N | N | P | N | PA | **F** |
| S-21 | Notes empty state | N | N | N | N | N | N | N | N |
| S-22 | Note create/edit modal | N | N | N | **F** | N | N | N | N |
| S-23 | Delete-note confirm | N | N | N | N | N | N | N | N |
| S-24 | Orders toolbar | P | **P** | **P** | PA | PA | N | PA | **F** |
| S-25 | Orders list table | P | N | N | N | PA | N | PA | **F** |
| S-26 | Dispatch Queue mode | P | N | N | N | PA | N | PA | **F** |
| S-27 | Orders expandable lines | P | N | N | N | N | N | PA | **F** |
| S-28 | Factory Ready toggle+note | P | N | N | N | N | N | N | **F** |
| S-29 | Orders empty state | N | N | N | N | N | N | N | N |
| S-30 | Order Detail header/KPI | P | N | N | N | N | N | N | **F** |
| S-31 | Order Detail line table | P | N | N | N | N | N | PA | **F** |
| S-32 | Per-line inventory expander | P | N | N | N | N | N | N | N |
| S-33 | Order Detail edit mode | P | N | N | N | N | N | N | **F** |
| S-34 | Edit-locked notice | N | N | N | N | N | N | N | N |
| S-35 | SO status change confirm | N | N | N | N | N | N | N | N |
| S-36 | Reservations / allocation | P | N | **P** | PA | N | N | N | **F** |
| S-37 | Allocation history table | P | N | N | N | N | N | PA | **F** |
| S-38 | Release reservation confirm | N | N | N | N | N | N | N | N |
| S-39 | Shipping capacity preview | P | N | N | N | N | N | N | N |
| S-40 | Order Detail notes card | P | N | N | N | N | N | N | **F** |
| S-41 | Expected Receipts toolbar | P | **P** | **P** | PA | **P** | N | **P** | **F** |
| S-42 | Expected Receipts table | P | N | N | N | P | N | **P** | **F** |
| S-43 | ER row actions | P | N | N | N | N | N | N | **F** |
| S-44 | ER empty state | N | N | N | N | N | N | N | N |
| S-45 | ER create/edit modal | N | N | PA | PA | N | N | N | N |
| S-46 | Supplies header + Request | P | N | N | N | N | N | N | **F** |
| S-47 | Supplies sub-tabs | P | N | PA | N | P | N | PA | **F** |
| S-48 | Supplies search field | P | **P** | **F** | PA | P | N | PA | **F** |
| S-49 | Supplies inventory table | P | N | N | N | P | N | PA | **F** |
| S-50 | Supply lot / incoming detail | P | N | N | N | N | N | N | **F** |
| S-51 | Supply Requests list | P | N | N | N | **F** | N | **F** | **F** |
| S-52 | Supply Requests feedback | N | N | N | N | N | N | N | N |
| S-53 | Request Supply modal | N | N | PA | PA | N | N | N | N |
| S-54 | Lot Detail side panel | P | N | N | N | N | N | N | **F** |
| S-55 | Lot disambiguation | P | N | N | N | N | N | N | **F** |
| S-56 | Product Detail panel | P | N | N | N | N | N | PA | **F** |
| S-57 | Sankey controls bar | **F** | N | N | N | **P** | N | N | · |
| S-58 | Sankey chart + legend | **F** | N | N | N | N | N | N | · |
| S-59 | Sankey banner / loading | **F** | N | N | N | N | N | N | · |
| S-60 | Summary strip | **F** | N | N | N | N | N | N | · |
| S-61 | Production lines grid | **F** | N | N | N | N | N | N | · |
| S-62 | Error / stale banners | **F** | N | N | N | N | N | N | · |
| S-63 | Lot search + type-ahead | **F** | **F** | **F** | **P** | **F** | PA | N | **P** |
| S-64 | Trace direction + Trace | **F** | N | N | N | **F** | N | N | **P** |
| S-65 | Recent lots strip | **F** | N | N | **P** | N | PA | N | **P** |
| S-66 | Trace legend | N | N | N | N | N | N | N | N |
| S-67 | Status bar | N | N | N | **P** | N | N | N | N |
| S-68 | Trace graph + zoom | **F** | N | N | N | N | N | PA | **P** |
| S-69 | Node tooltip | N | N | N | N | N | N | N | N |
| S-70 | Trace detail + exports | **F** | N | N | N | N | N | P | **P** |
| S-71 | Print audit report | N | N | N | N | N | N | N | · |
| S-72 | Topbar KPI strip | N | N | N | N | N | N | N | · |
| S-73 | Topbar actions | N | N | N | N | N | N | N | · |
| S-74 | Legend bar | N | N | N | N | N | N | N | · |
| S-75 | Delta panel | N | N | N | N | N | N | N | · |
| S-76 | Settings panel (form) | N | N | N | N | N | N | N | · |
| S-77 | FG on hand (disclosure) | N | N | N | N | N | N | N | · |
| S-78 | Bulk-bin WIP (disclosure) | N | N | N | N | N | N | N | · |
| S-79 | Product catalog (form) | N | N | N | **P** | N | N | N | · |
| S-80 | Orders in / plans out | N | N | N | N | N | N | N | · |
| S-81 | CSV import result | N | N | N | N | N | N | N | · |
| S-82 | "How this works" | N | N | N | N | N | N | N | · |
| S-83 | Production board table | N | N | N | N | N | N | N | · |
| S-84 | Board cell copy / more | N | N | N | N | N | N | N | · |
| S-85 | Schedule panel | N | N | N | N | N | N | N | · |
| S-86 | Pin modal | N | N | N | N | N | N | N | · |
| S-87 | Order book | N | N | N | N | PA | N | N | · |
| S-88 | Add order line form | N | N | N | **P** | N | N | N | · |
| S-89 | Add-order validation alert | N | N | N | N | N | N | N | · |
| S-90 | Order book empty state | N | N | N | N | N | N | N | · |
| S-91 | Scheduler print view | N | N | N | N | N | N | N | · |

## Matrix B — Lists, Tables & Dense Operational Data (DATA-001…012)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | N | N | N | N | N | N | N | N | N | N | N | N |
| S-02 | Mini-calendar strip | N | P | N | N | N | N | N | N | N | N | N | PA |
| S-03 | App header | N | N | N | N | N | N | N | N | N | N | N | **F** |
| S-04 | Global search + results | P | P | P | PA | PA | PA | PA | N | N | N | N | N |
| S-05 | Tab bar (7 tabs) | N | N | N | N | N | N | N | N | N | N | N | N |
| S-06 | Today So Far tile | P | P | P | P | PA | N | P | N | N | N | N | P |
| S-07 | Today So Far — error/retry | N | P | N | N | N | N | N | N | N | N | N | N |
| S-08 | Production Calendar | P | **P** | P | P | PA | N | P | **F** | N | PA | N | **F** |
| S-09 | Calendar day detail panel | P | P | P | P | PA | N | P | N | N | PA | N | **F** |
| S-10 | Finished Goods panels | P | P | P | P | **P** | **F** | **F** | **F** | **F** | PA | N | N |
| S-11 | FG per-product lot rows | P | P | P | P | PA | **F** | **F** | **F** | **F** | PA | N | N |
| S-12 | Batch Inventory | P | P | P | P | **P** | **F** | **F** | **F** | **F** | PA | N | N |
| S-13 | On-Hand Ingredients | P | P | P | P | **P** | **F** | **F** | **F** | **F** | PA | N | N |
| S-14 | Recent Entries feed | PA | P | **F** | P | PA | N | P | **F** | N | N | N | **P** |
| S-15 | Recent Entries states | N | P | N | N | N | N | N | N | N | N | N | N |
| S-16 | Daily Entries + toolbar | P | P | PA | P | PA | PA | PA | **F** | **F** | N | N | **F** |
| S-17 | Shipping log | P | P | PA | P | P | **F** | **F** | **F** | **F** | PA | N | **F** |
| S-18 | Receiving log | P | P | PA | P | P | **F** | **F** | **F** | **F** | PA | N | **F** |
| S-19 | Notes toolbar | N | P | N | N | **F** | N | N | N | N | N | N | N |
| S-20 | Notes list (cards) | P | P | P | P | **F** | N | P | **F** | N | N | N | **F** |
| S-21 | Notes empty state | N | P | N | N | N | N | N | N | N | N | N | N |
| S-22 | Note create/edit modal | N | P | N | N | N | N | N | N | N | N | N | N |
| S-23 | Delete-note confirm | N | N | N | N | N | N | N | N | N | N | N | N |
| S-24 | Orders toolbar | N | P | N | N | N | N | N | N | N | N | N | N |
| S-25 | Orders list table | P | P | **F** | PA | PA | PA | PA | **F** | **F** | PA | N | **F** |
| S-26 | Dispatch Queue mode | P | P | **F** | PA | PA | PA | PA | PA | **F** | PA | N | **F** |
| S-27 | Orders expandable lines | P | P | PA | PA | PA | **F** | PA | **F** | **F** | **P** | N | **F** |
| S-28 | Factory Ready toggle+note | P | P | P | P | N | N | P | N | N | N | N | N |
| S-29 | Orders empty state | N | P | N | N | N | N | N | N | N | N | N | N |
| S-30 | Order Detail header/KPI | P | P | P | P | P | N | P | N | N | N | N | **F** |
| S-31 | Order Detail line table | P | P | **F** | PA | **F** | PA | PA | **F** | **F** | PA | N | N |
| S-32 | Per-line inventory expander | P | P | P | P | P | PA | P | N | N | P | N | N |
| S-33 | Order Detail edit mode | P | P | PA | PA | **F** | PA | PA | **F** | **F** | N | N | N |
| S-34 | Edit-locked notice | N | P | N | N | N | N | N | N | N | N | N | N |
| S-35 | SO status change confirm | N | N | N | N | N | N | N | N | N | N | N | N |
| S-36 | Reservations / allocation | P | P | P | P | P | N | P | N | N | N | N | N |
| S-37 | Allocation history table | P | P | PA | P | PA | **F** | **F** | **F** | **F** | PA | N | **P** |
| S-38 | Release reservation confirm | N | N | N | N | N | N | N | N | N | N | N | N |
| S-39 | Shipping capacity preview | PA | P | P | P | PA | N | P | N | N | N | N | N |
| S-40 | Order Detail notes card | N | P | N | N | N | N | N | N | N | N | N | N |
| S-41 | Expected Receipts toolbar | N | P | N | N | N | N | N | N | N | N | N | N |
| S-42 | Expected Receipts table | P | P | PA | P | **P** | PA | P | **F** | **F** | N | N | P |
| S-43 | ER row actions | N | P | N | N | N | N | P | N | N | N | N | N |
| S-44 | ER empty state | N | P | N | N | N | N | N | N | N | N | N | N |
| S-45 | ER create/edit modal | N | P | N | N | N | N | N | N | N | N | N | N |
| S-46 | Supplies header + Request | N | P | N | N | N | N | N | N | N | N | N | N |
| S-47 | Supplies sub-tabs | N | P | N | N | N | N | N | N | N | N | N | N |
| S-48 | Supplies search field | N | P | N | N | N | N | N | N | N | N | N | N |
| S-49 | Supplies inventory table | P | P | P | P | **F** | PA | P | **F** | **F** | P | N | P |
| S-50 | Supply lot / incoming detail | PA | P | P | P | P | N | **F** | **F** | N | P | N | PA |
| S-51 | Supply Requests list | P | P | P | P | P | PA | P | **F** | **F** | N | N | PA |
| S-52 | Supply Requests feedback | N | P | N | N | N | N | N | N | N | N | N | N |
| S-53 | Request Supply modal | N | P | N | N | N | N | N | N | N | N | N | N |
| S-54 | Lot Detail side panel | P | P | P | P | PA | N | **F** | N | N | N | N | **F** |
| S-55 | Lot disambiguation | P | P | P | P | PA | N | **F** | N | N | N | N | N |
| S-56 | Product Detail panel | P | P | P | P | **F** | **F** | **F** | **F** | **F** | N | N | N |
| S-57 | Sankey controls bar | N | P | N | N | N | N | N | N | · | N | · | · |
| S-58 | Sankey chart + legend | P | **P** | P | PA | N | N | P | N | · | N | · | · |
| S-59 | Sankey banner / loading | N | P | N | N | N | N | N | N | · | N | · | · |
| S-60 | Summary strip | P | P | P | P | P | N | P | N | · | N | · | · |
| S-61 | Production lines grid | PA | P | P | P | P | N | P | N | · | N | · | · |
| S-62 | Error / stale banners | N | P | N | N | N | N | N | N | · | N | · | · |
| S-63 | Lot search + type-ahead | P | P | P | **F** | PA | PA | PA | N | N | N | N | N |
| S-64 | Trace direction + Trace | N | P | N | N | N | N | N | N | N | N | N | N |
| S-65 | Recent lots strip | PA | P | P | P | **F** | N | **F** | N | N | N | N | N |
| S-66 | Trace legend | N | P | N | N | N | N | N | N | N | N | N | N |
| S-67 | Status bar | N | P | N | N | N | N | N | N | N | N | N | N |
| S-68 | Trace graph + zoom | P | **P** | P | **F** | N | N | P | N | N | PA | N | N |
| S-69 | Node tooltip | N | P | P | **P** | N | N | P | N | N | N | N | N |
| S-70 | Trace detail + exports | P | P | PA | P | **P** | PA | P | **F** | **F** | PA | N | **P** |
| S-71 | Print audit report | P | P | PA | P | P | PA | P | N | N | PA | · | P |
| S-72 | Topbar KPI strip | P | P | P | P | P | N | P | N | · | N | · | P |
| S-73 | Topbar actions | N | P | N | N | N | N | N | N | · | N | · | N |
| S-74 | Legend bar | N | P | N | N | N | N | N | N | · | N | · | N |
| S-75 | Delta panel | P | P | P | P | P | N | P | N | · | N | · | N |
| S-76 | Settings panel (form) | N | P | N | N | N | N | N | N | · | P | · | N |
| S-77 | FG on hand (disclosure) | P | P | P | P | P | N | P | N | · | P | · | N |
| S-78 | Bulk-bin WIP (disclosure) | P | P | P | P | P | N | P | N | · | P | · | N |
| S-79 | Product catalog (form) | P | P | P | P | P | N | P | N | · | P | · | N |
| S-80 | Orders in / plans out | N | P | N | N | N | N | N | N | · | N | · | N |
| S-81 | CSV import result | N | P | N | N | N | N | N | N | · | N | · | N |
| S-82 | "How this works" | N | P | N | N | N | N | N | N | · | P | · | N |
| S-83 | Production board table | P | **P** | P | PA | P | P | P | N | · | N | · | P |
| S-84 | Board cell copy / more | P | P | P | PA | P | P | P | N | · | P | · | P |
| S-85 | Schedule panel | P | P | P | P | P | N | P | N | · | N | · | P |
| S-86 | Pin modal | N | P | N | N | N | N | N | N | · | N | · | P |
| S-87 | Order book | P | P | PA | PA | P | P | P | **P** | · | N | PA | P |
| S-88 | Add order line form | N | P | N | N | N | N | N | N | · | N | · | P |
| S-89 | Add-order validation alert | N | N | N | N | N | N | N | N | · | N | · | N |
| S-90 | Order book empty state | N | P | N | N | N | N | N | N | · | N | · | N |
| S-91 | Scheduler print view | P | P | P | PA | P | P | P | N | · | N | · | P |

## Matrix C — Charts & Dashboards (CHART-001…008)

Only screens that render a chart, diagram, or data-visual are listed; every other screen is **N** for all eight rules.

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 |
|---|---|---|---|---|---|---|---|---|---|
| S-02 | Mini-calendar strip | P | P | P | PA | PA | PA | PA | **F** |
| S-06 | Today So Far tile | **P** | P | P | PA | P | P | PA | PA |
| S-08 | Production Calendar | P | P | P | PA | P | P | **F** | PA |
| S-09 | Calendar day detail panel | P | P | P | PA | P | P | **F** | PA |
| S-58 | Sankey chart + legend | **P** | **P** | **P** | **F** | **F** | PA | **F** | **F** |
| S-60 | Summary strip | P | P | P | PA | P | P | P | · |
| S-61 | Production lines grid | P | P | P | PA | P | P | PA | · |
| S-66 | Trace legend | N | N | **P** | N | N | P | **F** | N |
| S-68 | Trace graph + zoom | **P** | **P** | **P** | **P** | **P** | PA | **F** | PA |
| S-69 | Node tooltip | N | P | P | N | **F** | PA | P | PA |
| S-70 | Trace detail + exports | P | P | P | **P** | **P** | P | P | PA |
| S-71 | Print audit report | P | P | P | P | **P** | P | P | · |
| S-72 | Topbar KPI strip | P | P | P | **P** | P | P | P | · |
| S-75 | Delta panel | P | P | P | P | P | P | P | · |
| S-83 | Production board table | P | P | P | P | **P** | P | **F** | · |

---

## Findings

---

### SEARCH-003 — The current search scope is unmistakable before typing — **CRITICAL**

**S-48 (Supplies search) — FAIL.** *Hard rule.*

**Evidence:** `dashboard/index.html:311-314` — placeholder *"Search products..."*; `dashboard/dashboard.js:3830-3834`:
```
const query = document.getElementById('supplies-search').value.trim().toLowerCase();
const rows = supplyItemsForTab().filter(item => item.name.toLowerCase().includes(query));
if (rows.length === 0) { container.innerHTML = `… ${query ? 'No products match this search.' : …}` }
```
`supplyItemsForTab()` (`3656-3665`) filters to the **currently selected sub-tab** before the search runs. So with the Ingredients sub-tab active, searching "gloves" returns *"No products match this search."* — a definitive negative for an item that exists one tab over under Packaging. Nothing in the placeholder, the toolbar, or the empty state says the search is scoped to the sub-tab.

This is precisely the failure the rule exists to prevent: *"Prevents an operator thinking a lot doesn't exist because they searched the wrong scope."* Luz concludes the item is not in the catalogue and files an unlisted supply request for something already tracked.

**Suggested fix:** Change the placeholder to *"Search ingredients…"* / *"Search packaging…"* (bound to the active sub-tab), and change the empty state to *"No ingredients match "gloves". [Search all supplies]"* with a control that switches to the All sub-tab and keeps the query.

**S-63 (Traceability lot search) — FAIL.** `dashboard/traceability.html:276` — placeholder *"Search by lot code, supplier lot, or product name..."*. The stated scope is "lots"; the actual scope is *"lots appearing in the last 100 transactions"* (`374-400` — `api('/transactions/history?limit=100')`). The placeholder describes a search over the lot table; the search is over a truncated transaction cache. A lot from last month returns *"No matches"* (`469`) with no indication that the index simply does not reach that far back. Compounds NAV-012 in `01-nav-layout.md`.

**S-04 (Global search) — PARTIAL.** `dashboard/index.html:33` — *"Search SKU, lot, SO, customer..."* names four scopes ✓, which is far better than a bare "Search". But `/search` returns exactly those four types (`dashboard.js:1526-1553`) and the user has no way to know it excludes expected receipts, supply items, supply requests, notes, suppliers, and transactions. The scope is also placeholder-only, so it disappears the moment they type (INPUT-002).

**S-47 — PARTIAL.** The sub-tab strip is the de-facto scope selector and it is visible ✓, but nothing visually connects it to the search field sitting in the same toolbar (`index.html:299-315`).

**PASS — cite these:** `#orders-customer-search` placeholder *"Filter by customer..."* (`index.html:223`) states both the verb and the scope ✓; `#er-text-filter` *"Filter by product / supplier / reference..."* (`267`) enumerates every field searched ✓ — the best scope statement in the product; the allocation lot select carries an explicit scope hint, *"Only positive on-hand lots for the selected product are shown."* (`dashboard.js:2924`) ✓.

---

### DATA-004 — Never truncate an identifier so two records look alike; keep a one-step path to the full value — **CRITICAL**

*Hard rule.*

**S-63 (Traceability search results) — FAIL.** `dashboard/traceability.html:108` — `.search-item .product-name { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }`, rendered at `465` as `<span class="product-name">${it.product_name}</span>` with **no `title` attribute**.

In the lot picker the product name is what distinguishes two lots whose codes share a prefix. *"Sweetened Toasted Coconut 25 lb – Blue Str…"* and *"Sweetened Toasted Coconut 25 lb – Red Str…"* both clip before the distinguishing suffix, and there is no tooltip, no expand, and no detail view. Arturo picks by eye and there is nothing to pick by.

**S-68 (Trace graph nodes) — FAIL on truncation direction.** `dashboard/traceability.html:1145-1148`:
```
function truncate(str, max) { return str.length > max ? str.slice(0, max - 1) + '…' : str; }
```
applied as `truncate(node.label, 22)` (`1115`) and `truncate(node.sublabel, 26)` (`1124`). This is **end**-truncation on lot codes. The rule is explicit: *"LAT codes and supplier lot numbers share prefixes and differ at the end — end-truncation ('LAT-260907-GRAN-…') hides exactly the distinguishing part; middle-truncate ('LAT-2609…-003') or guarantee the full code fits."*
The full value **is** available in one step — the hover tooltip (`1157-1164`) ✓ — but only on a pointer device; on touch there is no hover and `onNodeClick` does something else entirely (`1177-1194`). So: FAIL on direction, PARTIAL on the one-step path.

**S-83, S-87, S-91 (Scheduler) — PARTIAL.** `scheduler:153` — `.ord .o-product { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }` and `:151` `.o-line-main { overflow: hidden }`, with no `title` attribute on either. SKU-differentiating suffixes (10 LB vs 25 LB) sit at the end of product names.

**S-25, S-26, S-31, S-33 — PARTIAL.** `dashboard.css:2217` — `.er-notes` truncates but carries a `title` ✓ (`dashboard.js:3429`) — correct, and it is a note rather than an identifier. The problem elsewhere is the reverse: `.order-blockers-cell { max-width: 320px }` (`1325-1328`) and `.line-blockers-cell { max-width: 300px }` (`1818`) constrain a wrapping flex container, so blocker chips reflow rather than clip ✓ — no truncation, but the cell grows tall instead (DATA-003).

**Largest-text-size clause — FAIL app-wide.** Every `font-size` declaration in `dashboard.css` is in `px` (over 130 of them). The browser's text-size setting therefore has no effect on any text in the product, and the rule's requirement to *"show as much useful text at the largest text size as at the default"* cannot be assessed because the largest text size does not exist. Cross-referenced under ACCESS-001 and ACCESS-006 in `05-access-icon-other.md`.

**PASS — lot codes are never truncated in the dashboard.** `.lot-link` (`dashboard.css:840-848`) has no width constraint and no ellipsis; every table cell holding a lot code wraps rather than clips. Given the rule's Critical rating this is the right default and it is applied consistently.

---

### DATA-008 — Sortable columns with reversible direction, a visible active-sort indicator, and a sensible default — **HIGH**

**FAIL on every table in the product.**

**Evidence:** No `<th>` anywhere carries a click handler, an `aria-sort` attribute, or a sort affordance (verified by grep across `dashboard/*.html`, `dashboard/*.js`, `dashboard/scheduler/*.html`). There is no mobile sort control either. Twelve tables, zero sortable columns.

The rule's Factory Ledger example is exact about what is missing: *"Lot table sortable by age (FIFO check), qty, product; SO table by ship date, customer, status."* Luz cannot sort the orders table by ship date; she can only filter to "Overdue only" and read what the server returned. She cannot sort Expected Receipts by remaining quantity to find the biggest shortfall, or Finished Goods by cases to find what is nearly out.

**The fixed defaults are, in fairness, well chosen** — and this is why the absence is a High rather than a Critical failure:

| Table | Default order | Evidence | Right default? |
|---|---|---|---|
| Supply FIFO lots | lot date ascending | `dashboard.js:3729-3738` (`sortSupplyLots`) | ✓ FIFO is the task |
| Supply requests | `created_at` descending | `dashboard.js:3909-3911` | ✓ newest first |
| Supplies inventory | name, `localeCompare` | `dashboard.js:3708` | ✓ |
| Recent entries | server, newest entered first | `dashboard.js:611` | ✓ |
| Dispatch Queue | ready first, then ship date, then id | `dashboard.js:2066-2072` | ✓ excellent for the task |
| Trace detail | by graph layer | `traceability.html:1279` | ✓ |
| Scheduler order book | by status priority (late/won't-finish first) | `scheduler:1519` | ✓ **PASS** |
| Orders (non-dispatch) | server order, unspecified | `dashboard.js:2263` | unknown |
| Expected receipts | server order | `dashboard.js:3375` | unknown |
| Finished goods / batch / ingredients | server order | `dashboard.js:919`, `1003`, `1107` | unknown |
| Allocations | server order | `dashboard.js:2937` | unknown |

**S-87 — PASS.** The scheduler order book is the only list in the product whose sort is both deliberate and stated: `statusPriority = {late:0, inc:0, tight:1, ontime:2, early:2, stock:2}` puts what is at risk at the top ✓.

**Suggested fix:** Add click-to-sort with an `aria-sort` indicator to the four tables where the task demands it — Orders (ship date, customer, remaining), Expected Receipts (expected date, remaining), Finished Goods (cases), and Supplies (on hand). One shared `sortableTable()` helper covers all four.

---

### DATA-006 — One consistent table style; alternating shading on wide tables; selection stays identifiable — **MEDIUM**

**S-17, S-18 (Shipping and Receiving logs) — FAIL. The alternating shading never appears.**

**Evidence:** `dashboard/dashboard.css:775` — `.activity-table tr:nth-child(even) td { background: var(--row-alt) }`. But `renderShipments` (`dashboard.js:1196`, `1204`) and `renderReceipts` (`1249`, `1258`) emit **two** `<tr>` per record: the data row and an `.activity-detail` row. So the data rows are always the odd children (1st, 3rd, 5th…) and the even children are always the detail rows — which are `display: none` until expanded (`css:818`) and which set their own `background: var(--row-header)` when visible (`820`).

The result: **no visible row in either log is ever shaded.** A five-column table with up to 100 rows has no row-tracking aid at all, which is exactly the failure mode the rule names — *"row shading prevents reading the wrong row's value."*

**S-10…S-13 (Inventory tables) — FAIL, and the markup is invalid.**

**Evidence:** `dashboard.js:918` opens `<table class="inv-table"><thead>…<tbody>`, then for each product emits a `<tr class="expandable">` followed by `<tbody class="lot-breakdown" id="…">` (`930`) — a `<tbody>` **nested inside** the already-open `<tbody>`. Same pattern at `1019` (batch) and `1115` (ingredients). Nested `<tbody>` is invalid HTML; browsers recover by implicitly closing the outer one, so the DOM tree does not match the source and `tr:nth-child(even)` (`css:698`) counts within whichever `<tbody>` the parser produced. The striping is therefore unpredictable rather than alternating.

**S-27, S-37, S-56 — FAIL on wide tables with no shading at all.** `.order-lines-table` (`css:1376-1400`) and `.allocation-table` (`2029-2054`) define no `nth-child` rule. The allocation table is explicitly `min-width: 820px` (`2031`) — a wide multi-column table with no row shading, which is the rule's primary target. The product panel's two tables are built from inline styles with no shading (`dashboard.js:1462-1484`).

**Twelve table treatments — FAIL.** `.inv-table` (`674`), `.activity-table` (`750`), `.orders-table` (`1293`), `.order-lines-table` (`1376`), `.order-inventory-table` (`1685`), `.allocation-table` (`2029`), `.er-table` (`2214`, inherits `.orders-table`), `.supplies-table` (`2338`), `.supply-requests-table` (`2341`), `.order-readiness-table` (`1817`), `.detail-table` (`traceability.html:192`), plus two inline-styled tables (`dashboard.js:1462`, `1476`). Header font sizes range across 10px, 11px, and 12px; cell padding across `8px 12px`, `6px 10px`, `8px 10px`, `6px 0`, and `4px 8px`; `text-transform` is `uppercase` on most and `none` on `.order-inventory-table th` (`1708`).

**Grouping — PARTIAL.** Batch inventory groups by production family with `.batch-family-heading` when more than one family is present ✓ (`dashboard.js:1055-1061`) — a good implementation. Expected Receipts are **not** grouped by supplier or expected date, which is the rule's own Factory Ledger example (`932` of the standard).

**Selection — FAIL.** No table marks a selected row. See FEEDBACK-009 in `03-feedback-error-notify.md`.

**S-83, S-84, S-87, S-91 — PASS.** The scheduler board uses sticky headers, a sticky station column, weekend shading, and a consistent cell treatment throughout (`scheduler:102-126`) ✓.

---

### DATA-007 — One consistent row anatomy per record type — **MEDIUM**

**FAIL. The lot row — the most-repeated record type in the product — has seven different anatomies.**

| Where | Structure | Evidence |
|---|---|---|
| Finished Goods lot | `lot-link · qty` in a `.lot-row`, 32px indent, 4 columns | `dashboard.js:935-936`; `css:729-735` |
| Batch lot | same shape, 3 columns | `dashboard.js:1033-1034` |
| Ingredient lot | same shape, 2 columns | `dashboard.js:1120-1121` |
| Shipment / receipt detail | `lot-link — product: qty` as inline text separated by `<br>` — **not a row at all** | `dashboard.js:1211`, `1264` |
| Product panel active lots | `lot-link \| source \| on hand` in an inline-styled table | `dashboard.js:1465-1469` |
| Product panel depleted lots | same, in a second table with **no header row**, at `opacity: 0.6` | `dashboard.js:1476-1484` |
| Supplies FIFO | a `.supply-lot-card` with a `<dl>` of Lot code / Lot date / Remaining quantity | `dashboard.js:3785-3793` |
| Allocation Level cell | `<strong>Lot</strong><span>{lot_code}</span>` inside a cell | `dashboard.js:2941` |

None carries a leading status indicator, none carries a trailing accessory, and the quantity appears variously as `"1,200 lb"`, `"1,200 lb · 48 units"`, `"1,200 lb · 3.2 batches"`, and `"1,200 lb"` in a `<dd>`. The rule's example is exact: *"A lot row looks the same on the Lots screen, in trace results, and in the allocation picker: status dot · LAT code · product/qty · chevron."*

**S-04, S-25 — PARTIAL.** The **order row** has two anatomies: an 11-column table row (`dashboard.js:2319-2331`) and a search-dropdown item `order_number | customer | status` (`1544`).

**S-54, S-55, S-56 — FAIL.** The lot **panel** itself renders three unrelated structures depending on entry point: a `<dl class="lot-info-grid">` plus a `.timeline` for a lot (`1401-1431`), a stack of inline-styled buttons for a disambiguation (`1373-1385`), and a `<dl>` plus two inline-styled tables for a product (`1447-1491`) — all inside the same panel shell with the same title bar.

**PASS:** the order **line** row is consistent between the inline expander (`2393-2401`) and the detail table's non-edit mode (`3256-3278`) in column meaning, though not in column count ✓.

---

### SEARCH-002 — One global search finds every record type; section boxes are filters — **HIGH**

**S-04 — FAIL, with a live bug.**

**Bug: clicking a customer result silently applies no filter.**
`dashboard/dashboard.js:1596-1612`:
```
const custFilter = document.getElementById('orders-customer-filter');
if (custFilter) { custFilter.value = name; custFilter.dispatchEvent(new Event('change')); }
```
No element with the id `orders-customer-filter` exists. The real field is `orders-customer-search` (`dashboard/index.html:223`). The `if (custFilter)` guard swallows the miss entirely, so clicking "Restaurant Depot" in the global search dropdown switches to the Sales Orders tab, clears the search box (`1600`), and shows the **unfiltered** order list with no message. The user believes they searched for a customer and is looking at every order.
(Two further mismatches in the same handler: the dispatched event is `change`, while the real field listens for `input` — `3333`; and the real filter is applied by `renderOrdersList`, not by the change handler.)

**Coverage gap.** `/search` returns products, lots, orders, and customers (`dashboard.js:1526-1553`). It does **not** return expected receipts, supply items, supply requests, notes, suppliers, or transactions. Luz cannot find an expected receipt or a supply request from the one place the product asks her to search.

**S-63 — FAIL. A second, divergent "global" search.**
`dashboard/traceability.html:374-400` builds its own lot index client-side from `/transactions/history?limit=100`, and `filterLots` (`448-472`) searches it. The dashboard's `#global-search` queries the server's `/search?q=` (`dashboard.js:1514`). Searching the same lot code in the two places returns different results — the Traceability one being a strict subset limited to recent transactions. The rule is explicit: *"Never have two 'global' searches that return different things."*

**PASS on the filter clause — cite these:** `#orders-customer-search` (`dashboard.js:2034`, `2050-2052`), `#er-text-filter` (`3386-3394`), and `#supplies-search` (`3830-3831`) all behave strictly as filters over their own view ✓. Their placeholders even say "Filter by…" rather than "Search" (`index.html:223`, `267`) ✓.

**Suggested fix:** correct the element id in `dashboard.js:1606` and dispatch `input`; extend `/search` to cover expected receipts, supplies, and supply requests; replace Traceability's client-side index with the same `/search` endpoint scoped to lots.

---

### SEARCH-001 — Search has a primary, persistent position reachable from anywhere in one action — **HIGH**

**S-03, S-04 — PASS.** `dashboard/index.html:31-36` places `#global-search` in the centre of `.app-header`, which is `position: sticky; top: 48px` (`css:140-142`), so it is visible from every tab without navigating ✓. At ≤768px it moves to its own full-width row (`css:2184`) and stays visible ✓ — subject to the sticky-overlap bug documented under LAYOUT-003.

**S-57…S-70 — FAIL.** Global search exists **only on `index.html`**. The other three pages carry no search field at all:
- `dashboard/sankey.html:282-288` — header holds the mini-calendar and `#last-updated` only
- `dashboard/process-flow.html` — same
- `dashboard/traceability.html:264-269` — header holds the mini-calendar only; the page's own `#lotSearch` (`276`) searches lots within Traceability, not records globally

From Material Flow or Production Lines, reaching global search means navigating back to the Dashboard — a full page load and a loss of context. The rule's test is *"From any screen, can the user reach global search in one tap/click without leaving their context?"*

**Suggested fix:** move the header (brand, global search, health badge, theme, refresh) into the shared chrome alongside the site nav so all four pages carry it. This also resolves the four-way duplication recorded under LAYOUT-002 item 8.

---

### SEARCH-004 — Reduce typing with recents, suggestions, completions, and near-miss correction — **HIGH**

**S-63, S-65, S-67 — PASS. The reference implementation, and the only complete one in the product.**
`dashboard/traceability.html` implements all four clauses:
- **Recents before typing** — the recent-lots strip renders 20 lot pills from the transaction index on load (`285-287`, `404-423`), so a common target costs **zero** keystrokes ✓
- **Predict as they type** — a 200ms-debounced type-ahead grouped by transaction type, *Received Lots / Production Lots / Packed Lots / Shipped Lots* (`432-472`) ✓
- **Completion highlighting** — `highlight()` bolds the matched substring in accent colour (`474-478`) ✓
- **Near-miss correction** — on a 404, `suggestSimilar()` matches on the first six characters and renders *"Did you mean: 26-0907-C 26-0907-D"* as clickable links that select and trace in one click (`529-530`, `938-947`) ✓

The rule's test — *"Can the user reach a common target with 3 or fewer keystrokes?"* — is met at zero.

**S-04 — PARTIAL.** `#global-search` has a 300ms debounce and a 2-character minimum (`dashboard.js:1509`, `4220-4223`) and returns grouped results ✓, but has no recents, no completion, and no near-miss correction. On zero results it renders `'<div class="search-item">No results found</div>'` (`1556`) — a dead end with no suggestion and no way to widen the search.

**S-45 — PARTIAL.** `#er-product-search` is a real type-ahead (`dashboard.js:3516-3540`) ✓ with a 250ms debounce (`3648`), but no recents and no near-miss; on zero results, *"No products found"* (`3523`).

**S-79, S-88 — PASS.** The scheduler's `skuPicker` (`scheduler:1558`) is a select over the full catalogue ✓ — selection rather than typing, which satisfies the rule's intent directly.

**S-24, S-41, S-48, S-53 — PARTIAL.** Plain substring filters with no suggestion layer.

**S-22 — FAIL.** `#note-entity-id` is a free text field where a type-ahead over products, lots, customers and suppliers is exactly what the rule (and INPUT-011/014) asks for. See `02-actions-input-touch.md`.

---

### SEARCH-008 — Every primary record has a stable, shareable link that opens straight to it — **MEDIUM**

**S-03…S-56 (the entire dashboard) — FAIL.** *Situational idea, but with real operational cost here.*

**Evidence:** there is no URL state anywhere in `dashboard/dashboard.js` — verified by grep for `history`, `pushState`, `replaceState`, `location.hash`, and `searchParams`, which return **zero** matches in the file.
- `initTabs` (`494-508`) toggles CSS classes only; the active tab is not in the URL.
- `openOrderDetail` (`2526-2550`) sets `listView.style.display = 'none'` and renders; the order is not in the URL.
- `openLotPanel` (`1344-1370`) and `openProductPanel` (`1438-1504`) do not touch history.
- The Expected Receipt and Supply Request modals do not either.

Consequences:
1. **Nothing can be shared.** A WhatsApp message cannot link to SO-1234; the recipient opens the Operations tab and navigates by hand. A QR code on a pallet label cannot open its lot.
2. **The back button does nothing useful.** Browser Back from an order detail leaves the site entirely rather than returning to the list.
3. **Refresh loses the record**, compounding ERROR-002 in `03-feedback-error-notify.md`.

The standard connects this rule to NAV-001 (*"Deep links so a WhatsApp message can open a specific SO"*), and the dashboard's own architecture assumes cross-tool coordination — the Floor GPT, WhatsApp, and this dashboard all reference the same records.

**S-63, S-64, S-65, S-68, S-70 — PASS. The reference implementation.**
`dashboard/traceability.html:1365-1372` — `updateURL()` writes `?lot=26-0907-C&direction=forward&product_id=42` via `history.replaceState`, and `loadFromURL()` (`1374-1387`) reads them back on load, restores the direction toggle, enables the Trace button, and auto-runs the trace. A lot is fully addressable, and the `product_id` parameter even survives the disambiguation choice ✓. This is exactly what every dashboard record needs.

**Suggested fix:** adopt the Traceability pattern — `#tab=orders&order=1234`, `#tab=supplies`, `#lot=26-0907-C` — read on load and written on every open. It costs one helper function and resolves a NAV-001 gap, a SEARCH-008 failure, and part of ERROR-002 at once.

---

### SEARCH-007 — Support find-within-page on long content — **MEDIUM**

**S-17, S-18 — FAIL, and native Ctrl+F cannot rescue it.** `dashboard/dashboard.js:1177` and `1230` fetch `limit=100`; `overflowClass(idx)` (`1148-1150`) adds `.overflow-hidden` to every row past the fourth, and `dashboard.css:822` sets `.activity-table tr.overflow-hidden { display: none }`. Rows hidden with `display: none` are **not found by the browser's own find-in-page**. So of 100 shipments, 96 are unreachable by any search mechanism until the user clicks "Show all (96 more)" — and there is no filter field even then.

**S-51 — FAIL.** `dashboard.js:3908` fetches `limit=500` supply requests and the toolbar offers only a show-completed checkbox (`index.html:324-327`). There is no filter field of any kind. Finding a specific request among 500 rows means scrolling.

**S-14 — FAIL.** `limit=20` with no filter; low row count, but the feed is the audit trail and there is no way to find a transaction id within it.

**S-41, S-42 — PASS.** `#er-text-filter` searches product, SKU, supplier, reference **and** notes in one field (`dashboard.js:3391-3393`) ✓ — the most complete find-within-view in the product.

**S-24, S-25, S-26 — PARTIAL.** `#orders-customer-search` filters by **customer only** (`dashboard.js:2050-2052`). With up to 200 orders loaded, there is no way to find a specific SO number within the view; the user must use the global search, which navigates away and opens the detail rather than locating the row.

**S-49 — PARTIAL.** `#supplies-search` filters by name only and is scoped to the active sub-tab (SEARCH-003).

**S-70 — PASS.** The trace detail table is fully rendered plain text with nothing hidden, so the browser's own find works ✓ — a legitimate desktop answer for a static table.

---

### DATA-005 — Every column has a descriptive header with unit; every headerless list has a context label — **MEDIUM**

**S-31, S-33 (Order Detail line table) — FAIL.** `dashboard.js:3236-3238` emits `Product | Ordered | [Price] | Shipped Effective | Remaining | Allocated | Shortage | Blockers | Status`. **None of the five numeric columns carries a unit**, and the cells render mixed formats — `lineFmt` produces `"1,200 lb · 48 units"` when a unit count exists and `"1,200 lb"` when it does not (`3251`), while Allocated and Shortage use `fmtLbs` (`3275-3276`). "Price" carries no currency (see INPUT-008).

**S-49 (Supplies inventory) — FAIL.** `dashboard.js:3838` emits `Name | On Hand | Unit | Incoming | Status`. "On Hand" has no unit but a dedicated Unit column follows ✓ — a reasonable design. But **"Incoming" is rendered in pounds regardless of the row's unit**: `dashboard.js:3845-3848` computes `unitIsPounds` from the item's own unit and, when false, emits `` `${fmtWt(incomingTotal)} lb` ``. So a row measured in "box" shows On Hand in boxes and Incoming in pounds, side by side, under a Unit column that says "box". The two columns are not comparable and the header does not warn.

**S-19, S-20 — FAIL on the headerless-list clause.** `dashboard/index.html:193` — `<div id="notes-list"></div>` has no heading, no count, and no context label. The tab is named "Notes" but the list itself says nothing about what it contains or how many. Contrast `#supply-requests-list`, which is headed *"Supply Requests (3)"* (`index.html:322`) ✓, and `#recent-entries-feed`, headed *"Recent Entries"* with a subtitle and a live count (`115-120`) ✓.

**S-56 — FAIL.** `dashboard.js:1476` — the depleted-lots table opens straight into `<tr>` data rows with **no header row at all**, while the active-lots table immediately above it has one (`1463`). Three unlabelled columns of lot data.

**S-65 — FAIL.** `traceability.html:286` — the recent-lots strip is labelled only *"Recent:"* with no count and no statement of what "recent" means (it is the last 20 lots from the last 100 transactions).

**PASS — cite these:**
- **Expected Receipts** (`dashboard.js:3424`) — `Expected (lb) | Received (lb) | Remaining (lb)`: **every numeric column carries its unit**. The best header row in the product.
- **Ingredients** (`dashboard.js:1102-1110`) — the header reads `On Hand (lb)` when every row shares a unit and falls back to bare `On Hand` with per-cell units when they differ ✓ genuinely thoughtful.
- **Shipping / Receiving** (`1191`, `1244`) — `Occurred / Entered | Product(s) | Qty (lb) | Customer | Ref` ✓.
- **Orders expand columns** carry `aria-label="Expand"` and `aria-label="Factory Ready"` on their empty headers (`2313`) ✓.

**S-25, S-26 — PARTIAL.** "Effective Remaining" has no unit while its cells render `"1,200 lb · 48 units"` (`formatOrderRemaining`, `2112-2116`).
**S-16 — PARTIAL.** The first column is headed "Entered" (`1315`) but `#daily-entries-mode` can switch the view to event date; the header does not change with the mode.

---

### DATA-003 — Rows carry only identifying and decision-critical values; no row wraps past two lines — **HIGH**

**S-25, S-26 — FAIL.** `dashboard.js:2313` defines eleven columns. Three of them are multi-line by construction:
- **Blockers** — `renderBlockerChips` emits one `.readiness-chip` per blocker into a `flex-wrap` container (`css:1609-1614`) constrained to `max-width: 320px` (`1325-1328`), with `.readiness-chip-detail { white-space: normal }` (`1632`) so each chip wraps its own detail. Three blockers with details is four or more lines.
- **Ship By** — `.ship-by-date` + `.ship-by-weekday`, both `display: block` (`css:1523-1527`) = two lines.
- **Status** — a `.so-badge` plus a `.so-ready-pill` carrying up to three segments (`dashboard.js:2229-2235`).

**S-31, S-33 — FAIL.** `renderOrderDetail` calls `renderLineReadiness(l, true)` (`3277`) — the `showDetail` flag **on**, so every blocker chip renders its full detail line inside the table cell. Add `.pallet-secondary` sub-lines under three separate numeric cells (`3268`, `3274`, `3261`) and a line row reaches five or more lines.

**S-14 — FAIL.** A `.recent-entry-card` renders topline + direction + optional correction detail + a `<ul>` with one `<li>` per transaction line (each of which may itself wrap to a second line for the lot code) + a two-column dates grid + timing flags (`dashboard.js:589-598`). Height is unbounded by the number of lines on the transaction.

**S-42 — PARTIAL.** `dashboard.js:3429` — the product cell stacks name, SKU, and notes = three lines.
**S-16 — PARTIAL.** The Entered cell stacks date/time, provenance, and a late-lag line (`1325-1328`) = three lines, and uses `rowspan` across a transaction's lines — a structure the reader must decode.
**S-17, S-18 — PARTIAL.** Two lines (`1197`, `1250`) ✓ exactly at the limit.
**S-37 — PARTIAL.** Three cells each stack a `<strong>` and a `<span>` (`2941-2951`) = two lines each ✓ at the limit.

**PASS:** the Supplies inventory row (`3849-3852`), the supply requests row (`3937-3946`), the ER row's numeric cells, and the Today So Far rows are all single-line ✓.

---

### DATA-012 — One date/time format system-wide; time zones correct; live values update themselves — **MEDIUM**

**FAIL on format consistency. Thirteen date/time renderings coexist.**

| Formatter | Output | Evidence |
|---|---|---|
| `formatDateShort` | `09/07/26` | `dashboard.js:1971-1976` |
| `formatBusinessDate` | `Sep 7, 2026` | `dashboard.js:511-518` |
| `formatEnteredAt` | `Sep 7, 2026, 3:42 PM EDT` | `dashboard.js:520-531` |
| `formatSupplyRequestTime` | `Sep 7, 2026, 3:42 PM ET` | `dashboard.js:3872-3880` |
| `productionDetailDate` | `Monday, September 7, 2026` | `dashboard.js:804-810` |
| `formatReadyTime` | `3:42 PM` | `dashboard.js:1993-2002` |
| `refreshAll` timestamp | `3:42:15 PM ET` | `dashboard.js:4166-4170` |
| Calendar day cards | raw `2026-09-07` | `dashboard.js:768` |
| Activity tables | raw server `date` + `time` strings | `dashboard.js:1197`, `1250`, `1325` |
| Traceability | `Sep 7, 2026, 3:42 PM ET` | `traceability.html:1263` |
| Trace text export | `September 7, 2026 at 3:42:15 PM EDT` | `traceability.html:1310` |
| Sankey | `toLocaleTimeString()` — **viewer locale, no timezone** | `sankey.html:822-823` |
| Scheduler | `Mon 8` | `scheduler` (`fmtShort`) |

`09/07/26`, `Sep 7, 2026`, `2026-09-07`, and `Monday, September 7, 2026` all appear on the Operations tab alone. The Orders and Expected Receipts tables at least agree with each other, both using `formatShipByDate` ✓ (`dashboard.js:2325`, `3434`).

**Time zones — handled with real care in two places, and loosely in three.**

**Correct, and worth preserving:**
- `formatBusinessDate` (`511-518`) deliberately anchors business dates at `T12:00:00Z` and formats with `timeZone: 'UTC'`, with an explanatory comment — *"Business dates are calendar dates, never instants to be converted to ET."* This avoids the classic off-by-one ✓
- `formatEnteredAt` (`520-531`) refuses to invent an offset for a timezone-naive value and renders it raw instead, with a comment saying why ✓

**Loose:**
- `getLocalDateFromISO` (`1978-1983`) builds a **browser-local** Date and `formatShipByDate` derives the weekday from it (`1985-1991`), so a viewer outside ET can see the wrong weekday beside a correct date.
- `getCalendarParams` (`644-667`) round-trips through `new Date(now.toLocaleString('en-US', { timeZone: tz }))` — an implementation-dependent parse. Its `endDate`, `startDate`, and `fmt2` locals (`659-663`) are **dead code**: the function returns `` `days=${totalDays}` `` at `666` regardless, so the "past periods" branch does not actually compute a date range.
- `process-flow.html` `toET()` uses the same round-trip.
- `sankey.html:822-823` uses a bare `toLocaleTimeString()` with no timezone, so its "Updated" reads in the viewer's clock while the rest of the product is explicit about plant time.

**Live values — one correct implementation out of six candidates.**
- ✓ `updateAllocationCountdowns` re-renders every `.allocation-expiry` from `data-expires-at` on a 60-second interval (`dashboard.js:3153-3157`, `3177`) so *"23h 14m remaining"* stays true.
- ✗ `#last-refreshed` is an absolute timestamp set once (`4171`) — never *"3 min ago"*, never updated.
- ✗ `recentEntryLagChip` computes *"Entered 2d 4h later"* at render time and never recomputes (`533-550`).
- ✗ `formatSupplyRequestTime` renders an absolute time with no elapsed value (`3872-3880`).
- ✗ Process Flow's stale-banner time is captured at render.
- ✗ **No lot age anywhere.** The rule's own example — *"lot age ('14 d')"* — has no equivalent; the lot panel shows dates only (`1414`) and FIFO decisions are made by reading dates rather than ages.

---

### DATA-009, DATA-010, DATA-011 — remaining table findings

**DATA-009 — FAIL on every desktop table.** *Medium.* No table has resizable columns; there are no drag handles, no `resize` CSS, and no persisted widths. Several columns are hard-capped, so the values that get clipped cannot be recovered by widening: `.order-blockers-cell { min-width: 190px; max-width: 320px }` (`css:1325-1328`), `.line-blockers-cell { min-width: 210px; max-width: 300px }` (`1818`), `.er-notes { max-width: 320px }` (`2217`), `.supply-request-note { max-width: 300px }` (`2400`), `.order-inventory-table th { width: 120px }` (`1709`). The rule's example — *"Office user widens 'Customer' or 'Notes'"* — is exactly the `.er-notes` case, and the only recovery is the `title` tooltip.

**DATA-010 — PARTIAL, with one clear gap.** *Medium.* Expandable outlines are used well and widely: FG product → lots (`dashboard.js:923-941`), batch → lots (`1007-1039`), ingredient → lots (`1111-1126`), order → lines (`2333`, `2490-2524`), order line → inventory (`3280`, `2638-2662`), supply item → FIFO lots + incoming receipts (`3853-3855`), transaction → lot detail (`1204`, `1258`), and the four collapsible Activity panels (`index.html:130-134`) ✓.
**FAIL on the expand-all clause:** there is **no expand-all or collapse-all control anywhere in the product**. The rule states the need directly — *"Expand-all for recall investigations."* A recall means opening every lot row in Finished Goods one at a time.
**S-31, S-36 — PARTIAL.** Allocations are a **separate flat table** below the line table (`2936-2955`) rather than nested under their line; the only link is a `Line #12` span in the first cell (`2947`). The rule's example is *"SO → lines → allocated lots."*
**S-68, S-70 — PARTIAL.** The trace **graph** conveys the hierarchy visually ✓, but the trace **detail table** — which is what prints and what a screen reader gets (CHART-005) — flattens it to a list sorted by layer (`traceability.html:1279-1293`). Interestingly the **text export** does indent by layer (`1334`, `1341-1346`) ✓, so the exported artefact is closer to the rule than the on-screen one.
**S-76…S-79, S-82 — PASS.** The scheduler's left panel is a clean set of `<details>`-style disclosures ✓ (`scheduler:993-1005`).

**DATA-011 — N/A across the dashboard, PARTIAL on the scheduler.** *Medium, Situational idea.* No dashboard list has user-meaningful order — every list is ordered by date, name, status, or FIFO, all system-determined, which the rule explicitly exempts (*"Not needed where order is system-determined"*).
**S-87 — PARTIAL.** The scheduler board is the one place order is data. The user influences sequence only indirectly, through pins (`openPin`, `scheduler:1435-1476`) and include/exclude (`:1550-1552`). There is no drag to move a job between days and no "Move to…" menu — the rule's own example is *"Reorder the production board queue."* The pin mechanism is a legitimate and well-built indirect path (live feasibility feedback, see INPUT-007), so this is PARTIAL rather than FAIL; see also DRAG-001 below.

---

### DATA-001, DATA-002 — presentation form

**DATA-001 — PARTIAL (S-14, S-50, S-39, S-61).** *High.*
- **S-14 Recent Entries** — `.recent-entries-feed { display: grid; gap: 12px }` (`css:292`) inside `.recent-entries-page { max-width: 760px }` (`286`), rendering one `.recent-entry-card` per ledger event (`dashboard.js:589-598`). The rule permits *"compact stacked cards — one per record, consistently structured"* and these are consistently structured and single-column ✓ — not a mosaic. But this is the **desktop** presentation of a purely textual, comparable audit feed, and each card is tall enough that roughly four fit on screen. A table with Occurred / Entered / Type / Direction / Status / Lines columns would let Blubber scan timing anomalies down a column, which is the stated purpose of the screen.
- **S-50 Supply FIFO lots** — `.supply-lot-card` per lot (`dashboard.js:3785-3793`), each a `<dl>` in a three-column grid (`css:2376-2380`). FIFO comparison means reading lot dates in order; cards scatter that value across a grid instead of aligning it in a column.
- **S-39 Shipping preview** — `.preview-line` cards (`3121-3129`) for comparable numeric data (requested vs can-ship vs reserved-elsewhere).
- **S-61 Production lines** — `.line-card` per line with a stage pipeline; genuinely spatial (a process flow), so the card form is defensible ✓ PARTIAL only for the metrics row.
- **PASS everywhere else** ✓ — every operational record type is a table row.

**DATA-002 — PASS throughout.** *Medium.* All dashboard tables are conventional ✓. The four non-standard layouts are each justified and each carries a key:
- **S-08 Production Calendar** — the standard names this exception explicitly: *"The pack-format calendar is a justified exception."* ✓
- **S-58 Sankey** — genuinely relational flow data ✓ with an always-visible legend and column labels (`sankey.html:338-349`)
- **S-68 Trace graph** — genuinely relational ✓ with an always-visible legend covering node **and** edge types (`traceability.html:291-300`)
- **S-83 Production board** — a days × stations matrix is the conventional form for a schedule ✓ with a legend bar (`scheduler:14-24`)

---

### DRAG-001 — Support drag and drop where users will instinctively try it, and always provide a non-drag alternative

**The Critical clause passes vacuously; the High clause is an absence in two places.**

**Critical clause — *"every drag operation is also achievable via a button, menu, or keyboard"* — PASS everywhere.** There is no drag operation in the product, so there is no drag-only path to any action. Every operation is reachable by button or form. Recorded as PASS rather than N/A because the hard rule's guarantee — no user is locked out of an action because they cannot drag — genuinely holds.

**High clause — *"support drag and drop where users will instinctively try it"* — PARTIAL on the two surfaces where a user reaches for it:**

**S-83, S-87 (Scheduler board and order book) — PARTIAL.** A days × stations board with movable work is the canonical drag surface, and the standard names it twice: *"Scheduler: drag a job between days, and also 'Move to…'"* (DRAG-001) and *"Reorder the production board queue"* (DATA-011). Neither exists. Moving work to a different day means opening the pin modal on the target cell (`scheduler:1435-1476`), forcing a quantity there, and letting the optimizer replan around it — an indirect model that is well built (live feasibility text, an explicit "the pin will be honored but flagged red" warning at `:1461`) but is not what a user will try first. There is also no "Move to…" menu as the button alternative the rule pairs with drag.

**S-36 (Allocations) — PARTIAL.** The standard's second example is *"Allocations: drag a lot onto an SO line, and also an 'Allocate' button."* The Allocate button exists and is good (`dashboard.js:2917-2927`) ✓; the drag shortcut does not. Given that allocation is a four-field form repeated per line, drag would be a genuine saving — but the non-drag path is complete, which is what the hard rule requires.

**DRAG-002 … DRAG-011 — N/A on all 91 screens**, reason: *no drag-and-drop interaction exists in this product.* Nothing can move or copy, be undone after a drop, give drag feedback, snap back on failure, auto-scroll, show drop progress, be selected after a drop, be dragged in one motion, or cross views mid-drag.

*Recorded so a future implementation starts from the right constraints:* if drag is added to the scheduler board, DRAG-003 (Critical — undo or confirm), DRAG-005 (continuous validity feedback), and DRAG-006 (Critical — visible snap-back on failure) become live, and DRAG-001's alternative clause requires the "Move to…" menu to ship alongside.

---

### CHART-007 — The same item, metric, or dataset uses the same color and marks everywhere — **MEDIUM**

**FAIL. Four independent colour systems describe the same supply chain, and two of them use the same hex values for opposite meanings.**

| System | Palette | Evidence |
|---|---|---|
| Dashboard product families | coconut `#60a5fa`, granola `#fbbf24`, graham `#4ade80` | `dashboard.css:17-19` |
| Sankey — by **stage** | Ingredients `#1D9E75`, Production Lines `#7F77DD`, Finished Goods `#D85A30`, Customers `#378ADD` | `sankey.html:339-342` |
| Traceability — by **node type** | Supplier `#1D9E75`, Ingredient Lot `#378ADD`, Batch `#7F77DD`, Finished Goods `#D85A30`, Customer `#BA7517`, Data Gap `#E24B4A` | `traceability.html:28-33` |
| Scheduler — by **constraint state** | ok `#2e7d4f`, amber `#b46a00`, wip `#7b5ea7`, coco `#8a6d3b`, late `#bf2f24`, pin `#1f5fbf`, idle `#98a0aa` | `scheduler:9-16` |

**Sankey and Traceability share four hex values with different meanings:**
- `#1D9E75` = "Ingredients" in Sankey, **"Supplier"** in Traceability
- `#378ADD` = "Customers" in Sankey, **"Ingredient Lot"** in Traceability
- `#7F77DD` = "Production Lines" in Sankey, "Batch" in Traceability (compatible ✓)
- `#D85A30` = "Finished Goods" in both ✓

The two charts sit next to each other in the site nav. A user who learns that green means "ingredient" on the Material Flow page will read green as "ingredient" on Traceability, where it means "supplier" — the opposite end of the chain. The rule's whole purpose is that *"users transfer what they learn from one chart to the next."*

**Coconut — the standard's own example — has five colours.** The rule states *"Coconut is always the same color everywhere."* It is `#60a5fa` on the dashboard (`css:17`), uncoloured in Sankey (coloured by stage, not product), uncoloured in Traceability (coloured by node type), `#8a6d3b` in the scheduler (`scheduler:16`), and — via the Coconut Sweetened line card — inherits the generic card styling in Process Flow.

**S-66 — FAIL.** The Traceability legend correctly documents its own palette (`traceability.html:291-300`) ✓, but there is no shared legend and no shared token file, so the divergence is invisible until a user compares two pages.

**Suggested fix:** define one `--entity-*` palette in the shared stylesheet proposed under LAYOUT-002, assign one colour per **domain object** (coconut / granola / graham) and a separate axis for **chain stage**, and have all four surfaces read from it.

---

### CHART-004 — Pair every chart with a plain-language sentence stating the takeaway and what needs action — **HIGH**

**S-58 (Sankey) — FAIL.** `dashboard/sankey.html:283` — the title is *"Product Flow — Sankey"*, which names the chart type rather than the finding. `354-357` — the footer says only *"All volumes shown in pounds."* plus, when truncated, *"Showing most recent 100 transactions per type. Actual volumes may be higher."* There is no headline, no annotation, and nothing that says what needs action. Blubber opens the page and must derive the insight himself from a four-column diagram.

The rule's Factory Ledger example is a single sentence — *"Coconut line at 91% — 2 days of slack before RD order."* Every input for such a sentence is already computed: the top ingredient, the top customer, and the total volume are all in `processData` (`sankey.html`).

**S-68 (Trace graph) — PASS. The reference implementation.**
`traceability.html:1223-1249` — `renderCompleteness()` renders exactly the sentence the rule asks for, in three variants:
- *"✓ Complete trace — all links confirmed"*
- *"⚠ Partial trace — 2 legacy lot(s), 1 unknown supplier(s)"*
- *"✗ Incomplete — no confirmed links found"*
- and a note variant: *"⚠ This ingredient lot has not been consumed in any production batches yet."* (`736`, `688`)

It states the takeaway, quantifies the gap, and — via the legacy/unknown counts — says what needs attention. It carries a glyph as well as colour (FEEDBACK-011) and sits directly on the chart (`307`, `1224`).

**S-72 (Scheduler KPI strip) — PASS.** `scheduler:3-9` — five KPIs, each with a `k-sub` detail line and an `info` tooltip explaining the metric in plain language (*"The date the last open order is 100% packed under the current plan"*, *"Orders that finish late, exactly on their due date (tight), or don't finish inside the planning horizon"*). "Orders at risk" and "Dominant bottleneck" are themselves the actionable headline ✓.

**S-06, S-08, S-09, S-60, S-61 — PARTIAL.** Numbers with labels but no takeaway sentence. Today So Far shows *"Granola — 4 batches"* without saying whether four is behind, on, or ahead of plan. The Process Flow summary strip shows *"Avg Yield 94%"* with no statement of whether that is good.

**S-02 — PARTIAL.** Ship-date dots carry a per-day `title` (*"3 open Sales Orders ship by 2026-09-12"*) ✓ but the strip as a whole says nothing.

---

### CHART-005 — Every chart exposes its values to assistive technology or as an adjacent data table — **MEDIUM (Hard rule)**

**S-58 (Sankey) — FAIL.** `dashboard/sankey.html:350` — `<div id="sankey-chart"></div>`, filled with D3-generated SVG (`renderSankey`, `572`). There is:
- no `role="img"` and no `aria-label` on the container or the SVG
- no `<title>` or `<desc>` element inside the SVG
- no adjacent data table and no "view as table" toggle
- and the only route to a value is a hover tooltip which is `pointer-events: none`

Every number in the chart — the volume on each of up to ~40 links — exists solely as SVG geometry. It is unreadable by assistive technology and un-copyable by anyone.

**S-68, S-70, S-71 — PASS. The model Sankey should copy.** `traceability.html:314` — the `<svg id="graphSvg">` has no ARIA either, **but** `renderDetailPanel` (`1254-1297`) renders the identical dataset immediately below it as a real `<table class="detail-table">` with a proper header row (`Type | Lot / Name | Product | Quantity | Link | Detail`) and one row per node, sorted by layer. The rule accepts *"an adjacent data table"* ✓, and `downloadAuditText` (`1306-1360`) additionally produces the whole trace as plain text ✓. The print view (`240-246`) hides the graph and shows only the table — so the printed artefact is the accessible one ✓.

**S-83 — PASS.** The scheduler board is a real `<table>` with `<th>` station and day headers (`scheduler:1335-1345`) ✓ — the values are in the DOM as text.

**S-06, S-60, S-72, S-75 — PASS.** Numeric tiles rendered as text ✓. `#today-tile` additionally carries `aria-live="polite"` (`index.html:68`) ✓.

**S-69 — FAIL.** `traceability.html:217-221` — `.node-tooltip { pointer-events: none }`. The tooltip is the only place the full untruncated node label and its transaction ids appear (`1157-1164`), and it is unselectable, uncopyable, and unreachable by keyboard or touch. Cross-referenced under ERROR-010 and DATA-004.

**S-02 — PARTIAL.** `mini-calendar.js:58` — day cells are `<div>`s with `aria-current="date"` on today ✓ and a `title` carrying the count on days with shipments ✓, but the grid has no table semantics and the dots have no text equivalent in the accessibility tree.

---

### CHART-006, CHART-008 — sizing and drill-through

**CHART-006 — PARTIAL/FAIL.** *Medium.*
**S-58 — PARTIAL.** `sankey.html:588` sizes the chart `height = Math.max(500, links.length * 8)` so it grows with data ✓ and takes its width from `container.clientWidth || 900` ✓. But there is no compact layout at any width (LAYOUT-003), so at 375px the four columns, their labels (`344-349`), and the node labels are compressed into an illegible strip.
**S-68 — PARTIAL/FAIL.** The SVG starts at a fixed `height="500"` (`traceability.html:314`) and is then overridden to `Math.max(500, maxLayerSize * 76 + 80)` ✓ (`986-988`), so it grows vertically with the graph. But the **layout width is a hard 900px** — `layerCount * 180 + (layerCount - 1) * 60` (`992-993`) — positioned at `startX = Math.max(40, (width - totalW) / 2)` (`993`). On any viewport narrower than ~980px the graph exceeds its container from the first render and only pan/zoom recovers it.
**FAIL on label legibility:** link quantity labels are `font-size: 10` (`1051`), confidence glyphs `font-size: 9` (`1063`), and node quantity badges `font-size: 9` (`1131`) — all below any operational minimum, and all shrink further under a "Fit" that scales a 900px graph into a narrower viewport. Cross-referenced under ACCESS-006.
**S-02 — PARTIAL.** `mini-calendar.css:63-74` — day cells are 10px tall with 8px text, dropping to 9px tall at ≤1050px (`129-133`).
**PASS:** the Today So Far tile, the Process Flow summary strip, and the scheduler KPI strip are all correctly sized as glanceable tiles ✓.

**CHART-008 — PARTIAL/FAIL.** *Low/Contextual, Situational idea.*
**S-58 — FAIL.** `sankey.html:667-696` — hovering a link shows *"Ingredient → Line: 12,400 lb"* and hovering a node shows its total ✓ (macro and mid), but **neither is clickable**: there is no `.on('click')` anywhere in the render. From a flow of 12,400 lb there is no path to the transactions behind it.
**S-68 — PARTIAL.** `onNodeClick` (`traceability.html:1177-1194`) puts the node's lot code into the search box and auto-selects the direction ✓ — a drill *sideways* into another trace. It does not run the trace (the user must press Trace) and it does not reach the underlying transaction records; the tooltip shows *"Transaction #1284"* as inert text (`1163`).
**S-08, S-09 — PARTIAL.** A day card drills to a per-family made/packed breakdown ✓ (`dashboard.js:854-888`) — macro to mid — but not to the lots behind the day. The rule's example is *"tap-a-bar to see the lots that made up that day."*
**S-06 — PARTIAL.** The Today So Far tile has no click handler at all (`renderTodayTile`, `238-284`); its numbers are terminal. CHART-002's example describes exactly the missing drill: *"Today So Far tile shows one number and a sparkline; tapping opens the full chart with per-item breakdown."*
**S-02 — FAIL.** Ship-date dots are `<div>`s with no click handler (`mini-calendar.js:58`); a day showing three open orders offers no way to see which three.

---

### CHART-001, CHART-002, CHART-003 — passing rules worth recording

**CHART-001 — PASS app-wide.** *High.* Both charts answer a question a table could not:
- **Sankey** shows material moving Ingredients → Lines → Finished Goods → Customers — a four-stage flow whose whole content is the relationships and their relative magnitudes ✓
- **Trace graph** shows a supply-chain graph whose content is the connectivity and its gaps ✓

And, notably, **nothing in the product is charted merely to display numbers**. Today So Far is a number tile (`dashboard.js:238-284`), the Process Flow summary is three numbers (`process-flow.html`), the scheduler KPI strip is five numbers (`scheduler:3-9`), and the Production Calendar is a set of day cards with counts — none of them dressed up as a chart. The rule's failure mode (*"Don't chart a list of open SOs"*) does not occur anywhere.

*Observation, not a finding:* there is **no trend chart anywhere in the product**. The rule's example of a legitimate chart — *"Coconut utilization over 10-day cadence → line chart"* — has no equivalent, and the Production Calendar shows five discrete days as cards rather than a trend line. Blubber has no way to see whether output is rising or falling.

**CHART-002 — PASS.** *High.*
- **S-68 — exemplary.** `traceability.html:744-772` caps the forward-trace fan-out at 5 of N batches when `!fullChainMode`, renders a *"... and N more"* node, and reveals the "Show full chain" control (`771`, `312`, `1214-1218`). Simple default, richer mode on demand — exactly the rule ✓
- **S-58 — PASS.** Three top-N controls (products, customers, ingredients) let the viewer choose the level of detail (`sankey.html:306-330`) ✓

**CHART-003 — PASS, and this is a genuine strength.** *Medium.* Both novel visualizations carry an always-visible key:
- **Sankey** — a four-swatch legend (`sankey.html:338-343`) **plus** column labels naming each stage (`344-349`) ✓
- **Trace graph** — an eight-item legend covering six node types **and** two edge types, with the dashed-line convention shown (`traceability.html:291-300`) ✓
- **Scheduler board** — a constraint-key legend bar with six swatches and a plain-language instruction line, *"Click any cell to pin a quantity or crew · click a weekend header to add overtime"* (`scheduler:14-24`) ✓

The rule's test — *"Would an operator who has never seen this chart type understand it without help?"* — is met on all three by construction.

---

## N/A register

| Rule | Screens | Reason |
|---|---|---|
| DRAG-002 … DRAG-011 | **All 91** | **No drag-and-drop interaction exists in this product** — no `draggable`, no `dragstart`/`drop` handler, no `DataTransfer`, no sortable library, no pointer-based reorder. |
| SEARCH-001…008 | Modals, confirmations, states, legends, tooltips, print views, and the scheduler | No search surface, and no record type addressable from the screen. |
| SEARCH-006 | All screens | **No search history is stored anywhere** — there are no recents in the dashboard, and no authentication exists (`dashboard.js:1882` — one shared API key), so per-user history is not expressible. Traceability's recent-lots strip is **system** activity derived from the last 100 transactions (`traceability.html:374-423`), not personal history, so it leaks nothing user-specific; marked PARTIAL there only for the absence of a clear action. |
| DATA-001…012 | Toolbars, modals, confirmations, states, legends, chrome | No list or table on the screen. |
| DATA-008, DATA-009 | Screens with no table, and all Tier B screens where the rule is Medium | Nothing to sort or resize. |
| DATA-011 | All dashboard screens | Order is system-determined everywhere (date, name, status, FIFO) — explicitly exempted by the rule. |
| CHART-001…008 | All screens except the 15 listed in Matrix C | No chart, diagram, or data-visual on the screen. |

## Unverifiable from code

| Rule | Screens | What a browser check must confirm |
|---|---|---|
| DATA-004 | S-63, S-68, S-83, S-87 | The rendered width at which each truncation actually bites, and whether two real product names in the current catalogue clip to the same string. The CSS and the `truncate()` limits are verified; which records collide is a data question. |
| DATA-006 | S-10…S-13 | How each browser recovers the nested-`<tbody>` markup (`dashboard.js:930`, `1019`, `1115`) and therefore what `tr:nth-child(even)` actually shades. The invalidity is verified in code; the rendered result is parser-dependent. |
| DATA-012 | S-25, S-42 | Whether `getLocalDateFromISO` (`dashboard.js:1978-1983`) produces a wrong weekday in practice — it will for any viewer outside America/New_York, but the office's actual browser timezone settings need confirming. |
| CHART-005 | S-58, S-68 | What a screen reader actually announces for each SVG. The absence of ARIA is verified; the browser's default SVG handling is not. |
| CHART-006 | S-58, S-68, S-02 | Rendered legibility of the 8–10px chart labels at real device pixel ratios and after a "Fit" transform. |
| SEARCH-007 | S-17, S-18 | That `display: none` rows (`dashboard.css:822`) are genuinely skipped by the browser's find-in-page in Chrome and Safari — standard behaviour, but worth confirming given the 96-row consequence. |

---

*End of 04-search-data-drag-chart.md*

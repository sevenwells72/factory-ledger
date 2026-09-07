# 05 — Audit: Accessibility, Icons & Cross-Cutting (ACCESS, ICON, OTHER)

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) §10 (ACCESS-001…010), §15 (ICON-001…010), §16 (OTHER-001…011) — 31 rules
**Screens:** [00-screen-inventory.md](00-screen-inventory.md) — S-01…S-91
**Status:** Audit findings. **No application code was modified.**

---

## How to read this

Statuses and tiering are as defined in [01-nav-layout.md](01-nav-layout.md): **P** PASS · **PA** PARTIAL · **F** FAIL · **N** N/A · **U** UNVERIFIABLE-FROM-CODE · **·** out of tier scope.

### Rule importance (drives Tier B scope)

- **Critical:** ACCESS-006, ACCESS-008
- **High:** ACCESS-001, 002, 003, 005, 007, 010; ICON-001, 004; OTHER-003, 004, 006, 010
- **Medium:** ACCESS-004, 009; ICON-002, 003, 005; OTHER-001, 002, 005
- **Low/Contextual:** ICON-006, 007, 008, 009, 010; OTHER-007, 008, 009, 011

### Product-wide facts established for this group

Four facts, each verified by grep across `dashboard/*.html`, `dashboard/*.css`, `dashboard/*.js`, and `dashboard/scheduler/*.html`, set the scope for several rules and are stated once here rather than repeated per screen:

1. **Every `font-size` in the product is in `px`** — over 130 declarations in `dashboard.css` alone, plus every declaration in the four standalone HTML files. No `rem`, no root `font-size`. Drives ACCESS-001 and ACCESS-006.
2. **There are no raster images anywhere** — no `<img>` tag, no `background-image`, no image asset of any kind. Every "icon" is a Unicode character, a CSS shape, or (once) an inline SVG path. Drives ICON-005 and makes ICON-009 and ICON-010 N/A.
3. **No permission of any kind is ever requested** — no `getUserMedia`, no `geolocation`, no `Notification.requestPermission`. Makes OTHER-002 N/A.
4. **No sub-Regular font weight is used anywhere** — no `100`, `200`, `300`, or `lighter`. **ACCESS-007 passes outright across all 91 screens** and is recorded once below rather than in the matrix.

---

## Matrix A — Accessibility & Readability (ACCESS-001…010)

`006` and `008` are Critical. **ACCESS-007 is PASS on every screen** and is omitted from the columns.

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 008 | 009 | 010 |
|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | PA | **F** | P | P | PA | PA | PA | P | **P** |
| S-02 | Mini-calendar strip | **F** | **F** | PA | P | PA | **F** | **F** | P | **P** |
| S-03 | App header | **F** | **F** | PA | P | PA | PA | **F** | P | **F** |
| S-04 | Global search + results | **F** | **F** | **F** | PA | PA | PA | PA | P | N |
| S-05 | Tab bar (7 tabs) | **F** | **F** | P | P | PA | PA | **F** | P | N |
| S-06 | Today So Far tile | PA | **F** | P | P | PA | PA | P | P | N |
| S-07 | Today So Far — error/retry | PA | **F** | P | **P** | PA | PA | P | P | N |
| S-08 | Production Calendar | **F** | **F** | PA | P | PA | **F** | PA | P | **F** |
| S-09 | Calendar day detail panel | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-10 | Finished Goods panels | **F** | **F** | **F** | P | PA | PA | **F** | P | **F** |
| S-11 | FG per-product lot rows | **F** | **F** | **F** | P | PA | PA | **F** | P | **F** |
| S-12 | Batch Inventory | **F** | **F** | **F** | P | PA | PA | **F** | P | **F** |
| S-13 | On-Hand Ingredients | **F** | **F** | **F** | P | PA | PA | **F** | P | **F** |
| S-14 | Recent Entries feed | PA | **F** | P | **P** | PA | PA | PA | P | P |
| S-15 | Recent Entries states | P | **F** | P | **P** | PA | P | P | P | P |
| S-16 | Daily Entries + toolbar | **F** | **F** | PA | P | PA | PA | **F** | PA | P |
| S-17 | Shipping log | **F** | **F** | **F** | P | PA | PA | **F** | PA | **F** |
| S-18 | Receiving log | **F** | **F** | **F** | P | PA | PA | **F** | PA | **F** |
| S-19 | Notes toolbar | PA | **F** | P | P | PA | PA | PA | P | P |
| S-20 | Notes list (cards) | PA | **F** | PA | P | PA | **F** | **F** | P | **F** |
| S-21 | Notes empty state | P | **F** | P | P | PA | P | P | P | PA |
| S-22 | Note create/edit modal | PA | **F** | PA | PA | PA | PA | PA | P | P |
| S-23 | Delete-note confirm | P | **F** | P | **F** | N | P | P | P | N |
| S-24 | Orders toolbar | **F** | **F** | PA | P | PA | PA | PA | P | N |
| S-25 | Orders list table | **F** | **F** | **F** | **F** | PA | **F** | **F** | **F** | **F** |
| S-26 | Dispatch Queue mode | **F** | **F** | **F** | **F** | PA | **F** | **F** | **F** | **F** |
| S-27 | Orders expandable lines | **F** | **F** | PA | **F** | PA | **F** | **F** | **F** | **F** |
| S-28 | Factory Ready toggle+note | **F** | **F** | PA | P | PA | **F** | **F** | P | P |
| S-29 | Orders empty state | P | **F** | P | P | PA | P | P | P | PA |
| S-30 | Order Detail header/KPI | PA | **F** | P | P | PA | PA | PA | P | P |
| S-31 | Order Detail line table | **F** | **F** | PA | **F** | PA | **F** | **F** | **F** | P |
| S-32 | Per-line inventory expander | PA | **F** | P | P | PA | PA | PA | P | P |
| S-33 | Order Detail edit mode | **F** | **F** | PA | **F** | PA | **F** | **F** | **F** | P |
| S-34 | Edit-locked notice | P | **F** | P | **P** | PA | P | P | P | N |
| S-35 | SO status change confirm | P | **F** | P | **P** | N | P | P | P | N |
| S-36 | Reservations / allocation | PA | **F** | P | **P** | PA | **F** | PA | P | P |
| S-37 | Allocation history table | **F** | **F** | PA | P | PA | **F** | **F** | PA | P |
| S-38 | Release reservation confirm | P | **F** | P | **P** | N | P | P | P | N |
| S-39 | Shipping capacity preview | PA | **F** | P | P | PA | **F** | PA | P | P |
| S-40 | Order Detail notes card | P | **F** | P | P | PA | P | P | **P** | N |
| S-41 | Expected Receipts toolbar | **F** | **F** | PA | **P** | PA | PA | PA | P | N |
| S-42 | Expected Receipts table | **F** | **F** | PA | P | PA | PA | **F** | PA | P |
| S-43 | ER row actions | **F** | **F** | P | **P** | PA | PA | PA | P | P |
| S-44 | ER empty state | P | **F** | P | P | PA | P | P | P | PA |
| S-45 | ER create/edit modal | PA | **F** | **F** | PA | PA | PA | PA | P | P |
| S-46 | Supplies header + Request | P | **F** | P | **P** | PA | P | P | P | N |
| S-47 | Supplies sub-tabs | P | **PA** | **P** | P | PA | PA | PA | P | **P** |
| S-48 | Supplies search field | P | **PA** | **P** | P | PA | P | P | P | **P** |
| S-49 | Supplies inventory table | PA | **PA** | **P** | P | PA | PA | PA | P | **P** |
| S-50 | Supply lot / incoming detail | PA | **PA** | P | P | PA | **F** | PA | P | P |
| S-51 | Supply Requests list | PA | **F** | PA | P | PA | PA | PA | P | P |
| S-52 | Supply Requests feedback | P | **P** | P | **P** | PA | P | P | P | N |
| S-53 | Request Supply modal | PA | **PA** | PA | **P** | PA | P | P | P | **P** |
| S-54 | Lot Detail side panel | PA | **F** | PA | P | PA | PA | PA | P | P |
| S-55 | Lot disambiguation | PA | **F** | P | P | **F** | PA | **F** | P | P |
| S-56 | Product Detail panel | PA | **F** | **F** | P | **F** | **F** | **F** | P | P |
| S-57 | Sankey controls bar | **F** | **F** | P | P | PA | PA | PA | P | P |
| S-58 | Sankey chart + legend | **F** | **F** | **F** | P | PA | **F** | PA | P | **F** |
| S-59 | Sankey banner / loading | PA | **F** | P | P | PA | P | PA | P | P |
| S-60 | Summary strip | P | **F** | P | P | PA | P | P | P | P |
| S-61 | Production lines grid | P | **F** | P | P | PA | PA | PA | P | P |
| S-62 | Error / stale banners | P | **F** | P | **P** | PA | P | PA | P | P |
| S-63 | Lot search + type-ahead | **F** | **F** | **F** | P | PA | PA | PA | P | **F** |
| S-64 | Trace direction + Trace | **F** | **F** | P | P | PA | P | PA | P | P |
| S-65 | Recent lots strip | **F** | **F** | **F** | P | PA | **F** | PA | P | P |
| S-66 | Trace legend | P | **F** | P | P | PA | **F** | PA | P | **P** |
| S-67 | Status bar | P | **F** | PA | **P** | PA | P | PA | P | P |
| S-68 | Trace graph + zoom | **F** | **F** | **F** | P | PA | **F** | PA | P | **F** |
| S-69 | Node tooltip | **F** | **F** | **F** | P | PA | **F** | PA | P | N |
| S-70 | Trace detail + exports | PA | **F** | P | P | PA | **F** | PA | P | P |
| S-71 | Print audit report | PA | **F** | P | P | PA | **F** | **U** | P | P |
| S-72 | Topbar KPI strip | **F** | **F** | P | **P** | PA | **F** | PA | P | PA |
| S-73 | Topbar actions | **F** | **F** | P | P | PA | PA | PA | P | P |
| S-74 | Legend bar | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-75 | Delta panel | **F** | **F** | P | P | PA | **F** | **F** | P | P |
| S-76 | Settings panel (form) | **F** | **F** | P | P | PA | **F** | PA | P | **P** |
| S-77 | FG on hand (disclosure) | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-78 | Bulk-bin WIP (disclosure) | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-79 | Product catalog (form) | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-80 | Orders in / plans out | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-81 | CSV import result | **F** | **F** | P | **P** | PA | **F** | PA | P | P |
| S-82 | "How this works" | **F** | **F** | P | P | PA | PA | PA | P | P |
| S-83 | Production board table | **F** | **F** | **F** | P | PA | PA | PA | P | **P** |
| S-84 | Board cell copy / more | **F** | **F** | **F** | P | PA | **F** | PA | P | **P** |
| S-85 | Schedule panel | **F** | **F** | P | P | PA | PA | PA | P | **P** |
| S-86 | Pin modal | **F** | **F** | PA | **P** | PA | PA | PA | P | P |
| S-87 | Order book | **F** | **F** | PA | P | PA | **F** | PA | P | **P** |
| S-88 | Add order line form | **F** | **F** | P | P | PA | **F** | PA | P | P |
| S-89 | Add-order validation alert | P | **F** | P | **F** | N | P | P | P | N |
| S-90 | Order book empty state | **F** | **F** | P | P | PA | PA | PA | P | P |
| S-91 | Scheduler print view | PA | **F** | P | P | PA | **F** | **U** | P | P |

## Matrix B — Icons & Imagery (ICON-001…010)

`009` and `010` are **N** on all 91 screens (no images, no attachments) and are omitted from the columns.

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 |
|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | P | PA | P | P | P | P | U | N |
| S-02 | Mini-calendar strip | P | P | P | P | PA | P | U | N |
| S-03 | App header | P | **F** | PA | PA | PA | P | U | **P** |
| S-04 | Global search + results | N | N | N | PA | N | N | N | N |
| S-05 | Tab bar (7 tabs) | N | N | N | N | N | N | N | N |
| S-06 | Today So Far tile | N | N | N | N | N | N | N | N |
| S-07 | Today So Far — error/retry | N | N | N | N | N | N | N | N |
| S-08 | Production Calendar | P | P | P | P | P | P | U | N |
| S-09 | Calendar day detail panel | N | N | N | N | N | N | N | N |
| S-10 | Finished Goods panels | P | PA | P | PA | PA | P | U | **P** |
| S-11 | FG per-product lot rows | N | N | N | N | N | N | N | N |
| S-12 | Batch Inventory | N | N | N | N | N | N | N | N |
| S-13 | On-Hand Ingredients | P | PA | P | PA | PA | P | U | **P** |
| S-14 | Recent Entries feed | N | N | N | N | N | N | N | N |
| S-15 | Recent Entries states | N | N | N | N | N | N | N | N |
| S-16 | Daily Entries + toolbar | P | PA | P | PA | PA | P | U | **P** |
| S-17 | Shipping log | P | PA | P | PA | PA | P | U | **P** |
| S-18 | Receiving log | P | PA | P | PA | PA | P | U | **P** |
| S-19 | Notes toolbar | N | N | N | N | N | N | N | N |
| S-20 | Notes list (cards) | PA | **F** | PA | **F** | PA | P | U | N |
| S-21 | Notes empty state | PA | **F** | PA | P | **F** | P | U | N |
| S-22 | Note create/edit modal | P | P | PA | P | P | P | U | N |
| S-23 | Delete-note confirm | N | N | N | N | N | N | N | N |
| S-24 | Orders toolbar | N | N | N | N | N | N | N | N |
| S-25 | Orders list table | P | PA | PA | PA | PA | P | U | **P** |
| S-26 | Dispatch Queue mode | P | PA | PA | PA | PA | P | U | **P** |
| S-27 | Orders expandable lines | P | PA | PA | PA | PA | P | U | **P** |
| S-28 | Factory Ready toggle+note | P | P | P | P | P | P | U | N |
| S-29 | Orders empty state | PA | **F** | PA | P | **F** | P | U | N |
| S-30 | Order Detail header/KPI | P | P | P | P | P | P | U | N |
| S-31 | Order Detail line table | N | N | N | N | N | N | N | N |
| S-32 | Per-line inventory expander | N | N | N | N | N | N | N | N |
| S-33 | Order Detail edit mode | N | N | N | N | N | N | N | N |
| S-34 | Edit-locked notice | N | N | N | N | N | N | N | N |
| S-35 | SO status change confirm | N | N | N | N | N | N | N | N |
| S-36 | Reservations / allocation | N | N | N | N | N | N | N | N |
| S-37 | Allocation history table | N | N | N | N | N | N | N | N |
| S-38 | Release reservation confirm | N | N | N | N | N | N | N | N |
| S-39 | Shipping capacity preview | N | N | N | N | N | N | N | N |
| S-40 | Order Detail notes card | N | N | N | N | N | N | N | N |
| S-41 | Expected Receipts toolbar | N | N | N | N | N | N | N | N |
| S-42 | Expected Receipts table | N | N | N | N | N | N | N | N |
| S-43 | ER row actions | N | N | N | N | N | N | N | N |
| S-44 | ER empty state | PA | **F** | PA | P | **F** | P | U | N |
| S-45 | ER create/edit modal | P | P | PA | P | P | P | U | N |
| S-46 | Supplies header + Request | N | N | N | N | N | N | N | N |
| S-47 | Supplies sub-tabs | N | N | N | N | N | N | N | N |
| S-48 | Supplies search field | N | N | N | N | N | N | N | N |
| S-49 | Supplies inventory table | P | P | P | PA | PA | P | U | **P** |
| S-50 | Supply lot / incoming detail | N | N | N | N | N | N | N | N |
| S-51 | Supply Requests list | PA | **F** | PA | PA | **F** | P | U | N |
| S-52 | Supply Requests feedback | N | N | N | N | N | N | N | N |
| S-53 | Request Supply modal | P | P | PA | P | P | P | U | N |
| S-54 | Lot Detail side panel | P | P | P | P | P | P | U | N |
| S-55 | Lot disambiguation | N | N | N | N | N | N | N | N |
| S-56 | Product Detail panel | P | P | P | P | P | P | U | N |
| S-57 | Sankey controls bar | N | N | N | N | N | N | N | N |
| S-58 | Sankey chart + legend | **F** | **F** | PA | PA | PA | P | U | N |
| S-59 | Sankey banner / loading | PA | **F** | PA | P | **F** | P | U | N |
| S-60 | Summary strip | N | N | N | N | N | N | N | N |
| S-61 | Production lines grid | P | **F** | P | P | **P** | P | U | N |
| S-62 | Error / stale banners | N | N | N | N | N | N | N | N |
| S-63 | Lot search + type-ahead | P | **F** | PA | **P** | PA | P | U | N |
| S-64 | Trace direction + Trace | P | P | P | P | P | P | U | N |
| S-65 | Recent lots strip | N | N | N | N | N | N | N | N |
| S-66 | Trace legend | **P** | P | P | **P** | P | P | U | N |
| S-67 | Status bar | P | P | P | P | P | P | U | N |
| S-68 | Trace graph + zoom | **P** | P | PA | **P** | PA | P | U | N |
| S-69 | Node tooltip | P | P | P | **P** | PA | P | U | N |
| S-70 | Trace detail + exports | P | **F** | PA | **P** | PA | P | U | N |
| S-71 | Print audit report | P | P | P | **P** | P | P | U | N |
| S-72 | Topbar KPI strip | P | P | P | P | P | PA | U | N |
| S-73 | Topbar actions | N | N | N | N | N | N | N | N |
| S-74 | Legend bar | **P** | P | P | P | P | P | U | N |
| S-75 | Delta panel | N | N | N | N | N | N | N | N |
| S-76 | Settings panel (form) | P | P | P | P | P | PA | U | N |
| S-77 | FG on hand (disclosure) | P | P | P | P | P | P | U | N |
| S-78 | Bulk-bin WIP (disclosure) | P | P | P | P | P | P | U | N |
| S-79 | Product catalog (form) | P | P | P | P | P | P | U | N |
| S-80 | Orders in / plans out | N | N | N | N | N | N | N | N |
| S-81 | CSV import result | N | N | N | N | N | N | N | N |
| S-82 | "How this works" | P | P | P | P | P | PA | U | N |
| S-83 | Production board table | P | P | P | P | P | P | U | **P** |
| S-84 | Board cell copy / more | PA | P | P | PA | PA | P | U | **P** |
| S-85 | Schedule panel | P | P | P | P | P | P | U | N |
| S-86 | Pin modal | P | P | P | P | P | PA | U | N |
| S-87 | Order book | **F** | P | P | **F** | PA | P | U | N |
| S-88 | Add order line form | N | N | N | N | N | N | N | N |
| S-89 | Add-order validation alert | N | N | N | N | N | N | N | N |
| S-90 | Order book empty state | N | N | N | N | N | N | N | N |
| S-91 | Scheduler print view | P | P | P | P | P | P | U | N |

## Matrix C — Cross-cutting (OTHER-001…011)

`002` is **N** on all 91 screens (no permission is ever requested) and is omitted from the columns.

| # | Screen | 001 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | PA | P | P | P | U | P | N | N | **F** | N |
| S-02 | Mini-calendar strip | PA | P | P | PA | U | P | N | N | **F** | P |
| S-03 | App header | PA | **F** | P | **F** | U | P | N | **F** | **F** | P |
| S-04 | Global search + results | PA | **F** | P | **F** | U | P | N | **F** | **F** | N |
| S-05 | Tab bar (7 tabs) | PA | **F** | P | **F** | U | P | N | **F** | **F** | N |
| S-06 | Today So Far tile | PA | P | P | PA | U | P | N | **F** | **F** | P |
| S-07 | Today So Far — error/retry | PA | P | P | P | U | **P** | N | N | **F** | P |
| S-08 | Production Calendar | PA | P | P | PA | U | PA | N | **F** | **F** | P |
| S-09 | Calendar day detail panel | PA | P | P | PA | U | P | N | **F** | **F** | P |
| S-10 | Finished Goods panels | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-11 | FG per-product lot rows | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-12 | Batch Inventory | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-13 | On-Hand Ingredients | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-14 | Recent Entries feed | PA | P | P | P | U | P | N | **F** | **F** | P |
| S-15 | Recent Entries states | PA | P | P | P | U | **P** | N | N | **F** | P |
| S-16 | Daily Entries + toolbar | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-17 | Shipping log | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-18 | Receiving log | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-19 | Notes toolbar | PA | P | P | P | U | P | N | **F** | **F** | P |
| S-20 | Notes list (cards) | PA | PA | P | PA | U | P | N | **F** | **F** | P |
| S-21 | Notes empty state | PA | P | P | P | U | PA | N | N | **F** | P |
| S-22 | Note create/edit modal | PA | PA | PA | PA | U | P | N | **F** | **F** | P |
| S-23 | Delete-note confirm | PA | PA | P | P | U | P | N | N | N | P |
| S-24 | Orders toolbar | PA | **F** | P | **F** | U | P | **P** | **F** | **F** | P |
| S-25 | Orders list table | PA | **F** | PA | **F** | U | P | N | **F** | **F** | P |
| S-26 | Dispatch Queue mode | PA | **F** | PA | **F** | U | P | N | **F** | **F** | P |
| S-27 | Orders expandable lines | PA | **F** | PA | **F** | U | P | N | **F** | **F** | P |
| S-28 | Factory Ready toggle+note | PA | **F** | PA | **F** | U | P | N | **F** | **F** | P |
| S-29 | Orders empty state | PA | P | P | P | U | PA | N | N | **F** | P |
| S-30 | Order Detail header/KPI | PA | **F** | P | **F** | U | P | N | **F** | **F** | P |
| S-31 | Order Detail line table | PA | **F** | P | **F** | U | P | N | **F** | **F** | P |
| S-32 | Per-line inventory expander | PA | P | P | PA | U | P | N | **F** | **F** | P |
| S-33 | Order Detail edit mode | PA | **F** | P | **F** | U | P | N | **F** | **F** | P |
| S-34 | Edit-locked notice | PA | **P** | P | P | U | **P** | N | N | **F** | P |
| S-35 | SO status change confirm | PA | PA | P | P | U | P | N | N | N | P |
| S-36 | Reservations / allocation | PA | PA | P | PA | U | **P** | N | **F** | **F** | P |
| S-37 | Allocation history table | PA | PA | P | **F** | U | P | N | **F** | **F** | P |
| S-38 | Release reservation confirm | PA | **P** | P | P | U | **P** | N | N | N | P |
| S-39 | Shipping capacity preview | PA | P | P | PA | U | **P** | N | **F** | **F** | P |
| S-40 | Order Detail notes card | PA | P | P | P | U | P | N | N | **F** | P |
| S-41 | Expected Receipts toolbar | PA | P | **P** | **F** | U | P | N | **F** | **F** | P |
| S-42 | Expected Receipts table | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-43 | ER row actions | PA | **P** | P | PA | U | **P** | N | **F** | **F** | P |
| S-44 | ER empty state | PA | P | P | P | U | PA | N | N | **F** | P |
| S-45 | ER create/edit modal | PA | PA | P | PA | U | P | N | **F** | **F** | P |
| S-46 | Supplies header + Request | PA | P | P | P | U | P | N | **F** | **F** | P |
| S-47 | Supplies sub-tabs | PA | P | P | **P** | U | P | N | **F** | **F** | P |
| S-48 | Supplies search field | PA | P | P | **P** | U | P | N | **F** | **F** | P |
| S-49 | Supplies inventory table | PA | P | P | **P** | U | P | N | **F** | **F** | P |
| S-50 | Supply lot / incoming detail | PA | P | P | **P** | U | P | N | **F** | **F** | P |
| S-51 | Supply Requests list | PA | PA | P | PA | U | P | N | **F** | **F** | P |
| S-52 | Supply Requests feedback | PA | P | P | P | U | **P** | N | N | **F** | P |
| S-53 | Request Supply modal | PA | **F** | P | PA | U | P | N | **F** | **F** | P |
| S-54 | Lot Detail side panel | **P** | P | P | PA | U | P | N | **F** | **F** | P |
| S-55 | Lot disambiguation | **P** | P | P | PA | U | P | N | **F** | **F** | P |
| S-56 | Product Detail panel | **P** | P | P | PA | U | P | N | **F** | **F** | P |
| S-57 | Sankey controls bar | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-58 | Sankey chart + legend | PA | P | P | **F** | U | PA | N | **F** | **F** | P |
| S-59 | Sankey banner / loading | PA | **F** | P | **F** | U | P | N | **F** | **F** | P |
| S-60 | Summary strip | PA | P | P | P | U | P | N | **F** | **F** | P |
| S-61 | Production lines grid | PA | P | P | P | U | P | N | **F** | **F** | P |
| S-62 | Error / stale banners | PA | **F** | P | P | U | **P** | N | **F** | **F** | P |
| S-63 | Lot search + type-ahead | **P** | P | P | PA | U | P | N | **F** | **F** | P |
| S-64 | Trace direction + Trace | **P** | P | P | PA | U | P | N | **F** | **F** | P |
| S-65 | Recent lots strip | **P** | PA | P | PA | U | P | N | **F** | **F** | P |
| S-66 | Trace legend | **P** | P | P | P | U | P | N | **F** | **F** | P |
| S-67 | Status bar | **P** | P | P | P | U | **P** | N | **F** | **F** | P |
| S-68 | Trace graph + zoom | **P** | P | P | **F** | U | P | N | **F** | **F** | P |
| S-69 | Node tooltip | **P** | P | P | **F** | U | P | N | **F** | **F** | P |
| S-70 | Trace detail + exports | **P** | P | P | PA | U | P | **P** | **F** | **F** | P |
| S-71 | Print audit report | **P** | P | P | P | U | P | **P** | N | **F** | P |
| S-72 | Topbar KPI strip | PA | P | **P** | **F** | U | P | N | **P** | **F** | P |
| S-73 | Topbar actions | PA | P | P | **F** | U | P | **P** | N | **F** | P |
| S-74 | Legend bar | PA | P | **P** | **F** | U | P | N | N | **F** | P |
| S-75 | Delta panel | PA | P | P | **F** | U | P | N | **P** | **F** | PA |
| S-76 | Settings panel (form) | PA | P | **P** | **F** | U | P | N | **P** | **F** | P |
| S-77 | FG on hand (disclosure) | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-78 | Bulk-bin WIP (disclosure) | PA | P | P | **F** | U | P | N | **F** | **F** | P |
| S-79 | Product catalog (form) | PA | P | **P** | **F** | U | P | N | **P** | **F** | P |
| S-80 | Orders in / plans out | PA | PA | P | **F** | U | P | **P** | **F** | **F** | P |
| S-81 | CSV import result | PA | P | **P** | **F** | U | **P** | N | N | **F** | P |
| S-82 | "How this works" | PA | P | P | **F** | U | **P** | N | **P** | **F** | P |
| S-83 | Production board table | PA | P | **P** | **F** | U | P | N | **P** | **F** | P |
| S-84 | Board cell copy / more | PA | P | P | **F** | U | **P** | N | **F** | **F** | P |
| S-85 | Schedule panel | PA | P | P | **F** | U | **P** | **P** | **F** | **F** | P |
| S-86 | Pin modal | PA | P | **P** | **F** | U | **P** | N | **P** | **F** | P |
| S-87 | Order book | PA | **F** | P | **F** | U | P | N | **P** | **F** | P |
| S-88 | Add order line form | PA | PA | P | **F** | U | P | N | **F** | **F** | P |
| S-89 | Add-order validation alert | PA | PA | P | **F** | U | PA | N | N | N | P |
| S-90 | Order book empty state | PA | P | P | **F** | U | P | N | N | **F** | P |
| S-91 | Scheduler print view | PA | P | P | P | U | P | N | N | **F** | P |

---

## Findings

---

### ACCESS-008 — Meet WCAG AA contrast in light, dark, and increased-contrast modes — **CRITICAL**

*Hard rule.* Ratios below are computed from the declared hex values using the WCAG relative-luminance formula. Where a colour is layered over a translucent fill, the fill was first composited over its parent surface.

#### 1. `--text-dimmed` fails AA in **both** themes — nine selectors

| Pair | Ratio | Target |
|---|---|---|
| `--text-dimmed #64748b` on `--surface #1e293b` (dark) | **3.07 : 1** | 4.5 : 1 |
| `--text-dimmed #64748b` on `--row-header #162032` (dark) | **3.09 : 1** | 4.5 : 1 |
| `--text-dimmed #94a3b8` on `--surface #ffffff` (light) | **2.56 : 1** | 4.5 : 1 |

**Evidence:** `dashboard.css:14` (dark) and `52` (light). Applied to: `.panel-count` (`647`), `.pallet-ratio` (`648`), `.chevron::after` (`652`), `.no-production` (`530`), `.ingredient-header-count` (`955`), `.note-meta` (`1088`), `.note-priority-badge.p-low` (`1072`), `.notes-empty`/`.orders-empty` icons (`1216`, `2173`), `.search-group-label` (`traceability.html:100`), `.mini-calendar-dow` (`mini-calendar.css:77`). All are rendered at 10–12px, which requires the 4.5:1 body target rather than the 3:1 large-text one.

`.note-meta` is the significant one — it carries the **due date**, including the overdue state.

#### 2. Three light-theme surfaces are effectively invisible

These hard-code dark-mode values with no light variant, so in the light theme they render pale-on-pale:

| Selector | Composited pair | Ratio | Evidence |
|---|---|---|---|
| **`.so-ready-pill`** | `#86efac` on `rgba(34,197,94,.12)` over `#ffffff` ≈ `#e4f8ec` | **≈ 1.27 : 1** | `dashboard.css:1576-1588` |
| **`.order-edit-message.success`** | `#86efac` on `rgba(52,211,153,.10)` over `#ffffff` | **≈ 1.27 : 1** | `dashboard.css:1885-1890` |
| `.order-edit-message.error` | `#fca5a5` on `rgba(248,113,113,.10)` over `#ffffff` | **≈ 1.73 : 1** | `dashboard.css:1891-1898` |

The first is the **"✓ READY" pill** — the signal the floor acts on — rendered at 10px. The second is the **"Header saved."** confirmation. In light mode a user toggles Factory Ready and sees nothing, and saves an order header and sees nothing.

#### 3. Four token-based pairs fail in the **default** (dark) theme

| Pair | Ratio | Where | Evidence |
|---|---|---|---|
| `.tab.active` — `--primary #3b82f6` on `--surface #1e293b` | **3.98 : 1** at 14px | the active section label | `dashboard.css:279`, `265-271` |
| `.date-overdue` — `--danger #ef4444` on `--surface` | **3.89 : 1** at 13px | overdue ship dates | `dashboard.css:1671` |
| `.date-overdue` on `--row-alt #1a2536` (striped rows) | **4.10 : 1** | same, on alternate rows | `dashboard.css:1317`, `1671` |
| `.lot-link` — `--primary` on `--row-header #162032` | **4.44 : 1** at 12px | lot codes in expanded lot rows | `dashboard.css:840-848`, `730-734` |

The light theme passes for all four (`--primary #2563eb` on white = 5.17 : 1; `--danger #dc2626` on white = 4.83 : 1). The failures are in the theme that ships by default (`dashboard.js:47`).

#### 4. A red/green pair of near-identical luminance

**S-75 — FAIL.** `scheduler:50` — `.d-better { color: var(--ok) }` and `.d-worse { color: var(--late) }`, both `font-weight: 700`.

| Colour | Relative luminance |
|---|---|
| `--ok #2e7d4f` | **0.158** |
| `--late #bf2f24` | **0.132** |

The two are separated by hue alone at essentially the same luminance — precisely what the rule prohibits: *"Avoid color pairs colorblind users can't separate (red/green of similar luminance)."* This is the panel Blubber reads to decide whether a plan change helped or hurt. Cross-referenced under FEEDBACK-011 in `03-feedback-error-notify.md`.

#### 5. A near-white-on-white control

**S-55 — FAIL.** `dashboard.js:1379` — `background: var(--bg-card, #fff)` on the lot-disambiguation buttons. `--bg-card` is defined nowhere in `dashboard.css`, so the `#fff` fallback applies while the text inherits `--text: #f1f5f9` from the panel — roughly **1.1 : 1**. Full write-up under LAYOUT-002 item 7 in `01-nav-layout.md`.

#### 6. No increased-contrast mode at all

The rule requires AA *"in light, dark, and increased-contrast modes."* There is no `@media (prefers-contrast: more)` block and no `forced-colors` handling in any of the five style sources (verified by grep).

#### What passes

Body and heading text is genuinely strong: `--text` on `--surface` is **12.6 : 1** (dark) and **17.9 : 1** (light); `--text-secondary` is **9.3 : 1**; `--text-muted` is **5.7 : 1** (dark) and **4.76 : 1** (light) ✓. Every **token-based** status chip clears the target comfortably — `.readiness-chip.severity-block` computes to **7.3 : 1** in light and the health badge to **6.4 : 1** in dark ✓. Nothing scrolls under a translucent bar: every sticky surface is opaque (`dashboard.css:138`, `256`, `2280`) ✓.

**The pattern is clean and the fix follows from it: every failure above is a hard-coded value or an undefined token. Every colour that resolves through the theme system passes.**

---

### ACCESS-006 — Fixed minimum and default text sizes; nothing operational below the minimum — **CRITICAL**

*Hard rule.* The standard's proposed floors: **desktop body 14–16 px, nothing operational below 12 px, 11 px captions only for non-operational metadata; mobile body ≥ 17 px.**

**`body { font-size: 14px }`** (`dashboard.css:83`) sits at the bottom of the desktop range ✓ — and there is **no mobile override**, so on Arturo's or Luz's phone the body is 14 px against a 17 px floor. **FAIL on the mobile default.**

**At least twenty-two selectors set operational text below the 12 px floor:**

| Size | Selector | What it carries | Evidence |
|---|---|---|---|
| **8 px** | `.mini-calendar-dow`, `.mini-calendar-day` | **day numbers and ship-date indicators** | `mini-calendar.css:63-74`; still 8px at ≤1050px (`129-133`) |
| 9 px | trace link confidence glyph | ✓ / ⚠ per link | `traceability.html:1063` |
| 9 px | trace node quantity | lb per node | `traceability.html:1131` |
| 10 px | `.readiness-chip`, `.readiness-chip-detail` | **why an order cannot ship** | `dashboard.css:1624`, `1628` |
| 10 px | `.dispatch-pill` | Dispatch Ready / Blocked | `dashboard.css:1594` |
| 10 px | `.so-ready-pill` | Factory Ready + who + when | `dashboard.css:1584` |
| 10 px | `.allocation-table td > span` | **the lot code of a lot-level reservation** | `dashboard.css:2055-2062` |
| 10 px | `.allocation-source`, `.allocation-status` | reservation source and state | `dashboard.css:2074` |
| 10 px | `.allocation-table th` | column headers | `dashboard.css:2046` |
| 10 px | `.allocation-field-hint`, `.allocation-form-note` | "Auto FIFO expires after 48 hours…" | `dashboard.css:1977-1982` |
| 10 px | `.production-detail-sku` | **SKU** | `dashboard.css:614-619` |
| 10 px | `.order-lines-table th` | column headers | `dashboard.css:1381-1391` |
| 10 px | `.supply-detail-field dt` | FIFO lot field labels | `dashboard.css:2381-2388` |
| 10 px | `.day-section-label` | "Made" / "Packed" | `dashboard.css:478-486` |
| 10 px | `.note-cat-badge`, `.note-priority-badge` | category and priority | `dashboard.css:1050-1070` |
| 10 px | `.preview-allocation-warning strong` | the warning code | `dashboard.css:2116` |
| 10 px | trace link quantity labels, `.tt-label` | lb per link | `traceability.html:1051`, `223` |
| 10 / 10.5 px | scheduler `.k-label`, `.k-sub`, `#deltapanel th` | KPI labels and delta headers | `scheduler:33`, `35`, `48` |

**Plus fourteen selectors at 11 px carrying operational values**, where the standard permits 11 px only for *non-operational metadata*: `.ship-by-weekday` (the weekday of a ship date, `1533-1538`), `.note-meta` (the due date, `1082-1090`), `.readiness-none` ("No blockers", `1649-1652`), `.readiness-caption` (`1653-1659`), `.readiness-metrics` (allocated/shortage totals, `1800-1806`), `.pallet-secondary` (pallet counts, `1436-1444`), `.er-summary` (`2212`), `.er-over` (over-receipt quantity, `2218`), `.badge` (case and batch counts, `710-719`), `.so-badge` (order status, `1556-1565`), `.supplies-count-badge` (`2306-2320`), `.production-detail-count` (`620-627`), `.late-lag` (`810-816`), and every table `<th>` in the product (`679`, `755`, `1298`).

The mini-calendar is the sharpest case: **8 px** day numbers, in a **10 px**-tall cell, on a strip that is one of only two attention signals in the app chrome (NOTIFY-011). The rule's context clause — *"arm's length, gloves, dust, motion, glare"* — is not survivable at 8 px on any device.

**Suggested fix:** define `--text-xs: 12px` as the hard floor and `--text-caption: 11px` for metadata only, then raise the twenty-two sub-12px operational selectors to 12 px and the mini-calendar to a legible size (or replace it with a compact count). Add a mobile block raising `body` to 17 px. Pair with the `rem`-based scale under ACCESS-001.

---

### ACCESS-001 — Honor user text-size, bold-text, and light/dark settings; survive the largest text size — **HIGH**

*Hard rule.* Three clauses; one passes outright, two fail.

**Light/dark — PASS, and it is exemplary.** `dashboard.css:4-75` defines a complete, paired token set for both themes; `initTheme` / `toggleTheme` set `data-theme` on the root and persist the choice in `localStorage` (`dashboard.js:44-63`); a toggle sits in the app header (`index.html:40`). This is the right architecture and it works. *(The nine hard-coded values that escape it are an ACCESS-008 finding, not an ACCESS-001 one.)*

**User text size — FAIL.** Every `font-size` declaration in the product is in `px` — over 130 in `dashboard.css` and every declaration in the four standalone HTML files (verified by grep). There is no root `font-size`, no `rem`, and no `em` sizing. A browser text-size preference scales the root font; with nothing sized relative to the root, **it has no effect on any text in the product**. The rule's requirement — *"Text scales with system/browser text-size settings (relative units)"* — cannot be met by the current stylesheet.

**200 % browser zoom — FAIL.** Zoom does scale `px`, so text grows ✓ — but zoom also halves the effective CSS viewport. A 1440 px window at 200 % reports 720 px, crossing the 768 px breakpoint and triggering two failures at once:
- the sticky tab bar overlaps the wrapped app header (`dashboard.css:140-142`, `259-260`, `2181-2189` — LAYOUT-003), so the search field and Refresh disappear behind the tabs;
- the six tables with no horizontal-scroll wrapper (`index.html:240`, `283`; `dashboard.js:918`, `1002`, `1106`, `1191`, `1244`, `1315` — LAYOUT-003) push the page into horizontal scroll.

The rule names this case exactly: *"Desktop at 200% zoom doesn't turn tables into unusable overflow."*

**Columns do not reduce at larger sizes — FAIL.** There is no column-hiding logic anywhere; the 11-column orders table and the 9–10 column readiness table render every column at every size.

**Icons do scale with text — PASS, by happy accident.** Because every glyph in the product is a Unicode text character rather than a fixed-size SVG (see ICON-005), all of them inherit `font-size` and grow with the text around them ✓.

**Bold-text preference — N/A.** There is no CSS media feature for the OS bold-text setting on the web platform, so this clause is not expressible here.

---

### ACCESS-003 — Every task is operable by touch, keyboard, and pointer — **HIGH**

**FAIL. Several complete workflows cannot be performed from the keyboard at all.** *Hard rule.*

Six interactive element types carry click handlers with no `tabindex`, no role, and no key handler, so they are invisible to keyboard navigation:

| Element | What it does | Evidence |
|---|---|---|
| `.lot-link` (a `<span>`) | **opens the lot panel — the product's main drill path** | `dashboard.js:935`, `1033`, `1120`, `1211`, `1264`; bound at `1640-1647` |
| `tr.expandable` | expands a product's lot breakdown | `dashboard.js:923`, `1007`, `1111`, `1196`, `1249`; bound at `1624-1638` |
| `.order-row` | opens an order's detail page | `dashboard.js:2319`; bound at `2340-2347` |
| `.search-item` (a `<div>`) | **selects a global-search result** | `dashboard.js:1530`, `1537`, `1544`, `1551`; bound at `1563-1612` |
| `.er-product-option` (a `<div>`) | selects a product in the ER modal | `dashboard.js:3526`; bound at `3528-3533` |
| `.product-lot-row` | opens a lot from the product panel | `dashboard.js:1465`, `1478`; bound at `1496-1500` |

Traceability adds `.search-item` (a `<div>` with an inline `onclick`, `traceability.html:463`) and `.lot-pill` (a `<span>`, `411-421`). The scheduler adds `td.cell` (opens the pin modal, `:114`) and `.otbtn` (a `<span>`, `:1345`).

The consequence is not partial: **a keyboard-only user cannot open a lot, cannot expand an inventory row, cannot select a search result, and cannot open an order.** Those are the four most-used interactions in the dashboard.

**No modal traps focus** (`dashboard.js:1785-1802`, `3542-3571`, `3993-4003`), and the Note modal sets no initial focus at all. Full treatment under INPUT-006 in `02-actions-input-touch.md`.

**Touch-only completion — PARTIAL.** Three values are reachable on desktop and unreachable on touch, because they are `title`- or hover-only: the full note text in an ER row (`title` on `.er-notes`, `dashboard.js:3429`), the blocker detail in the orders list (`title` on `.readiness-chip`, `1944`), and the full node label in the trace graph (the `mouseenter` tooltip, `traceability.html:1084`, `1153-1168`).

**Scan / paste / pick — PARTIAL.** Pick is well provided for supplier, product, lot, and requester (INPUT-011) ✓; paste works natively but is never extracted (INPUT-020); **scan is absent entirely** (INPUT-016) — no `capture`, no `BarcodeDetector`, no scanner-input handling.

**S-47, S-48, S-49 — PASS. The reference implementation.** `dashboard.js:3849` gives the supplies row `tabindex="0"`, `role="button"`, and `aria-expanded`; `3863-3868` handles Enter and Space with `preventDefault`; `dashboard.css:2300-2304` gives it a `:focus-visible` outline. This is the one custom control in the product that is fully operable by keyboard, and it shows the pattern the other six need.

---

### ACCESS-010 — Every icon-only control and custom icon has an accessible text name — **HIGH**

*Hard rule.* Six icon-only controls compute their accessible name from the glyph rather than from a label.

A `title` attribute does **not** override text content when the accessible name is computed. Where a button's only content is a glyph character, that glyph *is* the name a screen reader announces.

| Control | Markup | Announced as | Evidence |
|---|---|---|---|
| Notes **Edit** | `<button class="note-action-btn edit" title="Edit">✎</button>` | **"✎"** | `dashboard.js:1732` |
| Notes **Delete** | `<button class="note-action-btn delete" title="Delete">✕</button>` | **"✕"** | `dashboard.js:1733` |
| Order expand | `<button class="order-expand-toggle" title="Show line items"><span class="order-expand-caret">▸</span></button>` | **"▸"** | `dashboard.js:2320` |
| Calendar previous | `<button id="cal-prev" class="btn-sm" title="Previous">←</button>` | **"←"** | `index.html:76` |
| Calendar next | `<button id="cal-next" class="btn-sm" title="Next">→</button>` | **"→"** | `index.html:78` |
| Theme toggle | `<button id="theme-toggle" class="btn-theme" title="Toggle light/dark mode">☾</button>` | **"☾"** | `index.html:40`; swapped at `dashboard.js:62` |
| Graph zoom in / out | `<button class="graph-ctrl-btn" onclick="zoomIn()">+</button>` | **"+"** / **"−"** — no `title` either | `traceability.html:310-311` |

**Decorative glyphs that leak into their parent's name — PARTIAL.** `.chevron::after` uses CSS `content` so it is announced by some engines (`dashboard.css:650-657`), and `.order-expand-caret` is a plain `<span>` (`dashboard.js:2320`). Contrast `.supply-row-caret`, which correctly carries `aria-hidden="true"` (`dashboard.js:3850`) ✓.

**`.collapsible-header` has no role at all — FAIL.** `index.html:130`, `150`, `162`; `dashboard.js:910`, `1096` — a clickable `<div>` containing an `<h2>`/`<h3>` and a chevron, with no `role="button"`, no `tabindex`, and no `aria-expanded`. Four Activity panels and every Finished Goods and Ingredients panel header.

**PASS — the correct pattern already exists in six places:** all four `.btn-close` buttons carry `aria-label="Close"` (`index.html:341`, `402`, `450`, `512`) ✓; `#navToggle` carries `aria-label="Menu"` (`17`) ✓; both `.mini-calendar-nav` buttons carry `aria-label="Previous month"` / `"Next month"` (`mini-calendar.js:68`, `72`) ✓; and the scheduler labels its icon controls **with the record name included** — `aria-label="Copy schedule for Mon 8"` (`scheduler:1343`), `aria-label="Exclude SO-1234 Granola 25 LB from plan"` (`:1536`), `aria-label="Delete SO-1234 Granola 25 LB order line"` (`:1537`) ✓. That last set is the best accessible-naming in the product.

**S-58 — FAIL.** The Sankey SVG has no `role`, no `aria-label`, and no `<title>` (`sankey.html:350`) — see CHART-005 in `04-search-data-drag-chart.md`.

---

### ACCESS-002 — Accessibility is a baseline requirement, not a later fix — **HIGH**

**FAIL, as a process finding for the whole product.** *Hard rule.*

The audit test is literal: *"Is there an accessibility checklist applied to every new screen before ship?"* No such checklist exists in the repository — `docs/design/` contains only the standards document and this audit; `CLAUDE.md` has no accessibility section; there is no axe, Lighthouse, or pa11y configuration, and no CI check.

**The code shows the signature of a per-screen retrofit rather than a baseline.** Accessibility support is present and absent in a pattern that tracks how recently each surface was built:

| Surface | ARIA / keyboard support | Evidence |
|---|---|---|
| **Supplies** (S-47…S-53) | `role="tablist"` / `role="tab"` / `aria-selected` (`index.html:300-309`), `aria-expanded` + `role="button"` + `tabindex` + Enter/Space handler on rows (`dashboard.js:3849`, `3863-3868`), `:focus-visible` styling (`dashboard.css:2300-2304`), `aria-live` on two regions (`index.html:316`, `331`), `role="status"` (`330`), `role="alert"` (`388`), a `.sr-only` label (`312`), and `aria-label` on every count badge, updated on every render (`dashboard.js:3677`, `3926`) | comprehensive |
| **Recent Entries** (S-14, S-15) | `aria-labelledby`, `role="status"`, `aria-live` (`index.html:111`, `120-121`), and the product's only 44 px hit targets (`dashboard.css:290`, `316`) | good |
| **Production Calendar** (S-08, S-09) | `aria-expanded` + `aria-controls` on real `<button>` day cards (`dashboard.js:764`), `aria-live` on the detail region (`index.html:83`), `:focus-visible` (`dashboard.css:436-441`) | good |
| **Orders, Expected Receipts, Notes, inventory, lot panels** | clickable `<tr>` and `<span>` with no roles, no tabindex, no keyboard handlers, no focus styles; icon-only buttons named by their glyph | little |

`.sr-only` is defined once (`dashboard.css:2244-2254`) and used **once** (`index.html:312`).

Recording this as a FAIL against every screen is deliberate: the rule is about the process, and its absence is what produced the twelve substantive accessibility findings in this file.

**Suggested fix:** a short pre-ship checklist covering the eight things this audit found repeatedly — 44 px targets, focus-visible, accessible name on every icon-only control, keyboard operability of every click target, contrast in **both** themes, no text below 12 px, colour never the sole carrier, and a label on every field — applied to each new screen and to the Supplies tab's own patterns as the reference.

---

### ACCESS-005 — A defined type scale with few named styles — **HIGH**

**PARTIAL across all screens.**

**What is right — and it is more than half the rule.** One primary typeface via a system stack (`--font`, `dashboard.css:39`) plus one monospace stack for identifiers and quantities (`--mono`, `40`) ✓ — exactly the "one typeface plus a tabular-figure style" the rule asks for, and the split is applied correctly: lot codes (`735`, `844`, `905`), SKUs (`1451`, `617`), order numbers (`1546`, `1766`), and every numeric cell (`704`, `1514`, `2052`) use `--mono`. `font-variant-numeric: tabular-nums` is applied on ten selectors (`706`, `718`, `305`, `370`, `1405`, `1515`, `1519`, `1868`, `2137`, `2342`) so digits align in columns ✓. This is genuinely well done.

**What is missing — the scale itself.** There is no named text-style set: no `--text-title` / `--text-body` / `--text-caption` tokens, no utility classes, no documented scale. Sizes are assigned ad hoc per component, producing **thirteen distinct values** — 8, 9, 10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 32 px — and, for one structural level, seven different treatments:

| Heading | Size | Evidence |
|---|---|---|
| `.recent-entries-header h2` | 22 px | `dashboard.css:288` |
| `.supplies-page-header h2` | 20 px | `dashboard.css:2263` |
| `.production-detail-header h3` | 17 px | `dashboard.css:556-561` |
| `.panel-header h3` | 15 px | `dashboard.css:878` |
| `.order-section-heading h3` | 15 px | `dashboard.css:1928-1932` |
| `.section > h2`, `.section-header h2` | 15 px | `dashboard.css:337-341` |
| `.collapsible-header h2, h3` | 14 px | `dashboard.css:642-646` |

**Non-standard weights.** `font-weight: 650` (`dashboard.css:560`) and `750` (`580`) are synthetic values that the system font stack will round to 700, so they add no hierarchy while implying two extra steps. The rule is explicit that *"Bold is an extra hierarchy step within a style, not a new style."*

**Cross-file divergence.** Each standalone page redefines its own `--font` and `--mono` (`traceability.html:25-26`, `sankey.html`, `process-flow.html`) and the scheduler uses a different stack entirely with `ui-monospace, SFMono-Regular, Menlo, Consolas` for `.num` (`scheduler:22`, `25`) and a root `font-size: 14px` (`:17`).

---

### ACCESS-007 — Use Regular through Bold weights only — **HIGH**

**PASS on all 91 screens.** *Hard rule.*

Verified by grep across `dashboard.css`, `mini-calendar.css`, and all five HTML files: there is **no** `font-weight: 100`, `200`, `300`, or `lighter` anywhere in the product. The lightest weight in use is `400` (`normal`), applied deliberately to de-emphasised text — `.label-hint` (`dashboard.css:1159`), `.readiness-caption` (`1658`), `.readiness-chip-detail` (`1630`), `.created-at-meta` (`783`), `.ingredient-header-count` (`956`), `.allocation-form select/input` (`1968`) — with **colour and size** carrying the de-emphasis, which is exactly what the rule prescribes: *"De-emphasize secondary text with color and size, not a lighter weight."*

Recorded here rather than in the matrix because the result is uniform and unambiguous.

---

### ACCESS-004, ACCESS-009 — copy and spacing

**ACCESS-004 — PASS broadly, with one systemic FAIL.** *Medium.*

The product's copy is unusually good. Messages state what happened and what to do, in the fewest exact words:
- *"Editing opens only while the order is New or Confirmed. Current status: Ready to Ship."* (`dashboard.js:2703`)
- *"Only 240 lb is coverable. Reduce the request to 240 lb or release a competing reservation."* (`2979`)
- *"That lot belongs to a different product. Choose a lot listed for this line."* (`2981`)
- *"Auto FIFO chooses its own lots. Clear the manual lot selection and try again."* (`2983`)
- *"Remaining = expected − posted receipts (never stored). Receipts auto-link by product + supplier, oldest expected date first."* (`index.html:281`)
- *"Lot list may be incomplete — only the first 500 inventory lots were loaded."* (`dashboard.js:2924`)
- *"This correction references the original transaction; no separate ledger lines were created."* (`587`)

**FAIL (S-25…S-33 and eleven other error sites): raw API bodies are shown to the user.** `fetchAPI` and `fetchSalesAPI` throw `` new Error(`HTTP ${res.status}: ${body}`) `` with the **entire raw response body** (`dashboard.js:488`, `1919`). A `parseApiErrorMessage` helper exists (`2668-2685`) and correctly extracts `detail.message` — but it is called on only five paths (`2741`, `2806`, `2836`, `2984`, `3147`). Thirteen other `catch` blocks concatenate `e.message` directly: `901`, `970`, `1086`, `1181`, `1234`, `1299`, `1672`, `1751`, `1779`, `2182`, `2221`, `2272`, `2548`, `3381`, `3493`, `3725`, `3916`. So Luz sees *"Failed to load sales orders: HTTP 500: {"detail":{"error_code":"DB_TIMEOUT","message":"...","trace_id":"..."}}"* where the rule asks for *"Lot not found"*. The extractor is already written; it simply is not used.

**FAIL (S-23, S-89):** `confirm('Delete this item?')` (`1771`) omits the one word that matters — which item. `alert("SKU, qty, and due date are required.")` (`scheduler:1565`) omits which of the three is missing.

**ACCESS-009 — PASS with three PARTIALs.** *Medium.* `body { line-height: 1.5 }` (`dashboard.css:84`) is a correct default ✓, and every long-reading surface is explicitly loosened: `.order-notes-card p` at 1.5 (`2161`), `.order-ready-note-text` at 1.45 (`1506`), `.order-notes-edit textarea` at 1.45 (`1874`), `.production-detail-name` at 1.4 (`610`) ✓.
**PARTIAL (S-25…S-33):** `.readiness-chip { line-height: 1.25 }` (`1625`) with `.readiness-chip-detail { white-space: normal }` (`1632`) inside a `max-width: 320px` cell (`1327`) — a blocker with a long detail wraps to three or more lines at tight spacing, which the rule prohibits (*"a row that needs three lines is redesigned"*).
**PARTIAL (S-16…S-18, S-37, S-42):** `.created-at-meta` at 1.25 (`784`) and `.ship-by-date`/`.ship-by-weekday` at 1.25 (`1526`) are two-line elements ✓ permitted, but they sit inside cells that also stack a third line (DATA-003).

---

### ICON-004 — Conventional glyphs; one glyph per meaning; FL-specific concepts get a text label — **HIGH**

*Hard rule.* **FAIL: the × glyph carries four different meanings.**

| Instance | Meaning | Conventional? | Evidence |
|---|---|---|---|
| `&times;` in `.btn-close` | **close** | ✓ correct | `index.html:341`, `402`, `450`, `512` |
| `✕` in `#schedule-close` | **close** | ✓ correct | `scheduler:243` |
| `&#10005;` in `.note-action-btn.delete` | **delete a record** | ✗ the convention is a trash can | `dashboard.js:1733` |
| `✕` in `.o-excl` | **exclude an order from the plan** | ✗ a third meaning | `scheduler:1536` |
| `✗` in `.completeness-badge` | **incomplete** (a status, not an action) | — a fourth | `traceability.html:1246` |

The rule states *"A glyph is never reused for a different action"* and its Factory Ledger clause is exact: *"'Void' does not use the X; it gets a labeled destructive treatment."* The Notes delete button is a 22 × 20 px `✕` four pixels from an `✎` (LAYOUT-012), and the scheduler's `✕` (reversible exclude) sits immediately beside `⌫` (irreversible delete) — two remove-shaped glyphs, one recoverable and one not.

**S-87 — FAIL.** `scheduler:1537` — `⌫` (the keyboard erase-left key) for "delete order line". Not a metaphor for record deletion in any convention.

**Two triangles for one meaning — PARTIAL.** `▶` (`dashboard.css:651`) and `▸` (`dashboard.js:2320`, `3850`) both mean "expand in place", in different components, at 10 px where they are nearly indistinguishable.

**Missing conventional glyphs — PARTIAL.** No trash can for delete; no magnifier on the dashboard's `#global-search` (Traceability has one ✓, `traceability.html:275`); no undo glyph because there is no undo (ERROR-003).

**PASS — and this is a real strength.** Every Factory-Ledger-specific concept is given a **text label** exactly as the rule's qualification requires: Allocate, Release, Trace, Factory Ready, Dispatch Ready, Blocked, Close, Cancel, Preview, Inventory, Show full chain, Pin & replan, Set as baseline. No unlabelled glyph was invented for any of them. The standard's warning — *"Don't invent an unlabeled icon for 'allocate'"* — has been heeded throughout.

**S-66, S-68, S-69, S-71 — PASS.** Traceability's glyph vocabulary is internally consistent and conventional: `✓` = confirmed, `⚠` = gap or legacy, `✗` = incomplete, each paired with text in both the legend (`291-300`) and the badge (`1237-1246`) ✓.

---

### ICON-002 — One consistent icon set — **MEDIUM**

*Hard rule.* **FAIL. Four visually incompatible families are mixed, sometimes in one view.**

1. **Geometric / dingbat Unicode**, rendered in the UI font at the surrounding weight: `☰ ▶ ▸ × ← → ‹ › ✎ ✕ ✓ ✗ ⚠ ⧉ ⌫ ∅`
2. **Full-colour emoji**, rendered as multi-colour bitmaps from the platform emoji font and ignoring `color` and `font-weight` entirely: `📝` (`dashboard.js:1680`), `📦` (`2306`, `3833`), `🚛` (`3417`), `🔍` (`traceability.html:265`), `🔎` (`275`), `🖨` (`324`), `💾` (`325`)
3. **One hand-drawn inline SVG**: the process-flow stage arrow, `<svg viewBox="0 0 16 16">` with `stroke-width="1.5"` — the only vector icon in the product
4. **CSS-drawn shapes**: `.chevron::after` (a glyph rotated 90°, `dashboard.css:650-657`), `.timeline li::before` (an 8 px circle, `914-927`), `.mini-calendar-day.has-shipments::after` (a 3 px circle, `mini-calendar.css:97-108`), `.legend-swatch` and `.sw` (squares)

The mixing is visible within single views: the Traceability header pairs a colour `🔍` with monochrome `✕` and `⚠` glyphs; its export row pairs colour `🖨` and `💾` with monochrome text buttons; the Sankey page title uses a monochrome `☰` above a colour `∅` empty state.

**S-20, S-21, S-29, S-44, S-51, S-59 — FAIL.** The five emoji empty-state icons are rendered at 32 px with `opacity: 0.5` (`dashboard.css:1214-1218`, `2171-2175`) — an opacity that has no effect on a colour bitmap glyph the way it does on monochrome text, so they read as fully saturated illustrations in an otherwise monochrome interface.

---

### ICON-001, ICON-003, ICON-005, ICON-006, ICON-008 — remaining icon findings

**ICON-001 — FAIL on two glyphs, PASS elsewhere.** *High · Hard rule.*
**S-58 — FAIL.** `sankey.html:283` — `<h1><span class="icon">&#9776;</span> Product Flow — Sankey</h1>`. The hamburger `☰` is used as a decorative "flow lines" glyph in the page title, **twenty pixels below the same glyph functioning as the menu button** in the site nav (`274`). This is the "clever glyph" the rule prohibits, and it collides with a conventional meaning.
**S-87 — FAIL.** `⌫` for delete (see ICON-004).
**S-84 — PARTIAL.** `⧉` for "copy day" (`scheduler:1343`) is abstract, but it loosely matches the document-on-document convention ✓ and carries a full `aria-label` ✓.
**S-59 — PARTIAL.** `∅` (`sankey.html:580`) is a mathematical symbol rather than a factory metaphor for "no data".
**PASS everywhere else:** `☰` menu, `×` close, `←` back, `✎` edit, `✓` done, `🖨` print, `💾` save, `‹ ›` paging, `⚠` warning, `🚛` incoming delivery, `📦` inventory, `i` information — all immediately recognisable ✓.

**ICON-003 — PARTIAL.** *Medium.* Because every glyph is a text character, all of them inherit `font-size` and scale with the text around them ✓ — the rule's scaling clause passes throughout. The weight clause is where it slips: emoji ignore `font-weight` entirely, so `📦` in a 14 px regular empty state and `✓` inside a 700-weight `.so-ready-pill` (`dashboard.css:1585`) have unrelated visual weights; and `.btn-close { font-size: 22px }` (`883`) sets the × at 1.5× the 15 px `h3` beside it (`878`).

**ICON-005 — PARTIAL.** *Medium.* Font glyphs are vector by definition ✓, but the seven emoji are typically colour **bitmap** glyphs (CBDT/sbix) on most platforms — not vector, and not simplified for small sizes. At the smallest rendered sizes the monochrome glyphs also lose their distinctiveness: `.chevron::after` and `.order-expand-caret` are both 10 px (`dashboard.css:652`, `1362`), where `▶` and `▸` are indistinguishable; the mini-calendar's ship dot is 3 px (`mini-calendar.css:100-104`); the trace link glyphs are 9–10 px (`traceability.html:1063`, `1073`).
**S-61 — PASS.** The process-flow stage arrow is the product's one true SVG icon, drawn at 16 × 16 with `stroke-linecap`/`stroke-linejoin` set ✓.

**ICON-006 — PASS.** *Low.* The only letter-based icon is the scheduler's `i` for information (`scheduler:3-9`), which reads correctly in both English and Spanish (*información*) ✓. No human figures are depicted anywhere; `🚛` and `📦` are culturally neutral ✓.

**ICON-008 — PASS.** *Low.* The only icon with a selected state is the expand caret, and all three implementations use **one consistent treatment** — a 90° rotation — rather than swapping in an alternate glyph: `.collapsible-header.expanded .chevron::after` (`dashboard.css:657`), `.order-expand-toggle.expanded .order-expand-caret` (`1366`), `.supply-item-row[aria-expanded="true"] .supply-row-caret` (`2345`) ✓. It is paired with the row's own content as the label ✓ and is not colour-alone ✓. A small thing, done right and done consistently.
The theme toggle swaps `☾` for `☀` (`dashboard.js:62`), which is a mode indicator rather than a selection state and follows the universal convention ✓.

**ICON-007 — UNVERIFIABLE-FROM-CODE on every screen.** *Low.* Optical centring can only be judged from a render. What the code shows: `.order-expand-toggle` centres geometrically with `display: inline-flex; align-items: center; justify-content: center` on a 22 × 22 box (`dashboard.css:1344-1349`) — correct as far as CSS can go, though the asymmetric `▸` will read left-heavy inside it; `.btn-close` has `font-size: 22px; line-height: 1` and **no padding** (`880-889`), so the `×` sits wherever its font metrics place it; `.supply-row-caret { width: 16px }` (`2346-2351`) has no centring at all.

---

### OTHER-010 — Semantic color tokens with light, dark, and high-contrast variants, shared by all surfaces — **HIGH**

**FAIL.** *Hard rule.* Four clauses; all four fail.

**1. Five palettes, no shared file.** Each surface declares its own `:root`:

| File | Vocabulary |
|---|---|
| `dashboard/dashboard.css:4-75` | `--bg`, `--surface`, `--surface-alt`, `--surface-hover`, `--border`, `--text`, `--text-secondary`, `--text-muted`, `--text-dimmed`, `--primary`, `--accent`, `--danger`, `--warning`, `--badge-*`, `--row-*`, `--category-*` |
| `dashboard/traceability.html:10-34` | its own `--surface2`, plus `--col-supplier/-ingredient/-batch/-finished/-customer/-gap` |
| `dashboard/sankey.html` | its own `:root` with a third set of chart colours |
| `dashboard/process-flow.html` | its own `:root`, using **`--text-dim`** where the dashboard uses `--text-dimmed` |
| `dashboard/scheduler/…:8-18` | an entirely different light "paper" palette — `--ink`, `--ink2`, `--paper`, `--card`, `--line`, `--ok`, `--amber`, `--late`, `--pin`, `--wip`, `--idle`, `--coco` |

The rule asks for *"One token file consumed by dashboard and mobile."* There are five, and the naming diverges (`--text-dim` vs `--text-dimmed`) so they cannot even be mechanically merged.

**2. No high-contrast variant.** The rule requires all three variants *"even if only one mode ships today."* There is no `@media (prefers-contrast: more)` and no `forced-colors` handling in any of the five files.

**3. Tokens are repurposed.** `--category-coconut #60a5fa` is byte-identical to `--primary-hover #60a5fa` (`dashboard.css:16-17`); `--category-granola #fbbf24` is byte-identical to `--badge-amber-text #fbbf24` (`19`, `28`). *"A token is never used for a purpose other than its name."* Full treatment under FEEDBACK-012 in `03-feedback-error-notify.md`.

**4. Raw values bypass the tokens in at least twelve places:**

| Selector | Raw value | Evidence |
|---|---|---|
| `.site-nav` | `background: #1a1a2e` | `dashboard.css:98` |
| `.site-nav-link.active` | `background: #3b82f6` | `dashboard.css:115` |
| `.missing-list` | `border-top: 1px solid #991b1b` | `dashboard.css:743` |
| `.error-msg` | `border-top: 1px solid #991b1b` | `dashboard.css:940` |
| `.so-badge.status-ready` | `#064e3b` / `#5eead4` | `dashboard.css:1570` |
| `.so-badge.status-partial_ship` | `#7c2d12` / `#fdba74` | `dashboard.css:1571` |
| `.so-badge.status-invoiced` | `#064e3b` / `#6ee7b7` | `dashboard.css:1573` |
| `.so-ready-pill` | `rgba(34,197,94,…)` / `#86efac` | `dashboard.css:1576-1588` |
| `.order-ready-checkbox` | `accent-color: #22c55e` | `dashboard.css:1341` |
| `.order-row.so-ready td:first-child` | `border-left: 3px solid #22c55e` | `dashboard.css:1321` |
| `.order-edit-message.success` / `.error` | `#86efac` / `#fca5a5` | `dashboard.css:1885-1898` |
| `.detail-card` | `background: #16213e` | `traceability.html:188` |
| `.mini-calendar-day.is-today.has-shipments::after` | `background: #93c5fd` | `mini-calendar.css:111` |
| `.disambig-btn` | `var(--bg-card, #fff)` — **a token that does not exist** | `dashboard.js:1379` |

Four of those are the ACCESS-008 contrast failures above. The rule's stated purpose — *"Tokens are what make consistent color semantics, contrast, and future dark mode achievable without rework"* — is exactly what the hard-coded values broke.

**What is right:** the `[data-theme]` structure itself is well built — a complete paired light/dark set defined once, persisted, and toggled (`dashboard.css:4-75`; `dashboard.js:44-63`) ✓. The failure is that it is neither shared, exhaustive, nor respected.

---

### OTHER-003 — Collect and show only what each role needs; guard against misuse — **HIGH**

**FAIL on both clauses.** *Hard rule.*

**There is no role model.** One shared `SALES_API_KEY = 'dashboard-key-2026'` (`dashboard.js:1882`), no login, no user identity, no permission check anywhere. Every screen renders every control for everyone. The rule's clause — *"Read-only roles have no mutation controls rendered"* — has nothing to act on.

**Data the floor does not need is rendered.** The rule's example is *"Floor role sees quantities and lots, not customer pricing."* The dashboard renders:
- `case_price` in the Order Detail read view (`dashboard.js:3270`) and as an editable `unit_price` in edit mode (`3264`, PATCHed at `2776`)
- every customer name in the orders list (`2323`) and in the Dispatch Queue
- customer `contact_name` and `email` in global search results (`1551`)

Arturo's own screens (Supplies, Traceability, lot lookup) carry no pricing ✓, but nothing prevents him opening the Sales Orders tab, and the tab bar sits in front of him on every page.

**Guards that exist — and several are good:**
- the two-step arm on ER Close / Cancel (`3470-3483`) ✓
- optimistic rollback on a failed Factory Ready write (`2482`) ✓
- server-side `OVER_ALLOCATION` and `LOT_PRODUCT_MISMATCH` enforcement, surfaced with a concrete remedy (`2977-2983`) ✓
- the header-edit status gate, `canEditOrderHeader` (`2664-2666`), with the reason stated in prose (`2703`) ✓
- the dispatch-queue read-only guard on Factory Ready (`postOrderReady`, `2408-2410`) ✓

**Guards that are missing:**
- the scheduler order-line delete has **no confirmation** (`scheduler:1553-1554`)
- Factory Ready — an outward-facing signal to the floor — writes from an unconfirmed 16 × 16 px checkbox (`2455-2486`)
- the SO status select offers **every** transition including `cancelled`, with a generic confirm (`2696-2698`, `2819`)
- the write-capable API key is public (INPUT-003), so anyone who can load the page can POST to the ledger

**Logging — mixed.** The ledger's own provenance is genuinely good: `created_at_source`, `entry_backfilled`, and correction events are recorded and surfaced (`dashboard.js:438-446`, `557-566`, `1319-1328`) ✓. But the dashboard's own writes carry `by: 'floor'` as a hard-coded literal (`2414`) and display it as though it were audit data (`2232`) — so the trail records a fiction (INPUT-017).

---

### OTHER-005 — Give each platform first-class attention — **MEDIUM**

**FAIL.** The screen inventory states it plainly and the code confirms it: *"No dedicated mobile surface exists. Every dashboard screen is the desktop layout reflowed by `max-width` rules."*

**Mobile is a shrunken desktop.** Three breakpoints exist in total — 768 px (`dashboard.css:121`, `2181`, `2426`) and 480 px (`318`, `382`, covering only the Recent Entries cards and the Today tile). There is no mobile-specific view, no bottom tab bar, no bottom action bar, no scan-first path, and no one-task-per-screen flow. Every commit control sits at the top of the screen (TOUCH-001).

**And the desktop is an un-optimised web page.** The rule's desktop strengths — *"dense tables, keyboard shortcuts, multi-record views"* — are largely absent:
- **no keyboard shortcuts at all.** Three `keydown` handlers exist in the entire product: the supplies row's Enter/Space (`dashboard.js:3863`), the Supply modal's Escape (`4103`), and Traceability's Enter-to-trace (`traceability.html:442`). No Ctrl/Cmd shortcut, no `/` to focus search, no `n` for new.
- **no split view** — opening an order replaces the list (NAV-009)
- **no column sorting** (DATA-008) and **no resizable columns** (DATA-009)
- **no multi-record view** and no bulk anything
- `.tab-content { max-width: 1200px }` (`dashboard.css:282`) caps dense tables on wide monitors (LAYOUT-011)

**S-47…S-50 — PASS.** The Supplies tab is the one surface with deliberate compact behaviour — a dedicated `@media (max-width: 768px)` block (`dashboard.css:2426-2436`) restructuring the toolbar, the sub-tabs, the search field, and the detail grid, plus keyboard operability ✓.

**S-57…S-91 — FAIL.** Sankey, Traceability, and the scheduler have no compact rules beyond the shared nav (`sankey.html:49`, `traceability.html:49-55`; the scheduler's only `@media` is `print` at `:185`).

---

### OTHER-009 — One standard help control per screen — **LOW**

**FAIL on the dashboard (absence), PASS on the scheduler (a good implementation).**

**There is no help control anywhere in the dashboard** — no "?", no help link, no documentation entry point on any of the 56 dashboard screens. Concepts that plainly need one:
- the LAT lot-code policy
- **Factory Ready vs Dispatch Ready** — two readiness concepts on the same screen, one human-set and one computed
- the three allocation modes and the 48-hour auto-FIFO TTL
- what "Effective Remaining" and "Shipped Effective" mean versus the recorded counters
- why an expected receipt auto-links by product + supplier

Several are explained inline as prose ✓ — `.section-hint` (`index.html:281`, `293`), the readiness caption (`dashboard.js:2896`), the allocation subtitle (`2914`), the preview subtitle (`2963`). That is good writing, but it is caption text competing for space, not a help control the user can reach when they need it.

**S-72, S-75, S-76, S-79, S-82, S-83, S-86, S-87 — PASS. The reference implementation.** The scheduler puts a consistent `.info` control immediately after each label it explains — five on the KPI strip (`scheduler:3-9`), one on the pin modal's quantity field (`:1448`), one on the order-book header (`:1525`) — each carrying a plain-language `title`, and a delegated click handler renders it as a positioned popover so it works on touch as well as hover (`:1583-1597`) ✓. One control type, one position, context-specific content. Plus a "How this works" disclosure explaining the whole scheduling model (`:1005`) ✓.

---

### OTHER-001, OTHER-004, OTHER-006, OTHER-007, OTHER-008, OTHER-011

**OTHER-001 — PARTIAL.** *Medium.* The audit test asks for *"a written list of core workflows per role."* None exists: the screen inventory records that *"the repo has no written statement of who uses which dashboard screen"* and that every Primary-user attribution is inferred.
On the second clause the product does better than the first suggests. The standard names Receive, Pack, Ship, Coconut log, and Trace as core (`1426`); of those only **Trace** lives in this dashboard, and it is measurably the most polished surface in the product — recents, type-ahead, near-miss correction, URL state, a completeness headline, an adjacent accessible data table, and a text export (S-63…S-71). The core workflow that is in scope did get the investment ✓.

**OTHER-004 — PASS, and it is the product's strongest area.** *High.* The vocabulary is the floor's throughout, with no invented terms: *Lot, Lot Code, Batch, Cases, Pallets, On Hand, Expected Receipt, Received, Remaining, Ship By, Factory Ready, Supplier, BOL, SKU, Made, Packed, Bulk, Retail, Pan, Bake, Repack, Crew, Overtime, Bottleneck.* Product families use the floor's names — Coconut, Granola, Graham (`dashboard.js:174-205`) — and pack formats use the floor's names — *"Granola bulk 10 lb"*, *"12x10 OZ Retail Cases (SS Line)"*, *"6x7 OZ Retail Cases (BS Line)"* (`411-417`). The scheduler names real stations: Bake, Coconut, Toasted, Pouch, Box 10, Box 25, Repack. Explanatory copy stays in the floor's terms (`index.html:281`).
**PARTIAL on the controls clause:** the six clickable non-controls (ACCESS-003) are not control types users have seen elsewhere in a form they can operate.
**PARTIAL on one term:** *"Dispatch Ready"* and *"Factory Ready"* are two different readiness concepts on the same row (`dashboard.js:1948-1952` vs `2229-2235`), and the readiness card has to spend a caption explaining that one is computed and is *"not a shipping gate"* (`2896`). That collision was introduced by the software, not the floor.

**OTHER-006 — UNVERIFIABLE-FROM-CODE on the testing clause, with strong indirect evidence against.** *High, process rule.* Whether these screens were tested on floor devices cannot be read from the repository. What the code shows is that several of the rule's named conditions would have failed immediately if they had been:

| Condition the rule names | What ships |
|---|---|
| *"the largest and smallest layouts"* | the sticky tab bar overlaps the app header at ≤768 px — visible on any phone (LAYOUT-003) |
| *"200 % zoom"* | triggers the same overlap plus horizontal table overflow (ACCESS-001) |
| *"light and dark modes"* | the "✓ READY" pill and the "Header saved." message are ~1.27 : 1 in light mode (ACCESS-008) |
| *"Spanish labels"* | fifteen `white-space: nowrap` declarations on growable text (LAYOUT-003) |
| *"dead-zone Wi-Fi"* | no fetch has a timeout; a hang spins forever (FEEDBACK-003) |
| *"bright, dim, and glare lighting"*, *"arm's length"* | 8 px mini-calendar text, 10 px blocker details (ACCESS-006) |

**PASS on the review clause** — *"schedule periodic review against these standards"* is what this audit is, and the standards document is dated the same day ✓.

**OTHER-007 — PASS, with three small PARTIALs.** *Low.* The tone is right: no celebratory animation, no confetti, no playful copy on consequential actions, and error copy that stays factual and steady (*"Only 240 lb is coverable. Reduce the request to 240 lb or release a competing reservation."*). Transitions are short (0.15–0.3 s) and limited to colour and opacity.
**PARTIAL:** `.day-card.today { box-shadow: 0 0 20px rgba(59,130,246,0.1) }` (`dashboard.css:453`) is decoration on top of a tinted date header that already marks today; five emoji empty-state icons at 32 px (`1214`, `2171`) are decoration, though they soften an empty screen; and two `scrollIntoView({ behavior: 'smooth' })` calls (`dashboard.js:887`, `traceability.html:1142`) add roughly 300 ms to a user-initiated action, with no `prefers-reduced-motion` guard anywhere in the product.

**OTHER-008 — PASS.** *Low.* Every download uses an anchor with `download` and a Blob or object URL, invoking the browser's own save behaviour ✓: `exportOrdersCsv` (`dashboard.js:2172-2180`), `exportOrdersMatrix` (`2206-2219`), `downloadAuditText` (`traceability.html:1353-1359`), the scheduler's schedule CSV export (`scheduler:1311-1319`). The single upload path — the scheduler's orders CSV import — uses a native `<input type="file">` with `FileReader` (`:1077-1101`) ✓. There is no custom file browser anywhere.

**OTHER-011 — PASS, with one note.** *Low.* The standard states the population: *"CNS staff are US-based English/Spanish speakers for whom red = stop and green = go are standard."* The palette follows it — red for blocked/error/overdue, amber for attention, green for ready/complete/healthy ✓ — and no other market is served.
**PARTIAL (S-75):** because the scheduler's red/green delta pair sits at near-identical luminance (0.132 vs 0.158, ACCESS-008), the convention is carried by hue alone, which fails for the roughly 8 % of men with red-green colour vision deficiency — a real subset of the stated population. That is an ACCESS-008 / FEEDBACK-011 problem rather than a cultural one, but it lands on this rule's audit test too.

---

## N/A register

| Rule | Screens | Reason |
|---|---|---|
| ICON-009 | **All 91** | **No raster image exists anywhere in the product** — no `<img>`, no `background-image`, no image asset. Nothing can be stretched. |
| ICON-010 | **All 91** | **No attachment or file-display feature exists.** The only file operations are downloads initiated by named text buttons. |
| OTHER-002 | **All 91** | **No permission of any kind is ever requested** — no camera, geolocation, notifications, or microphone. The only system capability touched is `navigator.clipboard.writeText` (`scheduler:1088`), which needs no prompt in a user-gesture context. *If INPUT-016 (barcode scanning) is implemented, this rule becomes live.* |
| ACCESS-005 | S-23, S-35, S-38, S-89 | Native browser dialogs; the product cannot style their text. |
| ACCESS-010 | Screens with no icon-only control | Nothing to name. |
| ICON-001…008 | Screens rendering no glyph, shape, or diagram | No icon present. |
| OTHER-008 | Screens with no upload or download path | Nothing to pick a file for. |
| OTHER-009 | Confirmations and native dialogs | Not a screen a help control can attach to. |
| OTHER-010, OTHER-011 | Native dialogs | Colour is the browser's, not the product's. |

## Unverifiable from code

| Rule | Screens | What a browser check must confirm |
|---|---|---|
| ACCESS-001 | All screens | Rendered behaviour at 200 % zoom and at the largest OS text size. The `px`-only sizing is verified in code and predicts the outcome, but the rendered result should be captured. |
| ACCESS-008 | S-71, S-91 (print views) | Contrast in the printed output. `traceability.html:243-244` and `scheduler:185-208` restyle substantially for print (`background: #fff; color: #000`), so the printed ratios differ from the screen ones computed above. |
| ACCESS-008 | All screens, both themes | The computed ratios above are derived from declared hex values; a rendered check with a contrast tool should confirm them, particularly the composited translucent fills (`.so-ready-pill`, `.order-edit-message.*`, `.readiness-chip.severity-*`). |
| ICON-007 | **All 91** | Optical centring can only be judged from a render. `.btn-close` (`dashboard.css:880-889`, no padding), `.order-expand-toggle` (`1344-1349`, geometric centring of an asymmetric `▸`), and `.supply-row-caret` (`2346-2351`, no centring) are the three to check. |
| ICON-002, ICON-005 | Every screen using an emoji glyph | How `📝 📦 🚛 🔍 🔎 🖨 💾` actually render on the office machines and the floor phones — colour bitmap versus monochrome outline varies by OS and font stack, and determines how badly they clash with the dingbat glyphs. |
| OTHER-006 | **All 91** | Whether these screens were tested on the real floor devices, in floor lighting, at both extremes of size, in both appearance modes, and with Spanish labels. Not answerable from the repository; the code evidence tabulated above argues they were not. |

---

*End of 05-access-icon-other.md*

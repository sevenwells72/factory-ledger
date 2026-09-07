# 02 — Audit: Actions, Data Entry & Touch (ACTION, INPUT, TOUCH)

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) §3 (ACTION-001…012), §4 (INPUT-001…022), §5 (TOUCH-001…005) — 39 rules
**Screens:** [00-screen-inventory.md](00-screen-inventory.md) — S-01…S-91
**Status:** Audit findings. **No application code was modified.**

---

## How to read this

Statuses and tiering are as defined in [01-nav-layout.md](01-nav-layout.md): **P** PASS · **PA** PARTIAL · **F** FAIL · **N** N/A · **U** UNVERIFIABLE-FROM-CODE · **·** out of tier scope.

### Rule importance (drives Tier B scope)

- **Critical:** ACTION-002, 003, 006, 008; INPUT-007, 008, 011, 017, 019, 022; TOUCH-003 · *(INPUT-003 is Low/Contextual but Critical where it applies)*
- **High:** ACTION-001, 005, 007, 010; INPUT-002, 006, 010, 014, 015, 016, 018, 021; TOUCH-001, 004
- **Medium:** ACTION-004, 009, 011, 012; INPUT-001, 004, 005, 009, 012, 013, 020; TOUCH-002, 005
- **Low/Contextual:** INPUT-003

### Mobile-platform rules

**INPUT-010**, **INPUT-012** (mobile clause), **INPUT-016**, **TOUCH-001**, **TOUCH-002**, **TOUCH-004**, and **TOUCH-005** are Mobile-platform rules. Per the audit brief they are weighted at full strength only where the primary user is **Arturo or Luz** *and* the inventory Mobile column is Yes or Partial; elsewhere they are **N** with the reason *"desktop-only surface."*
**TOUCH-003 is a Both-platform rule** and is therefore applied everywhere, stricter on mobile.

---

## Matrix A — Actions & Buttons (ACTION-001…012)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | P | PA | P | P | P | P | N | N | P | P | P | PA |
| S-02 | Mini-calendar strip | P | PA | N | P | P | P | N | N | P | PA | P | N |
| S-03 | App header | P | **F** | PA | PA | **F** | **F** | N | N | P | P | P | N |
| S-04 | Global search + results | P | **F** | N | N | P | P | PA | N | P | PA | **F** | N |
| S-05 | Tab bar (7 tabs) | P | **F** | P | PA | P | P | N | N | P | P | P | PA |
| S-06 | Today So Far tile | N | N | N | N | N | N | N | N | N | N | N | N |
| S-07 | Today So Far — error/retry | P | **F** | P | P | P | PA | N | N | P | P | P | N |
| S-08 | Production Calendar | **F** | **F** | P | PA | PA | P | N | N | P | PA | PA | N |
| S-09 | Calendar day detail panel | P | **F** | P | P | P | P | N | N | P | P | P | N |
| S-10 | Finished Goods panels | P | **F** | N | N | PA | P | N | N | P | **F** | **F** | N |
| S-11 | FG per-product lot rows | P | **F** | N | N | PA | P | N | N | P | **F** | **F** | N |
| S-12 | Batch Inventory | P | **F** | N | N | PA | P | N | N | P | **F** | **F** | N |
| S-13 | On-Hand Ingredients | P | **F** | N | N | PA | P | N | N | P | **F** | **F** | N |
| S-14 | Recent Entries feed | P | **F** | PA | P | PA | **F** | N | N | P | P | P | N |
| S-15 | Recent Entries states | P | **F** | P | P | P | P | N | N | P | P | P | N |
| S-16 | Daily Entries + toolbar | **F** | **F** | P | · | P | P | N | N | · | P | · | · |
| S-17 | Shipping log | P | **F** | P | · | PA | P | N | N | · | **F** | · | · |
| S-18 | Receiving log | P | **F** | P | · | PA | P | N | N | · | **F** | · | · |
| S-19 | Notes toolbar | P | **F** | PA | · | P | **F** | N | N | · | P | · | · |
| S-20 | Notes list (cards) | P | **F** | P | · | **F** | **F** | N | **F** | · | **F** | · | · |
| S-21 | Notes empty state | N | N | N | · | N | N | N | N | · | N | · | · |
| S-22 | Note create/edit modal | P | **F** | P | · | PA | **F** | **F** | N | · | P | PA | · |
| S-23 | Delete-note confirm | P | N | **F** | · | **F** | **F** | **F** | **F** | · | P | · | · |
| S-24 | Orders toolbar | PA | **F** | **F** | PA | PA | **F** | N | N | PA | P | P | N |
| S-25 | Orders list table | P | **F** | P | P | **F** | **F** | N | PA | P | **F** | **F** | N |
| S-26 | Dispatch Queue mode | PA | **F** | P | P | **F** | **F** | N | PA | P | **F** | **F** | N |
| S-27 | Orders expandable lines | P | **F** | P | P | PA | P | **F** | N | P | **F** | **F** | N |
| S-28 | Factory Ready toggle+note | P | **F** | P | P | PA | **F** | **F** | PA | P | PA | PA | N |
| S-29 | Orders empty state | N | N | N | N | N | N | N | N | N | N | N | N |
| S-30 | Order Detail header/KPI | P | **F** | P | PA | P | **F** | **F** | N | P | P | P | N |
| S-31 | Order Detail line table | P | **F** | P | P | PA | P | N | N | P | P | P | N |
| S-32 | Per-line inventory expander | P | **F** | P | P | PA | P | N | N | PA | P | P | N |
| S-33 | Order Detail edit mode | P | **F** | PA | **F** | **F** | **F** | **F** | N | P | P | PA | N |
| S-34 | Edit-locked notice | N | N | N | N | N | N | N | N | N | N | N | N |
| S-35 | SO status change confirm | P | N | **F** | · | **F** | **F** | **F** | **F** | P | P | · | N |
| S-36 | Reservations / allocation | P | **F** | P | PA | P | PA | **P** | N | P | P | P | N |
| S-37 | Allocation history table | P | **F** | P | P | P | PA | N | PA | P | P | P | N |
| S-38 | Release reservation confirm | P | N | **F** | · | **F** | **F** | **F** | **F** | P | P | · | N |
| S-39 | Shipping capacity preview | P | **F** | P | P | PA | P | N | N | P | P | P | N |
| S-40 | Order Detail notes card | N | N | N | N | N | N | N | N | N | N | N | N |
| S-41 | Expected Receipts toolbar | P | **F** | **F** | PA | PA | **F** | N | N | PA | P | P | N |
| S-42 | Expected Receipts table | P | **F** | P | P | P | P | N | N | P | **F** | **F** | N |
| S-43 | ER row actions | P | **F** | P | P | P | PA | N | PA | PA | P | P | N |
| S-44 | ER empty state | N | N | N | N | N | N | N | N | N | N | N | N |
| S-45 | ER create/edit modal | P | **F** | P | **F** | PA | PA | **F** | N | P | P | PA | N |
| S-46 | Supplies header + Request | P | **F** | P | P | P | PA | N | N | PA | P | P | N |
| S-47 | Supplies sub-tabs | P | **F** | P | PA | P | P | N | N | P | P | P | PA |
| S-48 | Supplies search field | P | N | N | N | N | N | PA | N | N | N | P | N |
| S-49 | Supplies inventory table | PA | **F** | P | P | P | P | N | N | P | PA | **P** | N |
| S-50 | Supply lot / incoming detail | N | N | N | N | N | N | N | N | N | N | N | N |
| S-51 | Supply Requests list | P | **F** | P | P | P | PA | N | N | P | P | P | N |
| S-52 | Supply Requests feedback | N | N | N | N | N | N | N | N | N | N | N | N |
| S-53 | Request Supply modal | P | **F** | P | **F** | PA | PA | **P** | N | P | P | PA | N |
| S-54 | Lot Detail side panel | P | **F** | P | P | P | P | N | N | P | P | P | N |
| S-55 | Lot disambiguation | P | **F** | **F** | P | P | P | **F** | N | P | PA | **F** | N |
| S-56 | Product Detail panel | P | **F** | P | P | PA | P | N | N | P | **F** | **F** | N |
| S-57 | Sankey controls bar | P | **F** | P | · | P | P | N | N | · | P | P | · |
| S-58 | Sankey chart + legend | N | N | N | · | N | N | N | N | · | N | · | · |
| S-59 | Sankey banner / loading | N | N | N | · | N | N | N | N | · | N | · | · |
| S-60 | Summary strip | N | N | N | · | N | N | N | N | · | N | · | · |
| S-61 | Production lines grid | N | N | N | · | N | N | N | N | · | N | · | · |
| S-62 | Error / stale banners | N | N | N | · | N | N | N | N | · | N | · | · |
| S-63 | Lot search + type-ahead | P | **F** | P | P | PA | P | **P** | N | P | PA | **F** | N |
| S-64 | Trace direction + Trace | P | **F** | **P** | P | P | P | P | N | P | P | P | **F** |
| S-65 | Recent lots strip | P | **F** | P | P | P | P | N | N | P | P | PA | N |
| S-66 | Trace legend | N | N | N | N | N | N | N | N | N | N | N | N |
| S-67 | Status bar | P | **F** | P | P | P | P | N | N | P | PA | PA | N |
| S-68 | Trace graph + zoom | P | **F** | P | P | PA | P | N | N | P | P | PA | N |
| S-69 | Node tooltip | N | N | N | N | N | N | N | N | N | N | N | N |
| S-70 | Trace detail + exports | P | **F** | P | P | P | P | N | N | PA | P | P | N |
| S-71 | Print audit report | N | N | N | · | N | N | N | N | · | N | · | · |
| S-72 | Topbar KPI strip | N | N | N | · | N | N | N | N | · | N | · | · |
| S-73 | Topbar actions | P | **F** | **P** | · | P | P | N | N | · | P | · | · |
| S-74 | Legend bar | N | N | N | · | N | N | N | N | · | N | · | · |
| S-75 | Delta panel | P | **F** | P | · | P | P | N | N | · | P | · | · |
| S-76 | Settings panel (form) | P | **F** | P | · | P | P | **F** | N | · | P | · | · |
| S-77 | FG on hand (disclosure) | P | **F** | P | · | P | P | N | N | · | PA | · | · |
| S-78 | Bulk-bin WIP (disclosure) | P | **F** | P | · | P | P | N | N | · | PA | · | · |
| S-79 | Product catalog (form) | P | **F** | P | · | P | P | **F** | N | · | P | · | · |
| S-80 | Orders in / plans out | P | **F** | P | · | P | **F** | N | **F** | · | P | · | · |
| S-81 | CSV import result | N | N | N | · | N | N | N | N | · | N | · | · |
| S-82 | "How this works" | P | **F** | P | · | P | P | N | N | · | PA | · | · |
| S-83 | Production board table | P | **F** | P | · | PA | P | N | N | · | **F** | · | · |
| S-84 | Board cell copy / more | P | **F** | P | · | P | P | N | N | · | **F** | · | · |
| S-85 | Schedule panel | P | **F** | P | · | P | P | N | N | · | P | · | · |
| S-86 | Pin modal | P | **F** | P | · | P | P | **F** | N | · | P | · | · |
| S-87 | Order book | P | **F** | P | · | **F** | **F** | N | **F** | · | **F** | · | · |
| S-88 | Add order line form | P | **F** | P | · | PA | P | **F** | N | · | P | · | · |
| S-89 | Add-order validation alert | P | N | **F** | · | **F** | P | **F** | N | · | P | · | · |
| S-90 | Order book empty state | N | N | N | · | N | N | N | N | · | N | · | · |
| S-91 | Scheduler print view | N | N | N | · | N | N | N | N | · | N | · | · |

## Matrix B — Forms & Data Entry (INPUT-001…022)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 | 016 | 017 | 018 | 019 | 020 | 021 | 022 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| S-02 | Mini-calendar strip | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | P | N |
| S-03 | App header | N | **F** | **F** | P | N | P | N | N | N | PA | N | **F** | **F** | N | N | N | N | N | N | N | P | PA |
| S-04 | Global search + results | P | **F** | N | P | N | **F** | N | N | PA | PA | PA | **F** | **F** | PA | N | N | N | N | N | N | P | PA |
| S-05 | Tab bar (7 tabs) | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| S-06 | Today So Far tile | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | N | N | N | N | P | N |
| S-07 | Today So Far — error/retry | N | N | N | N | N | P | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-08 | Production Calendar | N | PA | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | P | N | N | P | PA |
| S-09 | Calendar day detail panel | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-10 | Finished Goods panels | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-11 | FG per-product lot rows | N | PA | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-12 | Batch Inventory | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-13 | On-Hand Ingredients | N | **P** | N | N | N | **F** | N | **P** | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-14 | Recent Entries feed | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-15 | Recent Entries states | N | N | N | N | N | P | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-16 | Daily Entries + toolbar | N | PA | N | P | P | P | P | P | P | P | P | N | N | N | N | N | P | P | N | N | P | P |
| S-17 | Shipping log | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-18 | Receiving log | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-19 | Notes toolbar | N | PA | N | P | P | P | N | N | P | P | P | N | N | N | N | N | P | P | N | N | P | P |
| S-20 | Notes list (cards) | N | P | N | N | N | P | N | P | P | P | N | N | N | N | N | N | P | N | N | N | P | P |
| S-21 | Notes empty state | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| S-22 | Note create/edit modal | P | P | N | PA | P | PA | **PA** | P | P | P | **F** | N | N | **F** | N | N | P | P | **F** | N | P | PA |
| S-23 | Delete-note confirm | N | N | N | N | N | **F** | N | N | N | N | N | N | N | N | N | N | N | **F** | N | N | N | N |
| S-24 | Orders toolbar | N | **F** | N | P | P | P | N | N | P | PA | PA | **F** | **F** | PA | N | N | P | P | N | N | P | PA |
| S-25 | Orders list table | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | **F** | N | N | N | P | PA |
| S-26 | Dispatch Queue mode | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | **F** | N | N | N | P | PA |
| S-27 | Orders expandable lines | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-28 | Factory Ready toggle+note | **F** | PA | N | P | P | PA | N | N | P | P | P | **F** | N | N | N | N | **F** | N | N | N | **F** | PA |
| S-29 | Orders empty state | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| S-30 | Order Detail header/KPI | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-31 | Order Detail line table | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-32 | Per-line inventory expander | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-33 | Order Detail edit mode | P | **F** | N | P | PA | **F** | PA | **F** | P | P | P | N | N | N | N | N | P | P | PA | PA | P | PA |
| S-34 | Edit-locked notice | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | **P** | P |
| S-35 | SO status change confirm | N | N | N | N | N | **F** | N | N | N | N | P | N | N | N | N | N | N | **F** | N | N | N | N |
| S-36 | Reservations / allocation | **F** | P | N | **P** | P | P | PA | P | P | P | **P** | N | N | P | N | N | **P** | P | PA | PA | P | PA |
| S-37 | Allocation history table | N | PA | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-38 | Release reservation confirm | N | N | N | N | N | **F** | N | N | N | N | N | N | N | N | N | N | N | **F** | N | N | N | N |
| S-39 | Shipping capacity preview | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-40 | Order Detail notes card | P | P | N | P | P | P | N | N | P | P | P | N | N | N | N | N | P | N | N | N | P | P |
| S-41 | Expected Receipts toolbar | N | **F** | N | P | P | P | N | N | P | PA | PA | **F** | **F** | PA | N | N | P | P | N | N | P | PA |
| S-42 | Expected Receipts table | N | P | N | N | N | P | N | P | **P** | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-43 | ER row actions | N | N | N | N | N | P | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-44 | ER empty state | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| S-45 | ER create/edit modal | P | **F** | N | PA | P | PA | **F** | **F** | P | P | **P** | **F** | PA | PA | N | N | P | P | **F** | **F** | **F** | PA |
| S-46 | Supplies header + Request | N | N | N | N | N | P | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-47 | Supplies sub-tabs | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | P | N | N | P | P |
| S-48 | Supplies search field | P | PA | N | P | P | P | N | N | P | P | P | PA | **F** | N | N | N | P | P | N | N | P | P |
| S-49 | Supplies inventory table | N | PA | N | N | N | **P** | N | PA | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-50 | Supply lot / incoming detail | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-51 | Supply Requests list | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-52 | Supply Requests feedback | N | N | N | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-53 | Request Supply modal | P | **P** | N | PA | P | PA | **P** | **P** | P | P | PA | N | PA | PA | N | P | **F** | P | PA | PA | P | PA |
| S-54 | Lot Detail side panel | N | P | N | N | N | PA | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-55 | Lot disambiguation | N | P | N | N | N | **F** | N | N | P | N | **P** | N | N | N | N | N | P | N | N | N | P | **F** |
| S-56 | Product Detail panel | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-57 | Sankey controls bar | N | PA | N | P | P | P | PA | P | P | P | P | N | N | N | N | N | P | P | N | N | P | P |
| S-58 | Sankey chart + legend | N | P | N | N | N | **F** | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-59 | Sankey banner / loading | N | N | N | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-60 | Summary strip | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-61 | Production lines grid | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-62 | Error / stale banners | N | N | N | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-63 | Lot search + type-ahead | P | PA | N | P | P | **F** | PA | N | **F** | PA | PA | **F** | **P** | PA | N | **F** | P | P | N | N | P | PA |
| S-64 | Trace direction + Trace | N | P | N | N | N | P | N | N | P | N | P | N | N | N | N | N | P | P | P | N | **P** | P |
| S-65 | Recent lots strip | N | P | N | N | N | **F** | N | N | P | N | P | N | N | N | N | N | P | N | N | N | P | P |
| S-66 | Trace legend | N | N | N | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-67 | Status bar | N | N | N | N | N | **F** | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-68 | Trace graph + zoom | N | P | N | N | N | **F** | N | P | **P** | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-69 | Node tooltip | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-70 | Trace detail + exports | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-71 | Print audit report | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-72 | Topbar KPI strip | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-73 | Topbar actions | N | N | N | N | N | P | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-74 | Legend bar | N | N | N | N | N | N | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-75 | Delta panel | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-76 | Settings panel (form) | P | P | N | P | P | P | PA | PA | P | P | P | N | N | N | N | N | P | P | PA | N | P | P |
| S-77 | FG on hand (disclosure) | P | P | N | P | P | P | PA | PA | P | P | P | N | N | N | N | N | P | P | PA | N | P | P |
| S-78 | Bulk-bin WIP (disclosure) | P | P | N | P | P | P | PA | PA | P | P | P | N | N | N | N | N | P | P | PA | N | P | P |
| S-79 | Product catalog (form) | P | P | N | P | P | P | PA | PA | P | P | P | N | N | N | N | N | P | P | PA | N | P | P |
| S-80 | Orders in / plans out | N | P | N | N | N | P | N | N | P | N | P | N | N | N | N | P | P | N | N | N | P | P |
| S-81 | CSV import result | N | N | N | N | N | N | **P** | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-82 | "How this works" | N | N | N | N | N | P | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-83 | Production board table | N | P | N | N | N | **F** | N | P | PA | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-84 | Board cell copy / more | N | N | N | N | N | **F** | N | N | P | N | N | N | N | N | N | N | N | N | N | N | P | P |
| S-85 | Schedule panel | N | P | N | N | N | P | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-86 | Pin modal | P | **P** | N | P | P | P | **P** | **P** | P | P | P | N | N | N | **P** | N | **P** | P | PA | N | P | P |
| S-87 | Order book | N | P | N | N | N | P | N | P | PA | N | N | N | N | N | N | N | P | N | N | N | P | P |
| S-88 | Add order line form | P | P | N | P | P | P | **F** | PA | P | P | PA | N | N | P | N | N | PA | PA | **F** | N | P | P |
| S-89 | Add-order validation alert | N | N | N | N | N | **F** | **F** | N | P | N | N | N | N | N | N | N | N | N | **F** | N | N | N |
| S-90 | Order book empty state | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| S-91 | Scheduler print view | N | P | N | N | N | N | N | P | P | N | N | N | N | N | N | N | P | N | N | N | P | P |

## Matrix C — Mobile & Touch (TOUCH-001…005)

| # | Screen | 001 | 002 | 003 | 004 | 005 |
|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | N | P | **F** | N | N |
| S-02 | Mini-calendar strip | N | P | PA | N | N |
| S-03 | App header | **F** | P | **F** | N | N |
| S-04 | Global search + results | **F** | P | **F** | N | N |
| S-05 | Tab bar (7 tabs) | **F** | P | PA | N | N |
| S-06 | Today So Far tile | N | N | N | N | N |
| S-07 | Today So Far — error/retry | N | N | **F** | N | N |
| S-08 | Production Calendar | N | P | P | N | N |
| S-09 | Calendar day detail panel | N | P | **F** | N | N |
| S-10 | Finished Goods panels | **F** | P | **F** | N | N |
| S-11 | FG per-product lot rows | **F** | P | **F** | N | N |
| S-12 | Batch Inventory | N | P | **F** | N | N |
| S-13 | On-Hand Ingredients | **F** | P | **F** | N | N |
| S-14 | Recent Entries feed | N | P | **P** | N | N |
| S-15 | Recent Entries states | N | P | **P** | N | N |
| S-16 | Daily Entries + toolbar | **F** | · | **F** | **F** | · |
| S-17 | Shipping log | **F** | · | **F** | **F** | · |
| S-18 | Receiving log | **F** | · | **F** | **F** | · |
| S-19 | Notes toolbar | **F** | · | **F** | PA | · |
| S-20 | Notes list (cards) | **F** | · | **F** | **F** | · |
| S-21 | Notes empty state | N | · | N | N | · |
| S-22 | Note create/edit modal | PA | · | **F** | **F** | · |
| S-23 | Delete-note confirm | N | · | N | N | · |
| S-24 | Orders toolbar | **F** | P | **F** | PA | N |
| S-25 | Orders list table | **F** | P | **F** | **F** | PA |
| S-26 | Dispatch Queue mode | **F** | P | **F** | **F** | PA |
| S-27 | Orders expandable lines | **F** | P | **F** | **F** | N |
| S-28 | Factory Ready toggle+note | **F** | P | **F** | **F** | PA |
| S-29 | Orders empty state | N | N | N | N | N |
| S-30 | Order Detail header/KPI | **F** | P | **F** | **F** | N |
| S-31 | Order Detail line table | **F** | P | **F** | **F** | N |
| S-32 | Per-line inventory expander | **F** | P | **F** | **F** | N |
| S-33 | Order Detail edit mode | **F** | P | **F** | **F** | N |
| S-34 | Edit-locked notice | N | N | N | N | N |
| S-35 | SO status change confirm | N | N | N | N | N |
| S-36 | Reservations / allocation | **F** | P | **F** | **F** | N |
| S-37 | Allocation history table | **F** | P | **F** | **F** | N |
| S-38 | Release reservation confirm | N | N | N | N | N |
| S-39 | Shipping capacity preview | **F** | P | **F** | **F** | N |
| S-40 | Order Detail notes card | N | P | P | N | N |
| S-41 | Expected Receipts toolbar | **F** | P | **F** | PA | N |
| S-42 | Expected Receipts table | **F** | P | **F** | **F** | N |
| S-43 | ER row actions | **F** | P | **F** | **F** | N |
| S-44 | ER empty state | N | N | N | N | N |
| S-45 | ER create/edit modal | PA | P | **F** | **F** | N |
| S-46 | Supplies header + Request | **F** | P | **F** | N | N |
| S-47 | Supplies sub-tabs | **F** | P | **F** | N | N |
| S-48 | Supplies search field | **F** | P | PA | N | N |
| S-49 | Supplies inventory table | **F** | P | PA | N | N |
| S-50 | Supply lot / incoming detail | N | P | P | N | N |
| S-51 | Supply Requests list | **F** | P | **F** | **F** | N |
| S-52 | Supply Requests feedback | N | P | N | N | N |
| S-53 | Request Supply modal | PA | P | **F** | **F** | N |
| S-54 | Lot Detail side panel | **F** | P | **F** | N | N |
| S-55 | Lot disambiguation | PA | P | PA | **F** | N |
| S-56 | Product Detail panel | **F** | P | **F** | N | N |
| S-57 | Sankey controls bar | N | · | **F** | N | · |
| S-58 | Sankey chart + legend | N | · | N | N | · |
| S-59 | Sankey banner / loading | N | · | N | N | · |
| S-60 | Summary strip | N | · | N | N | · |
| S-61 | Production lines grid | N | · | N | N | · |
| S-62 | Error / stale banners | N | · | N | N | · |
| S-63 | Lot search + type-ahead | **F** | P | **F** | **F** | N |
| S-64 | Trace direction + Trace | **F** | P | **F** | **F** | N |
| S-65 | Recent lots strip | **F** | P | **F** | N | N |
| S-66 | Trace legend | N | P | N | N | N |
| S-67 | Status bar | N | P | **F** | N | N |
| S-68 | Trace graph + zoom | N | **P** | **F** | N | N |
| S-69 | Node tooltip | N | **F** | N | N | N |
| S-70 | Trace detail + exports | **F** | P | **F** | **F** | N |
| S-71 | Print audit report | N | · | N | N | · |
| S-72 | Topbar KPI strip | N | · | N | N | · |
| S-73 | Topbar actions | N | · | **F** | N | · |
| S-74 | Legend bar | N | · | N | N | · |
| S-75 | Delta panel | N | · | N | N | · |
| S-76 | Settings panel (form) | N | · | **F** | N | · |
| S-77 | FG on hand (disclosure) | N | · | **F** | N | · |
| S-78 | Bulk-bin WIP (disclosure) | N | · | **F** | N | · |
| S-79 | Product catalog (form) | N | · | **F** | N | · |
| S-80 | Orders in / plans out | N | · | **F** | N | · |
| S-81 | CSV import result | N | · | N | N | · |
| S-82 | "How this works" | N | · | **F** | N | · |
| S-83 | Production board table | N | · | P | N | · |
| S-84 | Board cell copy / more | N | · | **F** | N | · |
| S-85 | Schedule panel | N | · | **F** | N | · |
| S-86 | Pin modal | N | · | **F** | N | · |
| S-87 | Order book | N | · | **F** | N | · |
| S-88 | Add order line form | N | · | **F** | N | · |
| S-89 | Add-order validation alert | N | · | N | N | · |
| S-90 | Order book empty state | N | · | N | N | · |
| S-91 | Scheduler print view | N | · | N | N | · |

---

## Findings

---

### TOUCH-003 — Every tappable element has a hit region of at least 44×44 pt — **CRITICAL, both platforms**

**FAIL on essentially every interactive control in the product.** *Hard rule.*

The rule is explicit that it applies "regardless of input method and on desktop tables too (a 16-px glyph still gets a 44-pt hit area)". Measured heights from `dashboard/dashboard.css`:

| Class | Declaration | Approx. hit height | Where used |
|---|---|---|---|
| `.note-action-btn` | `padding: 3px 6px; font-size: 11px` (`1111-1116`) | **≈22 × 20px** | Notes Edit / **Delete** |
| `.order-ready-checkbox` | `width: 16px; height: 16px` (`1336-1342`) | **16 × 16px** | Factory Ready write |
| `.note-checkbox` | `width: 18px; height: 18px` (`1026-1033`) | **18 × 18px** | Mark note done |
| `.show-more-btn` | `padding: 0; font-size: 12px` (`827-838`) | **≈16px** | Reveal 96 hidden activity rows |
| `.lot-link` | inline span, `font-size: 12px` (`840-848`) | **≈15px** | Open lot panel (the app's main drill path) |
| `.btn-close` | `font-size: 22px`, no padding (`880-889`) | **≈22px** | Close all four modals/panels |
| `.order-expand-toggle` | `width: 22px; height: 22px` (`1344-1349`) | **22 × 22px** | Expand order lines |
| `.btn-sm` | `padding: 4px 10px; font-size: 12px` (`394-403`) | **≈24px** | ER Edit/Close/Cancel, Release, Done, Cancel ×3, calendar arrows, Retry, Inventory |
| `.btn-theme` | `padding: 5px 8px; font-size: 16px` (`223-233`) | **≈26px** | Theme toggle |
| `.notes-filter-btn` | `padding: 5px 12px; font-size: 12px` (`977-986`) | **≈28px** | Notes category filters |
| `.site-nav-link` | `padding: 5px 14px` (`109-113`); `8px 14px` at ≤768px (`129`) | **≈28 / 34px** | Top-level navigation |
| `.er-product-option` | `padding: 6px 10px` (`2237`) | **≈27px** | Choose a product in the ER modal |
| `.btn-refresh` | `padding: 6px 14px; font-size: 13px` (`236-246`) | **≈30px** | Every primary commit in the app |
| `.supplies-subtab` | `padding: 6px 10px; font-size: 13px` (`2284-2297`) | **≈32px** | Supplies category switch |
| `.search-item` | `padding: 8px 12px` (`194-201`) | **≈33px** | Global search results |
| `.tab` | `padding: 10px 20px; font-size: 14px` (`265-278`) | **≈41px** | Section switch |

Traceability: `.lot-pill` ≈20px (`traceability.html:133-137`), `.graph-ctrl-btn` ≈27px (`158-162`), `.export-btn` ≈29px (`209-213`), `.dir-btn` / `.trace-btn` ≈35px (`111-125`).
Scheduler: `#topbar button { padding: 5px 10px; font-size: 12px }` ≈27px (`:38`); `.copyday`, `.otbtn`, `.o-action` smaller still.

**Two controls in the whole product meet the rule**, and they were clearly deliberate:
- `.recent-entries-refresh { min-width: 44px; min-height: 44px }` (`dashboard.css:290`)
- `.recent-entries-retry { min-height: 44px }` (`dashboard.css:316`)
- (`.day-card-trigger { min-height: 150px }` at `424` and scheduler `td.cell { height: 72px }` also pass, but as a by-product of being large content blocks.)

**Focus/selection clause.** Only three selectors define a focus indicator: `.day-card-trigger:focus-visible` (`436-441`), `.supplies-subtab:focus-visible` and `.supply-item-row:focus-visible` (`2300-2304`). Every other button relies on the user agent's default ring. Worse, five interactive elements are **not focusable at all** — `.lot-link` (`dashboard.js:935`), `tr.expandable` (`923`), `.search-item` (`1530`), `.er-product-option` (`3526`), `.product-lot-row` (`1465`) — so they have no focus state to show.

**Suggested fix:** Add a single global rule — `button, [role="button"], .lot-link { min-height: 44px; min-width: 44px }` for touch, or the lighter-touch version that keeps desktop density: give every interactive class a `::before` pseudo-element expanding the hit area to 44×44 without changing the visual size. Then raise the five non-focusable controls to real `<button>`/`<a>` elements. This one change resolves TOUCH-003 across 60+ screens and materially reduces the LAYOUT-012 risk.

---

### ACTION-002 — Every button visibly shows pressed, hover, disabled, and selected states — **CRITICAL**

**FAIL, app-wide.** *Hard rule.*

**Evidence:** No `:active` rule exists in `dashboard/dashboard.css`, `dashboard/mini-calendar.css`, `dashboard/sankey.html`, `dashboard/process-flow.html`, `dashboard/traceability.html`, or `dashboard/scheduler/seven-wells-production-board.html` (verified by grep across all six files). Not one button in the product has a press state.

The rule's stated consequence is exact: *"Without a press state people can't tell whether the tap registered, so they tap again — the classic path to duplicate submissions."* Two commit paths in this app are non-idempotent and unguarded at the moment of the click:
- `submitAllocation` disables its button only **after** the handler starts (`dashboard.js:3069-3070`), so a double-tap inside the same frame can queue two reservations.
- `saveNote` (`dashboard.js:1809-1847`) **never disables** `#note-save-btn` — a double-tap creates two notes.

**Hover:** present on nearly everything ✓ (`dashboard.css:247`, `404`, `641`, `700`, `889`, `987`, `1121`, `1319`, `1358`, `1747`, `1837`, `2298`, `2344`).

**Disabled:** only two classes are styled — `.btn-refresh:disabled, .btn-sm:disabled { opacity: 0.55; cursor: not-allowed; pointer-events: none }` (`dashboard.css:2423-2424`) and `.trace-btn:disabled` (`traceability.html:125`). Nine other button classes have no disabled treatment. And `pointer-events: none` on the disabled rule means the `title` explaining *why* a control is disabled can never be shown — see INPUT-021.

**Selected/toggled:** correctly implemented — `.tab.active` (`279`), `.notes-filter-btn.active` (`988`), `.supplies-subtab.active` (`2299`), `.dir-btn.active` (`traceability.html:116`), `.day-card-trigger.selected` (`442-447`), `.order-expand-toggle.expanded` (`1366`). ✓ But `.notes-filter-btn.active` uses the same solid `--primary` fill as the primary button style, which the rule forbids: *"don't reuse [the selected look] as a plain button style."*

**Suggested fix:** One rule — `button:active, [role="button"]:active { transform: translateY(1px); filter: brightness(0.92) }` — plus disabling every commit button synchronously at the top of its handler.

---

### ACTION-003 — One prominent (filled) primary action per view; at most two — **CRITICAL**

**S-24 (Orders toolbar) — FAIL.** `dashboard/index.html:233-235` — Export CSV, Export Matrix (xlsx), and Refresh are all `.btn-refresh`, i.e. three filled accent buttons in one row. None of them is the view's primary action (which is "open the order that needs work"). The audit test — *"is it the only filled control in its area"* — fails three times over.

**S-41 (ER toolbar) — FAIL.** `dashboard/index.html:274-275` — "+ New Expected Receipt" and "Refresh" both filled. New is primary; Refresh is maintenance.

**S-03 (App header) — PARTIAL.** `dashboard/index.html:41` — the global Refresh is the only filled control on the page chrome, so it reads as the app's primary action. It is a fallback, not a task.

**S-14 (Recent Entries) — PARTIAL.** `dashboard/index.html:118` — same shape, filled Refresh on a read-only feed.

**S-19 (Notes toolbar) — PARTIAL.** `dashboard/index.html:189` — "+ New" is correctly the single filled button, but `.notes-filter-btn.active` (`dashboard.css:988`) is *also* a solid `--primary` pill sitting four inches away, so the view shows two filled accent controls.

**S-33 (Order Detail edit) — PARTIAL.** `dashboard/dashboard.js:2707-2709` — "Save Header" and "Save Lines" are both filled. Two is the rule's limit, so this is technically inside it, but both commit the same record and the user must compare rather than act.

**S-23, S-35, S-38, S-89 (native dialogs) — FAIL.** `dashboard/dashboard.js:1771`, `2819`, `3096`; `dashboard/scheduler/…:1565`, `:1321`. A native `confirm()`/`alert()` gives the app no control over which button is prominent; the browser makes **OK** both prominent and Enter-default, and OK is the destructive choice in all four. See ACTION-008.

**S-55 — FAIL.** `dashboard/dashboard.js:1377-1380` — the disambiguation buttons have no primary/secondary distinction and, because `--bg-card` is undefined, no legible fill at all (see `01-nav-layout.md` LAYOUT-002 item 7).

**Correct implementations — cite as the pattern:** Traceability (`traceability.html:283` — one filled `.trace-btn`, everything else outlined) and the scheduler topbar (`scheduler:11` — one `.primary` amber button, two neutral).

**Suggested fix:** Define `.btn-primary` / `.btn-secondary` / `.btn-danger` and reassign: Refresh and both Exports → secondary; "+ New" and Save/Submit → primary; Delete/Release/Cancel-record → danger.

---

### ACTION-006 — Fixed semantic button roles; red only on data-destroying actions — **CRITICAL**

**FAIL, app-wide.** *Hard rule.*

**There is no destructive button style.** Grepping `dashboard.css` finds no `.btn-danger`, `.btn-destructive`, or equivalent. The four destructive actions each invent something different:

| Action | Treatment | Evidence |
|---|---|---|
| Delete note | Neutral outline; red **only on hover** | `dashboard.css:1122` (`.note-action-btn.delete:hover`) |
| Release reservation | `.btn-sm` with red *text*, neutral border | `dashboard.css:2081` |
| Cancel expected receipt | `.btn-sm` neutral; turns **amber** when armed | `dashboard.css:2230` (`.er-armed`) |
| Delete scheduler order line | Neutral `<a>`; hover `#ece9e1` | `scheduler:158`, `:1537` |

The Expected Receipts case is the sharpest: **"Close" (benign) and "Cancel" (destructive) get the identical amber armed treatment** (`dashboard.js:3470-3483` applies `.er-armed` regardless of which verb was pressed). Amber therefore means "press again", not "this destroys data".

Changing an order's status to `cancelled` — arguably the most consequential single action in the dashboard — is an ordinary `<option>` in a neutral `<select>` (`dashboard.js:2696-2698`, `3190`) with no destructive styling whatsoever.

**Primary role is not reserved either.** `.btn-refresh` carries twelve different semantic roles (enumerated under LAYOUT-002 in `01-nav-layout.md`).

**Red is used for non-destructive meanings.** `--danger` appears on: overdue dates (`css:1671`, `1093`, `1541`), line shortages (`1819`), blocking readiness chips (`1634-1638`), the Blocked dispatch pill (`1604-1607`), error banners (`935-941`, `1462`, `2368`, `2397`), and traceability data gaps (`traceability.html:33`). Every one of those is legitimate *status* under FEEDBACK-013 — but with Release also red, the colour now carries two meanings in the same table, which is what the rule forbids.

**Suggested fix:** Introduce `.btn-danger` (red fill, white text) and apply it to Delete Note, Release, Cancel Expected Receipt, and the scheduler's Delete Order Line. Move the SO "Cancelled" transition out of the status select into a separate, confirmed, red action. Keep red-as-status on chips and text only.

---

### ACTION-008 — Destructive actions are unmistakable and never the default — **CRITICAL**

**S-23, S-35, S-38 (native confirms) — FAIL.** *Hard rule.*

**Evidence:** `dashboard/dashboard.js:1771` (`confirm('Delete this item?')`), `2819` (`window.confirm('Change SO-… status from Ready to Cancelled?')`), `3096` (`window.confirm('Release this 2400 lb reservation? …')`); `dashboard/scheduler/…:1321` (`confirm('Clear all orders, pins, inventory, and settings?')`).

The audit test asks: *"Is there any dialog or screen where Enter or the most prominent button destroys or reverses data?"* In all four cases the answer is yes. A native `confirm()` makes **OK** the visually dominant and Enter-activated button, and OK is always the destructive choice. The standard's own worked example inverts this: *"'Void this shipment?' → 'Keep Shipment' is primary/Enter; 'Void' is red and secondary."* A native `confirm()` cannot express that, which is why the rule effectively prohibits it for destructive choices.

The scheduler case is the worst: a stray Enter on `confirm('Clear all orders, pins, inventory, and settings?')` wipes an entire plan held only in `localStorage` (`scheduler:376-378`) with no undo.

**S-20 (Notes delete) — FAIL.** `dashboard/dashboard.js:1733`; `dashboard/dashboard.css:1122`. The Delete control is a 22×20px neutral glyph button that only turns red on hover — so on touch it is *never* red. It is also 4px from Edit (see LAYOUT-012). It is neither unmistakable nor placed to be noticed.

**S-87 (Scheduler order-line delete) — FAIL.** `dashboard/scheduler/…:1537` and `:1553-1554`. `⌫` sits immediately beside `✕` (Exclude), styled identically (`:158`), with **no confirmation and no undo**. Exclude is reversible (`data-incl` restores it, `:1551-1552`); Delete is not.

**S-80 (Scheduler reset) — FAIL.** `scheduler:1321` — as above.

**S-25, S-26, S-28 (Factory Ready) — PARTIAL.** `dashboard/dashboard.js:2452-2487`. Not destructive, but it writes an outward-facing signal the floor acts on, from a 16×16px checkbox, with no confirmation and no undo affordance. Recoverable by unticking, hence PARTIAL rather than FAIL.

**S-37, S-43 — PARTIAL.** Release has red text at rest (the closest the app gets to correct); ER Cancel has the two-step arm (the best confirmation pattern in the app) but the wrong colour.

**Suggested fix:** Replace all four native `confirm()` calls with the app's own modal shell, giving the *non*-destructive option the primary role and Enter binding, and the destructive option `.btn-danger` in the secondary position. Add the ER two-step arm to the scheduler's delete.

---

### ACTION-001 — Match the control type to the interaction

**S-08 (Calendar mode toggle) — FAIL.** *High · Hard rule.* `dashboard/index.html:79`; `dashboard/dashboard.js:676`, `683`, `4212-4216`. A two-state **selection** is implemented as an action `<button>` whose label shows the state you are *not* in ("Month View" while in 5-day mode). Button means "this happens now"; here it means "you are currently in the other mode".

**S-16 (Daily Entries mode) — FAIL.** `dashboard/index.html:138-141`. A two-option view selection in a `<select>` — a menu control for what the rule specifies as a segmented case.

**S-24, S-26 (Orders status filter) — PARTIAL.** `dashboard/index.html:206`; `dashboard/dashboard.js:2040-2047`, `2244-2259`. One `<select>` mixes a *mode* (Dispatch Queue — different endpoint, different columns, reveals a second control) with *statuses* (a filter over one dataset). The rule's own Factory Ledger example warns against exactly this shape: *"Don't add 'Refresh' as a fourth segment to a status filter."*

**S-49 — PARTIAL.** `dashboard/dashboard.js:3849` — a `<tr>` with `role="button"` used as a disclosure. `aria-expanded` is set correctly ✓ and keyboard support exists ✓, but `role="button"` on a table row removes the row from the table's semantics for assistive technology. `role="button"` should sit on a child control, or the row should expose `aria-expanded` without the button role.

**Passing:** checkbox for on/off state (Factory Ready, all "show completed" toggles, overdue filters) ✓; `<button>` groups with `.active` for segmented selection (notes filters, supplies sub-tabs, trace direction) ✓; the trace direction group is correctly kept separate from the Trace action button (`traceability.html:279-283`) ✓.

---

### ACTION-005 — Every button states its outcome; icon-only controls carry a tooltip **and** an accessible name

**S-33 — FAIL.** `dashboard/dashboard.js:2709` — the button labelled **"Done"** exits edit mode *without saving* (its handler at `2850-2856` only re-renders). The rule names "Done" as a label to avoid, and here it is worse than vague: a user who has typed changes and presses "Done" will reasonably expect them to be committed. They are silently discarded.

**Icon-only buttons whose accessible name is the glyph — FAIL:**

| Control | Evidence | Accessible name computes to |
|---|---|---|
| Notes Edit | `dashboard.js:1732` — `<button class="note-action-btn edit" title="Edit">✎</button>` | "✎" |
| Notes Delete | `dashboard.js:1733` — `title="Delete"`, content `✕` | "✕" |
| Order expand | `dashboard.js:2320` — `title="Show line items"`, content `▸` | "▸" |
| Calendar prev/next | `index.html:76`, `78` — `title="Previous"`/`"Next"`, content `←`/`→` | "←" / "→" |
| Theme toggle | `index.html:40` — `title="Toggle light/dark mode"`, content `☾` | "☾" |
| Graph zoom in/out | `traceability.html:310-311` — no title, no aria-label, content `+`/`−` | "+" / "−" |

A `title` attribute does not override text content when computing the accessible name, so screen readers announce the glyph. Cross-referenced under ACCESS-010.

**Correctly labelled icon-only controls — the existing good pattern:** all four `.btn-close` buttons carry `aria-label="Close"` (`index.html:341`, `402`, `450`, `512`) ✓; `#navToggle` has `aria-label="Menu"` (`17`) ✓; `.mini-calendar-nav` has `aria-label="Previous month"` / `"Next month"` (`mini-calendar.js:68`, `72`) ✓; the scheduler's `.copyday`, `.o-excl`, and `.o-del` all carry descriptive aria-labels including the record name (`scheduler:1343`, `1536-1537`) ✓.

**Vague labels — PARTIAL:** "Save" in the note and ER modals (`index.html:438`, `500`) where "Save Note" / "Save Expected Receipt" is available; "Inventory" as a button label (`dashboard.js:3257`) where the action is "show on-hand for this line".

**Ambiguous scope — PARTIAL:** five controls labelled "Refresh" with five different scopes — the whole app (`index.html:41`), Recent Entries (`118`), Orders (`235`), Expected Receipts (`275`), Supply Requests (`327`). Nothing on the label says which.

**S-20, S-23, S-35, S-38, S-87, S-89 — FAIL.** Confirmation dialogs do not repeat the action verb. `confirm('Delete this item?')` (`1771`) offers OK/Cancel, not "Delete"/"Keep". The rule: *"In a confirmation dialog the confirming button repeats the action ('Void')."* The ER two-step arm does this correctly — `"Cancel? Confirm"` (`3473`) ✓.

---

### ACTION-007 — Enter triggers the primary non-destructive action in every desktop form and dialog

**S-22 (Note modal) — FAIL.** `dashboard/index.html:452-503` — the modal body is a `<div>`, not a `<form>`, and no keydown handler is bound (`dashboard.js:1849-1877`). Enter in the title field does nothing.

**S-45 (ER modal) — FAIL.** `dashboard/index.html:404-441` — same shape (`<div class="note-modal-body">`), no form, no Enter handler (`dashboard.js:3628-3650`). Compounding this: `#er-product-search` opens a results list on input, and Enter neither selects the first result nor saves — so the fastest possible path (type "Frank", Enter, Enter) does nothing at all.

**S-33 (Order Detail edit) — FAIL.** `dashboard/dashboard.js:3256-3272` — quantity and price inputs sit in table cells with no form and no Enter binding. Editing ten line quantities requires ten mouse trips to "Save Lines".

**S-55 — FAIL.** `dashboard/dashboard.js:1387-1391` — the disambiguation buttons are click-only; there is no default and no keyboard path.

**S-76, S-79, S-86, S-88, S-89 (scheduler) — FAIL.** No form elements and no Enter bindings anywhere (`scheduler:1435-1476`, `:1555-1580`).

**Correct implementations:**
- Supply Request modal — a real `<form>` with `type="submit"` (`index.html:343`, `390`) and a submit handler (`dashboard.js:4097`) ✓
- Allocation form — `<form>` + `type="submit"` + `preventDefault` handler (`dashboard.js:2917`, `2926`, `3164-3167`) ✓
- Traceability search — explicit `keydown` binding: `if (e.key === 'Enter') { closeDropdown(); runTrace(); }` (`traceability.html:442-444`) ✓

**S-04 — PARTIAL.** `dashboard/dashboard.js:4219-4223` — the global search has no Enter handling. There is no results page to navigate to, but Enter should open the first result.

**"…and closes temporary views":** no modal in the product closes on Enter, and only the Supply Request modal closes on Escape (`dashboard.js:4103-4107`). See LAYOUT-002 item 3.

---

### ACTION-010 — Buttons look like buttons

**S-17, S-18 (Show more) — FAIL.** *High.* `dashboard/dashboard.css:827-838` — `.show-more-btn { background: none; border: none; padding: 0 }` renders as bare accent text. It is the only control that reveals up to 96 hidden rows (`dashboard.js:1152-1157`), it sits in a table footer (not a toolbar, menu, or dialog, so the rule's exception does not apply), and it has a 16px hit height.

**S-10…S-13, S-25…S-27, S-42, S-56 — FAIL.** Rows carrying the primary interaction with no button shape at all: `tr.expandable` (`dashboard.js:923`, `1007`, `1111`, `1196`, `1249`), `.order-row` (`2319`), `.er-row` (`3428`), `.product-lot-row` (`1465`). The only affordance is `cursor: pointer` (`dashboard.css:699`, `1318`, and inline styles), which does not exist on touch.

**S-83, S-84, S-87 (scheduler) — FAIL.** `scheduler:114-115` — `td.cell { cursor: pointer }` with a hover outline is the sole affordance for the board's primary interaction (open the pin modal). `.otbtn` is a `<span>` (`:1345`); `.o-action` are `<a href="#">` (`:1536-1537`).

**S-08 — PARTIAL.** `dashboard/dashboard.css:418-428` — `.day-card { border: 0; background: var(--surface) }` on a `.section` that is also `--surface` (`331`). A production day is a `<button>` and a no-production day is a `<div>` (`dashboard.js:764`, `766`), but they look nearly identical: the only differences are a 3px left border (`431-433`) and a 10px "View details" hint (`520-527`).

**S-49 — PARTIAL.** The supplies row is the best of the row-as-control implementations — `role="button"`, `tabindex="0"`, `aria-expanded`, keyboard handler, focus ring (`dashboard.js:3849`, `3863-3868`; `dashboard.css:2300-2304`) — but it still has no button shape.

**S-04, S-63 — PARTIAL.** Dropdown result items are `<div>`s (`dashboard.js:1530`, `traceability.html:463`). Inside a dropdown this is conventional, so the exception broadly applies; they still lack roles and keyboard access.

---

### ACTION-011 — Prefer standard/native controls; a custom control must replicate every standard state

**S-45 (ER product type-ahead) — PARTIAL.** *Medium.* `dashboard/dashboard.js:3516-3540`, `3644-3649`; `dashboard/index.html:407-410`. A hand-rolled combobox with no `role="combobox"`, no `aria-expanded`, no `aria-activedescendant`, no arrow-key navigation, no Enter-to-select, and no close-on-blur (the list is hidden only by `setErProduct`, `3513`). It replicates hover (`dashboard.css:2238`) and nothing else.

**S-04, S-63 — FAIL.** `dashboard/dashboard.js:1522-1613`; `dashboard/traceability.html:448-491`. Same shape, plus the Traceability version builds its handlers by string interpolation into an inline `onclick` attribute (`traceability.html:463`) using an `escAttr` (`446`) that escapes for HTML but not for the JavaScript string literal it is injected into — a lot code containing a backslash would break the handler.

**S-10…S-13, S-25…S-27, S-42, S-55, S-56 — FAIL.** Clickable `<tr>`/`<span>` elements with no role, no tabindex, no keyboard handler, and no focus state.

**S-22, S-45, S-53 — PARTIAL.** All three modals are custom `<div>` overlays rather than `<dialog>`. None traps focus. Initial focus is set in two of three (`dashboard.js:3570` ER, `4002` Supply) and not at all in the Note modal (`1785-1802`). Focus is restored on close in one of three (`4007`, Supply Request only).

**S-49 — PASS. Cite as the reference implementation.** `dashboard/dashboard.js:3849`, `3860-3869`; `dashboard/dashboard.css:2300-2304`. The supplies row is the only custom control in the product that replicates keyboard activation (Enter and Space, with `preventDefault`), focus-visible styling, and ARIA state.

**Native controls used well:** `<select>` for suppliers, products, requesters, lots, priority, entity type, status filters ✓; `<input type="date">` throughout ✓; `<input type="number">` for every quantity except `#er-qty` ✓; `<input type="checkbox">` and `type="radio"` ✓.

---

### ACTION-004, ACTION-009, ACTION-012 — remaining findings

**ACTION-004 (S-22, S-45, S-53) — FAIL.** *Medium.* `dashboard/index.html:390-391`, `438-439`, `500-501`. In every modal, Save is `.btn-refresh` (`padding: 6px 14px`, 13px, `dashboard.css:236-246`) and Cancel is `.btn-sm` (`padding: 4px 10px`, 12px, `394-403`) — a ~6px height difference within a two-button choice set. The rule: *"Buttons forming a set are the same size and height; the preferred one uses a more prominent style."*
**(S-33) — FAIL.** `dashboard/dashboard.js:2707-2709` — Save Header / Save Lines (`.btn-refresh`) beside Done (`.btn-secondary`, `padding: 6px 10px`, 12px, `dashboard.css:1828-1836`). Three sizes in one action row.
**(S-24, S-41, S-47) — PARTIAL.** Toolbar rows mixing `.btn-refresh` and `.btn-sm` (`index.html:327` — Supplies Refresh is `.btn-sm` while every other Refresh in the app is `.btn-refresh`).
**Correct:** ER row actions are three uniform `.btn-sm` (`dashboard.js:3438-3442`) ✓; trace direction buttons are two uniform `.dir-btn` ✓.

**ACTION-009 — PARTIAL, app-wide.** *Medium.* No button in the product uses a trailing ellipsis or any equivalent. Four buttons open further input with no signal: "+ New" (`index.html:189` → note modal), "+ New Expected Receipt" (`274` → ER modal), "Request Supply" (`295` → supply modal), and the ER row "Edit" (`dashboard.js:3439` → ER modal). The "+" prefix is a partial convention on the first two. Conversely `Allocate`, `Preview all remaining lines`, `Save`, and `Release` all act immediately ✓ and correctly carry no ellipsis.

**ACTION-012 — PARTIAL / FAIL.** *Medium.*
**(S-64) — FAIL.** `dashboard/traceability.html:280-281` — the two direction segments are `"Forward →"` and `"← Backward"`: text plus a glyph, with the glyph on **opposite sides** of the two segments, and unequal widths. The rule forbids mixing text and icon within one segmented control and requires uniform widths.
**(S-01, S-05, S-47) — PARTIAL.** Segment widths are content-sized, not equal: `.site-nav-link` (`css:109`), `.tab { flex: 0 0 auto }` (`277`) with labels ranging from "Notes" to "Expected Receipts", `.supplies-subtab` (`2284`) with a trailing count badge of variable width.
**(S-19) — PASS on content type** (all text), PARTIAL on widths.

---

### INPUT-002 — Every field has a persistent label naming the value and its unit — **High, Hard rule**

**S-03, S-24, S-41 (filter and search fields) — FAIL.** The rule is explicit that a placeholder *"can never be the sole identifier"* because it disappears on input.

| Field | Evidence | Identifier |
|---|---|---|
| `#global-search` | `index.html:33` | placeholder only — "Search SKU, lot, SO, customer..." |
| `#orders-customer-search` | `index.html:223` | placeholder only — "Filter by customer..." |
| `#er-text-filter` | `index.html:267` | placeholder only — "Filter by product / supplier / reference..." |
| `#orders-status-filter` | `index.html:205` | no label, no `aria-label` |
| `#daily-entries-mode` | `index.html:138` | no label |
| `#date-from` / `#date-to` | `sankey.html:302`, `304` | no labels, separated only by the word "to" |
| `#lotSearch` | `traceability.html:276` | placeholder + a leading 🔎 glyph |

Once Luz types a customer name into `#orders-customer-search`, nothing on screen says what that box filters. `#supplies-search` shows the correct pattern with an `.sr-only` label (`index.html:311-314`), and `#orders-dispatch-filter` shows it with an `aria-label` (`218`) — the convention exists and is applied inconsistently.

**S-33 (Order Detail edit inputs) — FAIL.** `dashboard/dashboard.js:3259-3266`. The line quantity and price inputs have **no labels at all**. Their only identifier is the column header, which on a horizontally-scrolled 10-column table (`css:1817` sets `min-width: 1040px`) may be off-screen. The unit sits *below* the field in `.pallet-secondary` (11px, `--text-muted`, `css:1436-1444`) and the price input carries **no currency symbol** — the read-only view renders `$` (`3270`) and the editable view does not.

**S-45 (ER modal in edit mode) — FAIL.** `dashboard/dashboard.js:3549-3557`. When editing an existing receipt, `#er-product-group` is hidden (`3551`) because the product is fixed. The result is that the modal titled *"Edit Expected Receipt #12"* shows a disabled supplier, a quantity, a date, a reference and notes — and **never names the product**. Luz edits an expected quantity without seeing what it is for. The rule's second clause is precisely this: *"Any dedicated entry screen or prompt carries a title naming what to enter, in what unit, and for which record."*

**Correct implementations — cite these:**
- `#er-qty` — label "Expected qty (lb)" **with the unit**, placeholder "e.g. 2000" as a format example (`index.html:418-419`). This is exactly what the rule asks for.
- `#supply-request-qty` — label plus a **live unit span** that updates from the selected product (`index.html:362-366`; `dashboard.js:3978-3985`).
- `.allocation-quantity-input` — "Quantity (lb)" (`dashboard.js:2923`).
- Scheduler pin modal — *"Force quantity (cases)"* under a heading that names the record and date: *"Pin — Bake · Mon 8 · planned: 42 cases"* (`scheduler:1447-1450`). Best-in-class.
- Ingredients table — the column header carries the unit when all rows share one, and each cell carries its own when they do not (`dashboard.js:1102-1110`) ✓.

---

### INPUT-007 — Validate as early as the check can be made reliably — **CRITICAL**

**S-45 (`#er-qty`) — FAIL.** *Hard rule.*

**Evidence:** `dashboard/index.html:419` — `<input type="text" id="er-qty" inputmode="decimal" placeholder="e.g. 2000">`; `dashboard/dashboard.js:3580-3581` — `const qty = parseFloat(document.getElementById('er-qty').value); if (!(qty > 0)) { showError(...); return; }`.

A quantity field declared as `type="text"`. Non-numeric characters are not blocked at keystroke, and the only check happens at commit — where `parseFloat` **silently coerces**. Typing `12O` (letter O for zero) yields `12`. Typing `2,000` yields `2`. Typing `2000 lb` yields `2000` (correct by luck). The rule's stated purpose is *"Prevents '12O' (letter O)"* and its hard clause is *"Invalid data is never silently accepted."* This field silently accepts a 1000× error on an incoming-delivery quantity.

**S-88, S-89 (Scheduler add order) — FAIL.** `scheduler:1565` — `if(!sku||!(qty>0)||!due) return alert("SKU, qty, and due date are required.")`. Commit-time only, via a native alert, with no indication of which of the three is missing.

**S-22 (Note modal) — PARTIAL.** `dashboard/dashboard.js:1812-1814` — the only validation is `if (!title) { alert('Title is required'); return; }` at commit. The field carries no `required` attribute (`index.html:463`) and Save is never disabled.

**S-36 (Allocation) — PARTIAL.** `dashboard/dashboard.js:3052-3059` validates that the quantity is a positive number and that a lot is chosen ✓ — but the **cross-field** check (requested quantity versus what is actually coverable) is left entirely to the server, which returns `OVER_ALLOCATION` with a `coverable_lb` (`2977-2980`). The client already holds `unallocated_need_lb` for the selected line and uses it to prefill the field (`3038-3039`), so the comparison could be made inline the moment both values exist — which is what the rule requires: *"Validate cross-field and consequential rules inline as soon as the inputs are present."* The error message it eventually shows is excellent (*"Only 240 lb is coverable. Reduce the request to 240 lb or release a competing reservation."*) — it simply arrives one round-trip too late.

**S-45 (`#er-product-search`) — PARTIAL.** `dashboard/dashboard.js:3644-3649`, `3597`. Every keystroke calls `setErProduct(null)`, clearing the hidden product id. A user who types the full product name but never clicks a result gets *"Pick a product from the search results."* only after filling supplier, quantity, date, reference and notes. A blur-time check would catch it immediately.

**S-16 — PASS.** Date and mode are native controls whose values cannot be invalid.

**S-86 (Scheduler pin modal) — PASS. This is the reference implementation for the whole rule.**
`scheduler:1457-1462` — a `feas()` function bound to `oninput` and `onchange` recomputes station capacity live and reports either *"Feasible: capacity ≈ 42 cases (3-worker crew, default tier ×100% day)."* or *"Over capacity: ≈ 30 cases available (3-worker crew…). To make 50 real: add a 4th bake worker (3rd pan) or overtime. The pin will be honored but flagged red."* Inline, cross-field, at the moment both values exist, with a remedy. Every commit form in the dashboard should be measured against this.

---

### INPUT-008 — Numeric fields accept only numbers and always show their unit and format — **CRITICAL**

**S-45 (`#er-qty`) — FAIL.** `dashboard/index.html:419`. `type="text"` accepts any character, including pasted text (see INPUT-020). Unit is in the label ✓; the input itself is unconstrained.

**S-33 (`.order-line-price-input`) — FAIL.** `dashboard/dashboard.js:3264`. `type="number"` ✓ constrains input, but the field shows **no currency at all** — no `$` prefix, no "USD" suffix, nothing in a label. Only an 11px muted `<small>` beneath it reading the raw `price_basis` value ("case price" / "unit price"). The read-only rendering two lines away at `3270` does show `'$' + Number(l.case_price).toFixed(2)`.

**S-33 (`.order-line-qty-input`) — PARTIAL.** `dashboard/dashboard.js:3259-3261`. `type="number" step="0.01"` ✓; unit is present but only in `.pallet-secondary` below the field at tertiary emphasis.

**S-49 (Supplies "Incoming" column) — PARTIAL.** `dashboard/dashboard.js:3845-3848`. `unitIsPounds` is computed from the item's own unit; when the item is *not* measured in pounds the incoming total is still rendered as `${fmtWt(total)} lb`. The result is an "On Hand" column in the item's unit sitting beside an "Incoming" column in pounds. The code is aware of the mismatch (it appends "lb" deliberately) but the two columns are not comparable and the header does not say so.

**S-88 — PARTIAL.** `scheduler:1567` — `#ao-qty` is `type="number" min="1"` ✓ but the unit ("case" or "lb", derived from the SKU at `:1572`) is never shown beside the field.

**Passing and strong:** `#supply-request-qty` (`index.html:364` — `type="number" min="0" step="any" inputmode="decimal"` with a live unit span) ✓; `.allocation-quantity-input` (`dashboard.js:2923` — `type="number" min="0.0001" step="0.0001"`) ✓; `#pin-qty` (`scheduler:1449`) ✓. Display formatting is consistent and unit-bearing throughout — `fmt`, `fmtInt`, `fmtWt`, `fmtLbs`, `fmtQty`, `fmtQtyCases`, `formatInventoryUnits` (`dashboard.js:66-105`, `393-399`, `1394-1398`, `2004-2007`) all emit an explicit unit. This is one of the codebase's genuine strengths.

*Note:* all formatters hardcode `'en-US'`. The rule warns against hardcoding locale assumptions; for a single US site this is a defensible deliberate choice, recorded here only so it is a decision rather than an accident.

---

### INPUT-011 — Prefer selection over typing — **CRITICAL**

**S-22 (`#note-entity-id`) — FAIL.** *Hard rule.*

**Evidence:** `dashboard/index.html:485-497` — "Pin to" is a `<select>` of entity **types** (Product / Lot / Customer / Supplier) and "Entity Name / ID" is a free `<input type="text">` with placeholder *"e.g. Granola Classic 25 LB"*. The typed string is stored verbatim (`dashboard.js:1821`) and rendered as a chip (`1720`).

The audit test is *"For every text field: is the set of valid values enumerable? If yes, why is it a text field?"* Every one of the four entity types is enumerable, and the app already has the endpoints to enumerate them: `/products/search` (used at `dashboard.js:3520`), `/suppliers` (`3499`), and the global `/search` which covers products, lots, orders and customers (`1514`). A note pinned to "Granola Clasic 25 LB" is a note nobody will ever find.

**S-53 (`#supply-request-unit`) — PARTIAL.** `dashboard/index.html:358`; `dashboard.js:4044`, `4053`. Free text for a unit ("e.g. box, roll, each"), concatenated into `item_text` as `"${item} (unit: ${unit})"`. Units are enumerable — every row in `state.supplies.inventory` already carries one (`3984`). The result never reconciles with the catalogue.

**S-63 (Traceability lot entry) — PARTIAL.** `dashboard/traceability.html:432-440`. A type-ahead exists, but free text is the *primary* path: `selectedLotCode = q` is assigned on every keystroke (`437`) and `#traceBtn` is enabled at two characters (`439`), so the user can trace a string that was never in the list. Combined with the 100-transaction index cap (NAV-012), typed input is often the only path available.

**S-24, S-41 — PARTIAL.** `#orders-customer-search` and `#er-text-filter` are free text over enumerable sets. As *filters* rather than record references this is acceptable; a picker would still prevent "Restaurent Depot" returning silently empty.

**S-88 — PARTIAL.** `scheduler:1557` — `#ao-cust` is free-text customer; `#ao-sku` correctly uses `skuPicker` (`:1558`).

**Reference implementations — cite these:**
- **Supply Request product** (`index.html:346-349`; `dashboard.js:3681-3691`) — a `<select>` over the enumerable catalogue **plus** an explicit `"__other__"` → "Product not listed" escape that routes to a free-text `item_text` on the *request* rather than creating a master record (`4042-4056`). This is precisely the INPUT-011 + INPUT-014 pattern the standard describes.
- **ER supplier** — `<select>` populated from `/suppliers` (`dashboard.js:3497-3507`) ✓
- **Allocation lot** — `<select>` populated from live inventory, filtered to positive on-hand lots for the selected product, with an explicit incompleteness warning (`dashboard.js:3001-3029`) ✓
- **Requested by** — `<select>` with an "Other" branch (`index.html:375-386`) ✓

---

### INPUT-017 — Never ask for data the system already knows or can derive — **CRITICAL**

**S-53 (`#supply-request-requested-by`) — FAIL.** *Hard rule.* `dashboard/index.html:374-386`; `dashboard.js:4024-4035`. Every supply request asks who is making it, from a four-option list plus a free-text "Other". The identity genuinely is not known to the app — there is no authentication, only a single shared `SALES_API_KEY` (`dashboard.js:1882`) — so this is a symptom, not the disease. Recorded as FAIL because the field exists and is required; the fix is authentication, not a form change.

**S-25, S-26, S-28 (`ready_by`) — FAIL.** `dashboard/dashboard.js:2414` — `postOrderReady` sends `by: 'floor'` as a hard-coded literal on every call. `dashboard.js:2232` then renders `order.ready_by` in the READY pill. The pill therefore always reads "floor" regardless of who actually clicked, and `updateCachedOrderReady` (`2423`) defaults the same way. The system is asserting an identity it does not have, and displaying it as if it were audit data.

**Also noted:** `order.ready_at` is stamped from the **client clock** optimistically (`dashboard.js:2469`) before the server responds. It is overwritten by the server value on success (`2422`) and rolled back on failure (`2482`) ✓, but the audit timestamp shown in the interim is the browser's.

**Correctly derived — cite these:** the allocation quantity prefilled from `unallocated_need_lb` (`dashboard.js:3038-3039`) ✓; the supply-request unit derived from the selected product (`3983-3984`) ✓; the ER supplier and product preserved and locked on edit (`3551-3553`) ✓; the Daily Entries date defaulted to today in plant time (`1279-1286`) ✓; the ingredients unit derived per row (`1109`) ✓.

---

### INPUT-019 — Required data is unmistakable and the commit action is gated — **CRITICAL**

**FAIL on all five forms.** Not one required field in the product is visually marked, and not one commit button is gated.

**Evidence:** `dashboard/index.html` contains exactly **two** `required` attributes in the entire file — `#supply-request-product` (`346`) and `#supply-request-requested-by` (`375`) — and both live inside a form declared `novalidate` (`343`), so the browser's own enforcement is switched off. There is no `.required` class, no asterisk convention, and no `aria-required` anywhere.

Meanwhile the *optional* fields are explicitly marked: `<span class="label-hint">(optional)</span>` appears on Quantity (`362`), Note (`369`), Expected date (`424`), Reference # (`428`), Notes (`433`), Details (`466`), Due Date (`479`), and Pin to (`485`). Requiredness is therefore communicated only by **absence of an optional marker** — which the user must notice, and which fails silently for anyone who does not.

| Screen | Required fields | Marked? | Commit gated? | Failure surfaces as | Evidence |
|---|---|---|---|---|---|
| S-22 Note modal | Title | No | No | `alert('Title is required')` | `dashboard.js:1812-1814`; `index.html:462-463` |
| S-45 ER modal | Product, Supplier, Expected qty | No | No | inline `#er-modal-error` ✓ | `dashboard.js:3581`, `3597-3598` |
| S-53 Supply Request | Product, Requested by (+ Item, Unit when unlisted) | `required` attr but `novalidate` | No | inline, field-specific ✓ | `index.html:343`, `346`, `375`; `dashboard.js:4014-4051` |
| S-36 Allocation | Quantity (mode-dependent), Lot (manual_lot only) | `required` attr, `novalidate` | No | inline `.allocation-feedback` ✓ | `dashboard.js:2917`, `2923`, `3040`, `3052-3059` |
| S-88 Scheduler add order | SKU, Qty, Due | No | No | `alert("SKU, qty, and due date are required.")` | `scheduler:1565` |

**S-33 — PARTIAL.** `dashboard/dashboard.js:2772-2775` enforces *"Price cannot be blank for a line edit."* — a required rule that exists nowhere in the UI until the user has typed and pressed Save.

**Suggested fix:** Add a `*` marker and `aria-required` to the eight required fields, and bind each form's commit button to a `checkValidity()`-driven disabled state. The Supply Request form is already 80% of the way there — it has the attributes and the field-specific messages; it just needs the markers and the gate.

---

### INPUT-022 — Hint text, entered values, and unavailable controls are visibly distinct — **CRITICAL**

**S-22, S-45, S-53 (modal fields) — PARTIAL.** *Hard rule.* `dashboard/dashboard.css:1161-1175` — the `.form-group input / textarea / select` block sets `color: var(--text)` and `background: var(--search-bg)` but defines **no `::placeholder` rule**. Every modal field therefore inherits the user agent's default placeholder rendering (typically the text colour at ~54% opacity), which against `--text: #f1f5f9` on `--search-bg: #283548` lands close to `--text-secondary: #cbd5e1` — the same treatment used for real secondary values elsewhere. The three toolbar search fields *do* set it explicitly (`css:166`, `1269`, `2335`), so the convention exists and was not carried into the forms.

**The rule's specific data-integrity example passes.** No numeric field uses a bare number as a placeholder: `#er-qty` is `"e.g. 2000"` (`index.html:419`), `#supply-request-item` is `"e.g. Nitrile gloves"` (`354`), `#supply-request-unit` is `"e.g. box, roll, each"` (`358`), `#note-entity-id` is `"e.g. Granola Classic 25 LB"` (`496`). The "e.g." prefix is the correct convention and is applied consistently ✓.

**Unavailable controls — PARTIAL.** `.btn-refresh:disabled, .btn-sm:disabled { opacity: 0.55; cursor: not-allowed; pointer-events: none }` (`css:2423-2424`) reads clearly as disabled ✓. But `pointer-events: none` means the `title` attribute explaining *why* can never be shown — see INPUT-021. Nine other button classes have no disabled treatment at all.

**S-08 — PARTIAL.** `dashboard/dashboard.css:418-428` vs `464-467` — an available day card (`<button>`, `background: var(--surface)`) and an unavailable one (`<div>`, `background: var(--bg)`) differ by one surface step plus a 3px left border. In the rolling 5-day view the `.empty` class is not even applied (`dashboard.js:761` applies it only in month mode), so the two are nearly identical.

**S-55 — FAIL.** `dashboard/dashboard.js:1377-1380` — the disambiguation buttons' text is illegible against the broken `#fff` fallback background, so hint, value, and control state are all indistinguishable. See `01-nav-layout.md` LAYOUT-002 item 7.

---

### INPUT-021 — Read-only values look read-only; editable values look editable

**S-45 (`#er-supplier` in edit mode) — FAIL.** *High.* `dashboard/dashboard.js:3552-3553` — `supplierSel.value = record.supplier_name; supplierSel.disabled = true;`. The supplier is not editable by design (it is half the auto-match key, per the comment at `3550`), and it is presented as a **greyed-out `<select>`** — exactly the "disabled-looking input" the rule forbids for a value that is simply fixed. It should be a plain label.

**S-28 (`.order-ready-note-input` in Dispatch Queue mode) — FAIL.** `dashboard/dashboard.js:2362-2363` — the note input and its Save button are rendered `disabled` with `title="Toggle Factory Ready from All Open Orders"`. Two problems compound: a greyed input invites a tap, and the explanation is in a `title` that (a) never appears on touch and (b) never appears on the button either, because `.btn-sm:disabled { pointer-events: none }` (`css:2424`) suppresses the tooltip. The reason is unreachable through any input method.

**S-25, S-26 — PARTIAL.** `dashboard/dashboard.js:2321` — the same disabled treatment on the Factory Ready checkbox, but here the `title` is duplicated onto the enclosing `<td>`, which is not `pointer-events: none`, so hovering the cell does surface the reason on desktop ✓.

**S-63 — PARTIAL.** `dashboard/traceability.html:283`, `125` — `#traceBtn` is disabled until a lot is selected, styled `opacity: 0.5; cursor: not-allowed` ✓, but with no explanation of what is missing. The status bar does say *"Enter a lot code above to trace its supply chain."* (`303`), which covers it ✓ — downgraded to PARTIAL only because the two are not visually connected.

**Correct — cite these:** the Order Detail read view renders quantities and status as plain text and swaps them for inputs only in edit mode (`dashboard.js:3189-3193`, `3256-3272`) ✓; the edit-locked notice states the rule and the current status in plain prose rather than showing greyed controls (`2702-2704`) ✓ — this is the right answer to "why can't I edit this".

---

### INPUT-006 — Focus/tab order follows the entry sequence and ends on the primary action

**S-33 — FAIL.** *High · Hard rule.* `dashboard/dashboard.js:3206` versus `3259-3266` and `3293`. `renderOrderEditActions` emits Save Header / Save Lines / Done inside `.order-detail-header`, which precedes the line table and the notes textarea in DOM order. Tabbing from the ship-date field therefore lands on the three commit buttons **before** the user has reached any line quantity, price, or the notes field. The commit controls sit in the middle of the entry sequence.

**No modal traps focus — FAIL (S-22, S-45, S-53, S-54, S-55).** `dashboard/dashboard.js:1785-1802`, `3542-3571`, `3993-4003`. The overlays are plain `<div>`s (`index.html:337`, `398`, `446`, `508`) and the page behind them remains in the DOM and fully tabbable. Tab from the last control in any modal moves focus behind the overlay.
- Initial focus is set in two of three: ER focuses `#er-qty` or `#er-product-search` (`3570`) ✓, Supply Request focuses the product select (`4002`) ✓, **the Note modal focuses nothing** (`1785-1802`).
- Focus is restored on close in one of three: only Supply Request records `lastFocusedElement` and restores it (`3994`, `4007`) ✓ — the reference implementation.

**Keyboard-unreachable interactions — FAIL (S-04, S-10…S-13, S-17, S-18, S-25…S-27, S-32, S-42, S-55, S-56, S-58, S-63, S-65, S-67, S-68, S-83, S-84).** These elements have click handlers and no `tabindex`, no role, and no key handler: `.lot-link` (`dashboard.js:935`, `1033`, `1120`, `1211`, `1264`), `tr.expandable` (`923`, `1007`, `1111`, `1196`, `1249`), `.order-row` (`2340`), `.er-row` action targets are real buttons ✓ but the row is not, `.search-item` (`1563`, `1571`, `1582`, `1596`), `.er-product-option` (`3528`), `.product-lot-row` (`1496`), `.disambig-btn` is a real `<button>` ✓, traceability `.search-item` (`463`) and `.lot-pill` (`411-421`) are `<span>`s, scheduler `td.cell` and `.otbtn`. Opening a lot — the dashboard's single most common drill action — cannot be done from the keyboard.

**Tab order within modals — PASS.** DOM order is fields → error region → `.form-actions` with Save before Cancel (`index.html:389-392`, `437-440`, `499-502`), so Tab from the last field reaches the primary action ✓. Cancel is last, which is acceptable.

**S-49 — PASS.** `dashboard/dashboard.js:3849`, `3863-3868` — `tabindex="0"` plus an Enter/Space handler with `preventDefault`. The only fully keyboard-operable custom control in the product.

---

### INPUT-012, INPUT-013 — search-field affordances

**INPUT-012 — FAIL (S-03, S-04, S-24, S-41, S-45, S-63).** *Medium.* No clear (×) affordance exists on any text field. Five of six search/filter fields are `type="text"` rather than `type="search"`, so they do not get the user-agent clear button either: `#global-search` (`index.html:33`), `#orders-customer-search` (`223`), `#er-text-filter` (`267`), `#er-product-search` (`407`), `#lotSearch` (`traceability.html:276`). Only `#supplies-search` uses `type="search"` (`index.html:313`) ✓ — a one-character fix, already precedented in the same file.

*Related inconsistency:* the global search box is cleared programmatically after selecting a product, order, or customer (`dashboard.js:1576`, `1586`, `1600`) but **not** after selecting a lot (`1563-1568`), so opening a lot leaves the stale query in the box with the dropdown hidden.

**INPUT-013 — FAIL (S-03, S-24, S-41, S-48).** *Medium.* The leading-purpose / trailing-function convention exists in exactly one place: `dashboard/traceability.html:275`, `87-90` — a leading 🔎 glyph at `left: 11px` inside a field padded `10px 14px 10px 36px` ✓. The dashboard's four search and filter fields have no leading icon and no trailing control at all. No field anywhere has a trailing scan, lookup, or clear control.

*Minor inversion:* `#supply-request-qty` places its unit span in the **trailing** (functions) position (`index.html:363-366`) where the convention reserves that edge for actions; a leading unit or an in-label unit would be more consistent.

---

### INPUT-001, INPUT-004, INPUT-005, INPUT-009, INPUT-018, INPUT-020 — remaining findings

**INPUT-001 (S-28) — FAIL.** *Medium.* `dashboard/dashboard.js:2362` — the Factory Ready note is an `<input type="text">` labelled *"Factory Ready note"* with placeholder *"Optional note for the floor"*. It is free-form prose in a single-line field. The display renderer confirms multi-line content was expected: `.order-ready-note-text { white-space: pre-wrap; word-break: break-word }` (`css:1502-1509`).
**(S-36) — FAIL.** `dashboard/dashboard.js:2925` — `<input class="allocation-note-input" type="text" maxlength="500" placeholder="Why this stock is reserved">`. Five hundred characters of explanation in a single-line field.
**Correct:** `#note-body` rows=3 (`index.html:467`), `#er-notes` rows=2 (`434`), `#supply-request-note` rows=3 (`370`), `.order-edit-notes` rows=4 (`dashboard.js:3293`) ✓.

**INPUT-004 (S-22, S-45, S-53) — PARTIAL.** *Medium.* `dashboard/dashboard.css:1161-1165` — `.form-group input { width: 100% }` applies to every field in all three modals, so a 4-digit pound quantity and a free-text note get the same 520px box. `.form-row { grid-template-columns: 1fr 1fr }` (`1195-1199`) then forces every paired field to exactly half, making "Expected qty (lb)" the same width as "Supplier" (`index.html:412-421`) and "Priority" the same width as "Due Date" (`469-482`).
**Correct:** `.allocation-form` sizes its four columns by role — `minmax(220px, 1.5fr) minmax(180px, 1fr) minmax(130px, 0.6fr) minmax(220px, 1fr)` (`css:1941`) ✓; `.order-edit-num-cell .order-edit-input { width: 110px }` (`1864`) ✓; `#orders-customer-search { width: 200px }` (`1266`) ✓.

**INPUT-005 (S-33) — PARTIAL.** *Medium.* `dashboard/dashboard.js:3256-3272` — the two edit inputs live in table cells with the column header as their only label. Pairing is positional, and on a table with `min-width: 1040px` inside a scroll wrapper (`css:1812-1817`) the header can scroll out of view while the input stays.
**Passing elsewhere:** `.form-group { margin-bottom: 12px }` with `label { display: block }` above the field (`css:1147-1157`) gives unambiguous pairing and even spacing ✓; `.form-row` collapses to one column at ≤768px (`2191`) satisfying the mobile single-column clause ✓; `.allocation-form` uses `label` elements wrapping their controls (`1948-1955`) ✓.

**INPUT-009 (S-63) — FAIL.** *Medium.* `dashboard/traceability.html:108` — `.search-item .product-name { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }`, and the span at `465` carries **no `title` attribute**. The product name that disambiguates two similar lot codes is clipped with no one-step way to see it. This is the exact failure the rule's Factory Ledger example describes.
**(S-68) — PARTIAL.** Graph node labels are truncated at 22 and 26 characters with an ellipsis (`traceability.html:1115`, `1124`, `1145-1148`) ✓ and the full value is in the hover tooltip (`1157-1164`) ✓ — but hover only, so on touch the full value is unreachable.
**(S-83, S-87) — PARTIAL.** `scheduler:153` — `.ord .o-product { overflow: hidden; text-overflow: ellipsis; white-space: nowrap }` with no title.
**Correct:** `.er-notes` truncates with an ellipsis **and** carries `title="${notes}"` (`css:2217`; `dashboard.js:3429`) ✓ — the pattern the others should copy.

**INPUT-018 (S-88) — PARTIAL.** *High.* `scheduler:1559` — `#ao-due` defaults to start-date + 7 days. That is convenient, not probable; the rule is explicit that *"a wrong default that gets accepted blindly is worse than no default."*
**Passing and well judged:** `#er-date` is deliberately left blank (`dashboard.js:3565`) because an expected delivery date is genuinely unknown ✓; `#note-due` blank ✓; `#supply-request-qty` blank (it is optional) ✓; `#orders-status-filter` defaults to "All Open Orders" (`index.html:207`) ✓; `#er-status-filter` to "Open" ✓; `.allocation-mode-select` to the safest of three modes (`dashboard.js:2922`) ✓; `#daily-entries-date` to today in plant time (`1279-1286`) ✓; `.allocation-quantity-input` prefilled to the exact unallocated need (`3038-3039`) ✓ — the most probable correct value, though it should be paired with the inline coverage check from INPUT-007.

**INPUT-020 (S-45) — FAIL.** *Medium.* No paste handler exists anywhere in the product (verified by grep for `onpaste` / `'paste'`). Because `#er-qty` is `type="text"` (`index.html:419`), pasting `"2,000 lb"` from a supplier email leaves that string in the field and `parseFloat` (`dashboard.js:3580`) extracts **2**. The rule asks for extraction; this is silent mis-extraction.
**(S-33, S-36, S-53) — PARTIAL.** Their `type="number"` fields reject a messy paste outright — a rejection rather than an extraction, which the rule discourages but which at least fails loudly.

---

### INPUT-003 — Mask secrets only; never prefill a credential

**S-03 and every API-calling screen — FAIL.** *Low/Contextual, but the rule states "Critical where applicable" — and it applies.*

**Evidence:** `dashboard/dashboard.js:1882` — `const SALES_API_KEY = 'dashboard-key-2026';`. The same literal is repeated in `dashboard/mini-calendar.js:8`, `dashboard/traceability.html:334`, `dashboard/sankey.html`, and `dashboard/process-flow.html`. It is sent as an `X-API-Key` header on every write path — notes create/update/delete (`dashboard.js:1747`, `1775`, `1830`, `1838`), order Factory Ready (`2411`), order header/line/status PATCH (`2734`, `2795`, `2828`), allocations create/release (`3073`, `3100`), expected receipts (`3486`, `3589`, `3599`), and supply requests (`3960`, `4062`).

The audit test is *"Is any credential shown in clear or prefilled?"* — a write-capable API key is shipped in clear text to every browser that loads the page, and is readable by anyone with view-source. The remedy is architectural (a session-based auth layer, which would also fix INPUT-017), not a UI change, so it is recorded here as a finding rather than a design fix.

**The operational half passes.** No quantity, lot code, weight, or price is masked anywhere ✓ — every value stays visible and checkable before commit, as the rule requires.

---

### INPUT-010, INPUT-014, INPUT-015, INPUT-016 — remaining findings

**INPUT-010 — PARTIAL (S-03, S-04, S-24, S-41, S-63).** *High, Mobile.* `type="text"` on the five search and filter fields (see INPUT-012) means a full QWERTY keyboard with no search-key affordance. Every numeric field correctly opens a numeric pad: `#er-qty` via `inputmode="decimal"` (`index.html:419`), `#supply-request-qty` via `type="number" inputmode="decimal"` (`364`), `.allocation-quantity-input` via both (`dashboard.js:2923`) ✓. `.order-line-qty-input` and `.order-line-price-input` are `type="number"` without `inputmode` (`3260`, `3264`), which iOS honours anyway — PASS. Weighted at full strength only on Luz/Arturo screens with Mobile Yes/Partial.

**INPUT-014 — FAIL (S-22).** *High.* `#note-entity-id` is the clearest case in the product of a field that should be a type-ahead select and is a plain text input — see INPUT-011.
**— PARTIAL (S-04, S-45, S-63).** The three type-aheads that do exist all work and none creates a master record ✓, but none supports keyboard navigation, `role="combobox"`, or Enter-to-select-first (`dashboard.js:1522-1613`, `3516-3540`; `traceability.html:448-491`).
**— PARTIAL (S-53).** `#supply-request-product` is a plain (non-searchable) `<select>` over the whole supply catalogue (`dashboard.js:3681-3691`). The rule permits a plain dropdown for shorter lists; at 50+ items a type-to-filter select would be better.
**— PASS on the "never silently creates a master record" clause, everywhere.** The Supply Request "Product not listed" branch (`dashboard.js:4042-4056`) is the explicit, correctly-handled exception. The ER modal surfaces the API's supplier candidates on a mismatch — *"Candidates: Franklin Baker, Franklin Baker Inc"* (`3619`) ✓.

**INPUT-015 — N/A everywhere.** *High.* There are no custom +/− stepper controls in the product. The `type="number"` fields render native spinners, which by construction sit adjacent to a labelled value and are always paired with a typed field, satisfying both of the rule's clauses. Recorded as N/A rather than PASS because the rule has nothing to test. Worth noting as a future gap: on mobile, where native spinners do not render, there is no "+1 case" affordance anywhere.

**INPUT-016 — FAIL (S-63).** *High, Mobile.* `dashboard/traceability.html:276` — lot codes must be typed. No `capture` attribute, no `BarcodeDetector`, no `getUserMedia`, and no scanner-input handling exists anywhere under `dashboard/` (verified by grep). Arturo's primary task on this surface is "find this lot", and the lot code is printed on a label he is holding. Weighted at full strength here because Traceability's primary user is Arturo with Mobile = Partial.
**— N/A elsewhere:** the other Arturo/Luz mobile screens either have no scannable field (the Supply Request modal uses a product select ✓) or are desktop-only surfaces.
**— Related:** the audit test also asks *"could a lookup have supplied it?"* — see INPUT-017 for `requested_by` and `ready_by`.

---

### TOUCH-001 — Primary and frequent controls live in the thumb zone — **Mobile**

**FAIL on every Luz/Arturo mobile screen.** *High.*

Every commit and every frequent control in the product is at the top of the screen:

| Control | Position | Evidence |
|---|---|---|
| Section switch (7 tabs) | Top, below the header | `index.html:47`; `dashboard.css:259-260` |
| Global search | Top header | `index.html:32` |
| Refresh (×5) | Top of each section | `index.html:41`, `118`, `235`, `275`, `327` |
| + New / + New Expected Receipt | Top toolbar | `index.html:189`, `274` |
| **Request Supply** (Arturo's one write path) | Page header | `index.html:295` |
| **Edit Order / Save Header / Save Lines** | Record header | `dashboard.js:3206` |

The rule's explicit prohibition is *"the primary commit action is absent from the top bar"* and its Factory Ledger example is *"'Post Pack Run' is never top-right."* Save Header and Save Lines are exactly that. There is no bottom tab bar and no bottom action bar anywhere in the product.

**S-22, S-45, S-53 — PARTIAL.** At ≤768px `.note-modal { width: 100%; max-width: 100vw; border-radius: 0 }` (`css:2192`) makes modals full-bleed, so Save lands at the bottom of the scrolling body — closer to the thumb by accident, but not anchored (`.note-modal-body { overflow-y: auto }`, `1142-1145`), so on a long form it scrolls off.

---

### TOUCH-004 — Phone primary action is full-width and bottom-anchored; row limits

**S-22, S-45, S-53 — FAIL.** *High.* `dashboard/dashboard.css:1201-1206` — `.form-actions { display: flex; gap: 8px; justify-content: flex-end }`, with no override in the ≤768px block (`2181-2209`). Even when the modal goes full-bleed, Save and Cancel stay right-aligned at their intrinsic widths (≈30px and ≈24px tall — see TOUCH-003) rather than becoming a full-width bottom-anchored primary.

**S-43 — FAIL.** `dashboard/dashboard.js:3438-3442` — **three text buttons in one row** (Edit, Close, Cancel), against the rule's limit of two. On a phone this is three ~24px targets 4px apart at the trailing edge of a table row.

**S-24, S-41 — PARTIAL.** `dashboard/dashboard.css:2194` sets `.orders-toolbar { flex-direction: column }` at ≤768px, but `.orders-toolbar-actions` keeps `display: flex; flex-shrink: 0` (`1237-1242`) with **no** `flex-direction: column` in the media block — so Export CSV, Export Matrix (xlsx), and Refresh remain three text buttons in one row on a phone.

**S-16…S-18, S-20, S-25…S-28, S-30…S-33, S-36, S-37, S-39, S-42, S-51, S-63, S-64, S-70 — FAIL** on the primary clause (no bottom-anchored primary anywhere).

**Correct:** `.order-detail-actions { flex-wrap: wrap }` (`css:1821-1826`) ✓ and `#topbar { flex-wrap: wrap }` (`scheduler:28`) ✓ both wrap rather than crowd.

---

### TOUCH-002, TOUCH-005 — passing rules

**TOUCH-002 — PASS.** *Medium, Mobile.* No custom gestures exist anywhere (verified by grep for `touchstart`, `touchmove`, `swipe`). The only gesture surface is Traceability's D3 zoom (`traceability.html:1008-1011`), which supports pinch and drag **and** provides visible Fit / + / − buttons (`309-311`) — the rule's requirement that every gesture have a visible equivalent is met exactly ✓.
**S-69 — FAIL.** `dashboard/traceability.html:1153-1172` — the node tooltip is bound to `mouseenter`/`mouseleave` only. On touch there is no hover, so the full node detail (which is also where truncated labels are recovered — see INPUT-009) is unreachable. `onNodeClick` (`1177-1194`) fires instead and re-targets the search box, so a tap does something *other* than reveal the detail.

**TOUCH-005 — N/A everywhere.** *Medium, Situational.* No multi-select, reorder, or bulk-delete operation exists anywhere in the product, so the rule has nothing to test.
**S-25, S-26, S-28 — PARTIAL.** The rule's audit test — *"Can a single accidental tap in a list start a bulk or destructive operation?"* — has a near-miss: a single tap on a 16×16px checkbox in the orders list writes the Factory Ready flag with no confirmation (`dashboard.js:2455-2486`). It is neither bulk nor destructive, so this is recorded as a related observation rather than a failure; the sizing and adjacency problems are captured under TOUCH-003 and LAYOUT-012.

---

## N/A register

| Rule | Screens | Reason |
|---|---|---|
| ACTION-001…012 | Read-only tiles, states, empty states, legends, tooltips, print views | No buttons or control groups present. |
| ACTION-007 | Non-form, non-dialog screens | No form or temporary view for Enter to act on. |
| ACTION-012 | Screens with no segmented control or tab strip | Nothing to assess. |
| INPUT-001…022 | Read-only screens and states | No input fields present. |
| INPUT-003 | All screens without a credential field | No secret is collected on the screen; the shipped API key is recorded once against S-03 as the app-wide instance. |
| INPUT-010, 012 (mobile clause), 016 | Blubber-primary screens; Mobile=No screens | **Desktop-only surface.** |
| INPUT-015 | All screens | No stepper control exists in the product. |
| TOUCH-001, 002, 004, 005 | Blubber-primary screens; Mobile=No screens | **Desktop-only surface.** |
| TOUCH-005 | All screens | No multi-select, reorder, or bulk-delete operation exists. |

## Unverifiable from code

| Rule | Screens | What a browser check must confirm |
|---|---|---|
| ACTION-002 (disabled clause) | All screens using `.btn-close`, `.note-action-btn`, `.order-expand-toggle`, `.btn-back`, `.btn-secondary`, `.show-more-btn`, `.tab`, `.notes-filter-btn`, `.btn-theme` | These classes are never disabled in the current code paths, so the absence of a `:disabled` rule is latent rather than active. A rendered check should confirm no state disables them. |
| INPUT-010 | S-22, S-45, S-53, S-63 (Luz/Arturo, Mobile Yes/Partial) | Which soft keyboard each field actually raises on iOS Safari and Android Chrome. `inputmode` and `type` are declared correctly for the numeric fields; the five `type="text"` search fields need confirming. |
| INPUT-022 | S-22, S-45, S-53 | The user agent's default placeholder contrast against `--search-bg` in both themes. No `::placeholder` rule exists for `.form-group` fields, so the rendered contrast ratio must be measured, not read. |
| TOUCH-003 | All screens | Rendered hit-box sizes in CSS pixels at the device pixel ratios in use. The declared paddings above are strong evidence but a rendered measurement is definitive. |

---

*End of 02-actions-input-touch.md*

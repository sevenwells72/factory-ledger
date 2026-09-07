# 01 — Audit: Navigation & Layout (NAV, LAYOUT)

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) §1 (NAV-001…012) and §2 (LAYOUT-001…021) — 33 rules
**Screens:** [00-screen-inventory.md](00-screen-inventory.md) — S-01…S-91
**Status:** Audit findings. **No application code was modified.**

---

## How to read this

### Statuses

| Code | Meaning |
|---|---|
| **P** | PASS — the audit test is met in code |
| **PA** | PARTIAL — met for some cases, or met with a material gap |
| **F** | FAIL — the audit test is not met |
| **N** | N/A — the rule cannot apply to this screen (reason given in the findings or the N/A register) |
| **U** | UNVERIFIABLE-FROM-CODE — needs a rendered browser check |
| **·** | Out of tier scope — Tier B screen, rule is Medium or Low importance |

### Tiering applied

- **Tier A (every applicable rule walked):** S-01…S-15, S-24…S-56, S-63…S-70.
- **Tier B (Critical and High rules only):** S-16…S-18 (Activity), S-19…S-23 (Notes), S-57…S-59 (Sankey), S-60…S-62 (Process Flow), S-71 (print), S-72…S-91 (scheduler).

**Note on Notes (S-19…S-23).** The brief places screens whose primary action mutates data in Tier A *and* names Notes in Tier B. The explicit naming is the more specific instruction, so Notes is audited at Tier B. Every Critical and High rule is still walked for it; only Medium/Low rules are marked `·`.

### Rule importance (drives Tier B scope)

- **Critical:** LAYOUT-003, LAYOUT-012, LAYOUT-020
- **High:** NAV-001, 002, 003, 004, 005, 007, 012; LAYOUT-001, 002, 004, 005, 006, 007, 010, 011, 015, 018, 021
- **Medium:** NAV-006, 008, 009, 010, 011; LAYOUT-008, 009, 013, 014, 017
- **Low/Contextual:** LAYOUT-016, LAYOUT-019

### Mobile-platform rules

Per the audit brief, Mobile-platform rules (here: **LAYOUT-014**, **LAYOUT-015**) are applied at full weight only where the primary user is **Arturo or Luz** *and* the inventory Mobile column is Yes or Partial. Everywhere else they are **N** with the reason *"desktop-only surface."* Blubber-primary screens are treated as desktop-only throughout.

---

## Matrix A — Navigation (NAV-001…012)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | P | N | N | P | N | P | P | P | N | N | N | P |
| S-02 | Mini-calendar strip | P | N | P | N | N | N | N | N | N | N | N | PA |
| S-03 | App header | PA | N | P | N | N | N | N | N | N | N | N | N |
| S-04 | Global search + results | P | N | P | N | PA | N | N | N | N | P | N | PA |
| S-05 | Tab bar (7 tabs) | P | P | N | **F** | PA | P | P | **F** | N | N | N | P |
| S-06 | Today So Far tile | P | N | P | N | P | N | N | N | N | N | N | PA |
| S-07 | Today So Far — error/retry | P | N | P | N | P | N | N | N | N | N | N | N |
| S-08 | Production Calendar | P | N | P | P | P | P | **F** | P | P | PA | N | P |
| S-09 | Calendar day detail panel | P | N | P | N | P | N | N | N | P | PA | **F** | P |
| S-10 | Finished Goods panels | P | N | P | N | P | P | N | P | P | **F** | N | P |
| S-11 | FG per-product lot rows | P | N | PA | N | P | N | N | N | PA | **F** | N | P |
| S-12 | Batch Inventory | P | N | P | N | P | P | N | P | P | **F** | N | P |
| S-13 | On-Hand Ingredients | P | N | P | N | P | P | N | P | P | **F** | N | PA |
| S-14 | Recent Entries feed | P | N | P | N | P | P | N | N | N | N | N | **F** |
| S-15 | Recent Entries states | P | N | P | N | P | N | N | N | N | N | N | N |
| S-16 | Daily Entries + toolbar | P | P | P | N | P | · | **F** | · | · | · | · | P |
| S-17 | Shipping log | P | P | P | N | P | · | N | · | · | · | · | P |
| S-18 | Receiving log | P | P | P | N | P | · | N | · | · | · | · | P |
| S-19 | Notes toolbar | P | P | P | P | P | · | P | · | · | · | · | P |
| S-20 | Notes list (cards) | P | P | P | N | P | · | N | · | · | · | · | P |
| S-21 | Notes empty state | P | P | P | N | P | · | N | · | · | · | · | N |
| S-22 | Note create/edit modal | P | P | **PA** | N | P | · | P | · | · | · | · | P |
| S-23 | Delete-note confirm | P | P | PA | N | P | · | N | · | · | · | · | N |
| S-24 | Orders toolbar | P | N | P | PA | P | P | **F** | P | N | N | N | P |
| S-25 | Orders list table | P | N | P | N | **F** | P | N | N | PA | PA | N | **F** |
| S-26 | Dispatch Queue mode | P | N | PA | PA | P | P | **F** | P | N | N | N | PA |
| S-27 | Orders expandable lines | P | N | P | N | **F** | P | N | N | P | PA | N | P |
| S-28 | Factory Ready toggle+note | P | N | P | N | **F** | P | N | N | N | N | N | P |
| S-29 | Orders empty state | P | N | P | N | P | P | N | N | N | N | N | P |
| S-30 | Order Detail header/KPI | P | N | P | N | P | P | N | P | PA | N | **F** | P |
| S-31 | Order Detail line table | P | N | P | N | P | P | N | P | P | P | N | P |
| S-32 | Per-line inventory expander | P | N | PA | N | P | P | N | N | P | **F** | N | P |
| S-33 | Order Detail edit mode | P | PA | P | N | PA | P | N | P | N | N | N | P |
| S-34 | Edit-locked notice | P | P | P | N | P | P | N | N | N | N | N | P |
| S-35 | SO status change confirm | P | P | P | N | P | P | N | N | N | N | N | N |
| S-36 | Reservations / allocation | P | PA | P | N | P | P | P | P | N | N | N | P |
| S-37 | Allocation history table | P | N | P | N | P | P | N | P | P | P | N | **P** |
| S-38 | Release reservation confirm | P | P | P | N | P | P | N | N | N | N | N | N |
| S-39 | Shipping capacity preview | P | N | P | N | P | P | N | N | N | N | N | PA |
| S-40 | Order Detail notes card | P | N | P | N | P | P | N | N | N | N | N | P |
| S-41 | Expected Receipts toolbar | P | N | P | P | P | P | **F** | P | N | N | N | P |
| S-42 | Expected Receipts table | P | N | P | N | P | P | N | N | PA | **F** | N | **F** |
| S-43 | ER row actions | P | P | P | N | P | P | N | N | N | N | N | P |
| S-44 | ER empty state | P | N | P | N | P | P | N | N | N | N | N | P |
| S-45 | ER create/edit modal | P | P | P | N | P | P | N | P | N | N | N | PA |
| S-46 | Supplies header + Request | P | N | P | N | P | P | N | N | N | N | N | P |
| S-47 | Supplies sub-tabs | P | N | P | P | P | P | P | P | N | N | N | P |
| S-48 | Supplies search field | P | N | P | N | PA | P | N | N | N | N | N | P |
| S-49 | Supplies inventory table | P | N | P | N | P | P | N | N | P | P | N | P |
| S-50 | Supply lot / incoming detail | P | N | P | N | P | P | N | N | P | P | N | P |
| S-51 | Supply Requests list | P | N | P | N | P | P | N | N | N | N | N | **F** |
| S-52 | Supply Requests feedback | P | N | P | N | P | P | N | N | N | N | N | N |
| S-53 | Request Supply modal | P | P | P | N | P | P | P | P | N | N | N | P |
| S-54 | Lot Detail side panel | P | N | P | N | P | P | N | N | P | P | **F** | P |
| S-55 | Lot disambiguation | P | P | P | N | P | P | P | P | N | N | N | P |
| S-56 | Product Detail panel | P | N | P | N | P | P | N | N | P | P | **F** | **PA** |
| S-57 | Sankey controls bar | P | N | P | P | P | · | PA | · | · | · | · | P |
| S-58 | Sankey chart + legend | P | N | P | N | P | · | N | · | · | · | · | P |
| S-59 | Sankey banner / loading | P | N | P | N | P | · | N | · | · | · | · | N |
| S-60 | Summary strip | P | N | P | N | P | · | N | · | · | · | · | P |
| S-61 | Production lines grid | P | N | P | N | P | · | N | · | · | · | · | **F** |
| S-62 | Error / stale banners | P | N | P | N | P | · | N | · | · | · | · | N |
| S-63 | Lot search + type-ahead | P | N | P | N | P | P | N | N | N | P | N | **F** |
| S-64 | Trace direction + Trace | P | P | P | P | P | P | P | P | N | N | N | P |
| S-65 | Recent lots strip | P | N | P | N | P | P | N | N | N | N | N | **F** |
| S-66 | Trace legend | P | N | P | N | P | P | N | N | N | N | N | P |
| S-67 | Status bar | P | N | P | N | P | P | N | N | N | N | N | P |
| S-68 | Trace graph + zoom | P | PA | P | N | P | P | P | P | PA | PA | N | P |
| S-69 | Node tooltip | P | N | P | N | P | P | N | N | N | N | N | P |
| S-70 | Trace detail + exports | P | N | P | N | P | P | N | N | N | N | N | P |
| S-71 | Print audit report | · | · | P | N | P | · | · | · | · | · | · | P |
| S-72 | Topbar KPI strip | P | N | PA | N | P | · | N | · | · | · | · | P |
| S-73 | Topbar actions | P | N | P | N | P | · | N | · | · | · | · | N |
| S-74 | Legend bar | P | N | P | N | P | · | N | · | · | · | · | P |
| S-75 | Delta panel | P | N | P | N | P | · | N | · | · | · | · | P |
| S-76 | Settings panel (form) | P | P | P | N | P | · | PA | · | · | · | · | P |
| S-77 | FG on hand (disclosure) | P | N | P | N | P | · | N | · | · | · | · | P |
| S-78 | Bulk-bin WIP (disclosure) | P | N | P | N | P | · | N | · | · | · | · | P |
| S-79 | Product catalog (form) | P | P | P | N | P | · | N | · | · | · | · | P |
| S-80 | Orders in / plans out | P | P | P | N | P | · | N | · | · | · | · | P |
| S-81 | CSV import result | P | N | P | N | P | · | N | · | · | · | · | P |
| S-82 | "How this works" | P | N | P | N | P | · | N | · | · | · | · | P |
| S-83 | Production board table | P | N | P | N | P | · | N | · | · | · | · | P |
| S-84 | Board cell copy / more | P | N | P | N | P | · | N | · | · | · | · | P |
| S-85 | Schedule panel | P | P | P | N | P | · | N | · | · | · | · | P |
| S-86 | Pin modal | P | P | P | N | P | · | P | · | · | · | · | P |
| S-87 | Order book | P | N | P | N | P | · | N | · | · | · | · | P |
| S-88 | Add order line form | P | P | PA | N | P | · | N | · | · | · | · | P |
| S-89 | Add-order validation alert | P | P | PA | N | P | · | N | · | · | · | · | N |
| S-90 | Order book empty state | P | N | P | N | P | · | N | · | · | · | · | N |
| S-91 | Scheduler print view | · | · | P | N | P | · | · | · | · | · | · | P |

## Matrix B — Layout & Visual Hierarchy (LAYOUT-001…021)

`003`, `012`, `020` are Critical. Columns are LAYOUT rule numbers.

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 | 016 | 017 | 018 | 019 | 020 | 021 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | P | PA | PA | P | P | P | P | P | P | PA | PA | P | PA | U | **F** | N | P | P | N | P | P |
| S-02 | Mini-calendar strip | P | P | PA | PA | **F** | P | P | P | P | PA | P | P | P | U | **F** | N | P | P | N | P | P |
| S-03 | App header | P | PA | **F** | P | P | P | P | PA | PA | PA | **F** | P | PA | U | **F** | N | P | PA | N | P | P |
| S-04 | Global search + results | P | P | P | P | P | P | P | P | PA | PA | P | P | P | U | PA | N | P | P | N | P | P |
| S-05 | Tab bar (7 tabs) | P | P | **F** | P | P | P | P | PA | P | PA | **F** | P | PA | U | **F** | N | P | P | N | P | P |
| S-06 | Today So Far tile | P | P | P | PA | PA | P | P | N | P | PA | P | P | P | N | N | N | P | P | N | P | P |
| S-07 | Today So Far — error/retry | P | PA | P | P | P | P | P | N | P | PA | P | P | P | N | N | N | P | P | N | P | P |
| S-08 | Production Calendar | P | P | PA | P | PA | P | PA | N | P | PA | P | P | PA | N | N | N | P | P | N | P | P |
| S-09 | Calendar day detail panel | P | P | P | P | PA | P | P | N | P | PA | P | P | P | N | N | N | PA | P | N | P | P |
| S-10 | Finished Goods panels | P | PA | **F** | P | PA | P | P | N | P | PA | **F** | P | **F** | U | **F** | N | PA | P | PA | P | P |
| S-11 | FG per-product lot rows | P | PA | **F** | P | PA | P | PA | N | P | PA | **F** | P | **F** | U | **F** | N | PA | P | PA | P | P |
| S-12 | Batch Inventory | P | PA | **F** | P | PA | P | P | N | P | PA | **F** | P | **F** | N | N | N | PA | P | PA | P | P |
| S-13 | On-Hand Ingredients | P | PA | **F** | P | PA | P | P | N | P | PA | **F** | P | **F** | U | **F** | N | PA | P | PA | P | P |
| S-14 | Recent Entries feed | P | P | P | P | P | P | P | N | P | PA | P | P | P | U | PA | N | P | P | P | **F** | P |
| S-15 | Recent Entries states | P | PA | P | P | P | P | P | N | P | PA | P | P | P | N | N | N | P | P | P | P | P |
| S-16 | Daily Entries + toolbar | P | PA | **F** | P | PA | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-17 | Shipping log | P | PA | **F** | P | PA | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-18 | Receiving log | P | PA | **F** | P | PA | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-19 | Notes toolbar | P | PA | P | P | P | P | PA | · | · | PA | P | P | · | · | PA | · | · | PA | · | P | P |
| S-20 | Notes list (cards) | P | PA | P | P | PA | P | PA | · | · | PA | P | **F** | · | · | PA | · | · | P | · | **F** | P |
| S-21 | Notes empty state | P | P | P | P | P | P | P | · | · | PA | P | P | · | · | P | · | · | P | · | P | P |
| S-22 | Note create/edit modal | P | PA | P | P | P | P | P | · | · | PA | P | PA | · | · | PA | · | · | PA | · | P | P |
| S-23 | Delete-note confirm | P | **F** | P | P | P | P | P | · | · | P | P | P | · | · | P | · | · | P | · | P | P |
| S-24 | Orders toolbar | PA | PA | P | P | P | P | P | P | PA | PA | P | P | PA | U | PA | N | P | **F** | N | P | PA |
| S-25 | Orders list table | PA | PA | **F** | PA | **F** | PA | PA | N | P | PA | **F** | **F** | **F** | U | **F** | N | PA | PA | **F** | **F** | P |
| S-26 | Dispatch Queue mode | PA | PA | **F** | PA | PA | PA | PA | N | P | PA | **F** | **F** | **F** | U | **F** | N | PA | PA | **F** | **F** | P |
| S-27 | Orders expandable lines | PA | PA | **F** | P | PA | P | PA | N | P | PA | **F** | P | **F** | U | **F** | N | PA | P | **F** | **F** | PA |
| S-28 | Factory Ready toggle+note | P | PA | **F** | P | PA | P | PA | N | PA | PA | **F** | **F** | **F** | U | **F** | N | PA | P | N | **F** | PA |
| S-29 | Orders empty state | P | P | P | P | P | P | P | N | P | PA | P | P | P | U | P | N | P | P | N | P | P |
| S-30 | Order Detail header/KPI | PA | PA | P | PA | P | **F** | P | N | P | PA | P | P | P | U | PA | P | P | P | N | P | PA |
| S-31 | Order Detail line table | PA | P | P | P | **F** | P | P | N | P | PA | P | P | P | U | PA | P | P | P | P | P | P |
| S-32 | Per-line inventory expander | P | PA | P | P | PA | P | PA | N | P | PA | P | P | P | U | PA | P | P | P | P | P | P |
| S-33 | Order Detail edit mode | PA | PA | P | P | PA | P | PA | N | PA | PA | P | PA | P | U | PA | P | P | **F** | P | P | PA |
| S-34 | Edit-locked notice | P | P | P | P | P | P | P | N | P | PA | P | P | P | U | P | P | P | P | P | P | P |
| S-35 | SO status change confirm | P | **F** | P | P | P | P | P | N | P | P | P | P | P | U | P | P | P | P | P | P | P |
| S-36 | Reservations / allocation | PA | PA | P | P | PA | P | P | N | P | PA | P | P | P | U | PA | P | P | **F** | P | P | P |
| S-37 | Allocation history table | P | PA | P | P | **F** | P | PA | N | P | PA | P | PA | P | U | PA | P | P | P | P | P | P |
| S-38 | Release reservation confirm | P | **F** | P | P | P | P | P | N | P | P | P | P | P | U | P | P | P | P | P | P | P |
| S-39 | Shipping capacity preview | P | PA | P | P | PA | P | P | N | P | PA | P | P | P | U | PA | P | P | P | P | P | P |
| S-40 | Order Detail notes card | P | P | P | P | P | P | P | N | P | PA | P | P | P | U | P | P | P | P | P | P | P |
| S-41 | Expected Receipts toolbar | P | PA | P | P | P | P | P | P | PA | PA | P | P | PA | U | PA | N | P | **F** | N | P | P |
| S-42 | Expected Receipts table | P | PA | **F** | P | PA | P | PA | N | P | PA | **F** | PA | **F** | U | **F** | N | PA | PA | **F** | P | P |
| S-43 | ER row actions | P | PA | **F** | P | P | P | PA | N | P | PA | **F** | **PA** | **F** | U | **F** | N | P | P | N | P | P |
| S-44 | ER empty state | P | P | P | P | P | P | P | N | P | PA | P | P | P | U | P | N | P | P | N | P | P |
| S-45 | ER create/edit modal | P | PA | P | P | P | P | P | N | P | PA | P | P | P | U | PA | P | P | PA | P | P | P |
| S-46 | Supplies header + Request | P | PA | P | P | P | P | P | N | P | PA | P | P | P | U | PA | N | P | PA | N | P | **PA** |
| S-47 | Supplies sub-tabs | P | P | P | P | P | P | P | P | P | PA | P | P | P | U | PA | N | P | **F** | N | P | P |
| S-48 | Supplies search field | P | P | P | P | P | P | P | P | P | PA | P | P | P | U | PA | N | P | P | N | P | P |
| S-49 | Supplies inventory table | P | P | P | P | PA | P | PA | P | P | PA | P | P | P | U | PA | N | PA | PA | P | **F** | P |
| S-50 | Supply lot / incoming detail | P | PA | P | P | PA | P | P | N | P | PA | P | P | P | U | PA | N | PA | P | P | PA | P |
| S-51 | Supply Requests list | P | P | P | P | PA | P | PA | N | P | PA | P | P | P | U | PA | N | P | P | P | P | P |
| S-52 | Supply Requests feedback | P | PA | P | P | P | P | P | N | P | PA | P | P | P | U | P | N | P | P | P | P | P |
| S-53 | Request Supply modal | P | PA | P | P | P | P | P | N | P | PA | P | P | P | U | PA | P | P | PA | P | P | P |
| S-54 | Lot Detail side panel | P | **F** | P | P | PA | P | P | N | P | PA | P | P | P | U | PA | P | P | P | P | P | P |
| S-55 | Lot disambiguation | P | **F** | P | P | P | P | PA | N | P | **F** | P | P | P | U | PA | P | PA | **F** | P | P | P |
| S-56 | Product Detail panel | P | **F** | P | P | **F** | P | PA | N | P | **F** | P | P | P | U | PA | P | PA | P | P | P | P |
| S-57 | Sankey controls bar | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-58 | Sankey chart + legend | P | PA | **F** | P | PA | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | PA | P |
| S-59 | Sankey banner / loading | P | **F** | P | P | PA | P | P | · | · | PA | P | P | · | · | P | · | · | PA | · | P | P |
| S-60 | Summary strip | P | **F** | P | P | P | P | P | · | · | PA | P | P | · | · | N | · | · | P | · | P | P |
| S-61 | Production lines grid | P | **F** | P | P | PA | P | P | · | · | PA | P | P | · | · | N | · | · | P | · | **F** | P |
| S-62 | Error / stale banners | P | **F** | P | P | P | P | P | · | · | PA | P | P | · | · | N | · | · | PA | · | P | P |
| S-63 | Lot search + type-ahead | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-64 | Trace direction + Trace | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-65 | Recent lots strip | P | **F** | **F** | P | PA | P | P | · | · | PA | **F** | P | · | · | **F** | · | · | P | · | P | P |
| S-66 | Trace legend | P | **F** | PA | P | P | P | P | · | · | PA | P | P | · | · | PA | · | · | P | · | P | P |
| S-67 | Status bar | P | **F** | P | P | P | P | P | · | · | PA | P | P | · | · | PA | · | · | PA | · | P | P |
| S-68 | Trace graph + zoom | P | **F** | **F** | P | PA | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | PA | P |
| S-69 | Node tooltip | P | **F** | **F** | P | P | P | P | · | · | PA | P | P | · | · | N | · | · | P | · | P | P |
| S-70 | Trace detail + exports | P | **F** | PA | P | PA | P | P | · | · | PA | P | P | · | · | PA | · | · | P | · | P | P |
| S-71 | Print audit report | P | PA | P | P | PA | P | P | · | · | PA | P | P | · | · | N | · | · | P | · | P | P |
| S-72 | Topbar KPI strip | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-73 | Topbar actions | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | PA | · | · | N | · | · | P | · | P | PA |
| S-74 | Legend bar | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-75 | Delta panel | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-76 | Settings panel (form) | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-77 | FG on hand (disclosure) | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-78 | Bulk-bin WIP (disclosure) | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-79 | Product catalog (form) | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-80 | Orders in / plans out | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-81 | CSV import result | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-82 | "How this works" | P | **F** | **F** | P | P | P | P | · | · | PA | **F** | P | · | · | N | · | · | P | · | P | P |
| S-83 | Production board table | P | **F** | **F** | P | PA | P | P | · | · | P | **F** | P | · | · | N | · | · | P | · | P | P |
| S-84 | Board cell copy / more | P | **F** | **F** | P | P | P | P | · | · | P | **F** | P | · | · | N | · | · | P | · | P | P |
| S-85 | Schedule panel | P | **F** | **F** | P | P | P | P | · | · | P | **F** | P | · | · | N | · | · | P | · | P | P |
| S-86 | Pin modal | P | **F** | PA | P | P | P | P | · | · | P | P | P | · | · | N | · | · | P | · | P | P |
| S-87 | Order book | P | **F** | **F** | P | PA | P | P | · | · | P | **F** | **F** | · | · | N | · | · | P | · | **F** | P |
| S-88 | Add order line form | P | **F** | **F** | P | P | P | P | · | · | P | **F** | P | · | · | N | · | · | P | · | P | P |
| S-89 | Add-order validation alert | P | **F** | P | P | P | P | P | · | · | P | P | P | · | · | N | · | · | P | · | P | P |
| S-90 | Order book empty state | P | **F** | **F** | P | P | P | P | · | · | P | **F** | P | · | · | N | · | · | P | · | P | P |
| S-91 | Scheduler print view | P | PA | P | P | P | P | P | · | · | P | P | P | · | · | N | · | · | P | · | P | P |

---

## Findings

Ordered by rule ID. Every FAIL and PARTIAL carries `file:line` evidence.

---

### NAV-004 — Tabs switch between views of one subject; primary navigation moves between sections

**S-05 (Tab bar) — FAIL.** *High · Strong recommendation*

**Evidence:** `dashboard/index.html:47-55`; `dashboard/dashboard.js:494-508` (`initTabs`).

The seven tabs — Operations, Recent Entries, Activity, Notes, Sales Orders, Expected Receipts, Supplies — are not seven facets of one subject. They are seven separate areas of the system with different data, different primary users, and different primary actions. The audit test asks directly: *"Is any app section reachable only through a tab strip?"* Sales Orders, Expected Receipts, Supplies, and Notes are reachable **only** through this tab strip. The site nav (`index.html:18-23`) carries only four page-level destinations and does not list them.

**Suggested fix:** Promote the four record-owning sections (Sales Orders, Expected Receipts, Supplies, Notes) into the site nav as top-level destinations, and keep a tab strip inside the Dashboard page for what genuinely is one subject — Operations / Recent Entries / Activity, all of which are views of today's ledger. This also resolves NAV-008.

**S-24, S-26 (Orders toolbar / Dispatch Queue) — PARTIAL.**

**Evidence:** `dashboard/index.html:205-217`; `dashboard/dashboard.js:1967-1969` (`isDispatchQueueMode`), `2040-2047`.

`orders-status-filter` mixes a **mode** (`dispatch_queue`, which switches the whole list to a different data source and a different set of columns, `dashboard.js:2244-2259`) into a list of **statuses**. Selecting it also reveals a second, otherwise-hidden select (`orders-dispatch-filter`, `index.html:218`). One control is doing two different jobs at two different levels.

**Suggested fix:** Split the mode out of the status list — a two-segment "All Orders | Dispatch Queue" switcher beside the status filter.

---

### NAV-005 — Keep each tab pane self-contained

**S-25, S-27, S-28 (Orders list, expandable lines, Factory Ready) — FAIL.** *High · Hard rule*

**Evidence:** `dashboard/dashboard.js:2472` and `2479` (`bindOrderReadyToggles` calls `renderOrdersList()`), `2441` (`bindOrderReadyNoteControls` calls `renderOrdersList()`), `2300-2337` (`renderOrdersList` rewrites `container.innerHTML`).

Ticking the Factory Ready checkbox on **one** row rebuilds the entire orders table twice — once optimistically before the request, once after it returns. Every other row's inline expansion collapses, and any text typed into another row's Factory Ready note input (`.order-ready-note-input`, `dashboard.js:2362`) is destroyed without warning. The rule's test — *"does switching panes lose unsaved input?"* — fails inside a single pane: one row's action silently discards another row's unsaved input.

**Suggested fix:** Patch the single changed row in place (`tr.dataset.orderId` → update the checkbox, the `so-ready` class, and the pill) instead of re-rendering the table. If a full re-render is unavoidable, re-apply the expanded set and preserve in-flight note text.

**S-04 (Global search) — PARTIAL.**

**Evidence:** `dashboard/dashboard.js:1596-1612`; `dashboard/index.html:223`.

Clicking a customer in the global search dropdown switches to the Orders tab and then looks for `document.getElementById('orders-customer-filter')` — an element that does not exist. The real field is `orders-customer-search`. The `if (custFilter)` guard swallows the miss, so the user lands on an **unfiltered** orders list with no indication that their search was dropped. Full write-up under SEARCH-002 in `04-search-data-drag-chart.md`.

**S-48 (Supplies search) — PARTIAL.** `dashboard/dashboard.js:3831`. The supplies search box filters the inventory table only. That is correct scoping, but the Supply Requests list directly below it is unaffected and there is no visual boundary saying so; the box sits in a sticky toolbar (`dashboard.css:2271-2282`) that reads as page-level.

**S-33 (Order Detail edit mode) — PARTIAL.** `dashboard/dashboard.js:2715-2745`, `2747-2811`. "Save Header" and "Save Lines" are two commits of one record, presented as two buttons side by side (`2707-2709`). A user who edits both and presses only one gets a partial save with no warning that the other half is still dirty.

---

### NAV-007 — Visible one-tap switcher for small sets; a dropdown only when there are too many

Four small mutually-exclusive view/filter sets are hidden behind dropdowns or ambiguous toggles.

**S-08 (Production Calendar mode) — FAIL.** *High*

**Evidence:** `dashboard/index.html:79`; `dashboard/dashboard.js:669-685` (`updateCalendarLabel`), `4212-4216`.

The rolling-5-day / month choice is a single button whose label shows the mode you are **not** in (`toggleBtn.textContent = '5-Day View'` when the month view is active). The current selection is never visible; the user must read the range label to infer it. Two mutually exclusive views is the textbook segmented-control case.

**Suggested fix:** `[ 5 Days | Month ]` segmented control with the active segment filled.

**S-24 (Orders dispatch filter) — FAIL.** `dashboard/index.html:218-222`. Three options (All Dispatch States / Dispatch Ready / Blocked) in a `<select>`.

**S-41 (Expected Receipts status filter) — FAIL.** `dashboard/index.html:261-266`. Four options (Open / Closed / Cancelled / All) in a `<select>`.

**S-16 (Daily Entries date mode) — FAIL.** `dashboard/index.html:138-141`. Two options (By event date / By entry date) in a `<select>`. This one matters more than its size suggests: event-date vs entry-date is the distinction the whole Recent Entries timing model rests on, and it is currently invisible until the user opens the menu.

**Suggested fix (all three):** Replace with segmented controls styled like the existing `.notes-filter-btn` group (`dashboard.css:977-988`), which is already the correct pattern in this codebase.

**S-57 (Sankey controls) — PARTIAL.** `dashboard/sankey.html:293-330`. Four selects, each with 4–5 options, all below the NAV-007 threshold. Four segmented controls side by side would crowd the bar, so the dropdowns are defensible — but the **Period** control (the one Blubber changes most) should be segmented and the three Top-N controls collapsed behind a single "Detail" popover.

**S-76 (Scheduler settings) — PARTIAL.** `dashboard/scheduler/seven-wells-production-board.html:956-992`. Numeric dials and small enumerations are rendered as bare inputs and selects in one long column; several are 2–3-value choices.

---

### NAV-008 — Limit tabs: at most 5 on phones, about 6 on wide screens; never wrap, truncate, or scroll

**S-05 — FAIL.** *Medium · Strong recommendation*

**Evidence:** `dashboard/index.html:47-55` (7 tabs); `dashboard/dashboard.css:262` (`overflow-x: auto`), `276-277` (`white-space: nowrap; flex: 0 0 auto`).

Seven tabs exceeds the ~6 wide-screen limit and the 5 phone limit, and the CSS deliberately implements the one behaviour the rule forbids: horizontal scrolling. On a 390px phone roughly three of the seven tabs are visible; the remaining four — including Sales Orders and Supplies — are off-screen with no scroll affordance (`::-webkit-scrollbar` is 8px and the bar is 2px taller than the content, `dashboard.css:90`).

**Suggested fix:** Fixed by the NAV-004 restructure. If the tab strip stays, cap it at five and put the rest behind an overflow menu.

---

### NAV-010 — One affordance for "go deeper", a different one for "show details"

**S-10, S-11, S-12, S-13 (inventory tables) — FAIL.** *Medium*

**Evidence:** `dashboard/dashboard.js:923` (`<tr class="expandable" data-expand="...">`), `1007`, `1111`; `dashboard/dashboard.css:699-700`.

Expandable inventory rows carry **no** affordance at all — no chevron, no disclosure triangle, no caret. The only cue that a row expands is `cursor: pointer` on hover, which does not exist on touch. Meanwhile three other affordances are in play for the same "expand in place" meaning: `.chevron::after` = `▶` (`dashboard.css:650-657`), `.order-expand-caret` = `▸` (`dashboard.js:2320`), `.supply-row-caret` = `▸` (`dashboard.js:3850`).

**S-42 (Expected Receipts table) — FAIL.** `dashboard/dashboard.js:3428`. Rows carry `data-er-id` and a `.er-row` class but no expand and no navigate affordance; the row is inert while three buttons inside it are not.

**S-32 (per-line inventory expander) — FAIL.** `dashboard/dashboard.js:3257`. A fourth pattern for the same job: a text button labelled "Inventory" inside a table cell.

**S-25, S-26, S-27 — PARTIAL.** `dashboard/dashboard.js:2320` vs `2340-2347`. The row *navigates* and the caret button *expands*, which is correctly two affordances — but navigation has no glyph (only `cursor: pointer`, `dashboard.css:1318`), and the glyph that does exist (`▸`, the conventional "go deeper" chevron) is assigned to expand-in-place. The two meanings are swapped relative to the rule.

**S-55, S-56 (disambiguation, product panel) — FAIL.** `dashboard/dashboard.js:1465`, `1478`. Product-lot rows use `cursor:pointer` inline styles with a `.lot-link` span inside; the row and the link both navigate, so one target has two overlapping affordances and the row itself has none.

**S-08, S-09 — PARTIAL.** `dashboard/dashboard.js:764`, `785`. Day cards use a text hint "View details" plus `aria-expanded` — a fifth pattern, though at least it is honest about what it does.

**Suggested fix:** One glyph per meaning across the app — `▸`/`▾` disclosure for expand-in-place, `›` chevron at the row's trailing edge for navigate, and nothing else. Add the disclosure glyph to the four inventory tables.

---

### NAV-011 — Next/previous between sibling records in a detail view

**S-30 (Order Detail) — FAIL.** *Medium · Situational idea*
**S-54, S-56 (Lot / Product panels) — FAIL.**
**S-09 (Calendar day detail) — FAIL.**

**Evidence:** `dashboard/index.html:247` (only control is "← Back to Orders"); `dashboard/dashboard.js:1400-1432` (`renderLotPanel` — no navigation controls); `dashboard/dashboard.js:854-888` (`updateProductionDaySelection` — only Close).

Luz's dispatch-queue workflow is explicitly sequential: filter to blocked orders, open one, read the blockers, go back, open the next. Every record costs a round-trip through the list, and the list is rebuilt on return. Same for Arturo walking a set of lots.

**Suggested fix:** Add `‹ Prev / Next ›` to the Order Detail header, scoped to the current filtered list; add the same to the lot panel scoped to the lot list it was opened from. Low effort, high daily payoff.

---

### NAV-012 — Signal hidden content with a count and a path to reveal it; a limited list never looks complete

This is the highest-density failure in the navigation group. Nine list endpoints impose a server-side limit and only three tell the user.

**Done correctly (reference implementations):**
- Allocation lot picker — `dashboard/dashboard.js:2924`, `2993`, `3014`: renders *"Lot list may be incomplete — only the first 500 inventory lots were loaded"* whenever the result count equals the limit.
- Sankey — `dashboard/sankey.html:809-820`: *"Showing most recent 100 transactions per type. Actual volumes may be higher."*
- Activity show-more — `dashboard/dashboard.js:1152-1157`: *"Show all (N more)"* with the exact hidden count.
- Traceability fan-out — `dashboard/traceability.html:764-772`: *"... and N more"* node plus a "Show full chain" control.

**S-14 (Recent Entries) — FAIL.** *High*
`dashboard/dashboard.js:611` (`/ledger/recent?limit=20`), `614` (status text). The status line reads *"Showing 20 most recently entered ledger events"* — which reads as a complete answer, not a truncated one. There is no total, no "View all", and no way to page back. The standard's own Factory Ledger example for this rule is *"Recent Entries shows the last 20 with 'View all.'"*

**S-25, S-26 (Orders) — FAIL.** `dashboard/dashboard.js:2247`, `2261-2263` (`limit=200`). No cue at 200. An office with more than 200 orders in the selected status silently loses the tail.

**S-42 (Expected Receipts) — FAIL.** `dashboard/dashboard.js:3374` (`limit=500`). The summary line (`3412-3413`) reports *"N shown · N open · N overdue"* — all computed from the truncated set, so the counts themselves are wrong past 500 without saying so.

**S-51 (Supply Requests) — FAIL.** `dashboard/dashboard.js:3908` (`limit=500`). Same shape; the open-count badge (`3923-3926`) is computed from the truncated set.

**S-63, S-65 (Traceability search and recent lots) — FAIL.** *This is the most consequential instance.*
`dashboard/traceability.html:376` (`/transactions/history?limit=100`), `448-472` (`filterLots`), `462` (`items.slice(0, 8)` per group), `469` (*"No matches"*).
The entire lot search index is built from the last **100 transactions**. A lot older than that returns *"No matches"* — the same message as a genuinely non-existent lot. Per-group results are additionally capped at 8 with no "more" cue. This is precisely the failure the rule names: *"The count stops Arturo concluding a lot doesn't exist because it's filtered out."* Today he would conclude exactly that.

**S-56 (Product Detail panel) — PARTIAL.** `dashboard/dashboard.js:1477` (`zeroLots.slice(0, 10)`), `1485-1487`. The count of hidden depleted lots is shown (*"...and N more depleted lots"*) but there is **no path to reveal them** — it is static text, not a control.

**S-61 (Production lines grid) — FAIL.** `dashboard/process-flow.html` (`refresh()`): the 100-record truncation is reported to `console.warn` only, never to the user.

**S-02 (Mini-calendar) — PARTIAL.** `dashboard/mini-calendar.js:108` (`limit=200`), `57` (dot title). Ship-date dots are drawn from a 200-order fetch; a day with orders beyond the limit shows no dot and looks clear.

**S-13 (Ingredients) — PARTIAL.** The inventory notes say a "show more" overflow cut exists, and `dashboard.js:1146` defines `ACTIVITY_PREVIEW_ROWS = 4` — but `renderIngredients` (`1090-1143`) never calls `overflowClass` or `showMoreFooter`. The truncation helper exists and is applied only to shipments and receipts (`1196`, `1249`).

**S-39 (Shipping preview) — PARTIAL.** `dashboard/dashboard.js:3115-3133`. The preview lists lines returned by the API with no statement of whether all remaining lines were covered.

**Suggested fix (one change, nine screens):** Add a shared `truncationNotice(returned, limit)` helper next to `fetchSalesAPI` that renders a standard "showing first N — M more not shown" line whenever `returned === limit`, and call it from every list renderer. For Traceability, replace the client-side index entirely with a server-side lot search endpoint.

---

### NAV-003 — Users always know which record they're on

**S-22 (Note modal) — PARTIAL.** `dashboard/dashboard.js:1787-1788`. The title is a generic *"Edit Item"*; it never names the note being edited. With the list rebuilt behind the modal, there is nothing on screen tying the form to a record.

**S-11, S-32 — PARTIAL.** Expanded lot rows (`dashboard.js:930-941`) and the per-line inventory table (`2631-2635`) render values with no heading naming the parent product or line; the association is positional only.

**S-26 — PARTIAL.** `dashboard/dashboard.js:2276-2289`. In Dispatch Queue mode the table looks identical to All Open Orders apart from a small grey summary span; the mode is not named anywhere in the content area.

**S-72, S-88, S-89 (scheduler) — PARTIAL.** `dashboard/scheduler/seven-wells-production-board.html:3-9`, `1555-1580`, `1565`. The KPI strip states outcomes without naming the plan or its horizon; the add-order form and its `alert()` never name the plan being edited.

---

### LAYOUT-002 — Once an element's appearance or behavior is established, apply it identically everywhere

*High · **Hard rule***. This is the single largest systemic finding in the audit. Six distinct patterns exist for jobs that should have exactly one each.

**1. No shared primary-button component — FAIL (S-03, S-07, S-15, S-19, S-22, S-24, S-33, S-41, S-45, S-46, S-52, S-53).**
`dashboard/dashboard.css:236-248` defines `.btn-refresh` as a filled accent button. It is then used for: Refresh (`index.html:118`, `235`, `275`), Export CSV (`233`), Export Matrix (`234`), + New (`189`), + New Expected Receipt (`274`), Request Supply (`295`), Save (`438`, `500`), Submit Request (`390`), Try again (`dashboard.js:618`), Edit Order / Save Header / Save Lines (`2707-2712`), Allocate (`2926`). Twelve different semantic roles share one filled accent style, so nothing on screen is distinguishable as *the* primary action. See ACTION-003 and ACTION-006 in `02-actions-input-touch.md`.

**2. Cancel is a different size from Save — FAIL (S-22, S-45, S-53).**
`dashboard/index.html:390-391`, `438-439`, `500-501`. Save is `.btn-refresh` (`padding: 6px 14px`, 13px); Cancel is `.btn-sm` (`padding: 4px 10px`, 12px, `dashboard.css:394-403`). The rule's companion ACTION-004 is explicit that style, not size, distinguishes the preferred option.

**3. Three modals, three dismissal contracts — FAIL (S-22, S-45, S-53).**
`dashboard/dashboard.js:4103-4107` binds Escape for the Supply Request modal only. The Note modal (`1869-1873`) and the ER modal (`3638-3642`) bind close-button and overlay-click but **not** Escape. The lot panel (`4236-4239`) likewise has no Escape.

**4. Four confirmation patterns — FAIL (S-23, S-35, S-38, S-43, S-87, S-89).**
- native `confirm()`: note delete (`dashboard.js:1771`), SO status change (`2819`), allocation release (`3096`), scheduler reset (`scheduler:1321`)
- inline two-step arm: ER Close / Cancel (`dashboard.js:3470-3483`) — the best of the four, and used once
- native `alert()`: scheduler add-order validation (`scheduler:1565`), note save errors (`dashboard.js:1813`, `1845`)
- **nothing at all**: scheduler order-line delete (`scheduler:1553-1554`)

**5. Six error-surfacing patterns — FAIL (app-wide).**
`showError()` red bar (`dashboard.js:471-477`); `.order-edit-message` (`2687-2693`); `.allocation-feedback` (`2966-2972`); `.supply-feedback` (`3893-3899`); `alert()` (`1813`, `1845`); inline `.order-lines-error` (`2520`). Full treatment in `03-feedback-error-notify.md`.

**6. Hand-rolled markup instead of the component classes — FAIL (S-54, S-55, S-56).**
`dashboard/dashboard.js:1373-1385` (disambiguation) and `1447-1491` (`openProductPanel`) build tables and buttons entirely from inline `style="..."` attributes rather than `.inv-table`, `.btn-sm`, and the token set. Nine inline `style` attributes across the two functions. One of them is actively broken — see LAYOUT-002 item 7.

**7. `--bg-card` does not exist — FAIL (S-55). Contrast-breaking.**
`dashboard/dashboard.js:1379`: `background: var(--bg-card, #fff)`. `--bg-card` is defined nowhere in `dashboard.css`, so the fallback `#fff` applies. The button inherits `color` from the panel, which in the default dark theme is `--text: #f1f5f9`. The disambiguation buttons therefore render **near-white text on a white background**. This is the screen Arturo lands on when a lot code matches more than one product — the one moment he must choose correctly. Cross-referenced as a Critical contrast failure under ACCESS-008 in `05-access-icon-other.md`.

**8. Four site-nav copies — PARTIAL (S-01).**
`dashboard/dashboard.css:96-130` (for `index.html`) versus hand-copied CSS in `sankey.html:42-55`, `process-flow.html`, and `traceability.html:42-55`, plus a duplicated `navToggle` inline script in each of the four files (`index.html:521`, `traceability.html:1398`, `sankey.html:864`, `process-flow.html`). `index.html` additionally wraps the brand in `.site-nav-left` (`index.html:14-16`) which the other three lack. Four copies of one component will drift.

**9. Sankey, Process Flow, Traceability and the scheduler share no stylesheet — FAIL (S-57…S-70, S-72…S-91).**
Each file re-declares its own `:root` token block: `traceability.html:10-34`, `sankey.html` (own `:root`), `process-flow.html` (own `:root`, using `--text-dim` where the dashboard uses `--text-dimmed`), `scheduler:8-18` (an entirely different light "paper" palette). Four token vocabularies for one product. See OTHER-010.

**Suggested fix:** Extract `dashboard.css`'s tokens, buttons, badges, tables, and modal shell into a shared `components.css` linked by all five pages; delete the per-file `:root` blocks and the inline-styled panels; standardise on one confirmation pattern (the ER two-step arm) and one error surface.

---

### LAYOUT-003 — Layout adapts to size, orientation, text size, and label length — **CRITICAL**

**S-03, S-05 (App header + tab bar) — FAIL. The sticky offsets are hard-coded to the desktop header height.**

**Evidence:** `dashboard/dashboard.css:85` (`body { padding-top: 48px }`), `140-142` (`.app-header { position: sticky; top: 48px }`), `259-260` (`.tab-bar { position: sticky; top: 91px }`), `2182-2184` (at ≤768px `.app-header { flex-wrap: wrap }`, `.header-right { width: 100% }`, `.header-center { flex-basis: 100% }`), `2189` (the ≤768px block re-states `.tab-bar { top: 91px }`).

`top: 91px` is 48px of site nav plus a 43px single-row header. At ≤768px the header is explicitly made to wrap to three rows — brand, then the full-width `.header-right` (which itself contains the three-month mini-calendar, `mini-calendar.css:136-142`), then the full-width search row. That header is well over 150px tall. The tab bar continues to stick at 91px, so **the tab bar sits on top of the app header** on every phone. The ≤768px block re-declares the same 91px value, so this is not an oversight of one rule — the compact case was written and the offset was not recomputed.

**Suggested fix:** Replace the magic numbers with a measured offset — set `--header-h` from a `ResizeObserver` on `.app-header`, or make the header + tab bar one `position: sticky; top: 48px` container so the offsets compose automatically.

**S-10…S-13, S-16…S-18, S-25…S-28, S-42, S-43 — FAIL. Six wide tables with no horizontal-scroll wrapper.**

**Evidence:**
- `#orders-table-container` — `dashboard/index.html:240`; the table it holds has 11 columns (`dashboard.js:2313`) and no wrapper with `overflow-x`.
- `#er-table-container` — `dashboard/index.html:283`; 9 columns (`dashboard.js:3424`).
- `.inv-table` ×3 — `dashboard.js:918` (Finished Goods, 4 cols), `1002` (Batch, 3 cols), `1106` (Ingredients, 2 cols); `dashboard.css:674-677` sets `width: 100%` only.
- `.activity-table` ×3 — `dashboard.js:1191`, `1244`, `1315` (5 cols each); `dashboard.css:750-753`.

The pattern **is** present elsewhere in the same file: `.order-detail-table-wrap` and `.allocation-table-wrap` (`dashboard.css:1812-1815`) and `.supplies-table-scroll` (`dashboard.css:2337`). Without it, a narrow window makes the whole page scroll horizontally, taking the sticky header and tab bar out of alignment with the content.

**Suggested fix:** Wrap the six tables in the existing `.supplies-table-scroll` class (or a shared `.table-scroll`) and give each an appropriate `min-width`, matching `.order-readiness-table { min-width: 1040px }` (`dashboard.css:1817`).

**S-57, S-58, S-63, S-64, S-65, S-68, S-69 (Sankey, Traceability) — FAIL. No compact rules beyond the nav.**

**Evidence:** `dashboard/sankey.html:49` and `dashboard/traceability.html:49-55` — the only `@media` block in either file styles `.site-nav`. Then: `traceability.html:75-77` (`.search-row { display: flex; gap: 12px }` holding an input plus two direction buttons plus Trace, no wrap), `traceability.html:314` (`<svg id="graphSvg" width="100%" height="500">` — fixed height), `traceability.html:975` and `992-993` (layout is `4 layers × 180px + 3 × 60px = 900px` wide, positioned at `startX = Math.max(40, (width - 900) / 2)`, so on a 375px viewport the graph starts at x=40 and runs 900px wide inside a 375px SVG).

**S-72…S-90 (scheduler) — FAIL. No responsive rules at all.**

**Evidence:** `dashboard/scheduler/seven-wells-production-board.html:185` is the file's only `@media` block and it is `print`. `:56` fixes `#main { grid-template-columns: 240px minmax(0,1fr) 340px }`; `:102` and `:114` set `min-width: 124px` per day column with `position: sticky` headers; `:32` sets `min-width: 110px` per KPI. On a 1280px laptop the centre board gets 700px for a 7–14 day × 7-station grid.

**All screens — PARTIAL on text size and label length.**
Fifteen `white-space: nowrap` declarations on text that can grow: `dashboard.css:103` (brand), `148` (`h1`), `205` (`.header-right`), `1286` (dispatch summary), `1520` (ship-by cell), `1564` (`.so-badge`), `1587` (`.so-ready-pill`), `1598` (`.dispatch-pill`), `1805` (`.readiness-metrics`), `2053` (`.allocation-table .num`), `2212` (`.er-summary`), `2217` (`.er-notes`, with `overflow: hidden; text-overflow: ellipsis`), `2228` (`.er-actions`), `2401` (`.supply-request-actions`), `786`/`815` (`.created-at-meta`, `.late-lag`). Plus `#cal-range-label { min-width: 160px }` (`406`). Spanish labels run 20–30% longer and none of these can wrap. No `rem`-based type scale exists — every `font-size` in `dashboard.css` is in `px` (over 130 declarations), so the browser text-size setting has no effect on the layout at all.

---

### LAYOUT-005 — Consistent text-emphasis hierarchy; decision-critical data never at tertiary or faint emphasis

**S-25, S-26, S-31 (blocker chips) — FAIL.** *High · Hard rule*

**Evidence:** `dashboard/dashboard.css:1624-1633` — `.readiness-chip { font-size: 10px }`, `.readiness-chip-detail { font-size: 10px; font-weight: 400; opacity: 0.9 }`; rendered at `dashboard.js:1941-1945`.

The blocker detail is *the reason an order cannot ship*. It is rendered at 10px regular weight at 90% opacity — the standard's own definition of tertiary emphasis, which the rule forbids for decision-critical data. `.readiness-none` ("No blockers") is 11px muted (`css:1649-1652`), so the good case and the bad case are both whispered.

**S-37 (Allocation history) — FAIL.** `dashboard/dashboard.css:2055-2062` — every secondary span inside an allocation cell is `font-size: 10px; color: var(--text-muted)`, including the **lot code** of a lot-level reservation (`dashboard.js:2941`) and the line number (`2947`). Which lot is reserved is the whole point of a lot-level reservation.

**S-25 (Factory Ready rows) — FAIL.** `dashboard/dashboard.css:1320` — `.orders-table tr.order-row.so-ready { opacity: 0.6 }`. Orders that are Factory Ready are dimmed to 60%. Ready orders are not finished orders; they are the orders about to be shipped. This inverts the hierarchy for the exact rows Luz is looking for. (Contrast the correct uses of the same device: `.er-inactive` at `2220`, `.allocation-status-released` at `2063-2065`, `.supply-request-done` at `2402`, `.note-card.done` at `1020` — all genuinely inactive.)

**S-56 (Product panel) — FAIL.** `dashboard/dashboard.js:1476` — the depleted-lots table is rendered at `opacity: 0.6` on top of `--text-muted`, compounding two dimming devices.

**S-06 (Today So Far) — PARTIAL.** `dashboard/dashboard.css:367`, `370` — the label span and the `<strong>` count are both 13px; the number Blubber is meant to read first has weight but no size advantage.

**S-08, S-09 (Calendar) — PARTIAL.** `dashboard/dashboard.css:620-627` vs `606-613` — `.production-detail-count` is 11px while `.production-detail-name` is 12px, so the value is smaller than its label. Same pattern at `509-514` (`.calendar-summary-row strong` at 12px).

**S-02 (Mini-calendar) — FAIL.** `dashboard/mini-calendar.css:63-74` — day numbers are `font-size: 8px` in a `height: 10px` cell, dropping to the same 8px at ≤1050px (`129-133`). Eight pixels is below any legible operational minimum (see ACCESS-006).

**S-49, S-50, S-51, S-70 — PARTIAL.** 10px `dt` labels (`dashboard.css:2381-2388`), 10px allocation source pills (`2074`), 11px `.detail-table td` in Traceability (`traceability.html:197-200`).

---

### LAYOUT-006 — Position by importance in reading order

**S-30 (Order Detail) — FAIL.** *High*

**Evidence:** `dashboard/dashboard.js:3226-3231` (the KPI row is **built**) and `3285` (`html += summaryHtml` — it is **appended after** the line-items table closed at `3282`).

Total Ordered / Shipped Effective / Remaining Effective / Pallets are the four headline numbers of the record. They are rendered *below* a nine-to-ten-column line table. On any order with more than a handful of lines, the user scrolls past the detail to reach the summary. The variable is even named `summaryHtml` and constructed before the table, so the intent was clearly to lead with it.

**Suggested fix:** Move line `3285` to immediately after `3209` (`renderOrderReadinessSummary`).

**S-25, S-26 — PARTIAL.** `dashboard/dashboard.js:2313` — column order is `[expand] [ready] SO# | Customer | Order Date | Ship By | Status | Dispatch | Blockers | Pallets | Effective Remaining`. The identifying column is third, behind two control columns (`dashboard.css:1331-1334`, 32px and 34px). The rule asks for the identifying column first.

---

### LAYOUT-007 — Group related items; controls always distinguishable from content

**S-49 (Supplies inventory rows) — PARTIAL.** `dashboard/dashboard.js:3849` — a `<tr>` carrying `role="button" tabindex="0" aria-expanded`. The entire data row is a control. It is at least announced correctly and has a focus ring (`dashboard.css:2300-2304`), which is more than the inventory tables get.

**S-11, S-27, S-32 — PARTIAL.** `dashboard/dashboard.js:923`, `dashboard.css:699-700` — `tr.expandable` is a clickable row with no role, no tabindex, and no glyph. Content and control are indistinguishable.

**S-10…S-13, S-17, S-18 — PARTIAL.** `dashboard/dashboard.js:935`, `1033`, `1120`, `1211`, `1264`; `dashboard.css:840-848` — `.lot-link` is a `<span>` styled as an underlined accent-coloured link with `cursor: pointer`. It looks exactly like a hyperlink and behaves like one on click, but it is not an `<a>`, is not focusable, and does not respond to Enter.

**S-19 — PARTIAL.** `dashboard/dashboard.css:977-988` — `.notes-filter-btn.active` becomes a solid `--primary` pill, visually identical to `.note-cat-badge.cat-note` (`1059`) which is a non-interactive status chip in the same view.

**S-55, S-56, S-37, S-51 — PARTIAL.** Buttons built with inline styles (`dashboard.js:1377-1380`) and `.btn-sm` action buttons sitting inside data cells with no separator (`3438-3442`, `3944`).

---

### LAYOUT-010 — Align to a shared grid with a fixed spacing scale; indentation shows nesting

**All screens — PARTIAL.** *High · Hard rule*

**Evidence:** `dashboard/dashboard.css:37-38` — the only spatial tokens in the file are `--radius: 12px` and `--radius-sm: 8px`. There is no `--space-*` scale. Grepping the padding and gap values in use yields 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 32, 40px — seventeen distinct steps with no ratio between them, e.g. `.today-tile { padding: 14px 16px 16px }` (`356`), `.recent-entry-card { padding: 16px }` (`293`), `.note-card { padding: 12px 16px }` (`1012`), `.section-header { padding: 12px 16px }` (`349`), `.panel-header { padding: 14px 20px }` (`875`), `.panel-body { padding: 16px 20px }` (`894`), `.supply-lot-card { padding: 9px 12px }` (`2371`).

**Nesting indentation is three different values for one level:** `.lot-row td { padding: 5px 12px 5px 32px }` (`730`), `.activity-detail td { padding: 8px 12px 8px 24px }` (`820`), `.supply-lot-detail { padding: 12px 16px 14px 32px }` (`2363`, dropping to 16px at ≤768px, `2434`).

**Numeric alignment — PARTIAL (S-10, S-12).** `dashboard/dashboard.js:926` — the Finished Goods "Cases" column has no `.num` class, so it is left-aligned between two right-aligned numeric columns ("On Hand (lb)" at `925`, "Pallets" at `927`). Same at `1010-1018` for Batch Inventory's "Est. Batches". The rule's own Factory Ledger example is *"Quantities right-aligned so they compare at a glance."*

**Suggested fix:** Define `--space-1: 4px … --space-6: 24px`, replace the seventeen ad-hoc values, and pick one nesting indent. Add `class="num"` to the two case-count columns.

---

### LAYOUT-011 — Content fills the viewport; nothing actionable is hidden behind fixed bars

**S-03, S-05 — FAIL (Critical clause).** The sticky tab bar overlaps the app header at ≤768px — see LAYOUT-003. On phones this hides the search field and the Refresh control behind the tab strip.

**S-25, S-26, S-42 — FAIL.** `dashboard/dashboard.css:282` — `.tab-content { max-width: 1200px; margin: 0 auto }`. On a 2560px monitor an 11-column orders table is squeezed into 1200px with 680px of dead margin on each side, while `.order-blockers-cell { min-width: 190px; max-width: 320px }` (`1325-1328`) fights for room inside it. LAYOUT-019 explicitly exempts dense tables from the reading-width cap.

**S-10…S-13, S-16…S-18, S-27, S-43, S-57, S-58, S-63…S-65, S-68 — FAIL.** Consequence of the missing scroll wrappers (LAYOUT-003): the page itself scrolls horizontally, so the fixed nav and sticky header stay put while the content slides under them.

**S-72…S-90 — FAIL.** `dashboard/scheduler/seven-wells-production-board.html:26` — `#app { display: grid; grid-template-rows: auto auto 1fr; height: 100vh }` with `#leftpanel,#rightpanel { overflow-y: auto }` and `#center { overflow: auto }`. Three independent scroll regions inside a locked `100vh` on a surface with fixed 240px/340px side panels; below ~1100px the centre board is unusable and there is no way to collapse a panel.

**Suggested fix:** Let `.tab-content` stay capped for prose sections but allow table sections to break out to full width (`max-width: none` on `#section-orders`, `#section-expected`), and give the scheduler's side panels a collapse control.

---

### LAYOUT-012 — Space and group controls so a consequential control can't be mis-hit — **CRITICAL**

**S-20 (Notes row actions) — FAIL.**

**Evidence:** `dashboard/dashboard.js:1732-1733` (Edit `✎` and Delete `✕` rendered adjacent); `dashboard/dashboard.css:1102-1108` (`.note-actions { gap: 4px }`), `1111-1116` (`.note-action-btn { padding: 3px 6px; font-size: 11px }` → roughly 22×20px), `2193` (at ≤768px `.note-actions { opacity: 1 }`).

Two 22×20px buttons, 4px apart, one of which destroys a record. On desktop they are revealed on hover, so the pointer is already near them when they appear. On touch they are permanently visible with no hover state and no size increase. `confirm()` is the only guard.

**S-25, S-26, S-28 (Orders row leading edge) — FAIL.**

**Evidence:** `dashboard/dashboard.js:2320` (expand toggle) and `2321` (Factory Ready checkbox) in adjacent cells; `dashboard.css:1331-1334` (`.order-expand-col`/`.order-expand-cell` 32px, `.order-ready-col`/`.order-ready-cell` 34px with `padding-left: 8px`), `1336-1342` (`.order-ready-checkbox { width: 16px; height: 16px }`), `1344-1357` (`.order-expand-toggle { width: 22px; height: 22px }`).

A 22px button and a 16px checkbox roughly 8px apart at the leading edge of every row. One expands a panel; the other writes a state that the floor reads as "this order is ready to ship". The rule's test names this shape exactly: *"Are there rows with two different actions within thumb-width at the same edge?"* There is no confirmation on the Factory Ready write (`dashboard.js:2455-2486`), and the mis-hit is an outward-facing signal to the floor.

**S-87 (Scheduler order book) — FAIL.**

**Evidence:** `dashboard/scheduler/seven-wells-production-board.html:1536-1537` — `✕` (Exclude from plan, reversible) and `⌫` (Delete order line, **not** reversible) rendered as adjacent `<a class="o-action">` elements; `:158` styles both identically. The delete handler at `:1553-1554` runs immediately with **no confirmation of any kind**. The two glyphs are ✕ and ⌫ — both "remove" symbols, side by side, one recoverable and one not.

**S-43 (ER row actions) — PARTIAL.**

**Evidence:** `dashboard/dashboard.js:3438-3442` (Edit / Close / Cancel); `dashboard.css:2229` (`.er-actions .btn-sm { margin-left: 4px }`).

Three identically styled buttons 4px apart, the last of which cancels the record. Mitigated — and this is the one place in the app that does it right — by the two-step arm at `dashboard.js:3470-3483`: the first click changes the label to "Cancel? Confirm" and adds `.er-armed` (`dashboard.css:2230`), reverting after 4 seconds. Downgraded from FAIL to PARTIAL on that basis; the spacing and identical styling are still wrong.

**S-33 — PARTIAL.** `dashboard/dashboard.js:2707-2709` — "Save Header" and "Save Lines" adjacent, both `.btn-refresh`, `gap: 8px` (`dashboard.css:1821-1826`). Different scopes, identical appearance, no spacing cue.

**S-37 — PARTIAL.** `dashboard/dashboard.js:2952` — the Release button is `.btn-sm` with `color: var(--danger)` (`dashboard.css:2081`) inside a dense 12px table row. Red text on a small neutral button is the app's only destructive treatment and does not match ACTION-006's "red fill".

**S-73 — PARTIAL.** `dashboard/scheduler/…:11-13` — "Set as baseline" (primary amber), "Print", "Copy week" adjacent in the topbar with `gap: 18px` on the container but no separation between the three.

**Suggested fix:** Move Delete out of the notes row into an overflow behind Edit; separate the orders-row expand toggle from the Factory Ready checkbox with the SO number column between them, or move Factory Ready to the row's trailing edge; add a confirmation (the ER two-step arm) to the scheduler's order-line delete.

---

### LAYOUT-015 — Respect safe areas and system UI — **Mobile**

**S-01, S-03, S-05, S-10…S-13, S-16…S-18, S-25…S-28, S-42, S-43, S-57, S-58, S-63…S-65 — FAIL.** *High · Hard rule*

**Evidence:** No occurrence of `viewport-fit`, `env(safe-area-inset-*)`, or `safe-area` in any file under `dashboard/` (verified by grep across `*.html`, `*.css`, `scheduler/*.html`). `dashboard/index.html:5` declares `<meta name="viewport" content="width=device-width, initial-scale=1.0">` with no `viewport-fit=cover`, and `dashboard/dashboard.css:96-101` sets `.site-nav { position: fixed; top: 0; left: 0; right: 0; padding: 0 20px }`.

On a notched phone in landscape the fixed nav's brand and hamburger sit under the camera housing. There is also no `scroll-margin-top` anywhere, so a form field focused inside a modal scrolls to the viewport top and lands **under** the sticky app header and tab bar (compounding LAYOUT-003).

**Suggested fix:** Add `viewport-fit=cover` to the four viewport metas and `padding-left/right: max(20px, env(safe-area-inset-left/right))` to `.site-nav`; add `scroll-margin-top: calc(var(--header-h) + 8px)` to focusable form controls.

**Marked N (desktop-only surface):** every Blubber-primary screen (S-06…S-09, S-14, S-15, S-57…S-62, S-70…S-91) and every screen whose inventory Mobile column is **No** (S-68, S-69, S-72…S-90).

---

### LAYOUT-018 — Use color sparingly and for meaning; keep chrome neutral

**S-24, S-41 (toolbars) — FAIL.** *High*

**Evidence:** `dashboard/index.html:233-235` and `274-275` — Export CSV, Export Matrix, Refresh, + New Expected Receipt are all `.btn-refresh`, i.e. all filled `--primary` (`dashboard.css:236-247`). The Orders toolbar therefore shows three filled accent buttons in a row and the ER toolbar shows two. Accent fill no longer means "this is the action to take"; it means "this is a button".

**S-47 (Supplies sub-tab badges) — FAIL.**

**Evidence:** `dashboard/index.html:302`, `305`, `308` — `<span class="supplies-count-badge">0</span>`; `dashboard/dashboard.css:2306-2320` — `.supplies-count-badge { background: var(--badge-amber-bg); color: var(--badge-amber-text) }` unconditionally; `dashboard/dashboard.js:3673-3678` sets only `textContent`.

Three permanently amber warning chips sit in the tab labels, showing amber even when the count is zero. Amber is the app's low-stock/overdue colour (`.supply-low-badge` at `css:2356`, `.er-badge-overdue` at `2225`, overdue row tint at `2222`). A chip that is always amber teaches the user to ignore amber. The same badge is used for the Supply Requests open count (`index.html:322`) with the same problem.

**Suggested fix:** Render the badge only when the count is greater than zero, or use `--badge-bg` (neutral blue) at zero and `--badge-amber-bg` above zero.

**S-33, S-36 — FAIL.** `dashboard/dashboard.js:2707-2712` and `2926` — Edit Order, Save Header, Save Lines, and Allocate are all filled accent, in a view that also carries filled accent Refresh in the chrome above it.

**S-25, S-26, S-42 — PARTIAL. Two colour systems collide.**

**Evidence:** `dashboard/dashboard.css:18-19` (`--category-granola: #fbbf24`, `--category-coconut: #60a5fa`) versus `21-22` (`--warning: #f59e0b`, `--danger: #ef4444`) and `28` (`--badge-amber-text: #fbbf24`).

`--category-granola` is byte-identical to `--badge-amber-text`. A granola product name (`dashboard.css:373`, `506`, `583`, `671`) is rendered in exactly the colour that elsewhere means "overdue" or "low stock". `--category-coconut: #60a5fa` is likewise `--primary-hover`, the interactive colour. Product family and system status share a palette. Full treatment under FEEDBACK-012.

**S-55 — FAIL.** `dashboard/dashboard.js:1379` — hard-coded `#fff` background outside the token system (see LAYOUT-002 item 7).

**S-19, S-22, S-45, S-46, S-53 — PARTIAL.** Accent-filled Save/Submit alongside accent-filled Refresh/New in the same view.

**S-59, S-62, S-67 — PARTIAL.** `dashboard/sankey.html` `.banner.warning` and `dashboard/process-flow.html:150` `.error-banner` use colour as the only differentiator between the warning and error states; `dashboard/traceability.html:175` `.status-bar.error` changes text colour only.

---

### LAYOUT-020 — Never move content under the user — **CRITICAL**

**S-14 (Recent Entries) — FAIL.**

**Evidence:** `dashboard/dashboard.js:626-631` (`startRecentEntriesPolling` — `setInterval(refreshRecentEntries, 60000)` while the tab is open), `602-624` (`refreshRecentEntries` sets `feed.innerHTML` to a loading block, then), `568-600` (`renderRecentEntries` replaces `feed.innerHTML` wholesale, newest first).

Every 60 seconds the feed is torn down to a "Loading the latest ledger events…" block and rebuilt. Any new entry inserts at the top and pushes every card down. The rule's Factory Ledger example names this feed explicitly: *"Same for the Recent Entries feed and Today So Far tile."* The current implementation is worse than an insertion — the whole list disappears mid-read.

**Suggested fix:** Poll into a buffer and show *"3 new entries — tap to refresh"* above the feed; never replace `innerHTML` on a background refresh.

**S-25, S-26, S-27, S-28 (Orders) — FAIL.**

**Evidence:** `dashboard/dashboard.js:2472` and `2479` — `renderOrdersList()` is called optimistically on checkbox change and again when the server responds. `2337` rewrites `container.innerHTML`.

Two full table rebuilds per checkbox click. Between them the rows re-sort in Dispatch Queue mode (`2066-2072`) and the `so-ready` opacity class changes, so a row the user is about to click can move. `bindOrderReadyNoteControls` (`2441`) does the same on note save.

**S-20 (Notes) — FAIL.** `dashboard/dashboard.js:1741-1753` — the checkbox `change` handler calls `refreshNotes()`, which refetches and re-renders the whole list (`1657-1674`). With "Show completed" unchecked (the default, `state.notesShowDone = false` at `1653`) the ticked item is removed from the list and every row below it jumps up. Ticking two items in a row is a mis-click waiting to happen.

**S-49 (Supplies inventory) — FAIL.** `dashboard/dashboard.js:3796-3814` — `loadSupplyLots` calls `renderSuppliesInventory()` once when it starts (`3799`) and once when it finishes (`3813`); `renderSuppliesInventory` rewrites `container.innerHTML` (`3858`). The lot fetch is asynchronous, so the whole table is rebuilt twice at an unpredictable moment after the user's tap, and the expanded detail row changes height (`3854`), moving everything below.

**S-61 (Process Flow) — FAIL.** `dashboard/process-flow.html` — `setInterval(refresh, REFRESH_MS)` with `REFRESH_MS = 60000`, and `renderDashboard` sets `grid.innerHTML = html`. The comment in the source says *"Build new HTML (update in-place if possible)"* — the in-place path was never written.

**S-87 (Scheduler order book) — FAIL.** `dashboard/scheduler/…:1550-1554` — exclude, include, and delete handlers each call `recalc()`, which re-runs the scheduler and re-renders the board and the order book, reordering rows by status priority (`:1519`).

**S-58, S-68 — PARTIAL.** `dashboard/sankey.html:851-856` — a window `resize` triggers a debounced **refetch and full redraw**; `dashboard/traceability.html:1142` — `container.scrollIntoView({ behavior: 'smooth' })` fires on every graph render. Both are user-initiated, so they fall inside the rule's exception, but the Sankey resize handler also re-runs the sample-data fallback path (see CHART findings).

**Passing:** `dashboard/dashboard.js:3153-3157` (`updateAllocationCountdowns`) updates only `el.textContent` on a fixed-width `.num` cell — the correct pattern for a live value.

---

### LAYOUT-001 — Show only what the primary task needs

**S-24, S-25, S-26, S-27 — PARTIAL.** `dashboard/index.html:203-237` — the Orders toolbar carries eight controls above the table: status select, dispatch select, customer search, two checkboxes, a summary span, and three buttons (two of which are exports). Exports are a monthly task competing for space with a daily one.

**S-30, S-33, S-36 — PARTIAL.** `dashboard/dashboard.js:3180-3306` — `renderOrderDetail` emits, in one pass and all expanded: header, readiness card with blocker chips, a 9–10 column line table with a per-line inventory expander, a 4-tile KPI row, a four-field allocation form, an allocation history table, a shipping-preview section, and a notes card. Nothing is progressively disclosed.

**Suggested fix:** Collapse Reservations and Shipping Preview behind disclosures that open on demand; move the two exports behind a "···" overflow in the Orders toolbar.

---

### LAYOUT-004, LAYOUT-008, LAYOUT-009, LAYOUT-013, LAYOUT-017, LAYOUT-019 — remaining PARTIALs

**LAYOUT-004 (S-06) — PARTIAL.** `dashboard/dashboard.css:367`, `370`. See LAYOUT-005.
**LAYOUT-004 (S-25, S-26, S-30) — PARTIAL.** `dashboard/dashboard.css:1778` — Ship By in the Order Detail header is 13px `--text-muted`; the rule's example says status and ship date should be the largest text on an SO detail. The order number is 18px (`1764`) and the status badge 11px (`1560`).

**LAYOUT-008 (S-03, S-05, S-24, S-41) — PARTIAL.** *Medium.* `dashboard/dashboard.css:133-143`, `253-263`, `2271-2282` — every sticky bar separates itself with a 1–2px border on an opaque surface. The rule asks for *"a subtle shadow, fade, or blur rather than a heavy background or no separation"*; the current treatment is the heavy-background variant. `.supplies-inventory-toolbar` has the same issue with a `z-index: 20` that is lower than `.tab-bar`'s 99, so at the top of a scroll the two stack ambiguously.

**LAYOUT-009 (S-03, S-24, S-41) — PARTIAL.** *Medium.* The inverse problem — ambiguous controls with **no** caption: `#orders-status-filter` (`index.html:205`) has neither a label nor an `aria-label` while its sibling `#orders-dispatch-filter` (`218`) has one; `#global-search` (`33`), `#orders-customer-search` (`223`), and `#er-text-filter` (`267`) rely on placeholders that vanish on input. `#supplies-search` (`311-314`) shows the correct pattern with an `.sr-only` label.
**LAYOUT-009 (S-33) — PARTIAL.** `dashboard/dashboard.js:3190` — the status `<select>` in edit mode has an `aria-label` but no visible label, and it sits where a read-only status badge was a moment ago.
**LAYOUT-009 (S-28) — PARTIAL.** `dashboard/dashboard.js:2361` — *"Factory Ready note"* labels a text input inside a drawer that is itself unlabelled; the caption explains the field but not the drawer.

**LAYOUT-013 — PARTIAL/FAIL.** *Medium.* `dashboard/dashboard.css` has exactly three breakpoints: 768px (`121`, `2181`, `2426`) and 480px (`318`, `382`, the latter two covering only the Recent Entries cards and the Today tile). There is no intermediate desktop breakpoint and no tertiary-panel-first strategy. At half-screen (~960px) the layout is still the full desktop layout, and the six unwrapped tables (LAYOUT-003) start forcing horizontal page scroll — the exact half-width case the rule's Factory Ledger example calls out. Marked **F** for the six table screens (S-10…S-13, S-25…S-28, S-42, S-43) and **PA** elsewhere.

**LAYOUT-017 — PARTIAL.** *Medium.* `dashboard/dashboard.css:6-33` defines seven surface tokens: `--bg #0f172a`, `--surface #1e293b`, `--surface-alt #1a2536`, `--surface-hover #283548`, `--row-alt #1a2536`, `--row-header #162032`, `--panel-header-bg #162032`. Two pairs are byte-identical duplicates (`--surface-alt` = `--row-alt`; `--row-header` = `--panel-header-bg`), leaving four distinct levels — one more than the rule allows. Worse, the levels do not nest monotonically: `--surface-alt` (#1a2536) and `--row-header` (#162032) are **darker** than `--surface` (#1e293b), so going one level deeper moves the surface *toward* the page background, inverting the containment cue. `dashboard/traceability.html:188` adds a fifth, hard-coded `.detail-card { background: #16213e }`.

**LAYOUT-019 — PARTIAL/FAIL.** *Low/Contextual.* Prose is correctly constrained: `.recent-entries-page { max-width: 760px }` (`css:286`), `.detail-panel { max-width: 1000px }` (`traceability.html:185`). But the blanket `.tab-content { max-width: 1200px }` (`css:282`) applies the same cap to the dense tables the rule exempts — marked **F** on S-25, S-26, S-27, S-42.

---

## N/A register

| Rule | Screens | Reason |
|---|---|---|
| NAV-002 | Most read-only views | No multi-step flow exists; nothing to escape from. |
| NAV-003 | S-01, S-05 | Chrome; carries no record. |
| NAV-004 | All non-tab screens | No tab strip or segmented control present. |
| NAV-005 | Overlays, modals, standalone pages | Not inside a tab pane. |
| NAV-006 | Screens with no tab/segment labels | No labels to assess. |
| NAV-009…011 | Non-hierarchical screens, states, toolbars, modals | No parent/child list or sibling sequence. |
| NAV-012 | Empty states, confirmations, toolbars | Nothing is hidden. |
| LAYOUT-008 | Non-persistent content areas | No persistent control layer on the screen. |
| LAYOUT-014, LAYOUT-015 | All Blubber-primary screens; all Mobile=No screens | **Desktop-only surface** — per the audit brief's Mobile-weighting rule. |
| LAYOUT-016 | Overlays, chrome, non-form screens | No bottom-anchored critical content. |
| LAYOUT-019 | Chrome, overlays, states | No prose or wide table to constrain. |

## Unverifiable from code

| Rule | Screens | What a browser check must confirm |
|---|---|---|
| LAYOUT-014 | S-01…S-05, S-10, S-11, S-13, S-24…S-56 (Luz/Arturo, Mobile Yes/Partial) | Whether rotation preserves entered form values and scroll position, and whether either orientation is unusable. No rotation handler or width-keyed state exists in code, which is necessary but not sufficient evidence. |

---

*End of 01-nav-layout.md*

# 03 — Audit: Feedback, Errors & Notifications (FEEDBACK, ERROR, NOTIFY)

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) §6 (FEEDBACK-001…014), §7 (ERROR-001…010), §12 (NOTIFY-001…012) — 36 rules
**Screens:** [00-screen-inventory.md](00-screen-inventory.md) — S-01…S-91
**Status:** Audit findings. **No application code was modified.**

*§11 Performance & Perceived Speed contains no standalone rules; the standard routes its constraints through FEEDBACK-001, 003, 006, 007, LAYOUT-020, ERROR-002, and CHART-002, all of which are audited in this file or in groups 01 and 04.*

---

## How to read this

Statuses and tiering are as defined in [01-nav-layout.md](01-nav-layout.md): **P** PASS · **PA** PARTIAL · **F** FAIL · **N** N/A · **U** UNVERIFIABLE-FROM-CODE · **·** out of tier scope.

### Rule importance (drives Tier B scope)

- **Critical:** FEEDBACK-001, 003, 011, 012; ERROR-001, 002, 003, 007; NOTIFY-004
- **High:** FEEDBACK-002, 007, 008, 009, 013; ERROR-004, 005, 008, 010; NOTIFY-001, 002, 005, 009, 011
- **Medium:** FEEDBACK-004, 005, 006, 010, 014; ERROR-006, 009; NOTIFY-003, 006, 007, 010, 012
- **Low/Contextual:** NOTIFY-008

### There is no notification channel in this product

Verified by grep across `dashboard/*.html`, `dashboard/*.js`, `dashboard/*.css`, and `dashboard/scheduler/*.html`: no `Notification` API, no `serviceWorker`, no push registration, no toast library, no email or SMS trigger. The only auto-dismissing message is `showSupplyFeedback` (`dashboard.js:3893-3899`), which is an in-app `role="status"` region, not a notification.

Consequently **NOTIFY-001, 002, 003, 006, 007, 008, 009, and 012 are N/A across all 91 screens**, with the reason *"no notification channel exists in this product."* They are recorded once in the N/A register rather than repeated in the matrix.

The four NOTIFY rules that **do** apply are audited in full:
- **NOTIFY-004** — errors must be in-app alerts, never notifications (vacuously satisfied, and the separation is implemented correctly — recorded as a positive finding)
- **NOTIFY-005** — new information surfaced in context while the user is in the app
- **NOTIFY-010** — badge counters
- **NOTIFY-011** — important information findable on app entry, independent of any transient channel. **With no notification channel at all, this rule carries the product's entire attention model.**

### Scheduler scoping

Per the audit brief, sync and notification rules are recorded as **N/A with reason** on the scheduler (`localStorage` only, no API — `scheduler:376-378`), not as failures. State-persistence is the exception: **ERROR-002 applies to the scheduler and it passes**, as the only surface in the product that survives a browser refresh.

---

## Matrix A — Feedback, Status & Progress (FEEDBACK-001…014)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | N | N | N | N | N | N | N | P | P | N | P | P | P | P |
| S-02 | Mini-calendar strip | PA | N | **F** | N | PA | P | PA | PA | P | N | PA | P | P | P |
| S-03 | App header | PA | PA | **F** | P | PA | P | **F** | PA | P | N | P | P | P | P |
| S-04 | Global search + results | PA | N | **F** | N | **F** | PA | N | PA | PA | N | P | P | P | P |
| S-05 | Tab bar (7 tabs) | N | N | N | N | N | N | N | P | P | N | P | P | P | P |
| S-06 | Today So Far tile | P | N | **F** | **P** | PA | P | **F** | PA | N | N | **F** | **F** | P | P |
| S-07 | Today So Far — error/retry | P | N | PA | P | PA | P | N | P | N | N | P | P | P | P |
| S-08 | Production Calendar | P | N | **F** | P | PA | P | **F** | P | **P** | N | **F** | **F** | P | PA |
| S-09 | Calendar day detail panel | P | N | **F** | P | PA | P | **F** | P | **P** | N | **F** | **F** | P | P |
| S-10 | Finished Goods panels | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | P | P |
| S-11 | FG per-product lot rows | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | P | P |
| S-12 | Batch Inventory | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | **F** | **F** | P | P |
| S-13 | On-Hand Ingredients | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | P | PA |
| S-14 | Recent Entries feed | PA | N | **F** | **P** | PA | **F** | **F** | PA | N | **F** | P | P | P | P |
| S-15 | Recent Entries states | P | N | PA | P | PA | P | N | P | N | N | P | P | P | P |
| S-16 | Daily Entries + toolbar | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | PA | P |
| S-17 | Shipping log | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | P | P |
| S-18 | Receiving log | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | P | P |
| S-19 | Notes toolbar | P | N | **F** | P | PA | P | **F** | PA | P | **F** | P | P | P | P |
| S-20 | Notes list (cards) | **F** | N | **F** | P | PA | P | **F** | **F** | PA | **F** | **F** | P | **F** | P |
| S-21 | Notes empty state | N | N | N | N | N | N | N | P | N | N | P | P | P | P |
| S-22 | Note create/edit modal | **F** | N | **F** | N | **F** | P | N | **F** | N | N | P | P | P | P |
| S-23 | Delete-note confirm | N | N | N | N | N | N | N | PA | N | **F** | P | P | P | P |
| S-24 | Orders toolbar | P | N | **F** | P | PA | P | **F** | PA | P | N | P | P | P | P |
| S-25 | Orders list table | P | N | **F** | P | PA | P | **F** | **F** | **F** | **F** | **F** | **F** | PA | P |
| S-26 | Dispatch Queue mode | P | N | **F** | P | PA | P | **F** | **F** | **F** | **F** | PA | **F** | PA | P |
| S-27 | Orders expandable lines | P | N | **F** | P | PA | P | **F** | PA | PA | **F** | PA | **F** | PA | P |
| S-28 | Factory Ready toggle+note | **F** | N | **F** | N | PA | P | **F** | **F** | PA | **F** | P | **F** | P | P |
| S-29 | Orders empty state | N | N | N | N | N | N | N | P | N | N | P | P | P | P |
| S-30 | Order Detail header/KPI | P | N | **F** | P | PA | P | **F** | PA | **F** | N | P | **F** | P | P |
| S-31 | Order Detail line table | P | N | **F** | P | PA | P | **F** | PA | P | N | **F** | **F** | **F** | P |
| S-32 | Per-line inventory expander | P | N | **F** | **F** | PA | P | N | P | P | N | P | P | P | P |
| S-33 | Order Detail edit mode | PA | **F** | **F** | N | PA | P | N | PA | P | N | P | **F** | P | P |
| S-34 | Edit-locked notice | N | N | N | N | N | N | N | **P** | N | N | P | P | P | P |
| S-35 | SO status change confirm | PA | N | N | N | PA | N | N | PA | N | N | P | P | P | P |
| S-36 | Reservations / allocation | PA | N | **F** | **P** | PA | **P** | N | P | P | **F** | P | P | P | P |
| S-37 | Allocation history table | P | N | **F** | P | PA | P | PA | P | P | **F** | P | P | **F** | P |
| S-38 | Release reservation confirm | PA | N | N | N | PA | N | N | PA | N | **F** | P | P | P | P |
| S-39 | Shipping capacity preview | PA | **F** | **F** | P | PA | P | N | P | N | N | P | P | P | P |
| S-40 | Order Detail notes card | N | N | N | N | N | N | N | P | N | N | P | P | P | P |
| S-41 | Expected Receipts toolbar | PA | N | **F** | P | PA | P | **F** | PA | P | N | P | P | P | P |
| S-42 | Expected Receipts table | PA | N | **F** | P | PA | P | **F** | PA | PA | **F** | P | P | PA | P |
| S-43 | ER row actions | **F** | N | **F** | N | PA | P | N | PA | P | **F** | P | P | P | P |
| S-44 | ER empty state | N | N | N | N | N | N | N | P | N | N | P | P | P | P |
| S-45 | ER create/edit modal | P | N | **F** | N | **F** | PA | N | PA | N | N | P | P | P | P |
| S-46 | Supplies header + Request | N | N | N | N | PA | N | N | P | N | N | P | P | P | P |
| S-47 | Supplies sub-tabs | P | N | **F** | P | PA | P | **F** | P | **P** | N | P | **F** | P | P |
| S-48 | Supplies search field | PA | N | N | N | **F** | PA | N | P | P | N | P | P | P | P |
| S-49 | Supplies inventory table | PA | N | **F** | P | PA | P | **F** | PA | **P** | **F** | P | P | P | P |
| S-50 | Supply lot / incoming detail | P | N | **F** | **P** | PA | **P** | N | P | P | N | P | P | P | P |
| S-51 | Supply Requests list | PA | N | **F** | P | PA | P | **F** | PA | P | **F** | P | P | P | P |
| S-52 | Supply Requests feedback | P | N | N | N | PA | P | N | **P** | N | N | P | P | P | P |
| S-53 | Request Supply modal | **P** | N | **F** | **P** | PA | P | N | **P** | N | N | P | P | P | P |
| S-54 | Lot Detail side panel | P | N | **F** | P | PA | P | N | P | **F** | N | P | P | P | P |
| S-55 | Lot disambiguation | P | N | **F** | N | PA | P | N | PA | P | N | P | P | P | P |
| S-56 | Product Detail panel | P | N | **F** | P | PA | P | N | P | **F** | N | P | P | P | P |
| S-57 | Sankey controls bar | P | N | **F** | P | PA | PA | PA | PA | P | · | P | P | · | · |
| S-58 | Sankey chart + legend | P | N | **F** | P | PA | PA | PA | **F** | P | · | PA | P | · | · |
| S-59 | Sankey banner / loading | P | N | **F** | P | PA | PA | PA | PA | P | · | PA | P | · | · |
| S-60 | Summary strip | PA | N | **P** | P | P | P | **P** | P | P | · | P | P | · | · |
| S-61 | Production lines grid | PA | N | **P** | P | P | P | PA | PA | P | · | P | P | · | · |
| S-62 | Error / stale banners | P | N | **P** | P | P | P | **P** | P | P | · | PA | P | · | · |
| S-63 | Lot search + type-ahead | PA | N | **F** | P | PA | PA | N | PA | PA | N | P | P | P | P |
| S-64 | Trace direction + Trace | **P** | **F** | **F** | **P** | P | P | N | P | **P** | N | P | P | P | P |
| S-65 | Recent lots strip | P | N | **F** | P | P | P | N | P | P | N | P | P | P | P |
| S-66 | Trace legend | N | N | N | N | N | N | N | P | N | N | **P** | P | P | P |
| S-67 | Status bar | **P** | N | **F** | **P** | **P** | P | N | P | N | N | **P** | P | P | P |
| S-68 | Trace graph + zoom | P | **F** | **F** | P | P | P | N | P | PA | N | **P** | P | P | P |
| S-69 | Node tooltip | N | N | N | N | N | N | N | P | N | N | **P** | P | P | P |
| S-70 | Trace detail + exports | P | N | **F** | P | P | P | N | P | N | N | **P** | P | P | P |
| S-71 | Print audit report | N | N | N | N | N | N | N | P | N | N | **F** | P | · | · |
| S-72 | Topbar KPI strip | N | N | N | N | P | N | N | P | N | · | PA | P | · | · |
| S-73 | Topbar actions | P | N | N | P | P | N | N | P | N | · | P | P | · | · |
| S-74 | Legend bar | N | N | N | N | N | N | N | P | N | · | **P** | P | · | · |
| S-75 | Delta panel | N | N | N | N | P | N | N | P | N | · | **F** | P | · | · |
| S-76 | Settings panel (form) | P | N | N | N | P | N | N | PA | N | · | P | P | · | · |
| S-77 | FG on hand (disclosure) | P | N | N | N | P | N | N | P | P | · | P | P | · | · |
| S-78 | Bulk-bin WIP (disclosure) | P | N | N | N | P | N | N | P | P | · | P | P | · | · |
| S-79 | Product catalog (form) | P | N | N | N | P | N | N | P | P | · | **P** | P | · | · |
| S-80 | Orders in / plans out | PA | **F** | N | P | P | N | N | P | N | · | P | P | · | · |
| S-81 | CSV import result | N | N | N | N | P | N | N | **P** | N | · | P | P | · | · |
| S-82 | "How this works" | N | N | N | N | P | N | N | P | P | · | P | P | · | · |
| S-83 | Production board table | P | N | N | N | P | N | N | P | P | · | PA | P | · | · |
| S-84 | Board cell copy / more | P | N | N | N | P | N | N | **P** | P | · | P | P | · | · |
| S-85 | Schedule panel | P | N | N | N | P | N | N | **P** | N | · | P | P | · | · |
| S-86 | Pin modal | P | N | N | N | P | N | N | **P** | P | · | P | P | · | · |
| S-87 | Order book | P | N | N | N | P | N | N | P | P | · | P | P | · | · |
| S-88 | Add order line form | P | N | N | N | P | N | N | PA | N | · | P | P | · | · |
| S-89 | Add-order validation alert | N | N | N | N | **F** | N | N | PA | N | · | P | P | · | · |
| S-90 | Order book empty state | N | N | N | N | N | N | N | P | N | · | P | P | · | · |
| S-91 | Scheduler print view | N | N | N | N | N | N | N | P | N | · | **F** | P | · | · |

## Matrix B — Errors, Confirmation & Recovery (ERROR-001…010)

| # | Screen | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-01 | Site navigation bar | N | N | N | N | N | N | N | N | N | N |
| S-02 | Mini-calendar strip | N | P | N | N | N | N | N | N | N | P |
| S-03 | App header | N | P | N | P | N | N | N | N | N | P |
| S-04 | Global search + results | N | P | N | P | N | N | N | N | N | PA |
| S-05 | Tab bar (7 tabs) | N | **F** | N | N | N | N | N | N | N | N |
| S-06 | Today So Far tile | N | P | N | P | N | N | N | N | N | P |
| S-07 | Today So Far — error/retry | N | P | **P** | P | N | N | N | N | N | P |
| S-08 | Production Calendar | N | **F** | N | P | N | N | N | N | N | P |
| S-09 | Calendar day detail panel | N | P | N | P | N | N | N | N | N | P |
| S-10 | Finished Goods panels | N | **F** | N | P | N | N | N | N | N | PA |
| S-11 | FG per-product lot rows | N | **F** | N | P | N | N | N | N | N | PA |
| S-12 | Batch Inventory | N | **F** | N | P | N | N | N | N | N | PA |
| S-13 | On-Hand Ingredients | N | **F** | N | P | N | N | N | N | N | PA |
| S-14 | Recent Entries feed | N | P | N | P | N | N | N | N | N | P |
| S-15 | Recent Entries states | N | P | **P** | P | N | N | N | N | N | P |
| S-16 | Daily Entries + toolbar | N | **F** | N | P | N | N | N | N | N | P |
| S-17 | Shipping log | N | **F** | N | P | N | N | N | N | N | PA |
| S-18 | Receiving log | N | **F** | N | P | N | N | N | N | N | PA |
| S-19 | Notes toolbar | N | P | N | P | N | N | N | N | N | P |
| S-20 | Notes list (cards) | N | **F** | **F** | P | P | N | N | N | N | P |
| S-21 | Notes empty state | N | N | N | N | N | N | N | N | N | P |
| S-22 | Note create/edit modal | N | **F** | PA | **F** | N | N | PA | N | N | **F** |
| S-23 | Delete-note confirm | N | N | **F** | P | **P** | **F** | P | P | P | **F** |
| S-24 | Orders toolbar | **F** | P | N | P | N | N | N | N | N | P |
| S-25 | Orders list table | N | **F** | PA | P | N | N | N | N | N | PA |
| S-26 | Dispatch Queue mode | N | **F** | PA | P | N | N | N | N | N | PA |
| S-27 | Orders expandable lines | N | **F** | PA | P | N | N | N | N | N | PA |
| S-28 | Factory Ready toggle+note | N | **F** | P | P | PA | N | N | N | N | P |
| S-29 | Orders empty state | N | N | N | N | N | N | N | N | N | P |
| S-30 | Order Detail header/KPI | N | **F** | N | P | N | N | N | N | N | PA |
| S-31 | Order Detail line table | N | **F** | N | P | N | N | N | N | N | P |
| S-32 | Per-line inventory expander | N | P | N | P | N | N | N | N | N | P |
| S-33 | Order Detail edit mode | **PA** | **F** | PA | P | N | N | PA | N | N | P |
| S-34 | Edit-locked notice | N | N | N | N | N | N | N | N | N | P |
| S-35 | SO status change confirm | N | N | P | P | PA | **P** | P | P | P | **F** |
| S-36 | Reservations / allocation | N | **F** | PA | P | N | N | PA | N | N | P |
| S-37 | Allocation history table | N | P | **F** | P | N | N | N | N | N | P |
| S-38 | Release reservation confirm | N | N | **F** | P | **P** | **P** | P | P | P | **F** |
| S-39 | Shipping capacity preview | N | P | N | P | N | N | N | N | N | P |
| S-40 | Order Detail notes card | N | **F** | N | P | N | N | N | N | N | P |
| S-41 | Expected Receipts toolbar | N | P | N | P | N | N | N | N | N | P |
| S-42 | Expected Receipts table | N | **F** | **F** | P | N | N | N | N | N | P |
| S-43 | ER row actions | N | P | **F** | P | **P** | PA | PA | P | P | P |
| S-44 | ER empty state | N | N | N | N | N | N | N | N | N | P |
| S-45 | ER create/edit modal | N | **F** | PA | **P** | N | N | PA | N | N | P |
| S-46 | Supplies header + Request | N | P | N | P | N | N | N | N | N | P |
| S-47 | Supplies sub-tabs | N | P | N | P | N | N | N | N | N | P |
| S-48 | Supplies search field | N | P | N | P | N | N | N | N | N | P |
| S-49 | Supplies inventory table | N | **F** | N | P | N | N | N | N | N | P |
| S-50 | Supply lot / incoming detail | N | P | N | P | N | N | N | N | N | P |
| S-51 | Supply Requests list | N | **F** | **F** | P | N | N | N | N | N | P |
| S-52 | Supply Requests feedback | N | P | N | **P** | N | N | N | N | N | P |
| S-53 | Request Supply modal | N | **F** | PA | **P** | N | N | PA | N | N | P |
| S-54 | Lot Detail side panel | N | P | N | P | N | N | N | N | N | PA |
| S-55 | Lot disambiguation | N | P | N | P | N | PA | N | **F** | P | PA |
| S-56 | Product Detail panel | N | P | N | P | N | N | N | N | N | PA |
| S-57 | Sankey controls bar | N | **F** | N | P | N | · | N | N | · | P |
| S-58 | Sankey chart + legend | N | **F** | N | P | N | · | N | N | · | P |
| S-59 | Sankey banner / loading | N | **F** | N | P | N | · | N | N | · | P |
| S-60 | Summary strip | N | **P** | N | P | N | · | N | N | · | P |
| S-61 | Production lines grid | N | **P** | N | P | N | · | N | N | · | P |
| S-62 | Error / stale banners | N | **P** | N | P | N | · | N | N | · | P |
| S-63 | Lot search + type-ahead | N | P | N | P | N | N | N | N | N | P |
| S-64 | Trace direction + Trace | **F** | P | N | P | N | N | N | N | N | P |
| S-65 | Recent lots strip | N | P | N | P | N | N | N | N | N | P |
| S-66 | Trace legend | N | N | N | N | N | N | N | N | N | P |
| S-67 | Status bar | N | P | N | P | N | N | N | N | PA | **P** |
| S-68 | Trace graph + zoom | N | P | N | P | N | N | N | N | N | P |
| S-69 | Node tooltip | N | N | N | N | N | N | N | N | N | **F** |
| S-70 | Trace detail + exports | N | P | N | P | N | N | N | N | N | **P** |
| S-71 | Print audit report | N | N | N | N | N | · | N | N | · | P |
| S-72 | Topbar KPI strip | N | **P** | N | N | N | · | N | N | · | P |
| S-73 | Topbar actions | N | **P** | N | P | N | · | N | N | · | **P** |
| S-74 | Legend bar | N | N | N | N | N | · | N | N | · | P |
| S-75 | Delta panel | N | **P** | **P** | P | N | · | N | N | · | P |
| S-76 | Settings panel (form) | N | **P** | PA | P | N | · | N | N | · | P |
| S-77 | FG on hand (disclosure) | N | **P** | PA | P | N | · | N | N | · | P |
| S-78 | Bulk-bin WIP (disclosure) | N | **P** | PA | P | N | · | N | N | · | P |
| S-79 | Product catalog (form) | N | **P** | PA | P | N | · | N | N | · | P |
| S-80 | Orders in / plans out | N | **P** | **F** | P | **P** | **P** | P | P | · | P |
| S-81 | CSV import result | N | **P** | N | P | N | · | N | N | · | P |
| S-82 | "How this works" | N | N | N | N | N | · | N | N | · | P |
| S-83 | Production board table | N | **P** | P | P | N | · | N | N | · | P |
| S-84 | Board cell copy / more | N | **P** | N | P | N | · | N | N | · | **P** |
| S-85 | Schedule panel | N | **P** | N | P | N | · | N | N | · | **P** |
| S-86 | Pin modal | N | **P** | **P** | P | P | · | PA | **P** | P | P |
| S-87 | Order book | N | **P** | **F** | P | **F** | · | **F** | N | · | P |
| S-88 | Add order line form | N | **F** | PA | **F** | N | · | N | N | · | P |
| S-89 | Add-order validation alert | N | N | N | **F** | N | **F** | N | N | N | **F** |
| S-90 | Order book empty state | N | N | N | N | N | · | N | N | · | P |
| S-91 | Scheduler print view | N | N | N | N | N | · | N | N | · | P |

## Matrix C — Notifications & Attention (applicable rules only)

NOTIFY-001, 002, 003, 006, 007, 008, 009 and 012 are **N** on all 91 screens — see the N/A register.

| # | Screen | 004 | 005 | 010 | 011 |
|---|---|---|---|---|---|
| S-01 | Site navigation bar | N | N | N | **F** |
| S-02 | Mini-calendar strip | N | P | PA | PA |
| S-03 | App header | **P** | N | N | **F** |
| S-04 | Global search + results | **P** | N | N | N |
| S-05 | Tab bar (7 tabs) | N | N | **F** | **F** |
| S-06 | Today So Far tile | **P** | N | N | **F** |
| S-07 | Today So Far — error/retry | **P** | N | N | P |
| S-08 | Production Calendar | **P** | N | N | N |
| S-09 | Calendar day detail panel | **P** | N | N | N |
| S-10 | Finished Goods panels | **P** | N | P | N |
| S-11 | FG per-product lot rows | **P** | N | N | N |
| S-12 | Batch Inventory | **P** | N | P | N |
| S-13 | On-Hand Ingredients | **P** | N | P | N |
| S-14 | Recent Entries feed | **P** | **F** | N | PA |
| S-15 | Recent Entries states | **P** | N | N | P |
| S-16 | Daily Entries + toolbar | **P** | N | N | N |
| S-17 | Shipping log | **P** | N | N | N |
| S-18 | Receiving log | **P** | N | N | N |
| S-19 | Notes toolbar | **P** | N | **F** | **F** |
| S-20 | Notes list (cards) | **P** | N | N | **F** |
| S-21 | Notes empty state | N | N | N | N |
| S-22 | Note create/edit modal | **F** | N | N | N |
| S-23 | Delete-note confirm | P | N | N | N |
| S-24 | Orders toolbar | **P** | N | **F** | **F** |
| S-25 | Orders list table | **P** | **F** | N | **F** |
| S-26 | Dispatch Queue mode | **P** | **F** | **F** | **F** |
| S-27 | Orders expandable lines | **P** | N | N | N |
| S-28 | Factory Ready toggle+note | **P** | **F** | N | N |
| S-29 | Orders empty state | N | N | N | N |
| S-30 | Order Detail header/KPI | **P** | N | N | N |
| S-31 | Order Detail line table | **P** | N | N | N |
| S-32 | Per-line inventory expander | **P** | N | N | N |
| S-33 | Order Detail edit mode | **P** | N | N | N |
| S-34 | Edit-locked notice | N | N | N | N |
| S-35 | SO status change confirm | P | N | N | N |
| S-36 | Reservations / allocation | **P** | N | N | N |
| S-37 | Allocation history table | **P** | **F** | N | N |
| S-38 | Release reservation confirm | P | N | N | N |
| S-39 | Shipping capacity preview | **P** | N | N | N |
| S-40 | Order Detail notes card | N | N | N | N |
| S-41 | Expected Receipts toolbar | **P** | N | PA | **F** |
| S-42 | Expected Receipts table | **P** | **F** | N | **F** |
| S-43 | ER row actions | **P** | N | N | N |
| S-44 | ER empty state | N | N | N | N |
| S-45 | ER create/edit modal | **P** | N | N | N |
| S-46 | Supplies header + Request | N | N | N | **F** |
| S-47 | Supplies sub-tabs | **P** | N | PA | **F** |
| S-48 | Supplies search field | N | N | N | N |
| S-49 | Supplies inventory table | **P** | **F** | N | **F** |
| S-50 | Supply lot / incoming detail | **P** | N | N | N |
| S-51 | Supply Requests list | **P** | **F** | **P** | **F** |
| S-52 | Supply Requests feedback | **P** | P | N | N |
| S-53 | Request Supply modal | **P** | N | N | N |
| S-54 | Lot Detail side panel | **P** | N | N | N |
| S-55 | Lot disambiguation | **P** | N | N | N |
| S-56 | Product Detail panel | **P** | N | N | N |
| S-57 | Sankey controls bar | **P** | N | N | N |
| S-58 | Sankey chart + legend | **P** | N | N | N |
| S-59 | Sankey banner / loading | **P** | N | N | P |
| S-60 | Summary strip | **P** | P | N | P |
| S-61 | Production lines grid | **P** | **F** | N | P |
| S-62 | Error / stale banners | **P** | P | N | **P** |
| S-63 | Lot search + type-ahead | **P** | N | N | N |
| S-64 | Trace direction + Trace | **P** | N | N | N |
| S-65 | Recent lots strip | **P** | N | N | N |
| S-66 | Trace legend | N | N | N | N |
| S-67 | Status bar | **P** | N | N | P |
| S-68 | Trace graph + zoom | **P** | N | N | N |
| S-69 | Node tooltip | N | N | N | N |
| S-70 | Trace detail + exports | **P** | N | N | N |
| S-71 | Print audit report | N | N | N | N |
| S-72 | Topbar KPI strip | N | N | N | **P** |
| S-73 | Topbar actions | N | N | N | N |
| S-74 | Legend bar | N | N | N | N |
| S-75 | Delta panel | N | P | N | P |
| S-76 | Settings panel (form) | N | N | N | N |
| S-77 | FG on hand (disclosure) | N | N | N | N |
| S-78 | Bulk-bin WIP (disclosure) | N | N | N | N |
| S-79 | Product catalog (form) | N | N | N | N |
| S-80 | Orders in / plans out | N | N | N | N |
| S-81 | CSV import result | **P** | P | N | P |
| S-82 | "How this works" | N | N | N | N |
| S-83 | Production board table | N | P | N | P |
| S-84 | Board cell copy / more | N | N | N | N |
| S-85 | Schedule panel | N | N | N | N |
| S-86 | Pin modal | N | N | N | N |
| S-87 | Order book | N | P | **P** | P |
| S-88 | Add order line form | N | N | N | N |
| S-89 | Add-order validation alert | **F** | N | N | N |
| S-90 | Order book empty state | N | N | N | N |
| S-91 | Scheduler print view | N | N | N | N |

---

## Findings

---

### FEEDBACK-003 — Progress indicators keep moving; a stall is explained with what to do next — **CRITICAL**

**FAIL on every network-bound screen in the product.** *Hard rule.*

**Evidence:** No fetch anywhere in `dashboard/` has a timeout, an `AbortController`, or a `signal`. Every network call is a bare `fetch()`:
- `fetchAPI` — `dashboard/dashboard.js:484-491`
- `fetchSalesAPI` — `dashboard/dashboard.js:1914-1925`
- `openLotPanel`'s direct fetch — `dashboard.js:1354`
- `refreshHealthBadge` — `dashboard.js:4114`
- `exportOrdersMatrix` — `dashboard.js:2199`
- `api()` — `dashboard/traceability.html:351-361`
- `fetchTransactions` — `dashboard/sankey.html`
- `apiFetch` — `dashboard/process-flow.html`

If the Railway API accepts the connection and then hangs — the common failure mode for a cold container or a saturated pool — every loading state in the product waits forever. The audit test is *"After N seconds without progress, does the UI change to explain the stall and offer a next step?"* There is no N anywhere.

**Compounding this: the dashboard's indicator does not move.** `.loading-indicator` is static centred text (`dashboard/dashboard.css:944-950`) — "Loading sales orders...". The rule opens with *"A visibly stationary indicator is treated by users as a freeze."* The only animated indicators in the product are `traceability.html:176-181` (`.loading-spinner` with `@keyframes spin`) and `sankey.html` (`.loading-overlay .spinner`). There is exactly one `@keyframes` declaration in the entire product.

**S-60, S-61, S-62 (Process Flow) — PASS. The only stall handling that exists.**
`dashboard/process-flow.html` tracks `consecutiveFailures`, calls `showStale()` after three, and renders *"Data may be stale — last updated 3:42 PM"* with the real last-success time. It also declines to blank the view (`else { return; // Keep showing last good data }`). Every other surface should adopt this.

**S-07, S-15 — PARTIAL.** Both offer a Retry control on the error path (`dashboard.js:296-297`, `618-619`) ✓ — the "next step" half of the rule — but neither has a timeout, so a hang never reaches the error path.

**Suggested fix:** Wrap `fetchAPI` and `fetchSalesAPI` in an `AbortController` with a ~10s timeout that rejects with a distinguishable error, and render the stall as *"Server not responding — showing data from 3:42 PM. [Retry]"* over the last-good view rather than replacing it (see ERROR-002).

---

### ERROR-002 — Preserve in-progress work across app switches, refreshes, rotation, and interruptions — **CRITICAL**

*Hard rule.* Two distinct failures, both app-wide.

#### 1. No form draft is persisted anywhere, and every exit path discards silently

**Evidence:** `localStorage` is used for exactly one value in the dashboard — the theme (`dashboard.js:46`, `56`). `sessionStorage` for exactly one — which collapsible panels are expanded (`dashboard.js:12`, `449`). No draft state exists for any form.

All three custom modals discard everything on **three** exit paths, with no warning and no confirmation:

| Modal | Close paths | Evidence |
|---|---|---|
| Note (S-22) | X button, Cancel, **backdrop click** | `dashboard.js:1804-1807`, `1869-1873` |
| Expected Receipt (S-45) | X button, Cancel, **backdrop click** | `dashboard.js:3573-3576`, `3638-3642` |
| Request Supply (S-53) | X button, Cancel, backdrop click, **Escape** | `dashboard.js:4005-4008`, `4098-4107` |

A stray click on the backdrop while writing an expected receipt destroys it. Arturo pressing Escape mid-supply-request loses it. The standard's own worked example for this situation is under ERROR-004: *"Leaving a half-filled receiving form → sheet: 'Save Draft / Discard / Cancel.'"* None of the three asks anything.

**S-33 (Order Detail edit) — FAIL.** `dashboard.js:3308-3319` (`closeOrderDetail` hides the view without checking for unsaved edits), `2850-2856` (the "Done" button re-renders, discarding), `1582-1592` (a global-search order click replaces the container outright). A browser refresh loses everything; the rule's example is *"a desktop SO edit survives an accidental refresh."*

**S-28 (Factory Ready note) — FAIL.** `dashboard.js:2472`, `2441` — typed note text is destroyed by any `renderOrdersList()`, including one triggered by a *different row's* checkbox. See NAV-005 in `01-nav-layout.md`.

#### 2. A failed background refresh destroys the last-good view

**Evidence — thirteen call sites all doing `container.innerHTML = ''` in their `catch` before showing an error:**
`dashboard.js:700` (production calendar), `899` (finished goods), `969` (batch inventory), `1085` (ingredients), `1180` (shipments), `1233` (receipts), `1298` (daily entries), `1671` (notes), `2271` (orders), `2547` (order detail), `3380` (expected receipts), `3724` (supplies inventory), `3915` (supply requests).

A transient network blip while Luz is reading the orders table blanks it and replaces it with a red bar. Nothing about the failure required destroying the data she was already looking at. This is the failure the rule calls *"the most expensive friction."*

**Related contradictory state:** `hideError('orders-error')` is called only at the top of `refreshOrders` (`2238`), never in `renderOrdersList`. So after a failed refresh, changing any filter re-renders a full, correct table *beneath* a red "Failed to load sales orders" banner (`dashboard.js:2300-2337` does not clear it). The screen asserts two contradictory things at once.

**S-72…S-87 (Scheduler) — PASS. The reference implementation.**
`dashboard/scheduler/seven-wells-production-board.html:376-378` — `saveState()` / `loadState()` persist the entire plan (orders, pins, on-hand, WIP, catalog overrides, settings, baseline) to `localStorage` and restore it on load. A browser refresh, a crash, or closing the tab loses nothing. The brief allows the scheduler to be N/A on persistence rules; this one applies and passes outright.

**S-60, S-61, S-62 (Process Flow) — PASS.** Keeps the last good render on failure rather than blanking.

**S-57, S-58, S-59 (Sankey) — FAIL.** Worse than blanking: on error it **substitutes fabricated sample data** (`sankey.html:824-828`) behind a small warning banner. Full treatment under CHART-004 / DATA in `04-search-data-drag-chart.md`.

**Suggested fix:** (a) Add a `beforeunload`/close guard and a Save Draft / Discard / Cancel sheet to the three modals and to Order Detail edit mode; (b) mirror form state into `sessionStorage` on input, keyed by record id; (c) change every `catch` to leave the container's existing content in place and render the error above it.

---

### ERROR-003 — Build forgiveness in: reversible actions, an equally easy correction path, no re-entry — **CRITICAL**

**FAIL, app-wide.** *Hard rule.*

**There is no undo anywhere in the product.** No toast with an Undo affordance, no action history, no return-to-previous-state (verified by grep — the only "undo"-adjacent mechanism is the scheduler's baseline/delta).

**Four one-way doors have no path back through the UI at all:**

| Action | Evidence | Why it is one-way |
|---|---|---|
| Close an Expected Receipt | `dashboard.js:3458`, `3437` | `renderExpectedReceipts` emits row actions **only** `if (r.status === 'open')`; a closed receipt shows an empty action cell. There is no "reopen". |
| Cancel an Expected Receipt | `dashboard.js:3461`, `3437` | Same. |
| Mark a Supply Request Done | `dashboard.js:3943-3945` | The Done button is rendered **only** `if (request.status === 'open')`. There is no "reopen". |
| Delete a scheduler order line | `scheduler:1553-1554` | Immediate `ST.orders.filter(...)` with no confirmation and no recovery. |

The API may well support reopening — the dashboard simply never offers it. `erSetStatus` (`3465`) takes an arbitrary status and would accept `'open'`; nothing calls it with that value.

**Every correction requires re-entering data.** `openErModal(record)` prefills from the cached record ✓ (`3554-3557`), and `openNoteModal(note)` prefills ✓ (`1794-1799`) — so *edits* preserve the current value. But nothing preserves the **previous** value, so undoing an edit means remembering and retyping it. There is no pre-filled compensating entry anywhere; the rule's example — *"a mis-posted pack run has a one-tap 'Reverse this run' that creates a pre-filled compensating entry"* — has no equivalent.

**The ledger's own correction model is visible but not actionable.** `recentStatus()` (`dashboard.js:557-566`) renders `amend`, `void`, and `restore` correction events with dedicated badges, and `renderRecentEntries` shows the reason (`583`). So the dashboard can *display* a correction and cannot *create* one. Corrections must be made through the ChatGPT Floor GPT. For Luz, whose primary surface is this dashboard, every mistake is a context switch.

**S-20, S-23 — FAIL.** Note delete is permanent behind a `confirm()`; the note's content is not recoverable.
**S-37, S-38 — FAIL.** Releasing a reservation cannot be undone in one step; re-reserving the same lot and quantity means re-opening the form, re-selecting the line, the mode, and the lot, and re-typing the quantity (`dashboard.js:3045-3091`).
**S-80 — FAIL.** `scheduler:1321` — Reset wipes orders, pins, inventory, and settings from `localStorage` with no export-first prompt, behind a single `confirm()`.

**Reversible and correct:** the Factory Ready toggle (untick, `dashboard.js:2455-2486`, with a proper optimistic rollback on failure at `2482`) ✓; the note done toggle ✓; SO status (any transition is offered, so it can be changed back) ✓.

**S-75 (Scheduler baseline/delta) — PASS. The one genuine forgiveness mechanism in the product.** `scheduler:11` (Set as baseline), `:948` (Clear baseline), `:25` (delta panel). Blubber can snapshot a plan, change anything, see exactly what got better or worse, and revert by reloading the baseline. Every planning change is cheap. This is what the rule is asking for, implemented once.

**Suggested fix:** (a) Add reopen paths for closed/cancelled expected receipts and completed supply requests — a one-line change to the two `if (status === 'open')` guards; (b) add an Undo affordance to the note delete and allocation release using the values already in `state`; (c) add a Reverse action on Recent Entries rows that opens a pre-filled correction.

---

### FEEDBACK-001 — Progress within ~1s; commit buttons non-repeatable until the server responds — **CRITICAL**

*Hard rule.* Loading states are broadly present and well-labelled; the **non-repeatable** clause is where this fails.

**Four commit paths are never disabled and can be fired twice:**

| Commit | State during flight | Evidence |
|---|---|---|
| **Save a note** (S-22) | `#note-save-btn` is never touched | `dashboard.js:1809-1847`, `1876` |
| **Toggle a note done** (S-20) | checkbox stays live; handler refetches | `dashboard.js:1741-1753` |
| **Toggle Factory Ready** (S-28) | checkbox stays live | `dashboard.js:2455-2486` |
| **Close / Cancel an ER** (S-43) | button stays live after arming | `dashboard.js:3465-3495` |

The consequence is not theoretical: with **no `:active` press state anywhere** (ACTION-002), the user gets zero confirmation that the first click registered, and the rule names the outcome exactly — *"nothing visibly changing invites a second tap and a duplicate submission."* Two rapid clicks on Save in the note modal creates two notes.

**Correctly guarded, with the in-progress label the rule asks for — cite these:**
- `submitSupplyRequest` — disables and sets "Submitting..." (`dashboard.js:4058-4060`), restores in `finally` (`4073-4074`) ✓
- `markSupplyRequestDone` — disables and sets "Saving..." (`3957-3958`) ✓
- `exportOrdersCsv` / `exportOrdersMatrix` — disable, set "Exporting...", `.loading` class, restore in `finally` (`2144-2146`/`2184-2186`, `2193-2195`/`2223-2225`) ✓
- `refreshAll` — `.loading` + "Refreshing..." (`4140-4141`, `4172-4173`) ✓

**Guarded but with no label change — PARTIAL:** `saveEr` (`3585-3586`, `3623-3625`), `submitAllocation` (`3069-3070`), `releaseAllocation` (`3097`), `saveOrderHeader` (`2729-2730`, `2743`), `saveOrderLines` (`2788-2789`, `2809`), `saveOrderStatus` (`2824`, `2838`), `bindOrderReadyNoteControls` (`2437`, `2446`), `previewOrderShipping` (`3137`, `3149`), `refreshRecentEntries` (`608-609`, `621-622`).

**Loading indicators suppressed on refresh — PARTIAL (S-41, S-42, S-49, S-51):** three refresh functions show a loading state **only on first load**, guarded by a `*Loaded` flag: `refreshExpectedReceipts` (`3371`), `refreshSuppliesInventory` (`3696-3698`), `refreshSupplyRequests` (`3904-3906`). Every subsequent refresh — filter change, Refresh button, post-save — runs silently. On a slow connection the user presses Refresh and nothing happens for several seconds.

**No progress at all — PARTIAL (S-04, S-48, S-63):** `performSearch` (`1507-1520`) and `searchErProducts` (`3516-3540`) show nothing while the request is in flight; the dropdown simply keeps its previous content or stays hidden.

**Best in class:** `populateAllocationLots` (`dashboard.js:3008-3009`) sets the select to *"Loading lots…"* and disables it, then restores in `finally` (`3026-3028`). Compact, in place, non-blocking, and impossible to double-fire.

---

### FEEDBACK-007 — Refresh automatically without shifting content; offer manual refresh; show last-updated where staleness matters — **HIGH**

**FAIL, app-wide, on both the auto-refresh and the last-updated clauses.**

#### Only two surfaces auto-refresh, and both violate LAYOUT-020 doing it

**Evidence:** `dashboard.js:626-631` (`startRecentEntriesPolling` — 60s) and `process-flow.html` (`setInterval(refresh, 60000)`). Nothing else in the product refreshes on its own. Sales Orders, Expected Receipts, Supplies, Supply Requests, Notes, and all four inventory tables are fetched once when the page loads and change only when a human presses a Refresh button.

The rule's Factory Ledger clause is explicit: *"SO lists, Expected Receipts, Today So Far, and the production board stay in sync between Luz and Arturo."* They do not. Luz can leave the Sales Orders tab open all afternoon while Arturo posts shipments through the Floor GPT, and see none of it. In a multi-user ledger this produces exactly the duplicate and conflicting transactions the rule exists to prevent.

And the two that *do* refresh both replace `innerHTML` wholesale (`dashboard.js:568-600`; `process-flow.html` `renderDashboard`) rather than buffering — the failure detailed under LAYOUT-020 in `01-nav-layout.md`.

#### The last-updated timestamp is inaccurate

**Evidence:** `dashboard.js:4163-4171` — `#last-refreshed` is set **only** at the end of `refreshAll`, after `await Promise.allSettled(ops)`.

Two consequences:
1. **It ignores failures.** `allSettled` resolves whether or not the thirteen operations succeeded. If nine of thirteen rejected, the header still reads *"Updated: 3:42:15 PM ET"* as though everything is fresh.
2. **Section refreshes never update it.** `refreshOrders` (`2237`), `refreshExpectedReceipts` (`3368`), `refreshSuppliesInventory` (`3693`), `refreshSupplyRequests` (`3901`), and `refreshRecentEntries` (`602`) all fetch new data and leave the header timestamp alone. Press the Orders Refresh button at 4:15 and the header still says 3:42.

The audit test is *"Is last-sync time visible where staleness could cause a wrong transaction?"* It is visible and it is wrong, which is worse than absent.

**S-60, S-61, S-62 (Process Flow) — PASS.** Sets `#last-updated` from the actual render time, tracks `lastSuccessTime` separately, and shows a stale banner naming that time after three consecutive failures. The correct model.

**S-57, S-58, S-59 (Sankey) — PARTIAL.** Sets `#last-updated` at `sankey.html:822-823` — but inside the success path only, so a fallback-to-sample-data render leaves the previous timestamp standing beside fabricated numbers.

**S-72…S-91 (Scheduler) — N/A, "no API; state is local to the browser."**

**Manual refresh — PASS.** Five refresh controls exist (`index.html:41`, `118`, `235`, `275`, `327`) ✓, and the rule's label clause is satisfied: the buttons say "Refresh" (the instruction) and the value lives in a separate `#last-refreshed` span ✓.

**Suggested fix:** Poll the operational lists on a visible-tab interval into a buffer, surface *"3 new — tap to refresh"* rather than re-rendering, and set the timestamp per section from the actual response time, marking any section whose last fetch failed.

---

### FEEDBACK-011 — Status, state, and interactivity are never communicated by color alone — **CRITICAL**

*Hard rule.* The audit test: *"If this screen were viewed in grayscale, would every status, state, and interactive element still be identifiable?"*

**Most of the product passes.** Status badges carry text (`.so-badge` at `dashboard.js:2326`, `.dispatch-pill` at `1950-1952`, `.er-badge-*` at `3397-3405`, `.supply-low-badge` "Low Stock" at `3852`, `.note-priority-badge` "High"/"Low" at `1705-1706`, `.allocation-status` at `2951`). The health badge carries its numeric score (`4118`). Late entries carry the words *"entered next day 4:12 PM"* (`1303-1307`). Factory Ready rows carry a "✓ READY" pill (`2229-2235`). Traceability is the strongest surface in the product: node type uses fill **plus** a dashed border for gaps **plus** a ⚠ glyph (`traceability.html:1096`, `1099-1106`); link confidence uses colour **plus** ✓/⚠ glyphs **plus** a dash pattern (`1042`, `1057-1074`); the completeness badge uses ✓/⚠/✗ **plus** full text (`1235-1248`).

**Five cues fail — colour (or colour plus weight) is the only carrier:**

| Cue | Evidence | What survives grayscale |
|---|---|---|
| **Overdue ship date** (S-25, S-26, S-42) | `dashboard.css:1671` — `.date-overdue { color: var(--danger); font-weight: 600 }`; applied at `dashboard.js:2325`, `3434` | 13px semibold vs 13px regular. No icon, no word, no "OVERDUE". |
| **Overdue weekday line** (S-25, S-42) | `dashboard.css:1540-1542` — `.date-overdue .ship-by-weekday { color: color-mix(danger 72%) }` at 11px weight 500 | Nothing. |
| **Line shortage** (S-31) | `dashboard.css:1819` — `.readiness-shortage { color: var(--danger) !important; font-weight: 700 }` on a number; applied at `dashboard.js:3276` | Bold vs regular on a numeric cell in a nine-column table. |
| **Note due-date overdue** (S-20) | `dashboard.css:1093` — `.note-due.overdue { color: var(--danger) }` at 11px regular (`1082-1090`); applied at `dashboard.js:1716-1717` | Nothing. |
| **Product family** (S-06, S-08, S-09, S-12) | `dashboard.css:372-374`, `505-507`, `582-584`, `670-672` — `--category-coconut/granola/graham` applied to the product **name text** | Nothing. A granola row and a coconut row are identical in grayscale. |

**S-75 (Scheduler delta panel) — FAIL.** `scheduler:50` — `.d-better { color: var(--ok); font-weight: 700 }`, `.d-worse { color: var(--late); font-weight: 700 }`, `.d-same { color: var(--idle) }`. Better and worse are both bold and differ only in hue. This is the panel Blubber reads to decide whether a plan change helped.

**S-71, S-91 (print views) — FAIL.** `traceability.html:240-246` and `scheduler:185-208` both print. Print is the grayscale case the rule names explicitly, and the five cues above lose their only carrier on paper. The traceability print view keeps its ✓/⚠ glyphs ✓ but forces `.detail-card { background: #fff; color: #000 }` (`243`), flattening the coloured `.confirmed-tag` / `.legacy-tag` to near-black — leaving only the glyph, which is enough ✓. The scheduler print view keeps the board's colour-coded cells with no glyph substitution.

**Interactivity in grayscale — PARTIAL.** `.lot-link` is underlined ✓ (`css:842`). `.order-link` is **not** underlined at rest, only on hover (`css:1549`, `1553`) — so in grayscale a clickable SO number is indistinguishable from the customer name beside it. `.tab.active` uses colour **plus** a 2px bottom border ✓. `.supplies-subtab.active` uses background **plus** border ✓. `.notes-filter-btn.active` uses a solid fill ✓.

**Suggested fix:** Add a word or glyph to each of the five: `⚠ 3d overdue` instead of a red date; `Short 40 lb` instead of a red number; a leading family initial or a small shape beside the product name. Add `text-decoration: underline` to `.order-link` at rest.

---

### FEEDBACK-012 — One meaning per color, app-wide; the interactive color only on interactive things; status colors immune to themes — **CRITICAL**

*Hard rule.* Four distinct violations, all verifiable from the token table at `dashboard/dashboard.css:6-75`.

#### 1. Amber carries two unrelated meanings — and the hex values are identical

`--category-granola: #fbbf24` (`css:19`) is **byte-identical** to `--badge-amber-text: #fbbf24` (`css:28`).

Amber-as-attention is used for: `--warning` (`22`), `.er-badge-overdue` (`2225`), `.er-overdue` row tint (`2222-2223`), `.supply-low-badge` (`2356`), `.supply-item-low` row tint (`2354-2355`), `.late-entry` row tint (`806-809`), `.readiness-chip.severity-warn` (`1639-1643`), `.allocation-expiry` (`2080`), `.allocation-source.source-auto_fifo` (`2078`), `.supplies-count-badge` (`2314-2315`), `.badge.unknown` (`726`), `.recent-entry-lag` (`312`), `.preview-allocation-warning` (`2104-2112`), and `.er-armed` (`2230`).

Amber-as-granola is used for: `.today-tile-row.family-granola > span` (`373`), `.production-category-label.category-granola` (`506`), `.production-detail-family.family-granola > h4` (`583`), `.batch-family-heading.family-granola` (`671`).

On the Today So Far tile the word **"Granola"** is rendered in exactly the colour that means "overdue" everywhere else in the app.

#### 2. The interactive colour is used on non-interactive text

`--category-coconut: #60a5fa` (`css:17`) is **byte-identical** to `--primary-hover: #60a5fa` (`css:16`) — the app's hover/interactive blue.

So a coconut **product name** (`css:372`, `505`, `582`, `670`) renders in the interactive colour while being plain text. Also non-interactive: `.day-card-detail-hint` (`523`) and `.production-detail-eyebrow` (`550`), both `--primary-hover`. The rule's own example is *"If the accent blue means link/button, non-interactive lot codes aren't blue."*

#### 3. Green carries three order states plus a transaction type

`.so-badge.status-ready` (`#064e3b`/`#5eead4`, `1570`), `.so-badge.status-shipped` (`--badge-green-*`, `1572`), and `.so-badge.status-invoiced` (`#064e3b`/`#6ee7b7`, `1573`) are three near-identical greens for three different states. `--accent: #10b981` additionally means "receive" in the lot timeline (`.timeline li.txn-receive::before`, `926`) and "success" in `.allocation-feedback.success` (`2016-2021`).

#### 4. Status colours are **not** theme-independent — nine are hard-coded for dark mode only

The `--badge-*` tokens are correctly redefined per theme (`css:23-30` dark, `61-68` light) ✓. But nine status colours bypass the token system entirely and therefore render dark-mode values on the light theme's white surfaces:

| Selector | Hard-coded value | Evidence | Light-theme result |
|---|---|---|---|
| `.so-badge.status-ready` | `bg #064e3b` / `color #5eead4` | `css:1570` | very dark green on white surface — legible but inconsistent |
| `.so-badge.status-partial_ship` | `bg #7c2d12` / `color #fdba74` | `css:1571` | dark brown chip |
| `.so-badge.status-invoiced` | `bg #064e3b` / `color #6ee7b7` | `css:1573` | dark green chip |
| **`.so-ready-pill`** | `color #86efac` on `rgba(34,197,94,0.12)` | `css:1576-1588` | **pale green on near-white ≈ 1.4:1 — effectively invisible** |
| **`.order-edit-message.success`** | `color #86efac` on `rgba(52,211,153,0.10)` | `css:1885-1890` | **same — the "Header saved." confirmation disappears** |
| `.order-edit-message.error` | `color #fca5a5` on `rgba(248,113,113,0.10)` | `css:1891-1898` | pale red on near-white — very low contrast |
| `.order-ready-checkbox` | `accent-color: #22c55e` | `css:1341` | fixed green regardless of theme |
| `.order-row.so-ready td:first-child` | `border-left: 3px solid #22c55e` | `css:1321` | fixed green |
| `.mini-calendar-day.is-today.has-shipments::after` | `background: #93c5fd` | `mini-calendar.css:111` | pale blue dot on a light surface |

The two marked in bold are the significant ones: in light mode, **the "✓ READY" pill and the "Header saved." success message are both very close to invisible**. Both are confirmations of a write. Cross-referenced as ACCESS-008 contrast failures in `05-access-icon-other.md`.

**Suggested fix:** Give the product-family palette its own hue range that does not overlap the status palette (or drop colour-coding families in favour of a leading initial — see FEEDBACK-011), move `--category-coconut` off `--primary-hover`, and replace the nine hard-coded values with theme-aware tokens.

---

### FEEDBACK-008 — Every state change gets a clear signal — **HIGH**

*Hard rule.* Three of the rule's four clauses fail.

**Content changes are never signalled.** There is exactly one `@keyframes` declaration in the entire product (`traceability.html:181`, the loading spinner). No highlight, flash, pulse, or transition marks a data change anywhere. When a note is ticked done it simply vanishes (`dashboard.js:1749` → `refreshNotes` → full re-render). When Factory Ready is set, the row's opacity drops and a pill appears — inside a full table rebuild (`2472`). The rule's example is *"When a row's status changes to Allocated, the chip changes and briefly highlights."*

**Success is signalled inconsistently — and four writes signal nothing at all.**

| Write | Success signal | Evidence |
|---|---|---|
| Save order header/lines/status | `.order-edit-message.success` ✓ | `dashboard.js:2687-2693`, `2739` |
| Create/release allocation | `.allocation-feedback.success` ✓ | `2966-2972`, `3084`, `3106` |
| Create supply request | `.supply-feedback.success` ✓ | `3893-3899`, `4068` |
| Mark supply request done | `.supply-feedback.success` ✓ | `3965` |
| **Save a note** | **none** — the modal just closes | `1842-1843` |
| **Delete a note** | **none** | `1777` |
| **Save an expected receipt** | **none** — the modal just closes | `3608-3609` |
| **Toggle Factory Ready** | **none** beyond the row re-render | `2474-2480` |

Three different success components exist and four writes use none of them.

**Disabled state with a visible reason — PARTIAL.** Nothing in the product is ever disabled with an inline explanation. The two Dispatch-Queue disabled controls put their reason in a `title` (`dashboard.js:2321`, `2362`), and one of those is unreachable because `.btn-sm:disabled { pointer-events: none }` (`css:2424`) suppresses the tooltip — see INPUT-021 in `02-actions-input-touch.md`.

**S-34 — PASS, and it is the right answer.** `dashboard.js:2702-2704` renders *"Editing opens only while the order is New or Confirmed. Current status: Ready to Ship."* as plain prose rather than showing greyed controls. This is exactly what the rule asks for and it is used once.

**Standard patterns for choices — FAIL.** Four different confirmation patterns coexist (native `confirm`, native `alert`, the ER two-step arm, and nothing at all). See ACTION-006 in `02-actions-input-touch.md` and LAYOUT-002 item 4 in `01-nav-layout.md`.

**Stale error banners contradict fresh content — FAIL (S-25, S-26, S-42, S-49, S-51).** `hideError` is called only at the top of each `refresh*` function, never in the corresponding `render*`. After a failed refresh, any filter change re-renders a complete table beneath a red *"Failed to load sales orders"* banner (`dashboard.js:2238` vs `2300-2337`; `3369` vs `3407`; `3694` vs `3827`; `3902` vs `3920`).

**S-52, S-81, S-84, S-85, S-86 — PASS.** `showSupplyFeedback` (`3893-3899`) uses a `role="status"` region with an auto-dismiss for success only ✓. The scheduler's CSV import result names exactly what was accepted, what SKUs were unknown, and what rows were skipped (`scheduler:1079-1083`, `:1101`) ✓ — a model state-change message. `copyScheduleText` changes the button label to "Copied" or "Copy failed" and restores it after 1.4s (`scheduler:1086-1094`) ✓.

---

### FEEDBACK-009 — Selection feedback matches the purpose of the list — **HIGH**

**S-25, S-26 (Orders list → detail) — FAIL.** `dashboard.js:2531` — `listView.style.display = 'none'`. Opening an order replaces the list entirely, so there is no "currently open item" to keep highlighted. The rule's Factory Ledger example is the opposite: *"Desktop split view: the open SO row stays highlighted (dimmed when the panel has focus) while its detail shows."* Related to NAV-009.

**S-54, S-56 (Lot / Product panel) — FAIL.** `dashboard.js:1344-1370`, `1438-1504` — `openLotPanel` and `openProductPanel` touch nothing in the underlying list. The panel is a 440px overlay on the right (`css:861-869`) with the source table fully visible behind it, and the row the user came from is not marked. With the panel open there is no way to see which lot row was clicked.

**S-30 — FAIL.** No indication of position within the filtered list, and no next/prev (NAV-011).

**S-08, S-09 — PASS. Reference implementation.** `dashboard.js:854-859` — `.day-card-trigger.selected` gets `box-shadow: inset 0 0 0 2px var(--primary)` plus `background: var(--surface-hover)` and `z-index: 1` so the ring is not clipped by neighbours (`css:442-447`), **and** `aria-expanded` is kept in sync. The selection persists while the detail panel below is open. This is exactly the navigation-list behaviour the rule describes.

**S-47, S-49 — PASS.** `.supplies-subtab.active` with `aria-selected` (`dashboard.js:4082-4086`) ✓; supply rows keep `aria-expanded="true"` with a rotated caret and stay in place (`3849`, `css:2345`) ✓.

**S-64 — PASS.** `.dir-btn.active` (`traceability.html:496-500`) ✓.

**Choice lists — PASS throughout.** Notes filter `.active` (`dashboard.js:1854`), tab `.active` (`499`), trace direction, supplies sub-tabs — all mark the chosen item persistently ✓.

**S-27 — PARTIAL.** `.order-expand-toggle.expanded` rotates the caret ✓ (`css:1366`) but the parent row receives no selected treatment, so with three rows expanded the user cannot tell at a glance which rows are open without reading the carets.

**S-04, S-63 — PARTIAL.** Search dropdowns have `:hover` only (`css:202`; `traceability.html:106`), no keyboard selection state — consistent with their lack of keyboard support (INPUT-006).

---

### FEEDBACK-002, FEEDBACK-004, FEEDBACK-005, FEEDBACK-006, FEEDBACK-010, FEEDBACK-013, FEEDBACK-014

**FEEDBACK-002 — FAIL (S-24 export, S-33, S-39, S-64, S-68, S-80).** *High.* Every indicator in the product is indeterminate, including three operations whose total is known:
- `exportOrdersCsv` → `loadOrderDetails(orders)` fetches one detail per filtered order through six workers (`dashboard.js:2118-2138`), up to 200 requests, behind a static "Exporting..." label. The rule's own example is *"matrix exports: 'Processing lot 12 of 40.'"*
- `saveOrderLines` PATCHes each changed line sequentially (`2794-2800`) with `savedCount` already tracked (`2798`) — the count exists and is never shown.
- `traceForward` awaits `findShipmentsForLot` once per batch in a loop (`traceability.html:749-762`), up to eight sequential round-trips behind one static spinner.
- `refreshAll` runs thirteen operations (`4143-4163`) behind "Refreshing...".
**Shape stability — PASS.** No indicator ever changes shape mid-operation ✓.

**FEEDBACK-004 — PARTIAL, app-wide.** *Medium.* Nineteen loading strings follow the pattern *"Loading <section>..."* (`dashboard.js:289`, `691`, `894`, `964`, `1080`, `1175`, `1228`, `1291`, `1350`, `1444`, `1660`, `2240`, `2534`, `3371`, `3697`, `3905`, `3777`) — domain-named, so better than a bare "Loading", but they name the *screen region* rather than the operation. Two are the vague case the rule prohibits: `bindOrderInventoryToggles` renders *"Loading…"* (`2652`) and `populateAllocationLots` renders *"Loading lots…"* (`3009`).
**Best in class — PASS:** `traceability.html:510` — *"Tracing 26-0907-C forward..."* names the record and the operation, which is precisely the rule's example. Also good: *"Loading today's production…"* (`289`), *"Loading the latest ledger events…"* (`607`), *"Loading FIFO lots..."* (`3777`).

**FEEDBACK-005 — PARTIAL/FAIL, app-wide.** *Medium.* No single convention exists. Action progress appears in three different places depending on the caller: on the button (`.loading` class plus a label change — `refreshAll`, both exports, supply request, mark-done, recent entries), inside the target container (`innerHTML` replaced with `.loading-indicator` — every list refresh), and in a dedicated status line (`#recent-entries-status` at `index.html:120`, `#statusBar` in Traceability, `#banner` in Sankey). Background/sync status has one slot in the dashboard (`#last-refreshed`, `index.html:42`) ✓ but Sankey and Process Flow each define their own `#last-updated` in their own headers and Traceability has none.
**S-89 — FAIL.** `scheduler:1565` — validation feedback arrives in a native `alert()`, a fourth location.

**FEEDBACK-006 — PARTIAL.** *Medium.* Background work never blocks the UI ✓ across the dashboard — except `sankey.html:763-772`, where `.loading-overlay` covers the whole chart container on every parameter change (S-57, S-58, S-59). And the dashboard's convention of replacing a section's `innerHTML` with a loading block is the opposite of a compact inline spinner: the 60-second Recent Entries poll tears the entire feed down to *"Loading the latest ledger events…"* (`dashboard.js:607`) — **FAIL for S-14**.
**S-36, S-50 — PASS. Reference implementations:** `populateAllocationLots` shows progress inside the select itself (`3008-3009`) and `renderSupplyLotDetail` shows it inside the expanded detail row only (`3776-3778`) ✓.

**FEEDBACK-010 — FAIL, app-wide.** *Medium.* No list insert, delete, or reorder is animated anywhere. `.note-card { transition: background-color 0.15s ease, opacity 0.2s ease }` (`css:1017`) would animate the `.done` dimming — but `refreshNotes()` (`dashboard.js:1749`) removes the row by full re-render before the transition can play. The user's own actions produce instant, untracked changes: a ticked note vanishes, a released reservation disappears from the table, a closed expected receipt loses its action buttons.

**FEEDBACK-013 — FAIL (S-20, S-31, S-37) / PARTIAL (S-16, S-25, S-26, S-27, S-42).** *High.* Colour-carried status on thin, small, regular-weight text:
- `.readiness-chip-detail` — 10px weight 400 at 90% opacity, coloured by severity (`css:1628-1633`). This is the blocker reason.
- `.allocation-expiry` — `color: var(--badge-amber-text)` at 10px (`css:2080` with `2055-2062`). This is the reservation TTL.
- `.note-due.overdue` — `color: var(--danger)` at 11px regular (`css:1093` with `1082-1090`).
- `.ship-by-weekday` under an overdue date — `color-mix(danger 72%)` at 11px weight 500 (`css:1533-1542`).
- `.late-lag` — warning colour at 11px weight 600 (`css:810-816`) — borderline.
**Passing:** every badge and chip in the product renders status as a filled shape with a text label ✓, and the six 3px status left-borders (`css:431`, `1023`, `1321`, `2223`, `2355`, `2366`) are all paired with a badge or a row tint rather than standing alone ✓.

**FEEDBACK-014 — PASS, with two PARTIALs.** *Medium.* Every tint in the product communicates and every one is low-saturation (12–14%): `.late-entry` (`css:806-809`), `.er-overdue` (`2222`), `.supply-item-low` (`2354`), `.readiness-chip.severity-*` (`1634-1648`), `.allocation-feedback.*` (`2010-2027`), `.preview-allocation-warning` (`2104-2113`) ✓. Long-lived screens are neutral: Process Flow and Sankey on `#0f172a`, the scheduler on `--paper: #f4f3ee` ✓.
**PARTIAL (S-08):** `.day-card.today { box-shadow: 0 0 20px rgba(59,130,246,0.1) }` (`css:453`) — a decorative glow carrying no status beyond "today", which the tinted date header already carries (`455-461`).
**PARTIAL (S-13):** `.missing-list { background: var(--badge-red-bg); border-top: 1px solid #991b1b }` (`css:740-746`) — a full-width saturated red band listing missing SKUs, permanently present whenever the API returns any, on a screen intended to stay open all shift.

---

### ERROR-004, ERROR-005, ERROR-006, ERROR-007, ERROR-008, ERROR-009

**ERROR-004 — FAIL (S-22, S-88, S-89).** *High · Hard rule.* Alerts are used where an inline form error belongs. `dashboard.js:1813` — `alert('Title is required')` and `1845` — `alert('Save failed: ' + err.message)`; `scheduler:1565` — `alert("SKU, qty, and due date are required.")`. A validation failure on a form the user just submitted is not an unexpected problem; it is the predictable outcome of their own action and belongs in the form. The ER and Supply Request modals already do this correctly with `#er-modal-error` (`index.html:436`) and `#supply-request-modal-error` (`388`, with `role="alert"`).
**FAIL on the choice-dialog clause (S-22, S-45, S-53, S-33).** Leaving a half-filled form is a user-initiated action that requires a clarifying choice, and the product offers none — it discards silently. See ERROR-002.
**PASS everywhere else:** no alert is used to present a choice ✓; `showSupplyFeedback`'s 6-second auto-dismiss (`3898`) carries **success only** (`3965`, `4068`), with errors routed to the persistent `showError` (`3970`, `4071`) ✓ — the correct separation, and it satisfies NOTIFY-004 as well.

**ERROR-005 — PASS, with one PARTIAL.** *High.* The product does **not** over-confirm. All five confirmations protect something genuinely consequential: note delete (`1771`), SO status change (`2819`), allocation release (`3096`), ER close/cancel (`3470`), scheduler reset (`scheduler:1321`) ✓. Nothing confirms an ordinary add or edit.
**PARTIAL (S-35):** `saveOrderStatus` confirms **every** transition (`dashboard.js:2818-2823`), including benign forward moves like New → Confirmed. Confirming the routine case is exactly what trains users to click through the case that matters (Ready → Cancelled). Confirm only on backward transitions and on `cancelled`.
**The real failure is the inverse and lives under ACTION-008:** consequential actions with no confirmation at all — the scheduler order-line delete (`scheduler:1553-1554`) and the irreversible ER Close (ERROR-003).

**ERROR-006 — PASS on three, FAIL on two.** *Medium.*
- `confirm('Change SO-1234 status from Ready to Ship to Cancelled?')` (`dashboard.js:2819`) — one line, names the record and both states ✓ **the best dialog title in the product.**
- `confirm('Release this 2400 lb reservation? This makes it available to other orders but does not move physical inventory.')` (`3096`) — title plus a genuinely non-obvious consequence ✓.
- `confirm('Clear all orders, pins, inventory, and settings?')` (`scheduler:1321`) — one line enumerating exactly what is lost ✓.
- **FAIL:** `confirm('Delete this item?')` (`1771`) — one line, but it does not name the note. With the list rebuilt behind the dialog there is nothing on screen identifying which item.
- **FAIL:** `alert("SKU, qty, and due date are required.")` (`scheduler:1565`) — does not say which of the three is missing.
- **PARTIAL (S-43):** the two-step arm renders only `"Cancel? Confirm"` (`3473`) without naming the record. The *error* path does name it (`Failed to cancel 2000 lb Coconut Chips from Franklin Baker`, `3493`); the confirmation does not.

**ERROR-007 — PASS on the custom modals, PARTIAL elsewhere.** *Critical · Hard rule.*
- All three custom modals have a Cancel in the same slot every time — right-aligned in `.form-actions`, second after Save (`index.html:391`, `439`, `501`; `css:1201-1206`) ✓ **consistent.**
- Native `confirm()` provides Cancel in the browser's fixed position ✓ (S-23, S-35, S-38). The separate ACTION-008 problem — that OK is the Enter default and OK is destructive — is recorded there.
- **PARTIAL (S-22, S-45, S-53):** the backdrop click is the fastest apparent "safe exit" and it is not safe — it discards the draft (`1871-1873`, `3640-3642`, `4100-4102`). The rule requires *"a Cancel that does nothing"*; here the most reachable exit does something irreversible.
- **PARTIAL (S-43):** once armed, the ER Close/Cancel button offers no explicit "no". The only ways out are to wait four seconds (`3475-3481`) or click elsewhere.
- **PARTIAL (S-33, S-36):** "Done" in Order Detail edit mode (`2709`) reads like a commit and behaves like a Cancel — the worst possible labelling for a safe exit.
- **FAIL (S-87):** `scheduler:1537`, `:1553-1554` — the order-line delete has no dialog at all, so there is no Cancel to place.
- **PARTIAL (S-86):** the pin modal's three buttons are `Clear pins | Cancel | Pin & replan` (`scheduler:1453`) — Cancel is present ✓ but sits in the middle, with the destructive "Clear pins" leftmost.

**ERROR-008 — FAIL (S-55) and on Traceability's disambiguation.** *High.*
- `renderLotDisambiguation` (`dashboard.js:1372-1391`) loops `for (const m of matches)` with **no cap**, rendering one button per matching product into `#lot-panel-body`, which is `overflow-y: auto` (`css:891-895`). Five matches means five choices in a scrolling panel — both clauses of the rule broken.
- `showDisambiguation` (`traceability.html:537-553`) does the same into `#statusBar`, a flex container with `min-height: 44px` and no scroll (`171-174`), so the buttons overflow the bar entirely.
- **PASS:** the four native confirms offer two options ✓; the scheduler pin modal offers two plus Cancel ✓ (`scheduler:1453`).

**ERROR-009 — PASS.** *Medium.* There are no menus anywhere in the product (no `⋯` overflow, no context menu), so a menu can never appear unexpectedly ✓, and every choice dialog appears in direct response to a user action ✓.
**PARTIAL (S-67):** `showDisambiguation` injects choice buttons into the **status bar** (`traceability.html:543-552`) rather than a dialog — a choice arising from an action presented in a status region.

---

### ERROR-010 — Error messages and identifiers are selectable and copyable in one gesture — **HIGH**

**PASS on rendered text.** There is no `user-select: none` anywhere in the product (verified by grep). Every error surface renders through `textContent` and is selectable: `showError` (`dashboard.js:473`), `.order-edit-message` (`2690`), `.allocation-feedback` (`2969`), `.supply-feedback` (`3896`), `.order-lines-error` (`2520`), `#er-modal-error` (`3622`) ✓. Lot codes, SO numbers, SKUs, and BOL references all render as plain text ✓.

**FAIL — native dialog text is not selectable (S-22, S-23, S-35, S-38, S-89).** `alert()` and `confirm()` bodies cannot be selected or copied in Chrome or Safari. The rule's Factory Ledger example is precisely this: *"An API error in the dashboard is copyable so it can be pasted to Claude Code verbatim."* `alert('Save failed: ' + err.message)` (`dashboard.js:1845`) puts a raw API error into the one surface it cannot be copied from.

**PARTIAL — click handlers swallow selection (S-10…S-13, S-17, S-18, S-25…S-27, S-30, S-54, S-55, S-56).** `bindLotLinks` binds `click` on `.lot-link` (`dashboard.js:1640-1647`); `.order-link` sits inside a click-bound `.order-row` (`2322`, `2340-2347`). A click-drag to select the text ends in a `click` event, so attempting to select a lot code opens the lot panel and attempting to select an SO number navigates to the order. The rule warns against *"controls that swallow selection."*

**FAIL — no copy affordance anywhere in the dashboard.** The rule asks for *"a copy affordance [on] a lot code for WhatsApp, customer email, or trace search."* There is none on any lot code, SO number, or SKU.

**S-69 — FAIL.** `traceability.html:217-221` — `.node-tooltip { pointer-events: none }`. The tooltip is the only place the full (untruncated) node label and transaction IDs are shown (`1157-1164`), and it is unselectable by construction, disappearing the moment the pointer moves toward it.

**S-70, S-73, S-84, S-85 — PASS. The reference implementations:**
- `downloadAuditText` (`traceability.html:1306-1360`) produces a complete, copyable, plain-text audit report including every transaction ID ✓.
- The scheduler's `copyScheduleText` (`scheduler:1086-1094`) uses `navigator.clipboard.writeText` with an `execCommand` fallback (`1079-1085`) and reports success or failure in the button label ✓ — a proper copy affordance, and the only one in the product.

**S-67 — PASS.** Traceability's status bar renders error text as selectable HTML (`traceability.html:924-936`) including *"No records found for lot code "26-0907-C". Check the lot code and try again."* with clickable near-miss suggestions ✓ (`938-947`).

---

### NOTIFY-011 — Important information is findable on app entry, independent of any transient channel — **HIGH**

**FAIL. With no notification channel in the product, this rule carries the entire attention model — and nothing is promoted to the entry screen.** *Hard rule.*

**Evidence.** On open, `initTabs` leaves `operations` active (`dashboard.js:10`; `index.html:48`, `58`). The Operations tab contains Today So Far, the Production Calendar, Finished Goods, Batch Inventory, and On-Hand Ingredients — five read-only production views. The only attention-bearing element in the page chrome is the health badge (`index.html:39`), which reports a **system-integrity** score from `/audit/integrity` (`dashboard.js:4114`), not an operational one, and the mini-calendar's ship-date dots (`mini-calendar.js:50-58`).

**Every operational attention item is behind a tab, and several behind a tab plus a filter change:**

| Attention item | Where it lives | Already computed at |
|---|---|---|
| Overdue expected receipts | Tab 6 | `dashboard.js:3411` — `overdueCount` |
| Open supply requests | Tab 7 | `dashboard.js:3923` — `openCount` |
| Low-stock supplies | Tab 7 | `dashboard.js:3674` — per-category counts |
| Overdue sales orders | Tab 5 + the "Overdue only" checkbox | `dashboard.js:2009-2015` — `isOrderOverdue` |
| Dispatch-blocked orders | Tab 5 + switching the status filter to Dispatch Queue | `dashboard.js:2287` |
| To-dos and reminders due today | Tab 4 | `dashboard.js:1716` — the overdue comparison |
| Failed background refreshes | Nowhere | `refreshAll` swallows them via `Promise.allSettled` (`4163`) |

Every one of those counts is **already calculated in the code**. They are simply computed inside their own tab's render function and never promoted. The standard's Factory Ledger example for this rule is exact: *"Today So Far / dashboard shows open holds, overdue receipts, missed pickups, and failed syncs even if every push was ignored."*

The audit test — *"With all notifications disabled, can a user discover every pending attention item within one screen of opening the app?"* — fails on six of seven items. And notifications are not merely disabled here; they do not exist.

**S-62, S-72 — PASS.** Process Flow surfaces its stale/error state on entry ✓. The scheduler's topbar KPI strip puts the five things that matter (done-by date, orders at risk, dominant bottleneck, labor utilization, late days) in the first thing you see (`scheduler:3-9`) ✓ — the model the dashboard needs.

**Suggested fix. This is the single highest-value change in the audit.** Add an attention strip at the top of the Operations tab, above Today So Far, rendering the seven counts as chips that deep-link to the relevant tab with the relevant filter pre-applied — for example *"3 overdue receipts · 2 blocked orders · 5 low stock · 1 supply request"*. Every number already exists; only the promotion is missing. It resolves NOTIFY-011, most of NOTIFY-005, and materially improves NAV-001.

---

### NOTIFY-005 — Surface new information in context and make the change perceptible, not as an interruption — **HIGH**

**FAIL (S-14, S-25, S-26, S-28, S-37, S-42, S-49, S-51, S-61).**

The rule has two halves, and the product fails both in opposite directions.

**Where data does arrive live, it is not perceptible — it is disruptive.** The two auto-refreshing surfaces both replace their container wholesale with no highlight: Recent Entries (`dashboard.js:626-631` → `568-600`) and the Process Flow grid (`process-flow.html`, `grid.innerHTML = html`). The rule asks for *"a new row, updated counter, with a brief highlight so the change is noticed"*; what happens instead is that the entire view is torn down and rebuilt (see LAYOUT-020).

**Everywhere else, data does not arrive at all.** Sales Orders, Expected Receipts, Supplies, Supply Requests, Notes, and every inventory table are static until a human presses Refresh (see FEEDBACK-007). The rule's Factory Ledger example — *"If Luz is on Receiving when a receipt is logged from the floor, the row appears; if she is on Sales Orders, a badge on the Receiving nav item increments"* — describes behaviour that does not exist in either branch. There is no cross-tab badge on any tab in the tab bar (`index.html:47-55`).

**S-52, S-60, S-62, S-75, S-81, S-83, S-87 — PASS.** In-context status regions that update in place without interrupting: `#supply-requests-feedback` with `role="status"` ✓, Process Flow's summary strip and banners ✓, the scheduler's delta panel and board (which re-render only in response to the user's own change) ✓.

---

### NOTIFY-010 — Badges show only counts of unhandled items, are always accurate, and are never faked — **MEDIUM**

*Hard rule.*

**S-26 (Dispatch summary) — FAIL on accuracy.** `dashboard.js:2286-2288`:
```
const total = summary ? Number(summary.total_orders_checked || 0) : state.ordersData.length;
const ready = state.ordersData.filter(order => order.dispatch_ready).length;
summaryEl.textContent = `${ready} ready · ${Math.max(0, total - ready)} blocked`;
```
`total` comes from the **server's** count of everything it checked; `ready` is counted from the **client's** list, which was fetched with `limit=200` (`2247`). The two are different populations, so `total - ready` overstates "blocked" by exactly the number of ready orders that fell past the limit. The rule requires badges to be *"always accurate."*

**S-05, S-19, S-24, S-41 — FAIL on absence.** The rule's model — *"'Receiving (3)' means three unhandled receipts"* — has no implementation. Not one of the seven tabs carries a count, despite every count already being computed inside the corresponding tab (see NOTIFY-011).

**S-47 — PARTIAL.** The three `.supplies-count-badge` low-stock counters (`index.html:302`, `305`, `308`; `dashboard.js:3667-3679`) do count items needing attention ✓ and update on every render ✓, but: they render as **permanently amber even at zero** (`css:2306-2320`; `dashboard.js:3676` sets only `textContent`), which is the "faked signal" the rule warns about — a chip that is always amber teaches the user to ignore amber; and they are computed from the `/supplies/inventory` response with no truncation check (NAV-012).

**S-41 — PARTIAL.** `#er-summary` renders *"12 shown · 8 open · 3 overdue"* (`dashboard.js:3412-3413`) from `state.erData`, which was fetched with `limit=500` (`3374`). Text rather than a badge oval ✓, but potentially inaccurate.

**S-02 — PARTIAL.** Mini-calendar ship-date dots (`mini-calendar.js:50-58`) carry an accurate per-day count in the `title` ✓, drawn from a `limit=200` fetch (`108`), so a busy day past the limit renders as clear.

**S-51, S-87 — PASS. Correct badges.** `#supply-requests-open-count` (`index.html:322`; `dashboard.js:3923-3926`) counts open requests, decrements the moment one is marked Done, and carries an `aria-label` with the full phrase ✓. The scheduler's order-book header count `"Order book (14)"` (`scheduler:1524`) counts included lines and updates on every change ✓.

**S-10, S-12, S-13 — PASS on the inverse clause.** `.panel-count` *"(4 SKUs)"* (`dashboard.js:913`) and `.ingredient-header-count` *"Total SKUs: 12"* (`1097`) are rendered as dimmed inline text (`css:647`, `953-958`), **not** as badge ovals — correctly honouring *"Inventory quantities are data, not red ovals."*

---

### NOTIFY-004 — Errors are shown as in-app alerts, never as notifications — **CRITICAL**

**PASS across the product, and the separation is deliberate.** *Hard rule.*

With no notification channel, the rule is vacuously satisfied — but the more interesting finding is that the one auto-dismissing message surface in the product is correctly restricted to successes. `showSupplyFeedback` (`dashboard.js:3893-3899`) hides itself after 6 seconds and is called **only** with success messages (`3965`, `4068`); every error path in the same module routes to `showError` (`3916`, `3970`, `4071`), which is persistent. The rule's concern — *"an error arriving as a push is easily missed"* — is architecturally avoided.

**S-22, S-89 — FAIL on the related clause.** `alert()` is a browser-modal interruption used for what should be an inline form error: `dashboard.js:1813` (*"Title is required"*), `1845` (*"Save failed: …"*), `scheduler:1565`. Not a notification, but the same failure mode — see ERROR-004.

---

## N/A register

| Rule | Screens | Reason |
|---|---|---|
| NOTIFY-001, 002, 003, 006, 007, 008, 009, 012 | **All 91** | **No notification channel exists in this product** — no Notification API, no service worker, no push, no email/SMS trigger. Nothing to send, dedupe, label, secure, or let the user opt out of. |
| FEEDBACK-001…007 | Read-only tiles, states, legends, tooltips, print views, and every scheduler screen | No network-bound operation on the screen; the scheduler computes synchronously from `localStorage` (`scheduler:376-378`). |
| FEEDBACK-007 | S-72…S-91 | **No API; state is local to the browser** — there is nothing to sync or timestamp. Per the audit brief. |
| FEEDBACK-009, 010 | Screens with no list or selection | Nothing to select or animate. |
| FEEDBACK-013, 014 | S-57…S-62, S-71…S-91 | Out of tier scope (Medium/High on Tier B where not Critical) or no colour-carried status present. |
| ERROR-001 | All screens except S-24, S-33, S-64 | No long or interruptible operation on the screen. |
| ERROR-002 | Chrome, legends, empty states, print views | No in-progress work to preserve. |
| ERROR-003 | Read-only screens | No user action to recover from. |
| ERROR-005, 006, 007, 008, 009 | Screens with no dialog | Nothing to confirm. |
| ERROR-010 | S-01, S-05 | No error text or identifier rendered. |
| NOTIFY-004, 005, 010, 011 | Screens carrying no error, no live data, no badge, and no attention item | Nothing in scope for the rule. |

## Unverifiable from code

| Rule | Screens | What a browser check must confirm |
|---|---|---|
| FEEDBACK-002 | S-24 (export), S-33, S-39, S-64, S-68, S-80 | Actual elapsed time for the multi-request operations. The absence of a determinate indicator is verified in code; whether users perceive the wait as stalled depends on real latency against the Railway API. |
| FEEDBACK-003 | All network-bound screens | The API's real hang-versus-error behaviour. The absence of any timeout is verified in code; how often a request hangs rather than failing fast is an operational question. |
| FEEDBACK-011 | S-71, S-91 (print views) | Rendered greyscale output. The colour-only cues are verified in code; the print stylesheets (`traceability.html:240-246`, `scheduler:185-208`) restyle enough that the rendered result needs checking. |
| FEEDBACK-012 | All screens, light theme | Rendered contrast of the nine hard-coded dark-mode status colours against the light palette. Two (`.so-ready-pill`, `.order-edit-message.success`) are computed here as roughly 1.4:1 and should be measured. |
| ERROR-002 | S-01…S-05, S-22, S-33, S-45, S-53 | Whether a rotation or app-switch loses form values in practice. No state is width- or visibility-keyed in code, which is necessary but not sufficient evidence. |
| ERROR-010 | S-10…S-13, S-25…S-27, S-54…S-56 | Whether a click-drag over `.lot-link` / `.order-link` actually fires the click handler and prevents selection in Chrome and Safari. The handler binding is verified; the browser's drag-versus-click threshold is not. |

---

*End of 03-feedback-error-notify.md*

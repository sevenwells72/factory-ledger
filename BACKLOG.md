# Backlog — Deferred Items

Items that surfaced during planner v2 development but were intentionally
deferred. Listed in rough priority order. None block daily use of the
planner.

## Pallet Charge / service item filtering in SO panel

The SO panel's Items column currently includes service line items
(Pallet Charge, Customer Pickup) as if they were products. They appear in
"Honey Nut 25 LB · Pallet Charge" style strings. Cosmetic noise, not a
math error — the SQL's `quantity_lb > 0` filter already excludes them
from batch calculations.

Resolution path: in `render_planner_v2.py`'s SO row builder, filter the
items list against a known service-item name list before rendering.

## Duplicate SO numbers in source data

`SO-260413-001` (RESTAURANT DEPOT/Jetro Haines City) appears on multiple
table rows with different ship dates and quantities (May 1: 36,400 lb,
Jun 1: 36,400 lb, etc.). Could be a real data issue (duplicate SO entry
in Odoo) or a multi-line SO being rendered as separate rows.

Resolution path: investigate the source table — check whether the SO has
multiple `requested_ship_date` values across line items, or whether the
SO was duplicated. Fix at source if duplicated.

## Over-capacity callout on planner_v2 week cards

Week 1 currently can show batch totals far above weekly capacity (recent
example: 138 granola batches vs 75 weekly cap, 103 coconut vs 57 cap).
The planner shows the demand correctly but doesn't flag visually that the
work physically can't fit in the week.

Resolution path: add a small text indicator (e.g. red "exceeds weekly
cap" line) below the metric cards when granola or coconut counts exceed
the capacity numbers stored in the YAML. Do NOT re-add capacity bars or
percentages — keep the visual minimal.

## Pouch line load computation

Discrepancy #14 — pouch line is shown as info-only on demand_plan v1.
Planner_v2 doesn't surface pouch line at all. Address only if pouch
bottlenecks actually surface in real planning use.

## Coconut "0/114 big green" visual

Discrepancy from v1: when coconut has 99 batches beyond the 14d horizon
but 0 in the 7d window, the v1 capacity card reads as comfortably green
when it shouldn't. v2 partially addresses this with the Beyond card.
Revisit only if this visual reappears or causes confusion.

## Open discrepancies tracking

`discrepancies.md` lists open data-quality items. Resolution is ongoing,
roughly one per week. No formal cadence.

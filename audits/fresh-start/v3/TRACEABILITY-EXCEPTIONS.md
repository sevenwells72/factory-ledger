# Traceability exception register — opening stock without an identified supplier lot

Permanent record (created 2026-10-27, owner-approved). Each lot below holds real inventory from the Oct 21–22, 2026
physical count whose supplier lot could not be identified. Their FL follow-ups were closed administratively as
**written off** (owner decision 2026-10-27): that closes the reminder only. Inventory, lot history and the lot's
`unidentified` status are unchanged, and nothing claims a supplier lot. Keep this register until each lot is
identified (supplier-lot correction with evidence) or fully consumed.

| FL exception | Product | Lot (FL id) | Opening qty (lb) | Receipt | Status |
| --- | --- | --- | ---: | --- | --- |
| 50 | Cinnamon Ground | 26-10-27-FOUND-001 (1648) | 20.00 | FND-261022-001 | unidentified |
| 51 | Sugar – 6X | 26-10-27-FOUND-002 (1649) | 152.49 | FND-261022-002 | unidentified |
| 53 | Glycol | 26-10-27-FOUND-003 (1650) | 972.00 | FND-261022-003 | unidentified |
| 55 | Chocolate Chips – Real – 1,000 CT | 26-10-27-FOUND-004 (1651) | 1,250.00 | FND-261022-004 | unidentified |
| 57 | Corn Starch | 26-10-27-FOUND-005 (1652) | 1,365.00 | FND-261022-005 | unidentified |
| 59 | Honey | 26-10-27-FOUND-006 (1653) | 1,558.20 | FND-261022-006 | unidentified |
| 65 | Coconut Flake Desiccated | 26-10-27-FOUND-011 (1658) | 159,250.00 | FND-261022-011 | unidentified — POOLED |

**Follow-ups closed 2026-10-27** by Blubber (owner) as written off (FL status `waived`) with the register notes; verified
afterwards: balances, `unidentified` status, no supplier lot and lot history unchanged. Final reconciliation: RESET COMPLETE.

## Recorded evidence (exact lot notes from FL)

**Cinnamon Ground — 26-10-27-FOUND-001**
> Opening stock — counted Oct 21–22, 2026. Possible sources: 2026-04-27, Dutch Valley, supplier lot 55279, BOL 55279; 2026-02-06 (no receipt record), supplier not recorded, supplier lot -, BOL -; exception OPEN-7, counted 2026-10-22T09:00:00-04:00; lot as printed (none)

**Sugar – 6X — 26-10-27-FOUND-002**
> Opening stock — counted Oct 21–22, 2026. Possible sources: 2026-09-30, New England, supplier lot 93026, BOL 262746; 2026-08-06, Sweet New England, supplier lot 072326L7, BOL 60526SNE; 2026-06-10, NEW ENGLAND, supplier lot 6926, BOL 51326S; exception OPEN-40, counted 2026-10-22T11:12:00-04:00; lot as printed (none)

**Glycol — 26-10-27-FOUND-003**
> Opening stock — counted Oct 21–22, 2026. Possible sources: 2026-09-28, Tilley, supplier lot 73867-260707, BOL 252784; 2026-04-24, Tilley, supplier lot 260416, BOL 260416; exception OPEN-29, counted 2026-10-22T11:10:00-04:00; lot as printed (none)

**Chocolate Chips – Real – 1,000 CT — 26-10-27-FOUND-004**
> Opening stock — counted Oct 21–22, 2026. Possible sources: none found; exception OPEN-73, counted 2026-10-22T09:00:00-04:00; lot as printed (none)

**Corn Starch — 26-10-27-FOUND-005**
> Opening stock — counted Oct 21–22, 2026. Possible sources: 2026-10-19, Essex Food, supplier lot 1246213301, BOL SHP211226100428; 2026-07-13, Essex Food, supplier lot 1246163300, BOL 2120050; exception OPEN-16, counted 2026-10-22T09:00:00-04:00; lot as printed (none)

**Honey — 26-10-27-FOUND-006**
> Opening stock — counted Oct 21–22, 2026. Possible sources: 2026-10-06, Dutch Gold, supplier lot 26426-D, BOL 091426; 2026-09-03, Dutch Gold, supplier lot 23326D, BOL 1412736; 2026-07-10, Dutch Gold, supplier lot 18126D, BOL 199702; 2026-04-28, Dutch Gold, supplier lot 03091, BOL 03091; exception OPEN-30, counted 2026-10-22T09:00:00-04:00; lot as printed (none)

**Coconut Flake Desiccated — 26-10-27-FOUND-011**
> Opening stock — counted Oct 21–22, 2026. Possible sources: 2026-05-14 (no receipt record), supplier not recorded, supplier lot -, BOL -; 2026-05-01, Franklin Baker, supplier lot 4062026, BOL 4062026; bags counted by supplier lot (lb): 6161 46,300; 6182 38,400; 6113 37,750; 6153 27,700; 6196 13,100; 6114 1,600; 6190 1,500; 6127 600; 6051 300; 6125 100 = 167,350 lb, of which the reset rule booked 8,100 lb to existing FL lot 6013; POOLED opening stock — not traceable to a single supplier lot; exception OPEN-12, counted 2026-10-22T09:00:00-04:00; lot as printed (none)

## Coconut Flake Desiccated — POOLED (highest priority)
- 159,250 lb opening lot 26-10-27-FOUND-011 holds bags of at least ten supplier lots (6161, 6182, 6113, 6153, 6196,
  6114, 6190, 6127, 6051, 6125 — 167,350 lb counted in total). A further 8,100 lb of those counted bags was booked by the
  reset rule to the existing FL lot 6013 (6,900 lb at the end of Oct 27 after Oct 23 production), so lot 6013 also
  holds bags of these supplier lots, not only 6013.
- Recall impact until resolved: a recall of any of those supplier lots covers the whole pooled lot and lot 6013.
- How to resolve without losing history: when bags are confirmed by supplier lot, post Found entries per supplier lot
  (each with its supplier lot code) and matching count_correction adjustments that reduce the pooled lot / lot 6013;
  never identify the pooled lot with one code (draft PR #154 makes FL refuse that).

## How any lot here gets resolved
1. Evidence (bag label, receipt, BOL) of the actual supplier lot.
2. Single-source lot: supplier-lot correction with that evidence (FL keeps the old state in lot history).
3. Mixed lot: split with Found + adjust corrections as above.
4. Update this register (date, who, receipt numbers).

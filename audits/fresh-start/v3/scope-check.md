# All-inventory scope check - v3

**Plain-English summary:** Count all physical inventory, by item, printed lot and actual location, while production continues. FL has **210 catalog products and 1199 lot records** in this read-only snapshot. Packaging is present in the catalog, but **30 of its 32 products have no lots or posted movement history**. Count what is found. Packaging is information only and excluded from the reset; Pallets 102 and Pallet Charge 176 are excluded billing items. Reset scope is **178 products**; the **32 packaging/billing products** have a separate information-only report. Missing FL records for reset-scope stock are holds, never permission to create lots or discard stock. This v3 count replaces the October 5 ingredient-total proposal and the v2 frozen-count instructions.

Snapshot: 2026-10-06T18:37:34.872652+00:00. Fixed query `queries/01-scope.sql`, `BEGIN; SET TRANSACTION READ ONLY; ... ROLLBACK;`, port 5432; rollback confirmed. No writes/API calls. The category groupings below are suggested work groups, **not a verified physical location map**. Arturo fills actual Area and Location on every sheet/row, duplicating blank sheets for additional areas with new sheet and row IDs.

| Suggested work group | Products | Active products | All lots | Active lots | Nonzero lots |
| --- | --- | --- | --- | --- | --- |
| 01 Ingredients | 68 | 68 | 202 | 202 | 90 |
| 02 Coconut raw materials | 9 | 9 | 40 | 40 | 18 |
| 03 Other WIP - container stock | 1 | 1 | 1 | 1 | 1 |
| 04 Coconut bulk WIP | 4 | 4 | 181 | 181 | 1 |
| 05 Granola bulk WIP | 21 | 21 | 226 | 225 | 10 |
| 06 Finished goods - all customers | 75 | 65 | 547 | 546 | 65 |
| 07 Packaging - bags, boxes, tape and pallets | 32 | 32 | 2 | 2 | 2 |

There are 78 ingredients (including raw coconut and one container-based WIP item), 25 batches (21 granola, four coconut), 75 finished products (65 active, ten inactive), 32 packaging products and **zero consumable products**. Every catalog item appears in the count packet; the former exclusions 171/209 are superseded for physical counting by the owner's ALL-inventory scope. Within reset scope, inactive, service, unresolved-unit and unmatched-lot adjustments remain held. Packaging never generates reset adjustments or holds. No catalog status changes were made.

## Packaging: which records exist?

| ID / SKU | Product | Catalog unit | Lots / printed codes | Posted lines |
| --- | --- | --- | --- | --- |
| 76 / 21001 | Bag Blue #10 | unit | 0 /  | 0 |
| 77 / 21002 | Bag Blue #25 | unit | 0 /  | 0 |
| 78 / 21003 | Bag Clear #10 | unit | 0 /  | 0 |
| 79 / 21004 | Bag Clear #25 | unit | 0 /  | 0 |
| 83 / 21024 | Bag – Printed – BS Almond Butter 7 oz | unit | 0 /  | 0 |
| 84 / 21025 | Bag – Printed – BS Dark Chocolate 7 oz | unit | 0 /  | 0 |
| 85 / 21026 | Bag – Printed – BS Hazelnut Butter 7 oz | unit | 0 /  | 0 |
| 86 / 21028 | Bag – Printed – BS Peanut Butter 7 oz | unit | 0 /  | 0 |
| 80 / 21005 | Bag – Printed – SS Chocolate Chip Granola | unit | 0 /  | 0 |
| 87 / 21061 | Bag – Printed – SS Chocolate Chip Granola Low Carb | unit | 0 /  | 0 |
| 81 / 21006 | Bag – Printed – SS Cranberry Granola | unit | 0 /  | 0 |
| 82 / 21007 | Bag – Printed – SS Original Granola | unit | 0 /  | 0 |
| 88 / 21060 | Bag – Printed – SS Original Granola Low Carb | unit | 0 /  | 0 |
| 89 / 21008 | Box 10 LB White Plain #13 | unit | 0 /  | 0 |
| 90 / 21009 | Box 10 LB White Plain #13 Flat | unit | 0 /  | 0 |
| 91 / 21010 | Box 14 12 | unit | 0 /  | 0 |
| 92 / 21011 | Box 24 | unit | 0 /  | 0 |
| 93 / 21012 | Box 24 Flat | unit | 0 /  | 0 |
| 94 / 21013 | Box 38F Toasted 25 LB | unit | 0 /  | 0 |
| 95 / 21014 | Box F28 | unit | 0 /  | 0 |
| 96 / 21015 | Box PL9 Crumb | unit | 0 /  | 0 |
| 97 / 21016 | Box Unipro Printed 10 LB | unit | 0 /  | 0 |
| 98 / 21017 | Box Unipro Printed 10 LB Flat | unit | 0 /  | 0 |
| 99 / 21018 | Box – CQ – Printed – Coconut | unit | 0 /  | 0 |
| 100 / 21019 | Box – CQ – Printed – Granola | unit | 0 /  | 0 |
| 101 / 21020 | Box – CQ – Printed – Granola – Flat | unit | 0 /  | 0 |
| 105 / 21029 | Case – Generic – 7 OZ Master | unit | 0 /  | 0 |
| 106 / 21030 | Case – Generic – 7 OZ Master – Flat | unit | 0 /  | 0 |
| 176 / None | Pallet Charge | each | 1 / 26-03-19-FOUN-001 | 12 |
| 102 / 21021 | Pallets | unit | 1 / INITIAL-260 | 11 |
| 103 / 21022 | Tape – 48mmx100m – Case 36 | unit | 0 /  | 0 |
| 104 / 21023 | Tape – 48mmx914m – Case 6 | unit | 0 /  | 0 |

Packaging values use unit/each counts, not pounds, even though the ledger column is named `quantity_lb`. The owner classifies Product 102 Pallets and 176 Pallet Charge as billing items: both are excluded from the reset regardless of catalog type. Count each physical pallet once for information; do not count a billing charge as an additional pallet. Bags, boxes, tape, cases and labels are likewise information only because FL does not deduct packaging on pack. No lot is created or reset for packaging, including products with no existing lots.

## Physical inventory that FL does not establish

No standalone label product, consumable product, location/bin master, container tare table or location-level balance was found. Bag/box names are catalog descriptions, not proof of what is actually in each room. `lots.found_location` is an occasional discovery note and `trace_events.biz_location` is a broad plant trace label; neither supplies current stock by shelf or area. Label rolls, film, adhesives, cleaning materials, miscellaneous supplies, unidentified WIP and any other unlisted stock must be captured on Other / not-in-catalog sheets. Their actual existence and quantities cannot be determined from a database alone; Arturo's walk-through establishes them. No unlisted material is assumed zero.

The complete identity inventory is in `scope-products-and-lots.csv` (office use). All historical/merged lot identities remain visible there; counters write the lot physically printed, with no FL quantities or suggested lot defaults on the floor sheet.

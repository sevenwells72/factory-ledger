# Product → SKU Mapping
_Generated 2026-05-19. Source of truth: products table, parent_batch_product_id link._

## Section 1: How to use this mapping

This mapping tells the GPT which finished SKUs map to which batch recipe (bake-side) and which packaging line (pack-side) so it can plan production capacity against incoming orders. Most finished products use a two-resource model: a bake-side step on either the **granola_oven** or the **coconut_line**, and — for granola SKUs only — an additional pack-side step on either the **pouch_line** (small consumer cases like 6x7 OZ, 6x8 OZ, or 12x10 OZ) or the **bulk_line** (10 LB / 25 LB foodservice cases). Coconut SKUs use the **coconut_line** as a single combined make-and-pack resource — there is no separate pack-side step, so the pack_line column is intentionally blank for coconut rows. Rows with `bake_equipment = resale` are copack or externally-sourced finished goods — never plan production for these; check inventory and trigger a PO workflow instead. SKU 70061 (Granola Fruit Nut) is a composite batch: its own parent batch (179) consumes a Classic granola batch (107) as a sub-batch, so Fruit Nut demand creates upstream Classic batch demand the GPT must account for. If a SKU you are asked about does not appear in this file, ASK — do not guess equipment, batch, or case size.

## Section 2: Batch recipes (lookup)

| parent_batch_id | batch_code | batch_name | equipment | batch_lb |
|---|---|---|---|---|
| 107 | 90002 | Batch Classic Granola #9 | granola_oven | 323 |
| 108 | 90001 | Batch Classic Chocolate Chip Granola #9 | granola_oven | 348 |
| 109 | 90019 | Batch Setton Cinnamon Almond Granola #14 | granola_oven | 330 |
| 111 | 90020 | Batch Setton Cocoa Crunch Granola #13 | granola_oven | 338 |
| 112 | 90024 | Batch Vanilla Crisp Granola #16(no almonds) | granola_oven | 370 |
| 113 | 90010 | Batch Granola Vanilla Almond 380 lb | granola_oven | 380 |
| 114 | 90011 | Batch SS Chocolate Chip Granola #2 | granola_oven | 393 |
| 116 | 90016 | Batch SS Original Granola #1 | granola_oven | 350 |
| 118 | 90013 | Batch SS Cranberry Granola #3 | granola_oven | 379 |
| 119 | 90014 | Batch SS Low Carb Chocolate Chip Granola #8 | granola_oven | 350 |
| 120 | 90015 | Batch SS Low Carb Original Granola #7 | granola_oven | 350 |
| 121 | 95002 | Batch BS Dark Chocolate Granola 350 | granola_oven | 350 |
| 122 | 95005 | Batch BS Peanut Butter Banana Granola | granola_oven | 452 |
| 123 | 95001 | Batch BS Almond Butter Granola 350 | granola_oven | 350 |
| 124 | 95003 | Batch BS Hazelnut Butter Granola 350 | granola_oven | 350 |
| 125 | 90003 | Batch Coconut Sweetened Fancy | coconut_line | 360 |
| 126 | 90004 | Batch Coconut Sweetened Flake | coconut_line | 360 |
| 127 | 90005 | Batch Coconut Sweetened Medium | coconut_line | 360 |
| 128 | 90007 | Batch Coconut Toasted Sweetened Flake | coconut_line | 300 |
| 179 | 90008 | Granola Fruit Nut Batch | granola_oven | 384.52 |

## Section 3: Finished SKUs

| sku | name | brand | bake_equipment | pack_line | parent_batch | batch_lb | case_size_lb | cases_per_pan | is_copack | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 70073 | BS Granola – Peanut Butter Banana – 6x7 OZ Case | Blue Stripes | granola_oven | pouch_line | 122 | 452.00 | 2.63 | 171.86 (rounds to 171) | False | |
| 70074 | BS Granola – Dark Chocolate – 6x7 OZ Case | Blue Stripes | granola_oven | pouch_line | 121 | 350.00 | 2.63 | 133.08 (rounds to 133) | False | |
| 70079 | BS Almond Butter Granola – 6x7 OZ Case | Blue Stripes | granola_oven | pouch_line | 123 | 350.00 | 2.63 | 133.08 (rounds to 133) | False | |
| 70080 | BS Granola – Hazelnut Butter – 6x7 OZ Case | Blue Stripes | granola_oven | pouch_line | 124 | 350.00 | 2.63 | 133.08 (rounds to 133) | False | |
| 10300 | Granola Crunchy CNS 10 LB Case | CNS | granola_oven | bulk_line | 107 | 323.00 | 10.00 | 32.30 (rounds to 32) | False | |
| 70012 | Granola Wheat Free 25 LB | CNS | granola_oven | bulk_line | 107 | 323.00 | 25.00 | 12.92 (rounds to 12) | False | |
| 70048 | Granola Vanilla Almond 25 LB | CNS | granola_oven | bulk_line | 113 | 380.00 | 25.00 | 15.20 (rounds to 15) | False | |
| 70050 | Granola Classic 25 LB | CNS | granola_oven | bulk_line | 107 | 323.00 | 25.00 | 12.92 (rounds to 12) | False | |
| 70052 | Granola Vanilla Crisp 25 LB (French Vanilla) | CNS | granola_oven | bulk_line | 112 | 370.00 | 25.00 | 14.80 (rounds to 14) | False | |
| 70053 | Granola Chocolate Chip 25 LB | CNS | granola_oven | bulk_line | 108 | 348.00 | 25.00 | 13.92 (rounds to 13) | False | |
| 70057 | Granola Cinnamon Almond 25 LB | CNS | granola_oven | bulk_line | 109 | 330.00 | 25.00 | 13.20 (rounds to 13) | False | |
| 70059 | Granola Cocoa Vibes 25 LB | CNS | granola_oven | bulk_line | 111 | 338.00 | 25.00 | 13.52 (rounds to 13) | False | |
| 70060 | Granola Honey Nut 25 LB | CNS | granola_oven | bulk_line | 116 | 350.00 | 25.00 | 14.00 (rounds to 14) | False | |
| 70061 | Granola Fruit Nut 25 LB | CNS | granola_oven | bulk_line | 179 | 384.52 | 25.00 | 15.38 (rounds to 15) | False | composite batch (consumes Classic 107) |
| 1614 | CQ Granola 10 LB | CQ | granola_oven | bulk_line | 107 | 323.00 | 10.00 | 32.30 (rounds to 32) | False | |
| 70056 | Granola Setton Cocoa Crunch 25 LB | Setton | granola_oven | bulk_line | 111 | 338.00 | 25.00 | 13.52 (rounds to 13) | False | |
| 70077 | Granola Setton Cinnamon Spice Almond 25 LB | Setton | granola_oven | bulk_line | 109 | 330.00 | 25.00 | 13.20 (rounds to 13) | False | |
| 70081 | Granola Setton Good Ol 25 LB | Setton | granola_oven | bulk_line | 107 | 323.00 | 25.00 | 12.92 (rounds to 12) | False | |
| 70082 | Granola Setton French Vanilla 25 LB | Setton | granola_oven | bulk_line | 112 | 370.00 | 25.00 | 14.80 (rounds to 14) | False | |
| 70002 | Granola SS Original 12x10 OZ Case | Sunshine | granola_oven | pouch_line | 116 | 350.00 | 7.50 | 46.67 (rounds to 46) | False | |
| 70003 | Granola SS Chocolate Chip 12x10 OZ Case | Sunshine | granola_oven | pouch_line | 114 | 393.00 | 7.50 | 52.40 (rounds to 52) | False | |
| 70004 | Granola SS Original Bulk per/lb | Sunshine | granola_oven | unknown | 116 | 350.00 | — | — | False | pack process to confirm |
| 70006 | Granola SS Mini 100 | Sunshine | granola_oven | unknown | 116 | 350.00 | — | — | False | pack process to confirm |
| 70010 | Granola SS Original Low Carb 12x10 OZ Case | Sunshine | granola_oven | pouch_line | 120 | 350.00 | 7.50 | 46.67 (rounds to 46) | False | |
| 70011 | Granola SS Cranberry 12x10 OZ Case | Sunshine | granola_oven | pouch_line | 118 | 379.00 | 7.50 | 50.53 (rounds to 50) | False | |
| 70070 | Granola SS Chocolate Chip Low Carb 12x10 OZ Case | Sunshine | granola_oven | pouch_line | 119 | 350.00 | 7.50 | 46.67 (rounds to 46) | False | |
| 70085 | BS Granola – Hazelnut Butter – 6x8 OZ Case | — | granola_oven | pouch_line | 124 | 350.00 | 3.00 | 116.67 (rounds to 116) | False | brand blank in DB — should be Blue Stripes |
| 70086 | BS Almond Butter Granola – 6x8 OZ Case | — | granola_oven | pouch_line | 123 | 350.00 | 3.00 | 116.67 (rounds to 116) | False | brand blank in DB — should be Blue Stripes |
| 70087 | BS Granola – Dark Chocolate – 6x8 OZ Case | — | granola_oven | pouch_line | 121 | 350.00 | 3.00 | 116.67 (rounds to 116) | False | brand blank in DB — should be Blue Stripes |
| 70088 | BS Granola – Peanut Butter Banana – 6x8 OZ Case | — | granola_oven | pouch_line | 122 | 452.00 | 3.00 | 150.67 (rounds to 150) | False | brand blank in DB — should be Blue Stripes |
| 10001 | Coconut Sweetened Flake CNS 10 LB | CNS | coconut_line | — | 126 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 10002 | Coconut Sweetened Medium CNS 10 LB | CNS | coconut_line | — | 127 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 10006 | Coconut Sweetened Fancy CNS 10 LB | CNS | coconut_line | — | 125 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 10007 | Coconut Sweetened Fancy CNS 25 LB | CNS | coconut_line | — | 125 | 360.00 | 25.00 | 14.40 (rounds to 14) | False | |
| 10010 | Coconut Toasted Sweetened Flake CNS 10 LB | CNS | coconut_line | — | 128 | 300.00 | 10.00 | 30.00 (rounds to 30) | False | |
| 10020 | Coconut Sweetened Flake CNS 25 LB | CNS | coconut_line | — | 126 | 360.00 | 25.00 | 14.40 (rounds to 14) | False | |
| 10029 | Coconut Toasted Sweetened Flake CNS 25 LB | CNS | coconut_line | — | 128 | 300.00 | 25.00 | 12.00 (rounds to 12) | False | |
| 893 | CQ Coconut Sweetened Flake 10 LB | CQ | coconut_line | — | 126 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 67470 | Coconut Sweetened Fancy UNIPRO 10 LB | UNIPRO | coconut_line | — | 125 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 67473 | Coconut Sweetened Medium UNIPRO 10 LB | UNIPRO | coconut_line | — | 127 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 67476 | Coconut Sweetened Flake UNIPRO 10 LB | UNIPRO | coconut_line | — | 126 | 360.00 | 10.00 | 36.00 (rounds to 36) | False | |
| 31012 | Graham Cracker Crumbs – 10 LB | CNS | resale | — | — | — | 10.00 | — | True | |
| 10045 | Desiccated Macaroon Bakers 50 LB | house | resale | — | — | — | 50.00 | — | False | |
| 10046 | Desiccated Fancy Bakers 50 LB | house | resale | — | — | — | 50.00 | — | False | |
| 10047 | Desiccated Flake 50 LB | house | resale | — | — | — | 50.00 | — | False | |
| 10048 | Desiccated Flake CNS 25 LB | house | resale | — | — | — | 25.00 | — | False | |
| 10049 | Desiccated Coconut Chips 25 LB | house | resale | — | — | — | 25.00 | — | False | |
| 10050 | Desiccated Coconut Chips 15 LB | house | resale | — | — | — | 15.00 | — | False | |
| 10051 | Desiccated Medium Bakers 50 LB | house | resale | — | — | — | 50.00 | — | False | |
| 10052 | Desiccated Medium CNS 25 LB | house | resale | — | — | — | 25.00 | — | False | |
| 10053 | Desiccated Macaroon Bakers 25 LB | house | resale | — | — | — | 25.00 | — | False | |
| 10054 | Desiccated Medium CNS 100 LB | house | resale | — | — | — | 100.00 | — | False | |
| 10055 | Desiccated Flake 55 LB | house | resale | — | — | — | 55.00 | — | False | |
| 10056 | Desiccated Flake 80 LB | house | resale | — | — | — | 80.00 | — | False | |
| 10058 | Desiccated Macaroon CNS 25 LB | house | resale | — | — | — | 25.00 | — | False | |
| 10059 | Desiccated Macaroon 55 LB | house | resale | — | — | — | 55.00 | — | False | |
| 10301 | Kookies & Kreme – 10 LB | house | resale | — | — | — | 10.00 | — | True | |
| 70051 | Coconut Chips 25 LB | house | resale | — | — | — | 25.00 | — | False | |
| 10302 | Sprinkles Rainbow 10 LB | — | resale | — | — | — | 10.00 | — | True | |
| 10303 | Sprinkles Chocolate 10 LB | — | resale | — | — | — | 10.00 | — | True | |
| 10304 | Kookies & Kreme – 25 LB | — | resale | — | — | — | 25.00 | — | True | |
| 10305 | Sprinkles Rainbow 25 LB | — | resale | — | — | — | 25.00 | — | True | |
| 10306 | Sprinkles Chocolate 25 LB | — | resale | — | — | — | 25.00 | — | True | |

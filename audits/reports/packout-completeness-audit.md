# Packout completeness audit — 2026-09-17

**Question:** are finished-product packouts being skipped (never entered), leaving bulk granola batch inventory overstated?
**Scope:** every `type='finished'` product whose `parent_batch_product_id` is a granola batch product (21 batch products, 36 finished SKUs), plus product 171 "Classic Granola 25 LB", which has no parent but is a granola FG with 2026 activity. Period 2026-01-01 → 2026-09-16, posted transactions only (`ledger_current_*` with `effective_status = 'posted'`).
**Method:** read-only queries via `~/.config/factory-ledger/db_url` rewritten to the port-5432 session pooler, every statement inside `BEGIN; SET TRANSACTION READ ONLY; … COMMIT;`. Cross-checked against QuickBooks invoices dated 2026-08-15 → 2026-09-17 (49 invoices, read-only). **Nothing was written.** No `/adjust`, no migrations; the only files touched are this report and the change logs.

---

## Verdict (plain English)

**Packouts missing: partly — but not in the period that matters, and not enough to explain the batch excess.**

1. **Since the 2026-08-14 physical count, no packout is missing on any packable granola SKU.** Every ship since Aug 14 was covered by a pack entered earlier; no SKU-month has shipped > packed + opening; no finished SKU is negative; no finished SKU has been trued up by an `adjust` or a "found inventory" `receive` since Aug 14; and the ledger's ship entries through Sep 9 match QuickBooks invoices case-for-case. The batch-side debit of every pack equals cases × case weight. **Missing packouts explain 0 lb of the 29,131 lb on-hand across batches 107 / 108 / 114 that `granola-batch-onhand-audit.md` calls unexplained.**

2. **The only "never entered" packouts are the ones that cannot be entered:** the two Sunshine per-pound bulk lines on SO-260817-001 (6,000 lb CC #9 bulk, 4,000 lb Classic #9 bulk, confirmed Aug 17, 0 lb shipped, still `confirmed/open`). That is the 10,000 lb already identified in the prior audit, and it is a product-setup problem (`case_size_lb IS NULL`, parented to never-made batches 283/284), not an operator skipping entries. It explains 6,000 of 108's 9,396 lb and 4,000 of 107's 14,878 lb. The remaining 19,131 lb (107: 10,878; 108: 3,396; 114: 4,857) is **not** a packout-entry problem.

3. **Before Aug 14, packouts were skipped and back-filled by adjustment or "found inventory" receive rather than by `/pack`, so the batch was never debited.** Total 2026 to date, excluding the February system-start baseline: **13,932 lb across 8 batch products** (107: 4,175; 123: 2,651; 122: 2,627; 119: 1,568; 120: 1,523; 118: 780; 121: 316; 114: 293). Every one of these dates on or before Aug 14, and every affected batch was written to a physical count on Jun 8 and/or Aug 14, so none of it survives in today's ledger. On 107 they account for ~15 % of the 27,725 lb written down at those two counts; on 114 for ~1 % of 29,207 lb.

4. **The opposite problem is live right now: ships, not packs, are what is not being entered.** SS Chocolate Chip 12x10 OZ (145) shows **3,830 cases (28,725 lb)** on hand and SS Original 12x10 OZ (146) shows **1,761 cases (13,208 lb)** — zero ships since Aug 14. Sunshine ships have only ever been entered in bulk at reconciliations (Jun 11: 12 + 4 ship txns, `legacy-unattributed`; Aug 14: 8 + 1, `inv-recon-2026-08-17`). QuickBooks has no Sunshine invoice between Aug 15 and Sep 17 either, so this is either 5,600 cases genuinely on the floor awaiting pickup or product that left unbilled and unrecorded. Also, QuickBooks invoices dated Sep 11–14 (Inter-County 40 cs, Grassland 150 cs, Juliette 110 cs mixed, International Gourmet 15 cs ≈ 4,900 lb) have no ledger ship yet. Both overstate **finished goods**; neither touches batch on-hand.

**Bottom line for the batch excess:** entering packouts is working. The batch overstatement comes from (a) the un-packable per-lb bulk (10,000 lb) and (b) something that is not a ledger-entry omission at all — the prior audit's uncredited bake-loss / shrink hypothesis (yield_multiplier = 1.0) remains the only candidate for the other 19,131 lb, and it needs a weigh-off and a container count, not more data entry.

---

## 1. Cases packed vs cases shipped, per SKU per month (2026-01-01 → 2026-09-16)

Columns: opening = posted balance at start of month (opening at 2026-01-01 is 0 lb on every SKU; the ledger starts in Feb 2026); packed = positive `pack` lines; shipped = negative `ship` lines; adj = net `adjust`; recv = `receive` (all six 2026 FG receives are "FOUND INVENTORY" / "PHYSICAL COUNT" / "Inventory Correction", i.e. stock appearing without a pack); close = balance at month end. Case counts in parentheses use `products.case_size_lb`. **FLAG** = shipped > packed + opening, with the shortfall in lb.

| SKU | Parent batch | Month | Opening lb | Packed lb (cs) | Shipped lb (cs) | Adj lb | Recv lb | Close lb | Flag |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 136 Granola Classic 25 LB | 107 | 2026-02 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 136 Granola Classic 25 LB | 107 | 2026-03 | 600 | 12750 (510) | 8125 (325) |  |  | 5225 |  |
| 136 Granola Classic 25 LB | 107 | 2026-04 | 5225 | 2950 (118) | 2100 (84) |  |  | 6075 |  |
| 136 Granola Classic 25 LB | 107 | 2026-05 | 6075 | 10900 (436) | 5925 (237) | -6550 |  | 4500 |  |
| 136 Granola Classic 25 LB | 107 | 2026-06 | 4500 | 2100 (84) | 4500 (180) |  |  | 2100 |  |
| 136 Granola Classic 25 LB | 107 | 2026-07 | 2100 | 4750 (190) | 5475 (219) |  |  | 1375 |  |
| 136 Granola Classic 25 LB | 107 | 2026-08 | 1375 | 7675 (307) | 3675 (147) | -925 |  | 4450 |  |
| 136 Granola Classic 25 LB | 107 | 2026-09 | 4450 | 5800 (232) | 5500 (220) |  |  | 4750 |  |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-03 | 0 | 1040 (104) | 890 (89) |  | 200 | 350 |  |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-04 | 350 | 100 (10) | 400 (40) |  |  | 50 |  |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-05 | 50 | 1100 (110) | 1660 (166) | 110 | 600 | 200 | **FLAG −510** |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-06 | 200 | 1260 (126) | 1400 (140) | 240 |  | 300 |  |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-07 | 300 | 1800 (180) | 1850 (185) |  |  | 250 |  |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-08 | 250 | 1450 (145) | 910 (91) | -160 |  | 630 |  |
| 137 Granola Crunchy CNS 10 LB Case | 107 | 2026-09 | 630 | 1400 (140) | 330 (33) |  |  | 1700 |  |
| 138 Granola Wheat Free 25 LB | 107 | 2026-03 | 0 | 1250 (50) | 1250 (50) |  |  | 0 |  |
| 138 Granola Wheat Free 25 LB | 107 | 2026-04 | 0 | 1000 (40) | 1000 (40) |  |  | 0 |  |
| 138 Granola Wheat Free 25 LB | 107 | 2026-05 | 0 | 2000 (80) | 3000 (120) |  | 1000 | 0 | **FLAG −1000** |
| 138 Granola Wheat Free 25 LB | 107 | 2026-06 | 0 | 1150 (46) | 1150 (46) |  |  | 0 |  |
| 138 Granola Wheat Free 25 LB | 107 | 2026-07 | 0 | 2000 (80) | 2000 (80) |  |  | 0 |  |
| 138 Granola Wheat Free 25 LB | 107 | 2026-08 | 0 | 2000 (80) | 2000 (80) |  |  | 0 |  |
| 144 CQ Granola 10 LB | 107 | 2026-02 | 0 | 20 (2) | 0 (0) | 5580 |  | 5600 |  |
| 144 CQ Granola 10 LB | 107 | 2026-03 | 5600 | 15400 (1540) | 22400 (2240) |  | 1400 | 0 | **FLAG −1400** |
| 144 CQ Granola 10 LB | 107 | 2026-04 | 0 | 9800 (980) | 0 (0) |  |  | 9800 |  |
| 144 CQ Granola 10 LB | 107 | 2026-05 | 9800 | 7000 (700) | 11200 (1120) |  |  | 5600 |  |
| 144 CQ Granola 10 LB | 107 | 2026-06 | 5600 | 23800 (2380) | 29400 (2940) |  |  | 0 |  |
| 144 CQ Granola 10 LB | 107 | 2026-07 | 0 | 81200 (8120) | 58800 (5880) |  |  | 22400 |  |
| 144 CQ Granola 10 LB | 107 | 2026-08 | 22400 | 22400 (2240) | 44800 (4480) |  |  | 0 |  |
| 144 CQ Granola 10 LB | 107 | 2026-09 | 0 | 5600 (560) | 0 (0) |  |  | 5600 |  |
| 143 Granola Chocolate Chip 25 LB | 108 | 2026-04 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 143 Granola Chocolate Chip 25 LB | 108 | 2026-05 | 600 | 300 (12) | 900 (36) |  |  | 0 |  |
| 140 Granola Cinnamon Almond 25 LB | 109 | 2026-04 | 0 | 600 (24) | 600 (24) |  |  | 0 |  |
| 140 Granola Cinnamon Almond 25 LB | 109 | 2026-05 | 0 | 300 (12) | 300 (12) |  |  | 0 |  |
| 140 Granola Cinnamon Almond 25 LB | 109 | 2026-06 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 140 Granola Cinnamon Almond 25 LB | 109 | 2026-07 | 600 | 0 (0) | 600 (24) |  |  | 0 |  |
| 132 Granola Setton Cocoa Crunch 25 LB | 111 | 2026-05 | 0 | 1500 (60) | 1500 (60) |  |  | 0 |  |
| 139 Granola Cocoa Vibes 25 LB | 111 | 2026-02 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 139 Granola Cocoa Vibes 25 LB | 111 | 2026-06 | 600 | 600 (24) | 0 (0) |  |  | 1200 |  |
| 139 Granola Cocoa Vibes 25 LB | 111 | 2026-07 | 1200 | 0 (0) | 600 (24) |  |  | 600 |  |
| 139 Granola Cocoa Vibes 25 LB | 111 | 2026-08 | 600 | 0 (0) | 0 (0) | -600 |  | 0 |  |
| 133 Granola Setton French Vanilla 25 LB | 112 | 2026-02 | 0 | 4500 (180) | 4500 (180) |  |  | 0 |  |
| 133 Granola Setton French Vanilla 25 LB | 112 | 2026-05 | 0 | 3000 (120) | 3000 (120) |  |  | 0 |  |
| 133 Granola Setton French Vanilla 25 LB | 112 | 2026-07 | 0 | 750 (30) | 0 (0) |  |  | 750 |  |
| 133 Granola Setton French Vanilla 25 LB | 112 | 2026-08 | 750 | 0 (0) | 0 (0) | -750 |  | 0 |  |
| 134 Granola Vanilla Crisp 25 LB (French Vanilla) | 112 | 2026-02 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 134 Granola Vanilla Crisp 25 LB (French Vanilla) | 112 | 2026-05 | 600 | 500 (20) | 0 (0) |  |  | 1100 |  |
| 134 Granola Vanilla Crisp 25 LB (French Vanilla) | 112 | 2026-06 | 1100 | 600 (24) | 0 (0) |  |  | 1700 |  |
| 134 Granola Vanilla Crisp 25 LB (French Vanilla) | 112 | 2026-07 | 1700 | 0 (0) | 1350 (54) |  |  | 350 |  |
| 134 Granola Vanilla Crisp 25 LB (French Vanilla) | 112 | 2026-08 | 350 | 0 (0) | 0 (0) | -350 |  | 0 |  |
| 134 Granola Vanilla Crisp 25 LB (French Vanilla) | 112 | 2026-09 | 0 | 500 (20) | 0 (0) |  |  | 500 |  |
| 135 Granola Vanilla Almond 25 LB | 113 | 2026-03 | 0 | 250 (10) | 250 (10) |  |  | 0 |  |
| 135 Granola Vanilla Almond 25 LB | 113 | 2026-04 | 0 | 600 (24) | 600 (24) |  |  | 0 |  |
| 135 Granola Vanilla Almond 25 LB | 113 | 2026-05 | 0 | 300 (12) | 300 (12) |  |  | 0 |  |
| 135 Granola Vanilla Almond 25 LB | 113 | 2026-06 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 135 Granola Vanilla Almond 25 LB | 113 | 2026-07 | 600 | 125 (5) | 725 (29) |  |  | 0 |  |
| 135 Granola Vanilla Almond 25 LB | 113 | 2026-09 | 0 | 375 (15) | 0 (0) |  |  | 375 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-02 | 0 | 7928 (1057) | 0 (0) |  |  | 7928 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-03 | 7928 | 5625 (750) | 0 (0) |  |  | 13553 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-04 | 13553 | 9000 (1200) | 0 (0) |  |  | 22553 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-05 | 22553 | 11115 (1482) | 0 (0) |  |  | 33668 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-06 | 33668 | 11820 (1576) | 34793 (4639) |  |  | 10695 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-07 | 10695 | 7365 (982) | 0 (0) |  |  | 18060 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-08 | 18060 | 23625 (3150) | 25328 (3377) | 293 |  | 16650 |  |
| 145 Granola SS Chocolate Chip 12x10 OZ Case | 114 | 2026-09 | 16650 | 12075 (1610) | 0 (0) |  |  | 28725 |  |
| 141 Granola Honey Nut 25 LB | 116 | 2026-03 | 0 | 1000 (40) | 1000 (40) |  |  | 0 |  |
| 141 Granola Honey Nut 25 LB | 116 | 2026-04 | 0 | 1600 (64) | 1600 (64) |  |  | 0 |  |
| 141 Granola Honey Nut 25 LB | 116 | 2026-06 | 0 | 1600 (64) | 1000 (40) |  |  | 600 |  |
| 141 Granola Honey Nut 25 LB | 116 | 2026-07 | 600 | 0 (0) | 600 (24) |  |  | 0 |  |
| 141 Granola Honey Nut 25 LB | 116 | 2026-08 | 0 | 1000 (40) | 0 (0) |  |  | 1000 |  |
| 141 Granola Honey Nut 25 LB | 116 | 2026-09 | 1000 | 375 (15) | 1000 (40) |  |  | 375 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-02 | 0 | 2753 (367) | 0 (0) |  |  | 2753 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-03 | 2753 | 2018 (269) | 0 (0) |  |  | 4770 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-04 | 4770 | 1980 (264) | 0 (0) |  |  | 6750 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-06 | 6750 | 2250 (300) | 6750 (900) |  |  | 2250 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-07 | 2250 | 2430 (324) | 0 (0) |  |  | 4680 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-08 | 4680 | 4650 (620) | 113 (15) | -4568 |  | 4650 |  |
| 146 Granola SS Original 12x10 OZ Case | 116 | 2026-09 | 4650 | 8558 (1141) | 0 (0) |  |  | 13208 |  |
| 183 Granola SS Original Bulk per/lb | 116 | 2026-08 | 0 | 1875 (lb) | 1875 (lb) |  |  | 0 |  |
| 147 Granola SS Cranberry 12x10 OZ Case | 118 | 2026-07 | 0 | 1343 (179) | 0 (0) |  |  | 1343 |  |
| 147 Granola SS Cranberry 12x10 OZ Case | 118 | 2026-08 | 1343 | 2025 (270) | 2123 (283) | 780 |  | 2025 |  |
| 148 Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 119 | 2026-03 | 0 | 1050 (140) | 0 (0) |  |  | 1050 |  |
| 148 Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 119 | 2026-06 | 1050 | 0 (0) | 1050 (140) |  |  | 0 |  |
| 148 Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 119 | 2026-07 | 0 | 1050 (140) | 0 (0) |  |  | 1050 |  |
| 148 Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 119 | 2026-08 | 1050 | 1365 (182) | 2618 (349) | 1568 |  | 1365 | **FLAG −203** |
| 149 Granola SS Original Low Carb 12x10 OZ Case | 120 | 2026-03 | 0 | 1065 (142) | 0 (0) |  |  | 1065 |  |
| 149 Granola SS Original Low Carb 12x10 OZ Case | 120 | 2026-06 | 1065 | 0 (0) | 1065 (142) |  |  | 0 |  |
| 149 Granola SS Original Low Carb 12x10 OZ Case | 120 | 2026-07 | 0 | 975 (130) | 0 (0) |  |  | 975 |  |
| 149 Granola SS Original Low Carb 12x10 OZ Case | 120 | 2026-08 | 975 | 2033 (271) | 2498 (333) | 1523 |  | 2033 |  |
| 150 BS Granola – Dark Chocolate – 6x7 OZ Case | 121 | 2026-03 | 0 | 3408 (1296) | 3408 (1296) |  |  | 0 |  |
| 150 BS Granola – Dark Chocolate – 6x7 OZ Case | 121 | 2026-04 | 0 | 3766 (1432) | 0 (0) |  |  | 3766 |  |
| 150 BS Granola – Dark Chocolate – 6x7 OZ Case | 121 | 2026-05 | 3766 | 0 (0) | 3932 (1495) | 166 |  | 0 | **FLAG −166** |
| 150 BS Granola – Dark Chocolate – 6x7 OZ Case | 121 | 2026-07 | 0 | 1136 (432) | 1286 (489) | 150 |  | 0 | **FLAG −150** |
| 151 BS Granola – Peanut Butter Banana – 6x7 OZ Case | 122 | 2026-02 | 0 | 0 (0) | 4176 (1588) | 4216 |  | 40 | **FLAG −4176** |
| 151 BS Granola – Peanut Butter Banana – 6x7 OZ Case | 122 | 2026-03 | 40 | 379 (144) | 419 (159) |  |  | 0 |  |
| 151 BS Granola – Peanut Butter Banana – 6x7 OZ Case | 122 | 2026-07 | 0 | 3614 (1374) | 6241 (2373) | 2627 |  | 0 | **FLAG −2627** |
| 209 BS Granola – Peanut Butter Banana – 6x8 OZ Case | 122 | 2026-03 | 0 | 2406 (802) | 0 (0) |  |  | 2406 |  |
| 209 BS Granola – Peanut Butter Banana – 6x8 OZ Case | 122 | 2026-09 | 2406 | 0 (0) | 0 (0) | -2406 |  | 0 |  |
| 152 BS Almond Butter Granola – 6x7 OZ Case | 123 | 2026-03 | 0 | 3708 (1410) | 6359 (2418) | 2651 |  | 0 | **FLAG −2651** |
| 153 BS Granola – Hazelnut Butter – 6x7 OZ Case | 124 | 2026-03 | 0 | 2456 (934) | 2428 (923) |  |  | 28 |  |
| 153 BS Granola – Hazelnut Butter – 6x7 OZ Case | 124 | 2026-05 | 28 | 4587 (1744) | 4581 (1742) |  |  | 33 |  |
| 153 BS Granola – Hazelnut Butter – 6x7 OZ Case | 124 | 2026-08 | 33 | 0 (0) | 0 (0) | -33 |  | 0 |  |
| 206 BS Granola – Hazelnut Butter – 6x8 OZ Case | 124 | 2026-03 | 0 | 2400 (800) | 2400 (800) |  |  | 0 |  |
| 142 Granola Fruit Nut 25 LB | 179 | 2026-02 | 0 | 25 (1) | 0 (0) |  |  | 25 |  |
| 142 Granola Fruit Nut 25 LB | 179 | 2026-05 | 25 | 500 (20) | 525 (21) |  |  | 0 |  |
| 142 Granola Fruit Nut 25 LB | 179 | 2026-06 | 0 | 600 (24) | 0 (0) |  |  | 600 |  |
| 142 Granola Fruit Nut 25 LB | 179 | 2026-07 | 600 | 0 (0) | 600 (24) |  |  | 0 |  |
| 142 Granola Fruit Nut 25 LB | 179 | 2026-09 | 0 | 500 (20) | 0 (0) |  |  | 500 |  |
| 285 Granola SS Classic #9 Bulk per/lb | 283 | 2026-08 | 0 | 12115 (lb) | 12115 (lb) |  |  | 0 |  |
| 286 Granola SS Classic #9 25 LB | 283 | 2026-08 | 0 | 1575 (63) | 0 (0) | -1575 |  | 0 |  |
| 171 Classic Granola 25 LB | — | 2026-02 | 0 | 0 (0) | 9100 (364) | 9105 |  | 5 | **FLAG −9100** |

### 1a. What covered each flagged shortfall

| SKU | Month | Shortfall lb | Covered by | Batch never debited |
|---|---|---:|---|---|
| 171 Classic Granola 25 LB (no parent) | 2026-02 | 9,100 | adjust +9,105 "Found inventory: found_during_count" Feb 6–9 (tx 110/111/131–134), same days as the Quali-Pack ships | system-start baseline, before the Feb 12 count |
| 151 BS PB Banana 6x7 | 2026-02 | 4,176 | adjust +756 found_during_count (Feb 10) + 3,459.75 "predates_system" (Feb 12) | system-start baseline |
| 144 CQ Granola 10 LB | 2026-03 | 1,400 | receive +1,400 shipper "PHYSICAL COUNT" Mar 23 (tx 516) | 107 |
| 152 BS Almond Butter 6x7 | 2026-03 | 2,651 | adjust +2,651.04 "Manual pack adjustment" Mar 17 (tx 460) | 123 |
| 137 Crunchy CNS 10 LB | 2026-05 | 510 | receive +600 "FOUND INVENTORY"/"FOUND" May 15/21 (tx 965/967/1031) + adjust +110 "count correction during order fulfillment" May 21 (tx 1032) | 107 |
| 138 Wheat Free 25 LB | 2026-05 | 1,000 | receive +1,000 "FOUND INVENTORY" May 26 (tx 1051) | 107 |
| 150 BS Dark Choc 6x7 | 2026-05 | 166 | adjust +165.69 "28220-I GRDC under-pack 63 cs" (recon, tx 1953) | 121 |
| 150 BS Dark Choc 6x7 | 2026-07 | 150 | adjust +149.91 "28337-I GRDC under-pack 57 cs" (recon, tx 1952) | 121 |
| 151 BS PB Banana 6x7 | 2026-07 | 2,627 | adjust +2,627.37 "28337-I GRPB under-pack 999 cs" (recon, tx 1951) | 122 |
| 148 SS CC Low Carb 12x10 | 2026-08 | 203 | adjust +1,567.5 "QB Low Carb under-pack 209 cs (964 billed)" Aug 14 (tx 1955) | 119 |

Under-packs that do not show at month granularity because the month's packs (entered after the Aug 14 recon ship) covered them, but which were nonetheless posted as `adjust` rather than `pack`: 149 SS Original Low Carb +1,522.5 lb (203 cs, tx 1954, batch 120); 147 SS Cranberry +780 lb (104 cs, tx 1956, batch 118); 145 SS CC +292.5 lb (39 cs, count 600 vs 561, tx 2002, batch 114); 136 Classic 25 LB +625 lb "count correction during order fulfillment" May 19 (tx 1005, batch 107); 137 Crunchy +240 lb "Inventory correction" Jun 18 (tx 1382, batch 107); 137 receive +200 "Inventory Correction" Mar 20 (tx 496, batch 107).

Not counted as missing packs: the May 19–20 zero-out / re-find pairs on 136 (−11,475 / +6,000, net −6,550, count corrections), the Aug 14 count-to-floor entries (136 −1,375/+125/+325, 146 −4,567.5, 133/139/134/137/153 to zero), the Mar 18 duplicate-and-void pair on 206 (+2,400/−2,400), and the Sep 15 write-off of the discontinued 6x8 format on 209 (−2,406).

**No FLAG, no positive adjust, and no found-inventory receive on any granola FG SKU after 2026-08-14.**

---

## 2. Current finished-goods on-hand per SKU vs open sales orders (posted, 2026-09-17)

Open demand = `sales_orders.state = 'open'` lines with `line_status IN ('pending','partial')`, quantity − shipped.

| SKU | Parent | On-hand lb | On-hand cases | Open SO lb | Open orders (requested ship) | Flag |
|---|---|---:|---:|---:|---|---|
| 136 Granola Classic 25 LB | 107 | 4,750 | 190 | 19,750 | Juliette 1,000 (Sep 22); Grassland 3,750 (Sep 23), 3,000 ×5 (Sep 28 → Nov 26) | Low vs total, but exactly covers the two orders due by Sep 23; QB invoiced 150 cs to Grassland and 40 to Juliette on Sep 14 with no ledger ship yet |
| 137 Granola Crunchy CNS 10 LB | 107 | 1,700 | 170 | 850 | Inter-County 400 (Sep 9) + 300 (Sep 28); Intl Gourmet 150 (Sep 24) | OK (QB shows Inter-County 40 cs and Intl Gourmet 15 cs invoiced Sep 11/14, not yet in ledger) |
| 138 Granola Wheat Free 25 LB | 107 | 0 | 0 | 0 | | OK |
| 144 CQ Granola 10 LB | 107 | 5,600 | 560 | 0 | | OK |
| 129 Setton Good Ol 25 LB | 107 | 0 | 0 | 0 | | no activity |
| 143 Granola Chocolate Chip 25 LB | 108 | 0 | 0 | 0 | | no activity since May |
| 131 / 140 (Cinnamon Almond) | 109 | 0 | 0 | 0 | | |
| 132 / 139 (Cocoa Crunch / Cocoa Vibes) | 111 | 0 | 0 | 0 | | |
| 133 Setton French Vanilla 25 LB | 112 | 0 | 0 | 0 | | |
| 134 Vanilla Crisp 25 LB | 112 | 500 | 20 | 2,375 | Juliette 500 (Sep 22); Grassland 625 ×3 (Sep 23, Oct 28, Nov 26) | **Low**: 625 lb short for Grassland Sep 23 unless packed this week; QB shows 20 cs invoiced to Juliette Sep 14 |
| 135 Vanilla Almond 25 LB | 113 | 375 | 15 | 375 | Juliette 375 (Sep 22) | OK (invoiced Sep 14, ship not entered) |
| 145 SS Chocolate Chip 12x10 OZ | 114 | **28,725** | **3,830** | 4,500 | Sunshine SO-260909-001 PO 09062026 (Sep 16) | **Implausibly high**: 3,230 cs packed since Aug 14, 0 shipped; no Sunshine QB invoice since Aug 15 |
| 141 Honey Nut 25 LB | 116 | 375 | 15 | 375 | Juliette 375 (Sep 22) | OK |
| 146 SS Original 12x10 OZ | 116 | **13,208** | **1,761** | 11,250 | Sunshine SO-260909-001 (Sep 16) | **High**: 1,561 cs packed since Aug 14, 0 shipped |
| 183 SS Original Bulk per/lb | 116 | 0 | — | 0 | | |
| 185 SS Mini 100 | 116 | 0 | — | 0 | | no weight, no activity |
| 147 SS Cranberry 12x10 OZ | 118 | 2,025 | 270 | 0 | | packed Aug 25+, unshipped |
| 148 SS CC Low Carb 12x10 OZ | 119 | 1,365 | 182 | 0 | | packed, unshipped |
| 149 SS Original Low Carb 12x10 OZ | 120 | 2,033 | 271 | 0 | | packed, unshipped |
| 150 / 208 BS Dark Choc 6x7 / 6x8 | 121 | 0 | 0 | 0 | | |
| 151 / 209 BS PB Banana 6x7 / 6x8 | 122 | 0 | 0 | 0 | | 209 written off Sep 15 |
| 152 / 207 BS Almond Butter | 123 | 0 | 0 | 0 | | |
| 153 / 206 BS Hazelnut | 124 | 0 | 0 | 0 | | |
| 142 Fruit Nut 25 LB | 179 | 500 | 20 | 500 | Juliette 500 (Sep 22) | OK |
| 285 SS Classic #9 Bulk per/lb | 283 | 0 | — | **4,000** | Sunshine SO-260817-001 (Aug 28) | **Cannot be packed** (NULL case weight); batch 107 never debited |
| 286 / 287 SS Classic #9 25 LB / 10 LB | 283 | 0 | 0 | 0 | | 286's one pack (Aug 12) was reversed as wrong output product |
| 288 SS Classic CC #9 Bulk per/lb | 284 | 0 | — | **6,000** | Sunshine SO-260817-001 (Aug 28) | **Cannot be packed**; batch 108 never debited |
| 289 / 290 SS CC #9 25 LB / 10 LB | 284 | 0 | 0 | 0 | | |
| 171 Classic Granola 25 LB | — | 5 | 0.2 | 0 | | dust |

**Negatives: none.** Nothing in this table is low in a way that suggests a physical pack happened without a ledger pack. The two Sunshine 12x10 SKUs are the reverse: the ledger has far more finished goods than any plausible floor, because Sunshine shipments are entered only at reconciliation time.

### 2a. Since the Aug 14 count — FG roll-forward and batch roll-forward, side by side

| Parent batch | FG SKUs | FG bal Aug 14 | FG packed since | FG shipped since | FG now | Batch bal Aug 14 | Batch made since | Batch packed out since | Batch now |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 107 Classic #9 | 136, 137, 138, 144 | 10,300 | 32,240 | 30,490 | 12,050 | 10,013 | 37,145 | 32,240 (+40 to Fruit Nut) | 14,878 |
| 108 Classic CC #9 | 143, (288) | 0 | 0 | 0 | 0 | 0 | 9,396 | 0 | 9,396 |
| 114 SS CC #2 | 145 | 4,500 | 24,225 | 0 | 28,725 | 3,930 | 25,152 | 24,225 | 4,857 |
| 116 SS Original #1 | 141, 146 | 1,500 | 13,083 | 1,000 | 13,583 | 0 | 15,050 | 13,083 | 1,968 |
| 118 Cranberry #3 | 147 | 0 | 2,025 | 0 | 2,025 | 379 | 1,895 | 2,025 | 249 |
| 119 LC CC #8 | 148 | 0 | 1,365 | 0 | 1,365 | 0 | 1,400 | 1,365 | 35 |
| 120 LC Original #7 | 149 | 0 | 2,033 | 0 | 2,033 | 0 | 2,100 | 2,033 | 68 |
| 112 Vanilla Crisp #16 | 134 | 0 | 500 | 0 | 500 | 0 | 740 | 500 | 240 |
| 113 Vanilla Almond | 135 | 0 | 375 | 0 | 375 | 0 | 380 | 375 | 5 |
| 179 Fruit Nut | 142 | 0 | 500 | 0 | 500 | 169 | 769 | 500 | 438 |

FG "packed since" equals batch "packed out since" on every row — every pack that credited a finished SKU debited its batch by the same pounds. On the small batches (118/119/120/113) the batch residue after packing is 5–68 lb, i.e. one batch ≈ one pack, entered. The excess is concentrated where the prior audit put it (107/108/114/116), and on those rows the packs are entered; what is not entered is either impossible (108 → 288 bulk) or not a pack at all.

---

## 3. Ship days with no pack of that SKU in the prior 14 days

14 SKU-days since 2026-01-01. "Balance before" = posted FG balance at start of the ship day.

| SKU | Ship date | Shipped lb | Customer | Last pack on/before | Balance before | Reading |
|---|---|---:|---|---|---:|---|
| 171 Classic Granola 25 LB | 2026-02-09 | 9,100 | Quali Pack | never | 120 | system-start; covered by +9,000 found_during_count same day |
| 151 BS PB Banana 6x7 | 2026-02-20 | 4,176 | Blue Stripes | never | 4,216 | system-start; covered by Feb 10/12 found/predates_system adjusts |
| 144 CQ Granola 10 LB | 2026-03-05 | 5,600 | Restaurant Depot Haines City | 2026-02-16 | 5,600 | stock on hand from Feb packs — OK |
| 150 BS Dark Choc 6x7 | 2026-05-12 | 3,932 | Blue Stripes | 2026-04-22 | 3,766 | **166 lb short** → recon adjust "under-pack 63 cs" |
| 132 Setton Cocoa Crunch 25 LB | 2026-05-21 | 1,500 | Setton Farms | 2026-05-06 | 1,500 | OK (packed 15 days prior) |
| 133 Setton French Vanilla 25 LB | 2026-05-21 | 3,000 | Setton Farms | 2026-05-06 | 3,000 | OK |
| 136 Granola Classic 25 LB | 2026-06-04 | 1,500 | Grassland | 2026-05-20 | 4,500 | OK |
| 146 SS Original 12x10 | 2026-06-11 | 6,750 | Sunshine | 2026-04-16 | 6,750 | OK — Feb–Apr packs held until the Jun 11 bulk ship entry |
| 148 SS CC Low Carb 12x10 | 2026-06-11 | 1,050 | Sunshine | 2026-03-05 | 1,050 | OK (3-month-old pack) |
| 149 SS Original Low Carb 12x10 | 2026-06-11 | 1,065 | Sunshine | 2026-03-04 | 1,065 | OK |
| 134 Vanilla Crisp 25 LB | 2026-07-22 | 750 | Grassland | 2026-06-30 | 1,100 | OK |
| 147 SS Cranberry 12x10 | 2026-08-14 | 2,123 | Sunshine | 2026-07-08 | 1,343 | **780 lb short** → recon adjust "under-pack 104 cs" |
| 148 SS CC Low Carb 12x10 | 2026-08-14 | 2,618 | Sunshine | 2026-07-08 | 1,050 | **1,568 lb short** → recon adjust "under-pack 209 cs" |
| 149 SS Original Low Carb 12x10 | 2026-08-14 | 2,498 | Sunshine | 2026-07-08 | 975 | **1,523 lb short** → recon adjust "under-pack 203 cs" |

Four of the fourteen are true under-packs (all in the Aug 14/17 reconciliation's "QB under-pack" set, plus the May 12 Dark Choc one), all filled by `adjust` instead of `pack`. The Sunshine rows show the entry pattern behind §2: packs are entered as they happen, ships are entered months later in one batch. **No ship since Aug 14 lacks a pack in the prior 14 days.**

---

## 4. Re-check: pack transactions whose batch debit ≠ cases × case weight

256 posted pack transactions since 2026-01-01 credit a granola FG SKU. For 254 of them the note's "Pack N cases", N × `case_size_lb`, the FG credit, and the batch debit are all equal to the pound. Exceptions:

| Tx | Date | FG | Cases × weight | FG credit lb | Batch/ingredient debit lb | Δ | Note |
|---|---|---|---:|---:|---:|---:|---|
| 501 | 2026-03-20 | 209 BS PB Banana 6x8 | 802 × 3.00 = 2,406 | 2,406 | 3,104.49 = Dark Choc batch 121: 2,406 (3 lots) + BS Peanut Butter Chips 457.62 + BS Banana Bites 240.87 | **+698.49 over-debit** | Legacy WIP-mix BOM: PBB was built from a Dark Choc base plus inclusions and both were debited. Over-debit, not under; pre-count; 121 and 122 both since counted. |
| 1964 | 2026-08-14 | 285 Classic #9 Bulk per/lb | no case count (per/lb) | 12,115 | 12,115 from 107 across 14 lots | 0 | recon true-up; debit = FG lb |
| 1966 | 2026-08-14 | 183 SS Original Bulk per/lb | no case count (per/lb) | 1,875 | 1,875 from 116 | 0 | recon true-up; debit = FG lb |

Packs whose batch debit landed on a batch other than the FG's parent (cross-batch, debit still equal to FG lb): tx 486 (Mar 19, 151 from 121), 501 (above), **512 and 513 (Mar 20, 136 Classic 25 LB 625 lb and 138 Wheat Free 1,000 lb debited from 121 Dark Choc instead of 107)**, 748 (Apr 29, 135 Vanilla Almond from 112 instead of 113), 1907 (Aug 12, 286 from 107 — reversed the same day, "system assigned the wrong output product"), 1964 (285 from 107 rather than its nominal parent 283 — correct in substance). Tx 512/513 mean 107 was under-debited 1,625 lb and 121 over-debited in March; both were zeroed by the Jun 8 count, so nothing survives.

**Conclusion for §4 unchanged from the prior audit:** the pack path debits the batch correctly. There is no pack whose batch debit is smaller than cases × case weight.

---

## 5. Shortfalls converted to pounds per batch product vs the unexplained excess

"Shortfall" here = finished goods that entered the ledger by `adjust` or found-inventory `receive` instead of by `pack` (so the batch was never debited), 2026-01-01 → 2026-09-16, excluding the February system-start baseline (171: 9,105; 151: 4,215.75; 144: net 5,580) which precedes the Feb 12 first count.

| Batch product | Missing-pack lb, Feb 12 → Aug 14 | Missing-pack lb after Aug 14 | Ledger on-hand now | Prior audit "unexplained" | Of which un-packable bulk (SO-260817-001) | Left after bulk | Explained by missing packs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 107 Classic #9 | 4,175 (136: 625; 137: 1,150; 138: 1,000; 144: 1,400) | 0 | 14,878 | 14,878 | 4,000 | 10,878 | **0** |
| 108 Classic CC #9 | 0 | 0 | 9,396 | 9,396 | 6,000 | 3,396 | **0** |
| 114 SS CC #2 | 293 | 0 | 4,857 | 4,857 | 0 | 4,857 | **0** |
| 116 SS Original #1 | 0 | 0 | 1,968 | — | 0 | — | 0 |
| 118 Cranberry #3 | 780 | 0 | 249 | — | | | 0 |
| 119 LC CC #8 | 1,568 | 0 | 35 | — | | | 0 |
| 120 LC Original #7 | 1,523 | 0 | 68 | — | | | 0 |
| 121 BS Dark Choc | 316 | 0 | 1,050 | — | | | 0 |
| 122 BS PB Banana | 2,627 | 0 | 2 | — | | | 0 |
| 123 BS Almond Butter | 2,651 | 0 | 0 | — | | | 0 |
| **Total** | **13,932** | **0** | **32,503** | **29,131** | **10,000** | **19,131** | **0** |

Why the pre-Aug-14 column does not carry forward: each of those batches was written to a physical count after the under-pack occurred (Jun 8 and Aug 14 for 107/114; Aug 14 for 118–123; 123 has been at 0 since). The under-packs were part of what those counts wrote off. For 107 they are 4,175 of the 27,725 lb written down on Jun 8 + Aug 14 (15 %); for 114, 293 of 29,207 (1 %). So even historically, skipped packouts were a minor contributor; the bulk of the write-downs was something else.

---

## 6. Evidence pointers

* Ship ↔ QuickBooks reconciliation Aug 15 → Sep 9 (ledger = QB exactly): Classic 25 LB 274 cs / 6,850 lb; Crunchy 10 LB 64 cs / 640 lb; CQ 10 LB 2,100 cs / 21,000 lb; Wheat Free 80 cs / 2,000 lb; Honey Nut 40 cs / 1,000 lb. QB invoices after Sep 9 with no ledger ship: 28368-I (Inter-County, 40 cs Crunchy), 28384-I (Grassland, 150 cs Classic), 28383-I (Juliette, 40 Classic / 20 French Vanilla / 15 Honey Nut / 15 Vanilla Almond / 20 Fruit Nut), 28388-I (Intl Gourmet, 15 cs Crunchy). No Sunshine invoice in the window.
* SO-260817-001 (id 298): lines 693 (288, 6,000 lb) / 694 (285, 4,000 lb), `sales_order_shipments` none, status `confirmed`, state `open`, updated 2026-09-10. SO-260909-001 (id 329): 145 4,500 lb + 146 11,250 lb, requested 2026-09-16, 0 shipped.
* Sunshine ship entry history for 145/146: 2026-06-11 (34,792.5 + 6,750 lb, 16 txns, `legacy-unattributed`), 2026-08-14 (25,327.5 + 112.5 lb, 9 txns, `inv-recon-2026-08-17`). Nothing else in 2026.
* FG adjust/receive evidence: tx 460, 496, 516, 965, 967, 1005, 1031, 1032, 1051, 1382, 1951–1956, 2002 (all listed in §1a).
* Pack debit check: tx 501 lines (121 lots 322/279/274, 69 lot 22, 56 lot 11, 209 lot 343); cross-batch packs 486, 512, 513, 748, 1907/1910.
* Prior report: `audits/reports/granola-batch-onhand-audit.md` §2 "Since the last count" table (107: 14,878; 108: 9,396; 114: 4,857) and §4a–4c.
* Query wrapper used: `BEGIN; SET TRANSACTION READ ONLY; … COMMIT;` on the 5432 session pooler, `psql -v ON_ERROR_STOP=1`.

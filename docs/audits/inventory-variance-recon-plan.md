# Inventory variance — reconciliation plan

**STATUS: PLAN ONLY. DO NOT POST.** No production writes.  
**Revised:** 2026-08-17 15:24 EDT (QB item-level scan + Arturo Aug 14 count file)  
**Prior draft:** 2026-08-17 15:14 EDT (superseded numbers struck in the changelog, not here)  
**Ledger snapshot:** production, read-only, 2026-08-17 15:10–15:24 EDT  
**Sources:** QB item-level billed quantities (this request); `docs/audits/physical-count-2026-08-14.md`; `docs/audits/inventory-variance-txn-history.md`; live `products` / posted lots.

Every proposed line is in the [ordered execution checklist](#6-ordered-execution-checklist). Do not execute until each line is accepted.

---

## 0. Naming (unchanged)

| SKU | id | Canonical name | QB |
|---|---:|---|---|
| **70073** | 151 | BS Granola – Peanut Butter Banana – 6x7 OZ Case | GRPB |
| **70074** | 150 | BS Granola – Dark Chocolate – 6x7 OZ Case | GRDC |
| **70080** | 153 | BS Granola – Hazelnut Butter – 6x7 OZ Case | GRHB |

70010 = SS Original Low Carb 12x10; 70070 = SS Chocolate Chip Low Carb 12x10; 70011 = SS Cranberry 12x10.  
70013 = Granola SS Classic #9 Bulk per/lb. 70004 = Granola SS Original Bulk per/lb (`#1 Bulk`).  
70006 = Granola SS Mini 100 (`case_size_lb` **NULL**, `uom` = lb, parent batch = **90016 SS Original #1**, **zero ledger lines**). There is **no** Mini 100 #9 or Mini 100 #9 ChocChip product.

---

## Ground rules

1. **Do not void** `SUNSHINE-RECON-2026` (txns 1325–1342). Net new Sunshine ships against it.
2. **Do not ship** open SOs `SO-260814-002`, `SO-260817-001`, `SO-260629-003` as the recon vehicle.
3. Customers: Sunshine Granola = 217; Blue Stripes = 17.
4. Case math: SS 12x10 = **7.5 lb**; BS 6x7 = **2.63 lb**; Classic 25 LB = **25**; coconut/graham 10 LB = **10**; coconut 25 LB = **25**.
5. Batches: 90002 = 323; 90001 = 348; 90011 = 393; 95002 = 350; 90016 = 350; 90013 = 379; 90020 = 338; 90024 = 370; 90010 = 380; 90019 = 330; 90015 = 350.
6. Re-snapshot on-hand immediately before any future write.

---

## Count floors (from `physical-count-2026-08-14.md`)

| SKU | Floor | Notes |
|---|---:|---|
| 70003 | 600 cs on BB081027 | |
| 70002 | 200 cs on BB081027 | |
| 70010 / 70070 / 70011 | 0 | |
| 70073 / 70074 / 70080 | 0 | |
| 70050 | 18 cs (MAY 13 ×5, JUL 21 ×13) | ledger lot is AUG 11 |
| 70082 / 70059 / 70052 / 1614 / 10300 | 0 | |
| 31012 | 280 cs unstamped (Clark) | ledger 420 on 608101 |
| 893 | 700 cs | AUG 12 280 / AUG 13 140 / AUG 14 280 |
| 67476 | **317 cs** | lot tags 140+97=237 — SKU total used, lot move held |
| 67470 | 26 cs AUG 13 | |
| 10020 | 0 | |
| 10029 | 50 cs JUL 27 | |
| 67473 | 60 cs AUG 13 | |
| 10002 | 16 cs AUG 04 | |
| 10001 / 10007 / 10006 | 0 | |
| 10010 | 6 cs JUL 27 | |
| 90002 | 31 batches (10,013 lb) | 10 containers |
| 90001 | 0 | |
| 90011 | 10 batches (3,930 lb) | 3 containers; equals lot AUG 14 |
| 95002 | 3 batches (1,050 lb) | 1 container |
| 90016 / 90020 / 90024 / 90010 / 90019 / 90015 | 0 | |
| 90013 | 1 batch (379 lb) | 0.25 container |
| 95005 | **not a granola floor** | 1 container banana/PB-chip mix |
| 70088 6x8 / coconut batches / 90008 | **no floor** | not on the sheet |

---

## 1. Backdated shipment entries (QB = physical ships)

### 1.1 Sunshine 70003 — corrected billed 8,016 cs

| | cases | lb |
|---|---:|---:|
| QB billed YTD (item-level scan) | **8,016** | 60,120.00 |
| Already posted via `SUNSHINE-RECON-2026` | 4,639 | 34,792.50 |
| **Additional ships** | **3,377** | **25,327.50** |

Prior draft used 7,716 / extra 3,077. That is obsolete.

Invoice refs still the listed set (no per-invoice qty): 28259, 28260, 28263, 28266, 28268, 28265, 28183, 28181, 28276, 28277, 28323, 28295, 28297, 28320, 28321, 28322, 28257. Header: `order_reference = QB-70003-YTD-NET`, date 2026-08-14, notes include *billed 8,016 − recon 4,639*.

FIFO on remaining lots:

| Line | Lot | Ship cs | Ship lb | Left |
|---|---|---:|---:|---:|
| S-70003-a | BB060827 | 376 | 2,820.00 | 0 |
| S-70003-b | BB061527 | 495 | 3,712.50 | 0 |
| S-70003-c | BB062227 | 555 | 4,162.50 | 0 |
| S-70003-d | BB070827 | 150 | 1,125.00 | 0 |
| S-70003-e | BB071327 | 382 | 2,865.00 | 0 |
| S-70003-f | BB072027 | 450 | 3,375.00 | 0 |
| S-70003-g | BB080327 | 557 | 4,177.50 | 0 |
| S-70003-h | BB081027 | **412** | **3,090.00** | **561 cs** |
| **Total** | | **3,377** | **25,327.50** | |

On-hand after this ship: **561 cs**. Count: **600**. Residual **+39 cs** (§3.1) — upward, not a further deduction.

### 1.2 Sunshine 70002 — unchanged from prior draft

QB still ~915 vs recon 900 → extra **15 cs / 112.50 lb** from BB061527. Confirm 915. After ship: 809 cs. Floor 200 → residual −609 (§3.2).

### 1.3 Sunshine Low Carb 10oz/12 — 964 cs billed (new)

QB does not split Original vs Choc Chip Low Carb. Ledger history:

| SKU | Lifetime packed | Recon shipped | On-hand | Never billed on ledger after recon |
|---|---:|---:|---:|---|
| 70010 Orig LC | 272 cs (142+130) | 142 | 130 on BB070827 | 130 |
| 70070 CC LC | 280 cs (140+140) | 140 | 140 on BB070827 | 140 |
| **Combined** | **552** | **282** | **270** | **270** |

Billed **964**. Already deducted **282**. Additional to record **682**. Packed only **552**, so billed exceeds lifetime pack by **412 cs** (same class as GRPB).

Allocation of the 964 by lifetime pack mix (272 : 280):

| | 70010 | 70070 | Total |
|---|---:|---:|---:|
| Allocated billed | 475 | 489 | 964 |
| Already in recon | 142 | 140 | 282 |
| Additional ship | 333 | 349 | 682 |
| On-hand available | 130 | 140 | 270 |
| **Pack under-record** | **203** | **209** | **412** |
| After pack+ and ship | 0 | 0 | 0 = floor |

Under-record lb: 203 × 7.5 = **1,522.50**; 209 × 7.5 = **1,567.50**; total **3,090.00**.  
Additional ship lb: 333 × 7.5 = **2,497.50**; 349 × 7.5 = **2,617.50**; total **5,115.00**.

Do not force 90015 / 90014 negative to feed the 412 cs. Those batches are already at (or going to) their count floors. FG-only pack+, then ship, same as GRPB.

Date 2026-08-14, `order_reference = QB-LOWCARB-YTD-NET`. If a later invoice split exists, replace 475/489 but keep the 964 total.

### 1.4 Sunshine Cranberry 70011 — 283 cs billed, zero ever shipped

| | cs | lb |
|---|---:|---:|
| Lifetime packed (txn 1590, 2026-07-08, BB070827) | 179 | 1,342.50 |
| Posted ships | 0 | 0 |
| QB billed | 283 | 2,122.50 |
| **Pack under-record** | **104** | **780.00** |
| Then ship | 283 | 2,122.50 |
| After | 0 | 0 = floor |

FG-only pack+ on BB070827 (or a billed lot if QB names one). 90013 is taken to its 1-batch floor separately; do not pull it negative for these 104 cs.

### 1.5 Blue Stripes — unchanged from prior draft

Missing-pack prefixes, then standalone ships. Lot-alias: 70074 invoice 1,432 on SW2611090 = ledger 1,008 `SW2611090` + 424 `MAR 20 2026`.

| Date | Invoice | SKU | Lot | cs | lb | First? |
|---|---|---|---|---:|---:|---|
| 2026-07-29 | 28337-I | 70073 | SW2620891 | +999 | +2,627.37 | pack+ (GRPB under-pack, §4) |
| 2026-07-24 | 28337-I | 70074 | SW2620590 | +57 | +149.91 | pack+ |
| 2026-05-12 | 28220-I | 70074 | SW2607890 | +63 | +165.69 | pack+ |
| 2026-05-12 | 28220-I | 70080 | SW2612692 | −1,742 | −4,581.46 | ship |
| 2026-05-12 | 28220-I | 70074 | SW2611090 | −1,008 | −2,651.04 | ship |
| 2026-05-12 | 28220-I | 70074 | MAR 20 2026 | −424 | −1,115.12 | ship |
| 2026-05-12 | 28220-I | 70074 | SW2607890 | −63 | −165.69 | ship |
| 2026-07-30 | 28337-I | 70073 | SW2620891 | −2,373 | −6,240.99 | ship |
| 2026-07-30 | 28337-I | 70074 | SW2620590 | −489 | −1,286.07 | ship |

---

## 2. Batch true-ups — billed bulk/mini first, generic consume only the remainder

### 2.1 Mini 100 unit weight — **does not exist on the product**

`70006 Granola SS Mini 100`: `case_size_lb` is NULL, `uom` is lb, no bag weight, no pack_format, **zero posted lines**. Parent batch is **90016 (#1)**, not Classic #9 and not CC #9.

QB: **4,186 units Mini 100 #9 ChocChip** and **995 units #9/Mini 100**. Those items are not in `products`.

**Cannot convert units → lb without inventing a weight.** Do not assume 1 oz/bag or 6.25 lb/case.

Checklist holds two designation lines (units only). Until a weight is supplied, **none** of the 17,748 / 24,303 lb batch drops is labeled mini. When a weight `W` arrives: move `4186 × W` off the 90001 generic consume onto a mini-ship, and `995 × W` off the 90002 generic consume. If `4186 × W` exceeds 17,748, the excess is a CC #9 make/pack under-record (GRPB pattern).

### 2.2 Classic #9 (90002) — 106.2415 → 31

Consume **24,303.00 lb**. Keep AUG 13 18.00 + AUG 12 13.00.

QB billed **12,115 lb “Original (#9) Bulk per/lb”** = **70013** (zero ledger lines). Designate that slice as a real bulk pack+ship. Remainder **12,188.00 lb** is generic unrecorded consumption.

Assume newest 31 stay. FIFO older lots:

**A. Source of the 70013 bulk pack (12,115 lb)** — oldest first, through 979 lb of JUL 27:

| Lot | lb into 70013 |
|---|---:|
| JUN 30 2026 | 224.00 |
| Jul 01 2026 | 753.00 |
| JUL 2 2026 | 429.00 |
| JUL 06 2026 | 1,075.00 |
| JUL 07 2026 | 1,726.00 |
| JUL 10 2026 | 9.00 |
| JUL 13 2026 | 1,075.00 |
| JUL 14 2026 | 321.00 |
| JUL 15 2026 | 1,938.00 |
| JUL15 2026 | 6.00 |
| JUL 16 2026 | 644.00 |
| JUL 17 2026 | 814.00 |
| JUL 21 2026 | 2,122.00 |
| JUL 27 2026 (partial) | 979.00 |
| **70013 pack** | **12,115.00** |

Then **ship 70013 −12,115** to Sunshine, `order_reference = QB-70013-BULK-YTD`, date 2026-08-14. 70013 on-hand after ship = 0 (not on the FG count sheet; sold as bulk).

**B. Generic unrecorded consume (12,188 lb)**

| Lot | lb − |
|---|---:|
| JUL 27 2026 (rest) | 3,543.00 |
| JUL 28 2026 | 6.00 |
| Jul 29 2026 | 214.00 |
| JUL 30 2026 | 537.00 |
| JUL 31 2026 | 1,399.00 |
| AUG 03 2026 | 465.00 |
| AUG 04 2026 | 968.00 |
| AUG 10 2026 | 8.00 |
| AUG 11 2026 | 3,433.00 |
| AUG 12 2026 (partial) | 1,615.00 |
| **Generic** | **12,188.00** |

Left: AUG 12 4,199 (13.00 b) + AUG 13 5,814 (18.00 b) = **31.00 / 10,013**.

Mini 995 units: held (no lb). If later applied, take that lb out of the generic 12,188 (starting at AUG 12 / AUG 11).

### 2.3 Classic CC #9 (90001) — 51 → 0

Consume **17,748.00 lb**, all lots. No CC #9 bulk was billed. Mini 4,186 units held (no lb). Entire 17,748 is generic unrecorded consumption until a mini weight exists.

| Lot | batches | lb |
|---|---:|---:|
| JUN 10 2026 | 6 | 2,088 |
| JUN 22 2026 | 5 | 1,740 |
| JUN 23 2026 | 5 | 1,740 |
| JUL 07 2026 | 4 | 1,392 |
| JUL 08 2026 | 2 | 696 |
| JUL 22 2026 | 18 | 6,264 |
| AUG 03 2026 | 11 | 3,828 |
| **Total** | **51** | **17,748** |

### 2.4 SS Original #1 (90016) — 0, via billed `#1 Bulk` 1,875 lb

SKU **70004** (zero ledger lines). Ledger batch on-hand **1,620.00 lb**. Billed **1,875** exceeds batch by **255.00 lb** (under-record). Count floor is 0.

1. adjust+ **255 lb** on 90016 (ref `QB-70004-BULK-YTD` under-record).  
2. pack 70004 **+1,875** from all 90016 lots (now 1,875).  
3. ship 70004 **−1,875** to Sunshine.

90016 after = **0**. 70004 after = 0.

### 2.5 Other counted batches — generic consume to the count

No QB dest was given for these leftovers. All `adjust−`, date 2026-08-14, reason `unrecorded_consumption / Aug 14 count`.

| SKU | Now lb / batches | Floor | Consume lb | Keep |
|---|---|---|---:|---|
| 90011 SS CC #2 | 17,232 / 43.8473 | 10 / 3,930 | **13,302.00** | AUG 14 10.00 exactly |
| 95002 BS Dark Choc 350 | 4,463.84 / 12.7538 | 3 / 1,050 | **3,413.84** | JUL 24 1,050 (3.00) |
| 90013 SS Cranberry #3 | 552.50 / 1.4578 | 1 / 379 | **173.50** | AUG 06 1.00 |
| 90020 Setton Cocoa #13 | 2,708 / 8.0118 | 0 | **2,708.00** | — |
| 90024 Vanilla Crisp #16 | 920 / 2.4865 | 0 | **920.00** | — |
| 90010 Vanilla Almond 380 | 245 / 0.6447 | 0 | **245.00** | — |
| 90019 Setton Cinnamon #14 | 150 / 0.4545 | 0 | **150.00** | — |
| 90015 SS LC Orig #7 | 75 / 0.2143 | 0 | **75.00** | — |

90011 consume lots: JUL 20 4,656 + AUG 05 4,323 + AUG 06 4,323 = 13,302.  
95002 consume lots: JUL 23 263.84 + JUL 24 3,150.00 = 3,413.84.  
90013 consume lot: JUN 24 173.50.

### 2.6 BS PBB mix — designation, not a 95005 true-up

95005 leftover is **2.38 lb**. Arturo counted **banana/PB-chip mix**, not granola. Do **not** write 95005 to 0 or to 3 batches.

Held: create (later) a mix/inclusion product and move or leave the 2.38 lb until that SKU exists. Not in the execution checklist as a granola consume.

### 2.7 Still not on the count sheet

90003/90004/90005/90007 (coconut batches), 90008 Fruit Nut — **no floor, no consume**.

---

## 3. Residual FG adjustments after (1)+(2) vs the Aug 14 count

### 3.1 70003 — upward

After the 3,377 cs ship, BB081027 holds **561**. Count **600**.  
**R-70003-UP: adjust+ 39 cs / 292.50 lb** on BB081027. Reason: `count_variance / Aug 14 600 vs 561 after billed ships`. This is pack-side (more on the floor than billed+recon implies), not a ship reversal.

### 3.2 70002

| Lot | cs − | lb − |
|---|---:|---:|
| BB061527 | 285 | 2,137.50 |
| BB070827 | 324 | 2,430.00 |
| **Total** | **609** | **4,567.50** |

Leaves BB081027 200 = floor.

### 3.3 70010 / 70070 / 70011 / 70073 / 70074

Already 0 after §1. No residual.

### 3.4 70080 leftover after 28220-I

| Lot | cs − | lb − |
|---|---:|---:|
| SW2612692 | 2 | 5.26 |
| SW2606892 | 10.654 | 28.02 |

### 3.5 Granola / graham FG now un-held

| SKU | Ledger | Count | Residual | Lots |
|---|---:|---:|---|---|
| 70050 Classic 25 | 55 cs AUG 11 | 18 (MAY 13×5, JUL 21×13) | **−55 AUG 11** then **+5 MAY 13 +13 JUL 21** (net −37 cs / −925 lb) | counted lots were fully shipped earlier; recreate them |
| 70082 Setton FV 25 | 30 JUL 21 | 0 | −30 cs / −750 lb | |
| 70059 Cocoa Vibes 25 | 24 JUN 30 | 0 | −24 cs / −600 lb | |
| 70052 Vanilla Crisp 25 | 14 JUN 29 | 0 | −14 cs / −350 lb | |
| 1614 CQ Granola 10 | 0 | 0 | none | |
| 31012 Graham 10 | 420 on 608101 | 280 unstamped | **−140 cs / −1,400 lb** on 608101 | |
| 10300 Crunchy 10 | 16 (AUG 10×11, AUG 11×5) | 0 | −11 / −110 and −5 / −50 | |

### 3.6 Coconut FG now un-held

| SKU | Ledger | Count | Residual |
|---|---|---|---|
| 893 CQ Flake 10 | 710 (280/140/290) | 700 (280/140/280) | **−10 cs / −100 lb on AUG 14** |
| 67476 UNIPRO Flake 10 | 317 (AUG05 237 + AUG06 80) | 317 header | **SKU residual 0.** Lot tags 140+97 held (§0 flag) |
| 67470 UNIPRO Fancy 10 | 119 AUG 13 | 26 AUG 13 | **−93 cs / −930 lb** |
| 10020 CNS Flake 25 | 118 AUG 06 | 0 | **−118 cs / −2,950 lb** |
| 10029 CNS Toasted 25 | 64 JUL 27 | 50 JUL 27 | **−14 cs / −350 lb** |
| 67473 UNIPRO Medium 10 | 62 AUG 13 | 60 AUG 13 | **−2 cs / −20 lb** |
| 10002 CNS Medium 10 | 96 (MAR 26×60 + AUG 04×36) | 16 AUG 04 | **−60 MAR 26, −20 AUG 04** (−80 cs / −800 lb) |
| 10001 CNS Flake 10 | 0 | 0 | none |
| 10007 CNS Fancy 25 | 5 | 0 | **−5 cs / −125 lb** |
| 10010 CNS Toasted 10 | 6 JUL 27 | 6 JUL 27 | none |
| 10006 CNS Fancy 10 | 0 | 0 | none |

### 3.7 70088 PBB 6x8 — not on the count sheet

802 cs / 2,406 lb still on `SW2607708`. **Not executed** unless someone confirms 6x8 was in scope and is physically 0.

---

## 4. GRPB anomaly (unchanged math)

28337-I billed **2,373 cs** GRPB on SW2620891. Lot only ever packed **1,374 cs**. Under-record **999 cs / 2,627.37 lb**. Pack+ then ship. 95005 cannot fund it (2.38 lb left; and that leftover is mix, not granola).

Same class, smaller: 70074 +57 on SW2620590; 70074 +63 on SW2607890; 70010 +203; 70070 +209; 70011 +104; 90016 +255.

---

## 5. Process-fix recommendations

### 5.1 Invoice-tied shipment posting

Unchanged: every QB invoice becomes a same-day standalone `/ship` (or an SO ship, never both) with the invoice number as `order_reference`. Weekly billed-vs-posted control by SKU. No more `SUNSHINE-RECON-*` lumps. Leave 1325–1342 posted as the first 4,639 / 900 / 142 / 140 cs of those SKUs.

New: Low Carb invoices must name 70010 vs 70070. Combined “Low Carb 10oz/12” is why 964 had to be allocated.

### 5.2 Mini 100 / bulk consume path

70004 and 70013 have **never** been packed or shipped. 70006 has **never** been used and has **no unit weight**. Bulk/mini left the silo as QB invoices while the ledger kept batch.

Needed later (not this write):

1. Set `case_size_lb` (or bag_oz × 100) on 70006. Add Mini 100 #9 and Mini 100 #9 ChocChip products if those are real SKUs, parented to 90002 / 90001 — 70006 is parented to #1 today.
2. A pack/repack that consumes batch (or 12x10) into bulk per-lb and Mini 100 in one txn.
3. Until then, the only honest recording is exactly this plan: pack+ dest FG from batch, then ship dest FG.

Also designate the banana/PB-chip mix so 95005 is not used as a granola silo.

---

## Projected on-hand after executable lines (ex-HOLDs)

| SKU | After all posted lines | Target |
|---|---|---|
| 70003 | 600 cs BB081027 | 600 |
| 70002 | 200 cs BB081027 | 200 |
| 70010 / 70070 / 70011 | 0 | 0 |
| 70073 / 70074 / 70080 | 0 | 0 |
| 70013 / 70004 | 0 (packed and shipped) | not on FG sheet |
| 70050 | 18 (MAY 13 5 + JUL 21 13) | 18 |
| 70082 / 70059 / 70052 / 10300 | 0 | 0 |
| 31012 | 280 on 608101 | 280 |
| 893 | 700 | 700 |
| 67476 | 317 (lots unchanged) | 317 |
| 67470 | 26 | 26 |
| 10020 / 10007 | 0 | 0 |
| 10029 | 50 | 50 |
| 67473 | 60 | 60 |
| 10002 | 16 AUG 04 | 16 |
| 10010 | 6 | 6 |
| 90002 | 31 | 31 |
| 90001 | 0 | 0 |
| 90011 | 10 on AUG 14 | 10 |
| 95002 | 3 on JUL 24 | 3 |
| 90016 / 90020 / 90024 / 90010 / 90019 / 90015 | 0 | 0 |
| 90013 | 1 on AUG 06 | 1 |

70088, 95005, coconut batches, Mini lb: unchanged / held.

---

## 6. Ordered execution checklist

Post ships via standalone `POST /ship`, not `commitShipOrder`.  
Pack missing FG via `/pack` if a source batch has stock; otherwise `/adjust+`.  
Batch generic consumes and FG residuals via `/adjust`.  
Notes on every txn: `INV-RECON-2026-08-17` plus the invoice or `physical-count-2026-08-14`.  
**Nothing below has been posted.**

### Phase 0 — preflight (read-only)

| # | Check | Approve |
|---|---|---|
| 0.1 | Re-query on-hand for every SKU in this plan. Abort if any number moved. | ☐ |
| 0.2 | Confirm 70073=PBB, 70080=Hazelnut. | ☐ |
| 0.3 | Confirm 70003 billed **8,016** (not 7,716) and 70002 ~915. | ☐ |
| 0.4 | Confirm Low Carb 964 allocation 475 / 489, or supply the invoice split. | ☐ |
| 0.5 | Confirm 70011 billed 283. | ☐ |
| 0.6 | Confirm 70013 = “Original (#9) Bulk” 12,115 lb and 70004 = “#1 Bulk” 1,875 lb. | ☐ |
| 0.7 | Mini 100: **no spec weight**. Leave 2.1 mini lines held, or supply oz/unit. | ☐ |
| 0.8 | Confirm Classic #9 keep-newest-31 (AUG 13 18 + AUG 12 13). | ☐ |
| 0.9 | Confirm 70074 MAR 20 2026 = remainder of SW2611090 (1,008+424). | ☐ |
| 0.10 | Confirm 67476 floor is SKU 317, not the 237 lot-sum. | ☐ |
| 0.11 | Confirm 70088 6x8 stays **unposted** (not on the sheet). | ☐ |
| 0.12 | Confirm SO-260629-003 / SO-260814-002 / SO-260817-001 are not shipped. | ☐ |
| 0.13 | Confirm recon txns 1325–1342 stay posted. | ☐ |
| 0.14 | Confirm 95005 is not consumed as granola. | ☐ |

### Phase 1 — create billed FG that was never packed

| # | Date | Type | SKU | Lot | Qty | lb | Ref | Approve |
|---|---|---|---|---|---:|---:|---|---|
| 1.1 | 2026-07-29 | pack/adjust+ | 70073 | SW2620891 | +999 cs | +2,627.37 | 28337-I GRPB | ☐ |
| 1.2 | 2026-07-24 | pack/adjust+ | 70074 | SW2620590 | +57 cs | +149.91 | 28337-I GRDC | ☐ |
| 1.3 | 2026-05-12 | pack/adjust+ | 70074 | SW2607890 | +63 cs | +165.69 | 28220-I GRDC | ☐ |
| 1.4 | 2026-08-14 | pack/adjust+ | 70010 | BB070827 | +203 cs | +1,522.50 | QB Low Carb under-pack | ☐ |
| 1.5 | 2026-08-14 | pack/adjust+ | 70070 | BB070827 | +209 cs | +1,567.50 | QB Low Carb under-pack | ☐ |
| 1.6 | 2026-08-14 | pack/adjust+ | 70011 | BB070827 | +104 cs | +780.00 | QB Cranberry under-pack | ☐ |
| 1.7 | 2026-08-14 | adjust+ | 90016 | (existing lot, prefer JUN 24) | +255 lb | +255.00 | #1 Bulk under-record | ☐ |

Gates: 70073 SW2620891 = 2,373 cs before 2.5; 70074 SW2620590 = 489 before 2.6; 70074 SW2607890 = 63 before 2.4; 70010 BB070827 = 333 cs before 2.16; 70070 BB070827 = 349 before 2.17; 70011 BB070827 = 283 before 2.18; 90016 = 1,875 lb before 3.B.

### Phase 2 — backdated standalone ships

| # | Date | Customer | Invoice | SKU | Lot | Qty | lb | Approve |
|---|---|---|---|---|---|---:|---:|---|
| 2.1 | 2026-05-12 | Blue Stripes | 28220-I | 70080 | SW2612692 | −1,742 cs | −4,581.46 | ☐ |
| 2.2 | 2026-05-12 | Blue Stripes | 28220-I | 70074 | SW2611090 | −1,008 cs | −2,651.04 | ☐ |
| 2.3 | 2026-05-12 | Blue Stripes | 28220-I | 70074 | MAR 20 2026 | −424 cs | −1,115.12 | ☐ |
| 2.4 | 2026-05-12 | Blue Stripes | 28220-I | 70074 | SW2607890 | −63 cs | −165.69 | ☐ |
| 2.5 | 2026-07-30 | Blue Stripes | 28337-I | 70073 | SW2620891 | −2,373 cs | −6,240.99 | ☐ |
| 2.6 | 2026-07-30 | Blue Stripes | 28337-I | 70074 | SW2620590 | −489 cs | −1,286.07 | ☐ |
| 2.7 | 2026-08-14 | Sunshine | QB-70003-YTD-NET | 70003 | BB060827 | −376 cs | −2,820.00 | ☐ |
| 2.8 | 2026-08-14 | Sunshine | same | 70003 | BB061527 | −495 cs | −3,712.50 | ☐ |
| 2.9 | 2026-08-14 | Sunshine | same | 70003 | BB062227 | −555 cs | −4,162.50 | ☐ |
| 2.10 | 2026-08-14 | Sunshine | same | 70003 | BB070827 | −150 cs | −1,125.00 | ☐ |
| 2.11 | 2026-08-14 | Sunshine | same | 70003 | BB071327 | −382 cs | −2,865.00 | ☐ |
| 2.12 | 2026-08-14 | Sunshine | same | 70003 | BB072027 | −450 cs | −3,375.00 | ☐ |
| 2.13 | 2026-08-14 | Sunshine | same | 70003 | BB080327 | −557 cs | −4,177.50 | ☐ |
| 2.14 | 2026-08-14 | Sunshine | same | 70003 | BB081027 | −412 cs | −3,090.00 | ☐ |
| 2.15 | 2026-08-14 | Sunshine | QB-70002-YTD-NET | 70002 | BB061527 | −15 cs | −112.50 | ☐ |
| 2.16 | 2026-08-14 | Sunshine | QB-LOWCARB-YTD-NET | 70010 | BB070827 | −333 cs | −2,497.50 | ☐ |
| 2.17 | 2026-08-14 | Sunshine | same | 70070 | BB070827 | −349 cs | −2,617.50 | ☐ |
| 2.18 | 2026-08-14 | Sunshine | QB-70011-YTD | 70011 | BB070827 | −283 cs | −2,122.50 | ☐ |

2.1–2.4 one header; 2.5–2.6 one; 2.7–2.14 one; 2.16–2.17 one.

### Phase 3 — batch: billed bulk/mini, then generic remainder

**3.A Classic #9 → 70013 bulk 12,115 lb (pack sources, then ship dest)**

| # | Lot (90002) | lb − from batch / + to 70013 | Approve |
|---|---|---:|---|
| 3.A1 | JUN 30 2026 | 224.00 | ☐ |
| 3.A2 | Jul 01 2026 | 753.00 | ☐ |
| 3.A3 | JUL 2 2026 | 429.00 | ☐ |
| 3.A4 | JUL 06 2026 | 1,075.00 | ☐ |
| 3.A5 | JUL 07 2026 | 1,726.00 | ☐ |
| 3.A6 | JUL 10 2026 | 9.00 | ☐ |
| 3.A7 | JUL 13 2026 | 1,075.00 | ☐ |
| 3.A8 | JUL 14 2026 | 321.00 | ☐ |
| 3.A9 | JUL 15 2026 | 1,938.00 | ☐ |
| 3.A10 | JUL15 2026 | 6.00 | ☐ |
| 3.A11 | JUL 16 2026 | 644.00 | ☐ |
| 3.A12 | JUL 17 2026 | 814.00 | ☐ |
| 3.A13 | JUL 21 2026 | 2,122.00 | ☐ |
| 3.A14 | JUL 27 2026 | 979.00 | ☐ |
| 3.A15 | ship 70013 Sunshine QB-70013-BULK-YTD | **−12,115.00** | ☐ |

**3.B #1 Bulk 70004 1,875 lb** (after 1.7)

| # | Action | Approve |
|---|---|---|
| 3.B1 | pack 70004 +1,875 from all 90016 lots (JUN 24, JUN 09, JUN 29, AUG 05 + the +255) | ☐ |
| 3.B2 | ship 70004 −1,875 Sunshine `QB-70004-BULK-YTD` | ☐ |

**3.C Mini 100 — held, no lb**

| # | QB item | Units | Source batch | Dest SKU | lb | Approve |
|---|---|---:|---|---|---|---|
| 3.C1 | Mini 100 #9 ChocChip | 4,186 | 90001 | **none in catalog** | **TBD** | ☐ HOLD |
| 3.C2 | #9 / Mini 100 | 995 | 90002 | **none in catalog** (70006 is #1) | **TBD** | ☐ HOLD |

**3.D Generic unrecorded consumption (count remainder)**

| # | SKU | Lot | lb − | Approve |
|---|---|---|---:|---|
| 3.D1 | 90002 | JUL 27 2026 (rest) | 3,543.00 | ☐ |
| 3.D2 | 90002 | JUL 28 2026 | 6.00 | ☐ |
| 3.D3 | 90002 | Jul 29 2026 | 214.00 | ☐ |
| 3.D4 | 90002 | JUL 30 2026 | 537.00 | ☐ |
| 3.D5 | 90002 | JUL 31 2026 | 1,399.00 | ☐ |
| 3.D6 | 90002 | AUG 03 2026 | 465.00 | ☐ |
| 3.D7 | 90002 | AUG 04 2026 | 968.00 | ☐ |
| 3.D8 | 90002 | AUG 10 2026 | 8.00 | ☐ |
| 3.D9 | 90002 | AUG 11 2026 | 3,433.00 | ☐ |
| 3.D10 | 90002 | AUG 12 2026 | 1,615.00 | ☐ |
| 3.D11 | 90001 | JUN 10 2026 | 2,088.00 | ☐ |
| 3.D12 | 90001 | JUN 22 2026 | 1,740.00 | ☐ |
| 3.D13 | 90001 | JUN 23 2026 | 1,740.00 | ☐ |
| 3.D14 | 90001 | JUL 07 2026 | 1,392.00 | ☐ |
| 3.D15 | 90001 | JUL 08 2026 | 696.00 | ☐ |
| 3.D16 | 90001 | JUL 22 2026 | 6,264.00 | ☐ |
| 3.D17 | 90001 | AUG 03 2026 | 3,828.00 | ☐ |
| 3.D18 | 90011 | JUL 20 2026 | 4,656.00 | ☐ |
| 3.D19 | 90011 | AUG 05 2026 | 4,323.00 | ☐ |
| 3.D20 | 90011 | AUG 06 2026 | 4,323.00 | ☐ |
| 3.D21 | 95002 | JUL 23 2026 | 263.84 | ☐ |
| 3.D22 | 95002 | JUL 24 2026 | 3,150.00 | ☐ |
| 3.D23 | 90013 | JUN 24 2026 | 173.50 | ☐ |
| 3.D24 | 90020 | JUN 03 2026 | 2,366.00 | ☐ |
| 3.D25 | 90020 | JUN 30 2026 | 76.00 | ☐ |
| 3.D26 | 90020 | MAY 01 2026 | 266.00 | ☐ |
| 3.D27 | 90024 | 26-05-06-VAN-002 | 765.00 | ☐ |
| 3.D28 | 90024 | JUL 21 2026 | 15.00 | ☐ |
| 3.D29 | 90024 | JUN 29 2026 | 140.00 | ☐ |
| 3.D30 | 90010 | JUN 29 2026 | 160.00 | ☐ |
| 3.D31 | 90010 | MAY 12 2026 | 85.00 | ☐ |
| 3.D32 | 90019 | JUN 30 2026 | 60.00 | ☐ |
| 3.D33 | 90019 | MAY 12 2026 | 90.00 | ☐ |
| 3.D34 | 90015 | JUN 24 2026 | 75.00 | ☐ |

Do not touch 90002 AUG 13, 90002 AUG 12 leftover 4,199, 90011 AUG 14, 95002 JUL 24 leftover 1,050, 90013 AUG 06.

### Phase 4 — residual count entries

| # | Date | Type | SKU | Lot | Qty | lb | Ref | Approve |
|---|---|---|---|---|---:|---:|---|---|
| 4.1 | 2026-08-14 | adjust+ | 70003 | BB081027 | +39 cs | +292.50 | count 600 vs 561 | ☐ |
| 4.2 | 2026-08-14 | adjust− | 70002 | BB061527 | −285 cs | −2,137.50 | count 200 | ☐ |
| 4.3 | 2026-08-14 | adjust− | 70002 | BB070827 | −324 cs | −2,430.00 | count 200 | ☐ |
| 4.4 | 2026-08-14 | adjust− | 70080 | SW2612692 | −2 cs | −5.26 | BS = 0 | ☐ |
| 4.5 | 2026-08-14 | adjust− | 70080 | SW2606892 | −10.654 cs | −28.02 | BS = 0 | ☐ |
| 4.6 | 2026-08-14 | adjust− | 70050 | AUG 11 2026 | −55 cs | −1,375.00 | wrong lot vs count | ☐ |
| 4.7 | 2026-08-14 | adjust+ | 70050 | MAY 13 2026 | +5 cs | +125.00 | counted lot | ☐ |
| 4.8 | 2026-08-14 | adjust+ | 70050 | JUL 21 2026 | +13 cs | +325.00 | counted lot | ☐ |
| 4.9 | 2026-08-14 | adjust− | 70082 | JUL 21 2026 | −30 cs | −750.00 | count 0 | ☐ |
| 4.10 | 2026-08-14 | adjust− | 70059 | JUN 30 2026 | −24 cs | −600.00 | count 0 | ☐ |
| 4.11 | 2026-08-14 | adjust− | 70052 | JUN 29 2026 | −14 cs | −350.00 | count 0 | ☐ |
| 4.12 | 2026-08-14 | adjust− | 31012 | 608101 | −140 cs | −1,400.00 | count 280 Clark | ☐ |
| 4.13 | 2026-08-14 | adjust− | 10300 | AUG 10 2026 | −11 cs | −110.00 | count 0 | ☐ |
| 4.14 | 2026-08-14 | adjust− | 10300 | AUG 11 2026 | −5 cs | −50.00 | count 0 | ☐ |
| 4.15 | 2026-08-14 | adjust− | 893 | AUG 14 2026 | −10 cs | −100.00 | count 700 | ☐ |
| 4.16 | 2026-08-14 | adjust− | 67470 | AUG 13 2026 | −93 cs | −930.00 | count 26 | ☐ |
| 4.17 | 2026-08-14 | adjust− | 10020 | AUG 06 2026 | −118 cs | −2,950.00 | count 0 | ☐ |
| 4.18 | 2026-08-14 | adjust− | 10029 | JUL 27 2026 | −14 cs | −350.00 | count 50 | ☐ |
| 4.19 | 2026-08-14 | adjust− | 67473 | AUG 13 2026 | −2 cs | −20.00 | count 60 | ☐ |
| 4.20 | 2026-08-14 | adjust− | 10002 | MAR 26 2026 | −60 cs | −600.00 | count 16 on AUG 04 only | ☐ |
| 4.21 | 2026-08-14 | adjust− | 10002 | AUG 04 2026 | −20 cs | −200.00 | 36 → 16 | ☐ |
| 4.22 | 2026-08-14 | adjust− | 10007 | 26-02-12-FOUND-014 | −5 cs | −125.00 | count 0 | ☐ |

### Phase 5 — held

| # | What | Why |
|---|---|---|
| 5.1 | Mini 100 4,186 + 995 → lb | 70006 has no case/bag weight; no #9 Mini SKU |
| 5.2 | 67476 lot move 237/80 → 140/97 | tags sum 237 ≠ header 317 |
| 5.3 | 70088 −802 cs | 6x8 not on Arturo’s sheet |
| 5.4 | 95005 / banana-PB mix product | counted as mix, not granola |
| 5.5 | Coconut batches 90003/4/5/7, 90008 | not on the batch sheet |
| 5.6 | Per-invoice split of 8,016 / 964 / 12,115 / 1,875 | line qtys not in this packet |
| 5.7 | SO-260629-003 disposition | 3,942 cs ≠ 28337-I 2,373 |
| 5.8 | Mini/bulk code path + 70006 spec | process, §5.2 |

### Phase 6 — post-write verification (only after an approved write)

| # | Check |
|---|---|
| 6.1 | 70003 = 600 cs on BB081027 only |
| 6.2 | 70002 = 200 cs on BB081027 only |
| 6.3 | 70010 = 70070 = 70011 = 0 |
| 6.4 | Posted 70003 ships YTD = 4,639 + 3,377 = 8,016 cs |
| 6.5 | Posted Low Carb ships YTD = 282 + 682 = 964 cs |
| 6.6 | Posted 70011 ships YTD = 283 cs |
| 6.7 | 70073 = 70074 = 70080 = 0; 70013 = 70004 = 0 |
| 6.8 | 90002 = 31 (AUG 12 13 + AUG 13 18); 90001 = 0; 90011 = 10 AUG 14; 90016 = 0; 95002 = 3; 90013 = 1 |
| 6.9 | Coconut/graham/classic 25 match §3 tables |
| 6.10 | Recon 1325–1342 still posted; no open SO gained shipped qty |

---

## What this plan is not

- Not a production write and not a migration.
- Not a void of the June 11 Sunshine recon.
- Not fulfillment of the August Sunshine SOs or SO-260629-003.
- Not a coconut-*batch* close (those silos were not counted).
- Not a Mini 100 posting — the catalog cannot convert those units to pounds.

When a line is rejected, strike it here and recompute §3 before any write.

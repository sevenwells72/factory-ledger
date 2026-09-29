# Granola batch on-hand audit — 2026-09-17

**Scope:** every `type='batch'` product whose name contains "Granola" (the dashboard "Batch Inventory On-Hand → Granola" section).
**Method:** read-only queries against the session pooler (port 5432) inside `BEGIN; SET TRANSACTION READ ONLY;`. On-hand = `SUM(quantity_lb)` from `ledger_current_transaction_lines` joined to `ledger_current_transactions` with `effective_status = 'posted'`. This is byte-for-byte the query the dashboard endpoint `/dashboard/api/inventory/batches` runs (`main.py` ~16215), so the dashboard is faithfully reporting the ledger. The ledger is what is wrong.
**Nothing was written.** No `/adjust`, no migrations, no commits other than this file and the change logs.

---

## 1. Per-product summary (posted, as of 2026-09-17)

| Product (id) | On-hand lb | ≈ batches | Non-zero lots | 2026 made lb | 2026 packed lb | 2026 adjust net lb | Packed / made |
|---|---:|---:|---:|---:|---:|---:|---:|
| Batch Classic Granola #9 (107) | **14,878.0** | 46.1 | 6 (SEP 01 – SEP 16) | 305,558 | 249,860 | −40,320 | 82 % |
| Batch Classic Chocolate Chip Granola #9 (108) | **9,396.0** | 27.0 | 4 (AUG 25 – SEP 16) | 47,328 | **900** | −37,032 | **2 %** |
| Batch SS Chocolate Chip Granola #2 (114) | **4,857.0** | 12.4 | 2 (SEP 09, SEP 10) | 122,616 | 88,553 | −29,207 | 72 % |
| Batch SS Original Granola #1 (116) | 1,967.5 | 5.6 | 5 (AUG 19 – SEP 14) | 39,200 | 32,088 | −5,145 | 82 % |
| Batch BS Dark Chocolate Granola 350 (121) | 1,050.0 | 3.0 | 1 (JUL 24) | 18,200 | 12,721 (+2,800 into PBB batch) | −1,629 | — |
| Granola Fruit Nut Batch (179) | 438.1 | 1.1 | 2 (JUN 29 169 lb, SEP 16 269 lb) | 2,063 | 1,625 | 0 | — |
| Batch SS Cranberry Granola #3 (118) | 249.0 | 0.7 | 2 (BB081727 245, AUG 06 4) | 5,685 | 3,368 | −2,069 | — |
| Batch Vanilla Crisp Granola #16 (112) | 240.0 | 0.6 | 1 (SEP 14) | 12,210 | 11,050 | −920 | — |
| Batch SS Low Carb Original Granola #7 (120) | 67.5 | 0.2 | 1 (AUG 27) | 5,600 | 4,073 | −1,460 | — |
| Batch SS Low Carb Chocolate Chip Granola #8 (119) | 35.0 | 0.1 | 1 (AUG 27) | 4,900 | 3,465 | −1,400 | — |
| Batch Granola Vanilla Almond 380 lb (113) | 5.0 | 0.0 | 1 (SEP 14) | 1,900 | 1,650 | −245 | — |
| Batch BS Peanut Butter Banana Granola (122) | 2.4 | 0.0 | 1 (JUL 27) | 3,616 | 3,614 | 0 | — |
| 109, 110, 111, 115, 117, 123, 124, 283, 284 | 0.0 | 0 | 0 | | | | |

"2026 made" = positive `make` lines; "packed" = negative `pack` lines; "adjust net" = sum of `adjust` lines. Each row reconciles: made − packed − ingredient use + adjust = on-hand (e.g. 107: 305,558 − 249,860 − 500 − 40,320 = 14,878).

### 1a. On-hand by lot — the three flagged products

| Product | Lot id | Lot code | Made (lb) | Consumed (lb) | **On-hand** | Last touched |
|---|---:|---|---:|---:|---:|---|
| 107 Classic #9 | 1378 | SEP 01 2026 | 4,522 | −2,013 | **2,509** | 2026-09-16 |
| 107 | 1387 | SEP 02 2026 | 5,168 | −4,750 | **418** | 2026-09-16 |
| 107 | 1429 | SEP 11 2026 | 3,230 | 0 | **3,230** | untouched |
| 107 | 1434 | SEP 14 2026 | 1,615 | 0 | **1,615** | untouched |
| 107 | 1441 | SEP 15 2026 | 5,168 | 0 | **5,168** | untouched |
| 107 | 1449 | SEP 16 2026 | 1,938 | 0 | **1,938** | untouched |
| 108 Classic CC #9 | 1334 | AUG 25 2026 | 3,132 | 0 | **3,132** | untouched |
| 108 | 1346 | AUG 26 2026 | 3,480 | 0 | **3,480** | untouched |
| 108 | 1377 | SEP 01 2026 | 696 | 0 | **696** | untouched |
| 108 | 1450 | SEP 16 2026 | 2,088 | 0 | **2,088** | untouched |
| 114 SS CC #2 | 1413 | SEP 09 2026 | 5,502 | −4,575 | **927** | 2026-09-16 |
| 114 | 1422 | SEP 10 2026 | 3,930 | 0 | **3,930** | untouched |

All older lots on these three products are at zero. Nothing stale is hiding in the totals; the excess is entirely in lots made since the last physical count (2026-08-14).

---

## 2. Pounds in vs out by month since 2026-01-01

`in` = make lines; `pack` = pack consumption; `adj` = adjustments (sign as posted). Blank = none.

### 107 Batch Classic Granola #9
| Month | Made in | Pack out | Adjust | Net |
|---|---:|---:|---:|---:|
| 2026-02 | 31,977 | −8,120 | +4,284 / −18,454 | +9,667 |
| 2026-03 | 26,486 | −28,815 | | −2,329 |
| 2026-04 | 19,703 | −13,850 | | +5,853 |
| 2026-05 | 34,238 | −21,000 | | +12,838 |
| 2026-06 | 30,362 | −28,310 | −15,537 (Jun 8 count) | −13,525 |
| 2026-07 | 96,900 | −89,750 | | +7,150 |
| 2026-08 | 44,251 | −47,215 | +1,575 / −12,188 (Aug 14 count) | −13,577 |
| 2026-09 (to 16th) | 21,641 | −12,800 | | +8,801 |

### 108 Batch Classic Chocolate Chip Granola #9
| Month | Made in | Pack out | Adjust | Net |
|---|---:|---:|---:|---:|
| 2026-02 | 2,784 | | −15 | +2,769 |
| 2026-03 | 2,088 | | | +2,088 |
| 2026-04 | 6,264 | −600 | | +5,664 |
| 2026-05 | 6,960 | −300 | | +6,660 |
| 2026-06 | 7,656 | | −19,269 (Jun 8 count) | −11,613 |
| 2026-07 | 8,352 | | | +8,352 |
| 2026-08 | 10,440 | | −17,748 (Aug 14 count) | −7,308 |
| 2026-09 (to 16th) | 2,784 | | | +2,784 |

### 114 Batch SS Chocolate Chip Granola #2
| Month | Made in | Pack out | Adjust | Net |
|---|---:|---:|---:|---:|
| 2026-02 | 15,720 | −7,928 | | +7,793 |
| 2026-03 | | −5,625 | | −5,625 |
| 2026-04 | 16,899 | −9,000 | | +7,899 |
| 2026-05 | 18,078 | −11,115 | | +6,963 |
| 2026-06 | 19,257 | −11,820 | −15,905 (Jun 8 count) | −8,468 |
| 2026-07 | 14,934 | −7,365 | | +7,569 |
| 2026-08 | 22,401 | −23,625 | −13,302 (Aug 14 count) | −14,526 |
| 2026-09 (to 16th) | 15,327 | −12,075 | | +3,252 |

### 116 Batch SS Original Granola #1 (for comparison — a "normal" product)
| Month | Made in | Pack out | Adjust | Net |
|---|---:|---:|---:|---:|
| 2026-02 | 2,800 | −2,753 | | +48 |
| 2026-03 | 3,150 | −3,018 | | +133 |
| 2026-04 | 5,250 | −3,580 | | +1,670 |
| 2026-05 | 4,550 | | | +4,550 |
| 2026-06 | 6,650 | −3,850 | +1,000 / −6,400 | −2,600 |
| 2026-07 | | −2,430 | | −2,430 |
| 2026-08 | 7,350 | −7,525 | +255 | +80 |
| 2026-09 (to 16th) | 9,450 | −8,933 | | +518 |

The pattern is the same on every granola batch: the ledger drifts upward between counts and is written down at each count (Feb 12, Jun 8, Aug 14 — reason text "unrecorded consumption"). Total 2026 write-downs on the three flagged products: **≈ 106,500 lb**.

### Since the last count (2026-08-14 → 2026-09-16)

| Product | Count floor Aug 14 | Made since | Packed since | Ledger now | Ledger as % of available |
|---|---:|---:|---:|---:|---:|
| 107 | 10,013 (31 batches) | 37,145 | 32,240 (+40 to Fruit Nut) | 14,878 | 32 % |
| 108 | 0 | 9,396 | **0** | 9,396 | 100 % |
| 114 | 3,930 (10 batches) | 25,152 | 24,225 | 4,857 | 17 % |

---

## 3. What was checked and ruled out

| Hypothesis | Result |
|---|---|
| Pack transactions that don't consume the batch | **Ruled out.** Every posted `pack` since Jan 1 has a negative batch line equal to the FG output (97 FG × source × month groups, all `fg_out_lb = batch_in_lb`). `/pack` inserts `−total_lb` on the source lots unconditionally (`main.py` ~8760). |
| Wrong BOM / consumption quantity | **Ruled out for packs.** `product_bom` is 25 lb → 25 LB case, 10 → 10 LB, 7.5 → 12x10 OZ; `/pack` uses `cases × case_size_lb` and that is what the lines show. |
| `yield_multiplier ≠ 1.0` | **Ruled out.** All 25 granola batch products have `yield_multiplier = 1`. `formula_lb` equals `default_batch_lb` on every one (323 / 348 / 393 / 350 …). |
| Duplicate makes | **Ruled out for Aug–Sep.** The only same-day/same-qty pairs in 2026 are three 16-batch Classic entries on 2026-02-05 and two 700-lb SS Original entries on 2026-04-27 (legacy backfill, identical `created_at`); both lots were zeroed by the Feb/Jun write-downs. |
| Stale old lots never consumed | **Ruled out for 107/108/114** (see §1a). Minor stale remnants elsewhere: 179 JUN 29 169 lb; 121 JUL 24 1,050 lb (untouched since the Aug 14 count set it to 3 batches); 118 AUG 06 4 lb; 113 SEP 14 5 lb; 122 2.4 lb (Aug 14 count says this is banana/PB-chip mix, not granola). |
| Dashboard query bug | **Ruled out.** Endpoint = posted-lines SUM by product and lot; matches the numbers above exactly. |
| Ingredient side of the 108 makes | The four Aug 25 – Sep 16 makes consumed real ingredients including **675 lb of Chocolate Chips – Real – 1,000 CT**, so they are genuine production events, not typos. |

---

## 4. Root cause, in plain English

### 4a. Classic Chocolate Chip #9 (108): the product it becomes has no ledger path — 100 % of the 9,396 lb

Chocolate Chip #9 is made for Sunshine Granola and leaves the building as **bulk per-pound** (and, per the Aug 14 recon, as **"Mini 100 #9 ChocChip" bags**). Neither destination can be recorded:

* **Sunshine SO 298 (created 2026-08-17, PO 08172026, status `confirmed`)** carries **6,000 lb of "Granola SS Classic Chocolate Chip #9 Bulk per/lb" (product 288)** and **4,000 lb of "Granola SS Classic #9 Bulk per/lb" (product 285)**. Both lines show **0 lb shipped**. No Sunshine `ship` transaction exists after Aug 14. The four 108 makes (Aug 25, Aug 26, Sep 1, Sep 16 = 9,396 lb) line up with that order.
* Products 285 and 288 have **`case_size_lb = NULL`**, so `/pack` refuses them ("Case weight required…") unless the caller passes `case_weight_lb`. The GPT only infers a case weight from a "N LB" name; "per/lb" has none. So the pack never gets entered, and the batch is never debited.
* 285/288 (and the 10 LB / 25 LB siblings 286/287/289/290) are **parented to the Kosher-Ignition batches 283/284**, which have **zero makes ever**. Real production is logged on 107/108. Even with a weight override, `/pack` from 108 into 288 raises a "Source batch mismatch" warning because 284's formula does not contain 108.
* "Mini 100" (product 185) is parented to SS Original #1, has no bag weight, and there is no ChocChip mini product at all. The Aug 14 recon plan already flagged 4,186 mini ChocChip units billed YTD in QuickBooks with nowhere to post them.

This is exactly the mechanism behind the 17,748 lb write-off on Aug 14 and the 19,269 lb write-off on Jun 8: 108 is baked, shipped as bulk/mini, and only ever cleared by a physical-count adjustment. The one FG that *can* be packed from 108 (Granola Chocolate Chip 25 LB, id 143) has had no transaction since May.

### 4b. Classic #9 (107): the same bulk hole (4,000 lb) plus an unexplained shrink

* 4,000 lb of SO 298 line 694 (Classic #9 bulk) is unrecorded for the same reason. On Aug 14 the recon had to synthesize a 12,115 lb pack+ship of product 70013 for the same SKU.
* Beyond that, 107 shows a persistent ~13–18 % gap between made and packed even in months with no bulk activity (Mar–May). The remaining 10,878 lb after the bulk true-up (≈ 34 batches, ≈ 11 containers) is *possible* on the floor — Grassland has 18,750 lb of Classic 25 LB confirmed and unpacked — but the Aug 14 count found 31 batches when the ledger said 106, so a count is required before believing it.

### 4c. SS Chocolate Chip #2 (114): systematic shrink, not a missing SKU

114 packs 1:1 into 12x10 OZ cases and has no bulk sink, yet **24 % of every pound made in 2026 has been written off at counts** (29,207 of 122,616). Since Aug 14 the ledger holds 4,857 of 29,082 lb available; if the floor really has ~1 container (≈ 3 batches, ≈ 1,180 lb) the shrink since the count is ≈ 3,700 lb ≈ 15 % of production. The batch is credited at the **raw formula weight** (393 lb, which includes 47 lb honey, 22 lb oil) with `yield_multiplier = 1.0`; bake moisture loss, hopper/line waste, QA samples and bag over-fill are all uncredited. The same ~18 % shows on 107 and 116. **This is a hypothesis** — it needs a weigh-off (baked lb out per batch vs 393) to confirm and to set a multiplier.

### 4d. Why it shows up as "huge numbers" now
The Aug 14 count zeroed everything old, so 100 % of what has accrued in 33 days is visible in 12 recent lots. Nothing has been written down since, and the two biggest daily makes (16 batches on Sep 15, 10 on Sep 11) are still fully intact because packs FIFO from the Sep 1/2 lots first.

---

## 5. Dashboard: is there an `/adjust` UI?

**No.** `grep` of `dashboard/*.js`, `dashboard/*.html`, `dashboard/scheduler/` finds `/adjust` nowhere. The only "adjust" strings are display labels for the Recent Entries feed (`adjusted_in: 'Stock added'`, `adjust: 'Adjustment'`) and the History page filter. All adjustments to date were posted through the GPT (`operator_id = legacy-shared-key`) or the Aug 17 recon script (`scripts/inv_recon_post_2026_08_17.py`, operator `inv-recon-2026-08-17`). Per memory, `/adjust` has no actor-key plumbing and is not on `DASHBOARD_KEY_ALLOWLIST`, so a dashboard adjust button would need backend work, not just UI.

---

## 6. Recommended fix

### Forward fix (stops the drift)

1. **Make per-pound SKUs packable.** In `/pack` (`main.py` ~8600 / ~8700), when `target.case_size_lb IS NULL` and the product name ends in "per/lb" (or `pack_format IS NULL` and `uom = 'lb'`), default `case_weight = 1.0` so `cases` = pounds. Alternatively set `case_size_lb = 1` on 183, 184, 285, 288 and treat "cases" as lb by convention. Either way the GPT can then log "pack 6000 lb bulk".
2. **Re-parent the Sunshine bulk/10 LB/25 LB SKUs** 285/286/287 → 107 and 288/289/290 → 108, *or* start logging makes on 283/284 when the kosher-supervised run is what actually happens. Today the parent points at batches that have never been made, which turns every correct pack into a mismatch warning and hides the FGs from the post-`/make` `pack_needed` prompt.
3. **Create the mini SKUs with a weight.** "Mini 100 #9 ChocChip" and "#9 / Mini 100" exist in QuickBooks, not in `products`. Add them parented to 108 / 107 with `case_size_lb` (bag oz × bags) so the Sunshine mini runs can be packed. Fix 185's NULL weight at the same time.
4. **Sales-order guard.** SO 298 has been `confirmed` for a month with 0 lb shipped while the batches were consumed. Surface "confirmed > 14 days, 0 % shipped" on the dashboard orders list, and have the GPT ask "was this shipped?" when a `/make` lands on a batch whose only open demand is a per/lb line.
5. **Yield.** Run a weigh-off on 3–5 batches each of 107, 114, 116 (baked pounds out of the oven vs formula weight). If the loss is real, set `yield_multiplier` (e.g. 0.85–0.90) — `/make` already multiplies (`total_output = formula_weight_lb × yield_multiplier`, `main.py` ~8093) and the dashboard batch_count already divides by `made_unit_size_lbs`, so no code change is needed. Until then, expect ~15 % of every granola make to be phantom.
6. **Count cadence.** Month-end batch container count with same-day write-down, using the Aug 14 recon plan as the template. Any container count older than ~30 days is worthless for these three products (they turn over 5–10 k lb/week).

### Proposed per-lot adjustment list — NOT executed, NOT approved

Two tiers. Tier A is evidence-backed today and should be posted as pack + ship (so the Sunshine order and traceability are correct), not as `/adjust`. Tier B needs a physical count first; the amounts shown are the maximum that could be written down and must be replaced by (ledger − counted) once Arturo counts.

**Tier A — record the Sunshine bulk on SO 298 (pack + ship, FIFO oldest lots)**

| # | Batch product | Lot | Action | lb | Result lot balance |
|---|---|---|---|---:|---:|
| A1 | 108 Classic CC #9 | AUG 25 2026 (1334) | pack → 288 CC #9 Bulk per/lb, `case_weight_lb = 1`, cases = lb | −3,132 | 0 |
| A2 | 108 | AUG 26 2026 (1346) | pack → 288 | −2,868 | 612 |
| A3 | 288 | (new lot AUG 25 2026) | ship 6,000 lb to Sunshine Granola against SO 298 line 693 | −6,000 | 0 |
| A4 | 107 Classic #9 | SEP 01 2026 (1378) | pack → 285 Classic #9 Bulk per/lb | −2,509 | 0 |
| A5 | 107 | SEP 02 2026 (1387) | pack → 285 | −418 | 0 |
| A6 | 107 | SEP 11 2026 (1429) | pack → 285 | −1,073 | 2,157 |
| A7 | 285 | (new lot SEP 01 2026) | ship 4,000 lb to Sunshine Granola against SO 298 line 694 | −4,000 | 0 |

Precondition: confirm with Sunshine / the BOL file that PO 08172026 actually shipped and on what date (QuickBooks shows no Sunshine invoice between Aug 15 and Sep 17, so it may be unbilled rather than unshipped). Use `occurred_at` = the real ship date.

**Tier B — pending physical count (max write-down shown; replace with ledger − count)**

| # | Batch product | Lot | Ledger after Tier A | Proposed | Note |
|---|---|---|---:|---|---|
| B1 | 108 | AUG 26 2026 (1346) | 612 | adjust − up to 612, reason `unrecorded_consumption / Sep count` | CC #9 floor was 0 on Aug 14; mini bags likely |
| B2 | 108 | SEP 01 2026 (1377) | 696 | adjust − up to 696 | same |
| B3 | 108 | SEP 16 2026 (1450) | 2,088 | **hold** | made yesterday; count it |
| B4 | 107 | SEP 11 2026 (1429) | 2,157 | adjust − (ledger − count), oldest first | |
| B5 | 107 | SEP 14 2026 (1434) | 1,615 | " | |
| B6 | 107 | SEP 15 2026 (1441) | 5,168 | " | 16 batches; Grassland 18,750 lb open — may be real |
| B7 | 107 | SEP 16 2026 (1449) | 1,938 | **hold** | made yesterday |
| B8 | 114 SS CC #2 | SEP 09 2026 (1413) | 927 | adjust − (ledger − count), oldest first | Sunshine SO 329 4,500 lb open |
| B9 | 114 | SEP 10 2026 (1422) | 3,930 | " | |
| B10 | 179 Fruit Nut | JUN 29 2026 (902) | 169 | adjust −169 if not on floor | 80-day-old remnant |
| B11 | 121 BS Dark Choc | JUL 24 2026 (1055) | 1,050 | count; adjust to count | set to 3 batches on Aug 14, untouched since |
| B12 | 118 / 113 / 122 | AUG 06 (4 lb) / SEP 14 (5 lb) / JUL 27 (2.4 lb) | 11.4 | adjust to 0 | dust; 122 is banana mix per Aug 14 sheet |

If the count comes back at roughly the Aug 14 proportions (about one third of the ledger), expect the Tier B write-down to be on the order of 7,000 lb on 107, 1,300 lb on 108, and 3,500 lb on 114.

---

## Appendix — key evidence pointers

* `/pack` consumption and per/lb refusal: `main.py` 8587–8700 (`case_weight <= 0 → 400`), 8760 (batch lines inserted `−qty`).
* `/make` output credit: `main.py` 8091–8093 (`total_output = formula_weight_lb × yield_multiplier`).
* Dashboard endpoint: `main.py` 16215 (`/dashboard/api/inventory/batches`), posted-lines SUM.
* Prior counts: `docs/audits/physical-count-2026-08-14.md`, `docs/audits/inventory-variance-recon-plan.md` §2.1–2.3 (mini units without weight, CC #9 51 batches → 0, 70013 bulk true-up).
* Sunshine order: `sales_orders` 298, lines 693 (288, 6,000 lb) and 694 (285, 4,000 lb), `sales_order_shipments` = none.
* Ingredient proof for the 108 makes: `ingredient_lot_consumption` on tx 2104, 2120, 2167, 2378 (chocolate chips 225 / 250 / 50 / 150 lb).

# Inventory variance — transaction history

**Pulled:** 2026-08-17 14:48 EDT; follow-up 2026-08-17 14:56 EDT; batch #9 follow-up 2026-08-17 15:01 EDT; 90-day in/out 2026-08-17 15:02 EDT  
**Source:** production Supabase, read-only  
**Ledger rule:** `ledger_current_transactions` + `ledger_current_transaction_lines`. On-hand and in/out totals use `effective_status = 'posted'`. Date key is `business_date`. Window for totals: `business_date >= 2026-01-01`.  
**No rows exist before 2026-01-01** for any of the four products.

Signed quantity is the product line on that SKU’s lots: pack/receive/adjust+ = inbound; ship/adjust− = outbound.

### Product naming verification (2026-08-17, live `products` table)

| SKU | id | Canonical name | case_size_lb |
|---|---:|---|---:|
| **70073** | 151 | **BS Granola – Peanut Butter Banana – 6x7 OZ Case** | 2.63 |
| **70074** | 150 | **BS Granola – Dark Chocolate – 6x7 OZ Case** | 2.63 |
| **70080** | 153 | **BS Granola – Hazelnut Butter – 6x7 OZ Case** | 2.63 |

QuickBooks abbreviations used in the recon plan: **GRPB = 70073**, **GRDC = 70074**, **GRHB = 70080**. Do not swap 70073 and 70080.

Every section below already used these names. **No historical row was relabeled.** A 70073=Hazelnut reading came from chat/summary wording, not from this extract or from `products`.

---

## Summary

| Product | SKU | On hand (posted) | Last outbound | Last outbound date | Inbound since 2026-01-01 | Outbound since 2026-01-01 | Net |
|---|---|---:|---|---|---:|---:|---:|
| Granola SS Chocolate Chip 12x10 OZ Case | 70003 | 29,535.00 lb (3,938 cs @ 7.5) | ship txn **1336** −1,125.00 lb lot `MAY 01 2026` → Sunshine Granola (recon) | **2026-06-11** 14:52 ET | 64,327.50 lb | 34,792.50 lb | 29,535.00 |
| CQ Granola 10 LB | 1614 | 0.00 lb | ship txn **1949** last line −4,200.00 lb lot `AUG 10 2026` (same txn also −5,600.00 lb another lot) → SO-260714-001 RD Streetsboro | **2026-08-17** 14:33 ET | 163,600.00 lb | 163,600.00 lb | 0.00 |
| CQ Coconut Sweetened Flake 10 LB | 893 | 7,100.00 lb (710 cs @ 10) | ship txn **1948** last line −1,400.00 lb lot `AUG 12 2026` (same txn also −5,600 / −4,200) → SO-260714-001 RD Streetsboro | **2026-08-17** 14:33 ET | 278,531.00 lb | 271,431.00 lb | 7,100.00 |
| BS Granola – Dark Chocolate – 6x7 OZ Case | 70074 | 4,902.32 lb (1,864 cs @ 2.63) | ship txn **573** −3,408.48 lb lot `SW2607890` → SO-260311-001 Blue Stripes | **2026-03-30** 14:32 ET | 8,310.80 lb | 3,408.48 lb | 4,902.32 |

---

## 1. Granola SS Chocolate Chip 12x10 OZ Case (id 145, SKU 70003)

- Type: finished. Case size: 7.5 lb. Label: private_label.
- Lots: 20. Posted lines: 43 (31 pack inbound, 12 ship outbound). Zero voids.
- Posted on-hand: **29,535.00 lb** = 3,938 cases.

### Last outbound / deduction

All 12 deductions are standalone ships on **2026-06-11**, customer **Sunshine Granola**, `order_reference = SUNSHINE-RECON-2026`. Notes on each: *“INTERIM — actual ship dates/orders pending backdated customer data. To be voided and replaced.”* No sales-order number. `created_at` on these rows is 2026-08-11 10:32 ET (migration 039 backfill), not the operational time.

**Last deduction line** (latest `occurred_at`):

| Field | Value |
|---|---|
| Date | 2026-06-11 |
| Time (occurred_at) | 14:52:34 ET |
| Type | ship |
| Qty | **−1,125.00 lb** (150 cases) |
| Lot | `MAY 01 2026` |
| Txn | 1336 |
| Shipment | 251 |
| Customer | Sunshine Granola |
| SO | none |

The same-day recon series (txns 1325–1336) totals **−34,792.50 lb** (4,639 cases). Nothing has been shipped from this SKU since.

### Inbound vs outbound since 2026-01-01 (posted)

| Direction | lb | cases @ 7.5 |
|---|---:|---:|
| Inbound (all pack) | 64,327.50 | 8,577 |
| Outbound (all ship) | 34,792.50 | 4,639 |
| Net / on-hand | 29,535.00 | 3,938 |

Remaining on-hand lots (posted): `BB081027` 7,297.5; `BB080327` 4,177.5; `BB062227` 4,162.5; `BB061527` 3,712.5; `BB072027` 3,375.0; `BB071327` 2,865.0; `BB060827` 2,820.0; `BB070827` 1,125.0.

### Full chronological list

| business_date | type | qty_lb | cases | txn | lot | reference |
|---|---|---:|---:|---:|---|---|
| 2026-02-17 | pack | +4,500.00 | 600 | 260 | BB021627 | pack from Batch SS Choc Chip #2 |
| 2026-02-18 | pack | +1,912.50 | 255 | 270 | FEB 13 2026 | pack |
| 2026-02-20 | pack | +952.50 | 127 | 298 | FEB 13 2026 | pack |
| 2026-02-24 | pack | +562.50 | 75 | 315 | BB022427 Lot | pack |
| 2026-03-02 | pack | +2,002.50 | 267 | 345 | FEB 18 2026 | pack |
| 2026-03-03 | pack | +247.50 | 33 | 364 | BB030327 | pack |
| 2026-03-04 | pack | +1,125.00 | 150 | 368 | 26-02-25 | pack |
| 2026-03-05 | pack | +2,250.00 | 300 | 384 | 26-02-25 | pack |
| 2026-04-14 | pack | +2,250.00 | 300 | 616 | BB041327 Lot | pack |
| 2026-04-15 | pack | +3,375.00 | 450 | 625 | BB041327 | pack |
| 2026-04-16 | pack | +1,125.00 | 150 | 638 | BB041327 | pack |
| 2026-04-17 | pack | +2,250.00 | 300 | 644 | BB041327 | pack |
| 2026-05-01 | pack | +1,125.00 | 150 | 786 | MAY 01 2026 | pack |
| 2026-05-04 | pack | +3,375.00 | 450 | 800 | BB050427 | pack |
| 2026-05-05 | pack | +4,365.00 | 582 | 813 | BB050427 | pack |
| 2026-05-28 | pack | +2,250.00 | 300 | 1100 | BB052627 | pack |
| 2026-06-01 | pack | +1,125.00 | 150 | 1126 | BB060127 | pack |
| 2026-06-11 | ship | −3,375.00 | 450 | 1325 | 26-02-25 | Sunshine Granola / SUNSHINE-RECON-2026 / shipment 240 |
| 2026-06-11 | ship | −4,500.00 | 600 | 1326 | BB021627 | Sunshine Granola / shipment 241 |
| 2026-06-11 | ship | −562.50 | 75 | 1327 | BB022427 Lot | Sunshine Granola / shipment 242 |
| 2026-06-11 | ship | −247.50 | 33 | 1328 | BB030327 | Sunshine Granola / shipment 243 |
| 2026-06-11 | ship | −6,750.00 | 900 | 1329 | BB041327 | Sunshine Granola / shipment 244 |
| 2026-06-11 | ship | −2,250.00 | 300 | 1330 | BB041327 Lot | Sunshine Granola / shipment 245 |
| 2026-06-11 | ship | −7,740.00 | 1,032 | 1331 | BB050427 | Sunshine Granola / shipment 246 |
| 2026-06-11 | ship | −2,250.00 | 300 | 1332 | BB052627 | Sunshine Granola / shipment 247 |
| 2026-06-11 | ship | −1,125.00 | 150 | 1333 | BB060127 | Sunshine Granola / shipment 248 |
| 2026-06-11 | ship | −2,865.00 | 382 | 1334 | FEB 13 2026 | Sunshine Granola / shipment 249 |
| 2026-06-11 | ship | −2,002.50 | 267 | 1335 | FEB 18 2026 | Sunshine Granola / shipment 250 |
| 2026-06-11 | ship | **−1,125.00** | 150 | **1336** | MAY 01 2026 | Sunshine Granola / shipment 251 — **last outbound** |
| 2026-06-11 | pack | +1,125.00 | 150 | 1346 | BB060827 | pack (same day, after the recon ships) |
| 2026-06-12 | pack | +1,695.00 | 226 | 1355 | BB060827 | pack |
| 2026-06-18 | pack | +2,250.00 | 300 | 1389 | BB061527 | pack |
| 2026-06-19 | pack | +1,462.50 | 195 | 1397 | BB061527 | pack |
| 2026-06-22 | pack | +2,250.00 | 300 | 1411 | BB062227 | pack |
| 2026-06-23 | pack | +1,912.50 | 255 | 1432 | BB062227 | pack |
| 2026-07-09 | pack | +1,125.00 | 150 | 1609 | BB070827 | pack |
| 2026-07-13 | pack | +2,865.00 | 382 | 1639 | BB071327 | pack |
| 2026-07-21 | pack | +3,375.00 | 450 | 1698 | BB072027 | pack |
| 2026-08-03 | pack | +562.50 | 75 | 1816 | BB080327 | pack |
| 2026-08-04 | pack | +2,812.50 | 375 | 1843 | BB080327 | pack |
| 2026-08-05 | pack | +802.50 | 107 | 1859 | BB080327 | pack |
| 2026-08-12 | pack | +3,442.50 | 459 | 1906 | BB081027 | pack |
| 2026-08-13 | pack | +3,855.00 | 514 | 1921 | BB081027 | pack |

No adjustments, receives, or voids on this SKU.

---

## 2. CQ Granola 10 LB (id 144, SKU 1614) — summary only

- Case size 10 lb. Posted on-hand: **0.00 lb**.
- Posted line mix: pack 57 / ship 41 / adjust 18 / receive 1. Two voided lines (txn 469 pack +10 and txn 470 pack −10 on 2026-03-18); excluded from posted totals (10 lb each).

### Last outbound

| Field | Value |
|---|---|
| Date | **2026-08-17** |
| Time | 14:33:55 ET |
| Type | ship |
| Txn | **1949** |
| Shipment | 332 |
| Customer | Restaurant Depot/Jetro-RD #436 Streetsboro OH |
| SO | **SO-260714-001** (status shipped) |
| Last line | **−4,200.00 lb** lot `AUG 10 2026` |
| Same txn other line | −5,600.00 lb (second lot) |
| Event total | **−9,800.00 lb** (980 cases) |

### Inbound vs outbound since 2026-01-01 (posted)

| Direction | lb | cases @ 10 |
|---|---:|---:|
| Inbound | 163,600.00 | 16,360 |
| Outbound | 163,600.00 | 16,360 |
| Net / on-hand | 0.00 | 0 |

Inbound mix (posted): pack 148,420.00; adjust+ 13,780.00; receive 1,400.00. Outbound: ship 155,400.00; adjust− 8,200.00.

---

## 3. CQ Coconut Sweetened Flake 10 LB (id 164, SKU 893) — summary only

- Case size 10 lb. Posted on-hand: **7,100.00 lb** (710 cases).
- Posted line mix: pack 126 / ship 82 / adjust 29 / receive 2. Three voided make lines (txns 83, 84, 177) with **0.00 lb** — no inventory effect.

### Last outbound

| Field | Value |
|---|---|
| Date | **2026-08-17** |
| Time | 14:33:55 ET |
| Type | ship |
| Txn | **1948** |
| Shipment | 332 |
| Customer | Restaurant Depot/Jetro-RD #436 Streetsboro OH |
| SO | **SO-260714-001** |
| Last line | **−1,400.00 lb** lot `AUG 12 2026` |
| Same txn other lines | −5,600.00 and −4,200.00 |
| Event total | **−11,200.00 lb** (1,120 cases) |

Same shipment header (332) as the CQ Granola 10 LB ship above.

### Inbound vs outbound since 2026-01-01 (posted)

| Direction | lb | cases @ 10 |
|---|---:|---:|
| Inbound | 278,531.00 | 27,853.1 |
| Outbound | 271,431.00 | 27,143.1 |
| Net / on-hand | 7,100.00 | 710 |

Inbound mix (posted): pack 220,830.00; adjust+ 55,691.00; receive 2,010.00. Outbound: ship 232,400.00; adjust− 39,031.00.

---

## 4. BS Granola – Dark Chocolate – 6x7 OZ Case (id 150, SKU 70074) — summary only

- Case size 2.63 lb. Posted on-hand: **4,902.32 lb** (1,864 cases).
- Posted lines: 6 pack inbound, 1 ship outbound. Zero voids.

### Last outbound

| Field | Value |
|---|---|
| Date | **2026-03-30** |
| Time | 14:32:52 ET |
| Type | ship |
| Qty | **−3,408.48 lb** (1,296 cases) |
| Lot | `SW2607890` |
| Txn | 573 |
| Shipment | 130 |
| Customer | Blue Stripes |
| SO | **SO-260311-001** |

Only outbound event on this SKU. `created_at` is 2026-08-11 10:32 ET (039 backfill).

### Inbound vs outbound since 2026-01-01 (posted)

| Direction | lb | cases @ 2.63 |
|---|---:|---:|
| Inbound (all pack) | 8,310.80 | 3,160 |
| Outbound (all ship) | 3,408.48 | 1,296 |
| Net / on-hand | 4,902.32 | 1,864 |

---

## Method notes

- Query joined current lines to current transaction headers so quantity/lot overlays and void status are the effective ones.
- In/out totals ignore voided lines. For these four SKUs that only changes CQ Granola 10 LB by ±10 lb (voided pair 469/470).
- `business_date` is the plant day. `occurred_at` is shown for last-outbound clocks. Several older ships have `created_at` = 2026-08-11 (Phase 1 backfill), not the event time.
- This file is a read-only extract. No ledger rows were written.

---

## 5. Follow-up — Sunshine SS shipments and the 2026-06-11 recon

SS-line finished SKUs checked: every `Granola SS %` finished product (70002, 70003, 70004, 70005, 70006, 70009, 70010, 70011, 70013–70018, 70070). Customer match: `customers.id = 217` Sunshine Granola, plus `customer_name` / `order_reference` / notes containing “sunshine”.

### All YTD Sunshine ships on SS SKUs

Every posted Sunshine outbound on the SS line is the **2026-06-11 `SUNSHINE-RECON-2026` series**. No other customer, no other date, no sales-order link.

| SKU | Product | Txn | Lot | qty_lb | cases @ 7.5 | shipment |
|---|---|---:|---|---:|---:|---:|
| 70003 | SS Chocolate Chip 12x10 | 1325 | 26-02-25 | −3,375.00 | 450 | 240 |
| 70003 | | 1326 | BB021627 | −4,500.00 | 600 | 241 |
| 70003 | | 1327 | BB022427 Lot | −562.50 | 75 | 242 |
| 70003 | | 1328 | BB030327 | −247.50 | 33 | 243 |
| 70003 | | 1329 | BB041327 | −6,750.00 | 900 | 244 |
| 70003 | | 1330 | BB041327 Lot | −2,250.00 | 300 | 245 |
| 70003 | | 1331 | BB050427 | −7,740.00 | 1,032 | 246 |
| 70003 | | 1332 | BB052627 | −2,250.00 | 300 | 247 |
| 70003 | | 1333 | BB060127 | −1,125.00 | 150 | 248 |
| 70003 | | 1334 | FEB 13 2026 | −2,865.00 | 382 | 249 |
| 70003 | | 1335 | FEB 18 2026 | −2,002.50 | 267 | 250 |
| 70003 | | 1336 | MAY 01 2026 | −1,125.00 | 150 | 251 |
| **70003 subtotal** | | | | **−34,792.50** | **4,639** | |
| 70070 | SS Choc Chip Low Carb 12x10 | 1337 | B26-0302-003 | −1,050.00 | 140 | 252 |
| 70002 | SS Original 12x10 | 1338 | BB022427 | −900.00 | 120 | 253 |
| 70002 | | 1339 | BB030327 | −2,017.50 | 269 | 254 |
| 70002 | | 1340 | BB041327 | −1,980.00 | 264 | 255 |
| 70002 | | 1341 | FEB 16 2026 | −1,852.50 | 247 | 256 |
| **70002 subtotal** | | | | **−6,750.00** | **900** | |
| 70010 | SS Original Low Carb 12x10 | 1342 | B26-0302-002 | −1,065.00 | 142 | 257 |
| **Recon total** | | | | **−43,657.50** | | |

No Sunshine ships on 70004, 70005, 70006, 70009, 70011 (Cranberry), or 70013–70018.

Each recon header is still `effective_status = posted`, `latest_correction_id` null. Notes: *“INTERIM — actual ship dates/orders pending backdated customer data. To be voided and replaced.”* `ledger_corrections` has **zero** rows targeting txns 1325–1342 or mentioning this recon.

### Were replacements posted?

**No.**

- No void/restore/amend on the 18 recon ships.
- No later Sunshine ship on any SS SKU (any date after 2026-06-11).
- No `sales_order_shipments` row tying those txns to an SO.
- Two Sunshine sales orders exist and are **unshipped**:

| SO | status | order_date | ship-by | Notes | Lines (ordered / shipped) |
|---|---|---|---|---|---|
| SO-260814-002 | confirmed | 2026-08-14 | 2026-08-26 | PO 81326 | 70003 13,500 lb (1,800 cs) / 0; 70002 4,500 / 0; 70011 2,250 / 0; 70010 1,875 / 0; 70070 1,875 / 0 |
| SO-260817-001 | confirmed | 2026-08-17 | 2026-08-28 | PO 08172026 | 70016 6,000 lb / 0; 70013 4,000 lb / 0 |

Those SOs are new demand, not replacements for the June 11 recon.

### 70003 on-hand if Sunshine shipping had continued

Facts that constrain the counterfactual:

- **Pre-June Sunshine ship count on 70003 is zero.** There is no dated pre-June Sunshine invoice series to compute a pace from.
- The only Sunshine outflow is the recon: **34,792.50 lb** booked on one day, lots spanning packs **2026-02-17 through 2026-06-01**.
- After the last recon ship (txn 1336), posted inbound is **29,535.00 lb** and posted outbound is **0**. That inbound is exactly current on-hand.

Two stated counterfactuals (not forecasts):

| Assumption | Extra outbound through 2026-08-17 | Implied on-hand |
|---|---:|---:|
| A. Sunshine kept taking **all** 70003 packs (what the recon did to stock through 6/1) | 29,535.00 | **0.00 lb** |
| B. Linearize recon over first pack → last pre-recon pack (2026-02-17 to 2026-06-01 = **104 days**) → **334.54 lb/day**, project **67 days** 2026-06-12 through 2026-08-17 | 22,414.40 | **7,120.60 lb** (950 cs @ 7.5) |

There is no third “pre-June actual invoice pace.” Using 0 lb/day (the ledger’s pre-June Sunshine rate) would leave on-hand at the current **29,535.00 lb**.

---

## 6. Follow-up — BS 6x7 SKUs vs Blue Stripes sales orders

SKUs: 70074 Dark Chocolate (id 150), 70080 Hazelnut (id 153), 70073 Peanut Butter Banana (id 151). Customer: `customers.id = 17` Blue Stripes. Window: `business_date` / `order_date` ≥ 2026-01-01.

### Posted outbound YTD (unique lines; join-dupes removed)

| Date | SKU | txn | qty_lb | cases @ 2.63 | lot | Linked SO | shipment | Notes |
|---|---|---:|---:|---:|---|---|---:|---|
| 2026-02-20 | 70073 | 297 | −756.00 | 287.5 | 26-02-10-FOUND-005 | **none** | 10 | standalone `order_reference=TO834` |
| 2026-02-20 | 70073 | 297 | −3,419.60 | 1,300.2 | 26-02-12-FOUND-001 | **none** | 10 | same txn |
| 2026-03-20 | 70080 | 503 | −2,428.40 | 923.3 | SW2606892 | SO-260318-003 | 36 | SO ship |
| 2026-03-30 | 70074 | 573 | −3,408.48 | 1,296.0 | SW2607890 | SO-260311-001 | 130 | SO ship |
| 2026-03-30 | 70073 | 575 | −40.15 | 15.3 | 26-02-12-FOUND-001 | SO-260311-001 | 130 | SO ship |
| 2026-03-30 | 70073 | 575 | −378.72 | 144.0 | SW2607708 | SO-260311-001 | 130 | SO ship |

| SKU | Posted outbound YTD | Last outbound |
|---|---:|---|
| 70073 | 4,594.47 lb | 2026-03-30 txn 575 (SO-260311-001) |
| 70074 | 3,408.48 lb | 2026-03-30 txn 573 (SO-260311-001) |
| 70080 | 2,428.40 lb | 2026-03-20 txn 503 (SO-260318-003) |

No posted outbound on these three SKUs after 2026-03-30.

### All Blue Stripes sales orders YTD

| SO | status | order_date | ship-by | PO / notes | 6x7 lines (ordered / shipped / line_status) | Matching posted ship txn? |
|---|---|---|---|---|---|---|
| SO-260206-011 | cancelled | 2026-02-06 | 2026-01-27 | S01079 / PO 1092 | 70074 2,880 / 0 / pending; 70073 2,016 / 0 / pending; 70080 2,592 / 0 / pending; also 70079 2,592 / 0 | **NO — flag** |
| SO-260311-001 | cancelled | 2026-03-11 | 2026-03-20 | PO1156 | 70074 7,488 / 3,408.48 / partial; 70073 2,165.80 / 418.87 / partial; also 70079 4,492.80 / 1,562.34 / partial | **YES** txns 573, 575 (and 574 for 70079) |
| SO-260318-003 | shipped | 2026-03-18 | — | packing-slip preview note | 70080 2,428.40 / 2,428.40 / fulfilled; also 70079 4,797 / 4,797; 70085 2,400 / 2,400 | **YES** txn 503 (and 502, 504 for other SKUs) |
| SO-260629-003 | confirmed | 2026-06-29 | 2026-07-10 | PO1229 | 70073 10,368 / 0 / pending (3,942 cs) | **NO — flag** |

### Flags

1. **SO-260206-011** — cancelled, `quantity_shipped_lb = 0` on every line, no `sales_order_shipments` rows. (Standalone Blue Stripes ship txn **297** on 2026-02-20 for 70073 **4,175.60 lb**, ref TO834, is **not** linked to this or any SO.)
2. **SO-260629-003** — still confirmed, 10,368 lb of 70073 ordered, **zero** shipped, **no** matching ship txn. Due 2026-07-10; now past due on the ledger.

SO-260311-001 is cancelled but **does** have matching ships (partial). It is not a “SO with no shipment” flag. It **is** an order whose remaining 70074 (4,079.52 lb) and 70073 (1,746.93 lb) will never ship unless a new SO is entered.

---

## 7. Follow-up — Classic #9 batch produced vs packed (2026-08-17 15:01 EDT)

Products (not the Kosher-Ignition SS twins 90025/90026):

| Product | id | SKU | lb/batch | Posted on-hand | System batches | Physical count | Gap |
|---|---:|---|---:|---:|---:|---:|---:|
| Batch Classic Granola #9 | 107 | 90002 | 323 | 34,316.00 lb | **106.24** | **31** | **75.24** (24,303 lb) |
| Batch Classic Chocolate Chip Granola #9 | 108 | 90001 | 348 | 17,748.00 lb | **51.00** | **0** | **51.00** (17,748 lb) |

Batch count = posted lb / `default_batch_lb`. Yield multiplier is 1.0 on both.

### Produced vs consumed (posted)

| | Classic #9 lb | Classic #9 batches | Choc Chip #9 lb | Choc Chip #9 batches |
|---|---:|---:|---:|---:|
| Added — make | 268,413.00 | 831.00 | 37,932.00 | 109.00 |
| Added — adjust+ | 5,859.00 | 18.14 | 0 | 0 |
| **Added total** | **274,272.00** | **849.14** | **37,932.00** | **109.00** |
| Consumed — pack | 205,505.00 | 636.24 | 900.00 | 2.59 |
| Consumed — adjust− | 33,991.00 | 105.24 | 19,284.00 | 55.41 |
| Consumed — make− | 460.00 | 1.42 | 0 | 0 |
| **Consumed total** | **239,956.00** | **742.90** | **20,184.00** | **58.00** |
| **Net on-hand** | **34,316.00** | **106.24** | **17,748.00** | **51.00** |

Voided (excluded above): Classic #9 txn 469/470 ±10 lb pack (2026-03-18); txn 863/864 ±2,907 lb make (2026-05-12, old reversal pair). Choc Chip #9: no voids.

Classic #9 pack destinations (from pack notes): CQ Granola 10 LB 148,420 lb (459.5 b); Granola Classic 25 LB 35,750 (110.7); Wheat Free 25 LB 6,400 (19.8); Crunchy CNS 10 LB 5,860 (18.1); SS Classic #9 25 LB 1,575 (4.9); unlabeled pack notes 7,500 (23.2, txns 94–98).

### Duplicate / near-duplicate production postings

**Classic #9 — same calendar day, more than one add:**

| date | txns | batches posted that day | Same lot? | Read |
|---|---|---:|---|---|
| 2026-02-05 | 72, 89, 92, 96 makes + 91 found-adjust | 16+16+8+16+4.64 = **60.64** | No — lots named FEB-03, FEB-04, FEB-02, FEB-05 | Backdated multi-day entry, not three copies of one run |
| 2026-02-10 | 159, 160 | 2+6 = 8 | No | Split, different lots |
| 2026-05-28 | 1077, 1095 | 7+10 = 17 | No (MAY 27 vs MAY 28) | Adjacent-day lots keyed same business_date |
| 2026-07-14 | 1644, 1649 | 19+8 = **27** | **Yes** `JUL 14 2026` | Two makes into one lot; not equal qty |
| 2026-07-15 | 1657, 1662 | 22+6 = **28** | Near (`JUL15` vs `JUL 15`) | Same pattern next day |
| 2026-07-16 | 1669, 1674 | 22+6 = **28** | **Yes** `JUL 16 2026` | Same pattern third day |
| 2026-08-12 | 1905 make + 1911 adjust+ | 18+4.88 | Different lots | Adjust onto AUG 11, not a second make |

Equal-qty pairs within 7 days are mostly the common **16-batch (5,168 lb)** day size (appears 11 times over 6 months) — that is a typical full bake day, not a cloned posting. July 14–16 are the only stretch that looks like **double-entry of a large day** (27–28 batches/day, twice the usual 16). Even if those three days were fully doubled (~40 extra batches), that does not reach the **75-batch** physical gap.

**Choc Chip #9:** 16 make txns, 16 distinct lot codes, no same-day doubles. Only near pairs: Jun 3 & Jun 10 both 6 batches; Jun 22 & Jun 23 both 5 batches (consecutive, plausible).

### Where the system on-hand sits

Classic #9 remaining **106.24 batches are almost all July–August lots** (105.55). Pre-July leftover is 0.69 batches (`JUN 30`). Feb–June production was packed/adjusted down to ~0. The physical-vs-system gap is therefore in **recent unconsumed lots**, not in the Feb 5 backfill.

Largest unused: AUG 12 18.00, AUG 13 18.00, JUL 27 14.00, AUG 11 10.63, JUL 21 6.57, … (see lot list in tables below).

Choc Chip #9 remaining **51.00 batches** are exactly the makes **after** the 2026-06-08 write-off (adjusts 1193–1201, −19,284 lb / −55.41 batches, which zeroed every older lot). Post-wipe makes: Jun 10 (6), Jun 22 (5), Jun 23 (5), Jul 7 (4), Jul 8 (2), Jul 22 (18), Aug 3 (11) = 51. **Zero pack deductions after June 8.** Lifetime pack from this batch SKU is only 900 lb (2.59 batches) into Granola Chocolate Chip 25 LB.

### Gap diagnosis

**Classic #9 (31 physical vs 106.24 system):** does **not** look like a pile of cloned make postings. Same-day extras are either backdated different lot-dates or July 14–16 split/double-shift entries (~40 batches at most). The leftover 106 is dated recent production that the ledger never fully packed off. That pattern fits **unrecorded or under-recorded packing/consumption of July–August batch** and/or **overstated batch counts on those make days**. It does **not** fit “the same 16-batch ticket posted twice and left sitting.” Distinguishing overstated makes vs missing packs needs a FG-side pack-weight check (CQ 10 LB / Classic 25 LB), which is outside this batch-only pull.

**Choc Chip #9 (0 physical vs 51 system):** also **not** same-day duplicate makes. After a June 8 cycle-count wipe, the ledger booked 51 new batches and **almost never deducted them into FG** (only 2.59 batches packed all year). With a physical zero, those 51 lots are either **makes that did not happen** or **batches that left the silo without a pack posting**. Because the pack path is nearly unused on this SKU, the gap is **missing consumption (or phantom production), not double-counted production tickets.**

Kosher SS twins 90025 / 90026 were not included.


### Classic #9 — every posted add and consume

#### Adds (74 posted lines)

| date | type | txn | lb | batches | lot | note |
|---|---|---:|---:|---:|---|---|
| 2026-02-03 | make | 58 | +1,292.0 | 4.00 | `JAN 30 2026` | 4 batch(es) of Batch Classic Granola #9 |
| 2026-02-04 | make | 67 | +646.0 | 2.00 | `B26-0204-001` | 2 batch(es) of Batch Classic Granola #9 |
| 2026-02-05 | make | 72 | +5,168.0 | 16.00 | `FEB-03-2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-02-05 | make | 89 | +5,168.0 | 16.00 | `FEB-04-26` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-02-05 | adjust | 91 | +1,500.0 | 4.64 | `26-02-05-FOUND-006` | Found inventory: predates_system |
| 2026-02-05 | make | 92 | +2,584.0 | 8.00 | `FEB-02-26` | 8 batch(es) of Batch Classic Granola #9 |
| 2026-02-05 | make | 96 | +5,168.0 | 16.00 | `FEB-05-2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-02-06 | make | 120 | +3,230.0 | 10.00 | `FEB 06 2026` | 10 batch(es) of Batch Classic Granola #9 |
| 2026-02-09 | make | 138 | +5,168.0 | 16.00 | `FEB 09 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-02-10 | make | 159 | +646.0 | 2.00 | `FEB 10 2026` | 2 batch(es) of Batch Classic Granola #9 |
| 2026-02-10 | make | 160 | +1,938.0 | 6.00 | `FEB 10 2026-02` | 6 batch(es) of Batch Classic Granola #9 |
| 2026-02-12 | adjust | 221 | +2,784.0 | 8.62 | `FEB 09 2026` | Adjustment: 2784.0 lb |
| 2026-02-26 | make | 339 | +969.0 | 3.00 | `FEB 26 2026` | 3 batch(es) of Batch Classic Granola #9 |
| 2026-03-03 | make | 362 | +4,522.0 | 14.00 | `MAR 03 2026` | 14 batch(es) of Batch Classic Granola #9 |
| 2026-03-04 | make | 366 | +1,615.0 | 5.00 | `B26-0304-001` | 5 batch(es) of Batch Classic Granola #9 |
| 2026-03-10 | make | 413 | +2,584.0 | 8.00 | `MAR 10 2026` | 8 batch(es) of Batch Classic Granola #9 |
| 2026-03-11 | make | 421 | +5,168.0 | 16.00 | `MAR 11 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-03-12 | make | 437 | +1,292.0 | 4.00 | `MAR 12 2026` | 4 batch(es) of Batch Classic Granola #9 |
| 2026-03-16 | make | 453 | +3,230.0 | 10.00 | `MAR 16 2026` | 10 batch(es) of Batch Classic Granola #9 |
| 2026-03-18 | make | 471 | +969.0 | 3.00 | `MAR 18 2026` | 3 batch(es) of Batch Classic Granola #9 |
| 2026-03-23 | make | 519 | +2,907.0 | 9.00 | `MAR 23 2026` | 9 batch(es) of Batch Classic Granola #9 |
| 2026-03-24 | make | 537 | +4,199.0 | 13.00 | `MAR 24 2026` | 13 batch(es) of Batch Classic Granola #9 |
| 2026-04-14 | make | 614 | +1,938.0 | 6.00 | `APR 13 2026` | 6 batch(es) of Batch Classic Granola #9 |
| 2026-04-16 | make | 633 | +3,230.0 | 10.00 | `2026-04-15` | 10 batch(es) of Batch Classic Granola #9 |
| 2026-04-21 | make | 672 | +969.0 | 3.00 | `2026-04-21` | 3 batch(es) of Batch Classic Granola #9 |
| 2026-04-23 | make | 691 | +4,522.0 | 14.00 | `ABR 23 2026` | 14 batch(es) of Batch Classic Granola #9 |
| 2026-04-28 | make | 739 | +2,584.0 | 8.00 | `APR 28 2026` | 8 batch(es) of Batch Classic Granola #9 |
| 2026-04-29 | make | 755 | +6,460.0 | 20.00 | `APR 29 2026` | 20 batch(es) of Batch Classic Granola #9 |
| 2026-05-04 | make | 795 | +1,615.0 | 5.00 | `26-05-04-GRAN-001` | 5 batch(es) of Batch Classic Granola #9 |
| 2026-05-05 | make | 811 | +1,938.0 | 6.00 | `26-05-05-GRAN-002` | 6 batch(es) of Batch Classic Granola #9 |
| 2026-05-06 | make | 821 | +969.0 | 3.00 | `26-05-06-GRAN-003` | 3 batch(es) of Batch Classic Granola #9 |
| 2026-05-08 | make | 844 | +2,907.0 | 9.00 | `MAY 08 2026` | 9 batch(es) of Batch Classic Granola #9 |
| 2026-05-12 | make | 865 | +1,938.0 | 6.00 | `MAY 12 2026` | 6 batch(es) of Batch Classic Granola #9 |
| 2026-05-13 | make | 882 | +4,522.0 | 14.00 | `MAY 13 2026` | 14 batch(es) of Batch Classic Granola #9 |
| 2026-05-14 | make | 950 | +646.0 | 2.00 | `MAY 14 2026` | 2 batch(es) of Batch Classic Granola #9 |
| 2026-05-19 | make | 1007 | +3,230.0 | 10.00 | `MAY 19 2026` | 10 batch(es) of Batch Classic Granola #9 |
| 2026-05-20 | make | 1025 | +5,814.0 | 18.00 | `MAY 20 2026` | 18 batch(es) of Batch Classic Granola #9 |
| 2026-05-21 | make | 1043 | +5,168.0 | 16.00 | `MAY 21 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-05-28 | make | 1077 | +2,261.0 | 7.00 | `MAY 27 2026` | 7 batch(es) of Batch Classic Granola #9 |
| 2026-05-28 | make | 1095 | +3,230.0 | 10.00 | `MAY 28 2026` | 10 batch(es) of Batch Classic Granola #9 |
| 2026-06-01 | make | 1128 | +5,168.0 | 16.00 | `JUN 01 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-06-02 | make | 1135 | +5,168.0 | 16.00 | `JUN 02 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-06-09 | make | 1295 | +1,292.0 | 4.00 | `JUN 09 2026` | 4 batch(es) of Batch Classic Granola #9 |
| 2026-06-25 | make | 1458 | +1,615.0 | 5.00 | `JUN 25 2026` | 5 batch(es) of Batch Classic Granola #9 |
| 2026-06-26 | make | 1464 | +4,845.0 | 15.00 | `JUN 26 2026` | 15 batch(es) of Batch Classic Granola #9 |
| 2026-06-28 | make | 1468 | +4,845.0 | 15.00 | `JUN 28 2026` | 15 batch(es) of Batch Classic Granola #9 |
| 2026-06-29 | make | 1484 | +3,230.0 | 10.00 | `JUN 29 2026` | 10 batch(es) of Batch Classic Granola #9 |
| 2026-06-30 | make | 1509 | +4,199.0 | 13.00 | `JUN 30 2026` | 13 batch(es) of Batch Classic Granola #9 |
| 2026-07-01 | make | 1534 | +3,553.0 | 11.00 | `Jul 01 2026` | 11 batch(es) of Batch Classic Granola #9 |
| 2026-07-02 | make | 1544 | +7,429.0 | 23.00 | `JUL 2 2026` | 23 batch(es) of Batch Classic Granola #9 |
| 2026-07-06 | make | 1553 | +8,075.0 | 25.00 | `JUL 06 2026` | 25 batch(es) of Batch Classic Granola #9 |
| 2026-07-07 | make | 1564 | +3,876.0 | 12.00 | `JUL 07 2026` | 12 batch(es) of Batch Classic Granola #9 |
| 2026-07-10 | make | 1614 | +4,199.0 | 13.00 | `JUL 10 2026` | 13 batch(es) of Batch Classic Granola #9 |
| 2026-07-13 | make | 1642 | +8,075.0 | 25.00 | `JUL 13 2026` | 25 batch(es) of Batch Classic Granola #9 |
| 2026-07-14 | make | 1644 | +6,137.0 | 19.00 | `JUL 14 2026` | 19 batch(es) of Batch Classic Granola #9 |
| 2026-07-14 | make | 1649 | +2,584.0 | 8.00 | `JUL 14 2026` | 8 batch(es) of Batch Classic Granola #9 |
| 2026-07-15 | make | 1657 | +7,106.0 | 22.00 | `JUL15 2026` | 22 batch(es) of Batch Classic Granola #9 |
| 2026-07-15 | make | 1662 | +1,938.0 | 6.00 | `JUL 15 2026` | 6 batch(es) of Batch Classic Granola #9 |
| 2026-07-16 | make | 1669 | +7,106.0 | 22.00 | `JUL 16 2026` | 22 batch(es) of Batch Classic Granola #9 |
| 2026-07-16 | make | 1674 | +1,938.0 | 6.00 | `JUL 16 2026` | 6 batch(es) of Batch Classic Granola #9 |
| 2026-07-17 | make | 1675 | +5,814.0 | 18.00 | `JUL 17 2026` | 18 batch(es) of Batch Classic Granola #9 |
| 2026-07-21 | make | 1694 | +4,522.0 | 14.00 | `JUL 21 2026` | 14 batch(es) of Batch Classic Granola #9 |
| 2026-07-27 | make | 1753 | +4,522.0 | 14.00 | `JUL 27 2026` | 14 batch(es) of Batch Classic Granola #9 |
| 2026-07-28 | make | 1777 | +3,876.0 | 12.00 | `JUL 28 2026` | 12 batch(es) of Batch Classic Granola #9 |
| 2026-07-29 | make | 1782 | +5,814.0 | 18.00 | `Jul 29 2026` | 18 batch(es) of Batch Classic Granola #9 |
| 2026-07-30 | make | 1789 | +6,137.0 | 19.00 | `JUL 30 2026` | 19 batch(es) of Batch Classic Granola #9 |
| 2026-07-31 | make | 1797 | +4,199.0 | 13.00 | `JUL 31 2026` | 13 batch(es) of Batch Classic Granola #9 |
| 2026-08-03 | make | 1812 | +1,615.0 | 5.00 | `AUG 03 2026` | 5 batch(es) of Batch Classic Granola #9 |
| 2026-08-04 | make | 1835 | +5,168.0 | 16.00 | `AUG 04 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-08-10 | make | 1886 | +5,168.0 | 16.00 | `AUG 10 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-08-11 | make | 1892 | +5,168.0 | 16.00 | `AUG 11 2026` | 16 batch(es) of Batch Classic Granola #9 |
| 2026-08-12 | make | 1905 | +5,814.0 | 18.00 | `AUG 12 2026` | 18 batch(es) of Batch Classic Granola #9 |
| 2026-08-12 | adjust | 1911 | +1,575.0 | 4.88 | `AUG 11 2026` | Adjustment: 1575.0 lb |
| 2026-08-13 | make | 1917 | +5,814.0 | 18.00 | `AUG 13 2026` | 18 batch(es) of Batch Classic Granola #9 |

#### Consumes (171 posted lines)

| date | type | txn | lb | batches | lot | dest / note |
|---|---|---:|---:|---:|---|---|
| 2026-02-05 | adjust | 93 | -1,500.0 | -4.64 | `26-02-05-FOUND-006` | Adjustment: -1500.0 lb |
| 2026-02-05 | pack | 94 | -1,292.0 | -4.00 | `JAN 30 2026` |  |
| 2026-02-05 | pack | 95 | -208.0 | -0.64 | `FEB-02-26` |  |
| 2026-02-05 | pack | 97 | -3,000.0 | -9.29 | `FEB-03-2026` |  |
| 2026-02-05 | pack | 98 | -3,000.0 | -9.29 | `FEB-04-26` |  |
| 2026-02-05 | adjust | 99 | -646.0 | -2.00 | `B26-0204-001` | Adjustment: -646.0 lb |
| 2026-02-05 | adjust | 100 | -2,376.0 | -7.36 | `FEB-02-26` | Adjustment: -2376.0 lb |
| 2026-02-05 | adjust | 101 | -2,168.0 | -6.71 | `FEB-03-2026` | Adjustment: -2168.0 lb |
| 2026-02-05 | adjust | 102 | -810.0 | -2.51 | `FEB-04-26` | Adjustment: -810.0 lb |
| 2026-02-05 | adjust | 103 | -1,358.0 | -4.20 | `FEB-04-26` | Adjustment: -1358.0 lb |
| 2026-02-05 | adjust | 104 | -42.0 | -0.13 | `FEB-05-2026` | Adjustment: -42.0 lb |
| 2026-02-06 | adjust | 123 | -1,400.0 | -4.33 | `FEB 06 2026` | Adjustment: -1400.0 lb |
| 2026-02-09 | adjust | 142 | -2,800.0 | -8.67 | `FEB 09 2026` | Adjustment: -2800.0 lb |
| 2026-02-12 | adjust | 219 | -5,126.0 | -15.87 | `FEB-05-2026` | Adjustment: -5126.0 lb |
| 2026-02-12 | adjust | 220 | -220.0 | -0.68 | `FEB 06 2026` | Adjustment: -220.0 lb |
| 2026-02-12 | adjust | 222 | -2.0 | -0.01 | `FEB 10 2026` | Adjustment: -2.0 lb |
| 2026-02-12 | adjust | 223 | -6.0 | -0.02 | `FEB 10 2026-02` | Adjustment: -6.0 lb |
| 2026-02-16 | pack | 245 | -20.0 | -0.06 | `FEB 06 2026` | CQ Granola 10 LB |
| 2026-02-19 | pack | 281 | -600.0 | -1.86 | `FEB 06 2026` | Granola Classic 25 LB |
| 2026-02-19 | make | 283 | -20.0 | -0.06 | `FEB 06 2026` | 1 batch(es) of Batch Granola Fruit Nut |
| 2026-03-03 | pack | 351 | -375.0 | -1.16 | `FEB 09 2026` | Granola Classic 25 LB |
| 2026-03-04 | pack | 371 | -1,500.0 | -4.64 | `B26-0304-001` | Granola Classic 25 LB |
| 2026-03-06 | pack | 387 | -40.0 | -0.12 | `FEB 06 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-03-10 | pack | 405 | -150.0 | -0.46 | `FEB 06 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-03-10 | pack | 415 | -780.0 | -2.41 | `FEB 06 2026` | CQ Granola 10 LB |
| 2026-03-10 | pack | 415 | -2,020.0 | -6.25 | `FEB 09 2026` | CQ Granola 10 LB |
| 2026-03-11 | pack | 422 | -1,250.0 | -3.87 | `FEB 09 2026` | Granola Classic 25 LB |
| 2026-03-11 | pack | 423 | -1,400.0 | -4.33 | `FEB 09 2026` | CQ Granola 10 LB |
| 2026-03-12 | pack | 427 | -107.0 | -0.33 | `FEB 09 2026` | Granola Classic 25 LB |
| 2026-03-12 | pack | 427 | -644.0 | -1.99 | `FEB 10 2026` | Granola Classic 25 LB |
| 2026-03-12 | pack | 427 | -499.0 | -1.54 | `FEB 10 2026-02` | Granola Classic 25 LB |
| 2026-03-12 | pack | 435 | -1,250.0 | -3.87 | `FEB 10 2026-02` | Granola Classic 25 LB |
| 2026-03-12 | pack | 440 | -183.0 | -0.57 | `FEB 10 2026-02` | CQ Granola 10 LB |
| 2026-03-12 | pack | 440 | -969.0 | -3.00 | `FEB 26 2026` | CQ Granola 10 LB |
| 2026-03-12 | pack | 440 | -248.0 | -0.77 | `MAR 03 2026` | CQ Granola 10 LB |
| 2026-03-12 | pack | 441 | -2,800.0 | -8.67 | `MAR 03 2026` | CQ Granola 10 LB |
| 2026-03-13 | pack | 447 | -1,474.0 | -4.56 | `MAR 03 2026` | CQ Granola 10 LB |
| 2026-03-13 | pack | 447 | -115.0 | -0.36 | `B26-0304-001` | CQ Granola 10 LB |
| 2026-03-13 | pack | 447 | -1,211.0 | -3.75 | `MAR 10 2026` | CQ Granola 10 LB |
| 2026-03-16 | pack | 456 | -1,373.0 | -4.25 | `MAR 10 2026` | CQ Granola 10 LB |
| 2026-03-16 | pack | 456 | -1,427.0 | -4.42 | `MAR 11 2026` | CQ Granola 10 LB |
| 2026-03-16 | pack | 457 | -1,400.0 | -4.33 | `MAR 11 2026` | CQ Granola 10 LB |
| 2026-03-24 | pack | 533 | -2,341.0 | -7.25 | `MAR 11 2026` | Granola Classic 25 LB |
| 2026-03-24 | pack | 533 | -159.0 | -0.49 | `MAR 12 2026` | Granola Classic 25 LB |
| 2026-03-24 | pack | 534 | -1,000.0 | -3.10 | `MAR 12 2026` | Granola Classic 25 LB |
| 2026-03-24 | pack | 535 | -133.0 | -0.41 | `MAR 12 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-03-24 | pack | 535 | -367.0 | -1.14 | `MAR 16 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-03-24 | pack | 536 | -350.0 | -1.08 | `MAR 16 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-03-27 | pack | 562 | -1,500.0 | -4.64 | `MAR 16 2026` | Granola Classic 25 LB |
| 2026-03-27 | pack | 563 | -1,013.0 | -3.14 | `MAR 16 2026` | Granola Classic 25 LB |
| 2026-03-27 | pack | 563 | -487.0 | -1.51 | `MAR 18 2026` | Granola Classic 25 LB |
| 2026-03-30 | pack | 580 | -250.0 | -0.77 | `MAR 18 2026` | Granola Wheat Free 25 LB |
| 2026-04-20 | pack | 660 | -232.0 | -0.72 | `MAR 18 2026` | Granola Wheat Free 25 LB |
| 2026-04-20 | pack | 660 | -768.0 | -2.38 | `MAR 23 2026` | Granola Wheat Free 25 LB |
| 2026-04-20 | pack | 661 | -250.0 | -0.77 | `MAR 23 2026` | Granola Classic 25 LB |
| 2026-04-21 | pack | 677 | -1,500.0 | -4.64 | `MAR 23 2026` | Granola Classic 25 LB |
| 2026-04-27 | pack | 702 | -389.0 | -1.20 | `MAR 23 2026` | Granola Classic 25 LB |
| 2026-04-27 | pack | 702 | -811.0 | -2.51 | `MAR 24 2026` | Granola Classic 25 LB |
| 2026-04-28 | pack | 735 | -100.0 | -0.31 | `MAR 24 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-04-29 | pack | 756 | -3,288.0 | -10.18 | `MAR 24 2026` | CQ Granola 10 LB |
| 2026-04-29 | pack | 756 | -1,938.0 | -6.00 | `APR 13 2026` | CQ Granola 10 LB |
| 2026-04-29 | pack | 756 | -3,230.0 | -10.00 | `2026-04-15` | CQ Granola 10 LB |
| 2026-04-29 | pack | 756 | -969.0 | -3.00 | `2026-04-21` | CQ Granola 10 LB |
| 2026-04-29 | pack | 756 | -375.0 | -1.16 | `ABR 23 2026` | CQ Granola 10 LB |
| 2026-05-01 | pack | 780 | -1,400.0 | -4.33 | `ABR 23 2026` | CQ Granola 10 LB |
| 2026-05-01 | pack | 787 | -1,500.0 | -4.64 | `ABR 23 2026` | Granola Classic 25 LB |
| 2026-05-06 | pack | 818 | -150.0 | -0.46 | `ABR 23 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-05-06 | pack | 823 | -1,097.0 | -3.40 | `ABR 23 2026` | Granola Classic 25 LB |
| 2026-05-06 | pack | 823 | -403.0 | -1.25 | `APR 28 2026` | Granola Classic 25 LB |
| 2026-05-06 | pack | 824 | -625.0 | -1.93 | `APR 28 2026` | Granola Classic 25 LB |
| 2026-05-06 | pack | 825 | -200.0 | -0.62 | `APR 28 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-05-12 | pack | 870 | -700.0 | -2.17 | `MAY 12 2026` | Granola Classic 25 LB |
| 2026-05-13 | pack | 883 | -1,356.0 | -4.20 | `APR 28 2026` | Granola Wheat Free 25 LB |
| 2026-05-13 | pack | 883 | -644.0 | -1.99 | `APR 29 2026` | Granola Wheat Free 25 LB |
| 2026-05-13 | pack | 884 | -1,500.0 | -4.64 | `MAY 08 2026` | Granola Classic 25 LB |
| 2026-05-13 | pack | 885 | -1,500.0 | -4.64 | `MAY 13 2026` | Granola Classic 25 LB |
| 2026-05-14 | make | 891 | -20.0 | -0.06 | `APR 29 2026` | 1 batch(es) of Batch Granola Fruit Nut |
| 2026-05-14 | make | 892 | -380.0 | -1.18 | `APR 29 2026` | 19 batch(es) of Batch Granola Fruit Nut |
| 2026-05-14 | pack | 951 | -3,000.0 | -9.29 | `MAY 13 2026` | Granola Classic 25 LB |
| 2026-05-20 | pack | 1011 | -150.0 | -0.46 | `MAY 19 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-05-20 | pack | 1012 | -575.0 | -1.78 | `MAY 19 2026` | Granola Classic 25 LB |
| 2026-05-21 | pack | 1044 | -600.0 | -1.86 | `MAY 19 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-05-26 | pack | 1057 | -1,905.0 | -5.90 | `MAY 19 2026` | CQ Granola 10 LB |
| 2026-05-26 | pack | 1057 | -895.0 | -2.77 | `APR 29 2026` | CQ Granola 10 LB |
| 2026-05-26 | pack | 1058 | -2,800.0 | -8.67 | `MAY 20 2026` | CQ Granola 10 LB |
| 2026-06-01 | pack | 1113 | -1,400.0 | -4.33 | `APR 29 2026` | CQ Granola 10 LB |
| 2026-06-01 | pack | 1114 | -1,400.0 | -4.33 | `APR 29 2026` | CQ Granola 10 LB |
| 2026-06-01 | pack | 1120 | -1,000.0 | -3.10 | `MAY 21 2026` | Granola Wheat Free 25 LB |
| 2026-06-01 | pack | 1121 | -560.0 | -1.73 | `MAY 21 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-06-04 | pack | 1169 | -2,800.0 | -8.67 | `MAY 21 2026` | CQ Granola 10 LB |
| 2026-06-04 | pack | 1170 | -2,800.0 | -8.67 | `MAY 28 2026` | CQ Granola 10 LB |
| 2026-06-05 | pack | 1177 | -1,400.0 | -4.33 | `JUN 01 2026` | CQ Granola 10 LB |
| 2026-06-05 | pack | 1178 | -430.0 | -1.33 | `MAY 28 2026` | CQ Granola 10 LB |
| 2026-06-05 | pack | 1178 | -1,721.0 | -5.33 | `APR 29 2026` | CQ Granola 10 LB |
| 2026-06-05 | pack | 1178 | -649.0 | -2.01 | `26-05-04-GRAN-001` | CQ Granola 10 LB |
| 2026-06-08 | adjust | 1182 | -966.0 | -2.99 | `26-05-04-GRAN-001` | Adjustment: -966 lb |
| 2026-06-08 | adjust | 1183 | -1,938.0 | -6.00 | `26-05-05-GRAN-002` | Adjustment: -1938 lb |
| 2026-06-08 | adjust | 1184 | -969.0 | -3.00 | `26-05-06-GRAN-003` | Adjustment: -969 lb |
| 2026-06-08 | adjust | 1185 | -1,407.0 | -4.36 | `MAY 08 2026` | Adjustment: -1407 lb |
| 2026-06-08 | adjust | 1186 | -1,238.0 | -3.83 | `MAY 12 2026` | Adjustment: -1238 lb |
| 2026-06-08 | adjust | 1187 | -22.0 | -0.07 | `MAY 13 2026` | Adjustment: -22 lb |
| 2026-06-08 | adjust | 1188 | -646.0 | -2.00 | `MAY 14 2026` | Adjustment: -646 lb |
| 2026-06-08 | adjust | 1189 | -3,014.0 | -9.33 | `MAY 20 2026` | Adjustment: -3014 lb |
| 2026-06-08 | adjust | 1190 | -808.0 | -2.50 | `MAY 21 2026` | Adjustment: -808 lb |
| 2026-06-08 | adjust | 1191 | -2,261.0 | -7.00 | `MAY 27 2026` | Adjustment: -2261 lb |
| 2026-06-08 | adjust | 1192 | -2,268.0 | -7.02 | `JUN 01 2026` | Adjustment: -2268 lb |
| 2026-06-08 | pack | 1180 | -1,500.0 | -4.64 | `JUN 01 2026` | Granola Classic 25 LB |
| 2026-06-08 | pack | 1181 | -300.0 | -0.93 | `JUN 02 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-06-19 | pack | 1403 | -300.0 | -0.93 | `JUN 02 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-06-19 | pack | 1404 | -150.0 | -0.46 | `JUN 02 2026` | Granola Wheat Free 25 LB |
| 2026-06-23 | pack | 1426 | -100.0 | -0.31 | `JUN 02 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-06-24 | pack | 1441 | -1,400.0 | -4.33 | `JUN 02 2026` | CQ Granola 10 LB |
| 2026-06-26 | pack | 1465 | -1,400.0 | -4.33 | `JUN 02 2026` | CQ Granola 10 LB |
| 2026-06-26 | pack | 1466 | -1,400.0 | -4.33 | `JUN 02 2026` | CQ Granola 10 LB |
| 2026-06-29 | pack | 1476 | -1,400.0 | -4.33 | `JUN 25 2026` | CQ Granola 10 LB |
| 2026-06-29 | pack | 1477 | -215.0 | -0.67 | `JUN 25 2026` | CQ Granola 10 LB |
| 2026-06-29 | pack | 1477 | -1,185.0 | -3.67 | `JUN 26 2026` | CQ Granola 10 LB |
| 2026-06-29 | pack | 1478 | -4,200.0 | -13.00 | `JUN 28 2026` | CQ Granola 10 LB |
| 2026-06-29 | pack | 1490 | -600.0 | -1.86 | `JUN 29 2026` | Granola Classic 25 LB |
| 2026-06-29 | make | 1494 | -40.0 | -0.12 | `JUN 02 2026` | 2 batch(es) of Granola Fruit Nut Batch |
| 2026-07-01 | pack | 1531 | -78.0 | -0.24 | `JUN 02 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-07-01 | pack | 1531 | -22.0 | -0.07 | `JUN 09 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-07-01 | pack | 1535 | -1,400.0 | -4.33 | `JUN 29 2026` | CQ Granola 10 LB |
| 2026-07-01 | pack | 1536 | -2,800.0 | -8.67 | `JUN 30 2026` | CQ Granola 10 LB |
| 2026-07-01 | pack | 1537 | -2,800.0 | -8.67 | `Jul 01 2026` | CQ Granola 10 LB |
| 2026-07-02 | pack | 1545 | -1,270.0 | -3.93 | `JUN 09 2026` | CQ Granola 10 LB |
| 2026-07-02 | pack | 1545 | -130.0 | -0.40 | `JUN 26 2026` | CQ Granola 10 LB |
| 2026-07-02 | pack | 1546 | -4,200.0 | -13.00 | `JUL 2 2026` | CQ Granola 10 LB |
| 2026-07-06 | pack | 1554 | -1,400.0 | -4.33 | `JUL 2 2026` | CQ Granola 10 LB |
| 2026-07-06 | pack | 1555 | -1,400.0 | -4.33 | `JUL 2 2026` | CQ Granola 10 LB |
| 2026-07-06 | pack | 1556 | -4,200.0 | -13.00 | `JUL 06 2026` | CQ Granola 10 LB |
| 2026-07-07 | pack | 1560 | -2,800.0 | -8.67 | `JUL 06 2026` | CQ Granola 10 LB |
| 2026-07-07 | pack | 1567 | -750.0 | -2.32 | `JUL 07 2026` | Granola Classic 25 LB |
| 2026-07-07 | pack | 1568 | -1,400.0 | -4.33 | `JUL 07 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-07-14 | pack | 1645 | -4,190.0 | -12.97 | `JUL 10 2026` | CQ Granola 10 LB |
| 2026-07-14 | pack | 1646 | -10.0 | -0.03 | `JUN 26 2026` | CQ Granola 10 LB |
| 2026-07-14 | pack | 1647 | -4,200.0 | -13.00 | `JUL 13 2026` | CQ Granola 10 LB |
| 2026-07-15 | pack | 1658 | -2,800.0 | -8.67 | `JUL 13 2026` | CQ Granola 10 LB |
| 2026-07-15 | pack | 1659 | -8,400.0 | -26.01 | `JUL 14 2026` | CQ Granola 10 LB |
| 2026-07-15 | pack | 1660 | -1,400.0 | -4.33 | `JUL15 2026` | CQ Granola 10 LB |
| 2026-07-16 | pack | 1671 | -5,700.0 | -17.65 | `JUL15 2026` | CQ Granola 10 LB |
| 2026-07-16 | pack | 1672 | -1,300.0 | -4.02 | `JUN 26 2026` | CQ Granola 10 LB |
| 2026-07-16 | pack | 1673 | -2,800.0 | -8.67 | `JUL 16 2026` | CQ Granola 10 LB |
| 2026-07-17 | pack | 1676 | -5,600.0 | -17.34 | `JUL 16 2026` | CQ Granola 10 LB |
| 2026-07-20 | pack | 1681 | -2,000.0 | -6.19 | `JUL 17 2026` | Granola Wheat Free 25 LB |
| 2026-07-21 | pack | 1696 | -3,000.0 | -9.29 | `JUL 17 2026` | Granola Classic 25 LB |
| 2026-07-21 | pack | 1697 | -1,000.0 | -3.10 | `JUL 21 2026` | Granola Classic 25 LB |
| 2026-07-27 | pack | 1754 | -300.0 | -0.93 | `JUN 26 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-07-28 | pack | 1781 | -1,400.0 | -4.33 | `JUL 21 2026` | CQ Granola 10 LB |
| 2026-07-29 | pack | 1783 | -1,920.0 | -5.94 | `JUN 26 2026` | CQ Granola 10 LB |
| 2026-07-29 | pack | 1783 | -645.0 | -2.00 | `JUN 28 2026` | CQ Granola 10 LB |
| 2026-07-29 | pack | 1783 | -235.0 | -0.73 | `JUN 29 2026` | CQ Granola 10 LB |
| 2026-07-29 | pack | 1784 | -2,800.0 | -8.67 | `JUL 28 2026` | CQ Granola 10 LB |
| 2026-07-30 | pack | 1791 | -1,070.0 | -3.31 | `JUL 28 2026` | CQ Granola 10 LB |
| 2026-07-30 | pack | 1792 | -330.0 | -1.02 | `JUN 29 2026` | CQ Granola 10 LB |
| 2026-07-30 | pack | 1793 | -5,600.0 | -17.34 | `Jul 29 2026` | CQ Granola 10 LB |
| 2026-07-30 | pack | 1794 | -2,800.0 | -8.67 | `JUL 30 2026` | CQ Granola 10 LB |
| 2026-07-31 | pack | 1798 | -2,800.0 | -8.67 | `JUL 30 2026` | CQ Granola 10 LB |
| 2026-07-31 | pack | 1799 | -2,800.0 | -8.67 | `JUL 31 2026` | CQ Granola 10 LB |
| 2026-08-03 | pack | 1802 | -665.0 | -2.06 | `JUN 29 2026` | CQ Granola 10 LB |
| 2026-08-03 | pack | 1802 | -735.0 | -2.28 | `JUN 30 2026` | CQ Granola 10 LB |
| 2026-08-03 | pack | 1815 | -750.0 | -2.32 | `AUG 03 2026` | Granola Classic 25 LB |
| 2026-08-04 | pack | 1844 | -400.0 | -1.24 | `AUG 03 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-08-07 | pack | 1877 | -4,200.0 | -13.00 | `AUG 04 2026` | CQ Granola 10 LB |
| 2026-08-11 | pack | 1893 | -4,200.0 | -13.00 | `AUG 10 2026` | CQ Granola 10 LB |
| 2026-08-11 | pack | 1894 | -960.0 | -2.97 | `AUG 10 2026` | CQ Granola 10 LB |
| 2026-08-11 | pack | 1895 | -440.0 | -1.36 | `JUN 30 2026` | CQ Granola 10 LB |
| 2026-08-11 | pack | 1896 | -110.0 | -0.34 | `AUG 11 2026` | Granola Crunchy CNS 10 LB Case |
| 2026-08-12 | pack | 1907 | -1,575.0 | -4.88 | `AUG 11 2026` | Granola SS Classic #9 25 LB |
| 2026-08-12 | pack | 1912 | -1,575.0 | -4.88 | `AUG 11 2026` | Granola Classic 25 LB |
| 2026-08-14 | pack | 1931 | -50.0 | -0.15 | `AUG 11 2026` | Granola Crunchy CNS 10 LB Case |



### Choc Chip #9 — every posted add and consume

#### Adds (16 posted lines)

| date | type | txn | lb | batches | lot | note |
|---|---|---:|---:|---:|---|---|
| 2026-02-10 | make | 162 | +1,740.0 | 5.00 | `FEB 10 2026` | 5 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-02-26 | make | 338 | +1,044.0 | 3.00 | `FEB 26 2026` | 3 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-03-10 | make | 400 | +2,088.0 | 6.00 | `MAR 09 2026` | 6 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-04-20 | make | 656 | +3,132.0 | 9.00 | `2026-04-20` | 9 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-04-27 | make | 700 | +696.0 | 2.00 | `APR 24 2026` | 2 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-04-28 | make | 740 | +2,436.0 | 7.00 | `APR 28 2026` | 7 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-05-11 | make | 852 | +3,828.0 | 11.00 | `MAY 11 2026` | 11 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-05-28 | make | 1078 | +3,132.0 | 9.00 | `MAY 27 2026` | 9 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-06-03 | make | 1149 | +2,088.0 | 6.00 | `JUN 03 2026` | 6 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-06-10 | make | 1313 | +2,088.0 | 6.00 | `JUN 10 2026` | 6 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-06-22 | make | 1410 | +1,740.0 | 5.00 | `JUN 22 2026` | 5 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-06-23 | make | 1431 | +1,740.0 | 5.00 | `JUN 23 2026` | 5 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-07-07 | make | 1566 | +1,392.0 | 4.00 | `JUL 07 2026` | 4 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-07-08 | make | 1584 | +696.0 | 2.00 | `JUL 08 2026` | 2 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-07-22 | make | 1716 | +6,264.0 | 18.00 | `JUL 22 2026` | 18 batch(es) of Batch Classic Chocolate Chip Granola #9 |
| 2026-08-03 | make | 1811 | +3,828.0 | 11.00 | `AUG 03 2026` | 11 batch(es) of Batch Classic Chocolate Chip Granola #9 |

#### Consumes (12 posted lines)

| date | type | txn | lb | batches | lot | dest / note |
|---|---|---:|---:|---:|---|---|
| 2026-02-12 | adjust | 224 | -15.0 | -0.04 | `FEB 10 2026` | Adjustment: -15.0 lb |
| 2026-04-27 | pack | 703 | -600.0 | -1.72 | `APR 24 2026` | Granola Chocolate Chip 25 LB |
| 2026-05-12 | pack | 868 | -300.0 | -0.86 | `MAY 11 2026` | Granola Chocolate Chip 25 LB |
| 2026-06-08 | adjust | 1193 | -1,725.0 | -4.96 | `FEB 10 2026` | Adjustment: -1725 lb |
| 2026-06-08 | adjust | 1194 | -1,044.0 | -3.00 | `FEB 26 2026` | Adjustment: -1044 lb |
| 2026-06-08 | adjust | 1195 | -2,088.0 | -6.00 | `MAR 09 2026` | Adjustment: -2088 lb |
| 2026-06-08 | adjust | 1196 | -3,132.0 | -9.00 | `2026-04-20` | Adjustment: -3132 lb |
| 2026-06-08 | adjust | 1197 | -96.0 | -0.28 | `APR 24 2026` | Adjustment: -96 lb |
| 2026-06-08 | adjust | 1198 | -2,436.0 | -7.00 | `APR 28 2026` | Adjustment: -2436 lb |
| 2026-06-08 | adjust | 1199 | -3,528.0 | -10.14 | `MAY 11 2026` | Adjustment: -3528 lb |
| 2026-06-08 | adjust | 1200 | -3,132.0 | -9.00 | `MAY 27 2026` | Adjustment: -3132 lb |
| 2026-06-08 | adjust | 1201 | -2,088.0 | -6.00 | `JUN 03 2026` | Adjustment: -2088 lb |

---

## 8. Follow-up — last 90 days in/out, all items, and whether SO ship deducts inventory

**Window:** `business_date` **2026-05-19 through 2026-08-17** (90 days back from 2026-08-17; 91 inclusive dates).  
**Unit of “entry”:** one posted `ledger_current_transaction_lines` row (signed qty on a product/lot). Inbound = `quantity_lb > 0`; outbound = `quantity_lb < 0`.  
**Pulled:** 2026-08-17 15:02 EDT, read-only.

### Plant-wide totals

| | Lines | lb |
|---|---:|---:|
| Inbound | 571 | 1,528,488.52 |
| Outbound | 1,786 | 1,537,082.79 |
| Ratio in/out | **0.32** | **0.99** |

Line count is inbound-light because one make writes one +output line and many −ingredient lines. **Pounds nearly balance** (in/out 0.99). That is not a plant that stopped deducting inventory.

By `products.type`:

| type | products | in_lines | out_lines | in/out lines | in_lb | out_lb |
|---|---:|---:|---:|---:|---:|---:|
| batch | 20 | 164 | 336 | 0.49 | 513,723 | 472,726 |
| finished | 37 | 304 | 395 | 0.77 | 544,457 | 526,243 |
| ingredient | 48 | 103 | 1,055 | 0.10 | 470,308 | 538,114 |

### Flag: inbound in window, zero outbound

**4 products** (of 105 with any posted activity):

| sku | type | in_lines | in_lb | How inbound | Outbound in window |
|---|---|---:|---:|---|---|
| 70074 | finished | 1 | 1,136.16 | pack 2026-07-24 txn 1747 (432 cs from Batch BS Dark Chocolate) | **none** (last ship was 2026-03-30, outside window) |
| 70073 | finished | 2 | 3,613.62 | pack 2026-07-29 txns 1786+1788 (1,374 cs from Batch BS PBB) | **none** (last ship 2026-03-30). Matches open SO-260629-003 still unshipped |
| 70011 | finished | 1 | 1,342.50 | pack 2026-07-08 txn 1590 (179 cs SS Cranberry) | **none** (never in Sunshine recon; SO-260814-002 still pending) |
| 11028 | ingredient | 2 | 225.00 | receive | **none** (not consumed in a make) |

21 other products had outbound only (drew down older stock). 70080 (BS Hazelnut 6x7) had **no** posted lines in the window.

Classic Chocolate Chip Granola #9 (90001) is **not** in the zero-outbound flag: it has 9 outbound lines, but they are **adjust−**, not pack (0 pack-out in window). Classic #9 (90002) does pack-out (92 outbound lines).

### Does the sales-order / shipment workflow write inventory deductions?

**Yes. The code path exists and it fires.**

Code (single writer `ship_order`, `main.py`):

1. `POST /sales/orders/{order_id}/ship` commit (`7136`) and `POST .../ship/commit` (`7440`) both run `ship_order`.
2. Insert `shipments` header (`7244–7248`).
3. For each **physical** (non-`is_service`) line with stock: insert `transactions` `type='ship'` (`7297–7298`) and `transaction_lines` with **negative** `quantity_lb` (`7307`).
4. Then bump `sales_order_lines.quantity_shipped_lb` (`7310`), insert `sales_order_shipments` (`7325`) and `shipment_lines` (`7328–7332`).
5. Service lines fulfill with **no** ledger movement (`7256–7272`). Zero physical ship rolls back the shipment header (`7346–7361`).

Standalone `POST /ship` commit does the same deduction (`3003–3020`) plus `shipments`/`shipment_lines` (`3028–3039`).

**Live check in this window:**

| Check | Result |
|---|---|
| `sales_order_shipments` whose txn `business_date` is in window | 230 rows / 230 distinct txns |
| Missing `transactions` row | **0** |
| Posted | 229 |
| Not posted | **1** — txn **1846**, SO-260805-001, 1 lb SKU 31012, `effective_status=voided` (disposable test ship from 2026-08-05) |
| `sos.quantity_lb` vs posted negative ledger lines on that product | **0 mismatches** |
| Posted SO-linked ship lb | 500,865.00 |
| Posted standalone ship txns | **18** / 43,657.50 lb — **exactly** the 2026-06-11 `SUNSHINE-RECON-2026` set |
| Shipped/partial SOs since 2026-05-19 with zero `sales_order_shipments` | **none** |
| Example today | 2026-08-17 SO-260730-001 / SO-260805-007 / SO-260814-001 and SO-260714-001 (CQ) all have matching posted −lb lines |

Conclusion: **SO dispatch is writing and has been writing inventory deductions.** The variance is not “shipments don’t hit the ledger.”

### Root-cause hypothesis

The books are not missing a ship-to-ledger hook. Posted pounds in/out over 90 days nearly match. The earlier SKU gaps sit in **specific workflows that never create a −line**, not in a dead ship path:

1. **Sunshine SS FG** — the only Sunshine/SS ships in 90 days are the 6/11 interim recon. No replacements. New Sunshine SOs (8/14, 8/17) are unshipped. 70003 still has 29,535 lb because **nothing has deducted it since the recon**; packing continued.
2. **Classic #9 batch** — makes and packs both fire (39 in / 92 out lines). System on-hand is leftover July–Aug lots, not a failed deduct-on-pack path.
3. **Choc Chip #9 batch** — makes fire; **pack almost never deducts this batch** (adjust wipe on 6/8, then 51 batches posted and not packed). Physical 0 ⇒ those makes never became consumed FG in the ledger.
4. **BS 6x7 FG (70073/70074)** — packs in July with **no ship in 90 days**, and SO-260629-003 (10,368 lb 70073) is still confirmed / 0 shipped. Inventory accumulates because **orders are not being dispatched**, not because dispatch doesn’t write.
5. **SS Cranberry 70011** — one July pack, never in the Sunshine recon, pending on SO-260814-002.

**Hypothesis:** overstated on-hand is mostly **production/pack recorded without a matching later consume** (batch not packed, or FG packed and not shipped / not recon-replaced), plus the **unreplaced Sunshine recon**. It is **not** that `ship_order` fails to insert negative `transaction_lines`.

### All 105 products with posted activity in the window

| type | sku | product | in_lines | out_lines | in/out lines | in_lb | out_lb |
|---|---|---|---:|---:|---:|---:|---:|
| batch | 95001 | Batch BS Almond Butter Granola 350 | 0 | 2 | 0.00 | 0.0 | 1,191.7 |
| batch | 95002 | Batch BS Dark Chocolate Granola 350 | 2 | 3 | 0.67 | 8,400.0 | 3,936.2 |
| batch | 95003 | Batch BS Hazelnut Butter Granola 350 | 0 | 1 | 0.00 | 0.0 | 356.9 |
| batch | 95005 | Batch BS Peanut Butter Banana Granola | 2 | 2 | 1.00 | 3,616.0 | 3,613.6 |
| batch | 90001 | Batch Classic Chocolate Chip Granola #9 | 9 | 9 | 1.00 | 22,968.0 | 19,269.0 |
| batch | 90002 | Batch Classic Granola #9 | 39 | 92 | 0.42 | 177,287.0 | 156,222.0 |
| batch | 90003 | Batch Coconut Sweetened Fancy | 8 | 10 | 0.80 | 14,385.6 | 12,810.0 |
| batch | 90004 | Batch Coconut Sweetened Flake | 48 | 101 | 0.48 | 177,422.4 | 162,010.0 |
| batch | 90005 | Batch Coconut Sweetened Medium | 7 | 15 | 0.47 | 5,994.0 | 5,010.0 |
| batch | 90007 | Batch Coconut Toasted Sweetened Flake | 19 | 44 | 0.43 | 31,275.0 | 33,375.0 |
| batch | 90010 | Batch Granola Vanilla Almond 380 lb | 1 | 2 | 0.50 | 760.0 | 725.0 |
| batch | 90011 | Batch SS Chocolate Chip Granola #2 | 13 | 30 | 0.43 | 52,269.0 | 48,814.5 |
| batch | 90013 | Batch SS Cranberry Granola #3 | 2 | 3 | 0.67 | 1,895.0 | 3,237.5 |
| batch | 90014 | Batch SS Low Carb Chocolate Chip Granola #8 | 1 | 2 | 0.50 | 1,050.0 | 2,450.0 |
| batch | 90015 | Batch SS Low Carb Original Granola #7 | 1 | 2 | 0.50 | 1,050.0 | 2,375.0 |
| batch | 90016 | Batch SS Original Granola #1 | 6 | 12 | 0.50 | 9,400.0 | 14,180.0 |
| batch | 90019 | Batch Setton Cinnamon Almond Granola #14 | 1 | 1 | 1.00 | 660.0 | 600.0 |
| batch | 90020 | Batch Setton Cocoa Crunch Granola #13 | 2 | 1 | 2.00 | 3,042.0 | 600.0 |
| batch | 90024 | Batch Vanilla Crisp Granola #16(no almonds) | 2 | 3 | 0.67 | 1,480.0 | 1,350.0 |
| batch | 90008 | Granola Fruit Nut Batch | 1 | 1 | 1.00 | 769.0 | 600.0 |
| finished | 70074 | BS Granola – Dark Chocolate – 6x7 OZ Case **FLAG** | 1 | 0 | ∞ | 1,136.2 | 0.0 |
| finished | 70073 | BS Granola – Peanut Butter Banana – 6x7 OZ Case **FLAG** | 2 | 0 | ∞ | 3,613.6 | 0.0 |
| finished | 893 | CQ Coconut Sweetened Flake 10 LB | 73 | 41 | 1.78 | 122,650.0 | 121,800.0 |
| finished | 1614 | CQ Granola 10 LB | 47 | 28 | 1.68 | 121,800.0 | 121,800.0 |
| finished | 67470 | Coconut Sweetened Fancy UNIPRO 10 LB | 10 | 23 | 0.43 | 12,810.0 | 12,400.0 |
| finished | 10001 | Coconut Sweetened Flake CNS 10 LB | 8 | 8 | 1.00 | 2,800.0 | 3,000.0 |
| finished | 10020 | Coconut Sweetened Flake CNS 25 LB | 7 | 17 | 0.41 | 9,400.0 | 9,400.0 |
| finished | 67476 | Coconut Sweetened Flake UNIPRO 10 LB | 13 | 22 | 0.59 | 27,160.0 | 29,720.0 |
| finished | 10002 | Coconut Sweetened Medium CNS 10 LB | 1 | 4 | 0.25 | 360.0 | 930.0 |
| finished | 67473 | Coconut Sweetened Medium UNIPRO 10 LB | 10 | 20 | 0.50 | 4,650.0 | 4,600.0 |
| finished | 10010 | Coconut Toasted Sweetened Flake CNS 10 LB | 6 | 9 | 0.67 | 2,550.0 | 2,490.0 |
| finished | 10029 | Coconut Toasted Sweetened Flake CNS 25 LB | 28 | 30 | 0.93 | 32,900.0 | 38,125.0 |
| finished | 10047 | Desiccated Flake 50 LB | 0 | 2 | 0.00 | 0.0 | 900.0 |
| finished | 31012 | Graham Cracker Crumbs – 10 LB | 17 | 28 | 0.61 | 25,840.0 | 22,640.0 |
| finished | 70057 | Granola Cinnamon Almond 25 LB | 1 | 1 | 1.00 | 600.0 | 600.0 |
| finished | 70050 | Granola Classic 25 LB | 11 | 31 | 0.35 | 16,375.0 | 28,600.0 |
| finished | 70059 | Granola Cocoa Vibes 25 LB | 1 | 1 | 1.00 | 600.0 | 600.0 |
| finished | 10300 | Granola Crunchy CNS 10 LB Case | 15 | 23 | 0.65 | 4,820.0 | 4,660.0 |
| finished | 70061 | Granola Fruit Nut 25 LB | 1 | 1 | 1.00 | 600.0 | 600.0 |
| finished | 70060 | Granola Honey Nut 25 LB | 2 | 2 | 1.00 | 1,600.0 | 1,600.0 |
| finished | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | 16 | 12 | 1.33 | 32,910.0 | 34,792.5 |
| finished | 70070 | Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 1 | 1 | 1.00 | 1,050.0 | 1,050.0 |
| finished | 70014 | Granola SS Classic #9 25 LB | 1 | 1 | 1.00 | 1,575.0 | 1,575.0 |
| finished | 70011 | Granola SS Cranberry 12x10 OZ Case **FLAG** | 1 | 0 | ∞ | 1,342.5 | 0.0 |
| finished | 70002 | Granola SS Original 12x10 OZ Case | 3 | 4 | 0.75 | 6,180.0 | 6,750.0 |
| finished | 70010 | Granola SS Original Low Carb 12x10 OZ Case | 1 | 1 | 1.00 | 975.0 | 1,065.0 |
| finished | 70056 | Granola Setton Cocoa Crunch 25 LB | 0 | 1 | 0.00 | 0.0 | 1,500.0 |
| finished | 70082 | Granola Setton French Vanilla 25 LB | 2 | 1 | 2.00 | 750.0 | 3,000.0 |
| finished | 70048 | Granola Vanilla Almond 25 LB | 2 | 2 | 1.00 | 725.0 | 725.0 |
| finished | 70052 | Granola Vanilla Crisp 25 LB (French Vanilla) | 1 | 3 | 0.33 | 600.0 | 1,350.0 |
| finished | 70012 | Granola Wheat Free 25 LB | 4 | 4 | 1.00 | 4,150.0 | 4,150.0 |
| finished | 10301 | Kookies & Kreme – 10 LB | 2 | 2 | 1.00 | 300.0 | 250.0 |
| finished | 10304 | Kookies & Kreme – 25 LB | 2 | 21 | 0.10 | 40,350.0 | 26,775.0 |
| finished | 10303 | Sprinkles Chocolate 10 LB | 1 | 11 | 0.09 | 1,760.0 | 2,890.0 |
| finished | 10306 | Sprinkles Chocolate 25 LB | 4 | 11 | 0.36 | 10,700.0 | 6,925.0 |
| finished | 10302 | Sprinkles Rainbow 10 LB | 4 | 15 | 0.27 | 28,650.0 | 13,530.0 |
| finished | 10305 | Sprinkles Rainbow 25 LB | 5 | 14 | 0.36 | 20,175.0 | 15,450.0 |
| ingredient | 11002 | Almonds – Diced | 2 | 2 | 1.00 | 675.0 | 237.0 |
| ingredient | 11003 | Almonds – Sliced | 1 | 26 | 0.04 | 4,000.0 | 3,649.8 |
| ingredient | 11004 | Almonds – Slivered | 2 | 3 | 0.67 | 950.0 | 354.0 |
| ingredient | 15002 | BS Almonds – Sliced – Raw | 0 | 2 | 0.00 | 0.0 | 1,096.8 |
| ingredient | 15003 | BS Banana Bites – Small | 0 | 2 | 0.00 | 0.0 | 280.3 |
| ingredient | 15005 | BS Cacao Nibs | 0 | 3 | 0.00 | 0.0 | 602.4 |
| ingredient | 15006 | BS Cacao Shell Flour | 2 | 2 | 1.00 | 91.2 | 91.2 |
| ingredient | 15007 | BS Chia Seeds | 0 | 2 | 0.00 | 0.0 | 182.4 |
| ingredient | 15008 | BS Cocoa Liquor – Chips | 1 | 3 | 0.33 | 1,800.0 | 364.8 |
| ingredient | 15009 | BS Coconut Sugar | 1 | 2 | 0.50 | 6,160.0 | 1,365.6 |
| ingredient | 15015 | BS Oil – Almond | 0 | 2 | 0.00 | 0.0 | 547.2 |
| ingredient | 15016 | BS Peanut Butter Chips | 0 | 2 | 0.00 | 0.0 | 532.6 |
| ingredient | 15019 | BS Pumpkin Seeds | 0 | 2 | 0.00 | 0.0 | 1,005.6 |
| ingredient | 15020 | BS Salt | 1 | 2 | 0.50 | 14.4 | 36.0 |
| ingredient | 25011 | Chocolate Chips – Real – 1,000 CT | 2 | 9 | 0.22 | 325.0 | 1,650.0 |
| ingredient | 25010 | Chocolate Chips – Real – 4,000 CT | 4 | 15 | 0.27 | 4,483.0 | 5,054.0 |
| ingredient | 11006 | Chocolate Chips – Sugar Free | 1 | 1 | 1.00 | 250.0 | 87.0 |
| ingredient | 11007 | Cinnamon Ground | 0 | 3 | 0.00 | 0.0 | 24.0 |
| ingredient | 11008 | Cocoa Powder | 0 | 2 | 0.00 | 0.0 | 31.5 |
| ingredient | 11010 | Coconut Fancy Desiccated | 0 | 8 | 0.00 | 0.0 | 7,200.0 |
| ingredient | 11012 | Coconut Flake Desiccated | 2 | 71 | 0.03 | 14,100.0 | 109,600.0 |
| ingredient | 11013 | Coconut Macaroon Desiccated | 24 | 69 | 0.35 | 6,978.0 | 11,010.0 |
| ingredient | 11014 | Coconut Medium Desiccated | 0 | 7 | 0.00 | 0.0 | 3,000.0 |
| ingredient | 11016 | Corn Starch | 1 | 82 | 0.01 | 2,250.0 | 2,995.0 |
| ingredient | 11017 | Cranberries Dried | 2 | 3 | 0.67 | 175.0 | 127.5 |
| ingredient | 11018 | Flavor – Almond | 1 | 48 | 0.02 | 800.0 | 1,098.0 |
| ingredient | 11020 | Flavor – Chocolate | 1 | 2 | 0.50 | 40.0 | 21.6 |
| ingredient | 11021 | Flavor – Cinnamon | 0 | 1 | 0.00 | 0.0 | 2.2 |
| ingredient | 11026 | Flavor – Vanilla | 1 | 3 | 0.33 | 80.0 | 60.0 |
| ingredient | 11027 | Flax – Ground | 1 | 2 | 0.50 | 40.0 | 90.0 |
| ingredient | 11028 | Flax – Seed **FLAG** | 2 | 0 | ∞ | 225.0 | 0.0 |
| ingredient | 11029 | Glycol | 0 | 81 | 0.00 | 0.0 | 4,193.0 |
| ingredient | 31011 | Graham Cracker Crumbs – 50 LB | 2 | 28 | 0.07 | 78,350.0 | 63,110.0 |
| ingredient | 11030 | Honey | 2 | 67 | 0.03 | 6,369.6 | 8,677.0 |
| ingredient | 11031 | Oat Bran | 20 | 20 | 1.00 | 4,669.0 | 4,669.0 |
| ingredient | 11032 | Oats | 6 | 83 | 0.07 | 168,000.0 | 175,578.6 |
| ingredient | 11033 | Oats – Gluten Free | 2 | 1 | 2.00 | 4,000.0 | 2,000.0 |
| ingredient | 11034 | Oil – Canola | 4 | 79 | 0.05 | 22,000.0 | 22,504.0 |
| ingredient | 11035 | Pumpkin Seeds | 2 | 3 | 0.67 | 183.0 | 153.0 |
| ingredient | 11036 | Raisins | 0 | 1 | 0.00 | 0.0 | 2.5 |
| ingredient | 11038 | Rice Flour | 1 | 4 | 0.25 | 500.0 | 210.0 |
| ingredient | 11039 | Salt | 2 | 135 | 0.01 | 7,350.0 | 2,565.7 |
| ingredient | 11040 | Sugar – 6X | 3 | 85 | 0.04 | 96,000.0 | 66,442.0 |
| ingredient | 11043 | Sugar – Invert(Cream) | 0 | 3 | 0.00 | 0.0 | 42.0 |
| ingredient | 11044 | Sugar – Light Brown | 5 | 56 | 0.09 | 31,950.0 | 31,165.0 |
| ingredient | 11045 | Sugar – Monk Fruit | 0 | 2 | 0.00 | 0.0 | 198.0 |
| ingredient | 11046 | Sunflower Seeds | 2 | 25 | 0.08 | 7,500.0 | 4,205.2 |
| ingredient | 11049 | Walnuts | 0 | 1 | 0.00 | 0.0 | 2.5 |

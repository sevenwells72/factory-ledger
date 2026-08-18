# Inventory variance — execution log

**Posted:** 2026-08-17 15:48 EDT  
**Status:** COMMITTED. In-transaction verification passed (all SKU and lot checks OK) before commit.  
**Operator:** `inv-recon-2026-08-17`  
**Plan:** `docs/audits/inventory-variance-recon-plan.md`  
**Count:** `docs/audits/physical-count-2026-08-14.md`  
**Poster:** `scripts/inv_recon_post_2026_08_17.py` (single `READ WRITE` transaction; rollback on any verify fail)

## Scope applied

- Mini 100 excluded (Classic #9 / CC #9 gaps posted as generic unrecorded consumption).
- 70088 6x8 excluded (still 802 cs / 2,406 lb).
- Coconut batches 90003/4/5/7 excluded.
- 67476 SKU total held at 317 cs; 80 cs on `STAGED-FEESERS-67476`.
- New WIP `15999` “WIP Banana / PB-chip mix (for PBB)” + 1.00 lb container token on `AUG 14 MIX`. `95005` leftover 2.38 lb not written off.
- `SUNSHINE-RECON-2026` txns 1325–1342 still `posted`.
- `SO-260629-003`, `SO-260814-002`, `SO-260817-001` unchanged (`confirmed`, `quantity_shipped_lb = 0`).

Pre-go fixes: 4.4/4.5 posted in exact pounds (−5.26 / −28.02) so 70080 landed on **0.0000 lb**. 4.16/4.19 used current 119/62 (8/17 ships already applied).

## Posted entries

| Checklist | Txn | Type | What |
|---|---:|---|---|
| C1 | product 291 | product create | `15999` ingredient, parent 95005 (id 122) |
| C2 | 1950 | adjust+ | 15999 `AUG 14 MIX` +1.00 |
| 1.1 | 1951 | adjust+ | 70073 SW2620891 +2,627.37 (28337-I) |
| 1.2 | 1952 | adjust+ | 70074 SW2620590 +149.91 (28337-I) |
| 1.3 | 1953 | adjust+ | 70074 SW2607890 +165.69 (28220-I) |
| 1.4 | 1954 | adjust+ | 70010 BB070827 +1,522.50 |
| 1.5 | 1955 | adjust+ | 70070 BB070827 +1,567.50 |
| 1.6 | 1956 | adjust+ | 70011 BB070827 +780.00 |
| 1.7 | 1957 | adjust+ | 90016 JUN 24 2026 +255.00 |
| 2.1–2.4 | 1958 / shipment 333 | ship | Blue Stripes **28220-I** 4 lines |
| 2.5–2.6 | 1959 / shipment 334 | ship | Blue Stripes **28337-I** 2 lines |
| 2.7–2.14 | 1960 / shipment 335 | ship | Sunshine **QB-70003-YTD-NET** (invoices 28259…28257) 8 lots, 3,377 cs |
| 2.15 | 1961 / shipment 336 | ship | Sunshine **QB-70002-YTD-NET** 15 cs |
| 2.16–2.17 | 1962 / shipment 337 | ship | Sunshine **QB-LOWCARB-YTD-NET** 70010+70070 |
| 2.18 | 1963 / shipment 338 | ship | Sunshine **QB-70011-YTD** 283 cs |
| 3.A | 1964 | pack | 90002 → 70013 `BULK-#9-YTD` 12,115 lb |
| 3.A16 | 1965 / shipment 339 | ship | Sunshine **QB-70013-BULK-YTD** −12,115 |
| 3.B | 1966 | pack | 90016 → 70004 `BULK-#1-YTD` 1,875 lb |
| 3.B3 | 1967 / shipment 340 | ship | Sunshine **QB-70004-BULK-YTD** −1,875 |
| 3.D1–3.D34 | 1968–2001 | adjust− | generic batch consumes to Aug 14 floors |
| 4.1 | 2002 | adjust+ | 70003 BB081027 +292.50 (+39 cs) |
| 4.2–4.3 | 2003–2004 | adjust− | 70002 to 200 cs on BB081027 |
| 4.4–4.5 | 2005–2006 | adjust− | 70080 exact −5.26 / −28.02 → 0.00 |
| 4.6–4.8 | 2007–2009 | adjust | 70050 lot-correct to MAY 13×5 + JUL 21×13 |
| 4.9–4.22 | 2010–2023 | adjust | remaining FG count residuals |
| 4.23–4.25 | 2024–2026 | adjust | 67476 lots 140 / 97 / `STAGED-FEESERS-67476` 80 |

All ship txns are standalone (`sales_orders` not updated). Notes include `INV-RECON-2026-08-17`. `business_date` / `occurred_at` are the checklist dates (noon America/New_York). `created_at` is the 2026-08-17 post time (trigger-enforced).

## Verification vs Aug 14 count

| SKU | Count expectation | Ledger after post | Delta | Result |
|---|---|---|---:|---|
| 70003 | 600 cs BB081027 | 600 cs / 4,500 lb on BB081027 | 0 | OK |
| 70002 | 200 cs BB081027 | 200 cs / 1,500 lb on BB081027 | 0 | OK |
| 70010 / 70070 / 70011 | 0 | 0 | 0 | OK |
| 70073 / 70074 / 70080 | 0 | 0.0000 lb | 0 | OK |
| 70050 | 18 (MAY 13×5, JUL 21×13) | 5 + 13 cs | 0 | OK |
| 70082 / 70059 / 70052 / 10300 | 0 | 0 | 0 | OK |
| 1614 | 0 | 0 | 0 | OK |
| 31012 | 280 unstamped | 280 cs / 2,800 lb on 608101 | 0 | OK |
| 893 | 700 (280/140/280) | 700 cs; AUG 14 = 280 | 0 | OK |
| 67476 | 317 | 317 cs (140 + 97 + 80 Feesers) | 0 | OK |
| 67470 | 26 | 26 cs / 260 lb | 0 | OK |
| 10020 | 0 | 0 | 0 | OK |
| 10029 | 50 JUL 27 | 50 cs / 1,250 lb | 0 | OK |
| 67473 | 60 | 60 cs / 600 lb | 0 | OK |
| 10002 | 16 AUG 04 | 16 cs / 160 lb | 0 | OK |
| 10001 / 10006 / 10007 | 0 | 0 | 0 | OK |
| 10010 | 6 JUL 27 | 6 cs / 60 lb | 0 | OK |
| 90002 | 31 batches | 31.00 (AUG 13 18 + AUG 12 13) | 0 | OK |
| 90001 | 0 | 0 | 0 | OK |
| 90011 | 10 | 10.00 on AUG 14 | 0 | OK |
| 95002 | 3 | 3.00 on JUL 24 | 0 | OK |
| 90016 / 90020 / 90024 / 90010 / 90019 / 90015 | 0 | 0 | 0 | OK |
| 90013 | 1 | 1.00 on AUG 06 | 0 | OK |
| 15999 | 1 container token | 1.00 lb on AUG 14 MIX | 0 | OK |
| 95005 | not written off | 2.38 lb JUL 27 | 0 | OK |
| 70088 | excluded | 802 cs / 2,406 lb | n/a | held |

70013 and 70004 are 0 (packed and shipped as billed bulk).

## Follow-ups still open

- Mini 100 4,186 + 995 units still unconverted (no catalog weight / #9 Mini SKU).
- `STAGED-FEESERS-67476` needs real lot dates before that 80 cs ships (trace).
- SO-260629-003 (3,942 cs 70073) still open and does not match 28337-I.
- 70006 Mini 100 product spec still has no `case_size_lb`.

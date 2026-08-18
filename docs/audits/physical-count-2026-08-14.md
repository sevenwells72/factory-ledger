# Physical inventory count — 2026-08-14

**Counted by:** Arturo  
**Count date:** 2026-08-14  
**Recorded in repo:** 2026-08-17 15:24 EDT  
**Scope:** finished-goods cases on the floor, plus granola batch containers. Coconut *batch* silos were not on this sheet.

SKU codes below are the live `products.odoo_code` values. Case sizes are ledger `case_size_lb`. Batch estimate rule on the sheet: **3 batches per full container**.

This file is the count source of truth for `docs/audits/inventory-variance-recon-plan.md`. It is not a ledger write.

---

## Finished goods — cases

| Count name | SKU | Counted cs | Counted lots (cs) | Ledger case lb |
|---|---|---:|---|---:|
| Granola Classic 25LB | 70050 | 18 | MAY 13 ×5, JUL 21 ×13 | 25 |
| Setton French Vanilla 25LB | 70082 | 0 | — | 25 |
| Cocoa Vibes 25LB | 70059 | 0 | — | 25 |
| Vanilla Crisp 25LB | 70052 | 0 | — | 25 |
| CQ Granola 10LB | 1614 | 0 | — | 10 |
| Graham Cracker Crumbs 10LB | 31012 | 280 | unstamped, all for Clark | 10 |
| Granola Crunchy CNS 10LB | 10300 | 0 | — | 10 |
| SS Choc Chip 12x10 | 70003 | 600 | BB081027 | 7.5 |
| SS Original 12x10 | 70002 | 200 | BB081027 | 7.5 |
| SS Cranberry | 70011 | 0 | — | 7.5 |
| SS CC Low Carb | 70070 | 0 | — | 7.5 |
| SS Orig Low Carb | 70010 | 0 | — | 7.5 |
| BS Dark Choc 6x7 | 70074 | 0 | — | 2.63 |
| BS Hazelnut 6x7 | 70080 | 0 | — | 2.63 |
| BS PB Banana 6x7 | 70073 | 0 | — | 2.63 |
| CQ Coconut Swt Flake 10LB | 893 | 700 | AUG 12 ×280, AUG 13 ×140, AUG 14 ×280 | 10 |
| Coconut Swt Flake UNIPRO 10LB | 67476 | 317 | AUG 05 ×140, AUG 06 ×97 | 10 |
| Swt Fancy UNIPRO 10LB | 67470 | 26 | AUG 13 | 10 |
| Swt Flake CNS 25LB | 10020 | 0 | — | 25 |
| Toasted Swt Flake CNS 25LB | 10029 | 50 | JUL 27 | 25 |
| Swt Medium UNIPRO 10LB | 67473 | 60 | AUG 13 | 10 |
| Swt Medium CNS 10LB | 10002 | 16 | AUG 04 | 10 |
| Swt Flake CNS 10LB | 10001 | 0 | — | 10 |
| Swt Fancy CNS 25LB | 10007 | 0 | — | 25 |
| Toasted Swt Flake CNS 10LB | 10010 | 6 | JUL 27 | 10 |
| Swt Fancy CNS 10LB | 10006 | 0 | — | 10 |

### Count-sheet flags (do not silently “fix”)

1. **67476 lot split vs total.** Header is 317 cs. Lot tags AUG 05 ×140 + AUG 06 ×97 = **237**, not 317. Ledger already holds 317 (AUG 05 ×237 + AUG 06 ×80). SKU floor used in the recon plan is **317**. Lot-level move is held until Arturo confirms the tags.
2. **70050 lot tags vs ledger.** Count is MAY 13 / JUL 21. Ledger on-hand is only `AUG 11 2026` (55 cs). Those older FG lots were shipped in June–August; the recon plan recreates the counted lots and clears AUG 11.
3. **6x8 BS (70088 etc.)** were not on this sheet. Only 6x7 was counted at 0.
4. **Graham 31012** counted 280 unstamped, reserved for Clark. Ledger lot `608101` holds 420 cs.

---

## Batches — containers / estimated batches

Rule written on the sheet: 3 batches per full container. Estimated batches are what the recon plan uses as the floor.

| Count name | SKU | Containers | Est. batches | lb/batch | Floor lb |
|---|---|---:|---:|---:|---:|
| Classic #9 | 90002 | 10 | **31** | 323 | 10,013 |
| Classic CC #9 | 90001 | 0 | **0** | 348 | 0 |
| SS CC #2 | 90011 | 3 | **10** | 393 | 3,930 |
| BS Dark Choc 350 | 95002 | 1 | **3** | 350 | 1,050 |
| SS Original #1 | 90016 | 0 | **0** | 350 | 0 |
| Setton Cocoa Crunch #13 | 90020 | 0 | **0** | 338 | 0 |
| Vanilla Crisp #16 | 90024 | 0 | **0** | 370 | 0 |
| SS Cranberry #3 | 90013 | 0.25 | **1** | 379 | 379 |
| Vanilla Almond 380 | 90010 | 0 | **0** | 380 | 0 |
| Setton Cinnamon #14 | 90019 | 0 | **0** | 330 | 0 |
| SS Low Carb Orig #7 | 90015 | 0 | **0** | 350 | 0 |

**BS PB Banana — not a granola batch on this count.** Arturo recorded **1 container of banana / PB-chip mix**, not Batch BS PBB (95005). That mix needs its own product designation. Do not treat it as 95005 granola on-hand.

### Not on this sheet (no floor)

Coconut batches 90003 / 90004 / 90005 / 90007, Fruit Nut 90008, SS Low Carb CC #8 (90014), SS Low Carb CC batch remnants, and every ingredient. The recon plan does not invent floors for them.

---

## How to use this file

- Finished-goods floor = counted cases at the SKU (and lot, when the lot tags add to the SKU total).
- Batch floor = estimated batches × `default_batch_lb`.
- 67476 lot tags do not add to the SKU total — do not force a lot move from this sheet alone.
- 95005 is not a counted granola batch.

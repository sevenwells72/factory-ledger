-- lot-code-cleanup-2026-08-25.sql  ── DRAFT, NOT EXECUTED
-- Cleanup for the 18 lot-code defects in section 6c of
-- docs/data-health-baseline-2026-08-24.md.
--
-- Verified against production (read-only, 2026-08-25) before drafting:
-- **NONE of the 18 defective codes exist as lot codes in FL.** Every typo
-- lives only in the paper/Google-Form free text; FL already carries the
-- corrected code for each affected day. Disposition of each defect is
-- documented below. While verifying, two FL-side typo lots of the same
-- classes WERE found (a 2027 year and a zero-padded day); this file fixes
-- the safe one and (per owner decision 2026-08-25) renames the colliding
-- one to a '-dup' suffix instead of merging.
--
-- Run with: psql session pooler (port 5432), -v ON_ERROR_STOP=1.
-- The transaction ends with a (product_id, code) uniqueness re-check.

-- ── Disposition of the 18 section-6c defects ──────────────────────────────
-- 1.  receiving 4/21  '260521PARK001'  sticker typo for 260421PARK001 (Parker
--     Flavor). No FL lot carries this code (FL receive lots use the
--     26-MM-DD-XXXX-nnn convention) → nothing to UPDATE; physical sticker only.
-- 2.  receiving 8/17  '260827ABAK'     sticker typo for 260817ABAK (A1 Baker,
--     BOL 69383). FL created its own correct lots 26-08-17-ABAK-001
--     (Chocolate Chips Real 4,000 CT) and 26-08-17-ABAK-002 (Cranberries
--     Dried, supplier lot 29997055) → no UPDATE. The PALLET STICKERS in the
--     plant read 260827ABAK — relabel physically to 260817ABAK.
-- 3.  coconut 1/2   case lot 'De 30 2025'  (= Dec 30 2025) — pre-FL era
--     (ledger starts 2026-01-28); no FL lot → no UPDATE.
-- 4.  coconut 1/5   batch lot 'Jan 05 2025' (= Jan 05 2026) — pre-FL → none.
-- 5.  coconut 1/5   case lot  'Jan 05 2025' — pre-FL → none.
-- 6.  coconut 1/5   case lot  'Dec 24 2025' (12 d drift) — pre-FL → none.
-- 7.  coconut 1/21  case lot  'Dec 15 2025' (37 d drift) — pre-FL → none.
-- 8.  coconut 3/9   case lot  'Feb 20 2026' (17 d drift) — FL's FEB 20 2026
--     lots (218/219/221) are genuine Feb-20 production; this is a day-
--     attribution question on the form, not a typo → no UPDATE.
-- 9.  coconut 3/26  sugar-6x lot '261230JA' (Dec 30 read as 2026 = future).
--     Neither '261230JA' nor the corrected '251230JA' exists as an FL lot,
--     and no FL receipt carries either reference, so the receipt date
--     cannot be confirmed → no UPDATE; sticker-only defect.
-- 10. coconut 6/9   batch lot 'Jan 09 2026' (= JUN 09 2026) — FL has the
--     correct JUN 09 2026 lots (789–794), no JAN 09 2026 lot → no UPDATE.
-- 11. coconut 6/9   case lot  'Jan 09 2026' — same as 10 → none.
-- 12. coconut 6/9   case lot  'Jan 09 2026' — same as 10 → none.
-- 13. coconut 7/27  case lot  'Jul07 2026' — spacing only; normalizes to
--     FL's existing JUL 07 2026 → no UPDATE.
-- 14. coconut 7/30  case lot  'Jul 30 3026' — FL lot is correctly
--     JUL 30 2026 (lot 1090, CQ) → no UPDATE; form-only typo.
-- 15. coconut 8/4   batch lot 'Aug 04 2027' — FL lots for 8/4 are correctly
--     AUG 04 2026 (1114–1121) → no UPDATE for this defect itself; but see
--     the same-class FL-side fix for lot 1329 below.
-- 16. coconut 8/6   case lot  'Aug 06 3026' — FL lot is correctly
--     AUG 06 2026 (lot 1137, UNIPRO) → no UPDATE.
-- 17. coconut 8/7   case lot  'Jul 30 2026' (8 d drift) — genuine old-stock
--     pack (the 79-case Jul-30-lot event, report §6b-iii) → not a typo,
--     no UPDATE.
-- 18. coconut 8/24  case lot  'Agu 24 2026' — FL lot is correctly
--     AUG 24 2026 (lot 1332, CQ, 3,690 lb on hand) → no UPDATE.

BEGIN;

-- ── FL-side fixes discovered while verifying the 18 ───────────────────────

-- (a) Lot 1329, Batch Classic Granola #9 (product 107), created 2026-08-24,
--     on-hand 3,230 lb: coded 'AUG 24 2027' — same year-typo class as
--     defect 15. Product 107 has NO existing 'AUG 24 2026' lot (the 8/24
--     lots 1330/1332 belong to products 126/164), so no collision.
UPDATE lots
   SET lot_code = 'AUG 24 2026'
 WHERE id = 1329
   AND lot_code = 'AUG 24 2027';   -- guard: exactly the typo row

-- (b) Lot 578, CQ Coconut Sweetened Flake 10 LB (product 164): coded
--     'MAY 011 2026' (zero-padded day). Correcting to 'MAY 11 2026' would
--     COLLIDE with existing lot 590 (product 164, 'MAY 11 2026'); both are
--     at 0.0 lb on hand. OWNER DECISION 2026-08-25: rename, not merge —
--     lot 578 becomes 'MAY 11 2026-dup' so it normalizes to the same date
--     for traceability searches while staying a distinct lot record (its
--     transaction_lines / ingredient_lot_consumption history is untouched).
UPDATE lots
   SET lot_code = 'MAY 11 2026-dup'
 WHERE id = 578
   AND lot_code = 'MAY 011 2026';  -- guard: exactly the typo row

-- ── Re-check (product_id, code) uniqueness before COMMIT ──────────────────
-- Expected: the pre-existing collisions from report §3i (159 codes shared
-- ACROSS products are by design of date-style codes; this checks the
-- stricter per-product duplicates, which must NOT grow from this script).
SELECT l.product_id,
       upper(btrim(l.lot_code))     AS code,
       count(*)                     AS n_lots,
       array_agg(l.id ORDER BY l.id) AS lot_ids
  FROM lots l
 GROUP BY l.product_id, upper(btrim(l.lot_code))
HAVING count(*) > 1
 ORDER BY l.product_id, code;

COMMIT;

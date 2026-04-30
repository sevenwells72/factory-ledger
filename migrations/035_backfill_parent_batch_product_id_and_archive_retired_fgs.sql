-- Migration 035: Backfill parent_batch_product_id mappings and archive retired FGs
--
-- Purpose:
--   Populate products.parent_batch_product_id for finished goods that lacked the
--   mapping, and archive three retired FG SKUs. Mappings were confirmed manually
--   against pack history and operator domain knowledge.
--
-- Coverage of the 68 active finished-good products:
--   * Group A (8):  Already mapped before this migration. Untouched here.
--   * Group B (33): Newly mapped. 33 UPDATE statements below.
--   * Group C (3):  Retired SKUs. Archived (active=false), mapping stays NULL.
--   * Group D (24): Dual-role SKUs (sold-as-is and/or used as batch ingredient).
--                   Intentional NULL parent_batch_product_id. See CLAUDE.md for
--                   the explicit list and rationale.
--
-- Post-migration expected state:
--   * 41 active FGs with parent_batch_product_id set (8 prior + 33 new)
--   * 24 active FGs with NULL parent_batch_product_id (Group D, by design)
--   * 65 active FGs total (68 - 3 archived)
--
-- This file is wrapped in BEGIN/COMMIT so it is all-or-nothing. A commented
-- rollback block is included at the bottom.

BEGIN;

-- =============================================================================
-- GROUP B: Newly mapped finished goods (33 rows)
-- =============================================================================

-- CNS house granolas
UPDATE products SET parent_batch_product_id = 107 WHERE id = 136;  -- Granola Classic 25 LB -> Batch Classic Granola #9
UPDATE products SET parent_batch_product_id = 107 WHERE id = 137;  -- Granola Crunchy CNS 10 LB Case -> Batch Classic Granola #9
UPDATE products SET parent_batch_product_id = 108 WHERE id = 143;  -- Granola Chocolate Chip 25 LB -> Batch Classic Chocolate Chip Granola #9
UPDATE products SET parent_batch_product_id = 179 WHERE id = 142;  -- Granola Fruit Nut 25 LB -> Batch Granola Fruit Nut

-- Sunshine (SS) granolas
UPDATE products SET parent_batch_product_id = 116 WHERE id = 146;  -- Granola SS Original 12x10 OZ Case -> Batch SS Original Granola #1
UPDATE products SET parent_batch_product_id = 114 WHERE id = 145;  -- Granola SS Chocolate Chip 12x10 OZ Case -> Batch SS Chocolate Chip Granola #2
UPDATE products SET parent_batch_product_id = 119 WHERE id = 148;  -- Granola SS Chocolate Chip Low Carb 12x10 OZ Case -> Batch SS Low Carb Chocolate Chip Granola #8
UPDATE products SET parent_batch_product_id = 120 WHERE id = 149;  -- Granola SS Original Low Carb 12x10 OZ Case -> Batch SS Low Carb Original Granola #7
UPDATE products SET parent_batch_product_id = 116 WHERE id = 183;  -- Granola SS Original Bulk per/lb -> Batch SS Original Granola #1
UPDATE products SET parent_batch_product_id = 116 WHERE id = 185;  -- Granola SS Mini 100 -> Batch SS Original Granola #1
UPDATE products SET parent_batch_product_id = 118 WHERE id = 147;  -- Granola SS Cranberry 12x10 OZ Case -> Batch SS Cranberry Granola #3

-- CNS house granolas (cross-recipe / variants)
UPDATE products SET parent_batch_product_id = 116 WHERE id = 141;  -- Granola Honey Nut 25 LB -> Batch SS Original Granola #1 (same recipe, different label)
UPDATE products SET parent_batch_product_id = 112 WHERE id = 134;  -- Granola Vanilla Crisp 25 LB (French Vanilla) -> Batch Vanilla Crisp Granola #16
UPDATE products SET parent_batch_product_id = 113 WHERE id = 135;  -- Granola Vanilla Almond 25 LB -> Batch Granola Vanilla Almond 380 lb (NOT Vanilla Crisp -- no almonds)
UPDATE products SET parent_batch_product_id = 107 WHERE id = 138;  -- Granola Wheat Free 25 LB -> Batch Classic Granola #9

-- Setton private label granolas
UPDATE products SET parent_batch_product_id = 109 WHERE id = 140;  -- Granola Cinnamon Almond 25 LB -> Batch Setton Cinnamon Almond Granola #14
UPDATE products SET parent_batch_product_id = 111 WHERE id = 139;  -- Granola Cocoa Vibes 25 LB -> Batch Setton Cocoa Crunch Granola #13
UPDATE products SET parent_batch_product_id = 112 WHERE id = 133;  -- Granola Setton French Vanilla 25 LB -> Batch Vanilla Crisp Granola #16
UPDATE products SET parent_batch_product_id = 109 WHERE id = 131;  -- Granola Setton Cinnamon Spice Almond 25 LB -> Batch Setton Cinnamon Almond Granola #14
UPDATE products SET parent_batch_product_id = 111 WHERE id = 132;  -- Granola Setton Cocoa Crunch 25 LB -> Batch Setton Cocoa Crunch Granola #13
UPDATE products SET parent_batch_product_id = 107 WHERE id = 129;  -- Granola Setton Good Ol 25 LB -> Batch Classic Granola #9

-- CQ private label granola
UPDATE products SET parent_batch_product_id = 107 WHERE id = 144;  -- CQ Granola 10 LB -> Batch Classic Granola #9

-- Coconut Sweetened Flake (multiple labels and pack sizes -> one batch)
UPDATE products SET parent_batch_product_id = 126 WHERE id = 156;  -- Coconut Sweetened Flake CNS 10 LB -> Batch Coconut Sweetened Flake
UPDATE products SET parent_batch_product_id = 126 WHERE id = 157;  -- Coconut Sweetened Flake CNS 25 LB -> Batch Coconut Sweetened Flake
UPDATE products SET parent_batch_product_id = 126 WHERE id = 162;  -- Coconut Sweetened Flake UNIPRO 10 LB -> Batch Coconut Sweetened Flake
UPDATE products SET parent_batch_product_id = 126 WHERE id = 164;  -- CQ Coconut Sweetened Flake 10 LB -> Batch Coconut Sweetened Flake

-- Coconut Toasted Sweetened Flake
UPDATE products SET parent_batch_product_id = 128 WHERE id = 159;  -- Coconut Toasted Sweetened Flake CNS 10 LB -> Batch Coconut Toasted Sweetened Flake
UPDATE products SET parent_batch_product_id = 128 WHERE id = 160;  -- Coconut Toasted Sweetened Flake CNS 25 LB -> Batch Coconut Toasted Sweetened Flake

-- Coconut Sweetened Fancy
UPDATE products SET parent_batch_product_id = 125 WHERE id = 161;  -- Coconut Sweetened Fancy UNIPRO 10 LB -> Batch Coconut Sweetened Fancy
UPDATE products SET parent_batch_product_id = 125 WHERE id = 154;  -- Coconut Sweetened Fancy CNS 10 LB -> Batch Coconut Sweetened Fancy
UPDATE products SET parent_batch_product_id = 125 WHERE id = 155;  -- Coconut Sweetened Fancy CNS 25 LB -> Batch Coconut Sweetened Fancy

-- Coconut Sweetened Medium
UPDATE products SET parent_batch_product_id = 127 WHERE id = 158;  -- Coconut Sweetened Medium CNS 10 LB -> Batch Coconut Sweetened Medium
UPDATE products SET parent_batch_product_id = 127 WHERE id = 163;  -- Coconut Sweetened Medium UNIPRO 10 LB -> Batch Coconut Sweetened Medium

-- =============================================================================
-- GROUP C: Archive retired finished goods (3 rows)
-- =============================================================================
-- These SKUs are no longer produced. Mapping intentionally stays NULL; only the
-- active flag changes.

UPDATE products SET active = false WHERE id = 130;  -- Granola Setton Morning Latte Crunch 25 LB
UPDATE products SET active = false WHERE id = 184;  -- Granola SS B'gan Chocolate per/lb
UPDATE products SET active = false WHERE id = 186;  -- Granola SS Evergreen 12 pack

-- =============================================================================
-- VERIFICATION
-- =============================================================================
-- These DO blocks raise an exception if the post-state does not match what we
-- expect. Because the whole file is in a single transaction, any failure rolls
-- back all changes above.

-- Expect: 41 active FGs with parent_batch_product_id IS NOT NULL
DO $$
DECLARE
    cnt int;
BEGIN
    SELECT COUNT(*) INTO cnt
    FROM products
    WHERE type = 'finished'
      AND active = true
      AND parent_batch_product_id IS NOT NULL;
    IF cnt <> 41 THEN
        RAISE EXCEPTION 'Verification failed: expected 41 active FGs with mapping set, got %', cnt;
    END IF;
END $$;

-- Expect: 24 active FGs with parent_batch_product_id IS NULL (Group D, dual-role)
DO $$
DECLARE
    cnt int;
BEGIN
    SELECT COUNT(*) INTO cnt
    FROM products
    WHERE type = 'finished'
      AND active = true
      AND parent_batch_product_id IS NULL;
    IF cnt <> 24 THEN
        RAISE EXCEPTION 'Verification failed: expected 24 active FGs without mapping, got %', cnt;
    END IF;
END $$;

-- Expect: 65 active FGs total (was 68, archived 3)
DO $$
DECLARE
    cnt int;
BEGIN
    SELECT COUNT(*) INTO cnt
    FROM products
    WHERE type = 'finished'
      AND active = true;
    IF cnt <> 65 THEN
        RAISE EXCEPTION 'Verification failed: expected 65 active FGs, got %', cnt;
    END IF;
END $$;

-- Expect: each of the 3 archived FGs is now active = false
DO $$
DECLARE
    cnt int;
BEGIN
    SELECT COUNT(*) INTO cnt
    FROM products
    WHERE id IN (130, 184, 186)
      AND active = false;
    IF cnt <> 3 THEN
        RAISE EXCEPTION 'Verification failed: expected all 3 archived FGs to be inactive, got %', cnt;
    END IF;
END $$;

COMMIT;

-- =============================================================================
-- ROLLBACK (commented out)
-- =============================================================================
-- If migration 035 needs to be reversed, run the following block in a single
-- transaction. Mappings revert to NULL for the 33 newly-mapped FGs, and the 3
-- archived FGs return to active.
--
-- Note: this rollback is safe to run only if migration 035 is the most recent
-- change to these specific products. If subsequent migrations have modified
-- parent_batch_product_id or active for any of these IDs, this block will
-- overwrite that work.
--
-- BEGIN;
--
-- -- Revert Group B mappings (33 rows)
-- UPDATE products SET parent_batch_product_id = NULL WHERE id IN (
--     129, 131, 132, 133, 134, 135, 136, 137, 138, 139,
--     140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
--     154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
--     164, 183, 185
-- );
--
-- -- Revert Group C archives (3 rows)
-- UPDATE products SET active = true WHERE id IN (130, 184, 186);
--
-- COMMIT;

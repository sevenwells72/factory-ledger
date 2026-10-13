-- 078: recipe corrections approved by Michael 2026-10-13
-- (source: ~/Documents/fl-audits/recipe-check-2026-10-13/recipe-check.md, rows 179 and 284;
--  catalog-cleanup.csv row 29 "Product 284 formula chip size", APPROVED 2026-10-07).
--
-- DATA ONLY. No DDL, no historical rows: transactions, transaction_lines, lots and
-- ingredient_lot_consumption are not read or written. Past makes, deductions and
-- lot balances stand as posted; stock true-ups are a separate, owner-approved step.
--
-- Apply as the table owner on port 5432, STAGING FIRST, production only with the
-- owner's approval:
--   BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
--   SET LOCAL search_path=public; \i migrations/078_recipe_corrections.sql; COMMIT;
-- Rerunnable: every UPDATE is a no-op once applied. The guard below accepts each row
-- in its recorded BEFORE state or its AFTER state and refuses anything else, so the
-- file can never be applied to a catalog it was not written for.
-- Down file: migrations/down/078_recipe_corrections_down.sql (restores the BEFORE values).
--
-- BEFORE (production, read-only 2026-10-13 17:43 UTC)        AFTER
-- products 179 (90008 Granola Fruit Nut Batch)
--   default_batch_lb                 384.5200                 328.3000
-- batch_formulas product_id=179 (what /make deducts)
--   id 179  107 Batch Classic Granola #9   20.0000            276.3000
--   id 180   49 Walnuts                     1.2500             13.0000
--   id 181    3 Almonds – Sliced            1.2500             13.0000
--   id 182   36 Raisins                     1.2500             13.0000
--   id 183   17 Cranberries Dried           1.2500             13.0000
--   total                                  25.0000            328.3000
-- product_bom finished_product_id=179 (planner / full-BOM loader), rewritten in place
--   id 117  107 Batch Classic Granola #9  323.0000 lb         276.3000 lb
--   id 118    3 Almonds – Sliced           15.3800 lb          13.0000 lb
--   id 119   49 Walnuts                    15.3800 lb          13.0000 lb
--   id 120   36 Raisins                    15.3800 lb          13.0000 lb
--   id 121   17 Cranberries Dried          15.3800 lb          13.0000 lb
--   total                                 384.5200            328.3000
-- batch_formulas product_id=284 (90026; named "Batch SS Classic Chocolate Chip Granola #9 (Kosher
--   Ignition)" at 17:43 UTC, renamed "… #9 — EXTRA-KOSHER (SS)" by 075 at 17:44:52 UTC; identity
--   is checked by odoo_code, not by name; rows re-read unchanged at 17:57 UTC after 075)
--   id 198  ingredient 73 Chocolate Chips – Real – 1,000 CT, 25.0000
--                                                        -> ingredient 72 Chocolate Chips – Real – 4,000 CT, 25.0000
--   the chips line must be deducted (exclude_from_inventory false/NULL) before and after: an
--   excluded chips line would post 348 lb with zero chips taken. Using 1,000 CT in place of
--   4,000 CT on a given make is a per-make A5 substitution (transaction_substitutions), never
--   a formula exclusion.
-- products 179 legacy case yields (Michael approved 2026-10-13 fix round; 328.3 / 25 = 13.1,
--   328.3 / 10 = 32.8)
--   yield_25lb_cases                 15                       13
--   yield_10lb_cases                 38                       32
-- Unchanged on purpose: product 142 (70061 Granola Fruit Nut 25 LB, case_size_lb 25,
-- parent 179) and its planner row product_bom id 116 (142 -> 179, 25 lb); 284's other
-- seven lines.

DO $$
DECLARE
    n integer;
BEGIN
    -- Identity of the rows this file was written for.
    IF (SELECT count(*) FROM products WHERE id = 179 AND odoo_code = '90008' AND type = 'batch') <> 1 THEN
        RAISE EXCEPTION '078 refused: product 179 is not 90008 Granola Fruit Nut Batch (type batch)';
    END IF;
    IF (SELECT count(*) FROM products WHERE id = 284 AND odoo_code = '90026' AND type = 'batch') <> 1 THEN
        RAISE EXCEPTION '078 refused: product 284 is not 90026 (SS Classic Chocolate Chip #9 kosher batch)';
    END IF;
    IF (SELECT count(*) FROM products WHERE id = 107 AND odoo_code = '90002' AND type = 'batch') <> 1 THEN
        RAISE EXCEPTION '078 refused: product 107 is not 90002 Batch Classic Granola #9';
    END IF;
    SELECT count(*) INTO n FROM products
     WHERE (id, odoo_code) IN ((3, '11003'), (49, '11049'), (36, '11036'), (17, '11017'), (72, '25010'))
       AND type = 'ingredient' AND COALESCE(active, true);
    IF n <> 5 THEN
        RAISE EXCEPTION '078 refused: an ingredient (3 almonds, 49 walnuts, 36 raisins, 17 cranberries, 72 chips 4,000 CT) is missing, renumbered or inactive';
    END IF;

    -- 179 make-recipe: exactly the five recorded lines, each at its BEFORE or AFTER value.
    SELECT count(*) INTO n FROM batch_formulas WHERE product_id = 179;
    IF n <> 5
       OR (SELECT count(*) FROM batch_formulas WHERE product_id = 179 AND ingredient_product_id IN (107, 3, 49, 36, 17)) <> 5 THEN
        RAISE EXCEPTION '078 refused: batch_formulas for 179 is not the recorded five-line recipe (107, 3, 49, 36, 17); found % rows', n;
    END IF;
    IF EXISTS (
        SELECT 1 FROM batch_formulas WHERE product_id = 179 AND (
            COALESCE(exclude_from_inventory, false)
            OR (ingredient_product_id = 107 AND quantity_lb NOT IN (20.0000, 276.3000))
            OR (ingredient_product_id IN (3, 49, 36, 17) AND quantity_lb NOT IN (1.2500, 13.0000))
        )
    ) THEN
        RAISE EXCEPTION '078 refused: a batch_formulas line for 179 is neither the recorded before-value nor the target';
    END IF;

    -- 179 batch size.
    IF (SELECT default_batch_lb FROM products WHERE id = 179) IS DISTINCT FROM 384.5200
       AND (SELECT default_batch_lb FROM products WHERE id = 179) IS DISTINCT FROM 328.3000 THEN
        RAISE EXCEPTION '078 refused: products.default_batch_lb for 179 is neither 384.52 nor 328.3';
    END IF;

    -- 179 legacy case yields.
    IF (SELECT yield_25lb_cases FROM products WHERE id = 179) IS DISTINCT FROM 15
       AND (SELECT yield_25lb_cases FROM products WHERE id = 179) IS DISTINCT FROM 13 THEN
        RAISE EXCEPTION '078 refused: products.yield_25lb_cases for 179 is neither 15 nor 13';
    END IF;
    IF (SELECT yield_10lb_cases FROM products WHERE id = 179) IS DISTINCT FROM 38
       AND (SELECT yield_10lb_cases FROM products WHERE id = 179) IS DISTINCT FROM 32 THEN
        RAISE EXCEPTION '078 refused: products.yield_10lb_cases for 179 is neither 38 nor 32';
    END IF;

    -- 179 planner rows: exactly the five recorded rows, each at its BEFORE or AFTER value.
    SELECT count(*) INTO n FROM product_bom WHERE finished_product_id = 179;
    IF n <> 5
       OR (SELECT count(*) FROM product_bom WHERE finished_product_id = 179 AND component_product_id IN (107, 3, 49, 36, 17)) <> 5 THEN
        RAISE EXCEPTION '078 refused: product_bom for 179 is not the recorded five planner rows (107, 3, 49, 36, 17); found % rows', n;
    END IF;
    IF EXISTS (
        SELECT 1 FROM product_bom WHERE finished_product_id = 179 AND (
            COALESCE(uom, 'lb') <> 'lb'
            OR (component_product_id = 107 AND quantity NOT IN (323.0000, 276.3000))
            OR (component_product_id IN (3, 49, 36, 17) AND quantity NOT IN (15.3800, 13.0000))
        )
    ) THEN
        RAISE EXCEPTION '078 refused: a product_bom row for 179 is neither the recorded before-value nor the target';
    END IF;

    -- 284 chips: exactly one chips line, 25 lb, deducted, on 73 (before) or 72 (after).
    SELECT count(*) INTO n FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id IN (72, 73);
    IF n <> 1 THEN
        RAISE EXCEPTION '078 refused: 284 must carry exactly one chips line (72 or 73); found %', n;
    END IF;
    IF (SELECT quantity_lb FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id IN (72, 73)) <> 25.0000 THEN
        RAISE EXCEPTION '078 refused: the 284 chips line is not 25 lb';
    END IF;
    IF (SELECT COALESCE(exclude_from_inventory, false) FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id IN (72, 73)) THEN
        RAISE EXCEPTION '078 refused: the 284 chips line is excluded from inventory (a make would post 348 lb with no chips deducted; record a 1,000 CT make as an A5 substitution instead)';
    END IF;
END $$;

-- 1. Fruit Nut make-recipe (what /make deducts): 276.3 lb Classic #9 + 13 lb each inclusion.
UPDATE batch_formulas SET quantity_lb = 276.3000
 WHERE product_id = 179 AND ingredient_product_id = 107 AND quantity_lb <> 276.3000;
UPDATE batch_formulas SET quantity_lb = 13.0000
 WHERE product_id = 179 AND ingredient_product_id IN (3, 49, 36, 17) AND quantity_lb <> 13.0000;

-- 2. Fruit Nut batch size (what /make posts per batch): 328.3 lb.
UPDATE products SET default_batch_lb = 328.3000
 WHERE id = 179 AND default_batch_lb IS DISTINCT FROM 328.3000;

-- 2b. Fruit Nut legacy case yields: 13 x 25 lb, 32 x 10 lb per 328.3 lb batch.
UPDATE products SET yield_25lb_cases = 13
 WHERE id = 179 AND yield_25lb_cases IS DISTINCT FROM 13;
UPDATE products SET yield_10lb_cases = 32
 WHERE id = 179 AND yield_10lb_cases IS DISTINCT FROM 32;

-- 3. Fruit Nut planner rows: rewritten in place (same ids, same uom), never deleted.
UPDATE product_bom SET quantity = 276.3000
 WHERE finished_product_id = 179 AND component_product_id = 107 AND quantity <> 276.3000;
UPDATE product_bom SET quantity = 13.0000
 WHERE finished_product_id = 179 AND component_product_id IN (3, 49, 36, 17) AND quantity <> 13.0000;

-- 4. 284 chips: 1,000 CT (73) -> 4,000 CT (72), still 25 lb (catalog-cleanup.csv row 29).
UPDATE batch_formulas SET ingredient_product_id = 72
 WHERE product_id = 284 AND ingredient_product_id = 73 AND quantity_lb = 25.0000;

-- Post-conditions: the recipe, the batch size and the planner rows agree at 328.3 lb;
-- 179 yields 13/32; 284 has one deducted 4,000 CT line and no 1,000 CT line.
DO $$
BEGIN
    IF (SELECT sum(quantity_lb) FROM batch_formulas WHERE product_id = 179) <> 328.3000 THEN
        RAISE EXCEPTION '078 post-check failed: 179 batch_formulas do not sum to 328.3';
    END IF;
    IF (SELECT default_batch_lb FROM products WHERE id = 179) <> 328.3000 THEN
        RAISE EXCEPTION '078 post-check failed: 179 default_batch_lb is not 328.3';
    END IF;
    IF (SELECT sum(quantity) FROM product_bom WHERE finished_product_id = 179) <> 328.3000 THEN
        RAISE EXCEPTION '078 post-check failed: 179 product_bom rows do not sum to 328.3';
    END IF;
    IF (SELECT (yield_25lb_cases, yield_10lb_cases) IS DISTINCT FROM (13, 32) FROM products WHERE id = 179) THEN
        RAISE EXCEPTION '078 post-check failed: 179 yields are not 13 x 25 lb / 32 x 10 lb';
    END IF;
    IF EXISTS (SELECT 1 FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id = 73)
       OR (SELECT count(*) FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id = 72 AND quantity_lb = 25.0000
                                          AND NOT COALESCE(exclude_from_inventory, false)) <> 1 THEN
        RAISE EXCEPTION '078 post-check failed: 284 chips line is not a single deducted 25 lb line on product 72';
    END IF;
END $$;

INSERT INTO migration_markers (name) VALUES ('078_recipe_corrections') ON CONFLICT (name) DO NOTHING;

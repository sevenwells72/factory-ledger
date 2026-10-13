-- 078 down: restore the production values recorded read-only on 2026-10-13 17:43 UTC
-- (see the BEFORE column in migrations/078_recipe_corrections.sql).
-- Data only; same recipe as the up file (table owner, port 5432, one transaction,
-- SET LOCAL lock_timeout='5s'). Strict: it runs only when every row 078 touches still
-- holds exactly the value 078 wrote, so it can never overwrite a later edit. A second
-- run refuses (the rows are no longer at 078's values); that is the intended no-op.
-- History (transactions, transaction_lines, lots, ingredient_lot_consumption) is never
-- read or written: makes posted while 078 was in force stand as posted.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM migration_markers WHERE name = '078_recipe_corrections') THEN
        RAISE EXCEPTION '078 down refused: marker 078_recipe_corrections is absent';
    END IF;
    IF (SELECT count(*) FROM products WHERE id = 179 AND odoo_code = '90008') <> 1
       OR (SELECT count(*) FROM products WHERE id = 284 AND odoo_code = '90026') <> 1 THEN
        RAISE EXCEPTION '078 down refused: products 179/284 are not the recorded catalog rows';
    END IF;

    -- 179 make-recipe: exactly ids 179–183 at the 078 values, none excluded.
    IF (SELECT count(*) FROM batch_formulas WHERE product_id = 179) <> 5
       OR (SELECT count(*) FROM batch_formulas
            WHERE product_id = 179 AND NOT COALESCE(exclude_from_inventory, false)
              AND ((id = 179 AND ingredient_product_id = 107 AND quantity_lb = 276.3000)
                OR (id = 180 AND ingredient_product_id = 49 AND quantity_lb = 13.0000)
                OR (id = 181 AND ingredient_product_id = 3 AND quantity_lb = 13.0000)
                OR (id = 182 AND ingredient_product_id = 36 AND quantity_lb = 13.0000)
                OR (id = 183 AND ingredient_product_id = 17 AND quantity_lb = 13.0000))) <> 5 THEN
        RAISE EXCEPTION '078 down refused: batch_formulas for 179 is not exactly the 078 recipe (276.3 + 13 x 4)';
    END IF;

    -- 179 batch size and yields.
    IF (SELECT (default_batch_lb, yield_25lb_cases, yield_10lb_cases) IS DISTINCT FROM (328.3000::numeric, 13, 32)
          FROM products WHERE id = 179) THEN
        RAISE EXCEPTION '078 down refused: product 179 batch size / yields are not exactly 328.3 / 13 / 32';
    END IF;

    -- 179 planner rows: exactly ids 117–121 at the 078 values.
    IF (SELECT count(*) FROM product_bom WHERE finished_product_id = 179) <> 5
       OR (SELECT count(*) FROM product_bom
            WHERE finished_product_id = 179 AND COALESCE(uom, 'lb') = 'lb'
              AND ((id = 117 AND component_product_id = 107 AND quantity = 276.3000)
                OR (id = 118 AND component_product_id = 3 AND quantity = 13.0000)
                OR (id = 119 AND component_product_id = 49 AND quantity = 13.0000)
                OR (id = 120 AND component_product_id = 36 AND quantity = 13.0000)
                OR (id = 121 AND component_product_id = 17 AND quantity = 13.0000))) <> 5 THEN
        RAISE EXCEPTION '078 down refused: product_bom for 179 is not exactly the 078 planner rows (276.3 + 13 x 4)';
    END IF;

    -- 284 chips: exactly one chips line, id 198, deducted, 25 lb of 72.
    IF (SELECT count(*) FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id IN (72, 73)) <> 1
       OR NOT EXISTS (SELECT 1 FROM batch_formulas
                       WHERE id = 198 AND product_id = 284 AND ingredient_product_id = 72
                         AND quantity_lb = 25.0000 AND NOT COALESCE(exclude_from_inventory, false)) THEN
        RAISE EXCEPTION '078 down refused: the 284 chips line is not exactly id 198 = 25 lb of 72 (4,000 CT), deducted';
    END IF;
END $$;

UPDATE batch_formulas SET quantity_lb = 20.0000 WHERE id = 179;
UPDATE batch_formulas SET quantity_lb = 1.2500 WHERE id IN (180, 181, 182, 183);
UPDATE products SET default_batch_lb = 384.5200, yield_25lb_cases = 15, yield_10lb_cases = 38 WHERE id = 179;
UPDATE product_bom SET quantity = 323.0000 WHERE id = 117;
UPDATE product_bom SET quantity = 15.3800 WHERE id IN (118, 119, 120, 121);
UPDATE batch_formulas SET ingredient_product_id = 73 WHERE id = 198;

DO $$
BEGIN
    IF (SELECT sum(quantity_lb) FROM batch_formulas WHERE product_id = 179) <> 25.0000
       OR (SELECT (default_batch_lb, yield_25lb_cases, yield_10lb_cases) IS DISTINCT FROM (384.5200::numeric, 15, 38) FROM products WHERE id = 179)
       OR (SELECT sum(quantity) FROM product_bom WHERE finished_product_id = 179) <> 384.5200
       OR (SELECT count(*) FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id = 73 AND quantity_lb = 25.0000) <> 1 THEN
        RAISE EXCEPTION '078 down post-check failed: before-values not restored';
    END IF;
END $$;

DELETE FROM migration_markers WHERE name = '078_recipe_corrections';

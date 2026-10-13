-- 078 down: restore the production values recorded read-only on 2026-10-13 17:43 UTC
-- (see the BEFORE column in migrations/078_recipe_corrections.sql).
-- Data only; same recipe as the up file (table owner, port 5432, one transaction,
-- SET LOCAL lock_timeout='5s'). Rerunnable. Makes posted while 078 was in force are
-- history and are NOT touched or reversed; their count is reported as a NOTICE so the
-- operator knows what was produced under the corrected recipe.
DO $$
DECLARE
    n integer;
    since timestamptz;
BEGIN
    IF (SELECT count(*) FROM products WHERE id = 179 AND odoo_code = '90008') <> 1
       OR (SELECT count(*) FROM products WHERE id = 284 AND odoo_code = '90026') <> 1 THEN
        RAISE EXCEPTION '078 down refused: products 179/284 are not the recorded catalog rows';
    END IF;
    IF (SELECT count(*) FROM batch_formulas WHERE product_id = 179 AND ingredient_product_id IN (107, 3, 49, 36, 17)) <> 5
       OR (SELECT count(*) FROM batch_formulas WHERE product_id = 179) <> 5 THEN
        RAISE EXCEPTION '078 down refused: batch_formulas for 179 is not the recorded five-line recipe';
    END IF;
    IF (SELECT count(*) FROM product_bom WHERE finished_product_id = 179 AND component_product_id IN (107, 3, 49, 36, 17)) <> 5
       OR (SELECT count(*) FROM product_bom WHERE finished_product_id = 179) <> 5 THEN
        RAISE EXCEPTION '078 down refused: product_bom for 179 is not the recorded five planner rows';
    END IF;
    SELECT count(*) INTO n FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id IN (72, 73);
    IF n <> 1 THEN
        RAISE EXCEPTION '078 down refused: 284 must carry exactly one chips line (72 or 73); found %', n;
    END IF;
    SELECT applied_at INTO since FROM migration_markers WHERE name = '078_recipe_corrections';
    IF since IS NOT NULL THEN
        SELECT count(DISTINCT t.id) INTO n
          FROM transactions t JOIN transaction_lines tl ON tl.transaction_id = t.id
         WHERE t.type = 'make' AND tl.product_id IN (179, 284) AND tl.quantity_lb > 0 AND t.created_at >= since;
        RAISE NOTICE '078 down: % make transaction(s) for 179/284 were posted since 078 was applied (%); they stand as history', n, since;
    END IF;
END $$;

UPDATE batch_formulas SET quantity_lb = 20.0000
 WHERE product_id = 179 AND ingredient_product_id = 107 AND quantity_lb <> 20.0000;
UPDATE batch_formulas SET quantity_lb = 1.2500
 WHERE product_id = 179 AND ingredient_product_id IN (3, 49, 36, 17) AND quantity_lb <> 1.2500;
UPDATE products SET default_batch_lb = 384.5200
 WHERE id = 179 AND default_batch_lb IS DISTINCT FROM 384.5200;
UPDATE product_bom SET quantity = 323.0000
 WHERE finished_product_id = 179 AND component_product_id = 107 AND quantity <> 323.0000;
UPDATE product_bom SET quantity = 15.3800
 WHERE finished_product_id = 179 AND component_product_id IN (3, 49, 36, 17) AND quantity <> 15.3800;
UPDATE batch_formulas SET ingredient_product_id = 73
 WHERE product_id = 284 AND ingredient_product_id = 72 AND quantity_lb = 25.0000;

DO $$
BEGIN
    IF (SELECT sum(quantity_lb) FROM batch_formulas WHERE product_id = 179) <> 25.0000
       OR (SELECT default_batch_lb FROM products WHERE id = 179) <> 384.5200
       OR (SELECT sum(quantity) FROM product_bom WHERE finished_product_id = 179) <> 384.5200
       OR (SELECT count(*) FROM batch_formulas WHERE product_id = 284 AND ingredient_product_id = 73 AND quantity_lb = 25.0000) <> 1 THEN
        RAISE EXCEPTION '078 down post-check failed: before-values not restored';
    END IF;
END $$;

DELETE FROM migration_markers WHERE name = '078_recipe_corrections';

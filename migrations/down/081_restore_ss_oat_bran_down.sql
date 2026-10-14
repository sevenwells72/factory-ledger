-- 081: Strict down: restore the three recorded zero values, only from the exact corrected recipe.
-- Verified production READ ONLY on 2026-10-14, port 5432, ROLLBACK.
-- Original source: _backup_20260305_batch_formulas; independently corroborated by
-- transactions 2267 (116: 56 lb / 2 batches), 2242 (114: 290 / 10), 2057 (118: 140 / 5).
-- BEFORE -> AFTER: formula 69 / product 116: 0 -> 28 lb; 56 / 114: 0 -> 29;
-- 82 / 118: 0 -> 28. Batch sizes stay 350, 393, 379 lb respectively.
-- Full before evidence: docs/deployments/pilot-triage-production-recipe-read.json.
-- DATA ONLY: no historical transactions, lines, consumption, lots or balances touched.
-- Apply explicitly in ONE transaction as table owner, port 5432, staging only:
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
-- SET LOCAL search_path=public; [this file]; COMMIT;
-- Formula ids are preserved; logical product/ingredient identity permits staging ids.
DO $$
BEGIN
    LOCK TABLE public.products, public.batch_formulas IN SHARE ROW EXCLUSIVE MODE;
    IF NOT EXISTS (SELECT 1 FROM migration_markers WHERE name='081_restore_ss_oat_bran') THEN
        RAISE EXCEPTION '081 down refused: applied marker missing';
    END IF;
    IF (SELECT count(*) FROM products WHERE (id,odoo_code,default_batch_lb) IN
        ((116,'90016',350),(114,'90011',393),(118,'90013',379))
        AND type='batch' AND COALESCE(active,true)) <> 3 THEN
        RAISE EXCEPTION '081 down refused: SS product identity or batch size changed';
    END IF;
    IF (SELECT count(*) FROM products WHERE id=31 AND odoo_code='11031'
        AND type='ingredient' AND COALESCE(active,true)) <> 1 THEN
        RAISE EXCEPTION '081 down refused: oat bran identity or active state changed';
    END IF;
    IF EXISTS (
        WITH expected(product_id,ingredient_product_id,quantity_lb) AS (VALUES
        (114, 3, 19.0000),
        (114, 13, 19.0000),
        (114, 30, 47.0000),
        (114, 31, 29.00),
        (114, 32, 200.0000),
        (114, 34, 22.0000),
        (114, 46, 19.0000),
        (114, 72, 38.0000),
        (116, 3, 21.0000),
        (116, 13, 21.0000),
        (116, 30, 50.0000),
        (116, 31, 28.00),
        (116, 32, 185.0000),
        (116, 34, 24.0000),
        (116, 46, 21.0000),
        (118, 3, 21.0000),
        (118, 7, 4.0000),
        (118, 13, 21.0000),
        (118, 17, 25.0000),
        (118, 30, 50.0000),
        (118, 31, 28.00),
        (118, 32, 185.0000),
        (118, 34, 24.0000),
        (118, 46, 21.0000)
        ), actual AS (SELECT * FROM batch_formulas WHERE product_id IN (114,116,118))
        SELECT 1 FROM expected e FULL JOIN actual b USING (product_id,ingredient_product_id)
        WHERE e.product_id IS NULL OR b.id IS NULL OR COALESCE(b.exclude_from_inventory,false)
           OR b.quantity_lb IS DISTINCT FROM e.quantity_lb
    ) OR (SELECT count(*) FROM batch_formulas WHERE product_id IN (114,116,118)) <> 24 THEN
        RAISE EXCEPTION '081 down refused: recipes differ from the recorded after values';
    END IF;
END $$;

UPDATE batch_formulas SET quantity_lb=0 WHERE product_id IN (114,116,118) AND ingredient_product_id=31;
DELETE FROM migration_markers WHERE name='081_restore_ss_oat_bran';

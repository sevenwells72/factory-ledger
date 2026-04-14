-- 032: Backfill has_bom for all products with BOM lines + rename product 166
-- Idempotent: both statements are safe to re-run

-- BUG 2: Sync has_bom flag for any product that has batch_formulas rows
UPDATE products
SET    has_bom = true
WHERE  id IN (SELECT DISTINCT product_id FROM batch_formulas)
  AND  has_bom = false;

-- BUG 3: Rename "Sweetened Flake Coconut" → "Coconut Sweetened Flake" for natural search
UPDATE products
SET    name = 'Coconut Sweetened Flake'
WHERE  id = 166
  AND  name = 'Sweetened Flake Coconut';

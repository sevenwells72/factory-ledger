-- Migration 004: Add exclude_from_inventory flag to batch_formulas
-- Purpose: Allow BOM ingredients (like Water from municipal supply) to remain
-- visible in formulas for operator reference without blocking production
-- or creating phantom inventory shortages.
--
-- This migration runs automatically at app startup via main.py.

ALTER TABLE batch_formulas
ADD COLUMN IF NOT EXISTS exclude_from_inventory BOOLEAN DEFAULT false;

-- Flag Water as excluded in all formulas
UPDATE batch_formulas bf
SET exclude_from_inventory = true
FROM products p
WHERE p.id = bf.ingredient_product_id
  AND LOWER(p.name) = 'water'
  AND bf.exclude_from_inventory = false;

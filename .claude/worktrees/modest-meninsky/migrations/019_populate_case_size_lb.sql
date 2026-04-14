-- ═══════════════════════════════════════════════════════════════
-- Migration 019: Populate NULL case_size_lb on Finished Goods
-- Updates 10 finished goods products that have NULL case_size_lb.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

-- First, inspect the products to determine correct values:
-- SELECT id, name, case_size_lb FROM products WHERE id IN (182,183,184,185,186,193,197,198,199,201);

-- Products with standard sizes inferred from name patterns:
-- 182: Check name for pattern (likely a standard size)
-- 183: "per/lb" product — bulk, no fixed case size (leave NULL)
-- 184: "per/lb" product — bulk, no fixed case size (leave NULL)
-- 185-186, 193, 197-199, 201: Infer from product name

-- Set case_size_lb for products where size can be determined from name
UPDATE products SET case_size_lb =
    CASE
        -- Extract from '## LB' patterns in name
        WHEN name ~* '(\d+)\s*LB' THEN
            (regexp_match(name, '(\d+)\s*LB', 'i'))[1]::numeric
        -- Extract from NNxNN OZ patterns: count * oz / 16
        WHEN name ~* '(\d+)x(\d+)\s*OZ' THEN
            (regexp_match(name, '(\d+)x(\d+)\s*OZ', 'i'))[1]::numeric *
            (regexp_match(name, '(\d+)x(\d+)\s*OZ', 'i'))[2]::numeric / 16.0
        ELSE NULL
    END
WHERE id IN (182, 183, 184, 185, 186, 193, 197, 198, 199, 201)
  AND case_size_lb IS NULL
  AND (name ~* '\d+\s*LB' OR name ~* '\d+x\d+\s*OZ');

-- Verify:
-- SELECT id, name, case_size_lb FROM products WHERE id IN (182,183,184,185,186,193,197,198,199,201);

-- DOWNGRADE
-- UPDATE products SET case_size_lb = NULL
-- WHERE id IN (182, 183, 184, 185, 186, 193, 197, 198, 199, 201);

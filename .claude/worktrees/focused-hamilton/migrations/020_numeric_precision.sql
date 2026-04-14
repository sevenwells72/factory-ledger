-- ═══════════════════════════════════════════════════════════════
-- Migration 020: Fix Floating Point Precision
-- Converts quantity columns from float/double precision to NUMERIC(14,4).
-- Cleans up existing floating point dust.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

-- Step 1: Convert transaction_lines.quantity_lb to NUMERIC(14,4)
ALTER TABLE transaction_lines
    ALTER COLUMN quantity_lb TYPE NUMERIC(14,4) USING ROUND(quantity_lb::numeric, 4);

-- Step 2: Convert ingredient_lot_consumption.quantity_lb to NUMERIC(14,4)
ALTER TABLE ingredient_lot_consumption
    ALTER COLUMN quantity_lb TYPE NUMERIC(14,4) USING ROUND(quantity_lb::numeric, 4);

-- Step 3: Convert batch_formulas.quantity_lb to NUMERIC(14,4)
ALTER TABLE batch_formulas
    ALTER COLUMN quantity_lb TYPE NUMERIC(14,4) USING ROUND(quantity_lb::numeric, 4);

-- Step 4: Convert sales_order_lines quantity columns
ALTER TABLE sales_order_lines
    ALTER COLUMN quantity_lb TYPE NUMERIC(14,4) USING ROUND(quantity_lb::numeric, 4);
ALTER TABLE sales_order_lines
    ALTER COLUMN quantity_shipped_lb TYPE NUMERIC(14,4) USING ROUND(quantity_shipped_lb::numeric, 4);

-- Step 5: Convert products.case_size_lb and default_batch_lb
ALTER TABLE products
    ALTER COLUMN case_size_lb TYPE NUMERIC(14,4) USING ROUND(case_size_lb::numeric, 4);
ALTER TABLE products
    ALTER COLUMN default_batch_lb TYPE NUMERIC(14,4) USING ROUND(default_batch_lb::numeric, 4);

-- Step 6: Clean up any remaining dust in transaction_lines
UPDATE transaction_lines
SET quantity_lb = ROUND(quantity_lb::numeric, 4)
WHERE ABS(quantity_lb - ROUND(quantity_lb::numeric, 4)) > 0;

-- DOWNGRADE
-- ALTER TABLE transaction_lines ALTER COLUMN quantity_lb TYPE DOUBLE PRECISION;
-- ALTER TABLE ingredient_lot_consumption ALTER COLUMN quantity_lb TYPE DOUBLE PRECISION;
-- ALTER TABLE batch_formulas ALTER COLUMN quantity_lb TYPE DOUBLE PRECISION;
-- ALTER TABLE sales_order_lines ALTER COLUMN quantity_lb TYPE DOUBLE PRECISION;
-- ALTER TABLE sales_order_lines ALTER COLUMN quantity_shipped_lb TYPE DOUBLE PRECISION;
-- ALTER TABLE products ALTER COLUMN case_size_lb TYPE DOUBLE PRECISION;
-- ALTER TABLE products ALTER COLUMN default_batch_lb TYPE DOUBLE PRECISION;

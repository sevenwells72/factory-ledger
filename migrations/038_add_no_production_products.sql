-- Mark resale/as-is products that should not enter the production scheduler.
-- Idempotent: safe to rerun after the column and flags have been applied.

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS no_production BOOLEAN NOT NULL DEFAULT false;

DO $$
DECLARE
    expected_skus CONSTANT TEXT[] := ARRAY[
        '10302', '10303', '10304', '10305', '10306', '31011',
        '10045', '10046', '10047', '10048', '10049', '10050', '10051',
        '10052', '10053', '10054', '10055', '10056', '10058', '10059',
        '11009', '11010', '11011', '11012', '11013', '11014'
    ];
    matched_count INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO matched_count
      FROM products
     WHERE odoo_code = ANY(expected_skus);

    IF matched_count <> 26 THEN
        RAISE EXCEPTION
            'Expected exactly 26 no-production product rows, found %',
            matched_count;
    END IF;

    UPDATE products
       SET no_production = true
     WHERE odoo_code = ANY(expected_skus);

    -- Explicit invariant: 31012 is produced/repacked and must remain schedulable.
    UPDATE products
       SET no_production = false
     WHERE odoo_code = '31012';

    IF (SELECT COUNT(*) FROM products WHERE odoo_code = ANY(expected_skus) AND no_production) <> 26 THEN
        RAISE EXCEPTION 'Failed to flag all 26 no-production products';
    END IF;

    IF EXISTS (SELECT 1 FROM products WHERE odoo_code = '31012' AND no_production) THEN
        RAISE EXCEPTION 'Invariant failed: SKU 31012 must remain schedulable';
    END IF;
END $$;

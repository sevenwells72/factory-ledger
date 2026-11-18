-- 095 down: put "Real" back in the three chocolate chip ingredient names.
-- Data only, products.name on ids 72 / 73 / 74; same recipe as the up file (table
-- owner, port 5432, one transaction, SET LOCAL lock_timeout='5s'). Revert the
-- dashboard/dashboard_config.json change in the same step, or the Inventory
-- ingredients panel lists the two chips under "Missing SKUs".
-- Strict: it runs only while every row still holds exactly the name 095 wrote, so it
-- can never overwrite a later rename. A second run refuses (the marker is gone); that
-- is the intended no-op. No history table is read or written.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM migration_markers WHERE name = '095_rename_chocolate_chips') THEN
        RAISE EXCEPTION '095 down refused: marker 095_rename_chocolate_chips is absent';
    END IF;
    IF (SELECT count(*) FROM products
         WHERE type = 'ingredient'
           AND (id, odoo_code, name) IN ((72, '25010', 'Chocolate Chips – 4,000 CT'),
                                         (73, '25011', 'Chocolate Chips – 1,000 CT'),
                                         (74, '25012', 'Chocolate Chips – 10,000 CT'))) <> 3 THEN
        RAISE EXCEPTION '095 down refused: products 72 / 73 / 74 do not all hold exactly the names 095 wrote';
    END IF;
    IF EXISTS (SELECT 1 FROM products
                WHERE id NOT IN (72, 73, 74)
                  AND name IN ('Chocolate Chips – Real – 4,000 CT',
                               'Chocolate Chips – Real – 1,000 CT',
                               'Chocolate Chips – Real – 10,000 CT')) THEN
        RAISE EXCEPTION '095 down refused: a recorded before-name is already used by another product';
    END IF;

    UPDATE products SET name = 'Chocolate Chips – Real – 4,000 CT'  WHERE id = 72;
    UPDATE products SET name = 'Chocolate Chips – Real – 1,000 CT'  WHERE id = 73;
    UPDATE products SET name = 'Chocolate Chips – Real – 10,000 CT' WHERE id = 74;
END $$;

DELETE FROM migration_markers WHERE name = '095_rename_chocolate_chips';

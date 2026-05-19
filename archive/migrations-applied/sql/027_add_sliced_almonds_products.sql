-- Migration 027: Add Sliced Almonds ingredient products
-- Run in Supabase SQL Editor (production: MyFirstProject, us-east-1)
--
-- Context: Arturo tried to receive "Almonds Slice" (sliced almonds from Blue Stripes)
-- and got "Product not found" because no sliced almonds product existed in the DB.
-- The dashboard_config.json references both variants as expected ingredients.

-- General sliced almonds ingredient (non-BS suppliers)
INSERT INTO products (name, type, uom, storage_type, active)
SELECT 'Almonds – Sliced', 'ingredient', 'lb', 'dry', true
WHERE NOT EXISTS (
    SELECT 1 FROM products WHERE LOWER(name) = LOWER('Almonds – Sliced')
);

-- Blue Stripes sliced almonds (raw, private label ingredient)
INSERT INTO products (name, type, uom, storage_type, label_type, active)
SELECT 'BS Almonds – Sliced – Raw', 'ingredient', 'lb', 'dry', 'private_label', true
WHERE NOT EXISTS (
    SELECT 1 FROM products WHERE LOWER(name) = LOWER('BS Almonds – Sliced – Raw')
);

-- ROLLBACK:
-- DELETE FROM products WHERE name IN ('Almonds – Sliced', 'BS Almonds – Sliced – Raw')
--   AND NOT EXISTS (SELECT 1 FROM transaction_lines tl JOIN lots l ON l.id = tl.lot_id WHERE l.product_id = products.id);

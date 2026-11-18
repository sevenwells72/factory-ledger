-- 095: rename the three "Chocolate Chips – Real – N CT" ingredients (owner request,
-- Michael, 2026-11-18). The word "Real" is dropped; nothing else changes.
--
-- DATA ONLY, products.name on three rows. No DDL. Product ids, odoo codes and every
-- other column stay as they are. No history table is read or written: transactions,
-- transaction_lines, lots, ingredient_lot_consumption, write_tickets and exceptions
-- keep whatever name text they recorded at the time. Recipes, lots, orders and
-- exceptions link by product id and follow the new name on their own.
--
-- Apply as the table owner on port 5432, STAGING FIRST, production only with the
-- owner's approval and a verified backup:
--   BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
--   SET LOCAL search_path=public; \i migrations/095_rename_chocolate_chips.sql; COMMIT;
--
-- ROLLOUT ORDER: dashboard/dashboard_config.json lists these ingredients by exact
-- name for the Inventory "Granola General Ingredients" panel. This file and the
-- config change in the same PR go out back to back: deploy the PR, then apply this
-- file as soon as the deploy is live. While only one of the two is in place the
-- panel shows the two listed chips (4,000 CT and 1,000 CT) under "Missing SKUs";
-- stock, recipes and entry screens are not affected because they use product ids.
--
-- Rerunnable: each row is accepted in its recorded BEFORE name or its AFTER name and
-- anything else is refused, so the file cannot be applied to a catalog it was not
-- written for. Down file: migrations/down/095_rename_chocolate_chips_down.sql.
--
-- BEFORE (production, read-only 2026-11-18)                 AFTER
--   72  25010  Chocolate Chips – Real – 4,000 CT            Chocolate Chips – 4,000 CT
--   73  25011  Chocolate Chips – Real – 1,000 CT            Chocolate Chips – 1,000 CT
--   74  25012  Chocolate Chips – Real – 10,000 CT           Chocolate Chips – 10,000 CT
-- Unchanged on purpose: 181 / 182 "Real Chocolate Chips 10000ct 50 LB / 30 LB"
-- (finished co-pack items), 6 "Chocolate Chips – Sugar Free", 53 "White Chocolate Chips".

DO $$
DECLARE
    r record;
    current_name text;
BEGIN
    FOR r IN
        SELECT * FROM (VALUES
            (72, '25010', 'Chocolate Chips – Real – 4,000 CT',  'Chocolate Chips – 4,000 CT'),
            (73, '25011', 'Chocolate Chips – Real – 1,000 CT',  'Chocolate Chips – 1,000 CT'),
            (74, '25012', 'Chocolate Chips – Real – 10,000 CT', 'Chocolate Chips – 10,000 CT')
        ) AS v(id, odoo_code, old_name, new_name)
    LOOP
        -- Identity of the row this file was written for.
        IF (SELECT count(*) FROM products
             WHERE id = r.id AND odoo_code = r.odoo_code AND type = 'ingredient') <> 1 THEN
            RAISE EXCEPTION '095 refused: product % is not ingredient % (%)', r.id, r.odoo_code, r.old_name;
        END IF;
        SELECT name INTO current_name FROM products WHERE id = r.id;
        IF current_name NOT IN (r.old_name, r.new_name) THEN
            RAISE EXCEPTION '095 refused: product % is named "%", neither the recorded name nor the target', r.id, current_name;
        END IF;
        -- products.name is unique; say which product holds the target name instead of
        -- failing on the index.
        IF EXISTS (SELECT 1 FROM products WHERE name = r.new_name AND id <> r.id) THEN
            RAISE EXCEPTION '095 refused: the name "%" is already used by another product', r.new_name;
        END IF;
    END LOOP;

    UPDATE products SET name = 'Chocolate Chips – 4,000 CT'
     WHERE id = 72 AND name = 'Chocolate Chips – Real – 4,000 CT';
    UPDATE products SET name = 'Chocolate Chips – 1,000 CT'
     WHERE id = 73 AND name = 'Chocolate Chips – Real – 1,000 CT';
    UPDATE products SET name = 'Chocolate Chips – 10,000 CT'
     WHERE id = 74 AND name = 'Chocolate Chips – Real – 10,000 CT';
END $$;

INSERT INTO migration_markers (name) VALUES ('095_rename_chocolate_chips') ON CONFLICT DO NOTHING;

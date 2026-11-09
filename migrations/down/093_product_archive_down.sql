-- Deploy code without product archive writes first. Refuses to drop the
-- attribution columns while any product carries an archive record: archive
-- history is never thrown away. products.active itself is untouched.
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM products WHERE archived_at IS NOT NULL OR archived_by IS NOT NULL) THEN
        RAISE EXCEPTION 'archived products carry attribution; keep migration 093';
    END IF;
END $$;
ALTER TABLE products DROP COLUMN IF EXISTS archived_at;
ALTER TABLE products DROP COLUMN IF EXISTS archived_by;
DELETE FROM migration_markers WHERE name='093_product_archive';

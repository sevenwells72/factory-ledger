-- Classify granola finished goods by pack format for dashboard reporting.
-- NULL means the classification is not applicable (for example coconut,
-- graham, or raw-material products).

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS pack_format TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conrelid = 'public.products'::regclass
           AND conname = 'products_pack_format_check'
    ) THEN
        ALTER TABLE products
            ADD CONSTRAINT products_pack_format_check
            CHECK (pack_format IS NULL OR pack_format IN ('10lb', '25lb', 'bagged'));
    END IF;
END $$;

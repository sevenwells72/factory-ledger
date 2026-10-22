-- Explicit apply only. No data changes or automatic startup application.
-- 085/086 are reserved by separate open PRs; apply this after 062 AND 063.
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public;
-- run this file with ON_ERROR_STOP; COMMIT.
DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='transaction_lot_confirmations'::regclass
        AND conname='transaction_lot_confirmations_method_check';
    IF position('supplier_lot' in COALESCE(definition,''))=0 THEN
        ALTER TABLE transaction_lot_confirmations
            DROP CONSTRAINT IF EXISTS transaction_lot_confirmations_method_check;
        IF definition IS NULL THEN
            definition := 'CHECK (method IN (''last4'',''full_code'',''scan'',''pallet''))';
        END IF;
        -- Preserve methods introduced by other branches (for example tap in
        -- reserved 085); applying this migration must never narrow the CHECK.
        EXECUTE 'ALTER TABLE transaction_lot_confirmations ADD CONSTRAINT '
            || 'transaction_lot_confirmations_method_check CHECK (('
            || substring(definition from 8 for length(definition)-8)
            || ') OR method = ''supplier_lot'')';
    END IF;
END $$;
-- Keep the original one-row-per-replaced-ingredient audit and its first actual
-- lot_id while retaining every consumed lot/pound when the replacement splits.
ALTER TABLE transaction_substitutions ADD COLUMN IF NOT EXISTS actual_lot_allocations jsonb NOT NULL DEFAULT '[]'::jsonb;
INSERT INTO migration_markers(name) VALUES ('087_supplier_lot_confirmation') ON CONFLICT DO NOTHING;

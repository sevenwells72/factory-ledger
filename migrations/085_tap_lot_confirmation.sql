-- Explicit operator tap evidence for reviewed Make/Pack ingredient lots.
-- Apply explicitly as table owner after 062, before enabling tap confirmation.
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='30s';
-- run this file with ON_ERROR_STOP; COMMIT. No startup hook or evidence rewrite.
DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='transaction_lot_confirmations'::regclass
        AND conname='transaction_lot_confirmations_method_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION 'transaction lot confirmation method constraint is missing';
    END IF;
    IF position('''tap''' in definition)=0 THEN
        ALTER TABLE transaction_lot_confirmations DROP CONSTRAINT transaction_lot_confirmations_method_check;
        EXECUTE 'ALTER TABLE transaction_lot_confirmations ADD CONSTRAINT transaction_lot_confirmations_method_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR method = ''tap'')';
    END IF;
END $$;
INSERT INTO migration_markers(name) VALUES ('085_tap_lot_confirmation') ON CONFLICT DO NOTHING;

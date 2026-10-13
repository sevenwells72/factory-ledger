-- Reverses 076_shift_confirmations. Refuses while any confirmation exists:
-- confirmations are the floor's attestation record and are never discarded
-- silently. Run in one transaction as the table owner on port 5432.
DO $$ BEGIN
    IF to_regclass('shift_confirmations') IS NOT NULL
       AND EXISTS (SELECT 1 FROM shift_confirmations) THEN
        RAISE EXCEPTION 'shift_confirmations holds rows; export them before reversing 076';
    END IF;
END $$;
DROP TABLE IF EXISTS shift_confirmations;
DELETE FROM migration_markers WHERE name = '076_shift_confirmations';

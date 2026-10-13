-- A5 part 2. Explicit apply only, STAGING ONLY for this builder.
-- App/table owner, one transaction: BEGIN; SET LOCAL search_path=public;
-- SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
-- include with ON_ERROR_STOP; COMMIT. No startup hook or historical rewrite.
-- Preserve every existing action; reuse 075 lot_events and 064 identity fields.
DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '079 requires the A1 write_tickets action constraint';
    END IF;
    IF position('update_supplier_lot' in definition)=0 THEN
        ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check;
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR action = ''update_supplier_lot'')';
    END IF;
END $$;
INSERT INTO migration_markers(name) VALUES ('079_supplier_lot_tickets') ON CONFLICT DO NOTHING;

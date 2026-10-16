DO $$ DECLARE definition text; BEGIN
    IF EXISTS (SELECT 1 FROM exceptions WHERE kind='ASSISTANT_OFFLINE') THEN
        RAISE EXCEPTION 'ASSISTANT_OFFLINE exceptions exist; remove them before rolling back 084';
    END IF;
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='exceptions'::regclass AND conname='exceptions_kind_check';
    IF position('ASSISTANT_OFFLINE' in definition)>0 THEN
        ALTER TABLE exceptions DROP CONSTRAINT exceptions_kind_check;
        EXECUTE 'ALTER TABLE exceptions ADD CONSTRAINT exceptions_kind_check '
            || replace(definition, ' OR (kind = ''ASSISTANT_OFFLINE''::text)', '');
    END IF;
END $$;
DROP INDEX IF EXISTS exceptions_one_assistant_outage_idx;
DELETE FROM migration_markers WHERE name='084_assistant_service_alert';

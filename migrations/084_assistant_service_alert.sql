-- Durable owner-visible notice for an assistant provider billing outage.
-- Apply explicitly; no provider or deployment configuration is changed.
DO $$ DECLARE definition text; BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='exceptions'::regclass AND conname='exceptions_kind_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION 'exceptions kind constraint is missing';
    END IF;
    IF position('ASSISTANT_OFFLINE' in definition)=0 THEN
        ALTER TABLE exceptions DROP CONSTRAINT exceptions_kind_check;
        EXECUTE 'ALTER TABLE exceptions ADD CONSTRAINT exceptions_kind_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR kind = ''ASSISTANT_OFFLINE'')';
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_one_assistant_outage_idx
    ON exceptions(kind) WHERE kind='ASSISTANT_OFFLINE' AND status IN ('open','escalated');
INSERT INTO migration_markers(name) VALUES ('084_assistant_service_alert') ON CONFLICT DO NOTHING;

-- D1/D2 receipt producers. Additive, idempotent; no historical backfill.
-- Explicit app/table-owner apply, staging only for this builder:
-- BEGIN; SET LOCAL search_path=public; SET LOCAL lock_timeout='5s';
-- SET LOCAL statement_timeout='60s'; include with ON_ERROR_STOP; COMMIT.
ALTER TABLE public.exceptions ADD COLUMN IF NOT EXISTS resolution_receipt_number text
    REFERENCES public.write_tickets(receipt_number);
DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '080 requires the A1 write_tickets action constraint';
    END IF;
    IF position('resolve_exception' in definition)=0 OR position('''void''' in definition)=0 THEN
        ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check;
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
            || substring(definition from 8 for length(definition)-8)
            || ') OR action IN (''resolve_exception'',''void''))';
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS write_tickets_void_correction_idx
    ON public.write_tickets ((result_ref->>'correction_id')) WHERE action='void' AND status='committed';
INSERT INTO public.migration_markers(name) VALUES ('080_resolution_receipts') ON CONFLICT DO NOTHING;

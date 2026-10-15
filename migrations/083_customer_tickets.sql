-- Create-customer tickets: additive, idempotent; direct routes unchanged.
-- Explicit app/table-owner apply BEFORE the application, staging first:
-- BEGIN; SET LOCAL search_path=public; SET LOCAL lock_timeout='5s';
-- SET LOCAL statement_timeout='60s'; include with ON_ERROR_STOP; COMMIT.
DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '083 requires the A1 write_tickets action constraint';
    END IF;
    IF position('create_customer' in definition)=0 THEN
        ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check;
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR action = ''create_customer'')';
    END IF;
END $$;
ALTER TABLE public.customers ADD COLUMN IF NOT EXISTS ticket_id bigint REFERENCES public.write_tickets(id);
CREATE INDEX IF NOT EXISTS customers_ticket_id_idx ON public.customers(ticket_id) WHERE ticket_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS write_tickets_customer_idx ON public.write_tickets(((result_ref->>'customer_id')::bigint))
    WHERE result_ref ? 'customer_id';
INSERT INTO public.migration_markers(name) VALUES ('083_customer_tickets') ON CONFLICT DO NOTHING;

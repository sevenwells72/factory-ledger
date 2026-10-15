-- Manual rollback after reverting the application. Export ticket evidence
-- first; existing create-customer tickets deliberately prevent narrowing.
-- App/table owner, explicit transaction + lock_timeout, ON_ERROR_STOP.
DO $$
DECLARE definition text; retained text;
BEGIN
    IF EXISTS (SELECT 1 FROM public.write_tickets WHERE action='create_customer') THEN
        RAISE EXCEPTION '083 rollback refused: create-customer ticket evidence exists';
    END IF;
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NOT NULL AND position('create_customer' in definition)>0 THEN
        -- Retain every other action, including any added by later migrations.
        SELECT string_agg(quote_literal(value), ',') INTO retained FROM
          (SELECT DISTINCT match[1] AS value FROM regexp_matches(definition, '''([^'']+)''', 'g') AS match
           WHERE match[1] <> 'create_customer') AS actions;
        IF retained IS NULL THEN
            RAISE EXCEPTION '083 rollback refused: prior ticket actions could not be read';
        END IF;
        ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check;
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (action IN (' || retained || '))';
    END IF;
END $$;
DROP INDEX IF EXISTS public.write_tickets_customer_idx;
DROP INDEX IF EXISTS public.customers_ticket_id_idx;
ALTER TABLE public.customers DROP COLUMN IF EXISTS ticket_id;
DELETE FROM public.migration_markers WHERE name='083_customer_tickets';

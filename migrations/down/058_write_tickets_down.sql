-- Manual rollback only, AFTER reverting the application; never automatic.
-- App/table owner, port 5432, ON_ERROR_STOP, explicit BEGIN/COMMIT wrapper.
-- SET LOCAL lock_timeout='5s' before including this file.
-- Refuses populated ticket/counter tables by default. Export tickets,
-- counters and transactions' ticket/receipt fields and verify the export
-- before SET LOCAL factory_ledger.confirm_ticket_export = 'yes'.
DO $$ BEGIN
    IF to_regclass('public.write_tickets') IS NOT NULL THEN
        LOCK TABLE public.write_tickets IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM public.write_tickets)
           AND current_setting('factory_ledger.confirm_ticket_export', true) IS DISTINCT FROM 'yes' THEN
            RAISE EXCEPTION '058 rollback refused: export and verify ticket evidence first';
        END IF;
    END IF;
    IF to_regclass('public.receipt_counters') IS NOT NULL THEN
        LOCK TABLE public.receipt_counters IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM public.receipt_counters)
           AND current_setting('factory_ledger.confirm_ticket_export', true) IS DISTINCT FROM 'yes' THEN
            RAISE EXCEPTION '058 rollback refused: export and verify receipt counters first';
        END IF;
    END IF;
END $$;
ALTER TABLE public.transactions
    DROP COLUMN IF EXISTS receipt_number,
    DROP COLUMN IF EXISTS ticket_id;
DROP TABLE IF EXISTS public.receipt_counters;
DROP TABLE IF EXISTS public.write_tickets;
DROP FUNCTION IF EXISTS public.write_ticket_preserve_committed();
DELETE FROM public.migration_markers WHERE name = '058_write_tickets';

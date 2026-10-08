-- Manual rollback only, AFTER reverting any application that reads these
-- objects (A3b+); never automatic. App/table owner, port 5432, ON_ERROR_STOP,
-- explicit BEGIN/COMMIT wrapper, SET LOCAL lock_timeout='5s' before including.
-- Refuses populated exceptions/shortage_flags tables by default: export and
-- verify them first, then SET LOCAL factory_ledger.confirm_exceptions_export = 'yes'.
-- The seed tables and the derived transactions.reason_code backfill are
-- dropped without a guard: rerunning 061 recreates them byte-for-byte from
-- adjust_reason/notes, which this rollback leaves untouched.
DO $$ BEGIN
    IF to_regclass('public.shortage_flags') IS NOT NULL THEN
        LOCK TABLE public.shortage_flags IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM public.shortage_flags)
           AND current_setting('factory_ledger.confirm_exceptions_export', true) IS DISTINCT FROM 'yes' THEN
            RAISE EXCEPTION '061 rollback refused: export and verify shortage_flags first';
        END IF;
    END IF;
    IF to_regclass('public.exceptions') IS NOT NULL THEN
        LOCK TABLE public.exceptions IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM public.exceptions)
           AND current_setting('factory_ledger.confirm_exceptions_export', true) IS DISTINCT FROM 'yes' THEN
            RAISE EXCEPTION '061 rollback refused: export and verify exceptions first';
        END IF;
    END IF;
END $$;
DROP TABLE IF EXISTS public.shortage_flags;
DROP TABLE IF EXISTS public.exceptions;
ALTER TABLE public.transactions DROP COLUMN IF EXISTS reason_code;
DROP TABLE IF EXISTS public.correction_reason_legacy_codes;
DROP TABLE IF EXISTS public.correction_reasons;
DELETE FROM public.migration_markers WHERE name = '061_exceptions_tables';

-- Manual rollback only, AFTER reverting the application that writes these
-- columns (A2+); never automatic. App/table owner, port 5432, ON_ERROR_STOP,
-- explicit BEGIN/COMMIT wrapper, SET LOCAL lock_timeout='5s' before including.
-- Dropping the columns discards the actor-id attribution recorded since 065;
-- operator_id (the name snapshot) stays, so no entry loses its person.
-- DROP COLUMN on transactions does not fire the append-only row trigger.
DROP INDEX IF EXISTS public.transactions_entered_by_actor_idx;
ALTER TABLE public.transactions DROP COLUMN IF EXISTS entered_by_actor_id;
ALTER TABLE public.ledger_corrections DROP COLUMN IF EXISTS entered_by_actor_id;
DELETE FROM public.migration_markers WHERE name = '065_entered_by';

-- Manual rollback only. Reverses migrations/056_actor_write_audit.sql.
-- Never run against production without separate owner approval.
--
-- SAFETY: dropping actor_write_audit permanently deletes named-actor
-- attribution for customer, lot and order edits. The guard locks the table
-- and refuses a non-empty table by default. Only after exporting its rows
-- and verifying the export, explicitly override the guard by executing:
--     SET LOCAL factory_ledger.confirm_audit_export = 'yes';
-- inside this script's transaction, after BEGIN and before the DO block.
-- The setting clears at COMMIT/ROLLBACK; no other value permits deletion.
-- Revert the backend first: while code that calls _record_actor_write() is
-- deployed, every named-actor metadata write fails closed (500 + rollback)
-- once this table is gone. Shared-key writes never touch the table.
-- Dropping the table also drops its index, append-only trigger and RLS flag.
-- DDL must go through port 5432 (direct/session), never the 6543 pooler.
BEGIN;

DO $$
BEGIN
    IF to_regclass('public.actor_write_audit') IS NOT NULL THEN
        LOCK TABLE public.actor_write_audit IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM public.actor_write_audit)
           AND current_setting('factory_ledger.confirm_audit_export', true)
               IS DISTINCT FROM 'yes' THEN
            RAISE EXCEPTION '056 rollback refused: actor_write_audit is not empty. Export and verify its rows before setting SET LOCAL factory_ledger.confirm_audit_export = ''yes'' in this transaction.';
        END IF;
    END IF;
END $$;

DROP TABLE IF EXISTS public.actor_write_audit;

DELETE FROM public.migration_markers WHERE name = '056_actor_write_audit';

COMMIT;

-- Manual rollback only. Reverses migrations/056_actor_write_audit.sql.
-- Never run against production without separate owner approval.
--
-- SAFETY: dropping actor_write_audit permanently deletes named-actor
-- attribution for customer, lot and order edits. Run ONLY while the table is
-- empty, or after its rows have been exported and the export verified:
--     SELECT count(*) FROM public.actor_write_audit;   -- must be 0, or exported
-- Revert the backend first: while code that calls _record_actor_write() is
-- deployed, every named-actor metadata write fails closed (500 + rollback)
-- once this table is gone. Shared-key writes never touch the table.
-- Dropping the table also drops its index, append-only trigger and RLS flag.
-- DDL must go through port 5432 (direct/session), never the 6543 pooler.
BEGIN;

DROP TABLE IF EXISTS public.actor_write_audit;

DELETE FROM public.migration_markers WHERE name = '056_actor_write_audit';

COMMIT;

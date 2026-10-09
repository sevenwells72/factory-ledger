-- Manual rollback only, AFTER reverting the application (A7+); never automatic.
-- App/table owner, port 5432, ON_ERROR_STOP, explicit BEGIN/COMMIT wrapper,
-- SET LOCAL lock_timeout='5s' before including this file.
-- Refuses while any order ticket exists: committed tickets are immutable
-- (058 trigger) and the action CHECK cannot be narrowed under them. Export
-- and verify the rows first, then SET LOCAL factory_ledger.confirm_ticket_export = 'yes'.
DO $$
DECLARE
    definition text;
BEGIN
    IF EXISTS (SELECT 1 FROM public.write_tickets WHERE action IN (
                 'create_order','add_order_lines','update_order_line','cancel_order_line',
                 'update_order_header','update_order_status','mark_order_ready',
                 'cancel_order','close_order','reopen_order'))
       AND current_setting('factory_ledger.confirm_ticket_export', true) IS DISTINCT FROM 'yes' THEN
        RAISE EXCEPTION '067 rollback refused: export and verify order ticket evidence first';
    END IF;
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NOT NULL AND position('create_order' in definition) > 0 THEN
        -- Restore the post-062 definition (058 actions + A5 move_lot).
        EXECUTE 'ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check';
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK ('
            || '(action IN (''receive'',''make'',''pack'',''adjust'',''found'')) OR action = ''move_lot'')';
    END IF;
END $$;
DROP INDEX IF EXISTS public.write_tickets_order_idx;
DROP INDEX IF EXISTS public.sales_orders_ticket_id_idx;
DROP INDEX IF EXISTS public.sales_order_lines_ticket_id_idx;
ALTER TABLE public.sales_orders DROP COLUMN IF EXISTS ticket_id;
ALTER TABLE public.sales_order_lines DROP COLUMN IF EXISTS ticket_id;
DELETE FROM public.migration_markers WHERE name = '067_order_tickets';

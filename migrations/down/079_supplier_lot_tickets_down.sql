-- Manual rollback AFTER reverting the app. Same transaction/timeouts as up.
-- Refuse ANY ticket of this action; never delete or detach audit/replay evidence.
DO $$
DECLARE definition text; restored text;
BEGIN
    LOCK TABLE public.write_tickets IN ACCESS EXCLUSIVE MODE;
    IF EXISTS (SELECT 1 FROM public.write_tickets WHERE action='update_supplier_lot') THEN
        RAISE EXCEPTION '079 rollback refused: supplier-lot tickets must be retained';
    END IF;
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF position('update_supplier_lot' in definition)>0 THEN
        restored := replace(definition, ' OR (action = ''update_supplier_lot''::text)', '');
        IF restored=definition OR position('update_supplier_lot' in restored)>0 THEN
            RAISE EXCEPTION '079 rollback refused: unexpected action constraint definition';
        END IF;
        ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check;
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check ' || restored;
    END IF;
END $$;
DELETE FROM migration_markers WHERE name='079_supplier_lot_tickets';

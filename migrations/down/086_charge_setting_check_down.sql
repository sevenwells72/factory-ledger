-- Roll back 086. Refuses once anyone has answered a check or confirmed an item:
-- those decisions are history. Open, unanswered checks are removed.
DO $$ DECLARE definition text; BEGIN
    IF EXISTS (SELECT 1 FROM public.exceptions WHERE kind='CHARGE_SETTING_CHECK' AND status NOT IN ('open','escalated')) THEN
        RAISE EXCEPTION 'Answered CHARGE_SETTING_CHECK exceptions exist; they are history, not rolled back';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name='products'
                 AND column_name='charge_setting_confirmed_at')
       AND EXISTS (SELECT 1 FROM public.products WHERE charge_setting_confirmed_at IS NOT NULL) THEN
        RAISE EXCEPTION 'Confirmed Charge/Product settings exist; they are history, not rolled back';
    END IF;
    DELETE FROM public.exceptions WHERE kind='CHARGE_SETTING_CHECK';
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.exceptions'::regclass AND conname='exceptions_kind_check';
    IF position('CHARGE_SETTING_CHECK' in definition)>0 THEN
        ALTER TABLE public.exceptions DROP CONSTRAINT exceptions_kind_check;
        EXECUTE 'ALTER TABLE public.exceptions ADD CONSTRAINT exceptions_kind_check '
            || replace(definition, ' OR (kind = ''CHARGE_SETTING_CHECK''::text)', '');
    END IF;
    IF EXISTS (SELECT 1 FROM public.actor_write_audit WHERE target_table='products') THEN
        RAISE EXCEPTION 'Catalog item writes are recorded in actor_write_audit; they are history, not rolled back';
    END IF;
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.actor_write_audit'::regclass AND conname='actor_write_audit_target_table_check';
    IF position('''products''' in definition)>0 THEN
        ALTER TABLE public.actor_write_audit DROP CONSTRAINT actor_write_audit_target_table_check;
        EXECUTE 'ALTER TABLE public.actor_write_audit ADD CONSTRAINT actor_write_audit_target_table_check '
            || replace(definition, ' OR (target_table = ''products''::text)', '');
    END IF;
END $$;
ALTER TABLE public.products DROP CONSTRAINT IF EXISTS products_batch_is_product_check;
-- Restore the legacy import behavior only after the history guards above pass.
ALTER TABLE public.products ALTER COLUMN is_service SET DEFAULT false;
DROP INDEX IF EXISTS public.exceptions_one_charge_check_per_product_idx;
DROP FUNCTION IF EXISTS public.charge_words_in_name(text);
ALTER TABLE public.products DROP COLUMN IF EXISTS charge_setting_confirmed_by_actor_id;
ALTER TABLE public.products DROP COLUMN IF EXISTS charge_setting_confirmed_at;
DELETE FROM public.migration_markers WHERE name='086_charge_setting_check';

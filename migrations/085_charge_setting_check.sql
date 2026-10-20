-- 085: Charge vs Product safety net (PR #143).
-- products.is_service stays the only rule for charge lines. This adds:
--   * products.charge_setting_confirmed_at / _by_actor_id — set when the owner
--     or office confirms an item's Charge/Product setting; a confirmed item is
--     never asked about again;
--   * exceptions kind CHARGE_SETTING_CHECK, at most one open per item;
--   * charge_words_in_name(text) — the same word list as charge_setting.py;
--   * one-time: open a check for every active, unconfirmed item whose name and
--     setting disagree (charge words but Product, or Charge without the words).
-- Nothing changes any item's setting. Apply before the code that reads the new
-- columns. Idempotent.
ALTER TABLE public.products ADD COLUMN IF NOT EXISTS charge_setting_confirmed_at timestamptz;
ALTER TABLE public.products ADD COLUMN IF NOT EXISTS charge_setting_confirmed_by_actor_id integer REFERENCES public.actors(id);

DO $$ DECLARE definition text; BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.exceptions'::regclass AND conname='exceptions_kind_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION 'exceptions kind constraint is missing';
    END IF;
    IF position('CHARGE_SETTING_CHECK' in definition)=0 THEN
        ALTER TABLE public.exceptions DROP CONSTRAINT exceptions_kind_check;
        EXECUTE 'ALTER TABLE public.exceptions ADD CONSTRAINT exceptions_kind_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR kind = ''CHARGE_SETTING_CHECK'')';
    END IF;
END $$;

-- Office catalog writes are attributed like other named-actor writes.
DO $$ DECLARE definition text; BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.actor_write_audit'::regclass AND conname='actor_write_audit_target_table_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION 'actor_write_audit target_table constraint is missing';
    END IF;
    IF position('''products''' in definition)=0 THEN
        ALTER TABLE public.actor_write_audit DROP CONSTRAINT actor_write_audit_target_table_check;
        EXECUTE 'ALTER TABLE public.actor_write_audit ADD CONSTRAINT actor_write_audit_target_table_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR target_table = ''products'')';
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS exceptions_one_charge_check_per_product_idx
    ON public.exceptions(product_id) WHERE kind='CHARGE_SETTING_CHECK' AND status IN ('open','escalated');

CREATE OR REPLACE FUNCTION public.charge_words_in_name(item_name text) RETURNS boolean
    LANGUAGE sql IMMUTABLE AS
$$ SELECT COALESCE(item_name, '') ~* '(fee|charge|delivery|pallet|freight|surcharge)' $$;

INSERT INTO public.exceptions(kind, severity, product_id, owner_actor_id, detail)
SELECT 'CHARGE_SETTING_CHECK', 'warn', p.id,
       (SELECT id FROM public.actors WHERE role='owner' AND active ORDER BY id LIMIT 1),
       jsonb_build_object('name', p.name,
                          'marked_as', CASE WHEN p.is_service THEN 'charge' ELSE 'product' END,
                          'reason', CASE WHEN p.is_service THEN 'charge_without_words' ELSE 'charge_words_marked_product' END,
                          'opened_by', '085_charge_setting_check')
FROM public.products p
WHERE COALESCE(p.active, true)
  AND p.charge_setting_confirmed_at IS NULL
  AND public.charge_words_in_name(p.name) <> p.is_service
  AND NOT EXISTS (SELECT 1 FROM public.exceptions e WHERE e.kind='CHARGE_SETTING_CHECK' AND e.product_id=p.id
                    AND e.status IN ('open','escalated'));

INSERT INTO public.migration_markers(name) VALUES ('085_charge_setting_check') ON CONFLICT DO NOTHING;

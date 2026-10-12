-- A7 review fixes. Apply AFTER 067, before the revised application.
-- App/table owner; explicit transaction; SET LOCAL lock_timeout='5s';
-- SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public.
-- Local tests only in this PR. No automatic production/staging apply.
-- Existing conflicting global references fail the index creation: reconcile
-- them explicitly before retrying, never silently pick or delete an order.
DO $$ BEGIN
    LOCK TABLE public.sales_orders, public.sales_order_create_receipts, public.write_tickets IN SHARE ROW EXCLUSIVE MODE;
    IF EXISTS (
        SELECT reference FROM (
            SELECT external_order_reference AS reference, id AS order_id FROM public.sales_orders
            WHERE external_order_reference IS NOT NULL
            UNION ALL
            SELECT external_order_reference, order_id FROM public.sales_order_create_receipts
            UNION ALL
            SELECT COALESCE(payload->>'external_order_reference',receipt_number), (result_ref->>'order_id')::bigint
            FROM public.write_tickets WHERE action='create_order' AND status='committed' AND receipt_number IS NOT NULL
        ) refs GROUP BY reference HAVING count(DISTINCT order_id)>1
    ) THEN
        RAISE EXCEPTION '072 refused: external references identify multiple orders; reconcile before retrying';
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS sales_orders_external_reference_uniq
    ON public.sales_orders(external_order_reference) WHERE external_order_reference IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS sales_order_receipts_reference_uniq
    ON public.sales_order_create_receipts(external_order_reference);
-- Original committed ticket evidence is immutable. Replay aliases deliberately
-- have no receipt_number and never compete with the original create receipt.
CREATE UNIQUE INDEX IF NOT EXISTS write_tickets_order_reference_uniq
    ON public.write_tickets((COALESCE(payload->>'external_order_reference', receipt_number)))
    WHERE action='create_order' AND status='committed' AND receipt_number IS NOT NULL;

-- Atomic per-day allocation. A prepare's savepoint can roll back its number,
-- but the row lock protects it against another prepare or a real commit.
CREATE TABLE IF NOT EXISTS public.sales_order_number_counters (
    order_day date PRIMARY KEY,
    last_number bigint NOT NULL CHECK (last_number > 0)
);
ALTER TABLE public.sales_order_number_counters ENABLE ROW LEVEL SECURITY;
CREATE OR REPLACE FUNCTION public.generate_order_number() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    today_prefix text := 'SO-' || to_char(CURRENT_DATE, 'YYMMDD');
    allocated bigint;
BEGIN
    INSERT INTO public.sales_order_number_counters(order_day, last_number)
    SELECT CURRENT_DATE, COALESCE(MAX(split_part(order_number, '-', 3)::bigint), 0) + 1
    FROM public.sales_orders WHERE order_number ~ ('^' || today_prefix || '-[0-9]+$')
    ON CONFLICT (order_day) DO UPDATE
        SET last_number = GREATEST(sales_order_number_counters.last_number + 1, EXCLUDED.last_number)
    RETURNING last_number INTO allocated;
    NEW.order_number := today_prefix || '-' || lpad(allocated::text, GREATEST(3, length(allocated::text)), '0');
    RETURN NEW;
END $$;

DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='public.write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NULL OR position('create_order' in definition)=0 THEN
        RAISE EXCEPTION '072 requires migration 067';
    END IF;
    IF position('create_expected_receipt' in definition)=0 THEN
        ALTER TABLE public.write_tickets DROP CONSTRAINT write_tickets_action_check;
        EXECUTE 'ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
            || substring(definition from 8 for length(definition)-8)
            || ') OR action IN (''create_expected_receipt'',''update_expected_receipt'',''cancel_expected_receipt''))';
    END IF;
END $$;
ALTER TABLE public.expected_receipts ADD COLUMN IF NOT EXISTS ticket_id bigint REFERENCES public.write_tickets(id);
CREATE INDEX IF NOT EXISTS expected_receipts_ticket_id_idx ON public.expected_receipts(ticket_id) WHERE ticket_id IS NOT NULL;
INSERT INTO public.migration_markers(name) VALUES ('072_order_ticket_review_fixes') ON CONFLICT (name) DO NOTHING;

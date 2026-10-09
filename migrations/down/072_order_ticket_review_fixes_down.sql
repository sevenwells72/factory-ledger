-- Stop revised A7 writes and revert its code first. Run before 067 down.
-- Explicit transaction as app/table owner, SET LOCAL lock_timeout='5s'.
-- Refuse any expected-receipt ticket evidence; archive it under an explicit
-- maintenance procedure before removing these actions (no silent deletion).
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM public.write_tickets WHERE action IN
        ('create_expected_receipt','update_expected_receipt','cancel_expected_receipt')) THEN
        RAISE EXCEPTION '072 rollback refused: expected-receipt ticket evidence exists';
    END IF;
END $$;
ALTER TABLE public.write_tickets DROP CONSTRAINT IF EXISTS write_tickets_action_check;
ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (action IN (
    'receive','make','pack','adjust','found','move_lot','create_order','add_order_lines',
    'update_order_line','cancel_order_line','update_order_header','update_order_status',
    'mark_order_ready','cancel_order','close_order','reopen_order'));
ALTER TABLE public.expected_receipts DROP COLUMN IF EXISTS ticket_id;
DROP INDEX IF EXISTS public.write_tickets_order_reference_uniq;
DROP INDEX IF EXISTS public.sales_order_receipts_reference_uniq;
DROP INDEX IF EXISTS public.sales_orders_external_reference_uniq;
CREATE OR REPLACE FUNCTION public.generate_order_number() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE today_prefix text; seq integer;
BEGIN
    today_prefix := 'SO-' || to_char(CURRENT_DATE, 'YYMMDD');
    SELECT COALESCE(MAX(CAST(SPLIT_PART(order_number, '-', 3) AS INTEGER)),0)+1 INTO seq
    FROM public.sales_orders WHERE order_number LIKE today_prefix || '-%';
    NEW.order_number := today_prefix || '-' || LPAD(seq::text,3,'0');
    RETURN NEW;
END $$;
DROP TABLE IF EXISTS public.sales_order_number_counters;
DELETE FROM public.migration_markers WHERE name='072_order_ticket_review_fixes';

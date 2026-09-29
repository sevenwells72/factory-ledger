-- Explicit rollback only, via port 5432 AFTER reverting the application.
-- Export receipts and new line/header metadata first if they must be retained.
-- No CASCADE; never drop ledger_current_* views.
DROP TABLE IF EXISTS public.sales_order_create_receipts;
DROP INDEX IF EXISTS public.sales_orders_customer_external_reference_uniq;
ALTER TABLE public.sales_orders DROP COLUMN IF EXISTS external_order_reference;
ALTER TABLE public.sales_order_lines
    DROP COLUMN IF EXISTS ordered_quantity,
    DROP COLUMN IF EXISTS ordered_unit,
    DROP COLUMN IF EXISTS ordered_case_weight_lb,
    DROP COLUMN IF EXISTS amount;

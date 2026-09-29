-- Explicit rollback only, AFTER reverting the application.
-- Run as the app's DB role (table owner), port 5432, with ON_ERROR_STOP:
--   \set ON_ERROR_STOP on
--   BEGIN;
--   SET LOCAL lock_timeout='5s';
--   \i migrations/down/057_order_create_contract_down.sql
--   COMMIT;
-- Export receipts and new line/header metadata first if they must be retained.
-- No CASCADE; never drop ledger_current_* views.
-- Dropping receipts also removes its reference-trimming CHECK.
DROP TABLE IF EXISTS public.sales_order_create_receipts;
DROP INDEX IF EXISTS public.sales_orders_customer_external_reference_uniq;
ALTER TABLE public.sales_orders DROP CONSTRAINT IF EXISTS sales_orders_external_reference_trimmed_check;
ALTER TABLE public.sales_orders DROP COLUMN IF EXISTS external_order_reference;
ALTER TABLE public.sales_order_lines
    DROP COLUMN IF EXISTS ordered_quantity,
    DROP COLUMN IF EXISTS ordered_unit,
    DROP COLUMN IF EXISTS ordered_case_weight_lb,
    DROP COLUMN IF EXISTS amount;

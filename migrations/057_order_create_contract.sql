-- Apply after 056, BEFORE deploying the order-create contract, via port 5432.
-- Additive only; no backfill, ledger view changes, or production data writes.
-- Rollback: stop contract writes, revert application, then run
-- migrations/down/057_order_create_contract_down.sql (new metadata is lost).
ALTER TABLE public.sales_orders
    ADD COLUMN IF NOT EXISTS external_order_reference text;

CREATE UNIQUE INDEX IF NOT EXISTS sales_orders_customer_external_reference_uniq
    ON public.sales_orders (customer_id, external_order_reference)
    WHERE external_order_reference IS NOT NULL;

-- NULL means a legacy line. Quantity is a count for services and the entered
-- quantity for physical goods; quantity_lb remains the inventory measure.
ALTER TABLE public.sales_order_lines
    ADD COLUMN IF NOT EXISTS ordered_quantity numeric,
    ADD COLUMN IF NOT EXISTS ordered_unit text,
    ADD COLUMN IF NOT EXISTS ordered_case_weight_lb numeric,
    ADD COLUMN IF NOT EXISTS amount numeric(18,2);

-- Keep the original response even after header/line edits. The original
-- customer/reference stays reserved if the order is moved to another customer.
CREATE TABLE IF NOT EXISTS public.sales_order_create_receipts (
    customer_id integer NOT NULL REFERENCES public.customers(id),
    external_order_reference text NOT NULL CHECK (btrim(external_order_reference) <> ''),
    order_id integer NOT NULL UNIQUE REFERENCES public.sales_orders(id),
    request_hash text NOT NULL CHECK (length(request_hash) = 64),
    response jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (customer_id, external_order_reference)
);

-- Match 056: API table owner can write; no public Supabase client policies.
ALTER TABLE public.sales_order_create_receipts ENABLE ROW LEVEL SECURITY;

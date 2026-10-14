-- Future orders only. Preserve the atomic per-day allocator from 072.
-- Apply explicitly in one transaction; no existing order or counter is rewritten.
CREATE OR REPLACE FUNCTION public.generate_order_number() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    plant_day date := (now() AT TIME ZONE 'America/New_York')::date;
    today_prefix text := 'SO-' || to_char(plant_day, 'YYMMDD');
    allocated bigint;
BEGIN
    INSERT INTO public.sales_order_number_counters(order_day, last_number)
    SELECT plant_day, COALESCE(MAX(split_part(order_number, '-', 3)::bigint), 0) + 1
    FROM public.sales_orders WHERE order_number ~ ('^' || today_prefix || '-[0-9]+$')
    ON CONFLICT (order_day) DO UPDATE
        SET last_number = GREATEST(sales_order_number_counters.last_number + 1, EXCLUDED.last_number)
    RETURNING last_number INTO allocated;
    NEW.order_number := today_prefix || '-' || lpad(allocated::text, GREATEST(3, length(allocated::text)), '0');
    RETURN NEW;
END $$;
ALTER TABLE public.sales_orders ALTER COLUMN order_date SET DEFAULT ((now() AT TIME ZONE 'America/New_York')::date);
INSERT INTO public.migration_markers(name) VALUES ('082_order_plant_date') ON CONFLICT (name) DO NOTHING;

-- Apply via port 5432. No transaction wrapper: follows 053/054 conventions.
-- Preserve legacy predicates, including production / finished_good helper filters.
-- Part B: today means business_date in America/New_York.
DO $migration$
DECLARE
    before_lb numeric;
    after_lb numeric;
    already boolean;
BEGIN
    CREATE TABLE IF NOT EXISTS public.migration_markers (
        name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    SELECT EXISTS (SELECT 1 FROM public.migration_markers WHERE name = '055_void_aware_views') INTO already;
    SELECT on_hand INTO before_lb FROM public.inventory_summary WHERE id = 128;

CREATE OR REPLACE VIEW public.inventory_summary AS
 SELECT p.id,
    p.name,
    p.type,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS on_hand
   FROM ((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
  WHERE (COALESCE(p.active, true) = true)
  GROUP BY p.id
  ORDER BY p.type, p.name;

CREATE OR REPLACE VIEW public.lot_balances AS
 SELECT l.id,
    l.lot_code,
    l.created_at,
    p.name AS product,
    p.type,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS balance
   FROM ((public.lots l
     JOIN public.products p ON ((p.id = l.product_id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
  GROUP BY l.id, p.id
 HAVING (COALESCE(sum(tl.quantity_lb), (0)::numeric) > (0)::numeric)
  ORDER BY l.created_at DESC;

CREATE OR REPLACE VIEW public.low_stock_alerts AS
 SELECT p.id,
    p.name,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS on_hand
   FROM ((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
  WHERE ((p.type = 'ingredient'::text) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id
 HAVING (COALESCE(sum(tl.quantity_lb), (0)::numeric) < (100)::numeric)
  ORDER BY COALESCE(sum(tl.quantity_lb), (0)::numeric);

CREATE OR REPLACE VIEW public.todays_transactions AS
 SELECT t.id,
    t.type,
    t."timestamp",
    t.notes,
    p.name AS product,
    l.lot_code,
    tl.quantity_lb::numeric(14,4) AS quantity_lb
   FROM (((public.ledger_current_transactions t
     JOIN public.ledger_current_transaction_lines tl ON ((tl.transaction_id = t.id)))
     JOIN public.lots l ON ((l.id = tl.lot_id)))
     JOIN public.products p ON ((p.id = tl.product_id)))
  WHERE t.effective_status = 'posted' AND (t.business_date = (now() AT TIME ZONE 'America/New_York')::date)
  ORDER BY t."timestamp" DESC;

CREATE OR REPLACE VIEW public.production_history AS
 SELECT t.id,
    t."timestamp",
    p.name AS product,
    l.lot_code,
    abs(sum(
        CASE
            WHEN (tl.quantity_lb > (0)::numeric) THEN tl.quantity_lb
            ELSE (0)::numeric
        END)) AS quantity_made
   FROM (((public.ledger_current_transactions t
     JOIN public.ledger_current_transaction_lines tl ON ((tl.transaction_id = t.id)))
     JOIN public.lots l ON ((l.id = tl.lot_id)))
     JOIN public.products p ON ((p.id = tl.product_id)))
  WHERE t.effective_status = 'posted' AND ((t.type = 'make'::text) AND (tl.quantity_lb > (0)::numeric))
  GROUP BY t.id, t."timestamp", p.id, l.id
  ORDER BY t."timestamp" DESC;

CREATE OR REPLACE VIEW public.v_lot_quantities AS
 SELECT l.id AS lot_id,
    l.lot_code,
    l.product_id,
    p.name AS product_name,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS quantity_on_hand,
    COALESCE(p.uom, 'lb'::text) AS uom
   FROM ((public.lots l
     JOIN public.products p ON ((p.id = l.product_id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
  GROUP BY l.id, l.lot_code, l.product_id, p.name, p.uom;

CREATE OR REPLACE VIEW public.v_batch_products_needing_setup AS
 SELECT p.id AS product_id,
    p.name AS product_name,
    p.product_category,
    p.production_context,
    p.verification_status,
    COALESCE(p.has_bom, false) AS has_bom,
    p.bom_status,
    p.customer_name,
    p.created_via,
    count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END) AS batch_count,
    COALESCE(sum(
        CASE
            WHEN ((t.type = 'production'::text) AND (tl.quantity_lb > (0)::numeric)) THEN tl.quantity_lb
            ELSE (0)::numeric
        END), (0)::numeric) AS total_produced
   FROM (((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.ledger_current_transactions t ON ((t.id = tl.transaction_id)))
  WHERE ((p.type = 'finished_good'::text) AND ((p.verification_status)::text = ANY (ARRAY[('unverified'::character varying)::text, ('incomplete'::character varying)::text])) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.verification_status, p.has_bom, p.bom_status, p.customer_name, p.created_via
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;

CREATE OR REPLACE VIEW public.v_products_missing_boms AS
 SELECT p.id AS product_id,
    p.name AS product_name,
    p.product_category,
    p.production_context,
    p.verification_status,
    p.customer_name,
    count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END) AS batch_count,
    COALESCE(sum(
        CASE
            WHEN ((t.type = 'production'::text) AND (tl.quantity_lb > (0)::numeric)) THEN tl.quantity_lb
            ELSE (0)::numeric
        END), (0)::numeric) AS total_produced,
    max(
        CASE
            WHEN (t.type = 'production'::text) THEN t."timestamp"
            ELSE NULL::timestamp without time zone
        END) AS last_produced
   FROM (((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.ledger_current_transactions t ON ((t.id = tl.transaction_id)))
  WHERE ((p.type = 'finished_good'::text) AND ((COALESCE(p.has_bom, false) = false) OR ((p.bom_status)::text = 'none'::text)) AND (COALESCE(p.active, true) = true) AND ((COALESCE(p.production_context, 'standard'::character varying))::text = 'standard'::text))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.verification_status, p.customer_name
 HAVING (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END) >= 1)
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;

CREATE OR REPLACE VIEW public.v_test_batches_for_review AS
 SELECT p.id AS product_id,
    p.name AS product_name,
    p.product_category,
    p.production_context,
    p.customer_name,
    count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END) AS batch_count,
    COALESCE(sum(
        CASE
            WHEN ((t.type = 'production'::text) AND (tl.quantity_lb > (0)::numeric)) THEN tl.quantity_lb
            ELSE (0)::numeric
        END), (0)::numeric) AS total_produced,
    max(
        CASE
            WHEN (t.type = 'production'::text) THEN t."timestamp"
            ELSE NULL::timestamp without time zone
        END) AS last_produced,
        CASE
            WHEN (count(DISTINCT
            CASE
                WHEN (t.type = 'production'::text) THEN t.id
                ELSE NULL::integer
            END) >= 3) THEN 'Consider promoting to standard'::text
            WHEN (count(DISTINCT
            CASE
                WHEN (t.type = 'production'::text) THEN t.id
                ELSE NULL::integer
            END) >= 1) THEN 'In testing'::text
            ELSE 'No batches yet'::text
        END AS recommendation
   FROM (((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN (SELECT cl.* FROM public.ledger_current_transaction_lines cl
         JOIN public.ledger_current_transactions ct ON ct.id = cl.transaction_id
         WHERE ct.effective_status = 'posted') tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.ledger_current_transactions t ON ((t.id = tl.transaction_id)))
  WHERE (((p.production_context)::text = ANY (ARRAY[('test_batch'::character varying)::text, ('sample'::character varying)::text, ('one_off'::character varying)::text])) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.customer_name
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;

    INSERT INTO public.migration_markers (name) VALUES ('055_void_aware_views') ON CONFLICT DO NOTHING;
    SELECT on_hand INTO after_lb FROM public.inventory_summary WHERE id = 128;
    RAISE NOTICE '055_void_aware_views: already applied = %, product 128 on_hand before = %, after = %', already, before_lb, after_lb;
END $migration$;

-- Manual rollback only, AFTER reverting the A3b application code; never
-- automatic. App/table owner, port 5432, ON_ERROR_STOP, explicit BEGIN/COMMIT
-- wrapper, SET LOCAL lock_timeout='5s' before including.
-- Refuses while any ticket is still 'awaiting_approval' (approve or reject
-- them first — they are the owner's decisions, not the migration's) and while
-- shortage or large-correction rows exist (export and verify them, then
-- SET LOCAL factory_ledger.confirm_exceptions_export = 'yes').
-- ROLLBACK ORDER: 069 before 065 or 061. The 069 view references
-- transactions.reason_code (061) and entered_by_actor_id (065); while it exists
-- those two DROP COLUMNs fail with DependentObjectsStillExist. This file puts
-- the view back in its 055 shape (two trailing columns removed). CREATE OR
-- REPLACE VIEW cannot drop columns, so the view is dropped CASCADE and it plus
-- its nine 055 dependents are recreated verbatim from the 2026-10-08 production
-- dump — dependent view OIDs change; nothing else does. The reason_code sweep
-- (§4) is history-only and is not reverted (061 down drops the column itself).
-- The widened actor_write_audit.target_table CHECK is kept: it is additive and
-- the audit rows written for /exceptions actions are append-only history.
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM public.write_tickets WHERE status = 'awaiting_approval') THEN
        RAISE EXCEPTION '069 rollback refused: tickets are still awaiting owner approval';
    END IF;
    IF (EXISTS (SELECT 1 FROM public.shortage_flags)
        OR EXISTS (SELECT 1 FROM public.exceptions WHERE kind IN ('SHORTAGE', 'LARGE_CORRECTION')))
       AND current_setting('factory_ledger.confirm_exceptions_export', true) IS DISTINCT FROM 'yes' THEN
        RAISE EXCEPTION '069 rollback refused: export and verify shortage_flags / exceptions first';
    END IF;
END $$;
DROP INDEX IF EXISTS public.exceptions_open_due_idx;
DROP INDEX IF EXISTS public.shortage_flags_open_due_idx;
DROP INDEX IF EXISTS public.shortage_flags_one_per_line_idx;
DROP INDEX IF EXISTS public.exceptions_one_shortage_per_line_idx;
DROP INDEX IF EXISTS public.exceptions_one_hold_per_ticket_idx;
DROP INDEX IF EXISTS public.write_tickets_awaiting_approval_idx;
ALTER TABLE public.write_tickets DROP CONSTRAINT IF EXISTS write_tickets_status_check;
ALTER TABLE public.write_tickets ADD CONSTRAINT write_tickets_status_check
    CHECK (status IN ('prepared', 'committed', 'expired', 'rejected', 'superseded'));
DELETE FROM public.migration_markers WHERE name = '069_exceptions_enforcement';

-- View back to the 055 shape (dependents recreated from tests/schema/schema.sql).
DROP VIEW IF EXISTS public.ledger_current_transactions CASCADE;
CREATE OR REPLACE VIEW public.ledger_current_transactions AS
 SELECT t.id,
    t.type,
    t."timestamp",
    t.notes,
    t.bol_reference,
    t.shipper_name,
    t.shipper_code,
    t.cases_received,
    t.case_size_lb,
    t.customer_name,
    t.order_reference,
    t.adjust_reason,
    t.adjust_reason_es,
    t.status,
    t.created_at,
    t.created_at_source,
    t.occurred_at,
    t.business_date,
    t.operator_id,
        CASE
            WHEN (correction.event_type = 'void'::text) THEN 'voided'::text
            WHEN (correction.event_type = 'restore'::text) THEN 'posted'::text
            WHEN (correction.event_type = 'amend'::text) THEN COALESCE((correction.replacement_values ->> 'status'::text), t.status, 'posted'::text)
            ELSE COALESCE(t.status, 'posted'::text)
        END AS effective_status,
    correction.id AS latest_correction_id,
    correction.event_type AS latest_correction_type,
    correction.created_at AS latest_correction_created_at,
    correction.operator_id AS latest_correction_operator_id,
    correction.replacement_values AS latest_replacement_values,
    ((to_jsonb(t.*) || COALESCE(correction.replacement_values, '{}'::jsonb)) || jsonb_build_object('status',
        CASE
            WHEN (correction.event_type = 'void'::text) THEN 'voided'::text
            WHEN (correction.event_type = 'restore'::text) THEN 'posted'::text
            WHEN (correction.event_type = 'amend'::text) THEN COALESCE((correction.replacement_values ->> 'status'::text), t.status, 'posted'::text)
            ELSE COALESCE(t.status, 'posted'::text)
        END)) AS effective_record
   FROM (public.transactions t
     LEFT JOIN LATERAL ( SELECT c.id,
            c.target_table,
            c.target_id,
            c.event_type,
            c.previous_values,
            c.replacement_values,
            c.reason,
            c.operator_id,
            c.created_at,
            c.created_at_source
           FROM public.ledger_corrections c
          WHERE ((c.target_table = 'transactions'::text) AND (c.target_id = t.id))
          ORDER BY c.created_at DESC, c.id DESC
         LIMIT 1) correction ON (true));

CREATE OR REPLACE VIEW public.inventory_summary AS
SELECT
    NULL::integer AS id,
    NULL::text AS name,
    NULL::text AS type,
    NULL::numeric AS on_hand;

CREATE OR REPLACE VIEW public.lot_balances AS
SELECT
    NULL::integer AS id,
    NULL::text AS lot_code,
    NULL::timestamp without time zone AS created_at,
    NULL::text AS product,
    NULL::text AS type,
    NULL::numeric AS balance;

CREATE OR REPLACE VIEW public.low_stock_alerts AS
SELECT
    NULL::integer AS id,
    NULL::text AS name,
    NULL::numeric AS on_hand;

CREATE OR REPLACE VIEW public.production_history AS
SELECT
    NULL::integer AS id,
    NULL::timestamp without time zone AS "timestamp",
    NULL::text AS product,
    NULL::text AS lot_code,
    NULL::numeric AS quantity_made;

CREATE OR REPLACE VIEW public.todays_transactions AS
 SELECT t.id,
    t.type,
    t."timestamp",
    t.notes,
    p.name AS product,
    l.lot_code,
    (tl.quantity_lb)::numeric(14,4) AS quantity_lb
   FROM (((public.ledger_current_transactions t
     JOIN public.ledger_current_transaction_lines tl ON ((tl.transaction_id = t.id)))
     JOIN public.lots l ON ((l.id = tl.lot_id)))
     JOIN public.products p ON ((p.id = tl.product_id)))
  WHERE ((t.effective_status = 'posted'::text) AND (t.business_date = ((now() AT TIME ZONE 'America/New_York'::text))::date))
  ORDER BY t."timestamp" DESC;

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
     LEFT JOIN ( SELECT cl.id,
            cl.transaction_id,
            cl.product_id,
            cl.lot_id,
            cl.quantity_lb,
            cl.created_at,
            cl.created_at_source,
            cl.latest_correction_id,
            cl.latest_correction_created_at,
            cl.latest_correction_operator_id,
            cl.effective_record
           FROM (public.ledger_current_transaction_lines cl
             JOIN public.ledger_current_transactions ct ON ((ct.id = cl.transaction_id)))
          WHERE (ct.effective_status = 'posted'::text)) tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.ledger_current_transactions t ON ((t.id = tl.transaction_id)))
  WHERE ((p.type = 'finished_good'::text) AND ((p.verification_status)::text = ANY (ARRAY[('unverified'::character varying)::text, ('incomplete'::character varying)::text])) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.verification_status, p.has_bom, p.bom_status, p.customer_name, p.created_via
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;

CREATE OR REPLACE VIEW public.v_lot_quantities AS
 SELECT l.id AS lot_id,
    l.lot_code,
    l.product_id,
    p.name AS product_name,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS quantity_on_hand,
    COALESCE(p.uom, 'lb'::text) AS uom
   FROM ((public.lots l
     JOIN public.products p ON ((p.id = l.product_id)))
     LEFT JOIN ( SELECT cl.id,
            cl.transaction_id,
            cl.product_id,
            cl.lot_id,
            cl.quantity_lb,
            cl.created_at,
            cl.created_at_source,
            cl.latest_correction_id,
            cl.latest_correction_created_at,
            cl.latest_correction_operator_id,
            cl.effective_record
           FROM (public.ledger_current_transaction_lines cl
             JOIN public.ledger_current_transactions ct ON ((ct.id = cl.transaction_id)))
          WHERE (ct.effective_status = 'posted'::text)) tl ON ((tl.lot_id = l.id)))
  GROUP BY l.id, l.lot_code, l.product_id, p.name, p.uom;

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
     LEFT JOIN ( SELECT cl.id,
            cl.transaction_id,
            cl.product_id,
            cl.lot_id,
            cl.quantity_lb,
            cl.created_at,
            cl.created_at_source,
            cl.latest_correction_id,
            cl.latest_correction_created_at,
            cl.latest_correction_operator_id,
            cl.effective_record
           FROM (public.ledger_current_transaction_lines cl
             JOIN public.ledger_current_transactions ct ON ((ct.id = cl.transaction_id)))
          WHERE (ct.effective_status = 'posted'::text)) tl ON ((tl.lot_id = l.id)))
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
     LEFT JOIN ( SELECT cl.id,
            cl.transaction_id,
            cl.product_id,
            cl.lot_id,
            cl.quantity_lb,
            cl.created_at,
            cl.created_at_source,
            cl.latest_correction_id,
            cl.latest_correction_created_at,
            cl.latest_correction_operator_id,
            cl.effective_record
           FROM (public.ledger_current_transaction_lines cl
             JOIN public.ledger_current_transactions ct ON ((ct.id = cl.transaction_id)))
          WHERE (ct.effective_status = 'posted'::text)) tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.ledger_current_transactions t ON ((t.id = tl.transaction_id)))
  WHERE (((p.production_context)::text = ANY (ARRAY[('test_batch'::character varying)::text, ('sample'::character varying)::text, ('one_off'::character varying)::text])) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.customer_name
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;

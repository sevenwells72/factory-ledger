--
-- PostgreSQL database dump
--

\restrict lLAWZcgkJFF1oXCgBXDD1uTvn4qEZYTsQa6SYb9objg1ubWzjefTNcURoaQtZPw

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.10 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA IF NOT EXISTS public;


--
-- Name: inventory_entry_source; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.inventory_entry_source AS ENUM (
    'received',
    'found_inventory',
    'transfer_in',
    'opening_balance',
    'production_output',
    'adjustment',
    'return'
);


--
-- Name: product_verification_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.product_verification_status AS ENUM (
    'verified',
    'incomplete',
    'unverified'
);


--
-- Name: production_context; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.production_context AS ENUM (
    'standard',
    'test_batch',
    'sample',
    'private_label',
    'one_off'
);


--
-- Name: generate_order_number(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.generate_order_number() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    today_prefix TEXT;
    seq INTEGER;
BEGIN
    today_prefix := 'SO-' || to_char(CURRENT_DATE, 'YYMMDD');
    SELECT COALESCE(MAX(
        CAST(SPLIT_PART(order_number, '-', 3) AS INTEGER)
    ), 0) + 1
    INTO seq
    FROM sales_orders
    WHERE order_number LIKE today_prefix || '-%';

    NEW.order_number := today_prefix || '-' || LPAD(seq::TEXT, 3, '0');
    RETURN NEW;
END;
$$;


--
-- Name: ledger_block_append_only_change(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.ledger_block_append_only_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; create a correction event instead', TG_TABLE_NAME
        USING ERRCODE = '23000';
END;
$$;


--
-- Name: ledger_enforce_created_at(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.ledger_enforce_created_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.created_at := clock_timestamp();
        IF TG_TABLE_NAME <> 'transactions'
           OR NEW.created_at_source IS DISTINCT FROM 'api_backfill' THEN
            NEW.created_at_source := 'database';
        END IF;
        IF TG_TABLE_NAME = 'transactions' THEN
            NEW.entry_backfilled := NEW.created_at_source = 'api_backfill';
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.created_at IS DISTINCT FROM OLD.created_at
       OR NEW.created_at_source IS DISTINCT FROM OLD.created_at_source THEN
        RAISE EXCEPTION 'created_at and created_at_source are immutable on %', TG_TABLE_NAME
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;


--
-- Name: ledger_fill_transaction_business_time(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.ledger_fill_transaction_business_time() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.occurred_at IS NULL THEN
        NEW.occurred_at := COALESCE(NEW."timestamp" AT TIME ZONE 'UTC', clock_timestamp());
    END IF;
    IF NEW.business_date IS NULL THEN
        NEW.business_date := timezone('America/New_York', NEW.occurred_at)::date;
    END IF;
    IF NEW.operator_id IS NULL OR btrim(NEW.operator_id) = '' THEN
        NEW.operator_id := COALESCE(NULLIF(current_setting('app.operator_id', true), ''), 'legacy-shared-key');
    END IF;
    RETURN NEW;
END;
$$;


--
-- Name: ledger_force_new_created_at(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.ledger_force_new_created_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.created_at := clock_timestamp();
    NEW.created_at_source := 'database';
    NEW.operator_id := COALESCE(NULLIF(current_setting('app.operator_id', true), ''), NEW.operator_id, 'legacy-shared-key');
    RETURN NEW;
END;
$$;


--
-- Name: quick_create_and_add_found_inventory(character varying, character varying, numeric, character varying, character varying, character varying, character varying, character varying, character varying, text, character varying); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.quick_create_and_add_found_inventory(p_product_name character varying, p_product_type character varying, p_quantity numeric, p_reason_code character varying, p_uom character varying DEFAULT 'lb'::character varying, p_storage_type character varying DEFAULT 'ambient'::character varying, p_found_location character varying DEFAULT NULL::character varying, p_estimated_age character varying DEFAULT 'unknown'::character varying, p_suspected_supplier character varying DEFAULT NULL::character varying, p_notes text DEFAULT NULL::text, p_performed_by character varying DEFAULT 'system'::character varying) RETURNS json
    LANGUAGE plpgsql
    AS $$
DECLARE
    v_product_id INTEGER;
    v_lot_result JSON;
BEGIN
    -- Create the product
    INSERT INTO products (
        name,
        type,
        uom,
        storage_type,
        verification_status,
        verification_notes,
        created_via
    ) VALUES (
        p_product_name,
        p_product_type,
        p_uom,
        p_storage_type,
        'unverified',
        'Quick-created during inventory count. ' || COALESCE(p_notes, ''),
        'quick_create_found_inventory'
    ) RETURNING id INTO v_product_id;
    
    -- Create audit record
    INSERT INTO product_verification_history (
        product_id,
        from_status, to_status,
        action, action_notes,
        performed_by
    ) VALUES (
        v_product_id,
        NULL, 'unverified',
        'created',
        'Quick-created during inventory count at ' || COALESCE(p_found_location, 'unknown location'),
        p_performed_by
    );
    
    -- Add found inventory
    SELECT add_found_inventory(
        v_product_id,
        p_quantity,
        p_uom,
        p_reason_code,
        p_found_location,
        p_estimated_age,
        p_suspected_supplier,
        NULL,
        p_notes,
        p_performed_by
    ) INTO v_lot_result;
    
    RETURN json_build_object(
        'success', true,
        'product_id', v_product_id,
        'product_name', p_product_name,
        'verification_status', 'unverified',
        'lot_id', v_lot_result->>'lot_id',
        'lot_code', v_lot_result->>'lot_code',
        'quantity', p_quantity
    );
END;
$$;


--
-- Name: supplier_name_norm(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.supplier_name_norm(text) RETURNS text
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
    AS $_$
    SELECT lower(
        regexp_replace(
            replace(replace(btrim($1), '’', ''''), '`', ''''),
            '\s+', ' ', 'g'
        )
    )
$_$;


--
-- Name: update_sales_order_timestamp(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.update_sales_order_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: _backup_20260305_batch_formulas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_batch_formulas (
    id integer,
    product_id integer,
    ingredient_product_id integer,
    quantity_lb numeric,
    exclude_from_inventory boolean
);


--
-- Name: _backup_20260305_ingredient_lot_consumption; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_ingredient_lot_consumption (
    id integer,
    transaction_id integer,
    ingredient_product_id integer,
    ingredient_lot_id integer,
    quantity_lb numeric(12,2),
    created_at timestamp without time zone
);


--
-- Name: _backup_20260305_lots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_lots (
    id integer,
    product_id integer,
    lot_code text,
    created_at timestamp without time zone,
    entry_source character varying(30),
    entry_source_notes text,
    estimated_age character varying(50),
    found_location character varying(200),
    entry_source_notes_es text,
    status text,
    merged_into_lot_id integer,
    merged_at timestamp with time zone,
    merge_reason text,
    supplier_lot_code text,
    lot_type text,
    received_at timestamp with time zone
);


--
-- Name: _backup_20260305_products; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_products (
    id integer,
    name text,
    type text,
    default_batch_lb integer,
    active boolean,
    odoo_code text,
    uom text,
    yield_25lb_cases integer,
    yield_10lb_cases integer,
    yield_retail_cases integer,
    yield_retail_bags integer,
    retail_bag_oz numeric(5,2),
    bags_per_case integer,
    brand text,
    verification_status character varying(20),
    verification_notes text,
    created_via character varying(50),
    verified_by character varying(100),
    verified_at timestamp with time zone,
    production_context character varying(20),
    customer_id uuid,
    customer_name character varying(200),
    has_bom boolean,
    bom_status character varying(50),
    product_category character varying(100),
    pack_size_lbs numeric(10,2),
    units_per_case integer,
    shelf_life_days integer,
    storage_type character varying(50),
    default_case_weight_lb numeric,
    verification_notes_es text,
    label_type text,
    yield_multiplier double precision,
    case_size_lb numeric(10,2)
);


--
-- Name: _backup_20260305_sales_order_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_sales_order_lines (
    id integer,
    sales_order_id integer,
    product_id integer,
    quantity_lb numeric(12,2),
    quantity_shipped_lb numeric(12,2),
    unit_price numeric(10,4),
    line_status text,
    notes text,
    created_at timestamp with time zone,
    notes_es text
);


--
-- Name: _backup_20260305_shipment_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_shipment_lines (
    id integer,
    shipment_id integer,
    transaction_id integer,
    sales_order_line_id integer,
    product_id integer,
    quantity_lb numeric
);


--
-- Name: _backup_20260305_shipments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_shipments (
    id integer,
    sales_order_id integer,
    customer_id integer,
    shipped_at timestamp with time zone
);


--
-- Name: _backup_20260305_transaction_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_transaction_lines (
    id integer,
    transaction_id integer,
    product_id integer,
    lot_id integer,
    quantity_lb numeric
);


--
-- Name: _backup_20260305_transactions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public._backup_20260305_transactions (
    id integer,
    type text,
    "timestamp" timestamp without time zone,
    notes text,
    bol_reference text,
    shipper_name text,
    shipper_code text,
    cases_received integer,
    case_size_lb numeric(10,2),
    customer_name text,
    order_reference text,
    adjust_reason text,
    adjust_reason_es text
);


--
-- Name: adjustment_reason_codes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.adjustment_reason_codes (
    code character varying(50) NOT NULL,
    description text NOT NULL,
    requires_location boolean DEFAULT false,
    adjustment_type character varying(50) NOT NULL,
    active boolean DEFAULT true
);


--
-- Name: allergens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.allergens (
    id integer NOT NULL,
    name text NOT NULL,
    display_order integer
);


--
-- Name: allergens_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.allergens_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: allergens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.allergens_id_seq OWNED BY public.allergens.id;


--
-- Name: batch_formulas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.batch_formulas (
    id integer NOT NULL,
    product_id integer NOT NULL,
    ingredient_product_id integer NOT NULL,
    quantity_lb numeric(14,4) NOT NULL,
    exclude_from_inventory boolean DEFAULT false
);


--
-- Name: batch_formulas_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.batch_formulas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: batch_formulas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.batch_formulas_id_seq OWNED BY public.batch_formulas.id;


--
-- Name: bom_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bom_lines (
    id integer NOT NULL,
    bom_id integer NOT NULL,
    ingredient_product_id integer NOT NULL,
    quantity_per_batch numeric(12,2),
    percentage numeric(5,2),
    uom character varying(20) DEFAULT 'lb'::character varying,
    sort_order integer DEFAULT 0
);


--
-- Name: bom_lines_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bom_lines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: bom_lines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bom_lines_id_seq OWNED BY public.bom_lines.id;


--
-- Name: boms; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.boms (
    id integer NOT NULL,
    product_id integer NOT NULL,
    version integer DEFAULT 1,
    status character varying(20) DEFAULT 'draft'::character varying,
    expected_yield_pct numeric(5,2),
    notes text,
    created_by character varying(100),
    created_from_batch_id integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: boms_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.boms_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: boms_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.boms_id_seq OWNED BY public.boms.id;


--
-- Name: certifications; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.certifications (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    business_date date NOT NULL,
    certified_at timestamp with time zone NOT NULL,
    operator_id text DEFAULT 'legacy-shared-key'::text NOT NULL,
    source_type text DEFAULT 'manual'::text NOT NULL,
    source_message_id text,
    notes text,
    supersedes_certification_id uuid,
    correction_reason text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    CONSTRAINT certification_correction_reason CHECK ((((supersedes_certification_id IS NULL) AND (correction_reason IS NULL)) OR ((supersedes_certification_id IS NOT NULL) AND (btrim(correction_reason) <> ''::text)))),
    CONSTRAINT certifications_source_type_check CHECK ((source_type = ANY (ARRAY['manual'::text, 'whatsapp_export'::text])))
);


--
-- Name: current_certifications; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.current_certifications AS
 SELECT DISTINCT ON (business_date) id AS certification_id,
    business_date,
    certified_at,
    operator_id,
    source_type,
    source_message_id,
    notes,
    supersedes_certification_id,
    correction_reason,
    created_at
   FROM public.certifications
  ORDER BY business_date, created_at DESC, id DESC;


--
-- Name: customer_aliases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.customer_aliases (
    id integer NOT NULL,
    customer_id integer NOT NULL,
    alias text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: customer_aliases_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.customer_aliases_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: customer_aliases_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.customer_aliases_id_seq OWNED BY public.customer_aliases.id;


--
-- Name: customers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.customers (
    id integer NOT NULL,
    name text NOT NULL,
    contact_name text,
    email text,
    phone text,
    address text,
    notes text,
    active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    notes_es text
);


--
-- Name: customers_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.customers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: customers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.customers_id_seq OWNED BY public.customers.id;


--
-- Name: expected_receipts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.expected_receipts (
    id integer NOT NULL,
    product_id integer NOT NULL,
    supplier_id integer NOT NULL,
    expected_qty numeric(14,4) NOT NULL,
    expected_date date,
    reference_number text,
    notes text,
    status text DEFAULT 'open'::text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_by text,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT expected_receipts_expected_qty_check CHECK ((expected_qty > (0)::numeric)),
    CONSTRAINT expected_receipts_status_check CHECK ((status = ANY (ARRAY['open'::text, 'closed'::text, 'cancelled'::text])))
);


--
-- Name: expected_receipts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.expected_receipts ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.expected_receipts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: ingredient_lot_consumption; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ingredient_lot_consumption (
    id integer NOT NULL,
    transaction_id integer,
    ingredient_product_id integer,
    ingredient_lot_id integer,
    quantity_lb numeric(14,4) NOT NULL,
    created_at timestamp without time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: ingredient_lot_consumption_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ingredient_lot_consumption_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ingredient_lot_consumption_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ingredient_lot_consumption_id_seq OWNED BY public.ingredient_lot_consumption.id;


--
-- Name: inventory_adjustments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.inventory_adjustments (
    id integer NOT NULL,
    lot_id integer,
    product_id integer NOT NULL,
    new_lot_code character varying(50),
    adjustment_type character varying(50) NOT NULL,
    quantity_before numeric(12,2),
    quantity_adjustment numeric(12,2) NOT NULL,
    quantity_after numeric(12,2) NOT NULL,
    uom character varying(20) NOT NULL,
    reason_code character varying(50) NOT NULL,
    reason_notes text,
    found_location character varying(200),
    estimated_age character varying(50),
    suspected_supplier character varying(200),
    suspected_receive_date date,
    adjusted_by character varying(100),
    adjusted_at timestamp with time zone DEFAULT now(),
    inventory_count_id integer,
    reason_notes_es text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: inventory_adjustments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.inventory_adjustments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: inventory_adjustments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.inventory_adjustments_id_seq OWNED BY public.inventory_adjustments.id;


--
-- Name: inventory_summary; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.inventory_summary AS
SELECT
    NULL::integer AS id,
    NULL::text AS name,
    NULL::text AS type,
    NULL::numeric AS on_hand;


--
-- Name: ledger_corrections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ledger_corrections (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    target_table text NOT NULL,
    target_id bigint NOT NULL,
    event_type text NOT NULL,
    previous_values jsonb NOT NULL,
    replacement_values jsonb NOT NULL,
    reason text NOT NULL,
    operator_id text DEFAULT 'legacy-shared-key'::text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    CONSTRAINT ledger_corrections_event_type_check CHECK ((event_type = ANY (ARRAY['amend'::text, 'void'::text, 'restore'::text]))),
    CONSTRAINT ledger_corrections_previous_values_check CHECK ((jsonb_typeof(previous_values) = 'object'::text)),
    CONSTRAINT ledger_corrections_reason_check CHECK ((btrim(reason) <> ''::text)),
    CONSTRAINT ledger_corrections_replacement_values_check CHECK ((jsonb_typeof(replacement_values) = 'object'::text)),
    CONSTRAINT ledger_corrections_supported_target CHECK ((target_table = ANY (ARRAY['transactions'::text, 'transaction_lines'::text])))
);


--
-- Name: transaction_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.transaction_lines (
    id integer NOT NULL,
    transaction_id integer NOT NULL,
    product_id integer NOT NULL,
    lot_id integer NOT NULL,
    quantity_lb numeric(14,4) NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: ledger_current_transaction_lines; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.ledger_current_transaction_lines AS
 SELECT tl.id,
    tl.transaction_id,
    COALESCE(((correction.replacement_values ->> 'product_id'::text))::integer, tl.product_id) AS product_id,
    COALESCE(((correction.replacement_values ->> 'lot_id'::text))::integer, tl.lot_id) AS lot_id,
    COALESCE(((correction.replacement_values ->> 'quantity_lb'::text))::numeric, tl.quantity_lb) AS quantity_lb,
    tl.created_at,
    tl.created_at_source,
    correction.id AS latest_correction_id,
    correction.created_at AS latest_correction_created_at,
    correction.operator_id AS latest_correction_operator_id,
    (to_jsonb(tl.*) || COALESCE(correction.replacement_values, '{}'::jsonb)) AS effective_record
   FROM (public.transaction_lines tl
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
          WHERE ((c.target_table = 'transaction_lines'::text) AND (c.target_id = tl.id))
          ORDER BY c.created_at DESC, c.id DESC
         LIMIT 1) correction ON (true));


--
-- Name: transactions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.transactions (
    id integer NOT NULL,
    type text NOT NULL,
    "timestamp" timestamp without time zone DEFAULT now(),
    notes text,
    bol_reference text,
    shipper_name text,
    shipper_code text,
    cases_received integer,
    case_size_lb numeric(10,2),
    customer_name text,
    order_reference text,
    adjust_reason text,
    adjust_reason_es text,
    status text DEFAULT 'posted'::text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    entry_backfilled boolean DEFAULT false NOT NULL,
    occurred_at timestamp with time zone NOT NULL,
    business_date date NOT NULL,
    operator_id text DEFAULT 'legacy-shared-key'::text NOT NULL,
    expected_receipt_id integer
);


--
-- Name: ledger_current_transactions; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.ledger_current_transactions AS
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


--
-- Name: line_capacity_modes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.line_capacity_modes (
    id integer NOT NULL,
    line_id integer NOT NULL,
    mode_name text NOT NULL,
    workers_required integer NOT NULL,
    batches_per_day integer,
    pallets_per_day integer,
    bags_per_day integer,
    pack_size_lb numeric,
    is_default boolean DEFAULT false,
    notes text
);


--
-- Name: line_capacity_modes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.line_capacity_modes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: line_capacity_modes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.line_capacity_modes_id_seq OWNED BY public.line_capacity_modes.id;


--
-- Name: lot_balances; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.lot_balances AS
SELECT
    NULL::integer AS id,
    NULL::text AS lot_code,
    NULL::timestamp without time zone AS created_at,
    NULL::text AS product,
    NULL::text AS type,
    NULL::numeric AS balance;


--
-- Name: lot_reassignments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lot_reassignments (
    id integer NOT NULL,
    lot_id integer NOT NULL,
    lot_code character varying(50) NOT NULL,
    from_product_id integer NOT NULL,
    from_product_name character varying(200) NOT NULL,
    to_product_id integer NOT NULL,
    to_product_name character varying(200) NOT NULL,
    quantity_affected numeric(12,2) NOT NULL,
    uom character varying(20) NOT NULL,
    reason_code character varying(50) NOT NULL,
    reason_notes text,
    reassigned_by character varying(100),
    reassigned_at timestamp with time zone DEFAULT now(),
    original_receive_id integer,
    reason_notes_es text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: lot_reassignments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.lot_reassignments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: lot_reassignments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.lot_reassignments_id_seq OWNED BY public.lot_reassignments.id;


--
-- Name: lot_supplier_codes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lot_supplier_codes (
    id integer NOT NULL,
    lot_id integer NOT NULL,
    supplier_lot_code text,
    supplier_name text,
    quantity_lb numeric,
    notes text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: lot_supplier_codes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.lot_supplier_codes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: lot_supplier_codes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.lot_supplier_codes_id_seq OWNED BY public.lot_supplier_codes.id;


--
-- Name: lots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lots (
    id integer NOT NULL,
    product_id integer NOT NULL,
    lot_code text NOT NULL,
    created_at timestamp without time zone DEFAULT clock_timestamp() NOT NULL,
    entry_source character varying(30) DEFAULT 'received'::character varying,
    entry_source_notes text,
    estimated_age character varying(50),
    found_location character varying(200),
    entry_source_notes_es text,
    status text DEFAULT 'active'::text,
    merged_into_lot_id integer,
    merged_at timestamp with time zone,
    merge_reason text,
    supplier_lot_code text,
    lot_type text,
    received_at timestamp with time zone,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: lots_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.lots_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: lots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.lots_id_seq OWNED BY public.lots.id;


--
-- Name: low_stock_alerts; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.low_stock_alerts AS
SELECT
    NULL::integer AS id,
    NULL::text AS name,
    NULL::numeric AS on_hand;


--
-- Name: notes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.notes (
    id integer NOT NULL,
    category text NOT NULL,
    title text NOT NULL,
    body text DEFAULT ''::text,
    priority text DEFAULT 'normal'::text NOT NULL,
    status text DEFAULT 'open'::text NOT NULL,
    due_date date,
    entity_type text,
    entity_id text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT notes_category_check CHECK ((category = ANY (ARRAY['note'::text, 'todo'::text, 'reminder'::text]))),
    CONSTRAINT notes_priority_check CHECK ((priority = ANY (ARRAY['low'::text, 'normal'::text, 'high'::text]))),
    CONSTRAINT notes_status_check CHECK ((status = ANY (ARRAY['open'::text, 'done'::text, 'dismissed'::text])))
);


--
-- Name: notes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.notes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: notes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.notes_id_seq OWNED BY public.notes.id;


--
-- Name: oauth_tokens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.oauth_tokens (
    id integer NOT NULL,
    email text NOT NULL,
    access_token text NOT NULL,
    refresh_token text NOT NULL,
    token_expiry timestamp without time zone NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


--
-- Name: oauth_tokens_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.oauth_tokens_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: oauth_tokens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.oauth_tokens_id_seq OWNED BY public.oauth_tokens.id;


--
-- Name: open_orders_summary; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.open_orders_summary AS
SELECT
    NULL::integer AS id,
    NULL::text AS order_number,
    NULL::text AS customer,
    NULL::date AS order_date,
    NULL::date AS requested_ship_date,
    NULL::text AS status,
    NULL::bigint AS line_count,
    NULL::numeric AS total_lb_ordered,
    NULL::numeric AS total_lb_shipped,
    NULL::numeric AS remaining_lb,
    NULL::boolean AS overdue;


--
-- Name: products; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.products (
    id integer NOT NULL,
    name text NOT NULL,
    type text DEFAULT 'ingredient'::text NOT NULL,
    default_batch_lb numeric(14,4),
    active boolean,
    odoo_code text,
    uom text,
    yield_25lb_cases integer,
    yield_10lb_cases integer,
    yield_retail_cases integer,
    yield_retail_bags integer,
    retail_bag_oz numeric(5,2),
    bags_per_case integer,
    brand text,
    verification_status character varying(20) DEFAULT 'verified'::character varying,
    verification_notes text,
    created_via character varying(50),
    verified_by character varying(100),
    verified_at timestamp with time zone,
    production_context character varying(20) DEFAULT 'standard'::character varying,
    customer_id uuid,
    customer_name character varying(200),
    has_bom boolean DEFAULT false,
    bom_status character varying(50) DEFAULT 'none'::character varying,
    product_category character varying(100),
    pack_size_lbs numeric(10,2),
    units_per_case integer,
    shelf_life_days integer,
    storage_type character varying(50) DEFAULT 'ambient'::character varying,
    default_case_weight_lb numeric,
    verification_notes_es text,
    label_type text DEFAULT 'house'::text,
    yield_multiplier double precision DEFAULT 1.0,
    case_size_lb numeric(14,4),
    parent_batch_product_id integer,
    is_service boolean DEFAULT false NOT NULL,
    is_copack boolean DEFAULT false NOT NULL,
    no_production boolean DEFAULT false NOT NULL,
    pack_format text,
    low_stock_threshold numeric,
    CONSTRAINT products_low_stock_threshold_check CHECK (((low_stock_threshold IS NULL) OR (low_stock_threshold >= (0)::numeric))),
    CONSTRAINT products_pack_format_check CHECK (((pack_format IS NULL) OR (pack_format = ANY (ARRAY['10lb'::text, '25lb'::text, 'bagged'::text])))),
    CONSTRAINT products_type_check CHECK ((type = ANY (ARRAY['ingredient'::text, 'packaging'::text, 'batch'::text, 'finished'::text, 'consumable'::text])))
);


--
-- Name: sales_order_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sales_order_lines (
    id integer NOT NULL,
    sales_order_id integer NOT NULL,
    product_id integer NOT NULL,
    quantity_lb numeric(14,4) NOT NULL,
    quantity_shipped_lb numeric(14,4) DEFAULT 0 NOT NULL,
    unit_price numeric(10,4),
    line_status text DEFAULT 'pending'::text NOT NULL,
    notes text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    notes_es text,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    CONSTRAINT sales_order_lines_line_status_check CHECK ((line_status = ANY (ARRAY['pending'::text, 'partial'::text, 'fulfilled'::text, 'cancelled'::text]))),
    CONSTRAINT sales_order_lines_quantity_lb_check CHECK ((quantity_lb >= (0)::numeric)),
    CONSTRAINT sales_order_lines_quantity_shipped_lb_check CHECK ((quantity_shipped_lb >= (0)::numeric))
);


--
-- Name: sales_orders; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sales_orders (
    id integer NOT NULL,
    order_number text NOT NULL,
    customer_id integer NOT NULL,
    order_date date DEFAULT CURRENT_DATE NOT NULL,
    requested_ship_date date,
    status text DEFAULT 'new'::text NOT NULL,
    notes text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT now(),
    notes_es text,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    CONSTRAINT sales_orders_status_check CHECK ((status = ANY (ARRAY['new'::text, 'confirmed'::text, 'in_production'::text, 'ready'::text, 'shipped'::text, 'partial_ship'::text, 'invoiced'::text, 'cancelled'::text])))
);


--
-- Name: order_line_details; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.order_line_details AS
 SELECT so.id AS order_id,
    so.order_number,
    c.name AS customer,
    so.status AS order_status,
    so.requested_ship_date,
    sol.id AS line_id,
    p.name AS product,
    sol.quantity_lb,
    sol.quantity_shipped_lb,
    (sol.quantity_lb - sol.quantity_shipped_lb) AS remaining_lb,
    sol.unit_price,
    sol.line_status
   FROM (((public.sales_orders so
     JOIN public.customers c ON ((c.id = so.customer_id)))
     JOIN public.sales_order_lines sol ON ((sol.sales_order_id = so.id)))
     JOIN public.products p ON ((p.id = sol.product_id)))
  ORDER BY so.order_number, sol.id;


--
-- Name: product_allergens; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.product_allergens (
    id integer NOT NULL,
    product_id integer NOT NULL,
    allergen_id integer NOT NULL,
    notes text
);


--
-- Name: product_allergens_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.product_allergens_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: product_allergens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.product_allergens_id_seq OWNED BY public.product_allergens.id;


--
-- Name: product_bom; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.product_bom (
    id integer NOT NULL,
    finished_product_id integer NOT NULL,
    component_product_id integer NOT NULL,
    quantity numeric(10,4) NOT NULL,
    uom text DEFAULT 'lb'::text,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: product_bom_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.product_bom_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: product_bom_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.product_bom_id_seq OWNED BY public.product_bom.id;


--
-- Name: product_line_assignments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.product_line_assignments (
    id integer NOT NULL,
    product_id integer NOT NULL,
    line_id integer NOT NULL
);


--
-- Name: product_line_assignments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.product_line_assignments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: product_line_assignments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.product_line_assignments_id_seq OWNED BY public.product_line_assignments.id;


--
-- Name: product_verification_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.product_verification_history (
    id integer NOT NULL,
    product_id integer NOT NULL,
    from_status character varying(20),
    to_status character varying(20) NOT NULL,
    action character varying(50) NOT NULL,
    action_notes text,
    old_name character varying(200),
    new_name character varying(200),
    merged_into_product_id integer,
    performed_by character varying(100),
    performed_at timestamp with time zone DEFAULT now(),
    action_notes_es text
);


--
-- Name: product_verification_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.product_verification_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: product_verification_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.product_verification_history_id_seq OWNED BY public.product_verification_history.id;


--
-- Name: production_history; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.production_history AS
SELECT
    NULL::integer AS id,
    NULL::timestamp without time zone AS "timestamp",
    NULL::text AS product,
    NULL::text AS lot_code,
    NULL::numeric AS quantity_made;


--
-- Name: production_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.production_lines (
    id integer NOT NULL,
    name text NOT NULL,
    line_code text NOT NULL,
    active boolean DEFAULT true
);


--
-- Name: production_lines_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.production_lines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: production_lines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.production_lines_id_seq OWNED BY public.production_lines.id;


--
-- Name: production_schedule; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.production_schedule (
    id integer NOT NULL,
    schedule_date date NOT NULL,
    line_id integer NOT NULL,
    product_id integer NOT NULL,
    planned_batches integer,
    planned_quantity_lb numeric,
    planned_bags integer,
    workers_assigned integer NOT NULL,
    status text DEFAULT 'planned'::text,
    linked_order_numbers text[],
    overproduction_lb numeric DEFAULT 0,
    overproduction_reason text,
    notes text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    confirmed_at timestamp with time zone,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    CONSTRAINT production_schedule_status_check CHECK ((status = ANY (ARRAY['planned'::text, 'confirmed'::text, 'in_progress'::text, 'completed'::text, 'cancelled'::text])))
);


--
-- Name: production_schedule_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.production_schedule_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: production_schedule_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.production_schedule_id_seq OWNED BY public.production_schedule.id;


--
-- Name: products_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.products_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: products_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.products_id_seq OWNED BY public.products.id;


--
-- Name: reassignment_reason_codes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.reassignment_reason_codes (
    code character varying(50) NOT NULL,
    description text NOT NULL,
    active boolean DEFAULT true
);


--
-- Name: sales_order_allocation_reactivations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sales_order_allocation_reactivations (
    transaction_id integer NOT NULL,
    sales_order_line_id integer NOT NULL,
    quantity_lb numeric(14,4) NOT NULL,
    correction_id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT soar_quantity_lb_check CHECK ((quantity_lb >= (0)::numeric))
);


--
-- Name: sales_order_allocations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sales_order_allocations (
    id bigint NOT NULL,
    sales_order_id integer NOT NULL,
    sales_order_line_id integer NOT NULL,
    product_id integer NOT NULL,
    lot_id integer,
    quantity_lb numeric(14,4) NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    source text NOT NULL,
    ship_transaction_id integer,
    last_ship_transaction_id integer,
    split_from_id bigint,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    created_by text,
    released_at timestamp with time zone,
    released_by text,
    release_reason text,
    expires_at timestamp with time zone,
    note text,
    CONSTRAINT allocations_released_has_time CHECK (((status <> 'released'::text) OR (released_at IS NOT NULL))),
    CONSTRAINT allocations_shipped_has_txn CHECK (((status = 'shipped'::text) = (ship_transaction_id IS NOT NULL))),
    CONSTRAINT soa_quantity_lb_check CHECK ((quantity_lb > (0)::numeric)),
    CONSTRAINT soa_source_check CHECK ((source = ANY (ARRAY['manual'::text, 'auto_fifo'::text, 'staged_lot'::text]))),
    CONSTRAINT soa_status_check CHECK ((status = ANY (ARRAY['active'::text, 'released'::text, 'shipped'::text, 'superseded'::text])))
);


--
-- Name: sales_order_allocations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.sales_order_allocations ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.sales_order_allocations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: sales_order_flags; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sales_order_flags (
    so_number text NOT NULL,
    ready boolean DEFAULT false NOT NULL,
    ready_at timestamp with time zone,
    ready_by text DEFAULT 'floor'::text,
    note text,
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: sales_order_lines_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sales_order_lines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sales_order_lines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sales_order_lines_id_seq OWNED BY public.sales_order_lines.id;


--
-- Name: sales_order_shipments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sales_order_shipments (
    id integer NOT NULL,
    sales_order_line_id integer NOT NULL,
    transaction_id integer NOT NULL,
    quantity_lb numeric(12,2) NOT NULL,
    shipped_at timestamp with time zone DEFAULT now(),
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL,
    CONSTRAINT sales_order_shipments_quantity_lb_check CHECK ((quantity_lb > (0)::numeric))
);


--
-- Name: sales_order_shipments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sales_order_shipments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sales_order_shipments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sales_order_shipments_id_seq OWNED BY public.sales_order_shipments.id;


--
-- Name: sales_orders_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sales_orders_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sales_orders_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sales_orders_id_seq OWNED BY public.sales_orders.id;


--
-- Name: scheduling_config; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scheduling_config (
    key text NOT NULL,
    value jsonb NOT NULL,
    description text
);


--
-- Name: shipment_lines; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.shipment_lines (
    id integer NOT NULL,
    shipment_id integer NOT NULL,
    transaction_id integer NOT NULL,
    sales_order_line_id integer,
    product_id integer NOT NULL,
    quantity_lb numeric NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: shipment_lines_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.shipment_lines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: shipment_lines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.shipment_lines_id_seq OWNED BY public.shipment_lines.id;


--
-- Name: shipments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.shipments (
    id integer NOT NULL,
    sales_order_id integer,
    customer_id integer,
    shipped_at timestamp with time zone NOT NULL,
    transaction_id integer,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    created_at_source text DEFAULT 'database'::text NOT NULL
);


--
-- Name: shipments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.shipments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: shipments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.shipments_id_seq OWNED BY public.shipments.id;


--
-- Name: suppliers; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.suppliers (
    id integer NOT NULL,
    name text NOT NULL,
    active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL
);


--
-- Name: suppliers_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.suppliers ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.suppliers_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: supply_requests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.supply_requests (
    id integer NOT NULL,
    product_id integer,
    item_text text,
    qty numeric,
    note text,
    requested_by text NOT NULL,
    status text DEFAULT 'open'::text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    done_at timestamp with time zone,
    CONSTRAINT supply_requests_done_at_check CHECK (((status = 'done'::text) = (done_at IS NOT NULL))),
    CONSTRAINT supply_requests_item_text_nonblank_check CHECK (((item_text IS NULL) OR (btrim(item_text) <> ''::text))),
    CONSTRAINT supply_requests_qty_check CHECK (((qty IS NULL) OR (qty > (0)::numeric))),
    CONSTRAINT supply_requests_status_check CHECK ((status = ANY (ARRAY['open'::text, 'done'::text]))),
    CONSTRAINT supply_requests_target_xor_check CHECK (((product_id IS NULL) <> (item_text IS NULL)))
);


--
-- Name: supply_requests_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.supply_requests ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.supply_requests_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: todays_transactions; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.todays_transactions AS
 SELECT t.id,
    t.type,
    t."timestamp",
    t.notes,
    p.name AS product,
    l.lot_code,
    tl.quantity_lb
   FROM (((public.transactions t
     JOIN public.transaction_lines tl ON ((tl.transaction_id = t.id)))
     JOIN public.lots l ON ((l.id = tl.lot_id)))
     JOIN public.products p ON ((p.id = tl.product_id)))
  WHERE ((t."timestamp")::date = CURRENT_DATE)
  ORDER BY t."timestamp" DESC;


--
-- Name: transaction_lines_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.transaction_lines_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: transaction_lines_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.transaction_lines_id_seq OWNED BY public.transaction_lines.id;


--
-- Name: transactions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.transactions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: transactions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.transactions_id_seq OWNED BY public.transactions.id;


--
-- Name: v_batch_ingredients; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_batch_ingredients AS
 SELECT b.id AS batch_id,
    b.name AS batch_name,
    b.odoo_code AS batch_ref,
    b.default_batch_lb AS batch_weight,
    i.id AS ingredient_id,
    i.name AS ingredient_name,
    i.odoo_code AS ingredient_ref,
    bf.quantity_lb
   FROM ((public.products b
     JOIN public.batch_formulas bf ON ((bf.product_id = b.id)))
     JOIN public.products i ON ((i.id = bf.ingredient_product_id)))
  WHERE (b.type = 'batch'::text)
  ORDER BY b.name, i.name;


--
-- Name: v_batch_products_needing_setup; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_batch_products_needing_setup AS
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
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.transactions t ON ((t.id = tl.transaction_id)))
  WHERE ((p.type = 'finished_good'::text) AND ((p.verification_status)::text = ANY (ARRAY[('unverified'::character varying)::text, ('incomplete'::character varying)::text])) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.verification_status, p.has_bom, p.bom_status, p.customer_name, p.created_via
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;


--
-- Name: v_batch_summary; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_batch_summary AS
 SELECT id,
    name,
    odoo_code AS internal_ref,
    default_batch_lb AS batch_weight,
    yield_25lb_cases,
    yield_10lb_cases,
    yield_retail_cases,
    yield_retail_bags,
    retail_bag_oz,
    bags_per_case,
    brand,
    active
   FROM public.products p
  WHERE (type = 'batch'::text)
  ORDER BY odoo_code;


--
-- Name: v_finished_product_bom; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_finished_product_bom AS
 SELECT fp.id AS finished_product_id,
    fp.name AS finished_product_name,
    fp.odoo_code AS finished_ref,
    cp.id AS component_id,
    cp.name AS component_name,
    cp.odoo_code AS component_ref,
    cp.type AS component_type,
    pb.quantity,
    pb.uom
   FROM ((public.products fp
     JOIN public.product_bom pb ON ((pb.finished_product_id = fp.id)))
     JOIN public.products cp ON ((cp.id = pb.component_product_id)))
  WHERE (fp.type = 'finished'::text)
  ORDER BY fp.name, cp.type, cp.name;


--
-- Name: v_lot_quantities; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_lot_quantities AS
 SELECT l.id AS lot_id,
    l.lot_code,
    l.product_id,
    p.name AS product_name,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS quantity_on_hand,
    COALESCE(p.uom, 'lb'::text) AS uom
   FROM ((public.lots l
     JOIN public.products p ON ((p.id = l.product_id)))
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
  GROUP BY l.id, l.lot_code, l.product_id, p.name, p.uom;


--
-- Name: v_products_missing_boms; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_products_missing_boms AS
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
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.transactions t ON ((t.id = tl.transaction_id)))
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


--
-- Name: v_recent_lot_reassignments; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_recent_lot_reassignments AS
 SELECT lr.id,
    lr.lot_code,
    lr.from_product_name,
    lr.to_product_name,
    lr.quantity_affected,
    lr.uom,
    lr.reason_code,
    rc.description AS reason_description,
    lr.reason_notes,
    lr.reassigned_by,
    lr.reassigned_at
   FROM (public.lot_reassignments lr
     LEFT JOIN public.reassignment_reason_codes rc ON (((rc.code)::text = (lr.reason_code)::text)))
  ORDER BY lr.reassigned_at DESC;


--
-- Name: v_test_batches_for_review; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.v_test_batches_for_review AS
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
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
     LEFT JOIN public.transactions t ON ((t.id = tl.transaction_id)))
  WHERE (((p.production_context)::text = ANY (ARRAY[('test_batch'::character varying)::text, ('sample'::character varying)::text, ('one_off'::character varying)::text])) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id, p.name, p.product_category, p.production_context, p.customer_name
  ORDER BY (count(DISTINCT
        CASE
            WHEN (t.type = 'production'::text) THEN t.id
            ELSE NULL::integer
        END)) DESC;


--
-- Name: allergens id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.allergens ALTER COLUMN id SET DEFAULT nextval('public.allergens_id_seq'::regclass);


--
-- Name: batch_formulas id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.batch_formulas ALTER COLUMN id SET DEFAULT nextval('public.batch_formulas_id_seq'::regclass);


--
-- Name: bom_lines id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bom_lines ALTER COLUMN id SET DEFAULT nextval('public.bom_lines_id_seq'::regclass);


--
-- Name: boms id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.boms ALTER COLUMN id SET DEFAULT nextval('public.boms_id_seq'::regclass);


--
-- Name: customer_aliases id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.customer_aliases ALTER COLUMN id SET DEFAULT nextval('public.customer_aliases_id_seq'::regclass);


--
-- Name: customers id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.customers ALTER COLUMN id SET DEFAULT nextval('public.customers_id_seq'::regclass);


--
-- Name: ingredient_lot_consumption id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingredient_lot_consumption ALTER COLUMN id SET DEFAULT nextval('public.ingredient_lot_consumption_id_seq'::regclass);


--
-- Name: inventory_adjustments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inventory_adjustments ALTER COLUMN id SET DEFAULT nextval('public.inventory_adjustments_id_seq'::regclass);


--
-- Name: line_capacity_modes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.line_capacity_modes ALTER COLUMN id SET DEFAULT nextval('public.line_capacity_modes_id_seq'::regclass);


--
-- Name: lot_reassignments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lot_reassignments ALTER COLUMN id SET DEFAULT nextval('public.lot_reassignments_id_seq'::regclass);


--
-- Name: lot_supplier_codes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lot_supplier_codes ALTER COLUMN id SET DEFAULT nextval('public.lot_supplier_codes_id_seq'::regclass);


--
-- Name: lots id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lots ALTER COLUMN id SET DEFAULT nextval('public.lots_id_seq'::regclass);


--
-- Name: notes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.notes ALTER COLUMN id SET DEFAULT nextval('public.notes_id_seq'::regclass);


--
-- Name: oauth_tokens id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.oauth_tokens ALTER COLUMN id SET DEFAULT nextval('public.oauth_tokens_id_seq'::regclass);


--
-- Name: product_allergens id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_allergens ALTER COLUMN id SET DEFAULT nextval('public.product_allergens_id_seq'::regclass);


--
-- Name: product_bom id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_bom ALTER COLUMN id SET DEFAULT nextval('public.product_bom_id_seq'::regclass);


--
-- Name: product_line_assignments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_line_assignments ALTER COLUMN id SET DEFAULT nextval('public.product_line_assignments_id_seq'::regclass);


--
-- Name: product_verification_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_verification_history ALTER COLUMN id SET DEFAULT nextval('public.product_verification_history_id_seq'::regclass);


--
-- Name: production_lines id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_lines ALTER COLUMN id SET DEFAULT nextval('public.production_lines_id_seq'::regclass);


--
-- Name: production_schedule id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_schedule ALTER COLUMN id SET DEFAULT nextval('public.production_schedule_id_seq'::regclass);


--
-- Name: products id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.products ALTER COLUMN id SET DEFAULT nextval('public.products_id_seq'::regclass);


--
-- Name: sales_order_lines id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_lines ALTER COLUMN id SET DEFAULT nextval('public.sales_order_lines_id_seq'::regclass);


--
-- Name: sales_order_shipments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_shipments ALTER COLUMN id SET DEFAULT nextval('public.sales_order_shipments_id_seq'::regclass);


--
-- Name: sales_orders id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_orders ALTER COLUMN id SET DEFAULT nextval('public.sales_orders_id_seq'::regclass);


--
-- Name: shipment_lines id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipment_lines ALTER COLUMN id SET DEFAULT nextval('public.shipment_lines_id_seq'::regclass);


--
-- Name: shipments id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipments ALTER COLUMN id SET DEFAULT nextval('public.shipments_id_seq'::regclass);


--
-- Name: transaction_lines id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transaction_lines ALTER COLUMN id SET DEFAULT nextval('public.transaction_lines_id_seq'::regclass);


--
-- Name: transactions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions ALTER COLUMN id SET DEFAULT nextval('public.transactions_id_seq'::regclass);


--
-- Name: adjustment_reason_codes adjustment_reason_codes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.adjustment_reason_codes
    ADD CONSTRAINT adjustment_reason_codes_pkey PRIMARY KEY (code);


--
-- Name: allergens allergens_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.allergens
    ADD CONSTRAINT allergens_name_key UNIQUE (name);


--
-- Name: allergens allergens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.allergens
    ADD CONSTRAINT allergens_pkey PRIMARY KEY (id);


--
-- Name: batch_formulas batch_formulas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.batch_formulas
    ADD CONSTRAINT batch_formulas_pkey PRIMARY KEY (id);


--
-- Name: bom_lines bom_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bom_lines
    ADD CONSTRAINT bom_lines_pkey PRIMARY KEY (id);


--
-- Name: boms boms_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.boms
    ADD CONSTRAINT boms_pkey PRIMARY KEY (id);


--
-- Name: certifications certifications_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.certifications
    ADD CONSTRAINT certifications_pkey PRIMARY KEY (id);


--
-- Name: customer_aliases customer_aliases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.customer_aliases
    ADD CONSTRAINT customer_aliases_pkey PRIMARY KEY (id);


--
-- Name: customers customers_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.customers
    ADD CONSTRAINT customers_name_key UNIQUE (name);


--
-- Name: customers customers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.customers
    ADD CONSTRAINT customers_pkey PRIMARY KEY (id);


--
-- Name: expected_receipts expected_receipts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.expected_receipts
    ADD CONSTRAINT expected_receipts_pkey PRIMARY KEY (id);


--
-- Name: ingredient_lot_consumption ingredient_lot_consumption_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingredient_lot_consumption
    ADD CONSTRAINT ingredient_lot_consumption_pkey PRIMARY KEY (id);


--
-- Name: inventory_adjustments inventory_adjustments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inventory_adjustments
    ADD CONSTRAINT inventory_adjustments_pkey PRIMARY KEY (id);


--
-- Name: ledger_corrections ledger_corrections_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ledger_corrections
    ADD CONSTRAINT ledger_corrections_pkey PRIMARY KEY (id);


--
-- Name: line_capacity_modes line_capacity_modes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.line_capacity_modes
    ADD CONSTRAINT line_capacity_modes_pkey PRIMARY KEY (id);


--
-- Name: lot_reassignments lot_reassignments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lot_reassignments
    ADD CONSTRAINT lot_reassignments_pkey PRIMARY KEY (id);


--
-- Name: lot_supplier_codes lot_supplier_codes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lot_supplier_codes
    ADD CONSTRAINT lot_supplier_codes_pkey PRIMARY KEY (id);


--
-- Name: lots lots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lots
    ADD CONSTRAINT lots_pkey PRIMARY KEY (id);


--
-- Name: lots lots_product_id_lot_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lots
    ADD CONSTRAINT lots_product_id_lot_code_key UNIQUE (product_id, lot_code);


--
-- Name: notes notes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.notes
    ADD CONSTRAINT notes_pkey PRIMARY KEY (id);


--
-- Name: oauth_tokens oauth_tokens_email_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.oauth_tokens
    ADD CONSTRAINT oauth_tokens_email_key UNIQUE (email);


--
-- Name: oauth_tokens oauth_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.oauth_tokens
    ADD CONSTRAINT oauth_tokens_pkey PRIMARY KEY (id);


--
-- Name: product_allergens product_allergens_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_allergens
    ADD CONSTRAINT product_allergens_pkey PRIMARY KEY (id);


--
-- Name: product_allergens product_allergens_product_id_allergen_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_allergens
    ADD CONSTRAINT product_allergens_product_id_allergen_id_key UNIQUE (product_id, allergen_id);


--
-- Name: product_bom product_bom_finished_product_id_component_product_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_bom
    ADD CONSTRAINT product_bom_finished_product_id_component_product_id_key UNIQUE (finished_product_id, component_product_id);


--
-- Name: product_bom product_bom_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_bom
    ADD CONSTRAINT product_bom_pkey PRIMARY KEY (id);


--
-- Name: product_line_assignments product_line_assignments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_line_assignments
    ADD CONSTRAINT product_line_assignments_pkey PRIMARY KEY (id);


--
-- Name: product_line_assignments product_line_assignments_product_id_line_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_line_assignments
    ADD CONSTRAINT product_line_assignments_product_id_line_id_key UNIQUE (product_id, line_id);


--
-- Name: product_verification_history product_verification_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_verification_history
    ADD CONSTRAINT product_verification_history_pkey PRIMARY KEY (id);


--
-- Name: production_lines production_lines_line_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_lines
    ADD CONSTRAINT production_lines_line_code_key UNIQUE (line_code);


--
-- Name: production_lines production_lines_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_lines
    ADD CONSTRAINT production_lines_name_key UNIQUE (name);


--
-- Name: production_lines production_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_lines
    ADD CONSTRAINT production_lines_pkey PRIMARY KEY (id);


--
-- Name: production_schedule production_schedule_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_schedule
    ADD CONSTRAINT production_schedule_pkey PRIMARY KEY (id);


--
-- Name: production_schedule production_schedule_schedule_date_line_id_product_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_schedule
    ADD CONSTRAINT production_schedule_schedule_date_line_id_product_id_key UNIQUE (schedule_date, line_id, product_id);


--
-- Name: products products_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_name_key UNIQUE (name);


--
-- Name: products products_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_pkey PRIMARY KEY (id);


--
-- Name: reassignment_reason_codes reassignment_reason_codes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reassignment_reason_codes
    ADD CONSTRAINT reassignment_reason_codes_pkey PRIMARY KEY (code);


--
-- Name: sales_order_allocation_reactivations sales_order_allocation_reactivations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocation_reactivations
    ADD CONSTRAINT sales_order_allocation_reactivations_pkey PRIMARY KEY (transaction_id, sales_order_line_id);


--
-- Name: sales_order_allocations sales_order_allocations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_pkey PRIMARY KEY (id);


--
-- Name: sales_order_flags sales_order_flags_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_flags
    ADD CONSTRAINT sales_order_flags_pkey PRIMARY KEY (so_number);


--
-- Name: sales_order_lines sales_order_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_lines
    ADD CONSTRAINT sales_order_lines_pkey PRIMARY KEY (id);


--
-- Name: sales_order_shipments sales_order_shipments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_shipments
    ADD CONSTRAINT sales_order_shipments_pkey PRIMARY KEY (id);


--
-- Name: sales_orders sales_orders_order_number_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_orders
    ADD CONSTRAINT sales_orders_order_number_key UNIQUE (order_number);


--
-- Name: sales_orders sales_orders_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_orders
    ADD CONSTRAINT sales_orders_pkey PRIMARY KEY (id);


--
-- Name: scheduling_config scheduling_config_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scheduling_config
    ADD CONSTRAINT scheduling_config_pkey PRIMARY KEY (key);


--
-- Name: shipment_lines shipment_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipment_lines
    ADD CONSTRAINT shipment_lines_pkey PRIMARY KEY (id);


--
-- Name: shipments shipments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipments
    ADD CONSTRAINT shipments_pkey PRIMARY KEY (id);


--
-- Name: suppliers suppliers_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.suppliers
    ADD CONSTRAINT suppliers_name_key UNIQUE (name);


--
-- Name: suppliers suppliers_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.suppliers
    ADD CONSTRAINT suppliers_pkey PRIMARY KEY (id);


--
-- Name: supply_requests supply_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.supply_requests
    ADD CONSTRAINT supply_requests_pkey PRIMARY KEY (id);


--
-- Name: transaction_lines transaction_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transaction_lines
    ADD CONSTRAINT transaction_lines_pkey PRIMARY KEY (id);


--
-- Name: transactions transactions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_pkey PRIMARY KEY (id);


--
-- Name: batch_formulas unique_batch_ingredient; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.batch_formulas
    ADD CONSTRAINT unique_batch_ingredient UNIQUE (product_id, ingredient_product_id);


--
-- Name: idx_certifications_business_date_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_certifications_business_date_created ON public.certifications USING btree (business_date, created_at, id);


--
-- Name: idx_customer_aliases_lower_alias; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_customer_aliases_lower_alias ON public.customer_aliases USING btree (lower(alias));


--
-- Name: idx_expected_receipts_open_match; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_expected_receipts_open_match ON public.expected_receipts USING btree (product_id, supplier_id, expected_date, id) WHERE (status = 'open'::text);


--
-- Name: idx_expected_receipts_status_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_expected_receipts_status_date ON public.expected_receipts USING btree (status, expected_date);


--
-- Name: idx_ilc_ingredient_lot; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ilc_ingredient_lot ON public.ingredient_lot_consumption USING btree (ingredient_lot_id);


--
-- Name: idx_ilc_ingredient_product; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ilc_ingredient_product ON public.ingredient_lot_consumption USING btree (ingredient_product_id);


--
-- Name: idx_ilc_transaction; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ilc_transaction ON public.ingredient_lot_consumption USING btree (transaction_id);


--
-- Name: idx_inventory_adjustments_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_inventory_adjustments_date ON public.inventory_adjustments USING btree (adjusted_at DESC);


--
-- Name: idx_inventory_adjustments_product; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_inventory_adjustments_product ON public.inventory_adjustments USING btree (product_id);


--
-- Name: idx_inventory_adjustments_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_inventory_adjustments_type ON public.inventory_adjustments USING btree (adjustment_type);


--
-- Name: idx_ledger_corrections_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ledger_corrections_created_at ON public.ledger_corrections USING btree (created_at);


--
-- Name: idx_ledger_corrections_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_ledger_corrections_target ON public.ledger_corrections USING btree (target_table, target_id, created_at, id);


--
-- Name: idx_lot_reassignments_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lot_reassignments_date ON public.lot_reassignments USING btree (reassigned_at DESC);


--
-- Name: idx_lot_reassignments_lot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lot_reassignments_lot_id ON public.lot_reassignments USING btree (lot_id);


--
-- Name: idx_lot_supplier_codes_lot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lot_supplier_codes_lot_id ON public.lot_supplier_codes USING btree (lot_id);


--
-- Name: idx_lot_supplier_codes_supplier_lot; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lot_supplier_codes_supplier_lot ON public.lot_supplier_codes USING btree (lower(supplier_lot_code));


--
-- Name: idx_lots_product_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lots_product_id ON public.lots USING btree (product_id);


--
-- Name: idx_lots_supplier_lot_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_lots_supplier_lot_code ON public.lots USING btree (lower(supplier_lot_code));


--
-- Name: idx_notes_category; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_notes_category ON public.notes USING btree (category);


--
-- Name: idx_notes_due_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_notes_due_date ON public.notes USING btree (due_date) WHERE (due_date IS NOT NULL);


--
-- Name: idx_notes_entity; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_notes_entity ON public.notes USING btree (entity_type, entity_id) WHERE (entity_type IS NOT NULL);


--
-- Name: idx_notes_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_notes_status ON public.notes USING btree (status);


--
-- Name: idx_product_allergens_product; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_product_allergens_product ON public.product_allergens USING btree (product_id);


--
-- Name: idx_product_bom_component; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_product_bom_component ON public.product_bom USING btree (component_product_id);


--
-- Name: idx_product_bom_finished; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_product_bom_finished ON public.product_bom USING btree (finished_product_id);


--
-- Name: idx_product_verification_history_product; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_product_verification_history_product ON public.product_verification_history USING btree (product_id);


--
-- Name: idx_products_name_trgm; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_name_trgm ON public.products USING gin (name public.gin_trgm_ops);


--
-- Name: idx_products_production_context; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_production_context ON public.products USING btree (production_context) WHERE ((production_context)::text <> 'standard'::text);


--
-- Name: idx_products_verification_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_verification_status ON public.products USING btree (verification_status) WHERE ((verification_status)::text <> 'verified'::text);


--
-- Name: idx_sales_order_lines_order; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_order_lines_order ON public.sales_order_lines USING btree (sales_order_id);


--
-- Name: idx_sales_order_lines_product; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_order_lines_product ON public.sales_order_lines USING btree (product_id);


--
-- Name: idx_sales_order_shipments_line; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_order_shipments_line ON public.sales_order_shipments USING btree (sales_order_line_id);


--
-- Name: idx_sales_order_shipments_txn; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_order_shipments_txn ON public.sales_order_shipments USING btree (transaction_id);


--
-- Name: idx_sales_orders_customer; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_orders_customer ON public.sales_orders USING btree (customer_id);


--
-- Name: idx_sales_orders_ship_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_orders_ship_date ON public.sales_orders USING btree (requested_ship_date);


--
-- Name: idx_sales_orders_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sales_orders_status ON public.sales_orders USING btree (status);


--
-- Name: idx_shipment_lines_product_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_shipment_lines_product_id ON public.shipment_lines USING btree (product_id);


--
-- Name: idx_shipment_lines_shipment_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_shipment_lines_shipment_id ON public.shipment_lines USING btree (shipment_id);


--
-- Name: idx_shipment_lines_transaction_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_shipment_lines_transaction_id ON public.shipment_lines USING btree (transaction_id);


--
-- Name: idx_shipments_customer_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_shipments_customer_id ON public.shipments USING btree (customer_id);


--
-- Name: idx_shipments_sales_order_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_shipments_sales_order_id ON public.shipments USING btree (sales_order_id);


--
-- Name: idx_shipments_shipped_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_shipments_shipped_at ON public.shipments USING btree (shipped_at);


--
-- Name: idx_sos_sales_order_line_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sos_sales_order_line_id ON public.sales_order_shipments USING btree (sales_order_line_id);


--
-- Name: idx_sos_shipped_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sos_shipped_at ON public.sales_order_shipments USING btree (shipped_at);


--
-- Name: idx_sos_transaction_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sos_transaction_id ON public.sales_order_shipments USING btree (transaction_id);


--
-- Name: idx_supply_requests_product; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_supply_requests_product ON public.supply_requests USING btree (product_id) WHERE (product_id IS NOT NULL);


--
-- Name: idx_supply_requests_status_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_supply_requests_status_created ON public.supply_requests USING btree (status, created_at DESC, id DESC);


--
-- Name: idx_transaction_lines_lot_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transaction_lines_lot_id ON public.transaction_lines USING btree (lot_id);


--
-- Name: idx_transactions_adjust_reason; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_adjust_reason ON public.transactions USING btree (adjust_reason) WHERE (adjust_reason IS NOT NULL);


--
-- Name: idx_transactions_bol_reference; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_bol_reference ON public.transactions USING btree (bol_reference) WHERE (bol_reference IS NOT NULL);


--
-- Name: idx_transactions_customer_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_customer_name ON public.transactions USING btree (customer_name) WHERE (customer_name IS NOT NULL);


--
-- Name: idx_transactions_expected_receipt; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_expected_receipt ON public.transactions USING btree (expected_receipt_id) WHERE (expected_receipt_id IS NOT NULL);


--
-- Name: idx_transactions_order_reference; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_order_reference ON public.transactions USING btree (order_reference) WHERE (order_reference IS NOT NULL);


--
-- Name: idx_transactions_shipper_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_shipper_code ON public.transactions USING btree (shipper_code) WHERE (shipper_code IS NOT NULL);


--
-- Name: idx_transactions_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_status ON public.transactions USING btree (status);


--
-- Name: idx_transactions_timestamp; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_transactions_timestamp ON public.transactions USING btree ("timestamp" DESC);


--
-- Name: products_odoo_code_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX products_odoo_code_unique ON public.products USING btree (odoo_code) WHERE (odoo_code IS NOT NULL);


--
-- Name: soa_active_lot_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX soa_active_lot_idx ON public.sales_order_allocations USING btree (lot_id) WHERE ((status = 'active'::text) AND (lot_id IS NOT NULL));


--
-- Name: soa_active_lot_uniq; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX soa_active_lot_uniq ON public.sales_order_allocations USING btree (sales_order_line_id, lot_id) WHERE ((status = 'active'::text) AND (lot_id IS NOT NULL));


--
-- Name: soa_active_product_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX soa_active_product_idx ON public.sales_order_allocations USING btree (product_id) WHERE (status = 'active'::text);


--
-- Name: soa_active_sku_uniq; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX soa_active_sku_uniq ON public.sales_order_allocations USING btree (sales_order_line_id) WHERE ((status = 'active'::text) AND (lot_id IS NULL));


--
-- Name: soa_order_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX soa_order_idx ON public.sales_order_allocations USING btree (sales_order_id);


--
-- Name: soa_ship_txn_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX soa_ship_txn_idx ON public.sales_order_allocations USING btree (ship_transaction_id) WHERE (ship_transaction_id IS NOT NULL);


--
-- Name: suppliers_name_norm_uidx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX suppliers_name_norm_uidx ON public.suppliers USING btree (public.supplier_name_norm(name));


--
-- Name: uq_certifications_original_business_date; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_certifications_original_business_date ON public.certifications USING btree (business_date) WHERE (supersedes_certification_id IS NULL);


--
-- Name: inventory_summary _RETURN; Type: RULE; Schema: public; Owner: -
--

CREATE OR REPLACE VIEW public.inventory_summary AS
 SELECT p.id,
    p.name,
    p.type,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS on_hand
   FROM ((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
  WHERE (COALESCE(p.active, true) = true)
  GROUP BY p.id
  ORDER BY p.type, p.name;


--
-- Name: lot_balances _RETURN; Type: RULE; Schema: public; Owner: -
--

CREATE OR REPLACE VIEW public.lot_balances AS
 SELECT l.id,
    l.lot_code,
    l.created_at,
    p.name AS product,
    p.type,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS balance
   FROM ((public.lots l
     JOIN public.products p ON ((p.id = l.product_id)))
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
  GROUP BY l.id, p.id
 HAVING (COALESCE(sum(tl.quantity_lb), (0)::numeric) > (0)::numeric)
  ORDER BY l.created_at DESC;


--
-- Name: low_stock_alerts _RETURN; Type: RULE; Schema: public; Owner: -
--

CREATE OR REPLACE VIEW public.low_stock_alerts AS
 SELECT p.id,
    p.name,
    COALESCE(sum(tl.quantity_lb), (0)::numeric) AS on_hand
   FROM ((public.products p
     LEFT JOIN public.lots l ON ((l.product_id = p.id)))
     LEFT JOIN public.transaction_lines tl ON ((tl.lot_id = l.id)))
  WHERE ((p.type = 'ingredient'::text) AND (COALESCE(p.active, true) = true))
  GROUP BY p.id
 HAVING (COALESCE(sum(tl.quantity_lb), (0)::numeric) < (100)::numeric)
  ORDER BY COALESCE(sum(tl.quantity_lb), (0)::numeric);


--
-- Name: open_orders_summary _RETURN; Type: RULE; Schema: public; Owner: -
--

CREATE OR REPLACE VIEW public.open_orders_summary AS
 SELECT so.id,
    so.order_number,
    c.name AS customer,
    so.order_date,
    so.requested_ship_date,
    so.status,
    count(sol.id) AS line_count,
    sum(sol.quantity_lb) AS total_lb_ordered,
    sum(sol.quantity_shipped_lb) AS total_lb_shipped,
    (sum(sol.quantity_lb) - sum(sol.quantity_shipped_lb)) AS remaining_lb,
        CASE
            WHEN ((so.requested_ship_date < CURRENT_DATE) AND (so.status <> ALL (ARRAY['shipped'::text, 'invoiced'::text, 'cancelled'::text]))) THEN true
            ELSE false
        END AS overdue
   FROM ((public.sales_orders so
     JOIN public.customers c ON ((c.id = so.customer_id)))
     JOIN public.sales_order_lines sol ON ((sol.sales_order_id = so.id)))
  WHERE (so.status <> ALL (ARRAY['invoiced'::text, 'cancelled'::text]))
  GROUP BY so.id, c.name
  ORDER BY so.requested_ship_date;


--
-- Name: production_history _RETURN; Type: RULE; Schema: public; Owner: -
--

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
   FROM (((public.transactions t
     JOIN public.transaction_lines tl ON ((tl.transaction_id = t.id)))
     JOIN public.lots l ON ((l.id = tl.lot_id)))
     JOIN public.products p ON ((p.id = tl.product_id)))
  WHERE ((t.type = 'make'::text) AND (tl.quantity_lb > (0)::numeric))
  GROUP BY t.id, p.id, l.id
  ORDER BY t."timestamp" DESC;


--
-- Name: certifications trg_certifications_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_certifications_append_only BEFORE DELETE OR UPDATE ON public.certifications FOR EACH ROW EXECUTE FUNCTION public.ledger_block_append_only_change();


--
-- Name: certifications trg_certifications_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_certifications_created_at BEFORE INSERT ON public.certifications FOR EACH ROW EXECUTE FUNCTION public.ledger_force_new_created_at();


--
-- Name: customers trg_customer_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_customer_updated BEFORE UPDATE ON public.customers FOR EACH ROW EXECUTE FUNCTION public.update_sales_order_timestamp();


--
-- Name: ingredient_lot_consumption trg_ingredient_lot_consumption_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_ingredient_lot_consumption_created_at BEFORE INSERT OR UPDATE ON public.ingredient_lot_consumption FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: inventory_adjustments trg_inventory_adjustments_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_inventory_adjustments_created_at BEFORE INSERT OR UPDATE ON public.inventory_adjustments FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: ledger_corrections trg_ledger_corrections_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_ledger_corrections_append_only BEFORE DELETE OR UPDATE ON public.ledger_corrections FOR EACH ROW EXECUTE FUNCTION public.ledger_block_append_only_change();


--
-- Name: ledger_corrections trg_ledger_corrections_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_ledger_corrections_created_at BEFORE INSERT ON public.ledger_corrections FOR EACH ROW EXECUTE FUNCTION public.ledger_force_new_created_at();


--
-- Name: lot_reassignments trg_lot_reassignments_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_lot_reassignments_created_at BEFORE INSERT OR UPDATE ON public.lot_reassignments FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: lot_supplier_codes trg_lot_supplier_codes_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_lot_supplier_codes_created_at BEFORE INSERT OR UPDATE ON public.lot_supplier_codes FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: lots trg_lots_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_lots_created_at BEFORE INSERT OR UPDATE ON public.lots FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: sales_orders trg_order_number; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_order_number BEFORE INSERT ON public.sales_orders FOR EACH ROW WHEN (((new.order_number IS NULL) OR (new.order_number = ''::text))) EXECUTE FUNCTION public.generate_order_number();


--
-- Name: production_schedule trg_production_schedule_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_production_schedule_created_at BEFORE INSERT OR UPDATE ON public.production_schedule FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: sales_order_lines trg_sales_order_lines_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_sales_order_lines_created_at BEFORE INSERT OR UPDATE ON public.sales_order_lines FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: sales_order_shipments trg_sales_order_shipments_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_sales_order_shipments_created_at BEFORE INSERT OR UPDATE ON public.sales_order_shipments FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: sales_orders trg_sales_order_updated; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_sales_order_updated BEFORE UPDATE ON public.sales_orders FOR EACH ROW EXECUTE FUNCTION public.update_sales_order_timestamp();


--
-- Name: sales_orders trg_sales_orders_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_sales_orders_created_at BEFORE INSERT OR UPDATE ON public.sales_orders FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: shipment_lines trg_shipment_lines_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_shipment_lines_created_at BEFORE INSERT OR UPDATE ON public.shipment_lines FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: shipments trg_shipments_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_shipments_created_at BEFORE INSERT OR UPDATE ON public.shipments FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: transaction_lines trg_transaction_lines_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_transaction_lines_created_at BEFORE INSERT OR UPDATE ON public.transaction_lines FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: transaction_lines trg_transaction_lines_original_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_transaction_lines_original_append_only BEFORE DELETE OR UPDATE ON public.transaction_lines FOR EACH ROW EXECUTE FUNCTION public.ledger_block_append_only_change();


--
-- Name: transactions trg_transactions_business_time; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_transactions_business_time BEFORE INSERT ON public.transactions FOR EACH ROW EXECUTE FUNCTION public.ledger_fill_transaction_business_time();


--
-- Name: transactions trg_transactions_created_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_transactions_created_at BEFORE INSERT OR UPDATE ON public.transactions FOR EACH ROW EXECUTE FUNCTION public.ledger_enforce_created_at();


--
-- Name: transactions trg_transactions_original_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_transactions_original_append_only BEFORE DELETE OR UPDATE ON public.transactions FOR EACH ROW EXECUTE FUNCTION public.ledger_block_append_only_change();


--
-- Name: batch_formulas batch_formulas_ingredient_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.batch_formulas
    ADD CONSTRAINT batch_formulas_ingredient_product_id_fkey FOREIGN KEY (ingredient_product_id) REFERENCES public.products(id);


--
-- Name: batch_formulas batch_formulas_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.batch_formulas
    ADD CONSTRAINT batch_formulas_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: bom_lines bom_lines_bom_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bom_lines
    ADD CONSTRAINT bom_lines_bom_id_fkey FOREIGN KEY (bom_id) REFERENCES public.boms(id);


--
-- Name: certifications certifications_supersedes_certification_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.certifications
    ADD CONSTRAINT certifications_supersedes_certification_id_fkey FOREIGN KEY (supersedes_certification_id) REFERENCES public.certifications(id);


--
-- Name: customer_aliases customer_aliases_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.customer_aliases
    ADD CONSTRAINT customer_aliases_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id) ON DELETE CASCADE;


--
-- Name: expected_receipts expected_receipts_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.expected_receipts
    ADD CONSTRAINT expected_receipts_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: expected_receipts expected_receipts_supplier_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.expected_receipts
    ADD CONSTRAINT expected_receipts_supplier_id_fkey FOREIGN KEY (supplier_id) REFERENCES public.suppliers(id);


--
-- Name: ingredient_lot_consumption ingredient_lot_consumption_ingredient_lot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingredient_lot_consumption
    ADD CONSTRAINT ingredient_lot_consumption_ingredient_lot_id_fkey FOREIGN KEY (ingredient_lot_id) REFERENCES public.lots(id);


--
-- Name: ingredient_lot_consumption ingredient_lot_consumption_ingredient_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingredient_lot_consumption
    ADD CONSTRAINT ingredient_lot_consumption_ingredient_product_id_fkey FOREIGN KEY (ingredient_product_id) REFERENCES public.products(id);


--
-- Name: ingredient_lot_consumption ingredient_lot_consumption_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ingredient_lot_consumption
    ADD CONSTRAINT ingredient_lot_consumption_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id) ON DELETE CASCADE;


--
-- Name: line_capacity_modes line_capacity_modes_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.line_capacity_modes
    ADD CONSTRAINT line_capacity_modes_line_id_fkey FOREIGN KEY (line_id) REFERENCES public.production_lines(id);


--
-- Name: lot_supplier_codes lot_supplier_codes_lot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lot_supplier_codes
    ADD CONSTRAINT lot_supplier_codes_lot_id_fkey FOREIGN KEY (lot_id) REFERENCES public.lots(id) ON DELETE CASCADE;


--
-- Name: lots lots_merged_into_lot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lots
    ADD CONSTRAINT lots_merged_into_lot_id_fkey FOREIGN KEY (merged_into_lot_id) REFERENCES public.lots(id);


--
-- Name: lots lots_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lots
    ADD CONSTRAINT lots_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: product_allergens product_allergens_allergen_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_allergens
    ADD CONSTRAINT product_allergens_allergen_id_fkey FOREIGN KEY (allergen_id) REFERENCES public.allergens(id) ON DELETE CASCADE;


--
-- Name: product_allergens product_allergens_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_allergens
    ADD CONSTRAINT product_allergens_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id) ON DELETE CASCADE;


--
-- Name: product_bom product_bom_component_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_bom
    ADD CONSTRAINT product_bom_component_product_id_fkey FOREIGN KEY (component_product_id) REFERENCES public.products(id);


--
-- Name: product_bom product_bom_finished_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_bom
    ADD CONSTRAINT product_bom_finished_product_id_fkey FOREIGN KEY (finished_product_id) REFERENCES public.products(id) ON DELETE CASCADE;


--
-- Name: product_line_assignments product_line_assignments_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_line_assignments
    ADD CONSTRAINT product_line_assignments_line_id_fkey FOREIGN KEY (line_id) REFERENCES public.production_lines(id);


--
-- Name: product_line_assignments product_line_assignments_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.product_line_assignments
    ADD CONSTRAINT product_line_assignments_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: production_schedule production_schedule_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_schedule
    ADD CONSTRAINT production_schedule_line_id_fkey FOREIGN KEY (line_id) REFERENCES public.production_lines(id);


--
-- Name: production_schedule production_schedule_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.production_schedule
    ADD CONSTRAINT production_schedule_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: products products_parent_batch_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_parent_batch_product_id_fkey FOREIGN KEY (parent_batch_product_id) REFERENCES public.products(id);


--
-- Name: sales_order_allocation_reactivations sales_order_allocation_reactivations_correction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocation_reactivations
    ADD CONSTRAINT sales_order_allocation_reactivations_correction_id_fkey FOREIGN KEY (correction_id) REFERENCES public.ledger_corrections(id);


--
-- Name: sales_order_allocation_reactivations sales_order_allocation_reactivations_sales_order_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocation_reactivations
    ADD CONSTRAINT sales_order_allocation_reactivations_sales_order_line_id_fkey FOREIGN KEY (sales_order_line_id) REFERENCES public.sales_order_lines(id);


--
-- Name: sales_order_allocation_reactivations sales_order_allocation_reactivations_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocation_reactivations
    ADD CONSTRAINT sales_order_allocation_reactivations_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id);


--
-- Name: sales_order_allocations sales_order_allocations_last_ship_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_last_ship_transaction_id_fkey FOREIGN KEY (last_ship_transaction_id) REFERENCES public.transactions(id);


--
-- Name: sales_order_allocations sales_order_allocations_lot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_lot_id_fkey FOREIGN KEY (lot_id) REFERENCES public.lots(id);


--
-- Name: sales_order_allocations sales_order_allocations_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: sales_order_allocations sales_order_allocations_sales_order_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_sales_order_id_fkey FOREIGN KEY (sales_order_id) REFERENCES public.sales_orders(id);


--
-- Name: sales_order_allocations sales_order_allocations_sales_order_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_sales_order_line_id_fkey FOREIGN KEY (sales_order_line_id) REFERENCES public.sales_order_lines(id);


--
-- Name: sales_order_allocations sales_order_allocations_ship_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_ship_transaction_id_fkey FOREIGN KEY (ship_transaction_id) REFERENCES public.transactions(id);


--
-- Name: sales_order_allocations sales_order_allocations_split_from_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_allocations
    ADD CONSTRAINT sales_order_allocations_split_from_id_fkey FOREIGN KEY (split_from_id) REFERENCES public.sales_order_allocations(id);


--
-- Name: sales_order_lines sales_order_lines_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_lines
    ADD CONSTRAINT sales_order_lines_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: sales_order_lines sales_order_lines_sales_order_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_lines
    ADD CONSTRAINT sales_order_lines_sales_order_id_fkey FOREIGN KEY (sales_order_id) REFERENCES public.sales_orders(id) ON DELETE CASCADE;


--
-- Name: sales_order_shipments sales_order_shipments_sales_order_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_shipments
    ADD CONSTRAINT sales_order_shipments_sales_order_line_id_fkey FOREIGN KEY (sales_order_line_id) REFERENCES public.sales_order_lines(id) ON DELETE CASCADE;


--
-- Name: sales_order_shipments sales_order_shipments_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_order_shipments
    ADD CONSTRAINT sales_order_shipments_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id);


--
-- Name: sales_orders sales_orders_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sales_orders
    ADD CONSTRAINT sales_orders_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: shipment_lines shipment_lines_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipment_lines
    ADD CONSTRAINT shipment_lines_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: shipment_lines shipment_lines_sales_order_line_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipment_lines
    ADD CONSTRAINT shipment_lines_sales_order_line_id_fkey FOREIGN KEY (sales_order_line_id) REFERENCES public.sales_order_lines(id);


--
-- Name: shipment_lines shipment_lines_shipment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipment_lines
    ADD CONSTRAINT shipment_lines_shipment_id_fkey FOREIGN KEY (shipment_id) REFERENCES public.shipments(id);


--
-- Name: shipment_lines shipment_lines_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipment_lines
    ADD CONSTRAINT shipment_lines_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id);


--
-- Name: shipments shipments_customer_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipments
    ADD CONSTRAINT shipments_customer_id_fkey FOREIGN KEY (customer_id) REFERENCES public.customers(id);


--
-- Name: shipments shipments_sales_order_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipments
    ADD CONSTRAINT shipments_sales_order_id_fkey FOREIGN KEY (sales_order_id) REFERENCES public.sales_orders(id);


--
-- Name: shipments shipments_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shipments
    ADD CONSTRAINT shipments_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id);


--
-- Name: supply_requests supply_requests_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.supply_requests
    ADD CONSTRAINT supply_requests_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: transaction_lines transaction_lines_lot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transaction_lines
    ADD CONSTRAINT transaction_lines_lot_id_fkey FOREIGN KEY (lot_id) REFERENCES public.lots(id);


--
-- Name: transaction_lines transaction_lines_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transaction_lines
    ADD CONSTRAINT transaction_lines_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: transaction_lines transaction_lines_transaction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transaction_lines
    ADD CONSTRAINT transaction_lines_transaction_id_fkey FOREIGN KEY (transaction_id) REFERENCES public.transactions(id);


--
-- Name: transactions transactions_expected_receipt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_expected_receipt_id_fkey FOREIGN KEY (expected_receipt_id) REFERENCES public.expected_receipts(id);


--
-- PostgreSQL database dump complete
--

\unrestrict lLAWZcgkJFF1oXCgBXDD1uTvn4qEZYTsQa6SYb9objg1ubWzjefTNcURoaQtZPw

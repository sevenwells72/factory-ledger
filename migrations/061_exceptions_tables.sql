-- A3a (design rev 3.4 §5.1, §7.1, R3): the exceptions queue, shortage flags
-- and the fixed correction-reason list. TABLES AND SEEDS ONLY — no
-- enforcement, no route change, no startup hook (apply explicitly, like 060).
-- Apply as the app/table owner after 060, port 5432, in one transaction
-- (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s',
-- SET LOCAL search_path=public, COMMIT). Safe to rerun: every statement is
-- IF NOT EXISTS / ON CONFLICT DO NOTHING / WHERE reason_code IS NULL.
--
-- The only write to existing rows is the one-time, history-only backfill of
-- the new nullable transactions.reason_code column on type='adjust' rows
-- (§5.1 "mapping of today's codes"). As in 046, the 039 append-only trigger
-- is disabled for that single statement inside this transaction and
-- re-enabled before COMMIT; no quantity, status, timestamp or ledger-line
-- value changes. Nothing in the application reads or writes these objects
-- until A3b; A5 and A6 build against them.

-- ---------------------------------------------------------------------------
-- 1. correction_reasons — Michael's fixed list of 8 (decided 2026-10-07, D4).
--    Edited by migration only, never from a screen. adjust_sign records the
--    §5.1 "adjust(+)" / "adjust(−)" restriction for A3b to enforce.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS correction_reasons (
    code text PRIMARY KEY CHECK (code ~ '^[a-z][a-z0-9_]{1,39}$'),
    label_en text NOT NULL CHECK (btrim(label_en) <> ''),
    label_es text NOT NULL CHECK (btrim(label_es) <> ''),
    applies_to text[] NOT NULL CHECK (
        cardinality(applies_to) > 0
        AND applies_to <@ ARRAY['adjust', 'found', 'void', 'rename_lot',
                                'update_supplier_lot', 'resolve_exception']::text[]),
    adjust_sign text NOT NULL DEFAULT 'any'
        CHECK (adjust_sign IN ('any', 'positive', 'negative')),
    note_required boolean NOT NULL DEFAULT false,
    active boolean NOT NULL DEFAULT true,
    sort_order smallint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO correction_reasons
    (code, label_en, label_es, applies_to, adjust_sign, note_required, sort_order)
VALUES
    ('physical_count', 'Physical count', 'Conteo físico',
        ARRAY['adjust', 'found', 'resolve_exception'], 'any', false, 10),
    ('missing_receipt', 'Missing receipt', 'Recepción no registrada',
        ARRAY['found', 'adjust', 'resolve_exception'], 'positive', false, 20),
    ('missing_production', 'Missing production', 'Producción no registrada',
        ARRAY['adjust', 'found', 'resolve_exception'], 'any', false, 30),
    ('wrong_lot', 'Wrong lot used', 'Lote equivocado',
        ARRAY['adjust', 'void', 'rename_lot', 'update_supplier_lot'], 'any', false, 40),
    ('damage_disposal', 'Damage/disposal', 'Daño o desecho',
        ARRAY['adjust'], 'negative', false, 50),
    ('unrecorded_usage', 'Unrecorded usage', 'Uso no registrado',
        ARRAY['adjust'], 'negative', false, 60),
    ('data_entry_error', 'Data-entry error', 'Error de captura',
        ARRAY['void', 'adjust', 'rename_lot', 'update_supplier_lot'], 'any', false, 70),
    ('unknown', 'Unknown', 'Desconocido',
        ARRAY['adjust', 'found'], 'any', true, 80)
ON CONFLICT (code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 2. correction_reason_legacy_codes — §5.1 "mapping of today's codes".
--    Source of the history backfill below AND the overlap-period translation
--    table for any client that still sends a legacy value (422 after A10).
--    legacy_code is stored normalised (lower, trimmed, single spaces) so the
--    lookup is lower(btrim(regexp_replace(value, '\s+', ' ', 'g'))). The
--    /reason-codes descriptions are included because POST /adjust stores
--    whatever text the caller sent. Reassignment codes (incorrect_receive,
--    product_merge, supplier_relabel) stay on the admin-only reassign route
--    and are deliberately absent.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS correction_reason_legacy_codes (
    source text NOT NULL CHECK (source IN ('adjust', 'found')),
    legacy_code text NOT NULL CHECK (
        legacy_code <> ''
        AND legacy_code = lower(btrim(regexp_replace(legacy_code, '\s+', ' ', 'g')))),
    reason_code text NOT NULL REFERENCES correction_reasons(code),
    note_prefill text,
    PRIMARY KEY (source, legacy_code)
);

INSERT INTO correction_reason_legacy_codes (source, legacy_code, reason_code, note_prefill)
VALUES
    -- POST /adjust codes (GET /reason-codes "adjustment_reasons")
    ('adjust', 'count_correction', 'physical_count', NULL),
    ('adjust', 'damage', 'damage_disposal', NULL),
    ('adjust', 'spoilage', 'damage_disposal', NULL),
    ('adjust', 'sample', 'unrecorded_usage', NULL),
    ('adjust', 'hydration_yield', 'unrecorded_usage', 'hydration/yield'),
    ('adjust', 'other', 'unknown', NULL),
    -- the same codes as their /reason-codes descriptions
    ('adjust', 'correction from physical count', 'physical_count', NULL),
    ('adjust', 'product damaged', 'damage_disposal', NULL),
    ('adjust', 'product spoiled or expired', 'damage_disposal', NULL),
    ('adjust', 'used for samples', 'unrecorded_usage', NULL),
    ('adjust', 'hydration/processing yield correction', 'unrecorded_usage', 'hydration/yield'),
    ('adjust', 'other reason (specify in notes)', 'unknown', NULL),
    -- POST /inventory/found codes (inventory_adjustments.reason_code; the
    -- ledger row carries them as notes 'Found inventory[ with new product]: <code>')
    ('found', 'found_during_count', 'physical_count', NULL),
    ('found', 'found_back_stock', 'missing_receipt', NULL),
    ('found', 'predates_system', 'missing_receipt', 'predates system'),
    ('found', 'unreceived_delivery', 'missing_receipt', NULL)
ON CONFLICT (source, legacy_code) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3. transactions.reason_code — nullable, FK to the fixed list, set at INSERT
--    by A3b for new corrections. History-only backfill of type='adjust' rows:
--    adjust_reason (or the found-inventory note code) → mapped code, anything
--    else → 'unknown' with the original text kept in adjust_reason/notes.
-- ---------------------------------------------------------------------------
ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS reason_code text REFERENCES correction_reasons(code);
CREATE INDEX IF NOT EXISTS transactions_reason_code_idx
    ON transactions (reason_code) WHERE reason_code IS NOT NULL;

DO $$
DECLARE
    guard_enabled boolean;
    backfilled integer;
BEGIN
    SELECT tgenabled <> 'D' INTO guard_enabled
      FROM pg_trigger
     WHERE tgrelid = 'public.transactions'::regclass
       AND tgname = 'trg_transactions_original_append_only';
    IF guard_enabled THEN
        ALTER TABLE public.transactions
            DISABLE TRIGGER trg_transactions_original_append_only;
    END IF;

    WITH src AS (
        SELECT id,
               CASE WHEN NULLIF(btrim(adjust_reason), '') IS NOT NULL THEN 'adjust'
                    WHEN notes LIKE 'Found inventory%: %' THEN 'found' END AS source,
               CASE WHEN NULLIF(btrim(adjust_reason), '') IS NOT NULL THEN adjust_reason
                    WHEN notes LIKE 'Found inventory%: %'
                        THEN substr(notes, position(': ' IN notes) + 2) END AS raw
          FROM public.transactions
         WHERE type = 'adjust' AND reason_code IS NULL
    )
    UPDATE public.transactions t
       SET reason_code = COALESCE(m.reason_code, already.code, 'unknown')
      FROM src
      LEFT JOIN public.correction_reason_legacy_codes m
             ON m.source = src.source
            AND m.legacy_code = lower(btrim(regexp_replace(src.raw, '\s+', ' ', 'g')))
      -- a row that already carries one of the eight new codes keeps it
      LEFT JOIN public.correction_reasons already
             ON already.code = replace(lower(btrim(regexp_replace(src.raw, '\s+', ' ', 'g'))), ' ', '_')
     WHERE t.id = src.id;
    GET DIAGNOSTICS backfilled = ROW_COUNT;
    RAISE NOTICE '061: reason_code backfilled on % adjust transaction(s)', backfilled;

    IF guard_enabled THEN
        ALTER TABLE public.transactions
            ENABLE TRIGGER trg_transactions_original_append_only;
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 4. exceptions — §7.1 verbatim (kinds, statuses, severities), FK types match
--    the referenced columns (transactions/products/lots/orders/shipments ids
--    are integer; write_tickets ids are bigint).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exceptions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN (
        'SHORTAGE', 'UNIDENTIFIED_LOT', 'LARGE_CORRECTION', 'LATE_ENTRY',
        'SHIPMENT_PROOF_MISSING', 'NEGATIVE_BALANCE', 'POSSIBLE_DUPLICATE_ACK',
        'UNSHIPPED_PAST_DUE', 'SUNSHINE_INVOICE_PENDING', 'SHIFT_DISCREPANCY')),
    status text NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved', 'waived', 'escalated')),
    severity text NOT NULL CHECK (severity IN ('info', 'warn', 'block')),
    product_id integer REFERENCES products(id),
    lot_id integer REFERENCES lots(id),
    transaction_id integer REFERENCES transactions(id),
    sales_order_id integer REFERENCES sales_orders(id),
    shipment_id integer REFERENCES shipments(id),
    receipt_number text,
    ticket_id bigint REFERENCES write_tickets(id),
    detail jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(detail) = 'object'),
    owner_actor_id integer REFERENCES actors(id),
    opened_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    due_at timestamptz,
    escalated_at timestamptz,
    resolved_at timestamptz,
    resolved_by_actor_id integer REFERENCES actors(id),
    resolution_kind text,
    resolution_note text,
    resolution_ticket_id bigint REFERENCES write_tickets(id),
    CONSTRAINT exceptions_resolved_needs_time
        CHECK (status NOT IN ('resolved', 'waived') OR resolved_at IS NOT NULL),
    CONSTRAINT exceptions_escalated_needs_time
        CHECK (status <> 'escalated' OR escalated_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS exceptions_open_kind_idx
    ON exceptions (kind, opened_at) WHERE status IN ('open', 'escalated');
CREATE INDEX IF NOT EXISTS exceptions_open_owner_idx
    ON exceptions (owner_actor_id, due_at) WHERE status IN ('open', 'escalated');
CREATE INDEX IF NOT EXISTS exceptions_lot_idx
    ON exceptions (lot_id) WHERE lot_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS exceptions_transaction_idx
    ON exceptions (transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS exceptions_ticket_idx
    ON exceptions (ticket_id) WHERE ticket_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS exceptions_receipt_idx
    ON exceptions (receipt_number) WHERE receipt_number IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 5. shortage_flags — R3 post-and-flag (§5 table, R3 row). One row per short
--    consumption; exception_id links the paired exceptions(SHORTAGE) row.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shortage_flags (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id integer NOT NULL REFERENCES transactions(id),
    product_id integer NOT NULL REFERENCES products(id),
    lot_id integer NOT NULL REFERENCES lots(id),
    short_lb numeric(14,4) NOT NULL CHECK (short_lb > 0),
    exception_id bigint REFERENCES exceptions(id),
    opened_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    due_at timestamptz NOT NULL,
    owner_actor_id integer REFERENCES actors(id),
    status text NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved', 'waived', 'escalated')),
    resolution_kind text
        CHECK (resolution_kind IN ('counted', 'missing_movement', 'voided')),
    resolution_ticket_id bigint REFERENCES write_tickets(id),
    resolved_at timestamptz,
    resolved_by_actor_id integer REFERENCES actors(id),
    CONSTRAINT shortage_flags_resolved_needs_kind
        CHECK ((status = 'resolved') = (resolution_kind IS NOT NULL)),
    CONSTRAINT shortage_flags_closed_needs_time
        CHECK (status NOT IN ('resolved', 'waived') OR resolved_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS shortage_flags_open_lot_idx
    ON shortage_flags (lot_id) WHERE status IN ('open', 'escalated');
CREATE INDEX IF NOT EXISTS shortage_flags_open_product_idx
    ON shortage_flags (product_id) WHERE status IN ('open', 'escalated');
CREATE INDEX IF NOT EXISTS shortage_flags_transaction_idx
    ON shortage_flags (transaction_id);

-- ---------------------------------------------------------------------------
-- 6. Same posture as 058/060: owner-only tables, RLS on, no public policies.
-- ---------------------------------------------------------------------------
ALTER TABLE correction_reasons ENABLE ROW LEVEL SECURITY;
ALTER TABLE correction_reason_legacy_codes ENABLE ROW LEVEL SECURITY;
ALTER TABLE exceptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE shortage_flags ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON correction_reasons FROM PUBLIC;
REVOKE ALL ON correction_reason_legacy_codes FROM PUBLIC;
REVOKE ALL ON exceptions FROM PUBLIC;
REVOKE ALL ON shortage_flags FROM PUBLIC;

-- Standalone migration marker (no startup gate reads it yet; 060 precedent).
INSERT INTO migration_markers (name) VALUES ('061_exceptions_tables')
ON CONFLICT (name) DO NOTHING;

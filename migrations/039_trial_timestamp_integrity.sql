-- Phase 1 / Trial v8.1: immutable database creation time, append-only
-- corrections, daily certifications, and cutoff support.
--
-- Forward-safe/idempotent: existing timestamps are preserved and labelled
-- legacy_unverified. Rows that had no creation timestamp receive the one
-- migration timestamp and are explicitly labelled migration_backfill_039.

BEGIN;

CREATE OR REPLACE FUNCTION ledger_enforce_created_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.created_at := clock_timestamp();
        NEW.created_at_source := 'database';
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

DO $$
DECLARE
    table_name text;
    ledger_tables text[] := ARRAY[
        'transactions',
        'transaction_lines',
        'shipments',
        'shipment_lines',
        'sales_order_shipments',
        'ingredient_lot_consumption',
        'inventory_adjustments',
        'lot_reassignments',
        'lots',
        'lot_supplier_codes',
        'sales_orders',
        'sales_order_lines',
        'production_schedule'
    ];
BEGIN
    FOREACH table_name IN ARRAY ledger_tables LOOP
        IF to_regclass('public.' || table_name) IS NULL THEN
            CONTINUE;
        END IF;

        EXECUTE format(
            'ALTER TABLE %I ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ',
            table_name
        );
        EXECUTE format(
            'ALTER TABLE %I ADD COLUMN IF NOT EXISTS created_at_source TEXT',
            table_name
        );
        EXECUTE format(
            'UPDATE %I SET created_at_source = CASE WHEN created_at IS NULL '
            'THEN ''migration_backfill_039'' ELSE ''legacy_unverified'' END, '
            'created_at = COALESCE(created_at, statement_timestamp()) '
            'WHERE created_at_source IS NULL OR created_at IS NULL',
            table_name
        );
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN created_at SET DEFAULT clock_timestamp()',
            table_name
        );
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN created_at SET NOT NULL',
            table_name
        );
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN created_at_source SET DEFAULT ''database''',
            table_name
        );
        EXECUTE format(
            'ALTER TABLE %I ALTER COLUMN created_at_source SET NOT NULL',
            table_name
        );
        EXECUTE format('DROP TRIGGER IF EXISTS trg_%I_created_at ON %I', table_name, table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_%I_created_at BEFORE INSERT OR UPDATE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION ledger_enforce_created_at()',
            table_name,
            table_name
        );
    END LOOP;
END;
$$;

ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS business_date DATE,
    ADD COLUMN IF NOT EXISTS operator_id TEXT;

UPDATE transactions
SET occurred_at = COALESCE(
        occurred_at,
        "timestamp" AT TIME ZONE 'UTC',
        created_at::timestamptz
    ),
    business_date = COALESCE(
        business_date,
        timezone(
            'America/New_York',
            COALESCE(occurred_at, "timestamp" AT TIME ZONE 'UTC', created_at::timestamptz)
        )::date
    ),
    operator_id = COALESCE(operator_id, 'legacy-unattributed')
WHERE occurred_at IS NULL OR business_date IS NULL OR operator_id IS NULL;

ALTER TABLE transactions
    ALTER COLUMN occurred_at SET NOT NULL,
    ALTER COLUMN business_date SET NOT NULL,
    ALTER COLUMN operator_id SET DEFAULT 'legacy-shared-key',
    ALTER COLUMN operator_id SET NOT NULL;

CREATE OR REPLACE FUNCTION ledger_fill_transaction_business_time()
RETURNS trigger
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

DROP TRIGGER IF EXISTS trg_transactions_business_time ON transactions;
CREATE TRIGGER trg_transactions_business_time
BEFORE INSERT ON transactions
FOR EACH ROW EXECUTE FUNCTION ledger_fill_transaction_business_time();

CREATE TABLE IF NOT EXISTS ledger_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_table TEXT NOT NULL,
    target_id BIGINT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('amend', 'void', 'restore')),
    previous_values JSONB NOT NULL CHECK (jsonb_typeof(previous_values) = 'object'),
    replacement_values JSONB NOT NULL CHECK (jsonb_typeof(replacement_values) = 'object'),
    reason TEXT NOT NULL CHECK (btrim(reason) <> ''),
    operator_id TEXT NOT NULL DEFAULT 'legacy-shared-key',
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    created_at_source TEXT NOT NULL DEFAULT 'database',
    CONSTRAINT ledger_corrections_supported_target CHECK (
        target_table IN ('transactions', 'transaction_lines')
    )
);

ALTER TABLE ledger_corrections
    DROP CONSTRAINT IF EXISTS ledger_corrections_supported_target;
ALTER TABLE ledger_corrections
    ADD CONSTRAINT ledger_corrections_supported_target CHECK (
        target_table IN ('transactions', 'transaction_lines')
    );

CREATE INDEX IF NOT EXISTS idx_ledger_corrections_target
    ON ledger_corrections (target_table, target_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_ledger_corrections_created_at
    ON ledger_corrections (created_at);

CREATE TABLE IF NOT EXISTS certifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_date DATE NOT NULL,
    certified_at TIMESTAMPTZ NOT NULL,
    operator_id TEXT NOT NULL DEFAULT 'legacy-shared-key',
    source_type TEXT NOT NULL DEFAULT 'manual' CHECK (source_type IN ('manual', 'whatsapp_export')),
    source_message_id TEXT,
    notes TEXT,
    supersedes_certification_id UUID REFERENCES certifications(id),
    correction_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    created_at_source TEXT NOT NULL DEFAULT 'database',
    CONSTRAINT certification_correction_reason CHECK (
        (supersedes_certification_id IS NULL AND correction_reason IS NULL)
        OR
        (supersedes_certification_id IS NOT NULL AND btrim(correction_reason) <> '')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_certifications_original_business_date
    ON certifications (business_date)
    WHERE supersedes_certification_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_certifications_business_date_created
    ON certifications (business_date, created_at, id);

CREATE OR REPLACE FUNCTION ledger_force_new_created_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.created_at := clock_timestamp();
    NEW.created_at_source := 'database';
    NEW.operator_id := COALESCE(NULLIF(current_setting('app.operator_id', true), ''), NEW.operator_id, 'legacy-shared-key');
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION ledger_block_append_only_change()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; create a correction event instead', TG_TABLE_NAME
        USING ERRCODE = '23000';
END;
$$;

DROP TRIGGER IF EXISTS trg_ledger_corrections_created_at ON ledger_corrections;
CREATE TRIGGER trg_ledger_corrections_created_at
BEFORE INSERT ON ledger_corrections
FOR EACH ROW EXECUTE FUNCTION ledger_force_new_created_at();
DROP TRIGGER IF EXISTS trg_ledger_corrections_append_only ON ledger_corrections;
CREATE TRIGGER trg_ledger_corrections_append_only
BEFORE UPDATE OR DELETE ON ledger_corrections
FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change();

DROP TRIGGER IF EXISTS trg_certifications_created_at ON certifications;
CREATE TRIGGER trg_certifications_created_at
BEFORE INSERT ON certifications
FOR EACH ROW EXECUTE FUNCTION ledger_force_new_created_at();
DROP TRIGGER IF EXISTS trg_certifications_append_only ON certifications;
CREATE TRIGGER trg_certifications_append_only
BEFORE UPDATE OR DELETE ON certifications
FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change();

CREATE OR REPLACE VIEW current_certifications AS
SELECT DISTINCT ON (business_date)
    id AS certification_id,
    business_date,
    certified_at,
    operator_id,
    source_type,
    source_message_id,
    notes,
    supersedes_certification_id,
    correction_reason,
    created_at
FROM certifications
ORDER BY business_date, created_at DESC, id DESC;

CREATE OR REPLACE VIEW ledger_current_transactions AS
SELECT
    t.*,
    CASE
        WHEN correction.event_type = 'void' THEN 'voided'
        WHEN correction.event_type = 'restore' THEN 'posted'
        WHEN correction.event_type = 'amend'
            THEN COALESCE(correction.replacement_values ->> 'status', t.status, 'posted')
        ELSE COALESCE(t.status, 'posted')
    END AS effective_status,
    correction.id AS latest_correction_id,
    correction.event_type AS latest_correction_type,
    correction.created_at AS latest_correction_created_at,
    correction.operator_id AS latest_correction_operator_id,
    correction.replacement_values AS latest_replacement_values,
    to_jsonb(t)
        || COALESCE(correction.replacement_values, '{}'::jsonb)
        || jsonb_build_object(
            'status',
            CASE
                WHEN correction.event_type = 'void' THEN 'voided'
                WHEN correction.event_type = 'restore' THEN 'posted'
                WHEN correction.event_type = 'amend'
                    THEN COALESCE(correction.replacement_values ->> 'status', t.status, 'posted')
                ELSE COALESCE(t.status, 'posted')
            END
        ) AS effective_record
FROM transactions t
LEFT JOIN LATERAL (
    SELECT c.*
    FROM ledger_corrections c
    WHERE c.target_table = 'transactions' AND c.target_id = t.id
    ORDER BY c.created_at DESC, c.id DESC
    LIMIT 1
) correction ON true;

CREATE OR REPLACE VIEW ledger_current_transaction_lines AS
SELECT
    tl.id,
    tl.transaction_id,
    COALESCE((correction.replacement_values ->> 'product_id')::integer, tl.product_id)
        AS product_id,
    COALESCE((correction.replacement_values ->> 'lot_id')::integer, tl.lot_id)
        AS lot_id,
    COALESCE((correction.replacement_values ->> 'quantity_lb')::numeric, tl.quantity_lb)
        AS quantity_lb,
    tl.created_at,
    tl.created_at_source,
    correction.id AS latest_correction_id,
    correction.created_at AS latest_correction_created_at,
    correction.operator_id AS latest_correction_operator_id,
    to_jsonb(tl) || COALESCE(correction.replacement_values, '{}'::jsonb)
        AS effective_record
FROM transaction_lines tl
LEFT JOIN LATERAL (
    SELECT c.*
    FROM ledger_corrections c
    WHERE c.target_table = 'transaction_lines' AND c.target_id = tl.id
    ORDER BY c.created_at DESC, c.id DESC
    LIMIT 1
) correction ON true;

DROP TRIGGER IF EXISTS trg_transactions_original_append_only ON transactions;
CREATE TRIGGER trg_transactions_original_append_only
BEFORE UPDATE OR DELETE ON transactions
FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change();

DROP TRIGGER IF EXISTS trg_transaction_lines_original_append_only ON transaction_lines;
CREATE TRIGGER trg_transaction_lines_original_append_only
BEFORE UPDATE OR DELETE ON transaction_lines
FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change();

COMMIT;

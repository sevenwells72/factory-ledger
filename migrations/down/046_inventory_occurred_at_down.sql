-- Down migration for 046_inventory_occurred_at.sql.
-- Restores the migration-039 trigger functions exactly.

BEGIN;

CREATE OR REPLACE FUNCTION public.ledger_fill_transaction_business_time()
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

CREATE OR REPLACE FUNCTION public.ledger_enforce_created_at()
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

COMMIT;

-- Optional inventory occurrence time and explicit API backfill provenance.
--
-- `occurred_at` is event time. `created_at` remains database entry time and
-- is still forced to clock_timestamp() for every insert.

BEGIN;

CREATE OR REPLACE FUNCTION public.ledger_fill_transaction_business_time()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- Preserve an event time supplied by the inventory write endpoint. The
    -- legacy timestamp-derived value remains the exact fallback for callers
    -- that omit occurred_at.
    IF NEW.occurred_at IS NULL THEN
        NEW.occurred_at := COALESCE(
            NEW."timestamp" AT TIME ZONE 'UTC',
            clock_timestamp()
        );
    END IF;
    IF NEW.business_date IS NULL THEN
        NEW.business_date := timezone('America/New_York', NEW.occurred_at)::date;
    END IF;
    IF NEW.operator_id IS NULL OR btrim(NEW.operator_id) = '' THEN
        NEW.operator_id := COALESCE(
            NULLIF(current_setting('app.operator_id', true), ''),
            'legacy-shared-key'
        );
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
        -- Entry time remains database-owned. Only the existing provenance
        -- marker is caller-selectable, and only for transaction backfills.
        NEW.created_at := clock_timestamp();
        IF TG_TABLE_NAME <> 'transactions'
           OR NEW.created_at_source IS DISTINCT FROM 'api_backfill' THEN
            NEW.created_at_source := 'database';
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

-- BEGIN 8/17 RECON BACKFILL MARKER
-- These transactions were intentionally posted on 2026-08-17 for earlier
-- business dates. Migration 039's created_at stamp is not usable as their
-- entry time, so give only this documented recon set the explicit backfill
-- provenance consumed by Activity. Both write guards are restored before the
-- transaction can commit.
ALTER TABLE public.transactions
    DISABLE TRIGGER trg_transactions_original_append_only;
ALTER TABLE public.transactions
    DISABLE TRIGGER trg_transactions_created_at;

UPDATE public.transactions
SET created_at_source = 'api_backfill'
WHERE created_at_source IN ('database', 'migration_backfill_039')
  AND operator_id = 'inv-recon-2026-08-17'
  AND notes LIKE '%INV-RECON-2026-08-17%'
  AND (created_at AT TIME ZONE 'America/New_York')::date = DATE '2026-08-17'
  AND business_date BETWEEN DATE '2026-07-24' AND DATE '2026-08-14';

ALTER TABLE public.transactions
    ENABLE TRIGGER trg_transactions_created_at;
ALTER TABLE public.transactions
    ENABLE TRIGGER trg_transactions_original_append_only;
-- END 8/17 RECON BACKFILL MARKER

COMMIT;

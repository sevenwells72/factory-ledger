-- Optional inventory occurrence time and explicit backfill provenance.
--
-- `occurred_at` is event time. `created_at` remains database entry time and
-- is still forced to clock_timestamp() for every insert.

BEGIN;

ALTER TABLE public.transactions
    ADD COLUMN IF NOT EXISTS entry_backfilled boolean NOT NULL DEFAULT false;

COMMENT ON COLUMN public.transactions.entry_backfilled IS
    'TRUE only for an intentional reconstruction/backfill. '
    'created_at_source=migration_backfill_039 does not imply a backfill; '
    'those rows retain their entry time in the legacy timestamp column.';

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

-- BEGIN 8/17 RECON BACKFILL MARKER
-- These transactions were intentionally posted on 2026-08-17 for earlier
-- business dates. They were inserted after migration 039, so their database-
-- owned created_at is the reliable entry time. (For rows that actually carry
-- created_at_source='migration_backfill_039', readers must instead use the
-- surviving legacy timestamp; that separate rule is implemented in Activity.)
-- Measured 2026-08-25: exactly 77 rows: 72 dated 2026-08-14, one each dated
-- 2026-07-24/29/30, and two dated 2026-05-12. Abort rather than partially
-- classify the session if production no longer matches that measured set.
ALTER TABLE public.transactions
    DISABLE TRIGGER trg_transactions_original_append_only;

DO $$
DECLARE
    recon_count integer;
BEGIN
    SELECT count(*)
      INTO recon_count
      FROM public.transactions
     WHERE operator_id = 'inv-recon-2026-08-17'
       AND notes LIKE '%INV-RECON-2026-08-17%'
       AND (created_at AT TIME ZONE 'America/New_York')::date = DATE '2026-08-17'
       AND business_date < DATE '2026-08-17';

    IF recon_count <> 77 THEN
        RAISE EXCEPTION
            'migration 046 expected 77 inventory-recon rows, found %',
            recon_count;
    END IF;

    UPDATE public.transactions
       SET entry_backfilled = true
     WHERE operator_id = 'inv-recon-2026-08-17'
       AND notes LIKE '%INV-RECON-2026-08-17%'
       AND (created_at AT TIME ZONE 'America/New_York')::date = DATE '2026-08-17'
       AND business_date < DATE '2026-08-17'
       AND NOT entry_backfilled;
END;
$$;

ALTER TABLE public.transactions
    ENABLE TRIGGER trg_transactions_original_append_only;
-- END 8/17 RECON BACKFILL MARKER

COMMIT;

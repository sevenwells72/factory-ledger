-- 051: Orthogonal sales-order state model
--      (docs/design/so-state-model-findings.md)
--
-- Adds a stored administrative State alongside the legacy operational
-- `status`, so an order can be taken off the board for a reason that has
-- nothing to do with how far it got operationally:
--
--   state         open | closed | cancelled
--   state_reason  why it left 'open' (constrained per state, see below)
--   state_note    free text; required when reason = 'other'
--   state_changed_at / state_changed_by
--   related_so_id the other order, for reason duplicate | superseded
--
-- `status` is NOT renamed, dropped, or altered, and its CHECK constraint is
-- untouched: every existing reader (§A2 of the findings) keeps working. The
-- application mirrors state onto status for the deprecation window, and
-- status_before_exit preserves the pre-exit status so reopen can restore it
-- (owner ruling 2) instead of flattening every reopened order to 'confirmed'.
--
-- Fulfillment and Health are DERIVED at read time and are deliberately not
-- stored here — they are functions of the ledger, not of administrator intent.
--
-- Deploy order (row-117 precedent, same as migration 050): apply to prod via
-- the session pooler on port 5432 BEFORE merging/deploying code that reads the
-- new columns; then re-dump tests/schema/schema.sql via
-- scripts/dump_prod_schema.sh.
--
-- Idempotent and re-runnable: every DDL step is IF NOT EXISTS, and the
-- backfill is skipped outright once its marker rows exist.

BEGIN;

-- ─────────────────────────────────────────────────────────────────
-- Columns
-- ─────────────────────────────────────────────────────────────────

ALTER TABLE public.sales_orders
    ADD COLUMN IF NOT EXISTS state text NOT NULL DEFAULT 'open',
    ADD COLUMN IF NOT EXISTS state_reason text,
    ADD COLUMN IF NOT EXISTS state_note text,
    ADD COLUMN IF NOT EXISTS state_changed_at timestamptz,
    ADD COLUMN IF NOT EXISTS state_changed_by text,
    ADD COLUMN IF NOT EXISTS related_so_id integer,
    ADD COLUMN IF NOT EXISTS status_before_exit text;

-- ─────────────────────────────────────────────────────────────────
-- Constraints
--
-- Added separately (not inline) so re-running the migration does not fail on
-- an already-present constraint: ADD COLUMN IF NOT EXISTS skips the column but
-- ADD CONSTRAINT has no IF NOT EXISTS.
-- ─────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'sales_orders_state_check') THEN
        ALTER TABLE public.sales_orders
            ADD CONSTRAINT sales_orders_state_check
            CHECK (state IN ('open', 'closed', 'cancelled'));
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'sales_orders_state_reason_check') THEN
        ALTER TABLE public.sales_orders
            ADD CONSTRAINT sales_orders_state_reason_check
            CHECK (state_reason IS NULL OR state_reason IN (
                'shipped_recorded', 'shipped_not_recorded', 'short_closed',
                'customer_cancelled', 'cns_declined', 'duplicate',
                'superseded', 'other'));
    END IF;

    -- The reason must match the state it explains: 'open' has nothing to
    -- explain, the three shipping-shaped reasons belong to 'closed', and the
    -- five intent-shaped reasons belong to 'cancelled'.
    --
    -- The IS NOT NULL on each terminal branch is load-bearing, not decoration.
    -- Without it, state='closed' with a NULL reason evaluates to
    -- (FALSE) OR (TRUE AND NULL) OR (FALSE) = NULL, and a CHECK admits NULL —
    -- only an explicit FALSE rejects a row. That would let a terminal state
    -- exist with no recorded reason, which is the one thing this column is for.
    -- The trailing IS TRUE makes the whole expression three-valued-safe even if
    -- a future branch is added without the same care.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'sales_orders_state_reason_matches_state') THEN
        ALTER TABLE public.sales_orders
            ADD CONSTRAINT sales_orders_state_reason_matches_state
            CHECK (
                (
                    (state = 'open'      AND state_reason IS NULL)
                 OR (state = 'closed'    AND state_reason IS NOT NULL
                                         AND state_reason IN (
                        'shipped_recorded', 'shipped_not_recorded', 'short_closed'))
                 OR (state = 'cancelled' AND state_reason IS NOT NULL
                                         AND state_reason IN (
                        'customer_cancelled', 'cns_declined', 'duplicate',
                        'superseded', 'other'))
                ) IS TRUE
            );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'sales_orders_related_so_id_fkey') THEN
        ALTER TABLE public.sales_orders
            ADD CONSTRAINT sales_orders_related_so_id_fkey
            FOREIGN KEY (related_so_id) REFERENCES public.sales_orders(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_sales_orders_state
    ON public.sales_orders (state);

-- ─────────────────────────────────────────────────────────────────
-- Applied-migration markers
--
-- This repo had no applied-migration table; migrations were re-run by hand and
-- each guarded itself. That works for pure DDL (IF NOT EXISTS) but not for a
-- data backfill, which cannot tell "already backfilled" from "backfilled, and
-- the rows have legitimately moved on since". Gating on the order rows
-- themselves is exactly that mistake: once a backfilled order is reopened or
-- re-closed by hand, a row-derived guard stops recognising its own work.
--
-- Introduced here, deliberately generic, so later migrations can use it.
-- ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.migration_markers (
    name       text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

COMMENT ON TABLE public.migration_markers IS
    'One row per applied one-shot migration step (data backfills especially). Durable and independent of the rows a backfill touched, so re-running a migration is a no-op even after those rows have changed.';

-- ─────────────────────────────────────────────────────────────────
-- Comments
-- ─────────────────────────────────────────────────────────────────

COMMENT ON COLUMN public.sales_orders.state IS
    'Administrative state, orthogonal to the operational status column: open | closed | cancelled. Fulfillment and health are derived at read time and are not stored.';
COMMENT ON COLUMN public.sales_orders.state_reason IS
    'Why the order left open. closed: shipped_recorded | shipped_not_recorded | short_closed. cancelled: customer_cancelled | cns_declined | duplicate | superseded | other. NULL while open.';
COMMENT ON COLUMN public.sales_orders.state_note IS
    'Free-text note on the state change; required by the API when state_reason = other.';
COMMENT ON COLUMN public.sales_orders.state_changed_by IS
    'Surface or SELF-REPORTED identity, never an authenticated one: the caller-supplied changed_by verbatim, else caller_source_tag() (dashboard | a caller tag | NULL). Never the shared-credential placeholder that _operator_id() returns. Real per-user attribution is blocked on FR-15 — do not use this as an audit control against a hostile actor.';
COMMENT ON COLUMN public.sales_orders.related_so_id IS
    'The other sales order, required by the API when state_reason is duplicate or superseded.';
COMMENT ON COLUMN public.sales_orders.status_before_exit IS
    'The operational status captured immediately before close/cancel mirrored over it; reopen restores from here (fallback confirmed) and clears it. NULL while open.';

-- ─────────────────────────────────────────────────────────────────
-- Backfill
--
-- Maps the legacy status onto the new state. For shipped/invoiced the split
-- between shipped_recorded and shipped_not_recorded is decided by the LEDGER,
-- not by sales_order_lines.quantity_shipped_lb: an order whose shipment was
-- later voided has recorded pounds that no longer exist in the ledger, and it
-- is the ledger that tells the truth. Effective remaining is therefore
-- computed exactly the way _line_shipped_effective() does in main.py —
-- through ledger_current_transactions with effective_status = 'posted' —
-- over non-cancelled lines only.
--
-- now() (transaction start) rather than clock_timestamp() is deliberate here:
-- every backfilled row should carry the same stamp, because they were all
-- decided by one event. The API path uses clock_timestamp() instead, so two
-- state changes in one request are distinguishable.
--
-- Idempotent: skipped entirely once any migration-051 marker row exists.
-- ─────────────────────────────────────────────────────────────────

DO $$
DECLARE
    n_open      integer := 0;
    n_recorded  integer := 0;
    n_not_rec   integer := 0;
    n_cancelled integer := 0;
    n_other     integer := 0;
BEGIN
    IF EXISTS (SELECT 1 FROM public.migration_markers
                WHERE name = '051_sales_order_state_backfill') THEN
        RAISE NOTICE '051 backfill: marker present, already applied — skipping.';
        RETURN;
    END IF;

    -- open: new | confirmed | in_production | ready | partial_ship
    UPDATE public.sales_orders so
       SET state             = 'open',
           state_reason      = NULL,
           state_note        = 'backfilled from legacy status=' || so.status,
           state_changed_by  = 'migration-051',
           state_changed_at  = now()
     WHERE so.status IN ('new', 'confirmed', 'in_production', 'ready', 'partial_ship')
       AND so.state_changed_at IS NULL;
    GET DIAGNOSTICS n_open = ROW_COUNT;

    -- closed: shipped | invoiced, split on effective remaining from the ledger.
    -- 0.0001 is main.py's BALANCE_EPSILON.
    WITH effective AS (
        SELECT sol.sales_order_id,
               SUM(GREATEST(
                   sol.quantity_lb - COALESCE((
                       SELECT SUM(ABS(tl.quantity_lb))
                         FROM public.sales_order_shipments sos
                         JOIN public.ledger_current_transactions ct
                           ON ct.id = sos.transaction_id
                          AND ct.effective_status = 'posted'
                          AND ct.type = 'ship'
                         JOIN public.ledger_current_transaction_lines tl
                           ON tl.transaction_id = sos.transaction_id
                          AND tl.product_id = sol.product_id
                        WHERE sos.sales_order_line_id = sol.id
                   ), 0),
                   0)) AS remaining_effective_lb
          FROM public.sales_order_lines sol
          JOIN public.products p ON p.id = sol.product_id
         WHERE sol.line_status <> 'cancelled'
           AND NOT COALESCE(p.is_service, false)
         GROUP BY sol.sales_order_id
    )
    UPDATE public.sales_orders so
       SET state            = 'closed',
           state_reason     = CASE
               WHEN COALESCE((SELECT e.remaining_effective_lb
                                FROM effective e
                               WHERE e.sales_order_id = so.id), 0) <= 0.0001
                   THEN 'shipped_recorded'
               ELSE 'shipped_not_recorded'
           END,
           state_note       = 'backfilled from legacy status=' || so.status,
           state_changed_by = 'migration-051',
           state_changed_at = now()
     WHERE so.status IN ('shipped', 'invoiced')
       AND so.state_changed_at IS NULL;

    SELECT count(*) FILTER (WHERE state_reason = 'shipped_recorded'),
           count(*) FILTER (WHERE state_reason = 'shipped_not_recorded')
      INTO n_recorded, n_not_rec
      FROM public.sales_orders
     WHERE state_changed_by = 'migration-051' AND state = 'closed';

    -- cancelled: reason 'other' — the legacy column never recorded why
    UPDATE public.sales_orders so
       SET state            = 'cancelled',
           state_reason      = 'other',
           state_note        = 'backfilled from legacy status=' || so.status,
           state_changed_by  = 'migration-051',
           state_changed_at  = now()
     WHERE so.status = 'cancelled'
       AND so.state_changed_at IS NULL;
    GET DIAGNOSTICS n_cancelled = ROW_COUNT;

    -- Nothing should be left: the status CHECK admits exactly these 8 values.
    SELECT count(*) INTO n_other
      FROM public.sales_orders
     WHERE state_changed_at IS NULL;

    INSERT INTO public.migration_markers (name)
         VALUES ('051_sales_order_state_backfill');

    RAISE NOTICE '051 backfill counts:';
    RAISE NOTICE '  status(new,confirmed,in_production,ready,partial_ship) -> state=open                             : %', n_open;
    RAISE NOTICE '  status(shipped,invoiced)                               -> state=closed reason=shipped_recorded    : %', n_recorded;
    RAISE NOTICE '  status(shipped,invoiced)                               -> state=closed reason=shipped_not_recorded: %', n_not_rec;
    RAISE NOTICE '  status(cancelled)                                      -> state=cancelled reason=other            : %', n_cancelled;
    RAISE NOTICE '  unmapped rows left untouched (expected 0)                                                         : %', n_other;
END $$;

COMMIT;

-- 053: production_runs + run_coverage — scheduling S1
--      (docs/design/scheduling-spec-draft.md, "Part 2 — proposed design")
--
-- A RUN is one finished SKU, one planned quantity, on one day. COVERAGE links
-- pounds of a run to individual sales_order_lines, so a later step (S2) can
-- tell "short and nobody is making it" from "short but scheduled".
--
-- NOTHING HERE IS READ BY ANY EXISTING QUERY. Health, readiness, availability
-- and every dashboard read are unchanged until S2; this migration only adds
-- the two tables the S1 endpoints write.
--
-- production_schedule / POST /schedule (migration 004) are NOT touched: not
-- migrated, not dropped, not depended on. production_lines and
-- product_line_assignments are reused as reference data only.
--
-- UNITS (owner ruling, 2026-09-14): planned_qty_lb is canonical and is what
-- coverage and (in S2) Health compare against remaining_lb. The run ALSO
-- records the quantity as the floor stated it — planned_qty + planned_unit
-- ('cases' | 'lb') — and case_size_lb_used, the products.case_size_lb the
-- API multiplied by, so "1,800 cases" survives as entered and the conversion
-- is auditable. run_coverage is pounds only: it is a comparison against a
-- line's remaining pounds and never needs a case count.
--
-- DELIBERATELY NOT MODELLED: run_kind / stage / parent_run_id. A future
-- migration adding kitchen/batch/packing relationships must not have to undo
-- anything here, so no constraint below assumes a run is single-stage.
-- Completion is human-confirmed (POST .../complete); no actual_transaction_id
-- column — evidence is derived at read time from the posted ledger.
--
-- APPLIED BY HAND WITH psql BEFORE THE CODE THAT WRITES IT MERGES:
--     psql "$(cat ~/.config/factory-ledger/db_url)" -v ON_ERROR_STOP=1 \
--          -f migrations/053_production_runs.sql
-- Deliberately no BEGIN/COMMIT (052 pattern): the file is also pasteable
-- into the Supabase SQL editor, which wraps each run in its own transaction.
--
-- Idempotent and re-runnable: CREATE TABLE IF NOT EXISTS, every constraint
-- added through a DO block that checks the catalog first, CREATE INDEX IF NOT
-- EXISTS, and the migration_markers row inserted ON CONFLICT DO NOTHING. A
-- second run performs no writes and reports 'already applied'.
-- No seed rows. No backfill.

-- ─────────────────────────────────────────────────────────────────
-- Tables
-- ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.production_runs (
    id                serial      PRIMARY KEY,
    product_id        integer     NOT NULL REFERENCES public.products(id),
    planned_qty_lb    numeric(14,4) NOT NULL,
    planned_qty       numeric(14,4),
    planned_unit      text,
    case_size_lb_used numeric(14,4),
    planned_date      date        NOT NULL,
    line_id           integer     REFERENCES public.production_lines(id),
    status            text        NOT NULL DEFAULT 'planned',
    notes             text,
    created_at        timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at        timestamptz NOT NULL DEFAULT clock_timestamp(),
    completed_at      timestamptz,
    created_by        text,
    updated_by        text,
    completed_by      text
);

CREATE TABLE IF NOT EXISTS public.run_coverage (
    id                  serial      PRIMARY KEY,
    run_id              integer     NOT NULL REFERENCES public.production_runs(id),
    sales_order_line_id integer     NOT NULL REFERENCES public.sales_order_lines(id),
    qty_lb              numeric(14,4) NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_by          text
);

-- ─────────────────────────────────────────────────────────────────
-- Constraints
--
-- Added separately rather than inline, so a re-run does not fail: CREATE
-- TABLE IF NOT EXISTS skips the whole statement including its constraints,
-- but a table created by an EARLIER, partially-applied run might be missing
-- one. ADD CONSTRAINT has no IF NOT EXISTS, hence the catalog checks.
-- ─────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'production_runs_planned_qty_lb_check'
                      AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs
            ADD CONSTRAINT production_runs_planned_qty_lb_check
            CHECK (planned_qty_lb > 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'production_runs_planned_qty_check'
                      AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs
            ADD CONSTRAINT production_runs_planned_qty_check
            CHECK (planned_qty IS NULL OR planned_qty > 0);
    END IF;

    -- The native unit vocabulary. Only the two units order intake accepts
    -- for a finished SKU today; widening it is a migration, not a config
    -- edit, so a typo cannot silently invent a third unit.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'production_runs_planned_unit_check'
                      AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs
            ADD CONSTRAINT production_runs_planned_unit_check
            CHECK (planned_unit IS NULL OR planned_unit IN ('cases', 'lb'));
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'production_runs_case_size_lb_used_check'
                      AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs
            ADD CONSTRAINT production_runs_case_size_lb_used_check
            CHECK (case_size_lb_used IS NULL OR case_size_lb_used > 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'production_runs_status_check'
                      AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs
            ADD CONSTRAINT production_runs_status_check
            CHECK (status IN ('planned', 'in_progress', 'done', 'cancelled'));
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'run_coverage_qty_lb_check'
                      AND conrelid = 'public.run_coverage'::regclass) THEN
        ALTER TABLE public.run_coverage
            ADD CONSTRAINT run_coverage_qty_lb_check
            CHECK (qty_lb > 0);
    END IF;

    -- One coverage row per (run, line). PUT .../coverage replaces the whole
    -- list, so a duplicate would only ever be a bug.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'run_coverage_run_id_sales_order_line_id_key'
                      AND conrelid = 'public.run_coverage'::regclass) THEN
        ALTER TABLE public.run_coverage
            ADD CONSTRAINT run_coverage_run_id_sales_order_line_id_key
            UNIQUE (run_id, sales_order_line_id);
    END IF;
END $$;

-- ─────────────────────────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_production_runs_planned_date
    ON public.production_runs (planned_date);
CREATE INDEX IF NOT EXISTS idx_production_runs_status
    ON public.production_runs (status);
CREATE INDEX IF NOT EXISTS idx_run_coverage_sales_order_line_id
    ON public.run_coverage (sales_order_line_id);

-- ─────────────────────────────────────────────────────────────────
-- Marker
--
-- ON CONFLICT DO NOTHING so a re-run is a true no-op rather than a duplicate
-- key error. Verify with:
--     select * from migration_markers where name = '053_production_runs';
-- ─────────────────────────────────────────────────────────────────

DO $$
DECLARE
    already boolean;
    n_runs integer;
BEGIN
    -- migration_markers is created by the application at startup
    -- (STARTUP_MIGRATION_MARKERS_DDL, main.py). Create it here too so this
    -- file can be applied to a database the app has not booted against yet.
    CREATE TABLE IF NOT EXISTS public.migration_markers (
        name       text PRIMARY KEY,
        applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );

    SELECT EXISTS (SELECT 1 FROM public.migration_markers
                    WHERE name = '053_production_runs') INTO already;

    INSERT INTO public.migration_markers (name)
         VALUES ('053_production_runs')
    ON CONFLICT (name) DO NOTHING;

    SELECT count(*) INTO n_runs FROM public.production_runs;

    IF already THEN
        RAISE NOTICE '053_production_runs: already applied — no changes made.';
    ELSE
        RAISE NOTICE '053_production_runs: applied.';
    END IF;
    RAISE NOTICE '  production_runs rows: % (0 is expected on first apply — no seed)', n_runs;
END $$;

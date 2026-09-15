-- S1 amendment: additive run types and auditable pan yield. Apply via port 5432.
-- No transaction wrapper: follows 053 / Supabase SQL editor conventions.
ALTER TABLE public.production_runs ADD COLUMN IF NOT EXISTS run_type text NOT NULL DEFAULT 'pack';
ALTER TABLE public.production_runs ADD COLUMN IF NOT EXISTS pan_yield_lb_used numeric(14,4);
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'production_runs_run_type_check'
                   AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs ADD CONSTRAINT production_runs_run_type_check
            CHECK (run_type IN ('bake','pack','coconut','other'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'production_runs_pan_yield_lb_used_check'
                   AND conrelid = 'public.production_runs'::regclass) THEN
        ALTER TABLE public.production_runs ADD CONSTRAINT production_runs_pan_yield_lb_used_check
            CHECK (pan_yield_lb_used IS NULL OR pan_yield_lb_used > 0);
    END IF;
END $$;
ALTER TABLE public.production_runs DROP CONSTRAINT IF EXISTS production_runs_planned_unit_check;
ALTER TABLE public.production_runs ADD CONSTRAINT production_runs_planned_unit_check
    CHECK (planned_unit IS NULL OR planned_unit IN ('cases','lb','pans'));
DO $$
DECLARE
    already boolean;
    n_runs integer;
    counts text;
BEGIN
    CREATE TABLE IF NOT EXISTS public.migration_markers (
        name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    SELECT EXISTS (SELECT 1 FROM public.migration_markers WHERE name='054_run_type') INTO already;
    INSERT INTO public.migration_markers (name) VALUES ('054_run_type') ON CONFLICT DO NOTHING;
    SELECT count(*) INTO n_runs FROM public.production_runs;
    SELECT string_agg(run_type || ': ' || n, ', ' ORDER BY run_type) INTO counts
      FROM (SELECT run_type, count(*) AS n FROM public.production_runs GROUP BY run_type) c;
    RAISE NOTICE '054_run_type: already applied = %, production_runs rows = %, counts = %', already, n_runs, COALESCE(counts, 'none');
END $$;

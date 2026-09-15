-- Manual rollback only. Never deletes runs or coverage.
BEGIN;
DO $$
DECLARE r record; blocked boolean := false;
BEGIN
    FOR r IN SELECT id, run_type, planned_unit FROM public.production_runs
              WHERE run_type <> 'pack' OR planned_unit = 'pans'
    LOOP
        blocked := true;
        RAISE NOTICE 'Rollback blocked: run %, type %, unit %', r.id, r.run_type, r.planned_unit;
    END LOOP;
    IF blocked THEN
        RAISE EXCEPTION '054 rollback refused: non-pack or pans runs exist. Any deletion requires separate explicit approval.';
    END IF;
END $$;
ALTER TABLE public.production_runs DROP CONSTRAINT IF EXISTS production_runs_planned_unit_check;
ALTER TABLE public.production_runs ADD CONSTRAINT production_runs_planned_unit_check
    CHECK (planned_unit IS NULL OR planned_unit IN ('cases','lb'));
ALTER TABLE public.production_runs DROP CONSTRAINT IF EXISTS production_runs_run_type_check;
ALTER TABLE public.production_runs DROP COLUMN run_type;
ALTER TABLE public.production_runs DROP CONSTRAINT IF EXISTS production_runs_pan_yield_lb_used_check;
ALTER TABLE public.production_runs DROP COLUMN pan_yield_lb_used;
DELETE FROM public.migration_markers WHERE name = '054_run_type';
COMMIT;

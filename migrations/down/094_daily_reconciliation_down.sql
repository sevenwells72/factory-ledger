-- Reverses 094_daily_reconciliation. Refuses while any run, snapshot or flag
-- exists: they are the nightly check's record and are never discarded silently
-- (export them first). Run in one transaction as the table owner on port 5432.
-- Touches only the three reconciliation tables and the marker.
DO $$
DECLARE t text; has_rows boolean;
BEGIN
    FOREACH t IN ARRAY ARRAY['reconciliation_flags', 'inventory_snapshots', 'reconciliation_runs'] LOOP
        IF to_regclass(t) IS NOT NULL THEN
            EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I)', t) INTO has_rows;
            IF has_rows THEN
                RAISE EXCEPTION '% holds rows; export them before reversing 094', t;
            END IF;
        END IF;
    END LOOP;
END $$;
DROP TABLE IF EXISTS reconciliation_flags;
DROP TABLE IF EXISTS inventory_snapshots;
DROP TABLE IF EXISTS reconciliation_runs;
DELETE FROM migration_markers WHERE name = '094_daily_reconciliation';

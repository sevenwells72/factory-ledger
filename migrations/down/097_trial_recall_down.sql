-- Reverses 097_trial_recall. Refuses while any saved run, Form row or link
-- exists: they are the record of what the jobs read and found and are never
-- discarded silently (export them first). Run in one transaction as the table
-- owner on port 5432. Touches only the four 097 tables and the marker.
DO $$
DECLARE t text; has_rows boolean;
BEGIN
    FOREACH t IN ARRAY ARRAY['receipt_bol_links', 'receiving_form_rows', 'receiving_form_sync_runs',
                             'trial_recall_runs'] LOOP
        IF to_regclass(t) IS NOT NULL THEN
            EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I)', t) INTO has_rows;
            IF has_rows THEN
                RAISE EXCEPTION '% holds rows; export them before reversing 097', t;
            END IF;
        END IF;
    END LOOP;
END $$;
DROP TABLE IF EXISTS receipt_bol_links;
DROP TABLE IF EXISTS receiving_form_rows;
DROP TABLE IF EXISTS receiving_form_sync_runs;
DROP TABLE IF EXISTS trial_recall_runs;
DELETE FROM migration_markers WHERE name = '097_trial_recall';

-- Deferred supplier-label backfill: NOT an A5 application prerequisite.
-- Apply only AFTER Michael approves the Dutch Valley duplicate / DUTC Valley
-- typo cleanup. Re-run scripts/dry_run_supplier_labels.py against that catalog.
-- Apply explicitly as table owner in ONE transaction, ON_ERROR_STOP, with:
-- SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public;
-- SET LOCAL factory_ledger.supplier_cleanup_confirmed='michael_approved';
-- No historical lot prefix or supplier FK is rewritten.
DO $$ BEGIN
    IF COALESCE(current_setting('factory_ledger.supplier_cleanup_confirmed', true), '') <> 'michael_approved' THEN
        RAISE EXCEPTION '066 deferred until Michael approves supplier cleanup'
            USING HINT='Clean up Dutch Valley duplicates and DUTC Valley typo, rerun the label dry-run, then explicitly acknowledge cleanup in this transaction.';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM migration_markers WHERE name='064_unidentified_lots') THEN
        RAISE EXCEPTION 'Apply A5 schema migration 064 before supplier label backfill';
    END IF;
END $$;

-- Allocate a unique display label, retaining the natural four-letter token
-- whenever available. Collisions receive another label; identity stays the ID.
CREATE OR REPLACE FUNCTION a5_supplier_display_code(supplier_name text) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE candidate text; base text; n integer := -1;
BEGIN
    PERFORM pg_advisory_xact_lock(6505);
    base := rpad(left(regexp_replace(upper(supplier_name),'[^A-Z]','','g'),4),4,'X');
    candidate := base;
    WHILE EXISTS(SELECT 1 FROM suppliers WHERE short_code=candidate) LOOP
        n := n+1;
        IF n >= 456976 THEN RAISE EXCEPTION 'Supplier display labels exhausted'; END IF;
        IF n < 26 THEN candidate := left(base,3) || chr(65+n);
        ELSE candidate := chr(65+(n/17576)%26) || chr(65+(n/676)%26) || chr(65+(n/26)%26) || chr(65+n%26);
        END IF;
    END LOOP;
    RETURN candidate;
END $$;
DO $$
DECLARE row record;
BEGIN
    FOR row IN SELECT id,name FROM suppliers WHERE short_code IS NULL
        AND btrim(supplier_name_norm(name)) <> ALL(ARRAY['found','found inventory','inventory found','physical count',
            'initial inventory','inventory correction','inventory intake','unknown']) ORDER BY id LOOP
        UPDATE suppliers SET short_code=a5_supplier_display_code(row.name) WHERE id=row.id;
    END LOOP;
END $$;
CREATE OR REPLACE FUNCTION a5_supplier_default_display_code() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.short_code IS NULL AND btrim(supplier_name_norm(NEW.name)) <> ALL(ARRAY['found','found inventory',
        'inventory found','physical count','initial inventory','inventory correction','inventory intake','unknown']) THEN
        NEW.short_code := a5_supplier_display_code(NEW.name);
    END IF;
    RETURN NEW;
END $$;
DO $$ BEGIN
    IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='suppliers'::regclass AND tgname='a5_supplier_default_display_code') THEN
        CREATE TRIGGER a5_supplier_default_display_code BEFORE INSERT ON suppliers
            FOR EACH ROW EXECUTE FUNCTION a5_supplier_default_display_code();
    END IF;
END $$;
INSERT INTO migration_markers(name) VALUES ('066_supplier_labels') ON CONFLICT DO NOTHING;

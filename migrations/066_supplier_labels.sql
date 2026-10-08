-- Deferred supplier-label backfill: NOT an A5 application prerequisite.
-- Apply only AFTER Michael's supplier cleanup (scripts/supplier_cleanup_dutch_2026_10_08.sql:
-- Dutch Valley duplicates + DUTC Valley typo merged into 16 Dutch Valley Foods = DUTV,
-- Dutch Gold merged into 13 Dutch Gold Honey = DUTG, duplicates deactivated).
-- Re-run scripts/dry_run_supplier_labels.py against that catalog first.
-- Apply explicitly as table owner in ONE transaction, ON_ERROR_STOP, with:
-- SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public;
-- SET LOCAL factory_ledger.supplier_cleanup_confirmed='michael_approved';
-- Rules: an explicit (non-NULL) short_code is never overwritten; inactive suppliers
-- are skipped (a label is assigned when a supplier becomes active); pseudo-suppliers
-- never get a label. No historical lot prefix or supplier FK is rewritten.
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
-- Explicit labels already in suppliers.short_code are reserved by the EXISTS probe.
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
-- Backfill: ACTIVE suppliers without a label, in id order. Explicit labels
-- (short_code IS NOT NULL) and deactivated suppliers are left untouched.
DO $$
DECLARE row record;
BEGIN
    FOR row IN SELECT id,name FROM suppliers WHERE short_code IS NULL AND active
        AND btrim(supplier_name_norm(name)) <> ALL(ARRAY['found','found inventory','inventory found','physical count',
            'initial inventory','inventory correction','inventory intake','unknown']) ORDER BY id LOOP
        UPDATE suppliers SET short_code=a5_supplier_display_code(row.name) WHERE id=row.id;
    END LOOP;
END $$;
-- Future suppliers: label on INSERT when active, or when a label-less supplier is
-- (re)activated. An explicit short_code supplied by the caller is kept as-is.
CREATE OR REPLACE FUNCTION a5_supplier_default_display_code() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.short_code IS NULL AND NEW.active AND btrim(supplier_name_norm(NEW.name)) <> ALL(ARRAY['found','found inventory',
        'inventory found','physical count','initial inventory','inventory correction','inventory intake','unknown']) THEN
        NEW.short_code := a5_supplier_display_code(NEW.name);
    END IF;
    RETURN NEW;
END $$;
-- Re-created on every run so an earlier INSERT-only version (staging, 2026-10-08) is replaced.
DROP TRIGGER IF EXISTS a5_supplier_default_display_code ON suppliers;
CREATE TRIGGER a5_supplier_default_display_code BEFORE INSERT OR UPDATE OF active ON suppliers
    FOR EACH ROW EXECUTE FUNCTION a5_supplier_default_display_code();
INSERT INTO migration_markers(name) VALUES ('066_supplier_labels') ON CONFLICT DO NOTHING;

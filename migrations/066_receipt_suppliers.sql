-- A5 part 4. Apply explicitly after 064 as the app/table owner, in one
-- transaction (ON_ERROR_STOP, SET LOCAL lock_timeout='5s', search_path=public).
-- Historical ledger/lot supplier links remain NULL; never guess from prefixes.
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS short_code text
    CHECK (short_code ~ '^[A-Z]{4}$');
CREATE UNIQUE INDEX IF NOT EXISTS suppliers_short_code_unique ON suppliers(short_code);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS supplier_id integer REFERENCES suppliers(id);
ALTER TABLE lots ADD COLUMN IF NOT EXISTS supplier_id integer REFERENCES suppliers(id);
ALTER TABLE lot_supplier_codes ADD COLUMN IF NOT EXISTS supplier_id integer REFERENCES suppliers(id);
CREATE INDEX IF NOT EXISTS transactions_supplier_idx ON transactions(supplier_id) WHERE supplier_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS lots_supplier_idx ON lots(supplier_id) WHERE supplier_id IS NOT NULL;

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
CREATE OR REPLACE FUNCTION a5_lot_supplier_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.supplier_id IS DISTINCT FROM OLD.supplier_id THEN
        RAISE EXCEPTION 'Lot supplier identity is set at insert; use a new lot' USING ERRCODE='23000';
    END IF;
    RETURN NEW;
END $$;
DO $$ BEGIN
    IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='lots'::regclass AND tgname='a5_lot_supplier_immutable') THEN
        CREATE TRIGGER a5_lot_supplier_immutable BEFORE UPDATE OF supplier_id ON lots
            FOR EACH ROW EXECUTE FUNCTION a5_lot_supplier_immutable();
    END IF;
END $$;
INSERT INTO migration_markers(name) VALUES ('066_receipt_suppliers') ON CONFLICT DO NOTHING;

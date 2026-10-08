-- A5 part 3. Explicit additive migration after 063, app/table owner, one
-- transaction with SET LOCAL lock_timeout='5s', search_path=public.
-- History remains NULL (unassessed); no historical lot is declared identified.
ALTER TABLE lots ADD COLUMN IF NOT EXISTS identity_status text
    CHECK (identity_status IN ('identified','unidentified'));
ALTER TABLE lots ADD COLUMN IF NOT EXISTS identify_by date;
CREATE INDEX IF NOT EXISTS lots_unidentified_due_idx ON lots(identify_by)
    WHERE identity_status='unidentified';
-- Run scripts/check_unidentified_lots_preapply.py first. Repeat the check here
-- so direct application also fails before creating the partial unique index.
DO $$ BEGIN
    IF EXISTS (SELECT lot_id FROM exceptions
               WHERE kind='UNIDENTIFIED_LOT' AND status IN ('open','escalated')
                 AND lot_id IS NOT NULL GROUP BY lot_id HAVING count(*) > 1) THEN
        RAISE EXCEPTION 'Duplicate open UNIDENTIFIED_LOT exceptions exist'
            USING HINT='Run scripts/check_unidentified_lots_preapply.py; reconcile duplicates before applying 064.';
    END IF;
END $$;
-- Reuse 061 exceptions, with a partial unique key for concurrent top-ups.
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_unidentified_open_lot_idx ON exceptions(lot_id)
    WHERE kind='UNIDENTIFIED_LOT' AND status IN ('open','escalated');
-- A5 part 4 schema only: receipt provenance works before any label backfill.
-- Historical supplier links stay NULL. Labels stay nullable until Michael's
-- supplier cleanup and the separately reviewed 066 migration; never infer IDs.
ALTER TABLE suppliers ADD COLUMN IF NOT EXISTS short_code text
    CHECK (short_code ~ '^[A-Z]{4}$');
CREATE UNIQUE INDEX IF NOT EXISTS suppliers_short_code_unique ON suppliers(short_code);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS supplier_id integer REFERENCES suppliers(id);
ALTER TABLE lots ADD COLUMN IF NOT EXISTS supplier_id integer REFERENCES suppliers(id);
ALTER TABLE lot_supplier_codes ADD COLUMN IF NOT EXISTS supplier_id integer REFERENCES suppliers(id);
CREATE INDEX IF NOT EXISTS transactions_supplier_idx ON transactions(supplier_id) WHERE supplier_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS lots_supplier_idx ON lots(supplier_id) WHERE supplier_id IS NOT NULL;

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
INSERT INTO migration_markers(name) VALUES ('064_unidentified_lots') ON CONFLICT DO NOTHING;

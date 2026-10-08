-- A5 part 2, additive. Apply after 062 in one transaction, app/table owner,
-- SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public; ON_ERROR_STOP.
CREATE TABLE IF NOT EXISTS transaction_substitutions (
    transaction_id integer NOT NULL REFERENCES transactions(id),
    ingredient_product_id integer NOT NULL REFERENCES products(id),
    substitute_product_id integer REFERENCES products(id),
    lot_id integer REFERENCES lots(id),
    reason_code text NOT NULL CHECK (btrim(reason_code)<>''),
    note text,
    actor_id integer REFERENCES actors(id),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (transaction_id,ingredient_product_id),
    CHECK ((substitute_product_id IS NULL) = (lot_id IS NULL)),
    CHECK (substitute_product_id IS DISTINCT FROM ingredient_product_id)
);
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='transaction_substitutions'::regclass AND tgname='a5_evidence_append_only') THEN
        CREATE TRIGGER a5_evidence_append_only BEFORE UPDATE OR DELETE ON transaction_substitutions
            FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change();
    END IF;
END $$;
ALTER TABLE transaction_substitutions ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON transaction_substitutions FROM PUBLIC;
INSERT INTO migration_markers(name) VALUES ('063_batch_substitutions') ON CONFLICT DO NOTHING;

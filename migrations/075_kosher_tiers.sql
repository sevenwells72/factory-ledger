-- A12, rev 3.8: explicit apply as app/table owner; STAGING ONLY for builder.
-- BEGIN; SET LOCAL search_path=public; SET LOCAL lock_timeout='5s';
-- SET LOCAL statement_timeout='60s'; run with ON_ERROR_STOP; COMMIT.
-- Additive. No startup hook. Formula 284 and historical ledger untouched.
ALTER TABLE products ADD COLUMN IF NOT EXISTS kosher_tier text NOT NULL DEFAULT 'regular'
    CHECK (kosher_tier IN ('regular','extra_kosher'));
ALTER TABLE lots ADD COLUMN IF NOT EXISTS kosher_tier text NOT NULL DEFAULT 'regular'
    CHECK (kosher_tier IN ('regular','extra_kosher'));
ALTER TABLE lots ADD COLUMN IF NOT EXISTS kosher_attested_by_actor_id integer REFERENCES actors(id);
ALTER TABLE lots ADD COLUMN IF NOT EXISTS kosher_attested_at timestamptz;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS kosher_tier text NOT NULL DEFAULT 'regular'
    CHECK (kosher_tier IN ('regular','extra_kosher'));
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS kosher_attested_by_actor_id integer REFERENCES actors(id);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS kosher_attested_at timestamptz;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS transaction_kosher_downgrade boolean NOT NULL DEFAULT false;

-- Catalog-only one-time update. Existing lots remain regular until the paper-log
-- cutover reconciliation; never infer historical procedure from today's SKU.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM migration_markers WHERE name='075_kosher_tiers') THEN
        UPDATE products SET kosher_tier='regular'
            WHERE id IN (107,108,136,144,137,138,129,179,143);
        UPDATE products SET kosher_tier='extra_kosher', name=CASE id
            WHEN 283 THEN 'Batch Classic Granola #9 — EXTRA-KOSHER (SS)'
            WHEN 284 THEN 'Batch Classic Chocolate Chip Granola #9 — EXTRA-KOSHER (SS)'
            ELSE regexp_replace(name, '\s*\(Kosher Ignition\)', '', 'gi') || ' — EXTRA-KOSHER (SS)' END
            WHERE id BETWEEN 283 AND 290;
    END IF;
END $$;

CREATE OR REPLACE FUNCTION lot_kosher_tier_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='INSERT' THEN
        SELECT kosher_tier INTO NEW.kosher_tier FROM products WHERE id=NEW.product_id;
    ELSIF NEW.kosher_tier IS DISTINCT FROM OLD.kosher_tier THEN
        RAISE EXCEPTION 'LOT_KOSHER_TIER_IMMUTABLE';
    END IF;
    IF TG_OP='UPDATE' AND OLD.kosher_attested_by_actor_id IS NOT NULL
       AND (NEW.kosher_attested_by_actor_id IS DISTINCT FROM OLD.kosher_attested_by_actor_id
            OR NEW.kosher_attested_at IS DISTINCT FROM OLD.kosher_attested_at) THEN
        RAISE EXCEPTION 'LOT_KOSHER_ATTESTATION_IMMUTABLE';
    END IF;
    RETURN NEW;
END $$;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='lots'::regclass AND tgname='lot_kosher_tier_guard') THEN
        CREATE TRIGGER lot_kosher_tier_guard BEFORE INSERT OR UPDATE ON lots
            FOR EACH ROW EXECUTE FUNCTION lot_kosher_tier_guard();
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS ticket_attestations (
    ticket_id bigint NOT NULL REFERENCES write_tickets(id),
    kind text NOT NULL CHECK (kind='kosher_procedure'),
    actor_id integer NOT NULL REFERENCES actors(id),
    attested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(ticket_id,kind)
);
CREATE TABLE IF NOT EXISTS lot_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lot_id integer NOT NULL REFERENCES lots(id),
    event text NOT NULL CHECK (event IN ('rename','supplier_lot','identified','move','ownership','kosher_downgrade')),
    old jsonb, new jsonb, reason_code text, happened_at timestamptz NOT NULL,
    entered_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    entered_by_actor_id integer REFERENCES actors(id),
    ticket_id bigint REFERENCES write_tickets(id),
    transaction_id integer REFERENCES transactions(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS lot_events_one_kosher_downgrade_idx
    ON lot_events(transaction_id,lot_id) WHERE event='kosher_downgrade';
DO $$ DECLARE definition text; t text; BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='exceptions'::regclass AND conname='exceptions_kind_check';
    IF position('KOSHER_ATTESTATION' in definition)=0 THEN
        ALTER TABLE exceptions DROP CONSTRAINT exceptions_kind_check;
        EXECUTE 'ALTER TABLE exceptions ADD CONSTRAINT exceptions_kind_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR kind = ''KOSHER_ATTESTATION'')';
    END IF;
    FOREACH t IN ARRAY ARRAY['ticket_attestations','lot_events'] LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid=t::regclass AND tgname='a12_append_only') THEN
            EXECUTE format('CREATE TRIGGER a12_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change()',t);
        END IF;
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC',t);
    END LOOP;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_one_kosher_hold_idx
    ON exceptions(ticket_id) WHERE kind='KOSHER_ATTESTATION';
INSERT INTO migration_markers(name) VALUES ('075_kosher_tiers') ON CONFLICT DO NOTHING;

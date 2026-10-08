-- A5 part 1. Explicit apply only, after 061, as app/table owner.
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public;
-- run this file with ON_ERROR_STOP; COMMIT. No startup hook or data rewrite.
-- Extend the ticket action constraint without narrowing any existing action.
DO $$
DECLARE definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='write_tickets'::regclass AND conname='write_tickets_action_check';
    IF position('move_lot' in definition)=0 THEN
        EXECUTE 'ALTER TABLE write_tickets DROP CONSTRAINT write_tickets_action_check';
        EXECUTE 'ALTER TABLE write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
            || substring(definition from 8 for length(definition)-8) || ') OR action = ''move_lot'')';
    END IF;
END $$;
CREATE TABLE IF NOT EXISTS lot_moves (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    lot_id integer NOT NULL REFERENCES lots(id),
    lot_code text NOT NULL,
    to_location text NOT NULL CHECK (to_location IN ('storage','staging','production')),
    moved_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    actor_id integer REFERENCES actors(id),
    ticket_id bigint NOT NULL UNIQUE REFERENCES write_tickets(id),
    method text NOT NULL CHECK (method IN ('last4','full_code','scan')),
    value text NOT NULL CHECK (btrim(value)<>'')
);
CREATE INDEX IF NOT EXISTS lot_moves_latest_idx ON lot_moves(lot_id,moved_at DESC,id DESC);
CREATE TABLE IF NOT EXISTS transaction_lot_confirmations (
    transaction_id integer NOT NULL REFERENCES transactions(id),
    lot_id integer NOT NULL REFERENCES lots(id),
    method text NOT NULL CHECK (method IN ('last4','full_code','scan','pallet')),
    value text NOT NULL CHECK (btrim(value)<>''),
    actor_id integer REFERENCES actors(id),
    move_id bigint REFERENCES lot_moves(id),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(transaction_id,lot_id),
    CHECK ((method='pallet') = (move_id IS NOT NULL))
);
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['lot_moves','transaction_lot_confirmations'] LOOP
        IF NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid=t::regclass AND tgname='a5_evidence_append_only') THEN
            EXECUTE format('CREATE TRIGGER a5_evidence_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION ledger_block_append_only_change()',t);
        END IF;
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',t);
        EXECUTE format('REVOKE ALL ON %I FROM PUBLIC',t);
    END LOOP;
END $$;
INSERT INTO migration_markers(name) VALUES ('062_lot_confirmation') ON CONFLICT DO NOTHING;

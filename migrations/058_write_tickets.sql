-- A1 part 1. Apply as the app/table owner after 057, port 5432, in one
-- transaction (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s', COMMIT).
-- Uses the application search_path (public); explicit SQL runners should
-- SET LOCAL search_path=public. Startup uses the same schema as other app SQL.
-- Additive only: no data sweeps, ledger mutations, or view replacement.
CREATE TABLE IF NOT EXISTS write_tickets (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticket_hash text NOT NULL UNIQUE CHECK (length(ticket_hash) = 64),
    action text NOT NULL CHECK (action IN ('receive','make','pack','adjust','found')),
    actor_id integer REFERENCES actors(id),
    operator_id text NOT NULL,
    key_kind text NOT NULL CHECK (key_kind IN ('actor','legacy_ledger','legacy_dashboard')),
    client_source text NOT NULL CHECK (client_source IN ('mcp','dashboard','fl_assistant','api')),
    payload jsonb NOT NULL,
    payload_hash text NOT NULL CHECK (length(payload_hash) = 64),
    state_hash text NOT NULL CHECK (length(state_hash) = 64),
    draft jsonb NOT NULL,
    warnings jsonb NOT NULL DEFAULT '[]',
    status text NOT NULL DEFAULT 'prepared'
        CHECK (status IN ('prepared','committed','expired','rejected','superseded')),
    prepared_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL,
    committed_at timestamptz,
    receipt_number text UNIQUE,
    result_ref jsonb,
    response jsonb,
    acknowledged jsonb NOT NULL DEFAULT '[]',
    reject_reason text,
    CHECK ((key_kind = 'actor') = (actor_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS write_tickets_actor_prepared_idx
    ON write_tickets (actor_id, prepared_at DESC);
CREATE INDEX IF NOT EXISTS write_tickets_open_idx
    ON write_tickets (status) WHERE status = 'prepared';
CREATE INDEX IF NOT EXISTS write_tickets_supersession_idx
    ON write_tickets (operator_id, action, payload_hash) WHERE status = 'prepared';

CREATE TABLE IF NOT EXISTS receipt_counters (
    prefix text NOT NULL,
    business_date date NOT NULL,
    next integer NOT NULL DEFAULT 1 CHECK (next > 0),
    PRIMARY KEY (prefix, business_date)
);
ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS receipt_number text,
    ADD COLUMN IF NOT EXISTS ticket_id bigint REFERENCES write_tickets(id);
-- Non-unique intentionally: §2 allows a later shipment to post multiple
-- transactions under one receipt. Ticket receipt numbers remain unique.
CREATE INDEX IF NOT EXISTS transactions_receipt_number_idx
    ON transactions (receipt_number) WHERE receipt_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS transactions_ticket_id_idx
    ON transactions (ticket_id) WHERE ticket_id IS NOT NULL;

CREATE OR REPLACE FUNCTION write_ticket_preserve_committed() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status = 'committed' THEN
        RAISE EXCEPTION 'Committed write tickets are immutable' USING ERRCODE = '23000';
    END IF;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END;
$$;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='write_tickets'::regclass
                   AND tgname='write_tickets_preserve_committed') THEN
        CREATE TRIGGER write_tickets_preserve_committed BEFORE UPDATE OR DELETE
            ON write_tickets FOR EACH ROW
            EXECUTE FUNCTION write_ticket_preserve_committed();
    END IF;
END $$;
ALTER TABLE write_tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE receipt_counters ENABLE ROW LEVEL SECURITY;
-- Standalone migration marker; the startup gate writes its own.
INSERT INTO migration_markers (name) VALUES ('058_write_tickets')
ON CONFLICT (name) DO NOTHING;

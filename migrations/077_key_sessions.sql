-- Pilot key sessions only; independent of parked 073 PIN sessions.
-- Apply explicitly as app/table owner. Additive, idempotent; staging only.
CREATE TABLE IF NOT EXISTS actor_sessions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    token_hash text NOT NULL UNIQUE,
    actor_id integer NOT NULL REFERENCES actors(id),
    actor_key_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_seen_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL,
    ended_at timestamptz,
    ended_reason text CHECK (ended_reason IN ('idle','logout','revoked')),
    CHECK ((ended_at IS NULL) = (ended_reason IS NULL))
);
CREATE INDEX IF NOT EXISTS actor_sessions_open_actor ON actor_sessions(actor_id) WHERE ended_at IS NULL;
CREATE TABLE IF NOT EXISTS key_session_attempts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_hash text NOT NULL,
    ok boolean NOT NULL,
    attempted_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS key_session_attempts_time ON key_session_attempts(attempted_at);
ALTER TABLE actor_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_session_attempts ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON actor_sessions,key_session_attempts FROM PUBLIC;
REVOKE ALL ON SEQUENCE actor_sessions_id_seq,key_session_attempts_id_seq FROM PUBLIC;
DO $$ DECLARE r text; BEGIN
    FOREACH r IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname=r) THEN
            EXECUTE format('REVOKE ALL ON actor_sessions,key_session_attempts FROM %I',r);
            EXECUTE format('REVOKE ALL ON SEQUENCE actor_sessions_id_seq,key_session_attempts_id_seq FROM %I',r);
        END IF;
    END LOOP;
END $$;
INSERT INTO migration_markers(name) VALUES ('077_key_sessions') ON CONFLICT DO NOTHING;

-- A11: additive identity/session storage. Apply as app owner, staging first,
-- in an explicit transaction with ON_ERROR_STOP and SET LOCAL lock_timeout='5s'.
-- No PINs, credentials, actor seeds, ledger writes or startup auto-apply.
ALTER TABLE actors ADD COLUMN IF NOT EXISTS pin_hash text,
    ADD COLUMN IF NOT EXISTS pin_set_at timestamptz,
    ADD COLUMN IF NOT EXISTS pin_locked_until timestamptz,
    ADD COLUMN IF NOT EXISTS pin_failed_attempts integer NOT NULL DEFAULT 0;
CREATE UNIQUE INDEX IF NOT EXISTS actors_pin_hash_unique ON actors(pin_hash)
    WHERE pin_hash IS NOT NULL;
CREATE TABLE IF NOT EXISTS actor_sessions (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    token_hash text NOT NULL UNIQUE CHECK (length(token_hash)=64),
    actor_id integer NOT NULL REFERENCES actors(id),
    auth_method text NOT NULL CHECK (auth_method IN ('pin','actor_key')),
    device_label text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_seen_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL,
    ended_at timestamptz,
    ended_reason text CHECK (ended_reason IN ('idle','logout','revoked','pin_reset'))
);
CREATE INDEX IF NOT EXISTS actor_sessions_actor_open ON actor_sessions(actor_id) WHERE ended_at IS NULL;
CREATE TABLE IF NOT EXISTS pin_rate_limits (
    source text PRIMARY KEY,
    failed_at timestamptz[] NOT NULL DEFAULT '{}',
    locked_until timestamptz
);
CREATE TABLE IF NOT EXISTS pin_attempts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source text NOT NULL,
    device_hash text NOT NULL,
    purpose text NOT NULL,
    ok boolean NOT NULL,
    blocked boolean NOT NULL DEFAULT false,
    actor_id integer REFERENCES actors(id),
    at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS pin_attempts_at_idx ON pin_attempts(at);
CREATE TABLE IF NOT EXISTS pin_management_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id integer NOT NULL REFERENCES actors(id),
    changed_by_actor_id integer NOT NULL REFERENCES actors(id),
    at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE actor_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pin_rate_limits ENABLE ROW LEVEL SECURITY;
ALTER TABLE pin_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pin_management_audit ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON actor_sessions, pin_rate_limits, pin_attempts, pin_management_audit FROM PUBLIC;
DO $$ DECLARE r text; BEGIN
    FOREACH r IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF EXISTS(SELECT FROM pg_roles WHERE rolname=r) THEN
            EXECUTE format('REVOKE ALL ON actor_sessions, pin_rate_limits, pin_attempts, pin_management_audit FROM %I',r);
        END IF;
    END LOOP;
END $$;
INSERT INTO migration_markers(name) VALUES ('072_pin_sessions') ON CONFLICT DO NOTHING;

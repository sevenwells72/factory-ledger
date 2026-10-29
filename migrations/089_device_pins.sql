-- Phone PIN sign-in on top of the live key sessions (077). A phone is enrolled
-- once with the person's personal key; afterwards a 6-digit PIN opens a
-- 12-hour session, on that phone only. Additive and idempotent: no change to
-- actors, roles, tickets or ledger rows. Apply as app/table owner before
-- setting PIN_SIGNIN_ENABLED=1; with the flag off nothing reads these tables.
CREATE TABLE IF NOT EXISTS key_devices (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id integer NOT NULL REFERENCES actors(id),
    -- sha256 of the 32-byte device secret held only by the phone.
    secret_hash text NOT NULL UNIQUE CHECK (length(secret_hash) = 64),
    -- The personal key that enrolled the phone; rotating that key retires it.
    actor_key_hash text NOT NULL,
    label text NOT NULL CHECK (char_length(label) BETWEEN 1 AND 60),
    -- PBKDF2 over the PIN, salted with a random salt AND the device secret,
    -- so these columns alone cannot be brute-forced back to a PIN.
    pin_hash text CHECK (pin_hash IS NULL OR length(pin_hash) = 64),
    pin_salt text CHECK (pin_salt IS NULL OR length(pin_salt) = 32),
    pin_iterations integer CHECK (pin_iterations IS NULL OR pin_iterations >= 1000),
    pin_set_at timestamptz,
    -- Owner "Reset PIN": the phone may choose a new PIN until this time.
    pin_reset_until timestamptz,
    failed_count integer NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
    locked_until timestamptz,
    enrolled_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_used_at timestamptz,
    revoked_at timestamptz,
    revoked_reason text CHECK (revoked_reason IN ('owner','self','replaced','key_rotated','deactivated','access_removed')),
    revoked_by integer REFERENCES actors(id),
    CHECK ((revoked_at IS NULL) = (revoked_reason IS NULL)),
    CHECK ((pin_hash IS NULL) = (pin_salt IS NULL) AND (pin_hash IS NULL) = (pin_iterations IS NULL))
);
CREATE INDEX IF NOT EXISTS key_devices_open_actor ON key_devices(actor_id) WHERE revoked_at IS NULL;

-- Attempts from phones the server does not recognise (no device row). They
-- share one bucket that never touches an enrolled phone's own counter.
CREATE TABLE IF NOT EXISTS key_device_attempts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    device_id bigint REFERENCES key_devices(id),
    ok boolean NOT NULL,
    attempted_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS key_device_attempts_time ON key_device_attempts(attempted_at);

-- Who enrolled, reset, revoked or (de)activated what, and when. No secrets.
CREATE TABLE IF NOT EXISTS key_device_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    actor_id integer NOT NULL REFERENCES actors(id),
    device_id bigint REFERENCES key_devices(id),
    action text NOT NULL CHECK (action IN ('enroll','pin_change','pin_reset','pin_reset_set','revoke','remove_access','deactivate','reactivate','blocked')),
    by_actor_id integer REFERENCES actors(id),
    at timestamptz NOT NULL DEFAULT clock_timestamp()
);

-- A session opened with a PIN names its phone, so revoking the phone ends it.
ALTER TABLE key_actor_sessions ADD COLUMN IF NOT EXISTS device_id bigint REFERENCES key_devices(id);

ALTER TABLE key_devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_device_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_device_audit ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON key_devices,key_device_attempts,key_device_audit FROM PUBLIC;
REVOKE ALL ON SEQUENCE key_devices_id_seq,key_device_attempts_id_seq,key_device_audit_id_seq FROM PUBLIC;
DO $$ DECLARE r text; BEGIN
    FOREACH r IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname=r) THEN
            EXECUTE format('REVOKE ALL ON key_devices,key_device_attempts,key_device_audit FROM %I',r);
            EXECUTE format('REVOKE ALL ON SEQUENCE key_devices_id_seq,key_device_attempts_id_seq,key_device_audit_id_seq FROM %I',r);
        END IF;
    END LOOP;
END $$;
INSERT INTO migration_markers(name) VALUES ('089_device_pins') ON CONFLICT DO NOTHING;

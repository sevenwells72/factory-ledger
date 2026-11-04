-- Simple user selection (FL-SIMPLE-USER-20261103). The owner approves a browser
-- once; on an approved browser staff tap one of the configured people. The
-- browser token authenticates the DEVICE (a routine capability, never owner
-- proof); the tapped name IDENTIFIES the person. Additive and idempotent: no
-- change to actors, roles, existing tickets or ledger rows. Apply as the
-- app/table owner before setting PERSON_SELECT_ENABLED=1; with the flag off
-- nothing reads or writes these objects.

-- One row per browser that asked for approval. Only sha256 hashes are stored.
CREATE TABLE IF NOT EXISTS trusted_browsers (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- sha256 of the random browser token held only in that browser.
    secret_hash text NOT NULL UNIQUE CHECK (length(secret_hash) = 64),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','revoked','expired')),
    -- sha256 of the short code the browser shows while pending; cleared on approval.
    pairing_code_hash text CHECK (pairing_code_hash IS NULL OR length(pairing_code_hash) = 64),
    pairing_expires_at timestamptz,
    label text CHECK (label IS NULL OR char_length(label) BETWEEN 1 AND 60),
    requested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    approved_at timestamptz,
    approved_by_actor_id integer REFERENCES actors(id),
    revoked_at timestamptz,
    revoked_by_actor_id integer REFERENCES actors(id),
    revoked_reason text CHECK (revoked_reason IS NULL OR revoked_reason IN ('owner','expired')),
    last_seen_at timestamptz,
    CHECK ((status = 'approved') = (approved_at IS NOT NULL AND revoked_at IS NULL)),
    CHECK ((status = 'approved') <= (approved_by_actor_id IS NOT NULL AND label IS NOT NULL)),
    CHECK ((status = 'pending') = (pairing_code_hash IS NOT NULL)),
    CHECK ((revoked_at IS NULL) = (revoked_reason IS NULL))
);
-- At most one live pending request per code (codes are random; this is a guard).
CREATE UNIQUE INDEX IF NOT EXISTS trusted_browsers_pending_code
    ON trusted_browsers(pairing_code_hash) WHERE status = 'pending';

-- Request/approval attempts for the shared rate limit. No secrets, no codes.
CREATE TABLE IF NOT EXISTS trusted_browser_attempts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('request','approve')),
    ok boolean NOT NULL,
    attempted_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS trusted_browser_attempts_time ON trusted_browser_attempts(attempted_at);

-- Who approved or revoked which browser, and when. Append-only.
CREATE TABLE IF NOT EXISTS trusted_browser_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    browser_id bigint NOT NULL REFERENCES trusted_browsers(id),
    action text NOT NULL CHECK (action IN ('approve','revoke')),
    by_actor_id integer NOT NULL REFERENCES actors(id),
    at timestamptz NOT NULL DEFAULT clock_timestamp()
);
DROP TRIGGER IF EXISTS trusted_browser_audit_append_only ON trusted_browser_audit;
CREATE TRIGGER trusted_browser_audit_append_only BEFORE DELETE OR UPDATE ON trusted_browser_audit
    FOR EACH ROW EXECUTE FUNCTION public.ledger_block_append_only_change();

-- The people offered on an approved browser, per environment. Rows are DATA,
-- inserted in a separately approved step (never by this migration). The
-- display name is UI only; ledger snapshots keep actors.name.
CREATE TABLE IF NOT EXISTS person_choices (
    actor_id integer PRIMARY KEY REFERENCES actors(id),
    display_name text NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 40),
    role_label text NOT NULL CHECK (char_length(role_label) BETWEEN 1 AND 40),
    sort_order integer NOT NULL UNIQUE,
    active boolean NOT NULL DEFAULT true
);

-- Assurance on the records a selected person creates. NULL = an existing
-- credential path (personal key, session, shared key) exactly as before.
ALTER TABLE write_tickets ADD COLUMN IF NOT EXISTS identity_via text
    CHECK (identity_via IS NULL OR identity_via = 'selection');
ALTER TABLE write_tickets ADD COLUMN IF NOT EXISTS trusted_browser_id bigint REFERENCES trusted_browsers(id);
ALTER TABLE actor_write_audit ADD COLUMN IF NOT EXISTS trusted_browser_id bigint REFERENCES trusted_browsers(id);
ALTER TABLE shift_confirmations ADD COLUMN IF NOT EXISTS identity_via text
    CHECK (identity_via IS NULL OR identity_via = 'selection');
ALTER TABLE shift_confirmations ADD COLUMN IF NOT EXISTS trusted_browser_id bigint REFERENCES trusted_browsers(id);

ALTER TABLE trusted_browsers ENABLE ROW LEVEL SECURITY;
ALTER TABLE trusted_browser_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE trusted_browser_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE person_choices ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON trusted_browsers,trusted_browser_attempts,trusted_browser_audit,person_choices FROM PUBLIC;
REVOKE ALL ON SEQUENCE trusted_browsers_id_seq,trusted_browser_attempts_id_seq,trusted_browser_audit_id_seq FROM PUBLIC;
DO $$ DECLARE r text; BEGIN
    FOREACH r IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname=r) THEN
            EXECUTE format('REVOKE ALL ON trusted_browsers,trusted_browser_attempts,trusted_browser_audit,person_choices FROM %I',r);
            EXECUTE format('REVOKE ALL ON SEQUENCE trusted_browsers_id_seq,trusted_browser_attempts_id_seq,trusted_browser_audit_id_seq FROM %I',r);
        END IF;
    END LOOP;
END $$;
INSERT INTO migration_markers(name) VALUES ('091_trusted_browsers') ON CONFLICT DO NOTHING;

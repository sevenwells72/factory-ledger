-- F1 transport state only. Apply explicitly, as the app/table owner, to staging.
-- 062–066 are A5/A2; 067 is reserved for the concurrent A3b lane.
CREATE TABLE IF NOT EXISTS assistant_sessions (
    id uuid PRIMARY KEY,
    actor_id integer NOT NULL REFERENCES actors(id),
    state jsonb NOT NULL DEFAULT '{"history":[],"resolved":{}}',
    lease_id uuid,
    lease_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS assistant_sessions_actor ON assistant_sessions(actor_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS assistant_turns (
    session_id uuid NOT NULL REFERENCES assistant_sessions(id),
    id uuid NOT NULL,
    request_hash text NOT NULL,
    response jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (session_id, id)
);
CREATE TABLE IF NOT EXISTS assistant_drafts (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES assistant_sessions(id),
    actor_id integer NOT NULL REFERENCES actors(id),
    ticket text NOT NULL UNIQUE,
    payload_hash text NOT NULL,
    card jsonb NOT NULL,
    prepare_body jsonb NOT NULL DEFAULT '{}',
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','cancelled','committed')),
    result jsonb,
    record_started_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS assistant_drafts_session ON assistant_drafts(session_id,created_at);
ALTER TABLE assistant_drafts ADD COLUMN IF NOT EXISTS record_started_at timestamptz;
ALTER TABLE assistant_drafts ADD COLUMN IF NOT EXISTS prepare_body jsonb NOT NULL DEFAULT '{}';
CREATE TABLE IF NOT EXISTS assistant_attachments (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES assistant_sessions(id),
    actor_id integer NOT NULL REFERENCES actors(id),
    filename text NOT NULL,
    media_type text NOT NULL CHECK (media_type IN ('image/jpeg','image/png','image/webp')),
    sha256 text NOT NULL,
    content bytea NOT NULL CHECK (octet_length(content) BETWEEN 1 AND 5242880),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS assistant_attachments_session ON assistant_attachments(session_id);
CREATE TABLE IF NOT EXISTS assistant_unmatched_words (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES assistant_sessions(id),
    actor_id integer NOT NULL REFERENCES actors(id),
    kind text NOT NULL,
    query text NOT NULL CHECK (length(query) <= 500),
    context jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE assistant_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_turns ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_drafts ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_attachments ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistant_unmatched_words ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON assistant_sessions,assistant_turns,assistant_drafts,
    assistant_attachments,assistant_unmatched_words FROM PUBLIC;
REVOKE ALL ON SEQUENCE assistant_unmatched_words_id_seq FROM PUBLIC;
-- Supabase grants new tables to these roles by default. Photos/tickets are private.
DO $$ DECLARE r text; BEGIN
    FOREACH r IN ARRAY ARRAY['anon','authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname=r) THEN
            EXECUTE format('REVOKE ALL ON assistant_sessions,assistant_turns,assistant_drafts,assistant_attachments,assistant_unmatched_words FROM %I',r);
            EXECUTE format('REVOKE ALL ON SEQUENCE assistant_unmatched_words_id_seq FROM %I',r);
        END IF;
    END LOOP;
END $$;
INSERT INTO migration_markers(name) VALUES ('068_fl_assistant') ON CONFLICT DO NOTHING;

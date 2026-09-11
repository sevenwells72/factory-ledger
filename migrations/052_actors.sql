-- 052: Actors — per-user attribution for dashboard writes (FR-15, step 5a)
--      (docs/design/so-state-model-findings.md, "FR-15 attribution")
--
-- One row per human who writes through the ledger. The API resolves an
-- inbound X-API-Key to an actor by sha256 and stamps that actor's NAME into
-- the existing attribution columns — sales_orders.state_changed_by,
-- sales_order_allocations.released_by, sales_order_flags.ready_by and the
-- created_by source tags — in place of today's 'dashboard' / NULL.
--
-- NO SCHEMA CHANGE OUTSIDE THIS TABLE. The attribution columns are already
-- text and already exist (migrations 044, 051, 037); this migration only adds
-- the vocabulary that fills them. Every existing reader keeps working, and an
-- API instance that has never heard of this table keeps working too: the two
-- legacy keys are checked FIRST and their behaviour is unchanged.
--
-- NO SEED ROWS. Keys are minted out of band and inserted separately:
--     python3 scripts/mint_actor_keys.py "Name:role" ...
-- writes actors_insert.sql containing hashes only. Plaintext keys are printed
-- once to that script's stdout and are never written to disk or to git.
--
-- key_hash is sha256(plaintext_key) as 64 lowercase hex characters. The
-- plaintext is never stored, so a leaked database dump cannot be replayed as
-- credentials. Revocation is `active = false` (the resolver requires
-- active = true); rotation is a re-INSERT with a new hash via ON CONFLICT.
--
-- APPLIED VIA THE SUPABASE SQL EDITOR — deliberately no BEGIN/COMMIT in this
-- file. The editor wraps each run in its own transaction, and a stray COMMIT
-- inside it aborts the run.
--
-- Idempotent and re-runnable: CREATE TABLE IF NOT EXISTS, every constraint and
-- index added through a DO block that checks the catalog first, and the
-- migration_markers row inserted ON CONFLICT DO NOTHING. A second run performs
-- no writes and reports 'already applied'.

-- ─────────────────────────────────────────────────────────────────
-- Table
-- ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.actors (
    id           serial PRIMARY KEY,
    name         text        NOT NULL,
    role         text        NOT NULL,
    key_hash     text        NOT NULL,
    active       boolean     NOT NULL DEFAULT true,
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz
);

-- ─────────────────────────────────────────────────────────────────
-- Constraints
--
-- Added separately rather than inline, so a re-run does not fail: CREATE TABLE
-- IF NOT EXISTS skips the whole statement including its constraints, but a
-- table created by an EARLIER, partially-applied run might be missing one.
-- ADD CONSTRAINT has no IF NOT EXISTS, hence the catalog checks.
-- ─────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'actors_name_key'
                      AND conrelid = 'public.actors'::regclass) THEN
        ALTER TABLE public.actors
            ADD CONSTRAINT actors_name_key UNIQUE (name);
    END IF;

    -- UNIQUE on the hash, not on the plaintext: two actors sharing a key would
    -- make attribution ambiguous in exactly the way this table exists to stop.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'actors_key_hash_key'
                      AND conrelid = 'public.actors'::regclass) THEN
        ALTER TABLE public.actors
            ADD CONSTRAINT actors_key_hash_key UNIQUE (key_hash);
    END IF;

    -- The role vocabulary. 'owner' | 'floor' | 'office' matches the three
    -- groups that write today; widening it is a migration, not a config edit,
    -- so a typo'd role cannot silently create a fourth class of user.
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                    WHERE conname = 'actors_role_check'
                      AND conrelid = 'public.actors'::regclass) THEN
        ALTER TABLE public.actors
            ADD CONSTRAINT actors_role_check
            CHECK (role IN ('owner', 'floor', 'office'));
    END IF;
END $$;

-- ─────────────────────────────────────────────────────────────────
-- Marker
--
-- ON CONFLICT DO NOTHING so a re-run is a true no-op rather than a duplicate
-- key error. Verify with:
--     select * from migration_markers where name = '052_actors';
-- ─────────────────────────────────────────────────────────────────

DO $$
DECLARE
    already boolean;
    n_actors integer;
BEGIN
    -- migration_markers is created by the application at startup
    -- (STARTUP_MIGRATION_MARKERS_DDL, main.py). Create it here too so this
    -- file can be applied to a database the app has not booted against yet.
    CREATE TABLE IF NOT EXISTS public.migration_markers (
        name       text PRIMARY KEY,
        applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );

    SELECT EXISTS (SELECT 1 FROM public.migration_markers
                    WHERE name = '052_actors') INTO already;

    INSERT INTO public.migration_markers (name)
         VALUES ('052_actors')
    ON CONFLICT (name) DO NOTHING;

    SELECT count(*) INTO n_actors FROM public.actors;

    IF already THEN
        RAISE NOTICE '052_actors: already applied — no changes made.';
    ELSE
        RAISE NOTICE '052_actors: applied.';
    END IF;
    RAISE NOTICE '  actors rows: % (0 is expected on first apply — mint keys with scripts/mint_actor_keys.py)', n_actors;
END $$;

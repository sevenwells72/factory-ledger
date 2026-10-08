-- A2 (design rev 3.7 §6.2): entered_by on every ledger write.
-- Apply as the app/table owner after 061, port 5432, in one transaction
-- (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s', SET LOCAL search_path=public,
-- COMMIT). STAGING FIRST; production only with owner approval.
--
-- Additive only: two nullable FK columns, one partial index, one marker. No
-- backfill, no trigger change, no view replacement — the append-only guard on
-- transactions is never disabled. ADD COLUMN of a nullable column without a
-- default is a catalog-only change (no table rewrite); it takes ACCESS
-- EXCLUSIVE on transactions / ledger_corrections only until COMMIT, bounded by
-- lock_timeout. Safe to rerun.
--
-- Meaning of the three entry fields on transactions after this migration:
--   happened_at  = occurred_at            (plant time; user-supplied, defaults to now)
--   entered_at   = created_at             (DB clock, immutable)
--   entered_by   = entered_by_actor_id    (authenticated person, FK actors; NULL for the
--                  two shared keys, whose operator_id stays 'legacy-shared-key' / the
--                  surface tag). operator_id (text name snapshot) is unchanged for the
--                  legacy readers; history before this migration is attributed by it.
ALTER TABLE transactions
    ADD COLUMN IF NOT EXISTS entered_by_actor_id integer REFERENCES actors(id);
CREATE INDEX IF NOT EXISTS transactions_entered_by_actor_idx
    ON transactions (entered_by_actor_id) WHERE entered_by_actor_id IS NOT NULL;

ALTER TABLE ledger_corrections
    ADD COLUMN IF NOT EXISTS entered_by_actor_id integer REFERENCES actors(id);

-- Standalone migration marker (no startup gate reads it; 060/061 precedent).
INSERT INTO migration_markers (name) VALUES ('065_entered_by')
ON CONFLICT (name) DO NOTHING;

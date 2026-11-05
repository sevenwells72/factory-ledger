-- Resolve a stock shortage without a physical count (owner decision, Michael,
-- 2026-11-05). Explicit apply only, as app/table owner, BEFORE or WITH the code:
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public;
-- run this file with ON_ERROR_STOP; COMMIT.
-- shortage_flags.resolution_kind gains 'unverified': the shortage was closed on a
-- note alone. No stock posts; the lot keeps its book balance. exceptions.resolution_kind
-- has no CHECK and needs no change. Additive: no row is rewritten.
-- Until the marker exists FL neither offers nor accepts 'unverified'
-- (UNVERIFIED_CLOSE_UNAVAILABLE, nothing recorded), so code without this fails safe.
ALTER TABLE shortage_flags DROP CONSTRAINT IF EXISTS shortage_flags_resolution_kind_check;
ALTER TABLE shortage_flags ADD CONSTRAINT shortage_flags_resolution_kind_check
    CHECK (resolution_kind IN ('counted', 'missing_movement', 'voided', 'unverified'));
INSERT INTO migration_markers(name) VALUES ('092_shortage_unverified_close') ON CONFLICT DO NOTHING;

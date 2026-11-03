-- Make-entry simplification (PR-1). Explicit apply only, as app/table owner,
-- BEFORE deploying the code that sends lot selections:
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public;
-- run this file with ON_ERROR_STOP; COMMIT.
-- 1. transaction_lot_confirmations.method gains 'selected': the worker tapped the
--    lot in the Make screen's list (no typed code). Existing methods unchanged;
--    the row keeps actor_id = who selected it. No data rewrite.
-- 2. An index for the Make screen's retry guard (write_tickets.payload
--    ->> 'entry_attempt_id'); the guard works without it, just slower.
-- Until the marker exists, FL refuses lot selections with LOT_SELECTION_UNAVAILABLE
-- (nothing posts), so a code deploy without this migration fails safe.
ALTER TABLE transaction_lot_confirmations DROP CONSTRAINT IF EXISTS transaction_lot_confirmations_method_check;
ALTER TABLE transaction_lot_confirmations ADD CONSTRAINT transaction_lot_confirmations_method_check
    CHECK (method IN ('last4','full_code','scan','pallet','selected'));
CREATE INDEX IF NOT EXISTS write_tickets_make_attempt_idx
    ON write_tickets ((payload->>'entry_attempt_id'))
    WHERE action = 'make' AND payload ? 'entry_attempt_id';
INSERT INTO migration_markers(name) VALUES ('090_make_lot_selection') ON CONFLICT DO NOTHING;

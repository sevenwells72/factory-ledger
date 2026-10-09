-- A3b follow-up (Codex re-check of PR #94, P1, 2026-10-09; design rev 3.7 §5 R3 resolution
-- rule): receipt evidence for a shortage must not be reusable. Every `missing_movement`
-- resolution records how many pounds of the evidence receipt it consumed, on the receipt's
-- lot; the code sums these under FOR UPDATE on the receipt's write_tickets row, so one
-- 20 lb receipt can never close 26 lb of shortages, and a receipt that covers only part
-- of what is open leaves the remainder open.
-- Apply as the app/table owner after 061 and 069, port 5432, in one transaction
-- (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s', SET LOCAL statement_timeout='60s',
-- SET LOCAL search_path=public, COMMIT). STAGING FIRST; production only with owner
-- approval and BEFORE the A3b code goes live there (the resolve route INSERTs into it).
--
-- Additive only, safe to rerun: one new table + two indexes + marker. No existing table,
-- view, trigger or constraint changes; nothing is backfilled (prod has no shortages yet).
CREATE TABLE IF NOT EXISTS shortage_evidence_claims (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    exception_id bigint NOT NULL REFERENCES exceptions(id),
    shortage_flag_id bigint REFERENCES shortage_flags(id),
    evidence_ticket_id bigint NOT NULL REFERENCES write_tickets(id),
    evidence_receipt_number text NOT NULL,
    lot_id integer NOT NULL REFERENCES lots(id),
    claimed_lb numeric(14,4) NOT NULL CHECK (claimed_lb > 0),
    claimed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    claimed_by_actor_id integer REFERENCES actors(id)
);
CREATE INDEX IF NOT EXISTS shortage_evidence_claims_receipt_idx
    ON shortage_evidence_claims (evidence_ticket_id, lot_id);
CREATE INDEX IF NOT EXISTS shortage_evidence_claims_exception_idx
    ON shortage_evidence_claims (exception_id);

-- Standalone migration marker (no startup gate reads it; 069/070 precedent).
INSERT INTO migration_markers (name) VALUES ('071_shortage_evidence_claims')
ON CONFLICT (name) DO NOTHING;

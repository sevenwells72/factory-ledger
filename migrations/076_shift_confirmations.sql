-- A9 (design rev 3.8 §7.3; contract docs/contracts/shift-summary.md §3.4):
-- end-of-shift confirmations. Apply as the app/table owner after 074 (075 is A12 kosher tiers), port
-- 5432, in one transaction (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s',
-- SET LOCAL statement_timeout='60s', SET LOCAL search_path=public, COMMIT).
-- STAGING FIRST; production only with owner approval and BEFORE the A9 code
-- goes live there (GET /reports/shift-summary reads this table for the
-- confirmation state; POST .../confirm inserts into it).
--
-- Additive only, safe to rerun: one new table, one index, one append-only
-- trigger (039 pattern), RLS on with zero policies (owner-only, like 058/074),
-- one marker. No existing table, view or trigger is touched. A confirmation is
-- not a ledger write: a second confirm for the same (date, actor) APPENDS a
-- row — the latest is the day's status (contract decision 3).
CREATE TABLE IF NOT EXISTS shift_confirmations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    business_date date NOT NULL,
    actor_id integer NOT NULL REFERENCES actors(id),
    outcome text NOT NULL CHECK (outcome IN ('match', 'discrepancy')),
    notes text NOT NULL DEFAULT '' CHECK (outcome <> 'discrepancy' OR btrim(notes) <> ''),
    receipts_seen integer NOT NULL CHECK (receipts_seen >= 0),
    summary_hash text NOT NULL CHECK (length(summary_hash) = 64),
    summary_snapshot jsonb NOT NULL CHECK (jsonb_typeof(summary_snapshot) = 'array'),
    exception_id bigint REFERENCES exceptions(id),
    confirmed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS shift_confirmations_day_actor_idx
    ON shift_confirmations (business_date, actor_id, confirmed_at DESC);
-- Append-only: the 039 guard function already exists on every ledger table.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid = 'shift_confirmations'::regclass
                   AND tgname = 'shift_confirmations_append_only') THEN
        CREATE TRIGGER shift_confirmations_append_only BEFORE UPDATE OR DELETE
            ON shift_confirmations FOR EACH ROW
            EXECUTE FUNCTION ledger_block_append_only_change();
    END IF;
END $$;
ALTER TABLE shift_confirmations ENABLE ROW LEVEL SECURITY;
-- Standalone migration marker; no startup gate reads it (060/061/074 precedent).
INSERT INTO migration_markers (name) VALUES ('076_shift_confirmations')
ON CONFLICT (name) DO NOTHING;

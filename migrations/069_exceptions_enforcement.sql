-- A3b (design rev 3.7 §5 R2/R3, §5.1, §7.1, §11 item 7; FOLLOWUPS P1.8, P1.11a):
-- exceptions ENFORCEMENT. The tables came with 061 (A3a); this migration adds
-- only what the enforcement code relies on. Apply as the app/table owner after
-- 061 and 065, port 5432, in one transaction (ON_ERROR_STOP, BEGIN,
-- SET LOCAL lock_timeout='5s', SET LOCAL statement_timeout='60s',
-- SET LOCAL search_path=public, COMMIT). STAGING FIRST; production only with
-- owner approval and BEFORE the A3b code goes live there (the code writes
-- write_tickets.status='awaiting_approval', relies on the unique indexes for
-- replay safety and reads reason_code through the view).
--
-- Additive only, safe to rerun:
--   1. write_tickets.status CHECK gains 'awaiting_approval' — a large
--      correction (> 500 lb) committed without a photo is HELD, not posted,
--      until the owner approves (POST /exceptions/{id}/approve) or a later
--      commit carries the photo. Held tickets never expire (scripts/
--      expire_tickets.py only touches 'prepared'). Same rebuild pattern as
--      062/067: the existing definition is replaced by the full known list.
--   2. Replay-safety unique indexes: one shortage_flags row and one
--      exceptions(SHORTAGE) row per (transaction_id, lot_id); one
--      exceptions(LARGE_CORRECTION) per held ticket. A retried commit replays
--      the stored receipt and never reaches the INSERTs; these indexes make
--      the "no second flag / exception" promise a database fact.
--   3. ledger_current_transactions gains two TRAILING columns, reason_code
--      and entered_by_actor_id (P1.8 / P1.11a: today both are visible only
--      inside effective_record). CREATE OR REPLACE VIEW keeps every existing
--      column in place; dependent views are untouched; no rewrite.
--   4. One-time sweep of transactions.reason_code on adjust rows still NULL —
--      the P1.8 window (tickets committed between the 061 apply and this
--      deploy), same four tiers as 061, same single-statement trigger dance.
--      From this deploy on, every adjust/found INSERT writes reason_code.
--   5. Two small partial indexes for the nightly overdue sweep.
--   6. actor_write_audit.target_table CHECK gains 'exceptions' so resolve /
--      approve / reject are attributed like every other non-ledger actor write
--      (056). Rebuild pattern as in §1; the audit rows themselves are append-only.
-- Lock profile: the CHECK rebuild and the view replacement take ACCESS
-- EXCLUSIVE on write_tickets / the view for the rest of the transaction
-- (milliseconds of work, bounded by lock_timeout); the sweep UPDATE touches
-- only rows WHERE reason_code IS NULL (0–few rows).

-- ---------------------------------------------------------------------------
-- 1. write_tickets.status: + 'awaiting_approval'
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='write_tickets'::regclass AND conname='write_tickets_status_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '069 requires migration 058 (write_tickets_status_check is missing)';
    END IF;
    IF position('awaiting_approval' in definition)=0 THEN
        EXECUTE 'ALTER TABLE write_tickets DROP CONSTRAINT write_tickets_status_check';
        EXECUTE 'ALTER TABLE write_tickets ADD CONSTRAINT write_tickets_status_check CHECK (status IN '
            || '(''prepared'',''committed'',''expired'',''rejected'',''superseded'',''awaiting_approval''))';
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS write_tickets_awaiting_approval_idx
    ON write_tickets (prepared_at) WHERE status = 'awaiting_approval';

-- ---------------------------------------------------------------------------
-- 2. Replay-safety unique indexes (061 tables)
-- ---------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS shortage_flags_one_per_line_idx
    ON shortage_flags (transaction_id, lot_id);
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_one_shortage_per_line_idx
    ON exceptions (transaction_id, lot_id) WHERE kind = 'SHORTAGE';
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_one_hold_per_ticket_idx
    ON exceptions (ticket_id) WHERE kind = 'LARGE_CORRECTION';

-- ---------------------------------------------------------------------------
-- 3. ledger_current_transactions + reason_code, entered_by_actor_id (trailing)
--    Body identical to the 055 definition as dumped from production on
--    2026-10-08 (tests/schema/schema.sql); only the two last columns are new.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.ledger_current_transactions AS
 SELECT t.id,
    t.type,
    t."timestamp",
    t.notes,
    t.bol_reference,
    t.shipper_name,
    t.shipper_code,
    t.cases_received,
    t.case_size_lb,
    t.customer_name,
    t.order_reference,
    t.adjust_reason,
    t.adjust_reason_es,
    t.status,
    t.created_at,
    t.created_at_source,
    t.occurred_at,
    t.business_date,
    t.operator_id,
        CASE
            WHEN (correction.event_type = 'void'::text) THEN 'voided'::text
            WHEN (correction.event_type = 'restore'::text) THEN 'posted'::text
            WHEN (correction.event_type = 'amend'::text) THEN COALESCE((correction.replacement_values ->> 'status'::text), t.status, 'posted'::text)
            ELSE COALESCE(t.status, 'posted'::text)
        END AS effective_status,
    correction.id AS latest_correction_id,
    correction.event_type AS latest_correction_type,
    correction.created_at AS latest_correction_created_at,
    correction.operator_id AS latest_correction_operator_id,
    correction.replacement_values AS latest_replacement_values,
    ((to_jsonb(t.*) || COALESCE(correction.replacement_values, '{}'::jsonb)) || jsonb_build_object('status',
        CASE
            WHEN (correction.event_type = 'void'::text) THEN 'voided'::text
            WHEN (correction.event_type = 'restore'::text) THEN 'posted'::text
            WHEN (correction.event_type = 'amend'::text) THEN COALESCE((correction.replacement_values ->> 'status'::text), t.status, 'posted'::text)
            ELSE COALESCE(t.status, 'posted'::text)
        END)) AS effective_record,
    t.reason_code,
    t.entered_by_actor_id
   FROM (public.transactions t
     LEFT JOIN LATERAL ( SELECT c.id,
            c.target_table,
            c.target_id,
            c.event_type,
            c.previous_values,
            c.replacement_values,
            c.reason,
            c.operator_id,
            c.created_at,
            c.created_at_source
           FROM public.ledger_corrections c
          WHERE ((c.target_table = 'transactions'::text) AND (c.target_id = t.id))
          ORDER BY c.created_at DESC, c.id DESC
         LIMIT 1) correction ON (true));

-- ---------------------------------------------------------------------------
-- 4. P1.8 sweep: adjust rows inserted with reason_code NULL since 061 ran.
--    Same tiers and the same single-statement guard toggle as 061 §3.
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    guard_enabled boolean;
    backfilled integer;
BEGIN
    SELECT tgenabled <> 'D' INTO guard_enabled
      FROM pg_trigger
     WHERE tgrelid = 'public.transactions'::regclass
       AND tgname = 'trg_transactions_original_append_only';
    IF guard_enabled THEN
        ALTER TABLE public.transactions
            DISABLE TRIGGER trg_transactions_original_append_only;
    END IF;

    WITH src AS (
        SELECT id,
               CASE WHEN NULLIF(btrim(adjust_reason), '') IS NOT NULL THEN 'adjust'
                    WHEN notes LIKE 'Found inventory%: %' THEN 'found' END AS source,
               CASE WHEN NULLIF(btrim(adjust_reason), '') IS NOT NULL THEN adjust_reason
                    WHEN notes LIKE 'Found inventory%: %'
                        THEN substr(notes, position(': ' IN notes) + 2) END AS raw
          FROM public.transactions
         WHERE type = 'adjust' AND reason_code IS NULL
    )
    UPDATE public.transactions t
       SET reason_code = COALESCE(
               m.reason_code, already.code,
               CASE WHEN src.source = 'adjust'
                     AND src.raw ~* '(physical.*count|physical inventory|inventory count|cycle count|count correction|recon)'
                    THEN 'physical_count' END,
               'unknown')
      FROM src
      LEFT JOIN public.correction_reason_legacy_codes m
             ON m.source = src.source
            AND m.legacy_code = lower(btrim(regexp_replace(src.raw, '\s+', ' ', 'g')))
      LEFT JOIN public.correction_reasons already
             ON already.code = replace(lower(btrim(regexp_replace(src.raw, '\s+', ' ', 'g'))), ' ', '_')
     WHERE t.id = src.id;
    GET DIAGNOSTICS backfilled = ROW_COUNT;
    RAISE NOTICE '069: reason_code swept on % adjust transaction(s) left NULL since 061', backfilled;

    IF guard_enabled THEN
        ALTER TABLE public.transactions
            ENABLE TRIGGER trg_transactions_original_append_only;
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 5. Nightly sweep indexes (overdue = open AND due_at < now()).
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS exceptions_open_due_idx
    ON exceptions (due_at) WHERE status = 'open' AND due_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS shortage_flags_open_due_idx
    ON shortage_flags (due_at) WHERE status = 'open';

-- ---------------------------------------------------------------------------
-- 6. actor_write_audit may name 'exceptions' as a target table.
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='actor_write_audit'::regclass AND conname='actor_write_audit_target_table_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '069 requires migration 056 (actor_write_audit_target_table_check is missing)';
    END IF;
    IF position('exceptions' in definition)=0 THEN
        EXECUTE 'ALTER TABLE actor_write_audit DROP CONSTRAINT actor_write_audit_target_table_check';
        EXECUTE 'ALTER TABLE actor_write_audit ADD CONSTRAINT actor_write_audit_target_table_check CHECK (('
            || substring(definition from 8 for length(definition)-8)
            || ') OR target_table = ''exceptions'')';
    END IF;
END $$;

-- Standalone migration marker (no startup gate reads it; 060/061/065/067 precedent).
INSERT INTO migration_markers (name) VALUES ('069_exceptions_enforcement')
ON CONFLICT (name) DO NOTHING;

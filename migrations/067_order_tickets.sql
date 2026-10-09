-- A7 (design rev 3.7 §1.2, §1.3, §4.3): order tickets.
-- Apply as the app/table owner after 065, port 5432, in one transaction
-- (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s', SET LOCAL
-- statement_timeout='30s', SET LOCAL search_path=public, COMMIT).
-- STAGING FIRST; production only with owner approval, BEFORE the A7 code
-- goes live there (the code INSERTs write_tickets rows with the new actions
-- and writes sales_orders.ticket_id / sales_order_lines.ticket_id).
--
-- Additive only, safe to rerun:
--   * write_tickets.action CHECK gains the ten order actions (same rebuild
--     pattern as 062's move_lot: the existing definition is kept and extended,
--     whatever earlier migrations put in it);
--   * sales_orders.ticket_id / sales_order_lines.ticket_id — nullable FK to the
--     ticket that CREATED the row (edits are UPDATEs; they are found through
--     write_tickets.result_ref->>'order_id', indexed below, and the existing
--     actor_write_audit rows);
--   * one expression index on write_tickets for "which receipts touched order X".
-- No backfill, no trigger change, no view replacement, no data sweep. ADD
-- COLUMN of a nullable column without a default is catalog-only (no rewrite);
-- the ACCESS EXCLUSIVE lock on sales_orders / sales_order_lines lasts until
-- COMMIT and is bounded by lock_timeout.
DO $$
DECLARE
    definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='write_tickets'::regclass AND conname='write_tickets_action_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '067 requires migration 058 (write_tickets_action_check is missing)';
    END IF;
    IF position('create_order' in definition)=0 THEN
        EXECUTE 'ALTER TABLE write_tickets DROP CONSTRAINT write_tickets_action_check';
        EXECUTE 'ALTER TABLE write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
            || substring(definition from 8 for length(definition)-8)
            || ') OR action IN (''create_order'',''add_order_lines'',''update_order_line'','
            || '''cancel_order_line'',''update_order_header'',''update_order_status'','
            || '''mark_order_ready'',''cancel_order'',''close_order'',''reopen_order''))';
    END IF;
END $$;

ALTER TABLE sales_orders
    ADD COLUMN IF NOT EXISTS ticket_id bigint REFERENCES write_tickets(id);
ALTER TABLE sales_order_lines
    ADD COLUMN IF NOT EXISTS ticket_id bigint REFERENCES write_tickets(id);
CREATE INDEX IF NOT EXISTS sales_orders_ticket_id_idx
    ON sales_orders (ticket_id) WHERE ticket_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS sales_order_lines_ticket_id_idx
    ON sales_order_lines (ticket_id) WHERE ticket_id IS NOT NULL;
-- Every committed order ticket records {order_id, ...} in result_ref.
CREATE INDEX IF NOT EXISTS write_tickets_order_idx
    ON write_tickets (((result_ref->>'order_id')::bigint))
    WHERE result_ref ? 'order_id';

-- Standalone migration marker (no startup gate reads it; 060/061/065 precedent).
INSERT INTO migration_markers (name) VALUES ('067_order_tickets')
ON CONFLICT (name) DO NOTHING;

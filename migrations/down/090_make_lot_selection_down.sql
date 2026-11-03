-- Deploy code without lot selection first. Refuses to narrow the constraint while
-- any 'selected' evidence exists (append-only evidence is never rewritten).
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM transaction_lot_confirmations WHERE method='selected') THEN
        RAISE EXCEPTION 'selected lot confirmations exist; keep migration 090';
    END IF;
END $$;
ALTER TABLE transaction_lot_confirmations DROP CONSTRAINT IF EXISTS transaction_lot_confirmations_method_check;
ALTER TABLE transaction_lot_confirmations ADD CONSTRAINT transaction_lot_confirmations_method_check
    CHECK (method IN ('last4','full_code','scan','pallet'));
DROP INDEX IF EXISTS write_tickets_make_attempt_idx;
DELETE FROM migration_markers WHERE name='090_make_lot_selection';

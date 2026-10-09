-- Reverses migrations/070_pre_make_adjust.sql. Run BEFORE the 069 and 061 downs
-- (061 drops the exceptions table; this file only needs it present).
-- Refuses while PRE_MAKE_ADJUST rows exist unless the session confirms they were
-- exported: SET LOCAL factory_ledger.confirm_exceptions_export = 'yes' (069 precedent).
-- Same wrapper as the up (BEGIN, lock_timeout, search_path=public, COMMIT).
DO $$
DECLARE
    tagged bigint;
BEGIN
    SELECT count(*) INTO tagged FROM exceptions WHERE kind = 'PRE_MAKE_ADJUST';
    IF tagged > 0 AND current_setting('factory_ledger.confirm_exceptions_export', true) IS DISTINCT FROM 'yes' THEN
        RAISE EXCEPTION '070 down refused: % PRE_MAKE_ADJUST exception row(s) exist; export them, then SET LOCAL factory_ledger.confirm_exceptions_export=''yes''', tagged;
    END IF;
    DELETE FROM exceptions WHERE kind = 'PRE_MAKE_ADJUST';
END $$;

DROP INDEX IF EXISTS exceptions_one_pre_make_tag_idx;

-- Restore the 061 kind list verbatim.
ALTER TABLE exceptions DROP CONSTRAINT IF EXISTS exceptions_kind_check;
ALTER TABLE exceptions ADD CONSTRAINT exceptions_kind_check CHECK (kind IN (
    'SHORTAGE', 'UNIDENTIFIED_LOT', 'LARGE_CORRECTION', 'LATE_ENTRY',
    'SHIPMENT_PROOF_MISSING', 'NEGATIVE_BALANCE', 'POSSIBLE_DUPLICATE_ACK',
    'UNSHIPPED_PAST_DUE', 'SUNSHINE_INVOICE_PENDING', 'SHIFT_DISCREPANCY'));

DELETE FROM migration_markers WHERE name = '070_pre_make_adjust';

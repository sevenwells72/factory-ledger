-- Reverses migrations/071_shortage_evidence_claims.sql. Run BEFORE the 070/069/061 downs
-- (the table references exceptions, shortage_flags, write_tickets, lots, actors).
-- Refuses while claims exist unless the session confirms they were exported:
-- SET LOCAL factory_ledger.confirm_exceptions_export = 'yes' (069/070 precedent).
DO $$
DECLARE
    claims bigint;
BEGIN
    IF to_regclass('shortage_evidence_claims') IS NOT NULL THEN
        SELECT count(*) INTO claims FROM shortage_evidence_claims;
        IF claims > 0 AND current_setting('factory_ledger.confirm_exceptions_export', true) IS DISTINCT FROM 'yes' THEN
            RAISE EXCEPTION '071 down refused: % shortage_evidence_claims row(s) exist; export them, then SET LOCAL factory_ledger.confirm_exceptions_export=''yes''', claims;
        END IF;
    END IF;
END $$;
DROP TABLE IF EXISTS shortage_evidence_claims;
DELETE FROM migration_markers WHERE name = '071_shortage_evidence_claims';

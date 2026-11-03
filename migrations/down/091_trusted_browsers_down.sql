-- Revert 091. Set PERSON_SELECT_ENABLED=0 and deploy the reverted backend
-- FIRST. This drops which browser and which assurance level recorded each
-- selection-made ticket, metadata edit and shift confirmation, so it refuses
-- while any such record exists unless the rows were exported and the operator
-- sets, inside this transaction:
--   SET LOCAL factory_ledger.confirm_selection_export = 'yes';
-- Actors, ledger rows, operator_id and entered_by_actor_id are untouched.
BEGIN;
LOCK TABLE trusted_browsers IN ACCESS EXCLUSIVE MODE;
DO $$ BEGIN
    IF (EXISTS (SELECT 1 FROM write_tickets WHERE identity_via IS NOT NULL OR trusted_browser_id IS NOT NULL)
        OR EXISTS (SELECT 1 FROM actor_write_audit WHERE trusted_browser_id IS NOT NULL)
        OR EXISTS (SELECT 1 FROM shift_confirmations WHERE identity_via IS NOT NULL OR trusted_browser_id IS NOT NULL)
        OR EXISTS (SELECT 1 FROM trusted_browser_audit))
       AND coalesce(current_setting('factory_ledger.confirm_selection_export', true), '') <> 'yes' THEN
        RAISE EXCEPTION '091 rollback refused: selection attribution rows exist; export them first';
    END IF;
END $$;
ALTER TABLE shift_confirmations DROP COLUMN IF EXISTS trusted_browser_id;
ALTER TABLE shift_confirmations DROP COLUMN IF EXISTS identity_via;
ALTER TABLE actor_write_audit DROP COLUMN IF EXISTS trusted_browser_id;
ALTER TABLE write_tickets DROP COLUMN IF EXISTS trusted_browser_id;
ALTER TABLE write_tickets DROP COLUMN IF EXISTS identity_via;
DROP TABLE IF EXISTS person_choices;
DROP TABLE IF EXISTS trusted_browser_audit;
DROP TABLE IF EXISTS trusted_browser_attempts;
DROP TABLE IF EXISTS trusted_browsers;
DELETE FROM migration_markers WHERE name='091_trusted_browsers';
COMMIT;

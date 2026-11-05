-- Deploy code without the no-count close first. Refuses to narrow the constraint
-- while any shortage was closed as 'unverified' (resolutions are never rewritten).
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM shortage_flags WHERE resolution_kind='unverified') THEN
        RAISE EXCEPTION 'unverified shortage resolutions exist; keep migration 092';
    END IF;
END $$;
ALTER TABLE shortage_flags DROP CONSTRAINT IF EXISTS shortage_flags_resolution_kind_check;
ALTER TABLE shortage_flags ADD CONSTRAINT shortage_flags_resolution_kind_check
    CHECK (resolution_kind IN ('counted', 'missing_movement', 'voided'));
DELETE FROM migration_markers WHERE name='092_shortage_unverified_close';

-- Deploy code without the destination rules first (or accept that the marker's
-- removal alone switches the application back to the old rules). Refuses while
-- any transaction carries one of the five destination reasons (history is never
-- rewritten) or while an owner email is still owed.
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM transactions WHERE reason_code IN
               ('shipped_unrecorded', 'sent_to_sunshine_bulk', 'used_in_production',
                'damaged_thrown_out', 'count_unknown')) THEN
        RAISE EXCEPTION 'adjustments recorded with a destination reason exist; keep migration 095';
    END IF;
    IF EXISTS (SELECT 1 FROM adjustment_alerts WHERE status = 'pending') THEN
        RAISE EXCEPTION 'owner emails for large adjustments are still pending; keep migration 095';
    END IF;
END $$;
DELETE FROM migration_markers WHERE name = '095_adjust_destinations';
DROP TABLE IF EXISTS adjustment_alerts;
DELETE FROM correction_reasons WHERE code IN
    ('shipped_unrecorded', 'sent_to_sunshine_bulk', 'used_in_production', 'damaged_thrown_out', 'count_unknown');
UPDATE correction_reasons SET active = true WHERE code IN ('damage_disposal', 'unrecorded_usage');
ALTER TABLE correction_reasons DROP CONSTRAINT IF EXISTS correction_reasons_order_needs_destination;
ALTER TABLE correction_reasons DROP CONSTRAINT IF EXISTS correction_reasons_destination_removes;
ALTER TABLE correction_reasons
    DROP COLUMN IF EXISTS sales_order_required,
    DROP COLUMN IF EXISTS removal_destination;

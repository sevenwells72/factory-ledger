-- A5 part 3. Explicit additive migration after 063, app/table owner, one
-- transaction with SET LOCAL lock_timeout='5s', search_path=public.
-- History remains NULL (unassessed); no historical lot is declared identified.
ALTER TABLE lots ADD COLUMN IF NOT EXISTS identity_status text
    CHECK (identity_status IN ('identified','unidentified'));
ALTER TABLE lots ADD COLUMN IF NOT EXISTS identify_by date;
CREATE INDEX IF NOT EXISTS lots_unidentified_due_idx ON lots(identify_by)
    WHERE identity_status='unidentified';
-- Reuse 061 exceptions, with a partial unique key for concurrent top-ups.
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_unidentified_open_lot_idx ON exceptions(lot_id)
    WHERE kind='UNIDENTIFIED_LOT' AND status IN ('open','escalated');
INSERT INTO migration_markers(name) VALUES ('064_unidentified_lots') ON CONFLICT DO NOTHING;

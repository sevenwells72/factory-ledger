-- Owner-approved receiving and hand-addition rules. No historical ledger writes.
-- Apply separately with owner approval, as table owner, in one transaction.
CREATE TABLE IF NOT EXISTS under_pack_counts (
 transaction_id integer PRIMARY KEY, lot_id integer NOT NULL, lb numeric(14,4) NOT NULL CHECK(lb>0),
 entered_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS under_pack_counts_lot ON under_pack_counts(lot_id,entered_at);
CREATE TABLE IF NOT EXISTS paired_lot_moves (
 transaction_id integer PRIMARY KEY, restored_lot_id integer NOT NULL, used_lot_id integer NOT NULL,
 CHECK(restored_lot_id<>used_lot_id)
);
ALTER TABLE under_pack_counts ENABLE ROW LEVEL SECURITY;
ALTER TABLE paired_lot_moves ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON under_pack_counts,paired_lot_moves FROM PUBLIC;
INSERT INTO migration_markers(name) VALUES ('098_no_stock_by_hand') ON CONFLICT(name) DO NOTHING;

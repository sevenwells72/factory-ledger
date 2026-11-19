-- Owner-approved receiving and hand-addition rules. No historical ledger writes.
-- Apply separately with owner approval, as table owner, in one transaction.
ALTER TABLE correction_reasons ADD COLUMN IF NOT EXISTS addition_allowed boolean NOT NULL DEFAULT false;
ALTER TABLE correction_reasons ADD COLUMN IF NOT EXISTS addition_label_en text;
ALTER TABLE correction_reasons ADD COLUMN IF NOT EXISTS addition_label_es text;
UPDATE correction_reasons SET addition_allowed=true,
 addition_label_en=CASE code
 WHEN 'physical_count' THEN 'Counted - same lot, FL was low'
 WHEN 'missing_production' THEN 'Made or packed but not recorded'
 WHEN 'wrong_lot' THEN 'Belongs on this lot - wrong lot was used'
 WHEN 'data_entry_error' THEN 'Typing mistake in an earlier entry' END,
 addition_label_es=CASE code
 WHEN 'physical_count' THEN 'Contado - mismo lote, FL tenía menos'
 WHEN 'missing_production' THEN 'Producido o empacado pero no registrado'
 WHEN 'wrong_lot' THEN 'Pertenece a este lote - se usó otro lote'
 WHEN 'data_entry_error' THEN 'Error de captura en una entrada anterior' END
 WHERE code IN ('physical_count','missing_production','wrong_lot','data_entry_error');
UPDATE correction_reasons SET applies_to=array_remove(applies_to,'adjust') WHERE code='missing_receipt';
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

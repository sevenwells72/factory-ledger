-- Keep lifetime usage and move identity for audit and safe reactivation.
DELETE FROM migration_markers WHERE name='098_no_stock_by_hand';
UPDATE correction_reasons SET applies_to=array_append(applies_to,'adjust')
 WHERE code='missing_receipt' AND NOT ('adjust'=ANY(applies_to));
ALTER TABLE correction_reasons DROP COLUMN IF EXISTS addition_allowed;
ALTER TABLE correction_reasons DROP COLUMN IF EXISTS addition_label_en;
ALTER TABLE correction_reasons DROP COLUMN IF EXISTS addition_label_es;

-- Rollback never deletes or relabels physical-tag evidence.
-- Use the same explicit transaction and timeouts as 085 up.
DO $$
DECLARE definition text; restored text;
BEGIN
    IF EXISTS (SELECT 1 FROM transaction_lot_confirmations WHERE method='tap') THEN
        RAISE EXCEPTION '085 down refused: tap confirmation evidence exists; retain its method constraint';
    END IF;
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='transaction_lot_confirmations'::regclass
        AND conname='transaction_lot_confirmations_method_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION 'transaction lot confirmation method constraint is missing';
    END IF;
    IF position('''tap''' in definition)>0 THEN
        restored := replace(definition, ' OR (method = ''tap''::text)', '');
        IF restored=definition OR position('''tap''' in restored)>0 THEN
            RAISE EXCEPTION '085 down refused: method constraint changed after 085; review it without rewriting evidence';
        END IF;
        ALTER TABLE transaction_lot_confirmations DROP CONSTRAINT transaction_lot_confirmations_method_check;
        EXECUTE 'ALTER TABLE transaction_lot_confirmations ADD CONSTRAINT transaction_lot_confirmations_method_check ' || restored;
    END IF;
END $$;
DELETE FROM migration_markers WHERE name='085_tap_lot_confirmation';

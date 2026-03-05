-- ═══════════════════════════════════════════════════════════════
-- Migration 015: Fix Lot 131 Negative Balance (-60 lb)
-- Posts an adjustment transaction to bring lot 131 to 0 lb.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

-- Pre-flight: verify lot 131 actually has the expected negative balance
DO $$
DECLARE
    v_balance NUMERIC;
    v_note TEXT;
BEGIN
    SELECT COALESCE(SUM(quantity_lb), 0) INTO v_balance
    FROM transaction_lines WHERE lot_id = 131;

    IF v_balance >= 0 THEN
        RAISE NOTICE 'Lot 131 balance is already >= 0 (%.4f lb). Skipping fix.', v_balance;
        RETURN;
    END IF;

    RAISE NOTICE 'Lot 131 current balance: %.4f lb. Posting +%.4f lb adjustment.', v_balance, ABS(v_balance);
    v_note := format('Audit fix 2026-03-05: correct negative lot balance on lot 131 (was %.4f lb)', v_balance);

    -- Post corrective adjustment transaction
    WITH new_txn AS (
        INSERT INTO transactions (type, timestamp, adjust_reason, notes)
        VALUES ('adjust', NOW(), v_note, v_note)
        RETURNING id
    )
    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
    SELECT new_txn.id, 126, 131, ABS(v_balance)
    FROM new_txn;

    -- Verify fix
    SELECT COALESCE(SUM(quantity_lb), 0) INTO v_balance
    FROM transaction_lines WHERE lot_id = 131;
    RAISE NOTICE 'Lot 131 balance after fix: %.4f lb', v_balance;
END $$;

-- DOWNGRADE
-- To reverse: delete the adjustment transaction created above.
-- Find it by its note text:
--   DELETE FROM transaction_lines WHERE transaction_id IN (
--     SELECT id FROM transactions
--     WHERE adjust_reason LIKE 'Audit fix 2026-03-05%lot 131%'
--   );
--   DELETE FROM transactions
--   WHERE adjust_reason LIKE 'Audit fix 2026-03-05%lot 131%';

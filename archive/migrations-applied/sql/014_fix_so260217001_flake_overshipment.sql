-- ═══════════════════════════════════════════════════════════════
-- Migration 014: Fix SO-260217-001 Flake UNIPRO over-shipment
--
-- Problem: System recorded 300 lb shipped for Coconut Sweetened
-- Flake UNIPRO 10 LB, but only 200 lb physically left the warehouse.
-- The extra 100 lb was deducted from lot "FEB 24 2026" (lot_id 242).
--
-- This migration corrects all affected tables in one transaction.
-- No records are deleted — only quantities and statuses are updated.
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- ── Pre-flight checks (these SELECTs verify we're targeting the right rows) ──

-- Verify transaction_line exists: txn 372, product 162, lot 242, qty = -100
DO $$
DECLARE
    v_qty NUMERIC;
BEGIN
    SELECT quantity_lb INTO v_qty
    FROM transaction_lines
    WHERE transaction_id = 372
      AND product_id = 162
      AND lot_id = 242;

    IF v_qty IS NULL THEN
        RAISE EXCEPTION 'Pre-flight FAILED: No transaction_line found for txn 372 / product 162 / lot 242';
    END IF;
    IF v_qty != -100 THEN
        RAISE EXCEPTION 'Pre-flight FAILED: Expected quantity_lb = -100, found %', v_qty;
    END IF;
    RAISE NOTICE 'Pre-flight OK: transaction_line qty = %', v_qty;
END $$;

-- Verify sales_order_line 48 currently shows 300 shipped
DO $$
DECLARE
    v_shipped NUMERIC;
    v_status  TEXT;
BEGIN
    SELECT quantity_shipped_lb, line_status INTO v_shipped, v_status
    FROM sales_order_lines
    WHERE id = 48;

    IF v_shipped != 300 THEN
        RAISE EXCEPTION 'Pre-flight FAILED: Expected quantity_shipped_lb = 300, found %', v_shipped;
    END IF;
    RAISE NOTICE 'Pre-flight OK: sales_order_line 48 shipped = %, status = %', v_shipped, v_status;
END $$;


-- ── 1. Fix transaction_lines: zero out the over-deduction ──
-- Lot 242 ("FEB 24 2026") was deducted -100 lb but nothing left from this lot.
-- Setting to 0 restores 100 lb to on-hand inventory for this lot.
UPDATE transaction_lines
SET quantity_lb = 0
WHERE transaction_id = 372
  AND product_id = 162
  AND lot_id = 242;

-- Add audit note to the transaction
UPDATE transactions
SET notes = notes || ' | CORRECTED 2026-03-05: reduced Flake ship qty from 300→200 lb (100 lb over-shipment reversed, lot FEB 24 2026 restored)'
WHERE id = 372;


-- ── 2. Fix shipment_lines: 300 → 200 lb ──
UPDATE shipment_lines
SET quantity_lb = 200
WHERE transaction_id = 372
  AND product_id = 162;


-- ── 3. Fix sales_order_shipments: 300 → 200 lb ──
UPDATE sales_order_shipments
SET quantity_lb = 200
WHERE transaction_id = 372
  AND sales_order_line_id = 48;


-- ── 4. Fix sales_order_lines: shipped 300 → 200, status → partial ──
UPDATE sales_order_lines
SET quantity_shipped_lb = 200,
    line_status = 'partial'
WHERE id = 48;


-- ── 5. Fix sales_orders: status → partial_ship ──
-- (Because one line is now partial, the order is no longer fully shipped)
UPDATE sales_orders
SET status = 'partial_ship'
WHERE id = 32;


-- ── Post-flight verification ──
DO $$
DECLARE
    v_lot_balance  NUMERIC;
    v_shipped      NUMERIC;
    v_line_status  TEXT;
    v_order_status TEXT;
    v_sl_qty       NUMERIC;
BEGIN
    -- Check lot 242 balance (should now be 200)
    SELECT COALESCE(SUM(quantity_lb), 0) INTO v_lot_balance
    FROM transaction_lines WHERE lot_id = 242;
    RAISE NOTICE 'Post-check: Lot 242 (FEB 24 2026) balance = % lb (expected 200)', v_lot_balance;

    -- Check sales_order_line 48
    SELECT quantity_shipped_lb, line_status INTO v_shipped, v_line_status
    FROM sales_order_lines WHERE id = 48;
    RAISE NOTICE 'Post-check: SO line 48 shipped = % lb, status = % (expected 200, partial)', v_shipped, v_line_status;

    -- Check sales_order 32
    SELECT status INTO v_order_status FROM sales_orders WHERE id = 32;
    RAISE NOTICE 'Post-check: SO 32 status = % (expected partial_ship)', v_order_status;

    -- Check shipment_lines
    SELECT quantity_lb INTO v_sl_qty
    FROM shipment_lines WHERE transaction_id = 372 AND product_id = 162;
    RAISE NOTICE 'Post-check: shipment_line qty = % lb (expected 200)', v_sl_qty;
END $$;

COMMIT;

-- ═══════════════════════════════════════════════════════════════
-- Migration 022: Close SO-260217-001 (Feeser's Food Distributors)
--
-- Problem: Order shows "Partial Ship" — Flake UNIPRO 10 LB had
-- 300 lb ordered but only 200 lb shipped. Remaining 100 lb will
-- not be shipped. Customer/business decision to close the order.
--
-- Fix: Reduce Flake ordered qty from 300 → 200 to match shipped,
-- mark line as fulfilled, update order status to shipped.
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- Pre-flight: verify Flake line on SO-260217-001 shows 200 shipped of 300 ordered
DO $$
DECLARE
    v_line_id INT;
    v_ordered NUMERIC;
    v_shipped NUMERIC;
    v_status  TEXT;
BEGIN
    SELECT sol.id, sol.quantity_lb, sol.quantity_shipped_lb, sol.line_status
    INTO v_line_id, v_ordered, v_shipped, v_status
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-001'
      AND p.name ILIKE '%Flake%UNIPRO%10 LB%';

    IF v_line_id IS NULL THEN
        RAISE EXCEPTION 'Flake line not found on SO-260217-001';
    END IF;
    IF v_shipped != 200 THEN
        RAISE EXCEPTION 'Flake shipped = %, expected 200', v_shipped;
    END IF;
    RAISE NOTICE 'Flake line: id=%, ordered=%, shipped=%, status=%', v_line_id, v_ordered, v_shipped, v_status;
END $$;

-- Close the Flake line: reduce ordered qty to match shipped (300 -> 200)
UPDATE sales_order_lines
SET quantity_lb = 200,
    line_status = 'fulfilled'
WHERE id = (
    SELECT sol.id
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-001'
      AND p.name ILIKE '%Flake%UNIPRO%10 LB%'
);

-- Update order status to shipped
UPDATE sales_orders
SET status = 'shipped'
WHERE order_number = 'SO-260217-001';

-- Post-flight
DO $$
DECLARE
    v_order_status TEXT;
    v_flake_ordered NUMERIC;
    v_flake_status TEXT;
BEGIN
    SELECT status INTO v_order_status FROM sales_orders WHERE order_number = 'SO-260217-001';
    IF v_order_status != 'shipped' THEN
        RAISE EXCEPTION 'Post-flight FAILED: order status = %', v_order_status;
    END IF;

    SELECT sol.quantity_lb, sol.line_status
    INTO v_flake_ordered, v_flake_status
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-001'
      AND p.name ILIKE '%Flake%UNIPRO%10 LB%';
    IF v_flake_ordered != 200 OR v_flake_status != 'fulfilled' THEN
        RAISE EXCEPTION 'Post-flight FAILED: Flake ordered=%, status=%', v_flake_ordered, v_flake_status;
    END IF;

    RAISE NOTICE 'Post-flight OK: order=shipped, Flake ordered=200/fulfilled';
END $$;

COMMIT;

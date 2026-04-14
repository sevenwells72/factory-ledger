-- ═══════════════════════════════════════════════════════════════
-- Migration 024: Close SO-260213-001 (Juliette Food LLC)
--
-- The shipment (BOL 28106-I, ship date 02/26/2026, customer pick up)
-- was never recorded as ship transactions in the system. All 4
-- granola products + 2 pallets were physically shipped per the
-- packing slip. Since no transactions exist, we mark the order
-- lines as fulfilled directly.
--
-- Products shipped per BOL:
--   - French Vanilla Granola 25lb: 24 cases = 600 lb
--   - Classic Granola 25LB: 24 cases = 600 lb
--   - Granola Fruits and Nuts 25 LB: 24 cases = 600 lb
--   - Granola Cocoa Vibes 25 LB: 24 cases = 600 lb
--   - Pallet Charge: 2
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- Pre-flight: verify order exists and shows 0 shipped
DO $$
DECLARE
    v_order_id INT;
    v_status   TEXT;
    v_shipped  NUMERIC;
BEGIN
    SELECT so.id, so.status INTO v_order_id, v_status
    FROM sales_orders so WHERE so.order_number = 'SO-260213-001';

    IF v_order_id IS NULL THEN RAISE EXCEPTION 'Order SO-260213-001 not found'; END IF;
    IF v_status NOT IN ('confirmed', 'new') THEN
        RAISE EXCEPTION 'Order status is %, expected confirmed', v_status;
    END IF;

    SELECT COALESCE(SUM(quantity_shipped_lb), 0) INTO v_shipped
    FROM sales_order_lines WHERE sales_order_id = v_order_id;

    IF v_shipped != 0 THEN
        RAISE EXCEPTION 'Order already has % lb shipped', v_shipped;
    END IF;
    RAISE NOTICE 'Pre-flight OK: id=%, status=%, shipped=%', v_order_id, v_status, v_shipped;
END $$;

-- Mark all lines as fully shipped
UPDATE sales_order_lines
SET quantity_shipped_lb = quantity_lb,
    line_status = 'fulfilled'
WHERE sales_order_id = (
    SELECT id FROM sales_orders WHERE order_number = 'SO-260213-001'
);

-- Update order status
UPDATE sales_orders
SET status = 'shipped'
WHERE order_number = 'SO-260213-001';

-- Post-flight
DO $$
DECLARE
    v_status TEXT;
    v_remaining NUMERIC;
    v_fulfilled INT;
    v_total INT;
BEGIN
    SELECT status INTO v_status FROM sales_orders WHERE order_number = 'SO-260213-001';
    IF v_status != 'shipped' THEN
        RAISE EXCEPTION 'Post-flight FAILED: order status = %', v_status;
    END IF;

    SELECT SUM(quantity_lb - quantity_shipped_lb), COUNT(*),
           COUNT(*) FILTER (WHERE line_status = 'fulfilled')
    INTO v_remaining, v_total, v_fulfilled
    FROM sales_order_lines
    WHERE sales_order_id = (SELECT id FROM sales_orders WHERE order_number = 'SO-260213-001');

    IF v_remaining != 0 THEN
        RAISE EXCEPTION 'Post-flight FAILED: % lb still remaining', v_remaining;
    END IF;

    RAISE NOTICE 'Post-flight OK: order=shipped, %/% lines fulfilled, 0 lb remaining', v_fulfilled, v_total;
END $$;

COMMIT;

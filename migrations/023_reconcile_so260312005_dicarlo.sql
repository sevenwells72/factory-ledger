-- ═══════════════════════════════════════════════════════════════
-- Migration 023: Reconcile SO-260312-005 with existing shipments
--
-- Problem: 4 ship transactions for DiCarlo Food Service on 2026-03-12
-- were created as standalone transactions before the sales order was
-- entered. The order shows 0 lb shipped / 3,200 lb remaining.
--
-- Products shipped (all match order lines exactly):
--   - Graham Cracker Crumbs – 10 LB: 400 lb
--   - Coconut Sweetened Fancy UNIPRO 10 LB: 400 lb
--   - Sprinkles Chocolate 25 LB: 600 lb
--   - Sprinkles Rainbow 25 LB: 1,800 lb
--
-- Fix: Link the existing ship transactions to the sales order by
-- creating shipments, sales_order_shipments, and shipment_lines
-- records, then updating line shipped quantities and order status.
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- STEP 0: Pre-flight — find the order and verify it shows 0 shipped
DO $$
DECLARE
    v_order_id INT;
    v_status   TEXT;
    v_shipped  NUMERIC;
BEGIN
    SELECT so.id, so.status
    INTO v_order_id, v_status
    FROM sales_orders so
    WHERE so.order_number = 'SO-260312-005';

    IF v_order_id IS NULL THEN
        RAISE EXCEPTION 'Order SO-260312-005 not found';
    END IF;
    IF v_status NOT IN ('confirmed', 'new') THEN
        RAISE EXCEPTION 'Order status is %, expected confirmed', v_status;
    END IF;

    SELECT COALESCE(SUM(sol.quantity_shipped_lb), 0)
    INTO v_shipped
    FROM sales_order_lines sol
    WHERE sol.sales_order_id = v_order_id;

    IF v_shipped != 0 THEN
        RAISE EXCEPTION 'Order already has % lb shipped, expected 0', v_shipped;
    END IF;

    RAISE NOTICE 'Order SO-260312-005: id=%, status=%, total_shipped=%', v_order_id, v_status, v_shipped;
END $$;

-- STEP 0b: Verify we can find the 4 standalone ship transactions
DO $$
DECLARE
    v_txn_count INT;
BEGIN
    SELECT COUNT(DISTINCT t.id)
    INTO v_txn_count
    FROM transactions t
    WHERE t.type = 'ship'
      AND t.customer_name ILIKE '%DiCarlo%'
      AND t.timestamp::date = '2026-03-12'
      AND COALESCE(t.status, 'posted') = 'posted';

    IF v_txn_count < 4 THEN
        RAISE EXCEPTION 'Found only % ship transactions for DiCarlo on 2026-03-12, expected at least 4', v_txn_count;
    END IF;
    RAISE NOTICE 'Found % DiCarlo ship transactions on 2026-03-12', v_txn_count;
END $$;


-- STEP 1: Create a shipments record to link everything
INSERT INTO shipments (sales_order_id, shipped_at, customer_id)
SELECT so.id, '2026-03-12 11:30:00-05'::timestamptz, so.customer_id
FROM sales_orders so
WHERE so.order_number = 'SO-260312-005';


-- STEP 2: For each product line, link the matching ship transaction

-- Helper: create a temp table mapping order lines to their ship transactions
CREATE TEMP TABLE line_txn_map AS
SELECT
    sol.id AS line_id,
    sol.product_id,
    sol.quantity_lb AS ordered_lb,
    t.id AS transaction_id,
    ABS(SUM(tl.quantity_lb)) AS shipped_lb
FROM sales_order_lines sol
JOIN sales_orders so ON so.id = sol.sales_order_id
JOIN transactions t ON t.type = 'ship'
    AND t.customer_name ILIKE '%DiCarlo%'
    AND t.timestamp::date = '2026-03-12'
    AND COALESCE(t.status, 'posted') = 'posted'
JOIN transaction_lines tl ON tl.transaction_id = t.id
    AND tl.product_id = sol.product_id
WHERE so.order_number = 'SO-260312-005'
GROUP BY sol.id, sol.product_id, sol.quantity_lb, t.id;

-- Verify we matched all 4 lines
DO $$
DECLARE
    v_count INT;
BEGIN
    SELECT COUNT(*) INTO v_count FROM line_txn_map;
    IF v_count < 4 THEN
        RAISE EXCEPTION 'Only matched % line-transaction pairs, expected 4', v_count;
    END IF;
    RAISE NOTICE 'Matched % line-transaction pairs', v_count;
END $$;

-- STEP 2a: Insert sales_order_shipments records
INSERT INTO sales_order_shipments (sales_order_line_id, transaction_id, quantity_lb)
SELECT line_id, transaction_id, shipped_lb
FROM line_txn_map;

-- STEP 2b: Insert shipment_lines records
INSERT INTO shipment_lines (shipment_id, transaction_id, sales_order_line_id, product_id, quantity_lb)
SELECT
    s.id,
    m.transaction_id,
    m.line_id,
    m.product_id,
    m.shipped_lb
FROM line_txn_map m
CROSS JOIN (
    SELECT s.id FROM shipments s
    JOIN sales_orders so ON so.id = s.sales_order_id
    WHERE so.order_number = 'SO-260312-005'
    ORDER BY s.id DESC LIMIT 1
) s;

-- STEP 2c: Update sales_order_lines shipped quantities
UPDATE sales_order_lines sol
SET quantity_shipped_lb = m.shipped_lb,
    line_status = CASE WHEN m.shipped_lb >= sol.quantity_lb THEN 'fulfilled' ELSE 'partial' END
FROM line_txn_map m
WHERE sol.id = m.line_id;

-- STEP 3: Update order status
UPDATE sales_orders
SET status = 'shipped'
WHERE order_number = 'SO-260312-005';

-- Cleanup temp table
DROP TABLE line_txn_map;

-- STEP 4: Post-flight verification
DO $$
DECLARE
    v_status TEXT;
    v_total_shipped NUMERIC;
    v_total_ordered NUMERIC;
    v_lines_fulfilled INT;
    v_total_lines INT;
BEGIN
    SELECT status INTO v_status FROM sales_orders WHERE order_number = 'SO-260312-005';
    IF v_status != 'shipped' THEN
        RAISE EXCEPTION 'Post-flight FAILED: order status = %', v_status;
    END IF;

    SELECT SUM(quantity_lb), SUM(quantity_shipped_lb), COUNT(*),
           COUNT(*) FILTER (WHERE line_status = 'fulfilled')
    INTO v_total_ordered, v_total_shipped, v_total_lines, v_lines_fulfilled
    FROM sales_order_lines sol
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260312-005';

    IF v_total_shipped != v_total_ordered THEN
        RAISE EXCEPTION 'Post-flight FAILED: shipped=% vs ordered=%', v_total_shipped, v_total_ordered;
    END IF;

    RAISE NOTICE 'Post-flight OK: order=shipped, %/% lines fulfilled, %/% lb shipped',
        v_lines_fulfilled, v_total_lines, v_total_shipped, v_total_ordered;
END $$;

COMMIT;

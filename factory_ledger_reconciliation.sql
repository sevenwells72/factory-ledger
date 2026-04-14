-- ============================================================================
-- FACTORY LEDGER — PRODUCTION RECONCILIATION REPORT
-- Run in Supabase SQL Editor as a single script.
-- Each section returns its own result set with a label column for clarity.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- TEST 1: Shipments with no transaction_lines (ghost shipments)
-- Catches: no_stock false-partial-ship creating empty shipment rows
-- Expected healthy result: 0 rows
-- ----------------------------------------------------------------------------
SELECT
    'T1: Ghost Shipments' AS test,
    s.id AS shipment_id,
    so.order_number,
    so.status AS order_status,
    s.shipped_at
FROM shipments s
JOIN sales_orders so ON so.id = s.sales_order_id
LEFT JOIN shipment_lines sl ON sl.shipment_id = s.id
WHERE sl.id IS NULL
ORDER BY s.shipped_at DESC;


-- ----------------------------------------------------------------------------
-- TEST 2: Order line shipped qty vs actual linked transaction_lines
-- Catches: quantity_shipped_lb drifting from ledger reality
-- Expected healthy result: 0 rows
-- ----------------------------------------------------------------------------
SELECT
    'T2: Shipped Qty Mismatch' AS test,
    sol.id AS line_id,
    so.order_number,
    so.status AS order_status,
    p.name AS product,
    sol.quantity_shipped_lb AS recorded_shipped,
    COALESCE(ABS(SUM(tl.quantity_lb)), 0) AS actual_shipped,
    sol.quantity_shipped_lb - COALESCE(ABS(SUM(tl.quantity_lb)), 0) AS drift_lb
FROM sales_order_lines sol
JOIN sales_orders so ON so.id = sol.sales_order_id
JOIN products p ON p.id = sol.product_id
LEFT JOIN sales_order_shipments sos ON sos.sales_order_line_id = sol.id
LEFT JOIN transaction_lines tl
    ON tl.transaction_id = sos.transaction_id
    AND tl.product_id = sol.product_id
GROUP BY sol.id, so.order_number, so.status, p.name
HAVING sol.quantity_shipped_lb != COALESCE(ABS(SUM(tl.quantity_lb)), 0)
ORDER BY ABS(sol.quantity_shipped_lb - COALESCE(ABS(SUM(tl.quantity_lb)), 0)) DESC;


-- ----------------------------------------------------------------------------
-- TEST 3: Standalone ship transactions for customers with open orders
-- Catches: force_standalone bypassing order linkage (Bin 1 — inventory
--          correct but order thinks it's unfulfilled)
-- Expected healthy result: 0 rows (or only rows with standalone_override noted)
-- ----------------------------------------------------------------------------
SELECT
    'T3: Standalone Ship w/ Open Orders' AS test,
    t.id AS txn_id,
    t.timestamp,
    t.customer_name,
    t.order_reference,
    t.notes,
    ABS(SUM(tl.quantity_lb)) AS shipped_lb,
    (SELECT COUNT(*)
     FROM sales_orders so2
     JOIN customers c2 ON c2.id = so2.customer_id
     WHERE c2.name = t.customer_name
       AND so2.status NOT IN ('invoiced', 'cancelled')
    ) AS open_order_count
FROM transactions t
JOIN transaction_lines tl ON tl.transaction_id = t.id
WHERE t.type = 'ship'
  AND t.id NOT IN (SELECT transaction_id FROM sales_order_shipments)
  AND t.customer_name IN (
      SELECT c.name
      FROM customers c
      JOIN sales_orders so ON so.customer_id = c.id
      WHERE so.status NOT IN ('invoiced', 'cancelled')
  )
GROUP BY t.id, t.timestamp, t.customer_name, t.order_reference, t.notes
ORDER BY t.timestamp DESC;


-- ----------------------------------------------------------------------------
-- TEST 4: Orders in partial_ship/shipped with zero actual shipment lines
-- Catches: status set without any real inventory movement
-- Expected healthy result: 0 rows
-- ----------------------------------------------------------------------------
SELECT
    'T4: Shipped Status w/ No Shipment Lines' AS test,
    so.id AS order_id,
    so.order_number,
    so.status,
    c.name AS customer
FROM sales_orders so
LEFT JOIN customers c ON c.id = so.customer_id
WHERE so.status IN ('partial_ship', 'shipped')
  AND NOT EXISTS (
      SELECT 1
      FROM sales_order_shipments sos
      JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
      WHERE sol.sales_order_id = so.id
  )
ORDER BY so.order_number DESC;


-- ----------------------------------------------------------------------------
-- TEST 5: Negative on-hand balances (smoke alarm for double-decrement)
-- Catches: partial commits or duplicate shipments
-- Expected healthy result: 0 rows
-- ----------------------------------------------------------------------------
SELECT
    'T5: Negative Balance' AS test,
    p.odoo_code,
    p.name AS product,
    l.lot_code,
    ROUND(SUM(tl.quantity_lb)::numeric, 2) AS balance_lb
FROM products p
JOIN lots l ON l.product_id = p.id
JOIN transaction_lines tl ON tl.lot_id = l.id
GROUP BY p.id, p.odoo_code, p.name, l.id, l.lot_code
HAVING SUM(tl.quantity_lb) < -0.01
ORDER BY SUM(tl.quantity_lb) ASC;


-- ----------------------------------------------------------------------------
-- TEST 6: Orphaned shipment references (linkage row points to empty txn)
-- Catches: broken join between sales_order_shipments and transaction_lines
-- Expected healthy result: 0 rows
-- ----------------------------------------------------------------------------
SELECT
    'T6: Orphaned Shipment Refs' AS test,
    sos.id AS shipment_link_id,
    sos.transaction_id,
    sos.quantity_lb AS linked_qty,
    so.order_number,
    p.name AS product
FROM sales_order_shipments sos
JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
JOIN sales_orders so ON so.id = sol.sales_order_id
JOIN products p ON p.id = sol.product_id
WHERE NOT EXISTS (
    SELECT 1
    FROM transaction_lines tl
    WHERE tl.transaction_id = sos.transaction_id
)
ORDER BY so.order_number;


-- ----------------------------------------------------------------------------
-- SUMMARY: Quick row counts (run this last for a dashboard view)
-- ----------------------------------------------------------------------------
SELECT 'SUMMARY' AS test, 'T1: Ghost Shipments' AS check_name,
    (SELECT COUNT(*) FROM shipments s
     LEFT JOIN shipment_lines sl ON sl.shipment_id = s.id
     WHERE sl.id IS NULL) AS row_count
UNION ALL
SELECT 'SUMMARY', 'T2: Shipped Qty Mismatch',
    (SELECT COUNT(*) FROM (
        SELECT sol.id
        FROM sales_order_lines sol
        LEFT JOIN sales_order_shipments sos ON sos.sales_order_line_id = sol.id
        LEFT JOIN transaction_lines tl
            ON tl.transaction_id = sos.transaction_id
            AND tl.product_id = sol.product_id
        GROUP BY sol.id, sol.quantity_shipped_lb
        HAVING sol.quantity_shipped_lb != COALESCE(ABS(SUM(tl.quantity_lb)), 0)
    ) sub)
UNION ALL
SELECT 'SUMMARY', 'T3: Standalone Ship w/ Open Orders',
    (SELECT COUNT(DISTINCT t.id)
     FROM transactions t
     WHERE t.type = 'ship'
       AND t.id NOT IN (SELECT transaction_id FROM sales_order_shipments)
       AND t.customer_name IN (
           SELECT c.name FROM customers c
           JOIN sales_orders so ON so.customer_id = c.id
           WHERE so.status NOT IN ('invoiced', 'cancelled')
       ))
UNION ALL
SELECT 'SUMMARY', 'T4: Shipped Status w/ No Shipment Lines',
    (SELECT COUNT(*) FROM sales_orders so
     WHERE so.status IN ('partial_ship', 'shipped')
       AND NOT EXISTS (
           SELECT 1 FROM sales_order_shipments sos
           JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
           WHERE sol.sales_order_id = so.id
       ))
UNION ALL
SELECT 'SUMMARY', 'T5: Negative Balances',
    (SELECT COUNT(*) FROM (
        SELECT l.id
        FROM lots l
        JOIN transaction_lines tl ON tl.lot_id = l.id
        GROUP BY l.id
        HAVING SUM(tl.quantity_lb) < -0.01
    ) sub)
UNION ALL
SELECT 'SUMMARY', 'T6: Orphaned Shipment Refs',
    (SELECT COUNT(*) FROM sales_order_shipments sos
     WHERE NOT EXISTS (
         SELECT 1 FROM transaction_lines tl
         WHERE tl.transaction_id = sos.transaction_id
     ));
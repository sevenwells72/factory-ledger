-- ═══════════════════════════════════════════════════════════════
-- Migration 018: Backfill Pre-Migration Shipments
-- Creates shipments + shipment_lines records for 11 ship transactions
-- (IDs: 296-329) from Feb 20-26, created before Migration 013.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

-- Step 1: Add transaction_id column to shipments for tracking
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS transaction_id INTEGER REFERENCES transactions(id);

-- Step 2: Allow standalone shipments (no sales order, no customer initially)
ALTER TABLE shipments ALTER COLUMN sales_order_id DROP NOT NULL;
ALTER TABLE shipments ALTER COLUMN customer_id DROP NOT NULL;

-- Step 3: Create shipment records for pre-migration ship transactions
-- These transactions may have sales_order_shipments records (SO-based)
-- or may be standalone (/ship endpoint). Handle both cases.
INSERT INTO shipments (transaction_id, sales_order_id, customer_id, shipped_at)
SELECT DISTINCT ON (t.id)
    t.id,
    NULL::integer,  -- will be resolved in Step 4 via sales_order_shipments
    c.id,
    t.timestamp
FROM transactions t
LEFT JOIN customers c ON LOWER(c.name) = LOWER(t.customer_name)
LEFT JOIN shipments s_existing ON s_existing.transaction_id = t.id
WHERE t.id IN (296, 297, 304, 306, 307, 308, 311, 313, 327, 328, 329)
  AND t.type = 'ship'
  AND s_existing.id IS NULL
ON CONFLICT DO NOTHING;

-- For rows where we couldn't resolve the customer, try with the actual
-- sales_order_id from sales_order_shipments
UPDATE shipments s
SET sales_order_id = so.id,
    customer_id = COALESCE(s.customer_id, so.customer_id)
FROM sales_order_shipments sos
JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
JOIN sales_orders so ON so.id = sol.sales_order_id
WHERE s.transaction_id = sos.transaction_id
  AND s.transaction_id IN (296, 297, 304, 306, 307, 308, 311, 313, 327, 328, 329);

-- Step 4: Create shipment_lines from transaction_lines
-- Make sales_order_line_id nullable for standalone shipments
ALTER TABLE shipment_lines ALTER COLUMN sales_order_line_id DROP NOT NULL;

INSERT INTO shipment_lines (shipment_id, transaction_id, sales_order_line_id, product_id, quantity_lb)
SELECT s.id, tl.transaction_id, sos.sales_order_line_id, tl.product_id, ABS(tl.quantity_lb)
FROM shipments s
JOIN transaction_lines tl ON tl.transaction_id = s.transaction_id
LEFT JOIN sales_order_shipments sos ON sos.transaction_id = tl.transaction_id
LEFT JOIN shipment_lines sl_existing ON sl_existing.shipment_id = s.id AND sl_existing.transaction_id = tl.transaction_id AND sl_existing.product_id = tl.product_id
WHERE s.transaction_id IN (296, 297, 304, 306, 307, 308, 311, 313, 327, 328, 329)
  AND tl.quantity_lb < 0  -- ship lines are negative
  AND sl_existing.id IS NULL
ON CONFLICT DO NOTHING;

-- Step 5: Verify
-- SELECT t.id, s.id as shipment_id, COUNT(sl.id) as line_count
-- FROM transactions t
-- LEFT JOIN shipments s ON s.transaction_id = t.id
-- LEFT JOIN shipment_lines sl ON sl.shipment_id = s.id
-- WHERE t.id IN (296, 297, 304, 306, 307, 308, 311, 313, 327, 328, 329)
-- GROUP BY t.id, s.id;

-- DOWNGRADE
-- DELETE FROM shipment_lines WHERE shipment_id IN (
--   SELECT id FROM shipments WHERE transaction_id IN (296,297,304,306,307,308,311,313,327,328,329)
-- );
-- DELETE FROM shipments WHERE transaction_id IN (296,297,304,306,307,308,311,313,327,328,329);
-- ALTER TABLE shipments DROP COLUMN IF EXISTS transaction_id;
-- ALTER TABLE shipments ALTER COLUMN sales_order_id SET NOT NULL;
-- ALTER TABLE shipments ALTER COLUMN customer_id SET NOT NULL;
-- ALTER TABLE shipment_lines ALTER COLUMN sales_order_line_id SET NOT NULL;

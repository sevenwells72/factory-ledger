-- ═══════════════════════════════════════════════════════════════
-- Migration 030: Backfill Standalone Shipment Records
-- Creates shipments + shipment_lines for all standalone ship
-- transactions that don't already have shipment records.
-- Part of GAP-3: unified shipping model.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

BEGIN;

-- Step 1: Create shipment headers for orphaned ship transactions
-- (ship transactions with no corresponding shipments row)
INSERT INTO shipments (transaction_id, shipped_at, customer_id)
SELECT
    t.id,
    t.timestamp,
    c.id
FROM transactions t
LEFT JOIN shipments s ON s.transaction_id = t.id
LEFT JOIN customers c ON LOWER(c.name) = LOWER(t.customer_name)
WHERE t.type = 'ship'
  AND COALESCE(t.status, 'posted') = 'posted'
  AND s.id IS NULL;

-- Step 2: Create shipment_lines from transaction_lines for newly created shipments
-- Uses ABS() since transaction_lines store negative quantities for shipments
INSERT INTO shipment_lines (shipment_id, transaction_id, product_id, quantity_lb)
SELECT
    s.id,
    tl.transaction_id,
    tl.product_id,
    ABS(tl.quantity_lb)
FROM shipments s
JOIN transaction_lines tl ON tl.transaction_id = s.transaction_id
LEFT JOIN shipment_lines sl ON sl.shipment_id = s.id
                            AND sl.transaction_id = tl.transaction_id
                            AND sl.product_id = tl.product_id
WHERE tl.quantity_lb < 0        -- only ship lines (negative)
  AND sl.id IS NULL             -- not already backfilled
  AND s.sales_order_id IS NULL; -- standalone shipments only

COMMIT;

-- VERIFICATION (run manually after migration):
--
-- Should return 0 rows (no orphaned ship transactions):
-- SELECT t.id, t.timestamp, t.customer_name
-- FROM transactions t
-- LEFT JOIN shipments s ON s.transaction_id = t.id
-- WHERE t.type = 'ship'
--   AND COALESCE(t.status, 'posted') = 'posted'
--   AND s.id IS NULL;
--
-- Count of backfilled records:
-- SELECT COUNT(*) FROM shipments WHERE sales_order_id IS NULL;
-- SELECT COUNT(*) FROM shipment_lines sl
-- JOIN shipments s ON s.id = sl.shipment_id
-- WHERE s.sales_order_id IS NULL;

-- DOWNGRADE:
-- DELETE FROM shipment_lines WHERE shipment_id IN (
--   SELECT id FROM shipments WHERE sales_order_id IS NULL
--     AND transaction_id NOT IN (296,297,304,306,307,308,311,313,327,328,329)
-- );
-- DELETE FROM shipments WHERE sales_order_id IS NULL
--   AND transaction_id NOT IN (296,297,304,306,307,308,311,313,327,328,329);

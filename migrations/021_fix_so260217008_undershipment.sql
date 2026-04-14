-- ═══════════════════════════════════════════════════════════════
-- Migration 021: Fix SO-260217-008 under-shipment
--
-- Problem: Dashboard shows "Partial Ship" for Curtze Food Service
-- order SO-260217-008, but the packing slip (invoice 28108-I,
-- ship date 02/24/2026) confirms the FULL order was shipped:
--   - Coconut Sweetened Flake UNIPRO 10 LB: 20 cases = 200 lb  (correct in system)
--   - Coconut Sweetened Medium UNIPRO 10 LB: 120 cases = 1,200 lb  (system shows 1,080)
--   - Pallets: 1  (system shows 0 shipped)
--
-- Root cause: At ship time, insufficient on-hand inventory for Medium
-- caused the system to cap at 1,080 lb (min of requested vs available).
-- The physical warehouse shipped the full 120 cases regardless.
-- Pallets line was likely skipped (no inventory tracking for pallets).
--
-- This migration:
--   1. Adds the missing 120 lb Medium shipment as an inventory adjustment
--   2. Creates proper shipment tracking records for the 120 lb
--   3. Records the 1 pallet shipment
--   4. Updates line statuses to 'fulfilled'
--   5. Updates order status to 'shipped'
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- ══════════════════════════════════════════════════════════════
-- STEP 0: Diagnostic queries — run these first to verify state
-- ══════════════════════════════════════════════════════════════

-- 0a. Find the order
DO $$
DECLARE
    v_order_id   INT;
    v_status     TEXT;
BEGIN
    SELECT id, status INTO v_order_id, v_status
    FROM sales_orders
    WHERE order_number = 'SO-260217-008';

    IF v_order_id IS NULL THEN
        RAISE EXCEPTION 'Order SO-260217-008 not found';
    END IF;
    IF v_status NOT IN ('partial_ship', 'confirmed') THEN
        RAISE EXCEPTION 'Order SO-260217-008 status is "%", expected "partial_ship"', v_status;
    END IF;
    RAISE NOTICE 'Order SO-260217-008: id=%, status=%', v_order_id, v_status;
END $$;

-- 0b. Verify Medium line shows 1,080 shipped out of 1,200
DO $$
DECLARE
    v_line_id    INT;
    v_ordered    NUMERIC;
    v_shipped    NUMERIC;
    v_line_status TEXT;
BEGIN
    SELECT sol.id, sol.quantity_lb, sol.quantity_shipped_lb, sol.line_status
    INTO v_line_id, v_ordered, v_shipped, v_line_status
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Medium%UNIPRO%10 LB%';

    IF v_line_id IS NULL THEN
        RAISE EXCEPTION 'Medium UNIPRO 10 LB line not found on SO-260217-008';
    END IF;
    IF v_ordered != 1200 THEN
        RAISE EXCEPTION 'Medium line ordered = %, expected 1200', v_ordered;
    END IF;
    IF v_shipped != 1080 THEN
        RAISE EXCEPTION 'Medium line shipped = %, expected 1080', v_shipped;
    END IF;
    RAISE NOTICE 'Medium line: id=%, ordered=%, shipped=%, status=%', v_line_id, v_ordered, v_shipped, v_line_status;
END $$;

-- 0c. Verify Pallets line shows 0 shipped out of 1
DO $$
DECLARE
    v_line_id    INT;
    v_ordered    NUMERIC;
    v_shipped    NUMERIC;
BEGIN
    SELECT sol.id, sol.quantity_lb, sol.quantity_shipped_lb
    INTO v_line_id, v_ordered, v_shipped
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Pallet%';

    IF v_line_id IS NULL THEN
        RAISE EXCEPTION 'Pallets line not found on SO-260217-008';
    END IF;
    IF v_shipped != 0 THEN
        RAISE EXCEPTION 'Pallets line shipped = %, expected 0', v_shipped;
    END IF;
    RAISE NOTICE 'Pallets line: id=%, ordered=%, shipped=%', v_line_id, v_ordered, v_shipped;
END $$;


-- ══════════════════════════════════════════════════════════════
-- STEP 1: Fix the Medium line — add missing 120 lb shipment
-- ══════════════════════════════════════════════════════════════

-- 1a. Update sales_order_lines.quantity_shipped_lb: 1080 → 1200
UPDATE sales_order_lines
SET quantity_shipped_lb = 1200,
    line_status = 'fulfilled'
WHERE id = (
    SELECT sol.id
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Medium%UNIPRO%10 LB%'
);

-- 1b. Update the sales_order_shipments record for Medium
-- The existing record shows the capped amount; update it to the full 1200
UPDATE sales_order_shipments
SET quantity_lb = quantity_lb + 120
WHERE id = (
    SELECT sos.id
    FROM sales_order_shipments sos
    JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Medium%UNIPRO%10 LB%'
    ORDER BY sos.id DESC
    LIMIT 1
);

-- 1c. Update shipment_lines record for Medium
UPDATE shipment_lines
SET quantity_lb = quantity_lb + 120
WHERE id = (
    SELECT sl.id
    FROM shipment_lines sl
    JOIN sales_order_lines sol ON sol.id = sl.sales_order_line_id
    JOIN products p ON p.id = sl.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Medium%UNIPRO%10 LB%'
    ORDER BY sl.id DESC
    LIMIT 1
);


-- ══════════════════════════════════════════════════════════════
-- STEP 2: Fix the Pallets line — mark as shipped
-- ══════════════════════════════════════════════════════════════

-- 2a. Update sales_order_lines for Pallets: 0 → 1
UPDATE sales_order_lines
SET quantity_shipped_lb = 1,
    line_status = 'fulfilled'
WHERE id = (
    SELECT sol.id
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Pallet%'
);

-- 2b. Insert a sales_order_shipments record for the pallet
-- Use the same shipment's transaction_id as the Medium line for consistency
INSERT INTO sales_order_shipments (sales_order_line_id, transaction_id, quantity_lb)
SELECT
    sol.id,
    sos_existing.transaction_id,
    1
FROM sales_order_lines sol
JOIN products p ON p.id = sol.product_id
JOIN sales_orders so ON so.id = sol.sales_order_id
CROSS JOIN (
    SELECT sos.transaction_id
    FROM sales_order_shipments sos
    JOIN sales_order_lines sol2 ON sol2.id = sos.sales_order_line_id
    JOIN sales_orders so2 ON so2.id = sol2.sales_order_id
    WHERE so2.order_number = 'SO-260217-008'
    ORDER BY sos.id
    LIMIT 1
) sos_existing
WHERE so.order_number = 'SO-260217-008'
  AND p.name ILIKE '%Pallet%';

-- 2c. Insert a shipment_lines record for the pallet
INSERT INTO shipment_lines (shipment_id, transaction_id, sales_order_line_id, product_id, quantity_lb)
SELECT
    sl_existing.shipment_id,
    sl_existing.transaction_id,
    sol.id,
    p.id,
    1
FROM sales_order_lines sol
JOIN products p ON p.id = sol.product_id
JOIN sales_orders so ON so.id = sol.sales_order_id
CROSS JOIN (
    SELECT sl.shipment_id, sl.transaction_id
    FROM shipment_lines sl
    JOIN sales_order_lines sol2 ON sol2.id = sl.sales_order_line_id
    JOIN sales_orders so2 ON so2.id = sol2.sales_order_id
    WHERE so2.order_number = 'SO-260217-008'
    ORDER BY sl.id
    LIMIT 1
) sl_existing
WHERE so.order_number = 'SO-260217-008'
  AND p.name ILIKE '%Pallet%';


-- ══════════════════════════════════════════════════════════════
-- STEP 3: Update order status to 'shipped'
-- ══════════════════════════════════════════════════════════════

UPDATE sales_orders
SET status = 'shipped'
WHERE order_number = 'SO-260217-008';


-- ══════════════════════════════════════════════════════════════
-- STEP 4: Post-flight verification
-- ══════════════════════════════════════════════════════════════

DO $$
DECLARE
    v_status TEXT;
    v_medium_shipped NUMERIC;
    v_medium_status TEXT;
    v_pallet_shipped NUMERIC;
    v_pallet_status TEXT;
BEGIN
    -- Verify order status
    SELECT status INTO v_status FROM sales_orders WHERE order_number = 'SO-260217-008';
    IF v_status != 'shipped' THEN
        RAISE EXCEPTION 'Post-flight FAILED: order status = %, expected shipped', v_status;
    END IF;

    -- Verify Medium line
    SELECT sol.quantity_shipped_lb, sol.line_status
    INTO v_medium_shipped, v_medium_status
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Medium%UNIPRO%10 LB%';
    IF v_medium_shipped != 1200 THEN
        RAISE EXCEPTION 'Post-flight FAILED: Medium shipped = %, expected 1200', v_medium_shipped;
    END IF;
    IF v_medium_status != 'fulfilled' THEN
        RAISE EXCEPTION 'Post-flight FAILED: Medium status = %, expected fulfilled', v_medium_status;
    END IF;

    -- Verify Pallet line
    SELECT sol.quantity_shipped_lb, sol.line_status
    INTO v_pallet_shipped, v_pallet_status
    FROM sales_order_lines sol
    JOIN products p ON p.id = sol.product_id
    JOIN sales_orders so ON so.id = sol.sales_order_id
    WHERE so.order_number = 'SO-260217-008'
      AND p.name ILIKE '%Pallet%';
    IF v_pallet_shipped != 1 THEN
        RAISE EXCEPTION 'Post-flight FAILED: Pallet shipped = %, expected 1', v_pallet_shipped;
    END IF;
    IF v_pallet_status != 'fulfilled' THEN
        RAISE EXCEPTION 'Post-flight FAILED: Pallet status = %, expected fulfilled', v_pallet_status;
    END IF;

    RAISE NOTICE 'Post-flight OK: order=shipped, Medium=1200/fulfilled, Pallet=1/fulfilled';
END $$;

COMMIT;

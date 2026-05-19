-- ═══════════════════════════════════════════════════════════════
-- Migration 029: Rename UNKNOWN lot to 25216 (Chocolate Sprinkles 25 LB)
--
-- Lot id=324, product_id=203 (Chocolate Sprinkles 25 LB) was created
-- with lot_code='UNKNOWN' because the actual code wasn't entered at
-- receive time. The supplier lot code (25216) was already backfilled
-- via the API, but the internal lot_code is still 'UNKNOWN'.
--
-- Schema audit: only lots.lot_code stores lot codes as strings.
-- All other tables (transaction_lines, shipment_lines,
-- ingredient_lot_consumption, sales_order_shipments) use integer
-- lot_id FKs, so only one UPDATE is needed.
--
-- ROLLBACK: UPDATE lots SET lot_code = 'UNKNOWN' WHERE id = 324;
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- Pre-flight: confirm this is the right lot and it's still UNKNOWN
DO $$
DECLARE
    v_lot_id     INT;
    v_lot_code   TEXT;
    v_product    TEXT;
    v_product_id INT;
BEGIN
    SELECT l.id, l.lot_code, p.name, l.product_id
    INTO v_lot_id, v_lot_code, v_product, v_product_id
    FROM lots l
    JOIN products p ON p.id = l.product_id
    WHERE l.id = 324;

    IF v_lot_id IS NULL THEN
        RAISE EXCEPTION 'Lot 324 not found — aborting';
    END IF;

    IF v_product_id != 203 THEN
        RAISE EXCEPTION 'Lot 324 product_id is %, expected 203 — aborting', v_product_id;
    END IF;

    IF v_lot_code != 'UNKNOWN' THEN
        -- Idempotent: if already renamed, skip silently
        IF v_lot_code = '25216' THEN
            RAISE NOTICE 'Lot 324 already renamed to 25216 — nothing to do';
            RETURN;
        ELSE
            RAISE EXCEPTION 'Lot 324 lot_code is %, expected UNKNOWN — aborting', v_lot_code;
        END IF;
    END IF;

    RAISE NOTICE 'Pre-flight OK: lot_id=%, product=%, lot_code=%', v_lot_id, v_product, v_lot_code;
END $$;

-- Check no conflict: 25216 must not already exist for product 203
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM lots WHERE lot_code = '25216' AND product_id = 203
    ) THEN
        RAISE EXCEPTION 'Lot code 25216 already exists for product 203 — aborting';
    END IF;
END $$;

-- Rename the lot
UPDATE lots
SET lot_code = '25216'
WHERE id = 324 AND product_id = 203 AND lot_code = 'UNKNOWN';

-- Post-flight: confirm update
DO $$
DECLARE
    v_lot_code TEXT;
BEGIN
    SELECT lot_code INTO v_lot_code
    FROM lots WHERE id = 324;

    IF v_lot_code != '25216' THEN
        RAISE EXCEPTION 'Post-flight FAILED: lot_code = %', v_lot_code;
    END IF;

    RAISE NOTICE 'Post-flight OK: lot 324 renamed to %', v_lot_code;
END $$;

COMMIT;

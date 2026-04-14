-- ═══════════════════════════════════════════════════════════════
-- Migration 025: Set supplier lot code on lot 26-03-10-FOUN-001
--
-- Lot 26-03-10-FOUN-001 (Sprinkles Rainbow 10 LB, id 286) was
-- used to ship 23 cases (230 lb) to International Gourmet Foods
-- (SO-260318-001, Shipment #32). Packing slip shows supplier
-- lot 550078168. Adding this cross-reference for traceability.
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- Pre-flight: verify lot exists with expected state
DO $$
DECLARE
    v_lot_id     INT;
    v_lot_code   TEXT;
    v_product    TEXT;
    v_current    TEXT;
BEGIN
    SELECT l.id, l.lot_code, p.name, l.supplier_lot_code
    INTO v_lot_id, v_lot_code, v_product, v_current
    FROM lots l
    JOIN products p ON p.id = l.product_id
    WHERE l.lot_code = '26-03-10-FOUN-001';

    IF v_lot_id IS NULL THEN
        RAISE EXCEPTION 'Lot 26-03-10-FOUN-001 not found';
    END IF;

    IF v_product NOT LIKE '%Sprinkles Rainbow%' THEN
        RAISE EXCEPTION 'Unexpected product: %', v_product;
    END IF;

    RAISE NOTICE 'Pre-flight OK: lot_id=%, product=%, current supplier_lot=%',
        v_lot_id, v_product, v_current;
END $$;

-- Update supplier lot code
UPDATE lots
SET supplier_lot_code = '550078168'
WHERE lot_code = '26-03-10-FOUN-001';

-- Post-flight: confirm update
DO $$
DECLARE
    v_supplier TEXT;
BEGIN
    SELECT supplier_lot_code INTO v_supplier
    FROM lots WHERE lot_code = '26-03-10-FOUN-001';

    IF v_supplier != '550078168' THEN
        RAISE EXCEPTION 'Post-flight FAILED: supplier_lot_code = %', v_supplier;
    END IF;

    RAISE NOTICE 'Post-flight OK: supplier_lot_code set to %', v_supplier;
END $$;

COMMIT;

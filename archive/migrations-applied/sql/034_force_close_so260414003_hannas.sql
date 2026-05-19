-- ═══════════════════════════════════════════════════════════════
-- Migration 034: Force-close SO-260414-003 (Hannas Gourmet) to shipped
--
-- Problem: Order was physically shipped (customer pick up on
-- 03/11/2026, invoice 28123-I dated 03/02/2026) but entered into
-- the system retroactively on 04/14/2026 without ledger
-- transactions. Business decision: flip status to 'shipped'
-- without creating underlying shipment/transaction records and
-- preserve the paper-trail details in the order notes.
--
-- ⚠️ WARNING — This migration intentionally bypasses the
-- PATCH /sales/orders/{id}/status guard (main.py:5375) that blocks
-- manual 'shipped' status changes. Consequences:
--   • sales_dashboard shipped_lb for this order will stay at 0
--   • No transaction_lines rows → on-hand inventory NOT decremented
--   • No sales_order_shipments / shipment_lines rows written
--   • Lot traceability unrecorded in the DB (only in notes)
--
-- Lot breakdown (for reference, NOT written to tables):
--   Granola Vanilla Almond 25 LB — 10 units
--     Lot: JAN 20 2026
--   Pallet Charge — 1 unit
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- Pre-flight: verify order exists and is in a flippable state
DO $$
DECLARE
    v_id     INT;
    v_status TEXT;
BEGIN
    SELECT id, status
    INTO v_id, v_status
    FROM sales_orders
    WHERE order_number = 'SO-260414-003';

    IF v_id IS NULL THEN
        RAISE EXCEPTION 'SO-260414-003 not found';
    END IF;
    IF v_status = 'shipped' THEN
        RAISE EXCEPTION 'SO-260414-003 already shipped — nothing to do';
    END IF;
    IF v_status IN ('cancelled', 'voided') THEN
        RAISE EXCEPTION 'SO-260414-003 is % — cannot flip to shipped', v_status;
    END IF;
    RAISE NOTICE 'Pre-flight OK: SO-260414-003 id=%, status=%', v_id, v_status;
END $$;

-- Flip status and append shipment paper-trail to notes
UPDATE sales_orders
SET status = 'shipped',
    notes  = COALESCE(notes, '') || E'\n\n' ||
             '— SHIPMENT RECORD (manually closed, not in ledger) —' || E'\n' ||
             'Bill/Ship To: Hannas Gourmet, 1330-14 Lincoln Ave, Holbrook, NY 11741' || E'\n' ||
             'Ship Date: 03/11/2026 (Customer Pick Up)' || E'\n' ||
             'Invoice: 28123-I, dated 03/02/2026' || E'\n' ||
             'PO: 2026-0099-SW' || E'\n' ||
             '' || E'\n' ||
             'Granola Vanilla Almond 25 LB — 10 units:' || E'\n' ||
             '  Lot: JAN 20 2026' || E'\n' ||
             'Pallet Charge — 1 unit'
WHERE order_number = 'SO-260414-003';

-- Post-flight
DO $$
DECLARE
    v_status TEXT;
    v_notes  TEXT;
BEGIN
    SELECT status, notes INTO v_status, v_notes
    FROM sales_orders WHERE order_number = 'SO-260414-003';
    IF v_status != 'shipped' THEN
        RAISE EXCEPTION 'Post-flight FAILED: status = %', v_status;
    END IF;
    IF v_notes NOT LIKE '%Invoice: 28123-I%' THEN
        RAISE EXCEPTION 'Post-flight FAILED: invoice note missing';
    END IF;
    RAISE NOTICE 'Post-flight OK: SO-260414-003 is shipped with paper-trail note';
END $$;

COMMIT;

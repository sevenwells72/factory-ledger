-- ═══════════════════════════════════════════════════════════════
-- Migration 033: Force-close SO-260326-002 (Ace Endico) to shipped
--
-- Problem: Order was physically shipped (customer pick up on
-- 04/13/2026, invoice 28159-I) but no ledger transactions were
-- written at the time. Business decision: flip status to 'shipped'
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
--   Unipro Fancy Sweet Shred Coconut 10 lb Case — 140 cases
--     MAR 30 2026 — 112 cases
--     MAR 31 2026 —  28 cases
--   Unipro Medium Sweetened Coconut 10 lb Case — 20 cases
--     MAR 26 2026 —   6 cases
--     MAR 27 2026 —  14 cases
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- Pre-flight: verify order exists and is in a flippable state
DO $$
DECLARE
    v_id     INT;
    v_status TEXT;
    v_notes  TEXT;
BEGIN
    SELECT id, status, notes
    INTO v_id, v_status, v_notes
    FROM sales_orders
    WHERE order_number = 'SO-260326-002';

    IF v_id IS NULL THEN
        RAISE EXCEPTION 'SO-260326-002 not found';
    END IF;
    IF v_status = 'shipped' THEN
        RAISE EXCEPTION 'SO-260326-002 already shipped — nothing to do';
    END IF;
    IF v_status IN ('cancelled', 'voided') THEN
        RAISE EXCEPTION 'SO-260326-002 is % — cannot flip to shipped', v_status;
    END IF;
    RAISE NOTICE 'Pre-flight OK: SO-260326-002 id=%, status=%', v_id, v_status;
END $$;

-- Flip status and append shipment paper-trail to notes
UPDATE sales_orders
SET status = 'shipped',
    notes  = COALESCE(notes, '') || E'\n\n' ||
             '— SHIPMENT RECORD (manually closed, not in ledger) —' || E'\n' ||
             'Bill/Ship To: Ace Endico, 80 International Blvd, Brewster, NY 10509' || E'\n' ||
             'Ship Date: 04/13/2026 (Customer Pick Up)' || E'\n' ||
             'Invoice: 28159-I, dated 03/26/2026' || E'\n' ||
             'PO: 624249' || E'\n' ||
             '' || E'\n' ||
             'Unipro Fancy Sweet Shred Coconut 10 lb Case — 140 cases:' || E'\n' ||
             '  MAR 30 2026 — 112 cases' || E'\n' ||
             '  MAR 31 2026 —  28 cases' || E'\n' ||
             'Unipro Medium Sweetened Coconut 10 lb Case — 20 cases:' || E'\n' ||
             '  MAR 26 2026 —  6 cases' || E'\n' ||
             '  MAR 27 2026 — 14 cases'
WHERE order_number = 'SO-260326-002';

-- Post-flight
DO $$
DECLARE
    v_status TEXT;
    v_notes  TEXT;
BEGIN
    SELECT status, notes INTO v_status, v_notes
    FROM sales_orders WHERE order_number = 'SO-260326-002';
    IF v_status != 'shipped' THEN
        RAISE EXCEPTION 'Post-flight FAILED: status = %', v_status;
    END IF;
    IF v_notes NOT LIKE '%Invoice: 28159-I%' THEN
        RAISE EXCEPTION 'Post-flight FAILED: invoice note missing';
    END IF;
    RAISE NOTICE 'Post-flight OK: SO-260326-002 is shipped with paper-trail note';
END $$;

COMMIT;

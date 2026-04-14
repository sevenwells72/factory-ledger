-- Migration 006: Add supplier_lot to lots for supplier traceability
-- Purpose: Store the supplier's internal lot number (e.g., Barry Callebaut lot 123168)
-- alongside the system-generated lot code. This enables full traceability from
-- supplier through production without relying on the BOL field.
--
-- supplier_lot is nullable (not all receives will have a supplier lot number)
-- and has no unique constraint (multiple receives could reference the same supplier lot).
--
-- This migration runs automatically at app startup via main.py.

ALTER TABLE lots
ADD COLUMN IF NOT EXISTS supplier_lot VARCHAR(100);

-- No index needed initially — the GPT search uses LIKE queries which won't benefit
-- from a btree index. If search performance becomes an issue, consider:
-- CREATE INDEX IF NOT EXISTS idx_lots_supplier_lot ON lots(supplier_lot) WHERE supplier_lot IS NOT NULL;

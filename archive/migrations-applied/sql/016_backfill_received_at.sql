-- ═══════════════════════════════════════════════════════════════
-- Migration 016: Backfill received_at on Historical Lots
-- Fixes 244 of 252 lots with NULL received_at for FIFO compliance.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

-- Step 1: Backfill from earliest receive transaction timestamp
WITH r AS (
    SELECT
        l.id AS lot_id,
        MIN(t.timestamp) AS recv_ts
    FROM lots l
    JOIN transaction_lines tl ON tl.lot_id = l.id
    JOIN transactions t ON t.id = tl.transaction_id
    WHERE t.type = 'receive'
      AND tl.quantity_lb > 0
    GROUP BY l.id
)
UPDATE lots l
SET received_at = r.recv_ts
FROM r
WHERE l.id = r.lot_id
  AND l.received_at IS NULL;

-- Step 2: Fallback for lots with no receive transaction (production output, found inventory)
UPDATE lots
SET received_at = created_at
WHERE received_at IS NULL;

-- Step 3: Verify
-- SELECT COUNT(*) FROM lots WHERE received_at IS NULL;
-- Expected: 0

-- DOWNGRADE
-- UPDATE lots SET received_at = NULL
-- WHERE received_at IS NOT NULL
--   AND id IN (SELECT id FROM lots WHERE received_at = created_at OR received_at IS NOT NULL);
-- Note: This is lossy — can't perfectly distinguish backfilled from original.

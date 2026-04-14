-- ═══════════════════════════════════════════════════════════════
-- Migration 017: Add transaction status column + void ghost makes
-- Status is for UI/history display ONLY — inventory SUM queries
-- must NEVER filter by status.
-- ═══════════════════════════════════════════════════════════════

-- UPGRADE

-- Step 1: Add status column
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'posted';
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);

-- Step 2: Void the ghost make transactions
UPDATE transactions
SET status = 'voided',
    notes = COALESCE(notes, '') || ' | Voided: audit 2026-03-05, ghost make (0 lb output, no ILC records)'
WHERE id IN (80, 83, 84, 177);

-- DOWNGRADE
-- DROP INDEX IF EXISTS idx_transactions_status;
-- ALTER TABLE transactions DROP COLUMN IF EXISTS status;

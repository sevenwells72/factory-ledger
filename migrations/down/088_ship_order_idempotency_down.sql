-- Receipts are a retry guard, not ledger evidence: the shipments they point at stay in transactions/shipments.
DROP TABLE IF EXISTS ship_order_commit_receipts;
DELETE FROM migration_markers WHERE name='088_ship_order_idempotency';

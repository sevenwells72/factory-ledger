-- Sales-order ship commit receipts: a retried commit returns the first receipt
-- instead of recording a second shipment (IDEMPOTENCY_KEY_PLAN.md §5, PR #146 review).
-- Additive only. Apply before deploying the code; until then the route behaves as before.
CREATE TABLE IF NOT EXISTS ship_order_commit_receipts (
    id              bigserial PRIMARY KEY,
    sales_order_id  integer     NOT NULL REFERENCES sales_orders(id),
    request_hash    text        NOT NULL CHECK (length(request_hash) = 64),
    shipment_id     integer     NOT NULL REFERENCES shipments(id),
    response_json   jsonb       NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS ship_order_commit_receipts_recent_idx
    ON ship_order_commit_receipts(sales_order_id, request_hash, created_at DESC);
ALTER TABLE ship_order_commit_receipts ENABLE ROW LEVEL SECURITY;
INSERT INTO migration_markers(name) VALUES ('088_ship_order_idempotency') ON CONFLICT DO NOTHING;

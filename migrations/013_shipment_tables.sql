-- ═══════════════════════════════════════════════════════════════
-- Migration 013: Shipment Tables
-- Creates shipments, shipment_lines, and sales_order_shipments
-- tables required by the sales order ship/commit endpoint.
-- ═══════════════════════════════════════════════════════════════

-- 1. shipments — One record per ship/commit execution
CREATE TABLE IF NOT EXISTS shipments (
    id              SERIAL PRIMARY KEY,
    sales_order_id  INTEGER NOT NULL REFERENCES sales_orders(id),
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    shipped_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shipments_sales_order_id
    ON shipments (sales_order_id);

CREATE INDEX IF NOT EXISTS idx_shipments_customer_id
    ON shipments (customer_id);

CREATE INDEX IF NOT EXISTS idx_shipments_shipped_at
    ON shipments (shipped_at);


-- 2. sales_order_shipments — Links shipment transactions to order lines
--    shipped_at is read in queries but not provided on INSERT,
--    so it defaults to now().
CREATE TABLE IF NOT EXISTS sales_order_shipments (
    id                    SERIAL PRIMARY KEY,
    sales_order_line_id   INTEGER NOT NULL REFERENCES sales_order_lines(id),
    transaction_id        INTEGER NOT NULL REFERENCES transactions(id),
    quantity_lb           NUMERIC NOT NULL,
    shipped_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sos_sales_order_line_id
    ON sales_order_shipments (sales_order_line_id);

CREATE INDEX IF NOT EXISTS idx_sos_transaction_id
    ON sales_order_shipments (transaction_id);

CREATE INDEX IF NOT EXISTS idx_sos_shipped_at
    ON sales_order_shipments (shipped_at);


-- 3. shipment_lines — Per-product detail within a shipment (v3.0.0)
CREATE TABLE IF NOT EXISTS shipment_lines (
    id                    SERIAL PRIMARY KEY,
    shipment_id           INTEGER NOT NULL REFERENCES shipments(id),
    transaction_id        INTEGER NOT NULL REFERENCES transactions(id),
    sales_order_line_id   INTEGER NOT NULL REFERENCES sales_order_lines(id),
    product_id            INTEGER NOT NULL REFERENCES products(id),
    quantity_lb           NUMERIC NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shipment_lines_shipment_id
    ON shipment_lines (shipment_id);

CREATE INDEX IF NOT EXISTS idx_shipment_lines_transaction_id
    ON shipment_lines (transaction_id);

CREATE INDEX IF NOT EXISTS idx_shipment_lines_product_id
    ON shipment_lines (product_id);

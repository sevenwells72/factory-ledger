-- Migration 012: Enable pg_trgm for fuzzy product search
-- Enables word-order-independent and typo-tolerant product matching

-- Enable the trigram extension (idempotent)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add trigram GIN index on product names for fast similarity queries
CREATE INDEX IF NOT EXISTS idx_products_name_trgm
    ON products USING gin (name gin_trgm_ops);

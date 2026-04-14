-- ═══════════════════════════════════════════════════════════════
-- Migration 005: Customer Aliases
-- Allows multiple name aliases per customer for reliable resolution.
-- Standalone /ship/commit resolves customer_name against both
-- customers.name AND customer_aliases.alias before checking for
-- open sales orders.
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS customer_aliases (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_customer_aliases_lower_alias
    ON customer_aliases (LOWER(alias));

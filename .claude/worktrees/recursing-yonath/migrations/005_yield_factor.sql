-- Migration 005: Add yield_factor to products for hydration yield tracking
-- Purpose: Sweetened coconut products (flake, shred, macaroon) absorb water during
-- hydration and gain weight. The yield_factor captures this multiplier so the system
-- can predict actual finished weight and auto-create the adjustment transaction,
-- eliminating the manual adjust step every production run.
--
-- yield_factor = 1.0 means no weight gain (default for all products).
-- yield_factor = 1.08 means 8% weight gain from hydration.
--
-- This migration runs automatically at app startup via main.py.
-- Exact yield factors for sweetened coconut products should be set by the operator
-- using PATCH /bom/{product_id} with {"yield_factor": 1.08}.

ALTER TABLE products
ADD COLUMN IF NOT EXISTS yield_factor NUMERIC(6,4) DEFAULT 1.0;

-- Note: No products are auto-flagged with yield_factor > 1.0.
-- The operator must provide the correct values for each product.
-- Products with "Sweetened" in the name are candidates — use PATCH /bom/{product_id}
-- to set the yield factor once the operator confirms the correct value.

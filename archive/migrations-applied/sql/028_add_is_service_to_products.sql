-- Migration 028: Add is_service flag to products table
-- Purpose: Distinguish service/charge line items (pallets, freight, etc.) from
--          actual product lines so weight calculations exclude them.
--
-- Run via Supabase SQL Editor on MyFirstProject (us-east-1)

ALTER TABLE products ADD COLUMN IF NOT EXISTS is_service BOOLEAN NOT NULL DEFAULT false;

-- Flag known service/charge products
UPDATE products
SET is_service = true
WHERE LOWER(name) LIKE '%pallet%'
   OR LOWER(name) LIKE '%freight%'
   OR LOWER(name) LIKE '%surcharge%'
   OR LOWER(name) LIKE '%delivery fee%'
   OR LOWER(name) LIKE '%charge%';

-- Verify
SELECT id, name, is_service FROM products WHERE is_service = true;

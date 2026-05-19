-- Migration 036: Drop products.bake_line column
-- Date: 2026-05-19
-- Slot history:
--   Originally authored in sibling worktree as 035_drop_bake_line.sql on
--   2026-05-19. Renumbered to 036 on 2026-05-19 12:?? ET to resolve a slot
--   conflict with main's 035_backfill_parent_batch_product_id_and_archive_retired_fgs.sql
--   (applied to production 2026-04-30, ~19 days earlier). Conflict resolved
--   by chronological-replay rule — production apply order is canonical,
--   regardless of which branch a migration was committed to first.
-- Context:
--   On 2026-05-18 a `bake_line` text column was added to `products` directly
--   via the Supabase SQL Editor (no migration file was committed for the add).
--   Seven Sunshine SKUs were set to bake_line='SS':
--     70002 Granola SS Original 12x10 OZ Case
--     70003 Granola SS Chocolate Chip 12x10 OZ Case
--     70004 Granola SS Original Bulk per/lb
--     70006 Granola SS Mini 100
--     70010 Granola SS Original Low Carb 12x10 OZ Case
--     70011 Granola SS Cranberry 12x10 OZ Case
--     70070 Granola SS Chocolate Chip Low Carb 12x10 OZ Case
--   The design changed: brand + parent_batch_product_id together capture the
--   same routing information (brand identifies the line owner, parent_batch_product_id
--   identifies the upstream bulk batch). The bake_line column is therefore redundant.
-- Pre-drop state (verified 2026-05-19 11:46 ET):
--   - products row count: 201
--   - bake_line NOT NULL count: 7 (matched the Sunshine SKUs above exactly)
--   - No code references to `bake_line` anywhere in the repo
--   - No views, functions, constraints, or indexes referenced the column
-- Note:
--   SET TRANSACTION READ WRITE is included as a no-op safety net for the
--   Supabase pooler quirk observed during prior DDL sessions where a stale
--   read-only flag occasionally lingers on a recycled pooled connection.

BEGIN;
SET TRANSACTION READ WRITE;
ALTER TABLE products DROP COLUMN bake_line;
COMMIT;

-- Post-commit verification (2026-05-19 11:47 ET):
--   - information_schema.columns WHERE column_name='bake_line': 0 rows
--   - products row count: 201 (unchanged)

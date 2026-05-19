-- Migration 034a (backfilled): Reconstructs the 2026-05-18 direct-SQL
-- session that was applied via Supabase SQL Editor at 15:23 ET before
-- migration discipline was enforced for this work stream. The transaction
-- below is the literal SQL that ran. Slotted at 034a (between existing
-- 034_force_close_so260414003_hannas.sql and
-- 035_backfill_parent_batch_product_id_and_archive_retired_fgs.sql) to
-- preserve real chronology for replay parity.
--
-- IMPORTANT: This SQL has already been applied to production. Do NOT run
-- against the live database. For migration replay against a fresh schema
-- only.
--
-- Note: 036_drop_bake_line.sql (originally authored as 035 in a sibling
-- worktree, renumbered 2026-05-19 to resolve a slot conflict with main's
-- 035_backfill_parent_batch_product_id_and_archive_retired_fgs.sql)
-- reverses the ADD COLUMN bake_line from this migration. Replay sequence
-- ..., 034, 034a, 035, 036 produces the correct current state (Fruit Nut
-- BOM present, is_copack present, bake_line absent).

BEGIN;

-- =========================================================================
-- TASK 1 — Fix Granola Fruit Nut batch product (id=179) + attach BOM
-- =========================================================================

-- 1a. Update product 179 in place
UPDATE products SET
    name                = 'Granola Fruit Nut Batch',           -- was 'Batch Granola Fruit Nut'
    brand               = 'CNS',                                -- was NULL  (mirror Classic)
    default_batch_lb    = 384.52,                               -- was 25.00 (THE fix)
    yield_25lb_cases    = 15,                                   -- was NULL  (floor(384.52/25))
    yield_10lb_cases    = 38,                                   -- was NULL  (floor(384.52/10))
    verification_status = 'verified',                           -- was 'unverified'
    verification_notes  = NULL,                                 -- clear quick-create stub note
    verified_at         = NOW()                                 -- record human verification time
WHERE id = 179;

-- 1b. Replace BOM for batch 179: clear-then-insert (clean replace, not append)
DELETE FROM product_bom WHERE finished_product_id = 179;

INSERT INTO product_bom (finished_product_id, component_product_id, quantity, uom) VALUES
    (179, 107, 323.00, 'lb'),   -- Classic granola batch (odoo_code 90002, sub-batch component)
    (179,   3,  15.38, 'lb'),   -- 11003 Almonds – Sliced     (4% inclusion)
    (179,  49,  15.38, 'lb'),   -- 11049 Walnuts              (4% inclusion)
    (179,  36,  15.38, 'lb'),   -- 11036 Raisins              (4% inclusion)
    (179,  17,  15.38, 'lb');   -- 11017 Cranberries Dried    (4% inclusion)

-- 1c. SKU 70061 (id=142) parent_batch_product_id is ALREADY 179. Verify only — no UPDATE.

-- =========================================================================
-- TASK 2 — Add bake_line column; mark 7 Sunshine SKUs as 'SS'
-- =========================================================================
ALTER TABLE products ADD COLUMN bake_line text;

UPDATE products SET bake_line = 'SS'
WHERE odoo_code IN ('70002','70003','70004','70006','70010','70011','70070');

-- =========================================================================
-- TASK 3 — Add is_copack column; mark 9 resale/co-pack SKUs
-- =========================================================================
ALTER TABLE products ADD COLUMN is_copack boolean NOT NULL DEFAULT false;

UPDATE products SET is_copack = true
WHERE odoo_code IN ('10301','10302','10303','10304','10305','10306',
                    '25013','25014','31012');

COMMIT;

-- ═══════════════════════════════════════════════════════════════
-- Migration 032: SKU backfill + merge duplicate BS cocoa products
--                + idempotent label_type guard on 70082
--
-- Context: product catalog cleanup. Three issues being resolved:
--
--   1. BACKFILL SKU 70089 on product id=171 ('Classic Granola 25 LB')
--      — odoo_code is currently NULL. Next-available code in the
--      700xx finished-goods range (max = 70088 + 1 = 70089).
--
--   2. MERGE two SKU-less duplicate ingredient products into the
--      canonical 15008 row:
--         id=61   odoo_code=15008   'BS Cocoa Liquor – Chips'  ← KEEP
--         id=167  odoo_code=NULL    'BS Cocoa Chips'           → merge
--         id=168  odoo_code=NULL    'BS Cocoa Liquor'          → merge
--      Reference scan (ids 167,168) found 9 rows to repoint across
--      4 tables:
--         lots.product_id                            (2 rows)
--         transaction_lines.product_id               (3 rows)
--         inventory_adjustments.product_id           (2 rows)
--         product_verification_history.product_id    (2 rows)
--      All other tables with a *product_id column had 0 rows.
--      Lot-code collision check: the two incoming lots
--      (26-01-30-FOUND-007, 26-01-30-FOUND-017) do NOT collide with
--      any existing (product_id=61, lot_code) row, so the
--      lots_product_id_lot_code_key UNIQUE constraint is satisfied.
--
--      product_verification_history handling: `action` column is
--      plain varchar with no CHECK constraint (current values are
--      'created' and 'verify'), so we set action='merged' and
--      merged_into_product_id=61 on the two existing rows. The
--      original 'created' action is overwritten — acceptable per
--      the user's instructions for this cleanup.
--
--   3. ASSERT label_type='private_label' on product id=133,
--      odoo_code=70082 ('Granola Setton French Vanilla 25 LB').
--      This is already the value in production; the UPDATE is
--      idempotent (WHERE label_type IS DISTINCT FROM 'private_label')
--      so this step is a 0-row no-op on current production but
--      will self-heal if a stale backup is ever restored.
--
-- Expected row count change: products goes from 203 → 201.
--
-- ROLLBACK: This migration deletes two product rows and repoints
-- nine FK references. A true rollback requires a pre-migration
-- backup of the products, lots, transaction_lines,
-- inventory_adjustments, and product_verification_history tables.
-- Inverse SQL (approximate, for reference only — will fail if the
-- original ids are reused):
--
--   BEGIN;
--   INSERT INTO products (id, name, odoo_code, type) VALUES
--     (167, 'BS Cocoa Chips',  NULL, 'ingredient'),
--     (168, 'BS Cocoa Liquor', NULL, 'ingredient');
--   UPDATE lots                          SET product_id = 167 WHERE id = 15;
--   UPDATE lots                          SET product_id = 168 WHERE id = 25;
--   UPDATE transaction_lines             SET product_id = 167 WHERE id = 1165;
--   UPDATE transaction_lines             SET product_id = 168 WHERE id IN (1175, 1229);
--   UPDATE inventory_adjustments         SET product_id = 167 WHERE id = 7;
--   UPDATE inventory_adjustments         SET product_id = 168 WHERE id = 17;
--   UPDATE product_verification_history  SET action = 'created', merged_into_product_id = NULL
--                                        WHERE product_id = 167 AND id = 3;
--   UPDATE product_verification_history  SET action = 'created', merged_into_product_id = NULL
--                                        WHERE product_id = 168 AND id = 4;
--   UPDATE products SET odoo_code = NULL WHERE id = 171 AND odoo_code = '70089';
--   COMMIT;
--
-- Run via Supabase SQL Editor on MyFirstProject (us-east-1).
-- ═══════════════════════════════════════════════════════════════

BEGIN;

-- ─────────────────────────────────────────────────────────────
-- PRE-FLIGHT: assert production is in the expected state.
-- Any deviation aborts the transaction before anything is touched.
-- ─────────────────────────────────────────────────────────────
DO $$
DECLARE
    v_canonical_id     INT;
    v_chips_id         INT;
    v_liquor_id        INT;
    v_classic_id       INT;
    v_classic_code     TEXT;
    v_70082_id         INT;
    v_70089_taken      INT;
    v_product_count    INT;
BEGIN
    -- Canonical BS cocoa product (must exist with code 15008)
    SELECT id INTO v_canonical_id
    FROM products WHERE odoo_code = '15008' AND name = 'BS Cocoa Liquor – Chips';
    IF v_canonical_id IS NULL THEN
        RAISE EXCEPTION 'Pre-flight: canonical BS cocoa product (15008) not found';
    END IF;
    IF v_canonical_id != 61 THEN
        RAISE EXCEPTION 'Pre-flight: canonical id changed — expected 61, got %', v_canonical_id;
    END IF;

    -- Two SKU-less duplicates (must exist with NULL odoo_code)
    SELECT id INTO v_chips_id
    FROM products WHERE name = 'BS Cocoa Chips' AND odoo_code IS NULL;
    IF v_chips_id IS NULL THEN
        RAISE NOTICE 'Pre-flight: BS Cocoa Chips with NULL odoo_code not found — '
                     'migration may have already been applied';
    ELSIF v_chips_id != 167 THEN
        RAISE EXCEPTION 'Pre-flight: BS Cocoa Chips id changed — expected 167, got %', v_chips_id;
    END IF;

    SELECT id INTO v_liquor_id
    FROM products WHERE name = 'BS Cocoa Liquor' AND odoo_code IS NULL;
    IF v_liquor_id IS NULL THEN
        RAISE NOTICE 'Pre-flight: BS Cocoa Liquor with NULL odoo_code not found — '
                     'migration may have already been applied';
    ELSIF v_liquor_id != 168 THEN
        RAISE EXCEPTION 'Pre-flight: BS Cocoa Liquor id changed — expected 168, got %', v_liquor_id;
    END IF;

    -- Classic Granola 25 LB must exist with NULL odoo_code (idempotent: allow already-set)
    SELECT id, odoo_code INTO v_classic_id, v_classic_code
    FROM products WHERE name = 'Classic Granola 25 LB';
    IF v_classic_id IS NULL THEN
        RAISE EXCEPTION 'Pre-flight: Classic Granola 25 LB not found';
    END IF;
    IF v_classic_id != 171 THEN
        RAISE EXCEPTION 'Pre-flight: Classic Granola 25 LB id changed — expected 171, got %', v_classic_id;
    END IF;
    IF v_classic_code IS NOT NULL AND v_classic_code != '70089' THEN
        RAISE EXCEPTION 'Pre-flight: Classic Granola 25 LB odoo_code is already % (expected NULL or 70089)', v_classic_code;
    END IF;

    -- 70089 must not be taken by anyone else
    SELECT COUNT(*) INTO v_70089_taken
    FROM products WHERE odoo_code = '70089' AND id != 171;
    IF v_70089_taken > 0 THEN
        RAISE EXCEPTION 'Pre-flight: odoo_code 70089 is already in use by another product';
    END IF;

    -- 70082 Setton FV must exist (label_type already private_label is OK)
    SELECT id INTO v_70082_id
    FROM products WHERE odoo_code = '70082' AND name = 'Granola Setton French Vanilla 25 LB';
    IF v_70082_id IS NULL THEN
        RAISE EXCEPTION 'Pre-flight: 70082 Granola Setton French Vanilla 25 LB not found';
    END IF;

    -- Lot-code collision: incoming lots must not collide with existing lots on product 61
    IF EXISTS (
        SELECT 1 FROM lots
        WHERE product_id = 61
          AND lot_code IN ('26-01-30-FOUND-007', '26-01-30-FOUND-017')
    ) THEN
        RAISE EXCEPTION 'Pre-flight: incoming lot_codes already exist on product 61 — UNIQUE constraint would fail';
    END IF;

    SELECT COUNT(*) INTO v_product_count FROM products;
    RAISE NOTICE 'Pre-flight OK — canonical=% chips=% liquor=% classic=% (odoo_code=%) 70082=% products_total=%',
        v_canonical_id, v_chips_id, v_liquor_id, v_classic_id, COALESCE(v_classic_code, 'NULL'),
        v_70082_id, v_product_count;
END $$;

-- ─────────────────────────────────────────────────────────────
-- STEP 1: Backfill SKU 70089 on Classic Granola 25 LB
-- ─────────────────────────────────────────────────────────────
UPDATE products
SET odoo_code = '70089'
WHERE id = 171
  AND name = 'Classic Granola 25 LB'
  AND odoo_code IS NULL;

-- ─────────────────────────────────────────────────────────────
-- STEP 2: Repoint all FK / *product_id refs from 167,168 → 61
-- ─────────────────────────────────────────────────────────────
UPDATE lots
SET product_id = 61
WHERE product_id IN (167, 168);

UPDATE transaction_lines
SET product_id = 61
WHERE product_id IN (167, 168);

UPDATE inventory_adjustments
SET product_id = 61
WHERE product_id IN (167, 168);

UPDATE product_verification_history
SET action = 'merged',
    merged_into_product_id = 61
WHERE product_id IN (167, 168);

-- ─────────────────────────────────────────────────────────────
-- STEP 3: Delete the now-orphaned duplicate product rows
-- ─────────────────────────────────────────────────────────────
DELETE FROM products
WHERE id IN (167, 168)
  AND odoo_code IS NULL
  AND name IN ('BS Cocoa Chips', 'BS Cocoa Liquor');

-- ─────────────────────────────────────────────────────────────
-- STEP 4: Assert label_type='private_label' on 70082 (idempotent)
-- ─────────────────────────────────────────────────────────────
UPDATE products
SET label_type = 'private_label'
WHERE odoo_code = '70082'
  AND name = 'Granola Setton French Vanilla 25 LB'
  AND label_type IS DISTINCT FROM 'private_label';

-- ─────────────────────────────────────────────────────────────
-- POST-FLIGHT: assert every invariant before COMMIT
-- ─────────────────────────────────────────────────────────────
DO $$
DECLARE
    v_classic_code    TEXT;
    v_bs_cocoa_left   INT;
    v_167_168_left    INT;
    v_70082_label     TEXT;
    v_product_count   INT;
    v_dangling_refs   INT;
BEGIN
    SELECT odoo_code INTO v_classic_code FROM products WHERE id = 171;
    IF v_classic_code != '70089' THEN
        RAISE EXCEPTION 'Post-flight FAIL: Classic Granola 25 LB odoo_code = %', v_classic_code;
    END IF;

    SELECT COUNT(*) INTO v_167_168_left
    FROM products WHERE id IN (167, 168);
    IF v_167_168_left != 0 THEN
        RAISE EXCEPTION 'Post-flight FAIL: % products with id IN (167,168) still exist', v_167_168_left;
    END IF;

    SELECT COUNT(*) INTO v_bs_cocoa_left
    FROM products WHERE name ILIKE 'BS Cocoa%';
    IF v_bs_cocoa_left != 1 THEN
        RAISE EXCEPTION 'Post-flight FAIL: expected 1 BS Cocoa* product, found %', v_bs_cocoa_left;
    END IF;

    SELECT label_type INTO v_70082_label
    FROM products WHERE odoo_code = '70082';
    IF v_70082_label IS DISTINCT FROM 'private_label' THEN
        RAISE EXCEPTION 'Post-flight FAIL: 70082 label_type = %', v_70082_label;
    END IF;

    SELECT COUNT(*) INTO v_product_count FROM products;
    IF v_product_count != 201 THEN
        RAISE EXCEPTION 'Post-flight FAIL: expected 201 products, found %', v_product_count;
    END IF;

    -- Final sweep: no remaining references to 167 or 168 in the data tables
    -- (product_verification_history intentionally retains 167/168 in product_id —
    --  it's an audit log with no FK and merged_into_product_id now points at 61)
    SELECT
      (SELECT COUNT(*) FROM lots                 WHERE product_id IN (167,168))
    + (SELECT COUNT(*) FROM transaction_lines    WHERE product_id IN (167,168))
    + (SELECT COUNT(*) FROM inventory_adjustments WHERE product_id IN (167,168))
    INTO v_dangling_refs;
    IF v_dangling_refs != 0 THEN
        RAISE EXCEPTION 'Post-flight FAIL: % dangling refs to 167/168 in data tables', v_dangling_refs;
    END IF;

    RAISE NOTICE 'Post-flight OK — products=% classic=70089 70082=private_label BS_cocoa_count=1',
        v_product_count;
END $$;

COMMIT;

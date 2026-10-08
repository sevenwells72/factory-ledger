-- Supplier cleanup — Dutch rows only (Michael's decision, 2026-10-08).
-- Two real companies: 13 Dutch Gold Honey (DUTG) and 16 Dutch Valley Foods (DUTV).
--   12 Dutch Gold                      → 13
--   11 DUTC Valley (typo), 14 Dutch Valley, 15 Dutch Valley Food Dist. → 16
-- Merge = repoint every FK reference to the canonical id, then DEACTIVATE the
-- duplicate (never delete), and add exact-match search aliases old name → canonical. Historical lot codes (e.g. "DUTC…") and free-text
-- supplier names are left unchanged. Run as table owner, ON_ERROR_STOP, in ONE
-- transaction; psql prints the UPDATE counts per statement:
--   psql "$URL" -v ON_ERROR_STOP=1 -X -f scripts/supplier_cleanup_dutch_2026_10_08.sql
-- Safe on both staging (which already carries auto labels from an earlier 066
-- build — DUTB/DUTF are replaced by the explicit codes, labels on the
-- deactivated rows are cleared) and production (labels are NULL today).
-- If ANY lots/transactions row references a duplicate, the immutable/append-only
-- triggers abort the transaction — that is intended: stop and re-plan.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path = public;

-- 0. Guard: ids and names must match the catalog this plan was written against.
DO $$ BEGIN
    IF (SELECT count(*) FROM suppliers WHERE (id, name) IN (
            (11,'DUTC Valley'), (12,'Dutch Gold'), (13,'Dutch Gold Honey'),
            (14,'Dutch Valley'), (15,'Dutch Valley Food Dist.'), (16,'Dutch Valley Foods'))) <> 6 THEN
        RAISE EXCEPTION 'Dutch supplier ids/names differ from the 2026-10-08 plan; re-run the dry-run';
    END IF;
    IF EXISTS (SELECT 1 FROM suppliers WHERE short_code IN ('DUTG','DUTV') AND id NOT IN (13,16)) THEN
        RAISE EXCEPTION 'DUTG/DUTV already held by another supplier';
    END IF;
END $$;

-- 1. Repoint references: duplicate → canonical (every table with an FK to suppliers.id).
UPDATE lot_supplier_codes       SET supplier_id = 13 WHERE supplier_id = 12;
UPDATE lot_supplier_codes       SET supplier_id = 16 WHERE supplier_id IN (11,14,15);
UPDATE expected_receipts        SET supplier_id = 13 WHERE supplier_id = 12;
UPDATE expected_receipts        SET supplier_id = 16 WHERE supplier_id IN (11,14,15);
UPDATE supplier_product_aliases SET supplier_id = 13 WHERE supplier_id = 12;
UPDATE supplier_product_aliases SET supplier_id = 16 WHERE supplier_id IN (11,14,15);
UPDATE search_aliases           SET supplier_id = 13 WHERE supplier_id = 12;
UPDATE search_aliases           SET supplier_id = 16 WHERE supplier_id IN (11,14,15);
UPDATE transactions             SET supplier_id = 13 WHERE supplier_id = 12;        -- append-only trigger: aborts if any row
UPDATE transactions             SET supplier_id = 16 WHERE supplier_id IN (11,14,15);
UPDATE lots                     SET supplier_id = 13 WHERE supplier_id = 12;        -- immutable trigger: aborts if any row
UPDATE lots                     SET supplier_id = 16 WHERE supplier_id IN (11,14,15);

-- 2. Deactivate the duplicates (never delete); clear any auto label they carry.
UPDATE suppliers SET active = false, short_code = NULL WHERE id IN (11,12,14,15);

-- 3. Explicit labels for the two real companies (066 never overwrites these).
UPDATE suppliers SET short_code = 'DUTG' WHERE id = 13;
UPDATE suppliers SET short_code = 'DUTV' WHERE id = 16;

-- 3b. Exact-match search aliases (kind='supplier', used by /resolve as tier 'alias';
--     receive still resolves by supplier name only). Idempotent via the unique index.
INSERT INTO search_aliases (kind, alias, supplier_id) VALUES
    ('supplier', 'Dutch Gold',              13),
    ('supplier', 'Dutch Valley',            16),
    ('supplier', 'Dutch Valley Food Dist.', 16),
    ('supplier', 'DUTC Valley',             16)
ON CONFLICT DO NOTHING;

-- 4. Verify: zero remaining references to the deactivated ids, labels + aliases in place.
DO $$ DECLARE remaining integer; BEGIN
    SELECT (SELECT count(*) FROM lots                     WHERE supplier_id IN (11,12,14,15))
         + (SELECT count(*) FROM transactions             WHERE supplier_id IN (11,12,14,15))
         + (SELECT count(*) FROM lot_supplier_codes       WHERE supplier_id IN (11,12,14,15))
         + (SELECT count(*) FROM expected_receipts        WHERE supplier_id IN (11,12,14,15))
         + (SELECT count(*) FROM supplier_product_aliases WHERE supplier_id IN (11,12,14,15))
         + (SELECT count(*) FROM search_aliases           WHERE supplier_id IN (11,12,14,15))
      INTO remaining;
    IF remaining <> 0 THEN RAISE EXCEPTION '% references to deactivated Dutch suppliers remain', remaining; END IF;
    IF (SELECT count(*) FROM suppliers WHERE id IN (11,12,14,15) AND NOT active AND short_code IS NULL) <> 4
       OR (SELECT short_code FROM suppliers WHERE id = 13) <> 'DUTG'
       OR (SELECT short_code FROM suppliers WHERE id = 16) <> 'DUTV' THEN
        RAISE EXCEPTION 'Dutch supplier end state not as planned';
    END IF;
    IF (SELECT count(*) FROM search_aliases WHERE kind='supplier' AND active
            AND ((supplier_id=13 AND alias_norm='dutch gold')
              OR (supplier_id=16 AND alias_norm IN ('dutch valley','dutch valley food dist.','dutc valley')))) <> 4 THEN
        RAISE EXCEPTION 'Dutch supplier search aliases missing';
    END IF;
END $$;
SELECT id, name, active, short_code FROM suppliers WHERE id IN (11,12,13,14,15,16) ORDER BY id;
COMMIT;

-- Removes the extra pack-source links. The code falls back to the usual batch link
-- (products.parent_batch_product_id) when the table is absent. Pack history is untouched:
-- posted packs name their products and lots directly, never this table.
DROP TABLE IF EXISTS product_pack_sources;
DELETE FROM migration_markers WHERE name='099_product_pack_sources';

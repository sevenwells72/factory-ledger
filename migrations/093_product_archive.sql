-- Product archive attribution (owner request, Michael, 2026-11-09).
-- Explicit apply only, as app/table owner, BEFORE the code that writes archives:
-- BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public;
-- run this file with ON_ERROR_STOP; COMMIT.
-- Archive reuses the existing products.active flag (false = archived; NULL and
-- true stay active). This only adds WHO archived it and WHEN. Additive: two
-- nullable columns, no default, no row rewritten, no data changed. Products
-- archived before this migration (e.g. the 6x8 OZ cases, row 152) keep NULLs.
-- Until the marker exists FL refuses archive/unarchive writes
-- (ARCHIVE_UNAVAILABLE, nothing changed), so code without this fails safe.
ALTER TABLE products ADD COLUMN IF NOT EXISTS archived_at timestamptz;
ALTER TABLE products ADD COLUMN IF NOT EXISTS archived_by text;
INSERT INTO migration_markers(name) VALUES ('093_product_archive') ON CONFLICT DO NOTHING;

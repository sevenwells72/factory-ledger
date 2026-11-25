-- Extra pack sources: a finished product may also be packed from the items listed
-- here, beside its usual batch in products.parent_batch_product_id (which holds one).
-- Why: a regular product packed from an Extra-Kosher (SS) twin batch could not be
-- chosen on the Pack form at all (2026-11-25: Granola Cocoa Vibes 25 LB, product 139,
-- from Batch Cocoa Crunch Granola #13 — EXTRA-KOSHER (SS), product 293).
-- Schema only. Each link row is catalog data added separately with owner approval.
-- Idempotent. Apply with owner approval, as table owner, in one transaction.
CREATE TABLE IF NOT EXISTS product_pack_sources (
 id serial PRIMARY KEY,
 finished_product_id integer NOT NULL REFERENCES products(id),
 source_product_id integer NOT NULL REFERENCES products(id),
 reason text NOT NULL CHECK (btrim(reason) <> ''),
 created_by text NOT NULL CHECK (btrim(created_by) <> ''),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 UNIQUE (finished_product_id, source_product_id),
 CHECK (finished_product_id <> source_product_id)
);
CREATE INDEX IF NOT EXISTS product_pack_sources_source ON product_pack_sources(source_product_id);
ALTER TABLE product_pack_sources ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON product_pack_sources FROM PUBLIC;
INSERT INTO migration_markers(name) VALUES ('099_product_pack_sources') ON CONFLICT(name) DO NOTHING;

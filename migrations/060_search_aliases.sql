-- A4 READ foundation only. No seed rows, log writes, startup hook or write API.
-- Apply only through a separately approved migration. Safe to rerun.
CREATE TABLE IF NOT EXISTS search_aliases (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('token', 'product', 'customer', 'supplier')),
    alias text NOT NULL CHECK (btrim(alias) <> ''),
    alias_norm text GENERATED ALWAYS AS
        (lower(btrim(regexp_replace(alias, '\s+', ' ', 'g')))) STORED,
    expansion text,
    product_id integer REFERENCES products(id),
    customer_id integer REFERENCES customers(id),
    supplier_id integer REFERENCES suppliers(id),
    language text NOT NULL DEFAULT 'any' CHECK (language IN ('any', 'en', 'es')),
    active boolean NOT NULL DEFAULT true,
    created_by integer REFERENCES actors(id),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    deactivated_by integer REFERENCES actors(id),
    deactivated_at timestamptz,
    CONSTRAINT search_aliases_target_check CHECK (
        (kind = 'token' AND expansion IS NOT NULL
            AND btrim(regexp_replace(expansion, '\s+', ' ', 'g')) <> ''
            AND product_id IS NULL AND customer_id IS NULL AND supplier_id IS NULL)
        OR (kind = 'product' AND product_id IS NOT NULL AND expansion IS NULL
            AND customer_id IS NULL AND supplier_id IS NULL)
        OR (kind = 'customer' AND customer_id IS NOT NULL AND expansion IS NULL
            AND product_id IS NULL AND supplier_id IS NULL)
        OR (kind = 'supplier' AND supplier_id IS NOT NULL AND expansion IS NULL
            AND product_id IS NULL AND customer_id IS NULL)
    ),
    CONSTRAINT search_aliases_normalized_nonblank CHECK (alias_norm <> '')
);
CREATE UNIQUE INDEX IF NOT EXISTS search_aliases_target_uniq
    ON search_aliases (kind, alias_norm, COALESCE(product_id, 0),
                       COALESCE(customer_id, 0), COALESCE(supplier_id, 0));
CREATE INDEX IF NOT EXISTS search_aliases_active_norm_idx
    ON search_aliases (alias_norm) WHERE active;
ALTER TABLE search_aliases ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON search_aliases FROM PUBLIC;

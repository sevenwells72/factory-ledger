CREATE TABLE IF NOT EXISTS sales_order_flags (
    so_number  text PRIMARY KEY,
    ready      bool NOT NULL DEFAULT false,
    ready_at   timestamptz,
    ready_by   text DEFAULT 'floor',
    note       text NULL,
    updated_at timestamptz DEFAULT now()
);

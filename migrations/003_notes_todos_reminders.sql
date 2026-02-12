-- Notes / To-Dos / Reminders table
-- Run this against the Supabase database before deploying

CREATE TABLE IF NOT EXISTS notes (
    id              SERIAL PRIMARY KEY,
    category        TEXT NOT NULL CHECK (category IN ('note', 'todo', 'reminder')),
    title           TEXT NOT NULL,
    body            TEXT DEFAULT '',
    priority        TEXT NOT NULL DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high')),
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done', 'dismissed')),
    due_date        DATE,
    entity_type     TEXT,          -- e.g. 'product', 'lot', 'customer', 'supplier'
    entity_id       TEXT,          -- the identifier (product name, lot code, customer name, etc.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_notes_category ON notes (category);
CREATE INDEX idx_notes_status ON notes (status);
CREATE INDEX idx_notes_due_date ON notes (due_date) WHERE due_date IS NOT NULL;
CREATE INDEX idx_notes_entity ON notes (entity_type, entity_id) WHERE entity_type IS NOT NULL;

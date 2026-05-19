-- ═══════════════════════════════════════════════════════════════
-- Migration 004: Production Scheduling Tables
-- Factory Ledger — 7-Day Tactical Production Scheduler
-- ═══════════════════════════════════════════════════════════════

-- 1. production_lines — Reference table for line definitions
CREATE TABLE IF NOT EXISTS production_lines (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    line_code TEXT NOT NULL UNIQUE,
    active BOOLEAN DEFAULT true
);

INSERT INTO production_lines (name, line_code) VALUES
    ('Granola Baking', 'granola'),
    ('Coconut Sweetened', 'coconut'),
    ('Bulk Packing', 'bulk_pack'),
    ('Pouch Line', 'pouch')
ON CONFLICT (line_code) DO NOTHING;


-- 2. line_capacity_modes — Capacity configurations per line
CREATE TABLE IF NOT EXISTS line_capacity_modes (
    id SERIAL PRIMARY KEY,
    line_id INTEGER NOT NULL REFERENCES production_lines(id),
    mode_name TEXT NOT NULL,
    workers_required INTEGER NOT NULL,
    batches_per_day INTEGER,
    pallets_per_day INTEGER,
    bags_per_day INTEGER,
    pack_size_lb NUMERIC,
    is_default BOOLEAN DEFAULT false,
    notes TEXT
);

-- Granola Baking: 2 workers = 9 batches/day, 3 workers = 16 batches/day
INSERT INTO line_capacity_modes (line_id, mode_name, workers_required, batches_per_day, is_default, notes)
SELECT id, '2-worker', 2, 9, true, '2 workers → 9 batches/day'
FROM production_lines WHERE line_code = 'granola'
ON CONFLICT DO NOTHING;

INSERT INTO line_capacity_modes (line_id, mode_name, workers_required, batches_per_day, is_default, notes)
SELECT id, '3-worker', 3, 16, false, '3 workers → 16 batches/day'
FROM production_lines WHERE line_code = 'granola'
ON CONFLICT DO NOTHING;

-- Coconut Sweetened: 2 workers = 12 batches/day
INSERT INTO line_capacity_modes (line_id, mode_name, workers_required, batches_per_day, is_default, notes)
SELECT id, 'standard', 2, 12, true, '2 workers → 12 batches/day'
FROM production_lines WHERE line_code = 'coconut'
ON CONFLICT DO NOTHING;

-- Bulk Packing 25lb: 2 workers = 4 pallets/day
INSERT INTO line_capacity_modes (line_id, mode_name, workers_required, pallets_per_day, pack_size_lb, is_default, notes)
SELECT id, '25lb-cases', 2, 4, 25, true, '25 lb cases → 4 pallets/day (2 workers full day)'
FROM production_lines WHERE line_code = 'bulk_pack'
ON CONFLICT DO NOTHING;

-- Bulk Packing 10lb: 2 workers = 9 pallets/day
INSERT INTO line_capacity_modes (line_id, mode_name, workers_required, pallets_per_day, pack_size_lb, is_default, notes)
SELECT id, '10lb-cases', 2, 9, 10, false, '10 lb cases → 9 pallets/day (2 workers full day)'
FROM production_lines WHERE line_code = 'bulk_pack'
ON CONFLICT DO NOTHING;

-- Pouch Line: 3 workers = 7500 bags/day (midpoint of 7000-8000)
INSERT INTO line_capacity_modes (line_id, mode_name, workers_required, bags_per_day, is_default, notes)
SELECT id, 'standard', 3, 7500, true, 'Min 3 workers → 7,000–8,000 bags/day'
FROM production_lines WHERE line_code = 'pouch'
ON CONFLICT DO NOTHING;


-- 3. product_line_assignments — Which products run on which lines
CREATE TABLE IF NOT EXISTS product_line_assignments (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    line_id INTEGER NOT NULL REFERENCES production_lines(id),
    UNIQUE(product_id, line_id)
);


-- 4. production_schedule — Saved confirmed schedules
CREATE TABLE IF NOT EXISTS production_schedule (
    id SERIAL PRIMARY KEY,
    schedule_date DATE NOT NULL,
    line_id INTEGER NOT NULL REFERENCES production_lines(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    planned_batches INTEGER,
    planned_quantity_lb NUMERIC,
    planned_bags INTEGER,
    workers_assigned INTEGER NOT NULL,
    status TEXT DEFAULT 'planned' CHECK (status IN ('planned', 'confirmed', 'in_progress', 'completed', 'cancelled')),
    linked_order_numbers TEXT[],
    overproduction_lb NUMERIC DEFAULT 0,
    overproduction_reason TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    confirmed_at TIMESTAMPTZ,
    UNIQUE(schedule_date, line_id, product_id)
);


-- 5. scheduling_config — Global scheduling parameters
CREATE TABLE IF NOT EXISTS scheduling_config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT
);

INSERT INTO scheduling_config (key, value, description) VALUES
    ('total_workers', '10', 'Total available workers'),
    ('friday_capacity_modifier', '0.5', 'Friday capacity multiplier'),
    ('work_days', '["Monday","Tuesday","Wednesday","Thursday","Friday"]', 'Working days'),
    ('default_horizon_days', '7', 'Default planning horizon')
ON CONFLICT (key) DO NOTHING;

-- Adjustment rules, owner decision 2026-11-18 (exceptions_enforcement.py,
-- adjustment_alerts.py). Apply as the app/table owner, port 5432, in one
-- transaction (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s',
-- SET LOCAL statement_timeout='60s', SET LOCAL search_path=public, COMMIT).
-- STAGING FIRST; production only with Michael's approval for that step.
--
-- What it does:
--   1. correction_reasons gains three flags and five NEW reasons: where the stock
--      went when an adjustment REMOVES it. One pick, no second step. Two existing
--      reasons stay usable on a removal because they say the stock was never
--      there: "Wrong lot used" and "Data-entry error", each with a required
--      note (owner, 2026-11-18).
--   2. Two old removal-only reasons that say the same thing as a new one are
--      retired for NEW entries (active = false). Their rows, labels and every
--      historical transaction that carries them are untouched.
--   3. adjustment_alerts: the email outbox for corrections over 500 lb.
--   4. The marker. Until it exists the application keeps the OLD rules whole
--      (old reason list, > 500 lb photo-or-hold), so code without this
--      migration, or this migration without the code, both fail safe.
--
-- Safe to rerun: IF NOT EXISTS / ON CONFLICT DO NOTHING / idempotent UPDATE.
-- No ledger table (transactions, transaction_lines, lots) is read, written or
-- locked: correction_reasons is only REFERENCED by transactions.reason_code,
-- and adjustment_alerts.transaction_id is a plain number with no foreign key
-- (the 094 precedent), so posting never waits on this file.

-- ---------------------------------------------------------------------------
-- 1. Removal flags on the fixed reason list.
--    removal_allowed      — a ticket adjustment that removes stock must use a
--                           reason with this flag: the five destinations, plus
--                           the two "it was never really there" reasons.
--    removal_destination  — this reason says where removed stock went.
--    sales_order_required — the entry must also name the sales order.
-- ---------------------------------------------------------------------------
ALTER TABLE correction_reasons
    ADD COLUMN IF NOT EXISTS removal_allowed boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS removal_destination boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS sales_order_required boolean NOT NULL DEFAULT false;

ALTER TABLE correction_reasons DROP CONSTRAINT IF EXISTS correction_reasons_destination_removes;
ALTER TABLE correction_reasons ADD CONSTRAINT correction_reasons_destination_removes
    CHECK (NOT removal_destination OR (adjust_sign = 'negative' AND 'adjust' = ANY (applies_to)));
ALTER TABLE correction_reasons DROP CONSTRAINT IF EXISTS correction_reasons_removal_can_remove;
ALTER TABLE correction_reasons ADD CONSTRAINT correction_reasons_removal_can_remove
    CHECK (NOT removal_allowed OR (adjust_sign <> 'positive' AND 'adjust' = ANY (applies_to)));
ALTER TABLE correction_reasons DROP CONSTRAINT IF EXISTS correction_reasons_destination_is_allowed;
ALTER TABLE correction_reasons ADD CONSTRAINT correction_reasons_destination_is_allowed
    CHECK (NOT removal_destination OR removal_allowed);
ALTER TABLE correction_reasons DROP CONSTRAINT IF EXISTS correction_reasons_order_needs_destination;
ALTER TABLE correction_reasons ADD CONSTRAINT correction_reasons_order_needs_destination
    CHECK (NOT sales_order_required OR removal_destination);

INSERT INTO correction_reasons
    (code, label_en, label_es, applies_to, adjust_sign, note_required, sort_order,
     removal_allowed, removal_destination, sales_order_required)
VALUES
    ('shipped_unrecorded', 'Shipped but not recorded', 'Enviado pero no registrado',
        ARRAY['adjust'], 'negative', false, 110, true, true, true),
    ('sent_to_sunshine_bulk', 'Sent to Sunshine as bulk', 'Enviado a Sunshine a granel',
        ARRAY['adjust'], 'negative', false, 120, true, true, false),
    ('used_in_production', 'Used in production but not recorded', 'Usado en producción pero no registrado',
        ARRAY['adjust'], 'negative', false, 130, true, true, false),
    ('damaged_thrown_out', 'Damaged or thrown out', 'Dañado o desechado',
        ARRAY['adjust'], 'negative', false, 140, true, true, false),
    ('count_unknown', 'Count correction — cause unknown', 'Corrección de conteo — causa desconocida',
        ARRAY['adjust'], 'negative', true, 150, true, true, false)
ON CONFLICT (code) DO NOTHING;

-- "Wrong lot used" and "Data-entry error" stay usable when stock is removed
-- (owner, 2026-11-18): they do not say where it went, they say it was never on
-- this lot. Only the new flag is set; their labels, sign and uses are unchanged.
-- On a removal the application requires a note with them (a removal_allowed
-- reason that is not a removal_destination needs a note). note_required itself
-- is NOT set: it would also force a note on voids, lot renames, supplier-lot
-- fixes and additions, which use the same two rows.
UPDATE correction_reasons SET removal_allowed = true
 WHERE code IN ('wrong_lot', 'data_entry_error') AND NOT removal_allowed;

-- ---------------------------------------------------------------------------
-- 2. Retire, for new entries only, the two removal-only reasons a new one
--    replaces word for word:
--      damage_disposal  ("Damage/disposal")  -> damaged_thrown_out
--      unrecorded_usage ("Unrecorded usage") -> used_in_production
--    The other six stay active: they are still used when stock is ADDED, for
--    found inventory, voids, lot renames, supplier-lot fixes and shortage
--    counts. Of those six only wrong_lot and data_entry_error are accepted when
--    a ticket adjustment removes stock (removal_allowed above).
--    The legacy translation table is not changed.
-- ---------------------------------------------------------------------------
UPDATE correction_reasons SET active = false
 WHERE code IN ('damage_disposal', 'unrecorded_usage') AND active;

-- ---------------------------------------------------------------------------
-- 3. adjustment_alerts — one row per correction over 500 lb, written in the
--    posting transaction inside a savepoint (a failure here never undoes the
--    post). The email is built from the ledger when it is sent; the row only
--    records that an email is owed and what happened to it. Not append-only:
--    the sender updates attempts / status.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS adjustment_alerts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id integer NOT NULL UNIQUE,          -- plain number, no FK (see header)
    action text NOT NULL CHECK (action IN ('adjust', 'found')),
    lb numeric(14,4) NOT NULL,                       -- signed pounds-equivalent that crossed the line
    queued_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_attempt_at timestamptz,
    last_error text,
    sent_at timestamptz,
    email_to text,
    provider_id text,
    CONSTRAINT adjustment_alerts_sent_needs_time CHECK ((status = 'sent') = (sent_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS adjustment_alerts_pending_idx
    ON adjustment_alerts (id) WHERE status = 'pending';

-- Same posture as 058/061/094: owner-only, RLS on, no public policies.
ALTER TABLE adjustment_alerts ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON adjustment_alerts FROM PUBLIC;

INSERT INTO migration_markers (name) VALUES ('096_adjust_destinations')
ON CONFLICT (name) DO NOTHING;

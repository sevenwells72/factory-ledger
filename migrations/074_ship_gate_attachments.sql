-- A6: additive; apply ONLY to staging as table owner, port 5432, with
-- ON_ERROR_STOP inside BEGIN; SET LOCAL lock_timeout='5s';
-- SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public; COMMIT.
CREATE TABLE IF NOT EXISTS attachments (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  attachment_ref text NOT NULL UNIQUE,
  kind text NOT NULL CHECK (kind IN ('bol','truck_photo','correction_photo','lot_tag','count_sheet','signed_packing_slip','other')),
  storage_path text NOT NULL UNIQUE,
  sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  content_type text NOT NULL CHECK (content_type IN ('image/jpeg','image/png','image/webp','application/pdf')),
  bytes integer NOT NULL CHECK (bytes > 0 AND bytes <= 15728640),
  uploaded_by_actor_id integer NOT NULL REFERENCES actors(id),
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  ticket_id bigint REFERENCES write_tickets(id),
  transaction_id integer REFERENCES transactions(id),
  shipment_id integer REFERENCES shipments(id),
  exception_id bigint REFERENCES exceptions(id),
  extracted jsonb NOT NULL DEFAULT '{}'::jsonb
);
ALTER TABLE attachments ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS attachments_ticket_idx ON attachments(ticket_id);
CREATE INDEX IF NOT EXISTS attachments_shipment_idx ON attachments(shipment_id);
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS bol_reference text;
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS proof_status text CHECK (proof_status IN ('complete','photo_pending'));
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS proof_due_at timestamptz;
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS ticket_id bigint REFERENCES write_tickets(id);
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS entered_by_actor_id integer REFERENCES actors(id);
ALTER TABLE shipments ADD COLUMN IF NOT EXISTS client_source text;
CREATE UNIQUE INDEX IF NOT EXISTS shipments_ticket_uniq ON shipments(ticket_id) WHERE ticket_id IS NOT NULL;
DO $$ DECLARE definition text; BEGIN
 SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
 WHERE conrelid='write_tickets'::regclass AND conname='write_tickets_action_check';
 IF position('ship_bulk' in definition)=0 THEN
  ALTER TABLE write_tickets DROP CONSTRAINT write_tickets_action_check;
  EXECUTE 'ALTER TABLE write_tickets ADD CONSTRAINT write_tickets_action_check CHECK (('
    || substring(definition from 8 for length(definition)-8)
    || ') OR action IN (''ship_order'',''ship_standalone'',''ship_bulk''))';
 END IF;
 SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
 WHERE conrelid='exceptions'::regclass AND conname='exceptions_kind_check';
 IF position('BOL_PHOTO_MISSING' in definition)=0 THEN
  ALTER TABLE exceptions DROP CONSTRAINT exceptions_kind_check;
  EXECUTE 'ALTER TABLE exceptions ADD CONSTRAINT exceptions_kind_check CHECK (('
    || substring(definition from 8 for length(definition)-8) || ') OR kind=''BOL_PHOTO_MISSING'')';
 END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_bol_photo_shipment_uniq ON exceptions(shipment_id)
 WHERE kind='BOL_PHOTO_MISSING';
-- Immutable provenance; linking a verified file later is the only mutable part.
CREATE OR REPLACE FUNCTION protect_attachment_provenance() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
 IF ROW(NEW.attachment_ref,NEW.kind,NEW.storage_path,NEW.sha256,NEW.content_type,NEW.bytes,NEW.uploaded_by_actor_id,NEW.created_at)
 IS DISTINCT FROM ROW(OLD.attachment_ref,OLD.kind,OLD.storage_path,OLD.sha256,OLD.content_type,OLD.bytes,OLD.uploaded_by_actor_id,OLD.created_at)
 OR (OLD.ticket_id IS NOT NULL AND NEW.ticket_id IS DISTINCT FROM OLD.ticket_id)
 OR (OLD.shipment_id IS NOT NULL AND NEW.shipment_id IS DISTINCT FROM OLD.shipment_id)
 OR (OLD.exception_id IS NOT NULL AND NEW.exception_id IS DISTINCT FROM OLD.exception_id) THEN
  RAISE EXCEPTION 'attachment provenance is immutable';
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS attachment_provenance ON attachments;
CREATE TRIGGER attachment_provenance BEFORE UPDATE ON attachments FOR EACH ROW EXECUTE FUNCTION protect_attachment_provenance();
INSERT INTO migration_markers(name) VALUES ('074_ship_gate_attachments') ON CONFLICT(name) DO NOTHING;

-- Empty staging/test rollback ONLY. Refuse to erase evidence or ship receipts.
DO $$ BEGIN
 IF EXISTS(SELECT 1 FROM attachments) OR EXISTS(SELECT 1 FROM shipments WHERE ticket_id IS NOT NULL)
 OR EXISTS(SELECT 1 FROM write_tickets WHERE action IN ('ship_order','ship_standalone','ship_bulk'))
 OR EXISTS(SELECT 1 FROM exceptions WHERE kind='BOL_PHOTO_MISSING') THEN
  RAISE EXCEPTION 'Export and preserve A6 evidence before rollback';
 END IF;
END $$;
DROP TABLE attachments;
DROP FUNCTION protect_attachment_provenance();
ALTER TABLE shipments DROP COLUMN bol_reference,DROP COLUMN proof_status,DROP COLUMN proof_due_at,
 DROP COLUMN ticket_id,DROP COLUMN entered_by_actor_id,DROP COLUMN client_source;
DROP INDEX exceptions_bol_photo_shipment_uniq;
-- Enum CHECK extensions are additive and intentionally retained while older
-- migration downs rebuild their own checks. No historical rows are rewritten.
DELETE FROM migration_markers WHERE name='074_ship_gate_attachments';

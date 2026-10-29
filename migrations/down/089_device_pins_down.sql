-- Set PIN_SIGNIN_ENABLED=0 and deploy first. Phone sessions end; personal-key
-- sessions, actors and ledger rows are untouched.
UPDATE key_actor_sessions SET ended_at=clock_timestamp(), ended_reason='revoked'
 WHERE device_id IS NOT NULL AND ended_at IS NULL;
ALTER TABLE key_actor_sessions DROP COLUMN IF EXISTS device_id;
DROP TABLE IF EXISTS key_device_audit;
DROP TABLE IF EXISTS key_device_attempts;
DROP TABLE IF EXISTS key_devices;
DELETE FROM migration_markers WHERE name='089_device_pins';

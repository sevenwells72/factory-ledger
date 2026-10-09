-- A3b follow-up (Codex review of PR #94, item 5; design rev 3.7 §5 R3 "pre_make_adjust",
-- §7.2 weekly-view item 3): a POSITIVE adjust by the same actor within 30 min before a
-- make/pack on an ingredient it consumes is tagged `pre_make_adjust` and listed for the
-- owner. The tag is stored as an exceptions row of a new kind, so the 061 kind CHECK
-- gains one value and one unique index keeps it to one tag per adjust posting.
-- Apply as the app/table owner after 061 and 069, port 5432, in one transaction
-- (ON_ERROR_STOP, BEGIN, SET LOCAL lock_timeout='5s', SET LOCAL statement_timeout='60s',
-- SET LOCAL search_path=public, COMMIT). STAGING FIRST; production only with owner
-- approval and BEFORE the A3b code goes live there (the code INSERTs this kind).
--
-- Additive only, safe to rerun:
--   1. exceptions.kind CHECK + 'PRE_MAKE_ADJUST' (rebuild pattern as 069 §1/§6: the
--      existing definition is extended, never narrowed).
--   2. UNIQUE INDEX exceptions_one_pre_make_tag_idx ON exceptions (transaction_id)
--      WHERE kind = 'PRE_MAKE_ADJUST' — an adjust followed by two makes is tagged once.
-- Lock profile: the CHECK rebuild takes ACCESS EXCLUSIVE on exceptions for the rest
-- of the transaction (milliseconds, bounded by lock_timeout); the index is small.

-- ---------------------------------------------------------------------------
-- 1. exceptions.kind: + 'PRE_MAKE_ADJUST'
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    definition text;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
      WHERE conrelid='exceptions'::regclass AND conname='exceptions_kind_check';
    IF definition IS NULL THEN
        RAISE EXCEPTION '070 requires migration 061 (exceptions_kind_check is missing)';
    END IF;
    IF position('PRE_MAKE_ADJUST' in definition)=0 THEN
        EXECUTE 'ALTER TABLE exceptions DROP CONSTRAINT exceptions_kind_check';
        EXECUTE 'ALTER TABLE exceptions ADD CONSTRAINT exceptions_kind_check CHECK (('
            || substring(definition from 8 for length(definition)-8)
            || ') OR kind = ''PRE_MAKE_ADJUST'')';
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 2. One tag per adjust posting
-- ---------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS exceptions_one_pre_make_tag_idx
    ON exceptions (transaction_id) WHERE kind = 'PRE_MAKE_ADJUST';

-- Standalone migration marker (no startup gate reads it; 069 precedent).
INSERT INTO migration_markers (name) VALUES ('070_pre_make_adjust')
ON CONFLICT (name) DO NOTHING;

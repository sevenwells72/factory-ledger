-- Lot identity (TRACEABILITY_DESIGN.md §3.1, CC-1) — identity ONLY: no trace
-- tables (those are migration 048), no emission changes.
--
--   * lot_uuid — the machine identity (future QR payload / API scan key).
--     Minted once at INSERT, survives PATCH /lots/{lot_id}/rename, never
--     ambiguous. The code stays human: UNIQUE (product_id, lot_code)
--     (lots_product_id_lot_code_key) remains the only exact-code constraint.
--   * Tier-1 HARD unique index (blocks the write): case + whitespace
--     normalization plus the trailing 'LOT' noise-token strip, per product —
--     the strongest normalization the historical data affirmatively supports
--     for a blocking constraint (validated against all 1,048 lots 2026-08-31).
--   * Tier-2 (aggressive: additionally strip ALL non-alphanumerics) stays OUT
--     of the database on purpose: it is a soft mint-time API warning
--     (suspicious_code_similarity, main.py) + the §8 T2 report, never a
--     constraint. Blocking a legitimate supplier receive at the dock is the
--     costlier failure mode; promoting tier 2 later is a cheap migration.
--   * Format CHECK is NOT VALID and is deliberately never VALIDATEd: it
--     disciplines newly minted codes only; historical codes stay untouched.
--
-- Pre-apply validation (docs/trace-preclean-worklist-2026-08.md): after the
-- two step-0 hygiene merges (1003→999, 401→410) the tier-1 index has ZERO
-- violations on production data — it builds clean with no further pre-clean.
-- Idempotent (IF NOT EXISTS / guarded DO blocks) so a re-run is a no-op.

BEGIN;

-- 1. Machine identity. Volatile default ⇒ the ADD COLUMN rewrites the table
--    and every existing row gets its own freshly minted UUID (~1k rows).
ALTER TABLE public.lots
    ADD COLUMN IF NOT EXISTS lot_uuid uuid NOT NULL DEFAULT gen_random_uuid();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'lots_lot_uuid_key'
          AND conrelid = 'public.lots'::regclass
    ) THEN
        ALTER TABLE public.lots
            ADD CONSTRAINT lots_lot_uuid_key UNIQUE (lot_uuid);
    END IF;
END;
$$;

COMMENT ON COLUMN public.lots.lot_uuid IS
    'Machine identity (QR payload / API scan key). Minted once at INSERT, '
    'survives rename, never reused. Human lookups keep using '
    '(product_id, lot_code).';

-- 2. Tier-1 HARD twin index.
-- Code twins within one product ("APR 10 2026 Lot" vs "APR 10 2026", "BB041327 Lot" vs
-- "BB041327"): upper-case, trim + collapse internal whitespace, drop a trailing 'LOT'
-- noise token, so casing/spacing/'Lot'-suffix variants of the same physical lot
-- collide at mint time.
-- Merged lots keep their lot_code (merge sets status='merged' without renaming), so the
-- index MUST exclude them — otherwise the first twin-merge leaves a colliding merged row
-- that makes the index impossible to build and future twin-merges impossible to complete.
CREATE UNIQUE INDEX IF NOT EXISTS lots_product_code_norm_uniq
    ON public.lots (product_id,
             regexp_replace(
                 regexp_replace(upper(btrim(lot_code)), '\s+', ' ', 'g'),  -- case + whitespace
                 '\s+LOT$', ''))                                           -- trailing 'LOT' token
    WHERE status IS DISTINCT FROM 'merged';

-- 3. Format discipline for newly minted codes, without invalidating history:
--    checks INSERT/UPDATE only; existing rows untouched, never VALIDATEd.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'lots_code_format_chk'
          AND conrelid = 'public.lots'::regclass
    ) THEN
        ALTER TABLE public.lots ADD CONSTRAINT lots_code_format_chk
            CHECK (lot_code ~ '^[A-Z0-9][A-Z0-9 ./#-]{2,49}$'
                   AND lot_code !~ '(ENE|ABR|AGO|DIC|SET) ')
            NOT VALID;
    END IF;
END;
$$;

COMMIT;

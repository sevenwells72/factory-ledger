-- Read-only preview of the 061 transactions.reason_code backfill. Run before
-- applying 061 (scripts/psql_ro.sh, inside BEGIN TRANSACTION READ ONLY) to
-- see how every type='adjust' row will be classified. The VALUES lists below
-- mirror the correction_reason_legacy_codes and correction_reasons seeds in
-- 061 — keep them in sync.
-- Output: one row per (source, raw value, mapped code) with a count; rows
-- mapped to 'unknown' keep their original text in adjust_reason/notes and
-- will show up as the first weekly view's backlog (§5.1).
WITH legacy(source, legacy_code, reason_code) AS (VALUES
    ('adjust', 'count_correction', 'physical_count'),
    ('adjust', 'damage', 'damage_disposal'),
    ('adjust', 'spoilage', 'damage_disposal'),
    ('adjust', 'sample', 'unrecorded_usage'),
    ('adjust', 'hydration_yield', 'unrecorded_usage'),
    ('adjust', 'other', 'unknown'),
    ('adjust', 'correction from physical count', 'physical_count'),
    ('adjust', 'product damaged', 'damage_disposal'),
    ('adjust', 'product spoiled or expired', 'damage_disposal'),
    ('adjust', 'used for samples', 'unrecorded_usage'),
    ('adjust', 'hydration/processing yield correction', 'unrecorded_usage'),
    ('adjust', 'other reason (specify in notes)', 'unknown'),
    ('found', 'found_during_count', 'physical_count'),
    ('found', 'found_back_stock', 'missing_receipt'),
    ('found', 'predates_system', 'missing_receipt'),
    ('found', 'unreceived_delivery', 'missing_receipt')),
new_codes(code) AS (VALUES ('physical_count'), ('missing_receipt'), ('missing_production'),
    ('wrong_lot'), ('damage_disposal'), ('unrecorded_usage'), ('data_entry_error'), ('unknown')),
src AS (
    SELECT id,
           CASE WHEN NULLIF(btrim(adjust_reason), '') IS NOT NULL THEN 'adjust'
                WHEN notes LIKE 'Found inventory%: %' THEN 'found' END AS source,
           CASE WHEN NULLIF(btrim(adjust_reason), '') IS NOT NULL THEN adjust_reason
                WHEN notes LIKE 'Found inventory%: %'
                    THEN substr(notes, position(': ' IN notes) + 2) END AS raw
      FROM transactions
     WHERE type = 'adjust')
SELECT COALESCE(src.source, '<no reason at all>') AS source,
       COALESCE(src.raw, '<null>') AS raw_value,
       COALESCE(m.reason_code, already.code, 'unknown') AS will_become,
       count(*) AS rows
  FROM src
  LEFT JOIN legacy m
         ON m.source = src.source
        AND m.legacy_code = lower(btrim(regexp_replace(src.raw, '\s+', ' ', 'g')))
  LEFT JOIN new_codes already
         ON already.code = replace(lower(btrim(regexp_replace(src.raw, '\s+', ' ', 'g'))), ' ', '_')
 GROUP BY 1, 2, 3
 ORDER BY 3, 4 DESC, 2;

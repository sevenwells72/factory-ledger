-- traceability_audit.sql — read-only data-entry audit, last 90 days.
-- Run against prod through the transaction pooler (port 6543):
--     psql "$DATABASE_URL" -f scripts/traceability_audit.sql
-- Every query is wrapped in BEGIN TRANSACTION READ ONLY ... COMMIT so nothing
-- here can write, and NO session-level GUCs are set (session read-only GUCs
-- poison shared 6543 pool connections — see CLAUDE.md hard rule / 2026-08-17
-- READONLY_TRIPWIRE incident).
--
-- Window: the 90 calendar days ending today (America/New_York business days).
-- Ledger event types read through ledger_current_transactions with
-- effective_status = 'posted' (voided/amended resolved). Entry-lag metrics
-- exclude created_at_source <> 'database' (migration_backfill_039 rows carry
-- the migration timestamp, not the real entry time).

\echo '=== [1] window sanity ==='
BEGIN TRANSACTION READ ONLY;
SELECT CURRENT_DATE AS today,
       (CURRENT_DATE - 89) AS window_start,
       now() AS db_now,
       current_setting('TimeZone') AS db_tz;
COMMIT;

-- ---------------------------------------------------------------------------
-- [2] Entries per day, by event type (90-day daily matrix).
-- Ledger types use business_date; side tables use created_at rendered as a
-- New York calendar date (lots.created_at is a naive UTC timestamp).
-- ---------------------------------------------------------------------------
\echo '=== [2] entries per day (90-day daily matrix) ==='
BEGIN TRANSACTION READ ONLY;
WITH days AS (
    SELECT generate_series(CURRENT_DATE - 89, CURRENT_DATE, interval '1 day')::date AS d
),
events AS (
    SELECT type AS event_type, business_date AS event_day
      FROM ledger_current_transactions
     WHERE effective_status = 'posted'
       AND type IN ('receive','ship','make','pack','adjust')
       AND business_date >= CURRENT_DATE - 89
    UNION ALL
    SELECT 'expected_receipt', timezone('America/New_York', created_at)::date
      FROM expected_receipts
     WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89
    UNION ALL
    SELECT 'so_allocation', timezone('America/New_York', created_at)::date
      FROM sales_order_allocations
     WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89
    UNION ALL
    SELECT 'fg_lot', timezone('America/New_York', created_at AT TIME ZONE 'UTC')::date
      FROM lots
     WHERE entry_source IN ('production_output','pack_output')
       AND timezone('America/New_York', created_at AT TIME ZONE 'UTC')::date >= CURRENT_DATE - 89
)
SELECT d.d AS day,
       to_char(d.d, 'Dy') AS dow,
       count(*) FILTER (WHERE e.event_type = 'receive')          AS receive,
       count(*) FILTER (WHERE e.event_type = 'expected_receipt') AS exp_rcpt,
       count(*) FILTER (WHERE e.event_type = 'ship')             AS ship,
       count(*) FILTER (WHERE e.event_type = 'so_allocation')    AS so_alloc,
       count(*) FILTER (WHERE e.event_type = 'make')             AS make,
       count(*) FILTER (WHERE e.event_type = 'pack')             AS pack,
       count(*) FILTER (WHERE e.event_type = 'fg_lot')           AS fg_lot,
       count(*) FILTER (WHERE e.event_type = 'adjust')           AS adjust
  FROM days d
  LEFT JOIN events e ON e.event_day = d.d
 GROUP BY d.d
 ORDER BY d.d;
COMMIT;

-- ---------------------------------------------------------------------------
-- [3] Per-type totals, active days, zero days (of the 90), weekday zero days.
-- ---------------------------------------------------------------------------
\echo '=== [3] totals / active days / zero days ==='
BEGIN TRANSACTION READ ONLY;
WITH days AS (
    SELECT generate_series(CURRENT_DATE - 89, CURRENT_DATE, interval '1 day')::date AS d
),
events AS (
    SELECT type AS event_type, business_date AS event_day
      FROM ledger_current_transactions
     WHERE effective_status = 'posted'
       AND type IN ('receive','ship','make','pack','adjust')
       AND business_date >= CURRENT_DATE - 89
    UNION ALL
    SELECT 'expected_receipt', timezone('America/New_York', created_at)::date
      FROM expected_receipts
     WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89
    UNION ALL
    SELECT 'so_allocation', timezone('America/New_York', created_at)::date
      FROM sales_order_allocations
     WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89
    UNION ALL
    SELECT 'fg_lot', timezone('America/New_York', created_at AT TIME ZONE 'UTC')::date
      FROM lots
     WHERE entry_source IN ('production_output','pack_output')
       AND timezone('America/New_York', created_at AT TIME ZONE 'UTC')::date >= CURRENT_DATE - 89
),
types(event_type) AS (
    VALUES ('receive'),('expected_receipt'),('ship'),('so_allocation'),
           ('make'),('pack'),('fg_lot'),('adjust')
),
daily AS (
    SELECT t.event_type, d.d,
           count(e.event_day) AS n
      FROM types t
     CROSS JOIN days d
      LEFT JOIN events e ON e.event_type = t.event_type AND e.event_day = d.d
     GROUP BY t.event_type, d.d
)
SELECT event_type,
       sum(n)                                            AS total_entries,
       count(*) FILTER (WHERE n > 0)                     AS active_days,
       count(*) FILTER (WHERE n = 0)                     AS zero_days,
       count(*) FILTER (WHERE n = 0
                          AND extract(isodow FROM d) < 6) AS zero_weekdays,
       max(n)                                            AS max_per_day
  FROM daily
 GROUP BY event_type
 ORDER BY event_type;
COMMIT;

-- ---------------------------------------------------------------------------
-- [4] Null / blank rates — receipts (transactions type='receive').
-- transaction_lines.lot_id / quantity_lb are NOT NULL by schema; the real
-- completeness risks are supplier fields, BOL, case counts, ER linkage,
-- and lots left without a supplier lot code.
-- ---------------------------------------------------------------------------
\echo '=== [4] null rates: receipts ==='
BEGIN TRANSACTION READ ONLY;
-- NOTE: prod's ledger_current_transactions view predates migration 041 and
-- does not expose expected_receipt_id; read it from the base table.
SELECT count(*)                                                             AS receive_txns,
       count(*) FILTER (WHERE ct.shipper_name IS NULL OR btrim(ct.shipper_name) = '')   AS null_shipper,
       count(*) FILTER (WHERE ct.bol_reference IS NULL OR btrim(ct.bol_reference) = '') AS null_bol,
       count(*) FILTER (WHERE ct.cases_received IS NULL)                    AS null_cases,
       count(*) FILTER (WHERE ct.case_size_lb IS NULL)                      AS null_case_size,
       count(*) FILTER (WHERE t.expected_receipt_id IS NULL)                AS unlinked_to_er
  FROM ledger_current_transactions ct
  JOIN transactions t ON t.id = ct.id
 WHERE ct.type = 'receive' AND ct.effective_status = 'posted'
   AND ct.business_date >= CURRENT_DATE - 89;
COMMIT;

\echo '=== [4b] receipts: line + supplier-lot-code coverage ==='
BEGIN TRANSACTION READ ONLY;
WITH r AS (
    SELECT id FROM ledger_current_transactions
     WHERE type = 'receive' AND effective_status = 'posted'
       AND business_date >= CURRENT_DATE - 89
)
SELECT (SELECT count(*) FROM r)                                            AS receive_txns,
       count(*) FILTER (WHERE tl.transaction_id IS NULL)                   AS txns_without_lines,
       count(DISTINCT tl.lot_id)                                           AS lots_received,
       count(DISTINCT tl.lot_id) FILTER (
           WHERE l.supplier_lot_code IS NULL
             AND NOT EXISTS (SELECT 1 FROM lot_supplier_codes lsc
                              WHERE lsc.lot_id = tl.lot_id))               AS lots_no_supplier_lot_code
  FROM r
  LEFT JOIN ledger_current_transaction_lines tl ON tl.transaction_id = r.id
  LEFT JOIN lots l ON l.id = tl.lot_id;
COMMIT;

-- ---------------------------------------------------------------------------
-- [5] Null rates — expected receipts.
-- ---------------------------------------------------------------------------
\echo '=== [5] null rates: expected receipts ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                                    AS expected_receipts,
       count(*) FILTER (WHERE expected_date IS NULL)               AS null_expected_date,
       count(*) FILTER (WHERE reference_number IS NULL
                           OR btrim(reference_number) = '')        AS null_reference,
       count(*) FILTER (WHERE created_by IS NULL)                  AS null_created_by,
       count(*) FILTER (WHERE status = 'open')                     AS still_open,
       count(*) FILTER (WHERE status = 'open'
                          AND expected_date < CURRENT_DATE - 7)    AS open_overdue_7d
  FROM expected_receipts
 WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89;
COMMIT;

-- ---------------------------------------------------------------------------
-- [6] Null rates — shipments (transactions type='ship' + shipments tables).
-- ---------------------------------------------------------------------------
\echo '=== [6] null rates: shipments ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                                                   AS ship_txns,
       count(*) FILTER (WHERE customer_name IS NULL OR btrim(customer_name) = '') AS null_customer,
       count(*) FILTER (WHERE order_reference IS NULL
                           OR btrim(order_reference) = '')                        AS null_order_ref,
       count(*) FILTER (WHERE bol_reference IS NULL)                              AS null_bol,
       count(*) FILTER (WHERE shipper_name IS NULL)                               AS null_shipper_name
  FROM ledger_current_transactions
 WHERE type = 'ship' AND effective_status = 'posted'
   AND business_date >= CURRENT_DATE - 89;
COMMIT;

\echo '=== [6b] shipments: lot lines + SO/shipment-record linkage ==='
BEGIN TRANSACTION READ ONLY;
WITH s AS (
    SELECT id FROM ledger_current_transactions
     WHERE type = 'ship' AND effective_status = 'posted'
       AND business_date >= CURRENT_DATE - 89
)
SELECT (SELECT count(*) FROM s)                                        AS ship_txns,
       count(*) FILTER (WHERE NOT EXISTS
           (SELECT 1 FROM transaction_lines tl WHERE tl.transaction_id = s.id)) AS txns_without_lot_lines,
       count(*) FILTER (WHERE NOT EXISTS
           (SELECT 1 FROM shipment_lines sl WHERE sl.transaction_id = s.id))    AS txns_without_shipment_lines,
       count(*) FILTER (WHERE NOT EXISTS
           (SELECT 1 FROM sales_order_shipments sos WHERE sos.transaction_id = s.id)) AS txns_not_linked_to_so
  FROM s;
COMMIT;

\echo '=== [6c] shipment_lines: sales_order_line linkage ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                                  AS shipment_lines,
       count(*) FILTER (WHERE sales_order_line_id IS NULL)       AS null_so_line
  FROM shipment_lines
 WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89;
COMMIT;

-- ---------------------------------------------------------------------------
-- [7] Null rates — sales-order allocations.
-- ---------------------------------------------------------------------------
\echo '=== [7] null rates: SO allocations ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                              AS allocations,
       count(*) FILTER (WHERE lot_id IS NULL)                AS null_lot,
       count(*) FILTER (WHERE created_by IS NULL)            AS null_created_by,
       count(*) FILTER (WHERE status = 'active')             AS active,
       count(*) FILTER (WHERE status = 'shipped')            AS shipped,
       count(*) FILTER (WHERE status = 'released')           AS released,
       count(*) FILTER (WHERE status = 'released'
                          AND (release_reason IS NULL
                               OR btrim(release_reason) = '')) AS released_no_reason,
       count(*) FILTER (WHERE status = 'superseded')         AS superseded
  FROM sales_order_allocations
 WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89;
COMMIT;

-- ---------------------------------------------------------------------------
-- [8] Null rates — production batches (make) and packaging (pack):
-- output lines and ingredient-lot consumption coverage.
-- ---------------------------------------------------------------------------
\echo '=== [8] make/pack: line + ingredient-lot-consumption coverage ==='
BEGIN TRANSACTION READ ONLY;
SELECT t.type,
       count(*)                                              AS txns,
       count(*) FILTER (WHERE NOT EXISTS
           (SELECT 1 FROM transaction_lines tl WHERE tl.transaction_id = t.id)) AS txns_without_lines,
       count(*) FILTER (WHERE NOT EXISTS
           (SELECT 1 FROM ingredient_lot_consumption ilc
             WHERE ilc.transaction_id = t.id))               AS txns_without_ilc
  FROM ledger_current_transactions t
 WHERE t.type IN ('make','pack') AND t.effective_status = 'posted'
   AND t.business_date >= CURRENT_DATE - 89
 GROUP BY t.type;
COMMIT;

\echo '=== [8b] ingredient_lot_consumption: null lot/product ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                              AS ilc_rows,
       count(*) FILTER (WHERE ingredient_lot_id IS NULL)     AS null_ingredient_lot,
       count(*) FILTER (WHERE ingredient_product_id IS NULL) AS null_ingredient_product,
       count(*) FILTER (WHERE transaction_id IS NULL)        AS null_transaction
  FROM ingredient_lot_consumption
 WHERE timezone('America/New_York', created_at AT TIME ZONE 'UTC')::date >= CURRENT_DATE - 89;
COMMIT;

-- ---------------------------------------------------------------------------
-- [9] Null rates — finished-goods lots (entry_source production/pack output).
-- ---------------------------------------------------------------------------
\echo '=== [9] null rates: finished-goods lots ==='
BEGIN TRANSACTION READ ONLY;
SELECT entry_source,
       count(*)                                             AS lots,
       count(*) FILTER (WHERE lot_code IS NULL
                           OR btrim(lot_code) = '')         AS null_lot_code,
       count(*) FILTER (WHERE status IS DISTINCT FROM 'active') AS non_active,
       count(*) FILTER (WHERE received_at IS NULL)          AS null_received_at
  FROM lots
 WHERE entry_source IN ('production_output','pack_output')
   AND timezone('America/New_York', created_at AT TIME ZONE 'UTC')::date >= CURRENT_DATE - 89
 GROUP BY entry_source;
COMMIT;

-- ---------------------------------------------------------------------------
-- [10] Null rates — inventory adjustments: adjust transactions + the
-- found-inventory audit table.
-- ---------------------------------------------------------------------------
\echo '=== [10] null rates: adjust transactions ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                                                 AS adjust_txns,
       count(*) FILTER (WHERE adjust_reason IS NULL OR btrim(adjust_reason) = '') AS null_reason,
       count(*) FILTER (WHERE notes IS NULL OR btrim(notes) = '')               AS null_notes
  FROM ledger_current_transactions
 WHERE type = 'adjust' AND effective_status = 'posted'
   AND business_date >= CURRENT_DATE - 89;
COMMIT;

\echo '=== [10b] inventory_adjustments (found-inventory audit rows) ==='
BEGIN TRANSACTION READ ONLY;
SELECT count(*)                                              AS n_rows,
       count(*) FILTER (WHERE lot_id IS NULL)                AS null_lot,
       count(*) FILTER (WHERE reason_code IS NULL)           AS null_reason_code,
       count(*) FILTER (WHERE adjusted_by IS NULL)           AS null_adjusted_by,
       count(*) FILTER (WHERE suspected_supplier IS NULL)    AS null_suspected_supplier
  FROM inventory_adjustments
 WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89;
COMMIT;

-- ---------------------------------------------------------------------------
-- [11] Entry lag — ledger types: NY calendar-day gap between business_date
-- (event day) and created_at (entry moment). Only created_at_source =
-- 'database' rows have a trustworthy created_at. Positive = entered late.
-- ---------------------------------------------------------------------------
\echo '=== [11] entry lag (days), ledger types ==='
BEGIN TRANSACTION READ ONLY;
WITH lag AS (
    SELECT type,
           (timezone('America/New_York', created_at)::date - business_date) AS lag_days
      FROM ledger_current_transactions
     WHERE effective_status = 'posted'
       AND type IN ('receive','ship','make','pack','adjust')
       AND business_date >= CURRENT_DATE - 89
       AND created_at_source = 'database'
)
SELECT type,
       count(*)                                             AS n,
       count(*) FILTER (WHERE lag_days > 0)                 AS entered_late,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY lag_days) AS p50,
       percentile_cont(0.9)  WITHIN GROUP (ORDER BY lag_days) AS p90,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY lag_days) AS p99,
       max(lag_days)                                        AS max_lag
  FROM lag
 GROUP BY type
 ORDER BY type;
COMMIT;

\echo '=== [11b] rows excluded from lag metrics (untrusted created_at) ==='
BEGIN TRANSACTION READ ONLY;
SELECT type, created_at_source, count(*)
  FROM ledger_current_transactions
 WHERE effective_status = 'posted'
   AND type IN ('receive','ship','make','pack','adjust')
   AND business_date >= CURRENT_DATE - 89
   AND created_at_source <> 'database'
 GROUP BY type, created_at_source;
COMMIT;

-- ---------------------------------------------------------------------------
-- [12] Entry lead/lag — expected receipts: days between logging the ER and
-- the expected delivery date (positive = logged in advance).
-- ---------------------------------------------------------------------------
\echo '=== [12] expected receipts: lead time (expected_date - entry day) ==='
BEGIN TRANSACTION READ ONLY;
WITH lead AS (
    SELECT (expected_date - timezone('America/New_York', created_at)::date) AS lead_days
      FROM expected_receipts
     WHERE expected_date IS NOT NULL
       AND timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89
)
SELECT count(*)                                              AS n,
       count(*) FILTER (WHERE lead_days < 0)                 AS logged_after_expected_date,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY lead_days) AS p50,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY lead_days) AS p90,
       min(lead_days)                                        AS min_lead,
       max(lead_days)                                        AS max_lead
  FROM lead;
COMMIT;

-- ---------------------------------------------------------------------------
-- [13] Corrections activity in-window (amend/void volume = rework signal).
-- ---------------------------------------------------------------------------
\echo '=== [13] ledger corrections in window ==='
BEGIN TRANSACTION READ ONLY;
SELECT event_type, count(*)
  FROM ledger_corrections
 WHERE timezone('America/New_York', created_at)::date >= CURRENT_DATE - 89
 GROUP BY event_type;
COMMIT;

-- Read-only production preflight for migration 039. Run before applying it.
BEGIN TRANSACTION READ ONLY;

SELECT current_database() AS database_name,
       current_user AS database_user,
       clock_timestamp() AS checked_at;

SELECT t.table_name,
       COUNT(*) AS row_count,
       COUNT(*) FILTER (WHERE cols.column_name = 'created_at') AS has_created_at_column
FROM information_schema.tables t
LEFT JOIN information_schema.columns cols
  ON cols.table_schema = t.table_schema
 AND cols.table_name = t.table_name
 AND cols.column_name = 'created_at'
WHERE t.table_schema = 'public'
  AND t.table_name IN (
      'transactions', 'transaction_lines', 'shipments', 'shipment_lines',
      'sales_order_shipments', 'ingredient_lot_consumption',
      'inventory_adjustments', 'lot_reassignments', 'lots',
      'lot_supplier_codes', 'sales_orders', 'sales_order_lines',
      'production_schedule'
  )
GROUP BY t.table_name
ORDER BY t.table_name;

SELECT COUNT(*) AS transactions_without_operational_timestamp
FROM transactions
WHERE "timestamp" IS NULL;

SELECT COALESCE(status, 'posted') AS status, COUNT(*)
FROM transactions
GROUP BY COALESCE(status, 'posted')
ORDER BY status;

SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('ledger_corrections', 'certifications')
ORDER BY table_name;

ROLLBACK;

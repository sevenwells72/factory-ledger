BEGIN;
SET TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '60s';
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
WITH p AS (
 SELECT * FROM products
), x AS (
 SELECT l.id line_id,l.transaction_id,l.product_id,l.lot_id,l.quantity_lb,
        l.created_at line_created_at,l.created_at_source line_created_at_source,
        l.latest_correction_created_at line_correction_at,
        t.type,t.occurred_at,t.business_date,t.created_at transaction_created_at,
        t.created_at_source transaction_created_at_source,t.effective_status,
        t.latest_correction_created_at transaction_correction_at,
        t.adjust_reason,t.notes,t.operator_id,
        raw.receipt_number,raw.ticket_id,raw.reason_code,raw.entered_by_actor_id,
        raw.kosher_tier transaction_kosher_tier,raw.kosher_attested_by_actor_id
 FROM ledger_current_transaction_lines l
 JOIN ledger_current_transactions t ON t.id=l.transaction_id
 JOIN transactions raw ON raw.id=t.id
 JOIN p ON p.id=l.product_id
), c AS (
 SELECT c.id,c.target_table,c.target_id,c.event_type,c.created_at,c.operator_id,c.reason
 FROM ledger_corrections c
 WHERE (c.target_table='transactions' AND c.target_id IN (SELECT transaction_id FROM x))
    OR (c.target_table='transaction_lines' AND c.target_id IN
        (SELECT id FROM ledger_current_transaction_lines
         WHERE transaction_id IN (SELECT transaction_id FROM x)))
)
SELECT json_build_object(
 'snapshot_at',current_timestamp, 'read_only',current_setting('transaction_read_only'),
 'catalog',COALESCE((SELECT json_agg(p ORDER BY id) FROM products p),'[]'::json),
 'transaction_line_counts',COALESCE((SELECT json_object_agg(transaction_id,n) FROM (SELECT transaction_id,count(*) n FROM ledger_current_transaction_lines GROUP BY transaction_id) counts),'{}'::json),
 'schema_columns',(SELECT json_agg(q) FROM (SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name,ordinal_position) q),
 'supply_requests',COALESCE((SELECT json_agg(q) FROM (SELECT id,product_id,item_text,qty,note,status,created_at FROM supply_requests) q),'[]'::json),
 'orders',COALESCE((SELECT json_agg(q) FROM sales_orders q),'[]'::json),
 'order_lines',COALESCE((SELECT json_agg(q) FROM sales_order_lines q),'[]'::json),
 'allocations',COALESCE((SELECT json_agg(q) FROM sales_order_allocations q),'[]'::json),
 'ship_transactions',COALESCE((SELECT json_agg(q) FROM ledger_current_transactions q WHERE type='ship'),'[]'::json),
 'products',COALESCE((SELECT json_agg(p ORDER BY id) FROM p),'[]'::json),
 'lots',COALESCE((SELECT json_agg(q ORDER BY product_id,id) FROM
   (SELECT l.id,l.product_id,l.lot_code,l.status,l.merged_into_lot_id,
           l.received_at, l.created_at AT TIME ZONE 'America/New_York' AS lot_created_at,
           l.created_at_source AS lot_created_at_source,l.entry_source,l.entry_source_notes,
           l.kosher_tier,l.kosher_attested_by_actor_id,l.kosher_attested_at
    FROM lots l JOIN p ON p.id=l.product_id)q),'[]'::json),
 'shortages',COALESCE((SELECT json_agg(q ORDER BY id) FROM
   (SELECT id,product_id,lot_id,status,resolution_kind,resolution_ticket_id,resolved_by_actor_id,
           detail->'resolution' AS resolution FROM exceptions WHERE kind='SHORTAGE')q),'[]'::json),
 'held_exceptions',COALESCE((SELECT json_agg(q ORDER BY id) FROM
   (SELECT id,kind,status,product_id,ticket_id,receipt_number,resolution_kind,resolved_by_actor_id
    FROM exceptions WHERE kind IN ('LARGE_CORRECTION','KOSHER_ATTESTATION'))q),'[]'::json),
 'correction_tickets',COALESCE((SELECT json_agg(q ORDER BY id) FROM
   (SELECT id,action,status,receipt_number,actor_id,operator_id,payload_hash
    FROM write_tickets WHERE action IN ('adjust','found'))q),'[]'::json),
 'actors',COALESCE((SELECT json_agg(q ORDER BY id) FROM (SELECT id,name,role,active FROM actors)q),'[]'::json),
 'lines',COALESCE((SELECT json_agg(x ORDER BY line_id) FROM x),'[]'::json),
 'corrections',COALESCE((SELECT json_agg(c ORDER BY created_at,id) FROM c),'[]'::json));
ROLLBACK;

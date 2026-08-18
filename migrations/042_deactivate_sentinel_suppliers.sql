-- FR-2 follow-up: the 041 supplier backfill copied every distinct shipper name
-- from the ledger, including pseudo-suppliers used for found/count/correction
-- receipts. Deactivate them so they never resolve for expected receipts,
-- never appear as candidates or in the dashboard dropdown, and never
-- auto-link a receipt. Rows are kept (not deleted) on purpose: a deleted
-- row would be recreated ACTIVE by a re-run of 041's backfill, whereas a
-- deactivated row survives re-runs (ON CONFLICT DO NOTHING).
--
-- Idempotent. Match on the normalised form so casing/spacing variants are covered.

BEGIN;

UPDATE public.suppliers
   SET active = false
 WHERE active
   AND public.supplier_name_norm(name) IN (
        'found',
        'found inventory',
        'inventory found',
        'physical count',
        'initial inventory',
        'inventory correction',
        'inventory intake',
        'unknown'
   );

COMMIT;

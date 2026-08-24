-- Void-time sales-order allocation reactivation quantities
-- (docs/designs/044-so-allocations-restore-addendum.md).
--
-- Restore-of-ship must consume exactly the reservation quantity that the
-- corresponding void returned to live allocations.  Ledger shipment pounds
-- are not that quantity: unallocated and partially allocated shipments are
-- valid.  One row is therefore recorded for every SO line on a voided ship,
-- including an explicit zero.
--
-- Idempotent and re-runnable.  No business_date and no session GUCs.

BEGIN;

CREATE TABLE IF NOT EXISTS public.sales_order_allocation_reactivations (
    transaction_id      integer NOT NULL REFERENCES public.transactions(id),
    sales_order_line_id integer NOT NULL REFERENCES public.sales_order_lines(id),
    quantity_lb         numeric(14,4) NOT NULL
                        CONSTRAINT soar_quantity_lb_check CHECK (quantity_lb >= 0),
    correction_id       uuid NOT NULL REFERENCES public.ledger_corrections(id),
    created_at          timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT sales_order_allocation_reactivations_pkey
        PRIMARY KEY (transaction_id, sales_order_line_id)
);

COMMENT ON TABLE public.sales_order_allocation_reactivations IS
    'Void-time SO allocation pounds reactivated per ship transaction and physical SO line; explicit zero means recorded with no reservation effect.';
COMMENT ON COLUMN public.sales_order_allocation_reactivations.quantity_lb IS
    'Reservation pounds returned to live sales_order_allocations by this void; restore consume target, including zero.';
COMMENT ON COLUMN public.sales_order_allocation_reactivations.correction_id IS
    'Append-only void correction that recorded or most recently overwrote this reactivation quantity.';

COMMIT;

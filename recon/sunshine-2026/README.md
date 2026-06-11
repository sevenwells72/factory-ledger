# Sunshine (SS line) Inventory Reconciliation — 2026

**Tag for the whole set: `SUNSHINE-RECON-2026`** (stored in `transactions.order_reference`)

## The situation

A physical count (June 2026) confirmed **zero on hand** for all Sunshine (SS line)
finished-goods SKUs, but the ledger showed **43,657.5 lb / 5,821 cases across 18 lots
and 4 SKUs** (SS Chocolate Chip, SS Original, SS Original Low Carb, SS Chocolate Chip
Low Carb — all 12x10 OZ Case). The product was actually shipped to the customer but the
shipments were never recorded: as of Phase 1, **no SS finished product had ever had a
ship transaction posted**. The other five SS finished products (Cranberry, Original
Bulk, B'gan Chocolate, Evergreen 12 pack, Mini 100) were already at zero and are not
part of this reconciliation.

## What Phase 1 did (2026-06-11)

Posted **18 interim standalone ship transactions** (one per lot, via `POST /ship`,
mode=commit) that shipped out the full live on-hand of every SS lot, bringing every
lot — and every SS product — to exactly 0.0000 lb (verified by posted-lines query
after posting).

- **Transactions: 1325–1342** (`type='ship'`, `status='posted'`)
- **Shipments: 240–257** (one `shipments` + one `shipment_lines` row per transaction)
- **Customer: `Sunshine Granola (RECON)`** — a placeholder customer auto-created for
  this purpose; not a real customer record
- Every transaction carries `order_reference = 'SUNSHINE-RECON-2026'` and the note:
  *"INTERIM — actual ship dates/orders pending backdated customer data. To be voided
  and replaced."*
- Full per-transaction detail: [`interim-ship-manifest.csv`](interim-ship-manifest.csv)
  (generated from a live query of the tagged set immediately after posting)

These transactions are **placeholders**. Their timestamps are the posting time
(2026-06-11), not the real ship dates. They exist so the ledger matches physical
reality while the customer's backdated shipment data is assembled.

## Phase 2 procedure (when backdated customer data arrives)

1. **Query the interim set by tag** (do not trust the ID range alone):

   ```sql
   SELECT t.id, s.id AS shipment_id
   FROM transactions t
   LEFT JOIN shipments s ON s.transaction_id = t.id
   WHERE t.order_reference = 'SUNSHINE-RECON-2026'
     AND t.type = 'ship'
     AND COALESCE(t.status, 'posted') = 'posted';
   ```

   Cross-check the result against `interim-ship-manifest.csv` — expect exactly 18.

2. **Void the set** — for each transaction id:
   `POST /void/{id}` with `{"reason": "Replaced by backdated actual shipments — SUNSHINE-RECON-2026 Phase 2"}`.
   Voiding flips `status='voided'`; the lines drop out of all balance math immediately
   (no reversal transactions are posted). This temporarily restores ~43,657.5 lb of
   phantom on-hand — proceed straight to step 4.

3. **Clean up shipment rows.** `POST /void` does NOT touch the `shipments` /
   `shipment_lines` rows created by `/ship`. Delete them for the voided transaction ids
   (`shipment_lines` first, then `shipments`, keyed on `transaction_id`), or they will
   linger in shipment-based views. Optionally deactivate the placeholder customer
   `Sunshine Granola (RECON)` once nothing references it.

4. **Post the real backdated ships** from the customer's data (real customer, real
   order references, real quantities per lot). Caveat: `POST /ship` always stamps
   `get_plant_now()` as the timestamp — backdating requires a follow-up
   `UPDATE transactions SET timestamp = <actual ship date>` on the new transaction ids
   (or direct SQL inserts mirroring the `/ship` row shape: negative `quantity_lb` in
   `transaction_lines`, plus `shipments`/`shipment_lines` rows).

5. **Verify net zero** — every SS lot back at 0.0000 posted on-hand, and the sum of the
   real ships per SKU equals what the interim set shipped (43,657.5 lb total):

   ```sql
   SELECT p.name, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) AS on_hand_lb
   FROM lots l
   JOIN products p ON p.id = l.product_id
   LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
     AND tl.transaction_id IN
         (SELECT id FROM transactions WHERE COALESCE(status,'posted') = 'posted')
   WHERE p.type = 'finished' AND p.name ILIKE '%SS %'
   GROUP BY p.name, l.lot_code
   HAVING ABS(COALESCE(SUM(tl.quantity_lb), 0)) > 0.0001;
   -- expect zero rows
   ```

   Show real query output for every step — receipt-anchored verification, no claimed
   success without it.

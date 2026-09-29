# Order create contract (row 158, stacked on PR #66)

Status: not deployed. Shared API keys and all named actors authorized by #66
can use `POST /sales/orders`. The dashboard key's existing create denial is
unchanged. Header PATCH keeps its existing key permissions and status gate
(`new` / `confirmed`); named header edits now have an atomic audit record.

```json
{
  "customer_id": 42,
  "customer_po": "062732",
  "external_order_reference": "ORD28100",
  "lines": [
    {"product_id": 107, "quantity": 24, "unit": "cases", "unit_price": 30, "amount": 720},
    {"product_id": 176, "quantity": 2, "unit": "each", "unit_price": 12.5, "amount": 25}
  ]
}
```

IDs above are illustrative except product 176, the Pallet Charge SKU. Supply
the approved customer and product IDs from lookup; the API validates existence
and uses those IDs directly, even if a stale display name is also supplied.
Existing name-only requests remain supported.

* `customer_po` must be a string or null. Non-empty text is saved exactly,
  including leading zeros and surrounding whitespace. Null, omitted, empty,
  or whitespace-only means No PO, visible in the order detail. Duplicate
  checks use the intake flow's existing case/whitespace normalization, include
  all order statuses, and serialize on the same transaction-scoped PO lock.
  A duplicate returns HTTP 409 `DUPLICATE_CUSTOMER_PO` with existing orders.
  Only an explicit boolean `allow_duplicate_po: true` overrides it.
* `external_order_reference` is optional, exact, case-sensitive text scoped
  to a customer. Blank is treated as absent. A database unique index prevents
  duplicate orders; a receipt primary key additionally reserves the original
  customer/reference after a customer edit. References on existing orders are
  immutable through the API. Reuse with changed content returns HTTP 409
  `EXTERNAL_ORDER_REFERENCE_CONFLICT`.
* An identical retry returns HTTP 200 with the original order ID, SO number,
  and saved response, even after the catalog or order header/lines change.
  Defaults and JSON key order are canonicalized; display names accompanying
  IDs and the duplicate-PO permission flag do not affect request identity.
  Line order and business content do. Retry comparison occurs before PO
  validation and product resolution. No second order or audit is written.
* Physical units support `lb`, `cases`, `bags`, and `boxes`. Missing case
  weights come from `products.case_size_lb`; an explicit weight remains
  supported. Prices follow the existing convention: per case when the product
  has a case weight, per pound otherwise. For case/bag/box inputs, the price
  is per entered unit. Conflicting explicit pounds are rejected.
* Service classification comes from `products.is_service`, including product
  176; it never depends on the product's name. Store service count in
  `ordered_quantity`, unit `each`, and zero in `quantity_lb`. Services contribute
  to money totals, never pounds, physical case totals, FIFO, or allocations.
  The existing ship-all path auto-fulfills them without inventory movement.
* Unit prices retain four-decimal database precision, including zero. Optional
  `amount` must equal priced quantity times price, rounded half-up to cents;
  otherwise the whole create is rejected. If omitted, it is calculated. If
  price is absent, amount is null and create `total` is null (unpriced order).
  Existing physical quantity/price edits keep stored amounts consistent.
  Service counts cannot be changed through the pounds-edit field.

The new create response includes `order_number`, `customer_po`,
`customer_po_status`, `external_order_reference`, `total_lb`, `total`, and
per-line `product_id`, `quantity`, `unit`, `quantity_lb`, `unit_price`, `amount`,
and `is_service`. Order detail also exposes saved commercial fields and shows
zero prices/totals. All header, line, audit and receipt writes commit in one
transaction. Audit or receipt failure rolls back the entire order.

`PATCH /sales/orders/{id}` accepts `customer_po` plus `allow_duplicate_po`.
Omitting PO preserves it; null or blank clears it. Moving a customer checks
the target PO and external reference for conflicts. The original create
receipt is never rewritten. A retry always reports the original creation;
GET returns the current order.

## Legacy compatibility

Requests that supply none of the new fields retain the original create
calculation, response shape, shared-key authorization and audit behavior.
There is no new mandatory PO, reference, or ID. No receipt is written without
a reference. New create fields, `each`, or product-derived case weights opt
into the durable commercial-line contract. Existing stored lines are not
backfilled. GET gains additive verification fields and faithfully displays
zero prices; header editing permissions and status restrictions are unchanged.
The GPT OpenAPI schema and its 30-operation limit are unchanged.

## Migration and rollback

Apply migration **056, then 057**, through **port 5432** before rolling out
this code, only with separate rollout authorization. This PR applies nothing
to a real database. 057 adds one nullable header field, a partial unique index
on customer/reference, four nullable line fields, and the receipt table. It
is additive and rerunnable, with no historical updates or ledger view changes.
`tests/schema/schema.sql` includes it for disposable databases.

To roll back, stop contract writes and revert the application first. Leaving
057 in place preserves metadata and is compatible with the old application.
If schema removal is required, export new metadata and run
`migrations/down/057_order_create_contract_down.sql` through port 5432; it
removes only 057 objects and necessarily discards their contents. It uses no
CASCADE and never drops `ledger_current_transactions` or
`ledger_current_transaction_lines`. Any future view change must use
`CREATE OR REPLACE VIEW`.

## Validation

Run the full Python suite with `TEST_DATABASE_URL` pointing to a disposable
local database loaded from `tests/schema/schema.sql`, using
`scripts/run_tests.sh`; run dashboard checks with `node --test tests/*.js`.
`tests/test_order_create_contract.py` exercises real PostgreSQL writes,
named/shared keys, original-response retries, concurrent HTTP requests,
database uniqueness, PO edits, product 176, service fulfillment, zero prices,
late-line/audit/receipt rollback, and migration up/down/rerun behavior.

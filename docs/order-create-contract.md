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

IDs above are illustrative except product 176, the Pallet Charge SKU. Product
102 (Pallets) is also a service product; both use the catalog's `is_service` flag. Supply
the approved customer and product IDs from lookup; the API validates existence
and uses those IDs directly, even if a stale display name is also supplied.
Existing name-only requests remain supported.

* `customer_po` must be a string or null. Only surrounding whitespace is
  stripped; leading zeros and internal whitespace are preserved. Null, omitted, empty,
  or whitespace-only means No PO, visible in the order detail. Duplicate
  checks use the intake flow's existing case/whitespace normalization, include
  all order statuses, and serialize on the same transaction-scoped PO lock.
  A duplicate returns HTTP 409 `DUPLICATE_CUSTOMER_PO` with existing orders.
  Only an explicit boolean `allow_duplicate_po: true` overrides it.
* `external_order_reference` is optional, case-sensitive text scoped
  to a customer, with surrounding whitespace stripped. Blank is treated as
  absent. Database CHECKs require trimmed references in orders and receipts.
  A database unique index prevents
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
  Referenced name-based creates also lock normalized customer name/reference
  before customer lookup/creation, so concurrent first requests share one customer.
* Physical units support `lb`, `cases`, `bags`, and `boxes`. Missing case
  weights come from `products.case_size_lb`; an explicit weight remains
  supported. Prices follow the existing convention: per case when the product
  has a case weight, per pound otherwise. For case/bag/box inputs, the price
  is per entered unit. Conflicting explicit pounds are rejected. A new-style
  priced line in pounds for a cased product must be a whole number of cases;
  otherwise a clear 422 asks for cases or whole-case pounds.
* Service classification comes from `products.is_service`, including products
  102 and 176; it never depends on a specific ID or the product's name. Store service count in
  `ordered_quantity`, unit `each`, and zero in `quantity_lb`. Services contribute
  to money totals, never pounds, physical case totals, FIFO, or allocations.
  Ship-all or explicit line shipping fulfills them without inventory movement,
  including a follow-up after the physical lines shipped. ZERO_SHIPMENT still
  rolls back while any physical line is unfulfilled. An order becomes shipped
  only when all non-cancelled lines are fulfilled. Packing slips show saved
  service counts, falling back to legacy quantity_lb for old lines.
* Unit prices retain four-decimal database precision, including zero. Optional
  `amount` must equal priced quantity times price, rounded half-up to cents;
  otherwise the whole create is rejected. If omitted, it is calculated. If
  price is absent, amount is null and create `total` is null (unpriced order).
  Existing physical quantity/price edits keep stored amounts consistent.
  Service counts cannot be changed through the pounds-edit field and display
  read-only in the dashboard; price edits remain available. Named line edits
  record actor audit in the same transaction as quantity and amount changes.

The new create response includes `order_number`, `customer_po`,
`customer_po_status`, `external_order_reference`, `total_lb`, `total`, and
per-line `product_id`, `quantity`, `unit`, `quantity_lb`, `unit_price`, `amount`,
and `is_service`. Order detail exposes saved commercial fields and shows
zero prices/totals for new-style lines. Legacy NULL and zero prices retain null
price/line-value reads and null totals on orders with no priced lines. NULL
prices remain NULL after quantity-only edits, too.
All header, line, audit and receipt writes commit in one
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
a reference. Only new fields (`customer_id`, `customer_po`,
`external_order_reference`, `allow_duplicate_po`, line `product_id` or `amount`)
opt into the durable commercial-line contract. Legacy case-unit requests with
no case weight still return their original 422; an omitted unit still warns
“Did you mean cases?” on both physical-line paths. Existing stored lines are
not backfilled. Lines with null `ordered_quantity` retain null reads for zero
`case_price`, `line_value`, `totals.total_value` and line-edit `unit_price`.
New-style lines preserve explicit zero. Header permissions/status gates remain unchanged.
The GPT OpenAPI schema and its 30-operation limit are unchanged.

## Migration and rollback

Apply migration **056, then 057**, through **port 5432** before rolling out
this code, only with separate rollout authorization. Run as the app's DB role
(table owner), using psql with `ON_ERROR_STOP` and
`BEGIN; SET LOCAL lock_timeout='5s';`, then `\i migrations/057_order_create_contract.sql`
and `COMMIT;`. The down script documents the equivalent rollback procedure.
This PR applies nothing
to a real database. 057 adds one nullable header field, a partial unique index
on customer/reference, four nullable line fields, and the receipt table. It
is additive and rerunnable, with no historical updates or ledger view changes.
Receipts have RLS enabled without FORCE or public policies, matching 056's
table-owner API access and denying public Supabase client access.
`tests/schema/schema.sql` includes it for disposable databases.
057 inserts `057_order_create_contract` into `migration_markers` with
`ON CONFLICT (name) DO NOTHING`, matching 056; rerunning retains the original marker.

To roll back, stop contract writes and revert the application first. Leaving
057 in place preserves metadata and is compatible with the old application.
If schema removal is required, export new metadata and run
`migrations/down/057_order_create_contract_down.sql` through port 5432; it
removes only 057 objects and its marker, and necessarily discards their contents. It uses no
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
The review tests additionally cover rendered packing-slip PDFs, line-by-line
service fulfillment, legacy GPT schema shapes/zero reads, audit rollback on
line edits, trimmed-reference retries, new-customer and header/create PO races,
whole-case pound validation, and dashboard read-only counts with editable prices.

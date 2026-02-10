# Factory Ledger — Sales API Reference (v2.3.1)

Complete documentation for all customer and sales order endpoints.

**Base URL:** `https://fastapi-production-b73a.up.railway.app`
**Auth:** All endpoints require `X-API-Key` header.

---

## 1. Customers

### Create Customer

```
POST /customers
```

Creates a new customer. Name must be unique.

**Request:**
```json
{
  "name": "Tropical Foods",
  "contact_name": "Maria Lopez",
  "email": "maria@tropicalfoods.com",
  "phone": "305-555-1234",
  "address": "123 Ocean Dr, Miami FL",
  "notes": "Preferred ship day: Tuesday"
}
```

Only `name` is required. All other fields are optional.

**Response:**
```json
{
  "customer_id": 12,
  "name": "Tropical Foods",
  "message": "Customer 'Tropical Foods' created"
}
```

**Errors:**
- `409` — Customer name already exists

**curl:**
```bash
curl -X POST "$BASE_URL/customers" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "Tropical Foods", "contact_name": "Maria Lopez", "phone": "305-555-1234"}'
```

---

### List Customers

```
GET /customers?active_only=true
```

Returns all customers, ordered by name.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `active_only` | bool | `true` | Only return active customers |

**Response:**
```json
{
  "customers": [
    {
      "id": 1,
      "name": "QUALI-PACK USA",
      "contact_name": "John Smith",
      "email": "john@qualipack.com",
      "phone": "305-555-0001",
      "active": true
    }
  ]
}
```

---

### Search Customers

```
GET /customers/search?q=quali
```

Fuzzy name search. Only searches active customers.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search term (min 1 character) |

**Response:**
```json
{
  "results": [
    {
      "id": 1,
      "name": "QUALI-PACK USA",
      "contact_name": "John Smith",
      "phone": "305-555-0001",
      "email": "john@qualipack.com"
    }
  ]
}
```

---

### Update Customer

```
PATCH /customers/{customer_id}
```

Partial update — only send the fields you want to change.

**Request:**
```json
{
  "phone": "305-555-9999",
  "notes": "Updated contact number"
}
```

**Response:**
```json
{
  "success": true,
  "customer_id": 12,
  "message": "Customer updated"
}
```

**Errors:**
- `400` — No fields provided
- `404` — Customer not found

---

## 2. Sales Orders — Lifecycle

### State Machine

Orders follow a strict status lifecycle. No skipping, no going backward.

```
                                    ┌─────────────┐
                                    │  cancelled   │ (terminal)
                                    └──────▲───────┘
                                           │ (from any non-terminal)
                                           │
┌─────┐     ┌───────────┐     ┌────────────────┐     ┌───────┐     ┌─────────┐     ┌──────────┐
│ new │ ──► │ confirmed │ ──► │ in_production  │ ──► │ ready │ ──► │ shipped │ ──► │ invoiced │
└─────┘     └───────────┘     └────────────────┘     └───┬───┘     └────▲────┘     └──────────┘
                                                         │              │           (terminal)
                                                         ▼              │
                                                   ┌──────────────┐    │
                                                   │ partial_ship │ ───┘
                                                   └──────────────┘
```

### Transition Rules

| From | Allowed Manual Transitions | Allowed Auto Transitions |
|------|---------------------------|-------------------------|
| `new` | confirmed, cancelled | — |
| `confirmed` | in_production, cancelled | — |
| `in_production` | ready, cancelled | — |
| `ready` | cancelled | shipped, partial_ship (via ship commit) |
| `partial_ship` | cancelled | shipped (via ship commit) |
| `shipped` | invoiced | — |
| `invoiced` | — (terminal) | — |
| `cancelled` | — (terminal) | — |

### Manual vs Automatic

- **Manual** (via `PATCH /sales/orders/{id}/status`): All transitions except `shipped` and `partial_ship`
- **Automatic** (via `POST /sales/orders/{id}/ship/commit`): Sets `shipped` when all lines fulfilled, `partial_ship` when some lines remain
- **Blocked**: You cannot manually set an order to `shipped` or `partial_ship` — the API rejects it with: *"shipped status is set automatically when an order is shipped. Use the ship endpoint instead."*
- **Terminal**: `invoiced` and `cancelled` accept no further status changes

---

## 3. Sales Orders — Endpoints

### Create Sales Order

```
POST /sales/orders
```

Creates a new sales order. Supports ordering in cases, bags, boxes, or pounds.

**Request — ordering in cases (auto-converts to lb):**
```json
{
  "customer_name": "Quali-Pack",
  "requested_ship_date": "2026-02-15",
  "lines": [
    {
      "product_name": "Classic Granola 25 LB",
      "quantity": 360,
      "unit": "cases",
      "case_weight_lb": 25,
      "unit_price": 4.50
    }
  ],
  "notes": "Rush order"
}
```

The API calculates: 360 cases x 25 lb = 9,000 lb.

**Request — ordering in lb (backward compatible):**
```json
{
  "customer_name": "Quali-Pack",
  "lines": [
    {
      "product_name": "Classic Granola 25 LB",
      "quantity_lb": 9000,
      "unit_price": 4.50
    }
  ]
}
```

**Request — cases with auto-lookup (no case_weight_lb needed if product has a default):**
```json
{
  "customer_name": "Quali-Pack",
  "lines": [
    {
      "product_name": "Classic Granola 25 LB",
      "quantity": 360,
      "unit": "cases",
      "unit_price": 4.50
    }
  ]
}
```

If `Classic Granola 25 LB` has `default_case_weight_lb = 25` in the products table, the API uses it automatically.

**Response:**
```json
{
  "order_id": 15,
  "order_number": "SO-260209-001",
  "customer": "QUALI-PACK USA",
  "requested_ship_date": "2026-02-15",
  "status": "new",
  "total_lb": 9000.0,
  "lines": [
    {
      "line_id": 42,
      "product": "Classic Granola 25 LB",
      "quantity_lb": 9000.0,
      "original_quantity": 360.0,
      "original_unit": "cases",
      "case_weight_lb": 25.0,
      "unit_price": 4.50
    }
  ],
  "warnings": null,
  "message": "Order SO-260209-001 created with 1 line(s)"
}
```

**Errors:**
- `400` — Multiple product matches, missing `case_weight_lb` with no default
- `404` — Product not found

**curl:**
```bash
curl -X POST "$BASE_URL/sales/orders" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Quali-Pack",
    "requested_ship_date": "2026-02-15",
    "lines": [
      {
        "product_name": "Classic Granola 25 LB",
        "quantity": 360,
        "unit": "cases",
        "unit_price": 4.50
      }
    ]
  }'
```

---

### List Sales Orders

```
GET /sales/orders
```

Returns orders with filtering and pagination.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | — | Filter by status (e.g., `confirmed`, `new`) |
| `customer` | string | — | Fuzzy match on customer name |
| `overdue_only` | bool | `false` | Only show orders past their ship date |
| `limit` | int | 50 | Max results (1–200) |

**Response:**
```json
{
  "orders": [
    {
      "order_id": 15,
      "order_number": "SO-260209-001",
      "customer": "QUALI-PACK USA",
      "order_date": "2026-02-09",
      "requested_ship_date": "2026-02-15",
      "status": "confirmed",
      "line_count": 1,
      "total_lb": 9000.0,
      "shipped_lb": 0.0,
      "remaining_lb": 9000.0,
      "overdue": false
    }
  ],
  "count": 1
}
```

**curl:**
```bash
curl "$BASE_URL/sales/orders?status=new&customer=quali&limit=10" \
  -H "X-API-Key: $API_KEY"
```

---

### Get Sales Order Detail

```
GET /sales/orders/{order_id}
```

Returns full order with lines, shipment history, and totals.

**Response:**
```json
{
  "order_id": 15,
  "order_number": "SO-260209-001",
  "customer": "QUALI-PACK USA",
  "order_date": "2026-02-09",
  "requested_ship_date": "2026-02-15",
  "status": "partial_ship",
  "notes": "Rush order",
  "created_date": "2026-02-09",
  "created_time": "10:30 AM ET",
  "lines": [
    {
      "line_id": 42,
      "product": "Classic Granola 25 LB",
      "quantity_lb": 9000.0,
      "quantity_shipped_lb": 5000.0,
      "remaining_lb": 4000.0,
      "unit_price": 4.50,
      "line_value": 40500.0,
      "line_status": "partial",
      "notes": null
    }
  ],
  "shipments": [
    {
      "shipment_id": 1,
      "line_id": 42,
      "product": "Classic Granola 25 LB",
      "quantity_lb": 5000.0,
      "shipped_date": "2026-02-10",
      "shipped_time": "02:15 PM ET",
      "transaction_id": 140
    }
  ],
  "totals": {
    "total_ordered_lb": 9000.0,
    "total_shipped_lb": 5000.0,
    "remaining_lb": 4000.0,
    "total_value": 40500.0
  }
}
```

---

### Update Order Status

```
PATCH /sales/orders/{order_id}/status
```

Moves an order to the next status. Enforces the state machine.

**Request:**
```json
{
  "status": "confirmed"
}
```

**Response:**
```json
{
  "order_number": "SO-260209-001",
  "previous_status": "new",
  "status": "confirmed",
  "message": "Order SO-260209-001: new → confirmed"
}
```

**Errors:**
- `400` — `"'shipped' status is set automatically when an order is shipped. Use the ship endpoint instead."`
- `400` — `"Invalid status transition: 'confirmed' → 'new'. Allowed transitions from 'confirmed': ['in_production', 'cancelled']."`
- `400` — `"Order SO-260209-001 is 'cancelled' — this is a terminal status. No further status changes are allowed."`

**curl:**
```bash
curl -X PATCH "$BASE_URL/sales/orders/15/status" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"status": "confirmed"}'
```

---

### Add Lines to Existing Order

```
POST /sales/orders/{order_id}/lines
```

Adds new lines to an existing order. Cannot add to shipped/invoiced/cancelled orders.

**Request:**
```json
{
  "lines": [
    {
      "product_name": "Granola Chocolate Chip 25 LB",
      "quantity": 100,
      "unit": "cases",
      "unit_price": 5.00
    }
  ]
}
```

**Response:**
```json
{
  "order_number": "SO-260209-001",
  "lines_added": [
    {
      "line_id": 43,
      "product": "Granola Chocolate Chip 25 LB",
      "quantity_lb": 2500.0,
      "original_quantity": 100.0,
      "original_unit": "cases",
      "case_weight_lb": 25.0
    }
  ],
  "message": "Added 1 line(s) to SO-260209-001"
}
```

---

### Update a Line

```
PATCH /sales/orders/{order_id}/lines/{line_id}/update?quantity_lb=10000
```

Updates quantity and/or price on an unfulfilled, non-cancelled line.

| Parameter | Type | Description |
|-----------|------|-------------|
| `quantity_lb` | float | New quantity in pounds |
| `unit_price` | float | New unit price |

At least one parameter is required.

**Response:**
```json
{
  "line_id": 42,
  "quantity_lb": 10000.0,
  "unit_price": 4.50
}
```

**Errors:**
- `400` — Nothing to update (no parameters provided)
- `404` — Line not found or already fulfilled/cancelled

---

### Cancel a Line

```
PATCH /sales/orders/{order_id}/lines/{line_id}/cancel
```

Cancels a line. Cannot cancel lines that are already fulfilled.

**Response:**
```json
{
  "line_id": 42,
  "line_status": "cancelled",
  "message": "Line cancelled"
}
```

---

## 4. Shipping Against Orders

### Ship Order Preview (Dry-Run)

```
POST /sales/orders/{order_id}/ship/preview
```

Checks inventory availability for each line without committing. Use this as a feasibility check.

**Request — ship everything remaining:**
```json
{
  "ship_all": true
}
```

**Request — ship specific lines:**
```json
{
  "lines": [
    {"line_id": 42, "quantity_lb": 5000}
  ]
}
```

**Request — no body (defaults to ship all):**
```bash
curl -X POST "$BASE_URL/sales/orders/15/ship/preview" \
  -H "X-API-Key: $API_KEY"
```

**Response:**
```json
{
  "order_number": "SO-260209-001",
  "customer": "QUALI-PACK USA",
  "status": "confirmed",
  "lines": [
    {
      "line_id": 42,
      "product": "Classic Granola 25 LB",
      "ordered_lb": 9000.0,
      "already_shipped_lb": 0.0,
      "remaining_lb": 9000.0,
      "requested_ship_lb": 9000.0,
      "can_ship_lb": 3200.0,
      "on_hand_lb": 3200.0,
      "short": 5800.0
    }
  ],
  "warnings": [
    "Classic Granola 25 LB: only 3200.0 lb on hand, need 9000.0 lb"
  ],
  "message": "Preview only — call /ship/commit to execute"
}
```

**Blocked statuses:**
- `new` — *"Cannot ship order — status is 'new'. Confirm the order first."*
- `invoiced`, `cancelled` — *"Cannot ship invoiced/cancelled order"*

---

### Ship Order Commit

```
POST /sales/orders/{order_id}/ship/commit
```

Executes the shipment. Deducts inventory from lots (FIFO by creation date), updates order balances, and auto-updates order status.

**Request:** Same as preview (ship_all or specific lines).

**Response:**
```json
{
  "order_number": "SO-260209-001",
  "customer": "QUALI-PACK USA",
  "order_status": "partial_ship",
  "lines_shipped": [
    {
      "line_id": 42,
      "product": "Classic Granola 25 LB",
      "requested_lb": 9000.0,
      "shipped_lb": 3200.0,
      "short_lb": 5800.0,
      "lots_used": [
        {"lot_code": "26-02-01-PROD-001", "quantity_lb": 1200.0},
        {"lot_code": "26-02-09-PROD-002", "quantity_lb": 2000.0}
      ],
      "transaction_id": 140,
      "line_status": "partial"
    }
  ],
  "message": "Shipped 1 line(s) for SO-260209-001"
}
```

**Auto-status update:**
- All lines fully shipped → order status set to `shipped`
- Some lines partially shipped or unfulfilled → order status set to `partial_ship`

**Partial shipments:** If inventory is insufficient, the system ships what's available and sets `line_status` to `partial`. Ship again later when more inventory is produced.

**curl:**
```bash
curl -X POST "$BASE_URL/sales/orders/15/ship/commit" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"ship_all": true}'
```

---

## 5. Sales Dashboard

```
GET /sales/dashboard
```

Returns a snapshot of all active orders, overdue items, upcoming due dates, and recent shipments.

**Response:**
```json
{
  "status_summary": {
    "new": 3,
    "confirmed": 2,
    "in_production": 1,
    "ready": 1,
    "partial_ship": 2
  },
  "overdue_orders": [
    {
      "order_number": "SO-260203-001",
      "customer": "Tropical Foods",
      "requested_ship_date": "2026-02-07",
      "remaining_lb": 2000.0
    }
  ],
  "overdue_count": 1,
  "due_this_week": [
    {
      "order_number": "SO-260209-001",
      "customer": "QUALI-PACK USA",
      "requested_ship_date": "2026-02-15",
      "remaining_lb": 9000.0
    }
  ],
  "due_this_week_count": 1,
  "recent_shipments_7d": [
    {
      "order_number": "SO-260209-001",
      "customer": "QUALI-PACK USA",
      "shipped_lb": 3200.0,
      "last_shipped_date": "2026-02-10",
      "last_shipped_time": "02:15 PM ET"
    }
  ],
  "as_of_date": "2026-02-10",
  "as_of_time": "03:30 PM ET"
}
```

**What each section shows:**
- **status_summary** — Count of orders by status (excludes invoiced and cancelled)
- **overdue_orders** — Ship date is past and order not yet shipped/invoiced/cancelled
- **due_this_week** — Ship date within next 7 days, not yet fulfilled
- **recent_shipments_7d** — All order shipments in the last 7 days

**curl:**
```bash
curl "$BASE_URL/sales/dashboard" \
  -H "X-API-Key: $API_KEY"
```

---

## 6. Fulfillment Check

```
GET /sales/orders/fulfillment-check
```

Read-only check across all open orders. Answers: "Which orders can I fulfill right now?" — without entering the shipping flow.

Scans orders in `confirmed`, `in_production`, and `ready` status. For each order, checks current inventory against every unfulfilled line.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `customer_name` | string | — | Fuzzy match on customer name |
| `status` | string | — | Filter to one status (`confirmed`, `in_production`, `ready`) |
| `order_id` | int | — | Check a single order by ID |

**Response:**
```json
{
  "summary": {
    "total_orders_checked": 8,
    "fulfillable": 5,
    "partially_fulfillable": 2,
    "blocked": 1
  },
  "orders": [
    {
      "order_id": 5,
      "order_number": "SO-260210-001",
      "customer": "QUALI-PACK USA",
      "status": "confirmed",
      "requested_ship_date": "2026-02-15",
      "fulfillable": true,
      "lines": [
        {
          "line_id": 12,
          "product": "Classic Granola 25 LB",
          "ordered_lb": 9000.0,
          "shipped_lb": 0.0,
          "remaining_lb": 9000.0,
          "on_hand_lb": 12500.0,
          "can_fulfill": true,
          "shortfall_lb": 0.0
        }
      ],
      "total_remaining_lb": 9000.0,
      "total_on_hand_lb": 12500.0,
      "total_shortfall_lb": 0.0
    }
  ]
}
```

**Summary classification:**
- **fulfillable** — All lines have enough inventory on hand
- **partially_fulfillable** — At least one line can be fulfilled, but not all
- **blocked** — No lines can be fulfilled (zero inventory for every product)

**Sort order:** Soonest ship date first, then fulfillable orders before blocked ones.

**Notes:**
- Strictly read-only — no inventory locks, reservations, or status changes
- Orders with all lines fully shipped (remaining = 0) are skipped
- Uses the same inventory query as `shipOrderPreview`

**Errors:**
- `400` — Invalid status filter (must be `confirmed`, `in_production`, or `ready`)

**curl examples:**
```bash
# All open orders
curl "$BASE_URL/sales/orders/fulfillment-check" \
  -H "X-API-Key: $API_KEY"

# Filter to one customer
curl "$BASE_URL/sales/orders/fulfillment-check?customer_name=Quali-Pack" \
  -H "X-API-Key: $API_KEY"

# Only ready-to-ship orders
curl "$BASE_URL/sales/orders/fulfillment-check?status=ready" \
  -H "X-API-Key: $API_KEY"

# Check a single order
curl "$BASE_URL/sales/orders/fulfillment-check?order_id=5" \
  -H "X-API-Key: $API_KEY"
```

---

## 7. Unit Handling & Sanity Checks

The system includes three warning mechanisms to catch common data entry errors. Warnings are returned in the response but do not block order creation.

### Unit Warning — No unit specified

When a line includes `quantity` but no `unit` field, the system defaults to pounds and returns a warning.

**Trigger:** `quantity` is provided, `unit` is `null` (not explicitly set)

**Example request:**
```json
{
  "lines": [{"product_name": "Classic Granola 25 LB", "quantity": 360}]
}
```

**Warning in response:**
```json
{
  "warnings": [
    "⚠️ 'Classic Granola 25 LB': No unit specified for quantity 360 — defaulting to lb. Did you mean cases?"
  ]
}
```

**How to avoid:** Always pass `"unit": "lb"` or `"unit": "cases"` explicitly.

---

### Quantity Warning — Abnormally low order

When a line's quantity is less than 25% of the customer's historical average for that product, the system flags it.

**Trigger:** `quantity_lb < (customer_avg_for_product * 0.25)`

**Example:** Quali-Pack's average order for Classic Granola is 9,000 lb. A new order for 100 lb triggers:

```json
{
  "warnings": [
    "⚠️ 'Classic Granola 25 LB': 100 lb is unusually low for QUALI-PACK USA. Their average order is 9,000 lb. Double-check the quantity."
  ]
}
```

**Notes:**
- Only triggers if the customer has prior orders for that product
- Excludes cancelled lines from the average calculation
- Does not block order creation — the order is still created

---

### Open Orders Warning — Standalone ship with open orders

When a standalone ship (`POST /ship/preview` or `/ship/commit`) is requested for a customer who has open sales orders, the system warns that you should ship against the order instead.

**Trigger:** Customer has unfulfilled sales orders (status not shipped/invoiced/cancelled, remaining quantity > 0)

**Warning in response:**
```json
{
  "open_orders_warning": "⚠️ WARNING: Quali-Pack has open sales order(s): SO-260209-001 (confirmed, 9,000 lb remaining). Consider using shipOrderPreview to ship against the order instead."
}
```

**Notes:**
- Uses fuzzy customer name matching (partial match)
- Appears in both preview and commit responses
- Does not block the standalone shipment — just warns

---

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/customers` | GET | List customers |
| `/customers/search` | GET | Search by name |
| `/customers` | POST | Create customer |
| `/customers/{id}` | PATCH | Update customer |
| `/sales/orders` | POST | Create sales order |
| `/sales/orders` | GET | List orders (with filters) |
| `/sales/orders/{id}` | GET | Get order detail |
| `/sales/orders/{id}/status` | PATCH | Update status (state machine enforced) |
| `/sales/orders/{id}/lines` | POST | Add lines to order |
| `/sales/orders/{id}/lines/{line_id}/update` | PATCH | Update line qty/price |
| `/sales/orders/{id}/lines/{line_id}/cancel` | PATCH | Cancel a line |
| `/sales/orders/{id}/ship/preview` | POST | Dry-run ship feasibility |
| `/sales/orders/{id}/ship/commit` | POST | Execute shipment |
| `/sales/orders/fulfillment-check` | GET | Bulk fulfillment feasibility check |
| `/sales/dashboard` | GET | Dashboard overview |

---

## 8. Bilingual Support (English + Spanish)

The system supports dual-language storage for all free-text fields. English is the system-of-record; Spanish is optional context.

### How It Works

Every free-text field (`notes`, `reason`, `reason_notes`) has an optional `_es` companion:

| English Field (required) | Spanish Companion (optional) |
|--------------------------|------------------------------|
| `reason` | `reason_es` |
| `notes` | `notes_es` |
| `reason_notes` | `reason_notes_es` |

### Validation Rules

1. **English always required** when Spanish is provided
2. **Spanish always optional** — can be omitted entirely
3. **Error if Spanish-only**: `"English version required. Provide 'reason' along with 'reason_es'."`

### Request Example

```json
{
  "product_name": "Classic Granola 25 LB",
  "lot_code": "26-02-09-PROD-001",
  "adjustment_lb": -25,
  "reason": "25 lb found damaged",
  "reason_es": "Encontramos 25 libras dañadas"
}
```

### Response Behavior

- If `_es` field exists in the database, it is included in the response
- If `_es` is null, only the English field is returned (no empty `_es` key)

```json
{
  "reason": "25 lb found damaged",
  "reason_es": "Encontramos 25 libras dañadas"
}
```

### Affected Endpoints

| Endpoint | Fields |
|----------|--------|
| `POST /adjust/commit` | `reason` / `reason_es` |
| `POST /products/quick-create` | `notes` / `notes_es` |
| `POST /products/quick-create-batch` | `notes` / `notes_es` |
| `POST /lots/{id}/reassign` | `reason_notes` / `reason_notes_es` |
| `POST /inventory/found` | `notes` / `notes_es` |
| `POST /inventory/found-with-new-product` | `notes` / `notes_es` |
| `POST /products/{id}/verify` | `notes` / `notes_es` |
| `POST /customers` | `notes` / `notes_es` |
| `PATCH /customers/{id}` | `notes` / `notes_es` |
| `POST /sales/orders` | `notes` / `notes_es` (order + lines) |
| `POST /sales/orders/{id}/lines` | `notes` / `notes_es` (per line) |
| `GET /sales/orders/{id}` | Returns `notes_es` if present |

### Reporting Rule

Only the English (base) field appears in exports, reports, and audit logs. The `_es` companion fields are internal/contextual only and are not included in any reporting output.
